"""Tests for distributed (DDP) support: nansense.distributed + session wiring.

The reduction math is covered single-process by packing two accumulators
("ranks") and reducing them by hand; the collective wiring is covered by
one real 2-rank gloo run spawned with torch.multiprocessing.
"""

from __future__ import annotations

from datetime import timedelta
from pathlib import Path
from typing import Any, cast

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.functional import cross_entropy
from torch.nn.parallel import DistributedDataParallel

import nansense
from nansense import distributed
from nansense.distributed import _unpack_reduced
from nansense.ui.app import serve
from nansense.watch import WatchAccumulator, bin_index

from tests.nansense.helpers import TinyNet


def _filled_accumulator(xs: list[torch.Tensor]) -> WatchAccumulator:
    acc = WatchAccumulator()
    for x in xs:
        acc.update(layer="L", phase="train", epoch=0, kind="activation", x=x)
    return acc


def _reduce_two(
    acc_a: WatchAccumulator, acc_b: WatchAccumulator
) -> distributed.ReducedStats:
    """Pack both accumulators against `acc_a`'s meta and reduce by hand."""
    meta = acc_a.reduce_meta(["L"])
    cpu = torch.device("cpu")
    ints_a, sums_a, mins_a, maxs_a = acc_a.export_for_reduce(meta, device=cpu)
    ints_b, sums_b, mins_b, maxs_b = acc_b.export_for_reduce(meta, device=cpu)
    return _unpack_reduced(
        meta,
        ints_a + ints_b,
        sums_a + sums_b,
        torch.minimum(mins_a, mins_b),
        torch.maximum(maxs_a, maxs_b),
    )


def test_reduced_stats_match_combined_accumulator() -> None:
    """Reducing two ranks' payloads equals feeding all data into one rank."""
    torch.manual_seed(0)
    xs_a = [torch.randn(2, 3, 4) for _ in range(2)]
    xs_b = [torch.randn(2, 3, 4) * 10 for _ in range(2)]
    acc_a = _filled_accumulator(xs_a)
    acc_b = _filled_accumulator(xs_b)
    combined = _filled_accumulator(xs_a + xs_b)

    act, grad = _reduce_two(acc_a, acc_b)[("L", "train", 0)]
    expected = (
        combined.snapshot(layers=["L"], include_patches=False)
        .stats[("L", "train", 0)]
        .activations
    )
    assert act.n == expected.n
    assert act.min == expected.min
    assert act.max == expected.max
    assert act.sum == pytest.approx(expected.sum, rel=1e-5)
    assert act.sum_sq == pytest.approx(expected.sum_sq, rel=1e-5)
    assert act.hist == expected.hist
    assert act.channel_hists == expected.channel_hists
    # Nothing ever fed the gradient stream: it reduces to a neutral zero.
    assert grad.n == 0 and grad.hist == tuple([0] * len(act.hist))


def test_reduce_with_missing_bucket_contributes_zeros() -> None:
    """A rank that never saw a bucket adds nothing to the reduction."""
    xs = [torch.ones(2, 3)]
    acc = _filled_accumulator(xs)
    own = acc.snapshot(layers=["L"], include_patches=False).stats[("L", "train", 0)]

    act, _ = _reduce_two(acc, WatchAccumulator())[("L", "train", 0)]
    assert act.n == own.activations.n
    assert act.hist == own.activations.hist
    assert act.channel_hists == own.activations.channel_hists


def test_reduce_channel_mismatch_drops_channel_hists() -> None:
    """Per-channel rows are dropped when a rank's channel count differs,
    while the universal histogram still sums both ranks exactly."""
    acc_a = _filled_accumulator([torch.ones(2, 3)])
    acc_b = _filled_accumulator([torch.ones(2, 5)])

    act, _ = _reduce_two(acc_a, acc_b)[("L", "train", 0)]
    assert act.channel_hists is None
    assert act.n == 16
    assert act.hist[bin_index(1.0)] == 16


class _FakeFollowerContext:
    is_leader = False


def _follower_session() -> nansense.Session:
    session = nansense.start(TinyNet(), epochs=1, phases={"train": 1})
    session._dist = cast(Any, _FakeFollowerContext())
    return session


def test_follower_serve_is_noop() -> None:
    session = _follower_session()
    assert not session.is_leader
    assert serve(session) is None


def test_distributed_time_travel_rejected(tmp_path: Path) -> None:
    session = _follower_session()
    with pytest.raises(RuntimeError, match="not supported with distributed"):
        session.training_restorer(cache_dir=tmp_path)


def test_unwrap_ddp_passes_plain_model_through() -> None:
    model = TinyNet()
    assert distributed.unwrap_ddp(model) is model


def _ddp_worker(rank: int, world_size: int, init_file: str) -> None:
    """One rank of the end-to-end gloo run; assertions fail the spawn."""
    dist.init_process_group(
        "gloo",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=world_size,
        timeout=timedelta(seconds=120),
    )
    try:
        model = TinyNet()
        ddp = DistributedDataParallel(model)
        session = nansense.start(ddp, epochs=1, phases={"train": 2})
        # The DDP wrapper is unwrapped: clean names, leader on rank 0 only.
        assert session.model is model
        assert session.layer_names == ["x", "fc1", "relu", "fc2"]
        assert session.is_leader == (rank == 0)
        if rank == 0:
            session.detach()
            assert session.watch("x")

        # Rank-distinguishable data: rank r feeds constant `r + 1`, so the
        # reduced stats are exactly predictable.
        for _ in range(2):
            x = torch.full((4, 4), float(rank + 1))
            y = torch.randint(0, 3, (4,))
            with session.batch(phase="train", epoch=0):
                ddp.zero_grad(set_to_none=True)
                cross_entropy(ddp(x), y).backward()

        key = ("x", "train", 0)
        if rank == 0:
            # The default update frequency published on the epoch's last
            # batch, which reduced the watch stats across both ranks.
            act = session.watch_snapshot(include_patches=False).stats[key].activations
            per_rank = 2 * 4 * 4  # batches x batch size x features
            assert act.n == world_size * per_rank
            assert act.min == 1.0
            assert act.max == 2.0
            assert act.sum == pytest.approx(per_rank * (1.0 + 2.0))
            assert act.hist[bin_index(1.0)] == per_rank
            assert act.hist[bin_index(2.0)] == per_rank
            assert act.channel_hists is not None and len(act.channel_hists) == 4
            assert all(sum(row) == 2 * 2 * 4 for row in act.channel_hists)
        else:
            # The follower picked the watched set up from the per-batch
            # broadcast and accumulated its own shard locally.
            assert session.watched_layers == frozenset({"x"})
            local = session._watch_accumulator.snapshot(
                layers=["x"], include_patches=False
            )
            assert local.stats[key].activations.n == 2 * 4 * 4
            assert local.stats[key].patches is None
        session.close()
    finally:
        dist.destroy_process_group()


def test_two_rank_gloo_end_to_end(tmp_path: Path) -> None:
    """Spawn a real 2-rank gloo group: watch-set sync + global reduction."""
    init_file = tmp_path / "ddp_init"
    mp.spawn(_ddp_worker, args=(2, str(init_file)), nprocs=2, join=True)
