"""Tests for distributed (DDP) support: nansense.distributed + session wiring.

The reduction math is covered single-process by packing two accumulators
("ranks") and reducing them by hand; the collective wiring is covered by
one real 2-rank gloo run spawned with torch.multiprocessing.

The spawned runs are hang-proofed in layers: waits *inside* a collective
trip gloo's 120s timeout, each worker arms a `faulthandler` watchdog that
dumps every thread's stack and dies if it wedges *outside* one, and the
parent's join carries a wall-clock ceiling as the last resort — so a wedged
run fails loudly with diagnostics instead of hanging pytest forever. A
finished worker leaves through `os._exit` rather than a normal return, which
sidesteps a torch teardown race (see `_worker_entry`).
"""

from __future__ import annotations

import faulthandler
import os
import sys
import time
from collections.abc import Callable
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

from tests.nansense.helpers import TinyNet, live_hist


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
    assert grad.n == 0 and live_hist(grad) == tuple([0] * len(live_hist(act)))


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
    assert live_hist(act)[bin_index(1.0)] == 16


class _FakeFollowerContext:
    is_leader = False
    rank = 1


def _follower_session() -> nansense.Session:
    session = nansense.start(TinyNet(), epochs=1, phases={"train": 1})
    session._dist = cast(Any, _FakeFollowerContext())
    return session


def test_follower_serve_is_noop() -> None:
    session = _follower_session()
    assert not session.is_leader
    assert serve(session) is None


def test_distributed_time_travel_restorer_allowed(tmp_path: Path) -> None:
    """The DDP guard is gone: a follower session can build a restorer.

    The follower's cache tags its files with its rank so the ranks never
    collide on disk (the real lockstep behaviour is covered end-to-end by
    `test_two_rank_gloo_time_travel_deterministic`).
    """
    session = _follower_session()  # _FakeFollowerContext has rank == 1
    restorer = session.training_restorer(cache_dir=tmp_path)
    assert restorer.cache.rank == 1
    assert restorer.cache.path_for(0).name == "epoch_0.rank1.pt"


def test_unwrap_ddp_passes_plain_model_through() -> None:
    model = TinyNet()
    assert distributed.unwrap_ddp(model) is model


# Both sit above the workers' 120s gloo timeout so a wait stuck inside a
# collective still surfaces as the gloo timeout error, not as a watchdog kill.
_WORKER_WATCHDOG_S = 150.0
_JOIN_CEILING_S = 240.0


def _arm_worker_watchdog() -> None:
    """Kill this worker with a full stack dump if it wedges for 150s.

    The gloo timeout only bounds waits inside a collective; a rank stuck
    anywhere else (e.g. a C-level driver call that never returns) would hang
    the spawn join — and the whole suite — indefinitely. `faulthandler`'s
    watchdog dumps every thread's traceback to stderr without needing the
    GIL, so it fires and exits even during such a hang.
    """
    faulthandler.dump_traceback_later(_WORKER_WATCHDOG_S, exit=True)


def _worker_entry(rank: int, worker: Callable[..., None], *args: object) -> None:
    """Run one rank's `worker`, then leave without interpreter finalization.

    `DistributedDataParallel` keeps torch's `ProcessGroupGloo` alive past
    `destroy_process_group()` — the backend's two `pt_gloo_runloop` threads
    are demonstrably still running after the group is destroyed, the wrapper
    is dropped and the cycle collector has run (reproducible in ten lines of
    plain torch, with no nansense in the picture). Those threads then race the
    C++ static teardown that follows `Py_Finalize`, and in roughly 1% of runs
    the race is lost: a worker that passed every assertion dies with
    "terminate called without an active exception" (SIGABRT), failing the
    parent's join. The abort lands after the interpreter is gone — a
    `faulthandler` dump of it shows no Python frame on any thread.

    A rank has nothing left to do once `worker` returns, so it exits with the
    success code `mp.spawn` checks for instead of finalizing. A raising worker
    never gets here: the exception propagates into torch's spawn wrapper,
    which reports the traceback to the parent as usual.
    """
    worker(rank, *args)
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


def _spawn_two_ranks(worker: Callable[..., None], *args: object) -> None:
    """`mp.spawn(nprocs=2, join=True)` with a wall-clock ceiling.

    The last line of defence behind the workers' own watchdogs: if a child
    wedges before it even arms one (e.g. during interpreter startup), the
    join here still kills it and fails the test instead of hanging pytest.
    """
    ctx = mp.spawn(_worker_entry, args=(worker, *args), nprocs=2, join=False)
    assert ctx is not None
    deadline = time.monotonic() + _JOIN_CEILING_S
    while not ctx.join(timeout=deadline - time.monotonic()):
        if time.monotonic() >= deadline:
            for process in ctx.processes:
                if process.is_alive():
                    process.kill()
            for process in ctx.processes:
                process.join()
            pytest.fail(
                f"2-rank gloo run still going after {_JOIN_CEILING_S:.0f}s; "
                "workers killed (their watchdog stack dumps are on stderr)"
            )


def _ddp_worker(rank: int, world_size: int, init_file: str) -> None:
    """One rank of the end-to-end gloo run; assertions fail the spawn."""
    _arm_worker_watchdog()
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
            # Suppress the per-epoch auto-publish (which now fires on the
            # epoch's *first* batch) so the run reduces exactly once, on the
            # explicit snapshot request below — after both batches accumulate.
            session.set_update_frequency(unit="batch", n=100)
            assert session.watch("x")

        # Rank-distinguishable data: rank r feeds constant `r + 1`, so the
        # reduced stats are exactly predictable.
        for i in range(2):
            x = torch.full((4, 4), float(rank + 1))
            y = torch.randint(0, 3, (4,))
            if rank == 0 and i == 1:
                session.request_snapshot()  # publish + reduce on the last batch
            with session.batch(phase="train", epoch=0):
                ddp.zero_grad(set_to_none=True)
                cross_entropy(ddp(x), y).backward()

        key = ("x", "train", 0)
        if rank == 0:
            # The snapshot request published on the epoch's last batch, which
            # reduced the watch stats across both ranks.
            act = session.watch_snapshot(include_patches=False).stats[key].activations
            per_rank = 2 * 4 * 4  # batches x batch size x features
            assert act.n == world_size * per_rank
            assert act.min == 1.0
            assert act.max == 2.0
            assert act.sum == pytest.approx(per_rank * (1.0 + 2.0))
            assert live_hist(act)[bin_index(1.0)] == per_rank
            assert live_hist(act)[bin_index(2.0)] == per_rank
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


def test_two_rank_gloo_end_to_end(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Spawn a real 2-rank gloo group: watch-set sync + global reduction."""
    # The workers are pure CPU/gloo; hide the GPU so no child ever
    # initializes a CUDA context (slow, and a driver-level wedge risk on a
    # GPU shared with a desktop — the CUDA RNG paths are covered by the
    # in-process time-travel tests).
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    init_file = tmp_path / "ddp_init"
    _spawn_two_ranks(_ddp_worker, 2, str(init_file))


def _stochastic_step(
    ddp: DistributedDataParallel, optimizer: torch.optim.Optimizer, rank: int
) -> None:
    """One DDP train step on data drawn from this rank's RNG.

    The input is `randn` (per-process RNG) so a deterministic replay depends
    on the per-rank RNG snapshot being restored, not just on the replicated
    model/optimizer state. DDP all-reduces the gradients, so the ranks stay
    in sync regardless of seeing different data.
    """
    x = torch.randn(4, 4) + rank  # rank-distinguishable, RNG-driven
    y = torch.randint(0, 3, (4,))
    optimizer.zero_grad(set_to_none=True)
    cross_entropy(ddp(x), y).backward()
    optimizer.step()


def _time_travel_worker(rank: int, world_size: int, init_file: str, cache_dir: str) -> None:
    """One rank of the time-travel determinism run; assertions fail the spawn.

    Every rank wraps a 3-epoch / 2-batch loop in a restorer. On the first
    attempt the leader arms a jump back to epoch 1 as it enters epoch 2; the
    jump is broadcast at the next batch-start barrier and EVERY rank raises
    `TimeTravelJump` in lockstep, then restores from its own per-rank
    checkpoint. The replayed epoch 1 must reproduce the original epoch 1's
    end-of-epoch weights bit-for-bit (model/opt restored + per-rank RNG
    restored => identical data => identical gradients). A hang would trip the
    gloo timeout and fail the spawn.
    """
    _arm_worker_watchdog()
    dist.init_process_group(
        "gloo",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=world_size,
        timeout=timedelta(seconds=120),
    )
    try:
        # Identical initial weights on every rank (DDP would broadcast them
        # anyway); seed the RNG per rank so the data streams differ.
        torch.manual_seed(0)
        model = TinyNet()
        ddp = DistributedDataParallel(model)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
        torch.manual_seed(100 + rank)

        epochs, batches = 3, 2
        session = nansense.start(
            ddp, epochs=epochs, phases={"train": batches}, optimizer=optimizer
        )
        if rank == 0:
            session.detach()  # free-running; the jump is armed programmatically
        restorer = session.training_restorer(cache_dir=Path(cache_dir))

        # End-of-epoch fc1 weights, per attempt, keyed by epoch.
        seen: list[tuple[int, int, torch.Tensor]] = []  # (attempt, epoch, weight)
        attempt = -1
        armed = False
        while restorer.pending():
            with restorer:
                attempt += 1
                # `request_time_travel` flips the leader to STEP mode; re-detach
                # each attempt so the single-threaded worker never pauses with
                # no UI to resume it (which would deadlock the followers).
                if rank == 0:
                    session.detach()
                for epoch in restorer.epochs():
                    # Arm the jump back to epoch 1 as we enter epoch 2 on the
                    # first attempt — epoch 1's checkpoint already exists.
                    if rank == 0 and attempt == 0 and epoch == 2 and not armed:
                        session.request_time_travel(1)
                        armed = True
                    for _ in session.batches(
                        range(batches), phase="train", epoch=epoch
                    ):
                        _stochastic_step(ddp, optimizer, rank)
                    seen.append(
                        (attempt, epoch, model.fc1.weight.detach().clone())
                    )

        # The leader armed a jump at epoch 2, so attempt 0 completed epochs
        # 0 and 1 (epoch 2 raised at its first batch, before logging), and
        # attempt 1 replayed epochs 1 and 2.
        epochs_per_attempt = [
            [e for a, e, _ in seen if a == att]
            for att in range(max(a for a, _, _ in seen) + 1)
        ]
        assert epochs_per_attempt == [[0, 1], [1, 2]], epochs_per_attempt

        # Determinism: the replayed epoch-1 weights match the original's on
        # EVERY rank (per-rank RNG restored => same data => same update).
        orig_ep1 = next(w for a, e, w in seen if a == 0 and e == 1)
        replay_ep1 = next(w for a, e, w in seen if a == 1 and e == 1)
        torch.testing.assert_close(replay_ep1, orig_ep1)
        assert restorer.finished
        session.close()
    finally:
        dist.destroy_process_group()


def test_two_rank_gloo_time_travel_deterministic(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Spawn a real 2-rank gloo group and jump back an epoch in lockstep.

    Exercises the full DDP time-travel path: the leader's armed jump is
    broadcast through `sync_batch_control`, every rank raises at the same
    barrier, each restores from its own per-rank checkpoint, and the replayed
    epoch reproduces the original deterministically — with no deadlock (a hang
    trips the 120s gloo timeout and fails the join).
    """
    # Without this, `capture_rng` in each child's epoch-start checkpoint
    # would initialize a full CUDA context just to snapshot GPU RNG the
    # worker never uses — the only concurrent CUDA init in the suite, and
    # the prime suspect in an observed indefinite two-rank wedge on a GPU
    # shared with a desktop. The CPU RNG snapshot is what the determinism
    # assertions exercise; CUDA RNG capture stays covered by the in-process
    # time-travel tests.
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    init_file = tmp_path / "tt_init"
    cache_dir = tmp_path / "tt_cache"
    _spawn_two_ranks(_time_travel_worker, 2, str(init_file), str(cache_dir))
    # Both ranks persisted their own epoch files (rank 0 canonical, rank 1
    # tagged), proving the per-rank checkpoint scheme actually wrote to disk.
    assert (cache_dir / "epoch_1.pt").exists()
    assert (cache_dir / "epoch_1.rank1.pt").exists()
