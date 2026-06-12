"""Multi-rank (DDP) awareness for nansense sessions.

In a `torch.distributed` run every rank constructs a session over the same
model, but the ranks play different roles:

- **Rank 0 (leader).** Behaves like a single-process session: serves the
  UI, publishes snapshots, pauses on captures, runs probes/experiments.
- **Other ranks (followers).** Never serve, publish, or pause. They follow
  the leader's watched-layer set, accumulate watch stats over their own
  data shard, and join the collective reduction that turns the leader's
  watch page into a *global* view.

Coordination happens at two points, both on the training thread:

1. **Per-batch control broadcast** (`sync_batch_control`, at
   `_BatchContext.__enter__`). A 2-int tensor from the leader: whether this
   batch's stats get reduced at `__exit__`, and the watched-set version.
   When the version changed since the last broadcast, a follow-up object
   broadcast carries the watched-layer list — so steady-state batches pay
   one tiny broadcast. This is also the pacing point: a leader paused in
   the UI holds every follower at its next batch start, mirroring how DDP
   itself would block them in the next gradient all-reduce.
2. **Stats reduction** (`reduce_watch_stats`, at `__exit__` of every batch
   the leader publishes — mode captures and frequency updates). The leader
   broadcasts the ordered bucket list, every rank packs its local
   accumulator state into four flat tensors (see
   `TensorAccumulator.reduce_payload`), and all-reduces combine them: SUM
   for counts/sums/histograms, MIN/MAX for the extremes. The leader
   unpacks the result into per-bucket `TensorStatsSnapshot`s that
   `Session.watch_snapshot` overlays on its local view.

The reduction is non-destructive — local accumulators are never mutated —
so repeated reductions can't double-count. Extreme-input patches stay
rank-local (gathering image crops across ranks isn't worth the traffic);
the min/max page shows the leader's shard.

Ranks must drive the same batch structure (same `session.batch` calls in
the same order), which DDP training loops do naturally. Time travel is
not supported in distributed mode: a jump would have to restore and rewind
every rank in lockstep.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import torch
import torch.distributed as dist
from torch import Tensor, nn
from torch.nn.parallel import DistributedDataParallel

from nansense.watch import N_BINS, TensorStatsSnapshot

if TYPE_CHECKING:
    from nansense.session import Session

# One reduction bucket: the accumulator key plus the leader's per-channel
# histogram channel counts for the activation and gradient streams (0 when
# the leader holds no per-channel buffer for that stream).
ReduceMeta = list[tuple[tuple[str, str, int], int, int]]
ReducedStats = dict[tuple[str, str, int], tuple[TensorStatsSnapshot, TensorStatsSnapshot]]


def unwrap_ddp(model: nn.Module) -> nn.Module:
    """Return the inner module of a DDP-wrapped model (or `model` as-is).

    Hooks installed on the inner module still fire through the wrapper's
    forward, while the inner module keeps clean layer names (no `module.`
    prefix) and stays fx-traceable — the wrapper itself is not.
    """
    if isinstance(model, DistributedDataParallel):
        return cast(nn.Module, model.module)
    return model


class DistContext:
    """This process's place in the distributed run, plus broadcast state."""

    def __init__(self) -> None:
        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        # NCCL collectives need device tensors; everything else uses CPU.
        backend = str(dist.get_backend()).lower()
        self.device = (
            torch.device("cuda", torch.cuda.current_device())
            if "nccl" in backend and torch.cuda.is_available()
            else torch.device("cpu")
        )
        # Watched-set version already shared via broadcast. Advances in
        # lockstep on every rank (each batch broadcasts the leader's current
        # version), so "did it change since the last batch" is a decision
        # every rank makes identically — the object broadcast below stays
        # collective without the leader knowing any follower's cache state.
        self._synced_version = -1

    @property
    def is_leader(self) -> bool:
        return self.rank == 0

    def broadcast_control(
        self, *, publish: bool, version: int, watched: list[str]
    ) -> None:
        """Leader side of the per-batch control sync."""
        t = torch.tensor([int(publish), version], dtype=torch.int64, device=self.device)
        dist.broadcast(t, src=0)
        if version != self._synced_version:
            dist.broadcast_object_list([watched], src=0, device=self.device)
            self._synced_version = version

    def recv_control(self) -> tuple[bool, list[str] | None]:
        """Follower side: returns (publish, watched-or-None-if-unchanged)."""
        t = torch.zeros(2, dtype=torch.int64, device=self.device)
        dist.broadcast(t, src=0)
        publish, version = (int(v) for v in t.cpu().tolist())
        watched: list[str] | None = None
        if version != self._synced_version:
            obj: list[object] = [None]
            dist.broadcast_object_list(obj, src=0, device=self.device)
            watched = cast(list[str], obj[0])
            self._synced_version = version
        return bool(publish), watched


def context() -> DistContext | None:
    """The process's distributed context, or `None` outside multi-rank runs.

    A single-rank process group needs no coordination, so it gets `None`
    too — the session behaves exactly like a non-distributed one.
    """
    if not dist.is_available() or not dist.is_initialized():
        return None
    if dist.get_world_size() < 2:
        return None
    return DistContext()


def sync_batch_control(session: Session, *, publish: bool) -> bool:
    """Per-batch control sync (all ranks, training thread, at batch start).

    The leader announces whether this batch ends in a stats reduction
    (`publish` and at least one layer watched) and shares watched-set
    changes; followers apply them — including dropping the accumulator
    buckets of layers that were unwatched, mirroring `Session.unwatch`.
    Returns the reduction flag on every rank.
    """
    ctx = session._dist
    assert ctx is not None
    if ctx.is_leader:
        with session._cv:
            version = session._watch_version
            watched = sorted(session._watched_layers)
        flag = publish and bool(watched)
        ctx.broadcast_control(publish=flag, version=version, watched=watched)
        return flag
    flag, new_watched = ctx.recv_control()
    if new_watched is not None:
        with session._cv:
            removed = session._watched_layers - set(new_watched)
            session._watched_layers = set(new_watched)
        for name in removed:
            session._watch_accumulator.forget_layer(name)
    return flag


def reduce_watch_stats(session: Session) -> None:
    """Collectively reduce watch stats (all ranks, training thread).

    The leader stores the reduced per-bucket stats on the session for
    `watch_snapshot` to overlay; followers just contribute and return.
    """
    ctx = session._dist
    assert ctx is not None
    acc = session._watch_accumulator
    if ctx.is_leader:
        with session._cv:
            layers = list(session._watched_layers)
        obj: list[object] = [acc.reduce_meta(layers)]
    else:
        obj = [None]
    dist.broadcast_object_list(obj, src=0, device=ctx.device)
    meta = cast(ReduceMeta, obj[0])
    if not meta:
        return
    ints, sums, mins, maxs = acc.export_for_reduce(meta, device=ctx.device)
    dist.all_reduce(ints, op=dist.ReduceOp.SUM)
    dist.all_reduce(sums, op=dist.ReduceOp.SUM)
    dist.all_reduce(mins, op=dist.ReduceOp.MIN)
    dist.all_reduce(maxs, op=dist.ReduceOp.MAX)
    if ctx.is_leader:
        session._dist_watch_stats = _unpack_reduced(meta, ints, sums, mins, maxs)


def _unpack_reduced(
    meta: ReduceMeta, ints: Tensor, sums: Tensor, mins: Tensor, maxs: Tensor
) -> ReducedStats:
    """Turn the all-reduced flat tensors back into per-bucket snapshots.

    Layout per (bucket, stream) pair, in `meta` order — the mirror of
    `TensorAccumulator.reduce_payload`:

    - `ints`  (SUM-reduced, int64): n, the 211-bin histogram, then
      `channels × 211` per-channel counts when the meta entry declared a
      channel count.
    - `sums`  (SUM, float32): sum, sum_sq.
    - `mins`  (MIN, float32): min, then the channel-ok flag — 1 unless some
      rank holds data whose per-channel buffer is missing or shaped
      differently, in which case the reduced per-channel rows are
      incomplete and are dropped (the universal histogram is still exact).
    - `maxs`  (MAX, float32): max.
    """
    ints_l = ints.cpu().tolist()
    sums_l = sums.cpu().tolist()
    mins_l = mins.cpu().tolist()
    maxs_l = maxs.cpu().tolist()
    out: ReducedStats = {}
    int_off = 0
    pair = 0
    for key, c_act, c_grad in meta:
        snaps: list[TensorStatsSnapshot] = []
        for channels in (c_act, c_grad):
            n = int(ints_l[int_off])
            hist = tuple(int(v) for v in ints_l[int_off + 1 : int_off + 1 + N_BINS])
            ch_start = int_off + 1 + N_BINS
            channel_hists: tuple[tuple[int, ...], ...] | None = None
            if channels > 0 and mins_l[2 * pair + 1] >= 0.5:
                channel_hists = tuple(
                    tuple(
                        int(v)
                        for v in ints_l[
                            ch_start + c * N_BINS : ch_start + (c + 1) * N_BINS
                        ]
                    )
                    for c in range(channels)
                )
            int_off = ch_start + channels * N_BINS
            snaps.append(
                TensorStatsSnapshot(
                    n=n,
                    sum=float(sums_l[2 * pair]),
                    sum_sq=float(sums_l[2 * pair + 1]),
                    min=float(mins_l[2 * pair]),
                    max=float(maxs_l[pair]),
                    hist=hist,
                    channel_hists=channel_hists,
                )
            )
            pair += 1
        out[key] = (snaps[0], snaps[1])
    return out
