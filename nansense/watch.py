"""Running stats for watched layers — activations and activation gradients.

For each watched layer, we accumulate per `(phase, epoch)`:

- scalar reductions: count, sum, sum_of_squares, min, max
- a signed-log histogram with 211 bins covering `(-1e6, 1e6)`
- a per-channel `(C, 211)` variant of the same histogram, where channels
  are dim 1 of the batch-first tensor (features act as channels for 2D
  activations, matching the patch accumulator's convention)

The histogram has 7 bins per decade in log10 space on each sign:

    bins  0 .. 104 : negative bins, from `-1e6` down to `-1e-9`
    bin   105      : the "zero band" covering `(-1e-9, +1e-9)`
    bins  106 .. 210 : positive bins, from `+1e-9` up to `+1e6`

Bin edges land on powers of 10 and at six intermediate log-spaced points
between consecutive powers, so axis labels at the powers of 10 line up
with bin boundaries instead of bisecting bins. The two end bins are
open-ended: anything below `-1e6` or above `+1e6` saturates into them,
and the UI marks these as overflow.

The universal histogram is ~2 KB and lives forever, but a per-channel
buffer is `C × 211` int64 (≈ 0.9 MB at 512 channels), so only the most
recent epoch per `(layer, phase)` keeps one: when a new epoch of a phase
starts, that same phase's older epochs collapse to the universal
histogram — the phase-scoped release rule the patch buffers already use.
Per-channel binning also turns off permanently for an accumulator whose
dim-1 size changes mid-stream (e.g. variable token counts) or whose
tensors are 1D.

Per-channel data — both these histograms and the patch galleries — is
capped to the first `channel_limit` channels (the patches store a
per-channel input image, so this is the dominant GPU VRAM cost). The
universal histogram and the scalar reductions always cover *all* channels,
so the layer-wide view stays accurate regardless of the cap.

All running stats live on the device of the first tensor seen for that
accumulator (typically the model's training device). Inputs are cast to
fp32 before reduction so bf16/fp16 training doesn't lose precision in
the running sums or bin-assignment math. A `snapshot()` method copies
the running state to CPU as immutable `*Snapshot` dataclasses suitable
for the UI to consume.
"""

from __future__ import annotations

import math
import threading
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import Literal

import torch
from torch import Tensor

from nansense.patches import (
    DEFAULT_SAMPLES_PER_CHANNEL,
    PatchAccumulator,
    PatchSnapshot,
)

# Default cap on how many of a watched layer's channels keep per-channel data
# (per-channel histograms and the extreme-input patch galleries). The patches
# store a per-channel input image, so this is the dominant GPU VRAM knob; the
# live value is a user-tunable "Performance" setting and can be disabled.
DEFAULT_CHANNEL_LIMIT: int = 16

BINS_PER_DECADE: int = 7
LOG10_MIN: int = -9
LOG10_MAX: int = 6
DECADES: int = LOG10_MAX - LOG10_MIN  # 15
N_POS: int = BINS_PER_DECADE * DECADES  # 105
N_BINS: int = 2 * N_POS + 1  # 211
ZERO_BIN: int = N_POS  # 105
_SMALLEST_POSITIVE: float = 10.0**LOG10_MIN  # 1e-9


Kind = Literal["activation", "gradient"]


def histogram_edges() -> list[float]:
    """The 212 edges of the 211-bin signed-log histogram, ordered low to high.

    The first edge is `-1e6`, the last is `+1e6`. Edges 105 and 106 are
    `-1e-9` and `+1e-9` — they bracket the zero band (bin index 105).
    """
    pos_edges = [
        10.0 ** (LOG10_MIN + i / BINS_PER_DECADE) for i in range(N_POS + 1)
    ]
    neg_edges = [-e for e in reversed(pos_edges)]
    return neg_edges + pos_edges


def bin_index(value: float) -> int:
    """Return the bin index for a single Python float. Used by tests."""
    if not math.isfinite(value):
        if math.isnan(value):
            return ZERO_BIN
        return N_BINS - 1 if value > 0 else 0
    abs_v = abs(value)
    if abs_v < _SMALLEST_POSITIVE:
        return ZERO_BIN
    log_idx = (math.log10(abs_v) - LOG10_MIN) * BINS_PER_DECADE
    pos = max(0, min(N_POS - 1, int(log_idx)))
    return ZERO_BIN + 1 + pos if value > 0 else ZERO_BIN - 1 - pos


def _bin_indices(x: Tensor) -> Tensor:
    """Vectorised version of `bin_index` for a flat fp32 tensor.

    Non-finite inputs are mapped exactly as `bin_index` does: `nan` to the
    zero bin, `+inf` to the top overflow bin, `-inf` to the bottom one. The
    finite path is computed unconditionally (`log10` of inf/nan underflows
    on `.long()`, but those lanes are overwritten below), so finite values
    are unaffected.
    """
    abs_x = x.abs()
    in_zero = abs_x < _SMALLEST_POSITIVE
    # `clamp_min` keeps log10 finite for zero/subnormal inputs; those are
    # picked up by `in_zero` and re-mapped to the zero bin below.
    log_abs = torch.log10(abs_x.clamp_min(_SMALLEST_POSITIVE))
    pos = ((log_abs - LOG10_MIN) * BINS_PER_DECADE).long().clamp_(0, N_POS - 1)
    pos_idx = ZERO_BIN + 1 + pos
    neg_idx = ZERO_BIN - 1 - pos
    idx = torch.where(x >= 0, pos_idx, neg_idx)
    zero = torch.full_like(idx, ZERO_BIN)
    idx = torch.where(in_zero, zero, idx)
    # Overwrite non-finite lanes, whose finite-path indices are garbage.
    idx = torch.where(torch.isnan(x), zero, idx)
    idx = torch.where(torch.isposinf(x), torch.full_like(idx, N_BINS - 1), idx)
    return torch.where(torch.isneginf(x), torch.zeros_like(idx), idx)


@dataclass(frozen=True)
class TensorStatsSnapshot:
    """Immutable CPU-side view of a single (layer, phase, epoch, kind) accumulator."""

    n: int
    sum: float
    sum_sq: float
    min: float
    max: float
    hist: tuple[int, ...]
    # Per-channel histogram rows summing to `hist`; `None` when per-channel
    # tracking is off for this accumulator (1D tensors, a dim-1 size change,
    # or an older epoch collapsed when a newer one started for its phase).
    channel_hists: tuple[tuple[int, ...], ...] | None = None
    # The dtype of the source tensor (before the fp32 reduction cast), so the
    # UI can place the dtype-aware under/overflow band on the histogram.
    # `None` when no data has been seen (or after a cross-rank reduction that
    # didn't carry it).
    dtype: torch.dtype | None = None
    # Dead-channel count carried over from a collapsed per-channel histogram
    # (an older epoch whose buffers a newer one released). `None` while
    # `channel_hists` is live — read `dead_channel_count`, which covers both —
    # or when the count was never knowable (per-channel tracking off).
    collapsed_dead_count: int | None = None

    @property
    def mean(self) -> float:
        return self.sum / self.n if self.n > 0 else float("nan")

    @property
    def variance(self) -> float:
        if self.n < 2:
            return float("nan")
        mean = self.mean
        v = self.sum_sq / self.n - mean * mean
        return max(v, 0.0)

    @property
    def std(self) -> float:
        v = self.variance
        return math.sqrt(v) if math.isfinite(v) and v > 0 else 0.0

    @property
    def median(self) -> float:
        """Histogram-derived median: midpoint of the bin that holds the median."""
        if self.n == 0:
            return float("nan")
        half = self.n / 2
        running = 0
        for i, count in enumerate(self.hist):
            running += count
            if running >= half:
                return bin_midpoint(i)
        return bin_midpoint(N_BINS - 1)

    @property
    def dead_channel_count(self) -> int | None:
        """How many channels only ever hit the zero bin, `None` when unknown.

        Live from `channel_hists` while the per-channel histogram exists;
        the count stored at collapse time for an evicted older epoch.
        """
        if self.channel_hists is not None:
            return len(dead_channel_indices(self.channel_hists))
        return self.collapsed_dead_count


@dataclass(frozen=True)
class LayerStatsSnapshot:
    layer: str
    phase: str
    epoch: int
    activations: TensorStatsSnapshot
    gradients: TensorStatsSnapshot
    # Per-channel extreme-activation input patches; `None` when the bucket
    # never saw an image-like input (or was evicted by a newer epoch).
    patches: PatchSnapshot | None = None


@dataclass(frozen=True)
class WatchSnapshot:
    """Immutable view of all accumulated stats at a point in time.

    Keyed by `(layer, phase, epoch)`. The UI is expected to filter to the
    layers it wants to display (typically the latest epoch for each phase).
    """

    stats: dict[tuple[str, str, int], LayerStatsSnapshot] = field(
        default_factory=dict
    )
    # Weight-tensor stats sampled once per (layer, epoch) — at that epoch's
    # first watched batch — mapping the full parameter name to its stats.
    # Weights don't vary by phase, so the key has none.
    weights: dict[tuple[str, int], dict[str, TensorStatsSnapshot]] = field(
        default_factory=dict
    )

    def latest_per_phase(self, layer: str) -> dict[str, LayerStatsSnapshot]:
        """For `layer`, return `phase -> stats` for the most recent epoch seen.

        Returns an empty dict if the layer has no entries yet.
        """
        result: dict[str, LayerStatsSnapshot] = {}
        for (l, ph, ep), s in self.stats.items():
            if l != layer:
                continue
            existing = result.get(ph)
            if existing is None or ep > existing.epoch:
                result[ph] = s
        return result

    def phase_history(self, layer: str, phase: str) -> list[LayerStatsSnapshot]:
        """`layer`'s buckets for `phase`, ordered by epoch.

        The epoch-by-epoch series behind the value-vs-epoch stats view.
        Older epochs carry universal-histogram stats only (their per-channel
        buffers collapsed when a newer epoch started).
        """
        return sorted(
            (
                s
                for (l, ph, _), s in self.stats.items()
                if l == layer and ph == phase
            ),
            key=lambda s: s.epoch,
        )

    def weight_history(
        self, layer: str
    ) -> dict[str, list[tuple[int, TensorStatsSnapshot]]]:
        """`layer`'s per-epoch weight samples, `param -> [(epoch, stats)]`.

        Each list is ordered by epoch — the series behind the GRAPHS view's
        weight plots. Empty for layers without parameters (fx
        intermediates, graph inputs).
        """
        out: dict[str, list[tuple[int, TensorStatsSnapshot]]] = {}
        for (l, epoch), params in sorted(
            self.weights.items(), key=lambda kv: kv[0][1]
        ):
            if l != layer:
                continue
            for name, stats in params.items():
                out.setdefault(name, []).append((epoch, stats))
        return out


def dead_channel_indices(
    channel_hists: tuple[tuple[int, ...], ...],
) -> list[int]:
    """Indices of channels whose every observed value landed in the zero bin.

    The zero bin holds exact zeros and sub-`1e-9` magnitudes (NaNs also land
    there), so this flags channels that never produced a meaningful
    activation — e.g. dead ReLUs. A channel that never saw a value is not
    reported as dead.
    """
    return [
        c
        for c, row in enumerate(channel_hists)
        if sum(row) > 0 and row[ZERO_BIN] == sum(row)
    ]


def bin_midpoint(idx: int) -> float:
    """Linear-space value at the geometric midpoint of the given bin.

    The two extreme bins are open-ended; we report their closed edge as a
    representative value (it's the only finite point we have).
    """
    if idx == ZERO_BIN:
        return 0.0
    if idx == 0:
        return -(10.0**LOG10_MAX)
    if idx == N_BINS - 1:
        return 10.0**LOG10_MAX
    if idx > ZERO_BIN:
        k = idx - ZERO_BIN - 1
        sign = 1.0
    else:
        k = ZERO_BIN - 1 - idx
        sign = -1.0
    lo = LOG10_MIN + k / BINS_PER_DECADE
    hi = LOG10_MIN + (k + 1) / BINS_PER_DECADE
    return sign * 10.0 ** ((lo + hi) / 2)


@dataclass
class _RunningStats:
    """Device-resident running reductions, allocated together on first use.

    Grouping them in one non-optional bundle means `TensorAccumulator`
    carries a single `_RunningStats | None` instead of seven `Tensor |
    None` fields that every method had to assert through.
    """

    n: Tensor
    sum: Tensor
    sum_sq: Tensor
    min: Tensor
    max: Tensor
    hist: Tensor

    @staticmethod
    def zeros(device: torch.device) -> _RunningStats:
        return _RunningStats(
            n=torch.zeros((), dtype=torch.int64, device=device),
            sum=torch.zeros((), dtype=torch.float32, device=device),
            sum_sq=torch.zeros((), dtype=torch.float32, device=device),
            min=torch.full((), float("inf"), dtype=torch.float32, device=device),
            max=torch.full((), float("-inf"), dtype=torch.float32, device=device),
            hist=torch.zeros(N_BINS, dtype=torch.int64, device=device),
        )


class TensorAccumulator:
    """Running stats for one tensor stream (activation OR gradient of a layer).

    All state lives on the device of the first non-empty tensor passed to
    `update()`. Reductions are computed in fp32; sums use fp32 (consumer GPUs
    don't always have fast fp64). For training runs measured in millions of
    elements this is comfortably precise — if you push much further you'd
    want Welford here.
    """

    def __init__(self) -> None:
        # Allocated by the first non-empty `update()`, on that tensor's device.
        self._stats: _RunningStats | None = None
        # `(C, N_BINS)` per-channel counts; allocated on the first update
        # with a usable channel axis, dropped by `collapse_channels`.
        self._channel_hist: Tensor | None = None
        self._channels_off = False
        # The buffer's final dead-channel count, kept when an epoch-eviction
        # collapse drops it (`collapse_channels(keep_dead_count=True)`), so
        # older epochs keep their point on the dead-neurons timeline.
        self._collapsed_dead_count: int | None = None
        # The source dtype, recorded on the first non-empty update (before the
        # fp32 reduction cast), so the UI can show the under/overflow band.
        self._dtype: torch.dtype | None = None

    def update(self, x: Tensor, *, channel_limit: int | None = None) -> None:
        if x.numel() == 0:
            return
        if self._stats is None:
            self._stats = _RunningStats.zeros(x.device)
        if self._dtype is None:
            self._dtype = x.dtype
        stats = self._stats
        flat = x.detach().to(torch.float32).reshape(-1)
        # Scalar reductions run over the finite values only. A single NaN
        # would otherwise poison min/max for good (torch.minimum/maximum
        # propagate NaN) and an inf would push sum/mean/variance to nan or
        # inf — so `n`, `sum`, `min`, `max` describe the finite population.
        # Non-finite values are still counted in the histogram below (NaN in
        # the zero bin, ±inf in the end bins via `_bin_indices`), so a
        # diverged layer stays visible there and in the dead-channels row.
        finite = flat[torch.isfinite(flat)]
        if finite.numel() > 0:
            stats.n += finite.numel()
            stats.sum += finite.sum()
            stats.sum_sq += finite.square().sum()
            stats.min = torch.minimum(stats.min, finite.min())
            stats.max = torch.maximum(stats.max, finite.max())
        idx = _bin_indices(flat)
        channels = self._usable_channels(x, channel_limit)
        if channels is None:
            stats.hist += torch.bincount(idx, minlength=N_BINS)
            return
        if channels == x.shape[1]:
            # No effective cap: one fused bincount over `channel * N_BINS +
            # bin` gives the per-channel counts and the universal histogram is
            # their sum, so the per-channel path costs one cheap reduction over
            # what the universal one already paid.
            counts = self._channel_counts(x, idx, channels)
            assert self._channel_hist is not None
            self._channel_hist += counts
            stats.hist += counts.sum(dim=0)
            return
        # Channel-capped: the universal histogram and scalars still cover every
        # channel (so the layer-wide view stays accurate), but per-channel rows
        # are kept only for the first `channels` of them.
        stats.hist += torch.bincount(idx, minlength=N_BINS)
        limited = x[:, :channels]
        lim_idx = _bin_indices(limited.detach().to(torch.float32).reshape(-1))
        counts = self._channel_counts(limited, lim_idx, channels)
        assert self._channel_hist is not None
        self._channel_hist += counts

    @staticmethod
    def _channel_counts(x: Tensor, idx: Tensor, channels: int) -> Tensor:
        """Per-channel `(channels, N_BINS)` counts for `x`'s flat bin `idx`."""
        ch_idx = (
            torch.arange(channels, device=x.device)
            .view([1, channels] + [1] * (x.ndim - 2))
            .expand(x.shape)
            .reshape(-1)
        )
        return torch.bincount(
            ch_idx * N_BINS + idx, minlength=channels * N_BINS
        ).reshape(channels, N_BINS)

    def _usable_channels(self, x: Tensor, channel_limit: int | None) -> int | None:
        """Channel count to keep per-channel rows for, managing the buffer.

        Caps the count to the first `channel_limit` channels (`None` keeps
        every channel). Returns `None` when per-channel tracking is off: 1D
        tensors have no channel axis, and a change in the (capped) row count
        mid-stream (variable token counts) makes per-channel rows meaningless,
        so either turns the tracking off for good and falls back to the
        universal histogram.
        """
        if self._channels_off:
            return None
        if x.ndim < 2:
            self.collapse_channels()
            return None
        channels = int(x.shape[1])
        if channel_limit is not None:
            channels = min(channels, channel_limit)
        if self._channel_hist is None:
            self._channel_hist = torch.zeros(
                channels, N_BINS, dtype=torch.int64, device=x.device
            )
        elif self._channel_hist.shape[0] != channels:
            self.collapse_channels()
            return None
        return channels

    def collapse_channels(self, *, keep_dead_count: bool = False) -> None:
        """Drop the per-channel histogram for good, keeping the universal one.

        Called by `WatchAccumulator` when a newer epoch starts for the same
        `(layer, phase)` — only the latest epoch renders per-channel — and
        internally when per-channel binning isn't applicable. The eviction
        path passes `keep_dead_count`: there the dropped buffer covers its
        whole epoch, so its final dead-channel count is stored for the
        dead-neurons timeline (one small GPU→CPU sync, once per epoch
        boundary). The mid-stream disables leave the count unset — a
        partial buffer's count would lie.
        """
        if keep_dead_count and self._channel_hist is not None:
            totals = self._channel_hist.sum(dim=1)
            dead = (totals > 0) & (self._channel_hist[:, ZERO_BIN] == totals)
            self._collapsed_dead_count = int(dead.sum().item())
        self._channel_hist = None
        self._channels_off = True

    def channel_count(self) -> int:
        """Rows in the per-channel histogram, or 0 when none is kept."""
        return 0 if self._channel_hist is None else int(self._channel_hist.shape[0])

    def reduce_payload(
        self, *, channels: int, device: torch.device
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Flat tensors for a cross-rank reduction of this accumulator.

        `channels` is the channel-row count the reduction leader declared
        for this stream (0 = no per-channel slots). Returns, on `device`:

        - ints  int64 `[1 + N_BINS + channels*N_BINS]`: n, the universal
          histogram, and the per-channel rows (zeros when this rank has
          none or its row count differs from `channels`).
        - sums  float32 `[2]`: sum, sum_sq.
        - mins  float32 `[2]`: min, and a channel-ok flag — 0 when this
          rank holds data but no per-channel rows matching `channels`, so
          a MIN-reduce tells the leader the combined rows are incomplete.
        - maxs  float32 `[1]`: max.

        An accumulator that never saw data contributes neutral values
        (zero counts, ±inf extremes, channel-ok 1). The local state is
        not mutated, so repeated reductions cannot double-count.
        """
        stats = self._stats
        if stats is None:
            return (
                torch.zeros(
                    1 + N_BINS + channels * N_BINS, dtype=torch.int64, device=device
                ),
                torch.zeros(2, dtype=torch.float32, device=device),
                torch.tensor([float("inf"), 1.0], dtype=torch.float32, device=device),
                torch.tensor([float("-inf")], dtype=torch.float32, device=device),
            )
        ch_ok = 1.0
        int_pieces = [stats.n.reshape(1), stats.hist]
        if channels > 0:
            ch = self._channel_hist
            if ch is not None and ch.shape[0] == channels:
                int_pieces.append(ch.reshape(-1))
            else:
                ch_ok = 0.0
                int_pieces.append(
                    torch.zeros(
                        channels * N_BINS, dtype=torch.int64, device=stats.hist.device
                    )
                )
        return (
            torch.cat(int_pieces).to(device),
            torch.stack([stats.sum, stats.sum_sq]).to(device),
            torch.stack(
                [
                    stats.min,
                    torch.tensor(ch_ok, dtype=torch.float32, device=stats.min.device),
                ]
            ).to(device),
            stats.max.reshape(1).to(device),
        )

    def snapshot(self) -> TensorStatsSnapshot:
        stats = self._stats
        if stats is None:
            return TensorStatsSnapshot(
                n=0,
                sum=0.0,
                sum_sq=0.0,
                min=float("inf"),
                max=float("-inf"),
                hist=tuple([0] * N_BINS),
                dtype=self._dtype,
            )
        # One sync per scalar group; the histogram lands separately because
        # it's int64 and the scalars are float32.
        scalars = torch.stack([stats.sum, stats.sum_sq, stats.min, stats.max]).cpu()
        n = int(stats.n.cpu().item())
        hist_cpu = stats.hist.cpu()
        channel_hists: tuple[tuple[int, ...], ...] | None = None
        if self._channel_hist is not None:
            channel_hists = tuple(
                tuple(row) for row in self._channel_hist.cpu().tolist()
            )
        return TensorStatsSnapshot(
            n=n,
            sum=float(scalars[0].item()),
            sum_sq=float(scalars[1].item()),
            min=float(scalars[2].item()),
            max=float(scalars[3].item()),
            hist=tuple(int(c) for c in hist_cpu.tolist()),
            channel_hists=channel_hists,
            dtype=self._dtype,
            collapsed_dead_count=self._collapsed_dead_count,
        )


@dataclass
class _LayerStats:
    activations: TensorAccumulator = field(default_factory=TensorAccumulator)
    gradients: TensorAccumulator = field(default_factory=TensorAccumulator)
    patches: PatchAccumulator = field(default_factory=PatchAccumulator)
    # Whether `update_patches` has touched this bucket yet — the trigger for
    # releasing the same phase's older-epoch patch buffers exactly once.
    patches_started: bool = False


def _evict_channel_hists(stats: _LayerStats) -> None:
    stats.activations.collapse_channels(keep_dead_count=True)
    stats.gradients.collapse_channels(keep_dead_count=True)


def _evict_patches(stats: _LayerStats) -> None:
    stats.patches.clear()


class WatchAccumulator:
    """Per-(layer, phase, epoch) accumulator store shared across threads.

    A single lock guards the bucket *map* — get-or-create on the training
    thread's `update`, the eviction passes, and the UI thread's `snapshot`
    walk. The heavy per-bucket tensor accumulation runs *outside* the lock
    (only the bucket lookup is serialised), so a `snapshot` may observe a
    bucket mid-update; the stats are monotonic running aggregates, so such a
    transient read simply reconciles on the next refresh. Hold time is
    dominated by the GPU→CPU sync `snapshot()` does once per UI refresh;
    per-batch `update()` only launches GPU kernels.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._stats: dict[tuple[str, str, int], _LayerStats] = {}
        # Weight-tensor samples, one bucket per (layer, epoch) captured at
        # that epoch's first watched batch: full param name -> accumulator
        # fed exactly one tensor. Untouched by `configure` — weights keep no
        # per-channel data, so the caps don't shape these buffers.
        self._weights: dict[tuple[str, int], dict[str, TensorAccumulator]] = {}
        # Performance config, applied to every accumulation. `channel_limit`
        # is `None` when the cap is disabled. Held under `_lock` so the read in
        # `update`/`update_patches` and a `configure` flush can't interleave —
        # a bucket can never be built under one config and fed under another.
        self._channel_limit: int | None = DEFAULT_CHANNEL_LIMIT
        self._samples_per_channel: int = DEFAULT_SAMPLES_PER_CHANNEL

    def configure(
        self, *, channel_limit: int | None, samples_per_channel: int
    ) -> bool:
        """Set the per-channel caps, flushing all buckets if they changed.

        The channel limit and samples-per-channel fix the per-channel buffer
        shapes, so a change can't be folded into existing buckets — every
        bucket is dropped and rebuilt under the new config on the next update.
        Returns whether anything changed (and was flushed).
        """
        with self._lock:
            if (
                channel_limit == self._channel_limit
                and samples_per_channel == self._samples_per_channel
            ):
                return False
            self._channel_limit = channel_limit
            self._samples_per_channel = samples_per_channel
            self._stats.clear()
            return True

    def update(
        self,
        *,
        layer: str,
        phase: str,
        epoch: int,
        kind: Kind,
        x: Tensor,
    ) -> None:
        key = (layer, phase, epoch)
        with self._lock:
            stats = self._bucket_locked(key)
            acc = stats.activations if kind == "activation" else stats.gradients
            channel_limit = self._channel_limit
        acc.update(x, channel_limit=channel_limit)

    def update_patches(
        self,
        *,
        layer: str,
        phase: str,
        epoch: int,
        act: Tensor,
        x: Tensor,
    ) -> None:
        """Fold one batch into `layer`'s extreme-patch buffers.

        Histogram stats are small enough to keep for every epoch, but a patch
        bucket holds `4 × channels × n_per_channel` image crops on the GPU — so
        the first patch update of a newer (layer, phase) epoch releases the
        older epochs' patch buffers. The UI only shows the latest epoch per
        phase, so nothing visible is lost. (Keyed off `patches_started` rather
        than bucket creation: `update` usually creates the bucket first, which
        must not skip the patch eviction.)
        """
        key = (layer, phase, epoch)
        with self._lock:
            stats = self._bucket_locked(key)
            if not stats.patches_started:
                stats.patches_started = True
                self._evict_older_locked(key, _evict_patches)
            acc = stats.patches
            channel_limit = self._channel_limit
            samples = self._samples_per_channel
        acc.update(
            act=act, x=x, channel_limit=channel_limit, n_per_channel=samples
        )

    def weights_pending(self, layer: str, epoch: int) -> bool:
        """Whether `(layer, epoch)` still needs its weight sample.

        The cheap per-batch check the session makes before materialising
        the parameter tensors: true only until the epoch's first watched
        batch captures them.
        """
        with self._lock:
            return (layer, epoch) not in self._weights

    def update_weights(
        self,
        *,
        layer: str,
        epoch: int,
        params: list[tuple[str, Tensor]],
    ) -> None:
        """Sample `layer`'s weight tensors for `epoch`, once.

        Each tensor folds into its own single-shot accumulator with
        per-channel tracking off (weights render no per-channel view), so
        the state stays on the training device — the GPU→CPU sync happens
        in `snapshot()`, off the training thread. A second call for the
        same `(layer, epoch)` is a no-op, which is what pins the sample to
        the epoch's first watched batch.
        """
        key = (layer, epoch)
        with self._lock:
            if key in self._weights:
                return
        bucket: dict[str, TensorAccumulator] = {}
        for name, tensor in params:
            acc = TensorAccumulator()
            acc.collapse_channels()
            acc.update(tensor)
            bucket[name] = acc
        # Built fully before publication so `snapshot` never iterates a
        # half-filled bucket; the training thread is the sole writer, so
        # the check above can't race another insert.
        with self._lock:
            self._weights[key] = bucket

    def _bucket_locked(self, key: tuple[str, str, int]) -> _LayerStats:
        """Get-or-create the (layer, phase, epoch) bucket (lock held).

        Creation means a new epoch of this phase begins: the *same* phase's
        older epochs release their per-channel histogram buffers (only the
        latest epoch per phase renders per-channel). Other phases keep
        theirs until their own next epoch starts.
        """
        stats = self._stats.get(key)
        if stats is None:
            stats = _LayerStats()
            self._stats[key] = stats
            self._evict_older_locked(key, _evict_channel_hists)
        return stats

    def _evict_older_locked(
        self,
        key: tuple[str, str, int],
        evict: Callable[[_LayerStats], None],
    ) -> None:
        """Run `evict` on the same (layer, phase)'s older-epoch buckets."""
        layer, phase, epoch = key
        for (l, ph, ep), other in self._stats.items():
            if l == layer and ph == phase and ep < epoch:
                evict(other)

    def forget_layer(self, layer: str) -> None:
        """Drop all stored stats for `layer` (e.g. on unwatch)."""
        with self._lock:
            for key in list(self._stats):
                if key[0] == layer:
                    del self._stats[key]
            for wkey in list(self._weights):
                if wkey[0] == layer:
                    del self._weights[wkey]

    def retain_layers(self, layers: Iterable[str]) -> None:
        """Drop buckets for any layer not in `layers`.

        Called by the training thread — the sole `update` caller — before
        its per-batch updates. `forget_layer` on `unwatch` runs on the UI
        thread and can race a batch: the training thread may already hold a
        snapshot of the watched set and recreate the just-forgotten bucket in
        `update`, leaving it stranded (the layer is no longer watched, so
        nothing ever forgets it again) and leaking its GPU buffers. Reaping
        unwatched layers here, where no `update` can resurrect them, closes
        that leak.
        """
        keep = set(layers)
        with self._lock:
            for key in list(self._stats):
                if key[0] not in keep:
                    del self._stats[key]
            for wkey in list(self._weights):
                if wkey[0] not in keep:
                    del self._weights[wkey]

    def forget_epochs_from(self, epoch: int) -> None:
        """Drop stats for `epoch` and later, across all layers and phases.

        Called when time travel rewinds to the start of `epoch`: the
        abandoned timeline's buckets must not absorb the re-run epochs'
        samples (the accumulators are additive) or linger on the watch page.
        """
        with self._lock:
            for key in list(self._stats):
                if key[2] >= epoch:
                    del self._stats[key]
            for wkey in list(self._weights):
                if wkey[1] >= epoch:
                    del self._weights[wkey]

    def reduce_meta(
        self, layers: Iterable[str]
    ) -> list[tuple[tuple[str, str, int], int, int]]:
        """Bucket list for a cross-rank reduction (the leader's side).

        Returns the sorted `(layer, phase, epoch)` keys restricted to
        `layers`, each with the activation and gradient streams' local
        per-channel row counts (0 = no per-channel buffer) — the layout
        contract every rank's `export_for_reduce` then packs against.
        """
        wanted = set(layers)
        with self._lock:
            keys = sorted(k for k in self._stats if k[0] in wanted)
            return [
                (
                    key,
                    self._stats[key].activations.channel_count(),
                    self._stats[key].gradients.channel_count(),
                )
                for key in keys
            ]

    def export_for_reduce(
        self,
        meta: list[tuple[tuple[str, str, int], int, int]],
        *,
        device: torch.device,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Pack the buckets named by `meta` into flat reduction tensors.

        Concatenates each bucket's activation- and gradient-stream
        `reduce_payload` in `meta` order; buckets this rank never created
        contribute neutral values. Runs on the training thread, which is
        the only writer, so the payload reads happen safely outside the
        lock.
        """
        with self._lock:
            buckets = [self._stats.get(key) for key, _, _ in meta]
        empty = _LayerStats()
        ints: list[Tensor] = []
        sums: list[Tensor] = []
        mins: list[Tensor] = []
        maxs: list[Tensor] = []
        for (_, c_act, c_grad), bucket in zip(meta, buckets):
            stats = bucket if bucket is not None else empty
            for acc, channels in (
                (stats.activations, c_act),
                (stats.gradients, c_grad),
            ):
                i, s, mn, mx = acc.reduce_payload(channels=channels, device=device)
                ints.append(i)
                sums.append(s)
                mins.append(mn)
                maxs.append(mx)
        return torch.cat(ints), torch.cat(sums), torch.cat(mins), torch.cat(maxs)

    def snapshot(
        self,
        *,
        layers: Iterable[str] | None = None,
        include_patches: bool = True,
    ) -> WatchSnapshot:
        """Snapshot all (or the requested subset of) layers' stats.

        `include_patches=False` skips the patch buffers' GPU→CPU copies —
        they're the bulk of the sync, and a caller showing only histograms
        has no use for them.
        """
        wanted = set(layers) if layers is not None else None
        with self._lock:
            keys = [k for k in self._stats if wanted is None or k[0] in wanted]
            stats_refs = [(k, self._stats[k]) for k in keys]
            weight_refs = [
                (k, dict(bucket))
                for k, bucket in self._weights.items()
                if wanted is None or k[0] in wanted
            ]
        # Compute snapshots outside the lock — `TensorAccumulator.snapshot`
        # does GPU→CPU syncs that we don't want serialised with updates.
        out: dict[tuple[str, str, int], LayerStatsSnapshot] = {}
        for (layer, phase, epoch), stats in stats_refs:
            out[(layer, phase, epoch)] = LayerStatsSnapshot(
                layer=layer,
                phase=phase,
                epoch=epoch,
                activations=stats.activations.snapshot(),
                gradients=stats.gradients.snapshot(),
                patches=stats.patches.snapshot() if include_patches else None,
            )
        weights = {
            key: {name: acc.snapshot() for name, acc in bucket.items()}
            for key, bucket in weight_refs
        }
        return WatchSnapshot(stats=out, weights=weights)


def single_batch_stats(
    *,
    layer: str,
    phase: str,
    epoch: int,
    activation: Tensor | None,
    gradient: Tensor | None,
    patch_source: Tensor | None,
    channel_limit: int | None,
    samples_per_channel: int,
    include_patches: bool,
) -> LayerStatsSnapshot:
    """Stats for a single batch's tensors, computed on the fly.

    Folds exactly one batch's activation (and gradient, and image patches)
    into throwaway accumulators — the `/stats` page's "Current batch" view,
    which reads the published `BatchSnapshot` so it works for *any* layer
    whether or not it is watched (the running `WatchAccumulator` only covers
    watched layers). Reusing the same accumulators the watch path uses means
    the result renders through the identical histogram / MIN-MAX code.
    """
    act_acc = TensorAccumulator()
    if activation is not None:
        act_acc.update(activation, channel_limit=channel_limit)
    grad_acc = TensorAccumulator()
    if gradient is not None:
        grad_acc.update(gradient, channel_limit=channel_limit)
    patches: PatchSnapshot | None = None
    if include_patches and activation is not None and patch_source is not None:
        patch_acc = PatchAccumulator()
        patch_acc.update(
            act=activation,
            x=patch_source,
            channel_limit=channel_limit,
            n_per_channel=samples_per_channel,
        )
        patches = patch_acc.snapshot()
    return LayerStatsSnapshot(
        layer=layer,
        phase=phase,
        epoch=epoch,
        activations=act_acc.snapshot(),
        gradients=grad_acc.snapshot(),
        patches=patches,
    )
