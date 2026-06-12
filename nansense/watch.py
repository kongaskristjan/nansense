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

from nansense.patches import PatchAccumulator, PatchSnapshot

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
    """Vectorised version of `bin_index` for a flat fp32 tensor."""
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
    return torch.where(in_zero, zero, idx)


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


class TensorAccumulator:
    """Running stats for one tensor stream (activation OR gradient of a layer).

    All state lives on the device of the first non-empty tensor passed to
    `update()`. Reductions are computed in fp32; sums use fp32 (consumer GPUs
    don't always have fast fp64). For training runs measured in millions of
    elements this is comfortably precise — if you push much further you'd
    want Welford here.
    """

    def __init__(self) -> None:
        self._device: torch.device | None = None
        self._n: Tensor | None = None
        self._sum: Tensor | None = None
        self._sum_sq: Tensor | None = None
        self._min: Tensor | None = None
        self._max: Tensor | None = None
        self._hist: Tensor | None = None
        # `(C, N_BINS)` per-channel counts; allocated on the first update
        # with a usable channel axis, dropped by `collapse_channels`.
        self._channel_hist: Tensor | None = None
        self._channels_off = False

    def _lazy_init(self, device: torch.device) -> None:
        if self._device is not None:
            return
        self._device = device
        self._n = torch.zeros((), dtype=torch.int64, device=device)
        self._sum = torch.zeros((), dtype=torch.float32, device=device)
        self._sum_sq = torch.zeros((), dtype=torch.float32, device=device)
        self._min = torch.full((), float("inf"), dtype=torch.float32, device=device)
        self._max = torch.full(
            (), float("-inf"), dtype=torch.float32, device=device
        )
        self._hist = torch.zeros(N_BINS, dtype=torch.int64, device=device)

    def update(self, x: Tensor) -> None:
        if x.numel() == 0:
            return
        self._lazy_init(x.device)
        assert self._sum is not None
        assert self._sum_sq is not None
        assert self._min is not None
        assert self._max is not None
        assert self._hist is not None
        assert self._n is not None
        flat = x.detach().to(torch.float32).reshape(-1)
        self._n += flat.numel()
        self._sum += flat.sum()
        self._sum_sq += flat.square().sum()
        self._min = torch.minimum(self._min, flat.min())
        self._max = torch.maximum(self._max, flat.max())
        idx = _bin_indices(flat)
        channels = self._usable_channels(x)
        if channels is None:
            self._hist += torch.bincount(idx, minlength=N_BINS)
            return
        # One fused bincount over `channel * N_BINS + bin` gives the
        # per-channel counts; the universal histogram is their sum, so the
        # per-channel path costs one cheap reduction over what the universal
        # one already paid.
        view = [1, channels] + [1] * (x.ndim - 2)
        ch_idx = (
            torch.arange(channels, device=x.device)
            .view(view)
            .expand(x.shape)
            .reshape(-1)
        )
        counts = torch.bincount(
            ch_idx * N_BINS + idx, minlength=channels * N_BINS
        ).reshape(channels, N_BINS)
        assert self._channel_hist is not None
        self._channel_hist += counts
        self._hist += counts.sum(dim=0)

    def _usable_channels(self, x: Tensor) -> int | None:
        """Channel count to bin `x` under, managing the per-channel buffer.

        Returns `None` when per-channel tracking is off: 1D tensors have no
        channel axis, and a dim-1 size change mid-stream (variable token
        counts) makes per-channel rows meaningless, so either turns the
        tracking off for good and falls back to the universal histogram.
        """
        if self._channels_off:
            return None
        if x.ndim < 2:
            self.collapse_channels()
            return None
        channels = x.shape[1]
        if self._channel_hist is None:
            self._channel_hist = torch.zeros(
                channels, N_BINS, dtype=torch.int64, device=x.device
            )
        elif self._channel_hist.shape[0] != channels:
            self.collapse_channels()
            return None
        return channels

    def collapse_channels(self) -> None:
        """Drop the per-channel histogram for good, keeping the universal one.

        Called by `WatchAccumulator` when a newer epoch starts for the same
        `(layer, phase)` — only the latest epoch renders per-channel — and
        internally when per-channel binning isn't applicable.
        """
        self._channel_hist = None
        self._channels_off = True

    def snapshot(self) -> TensorStatsSnapshot:
        if self._device is None:
            return TensorStatsSnapshot(
                n=0,
                sum=0.0,
                sum_sq=0.0,
                min=float("inf"),
                max=float("-inf"),
                hist=tuple([0] * N_BINS),
            )
        assert self._n is not None
        assert self._sum is not None
        assert self._sum_sq is not None
        assert self._min is not None
        assert self._max is not None
        assert self._hist is not None
        # One sync per scalar group; the histogram lands separately because
        # it's int64 and the scalars are float32.
        scalars = torch.stack([self._sum, self._sum_sq, self._min, self._max]).cpu()
        n = int(self._n.cpu().item())
        hist_cpu = self._hist.cpu()
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
    stats.activations.collapse_channels()
    stats.gradients.collapse_channels()


def _evict_patches(stats: _LayerStats) -> None:
    stats.patches.clear()


class WatchAccumulator:
    """Thread-safe per-(layer, phase, epoch) accumulator store.

    Training-thread writes (`update`) and UI-thread reads (`snapshot`) are
    serialised by a single lock. Hold time is dominated by the GPU→CPU sync
    that `snapshot()` does once per UI refresh; per-batch `update()` only
    mutates GPU-side tensors so it's nearly lock-free in practice.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._stats: dict[tuple[str, str, int], _LayerStats] = {}

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
        acc.update(x)

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

        Histogram stats are small enough to keep for every epoch, but a
        patch bucket holds `4 × channels × N_PER_CHANNEL` image crops on
        the GPU — so the first patch update of a newer (layer, phase)
        epoch releases the older epochs' patch buffers. The UI only shows
        the latest epoch per phase, so nothing visible is lost. (Keyed off
        `patches_started` rather than bucket creation: `update` usually
        creates the bucket first, which must not skip the patch eviction.)
        """
        key = (layer, phase, epoch)
        with self._lock:
            stats = self._bucket_locked(key)
            if not stats.patches_started:
                stats.patches_started = True
                self._evict_older_locked(key, _evict_patches)
            acc = stats.patches
        acc.update(act=act, x=x)

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
        return WatchSnapshot(stats=out)
