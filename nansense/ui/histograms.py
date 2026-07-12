"""Histogram math and Plotly figure construction for the stats views.

Pure functions over plotly + snapshot types — no UI state. The public
(non-underscore) names are also the contract `nansense.recording` renders
watch-view frames with, so recorded histograms match the page exactly.
"""

from __future__ import annotations

import bisect
import html
import math
from collections.abc import Callable, Sequence

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from nansense.watch import (
    BINS_PER_DECADE,
    LOG10_MAX,
    LOG10_MIN,
    N_BINS,
    ZERO_BIN,
    LayerStatsSnapshot,
    TensorStatsSnapshot,
    bin_midpoint,
    dead_channel_indices,
    histogram_edges,
)


_PHASE_COLORS: dict[str, str] = {
    "train": "#d97706",  # amber
    "val": "#3b82f6",  # blue
    "test": "#10b981",  # emerald — fallback if a user names their phases differently
}
_FALLBACK_COLORS: tuple[str, ...] = ("#a855f7", "#ef4444", "#14b8a6", "#6b7280")


def phase_color(phase: str, idx: int) -> str:
    return _PHASE_COLORS.get(phase, _FALLBACK_COLORS[idx % len(_FALLBACK_COLORS)])


def x_tick_layout() -> tuple[list[int], list[str]]:
    """Tick positions (bin indices) and labels for the signed-log x-axis.

    Labels are drawn only at powers of 10 (every 7th edge); the
    intermediate edges shape the bars but are unlabeled to keep the axis
    legible.
    """
    tick_vals: list[int] = [ZERO_BIN]
    tick_text: list[str] = ["0"]
    for k in range(LOG10_MIN, LOG10_MAX + 1):
        offset = (k - LOG10_MIN) * BINS_PER_DECADE
        label = "1" if k == 0 else f"1e{k}"
        tick_vals.append(ZERO_BIN + 1 + offset)
        tick_text.append(label)
        tick_vals.append(ZERO_BIN - 1 - offset)
        tick_text.append("-1" if k == 0 else f"-1e{k}")
    return tick_vals, tick_text


def _format_stat(value: float) -> str:
    """Format a scalar stat for the card header."""
    if not math.isfinite(value):
        return "—"
    if value == 0:
        return "0"
    abs_v = abs(value)
    if abs_v >= 1000 or abs_v < 0.01:
        return f"{value:.2e}"
    return f"{value:.3g}"


# Rows of the per-histogram stats table: label and how to format the value.
# The std guard matches the other rows' "—" on an empty stream (its snapshot
# property falls back to 0.0 there, where mean/median go nan and min/max ±inf).
_STAT_ROWS: tuple[tuple[str, Callable[[TensorStatsSnapshot], str]], ...] = (
    ("n", lambda s: f"{s.n:,}"),
    ("mean", lambda s: _format_stat(s.mean)),
    ("std", lambda s: _format_stat(s.std) if s.n > 0 else "—"),
    ("median", lambda s: _format_stat(s.median)),
    ("min", lambda s: _format_stat(s.min)),
    ("max", lambda s: _format_stat(s.max)),
)

# Cap on channel indices listed in the dead-channels hover tooltip.
_DEAD_CHANNELS_LISTED: int = 10


def dead_channels(stats: TensorStatsSnapshot) -> list[int] | None:
    """Indices of channels whose every observed value landed in the zero bin.

    See `watch.dead_channel_indices` for the dead notion. Returns `None`
    when per-channel histograms aren't tracked for this stream.
    """
    if stats.channel_hists is None:
        return None
    return dead_channel_indices(stats.channel_hists)


def _dead_channels_cell(stats: TensorStatsSnapshot) -> str:
    """The activation column's dead-channel cell: count, hover for indices.

    Shows "—" when per-channel histograms aren't available for the stream;
    a non-zero count gets a dotted underline plus a `title` tooltip listing
    the first `_DEAD_CHANNELS_LISTED` indices ("..." marks truncation).
    """
    dead = dead_channels(stats)
    if dead is None:
        return f'<td style="{_STATS_CELL_STYLE};color:#1e293b">—</td>'
    listed = ", ".join(str(c) for c in dead[:_DEAD_CHANNELS_LISTED])
    if len(dead) > _DEAD_CHANNELS_LISTED:
        listed += ", ..."
    hover = (
        f' title="channels: {listed}"'
        f' style="{_STATS_CELL_STYLE};color:#1e293b;'
        'text-decoration:underline dotted;cursor:help"'
        if dead
        else f' style="{_STATS_CELL_STYLE};color:#1e293b"'
    )
    return f"<td{hover}>{len(dead)}</td>"

_STATS_CELL_STYLE: str = "padding:2px 26px 2px 0;text-align:left"

# Light framed card around each stats table so it stands out from the page
# instead of floating as bare text.
_STATS_BOX_STYLE: str = (
    "display:inline-block;background:#f8fafc;border:1px solid #e2e8f0;"
    "border-radius:6px;padding:8px 14px"
)


def _stats_table_html(per_phase: dict[str, LayerStatsSnapshot]) -> str:
    """Scalar stats as an HTML table: activations and gradients side by side.

    One framed table per phase with data, its corner header ("train ep 0")
    tinted with the phase's trace color so it reads against the matching
    bars in the histograms below, and one value column per tensor kind. A
    kind with no samples yet keeps its column ("—" cells) so the shape
    stays stable — e.g. a val phase collects activations but never
    gradients. The dead-channels row (count, indices on hover) applies to
    activations only. Returns a plain "no data yet" note while no phase
    has samples.
    """
    blocks: list[str] = []
    for i, (phase, snap) in enumerate(per_phase.items()):
        if snap.activations.n == 0 and snap.gradients.n == 0:
            continue
        header = (
            f'<th style="{_STATS_CELL_STYLE};font-weight:700;'
            f"border-bottom:1px solid #e2e8f0;"
            f'color:{phase_color(phase, i)}">'
            f"{html.escape(phase)} ep {snap.epoch}</th>"
            + "".join(
                f'<th style="{_STATS_CELL_STYLE};font-weight:700;'
                f'border-bottom:1px solid #e2e8f0;color:#334155">{kind}</th>'
                for kind in ("activations", "gradients")
            )
        )
        rows = "".join(
            f'<tr><td style="{_STATS_CELL_STYLE};color:#64748b">{label}</td>'
            + "".join(
                f'<td style="{_STATS_CELL_STYLE};color:#1e293b">'
                f"{fmt(stats)}</td>"
                for stats in (snap.activations, snap.gradients)
            )
            + "</tr>"
            for label, fmt in _STAT_ROWS
        )
        rows += (
            f'<tr><td style="{_STATS_CELL_STYLE};color:#64748b">dead channels'
            f"</td>{_dead_channels_cell(snap.activations)}"
            f'<td style="{_STATS_CELL_STYLE};color:#1e293b">—</td></tr>'
        )
        blocks.append(
            f'<div style="{_STATS_BOX_STYLE}">'
            '<table style="border-collapse:collapse">'
            f"<thead><tr>{header}</tr></thead>"
            f"<tbody>{rows}</tbody></table></div>"
        )
    if not blocks:
        return '<span class="text-slate-500">no data yet</span>'
    return '<div class="flex flex-wrap gap-3">' + "".join(blocks) + "</div>"


# Plot height in px. Doubled from the original 220 so the distributions are
# easier to read.
_PLOT_HEIGHT: int = 440

# Tick-label exponent style for every Plotly axis in the library: powers of
# ten, never Plotly's default SI prefix letters — "30f" for 3e-14 reads as a
# unit, not a scale. Hover values inherit the same exponent style.
AXIS_EXPONENT_FORMAT: str = "power"

# Linear-space geometry of the signed-log bins, used when the x-axis is
# switched to a linear scale: each bar is drawn at the centre of its bin and
# given the bin's true (linear) width so the bars tile the value axis.
_HIST_EDGES: list[float] = histogram_edges()
BIN_CENTERS: list[float] = [
    (_HIST_EDGES[i] + _HIST_EDGES[i + 1]) / 2 for i in range(N_BINS)
]
BIN_WIDTHS: list[float] = [
    _HIST_EDGES[i + 1] - _HIST_EDGES[i] for i in range(N_BINS)
]


def _linear_bar_x(x_range: list[float] | None) -> list[float]:
    """Bar x-positions for the linear value axis, off-view bins blanked.

    All 211 bins keep their slot so a bar's index still equals its bin index
    (the per-channel sample hover maps `pointNumber` straight to a bin), but
    the centres of bins outside the visible `x_range` are set to NaN, which
    drops them from both the drawing and Plotly's hit-testing.

    Without this, the trace also carries the far-flung extreme-tail bins
    (centres out to ±1e6, widths ~1e5) while the axis is zoomed to a tiny
    window — e.g. an O(1e-3) gradient distribution. Plotly's bar
    `hovermode="x"` closest-bar search then degenerates and locks onto one of
    those off-screen bars, so the tooltip reports a fixed empty bin (count 0)
    no matter where the cursor is. Blanking the off-view centres confines the
    search to the bins actually on screen, so the hover tracks the cursor.
    `x_range` is `None` (full-span autorange) only when there's no data, so
    nothing is blanked there.
    """
    if x_range is None:
        return list(BIN_CENTERS)
    lo, hi = x_range
    return [c if lo <= c <= hi else math.nan for c in BIN_CENTERS]


# Hover labels for the signed-log view, where bars sit at plain bin indices:
# each bin's representative value (its geometric midpoint, the same notion
# the median stat uses) instead of the meaningless index.
_BIN_VALUE_LABELS: list[str] = [f"{bin_midpoint(i):.3g}" for i in range(N_BINS)]

# Axis trims may clip bins/bars holding up to this share of the data points
# (see `_trimmed_bin_bounds` / `_linear_y_range`). `_axis_ranges` starts at
# the base share and raises it in steps up to the max while the bars would
# fill less than `_MIN_FILL_FRACTION` of the plot area. The max is high
# because activation-gradient magnitudes routinely spread near-uniformly
# over several decades: on a linear value axis such a distribution stays a
# hairline spike until well over half the points are allowed off-view, and
# a clipped-but-readable plot beats an unclipped empty one (the loop stops
# escalating the moment the plot is readable, so compact distributions
# never get near the max).
_BASE_CLIP_SHARE: float = 0.005
_MAX_CLIP_SHARE: float = 0.5
_CLIP_SHARE_STEP: float = 0.005

# Minimum share of the plot area the bars should cover; below this the clip
# share keeps being raised (up to `_MAX_CLIP_SHARE`). A lone Gaussian-ish
# mode drawn over a ±4σ window covers ~0.2 of the plot, so 0.15 keeps such
# well-behaved distributions at the base trim while pushing spike-plus-tail
# shapes to zoom in further.
_MIN_FILL_FRACTION: float = 0.15

# A bar more than this many times taller than the runner-up in its phase is
# a freak spike (e.g. ReLU's exact zeros) and never anchors the y-scale.
_DOMINANCE_RATIO: float = 5.0


def use_density(log_x: bool) -> bool:
    """Whether bars show probability density instead of probabilities.

    On a linear value axis, per-bin probabilities are misleading: the
    signed-log bins differ in linear width by orders of magnitude, so a wide
    bin towers over a narrow one holding the same share of values. Density
    makes bar *area* proportional to probability, the honest reading of a
    distribution on a linear value axis. With the signed-log x-axis the bins
    render at uniform width, so plain probabilities are kept there.
    """
    return not log_x


def _probabilities(hist: tuple[int, ...]) -> list[float]:
    """Per-bin probability: count divided by the total count."""
    n = sum(hist)
    if n == 0:
        return [0.0] * len(hist)
    return [c / n for c in hist]


def _probability_densities(hist: tuple[int, ...]) -> list[float]:
    """Per-bin probability density: probability divided by the bin's width."""
    n = sum(hist)
    if n == 0:
        return [0.0] * len(hist)
    return [c / (n * w) for c, w in zip(hist, BIN_WIDTHS)]


def trace_heights(hist: tuple[int, ...], density: bool) -> list[float]:
    """Bar heights for one trace: probability densities or probabilities."""
    return _probability_densities(hist) if density else _probabilities(hist)


def _hover_customdata(hist: tuple[int, ...], density: bool) -> list[object]:
    """Per-bar hover payload, matching `_make_histogram_figure`'s templates.

    Density mode (linear x): the raw count alone — the bar's own x position
    is already the value. Signed-log mode: `[count, value-label]` pairs, so
    the hover can show the bin's value instead of its meaningless index.
    """
    if density:
        return list(hist)
    return [
        [count, label] for count, label in zip(hist, _BIN_VALUE_LABELS)
    ]


def _scale_bars(hist: tuple[int, ...], density: bool) -> list[tuple[float, int]]:
    """One trace's `(height, count)` bars that may anchor the y-scale.

    Sorted tallest-first, with a single drastically dominant bar — more than
    `_DOMINANCE_RATIO` times the runner-up — dropped no matter how many
    points it holds: a value the data hits exactly (e.g. ReLU zeros piling
    into the 2e-9-wide zero band) produces a bar that would otherwise
    flatten the rest of the distribution.
    """
    heights = trace_heights(hist, density)
    bars = sorted(((h, c) for h, c in zip(heights, hist) if c > 0), reverse=True)
    if len(bars) >= 2 and bars[0][0] > _DOMINANCE_RATIO * bars[1][0]:
        return bars[1:]
    return bars


def _linear_y_range(
    phase_hists: _PhaseHists,
    density: bool,
    clip_share: float = _BASE_CLIP_SHARE,
) -> list[float] | None:
    """Y-axis range on a linear y-axis, capped under a clip budget.

    Two clipping rules keep freak spikes from flattening the rest of the
    distribution (with **Log y** checked the axis autoranges instead, so
    everything is visible there):

    - Per phase, a single drastically dominant bar never anchors the scale
      (see `_scale_bars`).
    - Among the rest, bars clip tallest-first, but only as long as the
      clipped bars together hold less than `clip_share` of the pooled data
      points — the cap lands on the tallest bar that must stay fully
      visible.

    The same range is applied to every phase's subplot row so the rows stay
    comparable. Returns `None` (Plotly autorange) when there's no data.
    """
    bars: list[tuple[float, int]] = []
    total = 0
    for _, hist in phase_hists:
        total += sum(hist)
        bars.extend(_scale_bars(hist, density))
    if not bars:
        return None
    bars.sort(reverse=True)
    allowed = clip_share * total
    cap = bars[0][0]
    clipped = 0
    for height, count in bars:
        if clipped + count > allowed:
            cap = height
            break
        clipped += count
    return [0.0, cap * 1.05]


# Overflow markers: a bar taller than the capped y-axis is drawn cut off flat
# at the plot's top edge, which reads as if it ended there. A marker at the cap
# flags every such bar as continuing above — its true height/count stays in the
# bar's hover. Slate-900 so the glyph reads as an annotation against any bar
# color; drawn just inside the top edge so the whole triangle stays visible.
_OVERFLOW_MARKER_COLOR: str = "#0f172a"
_OVERFLOW_MARKER_Y_FRAC: float = 0.97


def _overflow_marks(
    phase_hists: _PhaseHists,
    x_values: Sequence[float],
    density: bool,
    y_top: float | None,
) -> list[tuple[list[float], list[float]]]:
    """Per-trace `(xs, ys)` overflow-marker positions for clipped bars.

    For each drawn trace, the bins whose bar height exceeds `y_top` (the
    capped axis top) get a marker just inside the top edge, so a clipped bar
    reads as "continues above" instead of looking like it ends at the plot
    boundary. `x_values` is the same x array the bars use (bin indices on the
    signed-log axis, bin centres — off-view ones `NaN` — on the linear axis),
    so the marks line up with their bars and inherit that blanking. Returns
    empty `(xs, ys)` per trace when there's no cap (`y_top` is `None`, e.g. the
    log-y autorange, where nothing is clipped) or nothing overflows it.
    """
    if y_top is None or y_top <= 0.0:
        return [([], []) for _ in phase_hists]
    mark_y = y_top * _OVERFLOW_MARKER_Y_FRAC
    marks: list[tuple[list[float], list[float]]] = []
    for _, hist in phase_hists:
        heights = trace_heights(hist, density)
        xs = [
            float(x_values[i])
            for i, h in enumerate(heights)
            if h > y_top and not math.isnan(x_values[i])
        ]
        marks.append((xs, [mark_y] * len(xs)))
    return marks


def kind_stats(layer_snap: LayerStatsSnapshot, kind: str) -> TensorStatsSnapshot:
    """The activation or gradient stats of a layer snapshot, by `kind`."""
    return layer_snap.activations if kind == "activation" else layer_snap.gradients


def _phases_with_data(
    per_phase: dict[str, LayerStatsSnapshot], kind: str
) -> list[str]:
    """Phases that have bins to draw for `kind`, in render order.

    This is exactly the set (and order) of traces `_make_histogram_figure`
    draws, so it doubles as the signature the panel uses to decide whether a
    refresh can restyle in place or must rebuild the figure. A bucket whose
    bins an epoch eviction collapsed (a time-travel rewind can leave one as
    a phase's latest) has samples but nothing drawable, so it is excluded
    like an empty one.
    """
    return [
        p
        for p, snap in per_phase.items()
        if kind_stats(snap, kind).n > 0 and kind_stats(snap, kind).hist is not None
    ]


# The figure's drawn traces for one `kind`: a `(phase, histogram)` pair per
# phase with data, in render order. The entry points (`_make_histogram_figure`,
# `axis_ranges`) extract this once via `_phase_hists` and the range helpers
# below all work over it instead of re-deriving it from the snapshots.
_PhaseHists = list[tuple[str, tuple[int, ...]]]


def _phase_hists(
    per_phase: dict[str, LayerStatsSnapshot], kind: str
) -> _PhaseHists:
    """The drawn traces' `(phase, histogram)` pairs (see `_PhaseHists`)."""
    pairs: _PhaseHists = []
    for p in _phases_with_data(per_phase, kind):
        hist = kind_stats(per_phase[p], kind).hist
        assert hist is not None  # _phases_with_data excludes collapsed buckets
        pairs.append((p, hist))
    return pairs


def _trimmed_bin_bounds(
    phase_hists: _PhaseHists,
    clip_share: float = _BASE_CLIP_SHARE,
    *,
    density: bool = False,
) -> tuple[int, int] | None:
    """Smallest/largest bin indices after trimming the extreme-tail bins.

    Pooled across the drawn traces, the outermost populated bins are dropped
    greedily while the dropped bins together hold less than `clip_share` of
    the data points, so the x-range keeps the bins holding the rest of the
    values and a lone outlier value no longer stretches the whole value
    axis. Which end gets trimmed each step depends on `density`:

    - `density=False` (the uniform-width signed-log axis): drop the lighter
      end first, minimising the mass clipped per step.
    - `density=True` (the linear value axis): drop the end that frees the
      most *value* span per point clipped. The signed-log bins differ in
      linear width by orders of magnitude, so trimming purely by mass eats
      one whole tail before touching the other and leaves a lopsided window;
      weighting by the width actually freed keeps a symmetric distribution
      symmetric and zooms past the wide outer bins where the bulk isn't.

    Returns `None` when there's no data.
    """
    counts = [0] * N_BINS
    for _, hist in phase_hists:
        for i, count in enumerate(hist):
            counts[i] += count
    total = sum(counts)
    if total == 0:
        return None
    lo = next(i for i, c in enumerate(counts) if c > 0)
    hi = next(i for i in range(N_BINS - 1, -1, -1) if counts[i] > 0)
    allowed = clip_share * total
    trimmed = 0
    while lo < hi:
        # Trimming an end pulls the bound in to the next populated bin.
        next_lo = next(i for i in range(lo + 1, hi + 1) if counts[i] > 0)
        next_hi = next(i for i in range(hi - 1, lo - 1, -1) if counts[i] > 0)
        if density:
            # Value span freed per point clipped, compared cross-multiplied;
            # ties go to the low end, which alternates ends on symmetric
            # tails. The span includes the width of any emptied bins skipped.
            gain_lo = _HIST_EDGES[next_lo] - _HIST_EDGES[lo]
            gain_hi = _HIST_EDGES[hi + 1] - _HIST_EDGES[next_hi + 1]
            trim_lo = gain_lo * counts[hi] >= gain_hi * counts[lo]
        else:
            trim_lo = counts[lo] <= counts[hi]
        side = lo if trim_lo else hi
        if trimmed + counts[side] > allowed:
            break
        trimmed += counts[side]
        lo, hi = (next_lo, hi) if side == lo else (lo, next_hi)
    return lo, hi


def _linear_x_range(bounds: tuple[int, int] | None) -> list[float] | None:
    """X-axis range (linear value space) covering the trimmed bins.

    On a linear axis the bars span the full `[-1e6, 1e6]` edge range, almost
    all of it empty. We zoom to the edges of the trimmed bin span (`bounds`,
    as computed by `_trimmed_bin_bounds`, plus a little padding) so the bulk
    of the distribution stays legible. Returns `None` (Plotly autorange)
    when there's no data (`bounds` is `None`).
    """
    if bounds is None:
        return None
    lo = _HIST_EDGES[bounds[0]]
    hi = _HIST_EDGES[bounds[1] + 1]
    pad = (hi - lo) * 0.05 or 1.0
    return [lo - pad, hi + pad]


def _log_x_range(bounds: tuple[int, int] | None) -> list[float] | None:
    """X-axis range (bin-index space) covering the trimmed bins.

    The signed-log view draws bars at integer bin indices, so the range
    brackets the trimmed span (`bounds`, as computed by
    `_trimmed_bin_bounds`) with half-bar margins plus a little padding.
    Returns `None` (Plotly autorange — the full 211-bin span) when there's
    no data (`bounds` is `None`).
    """
    if bounds is None:
        return None
    lo, hi = bounds
    pad = (hi - lo + 1) * 0.05
    return [lo - 0.5 - pad, hi + 0.5 + pad]


def _fill_fraction(
    phase_hists: _PhaseHists,
    density: bool,
    bounds: tuple[int, int],
    y_top: float,
) -> float:
    """Share of the plot area the bars cover at the given axis ranges.

    Bar areas are measured in the units the bars are drawn in (value space
    for density mode, bin-index space for the signed-log view) with heights
    clipped to the y-range top; the plot area is the x-span times the
    y-range top, averaged over the drawn traces (every subplot row shares
    the same ranges).
    """
    if y_top <= 0:
        return 1.0
    lo, hi = bounds
    span = (_HIST_EDGES[hi + 1] - _HIST_EDGES[lo]) if density else (hi - lo + 1)
    filled = 0.0
    for _, hist in phase_hists:
        heights = trace_heights(hist, density)
        for i in range(lo, hi + 1):
            width = BIN_WIDTHS[i] if density else 1.0
            filled += width * min(heights[i], y_top)
    return filled / (span * y_top * len(phase_hists))


def axis_ranges(
    per_phase: dict[str, LayerStatsSnapshot],
    kind: str,
    *,
    log_x: bool,
    log_y: bool,
) -> tuple[list[float] | None, list[float] | None]:
    """The histogram figure's `(x, y)` axis ranges.

    Extracts the drawn traces once (`_phase_hists`) and delegates to
    `_axis_ranges`. `_make_histogram_figure` calls `_axis_ranges` directly
    with the traces it already extracted.
    """
    return _axis_ranges(_phase_hists(per_phase, kind), log_x=log_x, log_y=log_y)


def _axis_ranges(
    phase_hists: _PhaseHists,
    *,
    log_x: bool,
    log_y: bool,
) -> tuple[list[float] | None, list[float] | None]:
    """`axis_ranges` over already-extracted `(phase, histogram)` traces.

    Both ranges come from the same clip budget: bins/bars holding up to a
    share of the data points may be cut off (x: outermost tail bins,
    `_trimmed_bin_bounds`; y: tallest bars, `_linear_y_range`).

    A tall near-zero peak next to a long thin tail can leave the plot
    nearly empty even after the base trims — the cap chases the peak's
    narrow neighbours while the tail stretches the x-span. So while the
    bars would cover less than `_MIN_FILL_FRACTION` of the plot area, the
    clip share is raised in `_CLIP_SHARE_STEP` increments (clipping more of
    the peak and trimming more of the tail), stopping once the plot is at
    least that full or the share reaches `_MAX_CLIP_SHARE`.

    With **Log y** the y-range is `None` (autorange) and the x-trim sticks
    to the base share — the log scale keeps the bars visible, so the fill
    heuristic doesn't apply. Both ranges are `None` when there's no data.
    """
    density = use_density(log_x)

    def x_range_at(bounds: tuple[int, int] | None) -> list[float] | None:
        return _log_x_range(bounds) if log_x else _linear_x_range(bounds)

    if log_y:
        bounds = _trimmed_bin_bounds(phase_hists, density=density)
        return x_range_at(bounds), None
    share = _BASE_CLIP_SHARE
    while True:
        bounds = _trimmed_bin_bounds(phase_hists, share, density=density)
        y_range = _linear_y_range(phase_hists, density, share)
        if bounds is None or y_range is None:
            return None, None
        if (
            share >= _MAX_CLIP_SHARE
            or _fill_fraction(phase_hists, density, bounds, y_range[1])
            >= _MIN_FILL_FRACTION
        ):
            return x_range_at(bounds), y_range
        share = min(share + _CLIP_SHARE_STEP, _MAX_CLIP_SHARE)


# --- Retaining axis ranges across a Log x / Log y / phase change ------------
#
# The "Retain axes" checkbox freezes the current view instead of auto-fitting
# to the data. Keeping the *same window* across a Log x toggle means
# re-expressing it between the two x-coordinate systems the figure uses — the
# linear value axis and the signed-log bin-index axis — and across a Log y
# toggle means re-expressing the linear y-cap on the log scale. Both edges
# meet at zero, which has no log; the helpers below handle that explicitly.


def _interp(values: list[float], pos: float) -> float:
    """Linear interpolation of `values` at a fractional index, clamped to ends."""
    last = len(values) - 1
    if pos <= 0:
        return values[0]
    if pos >= last:
        return values[last]
    i = int(pos)
    frac = pos - i
    return values[i] * (1.0 - frac) + values[i + 1] * frac


def _inverse_interp(values: list[float], target: float) -> float:
    """Fractional index `p` with `_interp(values, p) == target` (clamped).

    `values` must be strictly increasing.
    """
    last = len(values) - 1
    if target <= values[0]:
        return 0.0
    if target >= values[last]:
        return float(last)
    hi = bisect.bisect_right(values, target)
    lo = hi - 1
    span = values[hi] - values[lo]
    return (lo + (target - values[lo]) / span) if span else float(lo)


def _value_to_bin_coord(value: float) -> float:
    """Continuous bin-index coordinate of a linear value on the signed-log axis.

    Inverse of `_bin_coord_to_value`. A value of `0` lands at the centre of
    the zero band (`ZERO_BIN`); values past +/-1e6 clamp to the end bars.
    """
    return _inverse_interp(_HIST_EDGES, value) - 0.5


def _bin_coord_to_value(coord: float) -> float:
    """Linear value at a continuous bin-index coordinate (inverse of above)."""
    return _interp(_HIST_EDGES, coord + 0.5)


# A converted signed-log x-range never collapses below one bin: a view sitting
# entirely inside the (2e-9-wide) zero band would otherwise map to a
# zero-width range in bin-index space.
_MIN_LOG_X_SPAN: float = 1.0


def _x_range_linear_to_log(x_range: list[float]) -> list[float]:
    """A linear value x-range re-expressed on the signed-log bin-index axis."""
    lo = _value_to_bin_coord(x_range[0])
    hi = _value_to_bin_coord(x_range[1])
    if hi - lo < _MIN_LOG_X_SPAN:
        mid = (lo + hi) / 2.0
        lo, hi = mid - _MIN_LOG_X_SPAN / 2.0, mid + _MIN_LOG_X_SPAN / 2.0
    return [max(-0.5, lo), min(N_BINS - 0.5, hi)]


def _x_range_log_to_linear(x_range: list[float]) -> list[float]:
    """A signed-log bin-index x-range re-expressed on the linear value axis."""
    return [_bin_coord_to_value(x_range[0]), _bin_coord_to_value(x_range[1])]


def _min_positive_height(phase_hists: _PhaseHists, density: bool) -> float | None:
    """Smallest positive bar height across the drawn traces, or `None`."""
    best: float | None = None
    for _, hist in phase_hists:
        for h in trace_heights(hist, density):
            if h > 0.0 and (best is None or h < best):
                best = h
    return best


def _retained_y_range(
    top: float | None, *, log_y: bool, floor: float | None
) -> list[float] | None:
    """A retained linear y-cap re-expressed for the current y-scale.

    Linear: `[0, top]`. Log: `[log10(floor), log10(top)]`, where `floor` is
    the smallest positive bar so every bar stays visible — falling back to
    three decades below `top` when no positive bar is available or it isn't
    below `top`. That floor is the "reasonable near 0" handling: a linear
    bottom of 0 has no log. Returns `None` (Plotly autorange) without a cap.
    """
    if top is None or top <= 0.0:
        return None
    if not log_y:
        return [0.0, top]
    if floor is None or floor <= 0.0 or floor >= top:
        floor = top * 1e-3
    return [math.log10(floor), math.log10(top)]


# Dotted band-edge lines for the dtype-aware under/overflow regions (amber-700,
# reads against any bar color). Only edges that fall within the histogram's
# representable magnitude range (1e-9 .. 1e6) are drawn — fp32's bands, for
# instance, sit off either end of the axis, which correctly reads as "no
# under/overflow risk visible at these magnitudes".
_UNDER_OVER_LINE_COLOR: str = "#b45309"
_HIST_MIN_MAGNITUDE: float = 10.0**LOG10_MIN
_HIST_MAX_MAGNITUDE: float = 10.0**LOG10_MAX


def under_over_line_positions(
    band: tuple[float, float], log_x: bool
) -> list[tuple[float, str]]:
    """In-range band-edge `(x, label)` lines for the subnormal/overflow regions.

    `band` is `(tiny, overflow_edge)` (from `debugger.dtype_band`). Lines are
    placed at `±tiny` (the subnormal edge) and `±overflow_edge` (the
    near-saturation edge), but only when the edge magnitude lies strictly within
    the histogram's `1e-9 .. 1e6` span. The x-coordinate matches the figure's
    axis: a continuous bin-index coordinate on the signed-log axis, the plain
    value on the linear axis. Only the positive-side edge carries the text
    label, to keep the symmetric pair from doubling it.
    """
    tiny, maxv = band
    out: list[tuple[float, str]] = []
    for value, label in (
        (tiny, "subnormal"),
        (-tiny, ""),
        (maxv, "overflow"),
        (-maxv, ""),
    ):
        if _HIST_MIN_MAGNITUDE < abs(value) < _HIST_MAX_MAGNITUDE:
            x = _value_to_bin_coord(value) if log_x else value
            out.append((x, label))
    return out


def _add_under_over_lines(
    fig: go.Figure,
    band: tuple[float, float],
    *,
    log_x: bool,
    n_rows: int,
) -> None:
    """Draw the under/overflow band-edge lines across every subplot row."""
    for x, label in under_over_line_positions(band, log_x):
        for row in range(1, n_rows + 1):
            # Label only the top row's positive-side edge — passing annotation
            # kwargs at all would otherwise create an invisible (text-`None`)
            # annotation that shifts later annotation indices.
            annotation = (
                dict(
                    annotation_text=label,
                    annotation_position="top",
                    annotation_font_size=9,
                    annotation_font_color=_UNDER_OVER_LINE_COLOR,
                )
                if (row == 1 and label)
                else {}
            )
            fig.add_vline(
                x=x,
                row=row,
                col=1,
                line_color=_UNDER_OVER_LINE_COLOR,
                line_width=1,
                line_dash="dot",
                **annotation,
            )


def _make_histogram_figure(
    per_phase: dict[str, LayerStatsSnapshot],
    kind: str,
    title: str,
    *,
    log_x: bool = False,
    log_y: bool = False,
    trace_names: list[str] | None = None,
    override_ranges: tuple[list[float] | None, list[float] | None] | None = None,
    under_over_band: tuple[float, float] | None = None,
) -> tuple[go.Figure, tuple[list[float] | None, list[float] | None]]:
    """Plotly bar chart of the signed-log histogram, one subplot row per phase.

    Returns the figure together with the `(x_range, y_range)` pair applied
    to it (the most expensive part of the build, see `_axis_ranges`), so a
    caller that caches the ranges (`_HistPlot`) doesn't recompute them.

    `trace_names` overrides the default "phase (ep N)" trace/subplot names —
    the per-channel view appends the channel there.

    `kind` selects which of the two histograms on each `LayerStatsSnapshot`
    to plot ("activation" or "gradient"). `per_phase` may be empty (initial
    render before any data has been collected) — the figure is still
    returned, just with no traces.

    Each phase draws in its own stacked subplot row (titled with the phase
    and epoch, tinted with the trace color) rather than overlaying bars on
    shared axes, so one phase never obscures another. The rows share the
    x-axis and, on a linear y-axis, the same capped y-range
    (see `axis_ranges`), keeping the per-phase distributions directly
    comparable.

    `log_x` / `log_y` toggle the value (x) and probability (y) axes between a
    log-based and a linear scale (the "Log x" / "Log y" checkboxes on the
    Watching page — the checkbox alone decides the x-mode). With `log_x`
    off, bars show probability density instead of probabilities (see
    `use_density`).

    This builds the *whole* figure. Routine data refreshes don't call it —
    they restyle the existing figure in place (see `_HistPlot`) so client-side
    state like zoom survives; the figure is only rebuilt when the set of
    phases or the axis scale changes.

    `override_ranges` forces the `(x_range, y_range)` instead of fitting them
    to the data — used by the "Retain axes" toggle to carry the current view
    across a rebuild. The off-view bar blanking still tracks the applied
    x-range, so hovering stays correct.

    `under_over_band` is the `(tiny, overflow_edge)` dtype band (from
    `debugger.dtype_band`) for this stream's dtype. When given, dotted vertical
    lines mark the band edges that fall within the histogram's representable
    magnitude range, so the gradient distribution can be read against the
    dtype's subnormal and overflow thresholds.
    """
    phase_hists = _phase_hists(per_phase, kind)
    if override_ranges is not None:
        x_range, y_range = override_ranges
    else:
        x_range, y_range = _axis_ranges(phase_hists, log_x=log_x, log_y=log_y)
    # Signed-log mode draws at uniform bin indices; the linear value axis
    # draws at the bin centres but blanks the off-view tail bins so Plotly's
    # `hovermode="x"` hit-test stays on the bins on screen (see `_linear_bar_x`).
    x_values = list(range(N_BINS)) if log_x else _linear_bar_x(x_range)
    density = use_density(log_x)
    if density:
        hover = (
            "value %{x:.2e}<br>probability density %{y:.3g}"
            "<br>count %{customdata}<extra></extra>"
        )
    else:
        # Bars sit at bin indices on the signed-log axis; the hover shows
        # the bin's value (via customdata), not the index.
        hover = (
            "value ≈ %{customdata[1]}<br>probability %{y:.3g}"
            "<br>count %{customdata[0]}<extra></extra>"
        )
    names = trace_names or [
        f"{p} (ep {per_phase[p].epoch})" for p, _ in phase_hists
    ]
    fig = make_subplots(
        rows=max(1, len(phase_hists)),
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.06,
        subplot_titles=names or None,
    )
    for i, (phase, hist) in enumerate(phase_hists):
        fig.add_trace(
            go.Bar(
                x=x_values,
                y=trace_heights(hist, density),
                customdata=_hover_customdata(hist, density),
                width=None if log_x else BIN_WIDTHS,
                name=names[i],
                marker_color=phase_color(phase, i),
                opacity=0.85,
                hovertemplate=hover,
            ),
            row=i + 1,
            col=1,
        )
    # One overflow-marker trace per row, drawn after every bar trace so the
    # bars keep indices 0..n-1 (the restyle path and per-channel hover rely on
    # that). `hoverinfo="skip"` keeps them out of the bar hover. Empty when
    # nothing is clipped, but always present so the trace count stays 2n.
    marker_y_top = y_range[1] if (y_range is not None and not log_y) else None
    for i, (xs, ys) in enumerate(
        _overflow_marks(phase_hists, x_values, density, marker_y_top)
    ):
        fig.add_trace(
            go.Scatter(
                x=xs,
                y=ys,
                mode="markers",
                marker=dict(
                    symbol="triangle-up",
                    size=9,
                    color=_OVERFLOW_MARKER_COLOR,
                    line=dict(width=1, color="white"),
                ),
                hoverinfo="skip",
                showlegend=False,
            ),
            row=i + 1,
            col=1,
        )
    # Subplot titles double as the legend: phase name + epoch in the trace
    # color, sitting right above the row they describe.
    for i, annotation in enumerate(fig.layout.annotations):
        annotation.update(
            font=dict(size=11, color=phase_color(phase_hists[i][0], i))
        )
    # `shared_xaxes` only hides the upper rows' tick labels; matching the
    # x-axes proper keeps every row in lock-step when zooming/panning and
    # lets a single `xaxis.range` relayout retarget all rows at once.
    for row in range(2, len(phase_hists) + 1):
        fig.update_xaxes(matches="x", row=row, col=1)
    fig.update_layout(
        title=dict(text=title, x=0.0, font=dict(size=12)),
        bargap=0,
        margin=dict(l=50, r=20, t=40, b=40),
        height=_PLOT_HEIGHT * max(1, len(phase_hists)),
        plot_bgcolor="#f8fafc",
        paper_bgcolor="white",
        showlegend=False,
        # Hover by x-position instead of proximity: a short bar is hoverable
        # from anywhere in its column, which the per-channel sample strip
        # (and reading counts generally) depends on.
        hovermode="x",
    )
    if log_x:
        tick_vals, tick_text = x_tick_layout()
        fig.update_xaxes(
            range=x_range,
            tickvals=tick_vals,
            ticktext=tick_text,
            tickfont=dict(size=9),
            showgrid=False,
            zeroline=False,
        )
    else:
        fig.update_xaxes(
            range=x_range,
            tickfont=dict(size=9),
            exponentformat=AXIS_EXPONENT_FORMAT,
            showgrid=False,
            zeroline=True,
            zerolinecolor="#cbd5e1",
        )
    fig.update_yaxes(
        type="log" if log_y else "linear",
        # The cap is a linear-space range; on a log y-axis `_axis_ranges`
        # returns `None` (Plotly autorange, which shows 100% of the data
        # anyway) since Plotly would misread the range as log10 units.
        range=y_range,
        showgrid=True,
        gridcolor="#e2e8f0",
        tickfont=dict(size=9),
        exponentformat=AXIS_EXPONENT_FORMAT,
        title=dict(
            text="probability density" if density else "probability",
            font=dict(size=10),
        ),
    )
    if under_over_band is not None:
        _add_under_over_lines(
            fig, under_over_band, log_x=log_x, n_rows=max(1, len(phase_hists))
        )
    return fig, (x_range, y_range)
