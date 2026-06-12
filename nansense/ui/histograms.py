"""Histogram math and Plotly figure construction for the watch views.

Pure functions over plotly + snapshot types — no UI state. The public
(non-underscore) names are also the contract `nansense.recording` renders
watch-view frames with, so recorded histograms match the page exactly.
"""

from __future__ import annotations

import html
import math
from collections.abc import Callable

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
    if math.isnan(value):
        return "—"
    if value == 0:
        return "0"
    abs_v = abs(value)
    if abs_v >= 1000 or abs_v < 0.01:
        return f"{value:.2e}"
    return f"{value:.3g}"


# Rows of the per-histogram stats table: label and how to format the value.
_STAT_ROWS: tuple[tuple[str, Callable[[TensorStatsSnapshot], str]], ...] = (
    ("n", lambda s: f"{s.n:,}"),
    ("mean", lambda s: _format_stat(s.mean)),
    ("std", lambda s: _format_stat(s.std)),
    ("median", lambda s: _format_stat(s.median)),
    ("min", lambda s: _format_stat(s.min)),
    ("max", lambda s: _format_stat(s.max)),
)

_STATS_CELL_STYLE: str = "padding:2px 26px 2px 0;text-align:left"

# Light framed card around each stats table so it stands out from the page
# instead of floating as bare text.
_STATS_BOX_STYLE: str = (
    "display:inline-block;background:#f8fafc;border:1px solid #e2e8f0;"
    "border-radius:6px;padding:8px 14px"
)


def _stats_table_html(per_phase: dict[str, LayerStatsSnapshot], kind: str) -> str:
    """Scalar stats as an HTML table: one column per phase, one row per stat.

    The header of each phase column ("train ep 0") is tinted with the phase's
    trace color so it reads against the matching bars in the histogram below,
    and the whole table sits in a light framed box for visibility. Returns a
    plain "no data yet" note while the phase has no samples.
    """
    phases = _phases_with_data(per_phase, kind)
    if not phases:
        return '<span class="text-slate-500">no data yet</span>'
    header = "".join(
        f'<th style="{_STATS_CELL_STYLE};font-weight:700;'
        f"border-bottom:1px solid #e2e8f0;"
        f'color:{phase_color(p, i)}">'
        f"{html.escape(p)} ep {per_phase[p].epoch}</th>"
        for i, p in enumerate(phases)
    )
    rows = "".join(
        f'<tr><td style="{_STATS_CELL_STYLE};color:#64748b">{label}</td>'
        + "".join(
            f'<td style="{_STATS_CELL_STYLE};color:#1e293b">'
            f"{fmt(kind_stats(per_phase[p], kind))}</td>"
            for p in phases
        )
        + "</tr>"
        for label, fmt in _STAT_ROWS
    )
    return (
        f'<div style="{_STATS_BOX_STYLE}">'
        '<table style="border-collapse:collapse">'
        "<thead><tr>"
        f'<th style="border-bottom:1px solid #e2e8f0"></th>{header}'
        "</tr></thead>"
        f"<tbody>{rows}</tbody></table></div>"
    )


# Plot height in px. Doubled from the original 220 so the distributions are
# easier to read.
_PLOT_HEIGHT: int = 440

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
# Hover labels for the signed-log view, where bars sit at plain bin indices:
# each bin's representative value (its geometric midpoint, the same notion
# the median stat uses) instead of the meaningless index.
_BIN_VALUE_LABELS: list[str] = [f"{bin_midpoint(i):.3g}" for i in range(N_BINS)]

# Axis trims may clip bins/bars holding up to this share of the data points
# (see `_trimmed_bin_bounds` / `_linear_y_range`). `axis_ranges` starts at
# the base share and raises it in steps up to the max while the bars would
# fill less than `_MIN_FILL_FRACTION` of the plot area.
_BASE_CLIP_SHARE: float = 0.005
_MAX_CLIP_SHARE: float = 0.05
_CLIP_SHARE_STEP: float = 0.005

# Minimum share of the plot area the bars should cover; below this the clip
# share keeps being raised (up to `_MAX_CLIP_SHARE`).
_MIN_FILL_FRACTION: float = 0.05

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
    per_phase: dict[str, LayerStatsSnapshot],
    kind: str,
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
    for phase in _phases_with_data(per_phase, kind):
        hist = kind_stats(per_phase[phase], kind).hist
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


def kind_stats(layer_snap: LayerStatsSnapshot, kind: str) -> TensorStatsSnapshot:
    """The activation or gradient stats of a layer snapshot, by `kind`."""
    return layer_snap.activations if kind == "activation" else layer_snap.gradients


def _phases_with_data(
    per_phase: dict[str, LayerStatsSnapshot], kind: str
) -> list[str]:
    """Phases that have at least one sample for `kind`, in render order.

    This is exactly the set (and order) of traces `_make_histogram_figure`
    draws, so it doubles as the signature the panel uses to decide whether a
    refresh can restyle in place or must rebuild the figure.
    """
    return [p for p, snap in per_phase.items() if kind_stats(snap, kind).n > 0]


def _trimmed_bin_bounds(
    per_phase: dict[str, LayerStatsSnapshot],
    kind: str,
    clip_share: float = _BASE_CLIP_SHARE,
) -> tuple[int, int] | None:
    """Smallest/largest bin indices after trimming the extreme-tail bins.

    Pooled across the drawn traces, the outermost populated bins are dropped
    greedily — lighter end first — while the dropped bins together hold less
    than `clip_share` of the data points, so the x-range keeps the bins
    holding the rest of the values and a lone outlier value no longer
    stretches the whole value axis. Returns `None` when there's no data.
    """
    counts = [0] * N_BINS
    for phase in _phases_with_data(per_phase, kind):
        for i, count in enumerate(kind_stats(per_phase[phase], kind).hist):
            counts[i] += count
    total = sum(counts)
    if total == 0:
        return None
    lo = next(i for i, c in enumerate(counts) if c > 0)
    hi = next(i for i in range(N_BINS - 1, -1, -1) if counts[i] > 0)
    allowed = clip_share * total
    trimmed = 0
    while lo < hi:
        side = lo if counts[lo] <= counts[hi] else hi
        if trimmed + counts[side] > allowed:
            break
        trimmed += counts[side]
        if side == lo:
            lo += 1
            while counts[lo] == 0:
                lo += 1
        else:
            hi -= 1
            while counts[hi] == 0:
                hi -= 1
    return lo, hi


def _linear_x_range(
    per_phase: dict[str, LayerStatsSnapshot],
    kind: str,
    clip_share: float = _BASE_CLIP_SHARE,
) -> list[float] | None:
    """X-axis range (linear value space) covering the trimmed bins.

    On a linear axis the bars span the full `[-1e6, 1e6]` edge range, almost
    all of it empty. We zoom to the edges of the trimmed bin span
    (`_trimmed_bin_bounds`, plus a little padding) so the bulk of the
    distribution stays legible. Returns `None` (Plotly autorange) when
    there's no data.
    """
    bounds = _trimmed_bin_bounds(per_phase, kind, clip_share)
    if bounds is None:
        return None
    lo = _HIST_EDGES[bounds[0]]
    hi = _HIST_EDGES[bounds[1] + 1]
    pad = (hi - lo) * 0.05 or 1.0
    return [lo - pad, hi + pad]


def _log_x_range(
    per_phase: dict[str, LayerStatsSnapshot],
    kind: str,
    clip_share: float = _BASE_CLIP_SHARE,
) -> list[float] | None:
    """X-axis range (bin-index space) covering the trimmed bins.

    The signed-log view draws bars at integer bin indices, so the range
    brackets the trimmed span (`_trimmed_bin_bounds`) with half-bar margins
    plus a little padding. Returns `None` (Plotly autorange — the full
    211-bin span) when there's no data.
    """
    bounds = _trimmed_bin_bounds(per_phase, kind, clip_share)
    if bounds is None:
        return None
    lo, hi = bounds
    pad = (hi - lo + 1) * 0.05
    return [lo - 0.5 - pad, hi + 0.5 + pad]


def _fill_fraction(
    per_phase: dict[str, LayerStatsSnapshot],
    kind: str,
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
    phases = _phases_with_data(per_phase, kind)
    filled = 0.0
    for phase in phases:
        heights = trace_heights(kind_stats(per_phase[phase], kind).hist, density)
        for i in range(lo, hi + 1):
            width = BIN_WIDTHS[i] if density else 1.0
            filled += width * min(heights[i], y_top)
    return filled / (span * y_top * len(phases))


def axis_ranges(
    per_phase: dict[str, LayerStatsSnapshot],
    kind: str,
    *,
    log_x: bool,
    log_y: bool,
) -> tuple[list[float] | None, list[float] | None]:
    """The histogram figure's `(x, y)` axis ranges.

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

    def x_range_at(share: float) -> list[float] | None:
        return (
            _log_x_range(per_phase, kind, share)
            if log_x
            else _linear_x_range(per_phase, kind, share)
        )

    if log_y:
        return x_range_at(_BASE_CLIP_SHARE), None
    share = _BASE_CLIP_SHARE
    while True:
        bounds = _trimmed_bin_bounds(per_phase, kind, share)
        y_range = _linear_y_range(per_phase, kind, density, share)
        if bounds is None or y_range is None:
            return None, None
        if (
            share >= _MAX_CLIP_SHARE
            or _fill_fraction(per_phase, kind, density, bounds, y_range[1])
            >= _MIN_FILL_FRACTION
        ):
            return x_range_at(share), y_range
        share = min(share + _CLIP_SHARE_STEP, _MAX_CLIP_SHARE)


def _make_histogram_figure(
    per_phase: dict[str, LayerStatsSnapshot],
    kind: str,
    title: str,
    *,
    log_x: bool = False,
    log_y: bool = False,
    trace_names: list[str] | None = None,
) -> go.Figure:
    """Plotly bar chart of the signed-log histogram, one subplot row per phase.

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
    """
    x_values = list(range(N_BINS)) if log_x else BIN_CENTERS
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
    phases = _phases_with_data(per_phase, kind)
    names = trace_names or [f"{p} (ep {per_phase[p].epoch})" for p in phases]
    fig = make_subplots(
        rows=max(1, len(phases)),
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.06,
        subplot_titles=names or None,
    )
    for i, phase in enumerate(phases):
        stats = kind_stats(per_phase[phase], kind)
        fig.add_trace(
            go.Bar(
                x=x_values,
                y=trace_heights(stats.hist, density),
                customdata=_hover_customdata(stats.hist, density),
                width=None if log_x else BIN_WIDTHS,
                name=names[i],
                marker_color=phase_color(phase, i),
                opacity=0.85,
                hovertemplate=hover,
            ),
            row=i + 1,
            col=1,
        )
    # Subplot titles double as the legend: phase name + epoch in the trace
    # color, sitting right above the row they describe.
    for i, annotation in enumerate(fig.layout.annotations):
        annotation.update(font=dict(size=11, color=phase_color(phases[i], i)))
    # `shared_xaxes` only hides the upper rows' tick labels; matching the
    # x-axes proper keeps every row in lock-step when zooming/panning and
    # lets a single `xaxis.range` relayout retarget all rows at once.
    for row in range(2, len(phases) + 1):
        fig.update_xaxes(matches="x", row=row, col=1)
    fig.update_layout(
        title=dict(text=title, x=0.0, font=dict(size=12)),
        bargap=0,
        margin=dict(l=50, r=20, t=40, b=40),
        height=_PLOT_HEIGHT * max(1, len(phases)),
        plot_bgcolor="#f8fafc",
        paper_bgcolor="white",
        showlegend=False,
        # Hover by x-position instead of proximity: a short bar is hoverable
        # from anywhere in its column, which the per-channel sample strip
        # (and reading counts generally) depends on.
        hovermode="x",
    )
    x_range, y_range = axis_ranges(per_phase, kind, log_x=log_x, log_y=log_y)
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
            showgrid=False,
            zeroline=True,
            zerolinecolor="#cbd5e1",
        )
    fig.update_yaxes(
        type="log" if log_y else "linear",
        # The cap is a linear-space range; on a log y-axis `axis_ranges`
        # returns `None` (Plotly autorange, which shows 100% of the data
        # anyway) since Plotly would misread the range as log10 units.
        range=y_range,
        showgrid=True,
        gridcolor="#e2e8f0",
        tickfont=dict(size=9),
        title=dict(
            text="probability density" if density else "probability",
            font=dict(size=10),
        ),
    )
    return fig
