"""Value-vs-epoch stat series and figures for the GRAPHS view.

Pure functions over the watch snapshots — no UI state. The `/stats` page
builds one figure per tensor kind (activations, gradients) with a fixed
trace set, then restyles the trace arrays in place every refresh so
client-side state (legend selections, zoom) survives. Which stats are
shown is toggled through the Plotly legend.
"""

from __future__ import annotations

import math

import plotly.graph_objects as go

from nansense.ui.histograms import kind_stats
from nansense.watch import LayerStatsSnapshot, TensorStatsSnapshot

# The scalar stats drawn as value traces, in trace order, with their colors
# (hues distinct from the phase colors the histogram view keys on).
EPOCH_STAT_SPECS: tuple[tuple[str, str], ...] = (
    ("mean", "#0ea5e9"),  # sky
    ("std", "#f97316"),  # orange
    ("median", "#22c55e"),  # green
    ("min", "#64748b"),  # slate
    ("max", "#8b5cf6"),  # violet
)

# The activation figure's extra series: dead channels per epoch, drawn on a
# secondary count axis (a count would dwarf small activation values on the
# shared value axis).
DEAD_SERIES: str = "dead channels"
_DEAD_COLOR: str = "#ef4444"  # red

_EPOCH_PLOT_HEIGHT: int = 340


def _stat_point(stats: TensorStatsSnapshot, stat: str) -> float | None:
    """One stat as a plottable point; `None` (a Plotly gap) when undefined.

    An epoch with no samples yields no point at all — without the `n`
    guard its `std` would plot as a flat 0 (the snapshot property's
    empty-stream fallback) while the other stats gap. Non-finite values
    (nan, the ±inf extremes) also gap, which doubles as keeping the JSON
    pushed to the client valid.
    """
    if stats.n == 0:
        return None
    value = float(getattr(stats, stat))
    return value if math.isfinite(value) else None


def epoch_stat_series(
    history: list[LayerStatsSnapshot], kind: str
) -> tuple[list[int], dict[str, list[float | None]]]:
    """Per-stat series over `history`'s epochs for one tensor kind.

    Returns the epochs (x-axis) and a `stat -> values` map in
    `EPOCH_STAT_SPECS` order — the figure's trace order. The activation
    kind appends the dead-channels series, `None` where the count is
    unknown (per-channel tracking off for that epoch's accumulator).
    """
    series: dict[str, list[float | None]] = {
        stat: [_stat_point(kind_stats(s, kind), stat) for s in history]
        for stat, _ in EPOCH_STAT_SPECS
    }
    if kind == "activation":
        series[DEAD_SERIES] = [
            None if (dead := s.activations.dead_channel_count) is None
            else float(dead)
            for s in history
        ]
    return [s.epoch for s in history], series


def weight_stat_series(
    history: list[tuple[int, TensorStatsSnapshot]],
) -> tuple[list[int], dict[str, list[float | None]]]:
    """Per-stat series over one weight tensor's per-epoch samples.

    `history` is `WatchSnapshot.weight_history`'s `[(epoch, stats)]` list
    for a single parameter; the result matches `epoch_stat_series`'s shape
    so both feed the same figure/restyle path (weights have no
    dead-channels series).
    """
    return [epoch for epoch, _ in history], {
        stat: [_stat_point(stats, stat) for _, stats in history]
        for stat, _ in EPOCH_STAT_SPECS
    }


def epoch_axis_dtick(epochs: list[int]) -> int:
    """Integer x-tick spacing so fractional epoch ticks never appear.

    Roughly eight ticks across the span; at least 1, so a short run's
    autoranged axis doesn't label epoch 0.5.
    """
    if not epochs:
        return 1
    return max(1, math.ceil((epochs[-1] - epochs[0]) / 8))


def make_epoch_stats_figure(kind: str, title: str) -> go.Figure:
    """The value-vs-epoch figure for one tensor kind, with empty traces.

    Built once per plot; refreshes restyle the arrays in place (see the
    module docstring). The activation figure carries the dead-channels
    trace on a secondary count axis on the right. Only the mean starts
    enabled — the axes fit the one series most runs care about instead of
    being stretched by the extremes — and the rest wait in the legend as
    `"legendonly"`; the restyle path never writes `visible`, so what the
    user enables sticks across refreshes.
    """
    fig = go.Figure()
    for stat, color in EPOCH_STAT_SPECS:
        fig.add_trace(
            go.Scatter(
                x=[],
                y=[],
                mode="lines+markers",
                name=stat,
                visible=True if stat == "mean" else "legendonly",
                line=dict(color=color, width=2),
                marker=dict(size=6),
            )
        )
    if kind == "activation":
        fig.add_trace(
            go.Scatter(
                x=[],
                y=[],
                mode="lines+markers",
                name=DEAD_SERIES,
                visible="legendonly",
                yaxis="y2",
                line=dict(color=_DEAD_COLOR, width=2, dash="dot"),
                marker=dict(size=6),
            )
        )
    fig.update_layout(
        title=dict(text=title, x=0.0, font=dict(size=12)),
        height=_EPOCH_PLOT_HEIGHT,
        margin=dict(l=50, r=50, t=40, b=40),
        plot_bgcolor="#f8fafc",
        paper_bgcolor="white",
        # One shared hover row per epoch, so the stats read as a column.
        hovermode="x unified",
        legend=dict(font=dict(size=10)),
        xaxis=dict(
            title=dict(text="epoch", font=dict(size=10)),
            tickfont=dict(size=9),
            dtick=1,
            showgrid=False,
            zeroline=False,
        ),
        yaxis=dict(
            title=dict(text="value", font=dict(size=10)),
            tickfont=dict(size=9),
            showgrid=True,
            gridcolor="#e2e8f0",
            zeroline=True,
            zerolinecolor="#cbd5e1",
        ),
    )
    if kind == "activation":
        fig.update_layout(
            yaxis2=dict(
                title=dict(text="dead channels", font=dict(size=10)),
                tickfont=dict(size=9),
                overlaying="y",
                side="right",
                rangemode="tozero",
                showgrid=False,
            )
        )
    return fig
