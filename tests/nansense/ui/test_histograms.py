"""Tests for histogram figures, axis ranges, density, and stats tables in nansense.ui.histograms."""

from __future__ import annotations

import math
from dataclasses import replace

import pytest
import torch

from nansense.ui.histograms import (
    BIN_CENTERS,
    BIN_WIDTHS,
    _HIST_EDGES,
    _PLOT_HEIGHT,
    axis_ranges,
    _bin_coord_to_value,
    dead_channels,
    _fill_fraction,
    _linear_bar_x,
    _linear_x_range,
    _linear_y_range,
    _log_x_range,
    _make_histogram_figure,
    _min_positive_height,
    _OVERFLOW_MARKER_Y_FRAC,
    overflow_marks,
    phase_color,
    _phase_hists,
    _phases_with_data,
    _probabilities,
    _probability_densities,
    _retained_y_range,
    _stats_table_html,
    _trimmed_bin_bounds,
    under_over_line_positions,
    _value_to_bin_coord,
    _x_range_linear_to_log,
    _x_range_log_to_linear,
    use_density,
)
from nansense.watch import (
    BINS_PER_DECADE,
    N_BINS,
    ZERO_BIN,
    LayerStatsSnapshot,
    TensorStatsSnapshot,
    bin_midpoint,
)
from tests.nansense.helpers import _layer_snap, live_hist


# --- Watching histogram: trace structure / restyle signature --------------


def test_phases_with_data_in_render_order() -> None:
    per_phase = {"train": _layer_snap("train"), "val": _layer_snap("val")}
    assert _phases_with_data(per_phase, "activation") == ["train", "val"]


def test_phases_with_data_skips_empty_phases() -> None:
    per_phase = {"train": _layer_snap("train", n=5), "val": _layer_snap("val", n=0)}
    assert _phases_with_data(per_phase, "activation") == ["train"]


def test_histogram_traces_match_phases_with_data() -> None:
    per_phase = {"train": _layer_snap("train"), "val": _layer_snap("val", n=0)}
    fig, _ = _make_histogram_figure(per_phase, "activation", "activations")
    # Only phases with data get a trace, in order; all visible by default so a
    # rebuild never silently hides a series. (Bar traces only — each row also
    # carries an overflow-marker scatter trace.)
    bars = [t for t in fig.data if t.type == "bar"]
    assert [t.name for t in bars] == ["train (ep 0)"]
    assert all(t.visible in (True, None) for t in bars)


# --- Watching histogram: log/linear axis toggles (task 1) ------------------


@pytest.mark.parametrize("log_y", [True, False])
def test_histogram_log_y_toggles_y_axis_scale(log_y: bool) -> None:
    per_phase = {"train": _layer_snap("train")}
    fig, _ = _make_histogram_figure(
        per_phase, "activation", "activations", log_y=log_y
    )
    assert fig.layout.yaxis.type == ("log" if log_y else "linear")


def test_histogram_log_x_uses_bin_indices() -> None:
    per_phase = {"train": _layer_snap("train")}
    fig, _ = _make_histogram_figure(
        per_phase, "activation", "activations", log_x=True
    )
    # Log mode: evenly spaced bin indices, default (uniform) bar width.
    assert fig.data[0].x[0] == 0
    assert fig.data[0].width is None


def test_histogram_linear_x_uses_value_positions_and_widths() -> None:
    # A populated far-negative bin so its centre is genuinely negative and in
    # view (the all-zero-band default would zoom to a tiny range around 0).
    per_phase = {"train": _layer_snap("train", hist={ZERO_BIN - 40: 100})}
    fig, _ = _make_histogram_figure(
        per_phase, "activation", "activations", log_x=False
    )
    # Linear mode: in-view bars sit at their true bin centres (not indices),
    # with per-bin widths covering every bin slot.
    assert fig.data[0].x[ZERO_BIN - 40] == BIN_CENTERS[ZERO_BIN - 40] < 0
    assert isinstance(fig.data[0].width, tuple)
    assert len(fig.data[0].width) == N_BINS


def test_linear_x_range_none_when_empty() -> None:
    assert _linear_x_range(_trimmed_bin_bounds([])) is None


def test_linear_x_range_brackets_populated_bins() -> None:
    # Counts only in the zero band → a tight range straddling zero.
    per_phase = {"train": _layer_snap("train", n=7)}
    rng = _linear_x_range(
        _trimmed_bin_bounds(_phase_hists(per_phase, "activation"))
    )
    assert rng is not None
    lo, hi = rng
    assert lo < 0 < hi
    assert hi - lo < 1.0  # zoomed in, not the full +/-1e6 span


# --- Watching histogram: linear x-axis hover (off-view bins blanked) --------


def test_linear_bar_x_keeps_all_slots_blanking_off_view_bins() -> None:
    # A range covering only the central bins: bins on screen keep their true
    # centre, every other slot is NaN — but the array is still 211 long so a
    # bar's index still equals its bin index (the sample hover relies on that).
    rng = _linear_x_range((ZERO_BIN - 2, ZERO_BIN + 2))
    assert rng is not None
    x = _linear_bar_x(rng)
    assert len(x) == N_BINS
    assert x[ZERO_BIN] == BIN_CENTERS[ZERO_BIN]
    assert x[ZERO_BIN + 2] == BIN_CENTERS[ZERO_BIN + 2]
    # The far extreme-tail bins (centres ~±1e6) are the ones whose presence
    # breaks Plotly's zoomed-in `hovermode="x"` hit-test; they're blanked.
    assert math.isnan(x[0])
    assert math.isnan(x[N_BINS - 1])


def test_linear_bar_x_blanks_nothing_without_a_range() -> None:
    # No data → autorange (the full span), so every bin keeps its centre.
    assert _linear_bar_x(None) == BIN_CENTERS


def test_histogram_linear_blanks_far_tail_bins_for_hover() -> None:
    # A tiny O(1e-3) gradient-like distribution: the visible range is a narrow
    # window, so the far-flung extreme-tail bins must be blanked from the drawn
    # x (else Plotly's bar hover locks onto an off-screen empty bar, count 0).
    hist = {ZERO_BIN + 40: 100, ZERO_BIN - 40: 100}
    per_phase = {"train": _layer_snap("train", hist=hist)}
    fig, (x_range, _) = _make_histogram_figure(
        per_phase, "activation", "activations", log_x=False
    )
    x = fig.data[0].x
    assert x_range is not None
    # Populated, in-view bins keep their true centre; out-of-view bins (here
    # the extreme tails) are NaN. Every kept centre lies within the range.
    assert x[ZERO_BIN + 40] == BIN_CENTERS[ZERO_BIN + 40]
    assert math.isnan(x[0]) and math.isnan(x[N_BINS - 1])
    kept = [c for c in x if not math.isnan(c)]
    assert kept  # something is still drawn
    assert all(x_range[0] <= c <= x_range[1] for c in kept)


def test_histogram_log_x_x_positions_are_plain_indices() -> None:
    # The signed-log axis is unaffected by the blanking — bars stay at uniform
    # bin indices (no NaN), since that mode never spans the huge value domain.
    hist = {ZERO_BIN + 40: 100, ZERO_BIN - 40: 100}
    per_phase = {"train": _layer_snap("train", hist=hist)}
    fig, _ = _make_histogram_figure(
        per_phase, "activation", "activations", log_x=True
    )
    assert list(fig.data[0].x) == list(range(N_BINS))


# --- Watching histogram: x-range tail trimming ------------------------------


def test_trimmed_bin_bounds_drops_sparse_outlier_tail() -> None:
    # The 5-count outlier bin holds < 0.5% of the points, so the bounds
    # shrink to the bulk bin.
    hist = {ZERO_BIN + 10: 10_000, ZERO_BIN + 80: 5}
    per_phase = {"train": _layer_snap("train", hist=hist)}
    assert _trimmed_bin_bounds(_phase_hists(per_phase, "activation")) == (
        ZERO_BIN + 10,
        ZERO_BIN + 10,
    )


def test_trimmed_bin_bounds_keeps_heavy_tails() -> None:
    # Both bins hold 50% of the points — trimming either would drop far more
    # than the 0.5% budget, so the bounds cover them both.
    hist = {ZERO_BIN + 10: 500, ZERO_BIN + 80: 500}
    per_phase = {"train": _layer_snap("train", hist=hist)}
    assert _trimmed_bin_bounds(_phase_hists(per_phase, "activation")) == (
        ZERO_BIN + 10,
        ZERO_BIN + 80,
    )


def test_trimmed_bin_bounds_budget_spans_both_tails() -> None:
    # 3 + 4 outlier points (0.07% combined) trim away on both sides.
    hist = {ZERO_BIN - 40: 3, ZERO_BIN + 10: 10_000, ZERO_BIN + 40: 4}
    per_phase = {"train": _layer_snap("train", hist=hist)}
    assert _trimmed_bin_bounds(_phase_hists(per_phase, "activation")) == (
        ZERO_BIN + 10,
        ZERO_BIN + 10,
    )


def test_trimmed_bin_bounds_stops_when_budget_runs_out() -> None:
    # Trimming both tails (30 + 40 = 70 of 10070) would exceed the 0.5%
    # budget (~50 points), so only the lighter tail is dropped.
    hist = {ZERO_BIN - 40: 40, ZERO_BIN + 10: 10_000, ZERO_BIN + 40: 30}
    per_phase = {"train": _layer_snap("train", hist=hist)}
    assert _trimmed_bin_bounds(_phase_hists(per_phase, "activation")) == (
        ZERO_BIN - 40,
        ZERO_BIN + 10,
    )


def test_trimmed_bin_bounds_pools_phases() -> None:
    # val's lone outlier bin holds < 0.5% of the pooled points, so it trims
    # even though it's all the data val has in that bin.
    per_phase = {
        "train": _layer_snap("train", hist={ZERO_BIN + 10: 10_000}),
        "val": _layer_snap("val", hist={ZERO_BIN + 80: 5}),
    }
    assert _trimmed_bin_bounds(_phase_hists(per_phase, "activation")) == (
        ZERO_BIN + 10,
        ZERO_BIN + 10,
    )


def test_trimmed_bin_bounds_none_when_empty() -> None:
    assert _trimmed_bin_bounds([]) is None


def test_linear_x_range_excludes_sparse_outlier_tail() -> None:
    hist = {ZERO_BIN + 10: 10_000, ZERO_BIN + 80: 5}
    per_phase = {"train": _layer_snap("train", hist=hist)}
    rng = _linear_x_range(
        _trimmed_bin_bounds(_phase_hists(per_phase, "activation"))
    )
    assert rng is not None
    assert rng[1] < _HIST_EDGES[ZERO_BIN + 80]  # the outlier bin is outside


def test_log_x_range_brackets_trimmed_bins() -> None:
    hist = {ZERO_BIN + 10: 10_000, ZERO_BIN + 80: 5}
    per_phase = {"train": _layer_snap("train", hist=hist)}
    rng = _log_x_range(
        _trimmed_bin_bounds(_phase_hists(per_phase, "activation"))
    )
    assert rng is not None
    assert rng[0] < ZERO_BIN + 10 < rng[1]
    assert rng[1] < ZERO_BIN + 80  # the outlier bin is outside
    assert _log_x_range(_trimmed_bin_bounds([])) is None


def test_histogram_log_x_zooms_to_trimmed_bins() -> None:
    hist = {ZERO_BIN + 10: 10_000, ZERO_BIN + 80: 5}
    per_phase = {"train": _layer_snap("train", hist=hist)}
    fig, _ = _make_histogram_figure(
        per_phase, "activation", "activations", log_x=True
    )
    expected = _log_x_range(
        _trimmed_bin_bounds(_phase_hists(per_phase, "activation"))
    )
    assert expected is not None
    assert tuple(fig.layout.xaxis.range) == tuple(expected)


# --- Watching histogram: adaptive clip share (fill heuristic) ---------------


def test_fill_fraction_full_for_single_bar() -> None:
    # One bar spanning the whole x-range and reaching the y-top covers the
    # entire plot.
    per_phase = {"train": _layer_snap("train", hist={ZERO_BIN + 60: 100})}
    heights = _probability_densities(live_hist(per_phase["train"].activations))
    bounds = (ZERO_BIN + 60, ZERO_BIN + 60)
    frac = _fill_fraction(
        _phase_hists(per_phase, "activation"),
        True,
        bounds,
        heights[ZERO_BIN + 60],
    )
    assert frac == pytest.approx(1.0)


def test_axis_ranges_keep_base_share_when_plot_is_full() -> None:
    # A compact distribution fills the plot fine at the base budget, so the
    # adaptive loop leaves the base ranges untouched.
    hist = {ZERO_BIN + 60 + i: 100 for i in range(5)}
    per_phase = {"train": _layer_snap("train", hist=hist)}
    hists = _phase_hists(per_phase, "activation")
    x_range, y_range = axis_ranges(
        per_phase, "activation", log_x=False, log_y=False
    )
    assert x_range == _linear_x_range(_trimmed_bin_bounds(hists))
    assert y_range == _linear_y_range(hists, density=True)


def test_axis_ranges_raise_clip_share_when_plot_nearly_empty() -> None:
    # A huge exact-zero peak flanked by tall narrow near-zero bars plus a
    # long thin tail: at the base 0.5% budget the y-cap chases the narrow
    # near-zero bars and the tail stretches the x-span, leaving the bars
    # covering almost none of the plot. The budget then rises (up to 50%),
    # clipping the near-zero bars and trimming more of the tail.
    hist = {ZERO_BIN: 50_000, ZERO_BIN + 1: 600, ZERO_BIN + 2: 500}
    hist.update({ZERO_BIN + 60 + i: 250 for i in range(40)})
    per_phase = {"train": _layer_snap("train", hist=hist)}
    hists = _phase_hists(per_phase, "activation")
    base_x = _linear_x_range(_trimmed_bin_bounds(hists))
    base_y = _linear_y_range(hists, density=True)
    x_range, y_range = axis_ranges(
        per_phase, "activation", log_x=False, log_y=False
    )
    assert base_x is not None and base_y is not None
    assert x_range is not None and y_range is not None
    assert y_range[1] < base_y[1]  # cap dropped below the near-zero bars
    assert x_range[1] < base_x[1]  # tail trimmed harder


def test_axis_ranges_zoom_into_multi_decade_gradient_spike() -> None:
    # Activation-gradient magnitudes often spread near-uniformly over many
    # decades (equal counts per signed-log bin, here 1e-9..1e-4 on each
    # side). On a linear value axis the base trim leaves a hairline spike at
    # zero — the trimmed span is dominated by the outermost decade while
    # most points sit orders of magnitude closer to zero. The adaptive loop
    # must keep raising the clip budget until the bulk is readable.
    hist = {
        ZERO_BIN + offset * sign: 1_000
        for offset in range(1, 5 * BINS_PER_DECADE + 1)
        for sign in (1, -1)
    }
    per_phase = {"train": _layer_snap("train", hist=hist)}
    hists = _phase_hists(per_phase, "activation")
    base_x = _linear_x_range(_trimmed_bin_bounds(hists))
    x_range, y_range = axis_ranges(
        per_phase, "activation", log_x=False, log_y=False
    )
    assert base_x is not None and x_range is not None and y_range is not None
    # Zoomed an order of magnitude past the base trim, not just nudged.
    assert x_range[1] - x_range[0] < (base_x[1] - base_x[0]) / 10


def test_axis_ranges_log_y_keeps_base_trim_and_autorange() -> None:
    # With Log y everything is visible on the log scale, so the y-range
    # autoranges and the x-trim sticks to the base budget.
    hist = {ZERO_BIN: 50_000, ZERO_BIN + 60: 5}
    per_phase = {"train": _layer_snap("train", hist=hist)}
    x_range, y_range = axis_ranges(
        per_phase, "activation", log_x=False, log_y=True
    )
    assert y_range is None
    assert x_range == _linear_x_range(
        _trimmed_bin_bounds(_phase_hists(per_phase, "activation"))
    )


def test_axis_ranges_none_when_empty() -> None:
    assert axis_ranges({}, "activation", log_x=False, log_y=False) == (
        None,
        None,
    )


# --- Watching histogram: the Log x checkbox alone picks the x-mode ----------


def _multi_decade_snap() -> dict[str, LayerStatsSnapshot]:
    # Mass spread evenly over four decades on both signs (like real gradient
    # distributions).
    hist = {ZERO_BIN + s * i: 1000 for i in range(1, 29) for s in (1, -1)}
    hist[ZERO_BIN] = 2000
    return {"train": _layer_snap("train", hist=hist)}


def test_histogram_stays_linear_even_for_multi_decade_distribution() -> None:
    # The Log x checkbox is authoritative: with it off, even a gradient-like
    # multi-decade distribution renders on the linear value axis (no silent
    # signed-log fallback).
    fig, _ = _make_histogram_figure(
        _multi_decade_snap(), "activation", "activations", log_x=False
    )
    assert fig.data[0].width is not None  # linear per-bin widths
    assert fig.layout.yaxis.title.text == "probability density"
    assert fig.layout.xaxis.title.text is None


def test_histogram_compact_distribution_stays_linear() -> None:
    per_phase = {
        "train": _layer_snap(
            "train", hist={ZERO_BIN + 60 + i: 100 for i in range(5)}
        )
    }
    fig, _ = _make_histogram_figure(per_phase, "activation", "activations")
    assert fig.layout.yaxis.title.text == "probability density"
    assert fig.layout.xaxis.title.text is None


# --- Watching histogram: density mode (linear value axis) ------------------


def test_histogram_defaults_to_linear_density_mode() -> None:
    # Log x / Log y start unchecked, so the no-args figure is the density view.
    per_phase = {"train": _layer_snap("train")}
    fig, _ = _make_histogram_figure(per_phase, "activation", "activations")
    assert fig.layout.yaxis.type == "linear"
    assert fig.layout.yaxis.title.text == "probability density"


@pytest.mark.parametrize("log_x, expected", [(True, False), (False, True)])
def test_use_density_depends_only_on_log_x(log_x: bool, expected: bool) -> None:
    assert use_density(log_x) is expected


def test_probabilities_normalize_counts() -> None:
    hist = [0] * N_BINS
    hist[ZERO_BIN] = 4
    hist[ZERO_BIN + 10] = 6
    probs = _probabilities(tuple(hist))
    assert probs[ZERO_BIN] == pytest.approx(0.4)
    assert probs[ZERO_BIN + 10] == pytest.approx(0.6)
    assert sum(probs) == pytest.approx(1.0)


def test_probability_densities_normalize_by_count_and_width() -> None:
    hist = [0] * N_BINS
    hist[ZERO_BIN] = 4
    hist[ZERO_BIN + 10] = 6
    heights = _probability_densities(tuple(hist))
    assert heights[ZERO_BIN] == pytest.approx(0.4 / BIN_WIDTHS[ZERO_BIN])
    assert heights[ZERO_BIN + 10] == pytest.approx(0.6 / BIN_WIDTHS[ZERO_BIN + 10])
    assert heights[0] == 0
    # Bar areas integrate to 1 — it's a probability density.
    assert sum(h * w for h, w in zip(heights, BIN_WIDTHS)) == pytest.approx(1.0)


def test_probability_helpers_handle_empty_hist() -> None:
    empty = (0,) * N_BINS
    assert _probabilities(empty) == [0.0] * N_BINS
    assert _probability_densities(empty) == [0.0] * N_BINS


def test_linear_y_range_clips_bars_holding_under_coverage() -> None:
    # The zero-band spike holds 5 of 10005 points (< 0.5%), so it may clip;
    # the bulk bar holds the other 99.95% and must stay fully visible.
    hist = {ZERO_BIN: 5, ZERO_BIN + 50: 10_000}
    per_phase = {"train": _layer_snap("train", hist=hist)}
    heights = _probability_densities(live_hist(per_phase["train"].activations))
    rng = _linear_y_range(_phase_hists(per_phase, "activation"), density=True)
    assert rng is not None
    assert rng[0] == 0.0
    assert rng[1] == pytest.approx(heights[ZERO_BIN + 50] * 1.05)
    assert rng[1] < heights[ZERO_BIN]  # the sparse spike does clip


def test_linear_y_range_keeps_tall_bars_that_do_not_dominate() -> None:
    # Two adjacent narrow bins of similar height (ratio ~1.39, under the 5x
    # dominance cutoff) hold nearly all the points: neither the dominance
    # rule nor the 0.5% clip budget applies, so the cap reaches the tallest.
    hist = {ZERO_BIN + 1: 5_000, ZERO_BIN + 2: 5_000, ZERO_BIN + 50: 100}
    per_phase = {"train": _layer_snap("train", hist=hist)}
    heights = _probability_densities(live_hist(per_phase["train"].activations))
    rng = _linear_y_range(_phase_hists(per_phase, "activation"), density=True)
    assert rng is not None
    assert rng[1] == pytest.approx(max(heights) * 1.05)


def test_linear_y_range_excludes_dominant_spike_despite_count() -> None:
    # The zero-band spike holds half the points but towers >5x over the
    # runner-up, so it never anchors the scale; the cap lands on the bulk.
    hist = {ZERO_BIN: 10_000, ZERO_BIN + 50: 9_000, ZERO_BIN + 51: 1_000}
    per_phase = {"train": _layer_snap("train", hist=hist)}
    heights = _probability_densities(live_hist(per_phase["train"].activations))
    rng = _linear_y_range(_phase_hists(per_phase, "activation"), density=True)
    assert rng is not None
    assert rng[1] == pytest.approx(heights[ZERO_BIN + 50] * 1.05)
    assert rng[1] < heights[ZERO_BIN]  # the dominant spike clips


def test_linear_y_range_dominance_is_per_phase() -> None:
    # Each phase's own zero spike is dominant within that phase and is
    # dropped, even though pooled the two spikes are close in height.
    per_phase = {
        "train": _layer_snap("train", hist={ZERO_BIN: 6_000, ZERO_BIN + 30: 4_000}),
        "val": _layer_snap("val", hist={ZERO_BIN: 5_000, ZERO_BIN + 40: 5_000}),
    }
    train_heights = _probability_densities(live_hist(per_phase["train"].activations))
    val_heights = _probability_densities(live_hist(per_phase["val"].activations))
    rng = _linear_y_range(_phase_hists(per_phase, "activation"), density=True)
    assert rng is not None
    expected_cap = max(train_heights[ZERO_BIN + 30], val_heights[ZERO_BIN + 40])
    assert rng[1] == pytest.approx(expected_cap * 1.05)
    assert rng[1] < min(train_heights[ZERO_BIN], val_heights[ZERO_BIN])


def test_linear_y_range_pools_phases_weighted_by_count() -> None:
    # The clip budget is 0.5% of the pooled points across both traces:
    # train's sparse zero spike clips, and the cap lands on the tallest
    # bulk bar of either phase.
    per_phase = {
        "train": _layer_snap("train", hist={ZERO_BIN: 5, ZERO_BIN + 30: 5_000}),
        "val": _layer_snap("val", hist={ZERO_BIN + 40: 5_000}),
    }
    train_heights = _probability_densities(live_hist(per_phase["train"].activations))
    val_heights = _probability_densities(live_hist(per_phase["val"].activations))
    rng = _linear_y_range(_phase_hists(per_phase, "activation"), density=True)
    assert rng is not None
    expected_cap = max(train_heights[ZERO_BIN + 30], val_heights[ZERO_BIN + 40])
    assert rng[1] == pytest.approx(expected_cap * 1.05)
    assert rng[1] < train_heights[ZERO_BIN]


def test_linear_y_range_none_when_empty() -> None:
    assert _linear_y_range([], density=True) is None


# --- Watching histogram: overflow markers for bars clipped by the y-cap -----


# A dominant zero-band spike (dropped from the y-scale) flanked by a few
# ordinary bars: the spike towers far above the cap, the rest sit under it.
_CLIPPED_SPIKE_HIST: dict[int, int] = {
    ZERO_BIN: 10_000,
    **{ZERO_BIN + 10 + i: 100 for i in range(4)},
}


def _clipped_spike_per_phase() -> dict[str, LayerStatsSnapshot]:
    return {"train": _layer_snap("train", hist=_CLIPPED_SPIKE_HIST)}


def test_overflow_marks_flag_only_bars_above_the_cap() -> None:
    hists = _phase_hists(_clipped_spike_per_phase(), "activation")
    y_range = _linear_y_range(hists, density=True)
    assert y_range is not None
    y_top = y_range[1]
    (xs, ys), = overflow_marks(hists, list(range(N_BINS)), True, y_top)
    # Only the spike overflows the cap; the ordinary bars stay under it.
    assert xs == [float(ZERO_BIN)]
    # The marker sits just inside the top edge so the whole glyph is visible.
    assert ys == [pytest.approx(y_top * _OVERFLOW_MARKER_Y_FRAC)]


def test_overflow_marks_empty_without_a_cap() -> None:
    # No cap (log-y autorange) → nothing is clipped, so no markers.
    hists = _phase_hists(_clipped_spike_per_phase(), "activation")
    assert overflow_marks(hists, list(range(N_BINS)), True, None) == [([], [])]


def test_overflow_marks_skip_blanked_off_view_bins() -> None:
    # A clipped bin whose x is blanked (off-view, NaN) gets no marker — the
    # marks inherit the linear axis's off-view blanking from `x_values`.
    hists = _phase_hists(_clipped_spike_per_phase(), "activation")
    y_range = _linear_y_range(hists, density=True)
    assert y_range is not None
    (xs, ys), = overflow_marks(hists, [math.nan] * N_BINS, True, y_range[1])
    assert xs == [] and ys == []


def test_make_histogram_figure_adds_overflow_marker_trace() -> None:
    fig, (_, y_range) = _make_histogram_figure(
        _clipped_spike_per_phase(), "activation", "a"
    )
    # One bar trace plus one (always-present) overflow-marker trace per phase.
    assert len(fig.data) == 2
    marker = fig.data[1]
    assert marker.mode == "markers"
    assert marker.marker.symbol == "triangle-up"
    assert len(marker.x) >= 1  # the spike is flagged
    assert y_range is not None
    assert all(
        y == pytest.approx(y_range[1] * _OVERFLOW_MARKER_Y_FRAC)
        for y in marker.y
    )


def test_make_histogram_figure_no_overflow_markers_on_log_y() -> None:
    # Log y autoranges (nothing clipped), so the marker trace stays empty.
    fig, _ = _make_histogram_figure(
        _clipped_spike_per_phase(), "activation", "a", log_y=True
    )
    assert len(fig.data) == 2
    assert len(fig.data[1].x) == 0


def test_make_histogram_figure_no_markers_without_clipping() -> None:
    # A compact distribution with no bar above the cap → empty marker trace.
    per_phase = {"train": _layer_snap("train", hist={ZERO_BIN + 60: 100})}
    fig, _ = _make_histogram_figure(per_phase, "activation", "a")
    assert len(fig.data[1].x) == 0


def test_histogram_density_mode_plots_probability_density_with_capped_axis() -> None:
    hist = {ZERO_BIN: 50, ZERO_BIN + 10: 7}
    per_phase = {"train": _layer_snap("train", hist=hist)}
    fig, _ = _make_histogram_figure(
        per_phase, "activation", "activations", log_x=False, log_y=False
    )
    trace = fig.data[0]
    expected = _probability_densities(live_hist(per_phase["train"].activations))
    assert list(trace.y) == pytest.approx(expected)
    # Raw counts ride along for the hover text.
    assert trace.customdata[ZERO_BIN] == 50
    assert fig.layout.yaxis.title.text == "probability density"
    expected_range = _linear_y_range(
        _phase_hists(per_phase, "activation"), density=True
    )
    assert expected_range is not None
    assert tuple(fig.layout.yaxis.range) == tuple(expected_range)


def test_histogram_log_y_keeps_density_but_drops_range_cap() -> None:
    # Log y alone doesn't switch what's measured: bars stay probability
    # densities, but the cap (a linear-space range) gives way to autorange
    # so all of the data is visible on the log scale.
    per_phase = {"train": _layer_snap("train", n=9)}
    fig, _ = _make_histogram_figure(
        per_phase, "activation", "activations", log_x=False, log_y=True
    )
    expected = _probability_densities(live_hist(per_phase["train"].activations))
    assert list(fig.data[0].y) == pytest.approx(expected)
    assert fig.layout.yaxis.title.text == "probability density"
    assert fig.layout.yaxis.range is None


@pytest.mark.parametrize("log_y", [True, False])
def test_histogram_log_x_plots_probabilities(log_y: bool) -> None:
    per_phase = {"train": _layer_snap("train", n=9)}
    fig, _ = _make_histogram_figure(
        per_phase, "activation", "activations", log_x=True, log_y=log_y
    )
    assert fig.data[0].y[ZERO_BIN] == pytest.approx(1.0)
    assert tuple(fig.data[0].customdata[ZERO_BIN]) == (9, "0")
    assert fig.layout.yaxis.title.text == "probability"
    if log_y:
        assert fig.layout.yaxis.range is None
    else:
        # A linear probability axis is capped too (a single bar holding all
        # the points is its own scale anchor here).
        assert tuple(fig.layout.yaxis.range) == pytest.approx((0.0, 1.05))


def test_histogram_log_x_linear_y_excludes_dominant_probability_spike() -> None:
    # The dominant zero-band bar is excluded from the probability scale just
    # like in density mode: the cap lands on the tallest non-dominant bar.
    hist = {ZERO_BIN: 5_000, ZERO_BIN + 50: 600, ZERO_BIN + 51: 500}
    per_phase = {"train": _layer_snap("train", hist=hist)}
    fig, _ = _make_histogram_figure(
        per_phase, "activation", "activations", log_x=True, log_y=False
    )
    probs = _probabilities(live_hist(per_phase["train"].activations))
    assert tuple(fig.layout.yaxis.range) == pytest.approx(
        (0.0, probs[ZERO_BIN + 50] * 1.05)
    )
    assert fig.layout.yaxis.range[1] < probs[ZERO_BIN]


# --- Watching histogram: one subplot row per phase ---------------------------


def test_histogram_one_subplot_row_per_phase() -> None:
    per_phase = {"train": _layer_snap("train"), "val": _layer_snap("val", epoch=1)}
    fig, _ = _make_histogram_figure(per_phase, "activation", "activations")
    # Bar traces only — each row also carries an overflow-marker scatter trace,
    # added after all bars so the bars keep the leading indices.
    bars = [t for t in fig.data if t.type == "bar"]
    assert [t.name for t in bars] == ["train (ep 0)", "val (ep 1)"]
    # Each phase draws alone in its own stacked row: full opacity, no
    # overlay, separate y-axes.
    assert [t.opacity for t in bars] == [0.85, 0.85]
    assert bars[0].yaxis == "y"
    assert bars[1].yaxis == "y2"
    # Subplot titles carry phase + epoch, so the legend is dropped.
    assert [a.text for a in fig.layout.annotations] == ["train (ep 0)", "val (ep 1)"]
    assert fig.layout.showlegend is False
    # The rows' x-axes are matched so zooms and range relayouts stay in sync.
    assert fig.layout.xaxis2.matches == "x"


def test_histogram_rows_share_the_capped_y_range() -> None:
    per_phase = {"train": _layer_snap("train"), "val": _layer_snap("val")}
    fig, _ = _make_histogram_figure(per_phase, "activation", "activations")
    expected = _linear_y_range(
        _phase_hists(per_phase, "activation"), density=True
    )
    assert expected is not None
    assert tuple(fig.layout.yaxis.range) == tuple(expected)
    assert tuple(fig.layout.yaxis2.range) == tuple(expected)


# --- Watching histogram: dtype under/overflow band lines --------------------


def _fp16_band() -> tuple[float, float]:
    finfo = torch.finfo(torch.float16)
    return finfo.tiny, finfo.max


def test_under_over_line_positions_fp16_linear_uses_values() -> None:
    tiny, maxv = _fp16_band()
    pos = under_over_line_positions((tiny, maxv), log_x=False)
    values = [x for x, _ in pos]
    # Both edges of both signs land in the histogram's 1e-9..1e6 span; on the
    # linear axis the x-coordinate is the raw value.
    assert values == [tiny, -tiny, maxv, -maxv]
    # Only the positive edges carry a text label (the symmetric pair would
    # otherwise double it).
    assert {label for _, label in pos if label} == {"subnormal", "overflow"}


def test_under_over_line_positions_fp32_out_of_range() -> None:
    finfo = torch.finfo(torch.float32)
    # fp32's subnormal / saturation edges sit off both ends of the axis, so
    # nothing is drawn — which reads as "no under/overflow risk at this scale".
    assert under_over_line_positions((finfo.tiny, finfo.max), log_x=False) == []


def test_under_over_line_positions_log_uses_bin_coords() -> None:
    pos = under_over_line_positions(_fp16_band(), log_x=True)
    by_label = {label: x for x, label in pos if label}
    # Signed-log axis: x is the continuous bin coordinate — positive edges sit
    # right of the zero band, overflow further out than subnormal.
    assert by_label["subnormal"] > ZERO_BIN
    assert by_label["overflow"] > by_label["subnormal"]
    assert all(0.0 <= x <= N_BINS for x, _ in pos)


def test_make_histogram_figure_draws_band_lines() -> None:
    per_phase = {"train": _layer_snap("train", hist={ZERO_BIN + 40: 100})}
    fig, _ = _make_histogram_figure(
        per_phase, "gradient", "g", under_over_band=_fp16_band()
    )
    # Four in-range edges → four dotted vertical lines, positive ones labeled.
    lines = [s for s in fig.layout.shapes if s.type == "line"]
    assert len(lines) == 4
    assert all(s.line.dash == "dot" for s in lines)
    texts = {a.text for a in fig.layout.annotations}
    assert "subnormal" in texts and "overflow" in texts


def test_make_histogram_figure_no_band_lines_without_band() -> None:
    per_phase = {"train": _layer_snap("train", hist={ZERO_BIN + 40: 100})}
    fig, _ = _make_histogram_figure(per_phase, "gradient", "g")
    assert list(fig.layout.shapes) == []


def test_make_histogram_figure_band_lines_span_every_phase_row() -> None:
    per_phase = {
        "train": _layer_snap("train", hist={ZERO_BIN + 40: 100}),
        "val": _layer_snap("val", hist={ZERO_BIN + 40: 100}),
    }
    fig, _ = _make_histogram_figure(
        per_phase, "gradient", "g", under_over_band=_fp16_band()
    )
    # Four edges drawn on each of the two phase rows.
    assert len([s for s in fig.layout.shapes if s.type == "line"]) == 8


# --- Watching stats table ----------------------------------------------------


def test_stats_table_no_data() -> None:
    assert "no data yet" in _stats_table_html({})


def test_stats_table_has_kind_columns_and_stat_rows() -> None:
    per_phase = {
        "train": _layer_snap("train", epoch=2, n=1500),
        "val": _layer_snap("val", epoch=2, n=10),
    }
    table = _stats_table_html(per_phase)
    assert "<table" in table
    assert table.startswith("<div")  # framed box around each phase's table
    assert "train ep 2" in table and "val ep 2" in table
    assert ">activations</th>" in table and ">gradients</th>" in table
    for label in ("n", "mean", "std", "median", "min", "max"):
        assert f">{label}</td>" in table
    assert ">1,500</td>" in table  # n is comma-formatted


def test_stats_table_skips_phase_without_data() -> None:
    per_phase = {"train": _layer_snap("train", n=5), "val": _layer_snap("val", n=0)}
    table = _stats_table_html(per_phase)
    assert "train ep 0" in table
    assert "val" not in table


def test_stats_table_keeps_empty_kind_column() -> None:
    # Activations but no gradients (e.g. a val phase, which never runs
    # backward): the gradient column stays, its value cells em-dashes.
    empty = TensorStatsSnapshot(
        n=0, sum=0.0, sum_sq=0.0, min=math.inf, max=-math.inf,
        hist=tuple([0] * N_BINS),
    )
    snap = replace(_layer_snap("val", n=7), gradients=empty)
    table = _stats_table_html({"val": snap})
    assert ">activations</th>" in table and ">gradients</th>" in table
    assert ">0</td>" in table  # the gradient column's n
    assert ">—</td>" in table  # its mean/std/… (inf extremes included)


def test_stats_table_escapes_phase_names() -> None:
    per_phase = {"<b>": _layer_snap("<b>")}
    assert "<b>" not in _stats_table_html(per_phase)


@pytest.mark.parametrize(
    ("headings", "corner", "forbidden"),
    [
        # No mapping, or one that doesn't name the phase → the
        # "{phase} ep {epoch}" default.
        (None, "train ep 2", "override"),
        ({"val": "override"}, "train ep 2", "override"),
        # An override replaces the corner text verbatim.
        ({"train": "current batch — train ep 2"}, "current batch — train ep 2", None),
        # Override text is HTML-escaped like the default phase name.
        ({"train": "<b>train</b>"}, "&lt;b&gt;train&lt;/b&gt;", "<b>"),
    ],
)
def test_stats_table_headings_override_corner_header(
    headings: dict[str, str] | None, corner: str, forbidden: str | None
) -> None:
    table = _stats_table_html(
        {"train": _layer_snap("train", epoch=2)}, headings=headings
    )
    # The corner cell closes the tinted style with the phase color, which
    # stays keyed on the real phase name no matter what the heading says.
    assert f'color:{phase_color("train", 0)}">{corner}</th>' in table
    if forbidden is not None:
        assert forbidden not in table


def _snap_with_channel_hists(
    rows: list[dict[int, int]], phase: str = "train"
) -> LayerStatsSnapshot:
    """A layer snapshot whose activations carry per-channel `bin -> count` rows."""
    channel_hists = tuple(
        tuple(row.get(b, 0) for b in range(N_BINS)) for row in rows
    )
    hist = tuple(sum(col) for col in zip(*channel_hists))
    stats = TensorStatsSnapshot(
        n=sum(hist), sum=0.0, sum_sq=0.0, min=0.0, max=0.0,
        hist=hist, channel_hists=channel_hists,
    )
    return LayerStatsSnapshot(
        layer="L", phase=phase, epoch=0, activations=stats, gradients=stats
    )


@pytest.mark.parametrize(
    ("rows", "expected"),
    [
        # All mass in the zero bin = dead; any other bin keeps it alive.
        ([{ZERO_BIN: 4}, {ZERO_BIN: 3, ZERO_BIN + 9: 1}, {ZERO_BIN: 5}], [0, 2]),
        # A channel that never saw a value is not reported as dead.
        ([{}, {ZERO_BIN: 2}], [1]),
        ([{ZERO_BIN + 1: 7}], []),
    ],
)
def test_dead_channels_from_channel_hists(
    rows: list[dict[int, int]], expected: list[int]
) -> None:
    snap = _snap_with_channel_hists(rows)
    assert dead_channels(snap.activations) == expected


def test_dead_channels_none_without_channel_hists() -> None:
    assert dead_channels(_layer_snap("train").activations) is None


def test_stats_table_dead_channels_fill_activation_column_only() -> None:
    per_phase = {"train": _snap_with_channel_hists([{ZERO_BIN: 2}, {3: 1}])}
    table = _stats_table_html(per_phase)
    assert ">dead channels</td>" in table
    assert 'title="channels: 0"' in table  # hover lists the dead indices
    # Both kinds carry channel hists here, but only the activation column
    # reports dead channels — the gradient cell stays an em-dash.
    row = table.split(">dead channels</td>", 1)[1].split("</tr>", 1)[0]
    assert row.count("title=") == 1
    assert row.endswith("—</td>")


def test_stats_table_dead_channels_placeholder_without_channel_hists() -> None:
    table = _stats_table_html({"train": _layer_snap("train")})
    assert ">dead channels</td>" in table
    assert ">—</td>" in table
    assert "title=" not in table


def test_stats_table_dead_channels_hover_truncates_to_ten() -> None:
    per_phase = {"train": _snap_with_channel_hists([{ZERO_BIN: 1}] * 12)}
    table = _stats_table_html(per_phase)
    assert ">12</td>" in table
    listed = ", ".join(str(c) for c in range(10))
    assert f'title="channels: {listed}, ..."' in table


# --- Watching histogram: plot height (task 3) ------------------------------


def test_histogram_height_scales_with_phase_rows() -> None:
    assert _PLOT_HEIGHT == 440  # 2x the original 220, now per phase row
    fig, _ = _make_histogram_figure({}, "activation", "activations")
    assert fig.layout.height == _PLOT_HEIGHT
    two = {"train": _layer_snap("train"), "val": _layer_snap("val")}
    fig, _ = _make_histogram_figure(two, "activation", "activations")
    assert fig.layout.height == 2 * _PLOT_HEIGHT


def test_log_x_hover_shows_bin_value_not_index() -> None:
    hist = {ZERO_BIN: 5, ZERO_BIN + 1 + 9 * 7: 4}  # zero band + the 1.0 decade
    per_phase = {"train": _layer_snap("train", hist=hist)}
    fig, _ = _make_histogram_figure(
        per_phase, "activation", "activations", log_x=True
    )
    hover = fig.data[0].hovertemplate
    assert "bin %{x}" not in hover
    assert "value ≈ %{customdata[1]}" in hover
    assert "count %{customdata[0]}" in hover
    # Each bar carries (count, representative value); the value matches the
    # bin's geometric midpoint, the same notion the median stat reports.
    count, label = fig.data[0].customdata[ZERO_BIN + 1 + 9 * 7]
    assert count == 4
    assert label == f"{bin_midpoint(ZERO_BIN + 1 + 9 * 7):.3g}"


def test_linear_hover_keeps_value_from_bar_position() -> None:
    per_phase = {"train": _layer_snap("train", n=9)}
    fig, _ = _make_histogram_figure(
        per_phase, "activation", "activations", log_x=False
    )
    assert "value %{x:.2e}" in fig.data[0].hovertemplate
    assert fig.data[0].customdata[ZERO_BIN] == 9


# --- Retaining axes: value <-> bin-index coordinate conversion --------------


def test_value_to_bin_coord_maps_zero_to_zero_band_centre() -> None:
    # The value 0 sits at the centre of the zero band, ZERO_BIN.
    assert _value_to_bin_coord(0.0) == pytest.approx(ZERO_BIN)
    assert _bin_coord_to_value(float(ZERO_BIN)) == pytest.approx(0.0)


@pytest.mark.parametrize("coord", [5.0, 40.0, 105.0, 150.0, 205.0])
def test_value_bin_coord_round_trips(coord: float) -> None:
    value = _bin_coord_to_value(coord)
    assert _value_to_bin_coord(value) == pytest.approx(coord, abs=1e-6)


def test_value_to_bin_coord_clamps_beyond_the_span() -> None:
    # The bin-index axis runs from the left edge of bin 0 (-0.5) to the right
    # edge of the last bin (N_BINS - 0.5); values past +/-1e6 clamp there.
    assert _value_to_bin_coord(1e9) == pytest.approx(N_BINS - 0.5)
    assert _value_to_bin_coord(-1e9) == pytest.approx(-0.5)


# --- Retaining axes: x-range conversion across a Log x toggle ---------------


def test_x_range_linear_to_log_round_trips() -> None:
    linear = [-0.05, 0.05]
    back = _x_range_log_to_linear(_x_range_linear_to_log(linear))
    assert back[0] == pytest.approx(linear[0], rel=1e-6)
    assert back[1] == pytest.approx(linear[1], rel=1e-6)


def test_x_range_linear_to_log_straddles_the_zero_bin() -> None:
    log_range = _x_range_linear_to_log([-0.05, 0.05])
    assert log_range[0] < ZERO_BIN < log_range[1]


def test_x_range_linear_to_log_enforces_min_span_near_zero() -> None:
    # A window narrower than the zero band would collapse to ~zero width on
    # the bin axis; the guard keeps it at least one bin wide, centred on zero.
    log_range = _x_range_linear_to_log([-1e-12, 1e-12])
    assert log_range[1] - log_range[0] >= 1.0
    assert log_range[0] < ZERO_BIN < log_range[1]


def test_x_range_log_to_linear_zero_band_straddles_zero() -> None:
    linear = _x_range_log_to_linear([ZERO_BIN - 1.0, ZERO_BIN + 1.0])
    assert linear[0] < 0.0 < linear[1]


# --- Retaining axes: y-cap conversion across a Log y toggle -----------------


def test_retained_y_range_linear_keeps_top_from_zero() -> None:
    assert _retained_y_range(5.0, log_y=False, floor=0.01) == [0.0, 5.0]


def test_retained_y_range_log_spans_floor_to_top() -> None:
    assert _retained_y_range(100.0, log_y=True, floor=0.1) == [
        pytest.approx(math.log10(0.1)),
        pytest.approx(math.log10(100.0)),
    ]


@pytest.mark.parametrize("floor", [None, 0.0, -1.0, 5.0, 10.0])
def test_retained_y_range_log_floor_near_zero_falls_back(floor: float | None) -> None:
    # Without a usable positive floor below the top (the "near 0" case — a
    # linear bottom of 0 has no log), the bottom drops three decades.
    rng = _retained_y_range(5.0, log_y=True, floor=floor)
    assert rng is not None
    assert rng[1] == pytest.approx(math.log10(5.0))
    assert rng[0] == pytest.approx(math.log10(5.0 * 1e-3))


def test_retained_y_range_none_without_a_cap() -> None:
    assert _retained_y_range(None, log_y=False, floor=1.0) is None
    assert _retained_y_range(0.0, log_y=True, floor=1.0) is None


def test_min_positive_height_finds_smallest_positive_bar() -> None:
    per_phase = {
        "train": _layer_snap("train", hist={ZERO_BIN: 1000, ZERO_BIN + 50: 1})
    }
    hists = _phase_hists(per_phase, "activation")
    probs = _probabilities(live_hist(per_phase["train"].activations))
    expected = min(p for p in probs if p > 0)
    assert _min_positive_height(hists, density=False) == pytest.approx(expected)


def test_min_positive_height_none_when_empty() -> None:
    assert _min_positive_height([], density=True) is None


# --- Retaining axes: forcing the figure's ranges ---------------------------


def test_make_histogram_figure_uses_override_ranges() -> None:
    per_phase = {"train": _layer_snap("train", hist={ZERO_BIN + 10: 100})}
    fig, (x_range, y_range) = _make_histogram_figure(
        per_phase,
        "activation",
        "activations",
        log_x=False,
        log_y=False,
        override_ranges=([-0.3, 0.3], [0.0, 7.0]),
    )
    # The forced ranges are applied and returned verbatim (no data fit).
    assert x_range == [-0.3, 0.3]
    assert y_range == [0.0, 7.0]
    assert tuple(fig.layout.xaxis.range) == (-0.3, 0.3)
    assert tuple(fig.layout.yaxis.range) == (0.0, 7.0)
    # Off-view bars are still blanked to the applied x-range, so hover stays
    # confined to the bins on screen.
    kept = [c for c in fig.data[0].x if not math.isnan(c)]
    assert kept and all(-0.3 <= c <= 0.3 for c in kept)


def test_make_histogram_figure_override_carries_log_y_range() -> None:
    per_phase = {"train": _layer_snap("train", n=50)}
    fig, (_, y_range) = _make_histogram_figure(
        per_phase,
        "activation",
        "activations",
        log_x=False,
        log_y=True,
        override_ranges=([-0.3, 0.3], [-3.0, 0.0]),
    )
    assert fig.layout.yaxis.type == "log"
    assert y_range == [-3.0, 0.0]
    assert tuple(fig.layout.yaxis.range) == (-3.0, 0.0)


@pytest.mark.parametrize("log_y", [False, True])
def test_histogram_axes_use_power_exponents_not_si_prefixes(
    log_y: bool,
) -> None:
    # Plotly's default SI tick suffixes ("100M" densities, "500n" values)
    # read as units, not scales — exponents render as powers of ten.
    per_phase = {"train": _layer_snap("train")}
    fig, _ = _make_histogram_figure(
        per_phase, "activation", "a", log_y=log_y
    )
    assert fig.layout.yaxis.exponentformat == "power"
    assert fig.layout.xaxis.exponentformat == "power"  # linear value axis
