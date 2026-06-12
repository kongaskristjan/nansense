"""Tests for histogram figures, axis ranges, density, and stats tables in nansense.ui.histograms."""

from __future__ import annotations

import math

import pytest

from nansense.ui.histograms import (
    BIN_CENTERS,
    BIN_WIDTHS,
    _HIST_EDGES,
    _PLOT_HEIGHT,
    axis_ranges,
    dead_channels,
    _fill_fraction,
    _linear_bar_x,
    _linear_x_range,
    _linear_y_range,
    _log_x_range,
    _make_histogram_figure,
    _phase_hists,
    _phases_with_data,
    _probabilities,
    _probability_densities,
    _stats_table_html,
    _trimmed_bin_bounds,
    use_density,
)
from nansense.watch import (
    N_BINS,
    ZERO_BIN,
    LayerStatsSnapshot,
    TensorStatsSnapshot,
    bin_midpoint,
)
from tests.nansense.helpers import _layer_snap


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
    # rebuild never silently hides a series.
    assert [t.name for t in fig.data] == ["train (ep 0)"]
    assert all(t.visible in (True, None) for t in fig.data)


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
    heights = _probability_densities(per_phase["train"].activations.hist)
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
    # covering almost none of the plot. The budget then rises (up to 5%),
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
    heights = _probability_densities(per_phase["train"].activations.hist)
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
    heights = _probability_densities(per_phase["train"].activations.hist)
    rng = _linear_y_range(_phase_hists(per_phase, "activation"), density=True)
    assert rng is not None
    assert rng[1] == pytest.approx(max(heights) * 1.05)


def test_linear_y_range_excludes_dominant_spike_despite_count() -> None:
    # The zero-band spike holds half the points but towers >5x over the
    # runner-up, so it never anchors the scale; the cap lands on the bulk.
    hist = {ZERO_BIN: 10_000, ZERO_BIN + 50: 9_000, ZERO_BIN + 51: 1_000}
    per_phase = {"train": _layer_snap("train", hist=hist)}
    heights = _probability_densities(per_phase["train"].activations.hist)
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
    train_heights = _probability_densities(per_phase["train"].activations.hist)
    val_heights = _probability_densities(per_phase["val"].activations.hist)
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
    train_heights = _probability_densities(per_phase["train"].activations.hist)
    val_heights = _probability_densities(per_phase["val"].activations.hist)
    rng = _linear_y_range(_phase_hists(per_phase, "activation"), density=True)
    assert rng is not None
    expected_cap = max(train_heights[ZERO_BIN + 30], val_heights[ZERO_BIN + 40])
    assert rng[1] == pytest.approx(expected_cap * 1.05)
    assert rng[1] < train_heights[ZERO_BIN]


def test_linear_y_range_none_when_empty() -> None:
    assert _linear_y_range([], density=True) is None


def test_histogram_density_mode_plots_probability_density_with_capped_axis() -> None:
    hist = {ZERO_BIN: 50, ZERO_BIN + 10: 7}
    per_phase = {"train": _layer_snap("train", hist=hist)}
    fig, _ = _make_histogram_figure(
        per_phase, "activation", "activations", log_x=False, log_y=False
    )
    trace = fig.data[0]
    expected = _probability_densities(per_phase["train"].activations.hist)
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
    expected = _probability_densities(per_phase["train"].activations.hist)
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
    probs = _probabilities(per_phase["train"].activations.hist)
    assert tuple(fig.layout.yaxis.range) == pytest.approx(
        (0.0, probs[ZERO_BIN + 50] * 1.05)
    )
    assert fig.layout.yaxis.range[1] < probs[ZERO_BIN]


# --- Watching histogram: one subplot row per phase ---------------------------


def test_histogram_one_subplot_row_per_phase() -> None:
    per_phase = {"train": _layer_snap("train"), "val": _layer_snap("val", epoch=1)}
    fig, _ = _make_histogram_figure(per_phase, "activation", "activations")
    assert [t.name for t in fig.data] == ["train (ep 0)", "val (ep 1)"]
    # Each phase draws alone in its own stacked row: full opacity, no
    # overlay, separate y-axes.
    assert [t.opacity for t in fig.data] == [0.85, 0.85]
    assert fig.data[0].yaxis == "y"
    assert fig.data[1].yaxis == "y2"
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


# --- Watching stats table ----------------------------------------------------


def test_stats_table_no_data() -> None:
    assert "no data yet" in _stats_table_html({}, "activation")


def test_stats_table_has_phase_columns_and_stat_rows() -> None:
    per_phase = {
        "train": _layer_snap("train", epoch=2, n=1500),
        "val": _layer_snap("val", epoch=2, n=10),
    }
    table = _stats_table_html(per_phase, "activation")
    assert "<table" in table
    assert table.startswith("<div")  # framed box around the table
    assert "train ep 2" in table and "val ep 2" in table
    for label in ("n", "mean", "std", "median", "min", "max"):
        assert f">{label}</td>" in table
    assert ">1,500</td>" in table  # n is comma-formatted


def test_stats_table_skips_phase_without_data() -> None:
    per_phase = {"train": _layer_snap("train", n=5), "val": _layer_snap("val", n=0)}
    table = _stats_table_html(per_phase, "activation")
    assert "train ep 0" in table
    assert "val" not in table


def test_stats_table_escapes_phase_names() -> None:
    per_phase = {"<b>": _layer_snap("<b>")}
    assert "<b>" not in _stats_table_html(per_phase, "activation")


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


def test_stats_table_dead_channels_row_only_for_activations() -> None:
    per_phase = {"train": _snap_with_channel_hists([{ZERO_BIN: 2}, {3: 1}])}
    act = _stats_table_html(per_phase, "activation")
    assert ">dead channels</td>" in act
    assert 'title="channels: 0"' in act  # hover lists the dead indices
    assert "dead channels" not in _stats_table_html(per_phase, "gradient")


def test_stats_table_dead_channels_placeholder_without_channel_hists() -> None:
    table = _stats_table_html({"train": _layer_snap("train")}, "activation")
    assert ">dead channels</td>" in table
    assert ">—</td>" in table
    assert "title=" not in table


def test_stats_table_dead_channels_hover_truncates_to_ten() -> None:
    per_phase = {"train": _snap_with_channel_hists([{ZERO_BIN: 1}] * 12)}
    table = _stats_table_html(per_phase, "activation")
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
