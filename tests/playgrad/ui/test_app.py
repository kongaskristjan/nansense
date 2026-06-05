"""Tests for pure helpers in `playgrad.ui.app`."""

from __future__ import annotations

import pytest
import torch

import playgrad
from playgrad.schedule import BatchPosition, Schedule
from playgrad.session import BatchSnapshot
from playgrad.ui.app import (
    _BIN_WIDTHS,
    _HIST_EDGES,
    _PLOT_HEIGHT,
    _PLOTLY_CONFIG,
    _RenderCache,
    _compute_frame,
    _default_roles,
    _dims_from_roles,
    _figure_payload,
    _format_live_position,
    _input_img_tag,
    _linear_x_range,
    _linear_y_range,
    _log_x_range,
    _make_histogram_figure,
    _phases_with_data,
    _probabilities,
    _probability_densities,
    _role_options,
    _stats_table_html,
    _strip_html,
    _summarize_epoch_ranges,
    _trimmed_bin_bounds,
    _use_density,
    _validate_step_until_target,
    serve,
)
from playgrad.ui.render import (
    INPUT_IMAGE_SIZE,
    image_mime,
    render_image,
    render_strip,
)
from playgrad.watch import N_BINS, ZERO_BIN, LayerStatsSnapshot, TensorStatsSnapshot


def _tensor_stats(n: int, hist: dict[int, int] | None = None) -> TensorStatsSnapshot:
    """Stats with `n` values in the zero band, or an explicit `bin -> count` map."""
    counts = [0] * N_BINS
    if hist is None:
        counts[ZERO_BIN] = n
    else:
        for idx, count in hist.items():
            counts[idx] = count
        n = sum(hist.values())
    return TensorStatsSnapshot(
        n=n, sum=0.0, sum_sq=0.0, min=0.0, max=0.0, hist=tuple(counts)
    )


def _layer_snap(
    phase: str,
    epoch: int = 0,
    n: int = 10,
    hist: dict[int, int] | None = None,
) -> LayerStatsSnapshot:
    stats = _tensor_stats(n, hist)
    return LayerStatsSnapshot(
        layer="L",
        phase=phase,
        epoch=epoch,
        activations=stats,
        gradients=stats,
    )


def _snapshot_at(phase: str, epoch: int, batch_idx: int) -> BatchSnapshot:
    return BatchSnapshot(
        position=BatchPosition(
            phase=phase,
            epoch=epoch,
            batch_idx=batch_idx,
            is_last_in_phase=False,
            is_last_in_epoch=False,
            is_last_overall=False,
        ),
        activations={"x": torch.zeros(1)},
        activation_gradients={},
        weights={},
        weight_gradients={},
    )


@pytest.fixture
def schedule() -> Schedule:
    return Schedule(epochs=3, phases={"train": 5, "val": 2})


def test_validate_passes_for_future_position(schedule: Schedule) -> None:
    snap = _snapshot_at("train", 0, 1)
    assert (
        _validate_step_until_target(
            schedule=schedule, snapshot=snap, phase="val", epoch=0, batch_idx=0
        )
        is None
    )


def test_validate_passes_when_no_snapshot_yet(schedule: Schedule) -> None:
    assert (
        _validate_step_until_target(
            schedule=schedule, snapshot=None, phase="train", epoch=0, batch_idx=0
        )
        is None
    )


@pytest.mark.parametrize(
    "phase, epoch, batch_idx",
    [
        ("train", 0, 0),
        ("train", 0, 1),
        ("train", 0, 2),
    ],
)
def test_validate_rejects_position_at_or_before_current(
    schedule: Schedule, phase: str, epoch: int, batch_idx: int
) -> None:
    snap = _snapshot_at("train", 0, 2)
    msg = _validate_step_until_target(
        schedule=schedule, snapshot=snap, phase=phase, epoch=epoch, batch_idx=batch_idx
    )
    assert msg is not None
    assert "after the current" in msg


def test_validate_rejects_earlier_phase_in_same_epoch(schedule: Schedule) -> None:
    snap = _snapshot_at("val", 0, 0)
    msg = _validate_step_until_target(
        schedule=schedule, snapshot=snap, phase="train", epoch=0, batch_idx=4
    )
    assert msg is not None
    assert "after the current" in msg


def test_validate_rejects_unknown_phase(schedule: Schedule) -> None:
    msg = _validate_step_until_target(
        schedule=schedule, snapshot=None, phase="bogus", epoch=0, batch_idx=0
    )
    assert msg is not None
    assert "Unknown phase" in msg


def test_validate_rejects_epoch_out_of_range(schedule: Schedule) -> None:
    msg = _validate_step_until_target(
        schedule=schedule, snapshot=None, phase="train", epoch=3, batch_idx=0
    )
    assert msg is not None
    assert "Epoch" in msg


def test_validate_rejects_batch_out_of_range(schedule: Schedule) -> None:
    msg = _validate_step_until_target(
        schedule=schedule, snapshot=None, phase="train", epoch=0, batch_idx=5
    )
    assert msg is not None
    assert "Batch" in msg


def test_format_live_position() -> None:
    pos = BatchPosition(
        phase="val",
        epoch=3,
        batch_idx=7,
        is_last_in_phase=False,
        is_last_in_epoch=False,
        is_last_overall=False,
    )
    assert _format_live_position(pos) == "epoch 3 | val batch 7"


@pytest.mark.parametrize(
    "ndim, expected",
    [
        (1, ["x", "index"]),
        (2, ["x", "y", "index"]),
        (3, ["x", "y", "tile", "index"]),
        (4, ["x", "y", "tile", "index"]),
    ],
)
def test_role_options_scale_with_rank(ndim: int, expected: list[str]) -> None:
    assert list(_role_options(ndim)) == expected


@pytest.mark.parametrize(
    "ndim, roles",
    [
        (1, ["x"]),
        (2, ["y", "x"]),
        (3, ["tile", "y", "x"]),
        (4, ["index", "tile", "y", "x"]),
    ],
)
def test_default_roles_match_default_dims(ndim: int, roles: list[str]) -> None:
    assert _default_roles(ndim) == roles


def test_dims_from_roles_resolves_axes() -> None:
    assert _dims_from_roles(["index", "tile", "y", "x"]) == (3, 2, 1)
    assert _dims_from_roles(["x"]) == (0, None, None)
    assert _dims_from_roles(["index", "index"]) == (None, None, None)


# --- Watching histogram: trace structure / restyle signature --------------


def test_phases_with_data_in_render_order() -> None:
    per_phase = {"train": _layer_snap("train"), "val": _layer_snap("val")}
    assert _phases_with_data(per_phase, "activation") == ["train", "val"]


def test_phases_with_data_skips_empty_phases() -> None:
    per_phase = {"train": _layer_snap("train", n=5), "val": _layer_snap("val", n=0)}
    assert _phases_with_data(per_phase, "activation") == ["train"]


def test_histogram_traces_match_phases_with_data() -> None:
    per_phase = {"train": _layer_snap("train"), "val": _layer_snap("val", n=0)}
    fig = _make_histogram_figure(per_phase, "activation", "activations")
    # Only phases with data get a trace, in order; all visible by default so a
    # rebuild never silently hides a series.
    assert [t.name for t in fig.data] == ["train (ep 0)"]
    assert all(t.visible in (True, None) for t in fig.data)


# --- Watching histogram: log/linear axis toggles (task 1) ------------------


@pytest.mark.parametrize("log_y", [True, False])
def test_histogram_log_y_toggles_y_axis_scale(log_y: bool) -> None:
    per_phase = {"train": _layer_snap("train")}
    fig = _make_histogram_figure(
        per_phase, "activation", "activations", log_y=log_y
    )
    assert fig.layout.yaxis.type == ("log" if log_y else "linear")


def test_histogram_log_x_uses_bin_indices() -> None:
    per_phase = {"train": _layer_snap("train")}
    fig = _make_histogram_figure(
        per_phase, "activation", "activations", log_x=True
    )
    # Log mode: evenly spaced bin indices, default (uniform) bar width.
    assert fig.data[0].x[0] == 0
    assert fig.data[0].width is None


def test_histogram_linear_x_uses_value_positions_and_widths() -> None:
    per_phase = {"train": _layer_snap("train")}
    fig = _make_histogram_figure(
        per_phase, "activation", "activations", log_x=False
    )
    # Linear mode: bars positioned at true bin centres with per-bin widths.
    assert fig.data[0].x[0] < 0  # first bin centre is far negative
    assert isinstance(fig.data[0].width, tuple)
    assert len(fig.data[0].width) == N_BINS


def test_linear_x_range_none_when_empty() -> None:
    assert _linear_x_range({}, "activation") is None


def test_linear_x_range_brackets_populated_bins() -> None:
    # Counts only in the zero band → a tight range straddling zero.
    per_phase = {"train": _layer_snap("train", n=7)}
    rng = _linear_x_range(per_phase, "activation")
    assert rng is not None
    lo, hi = rng
    assert lo < 0 < hi
    assert hi - lo < 1.0  # zoomed in, not the full +/-1e6 span


# --- Watching histogram: x-range tail trimming ------------------------------


def test_trimmed_bin_bounds_drops_sparse_outlier_tail() -> None:
    # The 5-count outlier bin holds < 0.5% of the points, so the bounds
    # shrink to the bulk bin.
    hist = {ZERO_BIN + 10: 10_000, ZERO_BIN + 80: 5}
    per_phase = {"train": _layer_snap("train", hist=hist)}
    assert _trimmed_bin_bounds(per_phase, "activation") == (
        ZERO_BIN + 10,
        ZERO_BIN + 10,
    )


def test_trimmed_bin_bounds_keeps_heavy_tails() -> None:
    # Both bins hold 50% of the points — trimming either would drop far more
    # than the 0.5% budget, so the bounds cover them both.
    hist = {ZERO_BIN + 10: 500, ZERO_BIN + 80: 500}
    per_phase = {"train": _layer_snap("train", hist=hist)}
    assert _trimmed_bin_bounds(per_phase, "activation") == (
        ZERO_BIN + 10,
        ZERO_BIN + 80,
    )


def test_trimmed_bin_bounds_budget_spans_both_tails() -> None:
    # 3 + 4 outlier points (0.07% combined) trim away on both sides.
    hist = {ZERO_BIN - 40: 3, ZERO_BIN + 10: 10_000, ZERO_BIN + 40: 4}
    per_phase = {"train": _layer_snap("train", hist=hist)}
    assert _trimmed_bin_bounds(per_phase, "activation") == (
        ZERO_BIN + 10,
        ZERO_BIN + 10,
    )


def test_trimmed_bin_bounds_stops_when_budget_runs_out() -> None:
    # Trimming both tails (30 + 40 = 70 of 10070) would exceed the 0.5%
    # budget (~50 points), so only the lighter tail is dropped.
    hist = {ZERO_BIN - 40: 40, ZERO_BIN + 10: 10_000, ZERO_BIN + 40: 30}
    per_phase = {"train": _layer_snap("train", hist=hist)}
    assert _trimmed_bin_bounds(per_phase, "activation") == (
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
    assert _trimmed_bin_bounds(per_phase, "activation") == (
        ZERO_BIN + 10,
        ZERO_BIN + 10,
    )


def test_trimmed_bin_bounds_none_when_empty() -> None:
    assert _trimmed_bin_bounds({}, "activation") is None


def test_linear_x_range_excludes_sparse_outlier_tail() -> None:
    hist = {ZERO_BIN + 10: 10_000, ZERO_BIN + 80: 5}
    per_phase = {"train": _layer_snap("train", hist=hist)}
    rng = _linear_x_range(per_phase, "activation")
    assert rng is not None
    assert rng[1] < _HIST_EDGES[ZERO_BIN + 80]  # the outlier bin is outside


def test_log_x_range_brackets_trimmed_bins() -> None:
    hist = {ZERO_BIN + 10: 10_000, ZERO_BIN + 80: 5}
    per_phase = {"train": _layer_snap("train", hist=hist)}
    rng = _log_x_range(per_phase, "activation")
    assert rng is not None
    assert rng[0] < ZERO_BIN + 10 < rng[1]
    assert rng[1] < ZERO_BIN + 80  # the outlier bin is outside
    assert _log_x_range({}, "activation") is None


def test_histogram_log_x_zooms_to_trimmed_bins() -> None:
    hist = {ZERO_BIN + 10: 10_000, ZERO_BIN + 80: 5}
    per_phase = {"train": _layer_snap("train", hist=hist)}
    fig = _make_histogram_figure(
        per_phase, "activation", "activations", log_x=True
    )
    expected = _log_x_range(per_phase, "activation")
    assert expected is not None
    assert tuple(fig.layout.xaxis.range) == tuple(expected)


# --- Watching histogram: density mode (linear value axis) ------------------


def test_histogram_defaults_to_linear_density_mode() -> None:
    # Log x / Log y start unchecked, so the no-args figure is the density view.
    per_phase = {"train": _layer_snap("train")}
    fig = _make_histogram_figure(per_phase, "activation", "activations")
    assert fig.layout.yaxis.type == "linear"
    assert fig.layout.yaxis.title.text == "probability density"


@pytest.mark.parametrize("log_x, expected", [(True, False), (False, True)])
def test_use_density_depends_only_on_log_x(log_x: bool, expected: bool) -> None:
    assert _use_density(log_x) is expected


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
    assert heights[ZERO_BIN] == pytest.approx(0.4 / _BIN_WIDTHS[ZERO_BIN])
    assert heights[ZERO_BIN + 10] == pytest.approx(0.6 / _BIN_WIDTHS[ZERO_BIN + 10])
    assert heights[0] == 0
    # Bar areas integrate to 1 — it's a probability density.
    assert sum(h * w for h, w in zip(heights, _BIN_WIDTHS)) == pytest.approx(1.0)


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
    rng = _linear_y_range(per_phase, "activation", density=True)
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
    rng = _linear_y_range(per_phase, "activation", density=True)
    assert rng is not None
    assert rng[1] == pytest.approx(max(heights) * 1.05)


def test_linear_y_range_excludes_dominant_spike_despite_count() -> None:
    # The zero-band spike holds half the points but towers >5x over the
    # runner-up, so it never anchors the scale; the cap lands on the bulk.
    hist = {ZERO_BIN: 10_000, ZERO_BIN + 50: 9_000, ZERO_BIN + 51: 1_000}
    per_phase = {"train": _layer_snap("train", hist=hist)}
    heights = _probability_densities(per_phase["train"].activations.hist)
    rng = _linear_y_range(per_phase, "activation", density=True)
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
    rng = _linear_y_range(per_phase, "activation", density=True)
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
    rng = _linear_y_range(per_phase, "activation", density=True)
    assert rng is not None
    expected_cap = max(train_heights[ZERO_BIN + 30], val_heights[ZERO_BIN + 40])
    assert rng[1] == pytest.approx(expected_cap * 1.05)
    assert rng[1] < train_heights[ZERO_BIN]


def test_linear_y_range_none_when_empty() -> None:
    assert _linear_y_range({}, "activation", density=True) is None


def test_histogram_density_mode_plots_probability_density_with_capped_axis() -> None:
    hist = {ZERO_BIN: 50, ZERO_BIN + 10: 7}
    per_phase = {"train": _layer_snap("train", hist=hist)}
    fig = _make_histogram_figure(
        per_phase, "activation", "activations", log_x=False, log_y=False
    )
    trace = fig.data[0]
    expected = _probability_densities(per_phase["train"].activations.hist)
    assert list(trace.y) == pytest.approx(expected)
    # Raw counts ride along for the hover text.
    assert trace.customdata[ZERO_BIN] == 50
    assert fig.layout.yaxis.title.text == "probability density"
    expected_range = _linear_y_range(per_phase, "activation", density=True)
    assert expected_range is not None
    assert tuple(fig.layout.yaxis.range) == tuple(expected_range)


def test_histogram_log_y_keeps_density_but_drops_range_cap() -> None:
    # Log y alone doesn't switch what's measured: bars stay probability
    # densities, but the cap (a linear-space range) gives way to autorange
    # so all of the data is visible on the log scale.
    per_phase = {"train": _layer_snap("train", n=9)}
    fig = _make_histogram_figure(
        per_phase, "activation", "activations", log_x=False, log_y=True
    )
    expected = _probability_densities(per_phase["train"].activations.hist)
    assert list(fig.data[0].y) == pytest.approx(expected)
    assert fig.layout.yaxis.title.text == "probability density"
    assert fig.layout.yaxis.range is None


@pytest.mark.parametrize("log_y", [True, False])
def test_histogram_log_x_plots_probabilities(log_y: bool) -> None:
    per_phase = {"train": _layer_snap("train", n=9)}
    fig = _make_histogram_figure(
        per_phase, "activation", "activations", log_x=True, log_y=log_y
    )
    assert fig.data[0].y[ZERO_BIN] == pytest.approx(1.0)
    assert fig.data[0].customdata[ZERO_BIN] == 9
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
    fig = _make_histogram_figure(
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
    fig = _make_histogram_figure(per_phase, "activation", "activations")
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
    fig = _make_histogram_figure(per_phase, "activation", "activations")
    expected = _linear_y_range(per_phase, "activation", density=True)
    assert expected is not None
    assert tuple(fig.layout.yaxis.range) == tuple(expected)
    assert tuple(fig.layout.yaxis2.range) == tuple(expected)


def test_figure_payload_carries_plotly_config() -> None:
    # Autoscale would land on a different scale than the capped initial
    # render, so it's removed; double-click resets to the built ranges.
    payload = _figure_payload(_make_histogram_figure({}, "activation", "a"))
    assert set(payload) >= {"data", "layout", "config"}
    assert payload["config"] is _PLOTLY_CONFIG
    assert _PLOTLY_CONFIG["modeBarButtonsToRemove"] == ["autoScale2d"]
    assert _PLOTLY_CONFIG["doubleClick"] == "reset"


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


# --- Watching histogram: plot height (task 3) ------------------------------


def test_histogram_height_scales_with_phase_rows() -> None:
    assert _PLOT_HEIGHT == 440  # 2x the original 220, now per phase row
    fig = _make_histogram_figure({}, "activation", "activations")
    assert fig.layout.height == _PLOT_HEIGHT
    two = {"train": _layer_snap("train"), "val": _layer_snap("val")}
    fig = _make_histogram_figure(two, "activation", "activations")
    assert fig.layout.height == 2 * _PLOT_HEIGHT


def test_serve_on_disabled_session_is_noop() -> None:
    """`serve()` returns None without starting a server for a disabled session."""
    model = torch.nn.Linear(4, 2)
    session = playgrad.start(model, epochs=1, phases={"train": 1}, enabled=False)
    assert serve(session) is None


@pytest.mark.parametrize(
    ("epochs", "expected"),
    [
        ([0], "0"),
        ([0, 1, 2], "0–2"),
        ([0, 1, 2, 5, 7, 8], "0–2, 5, 7–8"),
        ([3, 5], "3, 5"),
    ],
)
def test_summarize_epoch_ranges(epochs: list[int], expected: str) -> None:
    assert _summarize_epoch_ranges(epochs) == expected


# --- Strip HTML assembly ----------------------------------------------------


def test_strip_html_scales_native_data_and_keeps_legend_crisp() -> None:
    strip = render_strip(torch.randn(1, 2, 8, 8), sample_idx=0)
    assert strip is not None
    html = _strip_html(strip)
    # One legend <img> (shown 1:1, no pixelated scaling) + one data <img>.
    assert html.count("<img") == 2
    assert html.count("image-rendering:pixelated") == 1
    assert html.count(f"data:{image_mime()};base64,") == 2
    assert f"width:{strip.width}px" in html
    assert f"height:{strip.height}px" in html


def test_strip_html_empty_for_none() -> None:
    assert _strip_html(None) == ""


def test_input_img_tag_scales_to_display_size() -> None:
    png = render_image(torch.rand(1, 3, 16, 16), sample_idx=0)
    assert png is not None
    html = _input_img_tag(png)
    assert f"width:{INPUT_IMAGE_SIZE}px" in html
    assert "image-rendering:pixelated" in html
    assert _input_img_tag(None) == ""


# --- Render cache + frame computation ---------------------------------------


def _frame_snapshot() -> BatchSnapshot:
    return BatchSnapshot(
        position=BatchPosition(
            phase="train",
            epoch=0,
            batch_idx=0,
            is_last_in_phase=False,
            is_last_in_epoch=False,
            is_last_overall=False,
        ),
        activations={"x": torch.rand(2, 3, 4, 4), "conv": torch.rand(2, 2, 4, 4)},
        activation_gradients={"conv": torch.rand(2, 2, 4, 4)},
        weights={},
        weight_gradients={},
    )


def test_render_cache_renders_once_per_key() -> None:
    cache = _RenderCache()
    snap = _frame_snapshot()
    calls = 0

    def render() -> str:
        nonlocal calls
        calls += 1
        return "html"

    assert cache.get_or_render(snap, ("a", "act", 0), render) == "html"
    assert cache.get_or_render(snap, ("a", "act", 0), render) == "html"
    assert calls == 1
    cache.get_or_render(snap, ("a", "act", 1), render)
    assert calls == 2  # a different sample is a different entry


def test_render_cache_resets_on_new_snapshot() -> None:
    cache = _RenderCache()
    calls = 0

    def render() -> str:
        nonlocal calls
        calls += 1
        return "html"

    cache.get_or_render(_frame_snapshot(), ("a", "act", 0), render)
    cache.get_or_render(_frame_snapshot(), ("a", "act", 0), render)
    assert calls == 2  # a new snapshot object invalidates the old entries


def test_compute_frame_renders_strips_and_input() -> None:
    snap = _frame_snapshot()
    rendered, input_html = _compute_frame(
        ["x", "conv", "missing"],
        snap,
        0,
        input_name="x",
        input_mean=None,
        input_std=None,
        cache=_RenderCache(),
    )
    act, grad = rendered["conv"]
    assert "<img" in act and "<img" in grad
    assert rendered["x"][1] == ""  # the input has no gradient captured
    assert rendered["missing"] == ("", "")
    assert "<img" in input_html


def test_compute_frame_reuses_cache_within_a_snapshot() -> None:
    cache = _RenderCache()
    snap = _frame_snapshot()

    def frame(sample_idx: int) -> tuple[dict[str, tuple[str, str]], str]:
        return _compute_frame(
            ["conv"],
            snap,
            sample_idx,
            input_name="x",
            input_mean=None,
            input_std=None,
            cache=cache,
        )

    first, input_first = frame(0)
    again, input_again = frame(0)
    # Cache hits return the exact same strings, not re-rendered copies.
    assert again["conv"][0] is first["conv"][0]
    assert input_again is input_first
    other_sample, _ = frame(1)
    assert other_sample["conv"][0] is not first["conv"][0]


def test_compute_frame_renders_more_layers_than_pool_workers() -> None:
    # Exercise the render pool's queueing: more layers than max_workers.
    names = [f"l{i}" for i in range(20)]
    snap = BatchSnapshot(
        position=BatchPosition(
            phase="train",
            epoch=0,
            batch_idx=0,
            is_last_in_phase=False,
            is_last_in_epoch=False,
            is_last_overall=False,
        ),
        activations={name: torch.rand(1, 2, 4, 4) for name in names},
        activation_gradients={},
        weights={},
        weight_gradients={},
    )
    rendered, _ = _compute_frame(
        names,
        snap,
        0,
        input_name=None,
        input_mean=None,
        input_std=None,
        cache=_RenderCache(),
    )
    assert set(rendered) == set(names)
    assert all("<img" in rendered[name][0] for name in names)
