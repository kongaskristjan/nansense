"""Tests for pure helpers in `playgrad.ui.app`."""

from __future__ import annotations

import pytest
import torch

import playgrad
from playgrad.probe import ProbeResult
from playgrad.schedule import BatchPosition, Schedule
from playgrad.session import BatchSnapshot
from playgrad.ui.app import (
    _BIN_WIDTHS,
    _DENSITY_TOP_BINS,
    _PLOT_HEIGHT,
    _PROBE_NO_GRADIENTS_HTML,
    _RenderCache,
    _compute_frame,
    _display_batch_size,
    _default_roles,
    _density_heights,
    _density_y_range,
    _dims_from_roles,
    _format_live_position,
    _input_img_tag,
    _linear_x_range,
    _make_histogram_figure,
    _phases_with_data,
    _role_options,
    _strip_html,
    _summarize_epoch_ranges,
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
def test_histogram_log_y_toggles_count_axis(log_y: bool) -> None:
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


# --- Watching histogram: density mode (both axes linear) -------------------


def test_histogram_defaults_to_linear_density_mode() -> None:
    # Log x / Log y start unchecked, so the no-args figure is the density view.
    per_phase = {"train": _layer_snap("train")}
    fig = _make_histogram_figure(per_phase, "activation", "activations")
    assert fig.layout.yaxis.type == "linear"
    assert fig.layout.yaxis.title.text == "density"


@pytest.mark.parametrize(
    "log_x, log_y, expected",
    [
        (True, True, False),
        (True, False, False),
        (False, True, False),
        (False, False, True),
    ],
)
def test_use_density_only_when_both_axes_linear(
    log_x: bool, log_y: bool, expected: bool
) -> None:
    assert _use_density(log_x, log_y) is expected


def test_density_heights_divide_counts_by_bin_width() -> None:
    hist = [0] * N_BINS
    hist[ZERO_BIN] = 4
    hist[ZERO_BIN + 10] = 6
    heights = _density_heights(tuple(hist))
    assert heights[ZERO_BIN] == pytest.approx(4 / _BIN_WIDTHS[ZERO_BIN])
    assert heights[ZERO_BIN + 10] == pytest.approx(6 / _BIN_WIDTHS[ZERO_BIN + 10])
    assert heights[0] == 0


def test_density_y_range_caps_at_20th_tallest_bar() -> None:
    # 30 populated bins: the cap sits at the 20th-tallest density, so the
    # taller bars (here, narrow near-zero bins) clip instead of stretching
    # the scale.
    hist = {ZERO_BIN + 1 + i: (i + 1) ** 2 for i in range(30)}
    per_phase = {"train": _layer_snap("train", hist=hist)}
    heights = sorted(
        (h for h in _density_heights(per_phase["train"].activations.hist) if h > 0),
        reverse=True,
    )
    rng = _density_y_range(per_phase, "activation")
    assert rng is not None
    assert rng[0] == 0.0
    assert rng[1] == pytest.approx(heights[_DENSITY_TOP_BINS - 1] * 1.05)
    assert rng[1] < heights[0]  # the tallest bar does clip


def test_density_y_range_pools_phases() -> None:
    # With two phases the 20-tallest pool spans both traces.
    per_phase = {
        "train": _layer_snap("train", hist={ZERO_BIN + 1 + i: 100 for i in range(15)}),
        "val": _layer_snap("val", hist={ZERO_BIN + 30 + i: 1 for i in range(15)}),
    }
    all_heights = sorted(
        (
            h
            for phase in per_phase.values()
            for h in _density_heights(phase.activations.hist)
            if h > 0
        ),
        reverse=True,
    )
    rng = _density_y_range(per_phase, "activation")
    assert rng is not None
    assert rng[1] == pytest.approx(all_heights[_DENSITY_TOP_BINS - 1] * 1.05)


def test_density_y_range_with_few_bars_uses_smallest() -> None:
    # Fewer than 20 populated bins: the "biggest 20" are all of them, so the
    # cap is the smallest populated bar (the giant zero-band spike clips).
    hist = {ZERO_BIN: 100, ZERO_BIN + 20: 5}
    per_phase = {"train": _layer_snap("train", hist=hist)}
    heights = [
        h for h in _density_heights(per_phase["train"].activations.hist) if h > 0
    ]
    rng = _density_y_range(per_phase, "activation")
    assert rng is not None
    assert rng[1] == pytest.approx(min(heights) * 1.05)


def test_density_y_range_none_when_empty() -> None:
    assert _density_y_range({}, "activation") is None


def test_histogram_density_mode_plots_density_with_capped_axis() -> None:
    hist = {ZERO_BIN: 50, ZERO_BIN + 10: 7}
    per_phase = {"train": _layer_snap("train", hist=hist)}
    fig = _make_histogram_figure(
        per_phase, "activation", "activations", log_x=False, log_y=False
    )
    trace = fig.data[0]
    expected = _density_heights(per_phase["train"].activations.hist)
    assert list(trace.y) == pytest.approx(expected)
    # Raw counts ride along for the hover text.
    assert trace.customdata[ZERO_BIN] == 50
    assert fig.layout.yaxis.title.text == "density"
    expected_range = _density_y_range(per_phase, "activation")
    assert expected_range is not None
    assert tuple(fig.layout.yaxis.range) == tuple(expected_range)


@pytest.mark.parametrize(
    "log_x, log_y", [(True, True), (True, False), (False, True)]
)
def test_histogram_keeps_counts_on_other_axis_combos(
    log_x: bool, log_y: bool
) -> None:
    per_phase = {"train": _layer_snap("train", n=9)}
    fig = _make_histogram_figure(
        per_phase, "activation", "activations", log_x=log_x, log_y=log_y
    )
    assert fig.data[0].y[ZERO_BIN] == 9
    assert fig.layout.yaxis.title.text == "count"
    assert fig.layout.yaxis.range is None


# --- Watching histogram: plot height (task 3) ------------------------------


def test_histogram_height_is_doubled() -> None:
    fig = _make_histogram_figure({}, "activation", "activations")
    assert _PLOT_HEIGHT == 440  # 2x the original 220
    assert fig.layout.height == _PLOT_HEIGHT


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
        None,
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
            None,
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


def _frame_probe() -> ProbeResult:
    return ProbeResult(
        input=torch.rand(2, 3, 4, 4),
        activations={"x": torch.rand(2, 3, 4, 4), "conv": torch.rand(2, 2, 4, 4)},
        mode="eval",
    )


def test_compute_frame_prefers_probe_over_snapshot() -> None:
    probe = _frame_probe()
    rendered, input_html = _compute_frame(
        ["x", "conv", "missing"],
        _frame_snapshot(),
        probe,
        0,
        input_name="x",
        input_mean=None,
        input_std=None,
        cache=_RenderCache(),
    )
    act, grad = rendered["conv"]
    assert "<img" in act
    # Probes are forward-only: every gradient strip is the placeholder note.
    assert grad == _PROBE_NO_GRADIENTS_HTML
    assert rendered["missing"][0] == ""
    assert "<img" in input_html


def test_display_batch_size_prefers_probe() -> None:
    snap = _frame_snapshot()  # batch size 2
    probe = ProbeResult(
        input=torch.rand(5, 3, 4, 4), activations={}, mode="eval"
    )
    assert _display_batch_size(snap, probe) == 5
    assert _display_batch_size(snap, None) == 2
    assert _display_batch_size(None, None) is None


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
        None,
        0,
        input_name=None,
        input_mean=None,
        input_std=None,
        cache=_RenderCache(),
    )
    assert set(rendered) == set(names)
    assert all("<img" in rendered[name][0] for name in names)
