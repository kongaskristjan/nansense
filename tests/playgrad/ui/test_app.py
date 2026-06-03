"""Tests for pure helpers in `playgrad.ui.app`."""

from __future__ import annotations

import pytest
import torch

import playgrad
from playgrad.schedule import BatchPosition, Schedule
from playgrad.session import BatchSnapshot
from playgrad.ui.app import (
    _PLOT_HEIGHT,
    _default_roles,
    _dims_from_roles,
    _format_live_position,
    _linear_x_range,
    _make_histogram_figure,
    _phases_with_data,
    _role_options,
    _validate_step_until_target,
    serve,
)
from playgrad.watch import N_BINS, ZERO_BIN, LayerStatsSnapshot, TensorStatsSnapshot


def _tensor_stats(n: int) -> TensorStatsSnapshot:
    hist = [0] * N_BINS
    hist[ZERO_BIN] = n
    return TensorStatsSnapshot(
        n=n, sum=0.0, sum_sq=0.0, min=0.0, max=0.0, hist=tuple(hist)
    )


def _layer_snap(phase: str, epoch: int = 0, n: int = 10) -> LayerStatsSnapshot:
    return LayerStatsSnapshot(
        layer="L",
        phase=phase,
        epoch=epoch,
        activations=_tensor_stats(n),
        gradients=_tensor_stats(n),
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
