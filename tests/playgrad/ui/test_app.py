"""Tests for pure helpers in `playgrad.ui.app`."""

from __future__ import annotations

import pytest
import torch

from playgrad.schedule import BatchPosition, Schedule
from playgrad.session import BatchSnapshot
from playgrad.ui.app import (
    _curve_number,
    _linear_x_range,
    _make_histogram_figure,
    _validate_step_until_target,
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


# --- Watching histogram: legend-selection persistence (task 2) -------------


def test_histogram_returns_traces_in_phase_order() -> None:
    per_phase = {"train": _layer_snap("train"), "val": _layer_snap("val")}
    fig, phases = _make_histogram_figure(per_phase, "activation", "activations")
    assert phases == ["train", "val"]
    assert [t.name for t in fig.data] == ["train (ep 0)", "val (ep 0)"]
    # No hidden phases → every trace is visible.
    assert all(t.visible in (True, None) for t in fig.data)


def test_histogram_skips_empty_phases() -> None:
    per_phase = {"train": _layer_snap("train", n=5), "val": _layer_snap("val", n=0)}
    _fig, phases = _make_histogram_figure(per_phase, "activation", "activations")
    assert phases == ["train"]


def test_histogram_hidden_phase_is_legendonly() -> None:
    per_phase = {"train": _layer_snap("train"), "val": _layer_snap("val")}
    fig, _phases = _make_histogram_figure(
        per_phase, "activation", "activations", hidden=frozenset({"train"})
    )
    by_phase = {t.name.split(" ")[0]: t.visible for t in fig.data}
    assert by_phase["train"] == "legendonly"  # stays hidden across refresh
    assert by_phase["val"] in (True, None)


@pytest.mark.parametrize(
    "args, expected",
    [
        ({"curveNumber": 0}, 0),
        ({"curveNumber": 2}, 2),
        ([1], 1),
        (3, 3),
        ({}, None),
        ({"curveNumber": True}, None),  # bool is not a curve index
        ("nope", None),
        (None, None),
    ],
)
def test_curve_number_parsing(args: object, expected: int | None) -> None:
    assert _curve_number(args) == expected


# --- Watching histogram: log/linear axis toggles (task 1) ------------------


@pytest.mark.parametrize("log_y", [True, False])
def test_histogram_log_y_toggles_count_axis(log_y: bool) -> None:
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
    per_phase = {"train": _layer_snap("train")}
    fig, _ = _make_histogram_figure(
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
