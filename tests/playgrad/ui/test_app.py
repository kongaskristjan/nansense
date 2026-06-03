"""Tests for pure helpers in `playgrad.ui.app`."""

from __future__ import annotations

import pytest
import torch

from playgrad.schedule import BatchPosition, Schedule
from playgrad.session import BatchSnapshot
from playgrad.ui.app import (
    _default_roles,
    _dims_from_roles,
    _format_live_position,
    _role_options,
    _validate_step_until_target,
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


