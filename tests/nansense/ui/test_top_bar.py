"""Tests for step-until validation, default positions, and epoch summaries in nansense.ui.top_bar."""

from __future__ import annotations

import pytest
import torch

from nansense.schedule import Schedule
from nansense.session import BatchSnapshot
from nansense.ui.top_bar import (
    _current_position,
    _summarize_epoch_ranges,
    _time_travel_default_index,
    _validate_step_until_target,
)
from tests.nansense.helpers import _make_snapshot, make_position


def _snapshot_at(phase: str, epoch: int, batch_idx: int) -> BatchSnapshot:
    return _make_snapshot(phase, epoch, batch_idx, activations={"x": torch.zeros(1)})


@pytest.fixture
def schedule() -> Schedule:
    return Schedule(epochs=3, phases={"train": 5, "val": 2})


def test_validate_passes_for_future_position(schedule: Schedule) -> None:
    snap = _snapshot_at("train", 0, 1)
    assert (
        _validate_step_until_target(
            schedule=schedule,
            live_position=None,
            snapshot=snap,
            phase="val",
            epoch=0,
            batch_idx=0,
        )
        is None
    )


def test_validate_passes_when_no_snapshot_yet(schedule: Schedule) -> None:
    assert (
        _validate_step_until_target(
            schedule=schedule,
            live_position=None,
            snapshot=None,
            phase="train",
            epoch=0,
            batch_idx=0,
        )
        is None
    )


def test_validate_uses_live_position_over_stale_snapshot(schedule: Schedule) -> None:
    """The live position runs ahead of the snapshot during step_epoch/run/detach.

    `step_until_position` only captures on an *exact* (phase, epoch, batch_idx)
    match against the live position, so a target between the stale snapshot and
    the live position can never be hit — it must be rejected, not silently let
    training run to the end.
    """
    snap = _snapshot_at("train", 0, 0)
    live = make_position("train", 1, 3)
    # Between snapshot (train,0,0) and live (train,1,3): already passed.
    assert (
        _validate_step_until_target(
            schedule=schedule,
            live_position=live,
            snapshot=snap,
            phase="train",
            epoch=0,
            batch_idx=4,
        )
        == "Target must be after the current position"
    )
    # Strictly ahead of the live position: accepted.
    assert (
        _validate_step_until_target(
            schedule=schedule,
            live_position=live,
            snapshot=snap,
            phase="train",
            epoch=2,
            batch_idx=0,
        )
        is None
    )


@pytest.mark.parametrize(
    ("current", "phase", "epoch", "batch_idx", "expected"),
    [
        pytest.param(
            ("train", 0, 2), "train", 0, 0, "after the current", id="before-current"
        ),
        pytest.param(
            ("train", 0, 2),
            "train",
            0,
            1,
            "after the current",
            id="just-before-current",
        ),
        pytest.param(
            ("train", 0, 2), "train", 0, 2, "after the current", id="at-current"
        ),
        pytest.param(
            ("val", 0, 0),
            "train",
            0,
            4,
            "after the current",
            id="earlier-phase-same-epoch",
        ),
        pytest.param(None, "bogus", 0, 0, "Unknown phase", id="unknown-phase"),
        pytest.param(None, "train", 3, 0, "Epoch", id="epoch-out-of-range"),
        pytest.param(None, "train", 0, 5, "Batch", id="batch-out-of-range"),
    ],
)
def test_validate_step_until_rejects_invalid_targets(
    schedule: Schedule,
    current: tuple[str, int, int] | None,
    phase: str,
    epoch: int,
    batch_idx: int,
    expected: str,
) -> None:
    """`current` is the last-captured position (None: nothing captured yet)."""
    snap = _snapshot_at(*current) if current is not None else None
    msg = _validate_step_until_target(
        schedule=schedule,
        live_position=None,
        snapshot=snap,
        phase=phase,
        epoch=epoch,
        batch_idx=batch_idx,
    )
    assert msg is not None
    assert expected in msg


@pytest.mark.parametrize(
    "live, snapshot, expected",
    [
        # Live position wins over the snapshot's frozen position.
        (("val", 1, 3), ("train", 0, 0), ("val", 1, 3)),
        # No live position yet: fall back to the snapshot.
        (None, ("train", 2, 4), ("train", 2, 4)),
        # Nothing published yet: keep the dialog's existing values.
        (None, None, None),
    ],
)
def test_current_position(
    live: tuple[str, int, int] | None,
    snapshot: tuple[str, int, int] | None,
    expected: tuple[str, int, int] | None,
) -> None:
    result = _current_position(
        make_position(*live) if live is not None else None,
        _snapshot_at(*snapshot) if snapshot is not None else None,
    )
    if expected is None:
        assert result is None
    else:
        assert result is not None
        assert (result.phase, result.epoch, result.batch_idx) == expected


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


@pytest.mark.parametrize(
    ("cached", "current_epoch", "expected"),
    [
        # Current epoch is cached: preselect it, not the last cached epoch
        # (epochs past a backwards jump keep their checkpoints on disk).
        pytest.param([0, 1, 2, 3, 4], 2, 2, id="current-cached"),
        pytest.param([0, 1, 2], 2, 2, id="current-is-last"),
        # Current epoch missing from a gappy cache: closest one before it.
        pytest.param([0, 1, 5, 6], 3, 1, id="gap-rounds-down"),
        # Everything cached is in the future: clamp to the earliest.
        pytest.param([4, 5], 2, 0, id="all-cached-ahead"),
        # No position published yet: fall back to the last cached epoch.
        pytest.param([0, 1, 2], None, 2, id="no-position-yet"),
    ],
)
def test_time_travel_default_index(
    cached: list[int], current_epoch: int | None, expected: int
) -> None:
    assert _time_travel_default_index(cached, current_epoch) == expected
