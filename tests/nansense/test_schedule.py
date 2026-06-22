"""Tests for the Schedule / BatchPosition machinery."""

from __future__ import annotations

import threading

import pytest

from nansense.schedule import Schedule, format_position
from tests.nansense.helpers import make_position


@pytest.mark.parametrize(
    ("phase", "epoch", "batch_idx", "total_epochs", "total_batches", "expected"),
    [
        # No totals known: bare, as before.
        ("train", 0, 0, None, None, "epoch 0 | train batch 0"),
        ("val", 12, 345, None, None, "epoch 12 | val batch 345"),
        # Both totals known: "/N" suffix on each (numbers stay 0-indexed).
        ("train", 26, 5, 50, 196, "epoch 26/50 | train batch 5/196"),
        # Only one total known: the other half stays bare.
        ("train", 3, 7, 50, None, "epoch 3/50 | train batch 7"),
        ("val", 3, 7, None, 40, "epoch 3 | val batch 7/40"),
    ],
)
def test_format_position(
    phase: str,
    epoch: int,
    batch_idx: int,
    total_epochs: int | None,
    total_batches: int | None,
    expected: str,
) -> None:
    assert (
        format_position(
            make_position(phase, epoch, batch_idx),
            total_epochs=total_epochs,
            total_batches=total_batches,
        )
        == expected
    )


def test_advance_through_full_run() -> None:
    schedule = Schedule(epochs=2, phases={"train": 3, "val": 2})

    positions = []
    for epoch in range(2):
        for phase, n in [("train", 3), ("val", 2)]:
            for _ in range(n):
                positions.append(schedule.advance(phase, epoch))

    assert [(p.phase, p.epoch, p.batch_idx) for p in positions] == [
        ("train", 0, 0), ("train", 0, 1), ("train", 0, 2),
        ("val", 0, 0), ("val", 0, 1),
        ("train", 1, 0), ("train", 1, 1), ("train", 1, 2),
        ("val", 1, 0), ("val", 1, 1),
    ]
    last_in_phase = [p.is_last_in_phase for p in positions]
    assert last_in_phase == [
        False, False, True,
        False, True,
        False, False, True,
        False, True,
    ]
    assert sum(p.is_last_in_epoch for p in positions) == 2
    assert sum(p.is_last_overall for p in positions) == 1
    assert positions[-1].is_last_overall


@pytest.mark.parametrize(
    ("epochs", "phases"),
    [
        (0, {"train": 1}),
        (-1, {"train": 1}),
        (1, {}),
        (1, {"train": 0}),
        (1, {"train": -2}),
    ],
)
def test_invalid_construction(epochs: int, phases: dict[str, int]) -> None:
    with pytest.raises(ValueError):
        Schedule(epochs=epochs, phases=phases)


def test_unknown_phase_raises() -> None:
    schedule = Schedule(epochs=1, phases={"train": 2})
    with pytest.raises(ValueError, match="unknown phase"):
        schedule.advance("val", 0)


def test_out_of_range_epoch_raises() -> None:
    schedule = Schedule(epochs=2, phases={"train": 2})
    with pytest.raises(ValueError, match="out of range"):
        schedule.advance("train", 2)


def test_too_many_batches_raises() -> None:
    schedule = Schedule(epochs=1, phases={"train": 2})
    schedule.advance("train", 0)
    schedule.advance("train", 0)
    with pytest.raises(ValueError, match="more batches than declared"):
        schedule.advance("train", 0)


def test_update_changes_counts() -> None:
    schedule = Schedule(epochs=1, phases={"train": 2})
    schedule.update(phases={"train": 5})
    assert schedule.phases == {"train": 5}
    schedule.advance("train", 0)
    schedule.advance("train", 0)
    schedule.advance("train", 0)


def test_last_phase_name_follows_insertion_order() -> None:
    schedule = Schedule(epochs=1, phases={"train": 1, "val": 1})
    assert schedule.last_phase_name == "val"
    schedule.update(phases={"val": 1, "train": 1})
    assert schedule.last_phase_name == "train"


def test_first_phase_name_follows_insertion_order() -> None:
    schedule = Schedule(epochs=1, phases={"train": 1, "val": 1})
    assert schedule.first_phase_name == "train"
    schedule.update(phases={"val": 1, "train": 1})
    assert schedule.first_phase_name == "val"


def test_rewind_to_epoch_resets_counters_for_that_epoch_onward() -> None:
    schedule = Schedule(epochs=3, phases={"train": 2})
    for epoch in range(3):
        schedule.advance("train", epoch)
        schedule.advance("train", epoch)

    schedule.rewind_to_epoch(1)
    # Epochs 1 and 2 advance from batch 0 again...
    assert schedule.advance("train", 1).batch_idx == 0
    assert schedule.advance("train", 2).batch_idx == 0
    # ...while epoch 0's counters survive the rewind.
    with pytest.raises(ValueError, match="more batches than declared"):
        schedule.advance("train", 0)


# -- lazy mode (phases discovered by observation) ---------------------------


def test_lazy_construction_starts_empty() -> None:
    schedule = Schedule()
    assert schedule.epochs is None
    assert schedule.phases == {}
    assert schedule.phase_order == []
    assert schedule.first_phase_name is None
    assert schedule.last_phase_name is None


def test_lazy_advance_registers_phases_without_error() -> None:
    schedule = Schedule(epochs=2)
    # Unknown phases are learned, not rejected; counts unknown so no is_last.
    pos = schedule.advance("train", 0)
    assert (pos.phase, pos.batch_idx) == ("train", 0)
    assert not pos.is_last_in_phase
    schedule.advance("train", 0)
    schedule.advance("val", 0)  # a second phase, discovered on first sight
    assert schedule.phase_order == ["train", "val"]
    # No counts learned yet, so `phases` (counted phases only) stays empty.
    assert schedule.phases == {}
    assert schedule.phase_count("train") is None


def test_lazy_counts_learned_after_first_epoch() -> None:
    schedule = Schedule(epochs=2)
    # Epoch 0: counts unknown, is_last_in_phase never fires.
    for _ in range(3):
        assert not schedule.advance("train", 0).is_last_in_phase
    schedule.record_phase_length("train", 0, 3)
    for _ in range(2):
        assert not schedule.advance("val", 0).is_last_in_phase
    schedule.record_phase_length("val", 0, 2)
    assert schedule.phases == {"train": 3, "val": 2}
    # Epoch 1: counts known → flags fire on the true last batches.
    train = [schedule.advance("train", 1) for _ in range(3)]
    assert [p.is_last_in_phase for p in train] == [False, False, True]
    assert not train[-1].is_last_in_epoch  # val still follows
    val = [schedule.advance("val", 1) for _ in range(2)]
    assert val[-1].is_last_in_phase and val[-1].is_last_in_epoch
    assert val[-1].is_last_overall  # last epoch, last phase, last batch


def test_lazy_tolerates_growing_dataset() -> None:
    schedule = Schedule(epochs=3)
    schedule.record_phase_length("train", 0, 2)  # learned 2 from epoch 0
    # Epoch 1 runs 3 batches (dataset grew): must not raise.
    for _ in range(3):
        schedule.advance("train", 1)
    schedule.record_phase_length("train", 1, 3)
    assert schedule.phase_count("train") == 3


def test_set_epochs_enables_is_last_overall() -> None:
    schedule = Schedule()
    schedule.record_phase_length("train", 0, 1)
    # epochs unknown → is_last_overall can't be True even on the last batch.
    assert not schedule.advance("train", 0).is_last_overall
    schedule.rewind_to_epoch(0)  # reset the counter to re-advance batch 0
    schedule.set_epochs(1)
    assert schedule.advance("train", 0).is_last_overall


def test_advance_is_thread_safe_under_concurrency() -> None:
    """`advance` runs on the training thread while the UI thread reads/updates
    the schedule; concurrent advances must not lose increments. Many threads
    sharing one (phase, epoch) should be handed every batch index exactly
    once — no duplicates from a torn read-modify-write of the counter."""
    n_threads, per_thread = 8, 50
    declared = n_threads * per_thread
    schedule = Schedule(epochs=1, phases={"train": declared})
    seen: list[int] = []
    seen_lock = threading.Lock()

    def worker() -> None:
        local = [schedule.advance("train", 0).batch_idx for _ in range(per_thread)]
        with seen_lock:
            seen.extend(local)

    threads = [threading.Thread(target=worker) for _ in range(n_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert sorted(seen) == list(range(declared))
