"""Tests for the visualization update frequency and auto experiment reruns."""

from __future__ import annotations

import pytest
import torch

from nansense.experiments import ExperimentResult
from nansense.session import BatchSnapshot, Session, UpdateFrequency
from tests.nansense.helpers import (
    TinyNet,
    make_position,
    make_session,
    paused_worker,
)


def _run_detached(
    session: Session, model: TinyNet, *, epochs: int, phases: dict[str, int]
) -> list[tuple[str, int, int]]:
    """Run the whole schedule detached, returning each publish's position."""
    session.detach()
    published: list[tuple[str, int, int]] = []
    for epoch in range(epochs):
        for phase, n in phases.items():
            for _ in range(n):
                before = session.snapshot
                with session.batch(phase=phase, epoch=epoch):
                    x = torch.randn(2, 4)
                    model.zero_grad(set_to_none=True)
                    model(x).sum().backward()
                snap = session.snapshot
                if snap is not None and snap is not before:
                    published.append(
                        (snap.position.phase, snap.position.epoch, snap.position.batch_idx)
                    )
    return published


def test_default_frequency_is_every_epoch() -> None:
    session, _ = make_session()
    assert session.update_frequency == UpdateFrequency(unit="epoch", n=1, phase=None)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"unit": "minute"}, "unknown frequency unit"),
        ({"unit": "batch", "n": 0}, "at least 1"),
        ({"unit": "epoch", "phase": "train"}, "only applies"),
        ({"unit": "batch", "phase": "nope"}, "unknown phase"),
    ],
)
def test_set_update_frequency_rejects_invalid(
    kwargs: dict[str, object], match: str
) -> None:
    session, _ = make_session()
    with pytest.raises(ValueError, match=match):
        session.set_update_frequency(**kwargs)  # ty: ignore[invalid-argument-type]


def test_epoch_frequency_publishes_at_nth_epoch_starts() -> None:
    phases = {"train": 2, "val": 2}
    session, model = make_session(epochs=4, phases=phases)
    session.set_update_frequency(unit="epoch", n=2)
    published = _run_detached(session, model, epochs=4, phases=phases)
    # Like Step Epoch, fires on the first batch of every 2nd epoch (0 and 2),
    # detected from the epoch number rather than the last batch.
    assert published == [("train", 0, 0), ("train", 2, 0)]


def test_epoch_frequency_fires_on_epoch_change_without_known_counts() -> None:
    """The epoch unit detects boundaries from the epoch number advancing — like
    Step Epoch — so it fires even during the first lazy epoch, where the batch
    count is unknown and `is_last_in_epoch` can never be set."""
    session, _ = make_session(epochs=3, phases={"train": 2})
    session.set_update_frequency(unit="epoch", n=1)
    # All is_last_* flags False, as during lazy phase-count discovery.
    fired = [
        session._should_freq_update(make_position("train", e, b))
        for e in range(3)
        for b in range(2)
    ]
    # First batch of each epoch fires; the rest of that epoch does not.
    assert fired == [True, False, True, False, True, False]


def test_batch_frequency_publishes_every_nth_batch() -> None:
    phases = {"train": 3, "val": 2}
    session, model = make_session(epochs=1, phases=phases)
    session.set_update_frequency(unit="batch", n=2)
    published = _run_detached(session, model, epochs=1, phases=phases)
    # Five batches overall; every 2nd one publishes.
    assert published == [("train", 0, 1), ("val", 0, 0)]


def test_batch_frequency_with_phase_counts_only_that_phase() -> None:
    phases = {"train": 3, "val": 3}
    session, model = make_session(epochs=1, phases=phases)
    session.set_update_frequency(unit="batch", n=2, phase="val")
    published = _run_detached(session, model, epochs=1, phases=phases)
    assert published == [("val", 0, 1)]


def test_capture_still_pauses_and_publishes_with_sparse_frequency() -> None:
    """Mode captures stay independent of the frequency setting."""
    session, model = make_session(epochs=1, phases={"train": 2})
    session.set_update_frequency(unit="batch", n=100)  # effectively never

    def loop() -> None:
        for _ in range(2):
            with session.batch(phase="train", epoch=0):
                x = torch.randn(2, 4)
                model.zero_grad(set_to_none=True)
                model(x).sum().backward()

    # The default STEP mode captures (and pauses) the very first batch.
    with paused_worker(session, loop, timeout=10.0):
        snap = session.snapshot
        assert snap is not None
        assert (snap.position.phase, snap.position.batch_idx) == ("train", 0)
        session.close()


def test_request_snapshot_publishes_only_the_next_free_running_batch() -> None:
    """The Refresh button (`request_snapshot`) arms a one-shot publish for the
    next free-running batch — refreshing activations, gradients, and weights —
    without pausing and without recomputing anything off the batch."""
    session, model = make_session(epochs=1, phases={"train": 3})
    session.set_update_frequency(unit="batch", n=100)  # effectively never
    session.detach()

    def run_batch(batch_idx: int) -> BatchSnapshot | None:
        before = session.snapshot
        with session.batch(phase="train", epoch=0):
            model.zero_grad(set_to_none=True)
            model(torch.randn(2, 4)).sum().backward()
        snap = session.snapshot
        return snap if snap is not None and snap is not before else None

    # Detached with no cadence due, a plain batch publishes nothing.
    assert run_batch(0) is None
    # Arming the request makes exactly the next batch publish a full snapshot.
    session.request_snapshot()
    published = run_batch(1)
    assert published is not None
    assert published.position.batch_idx == 1
    assert published.activations  # activations refreshed
    assert published.activation_gradients  # gradients refreshed
    assert published.weights  # weights refreshed
    # One-shot: the request is consumed, so the next batch publishes nothing.
    assert run_batch(2) is None


def test_auto_experiment_reruns_with_same_seq_on_updates() -> None:
    phases = {"train": 2}
    session, model = make_session(epochs=3, phases=phases)
    session.set_update_frequency(unit="epoch", n=1)
    seq = session.register_auto_experiment(
        "page-1",
        kind="deep_dream",
        layer="fc1",
        params={"steps": 1, "batch": 1, "mean": None, "std": None},
    )

    session.detach()
    results: list[ExperimentResult] = []
    fixed_x = torch.randn(2, 4)
    for epoch in range(3):
        session.touch_auto_experiment("page-1")
        for _ in range(2):
            with session.batch(phase="train", epoch=epoch):
                model.zero_grad(set_to_none=True)
                model(fixed_x).sum().backward()
        result = session.experiment_result_for(seq)
        assert result is not None
        assert result.done
        assert result.seq == seq  # reruns keep the seq
        results.append(result)

    # Each epoch-end update re-ran the experiment: a fresh result object.
    assert results[0] is not results[1] is not results[2]
    # The seeded noise start is derived from the seq, which never changes —
    # with identical input data, every rerun starts from the same tensor.
    first = results[0].reference
    last = results[2].reference
    assert first is not None and last is not None
    assert torch.equal(first, last)


def test_auto_experiment_expires_without_heartbeat() -> None:
    phases = {"train": 1}
    session, model = make_session(epochs=2, phases=phases)
    seq = session.register_auto_experiment(
        "page-1",
        kind="deep_dream",
        layer="fc1",
        params={"steps": 1, "batch": 1, "mean": None, "std": None},
    )
    session.detach()
    with session.batch(phase="train", epoch=0):
        model.zero_grad(set_to_none=True)
        model(torch.randn(2, 4)).sum().backward()
    first = session.experiment_result_for(seq)
    assert first is not None and first.done

    # Simulate the page going away: expire the registration, then update.
    with session._cv:
        session._auto_experiments["page-1"].expires_at = 0.0
    with session.batch(phase="train", epoch=1):
        model.zero_grad(set_to_none=True)
        model(torch.randn(2, 4)).sum().backward()
    assert session.experiment_result_for(seq) is first  # no rerun happened
    with session._cv:
        assert "page-1" not in session._auto_experiments


def test_pinned_auto_experiment_survives_expiry_check() -> None:
    session, model = make_session(epochs=2, phases={"train": 1})
    seq = session.register_auto_experiment(
        "page-1",
        kind="deep_dream",
        layer="fc1",
        params={"steps": 1, "batch": 1, "mean": None, "std": None},
    )
    assert session.pin_auto_experiment("page-1")
    session.detach()
    with session.batch(phase="train", epoch=0):
        model.zero_grad(set_to_none=True)
        model(torch.randn(2, 4)).sum().backward()
    first = session.experiment_result_for(seq)

    # Even with a stale heartbeat a pinned registration keeps re-running.
    with session.batch(phase="train", epoch=1):
        model.zero_grad(set_to_none=True)
        model(torch.randn(2, 4)).sum().backward()
    assert session.experiment_result_for(seq) is not first


def test_should_freq_update_mutates_counter_under_lock() -> None:
    """The per-batch counter advance happens inside the `_cv` critical section
    (alongside the modulo check), so `set_update_frequency`'s reset to 0 can't
    be clobbered by a concurrent unlocked increment. The cadence itself stays
    correct: every n-th frequency-eligible batch reports True."""
    session, _ = make_session(epochs=1, phases={"train": 6})
    session.set_update_frequency(unit="batch", n=3)
    fired = [
        session._should_freq_update(make_position("train", 0, i)) for i in range(6)
    ]
    assert fired == [False, False, True, False, False, True]


def test_rewind_resets_the_frequency_counter() -> None:
    """A time-travel jump restarts the per-batch cadence so post-jump frames
    fire on a clean phase rather than wherever the abandoned timeline left it."""
    session, _ = make_session(epochs=3, phases={"train": 4})
    session.set_update_frequency(unit="batch", n=2)
    session._should_freq_update(make_position("train", 0, 0))  # counter -> 1
    assert session._freq_counter == 1
    session._rewind_to_epoch(1)
    assert session._freq_counter == 0


def test_rewind_resets_epoch_frequency_tracking() -> None:
    """A jump clears the last-seen epoch, so re-entering it counts as a fresh
    boundary and fires again (the abandoned timeline's progress is forgotten)."""
    session, _ = make_session(epochs=3, phases={"train": 2})
    session.set_update_frequency(unit="epoch", n=1)
    assert session._should_freq_update(make_position("train", 1, 0))  # new epoch
    assert not session._should_freq_update(make_position("train", 1, 1))
    session._rewind_to_epoch(1)
    assert session._freq_epoch is None
    assert session._should_freq_update(make_position("train", 1, 0))  # fires anew
