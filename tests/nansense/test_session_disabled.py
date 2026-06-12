"""Tests for disabled sessions (enabled=False): no tracing, capture, or pauses."""

from __future__ import annotations

import nansense
from nansense.session import _BatchContext
from tests.nansense.helpers import TinyNet, train_step


def test_start_enabled_by_default() -> None:
    model = TinyNet()
    session = nansense.start(model, epochs=1, phases={"train": 1})
    assert session.enabled is True
    assert session.fx_traced is True  # TinyNet traces cleanly
    assert session.layer_names  # non-empty


def test_disabled_session_skips_trace_and_name_discovery() -> None:
    """`enabled=False` skips the fx trace and leaves the name lists empty."""
    model = TinyNet()
    session = nansense.start(model, epochs=1, phases={"train": 1}, enabled=False)

    assert session.enabled is False
    assert session.fx_traced is False  # would be True if we had traced
    assert session.input_names == []
    assert session.layer_names == []
    assert session.layer_weights == {}
    # Nothing is watchable on a disabled session.
    assert session.watch("anything") is False
    assert session.watched_layers == frozenset()


def test_disabled_session_batch_captures_nothing_and_never_pauses() -> None:
    """A disabled batch runs the user body but installs no hooks and never blocks.

    The whole loop runs on the main thread: if a disabled batch paused (as an
    enabled STEP-mode batch would), this test would hang instead of completing.
    """
    model = TinyNet()
    session = nansense.start(model, epochs=1, phases={"train": 2}, enabled=False)

    for _ in range(2):
        with session.batch(phase="train", epoch=0) as ctx:
            train_step(model)
            assert isinstance(ctx, _BatchContext)
            assert ctx.captured is False
            assert ctx.position is None

    assert session.snapshot is None
    assert session.live_position is None
    assert session.pause_count == 0
    # forward was never patched, no hooks left installed.
    assert "forward" not in model.__dict__


def test_disabled_session_does_not_advance_the_schedule() -> None:
    """Disabled batches skip `schedule.advance`, so the declared batch count
    is never enforced — an enabled session would raise on the 2nd batch here."""
    model = TinyNet()
    session = nansense.start(model, epochs=1, phases={"train": 1}, enabled=False)

    for _ in range(5):  # far more than the single declared batch
        with session.batch(phase="train", epoch=0):
            train_step(model)


def test_batches_yields_loader_items_unchanged_when_disabled() -> None:
    model = TinyNet()
    session = nansense.start(model, epochs=1, phases={"train": 2}, enabled=False)
    assert list(session.batches([1, 2, 3], phase="train", epoch=0)) == [1, 2, 3]
