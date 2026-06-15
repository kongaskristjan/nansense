"""Tests for the per-channel watch performance caps (`set_watch_performance`)."""

from __future__ import annotations

import nansense
from nansense.patches import DEFAULT_SAMPLES_PER_CHANNEL
from nansense.session import Session, WatchPerformance
from nansense.watch import DEFAULT_CHANNEL_LIMIT
from tests.nansense.helpers import TinyNet, paused_session


def test_default_watch_performance() -> None:
    session = nansense.start(TinyNet(), epochs=1, phases={"train": 1})
    perf = session.watch_performance
    assert perf == WatchPerformance(
        channel_limit_enabled=True,
        channel_limit=DEFAULT_CHANNEL_LIMIT,
        samples_per_channel=DEFAULT_SAMPLES_PER_CHANNEL,
    )


def test_partial_update_keeps_other_fields() -> None:
    session = nansense.start(TinyNet(), epochs=1, phases={"train": 1})
    session.set_watch_performance(channel_limit=8)
    perf = session.watch_performance
    assert perf.channel_limit == 8
    assert perf.channel_limit_enabled is True
    assert perf.samples_per_channel == DEFAULT_SAMPLES_PER_CHANNEL


def test_values_are_clamped_to_at_least_one() -> None:
    session = nansense.start(TinyNet(), epochs=1, phases={"train": 1})
    session.set_watch_performance(channel_limit=0, samples_per_channel=-3)
    perf = session.watch_performance
    assert perf.channel_limit == 1
    assert perf.samples_per_channel == 1


def test_flush_return_value() -> None:
    session = nansense.start(TinyNet(), epochs=1, phases={"train": 1})
    # A real change to the caps flushes; re-applying the same values does not.
    assert session.set_watch_performance(channel_limit=8) is True
    assert session.set_watch_performance(channel_limit=8) is False
    # Toggling the limit off changes the effective cap (8 -> None) -> flush.
    assert session.set_watch_performance(channel_limit_enabled=False) is True
    assert session.set_watch_performance(channel_limit_enabled=False) is False
    # The channel count is irrelevant while disabled, so editing it is a no-op.
    assert session.set_watch_performance(channel_limit=4) is False


def _fc1_channel_rows(session: Session) -> int | None:
    snap = session.watch_snapshot()
    for key, layer in snap.stats.items():
        if key[0] == "fc1":
            rows = layer.activations.channel_hists
            return None if rows is None else len(rows)
    return None


def test_channel_limit_caps_recorded_rows_end_to_end() -> None:
    """fc1 outputs 8 features; a limit of 3 keeps only 3 per-channel rows."""
    with paused_session(TinyNet()) as session:
        session.watch("fc1")
        session.set_watch_performance(channel_limit=3)
        session.step_batch()
        assert session.wait_until_paused(after_pauses=1, timeout=5)
        assert _fc1_channel_rows(session) == 3


def test_disabling_limit_records_all_channels_end_to_end() -> None:
    with paused_session(TinyNet()) as session:
        session.watch("fc1")
        session.set_watch_performance(channel_limit_enabled=False)
        session.step_batch()
        assert session.wait_until_paused(after_pauses=1, timeout=5)
        assert _fc1_channel_rows(session) == 8
