"""Settings components load without writes and apply complete user choices."""

from __future__ import annotations

import pytest
from nicegui import ui

from nansense.contracts.recording import MainView, RecordedView
from nansense.session import StatsScope
from nansense.ui.settings import SettingsDialog
from tests.nansense.helpers import make_session


def test_opening_settings_does_not_apply_partially_loaded_values(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session, _ = make_session(epochs=2, phases={"train": 2, "val": 1})
    session.set_stats_scope(StatsScope.ALL)
    session.set_auto_run_experiments(False)
    session.set_watch_performance(channel_limit=3, samples_per_channel=2)
    session.set_update_frequency(unit="batch", n=3, phase="val")
    session.set_debug_settings(enabled=True, interval=7, threshold=0.2)

    def unexpected_write(*args: object, **kwargs: object) -> None:
        pytest.fail("Opening settings must not write back to the session")

    for setter in (
        "set_stats_scope",
        "set_auto_run_experiments",
        "set_watch_performance",
        "set_update_frequency",
        "set_debug_settings",
    ):
        monkeypatch.setattr(session, setter, unexpected_write)
    with ui.card():
        dialog = SettingsDialog(session, None, lambda update: update())
    dialog.open()
    assert dialog.collection.scope_select.value == "all"
    assert not dialog.collection.auto_run_switch.value
    assert dialog.watch.channel_limit_input.value == 3
    assert dialog.frequency.phase_select.value == "val"
    assert dialog.debug.debug_interval.value == 7
    assert dialog.debug.debug_threshold.value == 20


def test_frequency_changes_validate_and_recordings_disable_the_controls() -> None:
    session, _ = make_session(epochs=2, phases={"train": 2, "val": 1})
    with ui.card():
        dialog = SettingsDialog(session, None, lambda update: update())
    dialog.open()
    frequency = dialog.frequency
    frequency.unit_select.value = "batch"
    frequency.phase_select.value = "val"
    frequency.n_input.value = 3
    assert session.update_frequency.phase == "val"
    assert session.update_frequency.n == 3
    frequency.unit_select.value = "epoch"
    assert session.update_frequency.phase is None
    assert not frequency.phase_select.visible
    frequency.n_input.value = 0
    assert session.update_frequency.n == 3
    assert frequency.error_label.text

    view = RecordedView("main", "Main", MainView())
    assert session.recording.start(view)
    try:
        dialog.open()
        assert not frequency.unit_select.enabled
        assert not frequency.n_input.enabled
        assert frequency.lock_note.visible
    finally:
        session.recording.delete_all()
    dialog.open()
    assert frequency.unit_select.enabled and not frequency.lock_note.visible
