"""Compose independent settings sections into the shared top-bar dialog."""

from __future__ import annotations

from collections.abc import Callable

from nicegui import ui

from nansense.recording import RecordedView
from nansense.session import Session
from nansense.ui.settings.recording import RecordingSettings
from nansense.ui.settings.sections import (
    CollectionSettings,
    DebugSettings,
    FrequencySettings,
    WatchSettings,
)


def add_settings_button(
    session: Session,
    record_view: Callable[[], RecordedView | None] | None,
    update: Callable[[Callable[[], None]], None],
) -> ui.button:
    if session.locked:
        with ui.dialog() as locked_dialog, ui.card().classes("max-w-md p-6 gap-3"):
            ui.label("Settings are locked").classes("text-lg font-bold")
            ui.label(
                "This hosted playground is shared by everyone viewing it, so "
                "the session-wide settings (stats collection, update "
                "frequency, performance caps, error checks, recording) and "
                "the shared probe state (pinning, perturbations, forward "
                "mode) are fixed. Everything per-tab — shown layers, "
                "experiments — works normally."
            ).classes("text-sm text-slate-600")
            with ui.row().classes("w-full justify-end"):
                ui.button("Close", on_click=locked_dialog.close).props("flat")
        button = ui.button(
            icon="settings", on_click=locked_dialog.open, color="slate-500"
        ).props('dense size=md aria-label="Settings"')
        button.tooltip("Settings (locked in this demo)")
        return button

    return SettingsDialog(session, record_view, update).button


class SettingsDialog:
    def __init__(
        self,
        session: Session,
        record_view: Callable[[], RecordedView | None] | None,
        update: Callable[[Callable[[], None]], None],
    ) -> None:
        self.session = session
        with ui.dialog() as self.dialog, ui.card().classes("min-w-[30rem] p-6 gap-3"):
            self.collection = CollectionSettings(session)
            ui.separator()
            self.watch = WatchSettings(session)
            self.frequency = FrequencySettings(session)
            ui.separator()
            self.debug = DebugSettings(session)
            ui.separator()
            self.recording = RecordingSettings(
                session, record_view, self.frequency.refresh_recording_lock, update
            )
            ui.separator()
            with ui.row().classes("w-full justify-end"):
                ui.button("Close", on_click=self.dialog.close).props("flat")
        self.button = ui.button(
            icon="settings", on_click=self.open, color="slate-500"
        ).props('dense size=md aria-label="Settings"')
        self.button.tooltip("Settings")
        with self.button:
            self.badge = ui.badge("").props("color=red floating")
        self.refresh_badge()
        ui.timer(0.5, self.refresh_badge)

    def open(self) -> None:
        self.collection.load()
        self.watch.load()
        self.frequency.load()
        self.debug.load()
        self.recording.rebuild()
        self.dialog.open()

    def refresh_badge(self) -> None:
        count = self.session.recording.count()
        self.badge.set_text(str(count))
        self.badge.set_visibility(count > 0)
