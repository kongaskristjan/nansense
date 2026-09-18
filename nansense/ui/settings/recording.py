"""Recording controls; file work runs outside the UI event loop."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from pathlib import Path
from typing import Literal

from nicegui import ui

from nansense.recording import RecordedView, release_recording_experiments
from nansense.session import Session


class RecordingSettings:
    def __init__(
        self,
        session: Session,
        record_view: Callable[[], RecordedView | None] | None,
        refresh_recording_lock: Callable[[], None],
        update: Callable[[Callable[[], None]], None],
    ) -> None:
        self.session = session
        self.record_view = record_view
        self.refresh_recording_lock = refresh_recording_lock
        self.update = update
        ui.label("Recording").classes("text-lg font-bold")
        ui.label(
            "Record the current view as an MP4, or save one frame as a PNG."
        ).classes("text-sm text-slate-600")
        self.recording_section = ui.column().classes("w-full gap-3")

    def add_view(self) -> None:
        view = self.record_view() if self.record_view is not None else None
        if view is None:
            ui.notify("Nothing to record on this page yet", type="warning")
            return
        if not self.session.recording.start(view):
            ui.notify("This view is already being recorded", type="warning")
            return
        if view.page == "experiment":
            self.session.pin_auto_experiment(view.auto_key)
        ui.notify(f"Recording into {self.session.recording.directory}/")
        self.refresh_recording_lock()
        self.rebuild()

    async def take_snapshot(self) -> None:
        view = self.record_view() if self.record_view is not None else None
        if view is None:
            ui.notify("Nothing to capture on this page yet", type="warning")
            return
        message: str
        kind: Literal["positive", "negative", "warning"]
        try:
            paths = await asyncio.to_thread(
                self.session.recording.snapshot, view, self.session
            )
        except Exception as e:  # noqa: BLE001 — reported, like a frame error
            message, kind = f"Snapshot failed: {type(e).__name__}: {e}", "negative"
        else:
            message, kind = (
                ("Saved " + ", ".join(str(p) for p in paths), "positive")
                if paths
                else ("Nothing to capture in this view yet", "warning")
            )
        self.update(lambda: ui.notify(message, type=kind))

    def finish_recording(
        self, key: str, view: RecordedView, *, keep: bool
    ) -> tuple[Path, ...]:
        try:
            if keep:
                return self.session.recording.end(key)
            self.session.recording.delete(key)
            return ()
        finally:
            release_recording_experiments(self.session, [view])

    async def end_view(self, key: str, view: RecordedView) -> None:
        await self.finish_view(key, view, keep=True)

    async def delete_view(self, key: str, view: RecordedView) -> None:
        await self.finish_view(key, view, keep=False)

    async def finish_view(self, key: str, view: RecordedView, *, keep: bool) -> None:
        message: str | None = None
        kind: Literal["info", "negative"] = "info"
        try:
            paths = await asyncio.to_thread(self.finish_recording, key, view, keep=keep)
        except Exception as error:  # noqa: BLE001 — show encoder/filesystem failures
            action = "Saving" if keep else "Deleting"
            message = f"{action} recording failed: {type(error).__name__}: {error}"
            kind = "negative"
        else:
            if keep:
                message = (
                    "Saved " + ", ".join(str(path) for path in paths)
                    if paths
                    else "Recording ended before any frame was captured"
                )

        def apply() -> None:
            self.refresh_recording_lock()
            self.rebuild()
            if message is not None:
                ui.notify(message, type=kind)

        self.update(apply)

    def rebuild(self) -> None:
        self.recording_section.clear()
        statuses = self.session.recording.statuses()
        current = self.record_view() if self.record_view is not None else None
        # Snapshot stays offered whatever the view is doing — a still costs
        # nothing and is just as useful mid-recording. "Record" is the one
        # that disappears once the view records, since it then carries a
        # "this view" entry in the list below; never both.
        current_recording = current is not None and self.session.recording.is_recording(
            current.key
        )
        with self.recording_section:
            if current is None:
                ui.label(
                    "Nothing capturable on this page yet — watch a layer or "
                    "run an experiment first."
                ).classes("text-sm text-slate-500 italic")
            else:
                with ui.row().classes("w-full items-center gap-2 no-wrap"):
                    ui.label(current.label).classes(
                        "text-sm font-medium truncate grow min-w-0"
                    )
                    ui.button(
                        "Snapshot",
                        icon="photo_camera",
                        on_click=self.take_snapshot,
                        color="grey-8",
                    ).props("dense size=sm no-caps").tooltip("Save this view as a PNG")
                    if not current_recording:
                        ui.button(
                            "Record",
                            icon="fiber_manual_record",
                            on_click=self.add_view,
                            color="red",
                        ).props("dense size=sm no-caps").tooltip(
                            "Record this view — one frame per update"
                        )
            if statuses:
                ui.label("Currently recording").classes(
                    "text-xs uppercase tracking-wider text-slate-400"
                )
                for status in statuses:
                    is_current = current is not None and status.view.key == current.key
                    with ui.row().classes("w-full items-center gap-2 no-wrap"):
                        with ui.column().classes("grow min-w-0 gap-0"):
                            label = status.view.label + (
                                "  (this view)" if is_current else ""
                            )
                            ui.label(label).classes("text-sm font-medium truncate")
                            note = f"{status.frames} frame" + (
                                "" if status.frames == 1 else "s"
                            )
                            if status.error is not None:
                                note += f" · {status.error}"
                            ui.label(note).classes(
                                "text-xs "
                                + (
                                    "text-red-600"
                                    if status.error is not None
                                    else "text-slate-500"
                                )
                            )
                        ui.button(
                            "Save & Finish",
                            on_click=lambda s=status: self.end_view(s.view.key, s.view),
                            color="grey-8",
                        ).props("dense size=sm no-caps").tooltip(
                            "Finish this view's MP4 file(s)"
                        )
                        ui.button(
                            "Delete",
                            on_click=lambda s=status: self.delete_view(
                                s.view.key, s.view
                            ),
                            color="red",
                        ).props("dense size=sm no-caps flat").tooltip(
                            "Discard this view's recording"
                        )
                ui.label(f"Files: {self.session.recording.directory}/").classes(
                    "text-xs text-slate-500 font-mono"
                )
