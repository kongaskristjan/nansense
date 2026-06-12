"""The shared top-bar/step controls and the dialogs they open."""

from __future__ import annotations

import asyncio
from collections.abc import Callable

from nicegui import ui

from nansense.recording import RecordedView
from nansense.restore import TimeTravelError
from nansense.schedule import BatchPosition, Schedule, format_position
from nansense.session import BatchSnapshot, Session


_TOP_BAR_CLASSES: str = (
    "w-full items-center gap-x-3 gap-y-0 px-3 py-2 shrink-0 "
    "border-b-2 border-slate-300 bg-slate-100 shadow-sm z-10"
)


def _top_bar_row() -> ui.row:
    """The shared top-bar row container used by every page."""
    return ui.row().classes(_TOP_BAR_CLASSES)


def _back_button() -> None:
    """The arrow-back button to the main page (every subpage's top bar).

    Rendered as a native link (`href` prop) rather than an `on_click`
    navigation so middle/ctrl-click opens the main page in a new tab —
    same pattern as the layer cards' Weights/Experiment buttons.
    """
    ui.button(
        icon="arrow_back",
        color="slate-500",
    ).props('dense size=md href="/"').tooltip("Back to the main page")


def _add_step_controls(
    session: Session,
    step_until_custom: ui.dialog,
) -> None:
    """Add the stepping buttons + a live-position label to the open row.

    Shared by every page's top bar so they all drive the session
    identically. The settings (gear) button lives in the top bar's
    right-side cluster instead — each page adds it via
    `_add_settings_button`. The label tracks the *live* training position
    via a timer registered here (see `format_position`); the 0.2s period is
    the throttle — rapid batches in step_epoch / step_run / detach coalesce
    into at most ~5 cheap label updates per second, and NiceGUI skips the
    write when the text is unchanged.
    """
    ui.button("Stop", on_click=session.stop, color="red").props(
        "dense size=md"
    ).tooltip("Pause at the next batch boundary")
    with ui.dropdown_button(
        "Step Batch",
        on_click=session.step_batch,
        split=True,
        auto_close=True,
        color="orange",
    ).props("dense size=md").tooltip("Advance one batch, then pause"):
        _step_menu_item(
            "Step epoch",
            "Run until the epoch changes, then pause",
            session.step_epoch,
        )
        _step_menu_item(
            "Step until end",
            "Run to the last batch of training, then pause",
            session.step_run,
        )
        _step_menu_item(
            "Step custom…",
            "Pick a phase/epoch/batch to pause at",
            step_until_custom.open,
        )
    _add_time_travel_button(session)
    position_label = ui.label("(waiting for first snapshot)").classes(
        "ml-3 font-mono text-sm"
    )

    def refresh_position() -> None:
        live = session.live_position
        if live is not None:
            position_label.text = format_position(live)

    ui.timer(0.2, refresh_position)


def _step_menu_item(
    title: str, caption: str, on_click: Callable[[], object]
) -> None:
    """One entry of the Step dropdown: an action title + explanatory caption."""
    with ui.item(on_click=on_click), ui.item_section():
        ui.item_label(title)
        ui.item_label(caption).props("caption")


def _add_time_travel_button(session: Session) -> None:
    """The blue Time Travel button (right of Detach) and its dialogs.

    The button is grayed out — with the reason as a hover tooltip — when the
    session has no training restorer, i.e. the training loop did not opt
    into time travel. The picker dialog rebuilds its content on every open,
    so the cached-epoch choices track the checkpoints written so far; a
    request the session rejects (e.g. a cache written by a different model)
    surfaces in a separate error dialog and no jump happens.
    """
    with ui.dialog() as error_dialog, ui.card().classes("min-w-96 p-6 gap-3"):
        ui.label("Time travel failed").classes("text-lg font-bold text-red-600")
        error_label = ui.label("").classes("text-sm text-slate-700")
        with ui.row():
            ui.button("Close", on_click=error_dialog.close)

    with ui.dialog() as dialog, ui.card().classes("min-w-96 p-6 gap-4"):
        ui.label("Time travel").classes("text-lg font-bold")
        content = ui.column().classes("w-full gap-3")

    def show_error(message: str) -> None:
        error_label.text = message
        error_dialog.open()

    def submit(value: object) -> None:
        if not isinstance(value, (int, float)):
            show_error("Pick an epoch to jump to.")
            return
        try:
            session.request_time_travel(int(value))
        except TimeTravelError as e:
            show_error(str(e))
            return
        dialog.close()

    def cache_run() -> None:
        # Runs training to the very end; with a restorer active, every
        # epoch start gets checkpointed along the way, filling the cache.
        session.step_run()
        dialog.close()

    def rebuild() -> None:
        content.clear()
        status = session.time_travel_status()
        cached = status.cached_epochs
        missing = sorted(set(range(status.total_epochs)) - set(cached))
        with content:
            if not status.available:
                ui.label(status.reason or "Time travel is unavailable.").classes(
                    "text-sm text-slate-600"
                )
                with ui.row():
                    ui.button("Close", on_click=dialog.close)
                return
            # The slider runs over *indices into the cached-epoch list*, so
            # epochs without a checkpoint are unselectable even when the
            # cached set has gaps; the label shows the mapped epoch number.
            epoch_slider: ui.slider | None = None
            if cached:
                ui.label("Jump back to the start of a cached epoch:").classes(
                    "text-sm"
                )
                with ui.row().classes("w-full items-center gap-4 no-wrap"):
                    epoch_label = ui.label(f"epoch {cached[-1]}").classes(
                        "font-mono text-sm w-20 shrink-0"
                    )
                    epoch_slider = ui.slider(
                        min=0,
                        max=len(cached) - 1,
                        step=1,
                        value=len(cached) - 1,
                        on_change=lambda e: epoch_label.set_text(
                            f"epoch {cached[int(e.value)]}"
                        ),
                    ).classes("grow")
            else:
                ui.label("No epochs have a cached model yet.").classes(
                    "text-sm text-slate-600"
                )
            ui.label(
                "Cached epochs: "
                + (_summarize_epoch_ranges(cached) if cached else "none")
            ).classes("text-xs text-slate-500")
            if missing:
                ui.label(
                    f"Uncached epochs: {_summarize_epoch_ranges(missing)}. "
                    "An epoch becomes available once training has passed "
                    "its start."
                ).classes("text-xs text-slate-500")
            with ui.row():
                ui.button("Cancel", on_click=dialog.close)
                if missing:
                    ui.button(
                        "Cache full training run", on_click=cache_run
                    ).tooltip(
                        "Run training to the end, checkpointing every epoch "
                        "start along the way"
                    )
                if epoch_slider is not None:
                    slider = epoch_slider
                    ui.button(
                        "Time travel",
                        on_click=lambda: submit(
                            cached[int(slider.value)]
                            if slider.value is not None
                            else None
                        ),
                        color="blue",
                    )

    def open_dialog() -> None:
        rebuild()
        dialog.open()

    status = session.time_travel_status()
    # Quasar suppresses pointer events on disabled buttons, so the tooltip
    # (which must explain *why* the button is off) lives on a wrapper div.
    tooltip = (
        "Jump training back to the start of a cached epoch"
        if status.available
        else (status.reason or "Time travel is unavailable.")
    )
    with ui.element("div").tooltip(tooltip):
        button = ui.button("Time Travel", on_click=open_dialog, color="blue").props(
            "dense size=md"
        )
        if not status.available:
            button.props("disable")


_FREQUENCY_UNIT_OPTIONS: dict[str, str] = {
    "epoch": "Every nth epoch",
    "batch": "Every nth batch",
}

# Phase-select sentinel for "count batches of every phase".
_ANY_PHASE: str = "(any phase)"


def _add_settings_button(
    session: Session,
    record_view: Callable[[], RecordedView | None] | None,
) -> ui.button:
    """The gear button (every top bar's right-side cluster) + settings dialog.

    Returned so the page can right-align it (`ml-auto`) when it is the
    first element of the cluster.

    The dialog hosts two sections. "Update frequency" configures
    `Session.set_update_frequency`: visualizations refresh every nth epoch
    (the default, n=1) or every nth batch, optionally counting only one
    phase's batches. The setting is locked while recordings are active —
    recording frames advance at this frequency, so changing it
    mid-recording would change the videos' time base.

    "Recording" offers "Add this view to recording" for the page's own view
    (built by `record_view` with the page's *current* parameters, frozen
    for the recording's lifetime; None when the page's current state can't
    be recorded yet) plus the list of all active recordings, each endable
    (finalize the MP4) or deletable (discard it), and end/delete-all
    buttons. A red badge on the gear carries the active-recording count.
    """
    phase_names = list(session.schedule.phases)

    with ui.dialog() as dialog, ui.card().classes("min-w-[30rem] p-6 gap-3"):
        ui.label("Update frequency").classes("text-lg font-bold")
        ui.label(
            "How often all visualizations refresh while training runs. "
            "They additionally refresh whenever training stops."
        ).classes("text-sm text-slate-600")
        with ui.row().classes("w-full gap-2 no-wrap items-start"):
            unit_select = ui.select(
                _FREQUENCY_UNIT_OPTIONS,
                label="Update every",
                value="epoch",
                on_change=lambda: sync_phase_visibility(),
            ).props("dense outlined").classes("flex-1")
            n_input = ui.number(
                label="n", value=1, min=1, step=1, format="%d"
            ).props("dense outlined").classes("w-20").tooltip(
                "Update on every nth epoch/batch"
            )
            phase_select = ui.select(
                [_ANY_PHASE] + phase_names,
                label="Phase",
                value=_ANY_PHASE,
            ).props("dense outlined").classes("flex-1").tooltip(
                "Count only this phase's batches (batch unit only)"
            )
        error_label = ui.label("").classes("text-red-500 text-sm min-h-4")
        lock_note = ui.label(
            "The frequency is locked while recordings are active — frames "
            "are recorded at this cadence."
        ).classes("text-xs text-amber-700")
        apply_button = ui.button("Apply", on_click=lambda: apply(), color="purple").props(
            "dense size=md"
        )
        ui.separator()
        ui.label("Recording").classes("text-lg font-bold")
        ui.label(
            "Each recorded view becomes an MP4 file — one frame per "
            "visualization update."
        ).classes("text-sm text-slate-600")
        recording_section = ui.column().classes("w-full gap-3")
        with ui.row().classes("w-full justify-end"):
            ui.button("Close", on_click=dialog.close)

    def sync_phase_visibility() -> None:
        phase_select.set_visibility(unit_select.value == "batch")

    def apply() -> None:
        unit = str(unit_select.value)
        phase = str(phase_select.value)
        try:
            n = int(n_input.value) if n_input.value is not None else 1
            session.set_update_frequency(
                unit=unit,
                n=n,
                phase=phase if unit == "batch" and phase != _ANY_PHASE else None,
            )
        except (TypeError, ValueError) as e:
            error_label.text = str(e)
            return
        error_label.text = ""
        ui.notify("Update frequency applied")

    def unpin(view: RecordedView) -> None:
        # An experiment recording pins the page's auto experiment so it
        # keeps re-running even after the page closes; ending the recording
        # puts it back on the page-heartbeat clock.
        if view.page == "experiment":
            session.unpin_auto_experiment(str(view.params.get("auto_key", "")))

    def add_view() -> None:
        view = record_view() if record_view is not None else None
        if view is None:
            ui.notify(
                "Nothing to record on this page yet", type="warning"
            )
            return
        if not session.recording.start(view):
            ui.notify("This view is already being recorded", type="warning")
            return
        if view.page == "experiment":
            session.pin_auto_experiment(str(view.params.get("auto_key", "")))
        ui.notify(f"Recording into {session.recording.directory}/")
        refresh_recording_lock()
        rebuild()

    # End/delete run via `asyncio.to_thread`: finalizing the MP4 writers
    # (ffmpeg) takes a moment and may briefly wait for an in-flight frame
    # append — blocking the event loop here would starve the websocket
    # keepalive and drop the connection.
    async def end_view(key: str, view: RecordedView) -> None:
        paths = await asyncio.to_thread(session.recording.end, key)
        unpin(view)
        if paths:
            ui.notify("Saved " + ", ".join(str(p) for p in paths))
        else:
            ui.notify("Recording ended before any frame was captured")
        refresh_recording_lock()
        rebuild()

    async def delete_view(key: str, view: RecordedView) -> None:
        await asyncio.to_thread(session.recording.delete, key)
        unpin(view)
        refresh_recording_lock()
        rebuild()

    async def end_all() -> None:
        for status in session.recording.statuses():
            unpin(status.view)
        paths = await asyncio.to_thread(session.recording.end_all)
        if paths:
            ui.notify("Saved " + ", ".join(str(p) for p in paths))
        refresh_recording_lock()
        rebuild()

    async def delete_all() -> None:
        for status in session.recording.statuses():
            unpin(status.view)
        await asyncio.to_thread(session.recording.delete_all)
        refresh_recording_lock()
        rebuild()

    def rebuild() -> None:
        recording_section.clear()
        statuses = session.recording.statuses()
        current = record_view() if record_view is not None else None
        with recording_section:
            ui.label("This view").classes(
                "text-xs uppercase tracking-wider text-slate-400"
            )
            if current is None:
                ui.label(
                    "Nothing recordable on this page yet — watch a layer or "
                    "run an experiment first."
                ).classes("text-sm text-slate-500 italic")
            else:
                with ui.row().classes("w-full items-center gap-2 no-wrap"):
                    with ui.column().classes("grow min-w-0 gap-0"):
                        ui.label(current.label).classes(
                            "text-sm font-medium truncate"
                        )
                        if session.recording.is_recording(current.key):
                            ui.label("recording — parameters frozen").classes(
                                "text-xs text-red-600"
                            )
                    if not session.recording.is_recording(current.key):
                        ui.button(
                            "Record",
                            icon="fiber_manual_record",
                            on_click=add_view,
                            color="red",
                        ).props("dense size=sm no-caps").tooltip(
                            "Add this view to recording — one frame per "
                            "visualization update"
                        )
            if statuses:
                ui.label("Currently recording").classes(
                    "text-xs uppercase tracking-wider text-slate-400"
                )
                for status in statuses:
                    with ui.row().classes("w-full items-center gap-2 no-wrap"):
                        with ui.column().classes("grow min-w-0 gap-0"):
                            ui.label(status.view.label).classes(
                                "text-sm font-medium truncate"
                            )
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
                            "End",
                            on_click=lambda s=status: end_view(s.view.key, s.view),
                            color="grey-8",
                        ).props("dense size=sm no-caps").tooltip(
                            "Finalize this view's MP4 file(s)"
                        )
                        ui.button(
                            "Delete",
                            on_click=lambda s=status: delete_view(
                                s.view.key, s.view
                            ),
                            color="red",
                        ).props("dense size=sm no-caps flat").tooltip(
                            "Discard this view's recording"
                        )
                with ui.row().classes("w-full gap-2"):
                    ui.button(
                        "End all", on_click=end_all, color="grey-8"
                    ).props("dense size=sm no-caps")
                    ui.button(
                        "Delete all", on_click=delete_all, color="red"
                    ).props("dense size=sm no-caps flat")
                ui.label(f"Files: {session.recording.directory}/").classes(
                    "text-xs text-slate-500 font-mono"
                )

    def refresh_recording_lock() -> None:
        locked = session.recording.count() > 0
        lock_note.set_visibility(locked)
        if locked:
            apply_button.disable()
        else:
            apply_button.enable()

    def open_dialog() -> None:
        freq = session.update_frequency
        unit_select.value = freq.unit
        n_input.value = freq.n
        phase_select.value = freq.phase if freq.phase is not None else _ANY_PHASE
        sync_phase_visibility()
        error_label.text = ""
        refresh_recording_lock()
        rebuild()
        dialog.open()

    button = ui.button(icon="settings", on_click=open_dialog, color="slate-500").props(
        "dense size=md"
    )
    button.tooltip("Settings — update frequency and MP4 recording")
    with button:
        badge = ui.badge("").props("color=red floating")

    def refresh_badge() -> None:
        n = session.recording.count()
        badge.set_text(str(n))
        badge.set_visibility(n > 0)

    refresh_badge()
    ui.timer(0.5, refresh_badge)
    return button


def _summarize_epoch_ranges(epochs: list[int]) -> str:
    """Compact "0–2, 5, 7–49" rendering of a sorted epoch list."""
    parts: list[str] = []
    start = prev = epochs[0]
    for e in epochs[1:]:
        if e == prev + 1:
            prev = e
            continue
        parts.append(f"{start}–{prev}" if prev > start else f"{start}")
        start = prev = e
    parts.append(f"{start}–{prev}" if prev > start else f"{start}")
    return ", ".join(parts)


def _build_step_until_custom_dialog(session: Session) -> ui.dialog:
    schedule = session.schedule
    phase_names = list(schedule.phases)

    with ui.dialog() as dialog, ui.card().classes("min-w-96 p-6 gap-4"):
        ui.label("Step until custom").classes("text-lg font-bold")
        with ui.row().classes("w-full gap-4 items-end no-wrap"):
            epoch_input = ui.number(
                label="Epoch", value=0, min=0, step=1, format="%d"
            ).classes("flex-1")
            phase_select = ui.select(
                phase_names, label="Phase", value=phase_names[0]
            ).classes("flex-1")
            batch_input = ui.number(
                label="Batch", value=0, min=0, step=1, format="%d"
            ).classes("flex-1")
        error_label = ui.label("").classes("text-red-500 text-sm min-h-4")

        def submit() -> None:
            try:
                epoch = int(epoch_input.value) if epoch_input.value is not None else 0
                batch_idx = int(batch_input.value) if batch_input.value is not None else 0
            except (TypeError, ValueError):
                error_label.text = "Invalid input"
                return
            phase = str(phase_select.value)
            error = _validate_step_until_target(
                schedule=schedule,
                snapshot=session.snapshot,
                phase=phase,
                epoch=epoch,
                batch_idx=batch_idx,
            )
            if error is not None:
                error_label.text = error
                return
            error_label.text = ""
            session.step_until_position(phase=phase, epoch=epoch, batch_idx=batch_idx)
            dialog.close()

        with ui.row():
            ui.button("Cancel", on_click=dialog.close)
            ui.button("Step", on_click=submit)

        def apply_current_position() -> None:
            position = _step_until_default_position(
                session.live_position, session.snapshot
            )
            if position is None:
                return
            epoch_input.value = position.epoch
            phase_select.value = position.phase
            batch_input.value = position.batch_idx
            error_label.text = ""

        dialog.on("before-show", apply_current_position)

    return dialog


def _step_until_default_position(
    live_position: BatchPosition | None, snapshot: BatchSnapshot | None
) -> BatchPosition | None:
    """Position to prefill the step-until dialog with on open.

    Prefilling with where training currently is means the user only tweaks
    the field they care about. `live_position` tracks every batch even in
    modes where `snapshot.position` is frozen between boundaries, so it wins;
    the snapshot covers the brief window before the first batch publishes a
    live position.
    """
    if live_position is not None:
        return live_position
    return snapshot.position if snapshot is not None else None


def _validate_step_until_target(
    *,
    schedule: Schedule,
    snapshot: BatchSnapshot | None,
    phase: str,
    epoch: int,
    batch_idx: int,
) -> str | None:
    phases = schedule.phases
    if phase not in phases:
        return f"Unknown phase {phase!r}"
    if not 0 <= epoch < schedule.epochs:
        return f"Epoch must be in [0, {schedule.epochs - 1}]"
    declared = phases[phase]
    if not 0 <= batch_idx < declared:
        return f"Batch must be in [0, {declared - 1}] for phase {phase!r}"
    if snapshot is not None:
        cur = snapshot.position
        target_rank = _position_rank(phases, phase, epoch, batch_idx)
        current_rank = _position_rank(phases, cur.phase, cur.epoch, cur.batch_idx)
        if target_rank <= current_rank:
            return "Target must be after the current position"
    return None


def _position_rank(
    phases: dict[str, int], phase: str, epoch: int, batch_idx: int
) -> tuple[int, int, int]:
    return (epoch, list(phases).index(phase), batch_idx)
