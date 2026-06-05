"""The "Input Selection" sidebar of the main page.

Hosts the per-sample spinner (moved out of the top bar), the batch-pinning
controls for probe runs (see `playgrad.probe`), and the input image. The
panel owns the per-connection view state the page's tick loop reads
(`sample_idx`) and forwards pin / probe-mode changes to the session; the
session reacts asynchronously by publishing a new `ProbeResult`, which the
tick loop picks up like a new snapshot.
"""

from __future__ import annotations

from collections.abc import Callable

from nicegui import ui

from playgrad.session import Session

_PROBE_MODE_OPTIONS: dict[str, str] = {
    "unchanged": "Unchanged",
    "eval": "Eval",
    "train": "Train",
}


class InputPanel:
    """Builds the sidebar's controls inside the currently open container.

    `on_change` marks the page dirty so the next tick re-renders the strips
    (sample flips and unpinning change what's displayed without a new
    snapshot or probe result arriving).
    """

    def __init__(self, *, session: Session, on_change: Callable[[], None]) -> None:
        self._session = session
        self._on_change = on_change
        self.sample_idx = 0
        self._spinner_max: int | None = None
        self._build()

    def _build(self) -> None:
        ui.label("Input Selection").classes("font-mono text-sm self-start")
        with ui.row().classes("w-full items-center gap-2 no-wrap"):
            ui.label("Viewing sample:").classes("text-sm")
            self._sample_input = ui.number(
                value=0,
                min=0,
                step=1,
                format="%d",
                on_change=self._on_sample_change,
            ).classes("w-20").props("dense")
        self._pin_switch = ui.switch(
            "Pin batch",
            value=self._session.is_pinned,
            on_change=self._on_pin_change,
        ).props("dense").tooltip(
            "Re-run the model on this fixed batch at every pause (a probe "
            "run), instead of showing the changing training batch"
        )
        self._pinned_caption = ui.label("").classes(
            "text-xs text-slate-500 font-mono self-start"
        )
        self._mode_toggle = ui.toggle(
            _PROBE_MODE_OPTIONS,
            value=self._session.probe_mode,
            on_change=self._on_mode_change,
        ).props("dense no-caps").tooltip(
            "Train/eval handling for probe forwards. Eval (default) uses "
            "BatchNorm running stats and disables dropout; Unchanged runs "
            "with whatever modes training left; Train uses batch stats and "
            "dropout. All modes restore the model's state afterwards."
        )
        # Probe mode only applies to probe runs, which need a pinned batch.
        self._mode_toggle.bind_enabled_from(self._pin_switch, "value")
        self._error_label = ui.label("").classes("text-xs text-red-600 self-start")
        self._image = ui.html("")

    def set_image(self, html: str) -> None:
        self._image.set_content(html)

    def sync_spinner_max(self, batch_size: int | None) -> None:
        """Clamp the sample spinner to the displayed batch's size."""
        if batch_size is None or batch_size <= 0:
            return
        new_max = batch_size - 1
        if self._spinner_max == new_max:
            return
        self._spinner_max = new_max
        self._sample_input.max = new_max
        if self.sample_idx > new_max:
            self.sample_idx = new_max
            self._sample_input.value = new_max
            self._on_change()

    def refresh_status(self) -> None:
        """Cheap per-tick text updates (NiceGUI skips unchanged writes)."""
        pos = self._session.pinned_position
        self._pinned_caption.text = (
            f"pinned at epoch {pos.epoch} | {pos.phase} batch {pos.batch_idx}"
            if pos is not None
            else ""
        )
        self._error_label.text = self._session.probe_error or ""

    @staticmethod
    def _defer(write: Callable[[], object]) -> None:
        # NiceGUI suppresses .value writes made from inside a value-change
        # handler; schedule the correction for the next event-loop iteration
        # so it actually reaches the client.
        ui.timer(0.0, write, once=True)

    def _on_sample_change(self, e: object) -> None:
        value = getattr(e, "value", None)
        idx = int(value) if value is not None else 0
        if idx < 0:
            idx = 0
        elif self._spinner_max is not None and idx > self._spinner_max:
            idx = self._spinner_max
        if idx != value:
            clamped = idx
            self._defer(lambda: self._sample_input.set_value(clamped))
        self.sample_idx = idx
        self._on_change()

    def _on_pin_change(self, e: object) -> None:
        if getattr(e, "value", False):
            if not self._session.pin_current_batch():
                ui.notify(
                    "Nothing to pin yet — run at least one batch first",
                    type="warning",
                )
                self._defer(lambda: self._pin_switch.set_value(False))
        else:
            self._session.unpin_batch()
            self._on_change()
        self.refresh_status()

    def _on_mode_change(self, e: object) -> None:
        value = getattr(e, "value", None)
        if value is not None:
            self._session.set_probe_mode(str(value))
