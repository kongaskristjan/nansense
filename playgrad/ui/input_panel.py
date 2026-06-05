"""The "Input Selection" sidebar of the main page.

Hosts the per-sample spinner (moved out of the top bar), the batch-pinning
and click-to-perturb controls for probe runs (see `playgrad.probe`), and the
input image. The panel owns the per-connection view state the page's tick
loop reads (`sample_idx`, `compare`) and forwards pin / probe-mode /
perturbation changes to the session; the session reacts asynchronously by
publishing a new `ProbeResult`, which the tick loop picks up like a new
snapshot.

The input image is a `ui.interactive_image`, so clicks arrive with
coordinates in the image's *native* pixel space regardless of the CSS
upscale. With "Click to perturb" on, a click writes the picked color —
back-transformed into model-input space via `normalized_color` — into that
pixel of the viewed sample on every subsequent probe input.
"""

from __future__ import annotations

from collections.abc import Callable

from nicegui import ui
from torch import Tensor

from playgrad.session import Session
from playgrad.ui.render import INPUT_IMAGE_SIZE

_PROBE_MODE_OPTIONS: dict[str, str] = {
    "unchanged": "Unchanged",
    "eval": "Eval",
    "train": "Train",
}


def normalized_color(
    hex_color: str,
    channels: int,
    mean: tuple[float, ...] | None,
    std: tuple[float, ...] | None,
) -> tuple[float, ...] | None:
    """Convert a `#rrggbb` display color to model-input channel values.

    Grayscale inputs (`channels == 1`) use the mean of the RGB components.
    When `mean` / `std` are given (the same stats used to *de*normalize the
    input for display), the color is back-transformed into normalized input
    space: `(c - mean) / std`. Returns `None` for unsupported channel
    counts, unparsable colors, or mismatched stats lengths.
    """
    if channels not in (1, 3):
        return None
    text = hex_color.strip().lstrip("#")
    if len(text) != 6:
        return None
    try:
        rgb = tuple(int(text[i : i + 2], 16) / 255.0 for i in (0, 2, 4))
    except ValueError:
        return None
    values: tuple[float, ...] = (sum(rgb) / 3.0,) if channels == 1 else rgb
    if mean is not None and std is not None:
        if len(mean) != channels or len(std) != channels:
            return None
        values = tuple((v - m) / s for v, m, s in zip(values, mean, std))
    return values


class InputPanel:
    """Builds the sidebar's controls inside the currently open container.

    `on_change` marks the page dirty so the next tick re-renders the strips
    (sample flips, unpinning, and the compare toggle change what's displayed
    without a new snapshot or probe result arriving).
    """

    def __init__(
        self,
        *,
        session: Session,
        input_name: str | None,
        input_mean: tuple[float, ...] | None,
        input_std: tuple[float, ...] | None,
        on_change: Callable[[], None],
    ) -> None:
        self._session = session
        self._input_name = input_name
        self._input_mean = input_mean
        self._input_std = input_std
        self._on_change = on_change
        self.sample_idx = 0
        self.compare = False
        self._color = "#000000"
        self._spinner_max: int | None = None
        self._build()

    def _build(self) -> None:
        # One compact column: the image first (it is what everything below
        # acts on), then its sample selector, then the "Probe" and "Perturb"
        # control sections.
        with ui.column().classes("w-full items-center gap-2"):
            ui.label("Input Selection").classes("font-mono text-sm self-start")
            self._image = ui.interactive_image(
                on_mouse=self._on_image_click, events=["mousedown"]
            ).style(f"width:{INPUT_IMAGE_SIZE}px; image-rendering:pixelated")
            with ui.row().classes("w-full items-center justify-between no-wrap"):
                ui.label("Viewing sample:").classes("text-sm")
                self._sample_input = ui.number(
                    value=0,
                    min=0,
                    step=1,
                    format="%d",
                    on_change=self._on_sample_change,
                ).classes("w-20").props("dense")
            self._error_label = ui.label("").classes(
                "text-xs text-red-600 self-start"
            )

            ui.separator()
            self._section_label("Probe")
            self._pin_switch = ui.switch(
                "Pin batch",
                value=self._session.is_pinned,
                on_change=self._on_pin_change,
            ).props("dense").classes("self-start").tooltip(
                "Re-run the model on this fixed batch at every pause (a probe "
                "run), instead of showing the changing training batch"
            )
            self._mode_toggle = ui.toggle(
                _PROBE_MODE_OPTIONS,
                value=self._session.probe_mode,
                on_change=self._on_mode_change,
            ).props("dense no-caps spread").classes("w-full").tooltip(
                "Train/eval handling for probe forwards. Eval (default) uses "
                "BatchNorm running stats and disables dropout; Unchanged runs "
                "with whatever modes training left; Train uses batch stats and "
                "dropout. All modes restore the model's state afterwards."
            )
            # Probe mode only applies to probe runs, which need a pinned batch.
            self._mode_toggle.bind_visibility_from(self._pin_switch, "value")
            self._pinned_caption = ui.label("").classes(
                "text-xs text-slate-500 font-mono self-start"
            )

            ui.separator()
            self._section_label("Perturb")
            with ui.row().classes("w-full items-center justify-between no-wrap"):
                self._perturb_switch = ui.switch(
                    "Click to perturb",
                    on_change=self._on_perturb_change,
                ).props("dense").tooltip(
                    "Clicking the input image paints the swatch color into "
                    "that pixel of the viewed sample on every probe input"
                )
                # A compact color swatch that opens the picker on click; the
                # button's background *is* the current color.
                self._color_button = ui.button().props("dense unelevated").style(
                    self._swatch_style()
                ).tooltip("Perturb color — click to change")
                with self._color_button:
                    picker = ui.color_picker(on_pick=self._on_pick_color)
                    # Hex only: normalized_color expects #rrggbb, so don't let
                    # the picker emit rgba()/hsl() strings.
                    picker.q_color.props("format-model=hex")
            self._compare_switch = ui.switch(
                "Compare with original",
                on_change=self._on_compare_change,
            ).props("dense").classes("self-start").tooltip(
                "Show each layer's activation diff (perturbed − original); "
                "with nothing perturbed the diff is zero and renders white"
            )
            # Compare belongs to editing mode (it shows white with zero
            # edits); the count/clear row only makes sense once a pixel has
            # actually been perturbed (`refresh_status` keeps it in sync).
            self._compare_switch.bind_visibility_from(
                self._perturb_switch, "value"
            )
            self._clear_row = ui.row().classes(
                "w-full items-center justify-between no-wrap"
            )
            self._clear_row.set_visibility(bool(self._session.perturbations))
            with self._clear_row:
                self._perturb_caption = ui.label("").classes(
                    "text-xs text-slate-500 font-mono"
                )
                ui.button(
                    "Clear",
                    on_click=self._on_clear,
                    color="slate-500",
                ).props("dense size=sm no-caps").tooltip(
                    "Remove all perturbations"
                )

    @staticmethod
    def _section_label(text: str) -> None:
        ui.label(text).classes(
            "text-[10px] uppercase tracking-wider text-slate-400 self-start"
        )

    def _swatch_style(self) -> str:
        return (
            f"background-color: {self._color} !important; "
            "width: 28px; height: 28px; min-height: 0; "
            "border: 1px solid #cbd5e1"
        )

    def _on_pick_color(self, e: object) -> None:
        color = str(getattr(e, "color", "") or "")
        if color:
            self._color = color
            self._color_button.style(self._swatch_style())

    def set_image(self, src: str) -> None:
        self._image.set_source(src)

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
        """Cheap per-tick text/visibility updates (no-op writes are skipped)."""
        pos = self._session.pinned_position
        self._pinned_caption.text = (
            f"pinned at epoch {pos.epoch} | {pos.phase} batch {pos.batch_idx}"
            if pos is not None
            else ""
        )
        n = len(self._session.perturbations)
        self._perturb_caption.text = (
            f"{n} perturbed pixel{'' if n == 1 else 's'}" if n else ""
        )
        has_perturbations = n > 0
        if self._clear_row.visible != has_perturbations:
            self._clear_row.set_visibility(has_perturbations)
        self._error_label.text = self._session.probe_error or ""

    def _current_input(self) -> Tensor | None:
        """The input batch the displayed image was rendered from."""
        probe = self._session.probe_result
        if probe is not None:
            return probe.input
        snap = self._session.snapshot
        if snap is None or self._input_name is None:
            return None
        return snap.activations.get(self._input_name)

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

    def _on_clear(self) -> None:
        self._session.clear_perturbations()
        self.refresh_status()  # hide compare/clear now, not on the next tick

    def _on_perturb_change(self, e: object) -> None:
        if getattr(e, "value", False):
            self._image.classes(add="cursor-crosshair")
        else:
            # Leaving editing mode discards the edits and the diff view: the
            # image and strips revert to the unperturbed input. Resetting the
            # (now hidden) compare switch fires its own change handler, which
            # clears `self.compare` and marks the page dirty.
            self._image.classes(remove="cursor-crosshair")
            self._compare_switch.set_value(False)
            self._on_clear()

    def _on_compare_change(self, e: object) -> None:
        self.compare = bool(getattr(e, "value", False))
        self._on_change()

    def _on_image_click(self, e: object) -> None:
        if not self._perturb_switch.value:
            return
        tensor = self._current_input()
        if tensor is None or tensor.ndim != 4:
            return
        values = normalized_color(
            self._color,
            int(tensor.shape[1]),
            self._input_mean,
            self._input_std,
        )
        if values is None:
            return
        h, w = int(tensor.shape[2]), int(tensor.shape[3])
        x = min(max(int(getattr(e, "image_x", 0)), 0), w - 1)
        y = min(max(int(getattr(e, "image_y", 0)), 0), h - 1)
        self._session.add_perturbation(
            sample=self.sample_idx, y=y, x=x, values=values
        )
