"""The "Input Selection" sidebar of the main page.

Hosts the per-sample spinner (moved out of the top bar), the batch-pinning
and click-to-perturb controls for probe runs (see `nansense.probe`), and the
input image. The panel owns the per-connection view state the page's tick
loop reads (`sample_idx`, plus the derived `compare`) and forwards pin /
probe-mode / perturbation changes to the session; the session reacts
asynchronously by publishing a new `ProbeResult`, which the tick loop picks
up like a new snapshot.

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

from nansense.session import Session
from nansense.ui.common import _defer_value_write, _set_controls_enabled

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
    (sample flips, unpinning, and clearing perturbations change what's
    displayed without a new snapshot or probe result arriving).
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
        self._color = "#000000"
        self._spinner_max: int | None = None
        self._frozen = False
        # Guards `_on_pin_change` while `refresh_status` writes the switch to
        # mirror shared session state (a pin from another tab) — otherwise the
        # programmatic write would re-fire pin/unpin.
        self._syncing_pin = False
        self._build()

    def _build(self) -> None:
        # One compact column: the image first (it is what everything below
        # acts on), then its sample selector, then the "Pin", "Forward mode"
        # and "Perturb" control sections.
        with ui.column().classes("w-full items-center gap-2"):
            # Full pane width, so the image scales with the (resizable)
            # pane; clicks stay in native pixel space regardless of CSS size.
            self._image = ui.interactive_image(
                on_mouse=self._on_image_click, events=["mousedown"]
            ).style("width:100%; image-rendering:pixelated")
            with ui.row().classes("w-full items-center justify-between no-wrap"):
                # The batch size is filled in once known (see
                # `sync_spinner_max`); the sample index itself is 0-based.
                self._sample_label = ui.label("Select sample in batch:").classes(
                    "text-sm"
                )
                self._sample_input = ui.number(
                    value=0,
                    min=0,
                    step=1,
                    format="%d",
                    on_change=self._on_sample_change,
                ).classes("w-16").props("dense")
            self._error_label = ui.label("").classes(
                "text-xs text-red-600 self-start"
            )

            ui.separator()
            self._section_label("Pin")
            self._pin_switch = ui.switch(
                "Pin batch",
                value=self._session.is_pinned,
                on_change=self._on_pin_change,
            ).props("dense").classes("self-start").tooltip(
                "Re-run the model on this fixed batch at every pause (a probe "
                "run), instead of showing the changing training batch"
            )
            self._pinned_caption = ui.label("").classes(
                "text-xs text-slate-500 font-mono self-start"
            )

            ui.separator()
            self._section_label("Forward mode")
            # Eval/Train re-run the model on the current batch under that mode
            # on their own (no pin needed); Unchanged only shows a re-run when
            # a batch is pinned or a pixel is perturbed.
            self._mode_toggle = ui.toggle(
                _PROBE_MODE_OPTIONS,
                value=self._session.probe_mode,
                on_change=self._on_mode_change,
            ).props("dense no-caps spread").classes("w-full").tooltip(
                "Train/eval handling for probe forwards. Unchanged (default) "
                "runs with whatever modes training left; Eval uses BatchNorm "
                "running stats and disables dropout; Train uses batch stats "
                "and dropout. Selecting Eval or Train re-runs the current "
                "batch under that mode even without a pin; all modes restore "
                "the model's state afterwards."
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
            # The count/clear row and the compare note only make sense once a
            # pixel has actually been perturbed (`refresh_status` keeps both
            # in sync).
            self._clear_row = ui.row().classes(
                "w-full items-center justify-between no-wrap"
            )
            self._clear_row.set_visibility(bool(self._session.perturbations))
            with self._clear_row:
                self._perturb_caption = ui.label("").classes(
                    "text-xs text-slate-500 font-mono"
                )
                self._clear_button = ui.button(
                    "Clear",
                    on_click=self._on_clear,
                    color="slate-500",
                ).props("dense size=sm no-caps").tooltip(
                    "Remove all perturbations"
                )
            self._compare_caption = ui.label(
                "Comparing with original: layer strips show the "
                "activation diff (perturbed − original)"
            ).classes("text-xs text-slate-500 self-start")
            self._compare_caption.set_visibility(
                bool(self._session.perturbations)
            )

    @property
    def compare(self) -> bool:
        """Whether the tick loop should render the diff view.

        Comparing is not a user toggle: it is active exactly while at least
        one pixel is perturbed, so the diff is never the all-zero (white)
        no-edit case.
        """
        return bool(self._session.perturbations)

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

    def set_frozen(self, frozen: bool) -> None:
        """Disable every control while the main view is being recorded.

        The recording renders with the live probe state (pinned batch,
        perturbations, sample), so the panel must not change it. No-op when
        the state didn't change, so the page can call this every tick.
        """
        if frozen == self._frozen:
            return
        self._frozen = frozen
        _set_controls_enabled(
            (
                self._sample_input,
                self._pin_switch,
                self._mode_toggle,
                self._perturb_switch,
                self._color_button,
                self._clear_button,
            ),
            not frozen,
        )

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
        self._sample_label.text = f"Select sample in batch ({batch_size}):"
        if self.sample_idx > new_max:
            self.sample_idx = new_max
            self._sample_input.value = new_max
            self._on_change()

    def refresh_status(self) -> None:
        """Cheap per-tick text/visibility updates (no-op writes are skipped).

        Also mirrors shared session state into this connection's controls so
        a pin / perturbation made in another tab shows up here immediately:
        the pin switch follows `is_pinned`, and the perturbation count / clear
        row / compare note follow `perturbations` (the perturbed image itself
        rides along on the shared probe result the page tick re-renders).
        """
        if self._pin_switch.value != self._session.is_pinned:
            self._syncing_pin = True
            self._pin_switch.set_value(self._session.is_pinned)
            self._syncing_pin = False
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
        if self._compare_caption.visible != has_perturbations:
            self._compare_caption.set_visibility(has_perturbations)
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

    def _on_sample_change(self, e: object) -> None:
        value = getattr(e, "value", None)
        idx = int(value) if value is not None else 0
        if idx < 0:
            idx = 0
        elif self._spinner_max is not None and idx > self._spinner_max:
            idx = self._spinner_max
        if idx != value:
            clamped = idx
            _defer_value_write(lambda: self._sample_input.set_value(clamped))
        self.sample_idx = idx
        self._on_change()

    def _on_pin_change(self, e: object) -> None:
        if self._syncing_pin:
            return  # programmatic mirror of another tab's pin, not a click
        if getattr(e, "value", False):
            if not self._session.pin_current_batch():
                ui.notify(
                    "Nothing to pin yet — run at least one batch first",
                    type="warning",
                )
                _defer_value_write(lambda: self._pin_switch.set_value(False))
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
        self.refresh_status()  # hide the compare note/clear row now
        self._on_change()  # drop the diff view now, not on the next probe

    def _on_perturb_change(self, e: object) -> None:
        if getattr(e, "value", False):
            self._image.classes(add="cursor-crosshair")
        else:
            # Leaving editing mode discards the edits and the diff view: the
            # image and strips revert to the unperturbed input (clearing the
            # perturbations also deactivates `compare`, which derives from
            # them).
            self._image.classes(remove="cursor-crosshair")
            self._on_clear()

    def _on_image_click(self, e: object) -> None:
        if self._frozen or not self._perturb_switch.value:
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
