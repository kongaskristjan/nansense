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
from nicegui.elements.mixins.disableable_element import DisableableElement
from torch import Tensor

from nansense.input_config import InputTransform, MeanStd, resolve_per_input
from nansense.session import Session
from nansense.ui.render import transform_preview_color
from nansense.ui.common import (
    _defer_value_write,
    _label_bar_html,
    _set_controls_enabled,
)

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
        input_names: list[str],
        input_mean: MeanStd | dict[str, MeanStd] | None,
        input_std: MeanStd | dict[str, MeanStd] | None,
        input_transform: InputTransform | dict[str, InputTransform] | None,
        on_change: Callable[[], None],
    ) -> None:
        self._session = session
        self._input_names = list(input_names)
        # Raw (possibly per-input-name) display config; resolved for whichever
        # input is selected by `_resolve_selected`.
        self._input_mean_cfg = input_mean
        self._input_std_cfg = input_std
        self._input_transform_cfg = input_transform
        # The input shown in the pane and targeted by perturbations. Equal to
        # the primary input until the (multi-input) dropdown changes it.
        self._selected_input = self._input_names[0] if self._input_names else None
        self._input_mean: MeanStd | None = None
        self._input_std: MeanStd | None = None
        self._input_transform: InputTransform | None = None
        self._resolve_selected()
        self._on_change = on_change
        self.sample_idx = 0
        self._color = "#000000"
        self._spinner_max: int | None = None
        self._frozen = False
        # Guards `_on_pin_change` while `refresh_status` writes the switch to
        # mirror shared session state (a pin from another tab) — otherwise the
        # programmatic write would re-fire pin/unpin.
        self._syncing_pin = False
        # Same guard for the perturb switch, which `refresh_status` mirrors
        # from the shared perturbations (another tab, or a rebuild after
        # navigating back from /stats). `_perturb_armed` is this connection's
        # local "perturb mode entered but nothing clicked yet" intent, kept
        # separate so the switch can stay on before the first perturbation and
        # so an external clear only switches off tabs that didn't arm it.
        self._syncing_perturb = False
        self._perturb_armed = False
        # The perturb value control adapts to the selected input: an RGB color
        # picker (`"color"`), one numeric field per channel for a non-RGB image
        # (`"channels"`), or nothing usable yet (`"none"`). `_sync_perturb_control`
        # rebuilds it when the kind / channel count changes.
        self._perturb_kind = "none"
        self._perturb_n = 0
        self._value_inputs: list[ui.number] = []
        self._value_preview: ui.element | None = None
        # Whether the pane currently shows a flat-input strip (vs an image),
        # which drives the image's fixed height and the legend's visibility.
        self._strip_mode = False
        # Controls disabled while recording, refreshed by each rebuild.
        self._value_controls: list[DisableableElement] = []
        self._build()

    def _build(self) -> None:
        # One compact column: the image first (it is what everything below
        # acts on), then its sample selector, then the "Pin", "Forward mode"
        # and "Perturb" control sections.
        with ui.column().classes("w-full items-center gap-2"):
            # A green "INPUT" banner (matching the experiment view's input
            # label / the activation markers) names the image — without it the
            # bare picture can be mistaken for an output or an activation map.
            ui.html(_label_bar_html("INPUT", color="#10b981")).classes("w-full")
            # Multi-input models get a picker for which input the pane shows
            # (and perturbations target); single-input models show no clutter.
            self._input_select = None
            if len(self._input_names) > 1:
                self._input_select = ui.select(
                    self._input_names,
                    value=self._selected_input,
                    label="Input",
                    on_change=self._on_input_select,
                ).props("dense outlined").classes("w-full").tooltip(
                    "Which model input to show and perturb"
                )
            # The image grows to the pane width; a flat (1D) input renders as a
            # colormapped strip with the scale bar beside it (hidden for image
            # inputs). Clicks stay in native pixel space regardless of CSS size.
            with ui.row().classes("w-full items-start no-wrap gap-1"):
                self._image = ui.interactive_image(
                    on_mouse=self._on_image_click, events=["mousedown"]
                ).classes("grow min-w-0").style("image-rendering:pixelated")
                self._input_legend = ui.image().classes("shrink-0").style(
                    "image-rendering:pixelated; height:48px; width:18px"
                )
                self._input_legend.set_visibility(False)
            # Shown in place of a blank image when the input can't be rendered
            # (e.g. an unsupported channel count needing an `input_transform`);
            # the text names the cause and the fix.
            self._input_warning_label = ui.label("").classes(
                "text-xs text-amber-600 self-start"
            )
            self._input_warning_label.set_visibility(False)
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
                    # On a rebuild (navigating back from /stats) the shared
                    # perturbations survive, so the switch must come up on to
                    # match the perturbed image/strips the page still renders.
                    value=bool(self._session.perturbations),
                    on_change=self._on_perturb_change,
                ).props("dense").tooltip(
                    "Click to modify a single pixel of the input image. "
                    "A diff compared to original is shown in the activations. "
                    "Useful for measuring receptive fields."
                )
                # The value control (color swatch or per-channel fields) is
                # built lazily for the selected input by `_sync_perturb_control`.
                self._perturb_value_slot = ui.row().classes(
                    "items-center gap-1 justify-end"
                )
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
            self._set_perturb_cursor(self._perturb_switch.value)

    def _set_perturb_cursor(self, active: bool) -> None:
        """Show the crosshair cursor on the image exactly while perturbing."""
        if active:
            self._image.classes(add="cursor-crosshair")
        else:
            self._image.classes(remove="cursor-crosshair")

    @property
    def selected_input(self) -> str | None:
        """The input name the pane shows and perturbations target."""
        return self._selected_input

    def _resolve_selected(self) -> None:
        """Refresh the resolved display config for the selected input."""
        name = self._selected_input
        self._input_mean = resolve_per_input(self._input_mean_cfg, name)
        self._input_std = resolve_per_input(self._input_std_cfg, name)
        self._input_transform = resolve_per_input(self._input_transform_cfg, name)

    def _on_input_select(self, e: object) -> None:
        value = getattr(e, "value", None)
        if value is None:
            return
        name = str(value)
        if name == self._selected_input:
            return
        self._selected_input = name
        self._resolve_selected()
        # The viewed input changed: re-render the pane (and its strips/diff)
        # against the new input on the next tick.
        self._on_change()

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

    def _desired_perturb_control(self, tensor: Tensor | None) -> tuple[str, int]:
        """The `(kind, n)` value control the selected input needs.

        `("color", c)` for an RGB/grayscale image with no transform,
        `("channels", c)` for any other 4D input (the pixel's `c` model-space
        values are edited directly), `("scalar", 1)` for a flat `[B, C]` input
        (a click writes one value to the clicked channel), or `("none", 0)`
        when nothing is shown.
        """
        if tensor is None:
            return ("none", 0)
        if tensor.ndim == 2:
            return ("scalar", 1)
        if tensor.ndim != 4:
            return ("none", 0)
        c = int(tensor.shape[1])
        if c in (1, 3) and self._input_transform is None:
            return ("color", c)
        return ("channels", c)

    def _sync_perturb_control(self) -> None:
        """Rebuild the perturb value control when the input's kind changes.

        Cheap when unchanged (the common per-tick case); the field-value
        preview updates on edit, not here.
        """
        kind, n = self._desired_perturb_control(self._current_input())
        if (kind, n) == (self._perturb_kind, self._perturb_n):
            return
        self._perturb_kind, self._perturb_n = kind, n
        self._value_inputs = []
        self._value_preview = None
        self._value_controls = []
        self._perturb_value_slot.clear()
        with self._perturb_value_slot:
            if kind == "color":
                self._build_color_control()
            elif kind == "channels":
                self._build_value_inputs(n, preview=self._input_transform is not None)
            elif kind == "scalar":
                self._build_value_inputs(1, preview=False)
        # Freshly built controls default to enabled; match the freeze state.
        if self._value_controls:
            _set_controls_enabled(self._value_controls, not self._frozen)

    def _build_color_control(self) -> None:
        # A compact color swatch that opens the picker on click; the button's
        # background *is* the current color.
        self._color_button = ui.button().props("dense unelevated").style(
            self._swatch_style()
        ).tooltip("Perturb color — click to change")
        with self._color_button:
            picker = ui.color_picker(on_pick=self._on_pick_color)
            # Hex only: normalized_color expects #rrggbb, so don't let the
            # picker emit rgba()/hsl() strings.
            picker.q_color.props("format-model=hex")
        self._value_controls = [self._color_button]

    def _build_value_inputs(self, n: int, *, preview: bool) -> None:
        # An optional swatch previewing the color these values map to via the
        # transform, then one field per channel (just one for a flat input,
        # whose clicked channel is chosen by the click). Fields wrap so a high
        # channel count stays usable.
        if preview:
            self._value_preview = ui.element("div").style(
                self._preview_style("#888888")
            ).tooltip("Preview: the color these channel values map to")
        with ui.row().classes("items-center gap-1 flex-wrap justify-end"):
            for i in range(n):
                tip = "Value to write" if n == 1 else f"Channel {i} value"
                field = ui.number(
                    value=0.0,
                    on_change=lambda _e: self._update_value_preview(),
                ).props("dense").classes("w-14").tooltip(tip)
                self._value_inputs.append(field)
        self._value_controls = list(self._value_inputs)
        self._update_value_preview()

    @staticmethod
    def _preview_style(color: str) -> str:
        return (
            f"background-color: {color}; width: 24px; height: 24px; "
            "border: 1px solid #cbd5e1; border-radius: 3px"
        )

    def _update_value_preview(self) -> None:
        """Repaint the preview swatch from the current channel field values."""
        if self._value_preview is None or self._input_transform is None:
            return
        values = self._value_tuple()
        color = transform_preview_color(self._input_transform, values)
        self._value_preview.style(self._preview_style(color or "#888888"))

    def _value_tuple(self) -> tuple[float, ...]:
        """The current per-channel field values (missing/blank read as 0)."""
        return tuple(
            float(inp.value if inp.value is not None else 0.0)
            for inp in self._value_inputs
        )

    def _perturb_values(self, channels: int) -> tuple[float, ...] | None:
        """The model-space values a click writes, or `None` if not ready.

        Color picks back-transform via `input_mean`/`input_std`; numeric
        fields are already in model-input space.
        """
        if self._perturb_kind == "color":
            return normalized_color(
                self._color, channels, self._input_mean, self._input_std
            )
        if self._perturb_kind == "channels" and len(self._value_inputs) == channels:
            return self._value_tuple()
        return None

    def set_frozen(self, frozen: bool) -> None:
        """Disable every control while the main view is being recorded.

        The recording renders with the live probe state (pinned batch,
        perturbations, sample), so the panel must not change it. No-op when
        the state didn't change, so the page can call this every tick.
        """
        if frozen == self._frozen:
            return
        self._frozen = frozen
        controls: list[DisableableElement] = [
            self._sample_input,
            self._pin_switch,
            self._mode_toggle,
            self._perturb_switch,
            self._clear_button,
            *self._value_controls,
        ]
        if self._input_select is not None:
            controls.append(self._input_select)
        _set_controls_enabled(controls, not frozen)

    def set_image(self, src: str) -> None:
        self._image.set_source(src)

    def set_input_legend(self, src: str | None) -> None:
        """Show/hide the flat-input scale bar and set the pane's strip mode.

        A non-empty `src` means the pane shows a flat-input strip: the image is
        stretched to a fixed strip height and the colorbar appears beside it.
        An empty `src` is an image input: natural height, legend hidden.
        """
        is_strip = bool(src)
        if is_strip != self._strip_mode:
            self._strip_mode = is_strip
            self._image.style(
                "image-rendering:pixelated; height:48px; object-fit:fill"
                if is_strip
                else "image-rendering:pixelated; height:auto; object-fit:contain"
            )
            self._input_legend.set_visibility(is_strip)
        if is_strip:
            self._input_legend.set_source(src)

    def set_input_warning(self, text: str | None) -> None:
        """Show or hide the blank-input hint under the image (no-op if same)."""
        new = text or ""
        if self._input_warning_label.text != new:
            self._input_warning_label.text = new
        visible = bool(new)
        if self._input_warning_label.visible != visible:
            self._input_warning_label.set_visibility(visible)

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
        the pin switch follows `is_pinned`, the perturb switch follows
        `perturbations` (or the local armed intent), and the perturbation
        count / clear row / compare note follow `perturbations` (the perturbed
        image itself rides along on the shared probe result the page tick
        re-renders). The perturb value control is rebuilt here when the
        selected input's channel kind changes.
        """
        self._sync_perturb_control()
        if self._pin_switch.value != self._session.is_pinned:
            self._syncing_pin = True
            self._pin_switch.set_value(self._session.is_pinned)
            self._syncing_pin = False
        # The switch is on while this tab is armed or any perturbation exists,
        # so a perturbation made elsewhere (another tab, or surviving a rebuild
        # after navigating back from /stats) turns the switch on, and an
        # external clear turns it back off in tabs that didn't arm it.
        perturb_on = self._perturb_armed or bool(self._session.perturbations)
        if self._perturb_switch.value != perturb_on:
            self._syncing_perturb = True
            self._perturb_switch.set_value(perturb_on)
            self._syncing_perturb = False
            self._set_perturb_cursor(perturb_on)
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
        """The selected input batch the displayed image was rendered from."""
        name = self._selected_input
        if name is None:
            return None
        probe = self._session.probe_result
        if probe is not None:
            return probe.shown_input(name)
        snap = self._session.snapshot
        if snap is None:
            return None
        return snap.activations.get(name)

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
        if self._syncing_perturb:
            return  # programmatic mirror of shared perturbations, not a click
        value = bool(getattr(e, "value", False))
        self._perturb_armed = value
        self._set_perturb_cursor(value)
        if not value:
            # Leaving editing mode discards the edits and the diff view: the
            # image and strips revert to the unperturbed input (clearing the
            # perturbations also deactivates `compare`, which derives from
            # them).
            self._on_clear()

    def _on_image_click(self, e: object) -> None:
        if self._frozen or not self._perturb_switch.value:
            return
        name = self._selected_input
        if name is None:
            return
        tensor = self._current_input()
        if tensor is None:
            return
        click_x = int(getattr(e, "image_x", 0))
        click_y = int(getattr(e, "image_y", 0))
        if tensor.ndim == 4:
            values = self._perturb_values(int(tensor.shape[1]))
            if values is None:
                return
            h, w = int(tensor.shape[2]), int(tensor.shape[3])
            index: tuple[int, ...] = (
                min(max(click_y, 0), h - 1),
                min(max(click_x, 0), w - 1),
            )
        elif tensor.ndim == 2:
            # The strip's native width is C, so image_x is the channel; the
            # single field holds the scalar written there.
            if self._perturb_kind != "scalar" or not self._value_inputs:
                return
            channel = min(max(click_x, 0), int(tensor.shape[1]) - 1)
            index = (channel,)
            values = self._value_tuple()
        else:
            return
        self._session.add_perturbation(
            input_name=name, sample=self.sample_idx, index=index, values=values
        )
