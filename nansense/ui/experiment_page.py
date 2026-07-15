"""The `/experiment` page: deep dream and Captum attributions."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from urllib.parse import quote
from uuid import uuid4

import torch
from nicegui import ui
from nicegui.elements.mixins.disableable_element import DisableableElement
from nicegui.elements.mixins.value_element import ValueElement
from torch import Tensor

from nansense.experiments import (
    _DEFAULT_DREAM_BATCH,
    EXPERIMENT_KINDS,
    ExperimentResult,
    available_experiment_kinds,
    layer_available,
)
from nansense.recording import RecordedView
from nansense.session import BatchSnapshot, Session
from nansense.ui.common import (
    _b64_img_src,
    _defer_value_write,
    _install_panel_resize,
    _label_bar_html,
    _page_scaffold,
    _resizable_pane_props,
    _resize_handle,
    _set_controls_enabled,
    _strip_html,
    _weights_placeholder,
)
from nansense.ui.render import (
    INPUT_IMAGE_SIZE,
    StripRender,
    render_attribution_overlay,
    render_image,
    render_strip,
    tensor_hw,
)
from nansense.ui.static import _STRIP_MARKER_CSS
from nansense.ui.top_bar import (
    _add_error_banner,
    _add_repo_logo,
    _add_settings_button,
    _add_share_button,
    _add_step_controls,
    _add_tour_button,
    _back_button,
    _build_step_until_custom_dialog,
    _top_bar_row,
)
from nansense.ui.tour import add_tour, experiment_tour_steps


@dataclass(frozen=True)
class _ExperimentParam:
    """One configurable knob of an experiment, rendered as a form widget."""

    key: str
    label: str
    kind: str  # "int" | "float" | "bool" | "select"
    default: object
    options: dict[str, str] | None = None
    minimum: float | None = None
    step: float | None = None
    tooltip: str = ""


# Shared knobs reused across kinds. A param is shared *by key*, so its value
# survives switching experiment type (point 1): set "channel" for Neuron
# Gradient and it carries over to Occlusion, "Inputs" carries everywhere, …
_CHANNEL_PARAM = _ExperimentParam(
    "channel",
    "Channel (-1 = whole layer)",
    "int",
    0,
    minimum=-1,
    tooltip="Which channel / feature of the selected layer to target",
)
_CHANNELS_PARAM = _ExperimentParam(
    "channels",
    "Channels",
    "int",
    _DEFAULT_DREAM_BATCH,
    minimum=1,
    tooltip=(
        "Dream on this many of the layer's first channels — one synthesized "
        "sample per channel (capped at the layer's channel count)"
    ),
)
_MINIMIZE_PARAM = _ExperimentParam(
    "minimize",
    "Minimize activations",
    "bool",
    False,
    tooltip=(
        "Descend the objective instead of ascending it — synthesize an input "
        "that suppresses each channel rather than excites it"
    ),
)
_SAMPLE_PARAM = _ExperimentParam(
    "sample",
    "Sample",
    "int",
    0,
    minimum=0,
    tooltip="Which input-batch sample every channel's dream starts from",
)
_TARGET_PARAM = _ExperimentParam(
    "target",
    "Target class (-1 = argmax)",
    "int",
    -1,
    minimum=-1,
    tooltip="Class index Grad-CAM explains; -1 uses each sample's prediction",
)
_BATCH_PARAM = _ExperimentParam(
    "batch",
    "Inputs",
    "int",
    _DEFAULT_DREAM_BATCH,
    minimum=1,
    tooltip=(
        "How many inputs to run on (defaults to the current batch size, "
        f"capped at {_DEFAULT_DREAM_BATCH})"
    ),
)
_START_PARAM = _ExperimentParam(
    "start",
    "Start from",
    "select",
    "noise",
    options={"noise": "Noise", "sample": "Current batch"},
    tooltip=(
        "Noise draws fresh inputs shaped and scaled like the network's real "
        "input — different on every run; Current batch starts from the real "
        "input batch itself"
    ),
)
_CLAMP_PARAM = _ExperimentParam(
    "clamp",
    "Clamp to displayable range",
    "bool",
    True,
    tooltip="Keep pixels inside the [0, 1] display range mapped through the input mean/std",
)
_DIFFUSION_PARAM = _ExperimentParam(
    "diffusion",
    "Diffusion",
    "float",
    0.05,
    minimum=0,
    step=0.01,
    tooltip="Per-step blend with a 3×3 blur; damps high-frequency noise",
)
_JITTER_PARAM = _ExperimentParam(
    "jitter",
    "Jitter (px)",
    "int",
    2,
    minimum=0,
    tooltip="Random shift each step, undone after the update; reduces pixel-grid artifacts",
)
_ZOOM_PARAM = _ExperimentParam(
    "zoom",
    "Zoom multiplier per step",
    "float",
    1.0,
    minimum=1,
    step=0.01,
    tooltip=(
        "Per-step center zoom-in factor (1 = no zoom; on small inputs it "
        "only takes effect above ~1 + 1/size)"
    ),
)

# Ordered per kind: the targeting knob first (deep dream's Channels, Captum's
# Channel/Target), then Inputs (Captum) or Start from + Sample (deep dream),
# then the method-specific knobs (point 1). The Layer selector is rendered
# above this list (point 2). Deep dream's Sample knob only shows when starting
# from the current batch (toggled in `rebuild_params`).
_EXPERIMENT_PARAMS: dict[str, list[_ExperimentParam]] = {
    "deep_dream": [
        _CHANNELS_PARAM,
        _START_PARAM,
        _SAMPLE_PARAM,
        _ExperimentParam("steps", "Steps", "int", 300, minimum=1),
        _ExperimentParam("lr", "Learning rate", "float", 0.05, minimum=0, step=0.01),
        _DIFFUSION_PARAM,
        _JITTER_PARAM,
        _ZOOM_PARAM,
        # The objective-direction toggle sits with the value-range knob below it.
        _MINIMIZE_PARAM,
        _CLAMP_PARAM,
    ],
    "gradcam": [_TARGET_PARAM, _BATCH_PARAM],
    "neuron_gradient": [_CHANNEL_PARAM, _BATCH_PARAM],
    "neuron_ig": [
        _CHANNEL_PARAM,
        _BATCH_PARAM,
        _ExperimentParam("ig_steps", "Integration steps", "int", 32, minimum=2),
    ],
    "occlusion": [
        _CHANNEL_PARAM,
        _BATCH_PARAM,
        _ExperimentParam(
            "window",
            "Window (px)",
            "int",
            4,
            minimum=1,
            tooltip="Side length of the occluding patch",
        ),
        _ExperimentParam("stride", "Stride (px)", "int", 2, minimum=1),
    ],
}

# Per kind: (short tooltip on the dropdown, long description shown at the
# bottom of the left pane) — point 4.
_EXPERIMENT_DESCRIPTIONS: dict[str, tuple[str, str]] = {
    "deep_dream": (
        "Synthesize one input per channel that maximally excites it.",
        "Deep Dream runs gradient ascent on the input to maximize each of the "
        "layer's first channels' mean activation — one synthesized sample per "
        "channel, a picture of what each unit 'wants' to see.",
    ),
    "gradcam": (
        "Coarse class heatmap localized onto the selected layer.",
        "Grad-CAM weights the selected layer's feature maps by the gradient "
        "of the target class score, giving a coarse heatmap of where that "
        "layer supports the class.",
    ),
    "neuron_gradient": (
        "Raw input gradient of one channel — tends to look grainy.",
        "Neuron Gradient is the raw input-space gradient of one channel's "
        "activation — the gradient view of its receptive field. It tends to "
        "produce noisy, high-frequency (grainy) attribution maps; Neuron "
        "Integrated Gradients is the smoother alternative.",
    ),
    "neuron_ig": (
        "Path-integrated input attribution of one channel (cleaner).",
        "Neuron Integrated Gradients integrates one channel's input gradient "
        "along a path from a zero baseline, giving a cleaner, less noisy "
        "version of Neuron Gradient.",
    ),
    "occlusion": (
        "Drop in a channel's activation as input regions are occluded.",
        "Occlusion slides a patch over the input and measures how much the "
        "selected layer-channel's mean activation drops — a perturbation "
        "view of that channel's receptive field.",
    ),
}


# Experiment cell caption colors, echoing the main view's activation (green) /
# gradient (purple) markers: the input is green, the attribution / overlay
# purple, and everything else (deep-dream channel images) the neutral slate the
# `CHANNEL n` column headers use.
_CAPTION_COLORS: dict[str, str] = {
    "input": "#10b981",
    "attribution": "#8b5cf6",
    "overlay": "#8b5cf6",
}


def _caption_bar_color(caption: str) -> str:
    """Bar color for an experiment cell caption (first word keys the map)."""
    head = caption.split(" ", 1)[0].lower()
    return _CAPTION_COLORS.get(head, "#64748b")


def _coerce_number(spec: _ExperimentParam, *candidates: object) -> int | float:
    """The first numeric `candidate` cast to the spec's type. A cleared number
    field reads back from NiceGUI as None, so callers pass the widget value, the
    persisted value and finally the always-numeric default — the cast then can
    never see a None."""
    for candidate in candidates:
        if isinstance(candidate, (int, float)):
            return int(candidate) if spec.kind == "int" else float(candidate)
    raise AssertionError(f"no numeric value for {spec.key!r}")  # default is numeric


def _experiment_status(result: ExperimentResult) -> str:
    state = "running"
    if result.done:
        state = "stopped early" if result.step < result.total_steps else "done"
    if result.error is not None:
        state = "failed"
    text = f"{EXPERIMENT_KINDS.get(result.kind, result.kind)} — {state}"
    if result.total_steps > 1:
        text += f" · step {result.step}/{result.total_steps}"
    if result.objective is not None:
        text += f" · objective {result.objective:.4g}"
    return text


def _experiment_img_html(image: bytes | None) -> str:
    """Input-space experiment image, CSS-upscaled like the input pane."""
    if image is None:
        return '<div class="text-xs text-slate-400 italic">not renderable</div>'
    return (
        f'<img src="{_b64_img_src(image)}" '
        f'style="width:{INPUT_IMAGE_SIZE}px; image-rendering:pixelated; '
        'display:block; max-width:none;" />'
    )


def _layer_channel_count(snap: BatchSnapshot | None, layer: str) -> int | None:
    """Channel count of `layer`'s last captured activation (None if unknown)."""
    act = snap.activations.get(layer) if snap is not None else None
    if act is None or act.ndim < 2:
        return None
    return int(act.shape[1])


def _attribution_vmax(attribution: Tensor) -> float:
    """Largest `|x|` over finite attribution values — the overlay's ±scale."""
    finite = attribution[torch.isfinite(attribution)]
    return float(finite.abs().max()) if finite.numel() else 0.0


@dataclass
class _ExperimentPageState:
    """Mutable page state shared by the form, Run/Cancel, and tick closures."""

    kind: str = "deep_dream"
    layer: str = ""
    # Parameter values persisted across kind switches (point 1): keyed by
    # param key, so a shared key keeps its value when the experiment changes.
    values: dict[str, object] = field(default_factory=dict)
    # Render Captum attributions blended over the input instead of beside it.
    overlay: bool = False
    # This page's own request; `None` until the first run.
    my_seq: int | None = None
    last_result: ExperimentResult | None = None
    # A run is needed (init, or a parameter / layer change). The tick coalesces
    # these and (re)registers at most once per tick, so a burst of edits never
    # floods the backend.
    dirty: bool = True
    # Last enable/disable flags pushed to the client (push only on change).
    frozen: bool | None = None
    run_enabled: bool | None = None
    cancel_enabled: bool | None = None


def _minmax_stats_href(layer: str) -> str:
    """Deep-link to `layer`'s MIN/MAX stats view — the real-input extremes
    that complement its synthesized dreams. `?view` opens straight on the
    grids rather than the histogram default."""
    return f"/stats?layer={quote(layer)}&view=minmax"


def _build_experiment_page(
    session: Session,
    layer: str,
    *,
    input_mean: tuple[float, ...] | None,
    input_std: tuple[float, ...] | None,
) -> None:
    """Per-layer experiments: deep dream and selected Captum attributions.

    The top bar carries the shared stepping controls — experiments execute on
    the paused training thread, so the user can pause right from this page.
    The left pane holds the experiment-kind dropdown, Run / Cancel, the
    selected kind's parameter form (headed by a layer selector and rebuilt on
    every dropdown change), a Captum overlay toggle, and a description of the
    chosen experiment. The right pane streams this page's *own* results
    (`experiment_result_for`, so concurrent tabs never overwrite each other)
    as one card per sample with captioned cells: the input image beside its
    deep-dream result, or the attribution map beside its input (the map first),
    or — when overlay is on — the attribution blended over the input (the
    MIN/MAX heat-overlay scheme). Attribution maps and overlays are sized to
    the input image they sit next to.
    """
    _page_scaffold("Experiment")
    _install_panel_resize()
    ui.add_head_html(_STRIP_MARKER_CSS)

    input_set = set(session.input_names)
    selectable_layers = [n for n in session.layer_names if n not in input_set]
    if not selectable_layers:
        with ui.column().classes("w-full h-screen no-wrap gap-0"):
            with _top_bar_row():
                _back_button()
                _add_repo_logo().classes("ml-auto")
            _weights_placeholder("No layers available to experiment on.")
        return
    initial_layer = layer if layer in selectable_layers else selectable_layers[0]
    add_tour(
        "experiment", experiment_tour_steps(locked=session.locked),
        locked=session.locked,
    )

    step_until_custom = _build_step_until_custom_dialog(session)
    widgets: dict[str, ui.element] = {}
    state = _ExperimentPageState(layer=initial_layer)
    # Seed persisted values with every kind's defaults so a freshly-shown
    # widget always has a value, even for a key the user hasn't touched.
    for specs in _EXPERIMENT_PARAMS.values():
        for spec in specs:
            state.values.setdefault(spec.key, spec.default)

    # This page's auto-experiment registration: a run registers the request so
    # it also re-runs on every visualization update (same seq → same seeded
    # noise); the page's tick heartbeats it and it expires when the page
    # closes, unless a recording pins it. The record key follows the *current*
    # layer so a recording captures the layer being viewed.
    page_key = f"experiment-page-{uuid4().hex}"

    def record_key() -> str:
        return f"experiment:{state.layer}"

    def collect_params() -> dict[str, object]:
        params: dict[str, object] = {"mean": input_mean, "std": input_std}
        for spec in _EXPERIMENT_PARAMS[state.kind]:
            value: object = getattr(widgets.get(spec.key), "value", None)
            if spec.kind in ("int", "float"):
                # `run` blocks the call while a numeric field is empty; the
                # persisted value / numeric-default fallbacks guard the rest.
                params[spec.key] = _coerce_number(
                    spec, value, state.values.get(spec.key), spec.default
                )
            elif spec.kind == "bool":
                params[spec.key] = bool(value)
            else:
                params[spec.key] = str(value if value is not None else spec.default)
        return params

    def run() -> None:
        if _refresh_param_error():  # an empty/non-numeric field — don't run
            return
        if state.my_seq is not None:  # a re-run replaces this page's request
            session.cancel_experiment(state.my_seq)
        state.my_seq = session.register_auto_experiment(
            page_key, kind=state.kind, layer=state.layer, params=collect_params()
        )
        state.last_result = None
        error_label.text = ""

    def cancel() -> None:
        if state.my_seq is None:
            return
        session.cancel_experiment(state.my_seq)
        # Stop the auto reruns too — unless a recording pinned the request
        # (Cancel is disabled while recorded, but another tab may differ).
        if not session.recording.is_recording(record_key()):
            session.unregister_auto_experiment(page_key)

    def record_view() -> RecordedView | None:
        if state.my_seq is None:
            return None  # nothing to record until an experiment has run
        kind = state.kind
        return RecordedView(
            key=record_key(),
            page="experiment",
            label=f"Experiment · {EXPERIMENT_KINDS.get(kind, kind)} · {state.layer}",
            params={
                "layer": state.layer,
                "seq": state.my_seq,
                "auto_key": page_key,
                "input_mean": input_mean,
                "input_std": input_std,
            },
        )

    def schedule_run() -> None:
        state.dirty = True

    def on_kind_change(e: object) -> None:
        value = getattr(e, "value", None)
        if value is None:
            return
        state.kind = str(value)
        refresh_layer_options()
        # Moving to a kind the current layer can't run keeps the form valid by
        # hopping to the first layer that can (point 2).
        if not layer_available(session, state.layer, state.kind):
            available = next(
                (n for n in selectable_layers if layer_available(session, n, state.kind)),
                None,
            )
            if available is not None:
                state.layer = available
                _defer_value_write(lambda: layer_select.set_value(available))
        rebuild_params()
        update_description()
        overlay_switch.set_visibility(state.kind != "deep_dream")
        compare_button.set_visibility(state.kind == "deep_dream")
        sync_compare_href()
        schedule_run()

    def on_layer_change(e: object) -> None:
        value = getattr(e, "value", None)
        if value is None:
            return
        if not layer_available(session, str(value), state.kind):
            # A disabled option shouldn't be selectable, but guard anyway.
            _defer_value_write(lambda: layer_select.set_value(state.layer))
            ui.notify(
                f"{EXPERIMENT_KINDS[state.kind]} can't run on {value}",
                type="warning",
            )
            return
        state.layer = str(value)
        sync_compare_href()
        clip_channel()
        schedule_run()

    def on_overlay_change(e: object) -> None:
        # A pure display toggle: re-render the current result, no backend re-run.
        state.overlay = bool(getattr(e, "value", False))
        if state.last_result is not None and state.last_result.error is None:
            render_result(state.last_result)

    def sync_compare_href() -> None:
        # Keep the compare link on the current layer. An `href` (not an
        # `on_click` navigate) renders the button as a real anchor, so
        # middle-click / ctrl-click open the stats view in a new tab.
        compare_button.props(f'href="{_minmax_stats_href(state.layer)}"')

    with ui.column().classes("w-full h-screen no-wrap gap-0"):
        with _top_bar_row():
            _back_button()
            _add_step_controls(session, step_until_custom)
            _add_settings_button(session, record_view).classes("ml-auto")
            _add_tour_button()
            _add_share_button()
            _add_repo_logo()

        _add_error_banner(session)

        with ui.row().classes("w-full grow min-h-0 no-wrap gap-0"):
            with ui.column().classes(
                "w-80 shrink-0 h-full overflow-auto p-4 gap-2 "
                "border-r-2 border-slate-300 bg-slate-50"
            ).props(_resizable_pane_props("experiment-controls")):
                ui.label("Experiment").classes("font-mono text-base font-bold")
                ui.separator()
                # `data-tour` marks the two selectors as the tour's arrow
                # targets (`tour.experiment_tour_steps`).
                kind_select = ui.select(
                    available_experiment_kinds(),
                    label="Experiment",
                    value=state.kind,
                    on_change=on_kind_change,
                ).props('dense outlined data-tour="kind"').classes("w-full")
                with kind_select:
                    kind_tooltip = ui.tooltip("")
                with ui.row().classes("w-full no-wrap gap-2"):
                    run_button = (
                        ui.button("Run", icon="science", on_click=run, color="yellow-8")
                        .props("dense size=md")
                        .classes("grow")
                        .tooltip(
                            "Run the experiment (training must be paused). "
                            "Disabled while auto-run is on or a run is in flight."
                        )
                    )
                    cancel_button = (
                        ui.button("Cancel", on_click=cancel, color="slate-500")
                        .props("dense size=md")
                        .classes("grow")
                        .tooltip("Abort this page's experiment and its automatic reruns")
                    )
                ui.separator()
                ui.label("Parameters").classes("font-mono text-sm")
                # The layer selector is the first configurable parameter
                # (point 2); it lives above the rebuilt-per-kind form so a
                # kind change never clears it. Unavailable layers are grayed
                # out via a `disable` flag on each option.
                layer_select = (
                    ui.select(
                        selectable_layers,
                        value=state.layer,
                        label="Layer",
                        on_change=on_layer_change,
                    )
                    .props(
                        "dense outlined options-dense option-disable=disable "
                        'data-tour="layer"'
                    )
                    .classes("w-full")
                )
                params_pane = ui.column().classes("w-full gap-2 p-0")
                # Flags an empty / non-numeric number field (which would
                # otherwise read back as None and crash the run).
                param_error_label = ui.label("").classes(
                    "text-xs text-red-600 whitespace-normal leading-snug"
                )
                overlay_switch = (
                    ui.switch("Overlay on input", value=state.overlay, on_change=on_overlay_change)
                    .props("dense")
                    .tooltip(
                        "Blend each attribution map over its input image "
                        "instead of showing them side by side"
                    )
                )
                overlay_switch.set_visibility(state.kind != "deep_dream")
                ui.space()
                # Deep-dream only: jump to the same layer's MIN/MAX stats — the
                # real-input extremes that complement the synthesized dreams
                # (point 3). Sits just above the kind description.
                compare_button = (
                    ui.button(
                        "Compare with MIN/MAX",
                        icon="bar_chart",
                        color="teal",
                    )
                    .props(
                        f'dense no-caps size=sm href="{_minmax_stats_href(state.layer)}"'
                    )
                    .classes("w-full")
                    .tooltip(
                        "Open this layer's MIN/MAX stats — the real inputs that "
                        "most excite the same channels"
                    )
                )
                compare_button.set_visibility(state.kind == "deep_dream")
                description_label = ui.label("").classes(
                    "text-xs text-slate-600 whitespace-normal leading-snug "
                    "border-t border-slate-300 pt-2 mt-1"
                )
            _resize_handle("experiment-controls", "left")
            with ui.column().classes(
                "grow min-w-0 h-full overflow-auto p-4 gap-3 bg-slate-200"
            ):
                status_label = ui.label(
                    "Adjust parameters — the experiment runs automatically "
                    "(training must be paused)."
                ).classes("text-sm text-slate-600")
                error_label = ui.label("").classes("text-sm text-red-600")
                results_col = ui.column().classes("gap-2 w-full")

    def _layer_options_with_disable() -> list[dict[str, object]]:
        return [
            {
                "value": index,
                "label": name,
                "disable": not layer_available(session, name, state.kind),
            }
            for index, name in enumerate(selectable_layers)
        ]

    def _patched_update_options() -> None:
        # NiceGUI's ChoiceElement.update regenerates `_props['options']` as
        # plain `{value, label}` dicts on every refresh, dropping a `disable`
        # flag. Reassigning the whole list here (rather than mutating it in
        # place) routes through Props change-tracking, so Quasar's
        # `option-disable` reliably grays the unavailable layers (point 2).
        before = layer_select.value
        layer_select._props["options"] = _layer_options_with_disable()
        layer_select._props[layer_select.VALUE_PROP] = (
            layer_select._value_to_model_value(before)
        )
        if not isinstance(before, list):
            layer_select.value = before if before in layer_select._values else None

    # Swap in the disable-aware option generator (ChoiceElement.update calls
    # the instance's `_update_options`); setattr keeps the static checker calm
    # about shadowing a method.
    setattr(layer_select, "_update_options", _patched_update_options)

    def refresh_layer_options() -> None:
        """Re-gray the layer options for the current kind (point 2)."""
        layer_select.update()

    def _clip_number(key: str, maximum: int) -> None:
        """Pin a number widget's max, clipping its value when the layer shrank."""
        widget = widgets.get(key)
        if not isinstance(widget, ui.number):
            return
        widget.max = maximum
        current = widget.value
        if isinstance(current, (int, float)) and current > maximum:
            state.values[key] = maximum
            _defer_value_write(lambda: widget.set_value(maximum))

    def clip_channel() -> None:
        """Pin the targeting widgets to the layer's channel count, clipping a
        value the new layer can no longer reach (point 2): deep dream's
        Channels is a count of the first N (max = channels), Captum's Channel
        is a single index (max = channels − 1)."""
        channels = _layer_channel_count(session.snapshot, state.layer)
        if channels is None:
            return
        _clip_number("channels", channels)
        _clip_number("channel", channels - 1)

    def _sync_sample_visibility() -> None:
        """Show deep dream's Sample knob only when starting from the current
        batch — noise has no input to pick (point 2)."""
        sample_widget = widgets.get("sample")
        start_widget = widgets.get("start")
        if sample_widget is None or start_widget is None:
            return
        sample_widget.set_visibility(getattr(start_widget, "value", None) == "sample")

    def _invalid_number_fields() -> list[str]:
        """Labels of numeric params whose widget holds no usable number — an
        empty or non-numeric field reads back from NiceGUI as None."""
        invalid: list[str] = []
        for spec in _EXPERIMENT_PARAMS[state.kind]:
            if spec.kind not in ("int", "float"):
                continue
            value = getattr(widgets.get(spec.key), "value", None)
            if not isinstance(value, (int, float)):
                invalid.append(spec.label)
        return invalid

    def _refresh_param_error() -> list[str]:
        """Sync the red hint with the current fields; return the invalid ones."""
        invalid = _invalid_number_fields()
        param_error_label.text = (
            "Enter a number for: " + ", ".join(invalid) if invalid else ""
        )
        return invalid

    def _on_param_change(key: str, widget: ui.element) -> None:
        value = getattr(widget, "value", None)
        # A cleared / non-numeric number field reads back as None; keep the last
        # good value rather than persisting it — `run` reports it as a red hint.
        if not (isinstance(widget, ui.number) and not isinstance(value, (int, float))):
            state.values[key] = value
            if key == "start":
                _sync_sample_visibility()
        _refresh_param_error()
        schedule_run()

    def rebuild_params() -> None:
        widgets.clear()
        params_pane.clear()
        with params_pane:
            for spec in _EXPERIMENT_PARAMS[state.kind]:
                initial = state.values.get(spec.key, spec.default)
                if spec.kind == "bool":
                    widget: ValueElement = ui.switch(
                        spec.label, value=bool(initial)
                    ).props("dense")
                elif spec.kind == "select":
                    widget = (
                        ui.select(spec.options or {}, label=spec.label, value=initial)
                        .props("dense outlined")
                        .classes("w-full")
                    )
                else:
                    default = initial
                    if spec.key == "batch" and not isinstance(default, (int, float)):
                        live = session.input_batch_size
                        default = min(_DEFAULT_DREAM_BATCH, live) if live else _DEFAULT_DREAM_BATCH
                    maximum: float | None = None
                    if spec.key in ("channel", "channels"):
                        channels = _layer_channel_count(session.snapshot, state.layer)
                        if channels is not None:
                            # Channel is a single index; Channels is a count.
                            maximum = channels - 1 if spec.key == "channel" else channels
                    elif spec.key == "sample":
                        live = session.input_batch_size
                        if live:
                            maximum = live - 1
                    default_number = default if isinstance(default, (int, float)) else 0
                    widget = (
                        ui.number(
                            label=spec.label,
                            value=default_number,
                            min=spec.minimum,
                            max=maximum,
                            step=1 if spec.kind == "int" else spec.step,
                            format="%d" if spec.kind == "int" else None,
                        )
                        .props("dense outlined")
                        .classes("w-full")
                    )
                if spec.tooltip:
                    widget.tooltip(spec.tooltip)
                widget.on_value_change(
                    lambda _e, k=spec.key, w=widget: _on_param_change(k, w)
                )
                widgets[spec.key] = widget
        _sync_sample_visibility()
        if state.frozen:
            _set_controls_enabled(_param_controls(), False)

    def _param_controls() -> list[DisableableElement]:
        controls = [w for w in widgets.values() if isinstance(w, DisableableElement)]
        if isinstance(layer_select, DisableableElement):
            controls.append(layer_select)
        return controls

    def update_description() -> None:
        short, long = _EXPERIMENT_DESCRIPTIONS.get(state.kind, ("", ""))
        description_label.text = long
        kind_tooltip.set_text(short)

    def _image_widget(tensor: Tensor | None, sample_idx: int) -> None:
        ui.html(
            _experiment_img_html(
                render_image(tensor, sample_idx, mean=input_mean, std=input_std)
            )
        )

    def _strip_widget(strip: StripRender | None) -> None:
        with ui.element("div").classes("max-w-full overflow-x-auto"):
            ui.html(_strip_html(strip, show_labels=True))

    def _captioned_cells(cells: list[tuple[str, Callable[[], None]]]) -> None:
        """A horizontal, scrollable row of captioned cells (caption over body),
        consistent with the watch / weights cards. Captions are filled color
        bars matching the main view's markers: input green, attribution/overlay
        purple, deep-dream channels slate (`_caption_bar_color`)."""
        with ui.row().classes("items-start gap-4 no-wrap w-full overflow-x-auto"):
            for caption, build in cells:
                with ui.column().classes("items-center gap-1 shrink-0"):
                    ui.html(
                        _label_bar_html(
                            caption.upper(), color=_caption_bar_color(caption)
                        )
                    ).classes("w-full")
                    build()

    def _sample_card(idx: int, cells: list[tuple[str, Callable[[], None]]]) -> None:
        """One Captum result card per sample: a label over a row of captioned
        cells."""
        with ui.card().classes("w-full p-3 gap-2"):
            ui.label(f"Sample {idx}").classes(
                "font-mono text-sm font-bold text-slate-600"
            )
            _captioned_cells(cells)

    def render_result(result: ExperimentResult) -> None:
        """Deep dream renders one card holding a single horizontal row — the
        starting input (only when dreaming from the current batch) followed by
        one dreamed image per channel (points 1–3). Captum keeps one card per
        sample: the attribution map beside its input (the map first, point 1),
        or — with overlay on — the attribution blended over the input.
        Attribution maps and overlays are sized to match the input (point 2)."""
        results_col.clear()
        with results_col:
            if result.image is not None:
                _render_image_row(result.reference, result.image)
            elif result.attribution is not None:
                _render_attribution_cards(result)

    def _render_image_row(reference: Tensor | None, image: Tensor) -> None:
        # One card, one horizontal row: the shared input (current-batch start
        # only) then one dreamed image per channel (points 2, 3).
        cells: list[tuple[str, Callable[[], None]]] = []
        if reference is not None:
            cells.append(("input", lambda: _image_widget(reference, 0)))
        for i in range(int(image.shape[0])):
            cells.append((f"channel {i}", lambda i=i: _image_widget(image, i)))
        with ui.card().classes("w-full p-3 gap-2"):
            _captioned_cells(cells)

    def _render_attribution_cards(result: ExperimentResult) -> None:
        attribution = result.attribution
        reference = result.reference
        assert attribution is not None
        attr = attribution  # narrowed; safe to index inside the cell closures
        n = int(attr.shape[0])
        if state.overlay and reference is not None:
            ref = reference  # narrowed for the closures below
            vmax = _attribution_vmax(attr)
            for i in range(n):
                _sample_card(
                    i,
                    [
                        (
                            "overlay",
                            lambda i=i: _strip_widget(
                                render_attribution_overlay(
                                    ref[i],
                                    attr[i],
                                    mean=input_mean,
                                    std=input_std,
                                    vmax=vmax,
                                    tile_px=INPUT_IMAGE_SIZE,
                                )
                            ),
                        )
                    ],
                )
            return
        input_hw = tensor_hw(reference)
        for i in range(n):
            # Attribution map first, input second (point 1).
            cells: list[tuple[str, Callable[[], None]]] = [
                (
                    "attribution",
                    lambda i=i: _strip_widget(
                        render_strip(
                            attr, i, input_hw=input_hw, tile_px=INPUT_IMAGE_SIZE
                        )
                    ),
                )
            ]
            if reference is not None:
                cells.append(("input", lambda i=i: _image_widget(reference, i)))
            _sample_card(i, cells)

    def update_controls(*, running: bool) -> None:
        run_ok = not state.frozen and not session.auto_run_experiments and not running
        cancel_ok = not state.frozen and running
        if run_ok != state.run_enabled:
            state.run_enabled = run_ok
            run_button.set_enabled(run_ok)
        if cancel_ok != state.cancel_enabled:
            state.cancel_enabled = cancel_ok
            cancel_button.set_enabled(cancel_ok)

    def tick() -> None:
        # Keep this page's auto experiment alive while the page is open.
        session.touch_auto_experiment(page_key)
        # While this experiment records, its request must stay as-is: a re-run
        # would replace the recorded seq and parameter edits would lie.
        frozen = session.recording.is_recording(record_key())
        if frozen != state.frozen:
            state.frozen = frozen
            _set_controls_enabled(
                [kind_select, overlay_switch, *_param_controls()], not frozen
            )

        result = session.experiment_result_for(state.my_seq) if state.my_seq else None
        running = state.my_seq is not None and not (result is not None and result.done)
        update_controls(running=running)

        # Auto-run: register (or re-register on change) without a manual Run.
        if session.auto_run_experiments and state.dirty and not frozen:
            state.dirty = False
            run()

        if state.my_seq is None:
            return
        if result is None:
            if state.last_result is None:
                status_label.text = (
                    "queued — waiting for training to pause, or for earlier "
                    "experiments to finish (use Stop / Step Batch above)"
                )
            return
        status_label.text = _experiment_status(result)
        error_label.text = result.error or ""
        if result is not state.last_result:
            state.last_result = result
            if result.error is None:
                render_result(result)

    refresh_layer_options()
    rebuild_params()
    update_description()
    ui.timer(0.2, tick)
