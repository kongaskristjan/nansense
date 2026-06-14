"""The `/experiment` page: deep dream and Captum attributions."""

from __future__ import annotations

from dataclasses import dataclass, field
from uuid import uuid4

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
    _page_scaffold,
    _resizable_pane_props,
    _resize_handle,
    _set_controls_enabled,
    _strip_html,
    _weights_placeholder,
)
from nansense.ui.input_panel import InputPanel
from nansense.ui.render import INPUT_IMAGE_SIZE, render_image, render_strip, tensor_hw
from nansense.ui.static import _STRIP_MARKER_CSS
from nansense.ui.top_bar import (
    _add_settings_button,
    _add_step_controls,
    _back_button,
    _build_step_until_custom_dialog,
    _top_bar_row,
)


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
# Gradient and it carries over to Deep Dream, "Inputs" carries everywhere, …
_CHANNEL_PARAM = _ExperimentParam(
    "channel",
    "Channel (-1 = whole layer)",
    "int",
    0,
    minimum=-1,
    tooltip="Which channel / feature of the selected layer to target",
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
        f"capped at {_DEFAULT_DREAM_BATCH}); ignored while 'Use viewed "
        "sample' is on"
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
_USE_VIEWED_PARAM = _ExperimentParam(
    "use_viewed",
    "Use viewed sample",
    "bool",
    False,
    tooltip=(
        "Run on the single sample the Input Selection is viewing (which may "
        "be pinned and/or perturbed) instead of a fresh batch. A perturbed "
        "sample shows the attribution diff (perturbed − original)"
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

# Ordered per kind: Channel/Target first, then Inputs, then (deep dream)
# Start from, then Use viewed sample, then the method-specific knobs (point 1
# / 1.5). The Layer selector is rendered above this list (point 2).
_EXPERIMENT_PARAMS: dict[str, list[_ExperimentParam]] = {
    "deep_dream": [
        _CHANNEL_PARAM,
        _BATCH_PARAM,
        _START_PARAM,
        _USE_VIEWED_PARAM,
        _ExperimentParam("steps", "Steps", "int", 100, minimum=1),
        _ExperimentParam("lr", "Learning rate", "float", 0.05, minimum=0, step=0.01),
        _DIFFUSION_PARAM,
        _JITTER_PARAM,
        _ZOOM_PARAM,
        _CLAMP_PARAM,
    ],
    "gradcam": [_TARGET_PARAM, _BATCH_PARAM, _USE_VIEWED_PARAM],
    "neuron_gradient": [_CHANNEL_PARAM, _BATCH_PARAM, _USE_VIEWED_PARAM],
    "neuron_ig": [
        _CHANNEL_PARAM,
        _BATCH_PARAM,
        _USE_VIEWED_PARAM,
        _ExperimentParam("ig_steps", "Integration steps", "int", 32, minimum=2),
    ],
    "occlusion": [
        _CHANNEL_PARAM,
        _BATCH_PARAM,
        _USE_VIEWED_PARAM,
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
        "Synthesize an input that maximally excites the selected channel.",
        "Deep Dream runs gradient ascent on the input to maximize the "
        "selected channel's (or the whole layer's) mean activation — a "
        "picture of what the unit 'wants' to see.",
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


@dataclass
class _ExperimentPageState:
    """Mutable page state shared by the form, Run/Cancel, and tick closures."""

    kind: str = "deep_dream"
    layer: str = ""
    # Parameter values persisted across kind switches (point 1): keyed by
    # param key, so a shared key keeps its value when the experiment changes.
    values: dict[str, object] = field(default_factory=dict)
    # This page's own request; `None` until the first run.
    my_seq: int | None = None
    last_result: ExperimentResult | None = None
    # A run is needed (init, or a parameter / layer / input change). The tick
    # coalesces these and (re)registers at most once per tick, so a burst of
    # edits never floods the backend.
    dirty: bool = True
    # Signature of the viewed input (pin / perturbations / sample) so the tick
    # re-runs when "use viewed sample" is on and the viewed input changes.
    last_input_sig: tuple[object, ...] | None = None
    # Last enable/disable flags pushed to the client (push only on change).
    frozen: bool | None = None
    run_enabled: bool | None = None
    cancel_enabled: bool | None = None
    # Identity of the last-rendered input image source, to skip redundant
    # re-renders of the embedded input pane.
    last_input_render: tuple[object, ...] | None = None


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
    every dropdown change), and a description of the chosen experiment. The
    right pane embeds the Input Selection pane (so the viewed sample is
    visible and steerable here too) next to this page's streamed results
    (`experiment_result_for`, so concurrent tabs never overwrite each other).
    Deep dream renders denormalized input-space images (live while it runs);
    attributions render with the diverging-colormap strips, one per sample.
    """
    _page_scaffold("Experiment")
    _install_panel_resize()
    ui.add_head_html(_STRIP_MARKER_CSS)

    input_name = session.input_names[0] if session.input_names else None
    input_set = set(session.input_names)
    selectable_layers = [n for n in session.layer_names if n not in input_set]
    if not selectable_layers:
        with ui.column().classes("w-full h-screen no-wrap gap-0"):
            with _top_bar_row():
                _back_button()
            _weights_placeholder("No layers available to experiment on.")
        return
    initial_layer = layer if layer in selectable_layers else selectable_layers[0]

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

    def _use_viewed() -> bool:
        widget = widgets.get("use_viewed")
        return bool(getattr(widget, "value", False))

    def collect_params() -> dict[str, object]:
        params: dict[str, object] = {"mean": input_mean, "std": input_std}
        for spec in _EXPERIMENT_PARAMS[state.kind]:
            value: object = getattr(widgets.get(spec.key), "value", None)
            if spec.kind in ("int", "float"):
                if not isinstance(value, (int, float)):
                    value = state.values.get(spec.key, spec.default)
                assert isinstance(value, (int, float))  # numeric specs only
                params[spec.key] = int(value) if spec.kind == "int" else float(value)
            elif spec.kind == "bool":
                params[spec.key] = bool(value)
            else:
                params[spec.key] = str(value if value is not None else spec.default)
        # The viewed-sample index is per-connection input-pane state, not a
        # form widget; the experiment reads it via the `sample` param.
        params["sample"] = input_panel.sample_idx
        return params

    def run() -> None:
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

    def on_input_change() -> None:
        # Sample flip / pin / clear from the input pane: re-run only when the
        # experiment actually consumes the viewed sample.
        if _use_viewed():
            schedule_run()

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
        clip_channel()
        schedule_run()

    with ui.column().classes("w-full h-screen no-wrap gap-0"):
        with _top_bar_row():
            _back_button()
            _add_step_controls(session, step_until_custom)
            _add_settings_button(session, record_view).classes("ml-auto")

        with ui.row().classes("w-full grow min-h-0 no-wrap gap-0"):
            with ui.column().classes(
                "w-80 shrink-0 h-full overflow-auto p-4 gap-2 "
                "border-r-2 border-slate-300 bg-slate-50"
            ).props(_resizable_pane_props("experiment-controls")):
                ui.label("Experiment").classes("font-mono text-base font-bold")
                ui.separator()
                kind_select = ui.select(
                    available_experiment_kinds(),
                    label="Experiment",
                    value=state.kind,
                    on_change=on_kind_change,
                ).props("dense outlined").classes("w-full")
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
                    .props("dense outlined options-dense option-disable=disable")
                    .classes("w-full")
                )
                params_pane = ui.column().classes("w-full gap-2 p-0")
                ui.space()
                description_label = ui.label("").classes(
                    "text-xs text-slate-600 whitespace-normal leading-snug "
                    "border-t border-slate-300 pt-2 mt-1"
                )
            _resize_handle("experiment-controls", "left")
            with ui.row().classes("grow min-w-0 h-full no-wrap gap-0"):
                with ui.column().classes(
                    "w-72 shrink-0 h-full overflow-auto p-3 gap-2 "
                    "border-r border-slate-300 bg-slate-100"
                ):
                    input_panel = InputPanel(
                        session=session,
                        input_name=input_name,
                        input_mean=input_mean,
                        input_std=input_std,
                        on_change=on_input_change,
                    )
                with ui.column().classes(
                    "grow min-w-0 h-full overflow-auto p-4 gap-3 bg-slate-200"
                ):
                    status_label = ui.label(
                        "Adjust parameters — the experiment runs automatically "
                        "(training must be paused)."
                    ).classes("text-sm text-slate-600")
                    error_label = ui.label("").classes("text-sm text-red-600")
                    results_row = ui.row().classes("gap-6 flex-wrap items-start")

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

    def clip_channel() -> None:
        """Pin the channel widget's max to the layer's channels, clipping the
        current value when the new layer has fewer (point 2)."""
        channel_widget = widgets.get("channel")
        if not isinstance(channel_widget, ui.number):
            return
        channels = _layer_channel_count(session.snapshot, state.layer)
        if channels is None:
            return
        channel_widget.max = channels - 1
        current = channel_widget.value
        if isinstance(current, (int, float)) and current > channels - 1:
            clipped = channels - 1
            state.values["channel"] = clipped
            _defer_value_write(lambda: channel_widget.set_value(clipped))

    def _on_param_change(key: str, widget: ui.element) -> None:
        state.values[key] = getattr(widget, "value", None)
        if key == "use_viewed":
            sync_viewed_dependent()
        schedule_run()

    def sync_viewed_dependent() -> None:
        """Gray out the batch / start knobs while a single viewed sample is
        used — neither applies then."""
        disabled = _use_viewed() and not state.frozen
        for key in ("batch", "start"):
            widget = widgets.get(key)
            if isinstance(widget, DisableableElement):
                widget.set_enabled(not disabled)

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
                    if spec.key == "channel":
                        channels = _layer_channel_count(session.snapshot, state.layer)
                        if channels is not None:
                            maximum = channels - 1
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
        sync_viewed_dependent()
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

    def render_batch_images(title: str, tensor: Tensor) -> None:
        """A labelled, wrapping grid of every sample in `tensor`."""
        rendered = [
            render_image(tensor, i, mean=input_mean, std=input_std)
            for i in range(int(tensor.shape[0]))
        ]
        with ui.column().classes("gap-1 min-w-0"):
            ui.label(title).classes("font-mono text-xs text-slate-600")
            if not any(r is not None for r in rendered):
                ui.html(_experiment_img_html(None))
                return
            with ui.row().classes("gap-2 flex-wrap"):
                for image in rendered:
                    ui.html(_experiment_img_html(image))

    def render_attribution_batch(result: ExperimentResult) -> None:
        """One diverging-colormap strip per sample (or the diff under a
        perturbed viewed sample)."""
        attribution = result.attribution
        if attribution is None:
            return
        title = (
            "Attribution diff (perturbed − original)"
            if result.is_diff
            else "Attribution"
        )
        input_hw = tensor_hw(result.reference)
        with ui.column().classes("gap-1 min-w-0"):
            ui.label(title).classes("font-mono text-xs text-slate-600")
            with ui.row().classes("gap-2 flex-wrap"):
                for i in range(int(attribution.shape[0])):
                    with ui.element("div").classes("max-w-full overflow-x-auto"):
                        ui.html(
                            _strip_html(render_strip(attribution, i, input_hw=input_hw))
                        )

    def render_result(result: ExperimentResult) -> None:
        results_row.clear()
        with results_row:
            if result.image is not None:
                render_batch_images("Result", result.image)
            if result.attribution is not None:
                render_attribution_batch(result)
            if result.reference is not None:
                render_batch_images("Input", result.reference)

    def _input_source() -> tuple[object, Tensor | None]:
        """The (identity, tensor) the embedded input image shows: the probe's
        perturbed/base input when one exists, else the live snapshot input."""
        probe = session.probe_result
        if probe is not None:
            tensor = (
                probe.perturbed_input
                if probe.perturbed_input is not None
                else probe.input
            )
            return probe, tensor
        snap = session.snapshot
        if snap is not None and input_name is not None:
            return snap, snap.activations.get(input_name)
        return None, None

    def refresh_input_image() -> None:
        source, tensor = _input_source()
        sig: tuple[object, ...] = (id(source), input_panel.sample_idx)
        if sig == state.last_input_render:
            return
        state.last_input_render = sig
        image = render_image(
            tensor, input_panel.sample_idx, mean=input_mean, std=input_std
        )
        input_panel.set_image(_b64_img_src(image) if image is not None else "")

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
        input_panel.refresh_status()
        refresh_input_image()
        input_panel.sync_spinner_max(session.input_batch_size)
        # While this experiment records, its request must stay as-is: a re-run
        # would replace the recorded seq and parameter edits would lie.
        frozen = session.recording.is_recording(record_key())
        if frozen != state.frozen:
            state.frozen = frozen
            input_panel.set_frozen(frozen)
            _set_controls_enabled([kind_select, *_param_controls()], not frozen)
            sync_viewed_dependent()
        # The viewed input changing (pin / perturb / sample) re-runs only when
        # the experiment uses it — perturbation clicks don't fire on_change.
        input_sig = (
            session.is_pinned,
            len(session.perturbations),
            input_panel.sample_idx,
        )
        if _use_viewed() and input_sig != state.last_input_sig:
            schedule_run()
        state.last_input_sig = input_sig

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
                    "queued — waiting for the training thread to pause or for "
                    "earlier experiments (use Stop / Step Batch above)"
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
