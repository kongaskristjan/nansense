"""The `/experiment` page: deep dream and Captum attributions."""

from __future__ import annotations

from dataclasses import dataclass
from uuid import uuid4

from nicegui import ui
from nicegui.elements.mixins.disableable_element import DisableableElement
from torch import Tensor

from nansense.experiments import (
    _DEFAULT_DREAM_BATCH,
    EXPERIMENT_KINDS,
    ExperimentResult,
    available_experiment_kinds,
)
from nansense.recording import RecordedView
from nansense.schedule import format_position
from nansense.session import BatchSnapshot, Session
from nansense.ui.common import (
    _b64_img_src,
    _set_controls_enabled,
    _strip_html,
    _tensor_hw,
    _weights_placeholder,
)
from nansense.ui.render import INPUT_IMAGE_SIZE, render_image, render_strip
from nansense.ui.static import _STRIP_MARKER_CSS
from nansense.ui.top_bar import (
    _add_settings_button,
    _add_step_controls,
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


_CHANNEL_PARAM = _ExperimentParam(
    "channel",
    "Channel (-1 = whole layer)",
    "int",
    0,
    minimum=-1,
    tooltip="Which channel / feature of this layer to target",
)
_SAMPLE_PARAM = _ExperimentParam(
    "sample",
    "Sample",
    "int",
    0,
    minimum=0,
    tooltip="Batch sample index of the input to work on",
)
_TARGET_PARAM = _ExperimentParam(
    "target",
    "Target class (-1 = argmax)",
    "int",
    -1,
    minimum=-1,
    tooltip="Class index the attribution explains; -1 uses the model's prediction",
)

_EXPERIMENT_PARAMS: dict[str, list[_ExperimentParam]] = {
    "deep_dream": [
        _CHANNEL_PARAM,
        _ExperimentParam("steps", "Steps", "int", 100, minimum=1),
        _ExperimentParam(
            "lr", "Learning rate", "float", 0.05, minimum=0, step=0.01
        ),
        _ExperimentParam(
            "diffusion",
            "Diffusion",
            "float",
            0.05,
            minimum=0,
            step=0.01,
            tooltip="Per-step blend with a 3×3 blur; damps high-frequency noise",
        ),
        _ExperimentParam(
            "jitter",
            "Jitter (px)",
            "int",
            2,
            minimum=0,
            tooltip=(
                "Random shift each step, undone after the update; "
                "reduces pixel-grid artifacts"
            ),
        ),
        _ExperimentParam(
            "zoom",
            "Zoom multiplier per step",
            "float",
            1.0,
            minimum=1,
            step=0.01,
            tooltip=(
                "Per-step center zoom-in factor (1 = no zoom; on small "
                "inputs it only takes effect above ~1 + 1/size)"
            ),
        ),
        _ExperimentParam(
            "batch",
            "Inputs",
            "int",
            1,
            minimum=1,
            tooltip=(
                "How many inputs to dream on; defaults to the size of the "
                f"currently processed batch, capped at {_DEFAULT_DREAM_BATCH}"
            ),
        ),
        _ExperimentParam(
            "start",
            "Start from",
            "select",
            "noise",
            options={"noise": "Noise", "sample": "Current batch"},
            tooltip=(
                "Noise draws fresh inputs shaped and scaled like the "
                "network's real input — different on every Run; Current "
                "batch starts from the real input batch itself"
            ),
        ),
        _ExperimentParam(
            "clamp",
            "Clamp to displayable range",
            "bool",
            True,
            tooltip=(
                "Keep pixels inside the [0, 1] display range mapped through "
                "the input mean/std"
            ),
        ),
    ],
    "gradcam": [_TARGET_PARAM, _SAMPLE_PARAM],
    "neuron_gradient": [_CHANNEL_PARAM, _SAMPLE_PARAM],
    "neuron_ig": [
        _CHANNEL_PARAM,
        _ExperimentParam("steps", "Integration steps", "int", 32, minimum=2),
        _SAMPLE_PARAM,
    ],
    "occlusion": [
        _TARGET_PARAM,
        _ExperimentParam(
            "window",
            "Window (px)",
            "int",
            4,
            minimum=1,
            tooltip="Side length of the occluding patch",
        ),
        _ExperimentParam("stride", "Stride (px)", "int", 2, minimum=1),
        _SAMPLE_PARAM,
    ],
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


def _build_experiment_page(
    session: Session,
    layer: str,
    *,
    input_mean: tuple[float, ...] | None,
    input_std: tuple[float, ...] | None,
) -> None:
    """Per-layer experiments: deep dream and selected Captum attributions.

    The top bar carries the shared stepping controls — experiments execute
    on the paused training thread, so the user can pause right from this
    page. The left pane is headed by the page title (with the layer name)
    and holds the experiment-kind dropdown (deep dream by
    default) with Run / Cancel right below it, followed by the selected
    kind's parameter form (rebuilt on every dropdown change); the right
    pane streams status and results for *this page's own request*
    (`experiment_result_for`), so concurrent tabs running their own
    experiments never overwrite each other — Run replaces only this page's
    previous request, Cancel aborts only this page's. Deep dream results
    render as a denormalized input-space image (updating live while the
    run progresses); attributions render with the shared
    diverging-colormap strip machinery, next to the input sample they
    explain.
    """
    title = f"Experiment · {layer}" if layer else "Experiment"
    ui.page_title(f"Nansense — {title}")
    ui.query(".nicegui-content").classes("p-0 h-screen overflow-hidden")
    ui.query("body").classes("overflow-hidden")
    ui.query("html").classes("overflow-hidden")
    ui.add_head_html(_STRIP_MARKER_CSS)

    step_until_custom = _build_step_until_custom_dialog(session)
    widgets: dict[str, ui.element] = {}
    kind_holder = {"kind": "deep_dream"}
    my_seq: list[int | None] = [None]  # this page's own request
    last_result: list[ExperimentResult | None] = [None]
    # This page's auto-experiment registration: Run registers the request
    # so it re-runs on every visualization update (same seq → same seeded
    # noise); the page's tick heartbeats it and it expires when the page
    # closes, unless a recording pins it.
    page_key = f"experiment-page-{uuid4().hex}"
    record_key = f"experiment:{layer}"
    frozen_state: dict[str, bool | None] = {"on": None}

    def collect_params() -> dict[str, object]:
        params: dict[str, object] = {"mean": input_mean, "std": input_std}
        for spec in _EXPERIMENT_PARAMS[kind_holder["kind"]]:
            value: object = getattr(widgets.get(spec.key), "value", None)
            if spec.kind in ("int", "float"):
                if not isinstance(value, (int, float)):
                    value = spec.default
                assert isinstance(value, (int, float))  # numeric specs only
                params[spec.key] = int(value) if spec.kind == "int" else float(value)
            elif spec.kind == "bool":
                params[spec.key] = bool(value)
            else:
                params[spec.key] = str(value if value is not None else spec.default)
        return params

    def run() -> None:
        if my_seq[0] is not None:  # a re-Run replaces this page's request
            session.cancel_experiment(my_seq[0])
        my_seq[0] = session.register_auto_experiment(
            page_key, kind=kind_holder["kind"], layer=layer, params=collect_params()
        )
        last_result[0] = None
        error_label.text = ""

    def cancel() -> None:
        if my_seq[0] is None:
            return
        session.cancel_experiment(my_seq[0])
        # Stop the auto reruns too — unless a recording pinned the request
        # (Cancel is disabled while recorded, but another tab may differ).
        if not session.recording.is_recording(record_key):
            session.unregister_auto_experiment(page_key)

    def record_view() -> RecordedView | None:
        if my_seq[0] is None:
            return None  # nothing to record until an experiment has run
        kind = kind_holder["kind"]
        return RecordedView(
            key=record_key,
            page="experiment",
            label=f"Experiment · {EXPERIMENT_KINDS.get(kind, kind)} · {layer}",
            params={
                "layer": layer,
                "seq": my_seq[0],
                "auto_key": page_key,
                "input_mean": input_mean,
                "input_std": input_std,
            },
        )

    def on_kind_change(e: object) -> None:
        value = getattr(e, "value", None)
        if value is not None:
            kind_holder["kind"] = str(value)
            rebuild_params()

    with ui.column().classes("w-full h-screen no-wrap gap-0"):
        with _top_bar_row():
            ui.button(
                icon="arrow_back",
                on_click=lambda: ui.navigate.to("/"),
                color="slate-500",
            ).props("dense size=md").tooltip("Back to the main page")
            position_label = _add_step_controls(session, step_until_custom)
            _add_settings_button(session, record_view).classes("ml-auto")

        with ui.row().classes("w-full grow min-h-0 no-wrap gap-0"):
            with ui.column().classes(
                "w-80 shrink-0 h-full overflow-auto p-4 gap-2 "
                "border-r-2 border-slate-300 bg-slate-50"
            ):
                with ui.row().classes("items-baseline gap-2 no-wrap"):
                    ui.label("Experiment").classes("font-mono text-base font-bold")
                    if layer:
                        ui.label(layer).classes(
                            "font-mono text-sm text-slate-500 truncate"
                        )
                ui.separator()
                # The experiment kind and its Run / Cancel sit above the
                # parameter form; only the form below is rebuilt when the
                # kind changes.
                kind_select = ui.select(
                    available_experiment_kinds(),
                    label="Experiment",
                    value=kind_holder["kind"],
                    on_change=on_kind_change,
                ).props("dense outlined").classes("w-full")
                with ui.row().classes("w-full no-wrap gap-2"):
                    run_button = ui.button(
                        "Run", icon="science", on_click=run, color="yellow-8"
                    ).props("dense size=md").classes("grow").tooltip(
                        "Run the experiment (training must be paused), then "
                        "re-run it automatically on every visualization update"
                    )
                    cancel_button = ui.button(
                        "Cancel", on_click=cancel, color="slate-500"
                    ).props("dense size=md").classes("grow").tooltip(
                        "Abort this page's experiment and its automatic reruns"
                    )
                ui.separator()
                params_pane = ui.column().classes("w-full gap-2 p-0")
            with ui.column().classes(
                "grow min-w-0 h-full overflow-auto p-4 gap-3 bg-slate-200"
            ):
                if layer not in session.layer_names:
                    _weights_placeholder(f"Unknown layer {layer!r}.")
                    return
                status_label = ui.label(
                    "Pick an experiment and press Run (training must be paused)."
                ).classes("text-sm text-slate-600")
                error_label = ui.label("").classes("text-sm text-red-600")
                results_row = ui.row().classes("gap-6 flex-wrap items-start")

    def rebuild_params() -> None:
        widgets.clear()
        params_pane.clear()
        with params_pane:
            ui.label("Parameters").classes("font-mono text-sm")
            for spec in _EXPERIMENT_PARAMS[kind_holder["kind"]]:
                if spec.kind == "bool":
                    widget: ui.element = ui.switch(
                        spec.label, value=bool(spec.default)
                    ).props("dense")
                elif spec.kind == "select":
                    widget = ui.select(
                        spec.options or {}, label=spec.label, value=spec.default
                    ).props("dense outlined").classes("w-full")
                else:
                    default = spec.default
                    if spec.key == "batch":  # tracks the live batch size
                        live = session.input_batch_size
                        if live is not None:
                            default = min(_DEFAULT_DREAM_BATCH, live)
                    maximum: float | None = None
                    if spec.key == "channel":  # bound to the layer's channels
                        channels = _layer_channel_count(session.snapshot, layer)
                        if channels is not None:
                            maximum = channels - 1
                    default_number = (
                        default if isinstance(default, (int, float)) else 0
                    )
                    widget = ui.number(
                        label=spec.label,
                        value=default_number,
                        min=spec.minimum,
                        max=maximum,
                        step=1 if spec.kind == "int" else spec.step,
                        format="%d" if spec.kind == "int" else None,
                    ).props("dense outlined").classes("w-full")
                if spec.tooltip:
                    widget.tooltip(spec.tooltip)
                widgets[spec.key] = widget
        if frozen_state["on"]:
            _set_controls_enabled(_param_controls(), False)

    def _param_controls() -> list[DisableableElement]:
        return [w for w in widgets.values() if isinstance(w, DisableableElement)]

    def render_batch_images(title: str, tensor: Tensor) -> None:
        """A labelled, wrapping grid of every sample in `tensor`.

        Non-image tensors (deep dream runs on the network's real input,
        whatever its shape) fall back to a single "not renderable" note.
        """
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

    def render_result(result: ExperimentResult) -> None:
        results_row.clear()
        with results_row:
            if result.image is not None:
                render_batch_images("Result", result.image)
            if result.attribution is not None:
                with ui.column().classes("gap-1 min-w-0"):
                    ui.label("Attribution").classes(
                        "font-mono text-xs text-slate-600"
                    )
                    with ui.element("div").classes("max-w-full overflow-x-auto"):
                        # `reference` is the input batch the attribution
                        # explains; its spatial size lets token-shaped
                        # attributions unflatten onto the patch grid.
                        ui.html(
                            _strip_html(
                                render_strip(
                                    result.attribution,
                                    0,
                                    input_hw=_tensor_hw(result.reference),
                                )
                            )
                        )
            if result.reference is not None:
                render_batch_images("Input", result.reference)

    def tick() -> None:
        live = session.live_position
        if live is not None:
            position_label.text = format_position(live)
        # Keep this page's auto experiment alive while the page is open.
        session.touch_auto_experiment(page_key)
        # While this experiment records, its request must stay as-is: Run
        # would replace the recorded seq and parameter edits would lie.
        frozen = session.recording.is_recording(record_key)
        if frozen != frozen_state["on"]:
            frozen_state["on"] = frozen
            _set_controls_enabled(
                [kind_select, run_button, cancel_button, *_param_controls()],
                not frozen,
            )
        if my_seq[0] is None:
            return  # nothing requested from this page yet
        result = session.experiment_result_for(my_seq[0])
        if result is None:
            if last_result[0] is None:
                status_label.text = (
                    "queued — waiting for the training thread to pause "
                    "or for earlier experiments (use Stop / Step Batch above)"
                )
            return
        status_label.text = _experiment_status(result)
        error_label.text = result.error or ""
        if result is not last_result[0]:
            last_result[0] = result
            if result.error is None:
                render_result(result)

    rebuild_params()
    ui.timer(0.2, tick)
