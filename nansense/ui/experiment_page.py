"""The `/experiment` page: deep dream and Captum attributions."""

from __future__ import annotations

from urllib.parse import quote

from nicegui import ui

from nansense.contracts.recording import ExperimentView
from nansense.experiments import (
    EXPERIMENT_DESCRIPTIONS,
    EXPERIMENT_KINDS,
    ExperimentQueueState,
    ExperimentResult,
    available_experiment_kinds,
    layer_available,
)
from nansense.recording import RecordedView
from nansense.session import Session
from nansense.ui.common import (
    _defer_value_write,
    _install_panel_resize,
    _page_scaffold,
    _resizable_pane_props,
    _resize_handle,
    _set_controls_enabled,
    _StatusChip,
    _StatusPill,
    _weights_placeholder,
)
from nansense.ui.components.experiment_form import ExperimentForm
from nansense.ui.components.experiment_results import ExperimentResults
from nansense.ui.controllers.experiment import ExperimentController
from nansense.ui.share import _add_share_button
from nansense.ui.static import _STRIP_MARKER_CSS
from nansense.ui.top_bar import (
    _add_error_banner,
    _add_repo_logo,
    _add_settings_button,
    _add_step_controls,
    _add_tour_button,
    _back_button,
    _back_href,
    _build_step_until_custom_dialog,
    _top_bar_row,
)
from nansense.ui.tour import add_tour, experiment_tour_steps


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


def _minmax_stats_href(layer: str) -> str:
    """Deep-link to `layer`'s MIN/MAX stats view — the real-input extremes
    that complement its synthesized dreams. `?view` opens straight on the
    grids rather than the histogram default."""
    return f"/stats?layer={quote(layer)}&view=minmax"


def _run_tooltip(locked: bool) -> str:
    """Run-button tooltip. Locked sessions expose no training controls, so
    the "training must be paused" advice only appears when the user can
    actually pause. Why the button is greyed out (auto-run, or a run in
    flight) is left to the results pane's own hint (`_status_text`)."""
    pause = "" if locked else " (training must be paused)"
    return f"Run the experiment{pause}"


def _status_text(auto_run: bool, locked: bool) -> str:
    """Results-pane hint. Same locked rule as `_run_tooltip`; the auto-run
    split tells the user whether pressing Run is ever needed."""
    pause = "" if locked else " (training must be paused)"
    if auto_run:
        return f"Adjust parameters — the experiment runs automatically{pause}."
    return (
        "The first experiment starts on its own — after changing "
        f"parameters, press Run{pause}."
    )


def _idle_chip(auto_run: bool, locked: bool) -> _StatusChip:
    """The pill before this page has an experiment of its own."""
    return _StatusChip("idle", "science", _status_text(auto_run, locked))


def _pending_chip(
    kind: str,
    queue: ExperimentQueueState,
    *,
    training_running: bool,
    locked: bool,
) -> _StatusChip:
    """The pill for a request that hasn't published a result yet.

    Captum methods publish once, at the end, so this covers whole runs —
    it has to say which kind of wait the user is in. Every wait here
    resolves on its own (an experiment page registers an auto experiment,
    which the training thread runs at its next visualization update as
    well as at a pause), so all of them spin (`icon=None`); only the
    request that will never run drops the spinner. Advancing training is
    the slow wait, so it names the controls that cut it short — unlocked
    only, a locked session has none (`_run_tooltip`).
    """
    title = EXPERIMENT_KINDS.get(kind, kind)
    if queue.stage == "running":
        return _StatusChip("running", None, f"{title} — running…")
    if queue.stage == "absent":
        # Cancelled, superseded, or dropped before the thread picked it up.
        return _StatusChip("idle", "block", f"{title} — stopped before it ran")
    if training_running:
        controls = "" if locked else "; Stop or Step Batch runs it now"
        return _StatusChip(
            "waiting",
            None,
            f"{title} — runs at the next visualization update{controls}",
        )
    if queue.ahead:
        plural = "" if queue.ahead == 1 else "s"
        return _StatusChip(
            "waiting",
            None,
            f"{title} — queued behind {queue.ahead} experiment{plural}",
        )
    return _StatusChip("running", None, f"{title} — starting…")


def _result_chip(result: ExperimentResult) -> _StatusChip:
    """The pill for a published result: still streaming, or its outcome."""
    text = _experiment_status(result)
    if result.error is not None:
        return _StatusChip("failed", "error", text)
    if not result.done:
        return _StatusChip("running", None, text)
    if result.step < result.total_steps:
        return _StatusChip("stopped", "stop_circle", text)
    return _StatusChip("done", "check_circle", text)


def _build_experiment_page(
    session: Session,
    layer: str,
    *,
    input_mean: tuple[float, ...] | None,
    input_std: tuple[float, ...] | None,
) -> None:
    _ExperimentPage(session, layer, input_mean=input_mean, input_std=input_std)


class _ExperimentPage:
    """NiceGUI adapter: builds components and synchronizes them with its controller."""

    def __init__(
        self,
        session: Session,
        layer: str,
        *,
        input_mean: tuple[float, ...] | None,
        input_std: tuple[float, ...] | None,
    ) -> None:
        self.input_mean = input_mean
        self.input_std = input_std
        self.session = session
        _page_scaffold("Experiment")
        _install_panel_resize()
        ui.add_head_html(_STRIP_MARKER_CSS)
        input_set = set(self.session.input_names)
        self.selectable_layers = [
            n for n in self.session.layer_names if n not in input_set
        ]
        if not self.selectable_layers:
            with ui.column().classes("w-full h-screen no-wrap gap-0"):
                with _top_bar_row():
                    _back_button()
                    _add_repo_logo().classes("ml-auto")
                _weights_placeholder("No layers available to experiment on.")
            return
        initial_layer = (
            layer if layer in self.selectable_layers else self.selectable_layers[0]
        )
        add_tour(
            "experiment",
            experiment_tour_steps(locked=self.session.locked),
            locked=self.session.locked,
        )
        self.step_until_custom = _build_step_until_custom_dialog(self.session)
        self.controller = ExperimentController(
            self.session, self.selectable_layers, initial_layer
        )
        self.state = self.controller.state
        self._build_layout()
        # NiceGUI regenerates options on update; preserve our disabled-layer flags.
        setattr(self.layer_select, "_update_options", self._patched_update_options)
        self.refresh_layer_options()
        self.form.rebuild_params()
        self.update_description()
        ui.timer(0.2, self.tick)

    def record_key(self) -> str:
        return self.controller.record_key

    def run(self) -> None:
        if self.form.validate():
            return
        try:
            self.controller.run(self.form.collect_params())
        except ValueError as error:
            self.error_label.text = str(error)
            return
        self.error_label.text = ""

    def cancel(self) -> None:
        self.controller.cancel()

    def record_view(self) -> RecordedView | None:
        if self.state.my_seq is None:
            return None
        kind = self.state.kind
        return RecordedView(
            key=self.record_key(),
            label=f"Experiment · {EXPERIMENT_KINDS.get(kind, kind)} · {self.state.layer}",
            config=ExperimentView(
                layer=self.state.layer,
                seq=self.state.my_seq,
                auto_key=self.controller.key,
                input_mean=self.input_mean,
                input_std=self.input_std,
                overlay=self.state.overlay,
            ),
        )

    def schedule_run(self) -> None:
        self.controller.schedule()

    def on_kind_change(self, e: object) -> None:
        value = getattr(e, "value", None)
        if value is None:
            return
        switched = self.controller.select_kind(str(value))
        self.refresh_layer_options()
        if switched is not None:
            old_layer, available = switched
            _defer_value_write(lambda: self.layer_select.set_value(available))
            ui.notify(
                f"{EXPERIMENT_KINDS[self.state.kind]} can't run on {old_layer} — switched to {available}",
                type="info",
            )
        self.form.rebuild_params()
        self.update_description()
        self.overlay_switch.set_visibility(self.state.kind != "deep_dream")
        self.compare_button.set_visibility(self.state.kind == "deep_dream")
        self.sync_compare_href()
        self.sync_back_href()
        self.schedule_run()

    def on_layer_change(self, e: object) -> None:
        value = getattr(e, "value", None)
        if value is None:
            return
        if not self.controller.select_layer(str(value)):
            _defer_value_write(lambda: self.layer_select.set_value(self.state.layer))
            ui.notify(
                f"{EXPERIMENT_KINDS[self.state.kind]} can't run on {value}",
                type="warning",
            )
            return
        self.sync_compare_href()
        self.sync_back_href()
        self.form.clip_channel()
        self.schedule_run()

    def on_overlay_change(self, e: object) -> None:
        self.state.overlay = bool(getattr(e, "value", False))
        if self.state.last_result is not None and self.state.last_result.error is None:
            self.results.render(self.state.last_result, overlay=self.state.overlay)

    def sync_compare_href(self) -> None:
        self.compare_button.props(f'href="{_minmax_stats_href(self.state.layer)}"')

    def sync_back_href(self) -> None:
        if self.session.locked:
            self.back_button.props(f'href="{_back_href(self.state.layer)}"')

    def _layer_options_with_disable(self) -> list[dict[str, object]]:
        return [
            {
                "value": index,
                "label": name,
                "disable": not layer_available(self.session, name, self.state.kind),
            }
            for index, name in enumerate(self.selectable_layers)
        ]

    def _patched_update_options(self) -> None:
        before = self.layer_select.value
        self.layer_select._props["options"] = self._layer_options_with_disable()
        self.layer_select._props[self.layer_select.VALUE_PROP] = (
            self.layer_select._value_to_model_value(before)
        )
        if not isinstance(before, list):
            self.layer_select.value = (
                before if before in self.layer_select._values else None
            )

    def refresh_layer_options(self) -> None:
        """Re-gray the layer options for the current kind."""
        self.layer_select.update()

    def update_description(self) -> None:
        short, long = EXPERIMENT_DESCRIPTIONS.get(self.state.kind, ("", ""))
        self.description_label.text = long
        self.kind_tooltip.set_text(short)

    def update_controls(self, *, running: bool) -> None:
        run_ok = (
            not self.state.frozen
            and (not self.session.auto_run_experiments)
            and (not running)
        )
        cancel_ok = not self.state.frozen and running
        if run_ok != self.state.run_enabled:
            self.state.run_enabled = run_ok
            self.run_button.set_enabled(run_ok)
        if cancel_ok != self.state.cancel_enabled:
            self.state.cancel_enabled = cancel_ok
            self.cancel_button.set_enabled(cancel_ok)

    def tick(self) -> None:
        frozen = self.controller.heartbeat()
        if frozen != self.state.frozen:
            self.state.frozen = frozen
            _set_controls_enabled(
                [
                    self.kind_select,
                    self.layer_select,
                    self.overlay_switch,
                    *self.form.controls(),
                ],
                not frozen,
            )
        result = (
            self.session.experiment_result_for(self.state.my_seq)
            if self.state.my_seq
            else None
        )
        running = self.state.my_seq is not None and (
            not (result is not None and result.done)
        )
        self.update_controls(running=running)
        if self.controller.consume_auto_run(frozen):
            self.run()
        if self.state.my_seq is None:
            return
        if result is None:
            if self.state.last_result is None:
                self.status_pill.show(
                    _pending_chip(
                        self.state.kind,
                        self.session.experiment_queue_state(self.state.my_seq),
                        training_running=self.session.is_running,
                        locked=self.session.locked,
                    )
                )
            return
        self.status_pill.show(_result_chip(result))
        self.error_label.text = result.error or ""
        if result is not self.state.last_result:
            self.state.last_result = result
            if result.error is None:
                self.results.render(result, overlay=self.state.overlay)

    def _build_header(self) -> None:
        with _top_bar_row():
            self.back_button = _back_button(
                self.state.layer if self.session.locked else None
            )
            _add_step_controls(self.session, self.step_until_custom)
            _add_settings_button(self.session, self.record_view).classes("ml-auto")
            _add_tour_button()
            _add_share_button(self.session)
            _add_repo_logo()

    def _build_content(self) -> None:
        with ui.row().classes("w-full grow min-h-0 no-wrap gap-0"):
            self._build_controls()
            _resize_handle("experiment-controls", "left")
            with ui.column().classes(
                "grow min-w-0 h-full overflow-auto p-4 gap-3 bg-slate-200"
            ):
                self.status_pill = _StatusPill(
                    _idle_chip(self.session.auto_run_experiments, self.session.locked)
                )
                self.error_label = ui.label("").classes("text-sm text-red-600")
                self.results = ExperimentResults(self.input_mean, self.input_std)

    def _build_controls(self) -> None:
        with (
            ui.column()
            .classes(
                "w-80 shrink-0 h-full overflow-auto p-4 gap-2 border-r-2 border-slate-300 bg-slate-50"
            )
            .props(_resizable_pane_props("experiment-controls"))
        ):
            ui.label("Experiment").classes("font-mono text-base font-bold")
            ui.separator()
            self.kind_select = (
                ui.select(
                    available_experiment_kinds(),
                    label="Experiment",
                    value=self.state.kind,
                    on_change=self.on_kind_change,
                )
                .props('dense outlined data-tour="kind"')
                .classes("w-full")
            )
            with self.kind_select:
                self.kind_tooltip = ui.tooltip("")
            with ui.row().classes("w-full no-wrap gap-2").props('data-tour="run"'):
                self.run_button = (
                    ui.button(
                        "Run", icon="science", on_click=self.run, color="yellow-8"
                    )
                    .props("dense size=md")
                    .classes("grow")
                    .tooltip(_run_tooltip(self.session.locked))
                )
                self.cancel_button = (
                    ui.button("Cancel", on_click=self.cancel, color="slate-500")
                    .props("dense size=md")
                    .classes("grow")
                    .tooltip("Abort experiment")
                )
            ui.separator()
            ui.label("Parameters").classes("font-mono text-sm")
            self.layer_select = (
                ui.select(
                    self.selectable_layers,
                    value=self.state.layer,
                    label="Layer",
                    on_change=self.on_layer_change,
                )
                .props(
                    'dense outlined options-dense option-disable=disable data-tour="layer"'
                )
                .classes("w-full")
            )
            self.form = ExperimentForm(
                self.session,
                self.state,
                self.schedule_run,
                self.input_mean,
                self.input_std,
            )
            self.overlay_switch = (
                ui.switch(
                    "Overlay on input",
                    value=self.state.overlay,
                    on_change=self.on_overlay_change,
                )
                .props("dense")
                .tooltip("Blend each map over its input instead of side by side")
            )
            self.overlay_switch.set_visibility(self.state.kind != "deep_dream")
            ui.space()
            self.compare_button = (
                ui.button("Compare with MIN/MAX", icon="bar_chart", color="teal")
                .props(
                    f'dense no-caps size=sm href="{_minmax_stats_href(self.state.layer)}"'
                )
                .classes("w-full")
                .tooltip("Open this layer's MIN/MAX stats")
            )
            self.compare_button.set_visibility(self.state.kind == "deep_dream")
            self.description_label = ui.label("").classes(
                "text-xs text-slate-600 whitespace-normal leading-snug border-t border-slate-300 pt-2 mt-1"
            )

    def _build_layout(self) -> None:
        with ui.column().classes("w-full h-screen no-wrap gap-0"):
            self._build_header()
            _add_error_banner(self.session)
            self._build_content()
