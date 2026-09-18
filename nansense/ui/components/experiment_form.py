"""Experiment parameter widgets and their widget-value validation."""

from __future__ import annotations

from collections.abc import Callable

from nicegui import ui
from nicegui.elements.mixins.disableable_element import DisableableElement
from nicegui.elements.mixins.value_element import ValueElement

from nansense.contracts.experiments import DEFAULT_BATCH
from nansense.experiments import EXPERIMENT_PARAMS, ExperimentParam
from nansense.session import BatchSnapshot, Session
from nansense.ui.common import _defer_value_write, _set_controls_enabled
from nansense.ui.controllers.experiment import ExperimentState


def _coerce_number(spec: ExperimentParam, *candidates: object) -> int | float:
    """The first numeric `candidate` cast to the spec's type. A cleared number
    field reads back from NiceGUI as None, so callers pass the widget value, the
    persisted value and finally the always-numeric default — the cast then can
    never see a None."""
    for candidate in candidates:
        if isinstance(candidate, (int, float)):
            return int(candidate) if spec.kind == "int" else float(candidate)
    raise AssertionError(f"no numeric value for {spec.key!r}")  # default is numeric


def _layer_channel_count(snap: BatchSnapshot | None, layer: str) -> int | None:
    """Channel count of `layer`'s last captured activation (None if unknown)."""
    act = snap.activations.get(layer) if snap is not None else None
    if act is None or act.ndim < 2:
        return None
    return int(act.shape[1])


class ExperimentForm:
    def __init__(
        self,
        session: Session,
        state: ExperimentState,
        on_change: Callable[[], None],
        mean: tuple[float, ...] | None,
        std: tuple[float, ...] | None,
    ) -> None:
        self.session = session
        self.state = state
        self.on_change = on_change
        self.mean = mean
        self.std = std
        self.widgets: dict[str, ui.element] = {}
        self.params_pane = ui.column().classes("w-full gap-2 p-0")
        self.param_error_label = ui.label("").classes(
            "text-xs text-red-600 whitespace-normal leading-snug"
        )

    def collect_params(self) -> dict[str, object]:
        params: dict[str, object] = {"mean": self.mean, "std": self.std}
        for spec in EXPERIMENT_PARAMS[self.state.kind]:
            value: object = getattr(self.widgets.get(spec.key), "value", None)
            if spec.kind in ("int", "float"):
                # `run` blocks the call while a numeric field is empty; the
                # persisted value / numeric-default fallbacks guard the rest.
                params[spec.key] = _coerce_number(
                    spec, value, self.state.values.get(spec.key), spec.default
                )
            elif spec.kind == "bool":
                params[spec.key] = bool(value)
            else:
                params[spec.key] = str(value if value is not None else spec.default)
        return params

    def _clip_number(self, key: str, maximum: int) -> None:
        """Pin a number widget's max, clipping its value when the layer shrank."""
        widget = self.widgets.get(key)
        if not isinstance(widget, ui.number):
            return
        widget.max = maximum
        current = widget.value
        if isinstance(current, (int, float)) and current > maximum:
            self.state.values[key] = maximum
            _defer_value_write(lambda: widget.set_value(maximum))

    def clip_channel(self) -> None:
        """Pin the targeting widgets to the layer's channel count, clipping a
        value the new layer can no longer reach: deep dream's
        Channels is a count of the first N (max = channels), Captum's Channel
        is a single index (max = channels − 1)."""
        channels = _layer_channel_count(self.session.snapshot, self.state.layer)
        if channels is None:
            return
        self._clip_number("channels", channels)
        self._clip_number("channel", channels - 1)

    def _sync_sample_visibility(self) -> None:
        """Show deep dream's Sample knob only when starting from the current
        batch — noise has no input to pick."""
        sample_widget = self.widgets.get("sample")
        start_widget = self.widgets.get("start")
        if sample_widget is None or start_widget is None:
            return
        sample_widget.set_visibility(getattr(start_widget, "value", None) == "sample")

    def _invalid_number_fields(self) -> list[str]:
        """Labels of numeric params whose widget holds no usable number — an
        empty or non-numeric field reads back from NiceGUI as None."""
        invalid: list[str] = []
        for spec in EXPERIMENT_PARAMS[self.state.kind]:
            if spec.kind not in ("int", "float"):
                continue
            value = getattr(self.widgets.get(spec.key), "value", None)
            if not isinstance(value, (int, float)):
                invalid.append(spec.label)
        return invalid

    def validate(self) -> list[str]:
        """Sync the red hint with the current fields; return the invalid ones."""
        invalid = self._invalid_number_fields()
        self.param_error_label.text = (
            "Enter a number for: " + ", ".join(invalid) if invalid else ""
        )
        return invalid

    def _on_param_change(self, key: str, widget: ui.element) -> None:
        value = getattr(widget, "value", None)
        # A cleared / non-numeric number field reads back as None; keep the last
        # good value rather than persisting it — `run` reports it as a red hint.
        if not (isinstance(widget, ui.number) and not isinstance(value, (int, float))):
            self.state.values[key] = value
            if key == "start":
                self._sync_sample_visibility()
        self.validate()
        self.on_change()

    def rebuild_params(self) -> None:
        self.widgets.clear()
        self.params_pane.clear()
        with self.params_pane:
            for spec in EXPERIMENT_PARAMS[self.state.kind]:
                initial = self.state.values.get(spec.key, spec.default)
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
                        live = self.session.input_batch_size
                        default = min(DEFAULT_BATCH, live) if live else DEFAULT_BATCH
                    maximum: float | None = None
                    if spec.key in ("channel", "channels"):
                        channels = _layer_channel_count(
                            self.session.snapshot, self.state.layer
                        )
                        if channels is not None:
                            # Channel is a single index; Channels is a count.
                            maximum = (
                                channels - 1 if spec.key == "channel" else channels
                            )
                    elif spec.key == "sample":
                        live = self.session.input_batch_size
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
                    lambda _e, k=spec.key, w=widget: self._on_param_change(k, w)
                )
                self.widgets[spec.key] = widget
        self._sync_sample_visibility()
        if self.state.frozen:
            _set_controls_enabled(self.controls(), False)

    def controls(self) -> list[DisableableElement]:
        controls = [
            w for w in self.widgets.values() if isinstance(w, DisableableElement)
        ]
        return controls
