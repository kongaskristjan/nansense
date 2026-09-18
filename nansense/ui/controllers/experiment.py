"""Experiment registration and per-tab lifecycle, independent of widgets."""

from __future__ import annotations

from dataclasses import dataclass, field
from uuid import uuid4

from nansense.experiments import ExperimentResult, default_param_values, layer_available
from nansense.session import Session


@dataclass
class ExperimentState:
    """Mutable page state shared by the form, Run/Cancel, and tick closures."""

    kind: str = "deep_dream"
    layer: str = ""
    # Parameter values persisted across kind switches: keyed by
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


class ExperimentController:
    def __init__(self, session: Session, layers: list[str], layer: str) -> None:
        self.session = session
        self.layers = tuple(layers)
        self.state = ExperimentState(layer=layer)
        self.state.values.update(default_param_values(session.experiment_defaults))
        self.key = f"experiment-page-{uuid4().hex}"

    @property
    def record_key(self) -> str:
        return f"experiment:{self.state.layer}"

    def schedule(self) -> None:
        self.state.dirty = True

    def select_kind(self, kind: str) -> tuple[str, str] | None:
        self.state.kind = kind
        old = self.state.layer
        if not layer_available(self.session, old, kind):
            self.state.layer = next(
                (
                    name
                    for name in self.layers
                    if layer_available(self.session, name, kind)
                ),
                old,
            )
        self.schedule()
        return (old, self.state.layer) if old != self.state.layer else None

    def select_layer(self, layer: str) -> bool:
        if layer not in self.layers or not layer_available(
            self.session, layer, self.state.kind
        ):
            return False
        self.state.layer = layer
        self.schedule()
        return True

    def run(self, params: dict[str, object]) -> None:
        previous = self.state.my_seq
        seq = self.session.register_auto_experiment(
            self.key, kind=self.state.kind, layer=self.state.layer, params=params
        )
        self.state.my_seq = seq
        if previous is not None:
            self.session.cancel_experiment(previous)
        self.state.last_result = None

    def cancel(self) -> None:
        if self.state.my_seq is None:
            return
        self.session.cancel_experiment(self.state.my_seq)
        if not self.session.recording.is_recording(self.record_key):
            self.session.unregister_auto_experiment(self.key)

    def heartbeat(self) -> bool:
        self.session.touch_auto_experiment(self.key)
        return self.session.recording.is_recording(self.record_key)

    def consume_auto_run(self, frozen: bool) -> bool:
        if (
            (self.session.auto_run_experiments or self.state.my_seq is None)
            and self.state.dirty
            and not frozen
        ):
            self.state.dirty = False
            return True
        return False
