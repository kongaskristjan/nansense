"""Lifecycle components work with explicit dependencies, without a Session."""

import threading
from collections.abc import Callable, Iterator

import torch

from nansense.contracts.experiments import DreamParams
from nansense.experiments import ExperimentManager, ExperimentRequest, ExperimentResult
from nansense.probe import ProbeManager
from nansense.session import BatchSnapshot
from tests.nansense.helpers import make_position


def test_probe_rejects_a_result_superseded_during_forward() -> None:
    snapshot = BatchSnapshot(
        position=make_position("train", 0, 0),
        activations={"x": torch.ones(1, 1)},
        activation_gradients={},
        weights={},
        weight_gradients={},
    )
    modes: list[str] = []

    def forward(inputs: dict[str, torch.Tensor], mode: str) -> dict[str, torch.Tensor]:
        modes.append(mode)
        if len(modes) == 1:
            manager.set_probe_mode("eval")
        return {name: tensor.clone() for name, tensor in inputs.items()}

    manager = ProbeManager(
        threading.Condition(),
        enabled=True,
        input_names=("x",),
        snapshot=lambda: snapshot,
        forward=forward,
        closed=lambda: False,
    )
    assert manager.pin_current_batch()
    assert manager.take_pending() == (True, [])
    manager.run_probe_guarded()
    assert manager.result is None
    assert manager.take_pending() == (True, [])
    manager.run_probe_guarded()
    assert manager.result is not None and manager.result.mode == "eval"
    assert modes == ["unchanged", "eval"]


def test_cancellation_after_dequeue_reaches_the_runner() -> None:
    def run(
        request: ExperimentRequest,
        aborted: Callable[[], bool],
    ) -> Iterator[ExperimentResult]:
        assert aborted()
        yield ExperimentResult(
            seq=request.seq,
            kind=request.kind,
            layer=request.layer,
            step=0,
            total_steps=1,
            done=True,
            error="cancelled",
        )

    manager = ExperimentManager(
        threading.Condition(),
        locked=lambda: False,
        control_state=lambda: (0, False, False),
        run=run,
        make_clip=lambda request: None,
    )
    seq = manager.request_experiment(layer="layer", params=DreamParams())
    request = manager.take_pending()
    assert request is not None
    manager.cancel_experiment(seq)
    manager.run_experiment_guarded(request)
    result = manager.result_for(seq)
    assert result is not None and result.error == "cancelled"
    assert manager.wait(timeout=0)
