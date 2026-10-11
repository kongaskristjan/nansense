"""User-registered experiment kinds (`Session.experiment`)."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import Any

import pytest
import torch
from torch import Tensor, nn

import nansense
from nansense import ExperimentContext, ExperimentOutput, ExperimentParam, Table
from nansense.contracts.experiments import CustomParams
from nansense.experiments import ExperimentResult
from nansense.session import Session
from tests.nansense.helpers import DynamicNet, TinyNet, paused_worker

_TOP = ExperimentParam("top", "Rows", "int", default=2, minimum=1)


@contextmanager
def _paused(model: nn.Module, register: Callable[[Session], None]) -> Iterator[Session]:
    """A session paused on its first batch, which came from `session.batches`
    (so the loader item is known), with `register` applied before training."""
    session = nansense.start(model, epochs=1, phases={"train": 2})
    register(session)
    items = [(torch.randn(2, 4), torch.tensor([0, 1])) for _ in range(2)]

    def loop() -> None:
        for x, y in session.batches(items, phase="train", epoch=0):
            nn.functional.cross_entropy(model(x), y).backward()

    with paused_worker(session, loop):
        yield session


def _run(
    session: Session, kind: str, layer: str = "fc1", **params: Any
) -> ExperimentResult:
    seq = session.request_experiment(kind=kind, layer=layer, params=params)
    assert session.wait_for_experiment(timeout=5.0)
    result = session.experiment_result_for(seq)
    assert result is not None and result.done
    return result


def test_a_custom_kind_runs_with_its_knobs_and_the_paused_batch() -> None:
    def register(session: Session) -> None:
        @session.experiment("rank_rows", params=[_TOP])
        def rank_rows(ctx: ExperimentContext) -> ExperimentOutput:
            """Largest activations of the layer.

            Lists the top rows."""
            assert isinstance(ctx.batch_item, tuple)
            _, y = ctx.batch_item
            assert isinstance(y, Tensor)
            act = ctx.activation(ctx.layer).detach().flatten()
            top = ctx.params["top"]
            assert isinstance(top, int)
            return ExperimentOutput(
                table=Table(("value",), [(float(v),) for v in act[:top]]),
                scalars={"targets": float(y.sum())},
            )

    with _paused(TinyNet(), register) as session:
        spec = session.experiment_spec("rank_rows")
        assert spec.label == "rank_rows" and spec.custom
        assert spec.summary == "Largest activations of the layer."
        assert "top rows" in spec.description
        result = _run(session, "rank_rows", top=0)  # clamped to the minimum
        assert result.error is None and result.payload == "table"
        assert result.table is not None and len(result.table.rows) == 1
        assert result.scalars == {"targets": 1.0}


@pytest.mark.parametrize("model", [TinyNet(), DynamicNet()], ids=["fx", "hooks"])
def test_edits_and_backward_in_user_code_never_reach_training(
    model: nn.Module,
) -> None:
    layer = "fc1" if isinstance(model, TinyNet) else "fc"
    observed: dict[str, Tensor] = {}

    def register(session: Session) -> None:
        @session.experiment("vandal")
        def vandal(ctx: ExperimentContext) -> ExperimentOutput:
            x = ctx.inputs[0]
            plain = ctx.forward(x)
            zeroed = ctx.forward(
                x, edit=lambda name, a: a * 0 if name == layer else None
            )
            assert isinstance(plain, Tensor) and isinstance(zeroed, Tensor)
            observed["changed"] = (plain - zeroed).abs().sum()
            for p in ctx.model.parameters():
                p.data.fill_(7.0)
            plain.sum().backward()
            return ExperimentOutput(note="done")

    with _paused(model, register) as session:
        before = [
            (p, p.detach().clone(), p.grad.clone())
            for p in model.parameters()
            if p.grad is not None
        ]
        result = _run(session, "vandal", layer=layer)
        assert result.error is None, result.error
        assert observed["changed"] > 0
        for p, value, grad in before:
            assert p.grad is not None
            assert torch.equal(p, value) and torch.equal(p.grad, grad)


def test_streamed_outputs_publish_progress_and_errors_name_the_line() -> None:
    def register(session: Session) -> None:
        @session.experiment("count")
        def count(ctx: ExperimentContext) -> Iterator[ExperimentOutput]:
            for i in range(3):
                yield ExperimentOutput(scalars={"i": i}, total_steps=3)

        @session.experiment("broken")
        def broken(ctx: ExperimentContext) -> ExperimentOutput:
            raise KeyError("missing")

    with _paused(TinyNet(), register) as session:
        result = _run(session, "count")
        assert (result.step, result.total_steps, result.scalars) == (3, 3, {"i": 2.0})
        error = _run(session, "broken").error
        assert error is not None and "KeyError" in error and "test_extensions.py" in error


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"name": "deep_dream"}, "already exists"),
        ({"name": "Bad Name"}, "snake_case"),
        ({"name": "x", "params": [ExperimentParam("n", "N", "int")]}, "default"),
        ({"name": "x", "params": [_TOP, _TOP]}, "taken"),
    ],
)
def test_invalid_registrations_raise(kwargs: dict[str, Any], match: str) -> None:
    session = nansense.start(TinyNet(), epochs=1, phases={"train": 1})
    name = kwargs.pop("name")
    with pytest.raises(ValueError, match=match):
        session.experiment(name, **kwargs)(lambda ctx: ExperimentOutput(note="x"))


def test_a_locked_session_offers_only_kinds_with_locked_limits_and_caps_them() -> None:
    session = nansense.start(TinyNet(), epochs=1, phases={"train": 1})
    for name, limits in (("private", None), ("public", {"top": 3})):
        session.experiment(name, params=[_TOP], locked_limits=limits)(
            lambda ctx: ExperimentOutput(note="x")
        )
    session.lock()
    assert "public" in session.experiment_specs
    assert "private" not in session.experiment_specs
    with pytest.raises(ValueError, match="unknown experiment kind"):
        session.request_experiment(kind="private", layer="fc1", params={})
    parsed = session.experiment_spec("public").parse({"top": 50}, True)
    assert isinstance(parsed, CustomParams) and parsed.values == {"top": 3}
    with pytest.raises(RuntimeError, match="locked"):
        session.experiment("late")


def test_a_disabled_session_returns_the_function_unregistered() -> None:
    session = nansense.start(TinyNet(), epochs=1, phases={"train": 1}, enabled=False)

    def fn(ctx: ExperimentContext) -> ExperimentOutput:
        return ExperimentOutput(note="x")

    assert session.experiment("noop")(fn) is fn
    assert "noop" not in session.experiment_specs
