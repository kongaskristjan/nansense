"""Editing optimizer hyperparameters mid-run (`Session.set_hyperparameters`)."""

from __future__ import annotations

import pytest
import torch

import nansense
from nansense import hyperparams
from nansense.hyperparams import HyperparameterEdit
from tests.nansense.helpers import (
    TinyNet,
    make_position,
    optimizer_train_step,
    paused_session,
)


def _two_group_sgd(model: TinyNet) -> torch.optim.SGD:
    return torch.optim.SGD(
        [
            {"params": model.fc1.parameters()},
            {"params": model.fc2.parameters(), "weight_decay": 0.0},
        ],
        lr=0.1,
        momentum=0.9,
        weight_decay=1e-4,
    )


@pytest.mark.parametrize(
    ("values", "group", "message"),
    [
        ({}, None, "No hyperparameters"),
        ({"lr": 0.1}, 2, "Param group 2 does not exist"),
        ({"nesterov": 1.0}, 0, "no numeric hyperparameter 'nesterov'"),
        ({"lr": float("nan")}, 0, "finite"),
        ({"lr": -1.0}, 0, "cannot be negative"),
    ],
)
def test_plan_changes_rejects(
    values: dict[str, float], group: int | None, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        hyperparams.plan_changes(_two_group_sgd(TinyNet()), values, group)


def test_apply_changes_keeps_integer_knobs_integral() -> None:
    optimizer = torch.optim.LBFGS(TinyNet().parameters(), max_iter=20)
    with pytest.raises(ValueError, match="whole number"):
        hyperparams.plan_changes(optimizer, {"max_iter": 2.5}, 0)
    changes = hyperparams.plan_changes(optimizer, {"max_iter": 5.0}, 0)
    assert hyperparams.apply_changes(optimizer, changes) == [(0, "max_iter", 20.0)]
    assert optimizer.param_groups[0]["max_iter"] == 5
    assert isinstance(optimizer.param_groups[0]["max_iter"], int)


def test_set_hyperparameters_while_paused() -> None:
    model = TinyNet()
    optimizer = _two_group_sgd(model)
    with paused_session(
        model, lambda m: optimizer_train_step(m, optimizer), optimizer=optimizer
    ) as session:
        assert session.optimizer_group_indices()["fc2.bias"] == 1
        session.set_hyperparameters({"lr": 0.01})
        assert [g["lr"] for g in optimizer.param_groups] == [0.01, 0.01]
        # The paused view shows the values the next step will use.
        snap = session.snapshot
        assert snap is not None
        assert snap.optimizer_hyperparams["fc2.weight"]["lr"] == 0.01
        session.set_hyperparameters({"weight_decay": 0.5}, group=1)
        edits = {(e.group, e.key): e.original for e in session.hyperparameter_edits}
        assert edits == {(0, "lr"): 0.1, (1, "lr"): 0.1, (1, "weight_decay"): 0.0}
        # Setting a value back to its original drops the record.
        session.set_hyperparameters({"lr": 0.1}, group=0)
        assert (0, "lr") not in {
            (e.group, e.key) for e in session.hyperparameter_edits
        }


@pytest.mark.parametrize("reason", ["no optimizer", "not paused", "locked"])
def test_set_hyperparameters_refusals(reason: str) -> None:
    model = TinyNet()
    optimizer = None if reason == "no optimizer" else _two_group_sgd(model)
    session = nansense.start(model, epochs=1, phases={"train": 1}, optimizer=optimizer)
    if reason == "locked":
        session.lock()
    with pytest.raises(ValueError):
        session.set_hyperparameters({"lr": 0.01})
    if optimizer is not None:
        assert optimizer.param_groups[0]["lr"] == 0.1


def test_time_travel_drops_edits_it_undoes() -> None:
    session = nansense.start(TinyNet(), epochs=3, phases={"train": 1})
    session._hparam_edits = tuple(
        HyperparameterEdit(0, key, 0.1, make_position("train", epoch, 0))
        for key, epoch in [("lr", 0), ("momentum", 1)]
    )
    session._rewind_to_epoch(1)
    assert [e.key for e in session.hyperparameter_edits] == ["lr"]
