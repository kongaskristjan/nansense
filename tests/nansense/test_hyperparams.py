"""Editing optimizer hyperparameters mid-run (`Session.set_hyperparameters`)."""

from __future__ import annotations

import io
from collections.abc import Callable

import pytest
import torch
from torch.optim.lr_scheduler import LRScheduler

import nansense
from nansense import hyperparams
from nansense.hyperparams import HyperparameterEdit, LrMultiplier
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


_SCHEDULERS: dict[str, Callable[[torch.optim.Optimizer], LRScheduler]] = {
    "lambda": lambda o: torch.optim.lr_scheduler.LambdaLR(o, lambda e: 1 / (e + 1)),
    "onecycle": lambda o: torch.optim.lr_scheduler.OneCycleLR(
        o, max_lr=1.0, total_steps=20
    ),
    "step": lambda o: torch.optim.lr_scheduler.StepLR(o, step_size=2, gamma=0.5),
    "cosine": lambda o: torch.optim.lr_scheduler.CosineAnnealingLR(o, T_max=8),
    "plateau": lambda o: torch.optim.lr_scheduler.ReduceLROnPlateau(o, patience=0),
}


def _step(scheduler: LRScheduler) -> None:
    scheduler.optimizer.step()
    if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
        scheduler.step(1.0)  # never improves, so it keeps reducing
    else:
        scheduler.step()


@pytest.mark.parametrize("kind", sorted(_SCHEDULERS))
def test_rescaler_scales_the_rest_of_the_schedule(kind: str) -> None:
    """Every later step is the unedited schedule times the factor, and the
    wrapper leaves no trace in the state dict and comes off at factor 1."""
    pairs = [
        (opt, _SCHEDULERS[kind](opt))
        for opt in (torch.optim.SGD(TinyNet().parameters(), lr=0.1) for _ in range(2))
    ]
    (_, reference), (optimizer, scheduler) = pairs
    for _ in range(3):
        _step(reference), _step(scheduler)
    rescaler = hyperparams.ScheduleRescaler(scheduler)
    optimizer.param_groups[0]["lr"] *= 0.25
    rescaler.set_factors({0: 0.25})
    assert type(scheduler) is not type(reference)
    for _ in range(6):
        _step(reference), _step(scheduler)
        expected = reference.get_last_lr()[0] * 0.25
        assert optimizer.param_groups[0]["lr"] == pytest.approx(expected)
        assert scheduler.get_last_lr()[0] == pytest.approx(expected)
    assert set(scheduler.state_dict()) == set(reference.state_dict())
    torch.save(scheduler.state_dict(), io.BytesIO())
    rescaler.set_factors({0: 1.0})
    assert type(scheduler) is type(reference)


def _onecycle_session_parts() -> tuple[TinyNet, torch.optim.SGD, LRScheduler]:
    model = TinyNet()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    return model, optimizer, _SCHEDULERS["onecycle"](optimizer)


def test_rescale_schedule_round_trip() -> None:
    model, optimizer, scheduler = _onecycle_session_parts()

    def step(m: TinyNet) -> None:
        optimizer_train_step(m, optimizer)
        scheduler.step()

    with paused_session(
        model, step, optimizer=optimizer, scheduler=scheduler
    ) as session:
        scheduled = optimizer.param_groups[0]["lr"]
        session.set_hyperparameters({"lr": scheduled * 2}, rescale_schedule=True)
        assert session.lr_multipliers == {0: pytest.approx(2.0)}
        assert [e.key for e in session.hyperparameter_edits] == ["lr"]
        # Back to the schedule's own value: the multiplier and the edit end.
        session.set_hyperparameters({"lr": scheduled}, rescale_schedule=True)
        assert session.lr_multipliers == {}
        assert session.hyperparameter_edits == ()
        assert optimizer.param_groups[0]["lr"] == scheduled


@pytest.mark.parametrize(
    ("values", "with_scheduler", "message"),
    [
        ({"lr": 0.5}, False, "needs the optimizer's scheduler"),
        ({"weight_decay": 0.0}, True, "include an lr value"),
    ],
)
def test_rescale_schedule_refusals(
    values: dict[str, float], with_scheduler: bool, message: str
) -> None:
    model, optimizer, scheduler = _onecycle_session_parts()
    with paused_session(
        model,
        lambda m: optimizer_train_step(m, optimizer),
        optimizer=optimizer,
        scheduler=scheduler if with_scheduler else None,
    ) as session:
        before = dict(optimizer.param_groups[0])
        with pytest.raises(ValueError, match=message):
            session.set_hyperparameters(values, rescale_schedule=True)
        assert optimizer.param_groups[0] == before


def test_time_travel_restores_the_multipliers_of_its_checkpoint() -> None:
    _, optimizer, scheduler = _onecycle_session_parts()
    session = nansense.start(
        TinyNet(), epochs=3, phases={"train": 1}, optimizer=optimizer, scheduler=scheduler
    )
    session._lr_multipliers = tuple(
        LrMultiplier(0, factor, make_position("train", epoch, 0))
        for factor, epoch in [(2.0, 0), (3.0, 1)]
    )
    session._set_rescaler_factors(session.lr_multipliers)
    version = session._hparam_version
    session._rewind_to_epoch(1)
    assert session.lr_multipliers == {0: 2.0}
    assert session._hparam_version == version + 1
    session._rewind_to_epoch(0)
    assert session.lr_multipliers == {}
    assert type(scheduler) is torch.optim.lr_scheduler.OneCycleLR
