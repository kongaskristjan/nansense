"""Editing an optimizer's param-group hyperparameters mid-run.

`Session.set_hyperparameters` validates a request here (`plan_changes`), then
writes it (`apply_changes`) while the training thread is paused. Validation
covers the whole request before anything is written, so a bad entry never
leaves a half-applied edit.

An edit is a plain write into `optimizer.param_groups`, which has two
consequences the docs spell out: a scheduler keeps driving the value from
there (a chainable one such as StepLR continues from the new value, one that
recomputes from its own schedule such as LambdaLR or OneCycleLR overwrites it
at its next step), and time travel restores the checkpoint's values.

An `lr` edit can instead set a per-group schedule multiplier
(`plan_multipliers`): new `lr` divided by the current, unscaled scheduled
`lr`. `ScheduleRescaler` applies it after every later scheduler step,
preserving the schedule's shape. Time travel restores the multipliers in
force at the checkpoint. Only `lr` is rescaled.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from nansense.schedule import BatchPosition

# Knobs whose negative values no torch optimizer accepts.
_NON_NEGATIVE: frozenset[str] = frozenset(
    {"lr", "weight_decay", "eps", "momentum", "dampening", "lr_decay"}
)


@dataclass(frozen=True)
class HyperparameterEdit:
    """A param-group hyperparameter that was changed from NaNsense.

    `original` is the value just before the first edit of this
    `(group, key)` — what the training script (or its scheduler) had set.
    `position` is where the run was paused when that first edit was made.
    """

    group: int
    key: str
    original: float
    position: BatchPosition


@dataclass(frozen=True)
class LrMultiplier:
    """A parameter group's learning-rate multiplier, set at `position`.

    `factor` applies until the next rescaling edit (1.0 ends it). The history
    lets time travel restore the multipliers in force at the checkpoint.
    """

    group: int
    factor: float
    position: BatchPosition


@dataclass(frozen=True)
class HyperparameterPayload:
    """What a distributed leader sends followers after an edit."""

    values: list[dict[str, float]]
    lr_multipliers: dict[int, float]


def active_multipliers(history: tuple[LrMultiplier, ...]) -> dict[int, float]:
    """Return active learning-rate multipliers by group, omitting factors of 1."""
    latest = {record.group: record.factor for record in history}
    return {group: factor for group, factor in latest.items() if factor != 1.0}


def numeric_hyperparams(group: Mapping[str, object]) -> dict[str, float]:
    """A param group's plain-numeric knobs (`lr`, `momentum`, ...).

    Booleans (`nesterov`, `amsgrad`) and tuples (`betas`) are left out; these
    are exactly the values the weights page shows and lets you edit.
    """
    return {
        key: float(value)
        for key, value in group.items()
        if key != "params"
        and isinstance(value, (int, float))
        and not isinstance(value, bool)
    }


def group_indices(model: nn.Module, optimizer: Optimizer | None) -> dict[str, int]:
    """Map each parameter name to the index of its param group."""
    if optimizer is None:
        return {}
    names = {id(p): n for n, p in model.named_parameters()}
    return {
        names[id(param)]: index
        for index, group in enumerate(optimizer.param_groups)
        for param in group["params"]
        if id(param) in names
    }


def plan_changes(
    optimizer: Optimizer, values: Mapping[str, float], group: int | None
) -> list[tuple[int, str, float]]:
    """Validate a request into `(group, key, value)` writes; `ValueError` if bad.

    `group=None` targets every param group, each of which must have the key.
    A value for an integer-valued knob must itself be a whole number.
    """
    groups = optimizer.param_groups
    if not values:
        raise ValueError("No hyperparameters provided; specify at least one key and value.")
    if group is None:
        targets = list(range(len(groups)))
    elif 0 <= group < len(groups):
        targets = [group]
    else:
        raise ValueError(
            f"Param group {group} does not exist; the optimizer has "
            f"{len(groups)} groups (indices 0–{len(groups) - 1})."
        )
    changes: list[tuple[int, str, float]] = []
    for index in targets:
        editable = numeric_hyperparams(groups[index])
        for key, value in values.items():
            if key not in editable:
                raise ValueError(
                    f"Param group {index} has no numeric hyperparameter {key!r}; "
                    f"editable keys: {', '.join(sorted(editable)) or 'none'}."
                )
            value = float(value)
            if not math.isfinite(value):
                raise ValueError(f"{key} must be finite; received {value}.")
            if key in _NON_NEGATIVE and value < 0:
                raise ValueError(f"{key} cannot be negative; received {value}.")
            if isinstance(groups[index][key], int) and not value.is_integer():
                raise ValueError(f"{key} requires a whole number; received {value}.")
            changes.append((index, key, value))
    return changes


def apply_changes(
    optimizer: Optimizer, changes: list[tuple[int, str, float]]
) -> list[tuple[int, str, float]]:
    """Write `changes` into the param groups; returns `(group, key, old)` triples.

    An integer-valued knob keeps its `int` type, so the optimizer sees exactly
    the kind of value the script gave it.
    """
    previous: list[tuple[int, str, float]] = []
    for index, key, value in changes:
        group = optimizer.param_groups[index]
        old = group[key]
        group[key] = int(value) if isinstance(old, int) else value
        previous.append((index, key, float(old)))
    return previous


def plan_multipliers(
    optimizer: Optimizer,
    changes: list[tuple[int, str, float]],
    active: Mapping[int, float],
) -> dict[int, float]:
    """Compute per-group schedule multipliers for the `lr` entries in `changes`.

    The current, unscaled scheduled `lr` is the live `lr` divided by the
    active multiplier; the new factor is the requested `lr` divided by that
    scheduled value. Factors close to 1 are rounded to 1, so restoring the
    scheduled value ends the multiplier. Raises `ValueError` if `changes`
    has no `lr` entry or the current scheduled `lr` is zero.
    """
    factors: dict[int, float] = {}
    for index, key, value in changes:
        if key != "lr":
            continue
        scheduled = float(optimizer.param_groups[index]["lr"]) / active.get(index, 1.0)
        if scheduled == 0.0:
            raise ValueError(
                f"The scheduled lr for parameter group {index} is zero and "
                "cannot be rescaled; set lr without rescale_schedule instead."
            )
        factor = value / scheduled
        factors[index] = 1.0 if math.isclose(factor, 1.0, rel_tol=1e-9) else factor
    if not factors:
        raise ValueError("Only lr can be rescaled; include an lr value.")
    return factors


class ScheduleRescaler:
    """Apply per-group factors to `lr` after each scheduler step.

    While any factor is active, the scheduler's class is swapped for a
    subclass whose `step` divides each scaled group's `lr` by its factor,
    runs the original step, then multiplies again — so the scheduler's own
    arithmetic (chainable or closed-form) only ever sees its own schedule.
    Swapping the class rather than setting an instance attribute keeps
    `state_dict()`, which copies `__dict__`, free of NaNsense state. With no
    active factor the original class is back and nothing is wrapped.
    """

    def __init__(self, scheduler: LRScheduler) -> None:
        self._scheduler = scheduler
        self._original: type[LRScheduler] = type(scheduler)
        self._rescaled: type[LRScheduler] | None = None
        self.factors: dict[int, float] = {}

    def set_factors(self, factors: Mapping[int, float]) -> None:
        """Set multipliers on the paused training thread; a factor of 1 ends one."""
        self.factors = {g: f for g, f in factors.items() if f != 1.0}
        patched = type(self._scheduler) is not self._original
        if self.factors and not patched:
            self._scheduler.__class__ = self._rescaled_class()
        elif not self.factors and patched:
            self._scheduler.__class__ = self._original

    def _rescaled_class(self) -> type[LRScheduler]:
        if self._rescaled is not None:
            return self._rescaled
        original = self._original
        rescaler = self

        def step(scheduler: LRScheduler, *args: Any, **kwargs: Any) -> None:
            groups = scheduler.optimizer.param_groups
            factors = dict(rescaler.factors)
            for index, factor in factors.items():
                groups[index]["lr"] /= factor
            try:
                original.step(scheduler, *args, **kwargs)
            finally:
                for index, factor in factors.items():
                    groups[index]["lr"] *= factor
                scheduler._last_lr = [group["lr"] for group in groups]

        self._rescaled = type(
            original.__name__,
            (original,),
            {
                "step": step,
                "__module__": original.__module__,
                "__qualname__": original.__qualname__,
            },
        )
        return self._rescaled
