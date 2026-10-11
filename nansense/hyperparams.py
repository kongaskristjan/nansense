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
from typing import Any, cast

from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from nansense.schedule import BatchPosition

# Knobs whose negative values no torch optimizer accepts (for a tuple, each
# element: Rprop's `step_sizes`).
_NON_NEGATIVE: frozenset[str] = frozenset(
    {"lr", "weight_decay", "eps", "momentum", "dampening", "lr_decay", "step_sizes"}
)

# Tuple-valued knobs whose elements must lie in [0, 1): Adam-style `betas`.
_UNIT_INTERVAL: frozenset[str] = frozenset({"betas"})


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


def _is_number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


@dataclass(frozen=True)
class _Entry:
    """Where an editable value lives: `group[key]`, or `group[key][index]`."""

    key: str
    index: int | None
    value: int | float


def _entries(group: Mapping[str, object]) -> dict[str, _Entry]:
    """Return a parameter group's editable numeric values by name.

    Scalar numbers keep their key. Each element of a nonempty, all-numeric
    tuple or list gets a name: the key with any trailing `s` removed, followed
    by a 1-based position (`beta1`, `beta2` for `betas`). If that name is
    already a group key, use `key[i]` with a zero-based index instead.
    """
    entries: dict[str, _Entry] = {}
    for key, value in group.items():
        if key == "params":
            continue
        if _is_number(value):
            entries[key] = _Entry(key, None, cast(int | float, value))
        elif isinstance(value, (tuple, list)) and value and all(map(_is_number, value)):
            for i, element in enumerate(value):
                name = f"{key.removesuffix('s')}{i + 1}"
                entries[name if name not in group else f"{key}[{i}]"] = _Entry(
                    key, i, cast(int | float, element)
                )
    return entries


def numeric_hyperparams(group: Mapping[str, object]) -> dict[str, float]:
    """Return editable numeric hyperparameters (`lr`, `momentum`, `beta1`, ...).

    Excludes booleans such as `nesterov` and `amsgrad`. Numeric tuples and
    lists expose one value per element: `betas` becomes `beta1` and `beta2`,
    or `betas[i]` (zero-based) if the corresponding name is already a group
    key. These are the settings shown and edited on the weights page.
    """
    return {name: float(entry.value) for name, entry in _entries(group).items()}


def source_key(group: Mapping[str, object], name: str) -> str:
    """Return the optimizer setting behind a `numeric_hyperparams` name.

    For example, `lr` stays `lr`, and `beta1` maps to `betas[0]`.
    """
    entry = _entries(group).get(name)
    if entry is None or entry.index is None:
        return name
    return f"{entry.key}[{entry.index}]"


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
    """Validate edits and return `(group, key, value)` writes.

    Keys are names returned by `numeric_hyperparams`; `beta2`, for example,
    targets `betas[1]`. `group=None` targets every parameter group, and each
    key must exist in every targeted group. Values must be finite, integer
    settings require whole numbers, and each beta must be in `[0, 1)`.
    Raises `ValueError` if any edit is invalid.
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
        editable = _entries(groups[index])
        for key, value in values.items():
            entry = editable.get(key)
            if entry is None:
                raise ValueError(
                    f"Param group {index} has no numeric hyperparameter {key!r}; "
                    f"editable keys: {', '.join(sorted(editable)) or 'none'}."
                )
            value = float(value)
            if not math.isfinite(value):
                raise ValueError(f"{key} must be finite; received {value}.")
            if entry.key in _NON_NEGATIVE and value < 0:
                raise ValueError(f"{key} cannot be negative; received {value}.")
            if entry.key in _UNIT_INTERVAL and not 0 <= value < 1:
                raise ValueError(f"{key} must be at least 0 and below 1; received {value}.")
            if isinstance(entry.value, int) and not value.is_integer():
                raise ValueError(f"{key} requires a whole number; received {value}.")
            changes.append((index, key, value))
    return changes


def apply_changes(
    optimizer: Optimizer, changes: list[tuple[int, str, float]]
) -> list[tuple[int, str, float]]:
    """Apply validated edits and return `(group, key, old_value)` triples.

    Keys are names returned by `numeric_hyperparams`. Integer settings keep
    their `int` type. Editing a tuple or list element preserves the container
    type and the other elements.
    """
    previous: list[tuple[int, str, float]] = []
    for index, name, value in changes:
        group = optimizer.param_groups[index]
        entry = _entries(group)[name]
        new = int(value) if isinstance(entry.value, int) else value
        if entry.index is None:
            group[entry.key] = new
        else:
            sequence = group[entry.key]
            elements = list(sequence)
            elements[entry.index] = new
            group[entry.key] = type(sequence)(elements)
        previous.append((index, name, float(entry.value)))
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
