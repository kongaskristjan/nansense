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
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass

from torch import nn
from torch.optim import Optimizer

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
