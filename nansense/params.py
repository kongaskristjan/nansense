"""Coercion helpers for loosely-typed `dict[str, object]` params.

Experiment requests and recorded views both carry user-shaped parameter
dicts whose values arrive as plain objects (UI widgets, persisted tuples).
These helpers coerce them defensively: a missing or wrongly-typed value
falls back to the default instead of raising.
"""

from __future__ import annotations


def float_param(params: dict[str, object], key: str, default: float) -> float:
    value = params.get(key, default)
    return float(value) if isinstance(value, (int, float)) else default


def int_param(params: dict[str, object], key: str, default: int = 0) -> int:
    value = params.get(key, default)
    return int(value) if isinstance(value, (int, float)) else default


def bool_param(params: dict[str, object], key: str, default: bool) -> bool:
    return bool(params.get(key, default))


def float_tuple(value: object, length: int | None = None) -> tuple[float, ...] | None:
    """`value` as a tuple of floats, or None if it isn't one (all-or-nothing).

    With `length` the tuple must additionally have exactly that many
    elements — e.g. per-channel normalization stats matching a channel
    count.
    """
    if not isinstance(value, (tuple, list)):
        return None
    if length is not None and len(value) != length:
        return None
    result: list[float] = []
    for v in value:
        if not isinstance(v, (int, float)):
            return None
        result.append(float(v))
    return tuple(result)


def str_tuple(value: object) -> tuple[str, ...]:
    if isinstance(value, (list, tuple)):
        return tuple(str(v) for v in value)
    return ()
