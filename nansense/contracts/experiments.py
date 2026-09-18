"""Immutable, validated parameters for each experiment kind."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, fields, replace
from typing import ClassVar, Literal

ExperimentKind = Literal[
    "deep_dream", "gradcam", "neuron_gradient", "neuron_ig", "occlusion"
]
DEFAULT_BATCH = 8
LOCKED_PARAM_LIMITS = {"steps": 300, "channels": 8, "batch": 8, "ig_steps": 64}


@dataclass(frozen=True)
class DisplayParams:
    mean: tuple[float, ...] | None = None
    std: tuple[float, ...] | None = None


@dataclass(frozen=True)
class DreamParams(DisplayParams):
    kind: ClassVar[Literal["deep_dream"]] = "deep_dream"
    channels: int = DEFAULT_BATCH
    start: Literal["noise", "sample"] = "noise"
    sample: int = 0
    steps: int = 300
    all_steps: bool = False
    lr: float = 0.05
    diffusion: float = 0.05
    jitter: int = 2
    zoom: float = 1.0
    minimize: bool = False
    clamp: bool = True


@dataclass(frozen=True)
class AttributionParams(DisplayParams):
    batch: int = DEFAULT_BATCH


@dataclass(frozen=True)
class GradCamParams(AttributionParams):
    kind: ClassVar[Literal["gradcam"]] = "gradcam"
    target: int = -1


@dataclass(frozen=True)
class NeuronGradientParams(AttributionParams):
    kind: ClassVar[Literal["neuron_gradient"]] = "neuron_gradient"
    channel: int = 0


@dataclass(frozen=True)
class NeuronIGParams(AttributionParams):
    kind: ClassVar[Literal["neuron_ig"]] = "neuron_ig"
    channel: int = 0
    ig_steps: int = 32


@dataclass(frozen=True)
class OcclusionParams(AttributionParams):
    kind: ClassVar[Literal["occlusion"]] = "occlusion"
    channel: int = 0
    window: int = 4
    stride: int = 2


ExperimentParams = (
    DreamParams
    | GradCamParams
    | NeuronGradientParams
    | NeuronIGParams
    | OcclusionParams
)
_DEFAULTS: dict[str, ExperimentParams] = {
    value.kind: value
    for value in (
        DreamParams(),
        GradCamParams(),
        NeuronGradientParams(),
        NeuronIGParams(),
        OcclusionParams(),
    )
}
_MINIMUMS = {
    "channels": 1,
    "batch": 1,
    "sample": 0,
    "steps": 1,
    "lr": 0,
    "diffusion": 0,
    "jitter": 0,
    "zoom": 1,
    "target": -1,
    "channel": -1,
    "ig_steps": 2,
    "window": 1,
    "stride": 1,
}


def default_params(kind: str) -> dict[str, object]:
    """External form defaults, derived from the same types used by runners."""
    try:
        value = _DEFAULTS[kind]
    except KeyError:
        raise ValueError(f"unknown experiment kind {kind!r}") from None
    return {field.name: getattr(value, field.name) for field in fields(value)}


def resolve_params(
    kind: str,
    overrides: Mapping[str, object],
    defaults: Mapping[str, object] | None = None,
) -> tuple[dict[str, object], list[str]]:
    """Merge known form keys; report foreign keys to external adapters."""
    values = default_params(kind)
    values.update({k: v for k, v in (defaults or {}).items() if k in values})
    values.update({k: v for k, v in overrides.items() if k in values})
    return values, [k for k in overrides if k not in values]


def _finite_number(value: object, key: str) -> int | float:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        try:
            if math.isfinite(value):
                return value
        except OverflowError:
            pass
    raise ValueError(f"{key} must be a finite number")


def _normalization(value: object, key: str) -> tuple[float, ...] | None:
    if value is None:
        return None
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{key} must be a finite numeric sequence or None")
    numbers: list[float] = []
    for number in value:
        numbers.append(float(_finite_number(number, key)))
    result = tuple(numbers)
    if key == "std" and any(v <= 0 for v in result):
        raise ValueError("std values must be positive")
    return result


def parse_params(
    kind: str, values: Mapping[str, object], *, locked: bool = False
) -> ExperimentParams:
    """Validate once before queuing; preserve numeric floors and demo ceilings."""
    defaults = default_params(kind)
    unknown = values.keys() - defaults.keys()
    if unknown:
        raise ValueError(f"unknown parameters for {kind}: {sorted(unknown)}")
    parsed: dict[str, object] = {}
    for key, default in defaults.items():
        value = values.get(key, default)
        if key in ("mean", "std"):
            parsed[key] = _normalization(value, key)
        elif isinstance(default, bool):
            if not isinstance(value, bool):
                raise ValueError(f"{key} must be a boolean")
            parsed[key] = value
        elif isinstance(default, (int, float)):
            number = max(_MINIMUMS[key], _finite_number(value, key))
            if key == "diffusion":
                number = min(1.0, number)
            if locked and key in LOCKED_PARAM_LIMITS:
                number = min(LOCKED_PARAM_LIMITS[key], number)
            parsed[key] = int(number) if isinstance(default, int) else float(number)
        else:
            if value not in ("noise", "sample"):
                raise ValueError("start must be noise or sample")
            parsed[key] = value
    return replace(_DEFAULTS[kind], **parsed)
