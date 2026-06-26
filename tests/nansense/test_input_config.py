"""Tests for per-input config resolution."""

from __future__ import annotations

import torch

from nansense.input_config import resolve_per_input


def test_single_value_applies_to_every_input() -> None:
    mean = (0.5, 0.5, 0.5)
    assert resolve_per_input(mean, "x") is mean
    assert resolve_per_input(mean, "other") is mean


def test_dict_resolves_by_input_name() -> None:
    cfg = {"img": (0.1,), "mask": (0.2,)}
    assert resolve_per_input(cfg, "img") == (0.1,)
    assert resolve_per_input(cfg, "mask") == (0.2,)
    assert resolve_per_input(cfg, "absent") is None  # no entry for this input


def test_none_config_or_name_resolves_to_none() -> None:
    assert resolve_per_input(None, "x") is None
    assert resolve_per_input((0.5,), None) is None


def test_resolves_a_callable_transform() -> None:
    def to_gray(x: torch.Tensor) -> torch.Tensor:
        return x.mean(dim=1, keepdim=True)

    # A bare callable applies everywhere; a dict selects per input.
    assert resolve_per_input(to_gray, "x") is to_gray
    assert resolve_per_input({"x": to_gray}, "x") is to_gray
    assert resolve_per_input({"x": to_gray}, "y") is None
