"""Tests for the loosely-typed param coercion helpers."""

from __future__ import annotations

import math

import pytest

from nansense.params import (
    bool_param,
    float_param,
    float_tuple,
    int_param,
    str_tuple,
)


@pytest.mark.parametrize(
    ("value", "expected"),
    [(3, 3), (3.7, 3), (True, 1), (-2.9, -2)],
)
def test_int_param_coerces_finite_numbers(value: object, expected: int) -> None:
    assert int_param({"k": value}, "k", default=99) == expected


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_int_param_falls_back_on_non_finite(value: float) -> None:
    # int(nan) raises ValueError and int(inf) raises OverflowError, both of
    # which used to slip through the isinstance(..., float) guard.
    assert int_param({"k": value}, "k", default=7) == 7


@pytest.mark.parametrize("value", ["x", None, [1], {}])
def test_int_param_falls_back_on_wrong_type(value: object) -> None:
    assert int_param({"k": value}, "k", default=7) == 7


def test_int_param_missing_key_uses_default() -> None:
    assert int_param({}, "k", default=5) == 5


def test_float_param_coerces_finite_numbers() -> None:
    assert float_param({"k": 2}, "k", default=9.0) == 2.0


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_float_param_falls_back_on_non_finite(value: float) -> None:
    result = float_param({"k": value}, "k", default=1.5)
    assert result == 1.5
    assert math.isfinite(result)


def test_bool_param_and_tuples_unaffected() -> None:
    assert bool_param({"k": 1}, "k", default=False) is True
    assert float_tuple([1, 2, 3], length=3) == (1.0, 2.0, 3.0)
    assert float_tuple([1, 2], length=3) is None
    assert str_tuple(["a", 2]) == ("a", "2")
