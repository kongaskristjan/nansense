"""Tests for pure helpers in `nansense.ui.input_panel`."""

from __future__ import annotations

import pytest

from nansense.ui.input_panel import normalized_color

CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD = (0.2470, 0.2435, 0.2616)


@pytest.mark.parametrize(
    "hex_color, channels, mean, std, expected",
    [
        # RGB without stats: plain [0, 1] channel values.
        ("#ff0000", 3, None, None, (1.0, 0.0, 0.0)),
        ("#000000", 3, None, None, (0.0, 0.0, 0.0)),
        # Leading/trailing whitespace tolerated.
        (" #ffffff ", 3, None, None, (1.0, 1.0, 1.0)),
        # Grayscale: mean of the RGB components.
        ("#ff0000", 1, None, None, (1.0 / 3.0,)),
        ("#ffffff", 1, (0.5,), (0.5,), ((1.0 - 0.5) / 0.5,)),
    ],
)
def test_normalized_color_converts_display_colors(
    hex_color: str,
    channels: int,
    mean: tuple[float, ...] | None,
    std: tuple[float, ...] | None,
    expected: tuple[float, ...],
) -> None:
    values = normalized_color(hex_color, channels, mean, std)
    assert values == pytest.approx(expected)


def test_normalized_color_back_transforms_with_stats() -> None:
    values = normalized_color("#ffffff", 3, CIFAR_MEAN, CIFAR_STD)
    assert values is not None
    expected = tuple((1.0 - m) / s for m, s in zip(CIFAR_MEAN, CIFAR_STD))
    assert values == pytest.approx(expected)


@pytest.mark.parametrize(
    "hex_color, channels, mean, std",
    [
        ("#12345", 3, None, None),  # truncated hex
        ("#zzzzzz", 3, None, None),  # not hex digits
        ("#ffffff", 2, None, None),  # unsupported channel count
        ("#ffffff", 3, (0.5, 0.5), (0.5, 0.5)),  # stats length mismatch
    ],
)
def test_normalized_color_rejects_bad_input(
    hex_color: str,
    channels: int,
    mean: tuple[float, ...] | None,
    std: tuple[float, ...] | None,
) -> None:
    assert normalized_color(hex_color, channels, mean, std) is None
