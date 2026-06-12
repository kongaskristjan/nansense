"""Tests for the vision example's dataset configs and transforms."""

from __future__ import annotations

import pytest
from PIL import Image

from examples.vision.data import DATASETS, build_transforms


def test_known_datasets() -> None:
    assert set(DATASETS) == {"mnist", "cifar10", "imagenette"}


@pytest.mark.parametrize("name", sorted(DATASETS))
def test_config_is_consistent(name: str) -> None:
    config = DATASETS[name]
    assert config.name == name
    assert config.num_classes == 10
    assert config.image_size % 8 == 0  # an 8x8 ViT patch grid must fit
    assert len(config.mean) == config.in_channels
    assert len(config.std) == config.in_channels


# Source images as the datasets deliver them: MNIST 28x28 grayscale,
# CIFAR10 at the target size, Imagenette 160px larger and non-square.
_SOURCE_IMAGES: dict[str, tuple[str, tuple[int, int]]] = {
    "mnist": ("L", (28, 28)),
    "cifar10": ("RGB", (32, 32)),
    "imagenette": ("RGB", (213, 160)),
}


@pytest.mark.parametrize("name", sorted(DATASETS))
@pytest.mark.parametrize("train", [True, False])
def test_transforms_output_shape(name: str, train: bool) -> None:
    config = DATASETS[name]
    mode, source_size = _SOURCE_IMAGES[name]
    transform = build_transforms(config, train=train)

    out = transform(Image.new(mode, source_size))

    assert out.shape == (config.in_channels, config.image_size, config.image_size)
