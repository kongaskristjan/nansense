"""Tests for the vision example's dataset configs and transforms."""

from __future__ import annotations

import pytest
from PIL import Image

from examples.vision.data import DATASETS, build_transforms


def test_known_datasets() -> None:
    assert set(DATASETS) == {"cifar10", "imagenette"}


@pytest.mark.parametrize("name", sorted(DATASETS))
def test_config_is_consistent(name: str) -> None:
    config = DATASETS[name]
    assert config.name == name
    assert config.num_classes == 10
    assert config.image_size % 8 == 0  # an 8x8 ViT patch grid must fit
    assert len(config.mean) == 3
    assert len(config.std) == 3


@pytest.mark.parametrize("name", sorted(DATASETS))
@pytest.mark.parametrize("train", [True, False])
def test_transforms_output_shape(name: str, train: bool) -> None:
    config = DATASETS[name]
    # CIFAR10 images arrive at the target size; Imagenette 160px images
    # arrive larger and non-square, e.g. shorter side 160.
    source_size = (32, 32) if name == "cifar10" else (213, 160)
    transform = build_transforms(config, train=train)

    out = transform(Image.new("RGB", source_size))

    assert out.shape == (3, config.image_size, config.image_size)
