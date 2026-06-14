"""Tests for the vision example's dataset configs and transforms."""

from __future__ import annotations

import pytest
from PIL import Image
from torchvision import transforms

from examples.standard.data import DATASETS, PADDING_MODES, build_transforms


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


@pytest.mark.parametrize("name", ["mnist", "cifar10"])
@pytest.mark.parametrize("padding", sorted(PADDING_MODES))
def test_padding_mode_threads_to_random_crop(name: str, padding: str) -> None:
    """The `--padding` choice must reach the train-time `RandomCrop`; the
    cropped datasets (mnist/cifar10) carry it on their `padding_mode`."""
    transform = build_transforms(DATASETS[name], train=True, padding=padding)

    crops = [t for t in transform.transforms if isinstance(t, transforms.RandomCrop)]
    assert len(crops) == 1
    assert crops[0].padding_mode == PADDING_MODES[padding]


@pytest.mark.parametrize("padding", sorted(PADDING_MODES))
def test_padding_modes_produce_valid_output(padding: str) -> None:
    """Every padding mode yields a correctly shaped crop (e.g. `reflect`
    rejects pads >= the image size, so the small pads here must stay legal)."""
    config = DATASETS["cifar10"]
    transform = build_transforms(config, train=True, padding=padding)

    out = transform(Image.new("RGB", (config.image_size, config.image_size)))

    assert out.shape == (config.in_channels, config.image_size, config.image_size)


def test_unknown_padding_is_rejected() -> None:
    with pytest.raises(KeyError):
        build_transforms(DATASETS["cifar10"], train=True, padding="bogus")
