"""Tests for the vision example entrypoint helpers."""

from __future__ import annotations

import io

import pytest
import torch

from examples import common
from examples.vision import main as main_module
from examples.vision.data import DATASETS
from examples.vision.lenet import LeNet
from examples.vision.resnet import PreActResNet
from examples.vision.vit import SimpleViT


def test_enable_line_buffering_sets_line_buffering(monkeypatch: pytest.MonkeyPatch) -> None:
    stream = io.TextIOWrapper(io.BytesIO(), line_buffering=False)
    monkeypatch.setattr(common.sys, "stdout", stream)

    common.enable_line_buffering()

    assert stream.line_buffering is True


def test_enable_line_buffering_tolerates_non_textiowrapper_stdout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A capture/proxy stdout (not a TextIOWrapper) must be left untouched, not raise."""
    monkeypatch.setattr(common.sys, "stdout", io.StringIO())

    common.enable_line_buffering()  # must be a no-op, not an error


@pytest.mark.parametrize("dataset", sorted(DATASETS))
@pytest.mark.parametrize(
    ("model_name", "model_cls"),
    [
        ("resnet", PreActResNet),
        ("resnet_deep", PreActResNet),
        ("vit", SimpleViT),
        ("lenet", LeNet),
    ],
)
def test_build_model(dataset: str, model_name: str, model_cls: type) -> None:
    config = DATASETS[dataset]
    model = main_module.build_model(model_name, config, blocks_per_stage=1)
    assert isinstance(model, model_cls)
    # Every architecture must accept the dataset's native input shape.
    x = torch.randn(2, config.in_channels, config.image_size, config.image_size)
    assert model(x).shape == (2, config.num_classes)
