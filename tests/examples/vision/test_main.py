"""Tests for the vision example entrypoint helpers."""

from __future__ import annotations

import io

import pytest
import torch

from examples.vision import main as main_module
from examples.vision.data import DATASETS
from examples.vision.resnet import ResNetCIFAR
from examples.vision.vit import SimpleViT


def test_enable_line_buffering_sets_line_buffering(monkeypatch: pytest.MonkeyPatch) -> None:
    stream = io.TextIOWrapper(io.BytesIO(), line_buffering=False)
    monkeypatch.setattr(main_module.sys, "stdout", stream)

    main_module.enable_line_buffering()

    assert stream.line_buffering is True


def test_enable_line_buffering_tolerates_non_textiowrapper_stdout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A capture/proxy stdout (not a TextIOWrapper) must be left untouched, not raise."""
    monkeypatch.setattr(main_module.sys, "stdout", io.StringIO())

    main_module.enable_line_buffering()  # must be a no-op, not an error


@pytest.mark.parametrize("dataset", sorted(DATASETS))
@pytest.mark.parametrize(
    ("model_name", "model_cls"),
    [("resnet", ResNetCIFAR), ("resnet_deep", ResNetCIFAR), ("vit", SimpleViT)],
)
def test_build_model(dataset: str, model_name: str, model_cls: type) -> None:
    config = DATASETS[dataset]
    model = main_module.build_model(model_name, config, blocks_per_stage=1)
    assert isinstance(model, model_cls)
    # Both architectures must accept the dataset's native input size.
    x = torch.randn(2, 3, config.image_size, config.image_size)
    assert model(x).shape == (2, config.num_classes)
