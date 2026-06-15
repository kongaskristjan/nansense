"""Tests for the standard example entrypoint helpers."""

from __future__ import annotations

import argparse
import io
import os
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
from torch.utils.data import DistributedSampler, TensorDataset

from examples import common
from examples.standard import data as data_module
from examples.standard import main as main_module
from examples.standard.data import DATASETS
from examples.standard.lenet import LeNet
from examples.standard.resnet import PreActResNet
from examples.standard.vit import SimpleViT


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


def test_build_optimizer_and_scheduler() -> None:
    model = torch.nn.Linear(4, 2)
    args = argparse.Namespace(lr=1e-3, weight_decay=0.05, epochs=10)
    optimizer, scheduler = main_module.build_optimizer_and_scheduler(model, args)
    assert isinstance(optimizer, torch.optim.AdamW)
    assert isinstance(scheduler, torch.optim.lr_scheduler.CosineAnnealingLR)


@pytest.mark.parametrize(
    ("dataset", "expected"),
    [("cifar10", 64), ("mnist", 64), ("imagenette", 32)],
)
def test_default_batch_size_is_dataset_dependent(dataset: str, expected: int) -> None:
    """Imagenette's 128x128 inputs get the smaller batch; the 32x32 datasets
    share 64 — all kept modest for low GPU memory."""
    assert dataset in DATASETS
    assert main_module.default_batch_size(dataset) == expected


def test_build_distributed_dataloaders_shards_with_sampler(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The distributed loaders must wrap a `DistributedSampler` (returned for
    `set_epoch`) without hitting the network — `_build_dataset` is stubbed."""
    dataset = TensorDataset(torch.randn(8, 1, 32, 32), torch.randint(0, 10, (8,)))
    monkeypatch.setattr(data_module, "_build_dataset", lambda *a, **k: dataset)

    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29571")
    dist.init_process_group("gloo", rank=0, world_size=1)
    try:
        train_loader, _test_loader, sampler = data_module.build_distributed_dataloaders(
            DATASETS["mnist"], data_dir=tmp_path, batch_size=4, num_workers=0, download=False
        )
        assert isinstance(sampler, DistributedSampler)
        assert train_loader.sampler is sampler
        inputs, _targets = next(iter(train_loader))
        assert inputs.shape[0] == 4
    finally:
        dist.destroy_process_group()
