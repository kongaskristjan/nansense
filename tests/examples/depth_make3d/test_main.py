"""Entrypoint, data-pairing, and synthetic train/eval-loop tests (no network)."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pytest
import scipy.io
import torch
from PIL import Image
from torch.utils.data import DataLoader, TensorDataset

import nansense
from examples.common import evaluate, train_one_epoch
from examples.depth_make3d import main as main_module
from examples.depth_make3d.data import (
    DatasetConfig,
    Make3DDataset,
    _build_index,
    _load_depth,
)
from examples.depth_make3d.losses import ScaleInvariantLogLoss, delta_accuracy
from examples.depth_make3d.model import build_model


def test_build_optimizer_and_scheduler() -> None:
    model = build_model(pretrained=False)
    args = argparse.Namespace(lr=1e-3, weight_decay=0.05, epochs=10)
    optimizer, scheduler = main_module.build_optimizer_and_scheduler(model, args)
    assert isinstance(optimizer, torch.optim.AdamW)
    assert isinstance(scheduler, torch.optim.lr_scheduler.CosineAnnealingLR)


@pytest.mark.parametrize(("backbone", "expected"), [("resnet18", 48), ("resnet34", 32)])
def test_default_batch_size_is_backbone_dependent(backbone: str, expected: int) -> None:
    """The deeper resnet34 encoder gets the smaller batch, keeping peak GPU
    memory for the 192x256 inputs around ~4 GB for either backbone."""
    assert main_module.default_batch_size(backbone) == expected


def test_build_optimizer_excludes_frozen_encoder() -> None:
    """With the encoder frozen, the optimizer must only own decoder params."""
    model = build_model(pretrained=False, freeze_encoder=True)
    args = argparse.Namespace(lr=1e-3, weight_decay=0.05, epochs=10)
    optimizer, _ = main_module.build_optimizer_and_scheduler(model, args)
    owned = sum(p.numel() for group in optimizer.param_groups for p in group["params"])
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert owned == trainable
    assert owned < sum(p.numel() for p in model.parameters())


def _write_synthetic_example(image_dir: Path, depth_dir: Path, example_id: str) -> None:
    """Write a tiny img-<id>.jpg + depth_sph_corr-<id>.mat pair to disk."""
    image_dir.mkdir(parents=True, exist_ok=True)
    depth_dir.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.zeros((40, 50, 3), dtype=np.uint8)).save(
        image_dir / f"img-{example_id}.jpg"
    )
    grid = np.zeros((55, 305, 4), dtype=np.float64)
    grid[:, :, 3] = 12.0  # constant 12 m depth
    scipy.io.savemat(depth_dir / f"depth_sph_corr-{example_id}.mat", {"Position3DGrid": grid})


def test_build_index_pairs_by_id(tmp_path: Path) -> None:
    """Only depth maps whose `img-<id>.jpg` exists are paired."""
    image_dir, depth_dir = tmp_path / "img", tmp_path / "depth"
    _write_synthetic_example(image_dir, depth_dir, "aaa-001")
    # An orphan depth with no image must be skipped.
    depth_dir.mkdir(parents=True, exist_ok=True)
    grid = np.zeros((55, 305, 4), dtype=np.float64)
    scipy.io.savemat(depth_dir / "depth_sph_corr-orphan.mat", {"Position3DGrid": grid})

    pairs = _build_index(image_dir, depth_dir)
    assert len(pairs) == 1
    assert pairs[0][0].name == "img-aaa-001.jpg"


def test_load_depth_shape_and_mask(tmp_path: Path) -> None:
    """Depth resizes to the target grid; non-positive pixels become 0 sentinels."""
    grid = np.full((55, 305, 4), 5.0, dtype=np.float64)
    grid[:, :, 3] = 8.0
    grid[0, :, 3] = -1.0  # invalid row -> must clip to 0
    grid[1, :, 3] = np.inf  # non-finite -> must become 0
    path = tmp_path / "depth_sph_corr-x.mat"
    scipy.io.savemat(path, {"Position3DGrid": grid})

    depth = _load_depth(path, size=(48, 64), max_depth=70.0)
    assert depth.shape == (1, 48, 64)
    assert torch.isfinite(depth).all()
    assert (depth >= 0).all() and depth.max() <= 70.0
    assert (depth > 0).any()  # the bulk of the map is valid


def test_dataset_returns_image_and_depth(tmp_path: Path) -> None:
    """The map-style dataset yields (image[3,H,W], depth[1,h,w]) with download off."""
    config = DatasetConfig()
    image_dir, depth_dir = tmp_path / "Train400Img", tmp_path / "Train400Depth"
    _write_synthetic_example(image_dir, depth_dir, "syn-001")

    dataset = Make3DDataset(config, tmp_path, train=True, download=False)
    image, depth = dataset[0]
    assert image.shape == (3, *config.image_size)
    assert depth.shape == (1, *config.depth_size)


def test_dataset_raises_when_empty(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError):
        Make3DDataset(DatasetConfig(), tmp_path, train=True, download=False)


def _synthetic_depth_loader(n: int = 8, batch_size: int = 4) -> DataLoader:
    """A handful of (image, depth) pairs as a DataLoader, depth in metres."""
    torch.manual_seed(0)
    images = torch.randn(n, 3, 192, 256)
    depths = torch.rand(n, 1, 48, 64) * 40.0 + 1.0  # 1..41 m, all valid
    return DataLoader(TensorDataset(images, depths), batch_size=batch_size)


def test_train_and_eval_loops_run_on_synthetic_data() -> None:
    """One epoch of train + eval through `examples.common` with a *disabled*
    nansense session must produce a finite loss and a delta accuracy in [0, 1]
    — the same wiring main.py uses, but offline with a random-init backbone."""
    torch.manual_seed(0)
    model = build_model(pretrained=False)
    device = torch.device("cpu")
    criterion = ScaleInvariantLogLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    loader = _synthetic_depth_loader()

    session = nansense.start(
        model, epochs=1, phases={"train": 2, "val": 2}, enabled=False
    )
    train_stats = train_one_epoch(
        model, loader, optimizer, criterion, device,
        session=session, metric_fn=delta_accuracy,
    )
    eval_stats = evaluate(
        model, loader, criterion, device, session=session, metric_fn=delta_accuracy,
    )
    session.close()

    assert np.isfinite(train_stats.loss)
    assert 0.0 <= train_stats.accuracy <= 1.0
    assert 0.0 <= eval_stats.accuracy <= 1.0


def test_training_reduces_loss_on_fixed_batch() -> None:
    """A few AdamW steps on a fixed synthetic batch must reduce the loss."""
    torch.manual_seed(0)
    model = build_model(pretrained=False)
    model.train()
    images = torch.randn(2, 3, 192, 256)
    depths = torch.rand(2, 1, 48, 64) * 40.0 + 1.0
    criterion = ScaleInvariantLogLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)

    initial = criterion(model(images), depths).item()
    for _ in range(5):
        optimizer.zero_grad(set_to_none=True)
        loss = criterion(model(images), depths)
        loss.backward()
        optimizer.step()
    final = criterion(model(images), depths).item()

    assert final < initial
