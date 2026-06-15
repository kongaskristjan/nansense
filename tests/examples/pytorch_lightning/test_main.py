"""Smoke tests for the minimal Lightning example."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch
from lightning.pytorch import Trainer, seed_everything
from torch.utils.data import DataLoader, TensorDataset

from examples.pytorch_lightning import main as main_module
from examples.pytorch_lightning.main import MNISTClassifier
from nansense.lightning import NansenseCallback, fit_with_time_travel


def _synthetic_loader(*, shuffle: bool) -> DataLoader:
    """Two MNIST-shaped batches of four samples."""
    x = torch.randn(8, 1, 28, 28)
    y = torch.randint(0, 10, (8,))
    return DataLoader(TensorDataset(x, y), batch_size=4, shuffle=shuffle)


def test_default_batch_size(monkeypatch: pytest.MonkeyPatch) -> None:
    """The documented default is kept modest for low GPU memory."""
    monkeypatch.setattr(sys, "argv", ["main.py"])
    assert main_module.parse_args().batch_size == 64


@pytest.mark.parametrize("batch_size", [1, 4])
def test_forward_shape(batch_size: int) -> None:
    module = MNISTClassifier()
    x = torch.randn(batch_size, 1, 28, 28)
    assert module(x).shape == (batch_size, 10)


def test_net_is_fx_traceable() -> None:
    """The callback's `model="net"` path needs a traceable submodule."""
    torch.fx.symbolic_trace(MNISTClassifier().net)


def test_fit_with_time_travel_runs_disabled(tmp_path: Path) -> None:
    """The example's exact wiring, on synthetic data with nansense disabled."""
    seed_everything(0)
    module = MNISTClassifier()
    callback = NansenseCallback(model="net", enabled=False)
    before = [p.detach().clone() for p in module.parameters()]

    fit_with_time_travel(
        lambda: Trainer(
            max_epochs=1,
            accelerator="cpu",
            devices=1,
            logger=False,
            enable_checkpointing=False,
            enable_progress_bar=False,
            enable_model_summary=False,
            num_sanity_val_steps=0,
            default_root_dir=tmp_path,
        ),
        module,
        callback=callback,
        train_dataloaders=_synthetic_loader(shuffle=True),
        val_dataloaders=_synthetic_loader(shuffle=False),
        cache_dir=tmp_path / "cache",
    )

    after = list(module.parameters())
    assert any(
        not torch.equal(b, a.detach()) for b, a in zip(before, after, strict=True)
    )
