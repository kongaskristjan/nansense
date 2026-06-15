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


@pytest.mark.parametrize("amp_dtype", [None, torch.float16, torch.bfloat16])
def test_training_step_autocasts_but_keeps_weights_fp32(amp_dtype: torch.dtype | None) -> None:
    """The step autocasts its forward/loss to `amp_dtype` while the weights
    stay fp32 — the same fp32-weight, no-grad-scaler contract as `--dtype`."""
    torch.manual_seed(0)
    module = MNISTClassifier(amp_dtype=amp_dtype)
    batch = (torch.randn(4, 1, 28, 28), torch.randint(0, 10, (4,)))

    loss = module.training_step(batch, batch_idx=0)

    assert loss.requires_grad and torch.isfinite(loss)
    # cross_entropy stays fp32 under autocast; the conv weights must too.
    assert module.net.conv1.weight.dtype == torch.float32
    loss.backward()
    grad = module.net.conv1.weight.grad
    assert grad is not None and grad.dtype == torch.float32


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
