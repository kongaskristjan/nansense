"""Shared helpers for the example smoke tests."""

from __future__ import annotations

import torch
from torch import Tensor, nn


def assert_training_reduces_loss(
    model: nn.Module, x: Tensor, y: Tensor, *, optimizer: torch.optim.Optimizer
) -> None:
    """Five full optimization steps on a fixed (x, y) batch must reduce the
    cross-entropy loss below its initial value."""
    criterion = nn.CrossEntropyLoss()
    model.train()
    initial = criterion(model(x), y).item()
    for _ in range(5):
        optimizer.zero_grad(set_to_none=True)
        loss = criterion(model(x), y)
        loss.backward()
        optimizer.step()
    final = criterion(model(x), y).item()

    assert final < initial
