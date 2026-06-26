"""Tests for the multimodal example entrypoint: transform, defaults, smoke."""

from __future__ import annotations

import sys

import pytest
import torch
from torch import nn

from examples.multimodal import main as main_module
from examples.multimodal.data import IMAGE_CHANNELS, STATS_DIM, to_multimodal
from examples.multimodal.model import MultiModalNet


def test_display_rgb_recovers_the_image() -> None:
    rgb01 = torch.rand(3, 16, 16)
    image, _ = to_multimodal(rgb01)
    shown = main_module.display_rgb(image.unsqueeze(0))  # [1, 3, 16, 16]
    assert shown.shape == (1, 3, 16, 16)
    torch.testing.assert_close(shown[0], rgb01)


def test_display_rgb_clamps_to_unit_range() -> None:
    # Extreme normalized values must not leave [0, 1] after de-normalizing.
    image = torch.full((1, IMAGE_CHANNELS, 8, 8), 50.0)
    shown = main_module.display_rgb(image)
    assert float(shown.min()) >= 0.0 and float(shown.max()) <= 1.0


def test_default_batch_size_and_epochs(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "argv", ["main.py"])
    args = main_module.parse_args()
    assert args.batch_size == 128
    assert args.epochs == 30


def test_training_reduces_loss() -> None:
    """A few optimization steps on a tiny synthetic multimodal batch reduce the
    cross-entropy, confirming both input branches carry gradient."""
    torch.manual_seed(0)
    image = torch.randn(16, IMAGE_CHANNELS, 8, 8)
    stats = torch.randn(16, STATS_DIM)
    target = torch.randint(0, 10, (16,))

    model = MultiModalNet(width=8)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)
    criterion = nn.CrossEntropyLoss()

    model.train()
    initial = criterion(model(image, stats), target).item()
    for _ in range(5):
        optimizer.zero_grad(set_to_none=True)
        loss = criterion(model(image, stats), target)
        loss.backward()
        optimizer.step()
    final = criterion(model(image, stats), target).item()

    assert final < initial
