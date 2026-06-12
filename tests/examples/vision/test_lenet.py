"""Smoke tests for the vision example's LeNet."""

from __future__ import annotations

import pytest
import torch
from torch import nn

from examples.vision.lenet import LeNet


@pytest.mark.parametrize("batch_size", [1, 4])
@pytest.mark.parametrize(
    ("in_channels", "image_size"),
    [(1, 32), (3, 32), (3, 128)],
)
def test_lenet_forward_shape(batch_size: int, in_channels: int, image_size: int) -> None:
    model = LeNet(num_classes=10, in_channels=in_channels, image_size=image_size)
    x = torch.randn(batch_size, in_channels, image_size, image_size)
    assert model(x).shape == (batch_size, 10)


def test_lenet_rejects_too_small_images() -> None:
    with pytest.raises(ValueError, match="too small"):
        LeNet(image_size=12)


def test_lenet_is_fx_traceable() -> None:
    """nansense's preferred capture path requires a successful symbolic trace."""
    torch.fx.symbolic_trace(LeNet())


def test_training_step_reduces_loss() -> None:
    torch.manual_seed(0)
    model = LeNet(num_classes=10)
    x = torch.randn(8, 1, 32, 32)
    y = torch.randint(0, 10, (8,))
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05, momentum=0.9)

    model.train()
    initial = criterion(model(x), y).item()
    for _ in range(5):
        optimizer.zero_grad(set_to_none=True)
        loss = criterion(model(x), y)
        loss.backward()
        optimizer.step()
    final = criterion(model(x), y).item()

    assert final < initial
