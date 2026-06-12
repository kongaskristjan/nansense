"""Smoke tests for the MNIST LeNet example."""

from __future__ import annotations

import pytest
import torch
from torch import nn

from examples.mnist_lenet.data import build_transforms
from examples.mnist_lenet.lenet import LeNet
from examples.common import evaluate, train_one_epoch


@pytest.mark.parametrize("batch_size", [1, 4])
def test_lenet_forward_shape(batch_size: int) -> None:
    model = LeNet(num_classes=10)
    x = torch.randn(batch_size, 1, 28, 28)
    assert model(x).shape == (batch_size, 10)


def test_lenet_is_fx_traceable() -> None:
    """nansense's preferred capture path requires a successful symbolic trace."""
    torch.fx.symbolic_trace(LeNet())


@pytest.mark.parametrize("train", [True, False])
def test_transforms_output_shape(train: bool) -> None:
    from PIL import Image

    transform = build_transforms(train=train)
    image = Image.new("L", (28, 28))
    out = transform(image)
    assert out.shape == (1, 28, 28)


def test_training_step_reduces_loss() -> None:
    torch.manual_seed(0)
    model = LeNet(num_classes=10)
    x = torch.randn(8, 1, 28, 28)
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


def test_train_and_eval_loops_run() -> None:
    torch.manual_seed(0)
    model = LeNet(num_classes=10)
    device = torch.device("cpu")
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05, momentum=0.9)

    inputs = torch.randn(16, 1, 28, 28)
    targets = torch.randint(0, 10, (16,))
    dataset = torch.utils.data.TensorDataset(inputs, targets)
    loader = torch.utils.data.DataLoader(dataset, batch_size=4)

    train_stats = train_one_epoch(model, loader, optimizer, criterion, device)
    eval_stats = evaluate(model, loader, criterion, device)

    assert 0.0 <= train_stats.accuracy <= 1.0
    assert 0.0 <= eval_stats.accuracy <= 1.0
    assert train_stats.loss > 0.0
    assert eval_stats.loss > 0.0
