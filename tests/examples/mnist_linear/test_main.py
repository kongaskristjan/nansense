"""Tests for the trivial MNIST linear example."""

from __future__ import annotations

import torch

from examples.mnist_linear.main import build_model
from tests.examples.helpers import assert_training_reduces_loss


def test_model_forward_shape() -> None:
    model = build_model()
    x = torch.randn(4, 1, 28, 28)
    assert model(x).shape == (4, 10)


def test_model_is_fx_traceable() -> None:
    """nansense's preferred capture path requires a successful symbolic trace."""
    torch.fx.symbolic_trace(build_model())


def test_training_step_reduces_loss() -> None:
    torch.manual_seed(0)
    model = build_model()
    x = torch.randn(16, 1, 28, 28)
    y = torch.randint(0, 10, (16,))
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    assert_training_reduces_loss(model, x, y, optimizer=optimizer)
