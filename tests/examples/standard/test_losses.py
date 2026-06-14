"""Tests for the standard example's selectable `--loss` criteria."""

from __future__ import annotations

import pytest
import torch
from torch import nn

from examples.standard import main as main_module
from examples.standard.losses import (
    LOSSES,
    BalancedMAELoss,
    MAELoss,
    MSELoss,
    build_criterion,
)


@pytest.mark.parametrize(
    ("name", "cls"),
    [
        ("cross_entropy", nn.CrossEntropyLoss),
        ("mse", MSELoss),
        ("mae", MAELoss),
        ("mae_30", BalancedMAELoss),
    ],
)
def test_build_criterion_returns_expected_type(name: str, cls: type) -> None:
    assert name in LOSSES
    assert isinstance(build_criterion(name, num_classes=10), cls)


def test_build_criterion_rejects_unknown() -> None:
    with pytest.raises(ValueError, match="unknown loss"):
        build_criterion("huber", num_classes=10)


@pytest.mark.parametrize("name", LOSSES)
def test_loss_is_finite_scalar_with_gradient(name: str) -> None:
    """Every loss must be a finite scalar that backpropagates into the logits."""
    criterion = build_criterion(name, num_classes=4)
    logits = torch.randn(8, 4, requires_grad=True)
    targets = torch.randint(0, 4, (8,))

    loss = criterion(logits, targets)

    assert loss.ndim == 0
    assert torch.isfinite(loss)
    loss.backward()
    assert logits.grad is not None and torch.isfinite(logits.grad).all()


def test_mse_and_mae_match_onehot_distance() -> None:
    """The symmetric one-hot losses are just the elementwise distance, meaned."""
    probs = torch.tensor([[0.2, 0.8]])
    onehot = torch.tensor([[1.0, 0.0]])

    torch.testing.assert_close(
        MSELoss(2)._elementwise(probs, onehot), torch.tensor([[0.64, 0.64]])
    )
    torch.testing.assert_close(
        MAELoss(2)._elementwise(probs, onehot), torch.tensor([[0.8, 0.8]])
    )


def test_balanced_mae_weights_under_prediction_more() -> None:
    """`mae_30`: weight 0.7 when real > pred (under-prediction), 0.3 when real < pred."""
    probs = torch.tensor([[0.2, 0.8]])
    onehot = torch.tensor([[1.0, 0.0]])  # real=1 underpredicted, real=0 overpredicted

    elementwise = BalancedMAELoss(2)._elementwise(probs, onehot)

    # class 0: |0.2 - 1| * 0.7 = 0.56 ; class 1: |0.8 - 0| * 0.3 = 0.24
    torch.testing.assert_close(elementwise, torch.tensor([[0.56, 0.24]]))


def test_balanced_mae_is_asymmetric() -> None:
    """Mirror-image errors of equal magnitude cost differently (0.7 vs 0.3)."""
    loss = BalancedMAELoss(1)
    under = loss._elementwise(torch.tensor([[0.6]]), torch.tensor([[1.0]]))  # real > pred
    over = loss._elementwise(torch.tensor([[0.4]]), torch.tensor([[0.0]]))  # real < pred

    # Same |error| of 0.4, but under-prediction weighted 0.7 vs over's 0.3.
    torch.testing.assert_close(under / over, torch.tensor([[0.7 / 0.3]]))


def test_loss_argument_defaults_and_choices(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("sys.argv", ["main.py"])
    assert main_module.parse_args().loss == "cross_entropy"

    for name in LOSSES:
        monkeypatch.setattr("sys.argv", ["main.py", "--loss", name])
        assert main_module.parse_args().loss == name
