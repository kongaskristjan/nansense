"""Tests for the scale-invariant log loss and the delta<1.25 metric."""

from __future__ import annotations

import pytest
import torch

from examples.depth_make3d.losses import ScaleInvariantLogLoss, delta_accuracy


def test_loss_decreases_on_synthetic_pair() -> None:
    """A learnable log-depth prediction must drive the loss down toward the
    target's log over a few optimizer steps (the loss is differentiable and
    well-behaved on a tiny masked pair)."""
    torch.manual_seed(0)
    target = torch.rand(2, 1, 8, 8) * 50.0 + 1.0  # 1..51 m, all valid
    pred = torch.zeros_like(target).requires_grad_(True)  # log-depth, learnable
    criterion = ScaleInvariantLogLoss()
    optimizer = torch.optim.SGD([pred], lr=0.5)

    initial = criterion(pred, target).item()
    for _ in range(20):
        optimizer.zero_grad(set_to_none=True)
        loss = criterion(pred, target)
        loss.backward()
        optimizer.step()
    final = criterion(pred, target).item()

    assert final < initial


def test_loss_ignores_invalid_pixels() -> None:
    """Pixels with target 0 (invalid) must not contribute: changing the
    prediction there leaves the loss unchanged."""
    target = torch.zeros(1, 1, 2, 2)
    target[0, 0, 0, 0] = 10.0  # one valid pixel
    criterion = ScaleInvariantLogLoss()

    pred_a = torch.full_like(target, 1.0)
    pred_b = pred_a.clone()
    pred_b[0, 0, 1, 1] = 999.0  # only an invalid pixel differs
    assert criterion(pred_a, target).item() == pytest.approx(criterion(pred_b, target).item())


def test_delta_accuracy_is_one_when_prediction_matches() -> None:
    """log(pred) == log(gt) everywhere valid -> ratio 1 < 1.25 -> accuracy 1.0."""
    target = torch.rand(2, 1, 4, 4) * 40.0 + 1.0
    pred = torch.log(target)
    assert delta_accuracy(pred, target) == pytest.approx(1.0)


def test_delta_accuracy_below_one_when_off() -> None:
    """A constant 2x scale error (ratio 2 > 1.25) yields 0 correct pixels."""
    target = torch.rand(2, 1, 4, 4) * 40.0 + 1.0
    pred = torch.log(target * 2.0)
    assert delta_accuracy(pred, target) < 1.0


def test_delta_accuracy_honours_mask() -> None:
    """Only valid pixels count: a perfect prediction on the lone valid pixel
    scores 1.0 even though the (invalid) pixels carry a wildly wrong value."""
    target = torch.zeros(1, 1, 2, 2)
    target[0, 0, 0, 0] = 10.0
    pred = torch.full_like(target, -5.0)  # wrong everywhere
    pred[0, 0, 0, 0] = torch.log(torch.tensor(10.0))  # correct on the valid pixel
    assert delta_accuracy(pred, target) == pytest.approx(1.0)


def test_delta_accuracy_zero_when_no_valid_pixels() -> None:
    target = torch.zeros(1, 1, 3, 3)
    pred = torch.randn(1, 1, 3, 3)
    assert delta_accuracy(pred, target) == 0.0
