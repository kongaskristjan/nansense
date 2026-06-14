"""Selectable classification losses for the standard example (`--loss`).

`cross_entropy` is the usual softmax + negative-log-likelihood. The other three
treat classification as *regression onto the one-hot target*: the model's logits
are turned into class probabilities with a softmax and compared, element by
element, against the one-hot encoding of the true label. This is a deliberately
pedagogical contrast — watch in the nansense views how an `mse` / `mae` run
shapes the logits and learning curve differently from `cross_entropy`.

`mae_30` is a *balanced* (asymmetric) absolute error: the pinball / quantile
loss at quantile 0.7. Each element is weighted 0.3 when the true value is below
the prediction and 0.7 when it is above, so under-prediction is penalised more
than over-prediction.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

# The supported `--loss` choices, listed in the order they appear in `--help`.
LOSSES: tuple[str, ...] = ("cross_entropy", "mse", "mae", "mae_30")

# `mae_30`: weight applied to the absolute error when the true (real) value is
# smaller than the prediction; the complementary `1 - _UNDER_WEIGHT` is used
# when it is larger. 0.3 / 0.7 makes this the quantile loss at quantile 0.7.
_UNDER_WEIGHT: float = 0.3


class _OneHotRegressionLoss(nn.Module):
    """Base for losses that regress softmax probabilities onto a one-hot target.

    Subclasses implement `_elementwise(probs, onehot)` returning a per-element
    error tensor; the mean over all elements is the loss. `targets` are integer
    class labels, matching `nn.CrossEntropyLoss`, so any choice is a drop-in
    swap in the training loop.
    """

    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.num_classes = num_classes

    def _elementwise(self, probs: Tensor, onehot: Tensor) -> Tensor:
        raise NotImplementedError

    def forward(self, logits: Tensor, targets: Tensor) -> Tensor:
        probs = logits.softmax(dim=1)
        onehot = F.one_hot(targets, self.num_classes).to(probs.dtype)
        return self._elementwise(probs, onehot).mean()


class MSELoss(_OneHotRegressionLoss):
    """Mean squared error between softmax probabilities and the one-hot target."""

    def _elementwise(self, probs: Tensor, onehot: Tensor) -> Tensor:
        return (probs - onehot) ** 2


class MAELoss(_OneHotRegressionLoss):
    """Mean absolute error between softmax probabilities and the one-hot target."""

    def _elementwise(self, probs: Tensor, onehot: Tensor) -> Tensor:
        return (probs - onehot).abs()


class BalancedMAELoss(_OneHotRegressionLoss):
    """Asymmetric absolute error (quantile loss): under-prediction costs more.

    With `error = pred - real`, the element is weighted `_UNDER_WEIGHT` (0.3)
    when `real < pred` (`error > 0`) and `1 - _UNDER_WEIGHT` (0.7) when
    `real > pred`, so the model is pushed towards over-predicting each class
    probability rather than under-predicting it.
    """

    def _elementwise(self, probs: Tensor, onehot: Tensor) -> Tensor:
        error = probs - onehot
        weight = torch.where(
            error > 0,
            torch.full_like(error, _UNDER_WEIGHT),
            torch.full_like(error, 1.0 - _UNDER_WEIGHT),
        )
        return weight * error.abs()


def build_criterion(name: str, num_classes: int) -> nn.Module:
    """Return the loss module for a `--loss` choice (see `LOSSES`)."""
    if name == "cross_entropy":
        return nn.CrossEntropyLoss()
    if name == "mse":
        return MSELoss(num_classes)
    if name == "mae":
        return MAELoss(num_classes)
    if name == "mae_30":
        return BalancedMAELoss(num_classes)
    raise ValueError(f"unknown loss {name!r}; choose from {', '.join(LOSSES)}")
