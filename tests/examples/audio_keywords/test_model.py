"""Smoke tests for the keyword-classification CNN."""

from __future__ import annotations

import pytest
import torch

from examples.audio_keywords.model import KeywordCNN
from tests.examples.helpers import assert_training_reduces_loss


@pytest.mark.parametrize("n_mels", [40, 64])
def test_forward_shape(n_mels: int) -> None:
    model = KeywordCNN(num_classes=8, in_channels=1)
    x = torch.randn(4, 1, n_mels, 101)  # [B, 1, n_mels, n_frames]
    assert model(x).shape == (4, 8)


def test_is_fx_traceable() -> None:
    """The block stack must trace statically (nansense captures activations
    through `torch.fx.symbolic_trace`)."""
    model = KeywordCNN(num_classes=8, in_channels=1)
    traced = torch.fx.symbolic_trace(model)
    x = torch.randn(2, 1, 40, 101)
    model.eval()
    assert torch.allclose(traced(x), model(x))


def test_rejects_zero_stages() -> None:
    with pytest.raises(ValueError):
        KeywordCNN(num_stages=0)


def test_training_step_reduces_loss() -> None:
    """CE loss drops on a tiny synthetic spectrogram batch."""
    torch.manual_seed(0)
    model = KeywordCNN(num_classes=8, in_channels=1, base_channels=8, num_stages=3)
    x = torch.randn(8, 1, 40, 101)
    y = torch.randint(0, 8, (8,))
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    assert_training_reduces_loss(model, x, y, optimizer=optimizer)
