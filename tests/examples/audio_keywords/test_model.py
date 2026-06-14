"""Smoke tests for the keyword-classification ResNet."""

from __future__ import annotations

import pytest
import torch

from examples.audio_keywords.model import BasicBlock, KeywordResNet
from tests.examples.helpers import assert_training_reduces_loss


@pytest.mark.parametrize("n_mels", [40, 64])
def test_forward_shape(n_mels: int) -> None:
    model = KeywordResNet(num_classes=8, in_channels=1)
    x = torch.randn(4, 1, n_mels, 101)  # [B, 1, n_mels, n_frames]
    assert model(x).shape == (4, 8)


def test_stem_downsamples_by_four() -> None:
    """The ImageNet-style stem (7x7 stride-2 conv + stride-2 max pool) gives a
    4x downsample, the wide receptive field the shallow CNN lacked."""
    model = KeywordResNet()
    x = torch.randn(1, 1, 40, 104)
    model.eval()
    stem_out = model.stem(x)
    assert stem_out.shape[-2:] == (10, 26)


def test_is_fx_traceable() -> None:
    """The block stack must trace statically (nansense captures activations
    through `torch.fx.symbolic_trace`)."""
    model = KeywordResNet(num_classes=8, in_channels=1)
    traced = torch.fx.symbolic_trace(model)
    x = torch.randn(2, 1, 40, 101)
    model.eval()
    assert torch.allclose(traced(x), model(x))


def test_same_shape_block_skips_downsample() -> None:
    """A stride-1, same-channel block adds the identity directly (no shortcut)."""
    assert BasicBlock(16, 16, stride=1).downsample is None
    assert BasicBlock(16, 32, stride=2).downsample is not None


@pytest.mark.parametrize("blocks_per_stage", [(), (2, 0, 2)])
def test_rejects_invalid_stage_spec(blocks_per_stage: tuple[int, ...]) -> None:
    with pytest.raises(ValueError):
        KeywordResNet(blocks_per_stage=blocks_per_stage)


def test_training_step_reduces_loss() -> None:
    """CE loss drops on a tiny synthetic spectrogram batch."""
    torch.manual_seed(0)
    model = KeywordResNet(
        num_classes=8, in_channels=1, base_channels=8, blocks_per_stage=(1, 1)
    )
    x = torch.randn(8, 1, 40, 101)
    y = torch.randint(0, 8, (8,))
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    assert_training_reduces_loss(model, x, y, optimizer=optimizer)
