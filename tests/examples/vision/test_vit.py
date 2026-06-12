"""Smoke tests for the simple Vision Transformer."""

from __future__ import annotations

import pytest
import torch

from examples.vision.vit import SelfAttention, SimpleViT, TransformerBlock
from tests.examples.helpers import assert_training_reduces_loss


@pytest.mark.parametrize(
    ("image_size", "patch_size"),
    [(32, 4), (128, 16)],
)
def test_vit_forward_shape(image_size: int, patch_size: int) -> None:
    model = SimpleViT(
        image_size=image_size, patch_size=patch_size, num_classes=10, dim=32, depth=2, num_heads=2
    )
    x = torch.randn(2, 3, image_size, image_size)
    assert model(x).shape == (2, 10)


def test_self_attention_preserves_shape() -> None:
    attn = SelfAttention(dim=32, num_heads=4)
    x = torch.randn(2, 16, 32)
    assert attn(x).shape == (2, 16, 32)


def test_transformer_block_preserves_shape() -> None:
    block = TransformerBlock(dim=32, num_heads=4)
    x = torch.randn(2, 16, 32)
    assert block(x).shape == (2, 16, 32)


@pytest.mark.parametrize(
    ("dim", "num_heads"),
    [(32, 5), (33, 4)],
)
def test_self_attention_rejects_indivisible_heads(dim: int, num_heads: int) -> None:
    with pytest.raises(ValueError):
        SelfAttention(dim=dim, num_heads=num_heads)


def test_vit_rejects_indivisible_patch_size() -> None:
    with pytest.raises(ValueError):
        SimpleViT(image_size=32, patch_size=5)


def test_vit_is_fx_traceable() -> None:
    """nansense's preferred capture path requires a successful symbolic trace."""
    model = SimpleViT(image_size=32, patch_size=4, dim=32, depth=2, num_heads=2)
    traced = torch.fx.symbolic_trace(model)
    x = torch.randn(2, 3, 32, 32)
    assert torch.allclose(traced(x), model(x))


def test_vit_traced_batch_size_is_dynamic() -> None:
    """The trace must not bake in the batch size it was traced with."""
    model = SimpleViT(image_size=32, patch_size=4, dim=32, depth=1, num_heads=2)
    traced = torch.fx.symbolic_trace(model)
    assert traced(torch.randn(3, 3, 32, 32)).shape == (3, 10)
    assert traced(torch.randn(7, 3, 32, 32)).shape == (7, 10)


def test_vit_training_step_reduces_loss() -> None:
    torch.manual_seed(0)
    model = SimpleViT(image_size=32, patch_size=4, num_classes=10, dim=32, depth=2, num_heads=2)
    x = torch.randn(8, 3, 32, 32)
    y = torch.randint(0, 10, (8,))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    assert_training_reduces_loss(model, x, y, optimizer=optimizer)
