"""Smoke tests for the Make3D depth model (no pretrained-weight download)."""

from __future__ import annotations

import pytest
import torch

from examples.depth_make3d.model import DepthNet, build_model

# Random-init backbones throughout: `pretrained=False` keeps every test offline.


@pytest.mark.parametrize("backbone", ["resnet18", "resnet34"])
def test_build_model_output_shape(backbone: str) -> None:
    """The decoder must map a 192x256 RGB input to a [B, 1, 48, 64] depth grid."""
    model = build_model(backbone=backbone, pretrained=False)
    x = torch.randn(2, 3, 192, 256)
    out = model(x)
    assert out.shape == (2, 1, 48, 64)


def test_build_model_returns_depthnet() -> None:
    assert isinstance(build_model(pretrained=False), DepthNet)


def test_build_model_rejects_unknown_backbone() -> None:
    with pytest.raises(ValueError):
        build_model(backbone="resnet999", pretrained=False)


def test_freeze_encoder_freezes_only_the_encoder() -> None:
    """`--freeze-encoder` must drop the encoder's grads but keep the decoder's."""
    model = build_model(pretrained=False, freeze_encoder=True)
    assert all(not p.requires_grad for p in model.encoder.parameters())
    decoder_params = [
        p for n, p in model.named_parameters() if not n.startswith("encoder.")
    ]
    assert decoder_params and all(p.requires_grad for p in decoder_params)


def test_model_is_fx_traceable() -> None:
    """The encoder feature extractor + U-Net decoder must symbolic-trace whole,
    so nansense can build its graph without the runtime hook fallback."""
    model = build_model(pretrained=False)
    traced = torch.fx.symbolic_trace(model)
    x = torch.randn(2, 3, 192, 256)
    model.eval()
    traced.eval()
    assert torch.allclose(traced(x), model(x), atol=1e-5)
