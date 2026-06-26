"""Tests for the multimodal example's two-input fusion network."""

from __future__ import annotations

import torch
import torch.fx

import nansense
from examples.multimodal.data import IMAGE_CHANNELS, NUM_CLASSES, STATS_DIM
from examples.multimodal.model import MultiModalNet


def test_forward_shape_and_fx_traceable() -> None:
    model = MultiModalNet(width=8)
    image = torch.randn(2, IMAGE_CHANNELS, 32, 32)
    stats = torch.randn(2, STATS_DIM)
    assert model(image, stats).shape == (2, NUM_CLASSES)
    # nansense traces the graph to name layers and inputs; tracing must succeed.
    torch.fx.symbolic_trace(model)


def test_nansense_sees_both_named_inputs() -> None:
    session = nansense.start(MultiModalNet(width=8), epochs=1, phases={"train": 1})
    assert session.input_names == ["image", "stats"]
