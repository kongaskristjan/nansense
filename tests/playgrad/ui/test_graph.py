"""Tests for the Mermaid graph builder."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from playgrad.ui.graph import CONFIG_HEADER, ROOT_ID, build_mermaid, slug


class TwoLayer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=3, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(),
        )
        self.fc = nn.Linear(8, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(self.stem(x).mean(dim=(-1, -2)))


class DynamicShape(nn.Module):
    """Data-dependent control flow defeats fx.symbolic_trace."""

    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.sum() > 0:
            return self.fc(x)
        return x


def test_includes_config_header_and_flowchart() -> None:
    src = build_mermaid(TwoLayer())
    assert src.startswith(CONFIG_HEADER)
    assert "flowchart TD" in src


def test_fx_emits_data_flow_edges() -> None:
    src = build_mermaid(TwoLayer())
    # fx unwraps the Sequential and gives a linear data-flow chain:
    # x -> stem.0 -> stem.1 -> stem.2 -> mean -> fc -> output
    assert "x --> stem_0" in src
    assert "stem_0 --> stem_1" in src
    assert "stem_1 --> stem_2" in src
    # mean is a tensor method call, name will be `mean`
    assert "stem_2 --> mean" in src
    assert "mean --> fc" in src


def test_fx_labels_modules_with_class_name() -> None:
    src = build_mermaid(TwoLayer())
    assert "Conv2d" in src
    assert "BatchNorm2d" in src
    assert "ReLU" in src
    assert "Linear" in src


class TwoBlocks(nn.Module):
    """Functional relus inside repeated submodules, like a ResNet block."""

    class Block(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.bn1 = nn.BatchNorm2d(4)
            self.bn2 = nn.BatchNorm2d(4)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return torch.relu(self.bn2(torch.relu(self.bn1(x))))

    def __init__(self) -> None:
        super().__init__()
        self.stem = nn.Conv2d(3, 4, kernel_size=3, padding=1)
        self.b0 = TwoBlocks.Block()
        self.b1 = TwoBlocks.Block()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.b1(self.b0(self.stem(x)))


def test_fx_scopes_repeated_function_nodes_by_submodule() -> None:
    src = build_mermaid(TwoBlocks())
    # Each block's two relus get scope-qualified ids and labels, so they are
    # distinguishable instead of a wall of identical "relu" ovals.
    assert 'b0_relu1(["b0.relu1"])' in src
    assert 'b0_relu2(["b0.relu2"])' in src
    assert 'b1_relu1(["b1.relu1"])' in src
    assert 'b1_relu2(["b1.relu2"])' in src
    # Edges reference the same scoped ids, so graph nodes still link to cards.
    assert "b0_bn1 --> b0_relu1" in src


def test_fx_node_shapes_by_kind() -> None:
    src = build_mermaid(TwoLayer())
    # Graph in/out are circles, weightless function/method calls (relu, add,
    # mean, ...) are ovals (stadiums), modules are rectangles.
    assert '  x(("in: x"))' in src
    assert '  output(("out"))' in src
    assert '  mean(["mean"])' in src
    assert "stem_0[" in src


def test_falls_back_to_hierarchy_for_untraceable_model() -> None:
    src = build_mermaid(DynamicShape())
    # The hierarchy fallback always emits a synthetic root and parent->child edges.
    assert "root --> fc" in src
    assert "Linear" in src


def test_hierarchy_fallback_root_label_can_be_customized() -> None:
    src = build_mermaid(DynamicShape(), root_label="my_model")
    assert '"my_model"' in src


def test_slug_replaces_non_alphanumeric_and_handles_empty() -> None:
    assert slug("stem.0") == "stem_0"
    assert slug("stage1.0.conv1") == "stage1_0_conv1"
    assert slug("relu") == "relu"
    assert slug("") == ROOT_ID


def test_handles_modules_with_tuple_args(monkeypatch: Any) -> None:
    # Walk recurses into tuples/lists, so calls like torch.cat([a, b], dim=1)
    # still produce edges from both inputs.
    class CatModel(nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return torch.cat([x, x], dim=1)

    src = build_mermaid(CatModel())
    # The single 'x' input should feed the cat node (appears twice).
    assert "x --> cat" in src
