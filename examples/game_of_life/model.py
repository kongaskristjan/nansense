"""Fully-convolutional residual net that predicts the K-steps-ahead board.

Game of Life is local: one step depends only on a cell's 3x3 neighbourhood, so
predicting ``steps`` (K) steps ahead needs a receptive field of radius K. The
network is a stack of 3x3 residual blocks with **no downsampling**, each conv
using ``padding_mode='circular'`` to match the toroidal boundary of the data,
so a cell's K-step light cone is reproduced exactly. The output is a per-cell
logit map of the same H x W as the input.

The architecture is deliberately free of data-dependent control flow so
``torch.fx.symbolic_trace`` succeeds (nansense traces the graph to name layers).
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


class ResidualBlock(nn.Module):
    """Conv-BN-ReLU x2 with an identity shortcut (shapes never change)."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(
            channels, channels, kernel_size=3, padding=1, padding_mode="circular", bias=False
        )
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(
            channels, channels, kernel_size=3, padding=1, padding_mode="circular", bias=False
        )
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x: Tensor) -> Tensor:
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return torch.relu(out + x)


class LifeNet(nn.Module):
    """Fully-convolutional residual predictor of the K-steps-ahead board.

    A 3x3 stem lifts the single-channel board to ``channels`` features, then
    ``depth`` residual blocks (each two 3x3 convs, so receptive radius grows by
    2 per block) refine them, and a 1x1 head emits one logit per cell. With no
    downsampling the output keeps the input's ``H x W``; ``[B, 1, H, W]`` in,
    ``[B, 1, H, W]`` logits out.
    """

    def __init__(self, channels: int = 32, depth: int = 4, in_channels: int = 1) -> None:
        super().__init__()
        if depth < 1:
            raise ValueError(f"depth must be >= 1, got {depth}")
        self.stem = nn.Conv2d(
            in_channels, channels, kernel_size=3, padding=1, padding_mode="circular", bias=False
        )
        self.blocks = nn.Sequential(*[ResidualBlock(channels) for _ in range(depth)])
        self.head = nn.Conv2d(channels, in_channels, kernel_size=1)
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: Tensor) -> Tensor:
        x = torch.relu(self.stem(x))
        x = self.blocks(x)
        return self.head(x)
