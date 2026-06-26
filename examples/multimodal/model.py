"""A two-input fusion network: a 5-channel image branch + a flat-stats branch.

`forward(self, image, stats)` gives the model two named inputs (`image`,
`stats`), which is what the nansense UI's input picker, multi-input probe
re-forwarding, and per-input perturbation key off. The image branch is a small
strided CNN; the stats branch is an MLP over the `[6]` feature vector
(BatchNorm-normalized first, so the raw features stay readable in the input
strip); their embeddings are concatenated and classified.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from examples.multimodal.data import IMAGE_CHANNELS, NUM_CLASSES, STATS_DIM


class ConvBlock(nn.Module):
    """Conv → BatchNorm → ReLU, optionally halving spatial size with stride."""

    def __init__(self, in_ch: int, out_ch: int, *, stride: int = 1) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, stride=stride)
        self.bn = nn.BatchNorm2d(out_ch)

    def forward(self, x: Tensor) -> Tensor:
        return torch.relu(self.bn(self.conv(x)))


class MultiModalNet(nn.Module):
    def __init__(
        self,
        *,
        num_classes: int = NUM_CLASSES,
        image_channels: int = IMAGE_CHANNELS,
        stats_dim: int = STATS_DIM,
        width: int = 32,
    ) -> None:
        super().__init__()
        self.stem = ConvBlock(image_channels, width)
        self.down1 = ConvBlock(width, width * 2, stride=2)  # 32 -> 16
        self.down2 = ConvBlock(width * 2, width * 4, stride=2)  # 16 -> 8
        self.pool = nn.AdaptiveAvgPool2d(1)

        # Normalize the raw stats here (not in the dataset) so the input strip
        # shows the human-readable feature values.
        self.stats_norm = nn.BatchNorm1d(stats_dim)
        self.stats_fc1 = nn.Linear(stats_dim, width)
        self.stats_fc2 = nn.Linear(width, width)

        self.classifier = nn.Linear(width * 4 + width, num_classes)

    def forward(self, image: Tensor, stats: Tensor) -> Tensor:
        img = self.pool(self.down2(self.down1(self.stem(image)))).flatten(1)
        s = torch.relu(self.stats_fc2(torch.relu(self.stats_fc1(self.stats_norm(stats)))))
        return self.classifier(torch.cat([img, s], dim=1))
