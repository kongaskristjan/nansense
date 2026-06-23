"""Pre-activation ResNet (ResNet v2) with ResNet-D shortcuts.

The block layout follows He et al. 2016, "Identity Mappings in Deep Residual
Networks": each block applies BN -> ReLU -> Conv -> BN -> ReLU -> Conv before
adding the shortcut, so the identity path stays free of nonlinearities and
batch-norms. Downsampling shortcuts use the ResNet-D scheme (He et al. 2018,
"Bag of Tricks for Image Classification"): an average pool followed by a 1x1
conv, which avoids the information loss of a strided 1x1 conv. The parameter
count stays essentially identical to the original CIFAR ResNet-20.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor, nn


class PreActBlock(nn.Module):
    """Pre-activation basic block: BN-ReLU-Conv x2 with an optional shortcut.

    A `shortcut` submodule is only registered when the residual path
    changes shape (stride or channel count); in the same-shape case the
    input is added directly without a wrapping `nn.Identity()`.
    """

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False
        )

        self.shortcut: nn.Module | None
        if stride != 1 or in_channels != out_channels:
            layers: list[nn.Module] = []
            if stride != 1:
                layers.append(nn.AvgPool2d(kernel_size=stride, stride=stride))
            layers.append(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False)
            )
            self.shortcut = nn.Sequential(*layers)
        else:
            self.shortcut = None

    def forward(self, x: Tensor) -> Tensor:
        out = torch.relu(self.bn1(x))
        out = self.conv1(out)
        out = torch.relu(self.bn2(out))
        out = self.conv2(out)
        residual = self.shortcut(x) if self.shortcut is not None else x
        return out + residual


class PreActResNet(nn.Module):
    """Pre-activation ResNet with a configurable number of blocks per stage.

    Stages are registered as `stage1` .. `stage{num_stages}`; the first runs
    at stride 1 and every later one downsamples by 2, doubling the channel
    count (16, 32, 64, ...). `blocks_per_stage` is either a single int applied
    to every stage or a per-stage sequence of length `num_stages` (e.g. a
    deeper final stage). With a uniform int, total depth is
    2 * num_stages * blocks_per_stage + 2 (ResNet-20 -> 3 stages of 3 blocks).
    """

    STEM_CHANNELS: int = 16

    def __init__(
        self,
        num_classes: int = 10,
        blocks_per_stage: int | Sequence[int] = 3,
        num_stages: int = 3,
        in_channels: int = 3,
    ) -> None:
        super().__init__()
        if num_stages < 1:
            raise ValueError(f"num_stages must be >= 1, got {num_stages}")
        blocks = (
            [blocks_per_stage] * num_stages
            if isinstance(blocks_per_stage, int)
            else list(blocks_per_stage)
        )
        if len(blocks) != num_stages:
            raise ValueError(
                f"blocks_per_stage must have {num_stages} entries, got {len(blocks)}"
            )
        self.num_stages = num_stages
        self.stem = nn.Conv2d(
            in_channels, self.STEM_CHANNELS, kernel_size=3, stride=1, padding=1, bias=False
        )
        in_channels = self.STEM_CHANNELS
        for i in range(num_stages):
            out_channels = self.STEM_CHANNELS * 2**i
            self.add_module(
                f"stage{i + 1}",
                self._make_stage(
                    in_channels, out_channels, blocks[i], stride=1 if i == 0 else 2
                ),
            )
            in_channels = out_channels
        self.head_bn = nn.BatchNorm2d(in_channels)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(in_channels, num_classes)

        self._init_weights()

    @staticmethod
    def _make_stage(
        in_channels: int, out_channels: int, num_blocks: int, stride: int
    ) -> nn.Sequential:
        layers: list[nn.Module] = [PreActBlock(in_channels, out_channels, stride=stride)]
        for _ in range(num_blocks - 1):
            layers.append(PreActBlock(out_channels, out_channels, stride=1))
        return nn.Sequential(*layers)

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x: Tensor) -> Tensor:
        x = self.stem(x)
        for i in range(self.num_stages):
            x = getattr(self, f"stage{i + 1}")(x)
        x = torch.relu(self.head_bn(x))
        x = self.pool(x).flatten(1)
        return self.fc(x)

