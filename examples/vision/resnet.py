"""Pre-activation CIFAR ResNet (ResNet v2) with ResNet-D shortcuts.

The block layout follows He et al. 2016, "Identity Mappings in Deep Residual
Networks": each block applies BN -> ReLU -> Conv -> BN -> ReLU -> Conv before
adding the shortcut, so the identity path stays free of nonlinearities and
batch-norms. Downsampling shortcuts use the ResNet-D scheme (He et al. 2018,
"Bag of Tricks for Image Classification"): an average pool followed by a 1x1
conv, which avoids the information loss of a strided 1x1 conv. The parameter
count stays essentially identical to the original CIFAR ResNet-20.
"""

from __future__ import annotations

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


class ResNetCIFAR(nn.Module):
    """Pre-activation CIFAR ResNet with `blocks_per_stage` blocks at 3 stages.

    Total depth = 6 * blocks_per_stage + 2 (e.g. ResNet-20 -> blocks_per_stage=3).
    """

    def __init__(self, num_classes: int = 10, blocks_per_stage: int = 3) -> None:
        super().__init__()
        self.stem = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.stage1 = self._make_stage(16, 16, blocks_per_stage, stride=1)
        self.stage2 = self._make_stage(16, 32, blocks_per_stage, stride=2)
        self.stage3 = self._make_stage(32, 64, blocks_per_stage, stride=2)
        self.head_bn = nn.BatchNorm2d(64)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(64, num_classes)

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
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = torch.relu(self.head_bn(x))
        x = self.pool(x).flatten(1)
        return self.fc(x)


def resnet20(num_classes: int = 10) -> ResNetCIFAR:
    return ResNetCIFAR(num_classes=num_classes, blocks_per_stage=3)
