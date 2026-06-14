"""LeNet-5-style convnet for 32x32 inputs.

Follows LeCun et al. 1998 in spirit — two conv+pool stages followed by
three fully connected layers — with the modern substitutions of ReLU for
tanh and max pooling for average pooling. Inputs arrive at the 32x32
resolution the original architecture assumed (the MNIST config scales its
28x28 digits up), so neither conv needs padding; other square sizes work
too and simply resize the first fully connected layer.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


class LeNet(nn.Module):
    def __init__(
        self, num_classes: int = 10, in_channels: int = 1, image_size: int = 32
    ) -> None:
        super().__init__()
        # Two (conv k5, pool /2) stages: image_size -> (size - 4) / 2, twice.
        feature_size = ((image_size - 4) // 2 - 4) // 2
        if feature_size < 1:
            raise ValueError(f"image_size={image_size} too small for LeNet")
        self.conv1 = nn.Conv2d(in_channels, 6, kernel_size=5)
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        self.fc1 = nn.Linear(16 * feature_size * feature_size, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x: Tensor) -> Tensor:
        x = self.pool1(torch.relu(self.conv1(x)))
        x = self.pool2(torch.relu(self.conv2(x)))
        x = x.flatten(1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)
