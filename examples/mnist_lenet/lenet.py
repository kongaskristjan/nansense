"""LeNet-5-style convnet for 28x28 MNIST digits.

Follows LeCun et al. 1998 in spirit — two conv+pool stages followed by
three fully connected layers — with the modern substitutions of ReLU for
tanh and max pooling for average pooling. The first conv pads by 2 so the
28x28 MNIST input matches the 32x32 input the original architecture
assumed.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


class LeNet(nn.Module):
    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, padding=2)
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x: Tensor) -> Tensor:
        x = self.pool1(torch.relu(self.conv1(x)))
        x = self.pool2(torch.relu(self.conv2(x)))
        x = x.flatten(1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)
