"""A small 2D CNN over log-mel spectrograms for keyword classification.

The spectrogram `[1, n_mels, n_frames]` is treated as a single-channel image:
stacked Conv-BN-ReLU blocks with stride-2 downsampling, a global average pool,
then a linear classifier. There is no data-dependent control flow in `forward`,
so the whole model is `torch.fx.symbolic_trace`-able — which nansense relies on
to capture per-layer activations.
"""

from __future__ import annotations

from torch import Tensor, nn


class ConvBlock(nn.Module):
    """Conv-BN-ReLU, optionally downsampling by a stride-2 convolution."""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: Tensor) -> Tensor:
        return self.act(self.bn(self.conv(x)))


class KeywordCNN(nn.Module):
    """Compact spectrogram CNN: a stem plus stride-2 blocks, global pool, Linear.

    Channels double at each downsampling stage starting from `base_channels`.
    """

    def __init__(
        self,
        num_classes: int = 8,
        in_channels: int = 1,
        base_channels: int = 32,
        num_stages: int = 4,
    ) -> None:
        super().__init__()
        if num_stages < 1:
            raise ValueError(f"num_stages must be >= 1, got {num_stages}")

        blocks: list[nn.Module] = [ConvBlock(in_channels, base_channels, stride=1)]
        channels = base_channels
        for _ in range(num_stages - 1):
            blocks.append(ConvBlock(channels, channels * 2, stride=2))
            channels *= 2
        self.features = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(channels, num_classes)

        self._init_weights()

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
        x = self.features(x)
        x = self.pool(x).flatten(1)
        return self.fc(x)
