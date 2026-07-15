"""An ImageNet-style ResNet over log-mel spectrograms for keyword classification.

The spectrogram `[1, n_mels, n_frames]` is treated as a single-channel image
and fed to a standard ResNet (He et al. 2016, "Deep Residual Learning"): a
7x7 stride-2 stem followed by a 3x3 stride-2 max pool (a 4x downsample that
gives every later layer a wide receptive field over the time-frequency grid),
then four stages of `BasicBlock`s, a global average pool, and a linear
classifier. This is the ResNet-18 layout by default — two 3x3 convs per block,
two blocks per stage, channels 64/128/256/512 — so the receptive field spans
the whole clip rather than the few frames a shallow stride-2 CNN would see.

There is no data-dependent control flow in `forward` (the per-block shortcut is
chosen at construction time), so the whole model is `torch.fx.symbolic_trace`-
able — which NaNsense relies on to capture per-layer activations.
"""

from __future__ import annotations

from torch import Tensor, nn


class BasicBlock(nn.Module):
    """ResNet basic block: two 3x3 convs (Conv-BN-ReLU, Conv-BN) plus a shortcut.

    A `downsample` submodule (1x1 stride-`stride` conv + BN) is only registered
    when the residual path changes shape (stride or channel count); otherwise
    the input is added directly. The final ReLU follows the residual add, as in
    the original post-activation ResNet.
    """

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU(inplace=True)

        self.downsample: nn.Module | None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.downsample = None

    def forward(self, x: Tensor) -> Tensor:
        out = self.act(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        residual = self.downsample(x) if self.downsample is not None else x
        return self.act(out + residual)


class KeywordResNet(nn.Module):
    """ImageNet-style ResNet for single-channel spectrograms.

    The stem (7x7 stride-2 conv + 3x3 stride-2 max pool) downsamples by 4, then
    `blocks_per_stage` blocks run per stage; the first stage keeps the stem's
    resolution and every later stage downsamples by 2, doubling channels from
    `base_channels`. The default `blocks_per_stage=(2, 2, 2, 2)` is ResNet-18.
    """

    def __init__(
        self,
        num_classes: int = 8,
        in_channels: int = 1,
        base_channels: int = 64,
        blocks_per_stage: tuple[int, ...] = (2, 2, 2, 2),
    ) -> None:
        super().__init__()
        if not blocks_per_stage:
            raise ValueError("blocks_per_stage must contain at least one stage")
        if any(n < 1 for n in blocks_per_stage):
            raise ValueError(f"every stage needs >= 1 block, got {blocks_per_stage}")
        self.num_stages = len(blocks_per_stage)

        self.stem = nn.Sequential(
            nn.Conv2d(
                in_channels, base_channels, kernel_size=7, stride=2, padding=3, bias=False
            ),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        channels = base_channels
        for i, num_blocks in enumerate(blocks_per_stage):
            out_channels = base_channels * 2**i
            self.add_module(
                f"stage{i + 1}",
                self._make_stage(
                    channels, out_channels, num_blocks, stride=1 if i == 0 else 2
                ),
            )
            channels = out_channels
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(channels, num_classes)

        self._init_weights()

    @staticmethod
    def _make_stage(
        in_channels: int, out_channels: int, num_blocks: int, stride: int
    ) -> nn.Sequential:
        layers: list[nn.Module] = [BasicBlock(in_channels, out_channels, stride=stride)]
        for _ in range(num_blocks - 1):
            layers.append(BasicBlock(out_channels, out_channels, stride=1))
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
        x = self.pool(x).flatten(1)
        return self.fc(x)
