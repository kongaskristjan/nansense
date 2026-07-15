"""Monocular depth network: a pretrained ResNet encoder + a U-Net decoder.

Transfer learning for a *dense* task. A torchvision classification backbone
(ResNet-18 by default) is repurposed as a multi-scale feature extractor via
`torchvision.models.feature_extraction.create_feature_extractor`, and a small
U-Net decoder upsamples the deepest features back toward the input resolution,
fusing the encoder's intermediate stages through skip connections.

The head predicts *log-depth* with a plain linear `1x1` convolution, so the
network's `forward` stays linear and data-independent (the exponentiation that
maps log-depth to metres lives in the loss/metric, not the model). Predicting
in log space matches the scale-invariant training objective and keeps the
output well-conditioned across the wide Make3D depth range (a few to ~70 m).

`create_feature_extractor` returns an fx `GraphModule`; calling it from the
parent's `forward` lets `torch.fx.symbolic_trace(build_model(...))` inline the
whole graph, so the model traces cleanly for NaNsense (no hook fallback needed).
"""

from __future__ import annotations

from collections.abc import Callable

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torchvision.models import ResNet, ResNet18_Weights, ResNet34_Weights, resnet18, resnet34
from torchvision.models._api import WeightsEnum
from torchvision.models.feature_extraction import create_feature_extractor

# Encoder stages returned by the feature extractor, deepest last. The names are
# the torchvision ResNet node names; the channel counts are identical for the
# ResNet-18 / ResNet-34 basic-block backbones supported here.
_RETURN_NODES: dict[str, str] = {
    "relu": "s1",  # 64ch, 1/2
    "layer1": "s2",  # 64ch, 1/4
    "layer2": "s3",  # 128ch, 1/8
    "layer3": "s4",  # 256ch, 1/16
    "layer4": "s5",  # 512ch, 1/32
}
# Channels of (s2, s3, s4, s5) for the basic-block ResNets.
_SKIP_CHANNELS: tuple[int, int, int, int] = (64, 128, 256, 512)

_BACKBONES: dict[str, tuple[Callable[..., ResNet], WeightsEnum]] = {
    "resnet18": (resnet18, ResNet18_Weights.DEFAULT),
    "resnet34": (resnet34, ResNet34_Weights.DEFAULT),
}


class UpBlock(nn.Module):
    """Upsample to the skip's resolution, concatenate, then two conv-BN-ReLU.

    The bilinear `F.interpolate` resizes the running decoder feature to match
    the encoder skip's spatial size before the channel-wise concatenation, so
    the decoder follows the encoder's exact stage resolutions regardless of the
    input size (no hard-coded shapes).
    """

    def __init__(self, in_channels: int, skip_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels + skip_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x: Tensor, skip: Tensor) -> Tensor:
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        return x


class DepthNet(nn.Module):
    """Pretrained ResNet encoder + U-Net decoder predicting log-depth.

    Decodes from the 1/32 deepest stage up to the 1/4 grid (matching the
    Make3D target resolution), fusing skips from the 1/16, 1/8, and 1/4
    encoder stages. The final `1x1` conv outputs a single log-depth channel.
    """

    def __init__(
        self,
        backbone: str = "resnet18",
        pretrained: bool = True,
        freeze_encoder: bool = False,
        decoder_channels: tuple[int, int, int] = (256, 128, 64),
    ) -> None:
        super().__init__()
        if backbone not in _BACKBONES:
            raise ValueError(f"unknown backbone {backbone!r}; choose from {sorted(_BACKBONES)}")
        builder, weights = _BACKBONES[backbone]
        encoder = builder(weights=weights if pretrained else None)
        self.encoder = create_feature_extractor(encoder, return_nodes=_RETURN_NODES)

        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad_(False)

        c2, c3, c4, c5 = _SKIP_CHANNELS
        d4, d3, d2 = decoder_channels
        self.up4 = UpBlock(c5, c4, d4)  # 1/32 -> 1/16
        self.up3 = UpBlock(d4, c3, d3)  # 1/16 -> 1/8
        self.up2 = UpBlock(d3, c2, d2)  # 1/8  -> 1/4
        self.head = nn.Conv2d(d2, 1, kernel_size=1)

    def forward(self, x: Tensor) -> Tensor:
        features = self.encoder(x)
        d = self.up4(features["s5"], features["s4"])
        d = self.up3(d, features["s3"])
        d = self.up2(d, features["s2"])
        return self.head(d)  # log-depth, [B, 1, h, w]


def build_model(
    backbone: str = "resnet18",
    pretrained: bool = True,
    freeze_encoder: bool = False,
) -> DepthNet:
    """Build the depth network.

    `pretrained=True` loads ImageNet weights for the encoder (a small download
    on first use); tests pass `pretrained=False` for random init and no network.
    `freeze_encoder=True` freezes the encoder so only the decoder trains.
    """
    return DepthNet(backbone=backbone, pretrained=pretrained, freeze_encoder=freeze_encoder)
