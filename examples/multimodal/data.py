"""Multimodal CIFAR-10: a 5-channel image plus a flat per-image stats vector.

Each example is `((image, stats), label)`:

- `image` is `[5, 32, 32]` — the three normalized RGB channels plus a luma
  channel and a Sobel edge-magnitude channel. Five channels is past what the UI
  can show directly, so `main.py` passes an `input_transform` that maps it back
  to the displayable RGB.
- `stats` is `[6]` — interpretable scalar features (mean R/G/B, luma contrast,
  edge density, bright-pixel fraction). A flat input, so the UI shows it as a
  per-channel strip you can click to perturb one feature.

Both modalities are derived from the same CIFAR image, so the model has a real
(if redundant) reason to fuse them.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms

# CIFAR-10 per-channel RGB normalization; the input_transform in `main.py`
# inverts it to recover the displayable image from the first three channels.
RGB_MEAN: tuple[float, float, float] = (0.4914, 0.4822, 0.4465)
RGB_STD: tuple[float, float, float] = (0.2470, 0.2435, 0.2616)

IMAGE_CHANNELS: int = 5
STATS_DIM: int = 6
NUM_CLASSES: int = 10

# Luma weights (Rec. 601) and Sobel kernels for the derived image channels.
_LUMA_WEIGHTS: Tensor = torch.tensor([0.299, 0.587, 0.114]).view(3, 1, 1)
_SOBEL_X: Tensor = torch.tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]])
_SOBEL_Y: Tensor = _SOBEL_X.t().contiguous()


def _edge_magnitude(luma: Tensor) -> Tensor:
    """Sobel gradient magnitude of a `[1, H, W]` luma map, size preserved.

    Reflection padding (not zero padding) keeps a flat region's edges at zero
    instead of lighting up the border against an implied black surround.
    """
    kernel = torch.stack([_SOBEL_X, _SOBEL_Y]).unsqueeze(1)  # [2, 1, 3, 3]
    padded = F.pad(luma.unsqueeze(0), (1, 1, 1, 1), mode="reflect")
    grad = F.conv2d(padded, kernel)[0]  # [2, H, W]
    return grad.pow(2).sum(dim=0, keepdim=True).sqrt()  # [1, H, W]


def to_multimodal(rgb01: Tensor) -> tuple[Tensor, Tensor]:
    """Turn a `[3, H, W]` image in `[0, 1]` into `(image[5,H,W], stats[6])`."""
    luma = (rgb01 * _LUMA_WEIGHTS).sum(dim=0, keepdim=True)  # [1, H, W]
    edges = _edge_magnitude(luma)  # [1, H, W]
    mean = torch.tensor(RGB_MEAN).view(3, 1, 1)
    std = torch.tensor(RGB_STD).view(3, 1, 1)
    rgb_norm = (rgb01 - mean) / std
    image = torch.cat([rgb_norm, luma, edges], dim=0)  # [5, H, W]
    stats = torch.tensor(
        [
            *rgb01.mean(dim=(1, 2)).tolist(),  # mean R, G, B
            float(luma.std()),  # contrast
            float(edges.mean()),  # edge density
            float((luma > 0.5).float().mean()),  # bright-pixel fraction
        ]
    )
    return image, stats


class MultiModalCIFAR(Dataset[tuple[tuple[Tensor, Tensor], int]]):
    """CIFAR-10 wrapped to yield `((image, stats), label)` per `to_multimodal`."""

    def __init__(self, root: Path, *, train: bool) -> None:
        self._base = datasets.CIFAR10(
            str(root), train=train, download=True, transform=transforms.ToTensor()
        )

    def __len__(self) -> int:
        return len(self._base)

    def __getitem__(self, index: int) -> tuple[tuple[Tensor, Tensor], int]:
        rgb01, label = self._base[index]
        return to_multimodal(rgb01), int(label)


def build_dataloaders(
    root: Path, *, batch_size: int, num_workers: int
) -> tuple[DataLoader, DataLoader]:
    train_loader = DataLoader(
        MultiModalCIFAR(root, train=True),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        MultiModalCIFAR(root, train=False),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    return train_loader, val_loader
