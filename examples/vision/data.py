"""MNIST / CIFAR10 / Imagenette data loading with standard augmentations."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms


@dataclass(frozen=True)
class DatasetConfig:
    """Static description of a supported dataset."""

    name: str
    image_size: int
    num_classes: int
    in_channels: int
    mean: tuple[float, ...]
    std: tuple[float, ...]


DATASETS: dict[str, DatasetConfig] = {
    # 28x28 grayscale digits, scaled up to the 32x32 the models train at.
    "mnist": DatasetConfig(
        name="mnist",
        image_size=32,
        num_classes=10,
        in_channels=1,
        mean=(0.1307,),
        std=(0.3081,),
    ),
    "cifar10": DatasetConfig(
        name="cifar10",
        image_size=32,
        num_classes=10,
        in_channels=3,
        mean=(0.4914, 0.4822, 0.4465),
        std=(0.2470, 0.2435, 0.2616),
    ),
    # The 160px Imagenette variant (shorter side 160), trained at 128x128.
    "imagenette": DatasetConfig(
        name="imagenette",
        image_size=128,
        num_classes=10,
        in_channels=3,
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
    ),
}


def build_transforms(config: DatasetConfig, train: bool) -> transforms.Compose:
    normalize = [transforms.ToTensor(), transforms.Normalize(config.mean, config.std)]
    size = config.image_size
    if config.name == "mnist":
        scale = [transforms.Resize((size, size))]
        if train:
            return transforms.Compose(
                [
                    *scale,
                    transforms.RandomCrop(size, padding=2),
                    transforms.RandomRotation(10),
                    *normalize,
                ]
            )
        return transforms.Compose([*scale, *normalize])
    if config.name == "cifar10":
        if train:
            return transforms.Compose(
                [
                    transforms.RandomCrop(size, padding=4),
                    transforms.RandomHorizontalFlip(),
                    *normalize,
                ]
            )
        return transforms.Compose(normalize)
    if train:
        return transforms.Compose(
            [
                transforms.RandomResizedCrop(size, scale=(0.35, 1.0)),
                transforms.RandomHorizontalFlip(),
                *normalize,
            ]
        )
    # Standard 0.875 eval crop ratio (e.g. 128 out of a 146 shorter side).
    return transforms.Compose(
        [
            transforms.Resize(round(size / 0.875)),
            transforms.CenterCrop(size),
            *normalize,
        ]
    )


def _build_dataset(config: DatasetConfig, data_dir: Path, train: bool, download: bool) -> Dataset:
    transform = build_transforms(config, train=train)
    if config.name == "mnist":
        return datasets.MNIST(
            root=str(data_dir), train=train, download=download, transform=transform
        )
    if config.name == "cifar10":
        return datasets.CIFAR10(
            root=str(data_dir), train=train, download=download, transform=transform
        )
    return datasets.Imagenette(
        root=str(data_dir),
        split="train" if train else "val",
        size="160px",
        download=download,
        transform=transform,
    )


def build_dataloaders(
    config: DatasetConfig,
    data_dir: Path,
    batch_size: int = 128,
    num_workers: int = 2,
    download: bool = True,
) -> tuple[DataLoader, DataLoader]:
    train_set = _build_dataset(config, data_dir, train=True, download=download)
    test_set = _build_dataset(config, data_dir, train=False, download=download)

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    return train_loader, test_loader
