"""MNIST / CIFAR10 / Imagenette data loading with standard augmentations."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from torch.utils.data import DataLoader, Dataset, DistributedSampler
from torchvision import datasets, transforms

from examples.common import default_num_workers
from examples.mirrors import cifar10


@dataclass(frozen=True)
class DatasetConfig:
    """Static description of a supported dataset."""

    name: str
    image_size: int
    num_classes: int
    in_channels: int
    mean: tuple[float, ...]
    std: tuple[float, ...]


# Crop-augmentation padding modes, mapped to torchvision's `padding_mode`.
# `zero` is torchvision's default ("constant" with fill 0).
PADDING_MODES: dict[str, str] = {
    "zero": "constant",
    "reflection": "reflect",
    "edge": "edge",
    "symmetric": "symmetric",
}


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


def build_transforms(
    config: DatasetConfig, train: bool, padding: str = "zero"
) -> transforms.Compose:
    normalize = [transforms.ToTensor(), transforms.Normalize(config.mean, config.std)]
    padding_mode = PADDING_MODES[padding]
    size = config.image_size
    if config.name == "mnist":
        scale = [transforms.Resize((size, size))]
        if train:
            return transforms.Compose(
                [
                    *scale,
                    transforms.RandomCrop(size, padding=2, padding_mode=padding_mode),
                    transforms.RandomRotation(10),
                    *normalize,
                ]
            )
        return transforms.Compose([*scale, *normalize])
    if config.name == "cifar10":
        if train:
            return transforms.Compose(
                [
                    transforms.RandomCrop(size, padding=4, padding_mode=padding_mode),
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


def _build_dataset(
    config: DatasetConfig, data_dir: Path, train: bool, download: bool, padding: str = "zero"
) -> Dataset:
    transform = build_transforms(config, train=train, padding=padding)
    if config.name == "mnist":
        return datasets.MNIST(
            root=str(data_dir), train=train, download=download, transform=transform
        )
    if config.name == "cifar10":
        return cifar10(data_dir, train=train, download=download, transform=transform)
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
    batch_size: int = 64,
    num_workers: int = default_num_workers(),
    download: bool = True,
    padding: str = "zero",
) -> tuple[DataLoader, DataLoader]:
    train_set = _build_dataset(config, data_dir, train=True, download=download, padding=padding)
    test_set = _build_dataset(config, data_dir, train=False, download=download, padding=padding)

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


def ensure_downloaded(config: DatasetConfig, data_dir: Path) -> None:
    """Fetch the dataset to disk if missing (no loaders built).

    Used by the distributed launcher so a single rank downloads first and
    the others read from the populated cache, avoiding a concurrent
    first-run download race.
    """
    _build_dataset(config, data_dir, train=True, download=True)
    _build_dataset(config, data_dir, train=False, download=True)


def build_distributed_dataloaders(
    config: DatasetConfig,
    data_dir: Path,
    batch_size: int = 64,
    num_workers: int = default_num_workers(),
    download: bool = True,
    padding: str = "zero",
) -> tuple[DataLoader, DataLoader, DistributedSampler]:
    """Per-rank loaders over `DistributedSampler` shards (returns the train
    sampler so the caller can `set_epoch` each epoch).

    Both phases are sharded with `drop_last`, so every rank runs the same
    number of batches per phase — distributed NaNsense sessions (like DDP
    itself) require the ranks to advance through batches in lockstep.
    """
    train_set = _build_dataset(config, data_dir, train=True, download=download, padding=padding)
    test_set = _build_dataset(config, data_dir, train=False, download=download, padding=padding)

    train_sampler = DistributedSampler(train_set, shuffle=True, drop_last=True)
    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        sampler=DistributedSampler(test_set, shuffle=False, drop_last=True),
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    return train_loader, test_loader, train_sampler
