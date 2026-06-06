"""MNIST data loading with basic augmentation."""

from __future__ import annotations

from pathlib import Path

from torch.utils.data import DataLoader
from torchvision import datasets, transforms

MNIST_MEAN: tuple[float, ...] = (0.1307,)
MNIST_STD: tuple[float, ...] = (0.3081,)


def build_transforms(train: bool) -> transforms.Compose:
    if train:
        return transforms.Compose(
            [
                transforms.RandomCrop(28, padding=2),
                transforms.RandomRotation(10),
                transforms.ToTensor(),
                transforms.Normalize(MNIST_MEAN, MNIST_STD),
            ]
        )
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(MNIST_MEAN, MNIST_STD),
        ]
    )


def build_dataloaders(
    data_dir: Path,
    batch_size: int = 256,
    num_workers: int = 2,
    download: bool = True,
) -> tuple[DataLoader, DataLoader]:
    train_set = datasets.MNIST(
        root=str(data_dir), train=True, download=download, transform=build_transforms(train=True)
    )
    test_set = datasets.MNIST(
        root=str(data_dir), train=False, download=download, transform=build_transforms(train=False)
    )

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
