"""Train a tiny convnet on MNIST with PyTorch Lightning + nansense.

The minimal Lightning wiring: a `NansenseCallback` on a stock `Trainer`,
run through `fit_with_time_travel` so the UI's Time Travel button works.
This is also the manual testbed for changes to `nansense.lightning` —
small enough to start in seconds, full enough to exercise the callback,
scheduler restore, and time travel.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from lightning.pytorch import LightningModule, Trainer, seed_everything
from lightning.pytorch.utilities.types import OptimizerLRScheduler
from torch import Tensor, nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from examples.common import add_dtype_arg, amp_dtype_from_name, autocast, enable_line_buffering
from nansense.lightning import NansenseCallback, fit_with_time_travel

MNIST_MEAN: tuple[float, ...] = (0.1307,)
MNIST_STD: tuple[float, ...] = (0.3081,)


class TinyConvNet(nn.Module):
    """Two conv+pool stages and a linear head — small but image-shaped."""

    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, 8, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        self.fc = nn.Linear(16 * 7 * 7, num_classes)

    def forward(self, x: Tensor) -> Tensor:
        x = self.pool1(torch.relu(self.conv1(x)))
        x = self.pool2(torch.relu(self.conv2(x)))
        return self.fc(x.flatten(1))


class MNISTClassifier(LightningModule):
    """The network lives in `self.net` so the callback's `model="net"` can
    trace and probe the actual convnet instead of this wrapper.

    `amp_dtype` mirrors the other examples' `--dtype`: the step methods autocast
    their forward/loss to it while the Trainer stays at fp32 precision, so the
    weights remain fp32 and Lightning installs no GradScaler. fp16 gradients are
    therefore left unscaled on purpose — an underflow demo, not a bug."""

    def __init__(self, lr: float = 0.05, amp_dtype: torch.dtype | None = None) -> None:
        super().__init__()
        self.net = TinyConvNet()
        self.lr = lr
        self.amp_dtype = amp_dtype

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)

    def training_step(self, batch: tuple[Tensor, Tensor], batch_idx: int) -> Tensor:
        x, y = batch
        with autocast(self.device, self.amp_dtype):
            return nn.functional.cross_entropy(self(x), y)

    def validation_step(self, batch: tuple[Tensor, Tensor], batch_idx: int) -> Tensor:
        x, y = batch
        with autocast(self.device, self.amp_dtype):
            logits = self(x)
            loss = nn.functional.cross_entropy(logits, y)
        acc = (logits.argmax(dim=1) == y).float().mean()
        self.log("val_acc", acc, prog_bar=True, logger=False)
        return loss

    def configure_optimizers(self) -> OptimizerLRScheduler:
        optimizer = torch.optim.SGD(self.parameters(), lr=self.lr, momentum=0.9)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}


def build_dataloaders(
    data_dir: Path,
    batch_size: int = 128,
    num_workers: int = 2,
    download: bool = True,
) -> tuple[DataLoader, DataLoader]:
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize(MNIST_MEAN, MNIST_STD)]
    )
    train_set = datasets.MNIST(
        root=str(data_dir), train=True, download=download, transform=transform
    )
    val_set = datasets.MNIST(
        root=str(data_dir), train=False, download=download, transform=transform
    )
    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    return train_loader, val_loader


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("./data"))
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size (default 64; the tiny convnet uses very little GPU memory).",
    )
    parser.add_argument("--lr", type=float, default=0.05)
    add_dtype_arg(parser)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path(".nansense_cache/pytorch_lightning"),
        help="Directory for time-travel epoch checkpoints (default .nansense_cache/pytorch_lightning).",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--nansense-port",
        type=int,
        default=8080,
        help="Port for the nansense UI (default 8080).",
    )
    parser.add_argument(
        "--disable-nansense",
        action="store_true",
        help="Disable nansense with near-zero overhead (run as plain training).",
    )
    return parser.parse_args()


def main() -> None:
    enable_line_buffering()
    args = parse_args()
    seed_everything(args.seed)

    train_loader, val_loader = build_dataloaders(
        args.data_dir, batch_size=args.batch_size, num_workers=args.num_workers
    )
    module = MNISTClassifier(lr=args.lr, amp_dtype=amp_dtype_from_name(args.dtype))
    callback = NansenseCallback(
        port=args.nansense_port,
        model="net",
        input_mean=MNIST_MEAN,
        input_std=MNIST_STD,
        enabled=not args.disable_nansense,
    )

    # Each time-travel jump consumes the running fit, so the trainer comes
    # from a factory. `logger=False` because metric loggers cannot
    # time-travel; checkpointing is off because the restorer keeps its own
    # epoch cache in `--cache-dir`.
    fit_with_time_travel(
        lambda: Trainer(
            max_epochs=args.epochs,
            accelerator="auto",
            devices=1,
            logger=False,
            enable_checkpointing=False,
        ),
        module,
        callback=callback,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
        cache_dir=args.cache_dir,
    )


if __name__ == "__main__":
    main()
