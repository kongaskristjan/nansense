"""Train a single linear layer on MNIST — the most trivial nansense example.

Everything lives in this one file: a `Flatten -> Linear` model, plain SGD,
and the minimal nansense wiring (`nansense.start` + `session.batches`).
No scheduler, no time travel, no checkpointing — see `examples/vision`
for the full wiring.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import nansense

MNIST_MEAN: tuple[float, ...] = (0.1307,)
MNIST_STD: tuple[float, ...] = (0.3081,)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("./data"))
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument(
        "--nansense-port",
        type=int,
        default=8080,
        help="Port for the nansense UI (default 8080).",
    )
    return parser.parse_args()


def build_model() -> nn.Module:
    """One linear layer over the flattened 28x28 image: logistic regression."""
    return nn.Sequential(nn.Flatten(), nn.Linear(28 * 28, 10))


def build_dataloaders(data_dir: Path, batch_size: int) -> tuple[DataLoader, DataLoader]:
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize(MNIST_MEAN, MNIST_STD)]
    )
    train_set = datasets.MNIST(str(data_dir), train=True, download=True, transform=transform)
    test_set = datasets.MNIST(str(data_dir), train=False, download=True, transform=transform)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=batch_size)
    return train_loader, test_loader


def main() -> None:
    args = parse_args()
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, test_loader = build_dataloaders(args.data_dir, args.batch_size)
    model = build_model().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)

    session = nansense.start(
        model,
        epochs=args.epochs,
        phases={"train": len(train_loader), "val": len(test_loader)},
        optimizer=optimizer,
        port=args.nansense_port,
        input_mean=MNIST_MEAN,
        input_std=MNIST_STD,
    )
    print(f"nansense UI at http://127.0.0.1:{args.nansense_port}", flush=True)

    for epoch in range(args.epochs):
        model.train()
        for inputs, targets in session.batches(train_loader, phase="train", epoch=epoch):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(inputs), targets)
            loss.backward()
            optimizer.step()

        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, targets in session.batches(test_loader, phase="val", epoch=epoch):
                inputs, targets = inputs.to(device), targets.to(device)
                preds = model(inputs).argmax(dim=1)
                correct += int((preds == targets).sum().item())
                total += targets.size(0)
        print(f"epoch {epoch + 1}/{args.epochs} test_acc={correct / total:.4f}", flush=True)

    session.close()


if __name__ == "__main__":
    main()
