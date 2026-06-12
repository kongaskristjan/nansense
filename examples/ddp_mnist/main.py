"""Train a small MLP on MNIST with DistributedDataParallel + nansense.

Launch with torchrun (one process per rank):

    uv run torchrun --nproc_per_node=2 -m examples.ddp_mnist.main --nansense-port 8080

The nansense wiring is identical to the single-process examples — every
rank calls `nansense.start` and wraps its batches — and the library sorts
out the roles: rank 0 serves the UI (open http://localhost:8080) and
drives pausing/stepping, the other ranks follow its pace and contribute
their data shard to the watch page's statistics, so histograms and the
stats table cover the *global* batch. Each rank trains on its own shard
via `DistributedSampler`.

Runs on CUDA with one GPU per rank (NCCL) when enough GPUs exist,
otherwise on CPU (gloo) — so a 2-rank run works on a single-GPU machine.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from torchvision import datasets, transforms

import nansense

MNIST_MEAN: tuple[float, ...] = (0.1307,)
MNIST_STD: tuple[float, ...] = (0.3081,)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("./data"))
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=256, help="Per-rank batch size.")
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument(
        "--nansense-port",
        type=int,
        default=8080,
        help="Port for the nansense UI, served by rank 0 (default 8080).",
    )
    return parser.parse_args()


def build_model() -> nn.Module:
    return nn.Sequential(
        nn.Flatten(), nn.Linear(28 * 28, 128), nn.ReLU(), nn.Linear(128, 10)
    )


def build_dataloaders(
    data_dir: Path, batch_size: int
) -> tuple[DataLoader, DataLoader, DistributedSampler]:
    """Per-rank loaders over `DistributedSampler` shards.

    Both phases are sharded, so every rank runs the same number of batches
    per phase — distributed nansense sessions (like DDP itself) require the
    ranks to advance through batches in lockstep.
    """
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize(MNIST_MEAN, MNIST_STD)]
    )
    train_set: Dataset = datasets.MNIST(
        str(data_dir), train=True, download=True, transform=transform
    )
    test_set: Dataset = datasets.MNIST(
        str(data_dir), train=False, download=True, transform=transform
    )
    train_sampler = DistributedSampler(train_set, shuffle=True)
    train_loader = DataLoader(train_set, batch_size=batch_size, sampler=train_sampler)
    test_loader = DataLoader(
        test_set, batch_size=batch_size, sampler=DistributedSampler(test_set, shuffle=False)
    )
    return train_loader, test_loader, train_sampler


def init_distributed() -> torch.device:
    """Join the process group and pick this rank's device.

    NCCL with one GPU per rank when the machine has enough GPUs for the
    world size; otherwise CPU with gloo, so multi-rank runs also work on
    single-GPU or CPU-only machines.
    """
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    use_cuda = torch.cuda.is_available() and torch.cuda.device_count() >= world_size
    dist.init_process_group("nccl" if use_cuda else "gloo")
    if use_cuda:
        torch.cuda.set_device(local_rank)
        return torch.device("cuda", local_rank)
    return torch.device("cpu")


def main() -> None:
    args = parse_args()
    device = init_distributed()
    rank = dist.get_rank()
    torch.manual_seed(0)

    train_loader, test_loader, train_sampler = build_dataloaders(
        args.data_dir, args.batch_size
    )
    model = DistributedDataParallel(build_model().to(device))
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)

    # Same call on every rank; nansense unwraps the DDP model itself and
    # serves the UI from rank 0 only.
    session = nansense.start(
        model,
        epochs=args.epochs,
        phases={"train": len(train_loader), "val": len(test_loader)},
        optimizer=optimizer,
        port=args.nansense_port,
        input_mean=MNIST_MEAN,
        input_std=MNIST_STD,
    )
    if rank == 0:
        print(f"nansense UI at http://127.0.0.1:{args.nansense_port}", flush=True)

    for epoch in range(args.epochs):
        train_sampler.set_epoch(epoch)
        model.train()
        for inputs, targets in session.batches(train_loader, phase="train", epoch=epoch):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(inputs), targets)
            loss.backward()
            optimizer.step()

        model.eval()
        hits = torch.zeros(2, device=device)  # correct, total
        with torch.no_grad():
            for inputs, targets in session.batches(test_loader, phase="val", epoch=epoch):
                inputs, targets = inputs.to(device), targets.to(device)
                preds = model(inputs).argmax(dim=1)
                hits += torch.stack(
                    [(preds == targets).sum(), torch.tensor(targets.size(0), device=device)]
                )
        dist.all_reduce(hits)
        if rank == 0:
            acc = float(hits[0]) / float(hits[1])
            print(f"epoch {epoch + 1}/{args.epochs} test_acc={acc:.4f}", flush=True)

    session.close()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
