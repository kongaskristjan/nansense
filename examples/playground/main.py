"""Hosted nansense playground: a locked, shared MNIST + LeNet demo.

Two modes, sharing one time-travel cache directory:

    # Train once, writing the per-epoch checkpoints (run at image build):
    uv run examples/playground/main.py --prepare

    # Serve the demo (run at container start):
    uv run examples/playground/main.py --nansense-port 7860 --host 0.0.0.0

Serving resumes from the cache at the last epoch and replays it with stats
collected for *every* layer (`StatsScope.ALL`), so histograms, min/max
patches, and per-epoch graphs are populated when the first visitor arrives.
It then parks paused on the run's final train batch with the session locked:
training sits still forever while visitors show/hide layers per tab, browse
stats, and run experiments — stepping, time travel, the shared probe state
(pinning/perturbation), and the global settings are disabled (see
`Session.lock`).
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
from torch import nn

import nansense
from examples.common import (
    enable_line_buffering,
    evaluate,
    select_device,
    train_one_epoch,
)
from examples.standard.data import DATASETS, DatasetConfig, build_dataloaders
from examples.standard.lenet import LeNet
from nansense.ui.render import set_strip_format

# The cards new visitors see by default: the two conv layers make the most
# visually interesting strips. Under the locked `all` scope this is only the
# per-tab seed — stats are collected for every layer regardless.
SHOWN_LAYERS: tuple[str, ...] = ("conv1", "conv2")

DEFAULT_CACHE_DIR = Path(".nansense_cache/playground")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--prepare",
        action="store_true",
        help="Train and write the epoch cache instead of serving the demo.",
    )
    parser.add_argument("--data-dir", type=Path, default=Path("./data"))
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=DEFAULT_CACHE_DIR,
        help="Epoch-checkpoint directory shared by --prepare and serving.",
    )
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument(
        "--device", type=str, default=None, help="cpu / cuda / mps; default auto"
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--nansense-port",
        type=int,
        default=7860,
        help="Port the demo UI serves on (default 7860, the HF Spaces port).",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="Bind address; a hosted deployment wants 0.0.0.0.",
    )
    return parser.parse_args()


def build_training(
    config: DatasetConfig,
    *,
    epochs: int,
    lr: float = 1e-3,
    weight_decay: float = 0.05,
) -> tuple[
    LeNet, nn.Module, torch.optim.Optimizer, torch.optim.lr_scheduler.LRScheduler
]:
    """The demo's model/criterion/optimizer/scheduler, identical in both modes.

    Serving rebuilds them fresh and restores the cached state into them, so
    the construction must match what --prepare checkpointed.
    """
    model = LeNet(
        num_classes=config.num_classes,
        in_channels=config.in_channels,
        image_size=config.image_size,
    )
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    return model, criterion, optimizer, scheduler


def make_demo_session(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    *,
    train_batches: int,
    config: DatasetConfig,
    port: int | None,
    host: str = "127.0.0.1",
) -> nansense.Session:
    """A session armed for the parked demo: run-to-end, then locked.

    PNG strips (internet bytes beat encode speed), the train phase declared
    up-front (so the run's final batch is detectable from batch 0 and
    `step_run` parks exactly there), the default-shown layers watched as the
    per-tab seed, `step_run()` armed, and the lock applied last — order
    matters, a locked session refuses all three.
    """
    set_strip_format("PNG")
    session = nansense.start(
        model,
        optimizer=optimizer,
        scheduler=scheduler,
        phases={"train": train_batches},
        port=port,
        host=host,
        open_browser=False,
        input_mean=config.mean,
        input_std=config.std,
    )
    for layer in SHOWN_LAYERS:
        session.watch(layer)
    session.step_run()
    session.lock()
    return session


def train_epochs(
    session: nansense.Session,
    *,
    model: nn.Module,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader | None,
    device: torch.device,
    epochs: int,
    cache_dir: Path,
    start_epoch: int = 0,
) -> None:
    """The shared epoch loop; serving passes `val_loader=None`.

    In serve mode the loop never completes: the armed `step_run` pauses the
    run's last train batch and the locked session keeps it parked, serving
    probe and experiment requests from the pause loop indefinitely.
    """
    for epoch in session.epochs(epochs, cache_dir=cache_dir, start_epoch=start_epoch):
        with session.restore_point():
            epoch_start = time.time()
            train_stats = train_one_epoch(
                model, train_loader, optimizer, criterion, device, session=session
            )
            message = (
                f"epoch {epoch + 1:2d}/{epochs} "
                f"train_loss={train_stats.loss:.4f} "
                f"train_acc={train_stats.accuracy:.4f}"
            )
            if val_loader is not None:
                val_stats = evaluate(
                    model, val_loader, criterion, device, session=session
                )
                message += (
                    f" val_loss={val_stats.loss:.4f} "
                    f"val_acc={val_stats.accuracy:.4f}"
                )
            scheduler.step()
            print(f"{message} ({time.time() - epoch_start:.1f}s)")


def main() -> None:
    enable_line_buffering()
    args = parse_args()
    torch.manual_seed(args.seed)
    config = DATASETS["mnist"]
    device = select_device(args.device)
    train_loader, val_loader = build_dataloaders(
        config,
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    model, criterion, optimizer, scheduler = build_training(
        config, epochs=args.epochs, lr=args.lr, weight_decay=args.weight_decay
    )
    model = model.to(device)

    if args.prepare:
        print(f"Preparing the playground cache in {args.cache_dir} ({device})")
        session = nansense.start(model, optimizer=optimizer, scheduler=scheduler)
        session.detach()  # plain training: no pauses, just epoch checkpoints
        train_epochs(
            session,
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            epochs=args.epochs,
            cache_dir=args.cache_dir,
        )
        session.close()
        print(f"Playground cache ready: {args.cache_dir}")
        return

    session = make_demo_session(
        model,
        optimizer,
        scheduler,
        train_batches=len(train_loader),
        config=config,
        port=args.nansense_port,
        host=args.host,
    )
    print(
        f"Replaying epoch {args.epochs}/{args.epochs} to fill the statistics, "
        "then parking locked at its last batch."
    )
    train_epochs(
        session,
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        train_loader=train_loader,
        val_loader=None,
        device=device,
        epochs=args.epochs,
        cache_dir=args.cache_dir,
        start_epoch=args.epochs - 1,
    )
    # Only reached if the session is closed externally; the parked demo
    # normally lives inside the loop above until the process exits.
    session.close()


if __name__ == "__main__":
    main()
