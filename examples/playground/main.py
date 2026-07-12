"""Hosted nansense playground: a locked, shared MNIST + LeNet demo.

Two modes, connected by one frozen-moment file:

    # Train once and freeze the final train batch (run at image build):
    uv run examples/playground/main.py --prepare

    # Serve the demo (run at container start):
    uv run examples/playground/main.py --nansense-port 7860 --host 0.0.0.0

`--prepare` trains the full run with statistics collected for *every* layer
(`StatsScope.ALL`) and freezes the complete debugger moment — the last train
batch's snapshot plus all running statistics — to disk with
`Session.freeze_moment`. The run ends at the last training phase (the final
epoch skips validation), so the frozen batch is the run's last
gradient-carrying one.

Serving needs no dataset, optimizer, or training loop: `nansense.load_moment`
rebuilds the frozen pause around a fresh LeNet in seconds, and the session
parks locked. Visitors show/hide layers per tab, browse stats, and run
experiments; stepping, time travel, the shared probe state
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
# per-tab seed — stats exist for every layer regardless.
SHOWN_LAYERS: tuple[str, ...] = ("conv1", "conv2")

DEFAULT_MOMENT = Path(".nansense_cache/playground/moment.pt")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--prepare",
        action="store_true",
        help="Train and freeze the demo moment instead of serving it.",
    )
    parser.add_argument("--data-dir", type=Path, default=Path("./data"))
    parser.add_argument(
        "--moment",
        type=Path,
        default=DEFAULT_MOMENT,
        help="Frozen-moment file written by --prepare and served otherwise.",
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


def build_model(config: DatasetConfig) -> LeNet:
    """The demo model, identical in both modes: serving rebuilds it fresh and
    `load_moment` restores the frozen weights into it, so the construction
    must match what `--prepare` froze."""
    return LeNet(
        num_classes=config.num_classes,
        in_channels=config.in_channels,
        image_size=config.image_size,
    )


def train_and_freeze(
    *,
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    device: torch.device,
    epochs: int,
    moment_path: Path,
    lr: float = 1e-3,
    weight_decay: float = 0.05,
) -> None:
    """The --prepare run: train fully, freeze the last train batch's moment.

    Stats collect for every layer from epoch 0 (scope `all`), so the frozen
    GRAPHS/HISTOGRAM/MIN-MAX views cover the whole run for the whole model;
    the watched seed layers only pick which cards new tabs show first. The
    final epoch skips validation — the moment freezes the run's last
    gradient-carrying batch, and nothing runs after it.
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    session = nansense.start(
        model,
        optimizer=optimizer,
        scheduler=scheduler,
        epochs=epochs,
        phases={"train": len(train_loader), "val": len(val_loader)},
    )
    for layer in SHOWN_LAYERS:
        session.watch(layer)
    session.set_stats_scope("all")
    session.freeze_moment(
        moment_path,
        phase="train",
        epoch=epochs - 1,
        batch_idx=len(train_loader) - 1,
    )
    session.detach()  # plain training: the armed freeze publishes, no pauses
    for epoch in range(epochs):
        epoch_start = time.time()
        train_stats = train_one_epoch(
            model, train_loader, optimizer, criterion, device,
            session=session, epoch=epoch,
        )
        message = (
            f"epoch {epoch + 1:2d}/{epochs} "
            f"train_loss={train_stats.loss:.4f} "
            f"train_acc={train_stats.accuracy:.4f}"
        )
        if epoch < epochs - 1:
            val_stats = evaluate(
                model, val_loader, criterion, device, session=session, epoch=epoch
            )
            message += (
                f" val_loss={val_stats.loss:.4f} val_acc={val_stats.accuracy:.4f}"
            )
        scheduler.step()
        print(f"{message} ({time.time() - epoch_start:.1f}s)")
    session.close()


def open_showcase(
    model: nn.Module,
    moment_path: Path,
    *,
    config: DatasetConfig,
    port: int | None,
    host: str = "127.0.0.1",
) -> nansense.Session:
    """The frozen moment reloaded and locked, ready to `park()`.

    PNG strips (internet bytes beat encode speed), then `load_moment` — which
    also loads the frozen weights and buffers into `model`, so experiments run
    against the exact frozen network — then the one-way lock. Everything else
    a demo needs (watched seed layers, statistics, schedule) already lives in
    the moment file.
    """
    set_strip_format("PNG")
    session = nansense.load_moment(
        model,
        moment_path,
        port=port,
        host=host,
        open_browser=False,
        input_mean=config.mean,
        input_std=config.std,
    )
    session.lock()
    return session


def main() -> None:
    enable_line_buffering()
    args = parse_args()
    torch.manual_seed(args.seed)
    config = DATASETS["mnist"]
    device = select_device(args.device)
    model = build_model(config).to(device)

    if args.prepare:
        print(f"Preparing the playground moment at {args.moment} ({device})")
        train_loader, val_loader = build_dataloaders(
            config,
            data_dir=args.data_dir,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )
        train_and_freeze(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            epochs=args.epochs,
            moment_path=args.moment,
            lr=args.lr,
            weight_decay=args.weight_decay,
        )
        return

    session = open_showcase(
        model,
        args.moment,
        config=config,
        port=args.nansense_port,
        host=args.host,
    )
    print("Serving the frozen playground moment; parking locked.")
    session.park()  # serves experiment/probe requests until the process ends


if __name__ == "__main__":
    main()
