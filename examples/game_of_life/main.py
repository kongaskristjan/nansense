"""Predict Conway's Game of Life K steps into the future, with full nansense wiring.

A fully-convolutional residual net learns the Game-of-Life rule from synthetic
random boards (no download): the input is a binary board ``[1, H, W]`` (a random
draw advanced one silent Game-of-Life step, so it looks less random) and the
target is that board advanced ``--steps`` (K) further steps under toroidal
boundaries.
Training minimises a per-cell ``BCEWithLogitsLoss`` and tracks per-cell accuracy.

    uv run examples/game_of_life/main.py --nansense-port 8080

The board-shaped input, target, activations and output make this a vivid
nansense demo: clicking to perturb a single cell shows its K-step influence
light-cone (the model's learned receptive field), deep-dream surfaces glider /
oscillator motifs, and time travel watches the rule itself being learned.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader

import nansense
from examples.common import (
    add_dtype_arg,
    amp_dtype_from_name,
    enable_line_buffering,
    evaluate,
    select_device,
    train_one_epoch,
)
from examples.game_of_life.life import GameOfLifeDataset
from examples.game_of_life.model import LifeNet


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("./data"))
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size (default 64, kept modest for low GPU memory).",
    )
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument(
        "--steps",
        type=int,
        default=2,
        help="K: how many Game-of-Life steps ahead to predict (default 2).",
    )
    parser.add_argument("--board-size", type=int, default=32)
    parser.add_argument(
        "--density",
        type=float,
        default=0.3,
        help="Probability each cell of a random board starts alive (default 0.3).",
    )
    parser.add_argument("--train-size", type=int, default=8192)
    parser.add_argument("--val-size", type=int, default=1024)
    parser.add_argument(
        "--channels", type=int, default=32, help="Feature width of the conv net (default 32)."
    )
    parser.add_argument(
        "--depth",
        type=int,
        default=None,
        help=(
            "Number of residual blocks. Default scales with --steps to cover the "
            "K-step light cone (each block has receptive radius 2)."
        ),
    )
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", type=str, default=None, help="cpu / cuda / mps; default auto")
    add_dtype_arg(parser)
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
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path(".nansense_cache/game_of_life"),
        help="Directory for time-travel epoch checkpoints (default .nansense_cache/game_of_life).",
    )
    return parser.parse_args()


def default_depth(steps: int) -> int:
    """Residual-block count that comfortably covers the K-step light cone.

    One Game-of-Life step needs receptive radius 1; K steps need radius K. Each
    residual block (two 3x3 convs) adds radius 2, so ``steps`` blocks already
    give radius 2K with margin. A floor of 4 keeps small-K models (including the
    default) deep enough to be an interesting net rather than a near-linear one.
    """
    return max(4, steps)


def build_model(channels: int = 32, depth: int | None = None, steps: int = 1) -> nn.Module:
    return LifeNet(channels=channels, depth=default_depth(steps) if depth is None else depth)


def build_dataloaders(args: argparse.Namespace) -> tuple[DataLoader, DataLoader]:
    """Train / val loaders over freshly sampled, deterministic GoL datasets.

    Train and val use different seeds so the val boards are unseen, while each
    split is itself fixed (seeded generator) for reproducible epochs and replay.
    """
    train_set = GameOfLifeDataset(
        size=args.train_size,
        board_size=args.board_size,
        steps=args.steps,
        density=args.density,
        seed=args.seed,
    )
    val_set = GameOfLifeDataset(
        size=args.val_size,
        board_size=args.board_size,
        steps=args.steps,
        density=args.density,
        seed=args.seed + 1,
    )
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    return train_loader, val_loader


def build_optimizer_and_scheduler(
    model: nn.Module, args: argparse.Namespace
) -> tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LRScheduler]:
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    return optimizer, scheduler


def per_cell_accuracy(output: Tensor, targets: Tensor) -> float:
    """Fraction of cells whose predicted state matches the true future board."""
    return ((output > 0) == (targets > 0.5)).float().mean().item()


def run(args: argparse.Namespace, device: torch.device) -> None:
    """Single-process training with the full nansense wiring and time travel."""
    amp_dtype = amp_dtype_from_name(args.dtype)
    print(f"Using device: {device} (dtype={args.dtype})")

    train_loader, val_loader = build_dataloaders(args)

    model = build_model(channels=args.channels, depth=args.depth, steps=args.steps).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer, scheduler = build_optimizer_and_scheduler(model, args)

    # Always create the session; `enabled=False` makes it a near-zero-overhead
    # no-op so the training loop below needs no nansense-specific branching.
    # The board is already in [0, 1] with a single channel, so the normalization
    # passed to nansense is the identity.
    session = nansense.start(
        model,
        epochs=args.epochs,
        phases={"train": len(train_loader), "val": len(val_loader)},
        enabled=not args.disable_nansense,
        optimizer=optimizer,
        scheduler=scheduler,
        port=args.nansense_port,
        input_mean=(0.0,),
        input_std=(1.0,),
    )

    # Opting into time travel: each epoch start is checkpointed to
    # `--cache-dir`, and a UI-requested jump unwinds to `with restorer:` and
    # re-enters the epoch loop at `restorer.start_epoch` with the cached
    # model / optimizer / scheduler / RNG state restored.
    restorer = session.training_restorer(cache_dir=args.cache_dir)
    best_acc = 0.0
    while restorer.pending():
        with restorer:
            best_acc = 0.0  # re-derived per attempt: a jump rewinds history
            for epoch in restorer.epochs():
                epoch_start = time.time()
                train_stats = train_one_epoch(
                    model, train_loader, optimizer, criterion, device,
                    amp_dtype=amp_dtype, session=session, epoch=epoch, metric_fn=per_cell_accuracy,
                )
                val_stats = evaluate(
                    model, val_loader, criterion, device,
                    amp_dtype=amp_dtype, session=session, epoch=epoch, metric_fn=per_cell_accuracy,
                )
                scheduler.step()

                elapsed = time.time() - epoch_start
                print(
                    f"epoch {epoch + 1:3d}/{args.epochs} "
                    f"train_loss={train_stats.loss:.4f} train_acc={train_stats.accuracy:.4f} "
                    f"val_loss={val_stats.loss:.4f} val_acc={val_stats.accuracy:.4f} "
                    f"lr={scheduler.get_last_lr()[0]:.4f} ({elapsed:.1f}s)"
                )

                if val_stats.accuracy > best_acc:
                    best_acc = val_stats.accuracy

    print(f"Best val per-cell accuracy: {best_acc:.4f}")

    session.close()


def main() -> None:
    enable_line_buffering()
    args = parse_args()
    torch.manual_seed(args.seed)
    run(args, select_device(args.device))


if __name__ == "__main__":
    main()
