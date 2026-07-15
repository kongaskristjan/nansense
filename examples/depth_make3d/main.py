"""Monocular depth estimation on Make3D via transfer learning.

Predict a per-pixel depth map from a single RGB photo with a pretrained ResNet
encoder and a small U-Net decoder, under the full NaNsense wiring (scheduler,
time travel, checkpoints):

    uv run examples/depth_make3d/main.py --nansense-port 8080

The first run downloads the Make3D archives (~0.9 GB; the host can be slow) and
the ImageNet encoder weights (~45 MB). Pass `--freeze-encoder` to train only the
decoder. NaNsense shows a *dense, non-classification* model end to end: the
ImageNet encoder's early activations stay structured (edges, textures), and the
predicted log-depth renders as an image strip alongside the input photo.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
from torch import nn

import nansense
from examples.common import (
    add_dtype_arg,
    amp_dtype_from_name,
    enable_line_buffering,
    evaluate,
    select_device,
    train_one_epoch,
)
from examples.depth_make3d.data import DatasetConfig, build_dataloaders
from examples.depth_make3d.losses import ScaleInvariantLogLoss, delta_accuracy
from examples.depth_make3d.model import build_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--backbone",
        choices=["resnet18", "resnet34"],
        default="resnet18",
        help="Pretrained encoder backbone (default resnet18).",
    )
    parser.add_argument("--data-dir", type=Path, default=Path("./data"))
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help=(
            "Batch size (default: 24 for resnet18, 16 for resnet34 — kept "
            "modest for low GPU memory)."
        ),
    )
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", type=str, default=None, help="cpu / cuda / mps; default auto")
    add_dtype_arg(parser)
    parser.add_argument(
        "--freeze-encoder",
        action="store_true",
        help="Freeze the pretrained encoder and train only the decoder.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path(".nansense_cache/depth_make3d"),
        help="Directory for time-travel epoch checkpoints (default .nansense_cache/depth_make3d).",
    )
    parser.add_argument(
        "--nansense-port",
        type=int,
        default=8080,
        help="Port for the NaNsense UI (default 8080).",
    )
    parser.add_argument(
        "--disable-nansense",
        action="store_true",
        help="Disable NaNsense with near-zero overhead (run as plain training).",
    )
    return parser.parse_args()


def default_batch_size(backbone: str) -> int:
    """Batch size kept modest for low GPU memory on the 192x256 inputs: the
    deeper resnet34 encoder uses a smaller batch (16) than resnet18 (24)."""
    return 16 if backbone == "resnet34" else 24


def build_optimizer_and_scheduler(
    model: nn.Module, args: argparse.Namespace
) -> tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LRScheduler]:
    # Only optimise parameters that require grad, so --freeze-encoder really
    # excludes the frozen encoder from AdamW's state.
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    return optimizer, scheduler


def run_single(args: argparse.Namespace, config: DatasetConfig, device: torch.device) -> None:
    """Single-process training with the full NaNsense wiring and time travel."""
    amp_dtype = amp_dtype_from_name(args.dtype)
    print(f"Using device: {device} (dtype={args.dtype})")

    train_loader, test_loader = build_dataloaders(
        config,
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    model = build_model(
        backbone=args.backbone, pretrained=True, freeze_encoder=args.freeze_encoder
    ).to(device)
    criterion = ScaleInvariantLogLoss()
    optimizer, scheduler = build_optimizer_and_scheduler(model, args)

    # Always create the session; `enabled=False` makes it a near-zero-overhead
    # no-op so the training loop below needs no NaNsense-specific branching.
    # `port=` serves the UI immediately (skipped automatically when disabled).
    session = nansense.start(
        model,
        enabled=not args.disable_nansense,
        # Optional: lets the weights page show per-parameter optimizer state
        # (Adam moments) and the group's live hyperparameters (lr, ...).
        optimizer=optimizer,
        # Optional: lets time-travel checkpoints restore the LR schedule.
        scheduler=scheduler,
        port=args.nansense_port,
        input_mean=config.mean,
        input_std=config.std,
    )

    # Opting into time travel: each epoch start is checkpointed to
    # `--cache-dir`, and a UI-requested jump unwinds to `with session.restore_point():`
    # and re-enters at the chosen epoch with the cached model / optimizer /
    # scheduler / RNG state restored.
    best_delta = 0.0
    for epoch in session.epochs(args.epochs, cache_dir=args.cache_dir):
        with session.restore_point():
            epoch_start = time.time()
            train_stats = train_one_epoch(
                model, train_loader, optimizer, criterion, device,
                amp_dtype=amp_dtype, session=session, metric_fn=delta_accuracy,
            )
            test_stats = evaluate(
                model, test_loader, criterion, device,
                amp_dtype=amp_dtype, session=session, metric_fn=delta_accuracy,
            )
            scheduler.step()

            elapsed = time.time() - epoch_start
            print(
                f"epoch {epoch + 1:3d}/{args.epochs} "
                f"train_loss={train_stats.loss:.4f} train_d1={train_stats.accuracy:.4f} "
                f"test_loss={test_stats.loss:.4f} test_d1={test_stats.accuracy:.4f} "
                f"lr={scheduler.get_last_lr()[0]:.4f} ({elapsed:.1f}s)"
            )

            best_delta = max(best_delta, test_stats.accuracy)

    print(f"Best test delta<1.25 accuracy: {best_delta:.4f}")

    session.close()


def main() -> None:
    enable_line_buffering()
    args = parse_args()
    torch.manual_seed(args.seed)

    config = DatasetConfig()
    if args.batch_size is None:
        args.batch_size = default_batch_size(args.backbone)
    run_single(args, config, select_device(args.device))


if __name__ == "__main__":
    main()
