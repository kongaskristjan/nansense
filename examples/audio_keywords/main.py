"""Spoken keyword classification from log-mel spectrograms (8 keywords).

Trains an ImageNet-style ResNet on Google's "mini Speech Commands" set (8
keywords: down, go, left, no, right, stop, up, yes) with the full NaNsense
wiring (scheduler, time travel, checkpoints). Each ~1 s 16 kHz clip is turned
into a `[1, n_mels, n_frames]` log-mel spectrogram in the dataset (via
torchaudio's `MelSpectrogram` front end) and fed to the ResNet as a
single-channel image:

    uv run examples/audio_keywords/main.py --nansense-port 8080

The first run downloads ~180 MB and extracts to `--data-dir`.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
from torch import nn

import nansense
from examples.audio_keywords.data import AudioConfig, build_dataloaders
from examples.audio_keywords.model import KeywordResNet
from examples.common import (
    add_dtype_arg,
    amp_dtype_from_name,
    enable_line_buffering,
    evaluate,
    select_device,
    train_one_epoch,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("./data"))
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size (default 64; the ResNet-18 peaks well under 4 GB at this size).",
    )
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", type=str, default=None, help="cpu / cuda / mps; default auto")
    add_dtype_arg(parser)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--n-mels", type=int, default=40, help="Mel bands in the log-mel front end (default 40)."
    )
    parser.add_argument(
        "--sample-rate", type=int, default=16_000, help="Audio sample rate (default 16000)."
    )
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path(".nansense_cache/audio_keywords"),
        help="Directory for time-travel epoch checkpoints (default .nansense_cache/audio_keywords).",
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


def build_model(config: AudioConfig) -> nn.Module:
    return KeywordResNet(num_classes=config.num_classes, in_channels=config.in_channels)


def build_optimizer_and_scheduler(
    model: nn.Module, args: argparse.Namespace
) -> tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LRScheduler]:
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    return optimizer, scheduler


def run_single(args: argparse.Namespace, config: AudioConfig, device: torch.device) -> None:
    """Single-process training with the full NaNsense wiring and time travel."""
    amp_dtype = amp_dtype_from_name(args.dtype)
    print(f"Using device: {device} (dtype={args.dtype})")

    train_loader, val_loader = build_dataloaders(
        config,
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    model = build_model(config).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer, scheduler = build_optimizer_and_scheduler(model, args)

    # Always create the session; `enabled=False` makes it a near-zero-overhead
    # no-op so the training loop below needs no NaNsense-specific branching.
    # `input_mean` / `input_std` are scalar log-mel stats passed as 1-tuples so
    # the single-channel spectrogram renders denormalized in the UI.
    session = nansense.start(
        model,
        enabled=not args.disable_nansense,
        optimizer=optimizer,
        scheduler=scheduler,
        port=args.nansense_port,
        phases={"train": len(train_loader), "val": len(val_loader)},
        input_mean=config.mean,
        input_std=config.std,
    )

    # Opting into time travel: each epoch start is checkpointed to
    # `--cache-dir`, and a UI-requested jump unwinds to `with session.restore_point():`
    # and re-enters at the chosen epoch with the cached model / optimizer /
    # scheduler / RNG state restored.
    best_acc = 0.0
    for epoch in session.epochs(args.epochs, cache_dir=args.cache_dir):
        with session.restore_point():
            epoch_start = time.time()
            train_stats = train_one_epoch(
                model, train_loader, optimizer, criterion, device,
                amp_dtype=amp_dtype, session=session,
            )
            val_stats = evaluate(
                model, val_loader, criterion, device,
                amp_dtype=amp_dtype, session=session,
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
                if args.checkpoint is not None:
                    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
                    torch.save(
                        {"model": model.state_dict(), "epoch": epoch + 1, "val_acc": best_acc},
                        args.checkpoint,
                    )

    print(f"Best val accuracy: {best_acc:.4f}")

    session.close()


def main() -> None:
    enable_line_buffering()
    args = parse_args()
    torch.manual_seed(args.seed)

    config = AudioConfig(n_mels=args.n_mels, sample_rate=args.sample_rate)
    run_single(args, config, select_device(args.device))


if __name__ == "__main__":
    main()
