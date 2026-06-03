"""Train the small CIFAR ResNet on CIFAR10."""

from __future__ import annotations

import argparse
import io
import sys
import time
from pathlib import Path

import torch
from torch import nn

import playgrad
from examples.cifar10.data import CIFAR10_MEAN, CIFAR10_STD, build_dataloaders
from examples.cifar10.resnet import ResNetCIFAR
from examples.cifar10.train import evaluate, train_one_epoch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("./data"))
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--blocks-per-stage", type=int, default=3, help="ResNet-(6n+2) depth knob")
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", type=str, default=None, help="cpu / cuda / mps; default auto")
    parser.add_argument(
        "--bf16",
        action="store_true",
        help="Use torch.autocast with bfloat16 for forward/loss (no GradScaler needed)",
    )
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--playgrad-port",
        type=int,
        default=8080,
        help="Port for the playgrad UI (default 8080).",
    )
    parser.add_argument(
        "--disable-playgrad",
        action="store_true",
        help="Disable playgrad with near-zero overhead (run as plain training).",
    )
    return parser.parse_args()


def enable_line_buffering() -> None:
    """Flush stdout on every newline so progress prints appear immediately.

    Python block-buffers stdout when it is not a TTY (e.g. redirected to a
    file or pipe), which can hide progress output until the buffer fills or
    the process exits. Reconfiguring to line buffering restores TTY-like
    behaviour regardless of how the script is launched.
    """
    if isinstance(sys.stdout, io.TextIOWrapper):
        sys.stdout.reconfigure(line_buffering=True)


def select_device(name: str | None) -> torch.device:
    if name is not None:
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main() -> None:
    enable_line_buffering()
    args = parse_args()
    torch.manual_seed(args.seed)

    device = select_device(args.device)
    amp_dtype = torch.bfloat16 if args.bf16 else None
    print(f"Using device: {device} (amp_dtype={amp_dtype})")

    train_loader, test_loader = build_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    model = ResNetCIFAR(num_classes=10, blocks_per_stage=args.blocks_per_stage).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        nesterov=True,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Always create the session; `enabled=False` makes it a near-zero-overhead
    # no-op so the training loop below needs no playgrad-specific branching.
    session = playgrad.start(
        model,
        epochs=args.epochs,
        phases={"train": len(train_loader), "val": len(test_loader)},
        enabled=not args.disable_playgrad,
    )
    if session.enabled:
        playgrad.serve(
            session,
            port=args.playgrad_port,
            input_mean=CIFAR10_MEAN,
            input_std=CIFAR10_STD,
        )
        print(f"playgrad UI at http://127.0.0.1:{args.playgrad_port}")

    best_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()
        train_stats = train_one_epoch(
            model, train_loader, optimizer, criterion, device,
            amp_dtype=amp_dtype, session=session, epoch=epoch - 1,
        )
        test_stats = evaluate(
            model, test_loader, criterion, device,
            amp_dtype=amp_dtype, session=session, epoch=epoch - 1,
        )
        scheduler.step()

        elapsed = time.time() - epoch_start
        print(
            f"epoch {epoch:3d}/{args.epochs} "
            f"train_loss={train_stats.loss:.4f} train_acc={train_stats.accuracy:.4f} "
            f"test_loss={test_stats.loss:.4f} test_acc={test_stats.accuracy:.4f} "
            f"lr={scheduler.get_last_lr()[0]:.4f} ({elapsed:.1f}s)"
        )

        if test_stats.accuracy > best_acc:
            best_acc = test_stats.accuracy
            if args.checkpoint is not None:
                args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
                torch.save(
                    {"model": model.state_dict(), "epoch": epoch, "test_acc": best_acc},
                    args.checkpoint,
                )

    print(f"Best test accuracy: {best_acc:.4f}")

    session.close()


if __name__ == "__main__":
    main()
