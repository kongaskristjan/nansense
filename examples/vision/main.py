"""Train a small ResNet or Vision Transformer on CIFAR10 or Imagenette."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
from torch import nn

import nansense
from examples.common import enable_line_buffering, evaluate, select_device, train_one_epoch
from examples.vision.data import DATASETS, DatasetConfig, build_dataloaders
from examples.vision.resnet import PreActResNet
from examples.vision.vit import SimpleViT


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        choices=sorted(DATASETS),
        default="cifar10",
        help="Dataset to train on (default cifar10).",
    )
    parser.add_argument(
        "--model",
        choices=["resnet", "resnet_deep", "vit"],
        default="resnet",
        help=(
            "Architecture: the small pre-activation ResNet, its five-stage "
            "variant, or the simple ViT (default resnet)."
        ),
    )
    parser.add_argument("--data-dir", type=Path, default=Path("./data"))
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument(
        "--blocks-per-stage",
        type=int,
        default=3,
        help="ResNet depth knob: 2 * stages * n + 2 layers (e.g. ResNet-20 at 3 stages)",
    )
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", type=str, default=None, help="cpu / cuda / mps; default auto")
    parser.add_argument(
        "--bf16",
        action="store_true",
        help="Use torch.autocast with bfloat16 for forward/loss (no GradScaler needed)",
    )
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("models/latest"),
        help="Directory for time-travel epoch checkpoints (default models/latest).",
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


def build_model(name: str, config: DatasetConfig, blocks_per_stage: int = 3) -> nn.Module:
    if name in ("resnet", "resnet_deep"):
        # resnet_deep adds two more downsampling stages (5 total, 256 channels).
        return PreActResNet(
            num_classes=config.num_classes,
            blocks_per_stage=blocks_per_stage,
            num_stages=5 if name == "resnet_deep" else 3,
        )
    # An 8x8 patch grid at either image size (32 -> patch 4, 128 -> patch 16).
    return SimpleViT(
        image_size=config.image_size,
        patch_size=config.image_size // 8,
        num_classes=config.num_classes,
    )


def main() -> None:
    enable_line_buffering()
    args = parse_args()
    torch.manual_seed(args.seed)

    device = select_device(args.device)
    amp_dtype = torch.bfloat16 if args.bf16 else None
    print(f"Using device: {device} (amp_dtype={amp_dtype})")

    config = DATASETS[args.dataset]
    train_loader, test_loader = build_dataloaders(
        config,
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    model = build_model(args.model, config, blocks_per_stage=args.blocks_per_stage).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Always create the session; `enabled=False` makes it a near-zero-overhead
    # no-op so the training loop below needs no nansense-specific branching.
    # `port=` serves the UI immediately (skipped automatically when disabled).
    session = nansense.start(
        model,
        epochs=args.epochs,
        phases={"train": len(train_loader), "val": len(test_loader)},
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
    if session.enabled:
        print(f"nansense UI at http://127.0.0.1:{args.nansense_port}")

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
                    amp_dtype=amp_dtype, session=session, epoch=epoch,
                )
                test_stats = evaluate(
                    model, test_loader, criterion, device,
                    amp_dtype=amp_dtype, session=session, epoch=epoch,
                )
                scheduler.step()

                elapsed = time.time() - epoch_start
                print(
                    f"epoch {epoch + 1:3d}/{args.epochs} "
                    f"train_loss={train_stats.loss:.4f} train_acc={train_stats.accuracy:.4f} "
                    f"test_loss={test_stats.loss:.4f} test_acc={test_stats.accuracy:.4f} "
                    f"lr={scheduler.get_last_lr()[0]:.4f} ({elapsed:.1f}s)"
                )

                if test_stats.accuracy > best_acc:
                    best_acc = test_stats.accuracy
                    if args.checkpoint is not None:
                        args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
                        torch.save(
                            {"model": model.state_dict(), "epoch": epoch + 1, "test_acc": best_acc},
                            args.checkpoint,
                        )

    print(f"Best test accuracy: {best_acc:.4f}")

    session.close()


if __name__ == "__main__":
    main()
