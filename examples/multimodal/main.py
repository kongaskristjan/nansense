"""Multimodal CIFAR-10 (image + tabular features), with full NaNsense wiring.

A two-input network classifies CIFAR-10 from a 5-channel image (normalized RGB
plus a luma and a Sobel edge channel) *and* a flat 6-feature stats vector
derived from the same image. It exists to exercise the NaNsense input pane on
inputs that aren't a plain RGB image:

    uv run examples/multimodal/main.py --nansense-port 8080

In the input pane you get a dropdown to switch between the two model inputs:

- `image` has 5 channels, so it can't be shown directly — the `input_transform`
  below maps it back to displayable RGB (drop it to see the channel-count hint
  instead). "Click to perturb" then edits all five channel values of a pixel.
- `stats` is a flat `[6]` vector, shown as a per-feature colormapped strip with
  a scale legend; clicking a cell perturbs that single feature.

Either way, pinning / perturbing re-runs the whole two-input model, so the
activation strips and the perturbation diff propagate through both branches.
"""

from __future__ import annotations

import argparse
import contextlib
import time
from pathlib import Path

import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader

import nansense
from examples.common import (
    add_dtype_arg,
    amp_dtype_from_name,
    autocast,
    enable_line_buffering,
    select_device,
)
from examples.multimodal.data import RGB_MEAN, RGB_STD, build_dataloaders
from examples.multimodal.model import MultiModalNet

# RGB de-normalization constants for the display transform (shaped to broadcast
# over a `[B, 3, H, W]` slice). Built once; the transform runs on every frame.
_DISPLAY_MEAN: Tensor = torch.tensor(RGB_MEAN).view(1, 3, 1, 1)
_DISPLAY_STD: Tensor = torch.tensor(RGB_STD).view(1, 3, 1, 1)


def display_rgb(image: Tensor) -> Tensor:
    """Map the 5-channel `image` input back to a displayable RGB `[B, 3, H, W]`.

    The first three channels are the normalized RGB; undo the normalization and
    clamp to `[0, 1]`. The luma / edge channels are dropped — they only feed the
    model. Passed to `nansense.start(input_transform={"image": display_rgb})`.
    """
    return (image[:, :3] * _DISPLAY_STD + _DISPLAY_MEAN).clamp(0.0, 1.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("./data"))
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--width", type=int, default=32, help="Conv/MLP width (default 32).")
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", type=str, default=None, help="cpu / cuda / mps; default auto")
    add_dtype_arg(parser)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--nansense-port", type=int, default=8080, help="Port for the NaNsense UI (default 8080)."
    )
    parser.add_argument(
        "--disable-nansense",
        action="store_true",
        help="Disable NaNsense with near-zero overhead (run as plain training).",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path(".nansense_cache/multimodal"),
        help="Directory for time-travel epoch checkpoints (default .nansense_cache/multimodal).",
    )
    return parser.parse_args()


def run_epoch(
    model: MultiModalNet,
    loader: DataLoader,
    *,
    train: bool,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    amp_dtype: torch.dtype | None,
    session: nansense.Session,
) -> tuple[float, float]:
    """One epoch over the two-input loader; returns `(mean_loss, mean_acc)`.

    A local loop rather than `examples.common.train_one_epoch`, since that
    helper feeds a single tensor — here each batch is `((image, stats), label)`
    and the model takes both. The body still runs inside `session.batches` so
    NaNsense captures it exactly like a single-input loop.
    """
    model.train(train)
    phase = "train" if train else "val"
    grad_ctx = contextlib.nullcontext() if train else torch.no_grad()
    total_loss = 0.0
    total_acc = 0.0
    n_batches = 0
    with grad_ctx:
        for (image, stats), target in session.batches(loader, phase=phase):
            image = image.to(device, non_blocking=True)
            stats = stats.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            with autocast(device, amp_dtype):
                logits = model(image, stats)
                loss = criterion(logits, target)
            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
            total_loss += loss.item()
            total_acc += (logits.argmax(dim=1) == target).float().mean().item()
            n_batches += 1
    return total_loss / n_batches, total_acc / n_batches


def run(args: argparse.Namespace, device: torch.device) -> None:
    amp_dtype = amp_dtype_from_name(args.dtype)
    print(f"Using device: {device} (dtype={args.dtype})")

    train_loader, val_loader = build_dataloaders(
        args.data_dir, batch_size=args.batch_size, num_workers=args.num_workers
    )
    model = MultiModalNet(width=args.width).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # The `image` input has 5 channels, so display it via `display_rgb`; `stats`
    # is flat and needs no transform (it renders as a strip). Per-input config is
    # keyed by input name — a single value would apply to every input instead.
    session = nansense.start(
        model,
        enabled=not args.disable_nansense,
        optimizer=optimizer,
        scheduler=scheduler,
        port=args.nansense_port,
        input_transform={"image": display_rgb},
    )

    best_acc = 0.0
    for epoch in session.epochs(args.epochs, cache_dir=args.cache_dir):
        with session.restore_point():
            epoch_start = time.time()
            train_loss, train_acc = run_epoch(
                model, train_loader, train=True, criterion=criterion,
                optimizer=optimizer, device=device, amp_dtype=amp_dtype, session=session,
            )
            val_loss, val_acc = run_epoch(
                model, val_loader, train=False, criterion=criterion,
                optimizer=optimizer, device=device, amp_dtype=amp_dtype, session=session,
            )
            scheduler.step()

            elapsed = time.time() - epoch_start
            print(
                f"epoch {epoch + 1:3d}/{args.epochs} "
                f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
                f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} "
                f"lr={scheduler.get_last_lr()[0]:.4f} ({elapsed:.1f}s)"
            )
            best_acc = max(best_acc, val_acc)

    print(f"Best val accuracy: {best_acc:.4f}")
    session.close()


def main() -> None:
    enable_line_buffering()
    args = parse_args()
    torch.manual_seed(args.seed)
    run(args, select_device(args.device))


if __name__ == "__main__":
    main()
