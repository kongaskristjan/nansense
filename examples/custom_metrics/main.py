"""Custom instruments demo: log your own metrics and tensors per layer.

A tiny CNN classifies which quadrant of a synthetic 16x16 image holds a
bright Gaussian blob (no download needed). The point of the example is the
NaNsense *instrument* wiring — user callbacks the session evaluates for every
watched layer, against the live activations/gradients/weights:

- ``@session.watch_metric``: scalars plotted in `/stats` -> GRAPHS, either
  every batch (``sparsity``) or reduced to one point per epoch
  (``grad_rms``).
- ``@session.watch_layer_tensor``: an activation-shaped tensor (``zscore``)
  rendered as an extra strip on the layer's card on the main page.
- ``@session.watch_weight_tensor``: a weight-shaped tensor (``adam_dir``,
  Adam's effective update direction) rendered on `/weights` next to the
  weight/gradient/optimizer strips.

    uv run examples/custom_metrics/main.py --nansense-port 8080

Two layers are watched from the script so every instrument shows data
immediately; watch more by clicking nodes in the architecture diagram.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, TensorDataset

import nansense
from examples.common import (
    enable_line_buffering,
    evaluate,
    select_device,
    train_one_epoch,
)


class BlobNet(nn.Module):
    """Two conv blocks + linear head over 16x16 single-channel images."""

    def __init__(self, channels: int = 8) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, channels, 3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels * 2, 3, padding=1)
        self.head = nn.Linear(channels * 2, 4)

    def forward(self, x: Tensor) -> Tensor:
        x = torch.relu(self.conv1(x))
        x = torch.max_pool2d(x, 2)
        x = torch.relu(self.conv2(x))
        x = x.mean(dim=(2, 3))
        return self.head(x)


def make_blob_dataset(
    size: int, *, image_size: int = 16, seed: int = 0
) -> TensorDataset:
    """`size` images of one Gaussian blob each; the label is its quadrant.

    Deterministic per seed and materialised up front — the dataset is tiny,
    so an in-memory `TensorDataset` keeps the example free of downloads.
    """
    generator = torch.Generator().manual_seed(seed)
    labels = torch.randint(0, 4, (size,), generator=generator)
    half = image_size // 2
    # Blob centres: a random point inside the labelled quadrant.
    offsets = torch.rand(size, 2, generator=generator) * (half - 2) + 1
    centers = offsets + torch.stack(
        [(labels // 2) * half, (labels % 2) * half], dim=1
    )
    coords = torch.arange(image_size, dtype=torch.float32)
    yy, xx = torch.meshgrid(coords, coords, indexing="ij")
    dist_sq = (yy - centers[:, 0, None, None]) ** 2 + (
        xx - centers[:, 1, None, None]
    ) ** 2
    images = torch.exp(-dist_sq / 4.0).unsqueeze(1)
    images += torch.randn(images.shape, generator=generator) * 0.05
    return TensorDataset(images, labels)


def register_instruments(
    session: nansense.Session,
) -> None:
    """The pedagogical core: three instrument kinds on one session."""

    @session.watch_metric("sparsity")
    def sparsity(ctx: nansense.LayerContext) -> float:
        """Fraction of positive entries — one point per batch."""
        return float((ctx.activation > 0).float().mean())

    @session.watch_metric("grad_rms", on="epoch", reduce="mean")
    def grad_rms(ctx: nansense.LayerContext) -> float | None:
        """Gradient RMS, averaged into one point per epoch.

        Validation forwards run without gradients; returning None simply
        skips those batches.
        """
        if ctx.gradient is None:
            return None
        return float(ctx.gradient.square().mean().sqrt())

    @session.watch_layer_tensor("zscore")
    def zscore(ctx: nansense.LayerContext) -> Tensor:
        """The activation standardized over the batch — same shape, so it
        renders as an extra strip under the activation/gradient pair."""
        a = ctx.activation
        return (a - a.mean()) / (a.std() + 1e-6)

    @session.watch_weight_tensor("adam_dir")
    def adam_dir(ctx: nansense.WeightContext) -> Tensor | None:
        """Adam's effective update direction `m / (sqrt(v) + eps)`.

        The optimizer state is lazily initialised, so the strip appears
        after the first `optimizer.step()`.
        """
        state = ctx.optimizer_state
        if "exp_avg" not in state or "exp_avg_sq" not in state:
            return None
        m, v = state["exp_avg"], state["exp_avg_sq"]
        assert isinstance(m, Tensor) and isinstance(v, Tensor)
        return m / (v.sqrt() + 1e-8)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--channels", type=int, default=8)
    parser.add_argument("--train-size", type=int, default=4096)
    parser.add_argument("--val-size", type=int, default=512)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--device", type=str, default=None, help="cpu / cuda / mps; default auto"
    )
    parser.add_argument("--seed", type=int, default=0)
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
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path(".nansense_cache/custom_metrics"),
        help="Directory for time-travel epoch checkpoints.",
    )
    return parser.parse_args()


def run(args: argparse.Namespace, device: torch.device) -> None:
    train_loader = DataLoader(
        make_blob_dataset(args.train_size, seed=args.seed),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=True,
    )
    val_loader = DataLoader(
        make_blob_dataset(args.val_size, seed=args.seed + 1),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    model = BlobNet(channels=args.channels).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    session = nansense.start(
        model,
        enabled=not args.disable_nansense,
        optimizer=optimizer,
        port=args.nansense_port,
        phases={"train": len(train_loader), "val": len(val_loader)},
        input_mean=(0.0,),
        input_std=(1.0,),
    )
    register_instruments(session)
    # Watch the conv layers up front so the instruments have layers to run
    # on from batch 0 (instruments only cover watched layers).
    session.watch("conv1")
    session.watch("conv2")

    for epoch in session.epochs(args.epochs, cache_dir=args.cache_dir):
        with session.restore_point():
            epoch_start = time.time()
            train_stats = train_one_epoch(
                model, train_loader, optimizer, criterion, device, session=session
            )
            val_stats = evaluate(
                model, val_loader, criterion, device, session=session
            )
            print(
                f"epoch {epoch + 1:3d}/{args.epochs} "
                f"train_loss={train_stats.loss:.4f} train_acc={train_stats.accuracy:.4f} "
                f"val_loss={val_stats.loss:.4f} val_acc={val_stats.accuracy:.4f} "
                f"({time.time() - epoch_start:.1f}s)"
            )

    session.close()


def main() -> None:
    enable_line_buffering()
    args = parse_args()
    torch.manual_seed(args.seed)
    run(args, select_device(args.device))


if __name__ == "__main__":
    main()
