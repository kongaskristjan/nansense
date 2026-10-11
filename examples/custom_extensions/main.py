"""Add custom experiments and tools to NaNsense.

A small CNN learns which quadrant of a synthetic image holds a blob, so no
download is required (the model and data are the custom-metrics example's).
The script registers ``ablate_channels``: an experiment that zeroes each
channel of the chosen layer in turn and lists how much the paused batch's
accuracy drops. Open a layer's Experiment page and pick "Channel ablation",
or run it from a coding agent with ``run_experiment``. The ``confusion`` tool
reports validation accuracy and a confusion matrix for up to ``batches``
batches. Pause training, then call it with
``nansense-cli --port 8080 confusion --batches 4`` or use the UI's Tools page.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Annotated

import pydantic
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader

import nansense
from examples.common import (
    enable_line_buffering,
    evaluate,
    select_device,
    train_one_epoch,
)
from examples.custom_metrics.main import BlobNet, make_blob_dataset


QUADRANTS = ("top left", "top right", "bottom left", "bottom right")


def _accuracy(logits: object, targets: Tensor) -> float:
    assert isinstance(logits, Tensor)
    return float((logits.argmax(dim=1) == targets).float().mean())


def register_extensions(session: nansense.Session, val_loader: DataLoader) -> None:
    """Register channel ablation and a validation confusion-matrix tool."""

    @session.experiment(
        "ablate_channels",
        label="Channel ablation",
        params=[
            nansense.ExperimentParam(
                "top",
                "Channels to list",
                "int",
                default=8,
                minimum=1,
                tooltip="Number of channels to list, ranked by accuracy drop on this batch",
            )
        ],
        locked_limits={"top": 32},
    )
    def ablate_channels(
        ctx: nansense.ExperimentContext,
    ) -> nansense.ExperimentOutput:
        """Accuracy drop when each channel of the layer is zeroed.

        Zeroes one channel of the selected layer at a time, reruns the captured
        inputs, and ranks channels by the drop in batch accuracy. A negative
        drop means accuracy improved when that channel was zeroed.
        """
        if not isinstance(ctx.batch_item, (tuple, list)):
            return nansense.ExperimentOutput(
                note="Needs a batch with targets; use session.batches or session.batch(item=...)."
            )
        targets = ctx.batch_item[1]
        assert isinstance(targets, Tensor)
        targets = targets.to(ctx.device)
        with torch.no_grad():
            baseline = _accuracy(ctx.forward(), targets)
            channels = ctx.activation(ctx.layer).shape[1]
            drops: list[tuple[int, float]] = []
            for channel in range(channels):
                if ctx.should_abort():
                    break

                def zero(name: str, act: Tensor, c: int = channel) -> Tensor | None:
                    if name != ctx.layer:
                        return None
                    act = act.clone()
                    act[:, c] = 0
                    return act

                ablated = _accuracy(ctx.forward(edit=zero), targets)
                drops.append((channel, baseline - ablated))
        drops.sort(key=lambda row: -row[1])
        top = ctx.params["top"]
        assert isinstance(top, int)
        return nansense.ExperimentOutput(
            table=nansense.Table(("channel", "accuracy drop"), drops[:top]),
            scalars={"baseline accuracy": baseline, "channels": channels},
        )

    @session.tool()
    def confusion(
        ctx: nansense.ToolContext,
        batches: Annotated[
            int, pydantic.Field(ge=1, description="Validation batches to evaluate")
        ] = 8,
    ) -> dict[str, object]:
        """Compute validation accuracy and a confusion matrix for up to `batches` batches.

        Include overall and per-quadrant accuracy. Matrix rows are true
        quadrants; columns are predicted quadrants.
        """
        matrix = torch.zeros(len(QUADRANTS), len(QUADRANTS), dtype=torch.long)
        with torch.no_grad():
            for i, (images, labels) in enumerate(val_loader):
                if i >= batches or ctx.should_abort():
                    break
                logits = ctx.model(images.to(ctx.device))
                assert isinstance(logits, Tensor)
                predicted = logits.argmax(dim=1).cpu()
                matrix.index_put_(
                    (labels, predicted), torch.ones_like(labels), accumulate=True
                )
        per_class = matrix.diag() / matrix.sum(dim=1).clamp(min=1)
        return {
            "accuracy": float(matrix.diag().sum() / matrix.sum().clamp(min=1)),
            "per_quadrant_accuracy": dict(zip(QUADRANTS, per_class.tolist())),
            "matrix": nansense.Table(
                ("true / predicted", *QUADRANTS),
                [(name, *row) for name, row in zip(QUADRANTS, matrix.tolist())],
            ),
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--channels", type=int, default=8)
    parser.add_argument("--train-size", type=int, default=4096)
    parser.add_argument("--val-size", type=int, default=512)
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
        help="Run the example without NaNsense.",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path(".nansense_cache/custom_extensions"),
        help="Directory for time-travel epoch checkpoints.",
    )
    return parser.parse_args()


def run(args: argparse.Namespace, device: torch.device) -> None:
    train_loader = DataLoader(
        make_blob_dataset(args.train_size, seed=args.seed),
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        make_blob_dataset(args.val_size, seed=args.seed + 1),
        batch_size=args.batch_size,
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
    register_extensions(session, val_loader)

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
