"""Hosted nansense playgrounds: locked, shared demos of trained networks.

Two playgrounds share this entrypoint — each is deployed as its own Hugging
Face Space (see deploy/README.md):

    mnist       LeNet-5 on MNIST, 20 epochs
    imagenette  the five-stage PreActResNet (resnet_deep) on Imagenette, 50 epochs

Two modes, connected by one frozen-moment file per playground:

    # Train once and freeze the final train batch (run locally, on a GPU):
    uv run --group cuda examples/playground/main.py --playground mnist --prepare --device cuda

    # Serve the demo (run at container start):
    uv run examples/playground/main.py --playground mnist --nansense-port 7860 --host 0.0.0.0

`--prepare` trains the full run with statistics collected for *every* layer
(`StatsScope.ALL`) and freezes the complete debugger moment — the last train
batch's snapshot plus all running statistics — to disk with
`Session.freeze_moment`. The run ends at the last training phase (the final
epoch skips validation), so the frozen batch is the run's last
gradient-carrying one.

Serving needs no dataset, optimizer, or training loop: `nansense.load_moment`
rebuilds the frozen pause around a fresh model in seconds, and the session
parks locked. Visitors show/hide layers per tab, browse stats, and run
experiments; stepping, time travel, the shared probe state
(pinning/perturbation), and the global settings are disabled (see
`Session.lock`).
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
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
from examples.standard.main import build_model
from nansense.ui.render import set_strip_format


@dataclass(frozen=True)
class PlaygroundSpec:
    """One hosted demo: a dataset, an architecture, and its training recipe.

    `model` is a `--model` name from examples/standard — the playground
    trains and serves exactly the network the standard example builds, so a
    visitor can reproduce the demo locally with the standard example.
    `shown_layers` only seeds the cards new tabs show first; under the locked
    `all` scope, statistics exist for every layer regardless.
    """

    dataset: str
    model: str
    epochs: int
    batch_size: int
    shown_layers: tuple[str, ...]
    lr: float = 1e-3
    weight_decay: float = 0.05

    @property
    def config(self) -> DatasetConfig:
        return DATASETS[self.dataset]

    def build(self) -> nn.Module:
        """A fresh demo model, identical in both modes: serving rebuilds it
        and `load_moment` restores the frozen weights into it, so the
        construction must match what `--prepare` froze."""
        return build_model(self.model, self.config)


PLAYGROUNDS: dict[str, PlaygroundSpec] = {
    "mnist": PlaygroundSpec(
        dataset="mnist",
        model="lenet",
        epochs=20,
        batch_size=64,
        # The two conv layers make the most visually interesting strips.
        shown_layers=("conv1", "conv2"),
    ),
    "imagenette": PlaygroundSpec(
        dataset="imagenette",
        model="resnet_deep",
        epochs=50,
        batch_size=32,
        shown_layers=("stem", "stage1.0.conv1"),
    ),
}


def default_moment_path(playground: str) -> Path:
    return Path(".nansense_cache/playground") / playground / "moment.pt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--playground",
        choices=sorted(PLAYGROUNDS),
        required=True,
        help="Which hosted demo to prepare or serve.",
    )
    parser.add_argument(
        "--prepare",
        action="store_true",
        help="Train and freeze the demo moment instead of serving it.",
    )
    parser.add_argument("--data-dir", type=Path, default=Path("./data"))
    parser.add_argument(
        "--moment",
        type=Path,
        default=None,
        help=(
            "Frozen-moment file written by --prepare and served otherwise "
            "(default .nansense_cache/playground/<playground>/moment.pt)."
        ),
    )
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


def train_and_freeze(
    spec: PlaygroundSpec,
    *,
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    device: torch.device,
    moment_path: Path,
) -> None:
    """The --prepare run: train fully, freeze the last train batch's moment.

    Stats collect for every layer from epoch 0 (scope `all`), so the frozen
    GRAPHS/HISTOGRAM/MIN-MAX views cover the whole run for the whole model;
    the watched seed layers only pick which cards new tabs show first. The
    final epoch skips validation — the moment freezes the run's last
    gradient-carrying batch, and nothing runs after it.
    """
    epochs = spec.epochs
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=spec.lr, weight_decay=spec.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    session = nansense.start(
        model,
        optimizer=optimizer,
        scheduler=scheduler,
        epochs=epochs,
        phases={"train": len(train_loader), "val": len(val_loader)},
    )
    for layer in spec.shown_layers:
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
    spec = PLAYGROUNDS[args.playground]
    moment_path = args.moment or default_moment_path(args.playground)
    device = select_device(args.device)
    model = spec.build().to(device)

    if args.prepare:
        print(f"Preparing the {args.playground} moment at {moment_path} ({device})")
        train_loader, val_loader = build_dataloaders(
            spec.config,
            data_dir=args.data_dir,
            batch_size=spec.batch_size,
            num_workers=args.num_workers,
        )
        train_and_freeze(
            spec,
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            moment_path=moment_path,
        )
        return

    session = open_showcase(
        model,
        moment_path,
        config=spec.config,
        port=args.nansense_port,
        host=args.host,
    )
    print(f"Serving the frozen {args.playground} moment; parking locked.")
    session.park()  # serves experiment/probe requests until the process ends


if __name__ == "__main__":
    main()
