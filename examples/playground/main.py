"""Hosted NaNsense playgrounds: locked, shared demos of trained networks.

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
    # Layers that collect extreme-input patch galleries (None = all layers).
    # uint8 payloads, average grids off by default, and the channel cap
    # below keep full-model galleries affordable, so no demo shortlists —
    # the knob remains for models that outgrow even that; every layer keeps
    # its histogram/graph statistics regardless.
    patch_layers: tuple[str, ...] | None = None
    # Extreme-patch samples kept per channel (None = session default).
    samples_per_channel: int | None = None
    # Channels tracked per layer by the per-channel histograms and patch
    # galleries (None = session default).
    channel_limit: int | None = None
    # Deep-dream form defaults served to visitors (None = the UI default).
    # Defaults only: visitors can still raise the values up to the locked
    # ceilings (300 steps / 8 channels, `experiments._LOCKED_PARAM_LIMITS`).
    dream_steps: int | None = None
    dream_channels: int | None = None

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
        # A single card keeps the first view focused; conv1 makes the most
        # visually interesting strips and the rest stay a diagram click away.
        shown_layers=("conv1",),
    ),
    "imagenette": PlaygroundSpec(
        dataset="imagenette",
        model="resnet_deep",
        epochs=50,
        # Also the frozen/replayed demo batch: 16 halves the serve-time
        # replay cost and the snapshot's RAM versus 32.
        batch_size=16,
        # A single card keeps the first view focused; the first residual
        # conv shows richer training dynamics than the stem.
        shown_layers=("stage1.0.conv1",),
        # Full-model galleries: uint8 payloads + average grids off by
        # default + the 8-channel cap keep the moment in the tens of MB.
        channel_limit=12,
        samples_per_channel=5,
        # 224x224 dreams through the deep resnet are the demo's costliest
        # request; halved steps and fewer channels keep the default run
        # snappy on the shared CPU host.
        dream_steps=150,
        dream_channels=4,
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
    if spec.samples_per_channel is not None:
        session.set_watch_performance(samples_per_channel=spec.samples_per_channel)
    if spec.channel_limit is not None:
        session.set_watch_performance(channel_limit=spec.channel_limit)
    if spec.patch_layers is not None:
        session.set_patch_layers(spec.patch_layers)
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
    spec: PlaygroundSpec,
    port: int | None,
    host: str = "127.0.0.1",
) -> nansense.Session:
    """The frozen moment reloaded and locked, ready to `park()`.

    PNG strips (internet bytes beat encode speed), then `load_moment` — which
    loads the frozen weights and buffers into `model` and replays the frozen
    batch through it (regenerating every activation and gradient the views
    show; the replay mirrors the prepare run's training step) — then the
    demo preferences armed ahead of the one-way lock: experiment re-runs
    wait for a manual Run (auto-run off — a shared queue shouldn't fill on
    parameter edits; a page's first experiment still starts on its own), the
    spec's cheaper deep-dream form defaults, if any, and the watched seed
    re-based to the spec's `shown_layers` — the moment froze the seed it was
    prepared with, so re-seeding here lets the default cards change without
    re-training. Everything else a demo needs (statistics, schedule) lives
    in the moment file.
    """
    config = spec.config
    set_strip_format("PNG")
    criterion = nn.CrossEntropyLoss()
    session = nansense.load_moment(
        model,
        moment_path,
        replay=lambda m, batch: criterion(m(batch[0]), batch[1]),
        port=port,
        host=host,
        open_browser=False,
        input_mean=config.mean,
        input_std=config.std,
    )
    session.set_auto_run_experiments(False)
    if spec.dream_steps is not None:
        session.set_experiment_defaults(steps=spec.dream_steps)
    if spec.dream_channels is not None:
        session.set_experiment_defaults(channels=spec.dream_channels)
    # Scope is `all` (restored from the moment), so the watched set only
    # picks which cards new tabs show first — never what collects stats.
    for name in session.watched_layers - set(spec.shown_layers):
        session.unwatch(name)
    for name in spec.shown_layers:
        session.watch(name)
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
        spec=spec,
        port=args.nansense_port,
        host=args.host,
    )
    print(f"Serving the frozen {args.playground} moment; parking locked.")
    session.park()  # serves experiment/probe requests until the process ends


if __name__ == "__main__":
    main()
