"""Train a small ResNet, Vision Transformer, or LeNet on MNIST, CIFAR10, or Imagenette.

Single-process by default, with the full nansense wiring (scheduler, time
travel, checkpoints):

    uv run examples/standard/main.py --nansense-port 8080

Pass `--distributed` and launch under torchrun for multi-rank
DistributedDataParallel training — one process per rank. nansense's wiring is
identical (every rank calls `nansense.start`); rank 0 serves the UI and drives
pausing/stepping while the other ranks follow and fold their data shard into the
watch-page statistics. Time travel works here too: every rank wraps its epoch
loop in a restorer, and a jump rewinds all ranks in lockstep.

    uv run torchrun --nproc_per_node=2 examples/standard/main.py --distributed --nansense-port 8080
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel

import nansense
from examples.common import (
    add_dtype_arg,
    amp_dtype_from_name,
    enable_line_buffering,
    evaluate,
    select_device,
    train_one_epoch,
)
from examples.standard.data import (
    DATASETS,
    PADDING_MODES,
    DatasetConfig,
    build_dataloaders,
    build_distributed_dataloaders,
    ensure_downloaded,
)
from examples.standard.lenet import LeNet
from examples.standard.losses import LOSSES, build_criterion
from examples.standard.resnet import PreActResNet
from examples.standard.vit import SimpleViT


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        choices=sorted(DATASETS),
        default="cifar10",
        help="Dataset to train on (default cifar10).",
    )
    parser.add_argument(
        "--padding",
        choices=sorted(PADDING_MODES),
        default="zero",
        help=(
            "Crop-augmentation padding mode for mnist/cifar10 (default zero). "
            "Zero padding leaks hard border seams into the hidden activations — "
            "see for yourself in the nansense layer views, then try reflection."
        ),
    )
    parser.add_argument(
        "--model",
        choices=["resnet", "resnet_deep", "vit", "lenet"],
        default="resnet",
        help=(
            "Architecture: the small pre-activation ResNet, its five-stage "
            "variant, the simple ViT, or LeNet-5 (default resnet)."
        ),
    )
    parser.add_argument(
        "--loss",
        choices=list(LOSSES),
        default="cross_entropy",
        help=(
            "Training loss (default cross_entropy). mse/mae/mae_30 instead "
            "regress softmax probabilities onto the one-hot label; mae_30 is a "
            "balanced (asymmetric) absolute error penalising under-prediction "
            "more — compare their effect in the nansense views."
        ),
    )
    parser.add_argument("--data-dir", type=Path, default=Path("./data"))
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help=(
            "Batch size (default: 64 for cifar10/mnist, 32 for imagenette's "
            "larger 128x128 inputs — kept modest for low GPU memory)."
        ),
    )
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
    add_dtype_arg(parser)
    parser.add_argument(
        "--distributed",
        action="store_true",
        help=(
            "Train with DistributedDataParallel; launch under torchrun "
            "(one process per rank). Time travel is supported."
        ),
    )
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help=(
            "Directory for time-travel epoch checkpoints "
            "(default .nansense_cache/standard/<model>)."
        ),
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


def default_batch_size(dataset: str) -> int:
    """Batch size kept modest for low GPU memory. Imagenette's 128x128 images
    cost ~16x the activation memory of the 32x32 cifar10 / mnist crops, so it
    drops to 32 where the smaller datasets use 64."""
    return 32 if dataset == "imagenette" else 64


def build_model(name: str, config: DatasetConfig, blocks_per_stage: int = 3) -> nn.Module:
    if name in ("resnet", "resnet_deep"):
        # resnet_deep adds two more downsampling stages (5 total, 256 channels).
        return PreActResNet(
            num_classes=config.num_classes,
            blocks_per_stage=blocks_per_stage,
            num_stages=5 if name == "resnet_deep" else 3,
            in_channels=config.in_channels,
        )
    if name == "lenet":
        return LeNet(
            num_classes=config.num_classes,
            in_channels=config.in_channels,
            image_size=config.image_size,
        )
    # An 8x8 patch grid at either image size (32 -> patch 4, 128 -> patch 16).
    return SimpleViT(
        image_size=config.image_size,
        patch_size=config.image_size // 8,
        num_classes=config.num_classes,
        in_channels=config.in_channels,
    )


def build_optimizer_and_scheduler(
    model: nn.Module, args: argparse.Namespace
) -> tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LRScheduler]:
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    return optimizer, scheduler


def run_single(args: argparse.Namespace, config: DatasetConfig, device: torch.device) -> None:
    """Single-process training with the full nansense wiring and time travel."""
    amp_dtype = amp_dtype_from_name(args.dtype)
    print(f"Using device: {device} (dtype={args.dtype})")

    train_loader, test_loader = build_dataloaders(
        config,
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        padding=args.padding,
    )

    model = build_model(args.model, config, blocks_per_stage=args.blocks_per_stage).to(device)
    criterion = build_criterion(args.loss, config.num_classes)
    optimizer, scheduler = build_optimizer_and_scheduler(model, args)

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


def init_distributed() -> tuple[torch.device, int, int]:
    """Join the process group; return (device, rank, world_size).

    NCCL with one GPU per rank when the machine has enough GPUs for the world
    size, otherwise CPU/gloo — so multi-rank runs also work on single-GPU or
    CPU-only machines.
    """
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    use_cuda = torch.cuda.is_available() and torch.cuda.device_count() >= world_size
    dist.init_process_group("nccl" if use_cuda else "gloo")
    if use_cuda:
        torch.cuda.set_device(local_rank)
        return torch.device("cuda", local_rank), dist.get_rank(), world_size
    return torch.device("cpu"), dist.get_rank(), world_size


@torch.no_grad()
def distributed_test_accuracy(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    *,
    session: nansense.Session,
    epoch: int,
) -> float:
    """Global test accuracy across ranks, with the val phase wrapped by
    `session.batches` so nansense captures it like the single-process path."""
    model.eval()
    hits = torch.zeros(2, device=device)  # correct, total
    for inputs, targets in session.batches(loader, phase="val", epoch=epoch):
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        preds = model(inputs).argmax(dim=1)
        hits += torch.stack(
            [(preds == targets).sum(), torch.tensor(targets.size(0), device=device)]
        )
    dist.all_reduce(hits)
    return float(hits[0]) / float(hits[1])


def run_distributed(args: argparse.Namespace, config: DatasetConfig) -> None:
    """Multi-rank DistributedDataParallel training with time travel.

    Every rank wraps its epoch loop in a restorer exactly like `run_single`:
    a UI-requested jump on rank 0 is broadcast to all ranks at the next
    batch-start barrier, where every rank raises `TimeTravelJump` in lockstep
    and restores from its own per-rank epoch checkpoint (model/optimizer/
    scheduler are replicated by DDP, RNG is per-rank). `set_epoch` reseeds the
    sampler shards deterministically, so a replayed epoch sees the same shards.
    """
    device, rank, world_size = init_distributed()
    amp_dtype = amp_dtype_from_name(args.dtype)
    if rank == 0:
        print(f"Distributed run: world_size={world_size}, device={device} (dtype={args.dtype})")

    # Avoid a concurrent first-run download race: rank 0 fetches, others wait.
    if rank == 0:
        ensure_downloaded(config, args.data_dir)
    dist.barrier()
    train_loader, test_loader, train_sampler = build_distributed_dataloaders(
        config,
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        download=False,
        padding=args.padding,
    )

    # nansense unwraps the DDP model itself and serves the UI from rank 0 only.
    model = DistributedDataParallel(
        build_model(args.model, config, blocks_per_stage=args.blocks_per_stage).to(device)
    )
    criterion = build_criterion(args.loss, config.num_classes)
    optimizer, scheduler = build_optimizer_and_scheduler(model, args)

    session = nansense.start(
        model,
        epochs=args.epochs,
        phases={"train": len(train_loader), "val": len(test_loader)},
        enabled=not args.disable_nansense,
        optimizer=optimizer,
        scheduler=scheduler,
        port=args.nansense_port,
        input_mean=config.mean,
        input_std=config.std,
    )

    # Opting into time travel under DDP: every rank wraps the same epoch loop
    # in a restorer. Each rank checkpoints its own state per epoch (rank 0 to
    # `epoch_<n>.pt`, followers to `epoch_<n>.rank<r>.pt`); a jump rewinds them
    # all in lockstep. `set_epoch(epoch)` is called inside the loop so a
    # replayed epoch reshuffles its shards identically.
    restorer = session.training_restorer(cache_dir=args.cache_dir)
    while restorer.pending():
        with restorer:
            for epoch in restorer.epochs():
                train_sampler.set_epoch(epoch)  # reshuffle shards each epoch
                epoch_start = time.time()
                train_stats = train_one_epoch(
                    model, train_loader, optimizer, criterion, device,
                    amp_dtype=amp_dtype, session=session, epoch=epoch,
                )
                test_acc = distributed_test_accuracy(
                    model, test_loader, device, session=session, epoch=epoch
                )
                scheduler.step()
                if rank == 0:
                    elapsed = time.time() - epoch_start
                    print(
                        f"epoch {epoch + 1:3d}/{args.epochs} "
                        f"train_loss={train_stats.loss:.4f} train_acc={train_stats.accuracy:.4f} "
                        f"test_acc={test_acc:.4f} lr={scheduler.get_last_lr()[0]:.4f} ({elapsed:.1f}s)"
                    )

    session.close()
    dist.destroy_process_group()


def main() -> None:
    enable_line_buffering()
    args = parse_args()
    torch.manual_seed(args.seed)

    config = DATASETS[args.dataset]
    if args.batch_size is None:
        args.batch_size = default_batch_size(args.dataset)
    if args.cache_dir is None:
        # The architecture is selectable here, so the cache is namespaced per
        # model — jumping timelines after switching --model never reloads a
        # checkpoint written for a differently shaped network.
        args.cache_dir = Path(".nansense_cache/standard") / args.model
    if args.distributed:
        run_distributed(args, config)
    else:
        run_single(args, config, select_device(args.device))


if __name__ == "__main__":
    main()
