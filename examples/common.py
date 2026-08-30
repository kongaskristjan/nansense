"""Support code shared by the examples: training-loop primitives and CLI utilities.

The pedagogical content — wiring NaNsense into a training script — lives in
each example's `main.py`; this module holds the plain training plumbing those
scripts would otherwise duplicate verbatim.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import sys
from collections.abc import Callable, Iterator
from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader

from nansense import Session


@dataclass
class EpochStats:
    loss: float
    # The metric_fn output averaged over batches — classification accuracy by
    # default, but any accuracy-like scalar (per-cell accuracy, a depth
    # threshold ratio, ...) when the caller passes its own `metric_fn`.
    accuracy: float


def _accuracy(output: Tensor, targets: Tensor) -> float:
    """Top-1 classification accuracy — the default metric for `train_one_epoch`
    / `evaluate`. Regression / dense tasks pass their own `metric_fn`."""
    preds = output.argmax(dim=1)
    return (preds == targets).float().mean().item()


# --dtype CLI choice -> the autocast compute dtype (None disables autocast, i.e.
# plain fp32). Model weights always stay fp32; this only sets the forward/loss
# compute dtype under `torch.autocast`.
_AMP_DTYPES: dict[str, torch.dtype | None] = {
    "fp32": None,
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
}

# Kept as a help-text constant so every example's --dtype flag reads identically
# and the "no GradScaler" caveat below stays in one place.
DTYPE_HELP = (
    "Compute precision (default fp32). This example does not use GradScaler, "
    "so fp16 also demonstrates gradient underflow in NaNsense."
)


def add_dtype_arg(parser: argparse.ArgumentParser) -> None:
    """Register the shared `--dtype {fp32,fp16,bf16}` flag (default fp32).

    See `DTYPE_HELP`: the flag picks the autocast compute dtype only — weights
    stay fp32 and no GradScaler is used (an intentional choice so fp16 underflow
    is observable rather than hidden). Pair with `amp_dtype_from_name`."""
    parser.add_argument(
        "--dtype",
        choices=list(_AMP_DTYPES),
        default="fp32",
        help=DTYPE_HELP,
    )


def amp_dtype_from_name(name: str) -> torch.dtype | None:
    """Map a `--dtype` choice to its autocast dtype; fp32 -> None (no autocast)."""
    return _AMP_DTYPES[name]


@contextlib.contextmanager
def autocast(device: torch.device, amp_dtype: torch.dtype | None) -> Iterator[None]:
    """Autocast the enclosed forward/loss to `amp_dtype`, or run unchanged when
    it is None (fp32). No GradScaler is paired with this on purpose — see
    `DTYPE_HELP`."""
    if amp_dtype is None:
        yield
        return
    with torch.autocast(device_type=device.type, dtype=amp_dtype):
        yield


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    amp_dtype: torch.dtype | None = None,
    *,
    session: Session,
    epoch: int | None = None,
    metric_fn: Callable[[Tensor, Tensor], float] = _accuracy,
) -> EpochStats:
    """One training epoch under `session.batches` (a disabled session is the
    no-branching off switch — the loop body runs inside the batch context
    either way, so hooks install before the forward pass and a time-travel
    jump surfaces from the `for` statement, not mid-body).

    The epoch `session.batches` records defaults to the one `session.epochs()`
    is on (the caller's loop); a plain `range`-driven loop passes `epoch=`
    explicitly instead (the playground's prepare run does).

    `metric_fn(output, targets) -> float` defaults to classification accuracy;
    regression / dense examples pass their own (it is only logged, never
    backpropagated). Loss and metric are averaged over batches."""
    model.train()
    total_loss = 0.0
    total_metric = 0.0
    n_batches = 0
    for inputs, targets in session.batches(loader, phase="train", epoch=epoch):
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with autocast(device, amp_dtype):
            output = model(inputs)
            loss = criterion(output, targets)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_metric += metric_fn(output, targets)
        n_batches += 1

    return EpochStats(loss=total_loss / n_batches, accuracy=total_metric / n_batches)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    amp_dtype: torch.dtype | None = None,
    *,
    session: Session,
    epoch: int | None = None,
    metric_fn: Callable[[Tensor, Tensor], float] = _accuracy,
) -> EpochStats:
    """Mirror of `train_one_epoch` for the val phase: forward-only, with loss
    and `metric_fn` averaged over batches."""
    model.eval()
    total_loss = 0.0
    total_metric = 0.0
    n_batches = 0
    for inputs, targets in session.batches(loader, phase="val", epoch=epoch):
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        with autocast(device, amp_dtype):
            output = model(inputs)
            loss = criterion(output, targets)

        total_loss += loss.item()
        total_metric += metric_fn(output, targets)
        n_batches += 1

    return EpochStats(loss=total_loss / n_batches, accuracy=total_metric / n_batches)


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


def default_num_workers() -> int:
    """The `--num-workers` default: 2, or 0 where DataLoader workers stall.

    On Linux, worker processes are forked and their teardown is free, so two
    of them overlap augmentation with the forward pass for nothing.

    macOS and Windows are the platforms where `torch.multiprocessing` offers
    only the `file_system` sharing strategy, and that strategy makes workers
    pathological for a training loop that starts a fresh iterator per phase.
    Every worker forks a `torch_shm_manager` helper which outlives it and
    inherits the worker's sentinel pipe, so the pipe never reaches EOF: the
    `join(timeout=MP_STATUS_CHECK_INTERVAL)` in PyTorch's worker shutdown
    always waits out its full 5 seconds. A train phase followed by a val
    phase therefore stops dead for `5s * num_workers` at each boundary — with
    the NaNsense UI still showing the last batch of the phase, so the run
    looks frozen — and leaks one orphaned `torch_shm_manager` per worker per
    phase. Loading in-process instead costs a fraction of that and leaks
    nothing; pass `--num-workers` explicitly to override.
    """
    if "file_descriptor" in torch.multiprocessing.get_all_sharing_strategies():
        return 2
    return 0


def add_num_workers_arg(parser: argparse.ArgumentParser) -> None:
    """Register the shared `--num-workers` flag (see `default_num_workers`)."""
    parser.add_argument(
        "--num-workers",
        type=int,
        default=default_num_workers(),
        help=(
            "DataLoader worker processes (default 2 on Linux, 0 on "
            "macOS/Windows, where per-phase worker shutdown costs 5s each)."
        ),
    )
