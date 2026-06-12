"""Support code shared by the examples: training-loop primitives and CLI utilities.

The pedagogical content — wiring nansense into a training script — lives in
each example's `main.py`; this module holds the plain training plumbing those
scripts would otherwise duplicate verbatim.
"""

from __future__ import annotations

import contextlib
import io
import sys
from collections.abc import Iterator
from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader

from nansense import Session


@dataclass
class EpochStats:
    loss: float
    accuracy: float


def _accuracy(logits: Tensor, targets: Tensor) -> float:
    preds = logits.argmax(dim=1)
    return (preds == targets).float().mean().item()


@contextlib.contextmanager
def _autocast(device: torch.device, amp_dtype: torch.dtype | None) -> Iterator[None]:
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
    epoch: int = 0,
) -> EpochStats:
    """One training epoch under `session.batches` (a disabled session is the
    no-branching off switch — the loop body runs inside the batch context
    either way, so hooks install before the forward pass and a time-travel
    jump surfaces from the `for` statement, not mid-body)."""
    model.train()
    total_loss = 0.0
    total_acc = 0.0
    n_batches = 0
    for inputs, targets in session.batches(loader, phase="train", epoch=epoch):
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with _autocast(device, amp_dtype):
            logits = model(inputs)
            loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_acc += _accuracy(logits, targets)
        n_batches += 1

    return EpochStats(loss=total_loss / n_batches, accuracy=total_acc / n_batches)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    amp_dtype: torch.dtype | None = None,
    *,
    session: Session,
    epoch: int = 0,
) -> EpochStats:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    for inputs, targets in session.batches(loader, phase="val", epoch=epoch):
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        with _autocast(device, amp_dtype):
            logits = model(inputs)
            loss = criterion(logits, targets)

        total_loss += loss.item() * targets.size(0)
        total_correct += int((logits.argmax(dim=1) == targets).sum().item())
        total_samples += targets.size(0)

    return EpochStats(
        loss=total_loss / total_samples,
        accuracy=total_correct / total_samples,
    )


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
