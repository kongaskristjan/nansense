"""Training loop primitives for the vision example."""

from __future__ import annotations

import contextlib
from collections.abc import Iterator
from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader

from playgrad import Session


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


def _batches(
    session: Session | None,
    loader: DataLoader,
    *,
    phase: str,
    epoch: int,
) -> Iterator[tuple[Tensor, Tensor]]:
    """Iterate `loader`, wrapped in `session.batches` when a session is given.

    The loop body runs inside the session's batch context (the generator is
    suspended there), so hooks are installed before the forward pass and a
    time-travel jump surfaces from the `for` statement, not mid-body.
    """
    if session is None:
        yield from loader
        return
    yield from session.batches(loader, phase=phase, epoch=epoch)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    amp_dtype: torch.dtype | None = None,
    *,
    session: Session | None = None,
    epoch: int = 0,
) -> EpochStats:
    model.train()
    total_loss = 0.0
    total_acc = 0.0
    n_batches = 0
    for inputs, targets in _batches(session, loader, phase="train", epoch=epoch):
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
    session: Session | None = None,
    epoch: int = 0,
) -> EpochStats:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    for inputs, targets in _batches(session, loader, phase="val", epoch=epoch):
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
