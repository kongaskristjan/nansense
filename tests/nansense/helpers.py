"""Shared harness helpers for the nansense test suite.

Stub models, train-step functions, and the start-thread -> wait-until-paused
-> detach -> join lifecycle that the session-level tests share.
"""

from __future__ import annotations

import threading
from collections.abc import Callable, Iterator
from contextlib import contextmanager

import torch
from torch import Tensor, nn

import nansense
from nansense.schedule import BatchPosition
from nansense.session import Session


def make_position(
    phase: str,
    epoch: int,
    batch_idx: int,
    *,
    is_last_in_phase: bool = False,
    is_last_in_epoch: bool = False,
    is_last_overall: bool = False,
) -> BatchPosition:
    """A `BatchPosition` literal whose `is_last_*` flags default to False."""
    return BatchPosition(
        phase=phase,
        epoch=epoch,
        batch_idx=batch_idx,
        is_last_in_phase=is_last_in_phase,
        is_last_in_epoch=is_last_in_epoch,
        is_last_overall=is_last_overall,
    )


class TinyNet(nn.Module):
    """Two-layer MLP (4 -> 8 -> 3) that fx-traces cleanly."""

    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(4, 8)
        self.fc2 = nn.Linear(8, 3)

    def forward(self, x: Tensor) -> Tensor:
        return self.fc2(torch.relu(self.fc1(x)))


class DynamicNet(nn.Module):
    """Data-dependent control flow: forces the hook-fallback capture path."""

    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(4, 2)

    def forward(self, x: Tensor) -> Tensor:
        if x.sum() > 0:
            return self.fc(x)
        return self.fc(-x)


def train_step(
    model: nn.Module,
    *,
    input_shape: tuple[int, ...] = (4,),
    num_classes: int = 3,
    batch_size: int = 2,
) -> None:
    """One zero_grad + forward + cross-entropy backward pass on random data."""
    x = torch.randn(batch_size, *input_shape)
    y = torch.randint(0, num_classes, (batch_size,))
    model.zero_grad(set_to_none=True)
    loss = nn.functional.cross_entropy(model(x), y)
    loss.backward()


def optimizer_train_step(model: nn.Module, optimizer: torch.optim.Optimizer) -> None:
    """`train_step` on TinyNet-shaped data, plus an optimizer update."""
    x = torch.randn(2, 4)
    y = torch.randint(0, 3, (2,))
    optimizer.zero_grad(set_to_none=True)
    loss = nn.functional.cross_entropy(model(x), y)
    loss.backward()
    optimizer.step()


def make_session(
    epochs: int = 2, phases: dict[str, int] | None = None
) -> tuple[Session, TinyNet]:
    """A fresh TinyNet session (no optimizer) over the given schedule."""
    if phases is None:
        phases = {"train": 2, "val": 2}
    model = TinyNet()
    return nansense.start(model, epochs=epochs, phases=phases), model


def run_in_thread(target: Callable[[], None]) -> threading.Thread:
    """Start `target` on a daemon thread and return the running thread."""
    thread = threading.Thread(target=target, daemon=True)
    thread.start()
    return thread


@contextmanager
def paused_worker(
    session: Session, loop: Callable[[], None], *, timeout: float = 5.0
) -> Iterator[threading.Thread]:
    """Run `loop` on a worker thread and yield once the session is paused.

    On exit (even when the body fails) the session is detached and the worker
    joined, so a stuck training thread cannot wedge the rest of the suite.
    """
    thread = run_in_thread(loop)
    try:
        assert session.wait_until_paused(timeout=timeout)
        yield thread
    finally:
        session.detach()
        thread.join(timeout=timeout)
        assert not thread.is_alive()


@contextmanager
def paused_session[M: nn.Module](
    model: M,
    step: Callable[[M], None] = train_step,
    *,
    epochs: int = 1,
    phases: dict[str, int] | None = None,
    optimizer: torch.optim.Optimizer | None = None,
    timeout: float = 5.0,
) -> Iterator[Session]:
    """Start a session whose worker runs the whole schedule with `step` per
    batch, yielding once the worker pauses on the first STEP-mode capture."""
    if phases is None:
        phases = {"train": 2}
    session = nansense.start(model, epochs=epochs, phases=phases, optimizer=optimizer)

    def loop() -> None:
        for epoch in range(epochs):
            for phase, n in phases.items():
                for _ in range(n):
                    with session.batch(phase=phase, epoch=epoch):
                        step(model)

    with paused_worker(session, loop, timeout=timeout):
        yield session
