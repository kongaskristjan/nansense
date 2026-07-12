"""Tests for the hosted-playground entrypoint (examples/playground/main.py).

Runs the prepare + serve flow end-to-end on a tiny synthetic dataset: prepare
must freeze the run's last train batch into a moment file (no validation
after it, no epoch checkpoints), and serving must reload that moment around a
fresh model and park locked with every layer's statistics browsable.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

import nansense
from examples.playground.main import (
    SHOWN_LAYERS,
    build_model,
    open_showcase,
    train_and_freeze,
)
from examples.standard.data import DATASETS
from nansense.session import StatsScope
from nansense.ui import render
from tests.nansense.helpers import run_in_thread

_CONFIG = DATASETS["mnist"]
_EPOCHS = 2
_BATCHES = 3
_DEVICE = torch.device("cpu")


@pytest.fixture(autouse=True)
def _restore_strip_format() -> Iterator[None]:
    """`open_showcase` flips the global strip format; undo it per test."""
    yield
    render.set_strip_format("BMP")


def _loaders() -> tuple[DataLoader, DataLoader]:
    x = torch.randn(_BATCHES * 4, _CONFIG.in_channels, 32, 32)
    y = torch.randint(0, _CONFIG.num_classes, (_BATCHES * 4,))
    train = DataLoader(TensorDataset(x, y), batch_size=4, shuffle=True)
    val = DataLoader(TensorDataset(x[:4], y[:4]), batch_size=4)
    return train, val


def _prepare(moment_path: Path) -> None:
    model = build_model(_CONFIG)
    train_loader, val_loader = _loaders()
    train_and_freeze(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=_DEVICE,
        epochs=_EPOCHS,
        moment_path=moment_path,
    )


def test_shown_layers_exist_on_the_lenet_graph() -> None:
    session = nansense.start(build_model(_CONFIG))
    assert set(SHOWN_LAYERS) <= set(session.layer_names)


def test_prepare_freezes_the_moment_and_nothing_else(tmp_path: Path) -> None:
    moment_path = tmp_path / "moment.pt"
    _prepare(moment_path)
    assert moment_path.exists()
    # No epoch cache: the frozen moment is the only artifact serving needs.
    assert not list(tmp_path.glob("epoch_*.pt"))


def test_serve_parks_locked_at_the_frozen_train_batch(tmp_path: Path) -> None:
    moment_path = tmp_path / "moment.pt"
    _prepare(moment_path)
    session = open_showcase(
        build_model(_CONFIG),
        moment_path,
        config=_CONFIG,
        port=None,  # no UI in tests; the parked, locked state is under test
    )
    assert session.locked
    assert session.stats_scope is StatsScope.ALL
    assert session.watched_layers == frozenset(SHOWN_LAYERS)

    thread = run_in_thread(session.park)
    try:
        assert session.wait_until_paused(timeout=10.0)
        position = session.live_position
        assert position is not None
        assert (position.phase, position.epoch) == ("train", _EPOCHS - 1)
        assert position.batch_idx == _BATCHES - 1
        # The prepare run collected stats for every layer, not just the seed.
        assert session.stats_layers == frozenset(session.layer_names)
        # The frozen snapshot is a train batch: gradients are populated.
        snapshot = session.snapshot
        assert snapshot is not None and snapshot.activation_gradients
        assert snapshot.position == position
        # The frozen schedule still reports the run's totals.
        assert session.schedule.epochs == _EPOCHS
        assert session.schedule.phase_count("train") == _BATCHES
        # Locked: a resume attempt leaves the park in place.
        session.step_batch()
        assert session.is_running is False
    finally:
        session.close()
        thread.join(timeout=10.0)
        assert not thread.is_alive()
