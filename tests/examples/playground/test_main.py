"""Tests for the hosted-playground entrypoint (examples/playground/main.py).

Runs the prepare + serve flow end-to-end on a tiny synthetic dataset: the
serve loop must resume from the baked cache, replay the final epoch with
all-layer stats, and park locked on the run's last train batch.
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
    build_training,
    make_demo_session,
    train_epochs,
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
    """`make_demo_session` flips the global strip format; undo it per test."""
    yield
    render.set_strip_format("BMP")


def _loaders() -> tuple[DataLoader, DataLoader]:
    x = torch.randn(_BATCHES * 4, _CONFIG.in_channels, 32, 32)
    y = torch.randint(0, _CONFIG.num_classes, (_BATCHES * 4,))
    train = DataLoader(TensorDataset(x, y), batch_size=4, shuffle=True)
    val = DataLoader(TensorDataset(x[:4], y[:4]), batch_size=4)
    return train, val


def _prepare(cache_dir: Path) -> None:
    model, criterion, optimizer, scheduler = build_training(
        _CONFIG, epochs=_EPOCHS
    )
    train_loader, val_loader = _loaders()
    session = nansense.start(model, optimizer=optimizer, scheduler=scheduler)
    session.detach()
    train_epochs(
        session,
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        train_loader=train_loader,
        val_loader=val_loader,
        device=_DEVICE,
        epochs=_EPOCHS,
        cache_dir=cache_dir,
    )
    session.close()


def test_shown_layers_exist_on_the_lenet_graph() -> None:
    model, _criterion, _optimizer, _scheduler = build_training(
        _CONFIG, epochs=_EPOCHS
    )
    session = nansense.start(model)
    assert set(SHOWN_LAYERS) <= set(session.layer_names)


def test_prepare_writes_every_epoch_checkpoint(tmp_path: Path) -> None:
    _prepare(tmp_path)
    assert sorted(p.name for p in tmp_path.glob("epoch_*.pt")) == [
        f"epoch_{e}.pt" for e in range(_EPOCHS)
    ]


def test_serve_parks_locked_at_the_final_train_batch(tmp_path: Path) -> None:
    _prepare(tmp_path)
    model, criterion, optimizer, scheduler = build_training(
        _CONFIG, epochs=_EPOCHS
    )
    train_loader, _val_loader = _loaders()
    session = make_demo_session(
        model,
        optimizer,
        scheduler,
        train_batches=len(train_loader),
        config=_CONFIG,
        port=None,  # no UI in tests; the parked pause is what's under test
    )
    assert session.locked
    assert session.stats_scope is StatsScope.ALL
    assert session.watched_layers == frozenset(SHOWN_LAYERS)

    thread = run_in_thread(
        lambda: train_epochs(
            session,
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            train_loader=train_loader,
            val_loader=None,
            device=_DEVICE,
            epochs=_EPOCHS,
            cache_dir=tmp_path,
            start_epoch=_EPOCHS - 1,
        )
    )
    try:
        assert session.wait_until_paused(timeout=30.0)
        position = session.live_position
        assert position is not None
        assert (position.phase, position.epoch) == ("train", _EPOCHS - 1)
        assert position.batch_idx == len(train_loader) - 1
        assert position.is_last_overall
        # The replay collected stats for every layer, not just the seed.
        assert session.stats_layers == frozenset(session.layer_names)
        # The parked snapshot is a train batch: gradients are populated.
        snapshot = session.snapshot
        assert snapshot is not None and snapshot.activation_gradients
        # Locked: a resume attempt leaves the park in place.
        session.step_batch()
        assert session.is_running is False
    finally:
        session.close()
        thread.join(timeout=10.0)
        assert not thread.is_alive()
