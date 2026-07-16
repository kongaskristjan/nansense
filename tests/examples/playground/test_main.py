"""Tests for the hosted-playground entrypoint (examples/playground/main.py).

Runs the prepare + serve flow end-to-end for every playground on a tiny
synthetic dataset: prepare must freeze the run's last train batch into a
moment file (no validation after it, no epoch checkpoints), and serving must
reload that moment around a fresh model and park locked with every layer's
statistics browsable.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Iterator
from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

import nansense
from examples.playground.main import (
    PLAYGROUNDS,
    PlaygroundSpec,
    open_showcase,
    train_and_freeze,
)
from nansense.session import StatsScope
from nansense.ui import render
from tests.nansense.helpers import run_in_thread

_EPOCHS = 2
_BATCHES = 3
# Both demo models accept 32x32 inputs: it is the real MNIST size, and the
# resnet is fully convolutional — so the imagenette flow tests cheaply too.
_IMAGE_SIZE = 32
_DEVICE = torch.device("cpu")

playgrounds = pytest.mark.parametrize(
    "spec", [PLAYGROUNDS[name] for name in sorted(PLAYGROUNDS)], ids=sorted(PLAYGROUNDS)
)


@pytest.fixture(autouse=True)
def _restore_strip_format() -> Iterator[None]:
    """`open_showcase` flips the global strip format; undo it per test."""
    yield
    render.set_strip_format("BMP")


def _loaders(spec: PlaygroundSpec) -> tuple[DataLoader, DataLoader]:
    config = spec.config
    x = torch.randn(_BATCHES * 4, config.in_channels, _IMAGE_SIZE, _IMAGE_SIZE)
    y = torch.randint(0, config.num_classes, (_BATCHES * 4,))
    train = DataLoader(TensorDataset(x, y), batch_size=4, shuffle=True)
    val = DataLoader(TensorDataset(x[:4], y[:4]), batch_size=4)
    return train, val


def _prepare(spec: PlaygroundSpec, moment_path: Path) -> PlaygroundSpec:
    spec = dataclasses.replace(spec, epochs=_EPOCHS)
    model = spec.build()
    train_loader, val_loader = _loaders(spec)
    train_and_freeze(
        spec,
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=_DEVICE,
        moment_path=moment_path,
    )
    return spec


@pytest.mark.parametrize(
    ("name", "layer"),
    [("mnist", "relu1"), ("imagenette", "stage1.0.conv1")],
)
def test_locked_playgrounds_default_to_a_single_card(name: str, layer: str) -> None:
    """One card on first load keeps a locked demo's landing view focused;
    every other layer stays a diagram click away."""
    assert PLAYGROUNDS[name].shown_layers == (layer,)


@playgrounds
def test_shown_layers_exist_on_the_model_graph(spec: PlaygroundSpec) -> None:
    session = nansense.start(spec.build())
    assert set(spec.shown_layers) <= set(session.layer_names)
    if spec.patch_layers is not None:  # no spec shortlists today; keep generic
        assert set(spec.patch_layers) <= set(session.layer_names)


def test_imagenette_dream_defaults_stay_under_the_locked_ceilings() -> None:
    """The cheaper imagenette dream defaults only seed the form — visitors
    may still raise them, up to the locked clamp, so the defaults must sit
    at or below those ceilings."""
    from nansense.experiments import _LOCKED_PARAM_LIMITS

    spec = PLAYGROUNDS["imagenette"]
    assert spec.dream_steps == 150
    assert spec.dream_channels == 4
    assert spec.dream_steps <= _LOCKED_PARAM_LIMITS["steps"]
    assert spec.dream_channels <= _LOCKED_PARAM_LIMITS["channels"]


@playgrounds
def test_prepare_freezes_the_moment_and_nothing_else(
    spec: PlaygroundSpec, tmp_path: Path
) -> None:
    moment_path = tmp_path / "moment.pt"
    _prepare(spec, moment_path)
    assert moment_path.exists()
    # No epoch cache: the frozen moment is the only artifact serving needs.
    assert not list(tmp_path.glob("epoch_*.pt"))


@playgrounds
def test_serve_parks_locked_at_the_frozen_train_batch(
    spec: PlaygroundSpec, tmp_path: Path
) -> None:
    moment_path = tmp_path / "moment.pt"
    spec = _prepare(spec, moment_path)
    session = open_showcase(
        spec.build(),
        moment_path,
        spec=spec,
        port=None,  # no UI in tests; the parked, locked state is under test
    )
    assert session.locked
    assert session.stats_scope is StatsScope.ALL
    assert session.watched_layers == frozenset(spec.shown_layers)


def test_serve_reseeds_the_watched_cards_from_the_spec(tmp_path: Path) -> None:
    """The moment freezes the watched seed it was prepared with; serving
    must re-base it on the spec's `shown_layers`, so the default cards can
    change without re-training the demo."""
    moment_path = tmp_path / "moment.pt"
    spec = _prepare(PLAYGROUNDS["mnist"], moment_path)
    served = dataclasses.replace(spec, shown_layers=("conv2",))
    session = open_showcase(
        served.build(),
        moment_path,
        spec=served,
        port=None,
    )
    assert session.watched_layers == frozenset({"conv2"})
    # Demo preferences armed ahead of the lock: experiments wait for a
    # manual Run, and the spec's deep-dream form defaults rode along.
    assert session.auto_run_experiments is False
    defaults = session.experiment_defaults
    assert defaults.get("steps") == (
        spec.dream_steps if spec.dream_steps is not None else None
    )
    assert defaults.get("channels") == (
        spec.dream_channels if spec.dream_channels is not None else None
    )
    # The spec's performance caps rode along in the moment file.
    perf = session.watch_performance
    if spec.channel_limit is not None:
        assert perf.channel_limit_enabled is True
        assert perf.channel_limit == spec.channel_limit
    if spec.samples_per_channel is not None:
        assert perf.samples_per_channel == spec.samples_per_channel

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
