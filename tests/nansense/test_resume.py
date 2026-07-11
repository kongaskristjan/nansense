"""Tests for resuming a run from a cache directory (`epochs(start_epoch=)`).

The cache directory is adopted as-is — `EpochCache.cached_epochs` scans the
disk, so checkpoints baked by an earlier process (e.g. into a deployment
image) feed both the resume and the time-travel dialog.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

import nansense
from nansense.restore import EpochCache, TimeTravelError
from nansense.session import Session
from tests.nansense.helpers import TinyNet, make_session, train_step

_EPOCHS = 3
_BATCHES = 2


def _run(session: Session, model: TinyNet, cache_dir: Path, **kwargs: int) -> list[int]:
    """Drive the full flat loop detached; returns the yielded epochs."""
    session.detach()
    seen: list[int] = []
    for epoch in session.epochs(_EPOCHS, cache_dir=cache_dir, **kwargs):
        with session.restore_point():
            seen.append(epoch)
            for _ in range(_BATCHES):
                with session.batch(phase="train", epoch=epoch):
                    train_step(model)
    return seen


def _bake_cache(cache_dir: Path) -> None:
    """A completed 3-epoch TinyNet run whose checkpoints land in `cache_dir`."""
    session, model = make_session(epochs=_EPOCHS, phases={"train": _BATCHES})
    assert _run(session, model, cache_dir) == list(range(_EPOCHS))


def test_resume_starts_at_the_requested_epoch_with_restored_state(
    tmp_path: Path,
) -> None:
    _bake_cache(tmp_path)
    session, model = make_session(epochs=_EPOCHS, phases={"train": _BATCHES})
    seen: list[int] = []
    session.detach()
    for epoch in session.epochs(_EPOCHS, cache_dir=tmp_path, start_epoch=2):
        with session.restore_point():
            seen.append(epoch)
            if len(seen) == 1:
                # Entering the first restore_point loaded epoch 2's baked
                # checkpoint into the fresh model.
                cached = EpochCache(tmp_path).load(2)["model"]
                for name, tensor in model.state_dict().items():
                    assert torch.equal(tensor, cached[name])
            for _ in range(_BATCHES):
                with session.batch(phase="train", epoch=epoch):
                    train_step(model)
    assert seen == [2]


def test_adopted_cache_feeds_the_time_travel_status(tmp_path: Path) -> None:
    _bake_cache(tmp_path)
    session, _model = make_session(epochs=_EPOCHS, phases={"train": _BATCHES})
    # Creating the loop restorer adopts the directory: the baked epochs are
    # offered before this process has trained a single batch.
    session.epochs(_EPOCHS, cache_dir=tmp_path)
    status = session.time_travel_status()
    assert status.available
    assert status.cached_epochs == list(range(_EPOCHS))


def test_resume_rejects_a_missing_epoch(tmp_path: Path) -> None:
    _bake_cache(tmp_path)
    session, _model = make_session(epochs=_EPOCHS, phases={"train": _BATCHES})
    with pytest.raises(TimeTravelError, match="out of range"):
        session.epochs(_EPOCHS, cache_dir=tmp_path, start_epoch=_EPOCHS)
    session2, _model2 = make_session(epochs=_EPOCHS, phases={"train": _BATCHES})
    with pytest.raises(TimeTravelError, match="no cached model"):
        session2.epochs(_EPOCHS, cache_dir=tmp_path / "empty", start_epoch=1)


def test_resume_rejects_a_mismatched_model(tmp_path: Path) -> None:
    _bake_cache(tmp_path)

    class OtherNet(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.fc = torch.nn.Linear(4, 3)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.fc(x)

    session = nansense.start(
        OtherNet(), epochs=_EPOCHS, phases={"train": _BATCHES}
    )
    with pytest.raises(TimeTravelError, match="does not match"):
        session.epochs(_EPOCHS, cache_dir=tmp_path, start_epoch=1)


def test_resume_is_ignored_on_a_disabled_session(tmp_path: Path) -> None:
    session = nansense.start(
        TinyNet(), epochs=_EPOCHS, phases={"train": _BATCHES}, enabled=False
    )
    # No cache exists, but a disabled session never touches the disk — the
    # loop just runs the plain range.
    seen: list[int] = []
    for epoch in session.epochs(_EPOCHS, cache_dir=tmp_path, start_epoch=2):
        with session.restore_point():
            seen.append(epoch)
    assert seen == list(range(_EPOCHS))
