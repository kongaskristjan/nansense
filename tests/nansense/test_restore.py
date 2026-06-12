"""Tests for time travel: epoch caching, restorer loop, and session jumps."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch import Tensor, nn

import nansense
from nansense.restore import (
    EpochCache,
    TimeTravelError,
    TrainingRestorer,
    restore_rng,
    validate_model_state,
)
from nansense.session import Session
from tests.nansense.helpers import TinyNet, optimizer_train_step, paused_worker


class OtherNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(4, 3)

    def forward(self, x: Tensor) -> Tensor:
        return self.fc(x)


def _make_training(
    tmp_path: Path, *, epochs: int = 3, enabled: bool = True
) -> tuple[
    Session,
    TrainingRestorer,
    TinyNet,
    torch.optim.SGD,
    torch.optim.lr_scheduler.StepLR,
]:
    model = TinyNet()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)
    session = nansense.start(
        model,
        epochs=epochs,
        phases={"train": 2},
        enabled=enabled,
        optimizer=optimizer,
        scheduler=scheduler,
    )
    restorer = session.training_restorer(cache_dir=tmp_path / "cache")
    return session, restorer, model, optimizer, scheduler


def test_epoch_cache_save_load_roundtrip(tmp_path: Path) -> None:
    cache = EpochCache(tmp_path / "cache")
    assert cache.cached_epochs() == []  # missing directory is fine

    model = TinyNet()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
    cache.save(2, model=model, optimizer=optimizer, scheduler=scheduler)

    assert cache.cached_epochs() == [2]
    payload = cache.load(2)
    assert payload["epoch"] == 2
    torch.testing.assert_close(payload["model"]["fc1.weight"], model.fc1.weight)
    assert isinstance(payload["optimizer"], dict)
    assert isinstance(payload["scheduler"], dict)


def test_epoch_cache_overwrites_existing_epoch(tmp_path: Path) -> None:
    cache = EpochCache(tmp_path / "cache")
    model = TinyNet()
    cache.save(0, model=model, optimizer=None, scheduler=None)
    with torch.no_grad():
        model.fc1.weight.add_(1.0)
    cache.save(0, model=model, optimizer=None, scheduler=None)

    payload = cache.load(0)
    torch.testing.assert_close(payload["model"]["fc1.weight"], model.fc1.weight)


def test_epoch_cache_load_missing_epoch_raises(tmp_path: Path) -> None:
    cache = EpochCache(tmp_path / "cache")
    with pytest.raises(TimeTravelError, match="no cached model for epoch 5"):
        cache.load(5)


@pytest.mark.parametrize(
    ("make_model", "expected"),
    [
        (TinyNet, None),
        (OtherNet, "does not match"),
    ],
)
def test_validate_model_state(
    tmp_path: Path, make_model: type[nn.Module], expected: str | None
) -> None:
    cache = EpochCache(tmp_path / "cache")
    cache.save(0, model=make_model(), optimizer=None, scheduler=None)
    error = validate_model_state(cache.load(0), TinyNet())
    if expected is None:
        assert error is None
    else:
        assert error is not None and expected in error


def test_restorer_loop_runs_once_without_jumps(tmp_path: Path) -> None:
    session, restorer, model, optimizer, _ = _make_training(tmp_path, epochs=2)
    session.detach()

    attempts: list[int] = []
    while restorer.pending():
        with restorer:
            attempts.append(restorer.start_epoch)
            for epoch in restorer.epochs():
                for _ in range(2):
                    with session.batch(phase="train", epoch=epoch):
                        optimizer_train_step(model, optimizer)

    assert attempts == [0]
    assert restorer.finished
    assert restorer.cache.cached_epochs() == [0, 1]


def test_restorer_pending_raises_when_with_block_is_missing(tmp_path: Path) -> None:
    _, restorer, _, _, _ = _make_training(tmp_path)
    assert restorer.pending()
    with pytest.raises(RuntimeError, match="with restorer"):
        restorer.pending()


def test_disabled_session_restorer_is_inert(tmp_path: Path) -> None:
    session, restorer, model, optimizer, _ = _make_training(
        tmp_path, epochs=2, enabled=False
    )

    attempts: list[int] = []
    while restorer.pending():
        with restorer:
            attempts.append(restorer.start_epoch)
            for epoch in restorer.epochs():
                for _ in range(2):
                    with session.batch(phase="train", epoch=epoch):
                        optimizer_train_step(model, optimizer)

    assert attempts == [0]
    assert restorer.cache.cached_epochs() == []  # nothing written to disk
    assert not (tmp_path / "cache").exists()
    status = session.time_travel_status()
    assert not status.available


def test_request_time_travel_without_restorer_raises() -> None:
    session = nansense.start(TinyNet(), epochs=2, phases={"train": 2})
    with pytest.raises(TimeTravelError, match="training restorer"):
        session.request_time_travel(0)


def test_request_time_travel_rejects_missing_epoch(tmp_path: Path) -> None:
    session, _, _, _, _ = _make_training(tmp_path)
    with pytest.raises(TimeTravelError, match="no cached model"):
        session.request_time_travel(1)


def test_request_time_travel_rejects_out_of_range_epoch(tmp_path: Path) -> None:
    session, _, _, _, _ = _make_training(tmp_path, epochs=3)
    with pytest.raises(TimeTravelError, match="out of range"):
        session.request_time_travel(7)


def test_request_time_travel_rejects_mismatched_model(tmp_path: Path) -> None:
    session, restorer, _, _, _ = _make_training(tmp_path)
    # Simulate a cache file left behind by a previous run of a different model.
    restorer.cache.save(0, model=OtherNet(), optimizer=None, scheduler=None)
    with pytest.raises(TimeTravelError, match="does not match"):
        session.request_time_travel(0)


def test_time_travel_status_reports_reason_and_cached_epochs(tmp_path: Path) -> None:
    session = nansense.start(TinyNet(), epochs=3, phases={"train": 2})
    status = session.time_travel_status()
    assert not status.available
    assert status.reason is not None and "restorer" in status.reason

    restorer = session.training_restorer(cache_dir=tmp_path / "cache")
    restorer.cache.save(1, model=session.model, optimizer=None, scheduler=None)
    restorer.cache.save(9, model=session.model, optimizer=None, scheduler=None)
    status = session.time_travel_status()
    assert status.available
    assert status.cached_epochs == [1]  # epoch 9 is outside the schedule
    assert status.total_epochs == 3


def test_second_restorer_for_same_session_raises(tmp_path: Path) -> None:
    session, _, _, _, _ = _make_training(tmp_path)
    with pytest.raises(RuntimeError, match="already has a training restorer"):
        session.training_restorer(cache_dir=tmp_path / "other")


def test_time_travel_jump_restores_and_replays_deterministically(
    tmp_path: Path,
) -> None:
    """End-to-end: jump back to epoch 1 and verify the replay is identical.

    The training thread logs (epoch, fc1 weights, lr) at the top of every
    epoch. After jumping from the pause at (train, 2, 0) back to epoch 1,
    the second attempt's epoch-1 *and* epoch-2 entries must match the first
    attempt's — model/optimizer/scheduler state and RNG are all restored, so
    the replayed epochs see identical data and produce identical weights.
    """
    session, restorer, model, optimizer, scheduler = _make_training(
        tmp_path, epochs=3
    )

    attempts: list[int] = []
    epoch_log: list[tuple[int, Tensor, float]] = []

    def loop() -> None:
        while restorer.pending():
            with restorer:
                attempts.append(restorer.start_epoch)
                for epoch in restorer.epochs():
                    epoch_log.append(
                        (
                            epoch,
                            model.fc1.weight.detach().clone(),
                            optimizer.param_groups[0]["lr"],
                        )
                    )
                    for _ in range(2):
                        with session.batch(phase="train", epoch=epoch):
                            optimizer_train_step(model, optimizer)
                    scheduler.step()

    # First batch pauses (STEP mode); run forward to the pause at (2, 0) so
    # epochs 0, 1, and 2 all have checkpoints.
    with paused_worker(session, loop, timeout=10.0):
        session.step_until_position(phase="train", epoch=2, batch_idx=0)
        assert session.wait_until_paused(after_pauses=1, timeout=10.0)
        assert restorer.cache.cached_epochs() == [0, 1, 2]

        # Jump back to epoch 1; the session enters STEP mode, so the first
        # batch of the replayed epoch pauses for inspection.
        pauses = session.pause_count
        session.request_time_travel(1)
        assert session.wait_until_paused(after_pauses=pauses, timeout=10.0)
        snap = session.snapshot
        assert snap is not None
        assert (snap.position.phase, snap.position.epoch, snap.position.batch_idx) == (
            "train",
            1,
            0,
        )

    assert restorer.finished

    assert attempts == [0, 1]
    # Attempt 1 logged epochs 0..2 (epoch 2 only began); attempt 2 logged 1..2.
    assert [e for e, _, _ in epoch_log] == [0, 1, 2, 1, 2]
    first_ep1, first_ep2 = epoch_log[1], epoch_log[2]
    replay_ep1, replay_ep2 = epoch_log[3], epoch_log[4]
    torch.testing.assert_close(replay_ep1[1], first_ep1[1])
    assert replay_ep1[2] == first_ep1[2]
    # Epoch 2's start state matches too: the replayed epoch 1 saw the same
    # RNG stream, so its training steps reproduced the original weights.
    torch.testing.assert_close(replay_ep2[1], first_ep2[1])
    assert replay_ep2[2] == first_ep2[2]


def _epoch_order(loader: torch.utils.data.DataLoader) -> list[int]:
    """The sample indices a fresh iterator over `loader` yields, in order."""
    return [int(i.item()) for (idx,) in loader for i in idx]


def test_session_batches_saves_pre_iter_rng_for_deterministic_replay(
    tmp_path: Path,
) -> None:
    """The epoch checkpoint must reproduce that epoch's shuffled data order.

    `Session.batches` does `for item in loader`, and `iter(loader)` draws the
    DataLoader's shuffle seed from the global RNG. The epoch-start checkpoint
    must capture the RNG *before* that draw — otherwise restoring it and
    building a fresh iterator yields a different order (the saved state was
    post-draw). This test records each epoch's order while iterating through
    `session.batches`, then replays each saved checkpoint's RNG and asserts the
    fresh iterator reproduces that same epoch's order. Fails pre-fix because
    epoch <n>'s checkpoint replays as epoch <n+1>'s order.
    """
    epochs = 3
    dataset = torch.utils.data.TensorDataset(torch.arange(8))
    model = TinyNet()
    session = nansense.start(model, epochs=epochs, phases={"train": 2})
    restorer = session.training_restorer(cache_dir=tmp_path / "cache")
    session.detach()

    torch.manual_seed(0)
    recorded: list[list[int]] = []
    for epoch in range(epochs):
        loader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=True)
        order: list[int] = []
        for (idx,) in session.batches(loader, phase="train", epoch=epoch):
            order.extend(int(i.item()) for i in idx)
        recorded.append(order)

    assert restorer.cache.cached_epochs() == [0, 1, 2]
    # The shuffle actually varies across epochs, so a wrong-epoch RNG would be
    # caught (the bug surfaced as epoch-1's checkpoint replaying epoch-2 order).
    assert recorded[0] != recorded[1]

    for epoch in range(epochs):
        payload = restorer.cache.load(epoch)
        restore_rng(payload["rng"])
        loader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=True)
        assert _epoch_order(loader) == recorded[epoch], (
            f"epoch {epoch} checkpoint did not reproduce its data order"
        )
