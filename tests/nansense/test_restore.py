"""Tests for time travel: epoch caching, restorer loop, and session jumps."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch import Tensor, nn

import nansense
from nansense.restore import (
    DEFAULT_CACHE_DIR,
    EpochCache,
    TimeTravelError,
    TrainingRestorer,
    capture_rng,
    release_cpu_memory,
    restore_rng,
    validate_model_state,
    validate_optimizer_state,
    validate_scheduler_state,
)
from nansense.session import Mode, Session
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


def test_default_cache_dir() -> None:
    # The library default lands in `.nansense_cache/` (gitignored), and a
    # restorer built without an explicit cache_dir picks it up. Construction is
    # lazy (no mkdir), so this asserts the path without touching disk.
    assert DEFAULT_CACHE_DIR == Path(".nansense_cache")
    assert TrainingRestorer().cache.directory == DEFAULT_CACHE_DIR


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


def test_epoch_cache_mmap_load_round_trips_and_validates(tmp_path: Path) -> None:
    # The validation path loads memory-mapped (no full copy in RAM); it must
    # still expose the keys, shapes, and values the validators read.
    cache = EpochCache(tmp_path / "cache")
    model = TinyNet()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    cache.save(0, model=model, optimizer=optimizer, scheduler=None)

    payload = cache.load(0, mmap=True)
    assert payload["epoch"] == 0
    torch.testing.assert_close(payload["model"]["fc1.weight"], model.fc1.weight)
    assert validate_model_state(payload, TinyNet()) is None
    assert validate_optimizer_state(payload, optimizer) is None


def test_release_cpu_memory_is_safe_to_call() -> None:
    # A best-effort trim: it must never raise, even where `malloc_trim` is
    # absent (musl, macOS) — the restore path calls it unconditionally.
    release_cpu_memory()


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


def test_validate_optimizer_state_accepts_matching(tmp_path: Path) -> None:
    cache = EpochCache(tmp_path / "cache")
    model = TinyNet()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    cache.save(0, model=model, optimizer=optimizer, scheduler=None)
    assert validate_optimizer_state(cache.load(0), optimizer) is None


def test_validate_optimizer_state_skips_when_absent(tmp_path: Path) -> None:
    cache = EpochCache(tmp_path / "cache")
    model = TinyNet()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    # No optimizer in the session: nothing to validate against.
    cache.save(0, model=model, optimizer=optimizer, scheduler=None)
    assert validate_optimizer_state(cache.load(0), None) is None
    # No optimizer in the checkpoint: nothing to load, so nothing to reject.
    cache.save(1, model=model, optimizer=None, scheduler=None)
    assert validate_optimizer_state(cache.load(1), optimizer) is None


def test_validate_optimizer_state_rejects_param_group_count(tmp_path: Path) -> None:
    cache = EpochCache(tmp_path / "cache")
    model = TinyNet()
    one_group = torch.optim.SGD(model.parameters(), lr=0.1)
    cache.save(0, model=model, optimizer=one_group, scheduler=None)
    # A live optimizer with two param groups can't load a one-group cache.
    two_groups = torch.optim.SGD(
        [
            {"params": model.fc1.parameters(), "lr": 0.1},
            {"params": model.fc2.parameters(), "lr": 0.2},
        ]
    )
    error = validate_optimizer_state(cache.load(0), two_groups)
    assert error is not None and "param group count" in error


def test_validate_optimizer_state_rejects_changed_class(tmp_path: Path) -> None:
    cache = EpochCache(tmp_path / "cache")
    model = TinyNet()
    sgd = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    optimizer_train_step(model, sgd)  # populate SGD's momentum_buffer state
    cache.save(0, model=model, optimizer=sgd, scheduler=None)
    # Same param-group layout, but Adam writes exp_avg/exp_avg_sq instead of
    # momentum_buffer — load_state_dict's count check passes, step() detonates.
    adam = torch.optim.Adam(model.parameters(), lr=0.1)
    optimizer_train_step(model, adam)
    error = validate_optimizer_state(cache.load(0), adam)
    assert error is not None and "state keys differ" in error


def test_validate_scheduler_state_accepts_matching(tmp_path: Path) -> None:
    cache = EpochCache(tmp_path / "cache")
    model = TinyNet()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
    cache.save(0, model=model, optimizer=optimizer, scheduler=scheduler)
    assert validate_scheduler_state(cache.load(0), scheduler) is None
    # No scheduler on either side is skipped.
    assert validate_scheduler_state(cache.load(0), None) is None
    cache.save(1, model=model, optimizer=optimizer, scheduler=None)
    assert validate_scheduler_state(cache.load(1), scheduler) is None


def test_request_time_travel_rejects_mismatched_optimizer(tmp_path: Path) -> None:
    """A cache whose optimizer layout differs is rejected up-front, intact.

    Mirrors the mismatched-model case but swaps the live optimizer for one
    with a different param-group structure after the checkpoint was written —
    `request_time_travel` must raise `TimeTravelError` on the UI thread
    (not a raw `ValueError` later on the training thread) and leave the live
    model and optimizer untouched (no partial mutation mid-restore).
    """
    session, restorer, model, _, _ = _make_training(tmp_path)
    # Epoch 0's checkpoint holds the single-group SGD layout.
    assert restorer.cache.cached_epochs() == []
    restorer.save_epoch_start(0)

    # Swap the live optimizer for a two-group one, incompatible with the cache.
    two_groups = torch.optim.SGD(
        [
            {"params": model.fc1.parameters(), "lr": 0.1},
            {"params": model.fc2.parameters(), "lr": 0.2},
        ]
    )
    session._optimizer = two_groups
    weight_before = model.fc1.weight.detach().clone()
    groups_before = len(two_groups.param_groups)

    with pytest.raises(TimeTravelError, match="param group count"):
        session.request_time_travel(0)

    # Nothing unwound: the model and optimizer are exactly as before.
    torch.testing.assert_close(model.fc1.weight, weight_before)
    assert len(two_groups.param_groups) == groups_before


def test_request_time_travel_allows_matching_optimizer(tmp_path: Path) -> None:
    """The matching-optimizer happy path still arms the jump (no over-reject)."""
    session, restorer, _, _, _ = _make_training(tmp_path)
    restorer.save_epoch_start(0)
    session.request_time_travel(0)  # must not raise
    assert session.mode is Mode.STEP


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
    assert status.reason is not None and "session.epochs()" in status.reason

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
        session.step_until_position(phase_index=0, epoch=2, batch_idx=0)
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


# -- flat loop API (`session.epochs()` + `session.restore_point()`) ----------


def _make_flat_session(
    tmp_path: Path, *, epochs: int = 3, enabled: bool = True
) -> tuple[
    Session,
    TinyNet,
    torch.optim.SGD,
    torch.optim.lr_scheduler.StepLR,
    Path,
]:
    """Like `_make_training`, but leaves the restorer to `session.epochs()`."""
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
    return session, model, optimizer, scheduler, tmp_path / "cache"


def test_session_epochs_runs_once_without_jumps(tmp_path: Path) -> None:
    session, model, optimizer, _, cache = _make_flat_session(tmp_path, epochs=2)
    session.detach()

    seen: list[int] = []
    for epoch in session.epochs(cache_dir=cache):
        with session.restore_point():
            seen.append(epoch)
            for _ in range(2):
                with session.batch(phase="train", epoch=epoch):
                    optimizer_train_step(model, optimizer)

    assert seen == [0, 1]
    assert session._loop_restorer is not None and session._loop_restorer.finished
    assert session._loop_restorer.cache.cached_epochs() == [0, 1]
    # The run is over, so time travel reports completed (not "no restorer").
    status = session.time_travel_status()
    assert not status.available
    assert status.reason is not None and "completed" in status.reason


def test_session_epochs_missing_restore_point_raises(tmp_path: Path) -> None:
    """Forgetting `with session.restore_point():` is caught, not silently run."""
    session, model, optimizer, _, cache = _make_flat_session(tmp_path, epochs=2)
    session.detach()

    with pytest.raises(RuntimeError, match="restore_point"):
        for epoch in session.epochs(cache_dir=cache):
            for _ in range(2):
                with session.batch(phase="train", epoch=epoch):
                    optimizer_train_step(model, optimizer)


def test_restore_point_outside_epochs_loop_raises(tmp_path: Path) -> None:
    session, _, _, _, _ = _make_flat_session(tmp_path)
    with pytest.raises(RuntimeError, match="inside a"):
        session.restore_point()


def test_session_batches_default_epoch_tracks_epochs_loop(tmp_path: Path) -> None:
    """`session.batches()` with no `epoch=` uses the epoch `epochs()` is on.

    Distinct per-epoch checkpoints prove the implicit epoch advanced — a
    stuck default (always epoch 0) would leave only `epoch_0` cached.
    """
    session, _, _, _, cache = _make_flat_session(tmp_path, epochs=3)
    session.detach()

    for _epoch in session.epochs(cache_dir=cache):
        with session.restore_point():
            for _ in session.batches([0, 1], phase="train"):
                pass

    assert session._current_epoch == 2
    assert session._loop_restorer is not None
    assert session._loop_restorer.cache.cached_epochs() == [0, 1, 2]


def test_disabled_session_epochs_is_inert(tmp_path: Path) -> None:
    session, model, optimizer, _, cache = _make_flat_session(
        tmp_path, epochs=2, enabled=False
    )

    seen: list[int] = []
    for epoch in session.epochs(cache_dir=cache):
        with session.restore_point():
            seen.append(epoch)
            for _ in range(2):
                with session.batch(phase="train", epoch=epoch):
                    optimizer_train_step(model, optimizer)

    assert seen == [0, 1]
    assert not cache.exists()  # nothing written to disk
    assert not session.time_travel_status().available


def test_session_epochs_jump_restores_and_replays_deterministically(
    tmp_path: Path,
) -> None:
    """End-to-end flat-loop jump: the replay must reproduce the original.

    Mirrors the nested-loop test, but driven by `session.epochs()` +
    `session.restore_point()`: jumping from the pause at (train, 2, 0) back
    to epoch 1 must re-yield epochs 1 and 2 with identical weights/lr, since
    model/optimizer/scheduler/RNG were all restored.
    """
    session, model, optimizer, scheduler, cache = _make_flat_session(
        tmp_path, epochs=3
    )
    epoch_log: list[tuple[int, Tensor, float]] = []

    def loop() -> None:
        for epoch in session.epochs(cache_dir=cache):
            with session.restore_point():
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

    with paused_worker(session, loop, timeout=10.0):
        session.step_until_position(phase_index=0, epoch=2, batch_idx=0)
        assert session.wait_until_paused(after_pauses=1, timeout=10.0)
        assert session._loop_restorer is not None
        assert session._loop_restorer.cache.cached_epochs() == [0, 1, 2]

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

    assert session._loop_restorer.finished
    # Attempt 1 logged epochs 0..2 (epoch 2 only began); the replay logged 1..2.
    assert [e for e, _, _ in epoch_log] == [0, 1, 2, 1, 2]
    first_ep1, first_ep2 = epoch_log[1], epoch_log[2]
    replay_ep1, replay_ep2 = epoch_log[3], epoch_log[4]
    torch.testing.assert_close(replay_ep1[1], first_ep1[1])
    assert replay_ep1[2] == first_ep1[2]
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


def test_capture_rng_includes_torch_cuda_and_mps_slots() -> None:
    rng = capture_rng()
    assert isinstance(rng["torch"], torch.Tensor)
    assert isinstance(rng["cuda"], list)
    # The MPS slot is always present (None off Apple Silicon) so the schema
    # is stable; it carries a tensor only when the backend is available.
    assert "mps" in rng
    mps_available = hasattr(torch, "mps") and torch.backends.mps.is_available()
    assert isinstance(rng["mps"], torch.Tensor) if mps_available else rng["mps"] is None


def test_restore_rng_round_trips_torch_state() -> None:
    rng = capture_rng()
    before = torch.rand(8)
    restore_rng(rng)
    after = torch.rand(8)
    assert torch.equal(before, after)


def test_restore_rng_tolerates_checkpoint_without_mps_slot() -> None:
    # Checkpoints written before MPS capture omit the key entirely.
    legacy = {"torch": torch.get_rng_state(), "cuda": []}
    restore_rng(legacy)  # must not raise


def test_restore_rng_ignores_mps_state_when_backend_unavailable() -> None:
    if hasattr(torch, "mps") and torch.backends.mps.is_available():
        pytest.skip("MPS available — cannot exercise the unavailable path")
    rng = capture_rng()
    rng["mps"] = torch.zeros(16, dtype=torch.uint8)
    restore_rng(rng)  # the bogus MPS state is skipped, not applied
