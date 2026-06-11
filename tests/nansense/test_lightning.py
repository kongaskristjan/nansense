"""Tests for the PyTorch Lightning integration (`nansense.lightning`)."""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
import torch
from lightning.pytorch import LightningModule, Trainer, seed_everything
from lightning.pytorch.utilities.types import OptimizerLRScheduler
from torch import Tensor, nn
from torch.utils.data import DataLoader, TensorDataset

from nansense.lightning import (
    NansenseCallback,
    _LightningCkptCache,
    fit_with_time_travel,
)
from nansense.restore import TimeTravelError
from nansense.session import Session


class BoringModule(LightningModule):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 2))

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)

    def training_step(self, batch: tuple[Tensor, Tensor], batch_idx: int) -> Tensor:
        x, y = batch
        return nn.functional.mse_loss(self(x), y)

    def validation_step(self, batch: tuple[Tensor, Tensor], batch_idx: int) -> Tensor:
        x, y = batch
        return nn.functional.mse_loss(self(x), y)

    def configure_optimizers(self) -> OptimizerLRScheduler:
        optimizer = torch.optim.SGD(self.parameters(), lr=0.1, momentum=0.9)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}


def _loader(*, shuffle: bool) -> DataLoader[tuple[Tensor, ...]]:
    """Two batches of four samples, drawn from the current global RNG."""
    x = torch.randn(8, 4)
    y = torch.randn(8, 2)
    return DataLoader(TensorDataset(x, y), batch_size=4, shuffle=shuffle)


def _make_trainer(root: Path, **overrides: Any) -> Trainer:
    kwargs: dict[str, Any] = {
        "accelerator": "cpu",
        "devices": 1,
        "max_epochs": 3,
        "logger": False,
        "enable_checkpointing": False,
        "enable_progress_bar": False,
        "enable_model_summary": False,
        "num_sanity_val_steps": 0,
        "default_root_dir": root,
    }
    kwargs.update(overrides)
    return Trainer(**kwargs)


def _wait_for_session(callback: NansenseCallback, *, timeout: float = 30.0) -> Session:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        session = callback.session
        if session is not None:
            return session
        time.sleep(0.01)
    raise AssertionError("session was not created within the timeout")


class _Driver:
    """Run UI-side driving logic in a thread; re-raise its failure in the test.

    On a driver failure the session is closed so the paused training thread
    (blocked inside `trainer.fit` on the main thread) is released instead of
    hanging the test.
    """

    def __init__(self, callback: NansenseCallback, target: Callable[[], None]) -> None:
        self._callback = callback
        self.error: BaseException | None = None
        self._thread = threading.Thread(target=self._run, args=(target,), daemon=True)
        self._thread.start()

    def _run(self, target: Callable[[], None]) -> None:
        try:
            target()
        except BaseException as e:  # noqa: BLE001 - re-raised in join()
            self.error = e
            session = self._callback.session
            if session is not None:
                session.close()

    def join(self, *, timeout: float = 60.0) -> None:
        self._thread.join(timeout)
        assert not self._thread.is_alive(), "driver thread did not finish"
        if self.error is not None:
            raise self.error


def test_callback_pauses_steps_and_completes(tmp_path: Path) -> None:
    seed_everything(0)
    module = BoringModule()
    train, val = _loader(shuffle=True), _loader(shuffle=False)
    callback = NansenseCallback(model="net")
    trainer = _make_trainer(
        tmp_path, max_epochs=2, num_sanity_val_steps=2, callbacks=[callback]
    )

    positions: list[tuple[str, int, int]] = []
    phases: list[dict[str, int]] = []

    def drive() -> None:
        session = _wait_for_session(callback)
        assert session.wait_until_paused(after_pauses=0, timeout=60.0)
        snap = session.snapshot
        assert snap is not None
        positions.append(
            (snap.position.phase, snap.position.epoch, snap.position.batch_idx)
        )
        phases.append(session.schedule.phases)
        # The forward ran through the fx-traced `net`, and Lightning's
        # batch-end hook fires after backward, so gradients are populated.
        assert "0" in snap.activations
        assert snap.weight_gradients
        session.step_batch()
        assert session.wait_until_paused(after_pauses=1, timeout=60.0)
        snap = session.snapshot
        assert snap is not None
        positions.append(
            (snap.position.phase, snap.position.epoch, snap.position.batch_idx)
        )
        session.detach()

    driver = _Driver(callback, drive)
    trainer.fit(module, train_dataloaders=train, val_dataloaders=val)
    driver.join()

    # The sanity check ran first (num_sanity_val_steps=2) but produced no
    # batches: the first pause is train batch 0, with both phases declared.
    assert positions == [("train", 0, 0), ("train", 0, 1)]
    assert phases == [{"train": 2, "val": 2}]
    session = callback.session
    assert session is not None
    assert session.closed  # on_fit_end closed it; the UI would stay up


def test_disabled_callback_is_inert(tmp_path: Path) -> None:
    seed_everything(0)
    module = BoringModule()
    callback = NansenseCallback(enabled=False)
    trainer = _make_trainer(tmp_path, max_epochs=1, callbacks=[callback])
    # Runs straight through: no pause, no captures, nothing to drive.
    trainer.fit(module, train_dataloaders=_loader(shuffle=False))
    session = callback.session
    assert session is not None
    assert not session.enabled
    assert session.snapshot is None

    with pytest.raises(RuntimeError, match="fresh NansenseCallback"):
        fit_with_time_travel(
            lambda: _make_trainer(tmp_path),
            module,
            callback=callback,
            train_dataloaders=_loader(shuffle=False),
            cache_dir=tmp_path / "cache",
        )


def test_mid_epoch_validation_rejected(tmp_path: Path) -> None:
    seed_everything(0)
    module = BoringModule()
    callback = NansenseCallback()
    trainer = _make_trainer(
        tmp_path, max_epochs=1, val_check_interval=0.5, callbacks=[callback]
    )
    with pytest.raises(RuntimeError, match="mid-epoch validation"):
        trainer.fit(
            module,
            train_dataloaders=_loader(shuffle=False),
            val_dataloaders=_loader(shuffle=False),
        )


def test_lightning_ckpt_cache_load_errors(tmp_path: Path) -> None:
    cache = _LightningCkptCache(tmp_path)
    with pytest.raises(TimeTravelError, match="no cached checkpoint"):
        cache.load(0)
    (tmp_path / "epoch_0.ckpt").write_bytes(b"not a checkpoint")
    assert cache.cached_epochs() == [0]
    with pytest.raises(TimeTravelError, match="failed to load"):
        cache.load(0)


def test_fit_with_time_travel_jump_and_deterministic_replay(tmp_path: Path) -> None:
    """End-to-end: jump from epoch 2 back to epoch 1, replay to completion.

    A straight 3-epoch run with the same seed is the ground truth: the
    time-traveled run trains epochs 0-1, begins epoch 2, jumps back to 1,
    and replays 1-2. With Lightning restoring model/optimizer/scheduler and
    the callback restoring RNG (shuffling included), the final weights must
    match the straight run exactly.
    """
    seed_everything(123)
    baseline = BoringModule()
    trainer = _make_trainer(tmp_path / "baseline", num_sanity_val_steps=2)
    trainer.fit(
        baseline,
        train_dataloaders=_loader(shuffle=True),
        val_dataloaders=_loader(shuffle=False),
    )

    seed_everything(123)
    module = BoringModule()
    train, val = _loader(shuffle=True), _loader(shuffle=False)
    callback = NansenseCallback(model="net")
    jump_positions: list[tuple[str, int, int]] = []

    def drive() -> None:
        session = _wait_for_session(callback)
        assert session.wait_until_paused(after_pauses=0, timeout=60.0)
        session.step_until_position(phase="train", epoch=2, batch_idx=0)
        assert session.wait_until_paused(after_pauses=1, timeout=60.0)
        # Epoch 0 was checkpointed at fit start, epochs 1 and 2 at the
        # ends of epochs 0 and 1.
        status = session.time_travel_status()
        assert status.available
        assert status.cached_epochs == [0, 1, 2]
        pauses = session.pause_count
        session.request_time_travel(1)
        assert session.wait_until_paused(after_pauses=pauses, timeout=60.0)
        snap = session.snapshot
        assert snap is not None
        jump_positions.append(
            (snap.position.phase, snap.position.epoch, snap.position.batch_idx)
        )
        session.detach()

    driver = _Driver(callback, drive)
    fit_with_time_travel(
        lambda: _make_trainer(tmp_path / "tt", num_sanity_val_steps=2),
        module,
        callback=callback,
        train_dataloaders=train,
        val_dataloaders=val,
        cache_dir=tmp_path / "cache",
    )
    driver.join()

    # The replayed epoch paused at its first batch, like a fresh session.
    assert jump_positions == [("train", 1, 0)]
    session = callback.session
    assert session is not None
    assert session.closed
    for p_base, p_tt in zip(
        baseline.parameters(), module.parameters(), strict=True
    ):
        torch.testing.assert_close(p_tt, p_base)
