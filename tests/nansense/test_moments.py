"""Tests for frozen debugger moments (nansense/moments.py).

Freeze a small training run at an exact batch, reload it around a fresh
model, and check the restored session shows exactly what the freezing one
did: snapshot, watch statistics (histograms, patches, weight history),
watched set, and schedule totals — plus the park loop that serves
experiments on a restored moment.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch import Tensor, nn

import nansense
from nansense.moments import MomentError
from nansense.session import Session, StatsScope
from nansense.watch import WatchSnapshot
from tests.nansense.helpers import TinyNet, run_in_thread

_EPOCHS = 2
_TRAIN_BATCHES = 3
_VAL_BATCHES = 2
_INPUT_SHAPE = (1, 8, 8)


class MomentNet(nn.Module):
    """Conv + BN + head on 8×8 grayscale images.

    BatchNorm proves buffer restore (running stats are not parameters), the
    conv feeds the extreme-patch galleries, and fx tracing exposes `relu` /
    `flatten` intermediates.
    """

    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, 4, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(4)
        self.fc = nn.Linear(4 * 8 * 8, 3)

    def forward(self, x: Tensor) -> Tensor:
        h = torch.relu(self.bn1(self.conv1(x)))
        return self.fc(h.flatten(1))


def _train_batch(model: nn.Module, optimizer: torch.optim.Optimizer) -> None:
    x = torch.randn(4, *_INPUT_SHAPE)
    y = torch.randint(0, 3, (4,))
    optimizer.zero_grad(set_to_none=True)
    loss = nn.functional.cross_entropy(model(x), y)
    loss.backward()
    optimizer.step()


def _freeze_run(moment_path: Path) -> tuple[Session, MomentNet]:
    """Train MomentNet for two epochs, freezing the last train batch.

    Mirrors the playground shape: all-layer stats, an armed freeze at the
    final train batch, and no validation after it.
    """
    torch.manual_seed(0)
    model = MomentNet()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    session = nansense.start(
        model,
        optimizer=optimizer,
        epochs=_EPOCHS,
        phases={"train": _TRAIN_BATCHES, "val": _VAL_BATCHES},
    )
    session.watch("conv1")
    session.set_stats_scope("all")
    session.freeze_moment(
        moment_path,
        phase="train",
        epoch=_EPOCHS - 1,
        batch_idx=_TRAIN_BATCHES - 1,
    )
    session.detach()
    for epoch in range(_EPOCHS):
        model.train()
        for _ in range(_TRAIN_BATCHES):
            with session.batch(phase="train", epoch=epoch):
                _train_batch(model, optimizer)
        if epoch < _EPOCHS - 1:
            model.eval()
            for _ in range(_VAL_BATCHES):
                with session.batch(phase="val", epoch=epoch), torch.no_grad():
                    model(torch.randn(4, *_INPUT_SHAPE))
    session.close()
    return session, model


def _assert_tensor_dicts_equal(
    left: dict[str, Tensor], right: dict[str, Tensor]
) -> None:
    assert set(left) == set(right)
    for name in left:
        assert torch.equal(left[name], right[name]), name


def _assert_watch_snapshots_equal(left: WatchSnapshot, right: WatchSnapshot) -> None:
    assert set(left.stats) == set(right.stats)
    for key in left.stats:
        l, r = left.stats[key], right.stats[key]
        # Frozen dataclasses of primitives — dtype included — compare exactly.
        assert l.activations == r.activations, key
        assert l.gradients == r.gradients, key
        if l.patches is None:
            assert r.patches is None, key
            continue
        assert r.patches is not None, key
        assert set(l.patches.by_type) == set(r.patches.by_type), key
        for ptype, lp in l.patches.by_type.items():
            rp = r.patches.by_type[ptype]
            for field in ("values", "patches", "heat", "top", "left"):
                assert torch.equal(getattr(lp, field), getattr(rp, field)), (
                    key,
                    ptype,
                    field,
                )
            assert lp.input_hw == rp.input_hw and lp.crop == rp.crop
    assert left.weights == right.weights


def test_freeze_and_load_round_trip(tmp_path: Path) -> None:
    moment_path = tmp_path / "moment.pt"
    session, model = _freeze_run(moment_path)
    assert moment_path.exists()

    restored_model = MomentNet()
    restored = nansense.load_moment(restored_model, moment_path)

    # The frozen batch: exact position and every snapshot tensor dict.
    original = session.snapshot
    snapshot = restored.snapshot
    assert original is not None and snapshot is not None
    assert snapshot.position == original.position
    assert (snapshot.position.phase, snapshot.position.epoch) == ("train", 1)
    assert snapshot.position.batch_idx == _TRAIN_BATCHES - 1
    _assert_tensor_dicts_equal(snapshot.activations, original.activations)
    _assert_tensor_dicts_equal(
        snapshot.activation_gradients, original.activation_gradients
    )
    _assert_tensor_dicts_equal(snapshot.weights, original.weights)
    _assert_tensor_dicts_equal(snapshot.weight_gradients, original.weight_gradients)
    assert set(snapshot.optimizer_state) == set(original.optimizer_state)
    assert restored.live_position == original.position

    # The model got the frozen parameters AND buffers (BatchNorm running
    # stats), so experiments run against the exact frozen network.
    _assert_tensor_dicts_equal(
        dict(restored_model.state_dict()), dict(model.state_dict())
    )

    # Every running statistic — histograms, patches, weight history.
    _assert_watch_snapshots_equal(restored.watch_snapshot(), session.watch_snapshot())
    # All-layer buckets exist across the run's epochs and phases.
    assert ("conv1", "train", 1) in restored.watch_snapshot().stats
    assert ("fc", "val", 0) in restored.watch_snapshot().stats

    # Watched seed, frozen stats scope, and the position label's totals.
    assert restored.watched_layers == frozenset({"conv1"})
    assert restored.stats_scope is StatsScope.NONE
    assert restored.stats_layers == frozenset(restored.layer_names)
    assert restored.schedule.epochs == _EPOCHS
    assert restored.schedule.phase_order == ["train", "val"]
    assert restored.schedule.phase_count("train") == _TRAIN_BATCHES


def test_load_moment_rejects_a_different_model(tmp_path: Path) -> None:
    moment_path = tmp_path / "moment.pt"
    _freeze_run(moment_path)
    with pytest.raises(MomentError, match="does not match"):
        nansense.load_moment(TinyNet(), moment_path)


@pytest.mark.parametrize(
    "make_file",
    [
        lambda path: None,  # missing file
        lambda path: path.write_bytes(b"not a torch file"),
        lambda path: torch.save({"kind": "something_else"}, path),
    ],
    ids=["missing", "corrupt", "wrong-kind"],
)
def test_load_moment_rejects_bad_files(tmp_path: Path, make_file) -> None:
    path = tmp_path / "moment.pt"
    make_file(path)
    with pytest.raises(MomentError):
        nansense.load_moment(MomentNet(), path)


def test_freeze_moment_is_refused_on_locked_and_disabled_sessions(
    tmp_path: Path,
) -> None:
    locked = nansense.start(MomentNet(), epochs=1, phases={"train": 1})
    locked.lock()
    locked.freeze_moment(tmp_path / "locked.pt", phase="train", epoch=0, batch_idx=0)
    assert locked._freeze_request is None

    disabled = nansense.start(MomentNet(), enabled=False)
    disabled.freeze_moment(
        tmp_path / "disabled.pt", phase="train", epoch=0, batch_idx=0
    )
    assert disabled._freeze_request is None


def test_close_reports_an_unreached_freeze_target(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    session = nansense.start(MomentNet(), epochs=1, phases={"train": 2})
    session.freeze_moment(tmp_path / "never.pt", phase="train", epoch=0, batch_idx=5)
    session.close()
    assert "never reached" in capsys.readouterr().out
    assert not (tmp_path / "never.pt").exists()


def test_parked_moment_serves_experiments_until_close(tmp_path: Path) -> None:
    moment_path = tmp_path / "moment.pt"
    _freeze_run(moment_path)
    session = nansense.load_moment(MomentNet(), moment_path)
    session.lock()
    thread = run_in_thread(session.park)
    try:
        assert session.wait_until_paused(timeout=10.0)
        assert session.is_running is False
        session.request_experiment(
            kind="deep_dream",
            layer="conv1",
            params={
                "channels": 2,
                "sample": 0,
                "steps": 2,
                "lr": 0.1,
                "diffusion": 0.1,
                "jitter": 1,
                "zoom": 1.0,
                "start": "sample",
                "clamp": True,
                "mean": (0.5,),
                "std": (0.25,),
            },
        )
        assert session.wait_for_experiment(timeout=10.0)
        result = session.experiment_result
        assert result is not None and result.error is None
        assert result.image is not None and result.image.shape == (2, *_INPUT_SHAPE)
    finally:
        session.close()
        thread.join(timeout=10.0)
    assert not thread.is_alive()
