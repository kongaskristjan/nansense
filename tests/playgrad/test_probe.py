"""Tests for probe runs: pinned-batch forwards between batches."""

from __future__ import annotations

import threading
from collections.abc import Callable

import pytest
import torch
from torch import Tensor, nn

import playgrad
from playgrad.session import Session


class BnDropNet(nn.Module):
    """Conv + BatchNorm + Dropout: exercises mode isolation (stats, RNG)."""

    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(3, 4, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(4)
        self.drop = nn.Dropout(p=0.5)
        self.fc = nn.Linear(4 * 4 * 4, 2)

    def forward(self, x: Tensor) -> Tensor:
        h = torch.relu(self.bn(self.conv(x)))
        h = self.drop(h)
        return self.fc(h.flatten(1))


class DynamicNet(nn.Module):
    """Data-dependent control flow: forces the hook-fallback capture path."""

    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(4, 2)

    def forward(self, x: Tensor) -> Tensor:
        if x.sum() > 0:
            return self.fc(x)
        return self.fc(-x)


def _bn_drop_step(model: BnDropNet) -> None:
    x = torch.randn(2, 3, 4, 4)
    y = torch.randint(0, 2, (2,))
    model.zero_grad(set_to_none=True)
    loss = nn.functional.cross_entropy(model(x), y)
    loss.backward()


def _dynamic_step(model: DynamicNet) -> None:
    x = torch.randn(2, 4)
    y = torch.randint(0, 2, (2,))
    model.zero_grad(set_to_none=True)
    loss = nn.functional.cross_entropy(model(x), y)
    loss.backward()


def _paused_session[M: nn.Module](
    model: M, step: Callable[[M], None], *, batches: int = 2
) -> tuple[Session, threading.Thread]:
    """Start a session, run `batches` STEP-mode batches on a thread, and
    return once the worker is paused on the first capture."""
    session = playgrad.start(model, epochs=1, phases={"train": batches})

    def loop() -> None:
        for _ in range(batches):
            with session.batch(phase="train", epoch=0):
                step(model)

    thread = threading.Thread(target=loop, daemon=True)
    thread.start()
    assert session.wait_until_paused(timeout=5)
    return session, thread


def _finish(session: Session, thread: threading.Thread) -> None:
    session.detach()
    thread.join(timeout=5)
    assert not thread.is_alive()


def test_pin_before_any_snapshot_returns_false() -> None:
    session = playgrad.start(BnDropNet(), epochs=1, phases={"train": 1})
    assert session.pin_current_batch() is False
    assert session.is_pinned is False
    assert session.probe_result is None


def test_pin_on_disabled_session_returns_false() -> None:
    session = playgrad.start(
        BnDropNet(), epochs=1, phases={"train": 1}, enabled=False
    )
    assert session.pin_current_batch() is False


def test_pin_while_paused_runs_probe_without_stepping() -> None:
    model = BnDropNet()
    session, thread = _paused_session(model, _bn_drop_step)
    snap = session.snapshot
    assert snap is not None
    assert session.pause_count == 1

    assert session.pin_current_batch() is True
    assert session.wait_for_probe(timeout=5)
    probe = session.probe_result
    assert probe is not None
    assert session.probe_error is None
    # The probe ran on the paused training thread: no extra pause happened.
    assert session.pause_count == 1
    # The pinned input is the snapshot's input batch, position included.
    torch.testing.assert_close(probe.input, snap.activations["x"])
    assert session.pinned_position == snap.position
    # Default mode is eval.
    assert probe.mode == "eval"
    # Every layer the snapshot knows shows up in the probe too, as
    # independent CPU clones.
    assert set(probe.activations) == set(session.layer_names)
    for name, tensor in probe.activations.items():
        assert tensor.device.type == "cpu", name
        assert not tensor.requires_grad, name

    _finish(session, thread)


def test_probe_reruns_on_every_capture_while_pinned() -> None:
    model = BnDropNet()
    session, thread = _paused_session(model, _bn_drop_step)
    assert session.pin_current_batch() is True
    assert session.wait_for_probe(timeout=5)
    first = session.probe_result
    assert first is not None
    first_count = session.probe_count

    session.step_batch()
    assert session.wait_until_paused(after_pauses=1, timeout=5)
    assert session.probe_count == first_count + 1
    second = session.probe_result
    assert second is not None and second is not first
    # Same pinned input, fresh activations against the stepped weights.
    torch.testing.assert_close(second.input, first.input)

    _finish(session, thread)


@pytest.mark.parametrize(
    "mode, model_training, dropout_active",
    [
        ("eval", True, False),
        ("train", False, True),
        ("unchanged", True, True),
        ("unchanged", False, False),
    ],
)
def test_probe_mode_controls_dropout(
    mode: str, model_training: bool, dropout_active: bool
) -> None:
    """In eval, dropout is identity (drop output == relu output); in train it
    rescales and zeroes; "unchanged" follows the model's current flags."""
    model = BnDropNet()
    session, thread = _paused_session(model, _bn_drop_step)
    model.train(model_training)  # safe: the training thread is paused
    session.set_probe_mode(mode)
    assert session.pin_current_batch() is True
    assert session.wait_for_probe(timeout=5)
    probe = session.probe_result
    assert probe is not None and probe.mode == mode

    identical = torch.equal(probe.activations["drop"], probe.activations["relu"])
    assert identical != dropout_active
    # The probe restored the flags it flipped.
    assert all(m.training == model_training for m in model.modules())

    _finish(session, thread)


@pytest.mark.parametrize("mode", ["unchanged", "eval", "train"])
def test_probe_leaves_no_trace_on_training_state(mode: str) -> None:
    """Probes restore BN buffers and the RNG stream in every mode — even
    "train", where the forward itself updates running stats in place."""
    model = BnDropNet()
    session, thread = _paused_session(model, _bn_drop_step)
    session.set_probe_mode(mode)

    running_mean = model.bn.running_mean
    running_var = model.bn.running_var
    batches_tracked = model.bn.num_batches_tracked
    assert running_mean is not None and running_var is not None
    assert batches_tracked is not None
    mean_before = running_mean.clone()
    var_before = running_var.clone()
    tracked_before = batches_tracked.clone()
    rng_before = torch.get_rng_state()
    flags_before = [m.training for m in model.modules()]

    assert session.pin_current_batch() is True
    assert session.wait_for_probe(timeout=5)
    assert session.probe_error is None

    torch.testing.assert_close(running_mean, mean_before)
    torch.testing.assert_close(running_var, var_before)
    assert torch.equal(batches_tracked, tracked_before)
    assert torch.equal(torch.get_rng_state(), rng_before)
    assert [m.training for m in model.modules()] == flags_before

    _finish(session, thread)


def test_probe_works_in_hook_fallback_mode() -> None:
    model = DynamicNet()
    session, thread = _paused_session(model, _dynamic_step)
    assert not session.fx_traced

    assert session.pin_current_batch() is True
    assert session.wait_for_probe(timeout=5)
    probe = session.probe_result
    assert probe is not None
    assert set(probe.activations) == {"x", "fc"}
    # The temporary probe hooks were removed again.
    assert session._hook_handles == []  # type: ignore[reportPrivateUsage]
    assert len(model._forward_hooks) == 0
    assert len(model._forward_pre_hooks) == 0

    _finish(session, thread)


def test_unpin_clears_probe_result() -> None:
    model = BnDropNet()
    session, thread = _paused_session(model, _bn_drop_step)
    assert session.pin_current_batch() is True
    assert session.wait_for_probe(timeout=5)
    assert session.probe_result is not None

    session.unpin_batch()
    assert session.is_pinned is False
    assert session.probe_result is None
    assert session.pinned_position is None

    _finish(session, thread)


def test_set_probe_mode_rejects_unknown_mode() -> None:
    session = playgrad.start(BnDropNet(), epochs=1, phases={"train": 1})
    with pytest.raises(ValueError, match="unknown probe mode"):
        session.set_probe_mode("bogus")


def test_failing_probe_publishes_error_not_crash() -> None:
    model = BnDropNet()
    session, thread = _paused_session(model, _bn_drop_step)
    assert session.pin_current_batch() is True
    assert session.wait_for_probe(timeout=5)

    # Sabotage the pinned input with an incompatible shape; the next probe
    # must fail gracefully into `probe_error` instead of killing the worker.
    session._pinned_input = torch.randn(2, 3, 9, 9)  # type: ignore[reportPrivateUsage]
    count = session.probe_count
    with session._cv:  # type: ignore[reportPrivateUsage]
        session._request_probe_locked()  # type: ignore[reportPrivateUsage]
    assert session.wait_for_probe(after_count=count, timeout=5)
    assert session.probe_error is not None
    # The worker is still paused and responsive.
    assert session.pause_count == 1
    session.step_batch()
    assert session.wait_until_paused(after_pauses=1, timeout=5)

    _finish(session, thread)
