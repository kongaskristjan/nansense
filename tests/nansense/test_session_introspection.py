"""Tests for live weight, gradient, and optimizer introspection on a Session."""

from __future__ import annotations

from collections.abc import Callable

import pytest
import torch
from torch import Tensor

import nansense
from tests.nansense.helpers import (
    TinyNet,
    make_session,
    optimizer_train_step,
    paused_session,
    train_step,
)


def test_current_weights_reads_live_params_without_a_snapshot() -> None:
    """current_weights() works before any batch runs (no snapshot needed) and
    returns independent CPU clones of every parameter."""
    session, model = make_session()
    assert session.snapshot is None  # nothing captured yet

    weights = session.current_weights()
    expected = dict(model.named_parameters())
    assert set(weights) == set(expected)
    for name, tensor in weights.items():
        assert tensor.device.type == "cpu"
        assert not tensor.requires_grad
        torch.testing.assert_close(tensor, expected[name].detach().cpu())
        # A clone, not a view onto the live parameter.
        assert tensor.data_ptr() != expected[name].data_ptr()


def test_current_weights_tracks_parameter_updates() -> None:
    """A fresh call reflects in-place parameter changes (e.g. an optimizer
    step), so the weights view can refresh mid-training."""
    session, model = make_session()
    before = session.current_weights()["fc1.weight"]
    with torch.no_grad():
        model.fc1.weight.add_(1.0)
    after = session.current_weights()["fc1.weight"]
    torch.testing.assert_close(after, before + 1.0)


def test_current_weight_gradients_empty_before_backward_then_live() -> None:
    """Before any backward pass there are no gradients; after one, every
    parameter's `.grad` is returned as an independent CPU clone."""
    session, model = make_session()
    assert session.current_weight_gradients() == {}

    train_step(model)  # zero_grad + forward + backward
    grads = session.current_weight_gradients()
    expected = {n: p.grad for n, p in model.named_parameters() if p.grad is not None}
    assert set(grads) == set(expected) == {n for n, _ in model.named_parameters()}
    for name, tensor in grads.items():
        assert tensor.device.type == "cpu"
        torch.testing.assert_close(tensor, expected[name].detach().cpu())
        assert tensor.data_ptr() != expected[name].data_ptr()

    model.zero_grad(set_to_none=True)
    assert session.current_weight_gradients() == {}


@pytest.mark.parametrize(
    "make_optimizer, expected_keys",
    [
        (
            lambda m: torch.optim.SGD(m.parameters(), lr=0.1, momentum=0.9),
            {"momentum_buffer"},
        ),
        (
            lambda m: torch.optim.AdamW(m.parameters(), lr=1e-3),
            {"step", "exp_avg", "exp_avg_sq"},
        ),
    ],
)
def test_current_optimizer_state_gathers_per_parameter_entries(
    make_optimizer: Callable[[TinyNet], torch.optim.Optimizer],
    expected_keys: set[str],
) -> None:
    """State is keyed back to parameter names generically — SGD and AdamW
    need no per-optimizer code. Empty before the first step (lazy init)."""
    model = TinyNet()
    optimizer = make_optimizer(model)
    session = nansense.start(
        model, epochs=1, phases={"train": 1}, optimizer=optimizer
    )
    assert session.current_optimizer_state() == {}

    optimizer_train_step(model, optimizer)
    state = session.current_optimizer_state()
    assert set(state) == {n for n, _ in model.named_parameters()}
    entry = state["fc1.weight"]
    assert set(entry) == expected_keys
    for tensor in entry.values():
        assert tensor.device.type == "cpu"
        if tensor.ndim > 0:
            assert tensor.shape == model.fc1.weight.shape
    # Clones, independent of the live optimizer state.
    live = optimizer.state[model.fc1.weight]
    for key, tensor in entry.items():
        if isinstance(live[key], Tensor) and live[key].ndim > 0:
            assert tensor.data_ptr() != live[key].data_ptr()


def test_current_optimizer_hyperparams_numeric_only_and_eager() -> None:
    """Group hyperparams are available before any step, contain only the
    numeric knobs, and map per parameter name."""
    model = TinyNet()
    optimizer = torch.optim.SGD(
        model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4, nesterov=True
    )
    session = nansense.start(
        model, epochs=1, phases={"train": 1}, optimizer=optimizer
    )
    hp = session.current_optimizer_hyperparams()
    assert set(hp) == {n for n, _ in model.named_parameters()}
    fc1 = hp["fc1.weight"]
    assert fc1["lr"] == pytest.approx(0.1)
    assert fc1["momentum"] == pytest.approx(0.9)
    assert fc1["weight_decay"] == pytest.approx(5e-4)
    assert "params" not in fc1
    assert "nesterov" not in fc1  # bools are flags, not numeric knobs


def test_optimizer_methods_empty_without_optimizer() -> None:
    session, model = make_session()
    train_step(model)
    assert session.current_optimizer_state() == {}
    assert session.current_optimizer_hyperparams() == {}


def test_snapshot_carries_optimizer_values_when_attached() -> None:
    model = TinyNet()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    with paused_session(
        model,
        lambda m: optimizer_train_step(m, optimizer),
        phases={"train": 1},
        optimizer=optimizer,
    ) as session:
        snap = session.snapshot
        assert snap is not None
        assert set(snap.optimizer_state["fc1.weight"]) == {"momentum_buffer"}
        assert snap.optimizer_hyperparams["fc1.weight"]["lr"] == pytest.approx(0.1)


def test_start_with_scheduler_exposes_it() -> None:
    model = TinyNet()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
    session = nansense.start(
        model,
        epochs=1,
        phases={"train": 1},
        optimizer=optimizer,
        scheduler=scheduler,
    )
    assert session.optimizer is optimizer
    assert session.scheduler is scheduler
