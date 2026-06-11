"""Tests for the Session state machine and per-batch capture."""

from __future__ import annotations

import threading
from collections.abc import Callable

import pytest
import torch
from torch import Tensor, nn

import nansense
from nansense.session import Mode, Session, _BatchContext


class TinyNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(4, 8)
        self.fc2 = nn.Linear(8, 3)

    def forward(self, x: Tensor) -> Tensor:
        return self.fc2(torch.relu(self.fc1(x)))


def _train_step(model: TinyNet) -> None:
    x = torch.randn(2, 4)
    y = torch.randint(0, 3, (2,))
    model.zero_grad(set_to_none=True)
    loss = nn.functional.cross_entropy(model(x), y)
    loss.backward()


def _make_session(epochs: int = 2, phases: dict[str, int] | None = None) -> tuple[Session, TinyNet]:
    if phases is None:
        phases = {"train": 2, "val": 2}
    model = TinyNet()
    return nansense.start(model, epochs=epochs, phases=phases), model


def _run_in_thread(target) -> threading.Thread:
    thread = threading.Thread(target=target, daemon=True)
    thread.start()
    return thread


def test_detach_skips_capture_for_every_batch() -> None:
    session, model = _make_session()
    session.detach()

    captured: list[bool] = []
    for epoch in range(2):
        for phase, n in [("train", 2), ("val", 2)]:
            for _ in range(n):
                with session.batch(phase=phase, epoch=epoch) as ctx:
                    _train_step(model)
                captured.append(ctx.captured)

    assert captured == [False] * 8
    # No batch *captured* (no pause), but the default update frequency
    # (every epoch) still publishes a snapshot at each epoch's last batch.
    snap = session.snapshot
    assert snap is not None
    assert snap.position.is_last_in_epoch
    assert snap.position.epoch == 1


def test_step_run_pauses_only_at_last_overall() -> None:
    session, model = _make_session(epochs=2, phases={"train": 2, "val": 2})

    captured_positions: list[tuple[str, int, int]] = []

    def loop() -> None:
        for epoch in range(2):
            for phase, n in [("train", 2), ("val", 2)]:
                for _ in range(n):
                    with session.batch(phase=phase, epoch=epoch) as ctx:
                        _train_step(model)
                    if ctx.captured and ctx.position is not None:
                        captured_positions.append(
                            (ctx.position.phase, ctx.position.epoch, ctx.position.batch_idx)
                        )

    session.step_run()
    thread = _run_in_thread(loop)

    assert session.wait_until_paused(timeout=5)
    session.close()
    thread.join(timeout=5)
    assert not thread.is_alive()
    assert captured_positions == [("val", 1, 1)]
    assert session.snapshot is not None
    assert session.snapshot.position.is_last_overall


def test_step_mode_pauses_on_every_batch() -> None:
    session, model = _make_session(epochs=1, phases={"train": 2})

    def loop() -> None:
        for _ in range(2):
            with session.batch(phase="train", epoch=0):
                _train_step(model)

    thread = _run_in_thread(loop)

    assert session.wait_until_paused(timeout=5)
    assert session.snapshot is not None
    assert session.snapshot.position.batch_idx == 0
    session.step_batch()

    assert session.wait_until_paused(after_pauses=1, timeout=5)
    assert session.snapshot is not None
    assert session.snapshot.position.batch_idx == 1
    session.detach()

    thread.join(timeout=5)
    assert not thread.is_alive()
    assert session.pause_count == 2


def test_until_phase_change_captures_only_phase_end() -> None:
    session, model = _make_session(epochs=1, phases={"train": 3, "val": 2})

    captured_positions: list[tuple[str, int, int]] = []

    def loop() -> None:
        for phase, n in [("train", 3), ("val", 2)]:
            for _ in range(n):
                with session.batch(phase=phase, epoch=0) as ctx:
                    _train_step(model)
                if ctx.captured and ctx.position is not None:
                    captured_positions.append(
                        (ctx.position.phase, ctx.position.epoch, ctx.position.batch_idx)
                    )

    session.step_phase()
    thread = _run_in_thread(loop)

    assert session.wait_until_paused(timeout=5)
    session.detach()

    thread.join(timeout=5)
    assert not thread.is_alive()
    assert captured_positions == [("train", 0, 2)]


def test_step_until_position_captures_only_at_target() -> None:
    session, model = _make_session(epochs=2, phases={"train": 2, "val": 2})

    captured_positions: list[tuple[str, int, int]] = []

    def loop() -> None:
        for epoch in range(2):
            for phase, n in [("train", 2), ("val", 2)]:
                for _ in range(n):
                    with session.batch(phase=phase, epoch=epoch) as ctx:
                        _train_step(model)
                    if ctx.captured and ctx.position is not None:
                        captured_positions.append(
                            (ctx.position.phase, ctx.position.epoch, ctx.position.batch_idx)
                        )

    session.step_until_position(phase="val", epoch=0, batch_idx=1)
    thread = _run_in_thread(loop)

    assert session.wait_until_paused(timeout=5)
    session.detach()

    thread.join(timeout=5)
    assert not thread.is_alive()
    assert captured_positions == [("val", 0, 1)]


def test_until_epoch_change_captures_only_epoch_end() -> None:
    session, model = _make_session(epochs=2, phases={"train": 2, "val": 2})

    captured_positions: list[tuple[str, int, int]] = []

    def loop() -> None:
        for epoch in range(2):
            for phase, n in [("train", 2), ("val", 2)]:
                for _ in range(n):
                    with session.batch(phase=phase, epoch=epoch) as ctx:
                        _train_step(model)
                    if ctx.captured and ctx.position is not None:
                        captured_positions.append(
                            (ctx.position.phase, ctx.position.epoch, ctx.position.batch_idx)
                        )

    session.step_epoch()
    thread = _run_in_thread(loop)

    assert session.wait_until_paused(timeout=5)
    session.detach()

    thread.join(timeout=5)
    assert not thread.is_alive()
    assert captured_positions == [("val", 0, 1)]


def test_live_position_starts_none_and_tracks_every_batch_under_detach() -> None:
    """Detach never captures a snapshot, yet `live_position` advances on every
    batch — this is what keeps the UI top bar moving when nothing is paused."""
    session, model = _make_session(epochs=1, phases={"train": 3})
    session.detach()
    assert session.live_position is None  # nothing has entered a batch yet

    seen: list[tuple[str, int, int]] = []
    for i in range(3):
        with session.batch(phase="train", epoch=0) as ctx:
            _train_step(model)
            assert ctx.captured is False  # detach: no capture
        if i < 2:
            # No snapshot until the default update frequency (every epoch)
            # publishes one at the epoch's last batch.
            assert session.snapshot is None
        lp = session.live_position
        assert lp is not None
        seen.append((lp.phase, lp.epoch, lp.batch_idx))

    assert seen == [("train", 0, 0), ("train", 0, 1), ("train", 0, 2)]
    assert session.snapshot is not None  # the epoch-end frequency update


def test_live_position_tracks_non_captured_batches_during_step_epoch() -> None:
    """STEP EPOCH captures only the epoch's last batch, but `live_position` is
    recorded for every batch the worker passes through — including the
    non-captured ones, so the top bar advances batch-by-batch."""
    session, model = _make_session(epochs=1, phases={"train": 2, "val": 2})

    observed: list[tuple[tuple[str, int, int], bool]] = []

    def loop() -> None:
        for phase, n in [("train", 2), ("val", 2)]:
            for _ in range(n):
                with session.batch(phase=phase, epoch=0) as ctx:
                    _train_step(model)
                lp = session.live_position
                assert lp is not None
                observed.append(((lp.phase, lp.epoch, lp.batch_idx), ctx.captured))

    session.step_epoch()
    thread = _run_in_thread(loop)

    assert session.wait_until_paused(timeout=5)
    session.detach()  # release the worker paused at the epoch boundary

    thread.join(timeout=5)
    assert not thread.is_alive()

    positions = [pos for pos, _ in observed]
    captured = [cap for _, cap in observed]
    assert positions == [
        ("train", 0, 0),
        ("train", 0, 1),
        ("val", 0, 0),
        ("val", 0, 1),
    ]
    # Only the epoch's final batch was captured; live_position tracked all four.
    assert captured == [False, False, False, True]


def test_snapshot_contains_all_four_tensor_categories() -> None:
    session, model = _make_session(epochs=1, phases={"train": 1})

    def loop() -> None:
        with session.batch(phase="train", epoch=0):
            _train_step(model)

    thread = _run_in_thread(loop)

    assert session.wait_until_paused(timeout=5)
    snap = session.snapshot
    assert snap is not None

    module_names = {"fc1", "fc2"}
    param_names = {"fc1.weight", "fc1.bias", "fc2.weight", "fc2.bias"}
    assert module_names <= set(snap.activations)
    assert module_names <= set(snap.activation_gradients)
    assert param_names <= set(snap.weights)
    assert param_names <= set(snap.weight_gradients)

    expected_param_shapes = {n: p.shape for n, p in model.named_parameters()}
    for name in param_names:
        assert snap.weights[name].shape == expected_param_shapes[name]
        assert snap.weight_gradients[name].shape == expected_param_shapes[name]

    # No optimizer was passed to start(): the optimizer fields stay empty.
    assert snap.optimizer_state == {}
    assert snap.optimizer_hyperparams == {}

    session.detach()
    thread.join(timeout=5)


def test_snapshot_captures_model_input_as_x() -> None:
    session, model = _make_session(epochs=1, phases={"train": 1})

    def loop() -> None:
        with session.batch(phase="train", epoch=0):
            _train_step(model)

    thread = _run_in_thread(loop)
    assert session.wait_until_paused(timeout=5)
    snap = session.snapshot
    assert snap is not None
    assert "x" in snap.activations
    assert snap.activations["x"].shape == (2, 4)
    assert session.input_names == ["x"]

    session.detach()
    thread.join(timeout=5)


def test_input_name_comes_from_forward_signature() -> None:
    class NamedInput(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.fc = nn.Linear(4, 3)

        def forward(self, image: Tensor) -> Tensor:
            return self.fc(image)

    model = NamedInput()
    session = nansense.start(model, epochs=1, phases={"train": 1})
    assert session.input_names == ["image"]

    def loop() -> None:
        with session.batch(phase="train", epoch=0):
            x = torch.randn(2, 4)
            y = torch.randint(0, 3, (2,))
            model.zero_grad(set_to_none=True)
            loss = nn.functional.cross_entropy(model(x), y)
            loss.backward()

    thread = _run_in_thread(loop)
    assert session.wait_until_paused(timeout=5)
    snap = session.snapshot
    assert snap is not None
    assert "image" in snap.activations
    assert "x" not in snap.activations

    session.detach()
    thread.join(timeout=5)


def test_fx_mode_captures_function_call_outputs() -> None:
    """When fx.symbolic_trace succeeds, call_function results show up too."""

    class BasicBlockLike(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.conv1 = nn.Conv2d(3, 4, kernel_size=3, padding=1)
            self.bn1 = nn.BatchNorm2d(4)

        def forward(self, x: Tensor) -> Tensor:
            return torch.relu(self.bn1(self.conv1(x)))

    model = BasicBlockLike()
    session = nansense.start(model, epochs=1, phases={"train": 1})
    assert session.fx_traced
    assert "relu" in session.layer_names
    assert "conv1" in session.layer_names
    assert "bn1" in session.layer_names

    def loop() -> None:
        with session.batch(phase="train", epoch=0):
            x = torch.randn(2, 3, 4, 4)
            y = torch.randint(0, 2, (2, 4, 4))
            model.zero_grad(set_to_none=True)
            logits = model(x)
            loss = nn.functional.cross_entropy(logits, y)
            loss.backward()

    thread = _run_in_thread(loop)
    assert session.wait_until_paused(timeout=5)
    snap = session.snapshot
    assert snap is not None
    assert "relu" in snap.activations
    # relu was applied to a tensor that requires grad, so we should also
    # have captured the gradient of its output.
    assert "relu" in snap.activation_gradients

    session.detach()
    thread.join(timeout=5)


def test_fx_mode_scopes_repeated_function_ops_by_submodule() -> None:
    """Two relus in a submodule capture under distinct scope-qualified keys."""

    class Block(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.bn1 = nn.BatchNorm2d(4)
            self.bn2 = nn.BatchNorm2d(4)

        def forward(self, x: Tensor) -> Tensor:
            return torch.relu(self.bn2(torch.relu(self.bn1(x))))

    class Wrapper(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.conv = nn.Conv2d(3, 4, kernel_size=3, padding=1)
            self.block = Block()

        def forward(self, x: Tensor) -> Tensor:
            return self.block(self.conv(x))

    model = Wrapper()
    session = nansense.start(model, epochs=1, phases={"train": 1})
    assert session.fx_traced
    # The two functional relus are disambiguated by their submodule scope.
    assert "block.relu1" in session.layer_names
    assert "block.relu2" in session.layer_names
    assert "relu" not in session.layer_names  # the bare fx name is gone

    def loop() -> None:
        with session.batch(phase="train", epoch=0):
            x = torch.randn(2, 3, 4, 4)
            y = torch.randint(0, 2, (2, 4, 4))
            model.zero_grad(set_to_none=True)
            loss = nn.functional.cross_entropy(model(x), y)
            loss.backward()

    thread = _run_in_thread(loop)
    assert session.wait_until_paused(timeout=5)
    snap = session.snapshot
    assert snap is not None
    assert "block.relu1" in snap.activations
    assert "block.relu2" in snap.activations
    assert session.watch("block.relu2") is True

    session.detach()
    thread.join(timeout=5)


def test_fx_mode_restores_original_forward_after_batch() -> None:
    """The interpreter patch is reverted before the worker pauses, so the
    user's original forward is the live one whenever the batch isn't actively
    running. (The patch is only in place between __enter__ and __exit__.)"""

    class Tiny(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.fc = nn.Linear(4, 2)

        def forward(self, x: Tensor) -> Tensor:
            return torch.relu(self.fc(x))

    model = Tiny()
    session = nansense.start(model, epochs=1, phases={"train": 1})
    original_forward = model.forward
    assert "forward" not in model.__dict__

    def loop() -> None:
        with session.batch(phase="train", epoch=0):
            x = torch.randn(2, 4)
            y = torch.randint(0, 2, (2,))
            model.zero_grad(set_to_none=True)
            loss = nn.functional.cross_entropy(model(x), y)
            loss.backward()

    thread = _run_in_thread(loop)
    try:
        assert session.wait_until_paused(timeout=5)
        assert "forward" not in model.__dict__
        assert model.forward == original_forward
    finally:
        session.detach()
    thread.join(timeout=5)
    assert not thread.is_alive()
    assert "forward" not in model.__dict__


def test_fx_failure_falls_back_to_hooks() -> None:
    class Dynamic(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.fc = nn.Linear(4, 2)

        def forward(self, x: Tensor) -> Tensor:
            if x.sum() > 0:
                return self.fc(x)
            return self.fc(-x)

    model = Dynamic()
    session = nansense.start(model, epochs=1, phases={"train": 1})
    assert not session.fx_traced
    # Hook-mode layer_names: inputs + module names.
    assert session.layer_names == ["x", "fc"]


def test_layer_weights_maps_modules_to_their_parameters_fx() -> None:
    """fx mode maps each call_module node to the params under its target and
    leaves weightless nodes (relu, input) with an empty list."""

    class ConvBlock(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.conv1 = nn.Conv2d(3, 4, kernel_size=3, padding=1)
            self.bn1 = nn.BatchNorm2d(4)

        def forward(self, x: Tensor) -> Tensor:
            return torch.relu(self.bn1(self.conv1(x)))

    session = nansense.start(ConvBlock(), epochs=1, phases={"train": 1})
    assert session.fx_traced
    lw = session.layer_weights
    # Every layer name has an entry; weightless ones map to [].
    assert set(lw) == set(session.layer_names)
    assert lw["conv1"] == ["conv1.bias", "conv1.weight"]
    assert lw["bn1"] == ["bn1.bias", "bn1.weight"]
    assert lw["relu"] == []
    assert lw["x"] == []


def test_layer_weights_covers_every_parameter_exactly() -> None:
    """In fx mode the per-layer mapping accounts for all of the model's
    parameters (TinyNet's two Linear layers, weight + bias each)."""
    session, model = _make_session()
    mapped = {p for params in session.layer_weights.values() for p in params}
    assert mapped == {n for n, _ in model.named_parameters()}
    assert session.layer_weights["fc1"] == ["fc1.bias", "fc1.weight"]
    assert session.layer_weights["fc2"] == ["fc2.bias", "fc2.weight"]


def test_layer_weights_detects_functional_parameter_use() -> None:
    """A parameter used through F.conv2d (not a submodule call) is picked up
    via the get_attr node feeding the call_function node."""

    class Functional(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = nn.Parameter(torch.randn(4, 3, 3, 3))

        def forward(self, x: Tensor) -> Tensor:
            return nn.functional.conv2d(x, self.weight, padding=1)

    session = nansense.start(Functional(), epochs=1, phases={"train": 1})
    assert session.fx_traced
    assert session.layer_weights["conv2d"] == ["weight"]


def test_layer_weights_uses_module_subtree_in_hook_fallback() -> None:
    """When fx tracing fails, a module maps to every parameter in its subtree."""

    class Dynamic(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.fc = nn.Linear(4, 2)

        def forward(self, x: Tensor) -> Tensor:
            if x.sum() > 0:
                return self.fc(x)
            return self.fc(-x)

    session = nansense.start(Dynamic(), epochs=1, phases={"train": 1})
    assert not session.fx_traced
    assert session.layer_weights == {"x": [], "fc": ["fc.bias", "fc.weight"]}


def test_layer_weights_keys_index_into_snapshot_weights() -> None:
    """Every parameter named by layer_weights is present in a snapshot."""
    session, model = _make_session(epochs=1, phases={"train": 1})

    def loop() -> None:
        with session.batch(phase="train", epoch=0):
            _train_step(model)

    thread = _run_in_thread(loop)
    assert session.wait_until_paused(timeout=5)
    snap = session.snapshot
    assert snap is not None
    for params in session.layer_weights.values():
        for name in params:
            assert name in snap.weights
    session.detach()
    thread.join(timeout=5)


def test_current_weights_reads_live_params_without_a_snapshot() -> None:
    """current_weights() works before any batch runs (no snapshot needed) and
    returns independent CPU clones of every parameter."""
    session, model = _make_session()
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
    session, model = _make_session()
    before = session.current_weights()["fc1.weight"]
    with torch.no_grad():
        model.fc1.weight.add_(1.0)
    after = session.current_weights()["fc1.weight"]
    torch.testing.assert_close(after, before + 1.0)


def test_current_weight_gradients_empty_before_backward_then_live() -> None:
    """Before any backward pass there are no gradients; after one, every
    parameter's `.grad` is returned as an independent CPU clone."""
    session, model = _make_session()
    assert session.current_weight_gradients() == {}

    _train_step(model)  # zero_grad + forward + backward
    grads = session.current_weight_gradients()
    expected = {n: p.grad for n, p in model.named_parameters() if p.grad is not None}
    assert set(grads) == set(expected) == {n for n, _ in model.named_parameters()}
    for name, tensor in grads.items():
        assert tensor.device.type == "cpu"
        torch.testing.assert_close(tensor, expected[name].detach().cpu())
        assert tensor.data_ptr() != expected[name].data_ptr()

    model.zero_grad(set_to_none=True)
    assert session.current_weight_gradients() == {}


def _opt_step(model: TinyNet, optimizer: torch.optim.Optimizer) -> None:
    x = torch.randn(2, 4)
    y = torch.randint(0, 3, (2,))
    optimizer.zero_grad()
    loss = nn.functional.cross_entropy(model(x), y)
    loss.backward()
    optimizer.step()


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

    _opt_step(model, optimizer)
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
    session, model = _make_session()
    _train_step(model)
    assert session.current_optimizer_state() == {}
    assert session.current_optimizer_hyperparams() == {}


def test_snapshot_carries_optimizer_values_when_attached() -> None:
    model = TinyNet()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    session = nansense.start(
        model, epochs=1, phases={"train": 1}, optimizer=optimizer
    )

    def loop() -> None:
        with session.batch(phase="train", epoch=0):
            _opt_step(model, optimizer)

    thread = _run_in_thread(loop)
    assert session.wait_until_paused(timeout=5)
    snap = session.snapshot
    assert snap is not None
    assert set(snap.optimizer_state["fc1.weight"]) == {"momentum_buffer"}
    assert snap.optimizer_hyperparams["fc1.weight"]["lr"] == pytest.approx(0.1)
    session.detach()
    thread.join(timeout=5)


def test_snapshot_tensors_are_cpu_and_independent() -> None:
    session, model = _make_session(epochs=1, phases={"train": 1})

    def loop() -> None:
        with session.batch(phase="train", epoch=0):
            _train_step(model)

    thread = _run_in_thread(loop)
    assert session.wait_until_paused(timeout=5)
    snap = session.snapshot
    assert snap is not None

    all_tensors = {
        **snap.activations,
        **snap.activation_gradients,
        **snap.weights,
        **snap.weight_gradients,
    }
    for name, t in all_tensors.items():
        assert t.device.type == "cpu", name
        assert not t.requires_grad, name

    live_weight = dict(model.named_parameters())["fc1.weight"]
    snap_weight = snap.weights["fc1.weight"]
    assert snap_weight.data_ptr() != live_weight.data_ptr()

    session.detach()
    thread.join(timeout=5)


def test_stop_then_step_pauses_at_next_batch() -> None:
    session, model = _make_session(epochs=1, phases={"train": 3})
    session.detach()

    captured: list[bool] = []

    def loop() -> None:
        for _ in range(3):
            with session.batch(phase="train", epoch=0) as ctx:
                _train_step(model)
            captured.append(ctx.captured)

    thread = _run_in_thread(loop)
    session.stop()  # next batch boundary should pause
    assert session.wait_until_paused(timeout=5)
    session.detach()
    thread.join(timeout=5)
    assert not thread.is_alive()
    assert sum(captured) >= 1


def test_set_schedule_mid_run() -> None:
    session, model = _make_session(epochs=1, phases={"train": 2})
    session.detach()

    with session.batch(phase="train", epoch=0):
        _train_step(model)

    session.set_schedule(phases={"train": 5})
    for _ in range(4):
        with session.batch(phase="train", epoch=0):
            _train_step(model)


def test_close_releases_waiter_and_is_idempotent() -> None:
    session, model = _make_session(epochs=1, phases={"train": 1})

    def loop() -> None:
        with session.batch(phase="train", epoch=0):
            _train_step(model)

    thread = _run_in_thread(loop)
    assert session.wait_until_paused(timeout=5)
    session.close()
    session.close()
    thread.join(timeout=5)
    assert not thread.is_alive()
    assert session.closed


def test_close_before_any_batch_is_safe() -> None:
    session, model = _make_session(epochs=1, phases={"train": 1})
    session.close()
    with session.batch(phase="train", epoch=0) as ctx:
        _train_step(model)
    assert not ctx.captured
    assert session.snapshot is None


def test_unknown_phase_raises_through_context() -> None:
    session, model = _make_session(epochs=1, phases={"train": 1})
    session.detach()
    with pytest.raises(ValueError, match="unknown phase"):
        with session.batch(phase="bogus", epoch=0):
            _train_step(model)


def test_user_exception_does_not_pause() -> None:
    session, model = _make_session(epochs=1, phases={"train": 1})
    # default mode is STEP: would normally pause, but user exception should
    # propagate without us blocking the worker.

    class Boom(Exception):
        pass

    def loop() -> None:
        with pytest.raises(Boom):
            with session.batch(phase="train", epoch=0):
                raise Boom

    thread = _run_in_thread(loop)
    thread.join(timeout=5)
    assert not thread.is_alive()
    assert session.snapshot is None
    assert session.mode == Mode.STEP


def test_hooks_removed_after_each_batch() -> None:
    session, model = _make_session(epochs=1, phases={"train": 2})

    def loop() -> None:
        for _ in range(2):
            with session.batch(phase="train", epoch=0):
                _train_step(model)

    thread = _run_in_thread(loop)
    assert session.wait_until_paused(timeout=5)
    # Between pauses, hooks should have been removed even though we still hold
    # the activations from the previous batch on the snapshot.
    assert session._hook_handles == []  # type: ignore[reportPrivateUsage]
    session.detach()
    thread.join(timeout=5)


def test_watch_accepts_any_layer_name_and_rejects_unknown() -> None:
    session, model = _make_session()
    # Modules, fx intermediates, and the input are all in layer_names.
    assert session.watch("fc1") is True
    assert session.watch("relu") is True  # fx intermediate
    assert session.watch("x") is True  # graph input
    assert session.watch("bogus") is False
    assert session.watched_layers == frozenset({"fc1", "relu", "x"})


def test_watch_accumulates_stats_while_detached() -> None:
    """Stats accumulate on every batch even when detach() means no captures."""
    session, model = _make_session(epochs=1, phases={"train": 3})
    session.watch("fc1")
    session.detach()

    for _ in range(3):
        with session.batch(phase="train", epoch=0) as ctx:
            _train_step(model)
            assert ctx.captured is False  # detach mode

    snap = session.watch_snapshot()
    assert ("fc1", "train", 0) in snap.stats
    layer_stats = snap.stats[("fc1", "train", 0)]
    # Three forward passes × 2 samples × 8 output features = 48 elements.
    assert layer_stats.activations.n == 48
    # Three backward passes' gradients aggregated too.
    assert layer_stats.gradients.n == 48


def test_watch_accumulates_stats_alongside_capture() -> None:
    """Stats also accumulate when the batch is being captured for the snapshot."""
    session, model = _make_session(epochs=1, phases={"train": 1})
    session.watch("fc1")

    def loop() -> None:
        with session.batch(phase="train", epoch=0):
            _train_step(model)

    thread = _run_in_thread(loop)
    assert session.wait_until_paused(timeout=5)
    snap = session.watch_snapshot()
    assert ("fc1", "train", 0) in snap.stats
    assert snap.stats[("fc1", "train", 0)].activations.n == 16
    session.detach()
    thread.join(timeout=5)


def test_unwatch_drops_collected_stats() -> None:
    session, model = _make_session(epochs=1, phases={"train": 1})
    session.watch("fc1")
    session.detach()
    with session.batch(phase="train", epoch=0):
        _train_step(model)
    assert ("fc1", "train", 0) in session.watch_snapshot().stats

    session.unwatch("fc1")
    assert session.watched_layers == frozenset()
    assert session.watch_snapshot().stats == {}


class TinyConvNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(3, 2, kernel_size=3, padding=1)
        self.fc = nn.Linear(2 * 8 * 8, 3)

    def forward(self, x: Tensor) -> Tensor:
        return self.fc(torch.relu(self.conv(x)).flatten(1))


def test_watch_gathers_patches_for_image_inputs() -> None:
    model = TinyConvNet()
    session = nansense.start(model, epochs=1, phases={"train": 2})
    session.watch("conv")
    session.detach()

    for _ in range(2):
        with session.batch(phase="train", epoch=0):
            x = torch.randn(2, 3, 8, 8)
            y = torch.randint(0, 3, (2,))
            model.zero_grad(set_to_none=True)
            nn.functional.cross_entropy(model(x), y).backward()

    patches = session.watch_snapshot().stats[("conv", "train", 0)].patches
    assert patches is not None
    tp = patches.by_type["max_pixel"]
    assert tp.values.shape[0] == 2  # one row per conv channel
    # Two batches × 2 samples = 4 candidates per channel made it in.
    assert torch.isfinite(tp.values[:, :4]).all()


def test_watch_skips_patches_without_image_input() -> None:
    session, model = _make_session(epochs=1, phases={"train": 1})
    session.watch("fc1")
    session.detach()
    with session.batch(phase="train", epoch=0):
        _train_step(model)
    # TinyNet's input is 2D — stats accumulate but no patches are gathered.
    layer_stats = session.watch_snapshot().stats[("fc1", "train", 0)]
    assert layer_stats.activations.n > 0
    assert layer_stats.patches is None


def test_watching_uses_full_capture_machinery_under_detach() -> None:
    """Watching engages the same hook path as capture, so fx intermediates work."""
    session, model = _make_session(epochs=1, phases={"train": 1})
    session.watch("fc1")
    session.detach()

    with session.batch(phase="train", epoch=0):
        # The exact handle count depends on fx-vs-hook mode; what matters
        # is that *something* got installed (capture machinery is live).
        # TinyNet (no Module-only ops between modules) traces cleanly, so
        # fx mode patches forward without registering RemovableHandles
        # — `_original_forward` is the signal there.
        installed = (
            len(session._hook_handles) > 0  # type: ignore[reportPrivateUsage]
            or session._original_forward is not None  # type: ignore[reportPrivateUsage]
        )
        assert installed
        _train_step(model)
    assert session._hook_handles == []  # type: ignore[reportPrivateUsage]
    assert session._original_forward is None  # type: ignore[reportPrivateUsage]


def test_watch_fx_intermediate_accumulates_stats() -> None:
    """Watching an fx-traced intermediate op (`relu`) produces stats."""
    session, model = _make_session(epochs=1, phases={"train": 1})
    assert session.fx_traced
    assert "relu" in session.layer_names
    session.watch("relu")
    session.detach()

    with session.batch(phase="train", epoch=0):
        _train_step(model)

    snap = session.watch_snapshot()
    assert ("relu", "train", 0) in snap.stats
    relu_stats = snap.stats[("relu", "train", 0)].activations
    # ReLU output is non-negative — the histogram's negative half is empty.
    from nansense.watch import ZERO_BIN
    neg_count = sum(relu_stats.hist[:ZERO_BIN])
    assert neg_count == 0
    assert relu_stats.n == 16  # batch 2 × 8 hidden features


def test_watch_input_x_accumulates_stats() -> None:
    """Watching the graph input `x` produces stats."""
    session, model = _make_session(epochs=1, phases={"train": 1})
    assert "x" in session.layer_names
    session.watch("x")
    session.detach()

    with session.batch(phase="train", epoch=0):
        _train_step(model)

    snap = session.watch_snapshot()
    assert ("x", "train", 0) in snap.stats
    x_stats = snap.stats[("x", "train", 0)].activations
    assert x_stats.n == 8  # batch 2 × 4 input features


def test_start_enabled_by_default() -> None:
    model = TinyNet()
    session = nansense.start(model, epochs=1, phases={"train": 1})
    assert session.enabled is True
    assert session.fx_traced is True  # TinyNet traces cleanly
    assert session.layer_names  # non-empty


def test_disabled_session_skips_trace_and_name_discovery() -> None:
    """`enabled=False` skips the fx trace and leaves the name lists empty."""
    model = TinyNet()
    session = nansense.start(model, epochs=1, phases={"train": 1}, enabled=False)

    assert session.enabled is False
    assert session.fx_traced is False  # would be True if we had traced
    assert session.input_names == []
    assert session.layer_names == []
    assert session.layer_weights == {}
    # Nothing is watchable on a disabled session.
    assert session.watch("anything") is False
    assert session.watched_layers == frozenset()


def test_disabled_session_batch_captures_nothing_and_never_pauses() -> None:
    """A disabled batch runs the user body but installs no hooks and never blocks.

    The whole loop runs on the main thread: if a disabled batch paused (as an
    enabled STEP-mode batch would), this test would hang instead of completing.
    """
    model = TinyNet()
    session = nansense.start(model, epochs=1, phases={"train": 2}, enabled=False)

    for _ in range(2):
        with session.batch(phase="train", epoch=0) as ctx:
            _train_step(model)
            assert isinstance(ctx, _BatchContext)
            assert ctx.captured is False
            assert ctx.position is None

    assert session.snapshot is None
    assert session.live_position is None
    assert session.pause_count == 0
    # forward was never patched, no hooks left installed.
    assert "forward" not in model.__dict__


def test_disabled_session_does_not_advance_the_schedule() -> None:
    """Disabled batches skip `schedule.advance`, so the declared batch count
    is never enforced — an enabled session would raise on the 2nd batch here."""
    model = TinyNet()
    session = nansense.start(model, epochs=1, phases={"train": 1}, enabled=False)

    for _ in range(5):  # far more than the single declared batch
        with session.batch(phase="train", epoch=0):
            _train_step(model)


def test_batches_runs_each_item_inside_a_batch_context() -> None:
    session, _ = _make_session(epochs=1, phases={"train": 3})
    session.detach()

    items = ["a", "b", "c"]
    positions: list[tuple[str, int, int]] = []
    for item in session.batches(items, phase="train", epoch=0):
        live = session.live_position
        assert live is not None
        positions.append((live.phase, live.epoch, live.batch_idx))

    # The body observed each batch's own position: it ran inside the context.
    assert positions == [("train", 0, 0), ("train", 0, 1), ("train", 0, 2)]
    # The schedule advanced three times — a fourth batch overflows.
    with pytest.raises(ValueError, match="more batches than declared"):
        with session.batch(phase="train", epoch=0):
            pass


def test_batches_yields_loader_items_unchanged_when_disabled() -> None:
    model = TinyNet()
    session = nansense.start(model, epochs=1, phases={"train": 2}, enabled=False)
    assert list(session.batches([1, 2, 3], phase="train", epoch=0)) == [1, 2, 3]


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
