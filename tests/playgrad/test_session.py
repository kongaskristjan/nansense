"""Tests for the Session state machine and per-batch capture."""

from __future__ import annotations

import threading

import pytest
import torch
from torch import Tensor, nn

import playgrad
from playgrad.session import Mode, Session


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
    return playgrad.start(model, epochs=epochs, phases=phases), model


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
    assert session.snapshot is None


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
    for _ in range(3):
        with session.batch(phase="train", epoch=0) as ctx:
            _train_step(model)
            assert ctx.captured is False  # detach: no capture
        assert session.snapshot is None  # ...and no snapshot ever published
        lp = session.live_position
        assert lp is not None
        seen.append((lp.phase, lp.epoch, lp.batch_idx))

    assert seen == [("train", 0, 0), ("train", 0, 1), ("train", 0, 2)]


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
    session = playgrad.start(model, epochs=1, phases={"train": 1})
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
    session = playgrad.start(model, epochs=1, phases={"train": 1})
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
    session = playgrad.start(model, epochs=1, phases={"train": 1})
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
    session = playgrad.start(model, epochs=1, phases={"train": 1})
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
    session = playgrad.start(model, epochs=1, phases={"train": 1})
    assert not session.fx_traced
    # Hook-mode layer_names: inputs + module names.
    assert session.layer_names == ["x", "fc"]


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
    from playgrad.watch import ZERO_BIN
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
