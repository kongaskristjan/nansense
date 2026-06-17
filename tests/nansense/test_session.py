"""Tests for the Session state machine and per-batch capture."""

from __future__ import annotations

import threading
from collections.abc import Callable

import pytest
from torch import Tensor, nn

import nansense
from nansense.session import Mode, Session
from tests.nansense.helpers import (
    TinyNet,
    make_session,
    paused_session,
    paused_worker,
    run_in_thread,
    train_step,
)


def _capture_positions_loop(
    session: Session,
    model: TinyNet,
    out: list[tuple[str, int, int]],
    *,
    epochs: int,
    phases: dict[str, int],
) -> Callable[[], None]:
    """A worker target that runs the schedule and records captured positions."""

    def loop() -> None:
        for epoch in range(epochs):
            for phase, n in phases.items():
                for _ in range(n):
                    with session.batch(phase=phase, epoch=epoch) as ctx:
                        train_step(model)
                    if ctx.captured and ctx.position is not None:
                        out.append(
                            (ctx.position.phase, ctx.position.epoch, ctx.position.batch_idx)
                        )

    return loop


def test_detach_skips_capture_for_every_batch() -> None:
    session, model = make_session()
    session.detach()

    captured: list[bool] = []
    for epoch in range(2):
        for phase, n in [("train", 2), ("val", 2)]:
            for _ in range(n):
                with session.batch(phase=phase, epoch=epoch) as ctx:
                    train_step(model)
                captured.append(ctx.captured)

    assert captured == [False] * 8
    # No batch *captured* (no pause), but the default update frequency
    # (every epoch) still publishes a snapshot at each epoch's last batch.
    snap = session.snapshot
    assert snap is not None
    assert snap.position.is_last_in_epoch
    assert snap.position.epoch == 1


@pytest.mark.parametrize(
    ("step", "epochs", "phases", "expected", "expect_last_overall"),
    [
        pytest.param(
            lambda s: s.step_run(),
            2,
            {"train": 2, "val": 2},
            [("val", 1, 1)],
            True,
            id="run",
        ),
        pytest.param(
            lambda s: s.step_phase(),
            1,
            {"train": 3, "val": 2},
            [("train", 0, 2)],
            False,
            id="until_phase_change",
        ),
        pytest.param(
            lambda s: s.step_until_position(phase="val", epoch=0, batch_idx=1),
            2,
            {"train": 2, "val": 2},
            [("val", 0, 1)],
            False,
            id="until_position",
        ),
        pytest.param(
            lambda s: s.step_epoch(),
            2,
            {"train": 2, "val": 2},
            [("val", 0, 1)],
            False,
            id="until_epoch_change",
        ),
    ],
)
def test_step_modes_capture_only_at_target(
    step: Callable[[Session], None],
    epochs: int,
    phases: dict[str, int],
    expected: list[tuple[str, int, int]],
    expect_last_overall: bool,
) -> None:
    """Each step mode (RUN / PHASE / UNTIL POSITION / EPOCH) captures exactly
    the batch it targets and nothing else."""
    session, model = make_session(epochs=epochs, phases=phases)

    captured_positions: list[tuple[str, int, int]] = []
    loop = _capture_positions_loop(
        session, model, captured_positions, epochs=epochs, phases=phases
    )

    step(session)
    with paused_worker(session, loop):
        # RUN pauses at the final batch: close() releases the worker there.
        # The other modes pause at a boundary; exit releases them via detach.
        if expect_last_overall:
            session.close()
    assert captured_positions == expected
    if expect_last_overall:
        assert session.snapshot is not None
        assert session.snapshot.position.is_last_overall


def test_step_mode_pauses_on_every_batch() -> None:
    model = TinyNet()
    with paused_session(model, phases={"train": 2}) as session:
        assert session.snapshot is not None
        assert session.snapshot.position.batch_idx == 0
        session.step_batch()

        assert session.wait_until_paused(after_pauses=1, timeout=5)
        assert session.snapshot is not None
        assert session.snapshot.position.batch_idx == 1
    assert session.pause_count == 2


def test_live_position_starts_none_and_tracks_every_batch_under_detach() -> None:
    """Detach never captures a snapshot, yet `live_position` advances on every
    batch — this is what keeps the UI top bar moving when nothing is paused."""
    session, model = make_session(epochs=1, phases={"train": 3})
    session.detach()
    assert session.live_position is None  # nothing has entered a batch yet

    seen: list[tuple[str, int, int]] = []
    for i in range(3):
        with session.batch(phase="train", epoch=0) as ctx:
            train_step(model)
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
    session, model = make_session(epochs=1, phases={"train": 2, "val": 2})

    observed: list[tuple[tuple[str, int, int], bool]] = []

    def loop() -> None:
        for phase, n in [("train", 2), ("val", 2)]:
            for _ in range(n):
                with session.batch(phase=phase, epoch=0) as ctx:
                    train_step(model)
                lp = session.live_position
                assert lp is not None
                observed.append(((lp.phase, lp.epoch, lp.batch_idx), ctx.captured))

    session.step_epoch()
    with paused_worker(session, loop):
        pass  # exit releases the worker paused at the epoch boundary

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
    model = TinyNet()
    with paused_session(model, phases={"train": 1}) as session:
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


def test_snapshot_captures_model_input_as_x() -> None:
    with paused_session(TinyNet(), phases={"train": 1}) as session:
        snap = session.snapshot
        assert snap is not None
        assert "x" in snap.activations
        assert snap.activations["x"].shape == (2, 4)
        assert session.input_names == ["x"]


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
            train_step(model)

    with paused_worker(session, loop):
        snap = session.snapshot
        assert snap is not None
        assert "image" in snap.activations
        assert "x" not in snap.activations


def test_snapshot_tensors_are_cpu_and_independent() -> None:
    model = TinyNet()
    with paused_session(model, phases={"train": 1}) as session:
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


def test_stop_then_step_pauses_at_next_batch() -> None:
    session, model = make_session(epochs=1, phases={"train": 3})
    session.detach()

    captured: list[bool] = []

    def loop() -> None:
        for _ in range(3):
            with session.batch(phase="train", epoch=0) as ctx:
                train_step(model)
            captured.append(ctx.captured)

    thread = run_in_thread(loop)
    session.stop()  # next batch boundary should pause
    assert session.wait_until_paused(timeout=5)
    session.detach()
    thread.join(timeout=5)
    assert not thread.is_alive()
    assert sum(captured) >= 1


def test_is_running_false_while_paused_and_after_each_step() -> None:
    """`is_running` is False whenever the worker sits paused at a captured
    batch — including after a Step resumes it onto the next one. The top bar
    grays Stop from this (and Run from its negation)."""
    with paused_session(TinyNet(), phases={"train": 3}) as session:
        assert session.is_running is False
        session.step_batch()
        assert session.wait_until_paused(after_pauses=1, timeout=5)
        assert session.is_running is False


def test_is_running_true_while_worker_advances_a_batch() -> None:
    """While the worker is actively inside a batch (here under detach, which
    never pauses), `is_running` is True — what grays out Run mid-run."""
    session, model = make_session(epochs=1, phases={"train": 1})
    session.detach()
    in_batch = threading.Event()
    release = threading.Event()

    def loop() -> None:
        with session.batch(phase="train", epoch=0):
            in_batch.set()
            assert release.wait(timeout=5)
            train_step(model)

    thread = run_in_thread(loop)
    try:
        assert in_batch.wait(timeout=5)
        assert session.is_running is True
    finally:
        release.set()
        thread.join(timeout=5)
    assert not thread.is_alive()


def test_set_schedule_mid_run() -> None:
    session, model = make_session(epochs=1, phases={"train": 2})
    session.detach()

    with session.batch(phase="train", epoch=0):
        train_step(model)

    session.set_schedule(phases={"train": 5})
    for _ in range(4):
        with session.batch(phase="train", epoch=0):
            train_step(model)


def test_close_releases_waiter_and_is_idempotent() -> None:
    with paused_session(TinyNet(), phases={"train": 1}) as session:
        session.close()
        session.close()
    assert session.closed


def test_close_before_any_batch_is_safe() -> None:
    session, model = make_session(epochs=1, phases={"train": 1})
    session.close()
    with session.batch(phase="train", epoch=0) as ctx:
        train_step(model)
    assert not ctx.captured
    assert session.snapshot is None


def test_unknown_phase_raises_through_context() -> None:
    session, model = make_session(epochs=1, phases={"train": 1})
    session.detach()
    with pytest.raises(ValueError, match="unknown phase"):
        with session.batch(phase="bogus", epoch=0):
            train_step(model)


def test_user_exception_does_not_pause() -> None:
    session, model = make_session(epochs=1, phases={"train": 1})
    # default mode is STEP: would normally pause, but user exception should
    # propagate without us blocking the worker.

    class Boom(Exception):
        pass

    def loop() -> None:
        with pytest.raises(Boom):
            with session.batch(phase="train", epoch=0):
                raise Boom

    thread = run_in_thread(loop)
    thread.join(timeout=5)
    assert not thread.is_alive()
    assert session.snapshot is None
    assert session.mode == Mode.STEP


def test_hooks_removed_after_each_batch() -> None:
    with paused_session(TinyNet(), phases={"train": 2}) as session:
        # Between pauses, hooks should have been removed even though we still
        # hold the activations from the previous batch on the snapshot.
        assert session._hook_handles == []  # type: ignore[reportPrivateUsage]


def test_batches_runs_each_item_inside_a_batch_context() -> None:
    session, _ = make_session(epochs=1, phases={"train": 3})
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
