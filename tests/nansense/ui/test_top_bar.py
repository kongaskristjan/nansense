"""Tests for step-until validation, default positions, and epoch summaries in nansense.ui.top_bar."""

from __future__ import annotations

import pytest
import torch

from nansense import debugger
from nansense.debugger import DebugError, LayerReport
from nansense.schedule import Schedule
from nansense.session import BatchSnapshot
from nansense.ui.top_bar import (
    _DEBUG_UNDER_OVER_TIP,
    _at_last_batch,
    _best_effort_ui_update,
    _current_position,
    _debug_banner_summary,
    _debug_pct,
    _summarize_epoch_ranges,
    _time_travel_default_index,
    _under_over_band_lines,
    _validate_step_until_target,
)
from tests.nansense.helpers import (
    TinyNet,
    _make_snapshot,
    make_position,
    make_session,
    paused_session,
    paused_worker,
    train_step,
)


def _snapshot_at(phase: str, epoch: int, batch_idx: int) -> BatchSnapshot:
    return _make_snapshot(phase, epoch, batch_idx, activations={"x": torch.zeros(1)})


@pytest.fixture
def schedule() -> Schedule:
    return Schedule(epochs=3, phases={"train": 5, "val": 2})


def test_validate_passes_for_future_position(schedule: Schedule) -> None:
    snap = _snapshot_at("train", 0, 1)
    assert (
        _validate_step_until_target(
            live_position=None,
            snapshot=snap,
            phase_order=schedule.phase_order,
            phase_index=1,  # val
            epoch=0,
            batch_idx=0,
        )
        is None
    )


def test_validate_passes_when_no_snapshot_yet(schedule: Schedule) -> None:
    assert (
        _validate_step_until_target(
            live_position=None,
            snapshot=None,
            phase_order=schedule.phase_order,
            phase_index=0,
            epoch=0,
            batch_idx=0,
        )
        is None
    )


def test_validate_allows_unknown_phase_and_overlarge_batch(
    schedule: Schedule,
) -> None:
    """Targets beyond the known schedule are warnings in the dialog, not hard
    rejects here — they simply match if training reaches them. Only the
    forward-progress rule is enforced."""
    for phase_index, epoch, batch_idx in [
        (5, 0, 0),  # phase index not observed yet
        (0, 99, 0),  # epoch beyond the total
        (0, 0, 999),  # batch beyond the learned count
    ]:
        assert (
            _validate_step_until_target(
                live_position=None,
                snapshot=None,
                phase_order=schedule.phase_order,
                phase_index=phase_index,
                epoch=epoch,
                batch_idx=batch_idx,
            )
            is None
        )


def test_validate_uses_live_position_over_stale_snapshot(schedule: Schedule) -> None:
    """The live position runs ahead of the snapshot during step_epoch/run/detach.

    `step_until_position` only captures on an *exact* (phase_index, epoch,
    batch_idx) match against the live position, so a target between the stale
    snapshot and the live position can never be hit — it must be rejected, not
    silently let training run to the end.
    """
    snap = _snapshot_at("train", 0, 0)
    live = make_position("train", 1, 3)
    # Between snapshot (train,0,0) and live (train,1,3): already passed.
    assert (
        _validate_step_until_target(
            live_position=live,
            snapshot=snap,
            phase_order=schedule.phase_order,
            phase_index=0,
            epoch=0,
            batch_idx=4,
        )
        == "Target must be after the current position"
    )
    # Strictly ahead of the live position: accepted.
    assert (
        _validate_step_until_target(
            live_position=live,
            snapshot=snap,
            phase_order=schedule.phase_order,
            phase_index=0,
            epoch=2,
            batch_idx=0,
        )
        is None
    )


@pytest.mark.parametrize(
    ("current", "phase_index", "epoch", "batch_idx"),
    [
        pytest.param(("train", 0, 2), 0, 0, 0, id="before-current"),
        pytest.param(("train", 0, 2), 0, 0, 1, id="just-before-current"),
        pytest.param(("train", 0, 2), 0, 0, 2, id="at-current"),
        pytest.param(("val", 0, 0), 0, 0, 4, id="earlier-phase-same-epoch"),
    ],
)
def test_validate_step_until_rejects_backward_targets(
    schedule: Schedule,
    current: tuple[str, int, int],
    phase_index: int,
    epoch: int,
    batch_idx: int,
) -> None:
    """A target at or before the current position is rejected (stepping is
    forward-only; going back is time travel). `current` is the last-captured
    position."""
    snap = _snapshot_at(*current)
    msg = _validate_step_until_target(
        live_position=None,
        snapshot=snap,
        phase_order=schedule.phase_order,
        phase_index=phase_index,
        epoch=epoch,
        batch_idx=batch_idx,
    )
    assert msg is not None
    assert "after the current" in msg


@pytest.mark.parametrize(
    "live, snapshot, expected",
    [
        # Live position wins over the snapshot's frozen position.
        (("val", 1, 3), ("train", 0, 0), ("val", 1, 3)),
        # No live position yet: fall back to the snapshot.
        (None, ("train", 2, 4), ("train", 2, 4)),
        # Nothing published yet: keep the dialog's existing values.
        (None, None, None),
    ],
)
def test_current_position(
    live: tuple[str, int, int] | None,
    snapshot: tuple[str, int, int] | None,
    expected: tuple[str, int, int] | None,
) -> None:
    result = _current_position(
        make_position(*live) if live is not None else None,
        _snapshot_at(*snapshot) if snapshot is not None else None,
    )
    if expected is None:
        assert result is None
    else:
        assert result is not None
        assert (result.phase, result.epoch, result.batch_idx) == expected


@pytest.mark.parametrize(
    ("epochs", "expected"),
    [
        ([0], "0"),
        ([0, 1, 2], "0–2"),
        ([0, 1, 2, 5, 7, 8], "0–2, 5, 7–8"),
        ([3, 5], "3, 5"),
    ],
)
def test_summarize_epoch_ranges(epochs: list[int], expected: str) -> None:
    assert _summarize_epoch_ranges(epochs) == expected


@pytest.mark.parametrize(
    ("cached", "current_epoch", "expected"),
    [
        # Current epoch is cached: preselect it, not the last cached epoch
        # (epochs past a backwards jump keep their checkpoints on disk).
        pytest.param([0, 1, 2, 3, 4], 2, 2, id="current-cached"),
        pytest.param([0, 1, 2], 2, 2, id="current-is-last"),
        # Current epoch missing from a gappy cache: closest one before it.
        pytest.param([0, 1, 5, 6], 3, 1, id="gap-rounds-down"),
        # Everything cached is in the future: clamp to the earliest.
        pytest.param([4, 5], 2, 0, id="all-cached-ahead"),
        # No position published yet: fall back to the last cached epoch.
        pytest.param([0, 1, 2], None, 2, id="no-position-yet"),
    ],
)
def test_time_travel_default_index(
    cached: list[int], current_epoch: int | None, expected: int
) -> None:
    assert _time_travel_default_index(cached, current_epoch) == expected


def test_at_last_batch_false_before_start_and_mid_training() -> None:
    """No live position yet, and a non-final captured batch, are both not-last —
    so Run / Step Batch advance without the step-over-the-end confirmation."""
    session, _ = make_session(epochs=1, phases={"train": 3})
    assert _at_last_batch(session) is False  # nothing has entered a batch yet

    model = TinyNet()
    with paused_session(model, phases={"train": 3}) as paused:
        assert paused.live_position is not None
        assert paused.live_position.is_last_overall is False
        assert _at_last_batch(paused) is False


def test_at_last_batch_true_when_paused_on_final_batch() -> None:
    """RUN pauses on the final overall batch; `_at_last_batch` flags it so the
    next Run / Step Batch click routes through the confirmation dialog."""
    session, model = make_session(epochs=1, phases={"train": 2})

    def loop() -> None:
        for _ in range(2):
            with session.batch(phase="train", epoch=0):
                train_step(model)

    session.step_run()
    with paused_worker(session, loop):
        assert session.live_position is not None
        assert session.live_position.is_last_overall is True
        assert _at_last_batch(session) is True


def test_best_effort_ui_update_runs_then_swallows_torn_down_client() -> None:
    """The deferred post-finalize refresh runs normally, but a client torn
    down during the await (NiceGUI raises RuntimeError) must not surface as an
    unhandled exception."""
    calls: list[str] = []
    _best_effort_ui_update(lambda: calls.append("ran"))
    assert calls == ["ran"]

    def torn_down() -> None:
        # NiceGUI's message when the element's slot/page no longer exists.
        raise RuntimeError("The parent element this slot belongs to has been deleted.")

    _best_effort_ui_update(torn_down)  # must not propagate


def test_debug_banner_summary_lists_reasons_and_position() -> None:
    error = DebugError(
        position=make_position("val", 2, 7),
        reasons=("nan", "underflow"),
        checks_used=(debugger.NAN_INF, debugger.UNDER_OVER),
        layers=(LayerReport("l", nan=1.0, inf=0.0, underflow=0.5, overflow=0.0),),
    )
    summary = _debug_banner_summary(error)
    # Reframed as a warning: "Numerical issue detected", not "error".
    assert summary.startswith("Numerical issue detected")
    assert "NaN" in summary
    # The "underflow" reason is displayed as "subnormal".
    assert "subnormal" in summary
    assert "epoch 2" in summary
    assert "val batch 7" in summary


def test_under_over_tip_names_modern_remedies() -> None:
    # The dialog's second paragraph points at modern PyTorch fixes.
    assert "GradScaler" in _DEBUG_UNDER_OVER_TIP
    assert "bfloat16" in _DEBUG_UNDER_OVER_TIP


def test_under_over_band_lines_name_dtype_and_magnitudes() -> None:
    import torch

    finfo = torch.finfo(torch.float16)
    error = DebugError(
        position=make_position("train", 0, 1),
        reasons=("underflow",),
        checks_used=(debugger.UNDER_OVER,),
        layers=(
            LayerReport(
                "l", nan=0.0, inf=0.0, underflow=0.5, overflow=0.0,
                dtype=torch.float16,
            ),
        ),
    )
    lines = _under_over_band_lines(error)
    assert len(lines) == 1
    assert "float16" in lines[0]
    assert "subnormal" in lines[0] and "overflow" in lines[0]
    # The real band magnitudes are spelled out: the subnormal edge (tiny) and
    # the early-warning overflow edge (max / headroom).
    assert f"{finfo.tiny:.2e}" in lines[0]
    assert f"{finfo.max / debugger.OVERFLOW_HEADROOM:.2e}" in lines[0]


def test_under_over_band_lines_one_per_distinct_dtype() -> None:
    import torch

    error = DebugError(
        position=make_position("train", 0, 1),
        reasons=("underflow",),
        checks_used=(debugger.UNDER_OVER,),
        layers=(
            LayerReport("a", 0.0, 0.0, 0.5, 0.0, dtype=torch.float16),
            LayerReport("b", 0.0, 0.0, 0.4, 0.0, dtype=torch.float16),
            LayerReport("c", 0.0, 0.0, 0.3, 0.0, dtype=torch.bfloat16),
        ),
    )
    lines = _under_over_band_lines(error)
    # Distinct dtypes only: float16 once (shared by a, b) plus bfloat16.
    assert len(lines) == 2


@pytest.mark.parametrize(
    "frac, expected",
    [
        (0.0, "—"),
        (0.0005, "<0.1%"),
        (0.5, "50.0%"),
        (1.0, "100.0%"),
    ],
)
def test_debug_pct(frac: float, expected: str) -> None:
    assert _debug_pct(frac) == expected
