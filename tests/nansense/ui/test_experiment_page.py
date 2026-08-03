"""Tests for the experiment form helpers in nansense.ui.experiment_page."""

from __future__ import annotations

import pytest
import torch

from tests.nansense.helpers import _frame_snapshot


@pytest.mark.parametrize(
    ("kind", "candidates", "expected"),
    [
        ("int", (3.7,), 3),  # cast to the spec's type
        ("float", (2,), 2.0),
        ("int", (None, 5), 5),  # a cleared field (None) skips to the next candidate
        ("float", (None, None, 0.5), 0.5),  # falls through to the numeric default
        ("int", ("abc", 4), 4),  # a non-numeric value is skipped too
    ],
)
def test_coerce_number_skips_non_numeric_candidates(
    kind: str, candidates: tuple[object, ...], expected: float
) -> None:
    from nansense.experiments import ExperimentParam
    from nansense.ui.experiment_page import _coerce_number

    spec = ExperimentParam("k", "K", kind, 0)
    result = _coerce_number(spec, *candidates)
    assert result == expected
    assert isinstance(result, int if kind == "int" else float)


def test_coerce_number_requires_a_numeric_candidate() -> None:
    from nansense.experiments import ExperimentParam
    from nansense.ui.experiment_page import _coerce_number

    spec = ExperimentParam("k", "K", "int", 0)
    with pytest.raises(AssertionError):
        _coerce_number(spec, None, "x")  # would otherwise crash the run with a cast


def test_layer_channel_count_reads_snapshot_activation() -> None:
    from nansense.ui.experiment_page import _layer_channel_count

    snap = _frame_snapshot()
    snap.activations["vec"] = torch.rand(5)  # channel-less activation
    assert _layer_channel_count(snap, "conv") == 2
    assert _layer_channel_count(snap, "vec") is None
    assert _layer_channel_count(snap, "missing") is None
    assert _layer_channel_count(None, "conv") is None


@pytest.mark.parametrize("locked", [False, True])
def test_run_tooltip_mentions_pausing_only_when_unlocked(locked: bool) -> None:
    from nansense.ui.experiment_page import _run_tooltip

    text = _run_tooltip(locked)
    # Why the button greys out is the results pane's job (`_status_text`);
    # the tooltip stays a single clause about what pressing Run does.
    assert "auto-run" not in text and "in flight" not in text
    assert text.startswith("Run the experiment")
    if locked:
        # Locked sessions have no training controls to point at.
        assert "paused" not in text and "training" not in text
    else:
        assert "training must be paused" in text


@pytest.mark.parametrize("locked", [False, True])
@pytest.mark.parametrize("auto_run", [False, True])
def test_status_text_locked_and_auto_run_variants(
    auto_run: bool, locked: bool
) -> None:
    from nansense.ui.experiment_page import _status_text

    text = _status_text(auto_run, locked)
    # The auto-run split survives the locked rewording.
    assert ("runs automatically" in text) == auto_run
    assert ("press Run" in text) == (not auto_run)
    if locked:
        assert "paused" not in text and "training" not in text
    else:
        assert "training must be paused" in text


@pytest.mark.parametrize("locked", [False, True])
def test_pending_chip_names_the_run_controls_only_when_there_are_some(
    locked: bool,
) -> None:
    """Advancing training is the slow wait — it still resolves itself at the
    next visualization update, so the pill spins and only *offers* the
    controls that cut it short. A locked session has none to offer."""
    from nansense.experiments import ExperimentQueueState
    from nansense.ui.experiment_page import _pending_chip

    chip = _pending_chip(
        "deep_dream",
        ExperimentQueueState("queued", 0),
        training_running=True,
        locked=locked,
    )
    assert chip.icon is None
    assert "Deep Dream" in chip.text and "next visualization update" in chip.text
    assert ("Step Batch" in chip.text) == (not locked)


@pytest.mark.parametrize(
    ("stage", "ahead", "expected"),
    [
        ("running", 0, "Deep Dream — running…"),
        ("queued", 0, "Deep Dream — starting…"),
        ("queued", 1, "Deep Dream — queued behind 1 experiment"),
        ("queued", 3, "Deep Dream — queued behind 3 experiments"),
    ],
)
def test_pending_chip_spins_through_the_waits_that_resolve_themselves(
    stage: str, ahead: int, expected: str
) -> None:
    from nansense.experiments import ExperimentQueueState
    from nansense.ui.experiment_page import _pending_chip

    chip = _pending_chip(
        "deep_dream",
        ExperimentQueueState(stage, ahead),
        training_running=False,
        locked=True,
    )
    assert chip.icon is None  # the spinner is the in-progress cue
    assert chip.text == expected


def test_pending_chip_reports_a_request_that_never_ran() -> None:
    # Cancel before the first publish: no result is ever coming, so a
    # spinner would lie.
    from nansense.experiments import ExperimentQueueState
    from nansense.ui.experiment_page import _pending_chip

    chip = _pending_chip(
        "occlusion",
        ExperimentQueueState("absent"),
        training_running=False,
        locked=False,
    )
    assert chip.icon is not None and "stopped before it ran" in chip.text


@pytest.mark.parametrize(
    ("step", "done", "error", "tone", "spins"),
    [
        (7, False, None, "running", True),
        (300, True, None, "done", False),
        (12, True, None, "stopped", False),  # cancelled or past the time limit
        (0, True, "boom", "failed", False),
    ],
)
def test_result_chip_tone_follows_the_outcome(
    step: int, done: bool, error: str | None, tone: str, spins: bool
) -> None:
    from nansense.experiments import ExperimentResult
    from nansense.ui.experiment_page import _result_chip

    chip = _result_chip(
        ExperimentResult(
            seq=1,
            kind="deep_dream",
            layer="conv",
            step=step,
            total_steps=300,
            done=done,
            error=error,
        )
    )
    assert chip.tone == tone
    assert (chip.icon is None) == spins


def test_every_chip_the_page_builds_has_a_defined_tone() -> None:
    # A tone with no entry in `_CHIP_TONES` raises only when the pill is
    # first shown — in the browser, mid-experiment.
    from nansense.experiments import ExperimentQueueState, ExperimentResult
    from nansense.ui.common import _CHIP_TONES
    from nansense.ui.experiment_page import _idle_chip, _pending_chip, _result_chip

    flags = (False, True)
    stages = (("running", 0), ("queued", 0), ("queued", 2), ("absent", 0))
    chips = [_idle_chip(auto_run, locked) for auto_run in flags for locked in flags]
    chips += [
        _pending_chip(
            "deep_dream",
            ExperimentQueueState(stage, ahead),
            training_running=running,
            locked=locked,
        )
        for stage, ahead in stages
        for running in flags
        for locked in flags
    ]
    chips += [
        _result_chip(
            ExperimentResult(
                seq=1,
                kind="deep_dream",
                layer="conv",
                step=step,
                total_steps=3,
                done=done,
                error=error,
            )
        )
        for step, done, error in (
            (1, False, None),  # streaming
            (3, True, None),  # done
            (1, True, None),  # stopped early
            (0, True, "x"),  # failed
        )
    ]
    assert {chip.tone for chip in chips} <= set(_CHIP_TONES)


def test_minmax_stats_href_encodes_layer_and_targets_minmax_view() -> None:
    # A real `href` (not an `on_click` navigate) keeps the compare button
    # middle-clickable; the link must land on the MIN/MAX grids directly.
    from nansense.ui.experiment_page import _minmax_stats_href

    assert _minmax_stats_href("fc1") == "/stats?layer=fc1&view=minmax"
    assert _minmax_stats_href("odd layer") == "/stats?layer=odd%20layer&view=minmax"
