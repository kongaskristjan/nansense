"""Tests for experiment parameter specs and layer-channel helpers in nansense.ui.experiment_page."""

from __future__ import annotations

import pytest
import torch

from tests.nansense.helpers import _frame_snapshot


def test_experiment_params_cover_every_kind() -> None:
    from nansense.experiments import EXPERIMENT_KINDS
    from nansense.ui.experiment_page import _EXPERIMENT_PARAMS

    assert set(_EXPERIMENT_PARAMS) == set(EXPERIMENT_KINDS)
    for kind, specs in _EXPERIMENT_PARAMS.items():
        assert specs, kind  # every experiment exposes at least one knob
        for spec in specs:
            assert spec.kind in ("int", "float", "bool", "select"), spec.key
            if spec.kind == "select":
                assert spec.options and spec.default in spec.options, spec.key
            if spec.kind in ("int", "float"):
                assert isinstance(spec.default, (int, float)), spec.key


def test_experiment_param_order_targeting_then_inputs() -> None:
    from nansense.ui.experiment_page import _EXPERIMENT_PARAMS

    for kind, specs in _EXPERIMENT_PARAMS.items():
        keys = [s.key for s in specs]
        if kind == "deep_dream":
            # Channels first; Start from with Sample directly below it (Sample's
            # visibility follows Start). Minimize sits directly above Clamp.
            assert keys[0] == "channels", kind
            start_idx = keys.index("start")
            assert keys[start_idx + 1] == "sample", kind
            assert keys[keys.index("clamp") - 1] == "minimize", kind
        else:
            # Captum: the targeting knob first, Inputs directly below it.
            assert keys[0] in ("channel", "target"), kind
            assert keys[1] == "batch", kind


def test_deep_dream_exposes_minimize_toggle() -> None:
    from nansense.ui.experiment_page import _EXPERIMENT_PARAMS

    specs = {s.key: s for s in _EXPERIMENT_PARAMS["deep_dream"]}
    assert "minimize" in specs
    minimize = specs["minimize"]
    assert minimize.kind == "bool" and minimize.default is False


def test_default_param_values_apply_session_overrides() -> None:
    from nansense.ui.experiment_page import (
        _EXPERIMENT_PARAMS,
        _default_param_values,
    )

    plain = _default_param_values({})
    assert plain["steps"] == 300 and plain["channels"] == 8
    seeded = _default_param_values({"steps": 150, "channels": 4})
    assert seeded["steps"] == 150 and seeded["channels"] == 4
    # Everything not overridden keeps its built-in default, and every knob
    # of every kind gets a value.
    assert seeded["lr"] == plain["lr"]
    every_key = {s.key for specs in _EXPERIMENT_PARAMS.values() for s in specs}
    assert set(seeded) == every_key


def test_experiment_descriptions_cover_every_kind() -> None:
    from nansense.experiments import EXPERIMENT_KINDS
    from nansense.ui.experiment_page import _EXPERIMENT_DESCRIPTIONS

    assert set(_EXPERIMENT_DESCRIPTIONS) == set(EXPERIMENT_KINDS)
    for short, long in _EXPERIMENT_DESCRIPTIONS.values():
        assert short and long  # both a tooltip and a pane description
    # Neuron Gradient calls out its grainy maps (point 4).
    assert "grain" in _EXPERIMENT_DESCRIPTIONS["neuron_gradient"][1].lower()


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
    from nansense.ui.experiment_page import _coerce_number, _ExperimentParam

    spec = _ExperimentParam("k", "K", kind, 0)
    result = _coerce_number(spec, *candidates)
    assert result == expected
    assert isinstance(result, int if kind == "int" else float)


def test_coerce_number_requires_a_numeric_candidate() -> None:
    from nansense.ui.experiment_page import _coerce_number, _ExperimentParam

    spec = _ExperimentParam("k", "K", "int", 0)
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
    # The disabled-state advice applies on locked pages too.
    assert "auto-run" in text and "in flight" in text
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


def test_minmax_stats_href_encodes_layer_and_targets_minmax_view() -> None:
    # A real `href` (not an `on_click` navigate) keeps the compare button
    # middle-clickable; the link must land on the MIN/MAX grids directly.
    from nansense.ui.experiment_page import _minmax_stats_href

    assert _minmax_stats_href("fc1") == "/stats?layer=fc1&view=minmax"
    assert _minmax_stats_href("odd layer") == "/stats?layer=odd%20layer&view=minmax"
