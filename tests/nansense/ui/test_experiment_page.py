"""Tests for experiment parameter specs and layer-channel helpers in nansense.ui.experiment_page."""

from __future__ import annotations

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


def test_experiment_param_order_channel_then_inputs() -> None:
    from nansense.ui.experiment_page import _EXPERIMENT_PARAMS

    for kind, specs in _EXPERIMENT_PARAMS.items():
        keys = [s.key for s in specs]
        # The targeting knob comes first, Inputs directly below it, and every
        # kind exposes the "use viewed sample" toggle (point 1).
        assert keys[0] in ("channel", "target"), kind
        assert keys[1] == "batch", kind
        assert "use_viewed" in keys, kind
    # Deep dream's "Start from" sits directly below Inputs.
    dd = [s.key for s in _EXPERIMENT_PARAMS["deep_dream"]]
    assert dd[dd.index("batch") + 1] == "start"


def test_experiment_descriptions_cover_every_kind() -> None:
    from nansense.experiments import EXPERIMENT_KINDS
    from nansense.ui.experiment_page import _EXPERIMENT_DESCRIPTIONS

    assert set(_EXPERIMENT_DESCRIPTIONS) == set(EXPERIMENT_KINDS)
    for short, long in _EXPERIMENT_DESCRIPTIONS.values():
        assert short and long  # both a tooltip and a pane description
    # Neuron Gradient calls out its grainy maps (point 4).
    assert "grain" in _EXPERIMENT_DESCRIPTIONS["neuron_gradient"][1].lower()


def test_layer_channel_count_reads_snapshot_activation() -> None:
    from nansense.ui.experiment_page import _layer_channel_count

    snap = _frame_snapshot()
    snap.activations["vec"] = torch.rand(5)  # channel-less activation
    assert _layer_channel_count(snap, "conv") == 2
    assert _layer_channel_count(snap, "vec") is None
    assert _layer_channel_count(snap, "missing") is None
    assert _layer_channel_count(None, "conv") is None
