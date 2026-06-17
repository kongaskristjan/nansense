"""Tests for patch grids, bin-sample strips, and figure payloads in nansense.ui.stats_page."""

from __future__ import annotations

import torch

from nansense.patches import PATCH_TYPES, PatchAccumulator
from nansense.session import BatchSnapshot
from nansense.ui.histograms import _make_histogram_figure
from nansense.ui.stats_page import (
    _ALL_LAYERS_MAX,
    _HOVER_EVENT,
    _LAYER_ALL,
    _PLOTLY_CONFIG,
    _bin_samples_html,
    _figure_payload,
    _filter_phase,
    _hover_attach_js,
    _layer_select_options,
    _patch_grids_html,
    _patch_grids_signature,
    _reconcile_selected_layer,
    _visible_layers,
    _watched_in_order,
)
from nansense.watch import ZERO_BIN, LayerStatsSnapshot, bin_index
from tests.nansense.helpers import _layer_snap, _make_snapshot, _tensor_stats


def test_hover_attach_js_uses_one_shared_event_with_element_id() -> None:
    """Each plot emits the single shared hover event with its element id in
    the payload — not a per-element event name. That's what lets one
    page-level handler serve every plot, so a card rebuild no longer piles up
    a `ui.on` handler (and the dead plot it closes over) per rebuild."""
    js = _hover_attach_js(42)
    assert f"emitEvent('{_HOVER_EVENT}'" in js
    assert "id: 42" in js
    assert "bin: p.pointNumber" in js
    # The old per-element event name must be gone.
    assert "nansense_hist_hover_42" not in js


def test_figure_payload_carries_plotly_config() -> None:
    # Autoscale would land on a different scale than the capped initial
    # render, so it's removed; double-click resets to the built ranges.
    fig, _ = _make_histogram_figure({}, "activation", "a")
    payload = _figure_payload(fig)
    assert set(payload) >= {"data", "layout", "config"}
    assert payload["config"] is _PLOTLY_CONFIG
    assert _PLOTLY_CONFIG["modeBarButtonsToRemove"] == ["autoScale2d"]
    assert _PLOTLY_CONFIG["doubleClick"] == "reset"


def _layer_snap_with_patches(phase: str, epoch: int = 0) -> LayerStatsSnapshot:
    acc = PatchAccumulator()
    acc.update(act=torch.randn(2, 2, 4, 4), x=torch.rand(2, 3, 8, 8))
    patches = acc.snapshot()
    assert patches is not None
    stats = _tensor_stats(4)
    return LayerStatsSnapshot(
        layer="L",
        phase=phase,
        epoch=epoch,
        activations=stats,
        gradients=stats,
        patches=patches,
    )


def test_patch_grids_html_renders_enabled_grids_per_phase() -> None:
    per_phase = {
        "train": _layer_snap_with_patches("train", epoch=1),
        "val": _layer_snap_with_patches("val", epoch=1),
    }
    html = _patch_grids_html(
        per_phase,
        enabled=list(PATCH_TYPES),
        heatmap=False,
        mean=None,
        std=None,
    )
    assert html.count("<img") == 2 * len(PATCH_TYPES)  # 4 grids × 2 phases
    for label in ("Max pixel", "Min pixel", "Max average", "Min average"):
        assert html.count(label) >= 2
    assert "train (ep 1)" in html
    assert "val (ep 1)" in html


def test_patch_grids_html_filters_to_enabled_types() -> None:
    per_phase = {"train": _layer_snap_with_patches("train")}
    html = _patch_grids_html(
        per_phase, enabled=["max_pixel"], heatmap=False, mean=None, std=None
    )
    assert html.count("<img") == 1
    assert "Max pixel" in html
    assert "Min pixel" not in html


def test_patch_grids_html_placeholder_without_patches() -> None:
    assert "no patches" in _patch_grids_html(
        {}, enabled=list(PATCH_TYPES), heatmap=False, mean=None, std=None
    )
    # A phase whose bucket has histogram stats but no patch data (e.g. a
    # non-image input) also falls back to the placeholder.
    assert "no patches" in _patch_grids_html(
        {"train": _layer_snap("train")},
        enabled=list(PATCH_TYPES),
        heatmap=False,
        mean=None,
        std=None,
    )


def test_patch_grids_html_labels_axes() -> None:
    # The fixture activations have 2 channels; the caption shows the count.
    per_phase = {"train": _layer_snap_with_patches("train")}
    html = _patch_grids_html(
        per_phase, enabled=["max_pixel"], heatmap=False, mean=None, std=None
    )
    assert "2 channels (one channel per column) &rarr;" in html
    assert "top samples (best first) &rarr;" in html
    assert "writing-mode:vertical-rl" in html
    assert "text-[15px] font-mono text-slate-600" in html


def test_patch_grids_signature_tracks_toggles_and_values() -> None:
    per_phase = {"train": _layer_snap_with_patches("train")}
    base = _patch_grids_signature(per_phase, list(PATCH_TYPES), False)
    assert base == _patch_grids_signature(per_phase, list(PATCH_TYPES), False)
    assert base != _patch_grids_signature(per_phase, list(PATCH_TYPES), True)
    assert base != _patch_grids_signature(per_phase, ["max_pixel"], False)
    other = {"train": _layer_snap_with_patches("train")}  # new random extremes
    assert base != _patch_grids_signature(other, list(PATCH_TYPES), False)


_NAMES = ["a", "b", "c", "d"]


def test_watched_in_order_follows_layer_names_not_set_order() -> None:
    # `watched` is an unordered set; order must come from `layer_names`.
    assert _watched_in_order(_NAMES, frozenset({"c", "a"})) == ["a", "c"]
    assert _watched_in_order(_NAMES, frozenset()) == []
    # A name not in `layer_names` is ignored.
    assert _watched_in_order(_NAMES, frozenset({"a", "z"})) == ["a"]


def test_layer_select_options_offers_all_only_when_few_watched() -> None:
    opts = _layer_select_options(["a", "b"])
    assert list(opts) == [_LAYER_ALL, "a", "b"]  # "all" first, in graph order
    assert opts["a"] == "a"
    # No layers → no options at all (not even "all").
    assert _layer_select_options([]) == {}


def test_layer_select_options_drops_all_at_threshold() -> None:
    below = [str(i) for i in range(_ALL_LAYERS_MAX - 1)]
    assert _LAYER_ALL in _layer_select_options(below)
    at = [str(i) for i in range(_ALL_LAYERS_MAX)]
    assert _LAYER_ALL not in _layer_select_options(at)
    assert list(_layer_select_options(at)) == at


def test_reconcile_selected_layer_defaults_to_first_watched() -> None:
    # Empty/unset selection (and any stale name) falls back to the first.
    assert _reconcile_selected_layer("", ["a", "b"]) == "a"
    assert _reconcile_selected_layer("gone", ["a", "b"]) == "a"
    # A still-watched layer is kept.
    assert _reconcile_selected_layer("b", ["a", "b"]) == "b"
    # Nothing watched → empty selection.
    assert _reconcile_selected_layer("a", []) == ""


def test_reconcile_selected_layer_keeps_all_only_below_threshold() -> None:
    below = [str(i) for i in range(_ALL_LAYERS_MAX - 1)]
    assert _reconcile_selected_layer(_LAYER_ALL, below) == _LAYER_ALL
    # At the threshold "all" is no longer offered, so it falls back to first.
    at = [str(i) for i in range(_ALL_LAYERS_MAX)]
    assert _reconcile_selected_layer(_LAYER_ALL, at) == "0"


def test_visible_layers_picks_one_or_all() -> None:
    assert _visible_layers("b", ["a", "b", "c"]) == ["b"]
    assert _visible_layers(_LAYER_ALL, ["a", "b", "c"]) == ["a", "b", "c"]
    # "all" past the threshold renders just the first (the safe fallback).
    at = [str(i) for i in range(_ALL_LAYERS_MAX)]
    assert _visible_layers(_LAYER_ALL, at) == ["0"]
    # A stale single selection falls back to the first watched layer.
    assert _visible_layers("gone", ["a", "b"]) == ["a"]
    assert _visible_layers("", []) == []


def test_filter_phase_narrows_to_selected_phase() -> None:
    per_phase = {"train": _layer_snap("train"), "val": _layer_snap("val")}
    assert set(_filter_phase(per_phase, "val")) == {"val"}
    assert set(_filter_phase(per_phase, "train")) == {"train"}
    assert _filter_phase(per_phase, "test") == {}


def test_patch_grids_html_adds_heat_legend_when_enabled() -> None:
    per_phase = {"train": _layer_snap_with_patches("train")}
    plain = _patch_grids_html(
        per_phase, enabled=["max_pixel"], heatmap=False, mean=None, std=None
    )
    heat = _patch_grids_html(
        per_phase, enabled=["max_pixel"], heatmap=True, mean=None, std=None
    )
    assert plain.count("<img") == 1
    assert heat.count("<img") == 2  # the grid plus its colorbar


def _hover_snapshot() -> BatchSnapshot:
    act = torch.zeros(2, 2, 4, 4)
    act[0, 1, 2, 3] = 5.0
    return _make_snapshot(
        "train", 3, 7, activations={"x": torch.rand(2, 3, 16, 16), "conv": act}
    )


def test_bin_samples_html_without_snapshot_notes_missing_batch() -> None:
    out = _bin_samples_html(None, "conv", "activation", 0, ZERO_BIN, "x", None, None)
    assert "no batch captured yet" in out


def test_bin_samples_html_names_the_source_batch() -> None:
    """The strip must make the last-batch-only population explicit."""
    out = _bin_samples_html(
        _hover_snapshot(), "conv", "activation", 1, bin_index(5.0), "x", None, None
    )
    assert "last captured batch only" in out
    assert "train ep 3, batch 7" in out
    assert "<img" in out  # the matching element rendered as an input crop
    assert "sample 0" in out


def test_bin_samples_html_empty_bin_notes_no_values() -> None:
    out = _bin_samples_html(
        _hover_snapshot(), "conv", "activation", 0, bin_index(5.0), "x", None, None
    )
    assert "no values in this bar" in out
    assert "last captured batch" in out


def test_bin_samples_html_missing_gradients_notes_kind() -> None:
    out = _bin_samples_html(
        _hover_snapshot(), "conv", "gradient", 0, ZERO_BIN, "x", None, None
    )
    assert "no captured gradients" in out
