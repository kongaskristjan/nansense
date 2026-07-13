"""Tests for patch grids, bin-sample strips, and figure payloads in nansense.ui.stats_page."""

from __future__ import annotations

import asyncio

import pytest
import torch

from nansense import debugger
from nansense.debugger import DebugError, LayerReport
from nansense.patches import PATCH_TYPES, PatchAccumulator, PatchType
from nansense.session import BatchSnapshot, StatsScope
from nansense.ui.histograms import _make_histogram_figure
from nansense.ui.stats_page import (
    _ALL_LAYERS_MAX,
    _HOVER_EVENT,
    _LAYER_ALL,
    _PATCH_TYPE_LABELS,
    _PHASE_CURRENT_BATCH,
    _PHASE_CURRENT_BATCH_LABEL,
    _PLOTLY_CONFIG,
    _RefreshGate,
    _VIEW_HISTOGRAM,
    _VIEW_MINMAX,
    _VIEW_GRAPHS,
    _apply_watch_param,
    _bin_samples_html,
    _deep_dream_href,
    _figure_payload,
    _filter_phase,
    _grid_type_options,
    _hover_attach_js,
    _initial_phase,
    _layer_select_options,
    _patch_grids_html,
    _patch_grids_signature,
    _phase_select_options,
    _reconcile_grid_type,
    _reconcile_selected_layer,
    _reconcile_selected_phase,
    _refresh_now,
    _selectable_layers,
    _should_show_bands,
    _visible_layers,
    _watched_in_order,
)
from nansense.watch import ZERO_BIN, LayerStatsSnapshot, bin_index
from tests.nansense.helpers import (
    TinyNet,
    _layer_snap,
    _make_snapshot,
    _tensor_stats,
    make_position,
    make_session,
    paused_session,
    train_step,
)


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


def _layer_snap_with_patches(
    phase: str, epoch: int = 0, *, average_patches: bool = True
) -> LayerStatsSnapshot:
    acc = PatchAccumulator()
    acc.update(
        act=torch.randn(2, 2, 4, 4),
        x=torch.rand(2, 3, 8, 8),
        average_patches=average_patches,
    )
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
    # 4 grids × 2 phases, each grid a cell <img> per (channel, sample) — the
    # fixture has 2 channels and 5 samples per channel.
    assert html.count("<img") == 2 * 5 * len(PATCH_TYPES) * 2
    for label in ("Max pixel", "Min pixel", "Max average", "Min average"):
        assert html.count(label) >= 2
    assert "train (ep 1)" in html
    assert "val (ep 1)" in html


def test_patch_grids_html_skips_absent_types() -> None:
    # A default (average_patches off) snapshot carries only the two pixel
    # grids; requesting all four types renders those two rather than raising
    # on the missing average keys.
    per_phase = {
        "train": _layer_snap_with_patches("train", average_patches=False)
    }
    html = _patch_grids_html(
        per_phase, enabled=list(PATCH_TYPES), heatmap=False, mean=None, std=None
    )
    assert html.count("<img") == 2 * 5 * 2  # 2 grids × 2 channels × 5 samples
    assert "Max pixel" in html and "Min pixel" in html
    assert "Max average" not in html and "Min average" not in html


def test_patch_grids_html_explains_uncollected_average_selection() -> None:
    # Selecting only an average grid against a default (average_patches off)
    # snapshot must say the type wasn't collected — not claim the model has
    # no image input (patches for the pixel types clearly exist).
    per_phase = {
        "train": _layer_snap_with_patches("train", average_patches=False)
    }
    html = _patch_grids_html(
        per_phase, enabled=["max_average"], heatmap=False, mean=None, std=None
    )
    assert "not collected" in html and "Performance settings" in html
    assert "image-like" not in html


def test_patch_grids_html_filters_to_enabled_types() -> None:
    per_phase = {"train": _layer_snap_with_patches("train")}
    html = _patch_grids_html(
        per_phase, enabled=["max_pixel"], heatmap=False, mean=None, std=None
    )
    assert html.count("<img") == 2 * 5  # 2 channels × 5 sample cells
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


def test_patch_grids_html_channels_scroll_within_card() -> None:
    # Many channels must scroll horizontally inside the layer card rather than
    # spill out of it: the channel columns sit in an `overflow-x-auto` flex
    # child made shrinkable below its content width with `min-width:0`.
    per_phase = {"train": _layer_snap_with_patches("train")}
    html = _patch_grids_html(
        per_phase, enabled=["max_pixel"], heatmap=False, mean=None, std=None
    )
    assert "overflow-x-auto" in html
    assert "min-width:0" in html


def test_patch_grids_html_labels_axes() -> None:
    # The grid is a table: CHANNEL n column headers and SAMPLE n row labels.
    # The fixture has 2 channels and 5 samples per channel.
    per_phase = {"train": _layer_snap_with_patches("train")}
    html = _patch_grids_html(
        per_phase, enabled=["max_pixel"], heatmap=False, mean=None, std=None
    )
    assert "CHANNEL 0" in html and "CHANNEL 1" in html
    assert "SAMPLE 0" in html and "SAMPLE 4" in html
    # The old free-text axis captions are gone.
    assert "one channel per column" not in html
    assert "top samples (best first)" not in html


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


def test_selectable_layers_depends_on_phase() -> None:
    watched = frozenset({"a", "c"})
    # A real phase offers only the watched layers (graph order).
    assert _selectable_layers("train", _NAMES, watched) == ["a", "c"]
    # "Current batch" reads the snapshot, which covers every layer.
    assert _selectable_layers(_PHASE_CURRENT_BATCH, _NAMES, watched) == _NAMES
    # Current batch offers all layers even when nothing is watched.
    assert _selectable_layers(_PHASE_CURRENT_BATCH, _NAMES, frozenset()) == _NAMES


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


def test_deep_dream_href_targets_the_visible_layer() -> None:
    # A real `href` (not an `on_click` navigate) keeps the compare button
    # middle-clickable; the target must track what the page actually shows.
    watched = frozenset({"a", "c"})
    # A selected watched layer links straight to it.
    assert _deep_dream_href("train", _NAMES, watched, "c") == "/experiment?layer=c"
    # "all" falls back to the first visible layer.
    assert (
        _deep_dream_href("train", _NAMES, watched, _LAYER_ALL)
        == "/experiment?layer=a"
    )
    # Nothing watched on a real phase → the stale selection stays the target.
    assert _deep_dream_href("train", _NAMES, frozenset(), "b") == "/experiment?layer=b"
    # Layer names are percent-encoded into the link.
    assert (
        _deep_dream_href(_PHASE_CURRENT_BATCH, ["odd layer"], frozenset(), "odd layer")
        == "/experiment?layer=odd%20layer"
    )


def test_apply_watch_param_watches_in_watched_scope_only() -> None:
    session, _ = make_session()
    # No flag → the watched set stays untouched.
    _apply_watch_param(session, "fc1", "")
    assert "fc1" not in session.watched_layers
    # `?watch=1` under the (default) watched scope starts collection.
    _apply_watch_param(session, "fc1", "1")
    assert "fc1" in session.watched_layers
    # Unknown layer names are refused rather than crashing the page.
    _apply_watch_param(session, "nope", "1")
    assert "nope" not in session.watched_layers
    # Other scopes already collect every layer (or are deliberately
    # paused), so the flag must leave the watched set alone there.
    session.set_stats_scope(StatsScope.ALL)
    _apply_watch_param(session, "fc2", "1")
    assert "fc2" not in session.watched_layers


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
    assert plain.count("<img") == 2 * 5  # 2 channels × 5 sample cells
    assert heat.count("<img") == 2 * 5 + 1  # the cells plus the colorbar


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


# --- Under/overflow band auto-check on page open ---------------------------


def _err(reasons: tuple[str, ...], checks_used: tuple[str, ...]) -> DebugError:
    return DebugError(
        position=make_position("train", 0, 1),
        reasons=reasons,
        checks_used=checks_used,
        layers=(LayerReport("l", nan=0.5, inf=0.0, underflow=0.5, overflow=0.0),),
    )


def test_should_show_bands_none_when_no_error() -> None:
    assert _should_show_bands(None) is False


def test_should_show_bands_true_when_under_over_tripped() -> None:
    # Any route to /stats while an under/overflow issue is active pre-checks
    # the band — it reads the live error, not a query flag.
    assert _should_show_bands(
        _err(("underflow",), (debugger.UNDER_OVER,))
    ) is True
    assert _should_show_bands(
        _err(("nan", "underflow"), (debugger.NAN_INF, debugger.UNDER_OVER))
    ) is True


def test_should_show_bands_false_for_nan_only_issue() -> None:
    # A NaN/Inf-only issue isn't about under/overflow, so the band stays off.
    assert _should_show_bands(_err(("nan",), (debugger.NAN_INF,))) is False


# --- Phase dropdown: options and reconciliation per view ---------------------


def test_phase_options_drop_current_batch_in_stats_view() -> None:
    names = ["train", "val"]
    for view in (_VIEW_HISTOGRAM, _VIEW_MINMAX):
        options = _phase_select_options(view, names)
        assert list(options) == ["train", "val", _PHASE_CURRENT_BATCH]
        assert options[_PHASE_CURRENT_BATCH] == _PHASE_CURRENT_BATCH_LABEL
    assert list(_phase_select_options(_VIEW_GRAPHS, names)) == ["train", "val"]


@pytest.mark.parametrize(
    ("selected", "view", "expected"),
    [
        # The stats view swaps "Current batch" (and anything stale) for the
        # first schedule phase — the epoch-aggregating counterpart.
        (_PHASE_CURRENT_BATCH, _VIEW_GRAPHS, "train"),
        ("bogus", _VIEW_GRAPHS, "train"),
        ("val", _VIEW_GRAPHS, "val"),
        # The other views keep valid picks and fall back to "Current batch".
        (_PHASE_CURRENT_BATCH, _VIEW_HISTOGRAM, _PHASE_CURRENT_BATCH),
        ("train", _VIEW_MINMAX, "train"),
        ("bogus", _VIEW_HISTOGRAM, _PHASE_CURRENT_BATCH),
    ],
)
def test_reconcile_selected_phase(
    selected: str, view: str, expected: str
) -> None:
    assert _reconcile_selected_phase(selected, view, ["train", "val"]) == expected


def test_reconcile_selected_phase_without_known_phases() -> None:
    # A lazy schedule that hasn't observed a phase yet: nothing to swap to.
    assert _reconcile_selected_phase(_PHASE_CURRENT_BATCH, _VIEW_GRAPHS, []) == ""


# --- Opening phase: the running phase when it has collected stats ------------


def test_initial_phase_before_any_batch_is_current_batch() -> None:
    # No live position and no snapshot yet — nothing to key a phase off.
    session, _ = make_session()
    assert _initial_phase(session, "") == _PHASE_CURRENT_BATCH


def test_initial_phase_prefers_the_running_phase_once_collected() -> None:
    with paused_session(TinyNet(), phases={"train": 4}) as session:
        # Paused on the first batch with nothing watched: no aggregates yet.
        assert session.stats_phases() == frozenset()
        assert _initial_phase(session, "") == _PHASE_CURRENT_BATCH
        assert session.watch("fc1")
        # Watching alone collects nothing until a batch is stepped.
        assert _initial_phase(session, "fc1") == _PHASE_CURRENT_BATCH
        session.step_batch()
        assert session.wait_until_paused(after_pauses=1, timeout=5.0)
        assert session.stats_phases() == frozenset({"train"})
        # Opening the page (bare, or on the watched layer's link) lands on
        # the running phase now that its aggregates hold stats.
        assert _initial_phase(session, "") == "train"
        assert _initial_phase(session, "fc1") == "train"
        # A link naming an unwatched layer stays on "Current batch" — the
        # only selection whose Layer dropdown can offer that layer.
        assert session.stats_phases("fc2") == frozenset()
        assert _initial_phase(session, "fc2") == _PHASE_CURRENT_BATCH


# --- MIN/MAX radio: average entries offered only while collected -------------


def test_grid_type_options_gate_average_entries() -> None:
    # Off (the default): only the pixel grids are ever collected, so the
    # radio must not offer the average entries.
    assert list(_grid_type_options(False)) == ["max_pixel", "min_pixel"]
    # On: all four grids collect and all four entries show.
    assert _grid_type_options(True) == _PATCH_TYPE_LABELS


@pytest.mark.parametrize(
    ("selected", "average_patches", "expected"),
    [
        ("min_pixel", False, "min_pixel"),  # still offered → kept
        ("max_average", True, "max_average"),  # still offered → kept
        ("max_average", False, "max_pixel"),  # entry gone → the default
        ("min_average", False, "max_pixel"),
    ],
)
def test_reconcile_grid_type(
    selected: PatchType, average_patches: bool, expected: PatchType
) -> None:
    options = _grid_type_options(average_patches)
    assert _reconcile_grid_type(selected, options) == expected


def test_refresh_gate_passes_only_on_new_publish_or_watched_change() -> None:
    # The page's periodic tick refreshes at the visualization update cadence:
    # the gate passes once per newly published snapshot (and on watched-set
    # changes), not merely because time passed while aggregates accumulated.
    with paused_session(TinyNet(), phases={"train": 4}) as session:
        gate = _RefreshGate()
        # The first tick consumes whatever state the page opened on.
        assert gate.should_refresh(session)
        assert not gate.should_refresh(session)
        # Watching a layer (e.g. from the main page) re-syncs the sidebar.
        assert session.watch("fc1")
        assert gate.should_refresh(session)
        assert not gate.should_refresh(session)
        # Stepping publishes a new snapshot -> exactly one refresh.
        before = session.snapshot
        session.step_batch()
        assert session.wait_until_paused(after_pauses=1, timeout=5.0)
        assert session.snapshot is not before
        assert gate.should_refresh(session)
        assert not gate.should_refresh(session)


def test_refresh_gate_passes_on_average_patches_flip() -> None:
    # Flipping the average-patches Performance setting flushes every
    # aggregate bucket and changes which grid types the MIN/MAX radio
    # offers, so the page must re-render on the next tick.
    session, _ = make_session()
    gate = _RefreshGate()
    assert gate.should_refresh(session)
    assert not gate.should_refresh(session)
    assert session.set_watch_performance(average_patches=True)
    assert gate.should_refresh(session)
    assert not gate.should_refresh(session)


def test_refresh_now_rerenders_and_arms_a_one_shot_publish() -> None:
    # The button must deliver fresh data while training runs freely: the
    # immediate refresh re-renders the running aggregates, and the armed
    # one-shot publish makes the next batch pass the page's gate — which is
    # what updates the snapshot-derived "Current batch" content.
    session, model = make_session(epochs=1, phases={"train": 3})
    session.set_update_frequency(unit="batch", n=100)  # cadence never due
    session.detach()
    gate = _RefreshGate()
    gate.should_refresh(session)  # consume the page-open state
    refreshed: list[bool] = []

    async def fake_refresh() -> None:
        refreshed.append(True)

    asyncio.run(_refresh_now(session, fake_refresh))
    assert refreshed == [True]  # the click re-renders immediately
    with session.batch(phase="train", epoch=0):
        train_step(model)
    assert session.snapshot is not None  # the one-shot published...
    assert gate.should_refresh(session)  # ...and the gate re-renders from it
