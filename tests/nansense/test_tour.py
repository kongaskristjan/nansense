"""Tests for the per-page guided tours (`nansense.ui.tour`).

The driver itself is browser JS; what's testable here is the contract it
relies on: the step data (short single sentences, resolvable selectors),
the config object, per-page seen keys, and the `data-tour` anchors
actually existing in the UI sources the selectors point at.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from nansense.ui.tour import (
    _TOUR_JS,
    SEEN_KEY_PREFIX,
    TourStep,
    experiment_tour_steps,
    main_tour_steps,
    seen_key,
    stats_tour_steps,
    tour_config,
    weights_tour_steps,
)

_UI_DIR = Path(__file__).resolve().parents[2] / "nansense" / "ui"

# Every page's steps, for both session flavors — the shape the pages
# actually install (see each page's `add_tour` call).
_ALL_PAGE_STEPS: list[tuple[str, bool, list[TourStep]]] = [
    (page, locked, steps)
    for locked in (True, False)
    for page, steps in [
        ("main", main_tour_steps("conv1", locked=locked)),
        ("stats", stats_tour_steps()),
        ("weights", weights_tour_steps()),
        ("experiment", experiment_tour_steps(locked=locked)),
    ]
]


def _ui_source() -> str:
    return "\n".join(
        p.read_text(encoding="utf-8") for p in _UI_DIR.glob("*.py")
    )


@pytest.mark.parametrize(("page", "locked", "steps"), _ALL_PAGE_STEPS)
def test_steps_are_short_and_skimmable(
    page: str, locked: bool, steps: list[TourStep]
) -> None:
    assert steps
    for step in steps:
        assert step.text.endswith(".")
        # Skimmable: at most two sentences and ~200 chars per bubble.
        assert step.text.count(". ") <= 1
        assert len(step.text) <= 200
        assert step.selectors


@pytest.mark.parametrize(
    "steps",
    [
        stats_tour_steps(),
        weights_tour_steps(),
        experiment_tour_steps(locked=False),
        experiment_tour_steps(locked=True),
    ],
)
def test_subpage_tours_stay_short(steps: list[TourStep]) -> None:
    """Subpage tours cover only the non-obvious bits: at most 4 steps."""
    assert 1 <= len(steps) <= 4


@pytest.mark.parametrize(
    ("locked", "main_count", "experiment_count", "has_step_controls"),
    [(True, 4, 2, False), (False, 5, 2, True)],
)
def test_step_controls_step_only_on_live_runs(
    locked: bool,
    main_count: int,
    experiment_count: int,
    has_step_controls: bool,
) -> None:
    for steps, count in [
        (main_tour_steps("conv1", locked=locked), main_count),
        (experiment_tour_steps(locked=locked), experiment_count),
    ]:
        assert len(steps) == count
        selectors = [sel for step in steps for sel in step.selectors]
        assert (
            '[data-tour="step-controls"]' in selectors
        ) == has_step_controls


@pytest.mark.parametrize("locked", [True, False])
def test_experiment_run_step_only_on_the_playground(locked: bool) -> None:
    """Only the locked playground turns experiment auto-run off, so only
    its tour points out the manual Run / Cancel pair."""
    selectors = [
        sel
        for step in experiment_tour_steps(locked=locked)
        for sel in step.selectors
    ]
    assert ('[data-tour="run"]' in selectors) == locked


@pytest.mark.parametrize(("page", "locked", "steps"), _ALL_PAGE_STEPS)
def test_data_tour_anchors_exist_in_ui_sources(
    page: str, locked: bool, steps: list[TourStep]
) -> None:
    """Every `data-tour` selector must have a matching attribute in the UI.

    This is the wiring test: a renamed or dropped anchor in any page module
    must fail here rather than silently leaving a tour step with no arrow.
    """
    source = _ui_source()
    for step in steps:
        for sel in step.selectors:
            # `search`, not `fullmatch`: the main tour's card anchors carry a
            # `[data-layer="…"] ` scope prefix (the mermaid-node selector has
            # no `data-tour` at all and is checked below).
            m = re.search(r'\[data-tour="([a-z-]+)"\]', sel)
            if m is None:
                continue
            assert f'data-tour="{m.group(1)}"' in source, sel


def test_diagram_selector_matches_findmermaidnode_scheme() -> None:
    (first, *_), *_ = (s.selectors for s in main_tour_steps("fc_1", locked=True))
    # Trailing dash included, so `fc_1` never matches an `fc_1x...` node id.
    assert first == 'g.node[id*="-flowchart-fc_1-"]'
    # No known layer: fall back to pointing at any diagram node.
    (fallback, *_), *_ = (
        s.selectors for s in main_tour_steps(None, locked=True)
    )
    assert fallback == "g.node"


@pytest.mark.parametrize("locked", [True, False])
def test_main_tour_stays_on_one_layer(locked: bool) -> None:
    """Every main-tour arrow belongs to the same layer.

    The first step points at a diagram node; each card-bound step after it
    is scoped to that layer's card (`data-layer="<slug>"`), so the tour can
    never point at a card the visitor didn't open — the auto-shown card and
    the diagram node the tour opened with are one and the same layer.
    """
    click, *rest = main_tour_steps("stage1_0_conv1", locked=locked)
    assert click.selectors == ('g.node[id*="-flowchart-stage1_0_conv1-"]',)
    for step in rest:
        for sel in step.selectors:
            if "data-tour" not in sel:
                continue
            if step.ensure_card:
                assert sel.startswith('[data-layer="stage1_0_conv1"] '), sel
            else:
                # The input pane and the step controls live outside the
                # cards, so they are page-wide anchors.
                assert "data-layer" not in sel, sel


def test_card_anchors_are_unscoped_without_a_layer() -> None:
    """A model with no captured layers has no card to scope to, so the
    anchors stay page-wide (matching the `g.node` diagram fallback)."""
    _, strips, *_ = main_tour_steps(None, locked=True)
    assert strips.selectors == ('[data-tour="strips"]',)


@pytest.mark.parametrize(
    ("has_weights", "anchors"),
    [
        (True, ("weights", "experiment", "stats")),
        # A ReLU/add card has no Weights button, so the step drops that
        # arrow instead of pointing it at some other layer's card.
        (False, ("experiment", "stats")),
    ],
)
def test_buttons_step_matches_the_buttons_the_card_has(
    has_weights: bool, anchors: tuple[str, ...]
) -> None:
    steps = main_tour_steps("relu1", locked=True, has_weights=has_weights)
    (buttons,) = [s for s in steps if s.ensure_card and "deeper" in s.text]
    assert buttons.selectors == tuple(
        f'[data-layer="relu1"] [data-tour="{a}"]' for a in anchors
    )
    # The message names exactly the buttons it points at.
    assert ("Weights" in buttons.text) == has_weights
    assert "Experiments" in buttons.text and "Stats" in buttons.text


def test_stats_steps_force_the_views_they_describe() -> None:
    """Each view-bound stats step names a real View-dropdown entry.

    The driver switches the page to `ensure_view` when the step shows, so
    the names must match the stats page's option constants exactly; the
    dropdowns step forces nothing.
    """
    from nansense.ui.stats_page import (
        _VIEW_GRAPHS,
        _VIEW_HISTOGRAM,
        _VIEW_MINMAX,
    )

    steps = stats_tour_steps()
    assert [s.ensure_view for s in steps] == [
        None,
        _VIEW_HISTOGRAM,
        _VIEW_MINMAX,
        _VIEW_GRAPHS,
    ]
    # The dropdowns step covers all three selectors in one message.
    assert len(steps[0].selectors) == 3
    # The histograms step draws one arrow per tensor kind.
    assert len(steps[1].selectors) == 2


def test_weights_tour_leads_with_the_weight_strip() -> None:
    """The one step's arrow lands on the weight row, and its message covers
    the gradient/optimizer strips below it."""
    (only,) = weights_tour_steps()
    assert only.selectors == ('[data-tour="weight-strips"]',)
    assert "gradient" in only.text and "optimizer" in only.text


@pytest.mark.parametrize("locked", [True, False])
def test_main_tour_opens_by_inviting_a_diagram_click(locked: bool) -> None:
    """The main tour leads with the diagram click the page turns on: one
    arrow, on a layer node, before any step talks about a card."""
    first, second, *_ = main_tour_steps("conv1", locked=locked)
    assert first.selectors == ('g.node[id*="-flowchart-conv1-"]',)
    # The invitation stands on its own — nothing is auto-shown before the
    # visitor has been told what a click does.
    assert not first.ensure_card
    # What the click produces is the next step's subject.
    assert "card" in second.text


def test_seen_keys_are_distinct_per_page() -> None:
    keys = [seen_key(p) for p in ("main", "stats", "weights", "experiment")]
    assert len(set(keys)) == len(keys)
    # The main page keeps the original unsuffixed key, so playground
    # visitors who already dismissed its tour aren't replayed.
    assert seen_key("main") == SEEN_KEY_PREFIX
    assert seen_key("stats") == f"{SEEN_KEY_PREFIX}-stats"


def test_config_carries_driver_contract() -> None:
    steps = main_tour_steps("conv1", locked=True)
    config = tour_config(
        steps, page="main", auto_start=True, auto_watch_slug="conv1"
    )
    assert config["autoStart"] is True
    assert config["autoWatchSlug"] == "conv1"
    assert config["seenKey"] == seen_key("main")
    payload = config["steps"]
    assert isinstance(payload, list)
    assert len(payload) == len(steps)
    # Exactly two steps may auto-show a card — the ones that talk about it:
    # the strips step and the buttons step.
    assert json.dumps(config).count('"ensureCard": true') == 2
    # Exactly one step re-opens the input pane (the sample step).
    assert json.dumps(config).count('"ensureInput": true') == 1
    subpage = tour_config(stats_tour_steps(), page="stats", auto_start=False)
    assert subpage["autoStart"] is False
    assert subpage["autoWatchSlug"] is None
    assert subpage["seenKey"] == seen_key("stats")
    # No subpage step auto-shows a layer card or re-opens the input pane —
    # those are main-view mechanics — but the view-bound stats steps
    # serialize their `ensureView`.
    assert '"ensureCard": true' not in json.dumps(subpage)
    assert '"ensureInput": true' not in json.dumps(subpage)
    assert '"ensureView": "HISTOGRAM"' in json.dumps(subpage)


def test_driver_js_uses_the_config_hooks() -> None:
    # The JS blob and the Python config must agree on their two globals,
    # and `?`-button clicks depend on the start function's name.
    assert "window.nansenseTourConfig" in _TOUR_JS
    assert "window.nansenseStartTour" in _TOUR_JS
    assert "cfg.seenKey" in _TOUR_JS
    assert "cfg.autoWatchSlug" in _TOUR_JS
    # View-bound steps reach the stats page through this event name.
    assert "nansense_tour_set_view" in _TOUR_JS
    # Card-needing steps auto-show a layer through a show-only event — the
    # toggle event a diagram click emits would hide an already-open card.
    assert "nansense_tour_show_layer" in _TOUR_JS
    assert "nansense_toggle_layer" not in _TOUR_JS
    # The sample step re-opens the input pane through this event name.
    assert "nansense_tour_show_input" in _TOUR_JS
    # Fresh runs are bracketed by start/end events — the stats page uses
    # them to restore the view its view-bound steps switched away.
    assert "nansense_tour_start" in _TOUR_JS
    assert "nansense_tour_end" in _TOUR_JS
    # Quasar fields get their data-tour forwarded to the inner native
    # control; the driver must widen matches to the whole field.
    assert ".q-field" in _TOUR_JS
    # The blob is injected as one <script>; it must not close itself early.
    assert _TOUR_JS.count("</script>") == 1


def test_ensure_card_step_defaults_off() -> None:
    step = TourStep("Text.", ("a",))
    assert step.ensure_card is False
    assert step.ensure_input is False
    assert step.ensure_view is None


@pytest.mark.parametrize("locked", [True, False])
def test_main_steps_ensure_their_targets_are_visible(locked: bool) -> None:
    """The card-bound steps (strips, buttons) auto-show the card their
    arrows need, and the sample step re-opens the input pane the top bar's
    image button can hide."""
    click, strips, buttons, sample, *_ = main_tour_steps("conv1", locked=locked)
    assert not click.ensure_card and not click.ensure_input
    assert strips.ensure_card and not strips.ensure_input
    assert buttons.ensure_card and not buttons.ensure_input
    # The pane and card mechanics stay separate per step.
    assert sample.ensure_input and not sample.ensure_card
