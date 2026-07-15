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
def test_steps_are_short_single_sentences(
    page: str, locked: bool, steps: list[TourStep]
) -> None:
    assert steps
    for step in steps:
        assert step.text.endswith(".")
        # One sentence: no sentence break inside the text.
        assert ". " not in step.text
        assert len(step.text) <= 140
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
            m = re.fullmatch(r'\[data-tour="([a-z-]+)"\]', sel)
            if m is None:
                continue  # the mermaid-node selector, checked below
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
    """The strips step comes first and its arrow lands on the weight row
    (the message covers the gradient/optimizer strips below it)."""
    first, second = weights_tour_steps()
    assert first.selectors == ('[data-tour="weight-strips"]',)
    assert "gradient" in first.text and "optimizer" in first.text
    assert second.selectors == ('[data-tour="axes"]',)


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
    # Exactly one step may auto-show a card (the main tour's strips step).
    assert json.dumps(config).count('"ensureCard": true') == 1
    subpage = tour_config(stats_tour_steps(), page="stats", auto_start=False)
    assert subpage["autoStart"] is False
    assert subpage["autoWatchSlug"] is None
    assert subpage["seenKey"] == seen_key("stats")
    # No subpage step auto-shows a layer card — that's a main-view mechanic —
    # but the view-bound stats steps serialize their `ensureView`.
    assert '"ensureCard": true' not in json.dumps(subpage)
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
    # Quasar fields get their data-tour forwarded to the inner native
    # control; the driver must widen matches to the whole field.
    assert ".q-field" in _TOUR_JS
    # The blob is injected as one <script>; it must not close itself early.
    assert _TOUR_JS.count("</script>") == 1


def test_ensure_card_step_defaults_off() -> None:
    step = TourStep("Text.", ("a",))
    assert step.ensure_card is False
    assert step.ensure_view is None
