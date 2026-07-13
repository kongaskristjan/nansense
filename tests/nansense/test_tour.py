"""Tests for the main view's guided tour (`nansense.ui.tour`).

The driver itself is browser JS; what's testable here is the contract it
relies on: the step data (short single sentences, resolvable selectors),
the config object, and the `data-tour` anchors actually existing in the UI
sources the selectors point at.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from nansense.ui.tour import (
    _TOUR_JS,
    SEEN_KEY,
    TourStep,
    tour_config,
    tour_steps,
)

_UI_DIR = Path(__file__).resolve().parents[2] / "nansense" / "ui"


def _ui_source() -> str:
    return "\n".join(
        p.read_text(encoding="utf-8") for p in _UI_DIR.glob("*.py")
    )


@pytest.mark.parametrize("locked", [True, False])
def test_steps_are_short_single_sentences(locked: bool) -> None:
    for step in tour_steps("conv1", locked=locked):
        assert step.text.endswith(".")
        # One sentence: no sentence break inside the text.
        assert ". " not in step.text
        assert len(step.text) <= 90
        assert step.selectors


@pytest.mark.parametrize(
    ("locked", "count", "has_step_controls"),
    [(True, 4, False), (False, 5, True)],
)
def test_step_controls_step_only_on_live_runs(
    locked: bool, count: int, has_step_controls: bool
) -> None:
    steps = tour_steps("conv1", locked=locked)
    assert len(steps) == count
    selectors = [sel for step in steps for sel in step.selectors]
    assert ('[data-tour="step-controls"]' in selectors) == has_step_controls


@pytest.mark.parametrize("locked", [True, False])
def test_data_tour_anchors_exist_in_ui_sources(locked: bool) -> None:
    """Every `data-tour` selector must have a matching attribute in the UI.

    This is the wiring test: a renamed or dropped anchor in
    `main_page.py` / `input_panel.py` / `top_bar.py` must fail here rather
    than silently leaving a tour step with no arrow.
    """
    source = _ui_source()
    for step in tour_steps("conv1", locked=locked):
        for sel in step.selectors:
            m = re.fullmatch(r'\[data-tour="([a-z-]+)"\]', sel)
            if m is None:
                continue  # the mermaid-node selector, checked below
            assert f'data-tour="{m.group(1)}"' in source, sel


def test_diagram_selector_matches_findmermaidnode_scheme() -> None:
    (first, *_), *_ = (s.selectors for s in tour_steps("fc_1", locked=True))
    # Trailing dash included, so `fc_1` never matches an `fc_1x...` node id.
    assert first == 'g.node[id*="-flowchart-fc_1-"]'
    # No known layer: fall back to pointing at any diagram node.
    (fallback, *_), *_ = (s.selectors for s in tour_steps(None, locked=True))
    assert fallback == "g.node"


def test_config_carries_driver_contract() -> None:
    config = tour_config(locked=True, layer_slug="conv1")
    assert config["autoStart"] is True
    assert config["autoWatchSlug"] == "conv1"
    assert config["seenKey"] == SEEN_KEY
    steps = config["steps"]
    assert isinstance(steps, list)
    assert len(steps) == len(tour_steps("conv1", locked=True))
    # Exactly one step may auto-show a card (the strips step).
    assert json.dumps(config).count('"ensureCard": true') == 1
    assert tour_config(locked=False, layer_slug=None)["autoStart"] is False


def test_driver_js_uses_the_config_hooks() -> None:
    # The JS blob and the Python config must agree on their two globals,
    # and `?`-button clicks depend on the start function's name.
    assert "window.nansenseTourConfig" in _TOUR_JS
    assert "window.nansenseStartTour" in _TOUR_JS
    assert "cfg.seenKey" in _TOUR_JS
    assert "cfg.autoWatchSlug" in _TOUR_JS
    # The blob is injected as one <script>; it must not close itself early.
    assert _TOUR_JS.count("</script>") == 1


def test_ensure_card_step_defaults_off() -> None:
    step = TourStep("Text.", ("a",))
    assert step.ensure_card is False
