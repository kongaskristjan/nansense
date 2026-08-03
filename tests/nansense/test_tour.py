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

_REPO = Path(__file__).resolve().parents[2]
_UI_DIR = _REPO / "nansense" / "ui"
# The docs pages' half of the seen-flag protocol (see `test_tour_flags_*`).
_EMBED_JS = _REPO / "docs" / "javascripts" / "playground-embed.js"

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
        # A visitor reads these seconds after landing: one sentence, one
        # line. Anything longer belongs on the page, not in the bubble.
        assert step.text.count(". ") == 0
        assert len(step.text) <= 100
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
    [(True, 5, 2, False), (False, 6, 2, True)],
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
    # The message names exactly the buttons it points at, with the labels
    # the card actually prints on them ("Experiment", not "Experiments").
    assert ("Weights" in buttons.text) == has_weights
    assert "Experiment and" in buttons.text or "Experiment, and" in buttons.text
    assert "Stats" in buttons.text
    assert "Experiments" not in buttons.text


def test_buttons_step_uses_the_cards_own_button_labels() -> None:
    """The names in the buttons step are the card's real button labels.

    The message is only useful if the visitor can match each word to a
    button under an arrow, so a renamed card button must fail here.
    """
    source = _ui_source()
    (buttons,) = [
        s
        for s in main_tour_steps("conv1", locked=True, has_weights=True)
        if s.ensure_card and "deeper" in s.text
    ]
    named = buttons.text.removesuffix(" go deeper.").replace(" and", "")
    labels = [word.strip() for word in named.split(",")]
    assert labels == ["Weights", "Experiment", "Stats"]
    for label in labels:
        assert re.search(rf'ui\.button\(\s*"{label}"', source), label


@pytest.mark.parametrize("locked", [True, False])
def test_card_steps_name_the_rows_and_the_color_scale(locked: bool) -> None:
    """The card's only written key: which row is which, and what the colors
    mean.

    The UI itself never says either — the markers read ACTIVATIONS /
    GRADIENTS with no top/bottom cue and the colorbar prints only `+x` /
    `0` / `-x` — so the two steps carry it, in that order, and the colors
    step rings the colorbar it describes.
    """
    _, strips, colors, *_ = main_tour_steps("conv1", locked=locked)
    assert strips.text.index("activations") < strips.text.index("gradients")
    assert "above" in strips.text and "below" in strips.text
    assert colors.selectors == ('[data-layer="conv1"] [data-tour="legend"]',)
    for word in ("Red", "positive", "blue", "negative", "white", "zero"):
        assert word in colors.text, word


def test_color_step_matches_the_renderers_colormap() -> None:
    """Red is the positive end and blue the negative one — asserted against
    the colormap itself, so flipping it must flip the tour's wording."""
    import numpy as np

    from nansense.ui.render import _diverging_colormap

    (positive, zero, negative) = _diverging_colormap(
        np.array([1.0, 0.0, -1.0], dtype=np.float32)
    )
    (colors,) = [
        s for s in main_tour_steps("conv1", locked=True) if "Red" in s.text
    ]
    # Channel order is R, G, B: the ends max out the channel the step names.
    assert positive.argmax() == 0 and colors.text.startswith("Red is positive")
    assert negative.argmax() == 2 and "blue negative" in colors.text
    assert tuple(zero) == (255, 255, 255) and "white zero" in colors.text


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


def test_tour_flags_ask_the_embedding_page_first() -> None:
    """The driver asks its embedder for the seen flag, and tells it on exit.

    The hosted playground swaps two Hugging Face Spaces — two origins —
    into one docs-page frame, so a flag kept only in the frame's own
    localStorage replays every tour when the visitor switches demos.
    """
    assert "window.parent.postMessage" in _TOUR_JS
    # Both directions: read the flag on load, record it on dismissal.
    assert "tellHost('get')" in _TOUR_JS
    assert "tellHost('set')" in _TOUR_JS
    # Replies are only taken from the embedder, and only for our own key.
    assert "e.source !== window.parent" in _TOUR_JS
    assert "data.key !== cfg.seenKey" in _TOUR_JS
    # Unembedded runs (local, a direct Space visit, huggingface.co's own
    # Space wrapper) must keep working off this origin's own flag.
    assert "localStorage.getItem(cfg.seenKey)" in _TOUR_JS
    assert "localStorage.setItem(cfg.seenKey" in _TOUR_JS


def test_tour_flags_match_the_docs_pages_store() -> None:
    """The docs half of the protocol agrees with the driver's half.

    `docs/javascripts/playground-embed.js` holds the flags for both demos
    (and for both embeds — the home page and `/playground/`). It answers
    only the playground origins and only keys the tour owns, so a renamed
    message or a new page key must fail here rather than in production.
    """
    host = _EMBED_JS.read_text(encoding="utf-8")
    for message in ("get", "set", "is"):
        assert f'"{message}"' in host, message
        assert f"'{message}'" in _TOUR_JS, message
    assert "nansenseTour" in host and "nansenseTour" in _TOUR_JS
    # Every page's key must pass the host's allowlist, which is what keeps
    # a frame from writing arbitrary keys into the docs origin.
    (allowed,) = re.findall(r"TOUR_KEY = /(\S+)/;", host)
    for page in ("main", "stats", "weights", "experiment"):
        assert re.fullmatch(allowed, seen_key(page)), page
    assert not re.fullmatch(allowed, "some-other-key")
    # Only the two demo Spaces are answered, and each reply goes back to
    # the asking origin rather than to `*`.
    assert host.count("hf.space") == 2
    assert "TOUR_ORIGINS.indexOf(event.origin) < 0" in host
    assert re.search(r"postMessage\(.*?event\.origin", host, re.S)
    assert '"*"' not in host


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
    # Exactly three steps may auto-show a card — the ones that talk about
    # it: the strips, colors, and buttons steps.
    assert json.dumps(config).count('"ensureCard": true') == 3
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
def test_sample_step_rings_the_image_and_its_selector(locked: bool) -> None:
    """"The input" is the picture plus the spinner that picks it, so the
    step draws an arrow to each rather than to the spinner alone."""
    _, _, _, _, sample, *_ = main_tour_steps("conv1", locked=locked)
    assert sample.selectors == (
        '[data-tour="input-image"]',
        '[data-tour="sample"]',
    )


@pytest.mark.parametrize("locked", [True, False])
def test_main_steps_ensure_their_targets_are_visible(locked: bool) -> None:
    """The card-bound steps (strips, colors, buttons) auto-show the card
    their arrows need, and the sample step re-opens the input pane the top
    bar's image button can hide."""
    click, strips, colors, buttons, sample, *_ = main_tour_steps(
        "conv1", locked=locked
    )
    assert not click.ensure_card and not click.ensure_input
    assert strips.ensure_card and not strips.ensure_input
    assert colors.ensure_card and not colors.ensure_input
    assert buttons.ensure_card and not buttons.ensure_input
    # The pane and card mechanics stay separate per step.
    assert sample.ensure_input and not sample.ensure_card
