"""Content checks for the docs site pages that ship hand-written HTML/JS.

The playground page (`docs/playground.md`) builds a custom header toolbar out
of inline HTML/CSS/JS, so a regression there is invisible to `mkdocs build`
(the markup is valid either way). These guard the pieces that matter.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

_DOCS = Path(__file__).parent.parent / "docs"

_SPACES = {
    "mnist": "https://kongaskristjan-nansense-playground-mnist.hf.space",
    "imagenette": "https://kongaskristjan-nansense-playground.hf.space",
}


def test_playground_iframe_delegates_clipboard_write() -> None:
    """Sharing moved from the docs header into the app's own Share dialog,
    which copies links via the async Clipboard API — inside the cross-origin
    playground iframe that only works if the embedding frame delegates
    `clipboard-write` through the `allow` attribute."""
    text = (_DOCS / "playground.md").read_text(encoding="utf-8")
    assert "pg-share-btn" not in text  # the header button is gone
    iframe = re.search(r"<iframe[^>]*>", text)
    assert iframe is not None
    assert 'allow="fullscreen; clipboard-write"' in iframe.group(0)


def test_home_embed_zooms_the_frame_instead_of_transforming_it() -> None:
    """The home page embeds the app, which is desktop-first: below 800px it pans
    instead of reflowing (`nansense/ui/static.py`), so the frame is rendered at a
    viewport wider than the column. Only `zoom` does that — `transform: scale()`
    shrinks what the app already drew at the column's width, leaving it just as
    cramped (and the frame short of its box)."""
    css = (_DOCS / "stylesheets" / "extra.css").read_text(encoding="utf-8")
    assert "zoom: var(--pg-scale" in css
    assert "transform: scale(var(--pg-scale" not in css


@pytest.mark.parametrize(
    ("page", "script"),
    [
        ("index.md", "javascripts/playground-embed.js"),  # embed on the home page
        ("playground.md", "playground.md"),  # full-page app, script inlined
    ],
)
def test_playground_switch_starts_on_the_easy_mnist_variant(page: str, script: str) -> None:
    """Both switches offer the same two hosted Spaces, and all four places that
    encode the starting variant — button order, `aria-pressed`, the iframe's
    `src` and the script's `current` — must agree on MNIST. It leads because a
    LeNet on digits is the readable one; the ResNet is the advanced follow-up."""
    text = (_DOCS / page).read_text(encoding="utf-8")
    script_text = (_DOCS / script).read_text(encoding="utf-8")

    buttons = re.findall(r"<button[^>]*data-variant=\"(\w+)\"[^>]*>([^<]*)</button>", text)
    assert [variant for variant, _ in buttons] == ["mnist", "imagenette"]
    labels = dict(buttons)
    assert labels["mnist"].startswith("Easy")
    assert labels["imagenette"].startswith("Advanced")

    assert dict(re.findall(r'(\w+): "(https://[^"]+\.hf\.space)"', script_text)) == _SPACES
    pressed = re.findall(r'data-variant="(\w+)" aria-pressed="true"', text)
    assert pressed == ["mnist"]
    iframe = re.search(r"<iframe[^>]*>", text)
    assert iframe is not None
    assert _SPACES["mnist"] in iframe.group(0)
    assert 'var current = "mnist"' in script_text


def _declarations(css: str, selector: str) -> str:
    """The declarations of the rule for `selector` in `css`."""
    rule = re.search(re.escape(selector) + r"\s*\{([^}]*)\}", css)
    assert rule is not None, f"no rule for {selector}"
    return rule.group(1)


def test_playground_collapses_the_nav_into_the_drawer_at_every_width() -> None:
    """A full-screen app has no room for the nav column, so the page hides it —
    but Material only offers the hamburger drawer below its 76.25em breakpoint,
    above which hiding the column drops the nav for good. The inline CSS
    re-creates the drawer above the breakpoint; without these pieces the wide
    playground is a dead end with no way back into the docs."""
    text = (_DOCS / "playground.md").read_text(encoding="utf-8")
    assert "  - navigation\n" in text  # front matter: no nav column

    desktop = re.search(r"@media screen and \(min-width: 76\.25em\) \{(.*?)\n  \}", text, re.S)
    assert desktop is not None
    assert "display: inline-block" in _declarations(desktop.group(1), '.md-header__button[for="__drawer"]')

    panel = _declarations(desktop.group(1), ".md-sidebar--primary")
    assert "display: block" in panel  # the front matter marks the sidebar hidden
    assert "position: fixed" in panel
    assert "left: -12.1rem" in panel  # parked off-canvas until opened
    opened = _declarations(desktop.group(1), '[data-md-toggle="drawer"]:checked ~ .md-container .md-sidebar--primary')
    assert "transform: translateX(12.1rem)" in opened


def test_playground_deep_link_names_the_non_default_variant() -> None:
    """`/playground/` keeps the advanced variant addressable as `#imagenette`;
    the default one needs no hash, so `#mnist` must not linger as a magic
    string that would strand the page on a variant it no longer starts with."""
    text = (_DOCS / "playground.md").read_text(encoding="utf-8")
    assert '"#imagenette"' in text
    assert '"#mnist"' not in text
