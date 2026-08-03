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


def test_playground_deep_link_names_the_non_default_variant() -> None:
    """`/playground/` keeps the advanced variant addressable as `#imagenette`;
    the default one needs no hash, so `#mnist` must not linger as a magic
    string that would strand the page on a variant it no longer starts with."""
    text = (_DOCS / "playground.md").read_text(encoding="utf-8")
    assert '"#imagenette"' in text
    assert '"#mnist"' not in text
