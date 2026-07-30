"""Content checks for the docs site pages that ship hand-written HTML/JS.

The playground page (`docs/playground.md`) builds a custom header toolbar out
of inline HTML/CSS/JS, so a regression there is invisible to `mkdocs build`
(the markup is valid either way). These guard the pieces that matter.
"""

from __future__ import annotations

import re
from pathlib import Path

_DOCS = Path(__file__).parent.parent / "docs"


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


def test_home_embed_switch_offers_both_hosted_spaces() -> None:
    """Every variant button must have a Space to switch to, and the iframe's
    initial `src` must be the variant the switch starts out pressing."""
    index = (_DOCS / "index.md").read_text(encoding="utf-8")
    script = (_DOCS / "javascripts" / "playground-embed.js").read_text(encoding="utf-8")
    variants = re.findall(r'data-variant="(\w+)"', index)
    assert set(variants) == {"imagenette", "mnist"}
    spaces = dict(re.findall(r'(\w+): "(https://[^"]+\.hf\.space)"', script))
    assert set(spaces) == set(variants)

    pressed = re.findall(r'data-variant="(\w+)" aria-pressed="true"', index)
    iframe = re.search(r'<iframe[^>]*class="pg-embed__frame"[^>]*>', index)
    assert iframe is not None
    assert len(pressed) == 1
    assert spaces[pressed[0]] in iframe.group(0)
