"""Content checks for the docs site pages that ship hand-written HTML/JS.

The playground page (`docs/playground.md`) builds a custom header toolbar out
of inline HTML/CSS/JS, so a regression there is invisible to `mkdocs build`
(the markup is valid either way). These guard the pieces that matter.
"""

from __future__ import annotations

from pathlib import Path

_DOCS = Path(__file__).parent.parent / "docs"


def test_playground_page_has_share_button() -> None:
    """The playground header carries a "Share playground" button (mirrors the
    app's "Share nansense" button). The shared URL is derived from the live
    location so it tracks the deployed version prefix and variant hash instead
    of pinning a hardcoded version."""
    text = (_DOCS / "playground.md").read_text(encoding="utf-8")
    assert 'class="pg-share-btn"' in text
    assert "Share playground" in text
    assert "location.origin + location.pathname + location.hash" in text
    assert "https://kongaskristjan.github.io/nansense/dev/playground/" not in text
