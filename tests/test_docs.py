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
