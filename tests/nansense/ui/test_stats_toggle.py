"""The top-bar stats toggle reflects collection state without rebuilding."""

from __future__ import annotations

import pytest
from nicegui import ui

from nansense.ui.components.stats_toggle import StatsToggle


@pytest.mark.parametrize("locked", [False, True])
def test_toggle_reflects_collection_and_count(locked: bool) -> None:
    clicks: list[int] = []
    with ui.card():
        toggle = StatsToggle(
            frozenset({"a"}), locked=locked, toggle=lambda: clicks.append(1)
        )
    assert toggle.count_label.text == "1"
    toggle.update(frozenset({"a", "b"}))
    assert toggle.count_label.text == "2"

    toggle.sync_collecting(False)
    assert "nansense-struck" in toggle.icon.classes
    assert "text-red-600" in toggle.icon.classes
    toggle.sync_collecting(True)
    assert "nansense-struck" not in toggle.icon.classes
    assert "text-green-600" in toggle.icon.classes
    # Locked: the button can't toggle, and its tooltip says why instead of
    # promising a click does something.
    assert toggle.button.enabled is not locked
    assert ("pinned" in toggle.tooltip.text) is locked
    if not locked:
        assert "click to pause" in toggle.tooltip.text
