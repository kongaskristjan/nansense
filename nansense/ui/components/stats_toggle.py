"""Top-bar stats-collection toggle with the shown-layer count."""

from __future__ import annotations

from collections.abc import Callable

from nicegui import ui

# Material has no "off" variant of `bar_chart`, so the paused state draws
# its own diagonal strike (top-left to bottom-right, like the `_off` icons).
STATS_TOGGLE_CSS: str = """
<style>
.nansense-stats-icon { position: relative; display: inline-flex; }
.nansense-stats-icon.nansense-struck::after {
  content: ''; position: absolute; left: 50%; top: -8%; width: 2px;
  height: 116%; margin-left: -1px; background: currentColor;
  transform: rotate(-45deg); border-radius: 1px;
}
</style>
"""

_ON_TIP = "Collecting stats for the shown layers — click to pause"
_OFF_TIP = "Stats collection is off — click to collect for the shown layers"
_LOCKED_TIP = "Stats collection is pinned on in this demo"


class StatsToggle:
    """One button: a stats glyph (green on, red struck-through off) + count.

    `sync_collecting` reflects the session's state and is cheap to call on
    every tick; it rewrites the DOM only when the state flips. A locked
    session can't toggle, so the button is disabled there.
    """

    def __init__(
        self, shown: frozenset[str], *, locked: bool, toggle: Callable[[], None]
    ) -> None:
        self.collecting: bool | None = None
        # The wrapper carries the tour anchor: Quasar's q-btn doesn't reliably
        # forward `data-*` attrs to its rendered DOM.
        with ui.element("div").props('data-tour="stats-toggle"').classes("ml-auto"):
            self.button = (
                ui.button(color="slate-100", on_click=toggle)
                .classes("text-amber-700 font-mono")
                .props("dense size=md no-caps")
            )
        with self.button:
            self.tooltip = ui.tooltip(_LOCKED_TIP if locked else _OFF_TIP)
            self.icon = ui.icon("bar_chart").classes(
                "text-base nansense-stats-icon"
            )
            self.count_label = ui.label(str(len(shown))).classes("ml-1")
        if locked:
            self.button.disable()
        self.locked = locked

    def update(self, shown: frozenset[str]) -> None:
        self.count_label.text = str(len(shown))

    def sync_collecting(self, collecting: bool) -> None:
        if collecting == self.collecting:
            return
        self.collecting = collecting
        self.icon.classes(
            remove="text-green-600 text-red-600 nansense-struck",
            add="text-green-600" if collecting else "text-red-600 nansense-struck",
        )
        if not self.locked:
            self.tooltip.set_text(_ON_TIP if collecting else _OFF_TIP)
