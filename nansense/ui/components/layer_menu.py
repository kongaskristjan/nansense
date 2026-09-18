"""Shown-layer menu and collection indicator."""

from __future__ import annotations

from collections.abc import Callable
from urllib.parse import quote

from nicegui import ui


class LayerMenu:
    def __init__(
        self,
        layer_names: list[str],
        shown: frozenset[str],
        *,
        locked: bool,
        show_all: Callable[[], object],
        clear_all: Callable[[], None],
        toggle_stats: Callable[[], None],
    ) -> None:
        self.layers = tuple(layer_names)
        self.collecting: bool | None = None
        self.watch_chip = (
            ui.button(
                color="slate-100",
            )
            .classes("ml-auto text-amber-700 font-mono")
            .props("dense size=md no-caps")
        )
        with self.watch_chip:
            # The menu opens bottom-right of the chip, so a default
            # (below-anchored) tooltip would cover its first item — anchor
            # the tooltip to the chip's left instead.
            ui.tooltip("Shown layers").props('anchor="center left" self="center right"')
            # Icon and count are built as button children (rather than the
            # button's `icon=` / text) so the eye icon carries its own colour
            # independently of the amber count. `sync_stats_icon` swaps it
            # between `visibility` (green, collecting) and `visibility_off`
            # (red, paused) — the slashed eye of the per-card Unwatch button.
            self.stats_icon = ui.icon("visibility").classes("text-base")
            self.watch_count_label = ui.label(str(len(shown))).classes("ml-1")
            with ui.menu().props("anchor='bottom right' self='top right'"):
                # Plain block container, NOT a flex column: Firefox fails to
                # position/size a QMenu whose content root is a flex column,
                # so the menu opens collapsed (height 0) and looks like it
                # never opened. See quasarframework/quasar#16167. Block-level
                # children stack vertically anyway.
                with ui.element("div").classes("min-w-64"):
                    ui.menu_item(
                        "Show all layers",
                        on_click=show_all,
                    ).classes("text-sm").tooltip("Show every layer's card")
                    ui.menu_item(
                        "Hide all layers",
                        on_click=clear_all,
                    ).classes("text-sm").tooltip(
                        "Hide every card and drop the collected stats"
                    )
                    # A locked session pins the stats scope, so the
                    # pause toggle would be a silent no-op — hide it.
                    if not locked:
                        ui.menu_item(
                            "Toggle collecting stats",
                            on_click=toggle_stats,
                            auto_close=False,
                        ).classes("text-sm").tooltip(
                            "Pause or resume stats collection; what is "
                            "already collected is kept"
                        )
                    ui.separator()
                    # "Current-batch stats" submenu: every layer (watched
                    # or not), each routing to that layer's stats view —
                    # the current-batch phase is what makes unwatched
                    # layers viewable there. The nested menu's content root
                    # is a block div, so the Firefox QMenu caveat above
                    # doesn't apply.
                    with ui.menu_item("Current-batch stats", auto_close=False).classes(
                        "text-sm"
                    ):
                        # Anchored left like the chip's own tooltip: the
                        # layer list opens to the right, and a default
                        # (below-anchored) tooltip would overlap it.
                        ui.tooltip("Open a layer's stats page").props(
                            "anchor='center left' self='center right'"
                        )
                        with ui.item_section().props("side"):
                            ui.icon("chevron_right")
                        with ui.menu().props("anchor='top end' self='top start'"):
                            with ui.element("div").classes(
                                "min-w-56 max-h-96 overflow-auto"
                            ):
                                for name in layer_names:
                                    ui.menu_item(name).props(
                                        f'href="/stats?layer={quote(name)}"'
                                    ).classes("font-mono text-sm")
                    ui.separator()
                    self.watch_list_container = ui.element("div").classes("py-1")

    def update(self, shown: frozenset[str]) -> None:
        self.watch_count_label.text = str(len(shown))
        self.watch_list_container.clear()
        with self.watch_list_container:
            if not shown:
                ui.label("No layers selected").classes(
                    "px-3 py-2 text-slate-500 text-sm italic"
                )
                return
            # Section header: each entry below opens the stats view
            # focused on that layer.
            ui.label("Open statistics").classes(
                "px-3 pt-1 pb-0.5 text-xs uppercase tracking-wider "
                "text-slate-400 select-none"
            )
            for layer in self.layers:
                if layer not in shown:
                    continue
                # A real anchor (href) instead of a JS navigate: the
                # browser natively opens middle/ctrl clicks in a new tab
                # and plain clicks in the current one.
                ui.menu_item(layer).props(
                    f'href="/stats?layer={quote(layer)}"'
                ).classes("font-mono text-sm")

    def sync_collecting(self, collecting: bool) -> None:
        if collecting == self.collecting:
            return
        self.collecting = collecting
        self.stats_icon.set_name("visibility" if collecting else "visibility_off")
        self.stats_icon.classes(
            remove="text-green-600 text-red-600",
            add="text-green-600" if collecting else "text-red-600",
        )
