"""The main page: architecture diagram, watched-layer cards, input pane."""

from __future__ import annotations

import asyncio
import json
import os
import threading
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from urllib.parse import quote

import torch
from nicegui import ui
from nicegui.events import GenericEventArguments
from torch import Tensor

from nansense.probe import ProbeResult
from nansense.recording import RecordedView
from nansense.session import BatchSnapshot, Session, StatsScope
from nansense.ui.common import (
    _b64_img_src,
    _install_panel_resize,
    _notice_banner,
    _page_scaffold,
    _refuse_unwatch_while_recording,
    _resizable_pane_props,
    _resize_handle,
    _strip_html,
    _strip_marker,
)
from nansense.ui.graph import slug_map
from nansense.ui.input_panel import InputPanel
from nansense.input_config import InputTransform, MeanStd, resolve_per_input
from nansense.ui.render import (
    input_blank_warning,
    probe_act_tensor,
    render_image,
    render_input_legend,
    render_strip,
    tensor_hw,
)
from nansense.ui.static import (
    _ARCHITECTURE_CLICK_CSS,
    _ARCHITECTURE_CLICK_JS,
    _STRIP_MARKER_CSS,
)
from nansense.ui.top_bar import (
    _add_error_banner,
    _add_repo_logo,
    _add_settings_button,
    _add_share_button,
    _add_step_controls,
    _add_tour_button,
    _build_step_until_custom_dialog,
    _refresh_button,
    _top_bar_row,
)
from nansense.ui.tour import add_tour, main_tour_steps


@dataclass
class _PageState:
    last_snapshot: BatchSnapshot | None = None
    last_probe: ProbeResult | None = None
    dirty: bool = False
    rendering: bool = False
    # The shown set this connection last reflected in its DOM (card
    # visibility, amber classes, chip). The tick compares it against the
    # current shown set so changes made elsewhere (another tab, in the
    # coupled scope) propagate here too.
    last_watched: frozenset[str] = frozenset()
    # This connection's own shown cards, used in the decoupled stats scopes
    # (`none` / `all`) where showing a card must not touch the session's
    # global watched set — so tabs are independent. Seeded from the watched
    # set when a decoupled scope is entered.
    shown: set[str] = field(default_factory=set)
    # The stats scope last seen by the tick, to detect scope switches.
    last_scope: StatsScope | None = None


# Shared pool for strip rendering. Per-layer renders are independent and the
# heavy parts (torch interpolate, numpy colormap, PIL PNG encode) release the
# GIL, so a new snapshot's strips render in parallel across cores. Workers
# spawn lazily, so the pool costs nothing until the first frame.
_RENDER_POOL = ThreadPoolExecutor(
    max_workers=min(8, os.cpu_count() or 1), thread_name_prefix="nansense-render"
)

# Resting width of the right-hand input pane. Proportional to the viewport so a
# narrow (phone) screen doesn't get a pane wide enough to crowd out the layer
# strips, floored at 11rem (below that the probe controls get unusable) and
# capped at 18rem — the old fixed `w-72`, so wide monitors are unchanged and the
# pane never bloats. Expressed as `width` (clamp) alone, never min-/max-width,
# so the drag handle's inline px width still overrides it freely: the cap bounds
# only the default, not how far the pane can be resized (see `_PANEL_RESIZE_JS`).
_INPUT_PANE_WIDTH: str = "w-[clamp(11rem,25vw,18rem)]"


class _RenderCache:
    """Strip-HTML cache for the main page, shared across connections.

    Valid for exactly one render source at a time — a `BatchSnapshot` or a
    `ProbeResult`: the cache holds a strong reference to the source it was
    filled against and resets whenever a different one shows up (identity
    comparison — both are frozen and every publish creates a new object, so
    identity is exactly "same capture"). Within a source, entries are keyed
    by `(name, kind, sample_idx)`, so flipping the sample spinner back to a
    value already seen, or a second browser tab on the same session, becomes
    a dict lookup instead of a re-render. `_MAX_ENTRIES` bounds a long
    sample-scrubbing session; overflowing simply resets the cache.
    """

    _MAX_ENTRIES: int = 4096

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._source: object | None = None
        self._entries: dict[tuple[str, str, int], str] = {}

    def get_or_render(
        self,
        source: object,
        key: tuple[str, str, int],
        render: Callable[[], str],
    ) -> str:
        with self._lock:
            if source is not self._source:
                self._source = source
                self._entries = {}
            entries = self._entries
            cached = entries.get(key)
        if cached is not None:
            return cached
        html = render()
        with self._lock:
            # Drop the result if a newer source displaced the cache while
            # this render was in flight — `entries` would be the stale dict.
            if self._source is source:
                if len(entries) >= self._MAX_ENTRIES:
                    entries.clear()
                entries[key] = html
        return html


def _layer_info_script(layer_info: dict[str, str], slugs: dict[str, str]) -> str:
    """Body script publishing the slug -> hyperparameter map the tooltip reads.

    Keyed by the same collision-free `slugs` the diagram and cards use, so a
    tooltip lookup lines up with the hovered node/card. Empty entries (graph
    inputs, relu, add, …) are dropped so the client can treat "no entry" as
    "no tooltip". The `</` escape keeps a pathological `extra_repr` from
    closing the script tag early.
    """
    payload = json.dumps(
        {
            slugs[name]: info
            for name, info in layer_info.items()
            if info and name in slugs
        }
    ).replace("</", "<\\/")
    return f"<script>window.nansenseLayerInfo = {payload};</script>"


def _seed_shown(
    watched: frozenset[str], focus_layer: str, layer_names: list[str]
) -> set[str]:
    """A new tab's decoupled shown set: the watched seed plus the deep link.

    `focus_layer` is the `?layer=` query param — the locked playground's
    subpages put the layer they show into their Back button's href, so
    returning to the main page shows the card the visitor came from even
    when it isn't part of the seed set. Unknown names are ignored.
    """
    shown = set(watched)
    if focus_layer in layer_names:
        shown.add(focus_layer)
    return shown


def _build_page(
    session: Session,
    mermaid_src: str,
    layer_names: list[str],
    *,
    focus_layer: str = "",
    input_names: list[str],
    input_mean: MeanStd | dict[str, MeanStd] | None,
    input_std: MeanStd | dict[str, MeanStd] | None,
    input_transform: InputTransform | dict[str, InputTransform] | None,
    render_cache: _RenderCache,
) -> None:
    # The primary input drives the token grid (its H×W); the user can view any
    # input in the pane via the dropdown, with its own resolved display config.
    input_name = input_names[0] if input_names else None
    state = _PageState()

    def decoupled() -> bool:
        """Whether card visibility is per-tab (stats scopes `none` / `all`).

        In the `watched` scope, visible ≡ watched — the global, cross-tab set
        whose members collect stats. In the other scopes the watched set does
        not drive collection, so clicks only touch this connection's `shown`
        set and tabs are independent.
        """
        return session.stats_scope is not StatsScope.WATCHED

    def shown_layers() -> frozenset[str]:
        """The layers whose cards this connection currently shows."""
        return frozenset(state.shown) if decoupled() else session.watched_layers

    state.last_scope = session.stats_scope
    if decoupled():
        # Only the locked playground emits `?layer=` links (the subpages'
        # Back buttons); elsewhere the param is ignored so a stray deep link
        # can't change what an unlocked session shows.
        state.shown = _seed_shown(
            session.watched_layers,
            focus_layer if session.locked else "",
            layer_names,
        )
    state.last_watched = shown_layers()
    layer_views: dict[str, _LayerView] = {}
    # One collision-free slug per layer, shared with the Mermaid diagram
    # (`graph.build_mermaid` keys node ids by the same `slug_map`). Using it
    # for the cards' `data-layer`, the click→toggle lookup, and the
    # JS watch/scroll calls keeps the diagram and the cards in lockstep even
    # when two distinct layer names would alias to the same bare slug
    # (e.g. `fc.1` and `fc_1`).
    slugs = slug_map(layer_names)

    def record_view() -> RecordedView | None:
        # `input_panel` is created further down; the dialog only calls this
        # after the page is fully built.
        if session.snapshot is None and session.probe_result is None:
            return None
        shown = shown_layers()
        watched = [n for n in layer_names if n in shown]
        plural = "" if len(watched) == 1 else "s"
        # Record exactly the input the pane is showing, with its resolved
        # display config (a same-process dict, so the transform travels too).
        selected = input_panel.selected_input
        return RecordedView(
            key="main",
            page="main",
            label=(
                f"Main view ({len(watched)} watched layer{plural}, "
                f"sample {input_panel.sample_idx})"
            ),
            params={
                "layers": tuple(watched),
                "sample_idx": input_panel.sample_idx,
                "input_name": selected or "",
                "input_mean": resolve_per_input(input_mean, selected),
                "input_std": resolve_per_input(input_std, selected),
                "input_transform": resolve_per_input(input_transform, selected),
            },
        )

    _page_scaffold()
    _install_panel_resize()
    ui.add_head_html(_ARCHITECTURE_CLICK_CSS)
    ui.add_head_html(_STRIP_MARKER_CSS)
    ui.add_body_html(_ARCHITECTURE_CLICK_JS)
    ui.add_body_html(_layer_info_script(session.layer_info, slugs))

    # The tour points at (and auto-shows, for its strips step) one layer;
    # preferring one that owns weights makes all three card buttons real
    # targets for the tour's three-arrow step. Auto-starts only on locked
    # (playground) sessions — local runs reach it via the `?` button.
    layer_weights = session.layer_weights
    tour_layer = next(
        (n for n in layer_names if layer_weights.get(n)),
        layer_names[0] if layer_names else None,
    )
    tour_slug = slugs[tour_layer] if tour_layer is not None else None
    add_tour(
        "main",
        main_tour_steps(tour_slug, locked=session.locked),
        locked=session.locked,
        auto_watch_slug=tour_slug,
    )

    step_until_custom = _build_step_until_custom_dialog(session)

    def watch_all() -> None:
        if decoupled():
            state.shown = set(layer_names)
        else:
            for name in layer_names:
                session.watch(name)
        sync_watch_ui()

    def clear_all() -> None:
        if decoupled():
            state.shown.clear()
        else:
            if _refuse_unwatch_while_recording(session):
                return
            for name in list(session.watched_layers):
                session.unwatch(name)
        sync_watch_ui()

    # Showing everything turns the lazy-rendering optimization off again:
    # every card renders on every pause (and in the `watched` scope, stats
    # accumulate for every layer on every batch). Worth an explicit
    # confirmation.
    watch_all_dialog = ui.dialog()
    with watch_all_dialog, ui.card().classes("max-w-md"):
        ui.label("Show all layers?").classes("text-lg font-medium")
        ui.label(
            "Every layer card will be rendered on every pause — and, while "
            "stats are collected for watched layers, per-layer statistics "
            "will accumulate on every batch. On larger models this can make "
            "the interface very slow and may even crash the browser tab."
        ).classes("text-sm text-slate-600")
        with ui.row().classes("w-full justify-end gap-2"):
            ui.button("Cancel", on_click=watch_all_dialog.close).props("flat")
            ui.button(
                "Show all",
                color="red",
                on_click=lambda: (watch_all(), watch_all_dialog.close()),
            )

    with ui.column().classes("w-full h-screen no-wrap gap-0"):
        with _top_bar_row():
            architecture_toggle = ui.button(
                icon="account_tree", color="slate-500"
            ).props("dense size=md").tooltip("Toggle architecture pane")
            _refresh_button(session)
            _add_step_controls(session, step_until_custom)
            watch_chip = ui.button(
                color="slate-100",
            ).classes(
                "ml-auto text-amber-700 font-mono"
            ).props("dense size=md no-caps").tooltip(
                "Shown layers — click a layer to open its stats view; "
                "use the menu to pause stats collection"
            )
            stats_icon: ui.icon
            watch_count_label: ui.label
            watch_list_container: ui.element
            with watch_chip:
                # Icon and count are built as button children (rather than the
                # button's `icon=` / text) so the eye icon carries its own colour
                # independently of the amber count. `sync_stats_icon` swaps it
                # between `visibility` (green, collecting) and `visibility_off`
                # (red, paused) — the slashed eye of the per-card Unwatch button.
                stats_icon = ui.icon("visibility").classes("text-base")
                watch_count_label = ui.label(
                    str(len(state.last_watched))
                ).classes("ml-1")
                with ui.menu().props("anchor='bottom right' self='top right'"):
                    # Plain block container, NOT a flex column: Firefox fails to
                    # position/size a QMenu whose content root is a flex column,
                    # so the menu opens collapsed (height 0) and looks like it
                    # never opened. See quasarframework/quasar#16167. Block-level
                    # children stack vertically anyway.
                    with ui.element("div").classes("min-w-64"):
                        ui.menu_item(
                            "Show all layers",
                            on_click=watch_all_dialog.open,
                        ).classes("text-sm").tooltip(
                            "Show every layer's card (while stats are "
                            "collected for watched layers, this also watches "
                            "them all)"
                        )
                        ui.menu_item(
                            "Hide all layers",
                            on_click=lambda: clear_all(),
                        ).classes("text-sm").tooltip(
                            "Hide every card (while stats are collected for "
                            "watched layers, this also unwatches them and "
                            "drops their collected stats)"
                        )
                        # A locked session pins the stats scope, so the
                        # pause toggle would be a silent no-op — hide it.
                        if not session.locked:
                            ui.menu_item(
                                "Toggle collecting stats",
                                on_click=lambda: toggle_stats(),
                                auto_close=False,
                            ).classes("text-sm").tooltip(
                                "Pause/resume the running stats — pausing "
                                "keeps everything collected so far, resuming "
                                "restores the previous collection scope (see "
                                "the settings gear). Cards stay visible "
                                "either way."
                            )
                        ui.separator()
                        # "Current batch" submenu: every layer (watched or not),
                        # each routing straight to that layer's current-batch
                        # stats view. The nested menu's content root is a block
                        # div, so the Firefox QMenu caveat above doesn't apply.
                        with ui.menu_item(
                            "Current batch", auto_close=False
                        ).classes("text-sm"):
                            with ui.item_section().props("side"):
                                ui.icon("chevron_right")
                            with ui.menu().props(
                                "anchor='top end' self='top start'"
                            ):
                                with ui.element("div").classes(
                                    "min-w-56 max-h-96 overflow-auto"
                                ):
                                    for name in layer_names:
                                        ui.menu_item(name).props(
                                            f'href="/stats?layer={quote(name)}"'
                                        ).classes("font-mono text-sm")
                        ui.separator()
                        watch_list_container = ui.element("div").classes("py-1")
            _add_settings_button(session, record_view)
            input_toggle = ui.button(
                icon="image", color="slate-500"
            ).props("dense size=md").tooltip(
                "Toggle input selection pane"
            )
            _add_tour_button()
            _add_share_button()
            _add_repo_logo()

        _add_error_banner(session)

        # The collection state last drawn into the icon, so the 200 ms tick only
        # touches the DOM when it actually flips (and never re-adds a tooltip —
        # an earlier per-tick `.tooltip()` here stacked dozens of them).
        stats_shown: bool | None = None

        def sync_stats_icon() -> None:
            """Reflect the collection state in the top-bar eye icon.

            `visibility` in green while collecting, the slashed `visibility_off`
            in red while paused (the per-card Unwatch button's glyph). Called on
            init, on toggle, and on the 200 ms tick so a toggle in one tab shows
            in every other — but it rewrites the icon only when the state flips.
            """
            nonlocal stats_shown
            collecting = session.stats_collecting
            if collecting == stats_shown:
                return
            stats_shown = collecting
            stats_icon.set_name("visibility" if collecting else "visibility_off")
            stats_icon.classes(
                remove="text-green-600 text-red-600",
                add="text-green-600" if collecting else "text-red-600",
            )

        def toggle_stats() -> None:
            session.toggle_stats_collecting()
            sync_stats_icon()

        def refresh_chip() -> None:
            shown = shown_layers()
            watch_count_label.text = str(len(shown))
            watch_list_container.clear()
            with watch_list_container:
                if not shown:
                    ui.label("No layers shown").classes(
                        "px-3 py-2 text-slate-500 text-sm italic"
                    )
                    return
                # Section header: each entry below opens the stats view
                # focused on that layer.
                ui.label("Open stats view").classes(
                    "px-3 pt-1 pb-0.5 text-xs uppercase tracking-wider "
                    "text-slate-400 select-none"
                )
                for layer in layer_names:
                    if layer not in shown:
                        continue
                    # A real anchor (href) instead of a JS navigate: the
                    # browser natively opens middle/ctrl clicks in a new tab
                    # and plain clicks in the current one.
                    ui.menu_item(layer).props(
                        f'href="/stats?layer={quote(layer)}"'
                    ).classes("font-mono text-sm")

        def sync_watch_ui() -> None:
            """Reflect the shown set in this connection's DOM.

            In the `watched` scope, visible is synonymous with watched (the
            global set); in the decoupled scopes it is this tab's own `shown`
            set. Either way: cards for newly shown layers appear (and get
            rendered on the next tick via the dirty flag), hidden ones
            disappear, the diagram's amber classes follow, and the chip menu /
            empty-pane hint refresh. Diffing against `state.last_watched`
            keeps the JS push proportional to the change, not the model size.
            """
            shown = shown_layers()
            added = shown - state.last_watched
            removed = state.last_watched - shown
            state.last_watched = shown
            for name in added | removed:
                view = layer_views.get(name)
                if view is not None:
                    view.set_visible(name in shown)
            if added or removed:
                changes = "; ".join(
                    f"window.nansenseSetWatched({json.dumps(slugs[n])}, "
                    f"{'true' if n in shown else 'false'})"
                    for n in added | removed
                )
                ui.run_javascript(changes)
                state.dirty = True
            empty_hint.set_visibility(not shown)
            refresh_chip()

        def toggle_layer(name: str) -> None:
            if decoupled():
                # Per-tab visibility only — the session is never touched, so
                # other tabs (and the collected stats) are unaffected.
                if name in state.shown:
                    state.shown.discard(name)
                elif name in layer_names:
                    state.shown.add(name)
                else:
                    return
            # Any name in `session.layer_names` is watchable (modules, fx
            # intermediates, graph inputs); False means an unknown name.
            elif name in session.watched_layers:
                if _refuse_unwatch_while_recording(session):
                    return
                session.unwatch(name)
            elif not session.watch(name):
                return
            sync_watch_ui()
            if name in shown_layers():
                ui.run_javascript(
                    f"window.nansenseScrollToCard({json.dumps(slugs[name])})"
                )

        with ui.row().classes("w-full no-wrap gap-0 grow min-h-0"):
            architecture_pane = ui.column().classes(
                "w-1/4 shrink-0 h-full overflow-auto p-2 "
                "border-r-2 border-slate-300 bg-slate-50"
            ).props(_resizable_pane_props("main-architecture"))
            architecture_handle = _resize_handle("main-architecture", "left")
            with architecture_pane:
                ui.mermaid(mermaid_src).classes("w-full")
            with ui.column().classes(
                "grow min-w-0 h-full overflow-auto p-3 bg-slate-200 gap-3"
            ):
                empty_hint = _notice_banner(
                    "No layers shown — click a node in the architecture "
                    "diagram to show a layer's activations and gradients "
                    "and start collecting stats.",
                    icon="touch_app",
                )
                empty_hint.set_visibility(not state.last_watched)
                # Every card is built once (cheap: header + empty strips) but
                # only watched ones are visible — and only visible cards get
                # strip data, so hidden layers cost neither render time nor
                # websocket bytes.
                for name in layer_names:
                    layer_views[name] = _LayerView(
                        name,
                        slug=slugs[name],
                        visible=name in state.last_watched,
                        decoupled=decoupled(),
                        weights=layer_weights.get(name, []),
                        on_toggle_watch=toggle_layer,
                    )
            input_handle = _resize_handle("main-input", "right")
            input_pane = ui.column().classes(
                f"{_INPUT_PANE_WIDTH} shrink-0 h-full overflow-auto p-3 "
                "border-l-2 border-slate-300 bg-slate-50 items-center"
            ).props(_resizable_pane_props("main-input"))
            with input_pane:

                def mark_dirty() -> None:
                    state.dirty = True

                input_panel = InputPanel(
                    session=session,
                    input_names=input_names,
                    input_mean=input_mean,
                    input_std=input_std,
                    input_transform=input_transform,
                    on_change=mark_dirty,
                )

        def toggle_architecture() -> None:
            visible = not architecture_pane.visible
            architecture_pane.set_visibility(visible)
            architecture_handle.set_visibility(visible)

        def toggle_input() -> None:
            visible = not input_pane.visible
            input_pane.set_visibility(visible)
            input_handle.set_visibility(visible)

        architecture_toggle.on_click(toggle_architecture)
        input_toggle.on_click(toggle_input)

    # Diagram clicks arrive as custom events carrying the node's slug; map
    # it back to the layer name and toggle. Unknown slugs (e.g. a node
    # whose label isn't a captured layer) are ignored. Inverting `slugs`
    # (rather than rebuilding via the bare `slug`) keeps this in step with
    # the diagram's disambiguated node ids.
    slug_to_name = {s: n for n, s in slugs.items()}

    def on_diagram_toggle(e: GenericEventArguments) -> None:
        name = slug_to_name.get(e.args)
        if name is not None:
            toggle_layer(name)

    ui.on("nansense_toggle_layer", on_diagram_toggle)

    # Populate the chip menu and, if anything is already watched, push the
    # set into JS so the MutationObserver applies the amber treatment to
    # mermaid nodes once Mermaid finishes rendering them client-side.
    refresh_chip()
    sync_stats_icon()
    initial_watched = list(state.last_watched)
    if initial_watched:
        slugs_js = json.dumps([slugs[n] for n in initial_watched])
        ui.timer(
            0.0,
            lambda: ui.run_javascript(
                f"({slugs_js}).forEach(s => window.nansenseSetWatched(s, true))"
            ),
            once=True,
        )
    # A locked `?layer=` deep link (a subpage's Back button) lands on the
    # card it names, not the top of the list.
    if session.locked and focus_layer in layer_names:
        scroll_js = (
            f"window.nansenseScrollToCard({json.dumps(slugs[focus_layer])})"
        )
        ui.timer(0.0, lambda: ui.run_javascript(scroll_js), once=True)

    async def tick() -> None:
        input_panel.refresh_status()
        # While the main view records, its render parameters (sample, pin,
        # perturbations, probe mode) are frozen: the recording renders with
        # the live probe state, so the input controls must not change it.
        input_panel.set_frozen(session.recording.is_recording("main"))
        # A stats-scope switch (from the settings gear, possibly in another
        # tab) re-bases the shown set: entering a decoupled scope seeds this
        # tab's own set from the global watched set, returning to `watched`
        # re-syncs to it. Switching between the decoupled scopes (the stats
        # pause toggle) keeps the tab's cards as they are.
        scope = session.stats_scope
        if scope is not state.last_scope:
            if scope is not StatsScope.WATCHED and (
                state.last_scope is StatsScope.WATCHED
            ):
                state.shown = set(session.watched_layers)
            state.last_scope = scope
            for view in layer_views.values():
                view.set_decoupled(scope is not StatsScope.WATCHED)
            sync_watch_ui()
        # Shown-set changes made elsewhere (another tab or the stats page, in
        # the coupled scope) propagate here: sync flips card visibility and
        # marks the frame dirty so newly visible cards render from the
        # current snapshot.
        elif shown_layers() != state.last_watched:
            sync_watch_ui()
        # Stats-collection state can change from another tab too; keep the
        # icon's colour/strike in sync (cheap class writes, no-op when stable).
        sync_stats_icon()
        snap = session.snapshot
        # With a probe result present (a batch is pinned, or an eval/train
        # forward mode is selected), the page renders the probe instead of the
        # snapshot: pinning tracks one fixed input across stepping and time
        # travel, while eval/train shows the current batch under that mode.
        probe = session.probe_result
        if snap is None and probe is None:
            return
        input_panel.sync_spinner_max(_display_batch_size(snap, probe))
        if state.rendering:
            return
        if (
            snap is not state.last_snapshot
            or probe is not state.last_probe
            or state.dirty
        ):
            # Mark this source/dirty as consumed up front so a clean render
            # doesn't re-fire. Every layer's render is isolated
            # (`_render_layers`), so a single bad layer can't reach here; this
            # guard covers the residual whole-frame failure modes (e.g. the
            # input image). On such a failure the frame is dropped but the
            # loop is never wedged: `rendering` is reset in `finally`, the bad
            # source stays marked as seen (so the timer doesn't busy-crash on
            # it every 200 ms), and the next published snapshot/probe — a new
            # object — renders cleanly, so the page recovers on its own.
            state.last_snapshot = snap
            state.last_probe = probe
            state.dirty = False
            state.rendering = True
            try:
                sample_idx = input_panel.sample_idx
                # Resolve the display config for whichever input the pane shows
                # (a per-input dict collapses to this one input's values).
                selected = input_panel.selected_input
                sel_mean = resolve_per_input(input_mean, selected)
                sel_std = resolve_per_input(input_std, selected)
                sel_transform = resolve_per_input(input_transform, selected)
                # Only the visible (= watched) layers render; hidden cards
                # keep whatever stale content they had, which is invisible
                # and re-rendered (cache-assisted) when they reappear.
                visible_names = [n for n in layer_names if n in state.last_watched]
                rendered, input_src = await asyncio.to_thread(
                    _compute_frame,
                    visible_names,
                    snap,
                    probe,
                    sample_idx,
                    compare=input_panel.compare,
                    input_name=input_name,
                    selected_input=selected,
                    input_mean=sel_mean,
                    input_std=sel_std,
                    input_transform=sel_transform,
                    cache=render_cache,
                )
            finally:
                state.rendering = False
            _apply_all(layer_views, rendered)
            input_panel.set_image(input_src)
            selected_tensor = _selected_input_tensor(snap, probe, selected)
            input_panel.set_input_warning(
                input_blank_warning(
                    selected_tensor,
                    sample_idx,
                    name=selected,
                    mean=sel_mean,
                    std=sel_std,
                    transform=sel_transform,
                )
            )
            input_panel.set_input_legend(
                _input_img_src(render_input_legend(selected_tensor, sample_idx))
            )

    ui.timer(0.2, tick)


def _snapshot_batch_size(snap: BatchSnapshot) -> int | None:
    for tensor in snap.activations.values():
        if tensor.ndim > 0:
            return int(tensor.shape[0])
    return None


def _zeros_like(tensor: Tensor | None) -> Tensor | None:
    return torch.zeros_like(tensor) if tensor is not None else None


def _selected_input_tensor(
    snap: BatchSnapshot | None, probe: ProbeResult | None, name: str | None
) -> Tensor | None:
    """The selected input's tensor as currently shown (probe wins over snap)."""
    if name is None:
        return None
    if probe is not None:
        return probe.shown_input(name)
    if snap is not None:
        return snap.activations.get(name)
    return None


def _display_batch_size(
    snap: BatchSnapshot | None, probe: ProbeResult | None
) -> int | None:
    """Batch size of whatever the page is currently rendering."""
    if probe is not None:
        return probe.batch_size()
    if snap is not None:
        return _snapshot_batch_size(snap)
    return None


# Shown in place of the GRADIENTS strip while a probe result is displayed:
# probes are forward-only, so there are no activation gradients to render.
_PROBE_NO_GRADIENTS_HTML: str = (
    '<div class="text-xs text-slate-400 italic py-1">'
    "no gradients on probe runs</div>"
)

# Strip pair shown for a layer whose render fell over. `render_strip` already
# returns `None` (hidden strip) for empty/unsupported tensors; this covers the
# residual cases (a genuinely unexpected render bug) so one bad layer degrades
# to a blank card instead of taking the whole frame down with it.
_EMPTY_STRIPS: tuple[str, str] = ("", "")


def _render_layers(
    layer_names: list[str], strips: Callable[[str], tuple[str, str]]
) -> dict[str, tuple[str, str]]:
    """Render every layer's strip pair concurrently, isolating failures.

    Layers fan out over `_RENDER_POOL`; a single layer that raises must not
    abort its siblings or drop the frame, so each render is guarded and a
    failed layer yields blank strips (`_EMPTY_STRIPS`) — the same empty
    result a layer absent from the snapshot gets.
    """

    def guarded(name: str) -> tuple[str, str]:
        try:
            return strips(name)
        except Exception:
            return _EMPTY_STRIPS

    return dict(zip(layer_names, _RENDER_POOL.map(guarded, layer_names), strict=True))


def _compute_frame(
    layer_names: list[str],
    snap: BatchSnapshot | None,
    probe: ProbeResult | None,
    sample_idx: int,
    *,
    compare: bool = False,
    input_name: str | None,
    selected_input: str | None = None,
    input_mean: tuple[float, ...] | None,
    input_std: tuple[float, ...] | None,
    input_transform: InputTransform | None = None,
    cache: _RenderCache,
) -> tuple[dict[str, tuple[str, str]], str]:
    """Render every layer's strip pair plus the input image source.

    With a probe result present it is the render source (pinned-batch /
    perturbed view, see `_compute_probe_frame`); otherwise the snapshot is.
    Layers render concurrently on `_RENDER_POOL`; each strip goes through
    `cache`, so only strips not already rendered for this source cost
    anything. `input_name` is the primary image input (its `H × W` sets the
    token grid for 2D activations); `selected_input` is the input shown in the
    pane — it defaults to `input_name` (the same one unless the user picked
    another from the multi-input dropdown).
    """
    if selected_input is None:
        selected_input = input_name
    if probe is not None:
        return _compute_probe_frame(
            layer_names,
            probe,
            sample_idx,
            compare=compare,
            input_name=input_name,
            selected_input=selected_input,
            input_mean=input_mean,
            input_std=input_std,
            input_transform=input_transform,
            cache=cache,
        )
    assert snap is not None  # tick only renders when at least one source exists
    input_hw = tensor_hw(snap.activations.get(input_name) if input_name else None)

    def strips(name: str) -> tuple[str, str]:
        if compare:
            # Diff view without any probe (perturb mode on, nothing clicked
            # yet, no pin): the diff is identically zero, rendered as a
            # white strip — same as a perturbation-free probe diff.
            act = cache.get_or_render(
                snap,
                (name, "act-diff", sample_idx),
                lambda: _strip_html(
                    render_strip(
                        _zeros_like(snap.activations.get(name)),
                        sample_idx,
                        input_hw=input_hw,
                    ),
                    show_labels=True,
                ),
            )
        else:
            act = cache.get_or_render(
                snap,
                (name, "act", sample_idx),
                lambda: _strip_html(
                    render_strip(
                        snap.activations.get(name), sample_idx, input_hw=input_hw
                    ),
                    show_labels=True,
                ),
            )
        grad = cache.get_or_render(
            snap,
            (name, "grad", sample_idx),
            lambda: _strip_html(
                render_strip(
                    snap.activation_gradients.get(name),
                    sample_idx,
                    input_hw=input_hw,
                )
            ),
        )
        return act, grad

    rendered = _render_layers(layer_names, strips)
    input_src = cache.get_or_render(
        snap,
        (selected_input or "", "input", sample_idx),
        lambda: _input_img_src(
            render_image(
                snap.activations.get(selected_input) if selected_input else None,
                sample_idx,
                mean=input_mean,
                std=input_std,
                transform=input_transform,
            )
        ),
    )
    return rendered, input_src


def _compute_probe_frame(
    layer_names: list[str],
    probe: ProbeResult,
    sample_idx: int,
    *,
    compare: bool,
    input_name: str | None,
    selected_input: str | None,
    input_mean: tuple[float, ...] | None,
    input_std: tuple[float, ...] | None,
    input_transform: InputTransform | None,
    cache: _RenderCache,
) -> tuple[dict[str, tuple[str, str]], str]:
    """The probe-sourced equivalent of the snapshot frame.

    Without perturbations the strips show the base activations. With
    perturbations they show the perturbed forward's activations, or — with
    `compare` on — the per-layer diff `perturbed − original`, whose nonzero
    extent traces how far the edit propagates (receptive field). The diff
    view renders even with nothing perturbed: an all-zero diff draws as a
    white strip, signalling "no differences" rather than falling back to a
    non-diff view. The input pane shows the perturbed copy of the selected
    input whenever one exists, so the edit is visible. Probe runs are
    forward-only, so every gradient strip shows a placeholder note instead of
    an image. `input_name` (the primary image input) sets the token grid;
    `selected_input` is the input shown in the pane.
    """
    kind = "probe-diff" if compare else (
        "probe-perturbed" if probe.perturbed_activations is not None else "probe-act"
    )
    input_hw = tensor_hw(probe.base_input(input_name))

    def strips(name: str) -> tuple[str, str]:
        act = cache.get_or_render(
            probe,
            (name, kind, sample_idx),
            lambda: _strip_html(
                render_strip(
                    probe_act_tensor(probe, name, compare=compare),
                    sample_idx,
                    input_hw=input_hw,
                ),
                show_labels=True,
            ),
        )
        return act, _PROBE_NO_GRADIENTS_HTML

    rendered = _render_layers(layer_names, strips)
    shown_input = probe.shown_input(selected_input)
    input_src = cache.get_or_render(
        probe,
        (selected_input or "", "probe-input", sample_idx),
        lambda: _input_img_src(
            render_image(
                shown_input,
                sample_idx,
                mean=input_mean,
                std=input_std,
                transform=input_transform,
            )
        ),
    )
    return rendered, input_src


def _apply_all(
    views: dict[str, _LayerView],
    rendered: dict[str, tuple[str, str]],
) -> None:
    for name, (act_html, grad_html) in rendered.items():
        views[name].apply(act_html, grad_html)


class _LayerView:
    """One card per layer, with activation + activation-gradient strips.

    Cards are built for every layer but shown only while the layer is in the
    page's shown set (`set_visible`) — the watched set in the coupled
    `watched` stats scope, the tab's own set otherwise — so the header
    carries a permanent hide button ("Unwatch" while coupled, since hiding
    then also drops the layer's stats; "Hide" while decoupled) and hidden
    cards receive no strip data at all.

    The strips are raw `<img>` elements (see `_strip_html`) with fixed CSS
    sizes and `flex:none`, so each strip renders at its display pixel width
    and the wrapping `overflow-x-auto` div produces a shared horizontal
    scrollbar inside the card. NiceGUI's `ui.image` uses Quasar's responsive
    q-img instead, which squishes the strip to the card width — not what we
    want here. The card has `min-w-0` so a wide strip doesn't push the
    column wider.

    Both strips use the same diverging colormap, so each one carries a
    labelled colored marker bar on its left edge to tell them apart
    (emerald ACTIVATIONS, violet GRADIENTS). The markers are `sticky
    left-0` so they stay visible while the strips are panned horizontally.
    """

    def __init__(
        self,
        name: str,
        *,
        slug: str,
        visible: bool,
        decoupled: bool,
        weights: list[str],
        on_toggle_watch: Callable[[str], None],
    ) -> None:
        self.name = name
        card = ui.element("div").classes(
            "w-full min-w-0 bg-white rounded border border-slate-300 shadow-sm "
            "hover:border-blue-400 transition-colors"
        )
        card.props(f'data-layer="{slug}"')
        with card:
            with ui.row().classes(
                "items-center w-full no-wrap gap-2 pl-3 pr-1 py-1 bg-slate-100 "
                "border-b border-slate-300 rounded-t"
            ):
                ui.label(name).classes(
                    "font-mono text-sm grow min-w-0 truncate"
                )
                # Each wrapper carries `data-card-action` so the
                # document-level click handler skips card→diagram navigation
                # when a header button is clicked. Quasar's q-btn doesn't
                # reliably pass arbitrary `data-*` attrs through to its
                # rendered DOM, so the attribute lives on these divs.
                # The Weights button only appears for layers that actually own
                # parameters; relu/add/input nodes have nothing to show.
                # href (not on_click navigation) renders the buttons as real
                # anchors, so the browser natively opens middle/ctrl clicks
                # in a new tab and plain clicks in the current one.
                # `data-tour` marks the wrappers as the tour's arrow targets
                # (`tour.py`) — on the divs for the same reason as
                # `data-card-action` above.
                if weights:
                    with ui.element("div").props(
                        'data-card-action data-tour="weights"'
                    ):
                        ui.button(
                            "Weights",
                            icon="grid_on",
                            color="blue",
                        ).props(
                            f'dense no-caps href="/weights?layer={quote(name)}"'
                        ).style(
                            "min-height: 0; padding: 1px 6px; font-size: 11px"
                        ).tooltip(
                            f"Inspect this layer's weights ({len(weights)})"
                        )
                with ui.element("div").props(
                    'data-card-action data-tour="experiment"'
                ):
                    ui.button(
                        "Experiment",
                        icon="science",
                        color="yellow-8",
                    ).props(
                        f'dense no-caps href="/experiment?layer={quote(name)}"'
                    ).style(
                        "min-height: 0; padding: 1px 6px; font-size: 11px"
                    ).tooltip(
                        "Run deep dream / Captum experiments on this layer"
                    )
                with ui.element("div").props(
                    'data-card-action data-tour="stats"'
                ):
                    ui.button(
                        "Stats",
                        icon="bar_chart",
                        color="teal",
                    ).props(
                        f'dense no-caps href="/stats?layer={quote(name)}"'
                    ).style(
                        "min-height: 0; padding: 1px 6px; font-size: 11px"
                    ).tooltip(
                        "Open this layer's stats view (histograms & min/max)"
                    )
                # The card only shows while the layer is shown, so the button
                # is always the "off" direction; its label reflects what
                # hiding does (see the class docstring) via `set_decoupled`.
                with ui.element("div").props("data-card-action"):
                    self._hide_button = ui.button(
                        icon="visibility_off",
                        on_click=lambda: on_toggle_watch(name),
                        color="red",
                    ).props("dense no-caps").style(
                        "min-height: 0; padding: 1px 6px; font-size: 11px"
                    )
                    with self._hide_button:
                        self._hide_tooltip = ui.tooltip("")
            with ui.element("div").classes("w-full overflow-x-auto p-2").props(
                'data-tour="strips"'
            ):
                # The max-content wrapper makes every row span the widest
                # strip. Without it a row is only as wide as the visible
                # container, so its sticky marker would be dragged out of
                # view once the strips are scrolled past that width.
                with ui.element("div").classes("w-max min-w-full"):
                    with ui.element("div").classes("flex no-wrap items-stretch"):
                        _strip_marker(
                            "bg-emerald-500", "ACTIVATIONS", header_gap=True
                        )
                        self.act_html = ui.html("")
                    ui.element("div").classes("h-1")
                    with ui.element("div").classes("flex no-wrap items-stretch"):
                        _strip_marker("bg-violet-500", "GRADIENTS")
                        self.grad_html = ui.html("")
        self._card = card
        self.set_decoupled(decoupled)
        # A page (re)built with layers already in the shown set (e.g.
        # after navigating back from `/stats`) shows those cards right away.
        self.set_visible(visible)

    def set_visible(self, visible: bool) -> None:
        self._card.set_visibility(visible)

    def set_decoupled(self, decoupled: bool) -> None:
        """Relabel the hide button for the current stats scope.

        Coupled (`watched` scope): hiding unwatches, which also drops the
        layer's collected stats — the label says so. Decoupled: hiding is
        pure per-tab visibility.
        """
        if decoupled:
            self._hide_button.set_text("Hide")
            self._hide_tooltip.set_text(
                "Hide this layer's card in this tab (stats are unaffected)"
            )
        else:
            self._hide_button.set_text("Unwatch")
            self._hide_tooltip.set_text(
                "Unwatch this layer and hide its card (drops its collected "
                "stats)"
            )

    def apply(self, act_html: str, grad_html: str) -> None:
        self.act_html.set_content(act_html)
        self.grad_html.set_content(grad_html)


def _input_img_src(image: bytes | None) -> str:
    """Data-URI source for the input pane's interactive image ("" when absent).

    The sizing (CSS upscale to `INPUT_IMAGE_SIZE` with nearest-neighbour
    rendering) lives on the `ui.interactive_image` element in `InputPanel`;
    only the native-resolution source travels per frame.
    """
    if image is None:
        return ""
    return _b64_img_src(image)
