"""The main page: architecture diagram, watched-layer cards, input pane."""

from __future__ import annotations

import asyncio
import json
import os
import threading
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import quote

import torch
from nicegui import ui
from nicegui.events import GenericEventArguments
from torch import Tensor

from nansense.contracts.recording import MainView
from nansense.input_config import InputTransform, MeanStd, resolve_per_input
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
from nansense.ui.components.layer_menu import LayerMenu
from nansense.ui.controllers.main import MainController
from nansense.ui.graph import slug_map
from nansense.ui.input_panel import InputPanel
from nansense.ui.render import (
    DEFAULT_RENDER_OPTIONS,
    RenderOptions,
    input_blank_warning,
    probe_act_tensor,
    render_image,
    render_input_legend,
    render_strip,
    tensor_hw,
)
from nansense.ui.share import _add_share_button
from nansense.ui.static import (
    _ARCHITECTURE_CLICK_CSS,
    _ARCHITECTURE_CLICK_JS,
    _STRIP_MARKER_CSS,
)
from nansense.ui.theme import ACTIVATIONS, CUSTOM_TENSOR, GRADIENTS
from nansense.ui.top_bar import (
    _add_error_banner,
    _add_repo_logo,
    _add_settings_button,
    _add_step_controls,
    _add_tour_button,
    _build_step_until_custom_dialog,
    _refresh_button,
    _top_bar_row,
)
from nansense.ui.tour import add_tour, main_tour_steps

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


def _pick_tour_layer(
    layer_names: list[str],
    shown: frozenset[str],
    layer_weights: dict[str, list[str]],
) -> str | None:
    """The layer the main tour points at (and auto-shows for its card steps).

    It must be a layer whose card the page already opened — the tour's first
    arrow lands on a diagram node, and pointing it at a layer other than the
    one card on screen reads as two unrelated subjects (the playground's
    seed is a single deliberate card, so the arrow has to name *that* one).
    Among several shown layers a weights-owning one wins: it makes all three
    of the card's buttons real targets for the buttons step. With nothing
    shown — an unwatched local run, where the tour's card steps open the
    card themselves — the same preference picks from the whole model.
    """
    candidates = [n for n in layer_names if n in shown] or layer_names
    if not candidates:
        return None
    return next((n for n in candidates if layer_weights.get(n)), candidates[0])


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
    _MainPage(
        session,
        mermaid_src,
        layer_names,
        focus_layer=focus_layer,
        input_names=input_names,
        input_mean=input_mean,
        input_std=input_std,
        input_transform=input_transform,
        render_cache=render_cache,
    )


class _MainPage:
    """Bind the main controller to widgets and per-client rendering."""

    def __init__(
        self,
        session: Session,
        mermaid_src: str,
        layer_names: list[str],
        *,
        focus_layer: str,
        input_names: list[str],
        input_mean: MeanStd | dict[str, MeanStd] | None,
        input_std: MeanStd | dict[str, MeanStd] | None,
        input_transform: InputTransform | dict[str, InputTransform] | None,
        render_cache: _RenderCache,
    ) -> None:
        self.session = session
        self.mermaid_src = mermaid_src
        self.layer_names = layer_names
        self.focus_layer = focus_layer
        self.input_names = input_names
        self.input_mean = input_mean
        self.input_std = input_std
        self.input_transform = input_transform
        self.render_cache = render_cache
        self.input_name = input_names[0] if input_names else None
        self.controller = MainController(session, layer_names, focus_layer)
        self.state = self.controller.state
        self.state.last_watched = self.shown_layers()
        self.layer_views: dict[str, _LayerView] = {}
        self.slugs = slug_map(layer_names)
        self.slug_to_name = {slug: name for name, slug in self.slugs.items()}
        self.layer_weights = session.layer_weights
        self._install_page()
        self._build_dialogs()
        with ui.column().classes("w-full h-screen no-wrap gap-0"):
            self._build_header()
            _add_error_banner(session)
            self._build_body()
        self._connect_events()
        ui.timer(0.2, self.tick)

    def _install_page(self) -> None:
        _page_scaffold()
        _install_panel_resize()
        ui.add_head_html(_ARCHITECTURE_CLICK_CSS)
        ui.add_head_html(_STRIP_MARKER_CSS)
        ui.add_body_html(_ARCHITECTURE_CLICK_JS)
        ui.add_body_html(_layer_info_script(self.session.layer_info, self.slugs))

        # The tour opens on one layer: the one this tab is already showing, so
        # its first arrow lands on the card the visitor is looking at rather than
        # opening a second, unrelated one. It stays the card steps' preference
        # and their fallback — they auto-show it only when no card is open at all
        # — but a visitor who opens another layer keeps it (`tour.py`).
        # Auto-starts only on locked (playground) sessions — local runs reach it
        # via the `?` button.
        tour_layer = _pick_tour_layer(
            self.layer_names, self.shown_layers(), self.layer_weights
        )
        tour_slug = self.slugs[tour_layer] if tour_layer is not None else None
        add_tour(
            "main",
            main_tour_steps(tour_slug, locked=self.session.locked),
            locked=self.session.locked,
            auto_watch_slug=tour_slug,
        )

    def _build_dialogs(self) -> None:
        self.step_until_custom = _build_step_until_custom_dialog(self.session)

        # Showing everything turns the lazy-rendering optimization off again:
        # every card renders on every pause (and in the `watched` scope, stats
        # accumulate for every layer on every batch). Worth an explicit
        # confirmation.
        self.watch_all_dialog = ui.dialog()
        with self.watch_all_dialog, ui.card().classes("max-w-md"):
            ui.label("Show every layer?").classes("text-lg font-medium")
            ui.label(
                "This can slow down large models and use a lot of browser memory."
            ).classes("text-sm text-slate-600")
            with ui.row().classes("w-full justify-end gap-2"):
                ui.button("Cancel", on_click=self.watch_all_dialog.close).props("flat")
                ui.button(
                    "Show all",
                    color="red",
                    on_click=lambda: (self.watch_all(), self.watch_all_dialog.close()),
                )

    def _build_header(self) -> None:
        with _top_bar_row():
            self.architecture_toggle = (
                ui.button(icon="account_tree", color="slate-500")
                .props("dense size=md")
                .tooltip("Toggle architecture pane")
            )
            _refresh_button(self.session)
            _add_step_controls(self.session, self.step_until_custom)
            self.layer_menu = LayerMenu(
                self.layer_names,
                self.state.last_watched,
                locked=self.session.locked,
                show_all=self.watch_all_dialog.open,
                clear_all=self.clear_all,
                toggle_stats=lambda: self.toggle_stats(),
            )
            _add_settings_button(self.session, self.record_view)
            self.input_toggle = (
                ui.button(icon="image", color="slate-500")
                .props("dense size=md")
                .tooltip("Show or hide the input panel")
            )
            _add_tour_button()
            _add_share_button(self.session)
            _add_repo_logo()

    def _build_body(self) -> None:
        with ui.row().classes("w-full no-wrap gap-0 grow min-h-0"):
            self.architecture_pane = (
                ui.column()
                .classes(
                    "w-1/4 shrink-0 h-full overflow-auto p-2 "
                    "border-r-2 border-slate-300 bg-slate-50"
                )
                .props(_resizable_pane_props("main-architecture"))
            )
            self.architecture_handle = _resize_handle("main-architecture", "left")
            with self.architecture_pane:
                ui.mermaid(self.mermaid_src).classes("w-full")
            with ui.column().classes(
                "grow min-w-0 h-full overflow-auto p-3 bg-slate-200 gap-3"
            ):
                self.empty_hint = _notice_banner(
                    "Select a layer in the architecture to inspect it.",
                    icon="touch_app",
                )
                self.empty_hint.set_visibility(not self.state.last_watched)
                # Every card is built once (cheap: header + empty strips) but
                # only watched ones are visible — and only visible cards get
                # strip data, so hidden layers cost neither render time nor
                # websocket bytes.
                for name in self.layer_names:
                    self.layer_views[name] = _LayerView(
                        name,
                        slug=self.slugs[name],
                        visible=name in self.state.last_watched,
                        decoupled=self.decoupled(),
                        weights=self.layer_weights.get(name, []),
                        on_toggle_watch=self.toggle_layer,
                    )
            self.input_handle = _resize_handle("main-input", "right")
            self.input_pane = (
                ui.column()
                .classes(
                    f"{_INPUT_PANE_WIDTH} shrink-0 h-full overflow-auto p-3 "
                    "border-l-2 border-slate-300 bg-slate-50 items-center"
                )
                .props(_resizable_pane_props("main-input"))
            )
            with self.input_pane:
                self.input_panel = InputPanel(
                    session=self.session,
                    input_names=self.input_names,
                    input_mean=self.input_mean,
                    input_std=self.input_std,
                    input_transform=self.input_transform,
                    on_change=self.mark_dirty,
                )

    def _connect_events(self) -> None:
        # Diagram clicks arrive as custom events carrying the node's slug; map
        # it back to the layer name and toggle. Unknown slugs (e.g. a node
        # whose label isn't a captured layer) are ignored. Inverting `slugs`
        # (rather than rebuilding via the bare `slug`) keeps this in step with
        # the diagram's disambiguated node ids.

        ui.on("nansense_toggle_layer", self.on_diagram_toggle)
        ui.on("nansense_tour_show_layer", self.on_tour_show_layer)
        # The tour's sample step re-opens the input pane the top bar's image
        # button may have hidden (a no-op when it is already visible).
        ui.on("nansense_tour_show_input", self.show_input)

        # Populate the chip menu and, if anything is already watched, push the
        # set into JS so the MutationObserver applies the amber treatment to
        # mermaid nodes once Mermaid finishes rendering them client-side.
        self.refresh_chip()
        self.sync_stats_icon()
        initial_watched = list(self.state.last_watched)
        if initial_watched:
            slugs_js = json.dumps([self.slugs[n] for n in initial_watched])
            ui.timer(
                0.0,
                lambda: ui.run_javascript(
                    f"({slugs_js}).forEach(s => window.nansenseSetWatched(s, true))"
                ),
                once=True,
            )
        # A locked `?layer=` deep link (a subpage's Back button) lands on the
        # card it names, not the top of the list.
        if self.session.locked and self.focus_layer in self.layer_names:
            scroll_js = f"window.nansenseScrollToCard({json.dumps(self.slugs[self.focus_layer])})"
            ui.timer(0.0, lambda: ui.run_javascript(scroll_js), once=True)
        self.architecture_toggle.on_click(self.toggle_architecture)
        self.input_toggle.on_click(self.toggle_input)

    def decoupled(self) -> bool:
        return self.controller.decoupled

    def shown_layers(self) -> frozenset[str]:
        return self.controller.shown_layers

    def record_view(self) -> RecordedView | None:
        if self.session.snapshot is None and self.session.probe_result is None:
            return None
        shown = self.shown_layers()
        watched = [n for n in self.layer_names if n in shown]
        plural = "" if len(watched) == 1 else "s"
        # Record exactly the input the pane is showing, with its resolved
        # display config (a same-process dict, so the transform travels too).
        selected = self.input_panel.selected_input
        return RecordedView(
            key="main",
            label=f"Main view ({len(watched)} watched layer{plural}, sample {self.input_panel.sample_idx})",
            config=MainView(
                layers=tuple(watched),
                sample_idx=self.input_panel.sample_idx,
                input_name=selected or "",
                input_mean=resolve_per_input(self.input_mean, selected),
                input_std=resolve_per_input(self.input_std, selected),
                input_transform=resolve_per_input(self.input_transform, selected),
                render_average=self.input_panel.render_options.average,
                render_values=self.input_panel.render_options.values,
            ),
        )

    def watch_all(self) -> None:
        self.controller.show_all()
        self.sync_watch_ui()

    def clear_all(self) -> None:
        if not self.controller.decoupled and _refuse_unwatch_while_recording(
            self.session
        ):
            return
        self.controller.clear()
        self.sync_watch_ui()

    def on_diagram_toggle(self, e: GenericEventArguments) -> None:
        name = self.slug_to_name.get(e.args)
        if name is not None:
            self.toggle_layer(name)

    def on_tour_show_layer(self, e: GenericEventArguments) -> None:
        # The tour's card-needing steps must never hide a card the visitor
        # already opened, so unlike a diagram click this is show-only;
        # `toggle_layer` still does the showing, keeping the watch/sync/
        # scroll logic in one place for both the decoupled and coupled
        # visibility flavors.
        name = self.slug_to_name.get(e.args)
        if name is not None and name not in self.shown_layers():
            self.toggle_layer(name)

    async def tick(self) -> None:
        self.input_panel.refresh_status()
        # While the main view records, its render parameters (sample, pin,
        # perturbations, probe mode) are frozen: the recording renders with
        # the live probe state, so the input controls must not change it.
        self.input_panel.set_frozen(self.session.recording.is_recording("main"))
        # A stats-scope switch (from the settings gear, possibly in another
        # tab) re-bases the shown set: entering a decoupled scope seeds this
        # tab's own set from the global watched set, returning to `watched`
        # re-syncs to it. Switching between the decoupled scopes (the stats
        # pause toggle) keeps the tab's cards as they are.
        scope = self.session.stats_scope
        if self.controller.reconcile_scope():
            for view in self.layer_views.values():
                view.set_decoupled(scope is not StatsScope.WATCHED)
            self.sync_watch_ui()
        # Shown-set changes made elsewhere (another tab or the stats page, in
        # the coupled scope) propagate here: sync flips card visibility and
        # marks the frame dirty so newly visible cards render from the
        # current snapshot.
        elif self.shown_layers() != self.state.last_watched:
            self.sync_watch_ui()
        # Stats-collection state can change from another tab too; keep the
        # icon's colour/strike in sync (cheap class writes, no-op when stable).
        self.sync_stats_icon()
        snap = self.session.snapshot
        # With a probe result present (a batch is pinned, an eval/train forward
        # mode is selected, or — per connection in a locked demo — this visitor
        # perturbed a pixel), the page renders the probe instead of the
        # snapshot: pinning tracks one fixed input across stepping and time
        # travel, while eval/train shows the current batch under that mode.
        # `client_key` is None on an unlocked session, so this is the shared
        # probe result there.
        probe = self.session.probe_result_for(self.input_panel.client_key)
        if snap is None and probe is None:
            return
        self.input_panel.sync_spinner_max(_display_batch_size(snap, probe))
        if self.state.rendering:
            return
        if (
            snap is not self.state.last_snapshot
            or probe is not self.state.last_probe
            or self.state.dirty
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
            self.state.last_snapshot = snap
            self.state.last_probe = probe
            self.state.dirty = False
            self.state.rendering = True
            try:
                sample_idx = self.input_panel.sample_idx
                # Resolve the display config for whichever input the pane shows
                # (a per-input dict collapses to this one input's values).
                selected = self.input_panel.selected_input
                sel_mean = resolve_per_input(self.input_mean, selected)
                sel_std = resolve_per_input(self.input_std, selected)
                sel_transform = resolve_per_input(self.input_transform, selected)
                # Only the visible (= watched) layers render; hidden cards
                # keep whatever stale content they had, which is invisible
                # and re-rendered (cache-assisted) when they reappear.
                visible_names = [
                    n for n in self.layer_names if n in self.state.last_watched
                ]
                rendered, input_src = await asyncio.to_thread(
                    _compute_frame,
                    visible_names,
                    snap,
                    probe,
                    sample_idx,
                    compare=self.input_panel.compare,
                    options=self.input_panel.render_options,
                    input_name=self.input_name,
                    selected_input=selected,
                    input_mean=sel_mean,
                    input_std=sel_std,
                    input_transform=sel_transform,
                    cache=self.render_cache,
                )
            finally:
                self.state.rendering = False
            _apply_all(self.layer_views, rendered)
            self.input_panel.set_image(input_src)
            selected_tensor = _selected_input_tensor(snap, probe, selected)
            self.input_panel.set_input_warning(
                input_blank_warning(
                    selected_tensor,
                    sample_idx,
                    name=selected,
                    mean=sel_mean,
                    std=sel_std,
                    transform=sel_transform,
                )
            )
            self.input_panel.set_input_legend(
                _input_img_src(render_input_legend(selected_tensor, sample_idx))
            )

    def sync_stats_icon(self) -> None:
        self.layer_menu.sync_collecting(self.session.stats_collecting)

    def toggle_stats(self) -> None:
        self.session.toggle_stats_collecting()
        self.sync_stats_icon()

    def refresh_chip(self) -> None:
        self.layer_menu.update(self.shown_layers())

    def sync_watch_ui(self) -> None:
        """Reflect the shown set in this connection's DOM.

        In the `watched` scope, visible is synonymous with watched (the
        global set); in the decoupled scopes it is this tab's own `shown`
        set. Either way: cards for newly shown layers appear (and get
        rendered on the next tick via the dirty flag), hidden ones
        disappear, the diagram's amber classes follow, and the chip menu /
        empty-pane hint refresh. Diffing against `state.last_watched`
        keeps the JS push proportional to the change, not the model size.
        """
        shown = self.shown_layers()
        added = shown - self.state.last_watched
        removed = self.state.last_watched - shown
        self.state.last_watched = shown
        for name in added | removed:
            view = self.layer_views.get(name)
            if view is not None:
                view.set_visible(name in shown)
        if added or removed:
            changes = "; ".join(
                f"window.nansenseSetWatched({json.dumps(self.slugs[n])}, "
                f"{'true' if n in shown else 'false'})"
                for n in added | removed
            )
            ui.run_javascript(changes)
            self.state.dirty = True
        self.empty_hint.set_visibility(not shown)
        self.refresh_chip()

    def toggle_layer(self, name: str) -> None:
        if (
            not self.controller.decoupled
            and name in self.session.watched_layers
            and _refuse_unwatch_while_recording(self.session)
        ):
            return
        if not self.controller.toggle(name):
            return
        self.sync_watch_ui()
        if name in self.shown_layers():
            ui.run_javascript(
                f"window.nansenseScrollToCard({json.dumps(self.slugs[name])})"
            )

    def toggle_architecture(self) -> None:
        visible = not self.architecture_pane.visible
        self.architecture_pane.set_visibility(visible)
        self.architecture_handle.set_visibility(visible)

    def show_input(self) -> None:
        # Show-only half of `toggle_input`, for the tour's sample step:
        # re-showing an already-visible pane is a no-op.
        self.input_pane.set_visibility(True)
        self.input_handle.set_visibility(True)

    def toggle_input(self) -> None:
        visible = not self.input_pane.visible
        self.input_pane.set_visibility(visible)
        self.input_handle.set_visibility(visible)

    def mark_dirty(self) -> None:
        self.state.dirty = True


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
    '<div class="text-xs text-slate-400 italic py-1">no gradients on probe runs</div>'
)

# One layer card's rendered content: the activation and gradient strips plus
# the custom layer-tensor strips (`Session.watch_layer_tensor`) as
# (instrument name, strip html) pairs.
_LayerStrips = tuple[str, str, tuple[tuple[str, str], ...]]

# Strips shown for a layer whose render fell over. `render_strip` already
# returns `None` (hidden strip) for empty/unsupported tensors; this covers the
# residual cases (a genuinely unexpected render bug) so one bad layer degrades
# to a blank card instead of taking the whole frame down with it.
_EMPTY_STRIPS: _LayerStrips = ("", "", ())


def _render_layers(
    layer_names: list[str], strips: Callable[[str], _LayerStrips]
) -> dict[str, _LayerStrips]:
    """Render every layer's strips concurrently, isolating failures.

    Layers fan out over `_RENDER_POOL`; a single layer that raises must not
    abort its siblings or drop the frame, so each render is guarded and a
    failed layer yields blank strips (`_EMPTY_STRIPS`) — the same empty
    result a layer absent from the snapshot gets.
    """

    def guarded(name: str) -> _LayerStrips:
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
    options: RenderOptions = DEFAULT_RENDER_OPTIONS,
    input_name: str | None,
    selected_input: str | None = None,
    input_mean: tuple[float, ...] | None,
    input_std: tuple[float, ...] | None,
    input_transform: InputTransform | None = None,
    cache: _RenderCache,
) -> tuple[dict[str, _LayerStrips], str]:
    """Render every layer's strips plus the input image source.

    With a probe result present it is the render source (pinned-batch /
    perturbed view, see `_compute_probe_frame`); otherwise the snapshot is.
    Layers render concurrently on `_RENDER_POOL`; each strip goes through
    `cache`, so only strips not already rendered for this source cost
    anything. `input_name` is the primary image input (its `H × W` sets the
    token grid for 2D activations); `selected_input` is the input shown in the
    pane — it defaults to `input_name` (the same one unless the user picked
    another from the multi-input dropdown). `options` is the viewer's render
    choice (magnitude, channel averaging) and rides in the cache keys, so
    switching it re-renders rather than serving the previous look.
    """
    if selected_input is None:
        selected_input = input_name
    if probe is not None:
        return _compute_probe_frame(
            layer_names,
            probe,
            sample_idx,
            compare=compare,
            options=options,
            input_name=input_name,
            selected_input=selected_input,
            input_mean=input_mean,
            input_std=input_std,
            input_transform=input_transform,
            cache=cache,
        )
    assert snap is not None  # tick only renders when at least one source exists
    input_hw = tensor_hw(snap.activations.get(input_name) if input_name else None)
    opt = options.cache_key

    def strips(name: str) -> _LayerStrips:
        if compare:
            # Diff view without any probe (perturb mode on, nothing clicked
            # yet, no pin): the diff is identically zero, rendered as a
            # white strip — same as a perturbation-free probe diff.
            act = cache.get_or_render(
                snap,
                (name, f"act-diff:{opt}", sample_idx),
                lambda: _strip_html(
                    render_strip(
                        _zeros_like(snap.activations.get(name)),
                        sample_idx,
                        input_hw=input_hw,
                        options=options,
                    ),
                    show_labels=True,
                ),
            )
        else:
            act = cache.get_or_render(
                snap,
                (name, f"act:{opt}", sample_idx),
                lambda: _strip_html(
                    render_strip(
                        snap.activations.get(name),
                        sample_idx,
                        input_hw=input_hw,
                        options=options,
                    ),
                    show_labels=True,
                ),
            )
        grad = cache.get_or_render(
            snap,
            (name, f"grad:{opt}", sample_idx),
            lambda: _strip_html(
                render_strip(
                    snap.activation_gradients.get(name),
                    sample_idx,
                    input_hw=input_hw,
                    options=options,
                )
            ),
        )
        # Custom layer-tensor strips ride below the pair; the diff view
        # skips them (a custom tensor has no perturbed counterpart to diff).
        custom: tuple[tuple[str, str], ...] = ()
        if not compare:
            custom = tuple(
                (
                    label,
                    cache.get_or_render(
                        snap,
                        (name, f"custom:{label}:{opt}", sample_idx),
                        lambda tensor=tensor: _strip_html(
                            render_strip(
                                tensor,
                                sample_idx,
                                input_hw=input_hw,
                                options=options,
                            )
                        ),
                    ),
                )
                for label, tensor in sorted(
                    snap.custom_activations.get(name, {}).items()
                )
            )
        return act, grad, custom

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
    options: RenderOptions,
    input_name: str | None,
    selected_input: str | None,
    input_mean: tuple[float, ...] | None,
    input_std: tuple[float, ...] | None,
    input_transform: InputTransform | None,
    cache: _RenderCache,
) -> tuple[dict[str, _LayerStrips], str]:
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
    kind = f"{kind}:{options.cache_key}"
    input_hw = tensor_hw(probe.base_input(input_name))

    def strips(name: str) -> _LayerStrips:
        act = cache.get_or_render(
            probe,
            (name, kind, sample_idx),
            lambda: _strip_html(
                render_strip(
                    probe_act_tensor(
                        probe, name, compare=compare, sample_idx=sample_idx
                    ),
                    0,
                    input_hw=input_hw,
                    options=options,
                ),
                show_labels=True,
            ),
        )
        # Custom layer tensors are snapshot cargo — a probe forward never
        # runs the instruments, so the rows disappear like the gradients do.
        return act, _PROBE_NO_GRADIENTS_HTML, ()

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
    rendered: dict[str, _LayerStrips],
) -> None:
    for name, (act_html, grad_html, custom) in rendered.items():
        views[name].apply(act_html, grad_html, custom)


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
                            f"This layer's weights ({len(weights)})"
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
                        "Deep dream and attribution experiments on this layer"
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
                    ).tooltip("Open this layer's stats")
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
                            ACTIVATIONS.css, "ACTIVATIONS", header_gap=True
                        )
                        self.act_html = ui.html("")
                    ui.element("div").classes("h-1")
                    with ui.element("div").classes("flex no-wrap items-stretch"):
                        _strip_marker(GRADIENTS.css, "GRADIENTS")
                        self.grad_html = ui.html("")
                    # Custom layer-tensor strips (`Session.watch_layer_tensor`)
                    # follow, one labelled row per instrument. The rows are
                    # (re)built by `apply` only when the instrument set
                    # changes; in between only their contents update.
                    self._custom_container = ui.element("div")
                    self._custom_rows: dict[str, ui.html] = {}
                    self._custom_labels: tuple[str, ...] = ()
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
            self._hide_tooltip.set_text("Hide this layer's card")
        else:
            self._hide_button.set_text("Unwatch")
            self._hide_tooltip.set_text(
                "Hide this card and drop the layer's collected stats"
            )

    def apply(
        self,
        act_html: str,
        grad_html: str,
        custom: tuple[tuple[str, str], ...] = (),
    ) -> None:
        self.act_html.set_content(act_html)
        self.grad_html.set_content(grad_html)
        labels = tuple(label for label, _ in custom)
        if labels != self._custom_labels:
            self._custom_labels = labels
            self._custom_rows.clear()
            self._custom_container.clear()
            with self._custom_container:
                for label in labels:
                    ui.element("div").classes("h-1")
                    with ui.element("div").classes(
                        "flex no-wrap items-stretch"
                    ):
                        _strip_marker(CUSTOM_TENSOR.css, label.upper())
                        self._custom_rows[label] = ui.html("")
        for label, content in custom:
            self._custom_rows[label].set_content(content)


def _input_img_src(image: bytes | None) -> str:
    """Data-URI source for the input pane's interactive image ("" when absent).

    The sizing (CSS upscale to `INPUT_IMAGE_SIZE` with nearest-neighbour
    rendering) lives on the `ui.interactive_image` element in `InputPanel`;
    only the native-resolution source travels per frame.
    """
    if image is None:
        return ""
    return _b64_img_src(image)
