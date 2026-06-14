"""The main page: architecture diagram, watched-layer cards, input pane."""

from __future__ import annotations

import asyncio
import json
import os
import threading
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from urllib.parse import quote

import torch
from nicegui import ui
from nicegui.events import GenericEventArguments
from torch import Tensor

from nansense.probe import ProbeResult
from nansense.recording import RecordedView
from nansense.session import BatchSnapshot, Session
from nansense.ui.common import (
    _b64_img_src,
    _install_panel_resize,
    _page_scaffold,
    _refuse_unwatch_while_recording,
    _resizable_pane_props,
    _resize_handle,
    _strip_html,
    _strip_marker,
)
from nansense.ui.graph import slug_map
from nansense.ui.input_panel import InputPanel
from nansense.ui.render import probe_act_tensor, render_image, render_strip, tensor_hw
from nansense.ui.static import (
    _ARCHITECTURE_CLICK_CSS,
    _ARCHITECTURE_CLICK_JS,
    _STRIP_MARKER_CSS,
)
from nansense.ui.top_bar import (
    _add_error_banner,
    _add_settings_button,
    _add_step_controls,
    _build_step_until_custom_dialog,
    _refresh_button,
    _top_bar_row,
)


@dataclass
class _PageState:
    last_snapshot: BatchSnapshot | None = None
    last_probe: ProbeResult | None = None
    dirty: bool = False
    rendering: bool = False
    # The watched set this connection last reflected in its DOM (card
    # visibility, amber classes, chip). The tick compares it against
    # `session.watched_layers` so changes made elsewhere (another tab)
    # propagate here too.
    last_watched: frozenset[str] = frozenset()


# Shared pool for strip rendering. Per-layer renders are independent and the
# heavy parts (torch interpolate, numpy colormap, PIL PNG encode) release the
# GIL, so a new snapshot's strips render in parallel across cores. Workers
# spawn lazily, so the pool costs nothing until the first frame.
_RENDER_POOL = ThreadPoolExecutor(
    max_workers=min(8, os.cpu_count() or 1), thread_name_prefix="nansense-render"
)


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


def _build_page(
    session: Session,
    mermaid_src: str,
    layer_names: list[str],
    *,
    input_name: str | None,
    input_mean: tuple[float, ...] | None,
    input_std: tuple[float, ...] | None,
    render_cache: _RenderCache,
) -> None:
    state = _PageState()
    state.last_watched = session.watched_layers
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
        watched = [n for n in layer_names if n in session.watched_layers]
        plural = "" if len(watched) == 1 else "s"
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
                "input_name": input_name or "",
                "input_mean": input_mean,
                "input_std": input_std,
            },
        )

    _page_scaffold()
    _install_panel_resize()
    ui.add_head_html(_ARCHITECTURE_CLICK_CSS)
    ui.add_head_html(_STRIP_MARKER_CSS)
    ui.add_body_html(_ARCHITECTURE_CLICK_JS)
    ui.add_body_html(_layer_info_script(session.layer_info, slugs))

    step_until_custom = _build_step_until_custom_dialog(session)

    def watch_all() -> None:
        for name in layer_names:
            session.watch(name)
        sync_watch_ui()

    def clear_all() -> None:
        if _refuse_unwatch_while_recording(session):
            return
        for name in list(session.watched_layers):
            session.unwatch(name)
        sync_watch_ui()

    # Watching everything turns the lazy-rendering optimization off again:
    # every card renders on every pause and stats accumulate for every
    # layer on every batch. Worth an explicit confirmation.
    watch_all_dialog = ui.dialog()
    with watch_all_dialog, ui.card().classes("max-w-md"):
        ui.label("Watch all layers?").classes("text-lg font-medium")
        ui.label(
            "Every layer card will be rendered on every pause and per-layer "
            "statistics will accumulate on every batch. On larger models this "
            "can make the interface very slow and may even crash the browser "
            "tab."
        ).classes("text-sm text-slate-600")
        with ui.row().classes("w-full justify-end gap-2"):
            ui.button("Cancel", on_click=watch_all_dialog.close).props("flat")
            ui.button(
                "Watch all",
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
                str(len(session.watched_layers)),
                icon="visibility",
                color="slate-100",
            ).classes(
                "ml-auto text-amber-700 font-mono"
            ).props("dense size=md no-caps").tooltip(
                "Watched layers — click to open the watch view or jump to a layer"
            )
            watch_list_container: ui.element
            with watch_chip:
                with ui.menu().props("anchor='bottom right' self='top right'"):
                    # Plain block container, NOT a flex column: Firefox fails to
                    # position/size a QMenu whose content root is a flex column,
                    # so the menu opens collapsed (height 0) and looks like it
                    # never opened. See quasarframework/quasar#16167. Block-level
                    # children stack vertically anyway.
                    with ui.element("div").classes("min-w-64"):
                        # A real anchor (href) instead of a JS navigate: the
                        # browser natively opens middle/ctrl clicks in a new
                        # tab and plain clicks in the current one.
                        ui.menu_item("Open watch view  →").props(
                            'href="/watch"'
                        ).classes("font-medium")
                        ui.separator()
                        ui.menu_item(
                            "Watch all layers…",
                            on_click=watch_all_dialog.open,
                        ).classes("text-sm")
                        ui.menu_item(
                            "Clear all watches",
                            on_click=lambda: clear_all(),
                        ).classes("text-sm")
                        ui.separator()
                        watch_list_container = ui.element("div").classes("py-1")
            _add_settings_button(session, record_view)
            input_toggle = ui.button(
                icon="image", color="slate-500"
            ).props("dense size=md").tooltip(
                "Toggle input selection pane"
            )

        _add_error_banner(session)

        def refresh_chip() -> None:
            watched = session.watched_layers
            watch_chip.text = str(len(watched))
            watch_list_container.clear()
            with watch_list_container:
                if not watched:
                    ui.label("No layers watched").classes(
                        "px-3 py-2 text-slate-500 text-sm italic"
                    )
                    return
                # Section header: the entries below scroll the main view to a
                # layer, which is easy to misread as further actions like the
                # "Open watch view" item above.
                ui.label("Jump to layer").classes(
                    "px-3 pt-1 pb-0.5 text-xs uppercase tracking-wider "
                    "text-slate-400 select-none"
                )
                for layer in layer_names:
                    if layer not in watched:
                        continue
                    ui.menu_item(
                        layer,
                        on_click=lambda n=layer: ui.run_javascript(
                            f"window.nansenseScrollToLayer({json.dumps(slugs[n])})"
                        ),
                    ).classes("font-mono text-sm")

        def sync_watch_ui() -> None:
            """Reflect `session.watched_layers` in this connection's DOM.

            Visible is synonymous with watched: cards for newly watched
            layers appear (and get rendered on the next tick via the dirty
            flag), unwatched ones hide, the diagram's amber classes follow,
            and the chip menu / empty-pane hint refresh. Diffing against
            `state.last_watched` keeps the JS push proportional to the
            change, not the model size.
            """
            watched = session.watched_layers
            added = watched - state.last_watched
            removed = state.last_watched - watched
            state.last_watched = watched
            for name in added | removed:
                view = layer_views.get(name)
                if view is not None:
                    view.set_visible(name in watched)
            if added or removed:
                changes = "; ".join(
                    f"window.nansenseSetWatched({json.dumps(slugs[n])}, "
                    f"{'true' if n in watched else 'false'})"
                    for n in added | removed
                )
                ui.run_javascript(changes)
                state.dirty = True
            empty_hint.set_visibility(not watched)
            refresh_chip()

        def toggle_layer(name: str) -> None:
            # Any name in `session.layer_names` is watchable (modules, fx
            # intermediates, graph inputs); False means an unknown name.
            if name in session.watched_layers:
                if _refuse_unwatch_while_recording(session):
                    return
                session.unwatch(name)
            elif not session.watch(name):
                return
            sync_watch_ui()
            if name in session.watched_layers:
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
            layer_weights = session.layer_weights
            with ui.column().classes(
                "grow min-w-0 h-full overflow-auto p-3 bg-slate-200 gap-3"
            ):
                empty_hint = ui.label(
                    "No layers shown — click a node in the architecture "
                    "diagram to show a layer's card and start watching it."
                ).classes("text-slate-500 italic text-sm p-2")
                empty_hint.set_visibility(not state.last_watched)
                # Every card is built once (cheap: header + empty strips) but
                # only watched ones are visible — and only visible cards get
                # strip data, so hidden layers cost neither render time nor
                # websocket bytes.
                for name in layer_names:
                    layer_views[name] = _LayerView(
                        name,
                        slug=slugs[name],
                        session=session,
                        weights=layer_weights.get(name, []),
                        on_toggle_watch=toggle_layer,
                    )
            input_handle = _resize_handle("main-input", "right")
            input_pane = ui.column().classes(
                "w-72 shrink-0 h-full overflow-auto p-3 "
                "border-l-2 border-slate-300 bg-slate-50 items-center"
            ).props(_resizable_pane_props("main-input"))
            with input_pane:

                def mark_dirty() -> None:
                    state.dirty = True

                input_panel = InputPanel(
                    session=session,
                    input_name=input_name,
                    input_mean=input_mean,
                    input_std=input_std,
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

    async def tick() -> None:
        input_panel.refresh_status()
        # While the main view records, its render parameters (sample, pin,
        # perturbations, probe mode) are frozen: the recording renders with
        # the live probe state, so the input controls must not change it.
        input_panel.set_frozen(session.recording.is_recording("main"))
        # Watched-set changes made elsewhere (another tab, the watch page)
        # propagate here: sync flips card visibility and marks the frame
        # dirty so newly visible cards render from the current snapshot.
        if session.watched_layers != state.last_watched:
            sync_watch_ui()
        snap = session.snapshot
        # With a probe result present (a batch is pinned), the page renders
        # the probe instead of the snapshot — that's the point of pinning:
        # the strips track one fixed input across stepping and time travel.
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
                    input_mean=input_mean,
                    input_std=input_std,
                    cache=render_cache,
                )
            finally:
                state.rendering = False
            _apply_all(layer_views, rendered)
            input_panel.set_image(input_src)

    ui.timer(0.2, tick)


def _snapshot_batch_size(snap: BatchSnapshot) -> int | None:
    for tensor in snap.activations.values():
        if tensor.ndim > 0:
            return int(tensor.shape[0])
    return None


def _zeros_like(tensor: Tensor | None) -> Tensor | None:
    return torch.zeros_like(tensor) if tensor is not None else None


def _display_batch_size(
    snap: BatchSnapshot | None, probe: ProbeResult | None
) -> int | None:
    """Batch size of whatever the page is currently rendering."""
    if probe is not None:
        return int(probe.input.shape[0]) if probe.input.ndim > 0 else None
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
    input_mean: tuple[float, ...] | None,
    input_std: tuple[float, ...] | None,
    cache: _RenderCache,
) -> tuple[dict[str, tuple[str, str]], str]:
    """Render every layer's strip pair plus the input image source.

    With a probe result present it is the render source (pinned-batch /
    perturbed view, see `_compute_probe_frame`); otherwise the snapshot is.
    Layers render concurrently on `_RENDER_POOL`; each strip goes through
    `cache`, so only strips not already rendered for this source cost
    anything.
    """
    if probe is not None:
        return _compute_probe_frame(
            layer_names,
            probe,
            sample_idx,
            compare=compare,
            input_mean=input_mean,
            input_std=input_std,
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
                    )
                ),
            )
        else:
            act = cache.get_or_render(
                snap,
                (name, "act", sample_idx),
                lambda: _strip_html(
                    render_strip(
                        snap.activations.get(name), sample_idx, input_hw=input_hw
                    )
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
        (input_name or "", "input", sample_idx),
        lambda: _input_img_src(
            render_image(
                snap.activations.get(input_name) if input_name else None,
                sample_idx,
                mean=input_mean,
                std=input_std,
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
    input_mean: tuple[float, ...] | None,
    input_std: tuple[float, ...] | None,
    cache: _RenderCache,
) -> tuple[dict[str, tuple[str, str]], str]:
    """The probe-sourced equivalent of the snapshot frame.

    Without perturbations the strips show the base activations. With
    perturbations they show the perturbed forward's activations, or — with
    `compare` on — the per-layer diff `perturbed − original`, whose nonzero
    extent traces how far the edit propagates (receptive field). The diff
    view renders even with nothing perturbed: an all-zero diff draws as a
    white strip, signalling "no differences" rather than falling back to a
    non-diff view. The input image shows the perturbed input whenever one
    exists, so the edit is visible. Probe runs are forward-only, so every
    gradient strip shows a placeholder note instead of an image.
    """
    kind = "probe-diff" if compare else (
        "probe-perturbed" if probe.perturbed_activations is not None else "probe-act"
    )
    input_hw = tensor_hw(probe.input)

    def strips(name: str) -> tuple[str, str]:
        act = cache.get_or_render(
            probe,
            (name, kind, sample_idx),
            lambda: _strip_html(
                render_strip(
                    probe_act_tensor(probe, name, compare=compare),
                    sample_idx,
                    input_hw=input_hw,
                )
            ),
        )
        return act, _PROBE_NO_GRADIENTS_HTML

    rendered = _render_layers(layer_names, strips)
    shown_input = (
        probe.perturbed_input if probe.perturbed_input is not None else probe.input
    )
    input_src = cache.get_or_render(
        probe,
        ("", "probe-input", sample_idx),
        lambda: _input_img_src(
            render_image(shown_input, sample_idx, mean=input_mean, std=input_std)
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

    Cards are built for every layer but shown only while the layer is
    watched (`set_visible`) — visible is synonymous with watched, so the
    header carries a permanent "Unwatch" button and hidden cards receive
    no strip data at all.

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
        session: Session,
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
                if weights:
                    with ui.element("div").props("data-card-action"):
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
                with ui.element("div").props("data-card-action"):
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
                # Visible is synonymous with watched: this card only shows
                # while the layer is watched, so the button is always the
                # "off" direction.
                with ui.element("div").props("data-card-action"):
                    ui.button(
                        "Unwatch",
                        icon="visibility_off",
                        on_click=lambda: on_toggle_watch(name),
                        color="red",
                    ).props("dense no-caps").style(
                        "min-height: 0; padding: 1px 6px; font-size: 11px"
                    ).tooltip("Unwatch this layer and hide its card")
            with ui.element("div").classes("w-full overflow-x-auto p-2"):
                # The max-content wrapper makes every row span the widest
                # strip. Without it a row is only as wide as the visible
                # container, so its sticky marker would be dragged out of
                # view once the strips are scrolled past that width.
                with ui.element("div").classes("w-max min-w-full"):
                    with ui.element("div").classes("flex no-wrap items-stretch"):
                        _strip_marker("bg-emerald-500", "ACTIVATIONS")
                        self.act_html = ui.html("")
                    ui.element("div").classes("h-1")
                    with ui.element("div").classes("flex no-wrap items-stretch"):
                        _strip_marker("bg-violet-500", "GRADIENTS")
                        self.grad_html = ui.html("")
        self._card = card
        # A page (re)built with layers already in the watched set (e.g.
        # after navigating back from `/watch`) shows those cards right away.
        self.set_visible(name in session.watched_layers)

    def set_visible(self, visible: bool) -> None:
        self._card.set_visibility(visible)

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
