"""NiceGUI app that visualizes a `Session`.

The app runs on a background daemon-thread-disabled uvicorn (signal handlers
disabled so it survives being started from a non-main thread). Layout:

- Top bar with the six control buttons and a position label.
- Left pane: the model architecture as a Mermaid diagram (built once at
  start). Clicking a node toggles its layer's visibility in the centre
  pane — visible is synonymous with watched (`session.watch`).
- Centre pane: one card per *watched* layer (empty by default — a hint
  points at the diagram). Each card holds two horizontally scrollable
  strips — activations on top, activation gradients below — sharing a
  single horizontal scrollbar so they pan together, and an "Unwatch"
  button that hides the card again. The watch-chip menu in the top bar
  offers "Watch all layers" (behind a performance-warning dialog, since
  watching everything renders every card and accumulates stats for every
  layer on every batch) and "Clear all watches".
- Right pane: the "Input Selection" sidebar (see `playgrad.ui.input_panel`)
  with the sample spinner, the batch-pinning probe controls, and the input
  image.

A `ui.timer` in each connection polls `session.snapshot` and
`session.probe_result`; when a new one is published, the page re-renders
the *watched* layers' strips against it (the probe — the pinned-batch view
— wins when present). Unwatched layers are never rendered or shipped to
the browser, which is what keeps large models responsive. The same timer
also refreshes the top-bar position label from `session.live_position` on
every tick, so the displayed epoch/batch keeps advancing during modes that
don't publish a snapshot every batch (step-epoch, step-custom, run,
detach) — a cheap label write, decoupled from the heavier strip rendering.
"""

from __future__ import annotations

import asyncio
import base64
import html
import json
import math
import os
import threading
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import quote

import plotly.graph_objects as go
import uvicorn
from fastapi import FastAPI
from plotly.subplots import make_subplots
from nicegui import ui
from nicegui.events import GenericEventArguments
import torch
from torch import Tensor

from playgrad.patches import PATCH_TYPES, PatchType
from playgrad.experiments import (
    _DEFAULT_DREAM_BATCH,
    EXPERIMENT_KINDS,
    ExperimentResult,
)
from playgrad.probe import ProbeResult
from playgrad.restore import TimeTravelError
from playgrad.schedule import BatchPosition, Schedule
from playgrad.session import BatchSnapshot, Session
from playgrad.ui.graph import build_mermaid, slug
from playgrad.ui.input_panel import InputPanel
from playgrad.ui.render import (
    INPUT_IMAGE_SIZE,
    PatchGridRender,
    StripRender,
    default_weight_dims,
    image_mime,
    render_image,
    render_patch_grid,
    render_strip,
    render_weight,
)
from playgrad.watch import (
    BINS_PER_DECADE,
    LOG10_MAX,
    LOG10_MIN,
    N_BINS,
    ZERO_BIN,
    LayerStatsSnapshot,
    TensorStatsSnapshot,
    WatchSnapshot,
    bin_midpoint,
    histogram_edges,
)

_ARCHITECTURE_CLICK_CSS: str = """
<style>
  g.node { cursor: pointer; }
  [data-layer] > :first-child { cursor: pointer; }
  [data-layer].playgrad-highlight {
    box-shadow: 0 0 0 3px rgb(96 165 250);
  }
  /* SVG nodes don't honour `box-shadow`, so the matching highlight uses
     an SVG filter that glows around the node's shape. */
  g.node.playgrad-highlight {
    filter: drop-shadow(0 0 4px rgb(96 165 250));
  }
  /* Watched: stronger, amber-tinted treatment that persists across hover.
     Distinct from the blue hover highlight so the two signals don't
     blur into one. */
  [data-layer].playgrad-watched {
    box-shadow:
      0 0 0 3px rgb(245 158 11),
      0 0 12px rgba(245, 158, 11, 0.55);
  }
  g.node.playgrad-watched {
    filter:
      drop-shadow(0 0 6px rgb(245 158 11))
      drop-shadow(0 0 3px rgb(245 158 11));
  }
  /* Watched + hovered: amber ring stays, blue layered around it. */
  [data-layer].playgrad-watched.playgrad-highlight {
    box-shadow:
      0 0 0 3px rgb(245 158 11),
      0 0 0 6px rgba(96, 165, 250, 0.6);
  }
  g.node.playgrad-watched.playgrad-highlight {
    filter:
      drop-shadow(0 0 6px rgb(245 158 11))
      drop-shadow(0 0 4px rgb(96 165 250));
  }
</style>
"""

# Mermaid SVG node ids look like "<element>-flowchart-<slug>-<counter>"; the
# matching layer card carries `data-layer="<slug>"` so we can cross-link
# the two. Hovering either side adds `.playgrad-highlight` to both ends
# of the pair. Clicking a diagram node emits `playgrad_toggle_layer` to
# the server, which toggles the layer's watched state (and with it the
# card's visibility); clicking a card header scrolls the diagram to the
# matching node. Scroll positions are computed directly instead of via
# `scrollIntoView`, because the latter leaves the target several dozen
# pixels below the column's top edge here (the previous item's tail stays
# visible), even with `block: 'start'`.
_ARCHITECTURE_CLICK_JS: str = """
<script>
(function() {
  const watchedSlugs = new Set();

  function slugFromMermaidId(id) {
    const m = /-flowchart-(.+)-\\d+$/.exec(id || '');
    return m ? m[1] : null;
  }
  function findMermaidNode(slug) {
    return document.querySelector(
      'g.node[id*="-flowchart-' + slug.replace(/"/g, '') + '-"]'
    );
  }
  function findCard(slug) {
    return document.querySelector(
      '[data-layer="' + slug.replace(/"/g, '') + '"]'
    );
  }
  function matchPair(el) {
    if (!el || !el.closest) return null;
    const node = el.closest('g.node');
    if (node) {
      const slug = slugFromMermaidId(node.id);
      if (!slug) return null;
      const card = findCard(slug);
      if (!card) return null;
      return { node: node, card: card };
    }
    const card = el.closest('[data-layer]');
    if (card) {
      const slug = card.getAttribute('data-layer');
      const node = findMermaidNode(slug);
      if (!node) return null;
      return { node: node, card: card };
    }
    return null;
  }
  function scrollableParent(el) {
    let p = el.parentElement;
    while (p) {
      const oy = getComputedStyle(p).overflowY;
      if ((oy === 'auto' || oy === 'scroll') && p.scrollHeight > p.clientHeight) {
        return p;
      }
      p = p.parentElement;
    }
    return null;
  }
  function scrollTargetToTop(target) {
    const container = scrollableParent(target);
    if (!container) return;
    const cRect = container.getBoundingClientRect();
    const tRect = target.getBoundingClientRect();
    const topPadding = 12;
    container.scrollTo({
      top: container.scrollTop + (tRect.top - cRect.top) - topPadding,
      behavior: 'smooth',
    });
  }

  let highlighted = null;
  function setHighlight(pair) {
    if (highlighted && pair && highlighted.node === pair.node) return;
    if (highlighted) {
      highlighted.node.classList.remove('playgrad-highlight');
      highlighted.card.classList.remove('playgrad-highlight');
    }
    highlighted = pair;
    if (pair) {
      pair.node.classList.add('playgrad-highlight');
      pair.card.classList.add('playgrad-highlight');
    }
  }
  document.addEventListener('mouseover', function(e) {
    setHighlight(matchPair(e.target));
  });
  document.addEventListener('mouseleave', function() {
    setHighlight(null);
  });

  document.addEventListener('click', function(e) {
    if (!e.target.closest) return;
    // Header action buttons (Watch, Weights) handle their own click; the
    // document-level navigation must not fire on top of them.
    if (e.target.closest('[data-card-action]')) return;
    const node = e.target.closest('g.node');
    if (node) {
      const slug = slugFromMermaidId(node.id);
      if (!slug) return;
      // Toggling watched state lives server-side (session.watch); the
      // server answers by updating card visibility and amber classes.
      emitEvent('playgrad_toggle_layer', slug);
      return;
    }
    const card = e.target.closest('[data-layer]');
    if (!card) return;
    // Only the card header (the first child) navigates back to the diagram;
    // clicks inside the strip area shouldn't trigger a jump.
    const header = card.firstElementChild;
    if (!header || !header.contains(e.target)) return;
    const slug = card.getAttribute('data-layer');
    const mNode = findMermaidNode(slug);
    if (!mNode) return;
    scrollTargetToTop(mNode);
  });

  // Toggle the `playgrad-watched` class on both the card and the matching
  // mermaid node. Mermaid renders the SVG asynchronously, so the node may
  // not exist yet when this runs; the MutationObserver below catches it.
  window.playgradSetWatched = function(slug, on) {
    if (on) { watchedSlugs.add(slug); } else { watchedSlugs.delete(slug); }
    const card = findCard(slug);
    if (card) card.classList.toggle('playgrad-watched', on);
    const node = findMermaidNode(slug);
    if (node) node.classList.toggle('playgrad-watched', on);
  };
  // Jump both panes to a layer: the right pane's card and the architecture
  // pane's mermaid node each scroll within their own container.
  window.playgradScrollToLayer = function(slug) {
    const card = findCard(slug);
    if (card) scrollTargetToTop(card);
    const node = findMermaidNode(slug);
    if (node) scrollTargetToTop(node);
  };
  // Card-only variant: used right after a diagram click reveals a card,
  // where also scrolling the diagram would yank the just-clicked node away
  // from under the cursor.
  window.playgradScrollToCard = function(slug) {
    const card = findCard(slug);
    if (card) scrollTargetToTop(card);
  };

  // Re-apply watched classes to any matching mermaid node / card that
  // appears after the initial render. Skips work when nothing is watched.
  const observer = new MutationObserver(function() {
    if (watchedSlugs.size === 0) return;
    for (const slug of watchedSlugs) {
      const card = findCard(slug);
      if (card && !card.classList.contains('playgrad-watched')) {
        card.classList.add('playgrad-watched');
      }
      const node = findMermaidNode(slug);
      if (node && !node.classList.contains('playgrad-watched')) {
        node.classList.add('playgrad-watched');
      }
    }
  });
  observer.observe(document.body, { childList: true, subtree: true });
})();
</script>
"""


def serve(
    session: Session,
    *,
    port: int = 8080,
    host: str = "127.0.0.1",
    log_level: str = "warning",
    input_mean: tuple[float, ...] | None = None,
    input_std: tuple[float, ...] | None = None,
) -> threading.Thread | None:
    """Start the NiceGUI app on a background thread and return that thread.

    Returns `None` without starting anything when `session` is disabled
    (`playgrad.start(..., enabled=False)`), so a training script can call
    `serve()` unconditionally and pay nothing when the UI is turned off.

    NiceGUI is mounted onto a bare FastAPI app via `ui.run_with`; the app is
    then served by uvicorn from a non-main thread, with signal handlers
    disabled so uvicorn doesn't try to wire SIGINT/SIGTERM from a thread
    that isn't the main one.

    `input_mean` / `input_std` are passed to the input-image pane so the
    sample is denormalized (`x * std + mean`) before display. When either
    is `None`, the renderer assumes the input is already in `[0, 1]`.
    """
    if not session.enabled:
        return None
    mermaid_src = build_mermaid(session.model)
    layer_names = session.layer_names
    input_name = session.input_names[0] if session.input_names else None

    fastapi_app = FastAPI()
    favicon_path = Path(__file__).resolve().parents[2] / "assets" / "logo_small.png"
    # One cache for all connections: two tabs on the same session share
    # rendered strips instead of re-rendering them per connection.
    render_cache = _RenderCache()

    @ui.page("/", favicon=str(favicon_path))
    def index() -> None:
        _build_page(
            session,
            mermaid_src,
            layer_names,
            input_name=input_name,
            input_mean=input_mean,
            input_std=input_std,
            render_cache=render_cache,
        )

    @ui.page("/watch", favicon=str(favicon_path))
    def watch_page() -> None:
        _build_watch_page(
            session, layer_names, input_mean=input_mean, input_std=input_std
        )

    @ui.page("/weights", favicon=str(favicon_path))
    def weights_page(layer: str = "") -> None:
        _build_weights_page(session, layer)

    @ui.page("/experiment", favicon=str(favicon_path))
    def experiment_page(layer: str = "") -> None:
        _build_experiment_page(
            session, layer, input_mean=input_mean, input_std=input_std
        )

    ui.run_with(fastapi_app, storage_secret="playgrad")

    config = uvicorn.Config(
        app=fastapi_app,
        host=host,
        port=port,
        log_level=log_level,
    )
    server = uvicorn.Server(config)
    setattr(server, "install_signal_handlers", lambda: None)

    thread = threading.Thread(target=server.run, name="playgrad-ui", daemon=False)
    thread.start()
    return thread


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
    max_workers=min(8, os.cpu_count() or 1), thread_name_prefix="playgrad-render"
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


_TOP_BAR_CLASSES: str = (
    "w-full items-center gap-x-3 gap-y-0 px-3 py-2 shrink-0 "
    "border-b-2 border-slate-300 bg-slate-100 shadow-sm z-10"
)


def _top_bar_row() -> ui.row:
    """The shared top-bar row container used by every page."""
    return ui.row().classes(_TOP_BAR_CLASSES)


def _add_step_controls(session: Session, step_until_custom: ui.dialog) -> ui.label:
    """Add the five stepping buttons + a live-position label to the open row.

    Shared by the main page and the weights page so both drive the session
    identically. The returned label is refreshed from `session.live_position`
    by each page's timer (see `_format_live_position`).
    """
    ui.button("Stop", on_click=session.stop, color="red").props(
        "dense size=md"
    ).tooltip("Pause at the next batch boundary")
    ui.button("Step Batch", on_click=session.step_batch, color="orange").props(
        "dense size=md"
    ).tooltip("Advance one batch, then pause")
    ui.button("Step Epoch", on_click=session.step_epoch, color="orange").props(
        "dense size=md"
    ).tooltip("Run until the epoch changes, then pause")
    ui.button(
        "Step Custom", on_click=step_until_custom.open, color="orange"
    ).props("dense size=md").tooltip("Pick a phase/epoch/batch to pause at")
    ui.button("Detach", on_click=session.detach, color="green").props(
        "dense size=md"
    ).tooltip("Release the training loop and stop capturing snapshots")
    _add_time_travel_button(session)
    return ui.label("(waiting for first snapshot)").classes(
        "ml-3 font-mono text-sm"
    )


def _format_live_position(live: BatchPosition) -> str:
    return f"epoch {live.epoch} | {live.phase} batch {live.batch_idx}"


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

    ui.page_title("PlayGrad")
    ui.query(".nicegui-content").classes("p-0 h-screen overflow-hidden")
    ui.query("body").classes("overflow-hidden")
    ui.query("html").classes("overflow-hidden")
    ui.add_head_html(_ARCHITECTURE_CLICK_CSS)
    ui.add_head_html(_STRIP_MARKER_CSS)
    ui.add_body_html(_ARCHITECTURE_CLICK_JS)

    step_until_custom = _build_step_until_custom_dialog(session)

    def watch_all() -> None:
        for name in layer_names:
            session.watch(name)
        sync_watch_ui()

    def clear_all() -> None:
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
            position_label = _add_step_controls(session, step_until_custom)
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
                        ui.menu_item(
                            "Open watch view  →",
                            on_click=lambda: ui.navigate.to("/watch", new_tab=True),
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
            input_toggle = ui.button(
                icon="image", color="slate-500"
            ).props("dense size=md").tooltip(
                "Toggle input selection pane"
            )

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
                            f"window.playgradScrollToLayer({json.dumps(slug(n))})"
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
                    f"window.playgradSetWatched({json.dumps(slug(n))}, "
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
                session.unwatch(name)
            elif not session.watch(name):
                return
            sync_watch_ui()
            if name in session.watched_layers:
                ui.run_javascript(
                    f"window.playgradScrollToCard({json.dumps(slug(name))})"
                )

        with ui.row().classes("w-full no-wrap gap-0 grow min-h-0"):
            architecture_pane = ui.column().classes(
                "w-1/4 shrink-0 h-full overflow-auto p-2 "
                "border-r-2 border-slate-300 bg-slate-50"
            )
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
                        session=session,
                        weights=layer_weights.get(name, []),
                        on_toggle_watch=toggle_layer,
                    )
            input_pane = ui.column().classes(
                "w-72 shrink-0 h-full overflow-auto p-3 "
                "border-l-2 border-slate-300 bg-slate-50 items-center"
            )
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
            architecture_pane.set_visibility(not architecture_pane.visible)

        def toggle_input() -> None:
            input_pane.set_visibility(not input_pane.visible)

        architecture_toggle.on_click(toggle_architecture)
        input_toggle.on_click(toggle_input)

    # Diagram clicks arrive as custom events carrying the node's slug; map
    # it back to the layer name and toggle. Unknown slugs (e.g. a node
    # whose label isn't a captured layer) are ignored.
    slug_to_name = {slug(n): n for n in layer_names}

    def on_diagram_toggle(e: GenericEventArguments) -> None:
        name = slug_to_name.get(e.args)
        if name is not None:
            toggle_layer(name)

    ui.on("playgrad_toggle_layer", on_diagram_toggle)

    # Populate the chip menu and, if anything is already watched, push the
    # set into JS so the MutationObserver applies the amber treatment to
    # mermaid nodes once Mermaid finishes rendering them client-side.
    refresh_chip()
    initial_watched = list(state.last_watched)
    if initial_watched:
        slugs_js = json.dumps([slug(n) for n in initial_watched])
        ui.timer(
            0.0,
            lambda: ui.run_javascript(
                f"({slugs_js}).forEach(s => window.playgradSetWatched(s, true))"
            ),
            once=True,
        )

    async def tick() -> None:
        # Top-bar position tracks the *live* training position, refreshed on
        # every tick independently of the (possibly much heavier) strip
        # rendering below. The 0.2s timer is the throttle: rapid batches in
        # step_epoch / step_run / detach coalesce into at most ~5 cheap label
        # updates per second, and NiceGUI skips the write when text is
        # unchanged. The strips still re-render only when a new snapshot lands.
        live = session.live_position
        if live is not None:
            position_label.text = _format_live_position(live)
        input_panel.refresh_status()
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


_PHASE_COLORS: dict[str, str] = {
    "train": "#d97706",  # amber
    "val": "#3b82f6",  # blue
    "test": "#10b981",  # emerald — fallback if a user names their phases differently
}
_FALLBACK_COLORS: tuple[str, ...] = ("#a855f7", "#ef4444", "#14b8a6", "#6b7280")


def _phase_color(phase: str, idx: int) -> str:
    return _PHASE_COLORS.get(phase, _FALLBACK_COLORS[idx % len(_FALLBACK_COLORS)])


def _x_tick_layout() -> tuple[list[int], list[str]]:
    """Tick positions (bin indices) and labels for the signed-log x-axis.

    Labels are drawn only at powers of 10 (every 7th edge); the
    intermediate edges shape the bars but are unlabeled to keep the axis
    legible.
    """
    tick_vals: list[int] = [ZERO_BIN]
    tick_text: list[str] = ["0"]
    for k in range(LOG10_MIN, LOG10_MAX + 1):
        offset = (k - LOG10_MIN) * BINS_PER_DECADE
        label = "1" if k == 0 else f"1e{k}"
        tick_vals.append(ZERO_BIN + 1 + offset)
        tick_text.append(label)
        tick_vals.append(ZERO_BIN - 1 - offset)
        tick_text.append("-1" if k == 0 else f"-1e{k}")
    return tick_vals, tick_text


def _format_stat(value: float) -> str:
    """Format a scalar stat for the card header."""
    if math.isnan(value):
        return "—"
    if value == 0:
        return "0"
    abs_v = abs(value)
    if abs_v >= 1000 or abs_v < 0.01:
        return f"{value:.2e}"
    return f"{value:.3g}"


# Rows of the per-histogram stats table: label and how to format the value.
_STAT_ROWS: tuple[tuple[str, Callable[[TensorStatsSnapshot], str]], ...] = (
    ("n", lambda s: f"{s.n:,}"),
    ("mean", lambda s: _format_stat(s.mean)),
    ("std", lambda s: _format_stat(s.std)),
    ("median", lambda s: _format_stat(s.median)),
    ("min", lambda s: _format_stat(s.min)),
    ("max", lambda s: _format_stat(s.max)),
)

_STATS_CELL_STYLE: str = "padding:2px 26px 2px 0;text-align:left"

# Light framed card around each stats table so it stands out from the page
# instead of floating as bare text.
_STATS_BOX_STYLE: str = (
    "display:inline-block;background:#f8fafc;border:1px solid #e2e8f0;"
    "border-radius:6px;padding:8px 14px"
)


def _stats_table_html(per_phase: dict[str, LayerStatsSnapshot], kind: str) -> str:
    """Scalar stats as an HTML table: one column per phase, one row per stat.

    The header of each phase column ("train ep 0") is tinted with the phase's
    trace color so it reads against the matching bars in the histogram below,
    and the whole table sits in a light framed box for visibility. Returns a
    plain "no data yet" note while the phase has no samples.
    """
    phases = _phases_with_data(per_phase, kind)
    if not phases:
        return '<span class="text-slate-500">no data yet</span>'
    header = "".join(
        f'<th style="{_STATS_CELL_STYLE};font-weight:700;'
        f"border-bottom:1px solid #e2e8f0;"
        f'color:{_phase_color(p, i)}">'
        f"{html.escape(p)} ep {per_phase[p].epoch}</th>"
        for i, p in enumerate(phases)
    )
    rows = "".join(
        f'<tr><td style="{_STATS_CELL_STYLE};color:#64748b">{label}</td>'
        + "".join(
            f'<td style="{_STATS_CELL_STYLE};color:#1e293b">'
            f"{fmt(_kind_stats(per_phase[p], kind))}</td>"
            for p in phases
        )
        + "</tr>"
        for label, fmt in _STAT_ROWS
    )
    return (
        f'<div style="{_STATS_BOX_STYLE}">'
        '<table style="border-collapse:collapse">'
        "<thead><tr>"
        f'<th style="border-bottom:1px solid #e2e8f0"></th>{header}'
        "</tr></thead>"
        f"<tbody>{rows}</tbody></table></div>"
    )


# Plot height in px. Doubled from the original 220 so the distributions are
# easier to read.
_PLOT_HEIGHT: int = 440

# Linear-space geometry of the signed-log bins, used when the x-axis is
# switched to a linear scale: each bar is drawn at the centre of its bin and
# given the bin's true (linear) width so the bars tile the value axis.
_HIST_EDGES: list[float] = histogram_edges()
_BIN_CENTERS: list[float] = [
    (_HIST_EDGES[i] + _HIST_EDGES[i + 1]) / 2 for i in range(N_BINS)
]
_BIN_WIDTHS: list[float] = [
    _HIST_EDGES[i + 1] - _HIST_EDGES[i] for i in range(N_BINS)
]
# Hover labels for the signed-log view, where bars sit at plain bin indices:
# each bin's representative value (its geometric midpoint, the same notion
# the median stat uses) instead of the meaningless index.
_BIN_VALUE_LABELS: list[str] = [f"{bin_midpoint(i):.3g}" for i in range(N_BINS)]

# Axis trims may clip bins/bars holding up to this share of the data points
# (see `_trimmed_bin_bounds` / `_linear_y_range`). `_axis_ranges` starts at
# the base share and raises it in steps up to the max while the bars would
# fill less than `_MIN_FILL_FRACTION` of the plot area.
_BASE_CLIP_SHARE: float = 0.005
_MAX_CLIP_SHARE: float = 0.05
_CLIP_SHARE_STEP: float = 0.005

# Minimum share of the plot area the bars should cover; below this the clip
# share keeps being raised (up to `_MAX_CLIP_SHARE`).
_MIN_FILL_FRACTION: float = 0.05

# A bar more than this many times taller than the runner-up in its phase is
# a freak spike (e.g. ReLU's exact zeros) and never anchors the y-scale.
_DOMINANCE_RATIO: float = 5.0


def _use_density(log_x: bool) -> bool:
    """Whether bars show probability density instead of probabilities.

    On a linear value axis, per-bin probabilities are misleading: the
    signed-log bins differ in linear width by orders of magnitude, so a wide
    bin towers over a narrow one holding the same share of values. Density
    makes bar *area* proportional to probability, the honest reading of a
    distribution on a linear value axis. With the signed-log x-axis the bins
    render at uniform width, so plain probabilities are kept there.
    """
    return not log_x


def _probabilities(hist: tuple[int, ...]) -> list[float]:
    """Per-bin probability: count divided by the total count."""
    n = sum(hist)
    if n == 0:
        return [0.0] * len(hist)
    return [c / n for c in hist]


def _probability_densities(hist: tuple[int, ...]) -> list[float]:
    """Per-bin probability density: probability divided by the bin's width."""
    n = sum(hist)
    if n == 0:
        return [0.0] * len(hist)
    return [c / (n * w) for c, w in zip(hist, _BIN_WIDTHS)]


def _trace_heights(hist: tuple[int, ...], density: bool) -> list[float]:
    """Bar heights for one trace: probability densities or probabilities."""
    return _probability_densities(hist) if density else _probabilities(hist)


def _hover_customdata(hist: tuple[int, ...], density: bool) -> list[object]:
    """Per-bar hover payload, matching `_make_histogram_figure`'s templates.

    Density mode (linear x): the raw count alone — the bar's own x position
    is already the value. Signed-log mode: `[count, value-label]` pairs, so
    the hover can show the bin's value instead of its meaningless index.
    """
    if density:
        return list(hist)
    return [
        [count, label] for count, label in zip(hist, _BIN_VALUE_LABELS)
    ]


def _scale_bars(hist: tuple[int, ...], density: bool) -> list[tuple[float, int]]:
    """One trace's `(height, count)` bars that may anchor the y-scale.

    Sorted tallest-first, with a single drastically dominant bar — more than
    `_DOMINANCE_RATIO` times the runner-up — dropped no matter how many
    points it holds: a value the data hits exactly (e.g. ReLU zeros piling
    into the 2e-9-wide zero band) produces a bar that would otherwise
    flatten the rest of the distribution.
    """
    heights = _trace_heights(hist, density)
    bars = sorted(((h, c) for h, c in zip(heights, hist) if c > 0), reverse=True)
    if len(bars) >= 2 and bars[0][0] > _DOMINANCE_RATIO * bars[1][0]:
        return bars[1:]
    return bars


def _linear_y_range(
    per_phase: dict[str, LayerStatsSnapshot],
    kind: str,
    density: bool,
    clip_share: float = _BASE_CLIP_SHARE,
) -> list[float] | None:
    """Y-axis range on a linear y-axis, capped under a clip budget.

    Two clipping rules keep freak spikes from flattening the rest of the
    distribution (with **Log y** checked the axis autoranges instead, so
    everything is visible there):

    - Per phase, a single drastically dominant bar never anchors the scale
      (see `_scale_bars`).
    - Among the rest, bars clip tallest-first, but only as long as the
      clipped bars together hold less than `clip_share` of the pooled data
      points — the cap lands on the tallest bar that must stay fully
      visible.

    The same range is applied to every phase's subplot row so the rows stay
    comparable. Returns `None` (Plotly autorange) when there's no data.
    """
    bars: list[tuple[float, int]] = []
    total = 0
    for phase in _phases_with_data(per_phase, kind):
        hist = _kind_stats(per_phase[phase], kind).hist
        total += sum(hist)
        bars.extend(_scale_bars(hist, density))
    if not bars:
        return None
    bars.sort(reverse=True)
    allowed = clip_share * total
    cap = bars[0][0]
    clipped = 0
    for height, count in bars:
        if clipped + count > allowed:
            cap = height
            break
        clipped += count
    return [0.0, cap * 1.05]


def _kind_stats(layer_snap: LayerStatsSnapshot, kind: str) -> TensorStatsSnapshot:
    """The activation or gradient stats of a layer snapshot, by `kind`."""
    return layer_snap.activations if kind == "activation" else layer_snap.gradients


def _phases_with_data(
    per_phase: dict[str, LayerStatsSnapshot], kind: str
) -> list[str]:
    """Phases that have at least one sample for `kind`, in render order.

    This is exactly the set (and order) of traces `_make_histogram_figure`
    draws, so it doubles as the signature the panel uses to decide whether a
    refresh can restyle in place or must rebuild the figure.
    """
    return [p for p, snap in per_phase.items() if _kind_stats(snap, kind).n > 0]


def _trimmed_bin_bounds(
    per_phase: dict[str, LayerStatsSnapshot],
    kind: str,
    clip_share: float = _BASE_CLIP_SHARE,
) -> tuple[int, int] | None:
    """Smallest/largest bin indices after trimming the extreme-tail bins.

    Pooled across the drawn traces, the outermost populated bins are dropped
    greedily — lighter end first — while the dropped bins together hold less
    than `clip_share` of the data points, so the x-range keeps the bins
    holding the rest of the values and a lone outlier value no longer
    stretches the whole value axis. Returns `None` when there's no data.
    """
    counts = [0] * N_BINS
    for phase in _phases_with_data(per_phase, kind):
        for i, count in enumerate(_kind_stats(per_phase[phase], kind).hist):
            counts[i] += count
    total = sum(counts)
    if total == 0:
        return None
    lo = next(i for i, c in enumerate(counts) if c > 0)
    hi = next(i for i in range(N_BINS - 1, -1, -1) if counts[i] > 0)
    allowed = clip_share * total
    trimmed = 0
    while lo < hi:
        side = lo if counts[lo] <= counts[hi] else hi
        if trimmed + counts[side] > allowed:
            break
        trimmed += counts[side]
        if side == lo:
            lo += 1
            while counts[lo] == 0:
                lo += 1
        else:
            hi -= 1
            while counts[hi] == 0:
                hi -= 1
    return lo, hi


def _linear_x_range(
    per_phase: dict[str, LayerStatsSnapshot],
    kind: str,
    clip_share: float = _BASE_CLIP_SHARE,
) -> list[float] | None:
    """X-axis range (linear value space) covering the trimmed bins.

    On a linear axis the bars span the full `[-1e6, 1e6]` edge range, almost
    all of it empty. We zoom to the edges of the trimmed bin span
    (`_trimmed_bin_bounds`, plus a little padding) so the bulk of the
    distribution stays legible. Returns `None` (Plotly autorange) when
    there's no data.
    """
    bounds = _trimmed_bin_bounds(per_phase, kind, clip_share)
    if bounds is None:
        return None
    lo = _HIST_EDGES[bounds[0]]
    hi = _HIST_EDGES[bounds[1] + 1]
    pad = (hi - lo) * 0.05 or 1.0
    return [lo - pad, hi + pad]


def _log_x_range(
    per_phase: dict[str, LayerStatsSnapshot],
    kind: str,
    clip_share: float = _BASE_CLIP_SHARE,
) -> list[float] | None:
    """X-axis range (bin-index space) covering the trimmed bins.

    The signed-log view draws bars at integer bin indices, so the range
    brackets the trimmed span (`_trimmed_bin_bounds`) with half-bar margins
    plus a little padding. Returns `None` (Plotly autorange — the full
    211-bin span) when there's no data.
    """
    bounds = _trimmed_bin_bounds(per_phase, kind, clip_share)
    if bounds is None:
        return None
    lo, hi = bounds
    pad = (hi - lo + 1) * 0.05
    return [lo - 0.5 - pad, hi + 0.5 + pad]


def _fill_fraction(
    per_phase: dict[str, LayerStatsSnapshot],
    kind: str,
    density: bool,
    bounds: tuple[int, int],
    y_top: float,
) -> float:
    """Share of the plot area the bars cover at the given axis ranges.

    Bar areas are measured in the units the bars are drawn in (value space
    for density mode, bin-index space for the signed-log view) with heights
    clipped to the y-range top; the plot area is the x-span times the
    y-range top, averaged over the drawn traces (every subplot row shares
    the same ranges).
    """
    if y_top <= 0:
        return 1.0
    lo, hi = bounds
    span = (_HIST_EDGES[hi + 1] - _HIST_EDGES[lo]) if density else (hi - lo + 1)
    phases = _phases_with_data(per_phase, kind)
    filled = 0.0
    for phase in phases:
        heights = _trace_heights(_kind_stats(per_phase[phase], kind).hist, density)
        for i in range(lo, hi + 1):
            width = _BIN_WIDTHS[i] if density else 1.0
            filled += width * min(heights[i], y_top)
    return filled / (span * y_top * len(phases))


def _axis_ranges(
    per_phase: dict[str, LayerStatsSnapshot],
    kind: str,
    *,
    log_x: bool,
    log_y: bool,
) -> tuple[list[float] | None, list[float] | None]:
    """The histogram figure's `(x, y)` axis ranges.

    Both ranges come from the same clip budget: bins/bars holding up to a
    share of the data points may be cut off (x: outermost tail bins,
    `_trimmed_bin_bounds`; y: tallest bars, `_linear_y_range`).

    A tall near-zero peak next to a long thin tail can leave the plot
    nearly empty even after the base trims — the cap chases the peak's
    narrow neighbours while the tail stretches the x-span. So while the
    bars would cover less than `_MIN_FILL_FRACTION` of the plot area, the
    clip share is raised in `_CLIP_SHARE_STEP` increments (clipping more of
    the peak and trimming more of the tail), stopping once the plot is at
    least that full or the share reaches `_MAX_CLIP_SHARE`.

    With **Log y** the y-range is `None` (autorange) and the x-trim sticks
    to the base share — the log scale keeps the bars visible, so the fill
    heuristic doesn't apply. Both ranges are `None` when there's no data.
    """
    density = _use_density(log_x)

    def x_range_at(share: float) -> list[float] | None:
        return (
            _log_x_range(per_phase, kind, share)
            if log_x
            else _linear_x_range(per_phase, kind, share)
        )

    if log_y:
        return x_range_at(_BASE_CLIP_SHARE), None
    share = _BASE_CLIP_SHARE
    while True:
        bounds = _trimmed_bin_bounds(per_phase, kind, share)
        y_range = _linear_y_range(per_phase, kind, density, share)
        if bounds is None or y_range is None:
            return None, None
        if (
            share >= _MAX_CLIP_SHARE
            or _fill_fraction(per_phase, kind, density, bounds, y_range[1])
            >= _MIN_FILL_FRACTION
        ):
            return x_range_at(share), y_range
        share = min(share + _CLIP_SHARE_STEP, _MAX_CLIP_SHARE)


def _make_histogram_figure(
    per_phase: dict[str, LayerStatsSnapshot],
    kind: str,
    title: str,
    *,
    log_x: bool = False,
    log_y: bool = False,
) -> go.Figure:
    """Plotly bar chart of the signed-log histogram, one subplot row per phase.

    `kind` selects which of the two histograms on each `LayerStatsSnapshot`
    to plot ("activation" or "gradient"). `per_phase` may be empty (initial
    render before any data has been collected) — the figure is still
    returned, just with no traces.

    Each phase draws in its own stacked subplot row (titled with the phase
    and epoch, tinted with the trace color) rather than overlaying bars on
    shared axes, so one phase never obscures another. The rows share the
    x-axis and, on a linear y-axis, the same capped y-range
    (see `_axis_ranges`), keeping the per-phase distributions directly
    comparable.

    `log_x` / `log_y` toggle the value (x) and probability (y) axes between a
    log-based and a linear scale (the "Log x" / "Log y" checkboxes on the
    Watching page — the checkbox alone decides the x-mode). With `log_x`
    off, bars show probability density instead of probabilities (see
    `_use_density`).

    This builds the *whole* figure. Routine data refreshes don't call it —
    they restyle the existing figure in place (see `_HistPlot`) so client-side
    state like zoom survives; the figure is only rebuilt when the set of
    phases or the axis scale changes.
    """
    x_values = list(range(N_BINS)) if log_x else _BIN_CENTERS
    density = _use_density(log_x)
    if density:
        hover = (
            "value %{x:.2e}<br>probability density %{y:.3g}"
            "<br>count %{customdata}<extra></extra>"
        )
    else:
        # Bars sit at bin indices on the signed-log axis; the hover shows
        # the bin's value (via customdata), not the index.
        hover = (
            "value ≈ %{customdata[1]}<br>probability %{y:.3g}"
            "<br>count %{customdata[0]}<extra></extra>"
        )
    phases = _phases_with_data(per_phase, kind)
    names = [f"{p} (ep {per_phase[p].epoch})" for p in phases]
    fig = make_subplots(
        rows=max(1, len(phases)),
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.06,
        subplot_titles=names or None,
    )
    for i, phase in enumerate(phases):
        stats = _kind_stats(per_phase[phase], kind)
        fig.add_trace(
            go.Bar(
                x=x_values,
                y=_trace_heights(stats.hist, density),
                customdata=_hover_customdata(stats.hist, density),
                width=None if log_x else _BIN_WIDTHS,
                name=names[i],
                marker_color=_phase_color(phase, i),
                opacity=0.85,
                hovertemplate=hover,
            ),
            row=i + 1,
            col=1,
        )
    # Subplot titles double as the legend: phase name + epoch in the trace
    # color, sitting right above the row they describe.
    for i, annotation in enumerate(fig.layout.annotations):
        annotation.update(font=dict(size=11, color=_phase_color(phases[i], i)))
    # `shared_xaxes` only hides the upper rows' tick labels; matching the
    # x-axes proper keeps every row in lock-step when zooming/panning and
    # lets a single `xaxis.range` relayout retarget all rows at once.
    for row in range(2, len(phases) + 1):
        fig.update_xaxes(matches="x", row=row, col=1)
    fig.update_layout(
        title=dict(text=title, x=0.0, font=dict(size=12)),
        bargap=0,
        margin=dict(l=50, r=20, t=40, b=40),
        height=_PLOT_HEIGHT * max(1, len(phases)),
        plot_bgcolor="#f8fafc",
        paper_bgcolor="white",
        showlegend=False,
    )
    x_range, y_range = _axis_ranges(per_phase, kind, log_x=log_x, log_y=log_y)
    if log_x:
        tick_vals, tick_text = _x_tick_layout()
        fig.update_xaxes(
            range=x_range,
            tickvals=tick_vals,
            ticktext=tick_text,
            tickfont=dict(size=9),
            showgrid=False,
            zeroline=False,
        )
    else:
        fig.update_xaxes(
            range=x_range,
            tickfont=dict(size=9),
            showgrid=False,
            zeroline=True,
            zerolinecolor="#cbd5e1",
        )
    fig.update_yaxes(
        type="log" if log_y else "linear",
        # The cap is a linear-space range; on a log y-axis `_axis_ranges`
        # returns `None` (Plotly autorange, which shows 100% of the data
        # anyway) since Plotly would misread the range as log10 units.
        range=y_range,
        showgrid=True,
        gridcolor="#e2e8f0",
        tickfont=dict(size=9),
        title=dict(
            text="probability density" if density else "probability",
            font=dict(size=10),
        ),
    )
    return fig


def _build_watch_page(
    session: Session,
    layer_names: list[str],
    *,
    input_mean: tuple[float, ...] | None = None,
    input_std: tuple[float, ...] | None = None,
) -> None:
    """The deep-dive page for watched layers.

    A header dropdown switches every layer card between two views, and a
    second dropdown picks which phase (train / val / …) the cards show —
    one phase at a time, in both views:

    - MIN/MAX (the default) — the extreme-activation patch grids
      (channels across, per-channel top samples down), one per patch
      type, each toggleable by its own checkbox, plus a heatmap checkbox
      that blends the stored activation maps over the patches.
    - HISTOGRAM — one plotly figure per tensor kind (activations and
      activation gradients) for the selected phase's latest epoch, plus
      the "Log x" / "Log y" axis-scale checkboxes.

    Each checkbox group is only visible while its view is selected. A
    `ui.timer` polls `session.watch_snapshot()` and refreshes the visible
    view in place. Layers can also be unwatched directly from the card
    header here, which drops the corresponding accumulator entry — the
    change is reflected on the main page on next navigation.
    """
    ui.page_title("PlayGrad — Watching")
    ui.query(".nicegui-content").classes("p-0 h-screen overflow-hidden")
    ui.query("body").classes("overflow-hidden")
    ui.query("html").classes("overflow-hidden")

    layer_panels: dict[str, _WatchLayerPanel] = {}
    count_label_holder: dict[str, ui.label] = {}
    body_container: ui.column
    # Whether the value (x) and probability (y) axes use a log-based scale.
    # Both default off — linear axes showing probability density (see
    # `_use_density`); the header checkboxes flip them and re-render every
    # plot immediately.
    axis_log = {"x": False, "y": False}
    # MIN/MAX view state (the default view): which of the four grids are
    # shown and whether the activation heatmap is blended over the patches.
    view_minmax = {"on": True}
    grid_on: dict[PatchType, bool] = dict.fromkeys(PATCH_TYPES, True)
    heat_on = {"on": False}
    # Every card shows one phase at a time, picked by a header dropdown
    # shared by both views; defaults to the schedule's first phase.
    phase_names = list(session.schedule.phases)
    selected_phase = {"name": phase_names[0] if phase_names else ""}

    async def set_axis_log(axis: str, value: bool) -> None:
        axis_log[axis] = value
        await refresh()

    async def set_mode(value: object) -> None:
        view_minmax["on"] = value == _VIEW_MINMAX
        hist_controls.set_visibility(not view_minmax["on"])
        minmax_controls.set_visibility(view_minmax["on"])
        await refresh()

    async def set_phase(value: object) -> None:
        selected_phase["name"] = str(value)
        await refresh()

    async def set_grid(ptype: PatchType, value: bool) -> None:
        grid_on[ptype] = value
        await refresh()

    async def set_heat(value: bool) -> None:
        heat_on["on"] = value
        await refresh()

    with ui.column().classes("w-full h-screen no-wrap gap-0"):
        with _top_bar_row():
            ui.button(
                icon="arrow_back",
                on_click=lambda: ui.navigate.to("/"),
                color="slate-500",
            ).props("dense size=md").tooltip("Back to the main page")
            ui.label("Watching").classes("font-mono text-base font-bold ml-2")
            count_label_holder["count"] = ui.label("").classes(
                "text-sm text-slate-500 ml-2"
            )
            ui.select(
                [_VIEW_MINMAX, _VIEW_HISTOGRAM],
                value=_VIEW_MINMAX,
                on_change=lambda e: set_mode(e.value),
            ).props("dense outlined options-dense").classes(
                "ml-4 text-sm"
            ).tooltip("What each layer card shows")
            ui.select(
                phase_names,
                value=selected_phase["name"],
                on_change=lambda e: set_phase(e.value),
            ).props("dense outlined options-dense").classes(
                "ml-2 text-sm"
            ).tooltip("Which phase the cards show")
            with ui.row().classes(
                "items-center gap-x-3 no-wrap"
            ) as hist_controls:
                ui.checkbox(
                    "Log x",
                    value=axis_log["x"],
                    on_change=lambda e: set_axis_log("x", bool(e.value)),
                ).props("dense").classes("text-sm ml-4").tooltip(
                    "Log-based (signed-log) scale on the value axis"
                )
                ui.checkbox(
                    "Log y",
                    value=axis_log["y"],
                    on_change=lambda e: set_axis_log("y", bool(e.value)),
                ).props("dense").classes("text-sm").tooltip(
                    "Log scale on the probability axis"
                )
            with ui.row().classes(
                "items-center gap-x-3 no-wrap"
            ) as minmax_controls:
                for i, ptype in enumerate(PATCH_TYPES):
                    ui.checkbox(
                        _PATCH_TYPE_LABELS[ptype],
                        value=grid_on[ptype],
                        on_change=lambda e, p=ptype: set_grid(p, bool(e.value)),
                    ).props("dense").classes(
                        "text-sm ml-4" if i == 0 else "text-sm"
                    ).tooltip(
                        f"Show the {_PATCH_TYPE_LABELS[ptype].lower()} grid"
                    )
                ui.checkbox(
                    "Enable heatmap",
                    value=heat_on["on"],
                    on_change=lambda e: set_heat(bool(e.value)),
                ).props("dense").classes("text-sm").tooltip(
                    "Blend each channel's activation strength over the "
                    "patches (red positive, blue negative), with a scale "
                    "next to each grid"
                )
            hist_controls.set_visibility(False)
            ui.button(
                icon="refresh",
                on_click=lambda: refresh(),
                color="slate-500",
            ).classes("ml-auto").props("dense size=md flat").tooltip("Refresh now")

        body_container = ui.column().classes(
            "w-full grow min-h-0 overflow-auto p-4 gap-3 bg-slate-200"
        )

    def rebuild_cards() -> None:
        layer_panels.clear()
        body_container.clear()
        watched = session.watched_layers
        with body_container:
            if not watched:
                with ui.column().classes("items-center gap-2 py-12 w-full"):
                    ui.icon("visibility_off", size="lg").classes("text-slate-400")
                    ui.label("No layers selected.").classes("text-slate-600")
                    ui.label(
                        "Go back and click the eye icon on a layer card "
                        "to start watching."
                    ).classes("text-slate-500 text-sm")
                return
            for name in layer_names:
                if name not in watched:
                    continue
                layer_panels[name] = _WatchLayerPanel(
                    name=name,
                    session=session,
                    on_unwatched=rebuild_cards,
                    axis_log=axis_log,
                    view_minmax=view_minmax,
                    grid_on=grid_on,
                    heat_on=heat_on,
                    selected_phase=selected_phase,
                    input_mean=input_mean,
                    input_std=input_std,
                )

    # Single-flight refresh: snapshotting and grid rendering run in a worker
    # thread so the event loop keeps serving websocket traffic (a blocked
    # loop starves keepalive pings and kills the connection). A toggle that
    # lands while a pass is in flight just marks it dirty — rapid Heatmap
    # clicks coalesce into one follow-up pass instead of queueing a full
    # re-render per click.
    refresh_state = {"running": False, "dirty": False}

    async def refresh() -> None:
        if refresh_state["running"]:
            refresh_state["dirty"] = True
            return
        refresh_state["running"] = True
        try:
            while True:
                refresh_state["dirty"] = False
                watched = session.watched_layers
                n = len(watched)
                count_label_holder["count"].text = (
                    f"{n} layer{'' if n == 1 else 's'}"
                )
                if set(layer_panels) != set(watched):
                    rebuild_cards()
                panels = dict(layer_panels)
                minmax = view_minmax["on"]

                def compute(
                    panels: dict[str, _WatchLayerPanel] = panels,
                    minmax: bool = minmax,
                ) -> tuple[
                    WatchSnapshot,
                    dict[str, tuple[tuple[object, ...], str] | None],
                ]:
                    # The GPU→CPU patch sync is only paid when the MIN/MAX
                    # view will actually consume it.
                    snap = session.watch_snapshot(include_patches=minmax)
                    grids: dict[str, tuple[tuple[object, ...], str] | None] = {}
                    if minmax:
                        for name, panel in panels.items():
                            grids[name] = panel.prepare_grids(snap)
                    return snap, grids

                snap, grids = await asyncio.to_thread(compute)
                for name, panel in panels.items():
                    if layer_panels.get(name) is panel:  # not rebuilt meanwhile
                        panel.update(snap, grids.get(name))
                if not refresh_state["dirty"]:
                    return
        finally:
            refresh_state["running"] = False

    ui.timer(0.0, refresh, once=True)
    ui.timer(2.0, refresh)


def _plotly_restyle(
    plot: ui.plotly,
    update: dict[str, object],
    indices: list[int],
    layout: dict[str, object] | None = None,
) -> None:
    """Update trace attributes of an existing figure in place.

    Calls `Plotly.update` on the live graph div instead of replacing the
    figure, so client-side state (legend visibility toggles, zoom/pan) is left
    untouched. `update` maps each attribute to a list of per-trace values
    (e.g. `{"y": [y0, y1]}`) aligned with `indices`. `layout` optionally
    carries relayout-style updates (e.g. `{"yaxis.range": [0, 5]}`) applied in
    the same call; note an explicit axis-range write does reset any user zoom
    on that axis.

    The guard makes this a no-op until Plotly's module has loaded and drawn the
    graph (a just-connected client can fire a timer tick before then); the next
    refresh re-applies the data, so nothing is lost.
    """
    ui.run_javascript(
        f"const gd = getHtmlElement({plot.id}); "
        f"if (gd && gd.data && window.Plotly) "
        f"window.Plotly.update(gd, {json.dumps(update)}, "
        f"{json.dumps(layout or {})}, {json.dumps(indices)});"
    )


# Plotly config shared by all watch histograms. "Autoscale" would expand the
# axes to fit every bar — including the freak spikes and outlier tails the
# capped ranges deliberately clip — landing on a different scale than the
# initial render, so the button is removed; "Reset axes" and double-click
# restore the ranges the figure was built with instead.
_PLOTLY_CONFIG: dict[str, object] = {
    "modeBarButtonsToRemove": ["autoScale2d"],
    "doubleClick": "reset",
}


def _figure_payload(fig: go.Figure) -> dict[str, object]:
    """The data/layout/config dict NiceGUI hands to `Plotly.react`."""
    return {**fig.to_plotly_json(), "config": _PLOTLY_CONFIG}


class _HistPlot:
    """One Plotly histogram figure that refreshes its data in place.

    The figure (one subplot row per phase) is built once and rebuilt only
    when the set of phases or the axis scale
    changes. Routine per-tick updates go through `Plotly.update`, which
    leaves client-side state — zoom/pan — untouched.
    """

    def __init__(self, kind: str, title: str, axis_log: dict[str, bool]) -> None:
        self._kind = kind
        self._title = title
        self._axis_log = axis_log
        # Signature of what's currently drawn, so `update` can tell a plain
        # data refresh (restyle) from a structural change (rebuild).
        self._phases: list[str] = []
        self._axis = self._current_axis()
        # Last axis ranges applied, so refreshes only push a relayout when a
        # cap actually moved (a range write resets zoom on that axis).
        self._y_range: list[float] | None = None
        self._x_range: list[float] | None = None
        self.element = ui.plotly(
            _figure_payload(
                _make_histogram_figure(
                    {}, kind, title, log_x=self._axis[0], log_y=self._axis[1]
                )
            )
        ).classes("w-full")

    def _current_axis(self) -> tuple[bool, bool]:
        return self._axis_log["x"], self._axis_log["y"]

    def update(self, per_phase: dict[str, LayerStatsSnapshot]) -> None:
        phases = _phases_with_data(per_phase, self._kind)
        axis = self._current_axis()
        log_x = axis[0]
        density = _use_density(log_x)
        if phases != self._phases or axis != self._axis:
            # A phase appeared/disappeared or an axis-scale checkbox
            # flipped — rebuild the whole figure.
            self.element.update_figure(
                _figure_payload(
                    _make_histogram_figure(
                        per_phase,
                        self._kind,
                        self._title,
                        log_x=axis[0],
                        log_y=axis[1],
                    )
                )
            )
            self._phases = phases
            self._axis = axis
            self._x_range, self._y_range = _axis_ranges(
                per_phase, self._kind, log_x=log_x, log_y=axis[1]
            )
        elif phases:
            # Same rows and axes — only counts (and the epoch label) moved.
            # Restyle in place so zoom/pan survives.
            hists = [_kind_stats(per_phase[p], self._kind).hist for p in phases]
            names = [f"{p} (ep {per_phase[p].epoch})" for p in phases]
            update: dict[str, object] = {
                "name": names,
                "y": [_trace_heights(h, density) for h in hists],
                "customdata": [_hover_customdata(h, density) for h in hists],
            }
            # The subplot titles carry the epoch, so refresh them with the
            # data (annotation order matches row order).
            layout: dict[str, object] = {
                f"annotations[{i}].text": name for i, name in enumerate(names)
            }
            # The caps follow the data; re-apply them only when they moved so
            # an idle refresh doesn't keep snapping the user's zoom back.
            x_range, y_range = _axis_ranges(
                per_phase, self._kind, log_x=log_x, log_y=axis[1]
            )
            if y_range is not None and y_range != self._y_range:
                # Every row gets the same range (row 1 is "yaxis", row n is
                # "yaxis{n}") so the subplots stay comparable.
                self._y_range = y_range
                for i in range(len(phases)):
                    axis_name = "yaxis" if i == 0 else f"yaxis{i + 1}"
                    layout[f"{axis_name}.range"] = y_range
            if x_range != self._x_range:
                # The rows' x-axes are matched, so one key updates every row.
                self._x_range = x_range
                layout["xaxis.range"] = x_range
            _plotly_restyle(
                self.element, update, list(range(len(phases))), layout
            )


class _WatchLayerPanel:
    """One card per watched layer — histograms or extreme-patch grids.

    Both views are built up front; `update` shows the one matching the
    header dropdown and only refreshes that view's content (hidden plotly
    figures and patch grids are left untouched until switched back). Patch
    grids re-render only when their cheap signature — toggles plus the
    stored extreme values — actually changes, so idle 2s refreshes don't
    re-encode images.
    """

    def __init__(
        self,
        *,
        name: str,
        session: Session,
        on_unwatched: Callable[[], None],
        axis_log: dict[str, bool],
        view_minmax: dict[str, bool],
        grid_on: dict[PatchType, bool],
        heat_on: dict[str, bool],
        selected_phase: dict[str, str],
        input_mean: tuple[float, ...] | None,
        input_std: tuple[float, ...] | None,
    ) -> None:
        self.name = name
        self._session = session
        self._view_minmax = view_minmax
        self._grid_on = grid_on
        self._heat_on = heat_on
        self._selected_phase = selected_phase
        self._input_mean = input_mean
        self._input_std = input_std
        self._grid_sig: tuple[object, ...] | None = None
        with ui.card().classes("w-full p-4 gap-2"):
            with ui.row().classes("w-full items-center gap-2 no-wrap"):
                ui.label(name).classes("font-mono text-base font-bold grow")
                ui.button(
                    icon="visibility_off",
                    color="amber-600",
                    on_click=lambda: (session.unwatch(name), on_unwatched()),
                ).props("dense size=sm flat round").tooltip("Stop watching")
            self._hist_section = ui.column().classes("w-full gap-3")
            with self._hist_section:
                ui.label("Activations").classes(
                    "font-mono text-sm text-slate-600"
                )
                self._act_stats = ui.html(
                    _stats_table_html({}, "activation")
                ).classes("font-mono text-sm")
                self._act = _HistPlot("activation", "activations", axis_log)
                ui.label("Gradients").classes(
                    "font-mono text-sm text-slate-600"
                )
                self._grad_stats = ui.html(
                    _stats_table_html({}, "gradient")
                ).classes("font-mono text-sm")
                self._grad = _HistPlot("gradient", "gradients", axis_log)
            self._patch_section = ui.column().classes("w-full gap-2")
            with self._patch_section:
                self._grids = ui.html(_NO_PATCHES_HTML).classes("w-full")
            self._hist_section.set_visibility(not view_minmax["on"])
            self._patch_section.set_visibility(view_minmax["on"])

    def update(
        self,
        snap: WatchSnapshot,
        grids: tuple[tuple[object, ...], str] | None = None,
    ) -> None:
        """Refresh the visible view. Runs on the UI event loop.

        `grids` is the output of `prepare_grids` (computed off the event
        loop by the page's refresh); `None` means the grids are unchanged.
        """
        per_phase = self._phase_view(snap)
        minmax = self._view_minmax["on"]
        self._hist_section.set_visibility(not minmax)
        self._patch_section.set_visibility(minmax)
        if minmax:
            if grids is not None:
                self._grid_sig, html = grids
                self._grids.set_content(html)
            return
        self._act_stats.set_content(_stats_table_html(per_phase, "activation"))
        self._grad_stats.set_content(_stats_table_html(per_phase, "gradient"))
        self._act.update(per_phase)
        self._grad.update(per_phase)

    def _phase_view(self, snap: WatchSnapshot) -> dict[str, LayerStatsSnapshot]:
        """The layer's latest-epoch stats, narrowed to the selected phase."""
        return _filter_phase(
            snap.latest_per_phase(self.name), self._selected_phase["name"]
        )

    def prepare_grids(
        self, snap: WatchSnapshot
    ) -> tuple[tuple[object, ...], str] | None:
        """Render this panel's patch grids if their signature changed.

        Pure compute — no UI element access — so the page's refresh can run
        it in a worker thread: blending heatmaps and encoding a card's worth
        of grid images would otherwise block the event loop long enough to
        starve websocket keepalive pings when toggles re-render every card.
        Returns `(signature, html)` for `update` to apply, or `None` when
        the current content is already up to date.
        """
        per_phase = self._phase_view(snap)
        enabled = [t for t in PATCH_TYPES if self._grid_on[t]]
        heatmap = self._heat_on["on"]
        sig = _patch_grids_signature(per_phase, enabled, heatmap)
        if sig == self._grid_sig:
            return None
        html = _patch_grids_html(
            per_phase,
            enabled=enabled,
            heatmap=heatmap,
            mean=self._input_mean,
            std=self._input_std,
        )
        return sig, html


def _filter_phase(
    per_phase: dict[str, LayerStatsSnapshot], phase: str
) -> dict[str, LayerStatsSnapshot]:
    """Narrow a `phase -> stats` mapping to the dropdown-selected phase."""
    return {p: s for p, s in per_phase.items() if p == phase}


_VIEW_HISTOGRAM: str = "HISTOGRAM"
_VIEW_MINMAX: str = "MIN/MAX"

_PATCH_TYPE_LABELS: dict[PatchType, str] = {
    "max_pixel": "Max pixel",
    "min_pixel": "Min pixel",
    "max_average": "Max average",
    "min_average": "Min average",
}

_NO_PATCHES_HTML: str = (
    '<div class="text-xs text-slate-400 italic py-1">no patches gathered '
    "yet — patches need an image-like (4D) model input</div>"
)


def _patch_grids_signature(
    per_phase: dict[str, LayerStatsSnapshot],
    enabled: list[PatchType],
    heatmap: bool,
) -> tuple[object, ...]:
    """Cheap change-detection key for a panel's patch grids.

    The stored extreme values identify the buffer contents: any accepted
    candidate changes its channel's value row, so unchanged values ⇒
    unchanged patches (within one epoch bucket, which the phase/epoch part
    of the key pins down).
    """
    parts: list[object] = [tuple(enabled), heatmap]
    for phase, layer_snap in per_phase.items():
        patches = layer_snap.patches
        if patches is None:
            parts.append((phase, layer_snap.epoch, None))
            continue
        for ptype in enabled:
            tp = patches.by_type.get(ptype)
            values = tp.values.numpy().tobytes() if tp is not None else None
            parts.append((phase, layer_snap.epoch, ptype, values))
    return tuple(parts)


def _patch_grids_html(
    per_phase: dict[str, LayerStatsSnapshot],
    *,
    enabled: list[PatchType],
    heatmap: bool,
    mean: tuple[float, ...] | None,
    std: tuple[float, ...] | None,
) -> str:
    """The MIN/MAX view body for one layer: per-phase blocks of grids."""
    blocks: list[str] = []
    for i, (phase, layer_snap) in enumerate(per_phase.items()):
        patches = layer_snap.patches
        if patches is None:
            continue
        rows: list[str] = []
        for ptype in enabled:
            tp = patches.by_type.get(ptype)
            if tp is None:
                continue
            grid = render_patch_grid(tp, mean=mean, std=std, heatmap=heatmap)
            if grid is None:
                continue
            rows.append(_patch_grid_row_html(_PATCH_TYPE_LABELS[ptype], grid))
        if not rows:
            continue
        color = _phase_color(phase, i)
        blocks.append(
            '<div class="flex flex-col gap-2 w-full">'
            f'<div class="font-mono text-xs font-bold" style="color:{color}">'
            f"{phase} (ep {layer_snap.epoch})</div>" + "".join(rows) + "</div>"
        )
    if not blocks:
        return _NO_PATCHES_HTML
    return '<div class="flex flex-col gap-4 w-full">' + "".join(blocks) + "</div>"


def _patch_grid_row_html(label: str, grid: PatchGridRender) -> str:
    """One labeled grid: channels as columns, top samples as rows.

    With the heatmap enabled the grid is flanked by its crisp
    display-resolution colorbar (the overlay's `±vmax` scale), which sits
    outside the scroll container so it stays visible on wide grids.
    `max-width:none` opts the images out of the preflight `max-width:100%`
    so wide grids scroll horizontally instead of being squashed.
    """
    legend = (
        f'<img src="{_b64_img_src(grid.heat_legend)}" '
        'style="display:block; flex:none; max-width:none;" />'
        if grid.heat_legend is not None
        else ""
    )
    return (
        '<div class="flex flex-col gap-0.5 w-full">'
        '<div class="text-[10px] uppercase tracking-wide text-slate-500 '
        f'font-mono">{label}</div>'
        '<div style="display:flex; align-items:flex-start;" class="w-full">'
        f"{legend}"
        '<div class="overflow-x-auto" style="flex:1; min-width:0;">'
        f'<img src="{_b64_img_src(grid.image, mime=grid.mime)}" '
        f'style="width:{grid.width}px; height:{grid.height}px; '
        'image-rendering:pixelated; display:block; max-width:none;" '
        f'title="{label} — columns: channels, rows: top samples '
        '(best first)" />'
        "</div></div></div>"
    )


_ROLE_LABELS: dict[str, str] = {"x": "X", "y": "Y", "tile": "Tile", "index": "Index"}


def _role_options(ndim: int) -> dict[str, str]:
    """Role choices offered per dimension, scaled to the weight's rank.

    A rank-1 weight can only map an axis to X; rank-2 adds Y; rank-3+ adds the
    tiling axis. Every rank can pin an axis to a single index.
    """
    roles = ["x"]
    if ndim >= 2:
        roles.append("y")
    if ndim >= 3:
        roles.append("tile")
    roles.append("index")
    return {r: _ROLE_LABELS[r] for r in roles}


def _dims_from_roles(roles: list[str]) -> tuple[int | None, int | None, int | None]:
    """Resolve a per-dimension role list to (x_dim, y_dim, tile_dim) axes."""
    x = y = tile = None
    for d, role in enumerate(roles):
        if role == "x":
            x = d
        elif role == "y":
            y = d
        elif role == "tile":
            tile = d
    return x, y, tile


def _default_roles(ndim: int) -> list[str]:
    """Per-dimension role list matching `render.default_weight_dims`."""
    dims = default_weight_dims(ndim)
    roles = ["index"] * ndim
    roles[dims.x_dim] = "x"
    if dims.y_dim is not None:
        roles[dims.y_dim] = "y"
    if dims.tile_dim is not None:
        roles[dims.tile_dim] = "tile"
    return roles


def _build_weights_page(session: Session, layer: str) -> None:
    """Per-layer weight viewer: kernel/image strips with selectable axes.

    Reuses the main page's stepping controls (minus the sample spinner — a
    weight has no batch axis) so the displayed weights track the same paused
    batch. One panel per parameter the layer owns; each panel lets the user
    remap which tensor axes become the X / Y / tiling axes and pins the rest
    by index.
    """
    title = f"Weights · {layer}" if layer else "Weights"
    ui.page_title(f"PlayGrad — {title}")
    ui.query(".nicegui-content").classes("p-0 h-screen overflow-hidden")
    ui.query("body").classes("overflow-hidden")
    ui.query("html").classes("overflow-hidden")
    ui.add_head_html(_STRIP_MARKER_CSS)

    weight_names = session.layer_weights.get(layer, [])
    shapes = {
        name: tuple(p.shape)
        for name, p in session.model.named_parameters()
        if name in set(weight_names)
    }
    step_until_custom = _build_step_until_custom_dialog(session)
    panels: list[_WeightPanel] = []

    with ui.column().classes("w-full h-screen no-wrap gap-0"):
        with _top_bar_row():
            ui.button(
                icon="arrow_back",
                on_click=lambda: ui.navigate.to("/"),
                color="slate-500",
            ).props("dense size=md").tooltip("Back to the main page")
            ui.label(title).classes(
                "font-mono text-base font-bold ml-2 truncate max-w-64"
            )
            position_label = _add_step_controls(session, step_until_custom)
            ui.button(
                icon="refresh",
                on_click=lambda: do_refresh(),
                color="slate-500",
            ).classes("ml-auto").props("dense size=md flat").tooltip(
                "Show the model's current weights (works while training)"
            )

        with ui.column().classes(
            "w-full grow min-h-0 overflow-auto p-4 gap-4 bg-slate-200"
        ):
            if layer not in session.layer_names:
                _weights_placeholder(f"Unknown layer {layer!r}.")
            elif not weight_names:
                _weights_placeholder(
                    f"Layer {layer!r} has no weights to show."
                )
            else:
                for name in weight_names:
                    panels.append(
                        _WeightPanel(name=name, shape=shapes[name], session=session)
                    )

    def do_refresh() -> None:
        # Read the model's live parameters instead of the last snapshot, so the
        # weights update even mid-training (detach / run modes never publish a
        # snapshot). The live view then persists until the next captured batch.
        weights = session.current_weights()
        gradients = session.current_weight_gradients()
        optimizer_state = session.current_optimizer_state()
        optimizer_hyperparams = session.current_optimizer_hyperparams()
        for panel in panels:
            panel.show_weights(
                weights,
                gradients,
                optimizer_state=optimizer_state,
                optimizer_hyperparams=optimizer_hyperparams,
            )

    def tick() -> None:
        live = session.live_position
        if live is not None:
            position_label.text = _format_live_position(live)
        snap = session.snapshot
        if snap is None:
            return
        for panel in panels:
            panel.maybe_render(snap)

    ui.timer(0.2, tick)


def _weights_placeholder(message: str) -> None:
    with ui.column().classes("items-center gap-2 py-12 w-full"):
        ui.icon("grid_off", size="lg").classes("text-slate-400")
        ui.label(message).classes("text-slate-600")


# Shown next to the GRADIENT marker before any backward pass has populated
# the parameter's gradient.
_NO_GRADIENT_HTML: str = (
    '<div class="text-xs text-slate-400 italic py-1">no gradient captured yet</div>'
)

# The marker's vertical label is hidden on strips too short to fit it
# (1D heatmap rows, the no-gradient placeholder); the tallest label is
# ~75px, so anything under 88px can't show it cleanly. The 128px conv
# tiles clear the threshold comfortably.
_STRIP_MARKER_CSS: str = """
<style>
  .playgrad-marker { container-type: size; }
  @container (max-height: 88px) {
    .playgrad-marker-label { display: none; }
  }
</style>
"""


class _WeightPanel:
    """One card per parameter — an axis-remappable kernel/image strip.

    The weight's rank is fixed, so the per-dimension role selects and index
    spinners are built once. Changing a role auto-demotes whichever other axis
    held that role (roles X/Y/Tile stay unique), then re-renders against the
    last snapshot; new snapshots re-render via `maybe_render`.
    """

    def __init__(self, *, name: str, shape: tuple[int, ...], session: Session) -> None:
        self.name = name
        self._shape = shape
        self._session = session
        self._ndim = len(shape)
        self._roles: list[str] = _default_roles(self._ndim)
        self._indices: dict[int, int] = {d: 0 for d in range(self._ndim)}
        self._last_snapshot: BatchSnapshot | None = None
        self._weights: dict[str, Tensor] | None = None
        self._gradients: dict[str, Tensor] | None = None
        self._opt_state: dict[str, dict[str, Tensor]] = {}
        self._opt_hparams: dict[str, dict[str, float]] = {}
        self._role_selects: list[ui.select] = []
        self._index_numbers: dict[int, ui.number] = {}

        options = _role_options(self._ndim)
        with ui.card().classes("w-full p-4 gap-3"):
            with ui.row().classes("w-full items-baseline gap-3 no-wrap"):
                ui.label(name).classes("font-mono text-base font-bold")
                ui.label(f"shape {tuple(shape)}").classes(
                    "font-mono text-xs text-slate-500"
                )
            with ui.row().classes("items-end gap-4 flex-wrap"):
                for d in range(self._ndim):
                    with ui.column().classes("gap-1"):
                        ui.label(f"dim {d} · {shape[d]}").classes(
                            "text-xs text-slate-500 font-mono"
                        )
                        with ui.row().classes("items-center gap-1 no-wrap"):
                            select = ui.select(
                                options=options,
                                value=self._roles[d],
                                on_change=lambda e, d=d: self._on_role(
                                    d, getattr(e, "value", None)
                                ),
                            ).props("dense outlined").classes("w-24")
                            self._role_selects.append(select)
                            number = ui.number(
                                value=0,
                                min=0,
                                max=shape[d] - 1,
                                step=1,
                                format="%d",
                                on_change=lambda e, d=d: self._on_index(
                                    d, getattr(e, "value", None)
                                ),
                            ).props("dense outlined").classes("w-20")
                            self._index_numbers[d] = number
            self._error = ui.label("").classes("text-amber-700 text-xs min-h-4")
            # Both strips share one horizontal scrollbar so they pan together,
            # and carry the same kind of labelled marker bars as the
            # activation/gradient pair on the main page's layer cards.
            with ui.element("div").classes("w-full overflow-x-auto"):
                # Same max-content wrapper as the layer cards: every row spans
                # the widest strip so the sticky markers stay in view across
                # the whole scroll range.
                with ui.element("div").classes("w-max min-w-full"):
                    with ui.element("div").classes("flex no-wrap items-stretch"):
                        _strip_marker("bg-sky-500", "WEIGHT")
                        self._img = ui.html("")
                    ui.element("div").classes("h-1")
                    with ui.element("div").classes("flex no-wrap items-stretch"):
                        _strip_marker("bg-violet-500", "GRADIENT")
                        self._grad_img = ui.html("")
                    # One marker-barred strip per tensor-valued optimizer
                    # state entry (momentum_buffer, exp_avg, …); rebuilt on
                    # each render. Stays empty — invisible — when the session
                    # has no optimizer.
                    self._opt_container = ui.element("div").classes("w-full")
            # Scalar optimizer values: 0-dim state entries (Adam's `step`) and
            # the param group's numeric hyperparameters (`lr`, …).
            self._opt_scalars = ui.label("").classes(
                "text-xs text-slate-500 font-mono"
            )
            self._opt_scalars.set_visibility(False)
        self._sync_index_visibility()

    def _on_role(self, dim: int, value: object) -> None:
        role = str(value) if value is not None else "index"
        if role in ("x", "y", "tile"):
            for other in range(self._ndim):
                if other != dim and self._roles[other] == role:
                    self._roles[other] = "index"
        self._roles[dim] = role
        # Writes to widget `.value` made from inside a value-change handler are
        # suppressed by NiceGUI; defer the select/visibility sync one loop tick
        # so demotions actually reach the client.
        ui.timer(0.0, self._apply_control_state, once=True)
        self._render_current()

    def _on_index(self, dim: int, value: float | None) -> None:
        idx = int(value) if value is not None else 0
        self._indices[dim] = max(0, min(idx, self._shape[dim] - 1))
        self._render_current()

    def _apply_control_state(self) -> None:
        for d, select in enumerate(self._role_selects):
            select.value = self._roles[d]
        self._sync_index_visibility()

    def _sync_index_visibility(self) -> None:
        for d, number in self._index_numbers.items():
            number.set_visibility(self._roles[d] == "index")

    def maybe_render(self, snap: BatchSnapshot) -> None:
        """Render snapshot weights, but only when the snapshot is new.

        A manual refresh (`show_weights` with live weights) leaves
        `_last_snapshot` untouched, so the live view persists until the next
        captured batch publishes a genuinely fresh snapshot.
        """
        if snap is self._last_snapshot:
            return
        self._last_snapshot = snap
        self.show_weights(
            snap.weights,
            snap.weight_gradients,
            optimizer_state=snap.optimizer_state,
            optimizer_hyperparams=snap.optimizer_hyperparams,
        )

    def show_weights(
        self,
        weights: dict[str, Tensor],
        gradients: dict[str, Tensor],
        *,
        optimizer_state: dict[str, dict[str, Tensor]],
        optimizer_hyperparams: dict[str, dict[str, float]],
    ) -> None:
        """Display weight, gradient, and optimizer values (snapshot or live)."""
        self._weights = weights
        self._gradients = gradients
        self._opt_state = optimizer_state
        self._opt_hparams = optimizer_hyperparams
        self._render()

    def _render_current(self) -> None:
        if self._weights is not None:
            self._render()

    def _render(self) -> None:
        tensor = self._weights.get(self.name) if self._weights is not None else None
        if tensor is None:
            self._show_error("no weights captured yet")
            return
        x_dim, y_dim, tile_dim = _dims_from_roles(self._roles)
        if x_dim is None:
            self._show_error("select an X dimension")
            return
        # A tiling axis only makes sense once a Y axis exists.
        tile = tile_dim if y_dim is not None else None
        fixed = {
            d: self._indices.get(d, 0)
            for d in range(self._ndim)
            if self._roles[d] == "index"
        }
        strip = render_weight(
            tensor, x_dim=x_dim, y_dim=y_dim, tile_dim=tile, fixed=fixed
        )
        if strip is None:
            self._show_error("invalid axis selection")
            return
        self._error.text = ""
        self._img.set_content(_strip_html(strip))
        # The gradient shares the weight's shape, so the same axis layout
        # applies; it's simply absent before the first backward pass.
        grad = self._gradients.get(self.name) if self._gradients is not None else None
        grad_strip = (
            render_weight(grad, x_dim=x_dim, y_dim=y_dim, tile_dim=tile, fixed=fixed)
            if grad is not None
            else None
        )
        self._grad_img.set_content(
            _strip_html(grad_strip) if grad_strip is not None else _NO_GRADIENT_HTML
        )
        self._render_optimizer_values(x_dim=x_dim, y_dim=y_dim, tile=tile, fixed=fixed)

    def _render_optimizer_values(
        self,
        *,
        x_dim: int,
        y_dim: int | None,
        tile: int | None,
        fixed: dict[int, int],
    ) -> None:
        """Rebuild the optimizer strips + scalar line below the gradient.

        Tensor state entries matching the weight's shape (momentum buffers,
        Adam moments) reuse the panel's axis layout; differently-shaped ones
        (e.g. factored second moments) fall back to their own rank's default
        axes. 0-dim entries (Adam's `step`) join the group hyperparameters
        (`lr`, …) on a single scalar line. With no optimizer attached both
        stay empty, leaving the panel exactly as before.
        """
        entries = dict(sorted(self._opt_state.get(self.name, {}).items()))
        self._opt_container.clear()
        scalar_parts: list[str] = []
        with self._opt_container:
            for key, tensor in entries.items():
                if tensor.ndim == 0:
                    scalar_parts.append(f"{key} = {_format_stat(float(tensor))}")
                    continue
                if tuple(tensor.shape) == self._shape:
                    strip = render_weight(
                        tensor, x_dim=x_dim, y_dim=y_dim, tile_dim=tile, fixed=fixed
                    )
                else:
                    dims = default_weight_dims(tensor.ndim)
                    strip = render_weight(
                        tensor,
                        x_dim=dims.x_dim,
                        y_dim=dims.y_dim,
                        tile_dim=dims.tile_dim,
                        fixed={d: 0 for d in dims.fixed_dims},
                    )
                if strip is None:
                    continue
                ui.element("div").classes("h-1")
                with ui.element("div").classes("flex no-wrap items-stretch"):
                    _strip_marker("bg-amber-600", key.upper())
                    ui.html(_strip_html(strip))
        scalar_parts += [
            f"{key} = {_format_stat(value)}"
            for key, value in sorted(self._opt_hparams.get(self.name, {}).items())
        ]
        self._opt_scalars.text = "  ·  ".join(scalar_parts)
        self._opt_scalars.set_visibility(bool(scalar_parts))

    def _show_error(self, message: str) -> None:
        self._error.text = message
        self._img.set_content("")
        self._grad_img.set_content("")
        self._opt_container.clear()
        self._opt_scalars.set_visibility(False)


@dataclass(frozen=True)
class _ExperimentParam:
    """One configurable knob of an experiment, rendered as a form widget."""

    key: str
    label: str
    kind: str  # "int" | "float" | "bool" | "select"
    default: object
    options: dict[str, str] | None = None
    minimum: float | None = None
    step: float | None = None
    tooltip: str = ""


_CHANNEL_PARAM = _ExperimentParam(
    "channel",
    "Channel (-1 = whole layer)",
    "int",
    0,
    minimum=-1,
    tooltip="Which channel / feature of this layer to target",
)
_SAMPLE_PARAM = _ExperimentParam(
    "sample",
    "Sample",
    "int",
    0,
    minimum=0,
    tooltip="Batch sample index of the input to work on",
)
_TARGET_PARAM = _ExperimentParam(
    "target",
    "Target class (-1 = argmax)",
    "int",
    -1,
    minimum=-1,
    tooltip="Class index the attribution explains; -1 uses the model's prediction",
)

_EXPERIMENT_PARAMS: dict[str, list[_ExperimentParam]] = {
    "deep_dream": [
        _CHANNEL_PARAM,
        _ExperimentParam("steps", "Steps", "int", 100, minimum=1),
        _ExperimentParam(
            "lr", "Learning rate", "float", 0.05, minimum=0, step=0.01
        ),
        _ExperimentParam(
            "diffusion",
            "Diffusion",
            "float",
            0.05,
            minimum=0,
            step=0.01,
            tooltip="Per-step blend with a 3×3 blur; damps high-frequency noise",
        ),
        _ExperimentParam(
            "jitter",
            "Jitter (px)",
            "int",
            2,
            minimum=0,
            tooltip=(
                "Random shift each step, undone after the update; "
                "reduces pixel-grid artifacts"
            ),
        ),
        _ExperimentParam(
            "zoom",
            "Zoom multiplier per step",
            "float",
            1.0,
            minimum=1,
            step=0.01,
            tooltip=(
                "Per-step center zoom-in factor (1 = no zoom; on small "
                "inputs it only takes effect above ~1 + 1/size)"
            ),
        ),
        _ExperimentParam(
            "batch",
            "Inputs",
            "int",
            1,
            minimum=1,
            tooltip=(
                "How many inputs to dream on; defaults to the size of the "
                f"currently processed batch, capped at {_DEFAULT_DREAM_BATCH}"
            ),
        ),
        _ExperimentParam(
            "start",
            "Start from",
            "select",
            "noise",
            options={"noise": "Noise", "sample": "Current batch"},
            tooltip=(
                "Noise draws fresh inputs shaped and scaled like the "
                "network's real input — different on every Run; Current "
                "batch starts from the real input batch itself"
            ),
        ),
        _ExperimentParam(
            "clamp",
            "Clamp to displayable range",
            "bool",
            True,
            tooltip=(
                "Keep pixels inside the [0, 1] display range mapped through "
                "the input mean/std"
            ),
        ),
    ],
    "gradcam": [_TARGET_PARAM, _SAMPLE_PARAM],
    "neuron_gradient": [_CHANNEL_PARAM, _SAMPLE_PARAM],
    "neuron_ig": [
        _CHANNEL_PARAM,
        _ExperimentParam("steps", "Integration steps", "int", 32, minimum=2),
        _SAMPLE_PARAM,
    ],
    "occlusion": [
        _TARGET_PARAM,
        _ExperimentParam(
            "window",
            "Window (px)",
            "int",
            4,
            minimum=1,
            tooltip="Side length of the occluding patch",
        ),
        _ExperimentParam("stride", "Stride (px)", "int", 2, minimum=1),
        _SAMPLE_PARAM,
    ],
}


def _experiment_status(result: ExperimentResult) -> str:
    state = "running"
    if result.done:
        state = "stopped early" if result.step < result.total_steps else "done"
    if result.error is not None:
        state = "failed"
    text = f"{EXPERIMENT_KINDS.get(result.kind, result.kind)} — {state}"
    if result.total_steps > 1:
        text += f" · step {result.step}/{result.total_steps}"
    if result.objective is not None:
        text += f" · objective {result.objective:.4g}"
    return text


def _experiment_img_html(image: bytes | None) -> str:
    """Input-space experiment image, CSS-upscaled like the input pane."""
    if image is None:
        return '<div class="text-xs text-slate-400 italic">not renderable</div>'
    return (
        f'<img src="{_b64_img_src(image)}" '
        f'style="width:{INPUT_IMAGE_SIZE}px; image-rendering:pixelated; '
        'display:block; max-width:none;" />'
    )


def _layer_channel_count(snap: BatchSnapshot | None, layer: str) -> int | None:
    """Channel count of `layer`'s last captured activation (None if unknown)."""
    act = snap.activations.get(layer) if snap is not None else None
    if act is None or act.ndim < 2:
        return None
    return int(act.shape[1])


def _build_experiment_page(
    session: Session,
    layer: str,
    *,
    input_mean: tuple[float, ...] | None,
    input_std: tuple[float, ...] | None,
) -> None:
    """Per-layer experiments: deep dream and selected Captum attributions.

    The top bar carries the experiment-kind dropdown (deep dream by
    default), Run / Cancel, and the shared stepping controls — experiments
    execute on the paused training thread, so the user can pause right from
    this page. The left pane holds the selected kind's parameter form
    (rebuilt on every dropdown change); the right pane streams status and
    results for *this page's own request* (`experiment_result_for`), so
    concurrent tabs running their own experiments never overwrite each
    other — Run replaces only this page's previous request, Cancel aborts
    only this page's. Deep dream results render as a denormalized
    input-space image (updating live while the run progresses);
    attributions render with the shared diverging-colormap strip machinery,
    next to the input sample they explain.
    """
    title = f"Experiment · {layer}" if layer else "Experiment"
    ui.page_title(f"PlayGrad — {title}")
    ui.query(".nicegui-content").classes("p-0 h-screen overflow-hidden")
    ui.query("body").classes("overflow-hidden")
    ui.query("html").classes("overflow-hidden")
    ui.add_head_html(_STRIP_MARKER_CSS)

    step_until_custom = _build_step_until_custom_dialog(session)
    widgets: dict[str, ui.element] = {}
    kind_holder = {"kind": "deep_dream"}
    my_seq: list[int | None] = [None]  # this page's own request
    last_result: list[ExperimentResult | None] = [None]

    def collect_params() -> dict[str, object]:
        params: dict[str, object] = {"mean": input_mean, "std": input_std}
        for spec in _EXPERIMENT_PARAMS[kind_holder["kind"]]:
            value: object = getattr(widgets.get(spec.key), "value", None)
            if spec.kind in ("int", "float"):
                if not isinstance(value, (int, float)):
                    value = spec.default
                assert isinstance(value, (int, float))  # numeric specs only
                params[spec.key] = int(value) if spec.kind == "int" else float(value)
            elif spec.kind == "bool":
                params[spec.key] = bool(value)
            else:
                params[spec.key] = str(value if value is not None else spec.default)
        return params

    def run() -> None:
        if my_seq[0] is not None:  # a re-Run replaces this page's request
            session.cancel_experiment(my_seq[0])
        my_seq[0] = session.request_experiment(
            kind=kind_holder["kind"], layer=layer, params=collect_params()
        )
        last_result[0] = None
        error_label.text = ""

    def cancel() -> None:
        if my_seq[0] is not None:
            session.cancel_experiment(my_seq[0])

    def on_kind_change(e: object) -> None:
        value = getattr(e, "value", None)
        if value is not None:
            kind_holder["kind"] = str(value)
            rebuild_params()

    with ui.column().classes("w-full h-screen no-wrap gap-0"):
        with _top_bar_row():
            ui.button(
                icon="arrow_back",
                on_click=lambda: ui.navigate.to("/"),
                color="slate-500",
            ).props("dense size=md").tooltip("Back to the main page")
            ui.label(title).classes(
                "font-mono text-base font-bold ml-2 truncate max-w-64"
            )
            position_label = _add_step_controls(session, step_until_custom)
            ui.select(
                dict(EXPERIMENT_KINDS),
                value=kind_holder["kind"],
                on_change=on_kind_change,
            ).props("dense outlined").classes("w-72 ml-3")
            ui.button("Run", icon="science", on_click=run, color="yellow-8").props(
                "dense size=md"
            ).tooltip("Run the experiment (training must be paused)")
            ui.button(
                "Cancel", on_click=cancel, color="slate-500"
            ).props("dense size=md").tooltip("Abort this page's experiment")

        with ui.row().classes("w-full grow min-h-0 no-wrap gap-0"):
            params_pane = ui.column().classes(
                "w-80 shrink-0 h-full overflow-auto p-4 gap-2 "
                "border-r-2 border-slate-300 bg-slate-50"
            )
            with ui.column().classes(
                "grow min-w-0 h-full overflow-auto p-4 gap-3 bg-slate-200"
            ):
                if layer not in session.layer_names:
                    _weights_placeholder(f"Unknown layer {layer!r}.")
                    return
                status_label = ui.label(
                    "Pick an experiment and press Run (training must be paused)."
                ).classes("text-sm text-slate-600")
                error_label = ui.label("").classes("text-sm text-red-600")
                results_row = ui.row().classes("gap-6 flex-wrap items-start")

    def rebuild_params() -> None:
        widgets.clear()
        params_pane.clear()
        with params_pane:
            ui.label("Parameters").classes("font-mono text-sm")
            for spec in _EXPERIMENT_PARAMS[kind_holder["kind"]]:
                if spec.kind == "bool":
                    widget: ui.element = ui.switch(
                        spec.label, value=bool(spec.default)
                    ).props("dense")
                elif spec.kind == "select":
                    widget = ui.select(
                        spec.options or {}, label=spec.label, value=spec.default
                    ).props("dense outlined").classes("w-full")
                else:
                    default = spec.default
                    if spec.key == "batch":  # tracks the live batch size
                        live = session.input_batch_size
                        if live is not None:
                            default = min(_DEFAULT_DREAM_BATCH, live)
                    maximum: float | None = None
                    if spec.key == "channel":  # bound to the layer's channels
                        channels = _layer_channel_count(session.snapshot, layer)
                        if channels is not None:
                            maximum = channels - 1
                    default_number = (
                        default if isinstance(default, (int, float)) else 0
                    )
                    widget = ui.number(
                        label=spec.label,
                        value=default_number,
                        min=spec.minimum,
                        max=maximum,
                        step=1 if spec.kind == "int" else spec.step,
                        format="%d" if spec.kind == "int" else None,
                    ).props("dense outlined").classes("w-full")
                if spec.tooltip:
                    widget.tooltip(spec.tooltip)
                widgets[spec.key] = widget

    def render_batch_images(title: str, tensor: Tensor) -> None:
        """A labelled, wrapping grid of every sample in `tensor`.

        Non-image tensors (deep dream runs on the network's real input,
        whatever its shape) fall back to a single "not renderable" note.
        """
        rendered = [
            render_image(tensor, i, mean=input_mean, std=input_std)
            for i in range(int(tensor.shape[0]))
        ]
        with ui.column().classes("gap-1 min-w-0"):
            ui.label(title).classes("font-mono text-xs text-slate-600")
            if not any(r is not None for r in rendered):
                ui.html(_experiment_img_html(None))
                return
            with ui.row().classes("gap-2 flex-wrap"):
                for image in rendered:
                    ui.html(_experiment_img_html(image))

    def render_result(result: ExperimentResult) -> None:
        results_row.clear()
        with results_row:
            if result.image is not None:
                render_batch_images("Result", result.image)
            if result.attribution is not None:
                with ui.column().classes("gap-1 min-w-0"):
                    ui.label("Attribution").classes(
                        "font-mono text-xs text-slate-600"
                    )
                    with ui.element("div").classes("max-w-full overflow-x-auto"):
                        # `reference` is the input batch the attribution
                        # explains; its spatial size lets token-shaped
                        # attributions unflatten onto the patch grid.
                        ui.html(
                            _strip_html(
                                render_strip(
                                    result.attribution,
                                    0,
                                    input_hw=_tensor_hw(result.reference),
                                )
                            )
                        )
            if result.reference is not None:
                render_batch_images("Input", result.reference)

    def tick() -> None:
        live = session.live_position
        if live is not None:
            position_label.text = _format_live_position(live)
        if my_seq[0] is None:
            return  # nothing requested from this page yet
        result = session.experiment_result_for(my_seq[0])
        if result is None:
            if last_result[0] is None:
                status_label.text = (
                    "queued — waiting for the training thread to pause "
                    "or for earlier experiments (use Stop / Step Batch above)"
                )
            return
        status_label.text = _experiment_status(result)
        error_label.text = result.error or ""
        if result is not last_result[0]:
            last_result[0] = result
            if result.error is None:
                render_result(result)

    rebuild_params()
    ui.timer(0.2, tick)


def _add_time_travel_button(session: Session) -> None:
    """The blue Time Travel button (right of Detach) and its dialogs.

    The button is grayed out — with the reason as a hover tooltip — when the
    session has no training restorer, i.e. the training loop did not opt
    into time travel. The picker dialog rebuilds its content on every open,
    so the cached-epoch choices track the checkpoints written so far; a
    request the session rejects (e.g. a cache written by a different model)
    surfaces in a separate error dialog and no jump happens.
    """
    with ui.dialog() as error_dialog, ui.card().classes("min-w-96 p-6 gap-3"):
        ui.label("Time travel failed").classes("text-lg font-bold text-red-600")
        error_label = ui.label("").classes("text-sm text-slate-700")
        with ui.row():
            ui.button("Close", on_click=error_dialog.close)

    with ui.dialog() as dialog, ui.card().classes("min-w-96 p-6 gap-4"):
        ui.label("Time travel").classes("text-lg font-bold")
        content = ui.column().classes("w-full gap-3")

    def show_error(message: str) -> None:
        error_label.text = message
        error_dialog.open()

    def submit(value: object) -> None:
        if not isinstance(value, (int, float)):
            show_error("Pick an epoch to jump to.")
            return
        try:
            session.request_time_travel(int(value))
        except TimeTravelError as e:
            show_error(str(e))
            return
        dialog.close()

    def cache_run() -> None:
        # Runs training to the very end; with a restorer active, every
        # epoch start gets checkpointed along the way, filling the cache.
        session.step_run()
        dialog.close()

    def rebuild() -> None:
        content.clear()
        status = session.time_travel_status()
        cached = status.cached_epochs
        missing = sorted(set(range(status.total_epochs)) - set(cached))
        with content:
            if not status.available:
                ui.label(status.reason or "Time travel is unavailable.").classes(
                    "text-sm text-slate-600"
                )
                with ui.row():
                    ui.button("Close", on_click=dialog.close)
                return
            # The slider runs over *indices into the cached-epoch list*, so
            # epochs without a checkpoint are unselectable even when the
            # cached set has gaps; the label shows the mapped epoch number.
            epoch_slider: ui.slider | None = None
            if cached:
                ui.label("Jump back to the start of a cached epoch:").classes(
                    "text-sm"
                )
                with ui.row().classes("w-full items-center gap-4 no-wrap"):
                    epoch_label = ui.label(f"epoch {cached[-1]}").classes(
                        "font-mono text-sm w-20 shrink-0"
                    )
                    epoch_slider = ui.slider(
                        min=0,
                        max=len(cached) - 1,
                        step=1,
                        value=len(cached) - 1,
                        on_change=lambda e: epoch_label.set_text(
                            f"epoch {cached[int(e.value)]}"
                        ),
                    ).classes("grow")
            else:
                ui.label("No epochs have a cached model yet.").classes(
                    "text-sm text-slate-600"
                )
            ui.label(
                "Cached epochs: "
                + (_summarize_epoch_ranges(cached) if cached else "none")
            ).classes("text-xs text-slate-500")
            if missing:
                ui.label(
                    f"Uncached epochs: {_summarize_epoch_ranges(missing)}. "
                    "An epoch becomes available once training has passed "
                    "its start."
                ).classes("text-xs text-slate-500")
            with ui.row():
                ui.button("Cancel", on_click=dialog.close)
                if missing:
                    ui.button(
                        "Cache full training run", on_click=cache_run
                    ).tooltip(
                        "Run training to the end, checkpointing every epoch "
                        "start along the way"
                    )
                if epoch_slider is not None:
                    slider = epoch_slider
                    ui.button(
                        "Time travel",
                        on_click=lambda: submit(
                            cached[int(slider.value)]
                            if slider.value is not None
                            else None
                        ),
                        color="blue",
                    )

    def open_dialog() -> None:
        rebuild()
        dialog.open()

    status = session.time_travel_status()
    # Quasar suppresses pointer events on disabled buttons, so the tooltip
    # (which must explain *why* the button is off) lives on a wrapper div.
    tooltip = (
        "Jump training back to the start of a cached epoch"
        if status.available
        else (status.reason or "Time travel is unavailable.")
    )
    with ui.element("div").tooltip(tooltip):
        button = ui.button("Time Travel", on_click=open_dialog, color="blue").props(
            "dense size=md"
        )
        if not status.available:
            button.props("disable")


def _summarize_epoch_ranges(epochs: list[int]) -> str:
    """Compact "0–2, 5, 7–49" rendering of a sorted epoch list."""
    parts: list[str] = []
    start = prev = epochs[0]
    for e in epochs[1:]:
        if e == prev + 1:
            prev = e
            continue
        parts.append(f"{start}–{prev}" if prev > start else f"{start}")
        start = prev = e
    parts.append(f"{start}–{prev}" if prev > start else f"{start}")
    return ", ".join(parts)


def _build_step_until_custom_dialog(session: Session) -> ui.dialog:
    schedule = session.schedule
    phase_names = list(schedule.phases)

    with ui.dialog() as dialog, ui.card().classes("min-w-96 p-6 gap-4"):
        ui.label("Step until custom").classes("text-lg font-bold")
        with ui.row().classes("w-full gap-4 items-end no-wrap"):
            epoch_input = ui.number(
                label="Epoch", value=0, min=0, step=1, format="%d"
            ).classes("flex-1")
            phase_select = ui.select(
                phase_names, label="Phase", value=phase_names[0]
            ).classes("flex-1")
            batch_input = ui.number(
                label="Batch", value=0, min=0, step=1, format="%d"
            ).classes("flex-1")
        error_label = ui.label("").classes("text-red-500 text-sm min-h-4")

        def submit() -> None:
            try:
                epoch = int(epoch_input.value) if epoch_input.value is not None else 0
                batch_idx = int(batch_input.value) if batch_input.value is not None else 0
            except (TypeError, ValueError):
                error_label.text = "Invalid input"
                return
            phase = str(phase_select.value)
            error = _validate_step_until_target(
                schedule=schedule,
                snapshot=session.snapshot,
                phase=phase,
                epoch=epoch,
                batch_idx=batch_idx,
            )
            if error is not None:
                error_label.text = error
                return
            error_label.text = ""
            session.step_until_position(phase=phase, epoch=epoch, batch_idx=batch_idx)
            dialog.close()

        with ui.row():
            ui.button("Cancel", on_click=dialog.close)
            ui.button("Step", on_click=submit)

    return dialog


def _validate_step_until_target(
    *,
    schedule: Schedule,
    snapshot: BatchSnapshot | None,
    phase: str,
    epoch: int,
    batch_idx: int,
) -> str | None:
    phases = schedule.phases
    if phase not in phases:
        return f"Unknown phase {phase!r}"
    if not 0 <= epoch < schedule.epochs:
        return f"Epoch must be in [0, {schedule.epochs - 1}]"
    declared = phases[phase]
    if not 0 <= batch_idx < declared:
        return f"Batch must be in [0, {declared - 1}] for phase {phase!r}"
    if snapshot is not None:
        cur = snapshot.position
        target_rank = _position_rank(phases, phase, epoch, batch_idx)
        current_rank = _position_rank(phases, cur.phase, cur.epoch, cur.batch_idx)
        if target_rank <= current_rank:
            return "Target must be after the current position"
    return None


def _position_rank(
    phases: dict[str, int], phase: str, epoch: int, batch_idx: int
) -> tuple[int, int, int]:
    return (epoch, list(phases).index(phase), batch_idx)


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


def _tensor_hw(tensor: Tensor | None) -> tuple[int, int] | None:
    """Spatial size of a `[B, C, H, W]` input, or `None` when not image-like.

    Threaded into `render_strip` as `input_hw` so 2D (token-shaped)
    activations can be unflattened back onto the input's patch grid.
    """
    if tensor is None or tensor.ndim != 4:
        return None
    return int(tensor.shape[-2]), int(tensor.shape[-1])


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
    input_hw = _tensor_hw(snap.activations.get(input_name) if input_name else None)

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

    rendered = dict(
        zip(layer_names, _RENDER_POOL.map(strips, layer_names), strict=True)
    )
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
    perturbed_acts = probe.perturbed_activations
    kind = "probe-diff" if compare else (
        "probe-perturbed" if perturbed_acts is not None else "probe-act"
    )
    input_hw = _tensor_hw(probe.input)

    def act_tensor(name: str) -> Tensor | None:
        base = probe.activations.get(name)
        if compare:
            if base is None:
                return None
            if perturbed_acts is None:
                return torch.zeros_like(base)
            pert = perturbed_acts.get(name)
            if pert is None or pert.shape != base.shape:
                return None
            return pert - base
        if perturbed_acts is not None:
            return perturbed_acts.get(name)
        return base

    def strips(name: str) -> tuple[str, str]:
        act = cache.get_or_render(
            probe,
            (name, kind, sample_idx),
            lambda: _strip_html(
                render_strip(act_tensor(name), sample_idx, input_hw=input_hw)
            ),
        )
        return act, _PROBE_NO_GRADIENTS_HTML

    rendered = dict(
        zip(layer_names, _RENDER_POOL.map(strips, layer_names), strict=True)
    )
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
        session: Session,
        weights: list[str],
        on_toggle_watch: Callable[[str], None],
    ) -> None:
        self.name = name
        card = ui.element("div").classes(
            "w-full min-w-0 bg-white rounded border border-slate-300 shadow-sm "
            "hover:border-blue-400 transition-colors"
        )
        card.props(f'data-layer="{slug(name)}"')
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
                if weights:
                    with ui.element("div").props("data-card-action"):
                        ui.button(
                            "Weights",
                            icon="grid_on",
                            on_click=lambda: ui.navigate.to(
                                f"/weights?layer={quote(name)}", new_tab=True
                            ),
                            color="blue",
                        ).props("dense no-caps").style(
                            "min-height: 0; padding: 1px 6px; font-size: 11px"
                        ).tooltip(
                            f"Inspect this layer's weights ({len(weights)})"
                        )
                with ui.element("div").props("data-card-action"):
                    ui.button(
                        "Experiment",
                        icon="science",
                        on_click=lambda: ui.navigate.to(
                            f"/experiment?layer={quote(name)}", new_tab=True
                        ),
                        color="yellow-8",
                    ).props("dense no-caps").style(
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


def _strip_marker(color_class: str, label: str) -> None:
    """A labelled colored bar marking which kind of strip sits next to it.

    Stretches to the strip's height via the flex row (and collapses to
    nothing when the strip is empty); `sticky left-0` keeps it pinned to the
    card's left edge while the strip scrolls underneath. A sticky element
    can only travel within its parent, so the caller must make the flex row
    span the full scrollable width (the `w-max min-w-full` wrapper around
    the rows) — otherwise the marker is dragged out of view once the strips
    are scrolled past the visible width. The label is drawn vertically,
    reading bottom-up, and is absolutely positioned so it adds no intrinsic
    height — otherwise a missing strip would leave a floating bar instead
    of an empty row. On strips too short to fit it the label is hidden via
    the container query in `_STRIP_MARKER_CSS`; the tooltip carries the
    full name regardless.
    """
    with ui.element("div").classes(
        f"playgrad-marker w-5 shrink-0 rounded mr-2 sticky left-0 z-10 "
        f"overflow-hidden {color_class}"
    ).tooltip(label.capitalize()):
        ui.label(label).classes(
            "playgrad-marker-label absolute text-white font-bold select-none"
        ).style(
            "writing-mode: vertical-rl; top: 50%; left: 50%; "
            "transform: translate(-50%, -50%) rotate(180deg); "
            "font-size: 9px; letter-spacing: 0.12em; line-height: 1; "
            "white-space: nowrap;"
        )


def _b64_img_src(image: bytes, *, mime: str | None = None) -> str:
    """Data-URI for an image, defaulting to the global `STRIP_FORMAT` mime."""
    encoded = base64.b64encode(image).decode("ascii")
    return f"data:{mime or image_mime()};base64,{encoded}"


def _strip_html(strip: StripRender | None) -> str:
    """HTML for one strip: a crisp legend `<img>` plus one native-res data `<img>`.

    The data image holds every tile (with native-resolution separators) at
    the tensor's native resolution; explicit CSS width/height plus
    `image-rendering: pixelated` make the browser do the nearest-neighbour
    upscale the renderer used to do server-side. The legend image is already
    at display resolution and renders 1:1, so its labels stay sharp.
    `flex:none` keeps the scroll container from squishing the images.
    """
    if strip is None:
        return ""
    return (
        '<div style="display:flex; align-items:flex-start;">'
        f'<img src="{_b64_img_src(strip.legend_image)}" '
        'style="display:block; flex:none; max-width:none;" />'
        f'<img src="{_b64_img_src(strip.data_image)}" '
        f'style="width:{strip.width}px; height:{strip.height}px; '
        'image-rendering:pixelated; display:block; flex:none; max-width:none;" />'
        "</div>"
    )


def _input_img_src(image: bytes | None) -> str:
    """Data-URI source for the input pane's interactive image ("" when absent).

    The sizing (CSS upscale to `INPUT_IMAGE_SIZE` with nearest-neighbour
    rendering) lives on the `ui.interactive_image` element in `InputPanel`;
    only the native-resolution source travels per frame.
    """
    if image is None:
        return ""
    return _b64_img_src(image)
