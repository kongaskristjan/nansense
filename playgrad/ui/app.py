"""NiceGUI app that visualizes a `Session`.

The app runs on a background daemon-thread-disabled uvicorn (signal handlers
disabled so it survives being started from a non-main thread). Layout:

- Top bar with the six control buttons, a position label, and a sample
  index spinner.
- Left pane: the model architecture as a Mermaid diagram (built once at
  start).
- Right pane: a one-column table with one row per submodule. Each row
  holds two horizontally scrollable strips — activations on top,
  activation gradients below — sharing a single horizontal scrollbar so
  they pan together.

A `ui.timer` in each connection polls `session.snapshot`; when a new
snapshot is published, the page re-renders all per-layer strips against
it. The same timer also refreshes the top-bar position label from
`session.live_position` on every tick, so the displayed epoch/batch keeps
advancing during modes that don't publish a snapshot every batch
(step-epoch, step-custom, run, detach) — a cheap label write, decoupled
from the heavier strip rendering.
"""

from __future__ import annotations

import asyncio
import base64
import json
import math
import threading
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import quote

import plotly.graph_objects as go
import uvicorn
from fastapi import FastAPI
from nicegui import ui
from torch import Tensor

from playgrad.restore import TimeTravelError
from playgrad.schedule import BatchPosition, Schedule
from playgrad.session import BatchSnapshot, Session
from playgrad.ui.graph import build_mermaid, slug
from playgrad.ui.render import (
    default_weight_dims,
    render_image,
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
# of the pair; clicking either side scrolls the *other* pane so the
# matching element lands at the top. Scroll positions are computed
# directly instead of via `scrollIntoView`, because the latter leaves the
# target several dozen pixels below the column's top edge here (the
# previous item's tail stays visible), even with `block: 'start'`.
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
      const card = findCard(slug);
      if (!card) return;
      scrollTargetToTop(card);
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

    @ui.page("/", favicon=str(favicon_path))
    def index() -> None:
        _build_page(
            session,
            mermaid_src,
            layer_names,
            input_name=input_name,
            input_mean=input_mean,
            input_std=input_std,
        )

    @ui.page("/watch", favicon=str(favicon_path))
    def watch_page() -> None:
        _build_watch_page(session, layer_names)

    @ui.page("/weights", favicon=str(favicon_path))
    def weights_page(layer: str = "") -> None:
        _build_weights_page(session, layer)

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
    sample_idx: int = 0
    last_snapshot: BatchSnapshot | None = None
    dirty: bool = False
    rendering: bool = False
    spinner_max: int | None = None


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
) -> None:
    state = _PageState()
    layer_views: dict[str, _LayerView] = {}

    ui.page_title("PlayGrad")
    ui.query(".nicegui-content").classes("p-0 h-screen overflow-hidden")
    ui.query("body").classes("overflow-hidden")
    ui.query("html").classes("overflow-hidden")
    ui.add_head_html(_ARCHITECTURE_CLICK_CSS)
    ui.add_head_html(_STRIP_MARKER_CSS)
    ui.add_body_html(_ARCHITECTURE_CLICK_JS)

    step_until_custom = _build_step_until_custom_dialog(session)

    with ui.column().classes("w-full h-screen no-wrap gap-0"):
        with _top_bar_row():
            architecture_toggle = ui.button(
                icon="account_tree", color="slate-500"
            ).props("dense size=md").tooltip("Toggle architecture pane")
            position_label = _add_step_controls(session, step_until_custom)
            ui.label("Viewing sample:").classes("ml-3 text-sm")
            sample_input = ui.number(value=0, min=0, step=1, format="%d").classes("w-20").props("dense")
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
                        watch_list_container = ui.element("div").classes("py-1")
            input_toggle = ui.button(
                icon="image", color="slate-500"
            ).props("dense size=md").tooltip(
                "Toggle input image pane"
            )

            def _defer_clamp_display(target: int) -> None:
                # NiceGUI suppresses .value writes made from inside a value-change
                # handler; schedule the display correction for the next event loop
                # iteration so it actually reaches the client.
                ui.timer(0.0, lambda: sample_input.set_value(target), once=True)

            def on_sample_change(e: object) -> None:
                value = getattr(e, "value", None)
                idx = int(value) if value is not None else 0
                if idx < 0:
                    idx = 0
                    _defer_clamp_display(idx)
                elif state.spinner_max is not None and idx > state.spinner_max:
                    idx = state.spinner_max
                    _defer_clamp_display(idx)
                state.sample_idx = idx
                state.dirty = True

            sample_input.on_value_change(on_sample_change)

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
                for layer in layer_names:
                    if layer not in watched:
                        continue
                    ui.menu_item(
                        layer,
                        on_click=lambda n=layer: ui.run_javascript(
                            f"window.playgradScrollToLayer({json.dumps(slug(n))})"
                        ),
                    ).classes("font-mono text-sm")

        def toggle_layer(name: str) -> None:
            was_watched = name in session.watched_layers
            if was_watched:
                session.unwatch(name)
                now_watched = False
            else:
                # `session.watch` returns False for non-modules (fx
                # intermediates, graph inputs). Treat that as a no-op.
                now_watched = session.watch(name)
            if was_watched == now_watched:
                return
            ui.run_javascript(
                f"window.playgradSetWatched({json.dumps(slug(name))}, "
                f"{'true' if now_watched else 'false'})"
            )
            refresh_chip()
            layer_views[name].refresh_eye()

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
                ui.label("Input").classes("font-mono text-sm self-start")
                input_html = ui.html("")

        def toggle_architecture() -> None:
            architecture_pane.set_visibility(not architecture_pane.visible)

        def toggle_input() -> None:
            input_pane.set_visibility(not input_pane.visible)

        architecture_toggle.on_click(toggle_architecture)
        input_toggle.on_click(toggle_input)

    # Populate the chip menu and, if anything is already watched, push the
    # set into JS so the MutationObserver applies the amber treatment to
    # mermaid nodes once Mermaid finishes rendering them client-side.
    refresh_chip()
    initial_watched = list(session.watched_layers)
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
        snap = session.snapshot
        if snap is None:
            return
        _sync_spinner_max(snap, state, sample_input)
        if state.rendering:
            return
        if snap is not state.last_snapshot or state.dirty:
            state.last_snapshot = snap
            state.dirty = False
            state.rendering = True
            try:
                sample_idx = state.sample_idx
                rendered, input_img = await asyncio.to_thread(
                    _compute_frame,
                    layer_views,
                    snap,
                    sample_idx,
                    input_name=input_name,
                    input_mean=input_mean,
                    input_std=input_std,
                )
            finally:
                state.rendering = False
            _apply_all(layer_views, rendered)
            input_html.set_content(input_img)

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


def _stats_summary(stats: TensorStatsSnapshot) -> str:
    """Compact one-line summary of the scalar stats shown above each histogram."""
    if stats.n == 0:
        return "no data yet"
    return (
        f"n={stats.n:,}  "
        f"mean={_format_stat(stats.mean)}  "
        f"std={_format_stat(stats.std)}  "
        f"median≈{_format_stat(stats.median)}  "
        f"min={_format_stat(stats.min)}  "
        f"max={_format_stat(stats.max)}"
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

# Density mode (both axes linear) caps the y-axis at the height of the
# N-th tallest bar; see `_density_y_range`.
_DENSITY_TOP_BINS: int = 20


def _use_density(log_x: bool, log_y: bool) -> bool:
    """Whether bars show density (count / bin width) instead of raw counts.

    On a linear value axis with a linear count axis, raw counts are
    misleading: the signed-log bins differ in linear width by orders of
    magnitude, so a wide bin towers over a narrow one holding the same
    share of values. Density makes bar *area* proportional to count, the
    honest reading of a distribution on linear axes. With either log axis
    the area intuition is gone anyway, so raw counts are kept there.
    """
    return not log_x and not log_y


def _density_heights(hist: tuple[int, ...]) -> list[float]:
    """Per-bin density: count divided by the bin's linear width."""
    return [c / w for c, w in zip(hist, _BIN_WIDTHS)]


def _density_y_range(
    per_phase: dict[str, LayerStatsSnapshot], kind: str
) -> list[float] | None:
    """Y-axis range for density mode, capped at the 20th-tallest bar.

    The bins near zero are extremely narrow (the zero band is 2e-9 wide), so
    even a handful of near-zero values produce densities that dwarf the rest
    of the distribution; autoranging to the global max would flatten
    everything else. Instead the axis tops out at the smallest of the
    `_DENSITY_TOP_BINS` tallest bars across the drawn traces (the tallest
    populated-bar height that the scale must still accommodate), letting the
    taller spikes clip. Returns `None` (Plotly autorange) when there's no
    data.
    """
    heights = sorted(
        (
            h
            for phase in _phases_with_data(per_phase, kind)
            for h in _density_heights(_kind_stats(per_phase[phase], kind).hist)
            if h > 0
        ),
        reverse=True,
    )
    if not heights:
        return None
    cap = heights[min(_DENSITY_TOP_BINS - 1, len(heights) - 1)]
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


def _linear_x_range(
    per_phase: dict[str, LayerStatsSnapshot], kind: str
) -> list[float] | None:
    """X-axis range (linear value space) covering the populated bins.

    On a linear axis the bars span the full `[-1e6, 1e6]` edge range, almost
    all of it empty. We zoom to the smallest/largest edges that actually hold
    counts (plus a little padding) so the populated part of the distribution
    stays legible. Returns `None` (Plotly autorange) when there's no data.
    """
    lo_idx = N_BINS
    hi_idx = -1
    for layer_snap in per_phase.values():
        for i, count in enumerate(_kind_stats(layer_snap, kind).hist):
            if count > 0:
                lo_idx = min(lo_idx, i)
                hi_idx = max(hi_idx, i)
    if hi_idx < 0:
        return None
    lo = _HIST_EDGES[lo_idx]
    hi = _HIST_EDGES[hi_idx + 1]
    pad = (hi - lo) * 0.05 or 1.0
    return [lo - pad, hi + pad]


def _make_histogram_figure(
    per_phase: dict[str, LayerStatsSnapshot],
    kind: str,
    title: str,
    *,
    log_x: bool = False,
    log_y: bool = False,
) -> go.Figure:
    """Plotly bar chart of the signed-log histogram, one trace per phase.

    `kind` selects which of the two histograms on each `LayerStatsSnapshot`
    to plot ("activation" or "gradient"). `per_phase` may be empty (initial
    render before any data has been collected) — the figure is still
    returned, just with no traces.

    `log_x` / `log_y` toggle the value (x) and count (y) axes between a
    log-based and a linear scale (the "Log x" / "Log y" checkboxes on the
    Watching page). With both off, bars show density instead of counts
    (see `_use_density`) and the y-axis is capped (see `_density_y_range`).

    This builds the *whole* figure. Routine data refreshes don't call it —
    they restyle the existing figure in place (see `_HistPlot`) so client-side
    state like legend toggles survives; the figure is only rebuilt when the
    set of phases or the axis scale changes.
    """
    x_values = list(range(N_BINS)) if log_x else _BIN_CENTERS
    density = _use_density(log_x, log_y)
    if log_x:
        hover = "bin %{x}<br>count %{y}<extra></extra>"
    elif density:
        hover = (
            "value %{x:.2e}<br>density %{y:.3g}"
            "<br>count %{customdata}<extra></extra>"
        )
    else:
        hover = "value %{x:.2e}<br>count %{y}<extra></extra>"
    fig = go.Figure()
    phases = _phases_with_data(per_phase, kind)
    for i, (phase, layer_snap) in enumerate(per_phase.items()):
        stats = _kind_stats(layer_snap, kind)
        if stats.n == 0:
            continue
        fig.add_trace(
            go.Bar(
                x=x_values,
                y=_density_heights(stats.hist) if density else list(stats.hist),
                customdata=list(stats.hist) if density else None,
                width=None if log_x else _BIN_WIDTHS,
                name=f"{phase} (ep {layer_snap.epoch})",
                marker_color=_phase_color(phase, i),
                opacity=0.55 if len(per_phase) > 1 else 0.85,
                hovertemplate=hover,
            )
        )
    has_data = bool(phases)
    fig.update_layout(
        title=dict(text=title, x=0.0, font=dict(size=12)),
        barmode="overlay",
        bargap=0,
        margin=dict(l=50, r=20, t=40, b=40),
        height=_PLOT_HEIGHT,
        plot_bgcolor="#f8fafc",
        paper_bgcolor="white",
        showlegend=has_data,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(size=10),
        ),
    )
    if log_x:
        tick_vals, tick_text = _x_tick_layout()
        fig.update_xaxes(
            tickvals=tick_vals,
            ticktext=tick_text,
            tickfont=dict(size=9),
            showgrid=False,
            zeroline=False,
        )
    else:
        fig.update_xaxes(
            range=_linear_x_range(per_phase, kind),
            tickfont=dict(size=9),
            showgrid=False,
            zeroline=True,
            zerolinecolor="#cbd5e1",
        )
    fig.update_yaxes(
        type="log" if log_y else "linear",
        range=_density_y_range(per_phase, kind) if density else None,
        showgrid=True,
        gridcolor="#e2e8f0",
        tickfont=dict(size=9),
        title=dict(text="density" if density else "count", font=dict(size=10)),
    )
    return fig


def _build_watch_page(session: Session, layer_names: list[str]) -> None:
    """The deep-dive page for watched layers — plotly histograms per layer.

    Each card renders one bar chart per tensor kind (activations and
    activation gradients) with train/val overlaid. A `ui.timer` polls
    `session.watch_snapshot()` and refreshes the figures in place.

    Layers can also be unwatched directly from the card header here, which
    drops the corresponding accumulator entry — the change is reflected on
    the main page on next navigation.
    """
    ui.page_title("PlayGrad — Watching")
    ui.query(".nicegui-content").classes("p-0 h-screen overflow-hidden")
    ui.query("body").classes("overflow-hidden")
    ui.query("html").classes("overflow-hidden")

    layer_panels: dict[str, _WatchLayerPanel] = {}
    count_label_holder: dict[str, ui.label] = {}
    body_container: ui.column
    # Whether the value (x) and count (y) axes use a log-based scale. Both
    # default off — linear axes showing density (see `_use_density`); the
    # header checkboxes flip them and re-render every plot immediately.
    axis_log = {"x": False, "y": False}

    def set_axis_log(axis: str, value: bool) -> None:
        axis_log[axis] = value
        refresh()

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
                "Log scale on the count axis"
            )
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
                )

    def refresh() -> None:
        watched = session.watched_layers
        n = len(watched)
        count_label_holder["count"].text = f"{n} layer{'' if n == 1 else 's'}"
        if set(layer_panels) != set(watched):
            rebuild_cards()
        snap = session.watch_snapshot()
        for panel in layer_panels.values():
            panel.update(snap)

    refresh()
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


class _HistPlot:
    """One Plotly histogram that refreshes its data in place.

    The figure is built once and rebuilt only when the set of phases or the
    axis scale changes. Routine per-tick updates go through `Plotly.restyle`,
    which leaves client-side state — legend toggles, zoom — untouched, so a
    series the user clicked off the legend stays hidden across refreshes.
    """

    def __init__(self, kind: str, title: str, axis_log: dict[str, bool]) -> None:
        self._kind = kind
        self._title = title
        self._axis_log = axis_log
        # Signature of what's currently drawn, so `update` can tell a plain
        # data refresh (restyle) from a structural change (rebuild).
        self._phases: list[str] = []
        self._axis = self._current_axis()
        # Last y-range applied in density mode, so refreshes only push a
        # relayout when the cap actually moved (a range write resets zoom).
        self._y_range: list[float] | None = None
        self.element = ui.plotly(
            _make_histogram_figure(
                {}, kind, title, log_x=self._axis[0], log_y=self._axis[1]
            )
        ).classes("w-full")

    def _current_axis(self) -> tuple[bool, bool]:
        return self._axis_log["x"], self._axis_log["y"]

    def update(self, per_phase: dict[str, LayerStatsSnapshot]) -> None:
        phases = _phases_with_data(per_phase, self._kind)
        axis = self._current_axis()
        if phases != self._phases or axis != self._axis:
            # A phase appeared/disappeared or an axis-scale checkbox flipped —
            # rebuild the whole figure.
            self.element.figure = _make_histogram_figure(
                per_phase, self._kind, self._title, log_x=axis[0], log_y=axis[1]
            )
            self.element.update()
            self._phases = phases
            self._axis = axis
            self._y_range = (
                _density_y_range(per_phase, self._kind)
                if _use_density(*axis)
                else None
            )
        elif phases:
            # Same traces and axes — only counts (and the epoch label) moved.
            # Restyle in place so legend toggles and zoom survive.
            hists = [_kind_stats(per_phase[p], self._kind).hist for p in phases]
            names = [f"{p} (ep {per_phase[p].epoch})" for p in phases]
            update: dict[str, object] = {"name": names}
            layout: dict[str, object] | None = None
            if _use_density(*axis):
                update["y"] = [_density_heights(h) for h in hists]
                update["customdata"] = [list(h) for h in hists]
                # The cap follows the data; re-apply it only when it moved so
                # an idle refresh doesn't keep snapping the user's zoom back.
                y_range = _density_y_range(per_phase, self._kind)
                if y_range != self._y_range:
                    self._y_range = y_range
                    layout = {"yaxis.range": y_range}
            else:
                update["y"] = [list(h) for h in hists]
            _plotly_restyle(
                self.element, update, list(range(len(phases))), layout
            )


class _WatchLayerPanel:
    """One card per watched layer — activations + gradients histograms."""

    def __init__(
        self,
        *,
        name: str,
        session: Session,
        on_unwatched: Callable[[], None],
        axis_log: dict[str, bool],
    ) -> None:
        self.name = name
        self._session = session
        with ui.card().classes("w-full p-4 gap-2"):
            with ui.row().classes("w-full items-center gap-2 no-wrap"):
                ui.label(name).classes("font-mono text-base font-bold grow")
                ui.button(
                    icon="visibility_off",
                    color="amber-600",
                    on_click=lambda: (session.unwatch(name), on_unwatched()),
                ).props("dense size=sm flat round").tooltip("Stop watching")
            with ui.column().classes("w-full gap-3"):
                ui.label("Activations").classes(
                    "font-mono text-sm text-slate-600"
                )
                self._act_stats = ui.label("no data yet").classes(
                    "font-mono text-xs text-slate-500"
                )
                self._act = _HistPlot("activation", "activations", axis_log)
                ui.label("Gradients").classes(
                    "font-mono text-sm text-slate-600"
                )
                self._grad_stats = ui.label("no data yet").classes(
                    "font-mono text-xs text-slate-500"
                )
                self._grad = _HistPlot("gradient", "gradients", axis_log)

    def update(self, snap: WatchSnapshot) -> None:
        per_phase = snap.latest_per_phase(self.name)
        self._act_stats.text = _combined_summary(per_phase, "activation")
        self._grad_stats.text = _combined_summary(per_phase, "gradient")
        self._act.update(per_phase)
        self._grad.update(per_phase)


def _combined_summary(
    per_phase: dict[str, LayerStatsSnapshot], kind: str
) -> str:
    if not per_phase:
        return "no data yet"
    parts: list[str] = []
    for phase, layer_snap in per_phase.items():
        stats = _kind_stats(layer_snap, kind)
        parts.append(f"[{phase} ep{layer_snap.epoch}] {_stats_summary(stats)}")
    return "    ".join(parts)


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
        png = render_weight(
            tensor, x_dim=x_dim, y_dim=y_dim, tile_dim=tile, fixed=fixed
        )
        if png is None:
            self._show_error("invalid axis selection")
            return
        self._error.text = ""
        self._img.set_content(_img_tag(png))
        # The gradient shares the weight's shape, so the same axis layout
        # applies; it's simply absent before the first backward pass.
        grad = self._gradients.get(self.name) if self._gradients is not None else None
        grad_png = (
            render_weight(grad, x_dim=x_dim, y_dim=y_dim, tile_dim=tile, fixed=fixed)
            if grad is not None
            else None
        )
        self._grad_img.set_content(
            _img_tag(grad_png) if grad_png is not None else _NO_GRADIENT_HTML
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
                    png = render_weight(
                        tensor, x_dim=x_dim, y_dim=y_dim, tile_dim=tile, fixed=fixed
                    )
                else:
                    dims = default_weight_dims(tensor.ndim)
                    png = render_weight(
                        tensor,
                        x_dim=dims.x_dim,
                        y_dim=dims.y_dim,
                        tile_dim=dims.tile_dim,
                        fixed={d: 0 for d in dims.fixed_dims},
                    )
                if png is None:
                    continue
                ui.element("div").classes("h-1")
                with ui.element("div").classes("flex no-wrap items-stretch"):
                    _strip_marker("bg-amber-600", key.upper())
                    ui.html(_img_tag(png))
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


def _sync_spinner_max(
    snap: BatchSnapshot,
    state: _PageState,
    sample_input: ui.number,
) -> None:
    batch_size = _snapshot_batch_size(snap)
    if batch_size is None or batch_size <= 0:
        return
    new_max = batch_size - 1
    if state.spinner_max == new_max:
        return
    state.spinner_max = new_max
    sample_input.max = new_max
    if state.sample_idx > new_max:
        state.sample_idx = new_max
        sample_input.value = new_max
        state.dirty = True


def _compute_frame(
    views: dict[str, _LayerView],
    snap: BatchSnapshot,
    sample_idx: int,
    *,
    input_name: str | None,
    input_mean: tuple[float, ...] | None,
    input_std: tuple[float, ...] | None,
) -> tuple[dict[str, tuple[str, str]], str]:
    rendered = {
        name: view.compute(
            snap.activations.get(name),
            snap.activation_gradients.get(name),
            sample_idx,
        )
        for name, view in views.items()
    }
    input_tensor = snap.activations.get(input_name) if input_name else None
    input_png = render_image(
        input_tensor, sample_idx, mean=input_mean, std=input_std
    )
    return rendered, _img_tag(input_png)


def _apply_all(
    views: dict[str, _LayerView],
    rendered: dict[str, tuple[str, str]],
) -> None:
    for name, (act_html, grad_html) in rendered.items():
        views[name].apply(act_html, grad_html)


class _LayerView:
    """One card per submodule, with activation + activation-gradient strips.

    The strips are raw `<img>` elements with `max-width: none`, so each PNG
    renders at its natural pixel width and the wrapping `overflow-x-auto`
    div produces a shared horizontal scrollbar inside the card. NiceGUI's
    `ui.image` uses Quasar's responsive q-img instead, which squishes the
    strip to the card width — not what we want here. The card has
    `min-w-0` so a wide strip doesn't push the column wider.

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
        self._session = session
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
                    self._eye_btn = ui.button(
                        "Watch",
                        icon="visibility",
                        on_click=lambda: on_toggle_watch(name),
                        color="green",
                    ).props("dense no-caps").style(
                        "min-height: 0; padding: 1px 6px; font-size: 11px"
                    ).tooltip("Watch this layer (toggle)")
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
        # Sync the icon now in case the page is being rebuilt with a layer
        # that's already in the watched set (e.g. after navigating from
        # `/watch` back to `/`).
        self.refresh_eye()

    def refresh_eye(self) -> None:
        on = self.name in self._session.watched_layers
        self._eye_btn.text = "Unwatch" if on else "Watch"
        self._eye_btn.icon = "visibility_off" if on else "visibility"
        self._eye_btn.props(f"color={'red' if on else 'green'}")

    def compute(
        self, activation: Tensor | None, gradient: Tensor | None, sample_idx: int
    ) -> tuple[str, str]:
        act_png = render_strip(activation, sample_idx)
        grad_png = render_strip(gradient, sample_idx)
        return _img_tag(act_png), _img_tag(grad_png)

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


def _img_tag(png: bytes | None) -> str:
    if png is None:
        return ""
    b64 = base64.b64encode(png).decode("ascii")
    return (
        f'<img src="data:image/png;base64,{b64}" '
        'style="display:block; max-width:none;" />'
    )
