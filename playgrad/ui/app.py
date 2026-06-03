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

import plotly.graph_objects as go
import uvicorn
from fastapi import FastAPI
from nicegui import ui
from torch import Tensor

from playgrad.schedule import Schedule
from playgrad.session import BatchSnapshot, Session
from playgrad.ui.graph import build_mermaid, slug
from playgrad.ui.render import render_image, render_strip
from playgrad.watch import (
    BINS_PER_DECADE,
    LOG10_MAX,
    LOG10_MIN,
    N_BINS,
    ZERO_BIN,
    LayerStatsSnapshot,
    TensorStatsSnapshot,
    WatchSnapshot,
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
    // The eye toggle inside a card handles its own click; don't navigate.
    if (e.target.closest('[data-watch-toggle]')) return;
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
  window.playgradScrollCardToTop = function(slug) {
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
) -> threading.Thread:
    """Start the NiceGUI app on a background thread and return that thread.

    NiceGUI is mounted onto a bare FastAPI app via `ui.run_with`; the app is
    then served by uvicorn from a non-main thread, with signal handlers
    disabled so uvicorn doesn't try to wire SIGINT/SIGTERM from a thread
    that isn't the main one.

    `input_mean` / `input_std` are passed to the input-image pane so the
    sample is denormalized (`x * std + mean`) before display. When either
    is `None`, the renderer assumes the input is already in `[0, 1]`.
    """
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
    ui.add_body_html(_ARCHITECTURE_CLICK_JS)

    step_until_custom = _build_step_until_custom_dialog(session)

    with ui.column().classes("w-full h-screen no-wrap gap-0"):
        with ui.row().classes(
            "w-full items-center gap-x-3 gap-y-0 px-3 py-2 shrink-0 "
            "border-b-2 border-slate-300 bg-slate-100 shadow-sm z-10"
        ):
            architecture_toggle = ui.button(
                icon="account_tree", color="slate-500"
            ).props("dense size=md").tooltip("Toggle architecture pane")
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
            position_label = ui.label("(waiting for first snapshot)").classes("ml-3 font-mono text-sm")
            ui.label("Viewing sample:").classes("ml-3 text-sm")
            sample_input = ui.number(value=0, min=0, step=1, format="%d").classes("w-20").props("dense")
            watch_chip = ui.button(
                str(len(session.watched_layers)),
                icon="visibility",
                color="slate-100",
            ).classes(
                "ml-auto text-amber-700 font-mono"
            ).props("dense size=md no-caps").tooltip(
                "Watched layers — click to open the watch view or jump to a card"
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
                            f"window.playgradScrollCardToTop({json.dumps(slug(n))})"
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
            with ui.column().classes(
                "grow min-w-0 h-full overflow-auto p-3 bg-slate-200 gap-3"
            ):
                for name in layer_names:
                    layer_views[name] = _LayerView(
                        name,
                        session=session,
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
            position_label.text = (
                f"epoch {live.epoch} | {live.phase} batch {live.batch_idx}"
            )
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


def _make_histogram_figure(
    per_phase: dict[str, LayerStatsSnapshot],
    kind: str,
    title: str,
) -> go.Figure:
    """Plotly bar chart of the signed-log histogram, one trace per phase.

    `kind` selects which of the two histograms on each `LayerStatsSnapshot`
    to plot ("activation" or "gradient"). `per_phase` may be empty (initial
    render before any data has been collected) — the figure is still
    returned, just with no traces.
    """
    tick_vals, tick_text = _x_tick_layout()
    x_indices = list(range(N_BINS))
    fig = go.Figure()
    has_data = False
    for i, (phase, layer_snap) in enumerate(per_phase.items()):
        stats = (
            layer_snap.activations
            if kind == "activation"
            else layer_snap.gradients
        )
        if stats.n == 0:
            continue
        has_data = True
        fig.add_trace(
            go.Bar(
                x=x_indices,
                y=list(stats.hist),
                name=f"{phase} (ep {layer_snap.epoch})",
                marker_color=_phase_color(phase, i),
                opacity=0.55 if len(per_phase) > 1 else 0.85,
                hovertemplate="bin %{x}<br>count %{y}<extra></extra>",
            )
        )
    fig.update_layout(
        title=dict(text=title, x=0.0, font=dict(size=12)),
        barmode="overlay",
        bargap=0,
        margin=dict(l=50, r=20, t=40, b=40),
        height=220,
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
    fig.update_xaxes(
        tickvals=tick_vals,
        ticktext=tick_text,
        tickfont=dict(size=9),
        showgrid=False,
        zeroline=False,
    )
    fig.update_yaxes(
        type="log",
        showgrid=True,
        gridcolor="#e2e8f0",
        tickfont=dict(size=9),
        title=dict(text="count", font=dict(size=10)),
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

    with ui.column().classes("w-full h-screen no-wrap gap-0"):
        with ui.row().classes(
            "w-full items-center gap-x-3 gap-y-0 px-3 py-2 shrink-0 "
            "border-b-2 border-slate-300 bg-slate-100 shadow-sm z-10"
        ):
            ui.button(
                icon="arrow_back",
                on_click=lambda: ui.navigate.to("/"),
                color="slate-500",
            ).props("dense size=md").tooltip("Back to the main page")
            ui.label("Watching").classes("font-mono text-base font-bold ml-2")
            count_label_holder["count"] = ui.label("").classes(
                "text-sm text-slate-500 ml-2"
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


class _WatchLayerPanel:
    """One card per watched layer — activations + gradients histograms."""

    def __init__(
        self,
        *,
        name: str,
        session: Session,
        on_unwatched: Callable[[], None],
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
                self._act_plot = ui.plotly(
                    _make_histogram_figure({}, "activation", "activations")
                ).classes("w-full")
                ui.label("Gradients").classes(
                    "font-mono text-sm text-slate-600"
                )
                self._grad_stats = ui.label("no data yet").classes(
                    "font-mono text-xs text-slate-500"
                )
                self._grad_plot = ui.plotly(
                    _make_histogram_figure({}, "gradient", "gradients")
                ).classes("w-full")

    def update(self, snap: WatchSnapshot) -> None:
        per_phase = snap.latest_per_phase(self.name)
        self._act_stats.text = _combined_summary(per_phase, "activation")
        self._grad_stats.text = _combined_summary(per_phase, "gradient")
        self._act_plot.figure = _make_histogram_figure(
            per_phase, "activation", "activations"
        )
        self._act_plot.update()
        self._grad_plot.figure = _make_histogram_figure(
            per_phase, "gradient", "gradients"
        )
        self._grad_plot.update()


def _combined_summary(
    per_phase: dict[str, LayerStatsSnapshot], kind: str
) -> str:
    if not per_phase:
        return "no data yet"
    parts: list[str] = []
    for phase, layer_snap in per_phase.items():
        stats = (
            layer_snap.activations
            if kind == "activation"
            else layer_snap.gradients
        )
        parts.append(f"[{phase} ep{layer_snap.epoch}] {_stats_summary(stats)}")
    return "    ".join(parts)


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
    """

    def __init__(
        self,
        name: str,
        *,
        session: Session,
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
                # The wrapper carries `data-watch-toggle` so the
                # document-level click handler skips card→diagram
                # navigation when the eye button is clicked. Quasar's q-btn
                # doesn't reliably pass arbitrary `data-*` attrs through to
                # its rendered DOM, so the attribute lives on this div.
                with ui.element("div").props("data-watch-toggle"):
                    self._eye_btn = ui.button(
                        "Watch",
                        icon="visibility",
                        on_click=lambda: on_toggle_watch(name),
                        color="green",
                    ).props("dense no-caps").style(
                        "min-height: 0; padding: 1px 6px; font-size: 11px"
                    ).tooltip("Watch this layer (toggle)")
            with ui.element("div").classes("w-full overflow-x-auto p-2"):
                self.act_html = ui.html("")
                ui.element("div").classes("h-1")
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
        act_png = render_strip(activation, sample_idx, kind="activation")
        grad_png = render_strip(gradient, sample_idx, kind="gradient")
        return _img_tag(act_png), _img_tag(grad_png)

    def apply(self, act_html: str, grad_html: str) -> None:
        self.act_html.set_content(act_html)
        self.grad_html.set_content(grad_html)


def _img_tag(png: bytes | None) -> str:
    if png is None:
        return ""
    b64 = base64.b64encode(png).decode("ascii")
    return (
        f'<img src="data:image/png;base64,{b64}" '
        'style="display:block; max-width:none;" />'
    )
