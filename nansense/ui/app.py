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
- Right pane: the "Input Selection" sidebar (see `nansense.ui.input_panel`)
  with the sample spinner, the batch-pinning probe controls, and the input
  image.

A `ui.timer` in each connection polls `session.snapshot` and
`session.probe_result`; when a new one is published, the page re-renders
the *watched* layers' strips against it (the probe — the pinned-batch view
— wins when present). Unwatched layers are never rendered or shipped to
the browser, which is what keeps large models responsive. A separate timer
(owned by the shared step controls, see `nansense.ui.top_bar`) refreshes
the top-bar position label from `session.live_position`, so the displayed
epoch/batch keeps advancing during modes that don't publish a snapshot
every batch (step-epoch, step-custom, run, detach) — a cheap label write,
decoupled from the heavier strip rendering.
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path

import uvicorn
from fastapi import FastAPI
from nicegui import ui

from nansense.session import Session
from nansense.ui.experiment_page import _build_experiment_page
from nansense.ui.graph import build_mermaid
from nansense.ui.main_page import _RenderCache, _build_page
from nansense.ui.watch_page import _build_watch_page
from nansense.ui.weights_page import _build_weights_page


class _DropListenerRerenderWarning(logging.Filter):
    """Drop NiceGUI's browser-relayed "Event listeners changed after initial
    definition" warning — adding listeners to live elements is routine here
    (e.g. cards rebuilt against a new snapshot), and the warning would repeat
    on every such update."""

    def filter(self, record: logging.LogRecord) -> bool:
        return "Event listeners changed after initial definition" not in record.getMessage()


logging.getLogger("nicegui").addFilter(_DropListenerRerenderWarning())


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
    (`nansense.start(..., enabled=False)`), so a training script can call
    `serve()` unconditionally and pay nothing when the UI is turned off —
    and likewise on the non-zero ranks of a distributed run, where the UI
    lives on rank 0.

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
    if not session.is_leader:
        # Non-zero ranks of a distributed run never serve: the UI lives on
        # rank 0, which presents the cross-rank-reduced watch stats. This
        # is also what keeps every rank from fighting over the same port.
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

    ui.run_with(fastapi_app, storage_secret="nansense")

    config = uvicorn.Config(
        app=fastapi_app,
        host=host,
        port=port,
        log_level=log_level,
    )
    server = uvicorn.Server(config)
    setattr(server, "install_signal_handlers", lambda: None)

    thread = threading.Thread(target=server.run, name="nansense-ui", daemon=False)
    thread.start()
    return thread
