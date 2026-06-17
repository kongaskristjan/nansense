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
import time
import webbrowser
from pathlib import Path
from typing import Protocol

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


def _display_url(host: str, port: int) -> str:
    """The address to show a user (and to open in their browser).

    `0.0.0.0` / `::` mean "bind every interface"; they are not routable from
    a browser, so loopback is shown instead.
    """
    shown_host = "127.0.0.1" if host in ("0.0.0.0", "::", "") else host
    return f"http://{shown_host}:{port}"


def _format_box(lines: list[str]) -> str:
    """Frame `lines` in a Unicode box so the address stands out in the busy
    training log it is printed amongst."""
    width = max(len(line) for line in lines)
    top = "┌" + "─" * (width + 2) + "┐"
    bottom = "└" + "─" * (width + 2) + "┘"
    body = [f"│ {line.ljust(width)} │" for line in lines]
    return "\n".join([top, *body, bottom])


def _announce(url: str) -> None:
    """Print the UI address inside a box, padded by blank lines so it is easy
    to spot between training-log lines."""
    box = _format_box(["nansense UI is running at:", url])
    print(f"\n{box}\n", flush=True)


class _Startable(Protocol):
    """The slice of `uvicorn.Server` the browser opener depends on — its
    `started` flag, flipped once startup completes."""

    started: bool


def _open_browser_when_ready(server: _Startable, url: str) -> None:
    """Open `url` in a browser tab once uvicorn has finished starting up.

    Runs on a daemon thread so it never blocks training. Waiting for
    `server.started` keeps the opened tab from racing the port bind (which
    would otherwise show a connection error). `new=2` asks for a new tab and
    `autoraise=True` brings the window to the front (focused) where the
    platform supports it. On a headless box there is no browser:
    `webbrowser.open` just returns `False` there, and any backend error is
    swallowed so a missing display never disrupts the run.
    """
    deadline = time.monotonic() + 10.0
    while not server.started and time.monotonic() < deadline:
        time.sleep(0.05)
    try:
        webbrowser.open(url, new=2, autoraise=True)
    except Exception:
        pass


def serve(
    session: Session,
    *,
    port: int = 8080,
    host: str = "127.0.0.1",
    log_level: str = "warning",
    open_browser: bool = True,
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

    Once the server thread is launched the UI address is printed inside a
    box (so it stands out in the training log) and, unless `open_browser` is
    `False`, opened in a browser tab — on a headless machine the open is a
    harmless no-op.

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
    # Past the guards we are committed to serving: a pause may now wait for the
    # UI indefinitely (an unserved session would instead detach on a timeout).
    session.mark_served()
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
    def watch_page(layer: str = "") -> None:
        _build_watch_page(
            session,
            layer_names,
            layer,
            input_mean=input_mean,
            input_std=input_std,
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

    url = _display_url(host, port)
    _announce(url)
    if open_browser:
        threading.Thread(
            target=_open_browser_when_ready,
            args=(server, url),
            name="nansense-open-browser",
            daemon=True,
        ).start()
    return thread
