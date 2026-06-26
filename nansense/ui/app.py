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
  layer on every batch) and "Clear all watched layers".
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
import shutil
import threading
import time
import warnings
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
from nansense.ui.stats_page import _build_stats_page
from nansense.ui.weights_page import _build_weights_page


class _DropBenignNiceguiNoise(logging.Filter):
    """Drop two benign NiceGUI log lines that would otherwise spam the run.

    Both originate inside NiceGUI, are harmless here, and would repeat often
    enough to bury real errors:

    - *"Event listeners changed after initial definition"* — a browser-relayed
      warning emitted whenever listeners are added to a live element, which is
      routine here (e.g. cards rebuilt against a new snapshot).

    - *"The parent slot of the element has been deleted"* — an unhandled
      ``RuntimeError`` from NiceGUI's per-connection ``ui.timer`` machinery
      (see this module's docstring on the polling timers). When a client goes
      away — a page reload, a navigation, or an abandoned load that NiceGUI
      later prunes — its element tree is torn down; a timer that was still
      waiting to start (parked in ``Timer._can_start`` on ``client.connected()``)
      is then woken by the client's deletion and walks straight into its
      element's parent-slot context (``Timer._get_context``), whose weakref'd
      slot is already gone. The error belongs to the connection that just left:
      training and every still-connected client keep working. But it is raised
      in NiceGUI's timer loop, *outside* our tick body and even before the
      loop's own ``_should_stop`` guard, so neither our in-tick guards nor a
      disconnect handler can pre-empt it — NiceGUI routes it through its global
      handler (``app.handle_exception`` → ``log.exception``), which logs a full
      traceback. Suppressing that one line is the only client-side lever; the
      durable fix is upstream (guarding the timer's context acquisition).
    """

    _BENIGN: tuple[str, ...] = (
        "Event listeners changed after initial definition",
        "The parent slot of the element has been deleted",
    )

    def filter(self, record: logging.LogRecord) -> bool:
        message = record.getMessage()
        return not any(noise in message for noise in self._BENIGN)


logging.getLogger("nicegui").addFilter(_DropBenignNiceguiNoise())


def _silence_reduce_op_future_warning() -> None:
    """Suppress the spurious `torch.distributed.reduce_op` FutureWarning.

    On startup NiceGUI walks `gc.get_objects()` to find the running uvicorn
    server, `isinstance`-checking every live object. PyTorch keeps a deprecated
    module-level instance, `torch.distributed.reduce_op`, whose
    `__getattribute__` emits a FutureWarning on *any* attribute access — and
    `isinstance` reads `__class__`. So the moment we serve (with
    `torch.distributed` imported) the walk trips a warning that is neither ours
    nor actionable. Silence just that message; everything else still surfaces.
    """
    warnings.filterwarnings(
        "ignore",
        message=r"`torch\.distributed\.reduce_op` is deprecated",
        category=FutureWarning,
    )


_silence_reduce_op_future_warning()


def _display_url(host: str, port: int) -> str:
    """The address to show a user (and to open in their browser).

    `0.0.0.0` / `::` mean "bind every interface"; they are not routable from
    a browser, so loopback is shown instead.
    """
    shown_host = "127.0.0.1" if host in ("0.0.0.0", "::", "") else host
    return f"http://{shown_host}:{port}"


def _format_box(lines: list[str], width: int) -> str:
    """Frame `lines` in a Unicode box `width` columns wide so the address
    stands out in the busy training log it is printed amongst.

    The interior is widened to span `width` (one space of padding on each
    side of the text), but never shrinks below the longest line.
    """
    inner = max(width - 4, max(len(line) for line in lines))
    rule = "─" * (inner + 2)
    body = [f"│ {line.ljust(inner)} │" for line in lines]
    return "\n".join([f"┌{rule}┐", *body, f"└{rule}┘"])


def _announce(url: str) -> None:
    """Print the UI address inside a box that spans the terminal width (a
    sensible default when output is redirected), padded by blank lines so it
    is easy to spot between training-log lines."""
    width = shutil.get_terminal_size().columns
    box = _format_box(["nansense UI is running at:", url], width)
    print(f"\n{box}\n", flush=True)


class _Startable(Protocol):
    """The slice of `uvicorn.Server` the announcer reads — its `started`
    flag, flipped once the port is bound and serving has begun."""

    started: bool


def _announce_when_ready(
    server: _Startable,
    server_thread: threading.Thread,
    url: str,
    open_browser: bool,
    *,
    timeout: float = 10.0,
) -> None:
    """Announce the UI — and, if `open_browser`, open a focused browser tab —
    but only once the server has actually bound its port.

    Runs on a daemon thread so it never blocks training. The wait is what
    makes this safe under a concurrent session: if another session already
    holds the port, uvicorn's bind fails on its own thread (it logs the
    `[Errno 98]` error and exits), so `server.started` never flips and that
    thread dies. We notice both, print nothing, and open nothing — no banner
    promising a URL we don't own, and no tab racing to a page served by the
    *other* session. On a clean bind we announce and (where supported) open
    the tab focused: `new=2` requests a new tab, `autoraise=True` raises it.
    On a headless box `webbrowser.open` is a harmless no-op, and any backend
    error is swallowed so a missing display never disrupts the run.
    """
    deadline = time.monotonic() + timeout
    while (
        not server.started
        and server_thread.is_alive()
        and time.monotonic() < deadline
    ):
        time.sleep(0.05)
    if not server.started:
        return  # a concurrent session holds the port — stay silent
    _announce(url)
    if open_browser:
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

    Once the server thread is launched, a daemon thread waits for the port to
    bind and then prints the UI address inside a box (so it stands out in the
    training log) and, unless `open_browser` is `False`, opens it in a focused
    browser tab. If a concurrent session already holds the port the bind
    fails, so the banner and the browser tab are both suppressed — only
    uvicorn's own `address already in use` error is shown. On a headless
    machine the bind still succeeds, so the banner prints and the browser open
    is a harmless no-op.

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
    favicon_path = (
        Path(__file__).resolve().parents[2] / "assets" / "logo" / "logo_small.png"
    )
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

    @ui.page("/stats", favicon=str(favicon_path))
    def stats_page(layer: str = "", phase: str = "") -> None:
        _build_stats_page(
            session,
            layer_names,
            layer,
            start_current_batch=phase == "current",
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

    threading.Thread(
        target=_announce_when_ready,
        args=(server, thread, _display_url(host, port), open_browser),
        name="nansense-announce",
        daemon=True,
    ).start()
    return thread
