"""Lazy entry points for the web UI and standalone tensor rendering."""

from __future__ import annotations

import threading
from importlib import import_module
from typing import TYPE_CHECKING

from nansense.input_config import InputTransform, MeanStd
from nansense.session import Session

if TYPE_CHECKING:
    from nansense.ui.graph import build_mermaid
    from nansense.ui.render import (
        RenderOptions,
        StripRender,
        render_image,
        render_strip,
    )

__all__ = [
    "RenderOptions",
    "StripRender",
    "build_mermaid",
    "render_image",
    "render_strip",
    "serve",
]


def __getattr__(name: str) -> object:
    if name == "build_mermaid":
        module = "nansense.ui.graph"
    elif name in {"RenderOptions", "StripRender", "render_image", "render_strip"}:
        module = "nansense.ui.render"
    else:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(module), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))


def serve(
    session: Session,
    *,
    port: int = 8080,
    host: str = "127.0.0.1",
    log_level: str = "warning",
    open_browser: bool = True,
    mcp: bool = True,
    input_mean: MeanStd | dict[str, MeanStd] | None = None,
    input_std: MeanStd | dict[str, MeanStd] | None = None,
    input_transform: InputTransform | dict[str, InputTransform] | None = None,
) -> threading.Thread | None:
    """Start the NiceGUI app on a background thread and return that thread.

    `port` / `host` pick the bind address (default `127.0.0.1:8080`).
    `log_level` is uvicorn's log level — `"warning"` by default, so routine
    request logging stays out of the training console.

    Returns `None` without starting anything when `session` is disabled
    (`nansense.start(..., enabled=False)`), so a training script can call
    `serve()` unconditionally and pay nothing when the UI is turned off —
    and likewise on the non-zero ranks of a distributed run, where the UI
    lives on rank 0.

    Web dependencies and server-specific noise filters load only when an
    enabled leader session is served.

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

    `mcp` (default `True`) also serves the MCP endpoint at `/mcp` on the same
    port, so a coding agent can drive the debugger through the same session the
    browser shows (`nansense.mcp_server`). Its route is registered *before*
    NiceGUI's catch-all mount at `/` — Starlette matches routes in order — and
    its lifespan is passed to the app at construction, since NiceGUI wraps
    whatever lifespan it finds and a mounted sub-app never receives one.

    `input_mean` / `input_std` are passed to the input-image pane so the
    sample is denormalized (`x * std + mean`) before display. When either
    is `None`, the renderer assumes the input is already in `[0, 1]`.
    `input_transform` maps a non-RGB input to a displayable 1-/3-channel
    image. Each of the three is either a single value applied to every input,
    or a `dict` keyed by input name for a multi-input model (see
    `nansense.input_config`); the stats and experiment panes use the primary
    input's resolved values, and so does the MCP server, whose image tools
    render the same views.
    """
    if not session.enabled or not session.is_leader:
        return None

    from nansense.ui.app import serve as start_server

    return start_server(
        session,
        port=port,
        host=host,
        log_level=log_level,
        open_browser=open_browser,
        mcp=mcp,
        input_mean=input_mean,
        input_std=input_std,
        input_transform=input_transform,
    )
