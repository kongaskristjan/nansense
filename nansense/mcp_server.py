"""An MCP server that lets a coding agent drive the NaNsense debugger.

The agent's counterpart to the browser UI: same `Session`, same published
snapshots, same control methods — reached over the Model Context Protocol
instead of a web page, so an agent debugging a training script can pause it,
inspect any layer's activations and gradients, and read the numerical-error
report without a human clicking anything.

`build_mount` returns the routes and lifespan for `nansense.ui.app.serve` to
graft onto the UI's own FastAPI app, so the MCP endpoint shares the UI's port
and lifetime (`http://<host>:<port>/mcp`). Nothing here holds session state of
its own: every tool reads the live `Session`, exactly as a page render does.

Two invariants shape every tool below:

- **Never block the event loop.** Uvicorn serves NiceGUI's websockets from the
  same loop, and their keepalive budget is only ~6 s — so anything that waits
  on the training thread (`wait_until_paused`) or copies tensors
  (`current_batch_stats`, `watch_snapshot`) goes through `asyncio.to_thread`,
  the same discipline the `/stats` page follows.
- **A refusal is an answer.** A locked or finished session silently no-ops its
  control methods; tools check for that up front and say so, because an agent
  that gets a plain "ok" from a command that did nothing will loop forever.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator, Callable, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any, Literal

from mcp.server import MCPServer
from starlette.routing import BaseRoute

from nansense.mcp_views import (
    architecture_view,
    debug_view,
    layer_stats_view,
    stats_history_view,
    status_view,
)
from nansense.session import Session, StatsScope

#: Where the MCP endpoint sits on the UI's server.
DEFAULT_MCP_PATH = "/mcp"

# How long a control tool waits for training to reach its next pause before
# reporting back. Long enough that a step over a normal batch answers in one
# call, short enough that a step across a slow epoch returns "still running"
# instead of holding the agent's tool call open.
_DEFAULT_WAIT_SECONDS = 30.0
_MAX_WAIT_SECONDS = 300.0

# Poll interval while waiting for a requested snapshot to be published.
_SNAPSHOT_POLL_SECONDS = 0.05

_INSTRUCTIONS = """\
NaNsense is a PyTorch debugger attached to a live training run. It pauses the
training loop between batches and exposes what every layer is doing: its
activations, its gradients, and statistics over both.

Typical loop: `get_status` to see where the run sits, `get_architecture` once
for the layer names, then `get_layer_stats` on the layers you suspect, and
`step` to advance. `get_debug_report` is the fastest first move when training
diverges — it names the layers where NaN/Inf or subnormal gradients appeared.

Two positions matter and are reported separately: `live_position` is where
training is now, `snapshot_position` is the batch the statistics describe.
They are the same while paused and diverge while the run advances freely.

Statistics come from two places. `get_layer_stats` reads the last captured
batch and works for any layer. `get_stats_history` reads the running
accumulators for the epoch-by-epoch trend, which only cover layers that are
being watched — call `watch_layers` first.
"""


@dataclass(frozen=True)
class McpMount:
    """The pieces `serve` needs to graft the MCP endpoint onto its own app.

    `routes` carries the endpoint (already bound to `path`) and `lifespan` runs
    the transport's session manager. Both must reach the *serving* app: the
    routes ahead of NiceGUI's catch-all mount at `/`, and the lifespan composed
    into the app's own, since a mounted sub-app never receives lifespan events.
    """

    path: str
    routes: Sequence[BaseRoute]
    lifespan: Callable[[Any], Any]


def _clamp_timeout(timeout: float) -> float:
    return max(0.0, min(float(timeout), _MAX_WAIT_SECONDS))


def _control_refusal(session: Session) -> dict[str, Any] | None:
    """Why a run-control command would do nothing here, or `None` if it works."""
    if session.locked:
        return {
            "error": "This session is locked (a shared demo); run controls are disabled.",
            "state": "locked",
        }
    if session.closed:
        return {
            "error": (
                "Training has finished. The last captured batch stays "
                "inspectable, but the run cannot be advanced."
            ),
            "state": "finished",
        }
    return None


def _settings_refusal(session: Session) -> dict[str, Any] | None:
    """Locked sessions pin their settings; say so rather than no-op silently."""
    if session.locked:
        return {
            "error": "This session is locked (a shared demo); settings are disabled.",
            "state": "locked",
        }
    return None


async def _await_pause(session: Session, *, after: int, timeout: float) -> bool:
    """Wait off-loop for the training thread's next pause."""
    return await asyncio.to_thread(
        session.wait_until_paused, after_pauses=after, timeout=timeout
    )


async def _run_control(
    session: Session,
    command: Callable[[], None],
    *,
    timeout: float,
    wait: bool = True,
) -> dict[str, Any]:
    """Issue a control command and report where the run ended up.

    Waiting is skipped when the command does not lead to a pause — `detach`
    never pauses again, and `pause` on an already-paused run has nothing to
    resume, so waiting for a *new* pause would just burn the timeout.
    """
    refusal = _control_refusal(session)
    if refusal is not None:
        return refusal
    before = session.pause_count
    command()
    reached = True
    if wait:
        reached = await _await_pause(
            session, after=before, timeout=_clamp_timeout(timeout)
        )
    view = status_view(session)
    if not reached:
        view["waiting"] = (
            f"Training was still running {timeout:g}s after the command. "
            "Call get_status to check again; the command remains in effect."
        )
    return view


def build_server(
    session: Session, *, mermaid: str | None = None, version: str = ""
) -> MCPServer:
    """An `MCPServer` whose tools drive and inspect `session`.

    `mermaid` is the architecture graph source, built once by the caller (it is
    fixed for the session's lifetime, so re-tracing per call would be waste).
    """
    server = MCPServer(
        name="nansense",
        title="NaNsense training debugger",
        instructions=_INSTRUCTIONS,
        version=version,
    )

    # ---- Orientation -------------------------------------------------

    @server.tool()
    async def get_status() -> dict[str, Any]:
        """Where the training run sits and what is being collected.

        Reports the run state (paused / running / finished), the current mode,
        both positions (live and the snapshot the statistics describe), the
        watched layers, and a one-line summary of any numerical warning.
        Start here.
        """
        return status_view(session)

    @server.tool()
    async def get_architecture(include_graph: bool = True) -> dict[str, Any]:
        """The model's layers, with hyperparameters and parameter names.

        The layer names returned here are the ones every other tool accepts.
        With `include_graph`, also returns the compute graph as Mermaid source
        (from `torch.fx` when the model traces, else the module hierarchy).
        """
        return architecture_view(
            session, mermaid=mermaid if include_graph else None
        )

    # ---- Inspection --------------------------------------------------

    @server.tool()
    async def get_layer_stats(
        layers: list[str], include_histogram: bool = False
    ) -> dict[str, Any]:
        """Activation and gradient statistics for the last captured batch.

        Works for any layer, watched or not. Reports count, shape, mean, std,
        min, max, median, and the dead-channel count (channels whose every
        value landed in the zero band — dead ReLUs). `include_histogram` adds
        the value distribution as `[value, count]` pairs over signed-log bins.

        Non-finite values are reported as the strings "nan", "inf" and "-inf".
        """
        return await asyncio.to_thread(
            layer_stats_view,
            session,
            layers=layers,
            include_histogram=include_histogram,
        )

    @server.tool()
    async def get_stats_history(
        layer: str, phase: str | None = None
    ) -> dict[str, Any]:
        """One layer's statistics per epoch — the trend across the run.

        Reads the running accumulators, so it only covers layers the current
        stats scope collects: call `watch_layers` first (or `set_stats_scope`
        with "all"). Restrict to one phase with `phase`, e.g. "train".
        """
        return await asyncio.to_thread(
            stats_history_view, session, layer=layer, phase=phase
        )

    @server.tool()
    async def get_debug_report() -> dict[str, Any]:
        """The numerical-error debugger: settings and any standing detection.

        Names the layers where NaN/Inf appeared, or where gradients collapsed
        into the subnormal band or saturated toward overflow, with per-layer
        percentages and the dtype-aware band edges. The first call to make when
        a run diverges or stops learning.
        """
        return debug_view(session)

    # ---- Run control -------------------------------------------------

    @server.tool()
    async def step(
        unit: Literal["batch", "phase", "epoch"] = "batch",
        timeout_seconds: float = _DEFAULT_WAIT_SECONDS,
    ) -> dict[str, Any]:
        """Advance training and pause again, then report the new position.

        `batch` pauses on the next batch; `phase` and `epoch` pause on the
        first batch of the next phase / epoch. Returns once training pauses, or
        after `timeout_seconds` with a note that it is still running (the
        command stays in effect — poll `get_status`).
        """
        commands = {
            "batch": session.step_batch,
            "phase": session.step_phase,
            "epoch": session.step_epoch,
        }
        return await _run_control(
            session, commands[unit], timeout=timeout_seconds
        )

    @server.tool()
    async def run(
        timeout_seconds: float = _DEFAULT_WAIT_SECONDS,
    ) -> dict[str, Any]:
        """Run training to its last batch, pausing there.

        Statistics keep accruing and the views refresh on the update cadence,
        but the run will not stop until the end — expect this to return "still
        running" on any real training run, then poll `get_status`. Use `pause`
        to stop earlier.
        """
        return await _run_control(
            session, session.step_run, timeout=timeout_seconds
        )

    @server.tool()
    async def run_until(
        phase: str,
        epoch: int,
        batch: int,
        timeout_seconds: float = _DEFAULT_WAIT_SECONDS,
    ) -> dict[str, Any]:
        """Run until an exact position, then pause there.

        `phase` is a phase name (e.g. "train"), `epoch` and `batch` are
        0-indexed. A position training has already passed will not be matched
        this run; a position it never reaches simply runs to the end.
        """
        refusal = _control_refusal(session)
        if refusal is not None:
            return refusal
        order = session.schedule.phase_order
        if phase not in order:
            return {
                "error": f"Unknown phase {phase!r}.",
                "known_phases": order,
                "hint": (
                    "Phases are learned as training first reaches them; only "
                    "the ones listed here can be targeted yet."
                ),
            }
        return await _run_control(
            session,
            lambda: session.step_until_position(
                phase_index=order.index(phase), epoch=epoch, batch_idx=batch
            ),
            timeout=timeout_seconds,
        )

    @server.tool()
    async def pause(
        timeout_seconds: float = _DEFAULT_WAIT_SECONDS,
    ) -> dict[str, Any]:
        """Stop training on the next batch and capture it.

        A no-op that returns immediately when the run is already paused.
        """
        return await _run_control(
            session,
            session.stop,
            timeout=timeout_seconds,
            wait=session.is_running,
        )

    @server.tool()
    async def detach() -> dict[str, Any]:
        """Let training run to completion without pausing again.

        Capture overhead drops to near zero. Statistics stop refreshing except
        on the update cadence; `pause` re-engages.
        """
        return await _run_control(
            session, session.detach, timeout=0.0, wait=False
        )

    @server.tool()
    async def refresh(timeout_seconds: float = 10.0) -> dict[str, Any]:
        """Publish a fresh snapshot from a freely running run, without pausing.

        Asks the next batch to publish what it already computed, so
        `get_layer_stats` reflects the current weights instead of the batch
        training last paused on. Pointless while paused (the snapshot is
        already current) and reports so.
        """
        if not session.is_running:
            view = status_view(session)
            view["refreshed"] = False
            view["note"] = (
                "Training is not advancing, so the last captured batch is "
                "already the current one."
            )
            return view
        before = session.snapshot
        session.request_snapshot()
        published = await asyncio.to_thread(
            _wait_for_snapshot, session, before, _clamp_timeout(timeout_seconds)
        )
        view = status_view(session)
        view["refreshed"] = published
        if not published:
            view["note"] = (
                "No new snapshot within the timeout — training may be between "
                "batches or moving slowly."
            )
        return view

    # ---- Collection settings ----------------------------------------

    @server.tool()
    async def watch_layers(layers: list[str]) -> dict[str, Any]:
        """Start collecting per-epoch statistics for these layers.

        Needed only for `get_stats_history`; `get_layer_stats` reads any layer
        without watching. Watching makes every batch pay capture cost, so watch
        the layers you are investigating rather than all of them.
        """
        refusal = _settings_refusal(session)
        if refusal is not None:
            return refusal
        unknown = [name for name in layers if not session.watch(name)]
        view = status_view(session)
        if unknown:
            view["unknown_layers"] = unknown
            view["hint"] = "Call get_architecture for the valid layer names."
        return view

    @server.tool()
    async def unwatch_layers(layers: list[str]) -> dict[str, Any]:
        """Stop watching these layers and drop the statistics collected for them."""
        refusal = _settings_refusal(session)
        if refusal is not None:
            return refusal
        for name in layers:
            session.unwatch(name)
        return status_view(session)

    @server.tool()
    async def set_stats_scope(
        scope: Literal["none", "watched", "all"],
    ) -> dict[str, Any]:
        """Choose which layers collect running statistics.

        "watched" (the default) collects for the watched layers, "all" for
        every layer — convenient but it makes every batch collect for the whole
        model — and "none" pauses collection while keeping what was already
        collected browsable.
        """
        refusal = _settings_refusal(session)
        if refusal is not None:
            return refusal
        session.set_stats_scope(StatsScope(scope))
        return status_view(session)

    @server.tool()
    async def configure_debug_checks(
        enabled: bool | None = None,
        interval_batches: int | None = None,
        check_nan_inf: bool | None = None,
        check_under_over: bool | None = None,
        threshold_fraction: float | None = None,
    ) -> dict[str, Any]:
        """Tune the numerical-error debugger; only the given fields change.

        `interval_batches` is the check cadence (1 = every batch, default 100).
        `threshold_fraction` is how much of a layer's summed |gradient| must
        land in the subnormal/overflow band to trip it. Lower the interval to
        catch a divergence closer to where it starts.
        """
        refusal = _settings_refusal(session)
        if refusal is not None:
            return refusal
        session.set_debug_settings(
            enabled=enabled,
            interval=interval_batches,
            check_nan_inf=check_nan_inf,
            check_under_over=check_under_over,
            threshold=threshold_fraction,
        )
        return debug_view(session)

    @server.tool()
    async def silence_debug_check(
        category: Literal["nan_inf", "under_over"],
    ) -> dict[str, Any]:
        """Turn off one check and clear its part of the standing warning.

        The agent equivalent of the banner's "Silence warning" button — use it
        once an issue is understood so later batches stop re-reporting it.
        """
        refusal = _settings_refusal(session)
        if refusal is not None:
            return refusal
        session.disable_debug_check(category)
        return debug_view(session)

    return server


def _wait_for_snapshot(session: Session, previous: object, timeout: float) -> bool:
    """Block until a snapshot other than `previous` is published, or time out.

    Polled rather than event-driven: snapshot publication is a plain atomic
    reference swap on the training thread with no condition-variable signal,
    and this runs on a worker thread where a short sleep costs nothing.
    """
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        current = session.snapshot
        if current is not None and current is not previous:
            return True
        time.sleep(_SNAPSHOT_POLL_SECONDS)
    return False


def build_mount(
    session: Session,
    *,
    mermaid: str | None = None,
    host: str = "127.0.0.1",
    path: str = DEFAULT_MCP_PATH,
    version: str = "",
) -> McpMount:
    """Build the MCP endpoint for `session` as routes plus a lifespan.

    The transport's Starlette app is built only to harvest its route (and the
    DNS-rebinding protection it configures for a loopback `host`); the route is
    then served by the UI's own app, which is what keeps the endpoint on the
    UI's port without a sub-mount's trailing-slash redirect on POST.
    """
    server = build_server(session, mermaid=mermaid, version=version)
    app = server.streamable_http_app(streamable_http_path=path, host=host)
    manager = server.session_manager

    @asynccontextmanager
    async def lifespan(_app: Any) -> AsyncIterator[None]:
        async with manager.run():
            yield

    return McpMount(path=path, routes=list(app.routes), lifespan=lifespan)
