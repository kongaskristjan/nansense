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
import json
import time
from collections.abc import AsyncIterator, Callable, Iterable, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any, Literal

from mcp.server import MCPServer
from starlette.routing import BaseRoute

from nansense import experiments
from nansense.input_config import InputDisplay, InputTransform, MeanStd
from nansense.mcp_images import (
    bin_samples_image,
    experiment_image,
    histogram_image,
    image_reply,
    input_image,
    layer_image,
    patches_image,
    weights_image,
)
from nansense.mcp_views import (
    architecture_view,
    default_phase,
    debug_view,
    experiment_catalog_view,
    experiment_result_view,
    layer_stats_view,
    metrics_view,
    probe_view,
    recordings_view,
    settings_view,
    stats_history_view,
    status_view,
    time_travel_view,
    weight_stats_view,
)
from nansense.patches import PATCH_TYPES
from nansense.restore import TimeTravelError
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

# Poll interval while waiting for one experiment request to finish. Coarser
# than the snapshot poll: an experiment takes seconds to minutes, and its
# progress publishes are already visible to `get_experiment_result`.
_EXPERIMENT_POLL_SECONDS = 0.2

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

The `render_*` tools return the same views as pictures — the strips, weight
maps, histograms and patch grids the browser draws. Statistics tell you how
big a problem is; a picture tells you where in the tensor it lives (one dead
channel, a saturated edge, a stuck kernel), which is often not visible in a
mean. Reach for one when the numbers say something is wrong but not what.
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
    wait: bool | None = True,
) -> dict[str, Any]:
    """Issue a control command and report where the run ended up.

    Waiting is skipped when the command does not lead to a pause — `detach`
    never pauses again, and `pause` on an already-paused run has nothing to
    resume, so waiting for a *new* pause would just burn the timeout. `None`
    defers that decision to here, where it can be made against the same
    `pause_count` the wait will use.
    """
    refusal = _control_refusal(session)
    if refusal is not None:
        return refusal
    before = session.pause_count
    # `wait=None` means "decide now": sampling `is_running` at the call site
    # would read it *before* `pause_count`, and a pause landing in between
    # would leave us waiting for a second one that never comes.
    should_wait = session.is_running if wait is None else wait
    command()
    reached = True
    waited = _clamp_timeout(timeout)
    if should_wait:
        reached = await _await_pause(session, after=before, timeout=waited)
    view = status_view(session)
    if not reached:
        view["waiting"] = (
            f"Training was still running {waited:g}s after the command. "
            "Call get_status to check again; the command remains in effect."
        )
    return view


def build_server(
    session: Session,
    *,
    mermaid: str | None = None,
    version: str = "",
    input_mean: MeanStd | dict[str, MeanStd] | None = None,
    input_std: MeanStd | dict[str, MeanStd] | None = None,
    input_transform: InputTransform | dict[str, InputTransform] | None = None,
) -> MCPServer:
    """An `MCPServer` whose tools drive and inspect `session`.

    `mermaid` is the architecture graph source, built once by the caller (it is
    fixed for the session's lifetime, so re-tracing per call would be waste).

    `input_mean` / `input_std` / `input_transform` are `serve`'s, and are used
    for the same thing: turning an input tensor back into a viewable image for
    the `render_*` tools. They live on the training script rather than on the
    session, so they have to be handed down here too.
    """
    server = MCPServer(
        name="nansense",
        title="NaNsense training debugger",
        instructions=_INSTRUCTIONS,
        version=version,
    )
    display = InputDisplay(
        mean=input_mean, std=input_std, transform=input_transform
    )
    input_names = session.input_names
    primary_input = input_names[0] if input_names else None

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

    # ---- Rendered views ----------------------------------------------

    @server.tool(structured_output=False)
    async def render_layer(
        layers: list[str],
        sample: int = 0,
        include_input: bool = False,
    ) -> list[Any]:
        """Picture of what these layers computed on the last captured batch.

        One row of per-channel tiles per layer for its activations and another
        for its gradients, on a shared symmetric scale — red positive, blue
        negative, NaN/±Inf left as transparent holes over a checkerboard. This
        is where a dead channel, a saturated border or a single diverged feature
        map is obvious in a way a mean never is. `sample` picks the sample
        within the batch; `include_input` adds the input image above.
        """
        return image_reply(
            await asyncio.to_thread(
                layer_image,
                session,
                layers=layers,
                sample=sample,
                display=display,
                input_name=primary_input,
                include_input=include_input,
            )
        )

    @server.tool(structured_output=False)
    async def render_input(
        sample: int = 0, input_name: str | None = None
    ) -> list[Any]:
        """Picture of one sample of the model's input.

        Denormalized with the statistics the training script passed to `serve`.
        `input_name` selects among a multi-input model's inputs (default: the
        first, which is the one the pages call primary).
        """
        return image_reply(
            await asyncio.to_thread(
                input_image,
                session,
                sample=sample,
                display=display,
                input_name=input_name or primary_input,
            )
        )

    @server.tool(structured_output=False)
    async def render_weights(
        layer: str,
        parameters: list[str] | None = None,
        index: int = 0,
        x_dim: int | None = None,
        y_dim: int | None = None,
        tile_dim: int | None = None,
    ) -> list[Any]:
        """Picture of a layer's parameters, gradients and optimizer state.

        Conv kernels are drawn as `kH×kW` tiles laid out across the input
        channels, so a filter that has gone flat or saturated shows up
        directly; a linear weight is one `[out, in]` image. Optimizer state
        shaped like the parameter is drawn the same way (Adam's `exp_avg`
        beside the weight it moves), and scalar state and hyperparameters —
        including the live learning rate — come back as a text line.

        Defaults to every parameter of the layer. `index` pins the axes the
        layout does not show, which for a conv weight is the output channel:
        raise it to page through filters. `x_dim` / `y_dim` / `tile_dim`
        re-assign the axes themselves when the default view is the wrong cut.
        """
        return image_reply(
            await asyncio.to_thread(
                weights_image,
                session,
                layer=layer,
                parameters=parameters,
                index=index,
                x_dim=x_dim,
                y_dim=y_dim,
                tile_dim=tile_dim,
            )
        )

    @server.tool(structured_output=False)
    async def render_histogram(
        layers: list[str],
        phase: str | None = None,
        log_x: bool = False,
        log_y: bool = False,
    ) -> list[Any]:
        """Picture of the value distributions of watched layers.

        Activations and gradients, one subplot each, over the signed-log bins.
        Shape is the point: a gradient histogram collapsing toward zero, a
        bimodal activation, a spike in the overflow bin. Covers watched layers
        only — call `watch_layers` first. `log_x` spreads the bins evenly by
        magnitude, `log_y` reveals sparse tails. Defaults to the newest phase
        with data.
        """
        return image_reply(
            await asyncio.to_thread(
                histogram_image,
                session,
                layers=layers,
                phase=phase,
                log_x=log_x,
                log_y=log_y,
            )
        )

    @server.tool(structured_output=False)
    async def render_extreme_patches(
        layer: str,
        phase: str | None = None,
        grids: list[Literal["max_pixel", "min_pixel", "max_average", "min_average"]]
        | None = None,
        heatmap: bool = False,
    ) -> list[Any]:
        """Picture of the inputs that most excite (or least excite) each channel.

        Columns are channels, rows the top-scoring samples for that channel —
        the view that answers "what has this unit learned to look for", and the
        one that shows a channel responding to nothing at all. Needs a watched
        layer and an image-like input. `heatmap` blends the channel's activation
        map over each patch; the `*_average` grids need
        `set_watch_performance(average_patches=True)`.
        """
        return image_reply(
            await asyncio.to_thread(
                patches_image,
                session,
                layer=layer,
                phase=phase,
                grids=PATCH_TYPES if grids is None else grids,
                heatmap=heatmap,
                display=display,
                input_name=primary_input,
            )
        )

    @server.tool(structured_output=False)
    async def render_bin_samples(
        layer: str,
        channel: int,
        value: float,
        kind: Literal["activation", "gradient"] = "activation",
        count: int = 4,
    ) -> list[Any]:
        """Picture of the inputs behind one histogram bar.

        A histogram says how many values fell in a bin; this says *which* —
        random elements of `(layer, channel)` near `value`, each with the input
        crop around where it came from. The way to get from "there is a spike
        in the overflow bin" to "these three inputs cause it".

        Pass a value straight from the `histogram` pairs of
        `get_layer_stats(include_histogram=True)` (or `render_histogram`); it
        is snapped to the bin it came from. Sampled from the last captured
        batch only — the one whose activations and input still exist — while
        the bar itself may count a whole epoch.
        """
        rendered, values = await asyncio.to_thread(
            bin_samples_image,
            session,
            layer=layer,
            channel=channel,
            value=value,
            kind=kind,
            count=count,
            display=display,
            input_name=primary_input,
        )
        reply = image_reply(rendered)
        if values:
            reply.insert(1, json.dumps({"samples": values}))
        return reply

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
            session, session.stop, timeout=timeout_seconds, wait=None
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

    @server.tool()
    async def get_settings() -> dict[str, Any]:
        """The visualization-update and watch-memory knobs, and their meaning."""
        return settings_view(session)

    @server.tool()
    async def set_update_frequency(
        unit: Literal["epoch", "batch"] = "epoch",
        n: int = 1,
        phase: str | None = None,
    ) -> dict[str, Any]:
        """How often the views refresh while training runs, without pausing.

        Each update publishes a snapshot, re-runs the probe and any auto
        experiments, and appends a frame to every recording — so this is also
        a recording's frame rate. `phase` restricts the batch count to one
        phase and only applies with `unit="batch"`.
        """
        refusal = _settings_refusal(session)
        if refusal is not None:
            return refusal
        try:
            session.set_update_frequency(unit=unit, n=n, phase=phase)
        except ValueError as error:
            return {"error": str(error), "known_phases": session.schedule.phase_order}
        return settings_view(session)

    @server.tool()
    async def set_watch_performance(
        channel_limit_enabled: bool | None = None,
        channel_limit: int | None = None,
        samples_per_channel: int | None = None,
        average_patches: bool | None = None,
    ) -> dict[str, Any]:
        """Bound what watching a layer costs in memory; only given fields change.

        Watched layers keep a histogram and a gallery of extreme input patches
        *per channel*, so cost scales with the channel count.
        `channel_limit` caps the per-channel data to the first N channels (the
        layer-wide histogram and scalars always cover all of them);
        `average_patches` enables the whole-input grids, off by default.

        Every one of these fixes a buffer shape, so changing any of them
        **discards all statistics collected so far** and starts over.
        """
        refusal = _settings_refusal(session)
        if refusal is not None:
            return refusal
        flushed = session.set_watch_performance(
            channel_limit_enabled=channel_limit_enabled,
            channel_limit=channel_limit,
            samples_per_channel=samples_per_channel,
            average_patches=average_patches,
        )
        view = settings_view(session)
        view["statistics_flushed"] = flushed
        if flushed:
            view["note"] = (
                "The buffer shapes changed, so every collected statistic was "
                "dropped. Let training advance before reading them again."
            )
        return view

    @server.tool()
    async def set_auto_run_experiments(enabled: bool) -> dict[str, Any]:
        """Whether open experiment *pages* re-run on every parameter change.

        A session-wide preference shared with the browser. Worth turning off
        while working: an auto-running page competes for the same paused
        training thread your own `run_experiment` needs.
        """
        refusal = _settings_refusal(session)
        if refusal is not None:
            return refusal
        session.set_auto_run_experiments(enabled)
        return settings_view(session)

    @server.tool()
    async def get_weight_stats(
        layer: str, parameters: list[str] | None = None
    ) -> dict[str, Any]:
        """A layer's parameters as numbers: values, gradients, optimizer state.

        Covers what `get_layer_stats` does not: the parameters themselves
        rather than what they produced. Includes each parameter's optimizer
        state (Adam's `exp_avg` / `exp_avg_sq`, SGD's `momentum_buffer`) and
        the live hyperparameters — so a learning rate a scheduler has driven to
        zero, or a second-moment estimate that has blown up, is visible here.
        """
        return await asyncio.to_thread(
            weight_stats_view, session, layer=layer, parameters=parameters
        )

    @server.tool()
    async def get_metrics(layers: list[str] | None = None) -> dict[str, Any]:
        """Custom scalar metrics the training script registered.

        These come from `session.watch_metric(...)` in the user's own code, so
        what exists here is specific to this project — and worth reading early,
        since a metric someone bothered to add usually marks what they were
        worried about.
        """
        return await asyncio.to_thread(metrics_view, session, layers=layers)

    # ---- Time travel -------------------------------------------------

    @server.tool()
    async def get_time_travel_status() -> dict[str, Any]:
        """Whether the run can jump back to an earlier epoch, and to which."""
        return time_travel_view(session)

    @server.tool()
    async def time_travel(
        epoch: int, timeout_seconds: float = _DEFAULT_WAIT_SECONDS
    ) -> dict[str, Any]:
        """Restart training at the beginning of `epoch` and pause there.

        Restores the model, optimizer and scheduler state checkpointed at that
        epoch's start, so the run continues from there and everything after it
        is discarded. This is how you get *back* to the batch before a
        divergence once you have run past it — with `configure_debug_checks`
        tightened, or a layer newly watched, so the second pass sees what the
        first one missed. Needs the training loop to be driven by
        `session.epochs()` with `session.restore_point()`.
        """
        before = session.pause_count
        try:
            session.request_time_travel(epoch)
        except TimeTravelError as error:
            view = time_travel_view(session)
            view["error"] = str(error)
            return view
        reached = await _await_pause(
            session, after=before, timeout=_clamp_timeout(timeout_seconds)
        )
        view = status_view(session)
        view["travelled_to_epoch"] = epoch
        if not reached:
            view["waiting"] = (
                "The jump is armed but training has not reached a batch "
                "boundary yet. Poll get_status."
            )
        return view

    # ---- Probes and perturbations ------------------------------------

    @server.tool()
    async def get_probe_status() -> dict[str, Any]:
        """What fixed input the model is being re-run on, and how."""
        return probe_view(session)

    @server.tool()
    async def pin_batch(timeout_seconds: float = 10.0) -> dict[str, Any]:
        """Hold the current input and re-run the model on it at every capture.

        Stepping then shows how the network's response to one *constant*
        stimulus evolves as the weights change, instead of confounding that
        with a different batch each step. Probes are forward-only, so pinned
        views have activations but no gradients.
        """
        refusal = _probe_refusal(session)
        if refusal is not None:
            return refusal
        before = session.probe_count
        if not session.pin_current_batch():
            view = probe_view(session)
            view["error"] = (
                "Nothing to pin: no batch has been captured yet, or the "
                "snapshot carries no input tensor."
            )
            return view
        return await _probe_result(session, after=before, timeout=timeout_seconds)

    @server.tool()
    async def unpin_batch(timeout_seconds: float = 10.0) -> dict[str, Any]:
        """Release the pinned input; captures go back to showing the live batch."""
        refusal = _probe_refusal(session)
        if refusal is not None:
            return refusal
        before = session.probe_count
        session.unpin_batch()
        return await _probe_result(session, after=before, timeout=timeout_seconds)

    @server.tool()
    async def set_probe_mode(
        mode: Literal["unchanged", "eval", "train"],
        timeout_seconds: float = 10.0,
    ) -> dict[str, Any]:
        """Choose the train/eval mode probe forwards run under.

        `"eval"` is the useful one: BatchNorm uses its running statistics and
        dropout is off, which is how the model will actually behave at
        inference — and a model that looks healthy in train mode and broken in
        eval usually has BatchNorm statistics that have drifted. Selecting a
        mode activates probing on its own, no pin required. Training state is
        always restored afterwards.
        """
        refusal = _probe_refusal(session)
        if refusal is not None:
            return refusal
        before = session.probe_count
        # Re-selecting the mode already in force returns early inside the
        # session and arms no probe; waiting for one would burn the timeout.
        rearmed = session.probe_mode != mode
        session.set_probe_mode(mode)
        return await _probe_result(
            session, after=before, timeout=timeout_seconds, expected=rearmed
        )

    @server.tool()
    async def add_perturbation(
        index: list[int],
        values: list[float],
        sample: int = 0,
        input_name: str | None = None,
        timeout_seconds: float = 10.0,
    ) -> dict[str, Any]:
        """Pin one position of the probe input to fixed values and re-run.

        Counterfactuals on a paused model: edit a pixel, see which layers move.
        `index` is `[y, x]` for an image input (with `values` its length-`C`
        channel vector) or `[channel]` for a flat one (a single value).
        `values` are in the model's own input space — already normalized — so
        `get_layer_stats` on the input layer tells you the range to aim at.

        While any perturbation is active the layer strips switch to showing
        `perturbed − original`, which is what makes the affected path visible.
        """
        refusal = _probe_refusal(session)
        if refusal is not None:
            return refusal
        target = input_name or primary_input or ""
        # Check the edit *before* recording it. A misfit is skipped silently at
        # apply time and stays in the map, so afterwards nothing distinguishes
        # "your edit was dropped" from "someone else's edit landed" — and the
        # agent would be told its perturbation is active when it is inert.
        misfit = _perturbation_misfit(
            session, input_name=target, sample=sample, index=index, values=values
        )
        if misfit is not None:
            view = probe_view(session)
            view["error"] = misfit
            return view
        before = session.probe_count
        session.add_perturbation(
            input_name=target,
            sample=sample,
            index=tuple(index),
            values=tuple(values),
        )
        return await _probe_result(session, after=before, timeout=timeout_seconds)

    @server.tool()
    async def clear_perturbations(timeout_seconds: float = 10.0) -> dict[str, Any]:
        """Drop every perturbation; the probe goes back to the unedited input."""
        refusal = _probe_refusal(session)
        if refusal is not None:
            return refusal
        before = session.probe_count
        session.clear_perturbations()
        return await _probe_result(session, after=before, timeout=timeout_seconds)

    # ---- Experiments -------------------------------------------------

    @server.tool()
    async def list_experiments() -> dict[str, Any]:
        """Every experiment kind, what it shows, and the knobs it takes.

        Call before `run_experiment` — the parameter keys, defaults and the
        layers each kind accepts all come from here.
        """
        return await asyncio.to_thread(experiment_catalog_view, session)

    @server.tool()
    async def run_experiment(
        kind: Literal[
            "deep_dream", "gradcam", "neuron_gradient", "neuron_ig", "occlusion"
        ],
        layer: str,
        params: dict[str, Any] | None = None,
        timeout_seconds: float = 120.0,
    ) -> dict[str, Any]:
        """Run an interpretability experiment on the paused model.

        Deep dream synthesizes the input that most excites each channel — what
        the unit "wants" to see; the Captum methods attribute a prediction back
        onto the input. Both answer questions statistics cannot: whether a
        channel has learned anything, and what the model is actually looking at.

        Runs on the training thread, so **training must be paused** — otherwise
        it is queued until the next pause. Returns once the run finishes,
        `timeout_seconds` elapses, or the server's own wall-clock ceiling stops
        it. `render_experiment` draws the result.
        """
        return await _run_experiment(
            session,
            kind=kind,
            layer=layer,
            params=params,
            display=display,
            input_name=primary_input,
            timeout=timeout_seconds,
        )

    @server.tool()
    async def get_experiment_result(seq: int) -> dict[str, Any]:
        """The latest progress or outcome published for one experiment request."""
        return experiment_result_view(session, seq=seq)

    @server.tool()
    async def cancel_experiment(seq: int | None = None) -> dict[str, Any]:
        """Cancel one queued or running experiment, or every one when `seq` is
        omitted. A running experiment stops at its next abort check."""
        session.cancel_experiment(seq)
        return {"cancelled": "all" if seq is None else seq}

    @server.tool(structured_output=False)
    async def render_experiment(seq: int) -> list[Any]:
        """Picture of an experiment's result: the synthesized inputs or the
        attribution maps, beside the inputs they came from."""
        return image_reply(
            await asyncio.to_thread(
                experiment_image,
                session,
                seq=seq,
                display=display,
                input_name=primary_input,
            )
        )

    # ---- Recordings --------------------------------------------------

    @server.tool()
    async def list_recordings() -> dict[str, Any]:
        """Recordings in progress, with their frame counts and output files."""
        return recordings_view(session)

    @server.tool()
    async def start_recording(
        view: Literal["layers", "weights", "histograms", "patches", "experiment"],
        layers: list[str] | None = None,
        layer: str | None = None,
        phase: str | None = None,
        sample: int = 0,
        heatmap: bool = False,
        log_x: bool = False,
        log_y: bool = False,
        kind: str | None = None,
        params: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Record a view to MP4, one frame per visualization update.

        The way to show a human *change over time* rather than a single
        moment — a channel dying, a weight distribution drifting, a deep dream
        sharpening epoch by epoch. Start it, let training run, then
        `stop_recording` for the file path.

        Frames come from visualization updates, so `set_update_frequency` is
        the frame rate and a paused run records nothing. Each view takes its
        own arguments: `layers` for "layers"/"histograms"/"patches", `layer`
        for "weights"/"experiment", `phase` for "histograms"/"patches", and
        `kind` + `params` for "experiment" (which registers its own
        continuously re-running experiment, as the page does).
        """
        refusal = _settings_refusal(session)
        if refusal is not None:
            return refusal
        return await asyncio.to_thread(
            _start_recording,
            session,
            view=view,
            layers=layers,
            layer=layer,
            phase=phase,
            sample=sample,
            heatmap=heatmap,
            log_x=log_x,
            log_y=log_y,
            kind=kind,
            params=params,
            display=display,
            input_name=primary_input,
        )

    @server.tool()
    async def stop_recording(key: str | None = None) -> dict[str, Any]:
        """Finalize a recording and return its file path (all of them if
        `key` is omitted). `list_recordings` has the keys."""
        return await asyncio.to_thread(_stop_recording, session, key=key)

    return server


def _probe_refusal(session: Session) -> dict[str, Any] | None:
    """Why arming a probe would do nothing here, or `None` if it works.

    Locked sessions no-op the probe setters. A *closed* one is worse than a
    no-op: the setters still record their state, but the pause loop that runs
    probes is gone, so the request is armed and never served — and
    `wait_for_probe` returns immediately on a closed session, which would make
    that look like a completed run.
    """
    if session.locked:
        return {
            "error": "This session is locked (a shared demo); probes are disabled.",
            "state": "locked",
        }
    if session.closed:
        return {
            "error": (
                "Training has finished, so no further probe can run — probes "
                "execute on the training thread's pause loop. The last "
                "captured batch stays inspectable."
            ),
            "state": "finished",
        }
    return None


def _perturbation_misfit(
    session: Session,
    *,
    input_name: str,
    sample: int,
    index: Sequence[int],
    values: Sequence[float],
) -> str | None:
    """Why this edit would not reach the probe input, or `None` if it fits.

    Checked against the tensor the probe will actually run on: the pinned batch
    when one is held, otherwise the current snapshot's input.
    """
    from nansense.probe import perturbation_fits

    probe = session.probe_result
    base = None
    if probe is not None:
        base = probe.base_input(input_name)
    if base is None:
        snapshot = session.snapshot
        base = None if snapshot is None else snapshot.activations.get(input_name)
    if base is None:
        return (
            f"No input named {input_name!r} to perturb. Known inputs: "
            f"{session.input_names}."
        )
    if perturbation_fits(base, sample, tuple(index), tuple(values)):
        return None
    shape = list(base.shape)
    channels = shape[1] if base.ndim == 4 else 0
    wanted = (
        f"index [y, x] within {shape[2:]} and "
        f"{channels} value{'' if channels == 1 else 's'} (one per channel)"
        if base.ndim == 4
        else f"index [channel] within [{shape[1]}] and 1 value"
        if base.ndim == 2
        else "a 4-D image or 2-D flat input, which this is not"
    )
    return (
        f"That perturbation does not fit input {input_name!r} of shape {shape}: "
        f"it needs sample < {shape[0]}, {wanted}. "
        f"Given sample {sample}, index {list(index)}, {len(values)} value(s)."
    )


def _probe_will_run(session: Session) -> bool:
    """Whether the mutation just made will actually produce a probe run.

    Probes only execute on the training thread, and only when something wants
    one. Waiting when nothing is pinned, perturbed or mode-forced — or while
    training is running free, where the run lands at the next capture rather
    than now — would just burn the timeout for a result that was already final.
    """
    if session.is_running:
        return False
    return (
        session.is_pinned
        or bool(session.perturbations)
        or session.probe_mode != "unchanged"
    )


async def _probe_result(
    session: Session, *, after: int, timeout: float, expected: bool = True
) -> dict[str, Any]:
    """Wait for the probe the caller's change triggered, then report it.

    `expected` is the caller's own knowledge that its change armed a probe at
    all — a setter given the value already in force returns early and arms
    nothing, and no amount of state inspection afterwards can distinguish that
    from a probe still pending.
    """
    if expected and _probe_will_run(session):
        ran = await asyncio.to_thread(
            session.wait_for_probe,
            after_count=after,
            timeout=_clamp_timeout(timeout),
        )
        view = probe_view(session)
        if not ran:
            view["waiting"] = (
                "No probe run completed within the timeout. Probes run on the "
                "training thread; poll get_probe_status."
            )
        return view
    view = probe_view(session)
    if expected and session.is_running:
        view["waiting"] = (
            "Training is advancing, so the probe will run at the next capture "
            "rather than now. Call pause() to run it immediately."
        )
    return view


def _experiment_params(
    session: Session,
    *,
    kind: str,
    overrides: dict[str, Any] | None,
    display: InputDisplay,
    input_name: str | None,
) -> tuple[dict[str, Any], list[str]]:
    """The full parameter set for one run, plus any keys this kind ignores.

    Layered the way the page layers them: every knob's declared default, then
    the session's own overrides (a hosted playground seeds cheaper ones), then
    the caller's. The display statistics ride along because the runners need
    them to clamp and denormalize into input space.
    """
    known = {spec.key for spec in experiments.EXPERIMENT_PARAMS[kind]}
    params: dict[str, Any] = {
        spec.key: spec.default for spec in experiments.EXPERIMENT_PARAMS[kind]
    }
    params.update(
        {
            key: value
            for key, value in session.experiment_defaults.items()
            if key in known
        }
    )
    unknown: list[str] = []
    for key, value in (overrides or {}).items():
        if key in known:
            params[key] = value
        else:
            unknown.append(key)
    mean, std = display.stats(input_name)
    params["mean"] = mean
    params["std"] = std
    return params, unknown


def _await_experiment(
    session: Session, *, seq: int, timeout: float
) -> bool:
    """Block until request `seq` publishes a final result, or time out.

    Polled per-seq rather than through `wait_for_experiment`, which waits on
    the *latest* request: a concurrent browser tab arming its own experiment
    would otherwise satisfy this wait with someone else's result.
    """
    deadline = time.monotonic() + timeout
    while True:
        result = session.experiment_result_for(seq)
        if result is not None and result.done:
            return True
        # Experiments run on the training thread's pause loop; once the run is
        # closed nothing will ever pick this request up.
        if session.closed or time.monotonic() >= deadline:
            # One last look: a result published during the final sleep would
            # otherwise be reported as a timeout beside its own `done: true`.
            result = session.experiment_result_for(seq)
            return result is not None and result.done
        time.sleep(_EXPERIMENT_POLL_SECONDS)


async def _run_experiment(
    session: Session,
    *,
    kind: str,
    layer: str,
    params: dict[str, Any] | None,
    display: InputDisplay,
    input_name: str | None,
    timeout: float,
) -> dict[str, Any]:
    """Validate, arm and await one experiment; report whatever it published."""
    if kind not in experiments.EXPERIMENT_KINDS:
        return {
            "error": f"Unknown experiment kind {kind!r}.",
            "known_kinds": sorted(experiments.EXPERIMENT_KINDS),
        }
    if layer not in set(session.layer_names):
        return {
            "error": f"Unknown layer {layer!r}.",
            "hint": "Call get_architecture for the valid layer names.",
        }
    if not experiments.layer_available(session, layer, kind):
        return {
            "error": (
                f"{kind} cannot run on {layer!r}: it needs the layer's "
                "nn.Module, and this is an fx intermediate (or the model did "
                "not trace, leaving only named modules hookable)."
            ),
            "hint": "list_experiments reports the layers each kind accepts.",
        }
    resolved, unknown = _experiment_params(
        session, kind=kind, overrides=params, display=display, input_name=input_name
    )
    running = session.is_running
    seq = session.request_experiment(kind=kind, layer=layer, params=resolved)
    finished = await asyncio.to_thread(
        _await_experiment, session, seq=seq, timeout=_clamp_timeout(timeout)
    )
    view = experiment_result_view(session, seq=seq)
    view["seq"] = seq
    view["params"] = {
        key: value for key, value in resolved.items() if key not in ("mean", "std")
    }
    if session.locked:
        view["params_note"] = (
            "This is a locked demo, which caps the heavier knobs (steps, "
            "channels, inputs); the run may have used lower values than these."
        )
    if unknown:
        view["ignored_params"] = unknown
        view["hint"] = f"{kind} takes only its own knobs; see list_experiments."
    if not finished:
        view["waiting"] = (
            "Training was still running, so the experiment is queued until the "
            "next pause. Call pause(), then get_experiment_result."
            if running
            else "Did not finish within the timeout. Poll get_experiment_result "
            f"with seq {seq}, or cancel_experiment({seq})."
        )
    return view


def _recorded_view(
    session: Session,
    *,
    view: str,
    layers: Sequence[str] | None,
    layer: str | None,
    phase: str | None,
    sample: int,
    heatmap: bool,
    log_x: bool,
    log_y: bool,
    kind: str | None,
    params: dict[str, Any] | None,
    display: InputDisplay,
    input_name: str | None,
) -> Any:
    """The `RecordedView` for one agent-facing view name, or an error dict.

    The page equivalents build these from their own widget state; here the
    arguments come from the tool call, but the `params` payloads must match
    exactly — `nansense.recording` unpacks them by key.
    """
    from nansense.recording import RecordedView

    mean, std = display.stats(input_name)
    if view == "layers":
        # The main page records the cards on screen, i.e. the watched layers —
        # *not* every layer with statistics, which under stats scope "all" is
        # the whole model and would compose a frame thousands of pixels tall.
        chosen = _ordered(session, layers if layers else session.watched_layers)
        if not chosen:
            return {
                "error": (
                    "Nothing to record: give `layers`, or watch some first — "
                    "this view records the layers being watched."
                )
            }
        return RecordedView(
            key="main",
            page="main",
            label=f"Main view ({len(chosen)} layers, sample {sample})",
            params={
                "layers": tuple(chosen),
                "sample_idx": sample,
                "input_name": input_name or "",
                "input_mean": mean,
                "input_std": std,
                "input_transform": display.transform(input_name),
            },
        )
    if view == "weights":
        if layer is None:
            return {"error": "Give `layer` — a weights recording covers one layer."}
        parameters = session.layer_weights.get(layer, [])
        if not parameters:
            return {"error": f"Layer {layer!r} has no parameters to record."}
        return RecordedView(
            key=f"weights:{layer}",
            page="weights",
            label=f"Weights · {layer}",
            params={
                "layer": layer,
                # `(name, roles, indices)` per panel; empty roles mean the
                # default axis layout, the same thing the page opens with.
                "panels": tuple((name, (), ()) for name in parameters),
            },
        )
    if view in ("histograms", "patches"):
        # These read the watch accumulators, whose browsable universe is the
        # `/stats` page's own: collecting layers plus any whose buckets are
        # still retained.
        chosen = _ordered(session, layers if layers else session.stats_layers)
        if not chosen:
            return {
                "error": (
                    "Nothing to record: these views read the watch "
                    "accumulators, so watch some layers first."
                )
            }
        resolved_phase = phase or _newest_phase(session, chosen)
        if resolved_phase is None:
            return {
                "error": (
                    f"No statistics collected for {chosen} yet. Let training "
                    "advance at least one batch after watching."
                )
            }
        if view == "histograms":
            return RecordedView(
                key="watch_histogram",
                page="watch_histogram",
                label=f"Watch · histograms ({resolved_phase})",
                params={
                    "layers": tuple(chosen),
                    "phase": resolved_phase,
                    "log_x": log_x,
                    "log_y": log_y,
                },
            )
        return RecordedView(
            key="watch_minmax",
            page="watch_minmax",
            label=f"Watch · MIN/MAX grids ({resolved_phase})",
            params={
                "layers": tuple(chosen),
                "phase": resolved_phase,
                "grids": PATCH_TYPES,
                "heatmap": heatmap,
                "input_mean": mean,
                "input_std": std,
            },
        )
    # "experiment": the page keeps its request alive across updates with an
    # auto experiment so each frame is a fresh rerun of the *same* seq (deep
    # dream then redraws the same seeded noise); do the same here.
    if layer is None or kind is None:
        return {"error": "Give `kind` and `layer` for an experiment recording."}
    if kind not in experiments.EXPERIMENT_KINDS:
        return {
            "error": f"Unknown experiment kind {kind!r}.",
            "known_kinds": sorted(experiments.EXPERIMENT_KINDS),
        }
    if not experiments.layer_available(session, layer, kind):
        return {"error": f"{kind} cannot run on {layer!r}; see list_experiments."}
    key = f"experiment:{layer}"
    # Registering replaces any entry under `key` with a *new* seq, and the
    # recording already running holds the old one in its frozen params. So
    # check for the duplicate here, before mutating: letting `_start_recording`
    # discover it afterwards would leave the live recording pointed at a seq
    # nothing reruns any more, and it would quietly stop producing frames.
    if session.recording.is_recording(key):
        return {
            "error": f"{key!r} is already recording.",
            "hint": "One recording per view; stop_recording ends it.",
        }
    resolved, _ = _experiment_params(
        session, kind=kind, overrides=params, display=display, input_name=input_name
    )
    seq = session.register_auto_experiment(
        key, kind=kind, layer=layer, params=resolved
    )
    session.pin_auto_experiment(key)
    return RecordedView(
        key=key,
        page="experiment",
        label=f"Experiment · {experiments.EXPERIMENT_KINDS[kind]} · {layer}",
        params={
            "layer": layer,
            "seq": seq,
            "auto_key": key,
            "input_mean": mean,
            "input_std": std,
        },
    )


def _ordered(session: Session, layers: Iterable[str]) -> list[str]:
    """`layers` in the model's own order, deduplicated.

    The pages lay layers out in graph order and record them that way; a set
    from `watched_layers` / `stats_layers` would otherwise stack a recording's
    strips alphabetically, so the video would not match the page it mirrors.
    """
    wanted = set(layers)
    return [name for name in session.layer_names if name in wanted]


def _newest_phase(session: Session, layers: Sequence[str]) -> str | None:
    """The phase to record when the caller named none (see `default_phase`)."""
    return default_phase(session, layers)


def _start_recording(
    session: Session,
    *,
    view: str,
    layers: Sequence[str] | None,
    layer: str | None,
    phase: str | None,
    sample: int,
    heatmap: bool,
    log_x: bool,
    log_y: bool,
    kind: str | None,
    params: dict[str, Any] | None,
    display: InputDisplay,
    input_name: str | None,
) -> dict[str, Any]:
    recorded = _recorded_view(
        session,
        view=view,
        layers=layers,
        layer=layer,
        phase=phase,
        sample=sample,
        heatmap=heatmap,
        log_x=log_x,
        log_y=log_y,
        kind=kind,
        params=params,
        display=display,
        input_name=input_name,
    )
    if isinstance(recorded, dict):
        return recorded
    if not session.recording.start(recorded):
        return {
            "error": f"{recorded.key!r} is already recording.",
            "hint": "One recording per view; stop_recording ends it.",
        }
    result = recordings_view(session)
    result["started"] = recorded.key
    return result


def _auto_keys(views: Iterable[Any]) -> list[str]:
    """The auto-experiment registrations these recorded views are holding open.

    An experiment recording pins its auto-rerun so the request survives without
    a page heartbeat, and the registration is keyed by the view's own
    `auto_key` — which is *not* the recording key. A browser-started recording
    uses a per-page uuid there, so unpinning by recording key would silently
    leave it pinned and re-running for the rest of the training run.
    """
    keys: list[str] = []
    for view in views:
        auto_key = view.params.get("auto_key")
        if isinstance(auto_key, str) and auto_key:
            keys.append(auto_key)
    return keys


def _stop_recording(session: Session, *, key: str | None) -> dict[str, Any]:
    manager = session.recording
    # Ask *before* ending: a recording that captured no frames finalizes to no
    # files at all, which is indistinguishable afterwards from a key that was
    # never recording — and the views carry the `auto_key`s to release.
    active = manager.statuses()
    if key is None:
        views = [status.view for status in active]
        paths = manager.end_all()
    else:
        views = [status.view for status in active if status.view.key == key]
        if not views:
            return {
                "error": f"Nothing was recording under {key!r}.",
                "hint": "list_recordings has the active keys.",
            }
        paths = manager.end(key)
    for auto_key in _auto_keys(views):
        session.unpin_auto_experiment(auto_key)
        session.unregister_auto_experiment(auto_key)
    stopped = [view.key for view in views]
    result: dict[str, Any] = {
        "stopped": stopped,
        "files": [str(path) for path in paths],
    }
    if not stopped:
        result["note"] = "Nothing was recording."
    elif not paths:
        result["note"] = (
            "Empty file list: the recording captured no frames. Frames come "
            "from visualization updates, so a run that stayed paused produces "
            "none."
        )
    return result


def _wait_for_snapshot(session: Session, previous: object, timeout: float) -> bool:
    """Block until a snapshot other than `previous` is published, or time out.

    Polled rather than event-driven: snapshot publication is a plain atomic
    reference swap on the training thread with no condition-variable signal,
    and this runs on a worker thread where a short sleep costs nothing.
    """
    deadline = time.monotonic() + timeout
    while True:
        current = session.snapshot
        if current is not None and current is not previous:
            return True
        if time.monotonic() >= deadline:
            # A snapshot published during the final sleep would otherwise come
            # back as `refreshed: false` next to the fresh numbers themselves.
            current = session.snapshot
            return current is not None and current is not previous
        time.sleep(_SNAPSHOT_POLL_SECONDS)


def build_mount(
    session: Session,
    *,
    mermaid: str | None = None,
    host: str = "127.0.0.1",
    path: str = DEFAULT_MCP_PATH,
    version: str = "",
    input_mean: MeanStd | dict[str, MeanStd] | None = None,
    input_std: MeanStd | dict[str, MeanStd] | None = None,
    input_transform: InputTransform | dict[str, InputTransform] | None = None,
) -> McpMount:
    """Build the MCP endpoint for `session` as routes plus a lifespan.

    The transport's Starlette app is built only to harvest its route (and the
    DNS-rebinding protection it configures for a loopback `host`); the route is
    then served by the UI's own app, which is what keeps the endpoint on the
    UI's port without a sub-mount's trailing-slash redirect on POST.

    The `input_*` arguments are `serve`'s, forwarded to `build_server` for the
    image tools.
    """
    server = build_server(
        session,
        mermaid=mermaid,
        version=version,
        input_mean=input_mean,
        input_std=input_std,
        input_transform=input_transform,
    )
    app = server.streamable_http_app(streamable_http_path=path, host=host)
    manager = server.session_manager

    @asynccontextmanager
    async def lifespan(_app: Any) -> AsyncIterator[None]:
        async with manager.run():
            yield

    return McpMount(path=path, routes=list(app.routes), lifespan=lifespan)
