"""MCP server: the JSON views and a real client round-trip over the SDK.

Two layers, matching the module split. `mcp_views` is exercised directly
(pure functions over a live session), while the tool surface is driven through
an actual `mcp.client.Client` on the in-memory transport — so the registration,
argument schemas, and JSON serialization are covered by the same path a coding
agent would take, not by calling the handlers as plain functions.

The client is async and the rest of the suite is not, so each round-trip test
wraps its coroutine in `asyncio.run` rather than pulling in an async plugin.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Coroutine
from typing import Any, TypeVar

import pytest
import torch
from fastapi import FastAPI
from fastapi.testclient import TestClient
from mcp.client import Client
from mcp.types import TextContent

import nansense
from nansense.debugger import DebugError, LayerReport
from nansense.mcp_server import build_mount, build_server
from nansense.mcp_views import (
    _num,
    debug_view,
    layer_stats_view,
    stats_history_view,
    status_view,
    tensor_stats_view,
)
from nansense.session import Session
from nansense.watch import N_BINS, ZERO_BIN, TensorStatsSnapshot

from .helpers import TinyNet, make_position, paused_session

_T = TypeVar("_T")


def _run(coro: Coroutine[Any, Any, _T]) -> _T:
    return asyncio.run(coro)


def _step_and_wait(session: Session, *, timeout: float = 5.0) -> None:
    """Advance one batch and block until the training thread has parked again.

    The pause counter has to be sampled *before* the command: sampling after it
    can already include the pause we are waiting for, and the wait then returns
    without anything having happened.
    """
    before = session.pause_count
    session.step_batch()
    assert session.wait_until_paused(after_pauses=before, timeout=timeout)


def _call(session: Session, name: str, arguments: dict[str, Any] | None = None) -> Any:
    """Call one tool through a real MCP client and parse its JSON payload."""

    async def go() -> Any:
        async with Client(build_server(session, mermaid="graph TD")) as client:
            result = await client.call_tool(name, arguments or {})
            assert result.content, f"{name} returned no content"
            block = result.content[0]
            assert isinstance(block, TextContent)
            return json.loads(block.text)

    return _run(go())


# --- number rendering -------------------------------------------------


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (1.5, 1.5),
        (float("nan"), "nan"),
        (float("inf"), "inf"),
        (float("-inf"), "-inf"),
        (None, None),
    ],
)
def test_num_keeps_non_finite_values_readable(
    value: float | None, expected: float | str | None
) -> None:
    """JSON has no NaN literal, and dropping to null would erase the finding."""
    assert _num(value) == expected


def test_num_trims_float_noise() -> None:
    assert _num(1 / 3) == 0.333333


# --- views ------------------------------------------------------------


def test_status_reports_paused_position_and_totals() -> None:
    with paused_session(TinyNet(), epochs=1, phases={"train": 2}) as session:
        view = status_view(session)
        assert view["state"] == "paused"
        assert view["live_position"]["text"] == "epoch 0/1 | train batch 0/2"
        assert view["snapshot_position"]["phase"] == "train"
        assert view["total_epochs"] == 1
        assert view["layer_count"] == len(session.layer_names)


def test_status_distinguishes_a_loop_that_never_started() -> None:
    """`is_running` is True before the first batch; an agent must not read that
    as "training is advancing" and wait for a pause nothing is driving toward."""
    session = nansense.start(TinyNet(), epochs=1, phases={"train": 1})
    view = status_view(session)
    assert view["state"] == "not_started"
    assert view["live_position"] is None
    assert "first batch" in view["hint"]


def test_layer_stats_cover_any_layer_with_shapes() -> None:
    with paused_session(TinyNet()) as session:
        view = layer_stats_view(session, layers=["fc1", "fc2"])
        by_name = {entry["layer"]: entry for entry in view["layers"]}
        assert set(by_name) == {"fc1", "fc2"}
        activations = by_name["fc1"]["activations"]
        assert activations["shape"] == [2, 8]
        assert activations["count"] == 16
        assert isinstance(activations["mean"], float)
        # The layer was never watched: reading the published snapshot is what
        # makes any layer inspectable without arranging collection first.
        assert not session.watched_layers


def test_layer_stats_flag_unknown_names_instead_of_silently_dropping_them() -> None:
    with paused_session(TinyNet()) as session:
        view = layer_stats_view(session, layers=["fc1", "nope"])
        assert view["unknown_layers"] == ["nope"]
        assert [entry["layer"] for entry in view["layers"]] == ["fc1"]


def test_layer_stats_histogram_is_sparse_pairs() -> None:
    with paused_session(TinyNet()) as session:
        view = layer_stats_view(session, layers=["fc1"], include_histogram=True)
        histogram = view["layers"][0]["activations"]["histogram"]
        # 211 fixed bins, but only the populated ones are shipped.
        assert 0 < len(histogram) < 211
        assert sum(count for _, count in histogram) == 16


def test_empty_tensor_stats_report_a_zero_count_not_a_row_of_nan() -> None:
    empty = TensorStatsSnapshot(n=0, sum=0.0, sum_sq=0.0, min=0.0, max=0.0, hist=None)
    view = tensor_stats_view(empty)
    assert view["count"] == 0
    assert "mean" not in view
    assert "no values captured" in view["note"]


def _hist(**bins: int) -> tuple[int, ...]:
    counts = [0] * N_BINS
    for index, count in bins.items():
        counts[int(index.removeprefix("b"))] = count
    return tuple(counts)


def test_fully_diverged_stats_are_not_reported_as_empty() -> None:
    """The accumulator's `n` counts finite values only, so an all-Inf tensor
    arrives with `n == 0` — indistinguishable from an unused one unless the
    histogram total is consulted."""
    diverged = TensorStatsSnapshot(
        n=0,
        sum=0.0,
        sum_sq=0.0,
        min=0.0,
        max=0.0,
        # ±Inf saturates into the two end bins.
        hist=_hist(b0=2, **{f"b{N_BINS - 1}": 4}),
    )
    view = tensor_stats_view(diverged)
    assert view["count"] == 6
    assert view["non_finite_count"] == 6
    assert view["finite_count"] == 0
    assert "non-finite" in view["note"]
    # Nothing finite to average, so no misleading scalars.
    assert "mean" not in view


def test_partly_diverged_stats_say_which_population_the_mean_describes() -> None:
    partial = TensorStatsSnapshot(
        n=2,
        sum=4.0,
        sum_sq=8.0,
        min=2.0,
        max=2.0,
        hist=_hist(b0=1, **{f"b{ZERO_BIN + 1}": 2}),
    )
    view = tensor_stats_view(partial)
    assert view["count"] == 3
    assert view["finite_count"] == 2
    assert view["non_finite_count"] == 1
    assert view["mean"] == 2.0
    assert "only the finite values" in view["note"]


def test_stats_history_points_at_watching_when_nothing_is_collected() -> None:
    with paused_session(TinyNet()) as session:
        view = stats_history_view(session, layer="fc1")
        assert view["history"] == {}
        assert "watch_layers" in view["hint"]


def test_stats_history_returns_per_epoch_series_for_a_watched_layer() -> None:
    with paused_session(TinyNet(), epochs=2, phases={"train": 2}) as session:
        assert session.watch("fc1")
        # Four batches in the schedule and we are parked on the first, so three
        # steps carry the run into its second epoch.
        for _ in range(3):
            _step_and_wait(session)
        view = stats_history_view(session, layer="fc1", phase="train")
        points = view["history"]["train"]["activations"]
        assert [point["epoch"] for point in points] == [0, 1]
        assert view["phases_with_data"] == ["train"]
        assert all(point["count"] > 0 for point in points)


def test_debug_view_reports_settings_and_a_standing_detection() -> None:
    with paused_session(TinyNet()) as session:
        assert debug_view(session)["detected"] is None
        session._debug_error = DebugError(
            position=make_position("train", 0, 3),
            reasons=("nan",),
            checks_used=("nan_inf",),
            layers=(
                LayerReport(
                    layer="fc1",
                    nan=0.25,
                    inf=0.0,
                    underflow=0.0,
                    overflow=0.0,
                    dtype=torch.float32,
                ),
            ),
        )
        detected = debug_view(session)["detected"]
        assert detected["reasons"] == "NaN"
        assert detected["first_detected_at"]["batch"] == 3
        layer = detected["layers"][0]
        assert layer["layer"] == "fc1"
        assert layer["NaN_percent"] == 25.0
        assert layer["gradient_dtype"] == "float32"


# --- tools over a real MCP client -------------------------------------


def test_every_tool_is_registered_with_a_description() -> None:
    session = nansense.start(TinyNet(), epochs=1, phases={"train": 1})

    async def go() -> None:
        async with Client(build_server(session, mermaid="graph TD")) as client:
            tools = (await client.list_tools()).tools
            names = {tool.name for tool in tools}
            assert {
                "get_status",
                "get_architecture",
                "get_layer_stats",
                "get_stats_history",
                "get_debug_report",
                "step",
                "run",
                "run_until",
                "pause",
                "detach",
                "refresh",
                "watch_layers",
                "unwatch_layers",
                "set_stats_scope",
                "configure_debug_checks",
                "silence_debug_check",
            } <= names
            # The descriptions are the agent's only documentation.
            assert all(tool.description for tool in tools)

    _run(go())


def test_get_architecture_carries_the_graph_and_layer_names() -> None:
    with paused_session(TinyNet()) as session:
        view = _call(session, "get_architecture")
        assert view["mermaid"] == "graph TD"
        assert view["inputs"] == ["x"]
        names = [layer["name"] for layer in view["layers"]]
        assert {"fc1", "fc2"} <= set(names)
        by_name = {layer["name"]: layer for layer in view["layers"]}
        assert set(by_name["fc1"]["parameters"]) == {"fc1.weight", "fc1.bias"}
        assert "in_features=4" in by_name["fc1"]["hyperparameters"]


def test_get_architecture_can_omit_the_graph() -> None:
    with paused_session(TinyNet()) as session:
        assert "mermaid" not in _call(
            session, "get_architecture", {"include_graph": False}
        )


def test_step_advances_the_run_and_reports_the_new_position() -> None:
    with paused_session(TinyNet(), epochs=1, phases={"train": 3}) as session:
        before = status_view(session)["live_position"]["batch"]
        view = _call(session, "step", {"unit": "batch", "timeout_seconds": 5})
        assert view["state"] == "paused"
        assert view["live_position"]["batch"] == before + 1
        assert "waiting" not in view


def test_watch_layers_reports_unknown_names_and_updates_status() -> None:
    with paused_session(TinyNet()) as session:
        view = _call(session, "watch_layers", {"layers": ["fc1", "ghost"]})
        assert view["watched_layers"] == ["fc1"]
        assert view["unknown_layers"] == ["ghost"]
        assert session.watched_layers == frozenset({"fc1"})


def test_unwatch_layers_drops_the_layer() -> None:
    with paused_session(TinyNet()) as session:
        _call(session, "watch_layers", {"layers": ["fc1"]})
        view = _call(session, "unwatch_layers", {"layers": ["fc1"]})
        assert view["watched_layers"] == []


def test_set_stats_scope_switches_collection() -> None:
    with paused_session(TinyNet()) as session:
        view = _call(session, "set_stats_scope", {"scope": "none"})
        assert view["stats_scope"] == "none"
        assert view["stats_collecting"] is False


def test_run_until_rejects_an_unknown_phase_with_the_known_ones() -> None:
    with paused_session(TinyNet(), epochs=1, phases={"train": 2}) as session:
        view = _call(
            session, "run_until", {"phase": "val", "epoch": 0, "batch": 0}
        )
        assert "Unknown phase" in view["error"]
        assert view["known_phases"] == ["train"]


def test_configure_debug_checks_updates_only_what_was_passed() -> None:
    with paused_session(TinyNet()) as session:
        view = _call(session, "configure_debug_checks", {"interval_batches": 1})
        assert view["settings"]["interval_batches"] == 1
        assert view["settings"]["check_nan_inf"] is True
        assert session.debug_settings.interval == 1


def test_silence_debug_check_clears_that_part_of_the_warning() -> None:
    with paused_session(TinyNet()) as session:
        session._debug_error = DebugError(
            position=make_position("train", 0, 0),
            reasons=("nan",),
            checks_used=("nan_inf",),
            layers=(
                LayerReport(
                    layer="fc1", nan=1.0, inf=0.0, underflow=0.0, overflow=0.0
                ),
            ),
        )
        view = _call(session, "silence_debug_check", {"category": "nan_inf"})
        assert view["detected"] is None
        assert view["settings"]["check_nan_inf"] is False
        assert session.debug_error is None


def test_refresh_is_a_no_op_while_paused() -> None:
    with paused_session(TinyNet()) as session:
        view = _call(session, "refresh")
        assert view["refreshed"] is False
        assert "already the current one" in view["note"]


def test_control_tools_refuse_on_a_locked_session() -> None:
    """A locked session no-ops its control methods; an agent that reads that as
    success would step forever without moving."""
    # Deliberately no worker thread: `lock()` refuses `detach()` too, so a
    # locked session cannot be released the way `paused_session` tears down.
    session = nansense.start(TinyNet(), epochs=1, phases={"train": 1})
    session.lock()
    for name, arguments in (
        ("step", {"timeout_seconds": 1}),
        ("pause", {"timeout_seconds": 1}),
        ("detach", {}),
        ("watch_layers", {"layers": ["fc1"]}),
        ("set_stats_scope", {"scope": "all"}),
        ("configure_debug_checks", {"interval_batches": 5}),
    ):
        view = _call(session, name, arguments)
        assert view["state"] == "locked", name
        assert "locked" in view["error"], name
    # Inspection still works — that is the point of a locked demo.
    assert _call(session, "get_status")["locked"] is True


def test_control_tools_refuse_once_training_has_finished() -> None:
    session, _ = _finished_session()
    view = _call(session, "step", {"timeout_seconds": 1})
    assert view["state"] == "finished"
    assert "finished" in view["error"]
    # The last captured batch stays inspectable.
    assert _call(session, "get_layer_stats", {"layers": ["fc1"]})["layers"]


def _finished_session() -> tuple[Session, TinyNet]:
    """A session driven to completion, so `closed` is True with a snapshot."""
    model = TinyNet()
    with paused_session(model, epochs=1, phases={"train": 1}) as session:
        pass
    session.close()
    return session, model


# --- the HTTP mount ---------------------------------------------------


def _mounted_app(session: Session) -> FastAPI:
    """A FastAPI app wired exactly the way `serve` wires its own."""
    mount = build_mount(session, mermaid="graph TD")
    app = FastAPI(lifespan=mount.lifespan)
    app.router.routes.extend(mount.routes)
    return app


def _initialize_request() -> dict[str, Any]:
    return {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "initialize",
        "params": {
            "protocolVersion": "2025-06-18",
            "capabilities": {},
            "clientInfo": {"name": "test", "version": "1"},
        },
    }


def test_mounted_route_answers_the_protocol_and_runs_its_lifespan() -> None:
    """The transport only works if the app's lifespan started its session
    manager — a mounted sub-app would never get one, which is why `serve`
    passes the lifespan to the app it actually serves."""
    session = nansense.start(TinyNet(), epochs=1, phases={"train": 1})
    with TestClient(_mounted_app(session), base_url="http://127.0.0.1:8080") as client:
        response = client.post(
            "/mcp",
            json=_initialize_request(),
            headers={"Accept": "application/json, text/event-stream"},
        )
    assert response.status_code == 200
    assert response.headers.get("mcp-session-id")
    assert '"serverInfo"' in response.text
    assert "nansense" in response.text


def test_loopback_mount_refuses_a_foreign_host_header() -> None:
    """The transport turns on DNS-rebinding protection for a loopback bind, so
    a page on another origin cannot drive a developer's training run."""
    session = nansense.start(TinyNet(), epochs=1, phases={"train": 1})
    with TestClient(_mounted_app(session), base_url="http://evil.example") as client:
        response = client.post(
            "/mcp",
            json=_initialize_request(),
            headers={"Accept": "application/json, text/event-stream"},
        )
    assert response.status_code >= 400


def test_get_layer_stats_before_any_capture_explains_itself() -> None:
    session = nansense.start(TinyNet(), epochs=1, phases={"train": 1})
    view = _call(session, "get_layer_stats", {"layers": ["fc1"]})
    assert "No batch has been captured" in view["error"]
    assert "hint" in view


def test_non_finite_activations_survive_to_the_agent() -> None:
    """The whole point of the debugger: an Inf must not be flattened to null."""

    class Diverged(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.fc = torch.nn.Linear(4, 3)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.fc(x) * float("inf")

    def step(model: Diverged) -> None:
        x = torch.randn(2, 4)
        model.zero_grad(set_to_none=True)
        model(x).sum().backward()

    with paused_session(Diverged(), step, epochs=1, phases={"train": 1}) as session:
        view = layer_stats_view(session, layers=session.layer_names)
        by_name = {entry["layer"]: entry for entry in view["layers"]}
        # `mul` is the fx node carrying the divergence; every one of its values
        # is non-finite, which must not be reported as an empty tensor.
        activations = by_name["mul"]["activations"]
        assert activations["count"] == 6
        assert activations["non_finite_count"] == 6
        assert "non-finite" in activations["note"]


def test_the_server_advertises_the_package_version() -> None:
    """`version` was accepted but never supplied, so clients saw an empty
    string in the handshake."""
    from nansense.mcp_server import _package_version

    session = nansense.start(TinyNet(), epochs=1, phases={"train": 1})
    assert build_server(session).version == _package_version()
    assert _package_version()  # the installed distribution is discoverable
    assert build_server(session, version="1.2.3").version == "1.2.3"
