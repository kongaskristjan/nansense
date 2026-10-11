"""User-registered tools (`Session.tool`), through a real MCP client."""

from __future__ import annotations

import asyncio
import json
from typing import Annotated, Any, Literal

import pydantic
import pytest
import torch
from mcp.client import Client
from mcp.types import CallToolResult, TextContent

import nansense
from nansense import Table, ToolContext
from nansense.mcp_server import build_server
from nansense.session import Session
from tests.nansense.helpers import TinyNet, paused_session


def _register(session: Session) -> None:
    @session.tool()
    def weight_norms(
        ctx: ToolContext,
        layer: Annotated[str, pydantic.Field(description="Module to measure")] = "fc1",
        order: Literal[1, 2] = 2,
    ) -> Table:
        """Norm of each parameter of one module."""
        module = dict(ctx.model.named_modules())[layer]
        for p in module.parameters():
            p.data.zero_()  # restored after the call
        return Table(("parameter", "norm"), [("weight", 0.0)])

    @session.tool()
    def count(ctx: ToolContext) -> int:
        """How many inputs the snapshot holds."""
        return int(ctx.inputs[0].shape[0])

    @session.tool()
    def broken(ctx: ToolContext) -> None:
        raise RuntimeError("nope")


async def _call(session: Session, name: str, args: dict[str, Any]) -> CallToolResult:
    async with Client(build_server(session)) as client:
        return await client.call_tool(name, args)


def _json(result: CallToolResult) -> Any:
    block = result.content[0]
    assert isinstance(block, TextContent)
    return json.loads(block.text)


def test_a_custom_tool_is_listed_with_its_schema_and_runs_while_paused() -> None:
    model = TinyNet()
    with paused_session(model) as session:
        _register(session)
        before = model.fc1.weight.detach().clone()

        async def go() -> tuple[Any, ...]:
            async with Client(build_server(session)) as client:
                tools = {t.name: t for t in (await client.list_tools()).tools}
                norms = await client.call_tool("weight_norms", {"order": 1})
                count = await client.call_tool("count", {})
                broken = await client.call_tool("broken", {})
                return tools, norms, count, broken

        tools, norms, count, broken = asyncio.run(go())
        schema = tools["weight_norms"].input_schema["properties"]
        assert schema["layer"]["description"] == "Module to measure"
        assert schema["order"]["enum"] == [1, 2]
        assert tools["weight_norms"].description == "Norm of each parameter of one module."
        assert _json(norms) == {
            "columns": ["parameter", "norm"],
            "rows": [{"parameter": "weight", "norm": 0.0}],
        }
        assert torch.equal(model.fc1.weight, before)
        assert _json(count) == {"result": 2}
        assert broken.is_error and "RuntimeError: nope" in _json(broken)["error"]


def test_a_training_thread_tool_waits_for_a_pause_but_a_snapshot_tool_does_not() -> None:
    session = nansense.start(TinyNet(), epochs=1, phases={"train": 1})

    @session.tool(on_training_thread=False)
    def quick(ctx: ToolContext) -> str:
        return "ok" if ctx.snapshot is None else "snapshot"

    @session.tool(timeout=0.1)
    def slow(ctx: ToolContext) -> str:
        return "ran"

    assert _json(asyncio.run(_call(session, "quick", {}))) == {"result": "ok"}
    queued = _json(asyncio.run(_call(session, "slow", {})))
    assert queued["stage"] == "queued"
    assert session.tool_call(queued["seq"]).stage == "queued"
    session.close()
    finished = asyncio.run(_call(session, "slow", {}))
    assert finished.is_error and "finished" in _json(finished)["error"]


def test_tools_registered_after_the_server_starts_are_served() -> None:
    session = nansense.start(TinyNet(), epochs=1, phases={"train": 1})
    server = build_server(session)

    @session.tool(on_training_thread=False)
    def late(ctx: ToolContext) -> str:
        return "here"

    async def go() -> list[str]:
        async with Client(server) as client:
            return [t.name for t in (await client.list_tools()).tools]

    names = asyncio.run(go())
    assert "late" in names and "get_tool_result" in names
    with pytest.raises(ValueError, match="already used by a NaNsense"):
        session.tool("get_status")(lambda ctx: None)


class _Opaque:
    pass


def _opaque_argument(ctx: ToolContext, thing: _Opaque) -> None:
    pass


@pytest.mark.parametrize(
    ("fn", "match"),
    [
        (lambda: None, "ToolContext"),
        (_opaque_argument, "JSON schema"),
        (lambda ctx, *rest: None, "args"),
    ],
)
def test_undescribable_tools_are_rejected(fn: Any, match: str) -> None:
    session = nansense.start(TinyNet(), epochs=1, phases={"train": 1})
    with pytest.raises(ValueError, match=match):
        session.tool("bad")(fn)


def test_a_locked_session_serves_only_public_tools() -> None:
    session = nansense.start(TinyNet(), epochs=1, phases={"train": 1})
    for name, public in (("private_tool", False), ("public_tool", True)):
        session.tool(name, on_training_thread=False, public=public)(lambda ctx: name)
    session.lock()
    assert set(session.tool_specs) == {"public_tool"}
    refused = asyncio.run(_call(session, "private_tool", {}))
    assert refused.is_error and "locked" in _json(refused)["error"]
    with pytest.raises(RuntimeError, match="locked"):
        session.tool("late")
