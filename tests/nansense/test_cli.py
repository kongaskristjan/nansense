"""`nansense-cli`: flags built from the live tool schemas, and each command
run against an in-process server through the same `run` the CLI uses."""

from __future__ import annotations

import asyncio
import base64
import json
import socket
from pathlib import Path
from typing import Any

import pytest
from mcp.client import Client
from mcp.types import CallToolResult, ImageContent, TextContent, Tool

import nansense
from nansense.cli import main, print_result, run, tool_parser, top_parser
from nansense.mcp_server import build_server
from nansense.session import Session

from .helpers import TinyNet, paused_session


@pytest.fixture(scope="module")
def tools() -> dict[str, Tool]:
    session = nansense.start(TinyNet(), epochs=1, phases={"train": 1})

    async def go() -> list[Tool]:
        async with Client(build_server(session)) as client:
            return (await client.list_tools()).tools

    return {tool.name: tool for tool in asyncio.run(go())}


@pytest.mark.parametrize(
    ("tool", "argv", "expected"),
    [
        # Omitted flags stay omitted, so the server's defaults apply.
        ("step", [], {}),
        ("step", ["--unit", "epoch"], {"unit": "epoch"}),
        (
            "get_layer_stats",
            ["--layers", "fc1", "fc2", "--include_histogram", "--channel", "3"],
            {"layers": ["fc1", "fc2"], "include_histogram": True, "channel": 3},
        ),
        ("render_layer", ["--layers", "fc1", "--no-include_input"],
         {"layers": ["fc1"], "include_input": False}),
        (
            "add_perturbation",
            ["--index", "0", "2", "--values", "-0.5", "1"],
            {"index": [0, 2], "values": [-0.5, 1.0]},
        ),
        (
            "run_experiment",
            ["--kind", "deep_dream", "--layer", "fc1", "--params", '{"all_steps": true}'],
            {"kind": "deep_dream", "layer": "fc1", "params": {"all_steps": True}},
        ),
    ],
)
def test_flags_follow_the_tool_schema(
    tools: dict[str, Tool], tool: str, argv: list[str], expected: dict[str, Any]
) -> None:
    assert vars(tool_parser(tools[tool]).parse_args(argv)) == expected


@pytest.mark.parametrize(
    ("tool", "argv"),
    [("step", ["--unit", "week"]), ("get_layer_stats", [])],  # bad enum, missing required
)
def test_invalid_flags_are_rejected(
    tools: dict[str, Tool], tool: str, argv: list[str]
) -> None:
    with pytest.raises(SystemExit):
        tool_parser(tools[tool]).parse_args(argv)


def _run(session: Session, argv: list[str]) -> int:
    top = top_parser().parse_args(["--port", "8123", *argv])
    return asyncio.run(run(build_server(session), top.command, top.rest, top))


def test_tools_prints_the_guide_and_every_tool(
    tools: dict[str, Tool], capsys: pytest.CaptureFixture[str]
) -> None:
    session = nansense.start(TinyNet(), epochs=1, phases={"train": 1})
    assert _run(session, ["tools"]) == 0
    out = capsys.readouterr().out
    assert out.startswith("NaNsense is a PyTorch debugger")
    assert "nansense-cli --port 8123 <tool> --help" in out
    assert all(f"  {name} " in out for name in tools)


def test_a_call_prints_the_tool_json(capsys: pytest.CaptureFixture[str]) -> None:
    with paused_session(TinyNet()) as session:
        assert _run(session, ["get_layer_stats", "--layers", "fc1"]) == 0
    layers = json.loads(capsys.readouterr().out)["layers"]
    assert [layer["layer"] for layer in layers] == ["fc1"]


def test_help_shows_the_docstring_without_calling(
    capsys: pytest.CaptureFixture[str],
) -> None:
    session = nansense.start(TinyNet(), epochs=1, phases={"train": 1})
    assert _run(session, ["get_layer_stats", "--help"]) == 0
    out = capsys.readouterr().out
    assert "usage: nansense-cli get_layer_stats" in out
    assert "\nWorks for any layer" in out  # dedented


def test_unknown_tool_points_at_the_listing(capsys: pytest.CaptureFixture[str]) -> None:
    session = nansense.start(TinyNet(), epochs=1, phases={"train": 1})
    assert _run(session, ["frobnicate"]) == 2
    assert "nansense-cli --port 8123 tools" in capsys.readouterr().err


def test_images_are_written_to_out(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    png = b"\x89PNG fake"
    result = CallToolResult(
        content=[
            TextContent(type="text", text="note"),
            ImageContent(
                type="image", data=base64.b64encode(png).decode(), mime_type="image/png"
            ),
        ]
    )
    assert print_result(result, "render_input", tmp_path / "shots") == 0
    out = capsys.readouterr().out.splitlines()
    assert out[0] == "note"
    path = Path(out[1].removeprefix("Image: "))
    assert path.parent == tmp_path / "shots"
    assert path.name.startswith("render_input-")
    assert path.read_bytes() == png


def test_a_tool_error_goes_to_stderr_with_a_failing_exit(
    capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    result = CallToolResult(content=[TextContent(type="text", text="bad")], is_error=True)
    assert print_result(result, "step", tmp_path) == 1
    assert capsys.readouterr().err == "bad\n"


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def test_nothing_serving_names_the_port_even_when_given_after_the_tool(
    capsys: pytest.CaptureFixture[str],
) -> None:
    port = _free_port()
    assert main(["get_status", "--port", str(port)]) == 2
    assert f"nothing is serving on 127.0.0.1:{port}" in capsys.readouterr().err
