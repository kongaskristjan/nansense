"""`nansense-cli`: the MCP tools as shell commands, for coding agents.

A top-level package rather than `nansense.cli` so a call never imports
`nansense` — torch and NiceGUI would add seconds to every command.

A thin client over the `/mcp` endpoint `serve` mounts. Nothing here knows any
tool by name: the tool list, each tool's flags (built from its input schema)
and its help text are fetched from the running server, so the CLI stays at
parity with the MCP surface without being touched. The run is addressed by
`--port` on every call, so parallel agents debugging parallel runs never share
any registration.

    nansense-cli --port 8080 tools                  # the guide + one line per tool
    nansense-cli --port 8080 get_layer_stats --help  # one tool's docstring and flags
    nansense-cli --port 8080 get_layer_stats --layers conv1 fc

JSON results are printed as-is; pictures and files (`render_*` PNGs, the
`export_tensors` `.pt`) are written to `--out` and their paths printed, so the
caller opens only the ones it wants.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import inspect
import json
import os
import socket
import sys
import time
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from mcp.client import Client
from mcp.types import (
    BlobResourceContents,
    CallToolResult,
    EmbeddedResource,
    ImageContent,
    TextContent,
    Tool,
)

if TYPE_CHECKING:
    from mcp.server import MCPServer

_PROG = "nansense-cli"
# `nansense.mcp_server.DEFAULT_MCP_PATH`, which this must not import.
MCP_PATH = "/mcp"
_SCALARS: dict[str, type] = {"integer": int, "number": float, "string": str}
_WAIT_POLL_SECONDS = 0.25


def _common_options(defaults: bool) -> argparse.ArgumentParser:
    """`--host` / `--port` / `--out` / `--wait`, accepted before or after the command.

    The copy a tool parser inherits suppresses its defaults, so a flag given
    only before the command is not reset by the second parse.
    """
    parser = argparse.ArgumentParser(add_help=False)
    keep = argparse.SUPPRESS
    parser.add_argument(
        "--host", default="127.0.0.1" if defaults else keep,
        help="Host the training run serves on (default 127.0.0.1).",
    )
    parser.add_argument(
        "--port", type=int, default=8080 if defaults else keep,
        help="The run's --nansense-port (default 8080).",
    )
    parser.add_argument(
        "--out", type=Path, default=Path(".nansense-cli") if defaults else keep,
        help=(
            "Directory for returned files (render_* PNGs and export_tensors "
            ".pt files; default .nansense-cli)."
        ),
    )
    parser.add_argument(
        "--wait", type=float, metavar="SECONDS", default=0.0 if defaults else keep,
        help=(
            "Wait up to SECONDS for the server to accept connections, e.g. "
            "right after launching the run (default 0: fail at once)."
        ),
    )
    return parser


def top_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=_PROG,
        parents=[_common_options(defaults=True)],
        description=(
            "Drive a NaNsense training run from the shell. Start with "
            f"`{_PROG} --port <port> tools` for a guide and the tool list."
        ),
    )
    parser.add_argument("command", nargs="?", help="`tools`, or a tool name.")
    parser.add_argument("rest", nargs=argparse.REMAINDER, help=argparse.SUPPRESS)
    return parser


def _nullable(schema: dict[str, Any]) -> dict[str, Any]:
    """`T | None` arrives as `anyOf: [T, null]`; the flag takes `T`."""
    options = [s for s in schema.get("anyOf", [schema]) if s.get("type") != "null"]
    return {**options[0], **{k: v for k, v in schema.items() if k != "anyOf"}}


def _add_flag(
    parser: argparse.ArgumentParser, name: str, schema: dict[str, Any], required: bool
) -> None:
    schema = _nullable(schema)
    kind = schema.get("type")
    flag: dict[str, Any] = {"help": schema.get("description"), "dest": name}
    if "default" in schema and schema["default"] is not None:
        flag["help"] = f"{flag['help'] or ''} (default {schema['default']})".strip()
    if kind == "boolean":
        flag["action"] = argparse.BooleanOptionalAction
    elif kind == "array":
        items = schema.get("items", {})
        flag.update(nargs="+", type=_SCALARS.get(items.get("type", ""), str))
        flag["choices"] = items.get("enum")
    elif kind in _SCALARS:
        flag.update(type=_SCALARS[kind], choices=schema.get("enum"))
    else:
        flag.update(type=json.loads, metavar="JSON")
    parser.add_argument(f"--{name}", required=required, **flag)


def _type_text(schema: dict[str, Any]) -> str:
    schema = _nullable(schema)
    if "const" in schema:
        return json.dumps(schema["const"])
    if "enum" in schema:
        return " | ".join(json.dumps(value) for value in schema["enum"])
    if schema.get("type") == "array":
        return f"list of {_type_text(schema.get('items', {}))}"
    return schema.get("type", "JSON")


def _variants_text(schema: dict[str, Any]) -> str:
    """The `$defs` behind a union-typed flag, spelled out for `--help`.

    An MCP client reads these off the schema; here they are the only way an
    agent learns which fields each JSON variant takes.
    """
    lines = ["JSON variants:"]
    for name, variant in schema.get("$defs", {}).items():
        lines.append(f"  {name}: {variant.get('description', '')}".rstrip())
        required = set(variant.get("required", []))
        for field, prop in variant.get("properties", {}).items():
            default = prop.get("default")
            suffix = " (required)" if field in required else (
                f" = {json.dumps(default)}"
                if default is not None and "const" not in prop
                else ""
            )
            lines.append(f"    {field}: {_type_text(prop)}{suffix}")
            if prop.get("description"):
                lines.append(f"        {prop['description']}")
    return "\n".join(lines) if len(lines) > 1 else ""


def tool_parser(tool: Tool) -> argparse.ArgumentParser:
    """An argparse parser for one tool, with its docstring as the help text.

    Only flags actually given end up in the parsed namespace, so every omitted
    argument keeps the server's default rather than a guessed one here.
    """
    parser = argparse.ArgumentParser(
        prog=f"{_PROG} {tool.name}",
        parents=[_common_options(defaults=False)],
        description=inspect.cleandoc(tool.description or ""),
        epilog=_variants_text(tool.input_schema),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        argument_default=argparse.SUPPRESS,
    )
    schema = tool.input_schema
    required = set(schema.get("required", []))
    for name, prop in schema.get("properties", {}).items():
        _add_flag(parser, name, prop, name in required)
    return parser


def _summary(description: str | None) -> str:
    """A docstring's first paragraph on one line — a summary may wrap."""
    first = inspect.cleandoc(description or "").split("\n\n")[0]
    return " ".join(first.split())


def tool_listing(instructions: str | None, tools: Sequence[Tool], port: int) -> str:
    """The server's guide, then each tool's one-line summary."""
    width = max(len(t.name) for t in tools)
    lines = [f"  {t.name:<{width}}  {_summary(t.description)}" for t in tools]
    header = f"Tools (`{_PROG} --port {port} <tool> --help` for a tool's flags):"
    return "\n".join([(instructions or "").strip(), "", header, *lines])


def _stamp() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S-%f")[:-3]


def _image_path(out: Path, tool: str, index: int) -> Path:
    return out / f"{tool}-{_stamp()}-{index}.png"


def _file_path(out: Path, uri: str) -> Path:
    """Where a returned file goes: its server-given name, stamped so repeated
    exports never overwrite each other. Only the last path segment is used, so
    a name cannot escape `out`."""
    name = Path(str(uri).rsplit("/", 1)[-1]).name or "file"
    stem, dot, suffix = name.rpartition(".")
    return out / (f"{stem}-{_stamp()}.{suffix}" if dot else f"{name}-{_stamp()}")


def print_result(result: CallToolResult, tool: str, out: Path) -> int:
    """Print the result's text, save its pictures and files, and return the
    exit code."""
    stream = sys.stderr if result.is_error else sys.stdout
    images = 0
    for block in result.content:
        if isinstance(block, TextContent):
            print(block.text, file=stream)
        elif isinstance(block, ImageContent):
            images += 1
            out.mkdir(parents=True, exist_ok=True)
            path = _image_path(out, tool, images)
            path.write_bytes(base64.b64decode(block.data))
            print(f"Image: {path}", file=stream)
        elif isinstance(block, EmbeddedResource) and isinstance(
            block.resource, BlobResourceContents
        ):
            out.mkdir(parents=True, exist_ok=True)
            path = _file_path(out, str(block.resource.uri))
            path.write_bytes(base64.b64decode(block.resource.blob))
            print(f"File: {path}", file=stream)
    return 1 if result.is_error else 0


async def run(
    server: str | MCPServer, command: str, rest: list[str], top: argparse.Namespace
) -> int:
    """Run one command against `server` — the endpoint URL, or (in tests) an
    in-process server."""
    async with Client(server) as client:
        tools = (await client.list_tools()).tools
        if command == "tools":
            print(tool_listing(client.instructions, tools, top.port))
            return 0
        tool = next((t for t in tools if t.name == command), None)
        if tool is None:
            print(
                f"{_PROG}: unknown tool {command!r}; "
                f"`{_PROG} --port {top.port} tools` lists them.",
                file=sys.stderr,
            )
            return 2
        try:
            arguments = vars(tool_parser(tool).parse_args(rest))
        except SystemExit as exit:  # --help, or a bad flag
            return int(exit.code or 0)
        out = arguments.pop("out", top.out)
        for option in ("host", "port", "wait"):
            arguments.pop(option, None)
        result = await client.call_tool(tool.name, arguments)
        return print_result(result, tool.name, out)


def _serving(host: str, port: int) -> bool:
    try:
        socket.create_connection((host, port), timeout=2).close()
    except OSError:
        return False
    return True


def _await_serving(host: str, port: int, wait: float) -> bool:
    """Whether the run serves, retrying for up to `wait` seconds.

    The run prints its banner only once the endpoint accepts connections, so
    an agent that launched it can either watch for the banner or pass
    `--wait` and let the first command block until then.
    """
    deadline = time.monotonic() + max(0.0, wait)
    while not _serving(host, port):
        if time.monotonic() >= deadline:
            return False
        time.sleep(_WAIT_POLL_SECONDS)
    return True


def main(argv: Sequence[str] | None = None) -> int:
    top = top_parser()
    args = top.parse_args(argv)
    if args.command is None:
        top.print_help()
        return 0
    # Options given after the command (`get_status --port 8081`) count too.
    late, _ = _common_options(defaults=False).parse_known_args(args.rest)
    for key, value in vars(late).items():
        setattr(args, key, value)
    if not _await_serving(args.host, args.port, args.wait):
        waited = f" after waiting {args.wait:g}s" if args.wait > 0 else ""
        print(
            f"{_PROG}: nothing is serving on {args.host}:{args.port}{waited}. "
            "Start the training run with NaNsense enabled, and pass its "
            "--nansense-port as --port (use --wait SECONDS to wait for startup).",
            file=sys.stderr,
        )
        return 2
    url = f"http://{args.host}:{args.port}{MCP_PATH}"
    try:
        code = asyncio.run(run(url, args.command, args.rest, args))
        # Flushed here, so a closed pipe surfaces below rather than at exit.
        sys.stdout.flush()
        return code
    except BrokenPipeError:
        # The reader went away (`| head`): stop quietly, as shell tools do,
        # and point stdout at /dev/null so the interpreter's final flush
        # does not raise the same error again.
        os.dup2(os.open(os.devnull, os.O_WRONLY), sys.stdout.fileno())
        return 0
