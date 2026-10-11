"""Tensor export: the `export_tensors` tool and the UI's download route.

Both front-ends build the same file from the published snapshot, so the tool
is driven through a real MCP client (the bytes an agent receives) and the route
through FastAPI's test client (the bytes a browser receives).
"""

from __future__ import annotations

import asyncio
import base64
import io
import json
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import pytest
import torch
from fastapi import FastAPI
from fastapi.testclient import TestClient
from mcp.client import Client
from mcp.types import BlobResourceContents, EmbeddedResource, TextContent

import nansense
from nansense.export import ExportError, build_export
from nansense.mcp_server import build_server
from nansense.session import Session
from nansense.ui.export import add_export_route, export_href

from .helpers import TinyNet, paused_session, train_step


@contextmanager
def _trained_session() -> Iterator[Session]:
    """Paused after a batch whose optimizer stepped, so state is populated."""
    model = TinyNet()
    optimizer = torch.optim.Adam(model.parameters())

    def step(m: TinyNet) -> None:
        train_step(m)
        optimizer.step()

    with paused_session(model, step, optimizer=optimizer) as session:
        yield session


def _load(data: bytes) -> dict[str, Any]:
    return torch.load(io.BytesIO(data), weights_only=True)


def _export_tool(session: Session, arguments: dict[str, Any]) -> tuple[Any, bytes | None]:
    async def go() -> tuple[Any, bytes | None]:
        async with Client(build_server(session)) as client:
            result = await client.call_tool("export_tensors", arguments)
        text = result.content[0]
        assert isinstance(text, TextContent)
        blobs = [
            base64.b64decode(b.resource.blob)
            for b in result.content
            if isinstance(b, EmbeddedResource)
            and isinstance(b.resource, BlobResourceContents)
        ]
        return json.loads(text.text), (blobs[0] if blobs else None)

    return asyncio.run(go())


def test_the_tool_returns_the_snapshot_tensors_and_a_manifest() -> None:
    with _trained_session() as session:
        manifest, data = _export_tool(session, {"layers": ["fc1", "x", "nope"]})
        snapshot = session.snapshot
        assert snapshot is not None and data is not None
        payload = _load(data)
        fc1 = payload["layers"]["fc1"]
        assert torch.equal(fc1["activation"], snapshot.activations["fc1"])
        assert torch.equal(fc1["gradient"], snapshot.activation_gradients["fc1"])
        weight = fc1["parameters"]["fc1.weight"]
        assert torch.equal(weight["value"], snapshot.weights["fc1.weight"])
        assert set(weight["optimizer_state"]) == {"step", "exp_avg", "exp_avg_sq"}
        assert weight["hyperparameters"]["lr"] == pytest.approx(1e-3)
        # The input batch is exported by its input name, and has no weights.
        assert set(payload["layers"]["x"]) == {"activation"}
        assert payload["position"] == {"phase": "train", "epoch": 0, "batch_idx": 0}
        assert manifest["unknown_layers"] == ["nope"]
        assert manifest["layers"]["fc1"]["activation"] == {
            "shape": [2, 8], "dtype": "float32"
        }
        assert manifest["file"] == "nansense-2-layers-train-e0-b0.pt"


@pytest.mark.parametrize(
    ("activations", "weights", "keys"),
    [
        (True, False, {"activation", "gradient"}),
        (False, True, {"parameters"}),
    ],
)
def test_the_flags_pick_the_activation_or_weight_side(
    activations: bool, weights: bool, keys: set[str]
) -> None:
    with paused_session(TinyNet()) as session:
        built = build_export(
            session, layers=["fc1"], activations=activations, weights=weights
        )
        assert set(built.payload["layers"]["fc1"]) == keys


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"layers": ["nope"]}, "No known layer"),
        ({"layers": ["fc1"], "activations": False, "weights": False}, "Nothing selected"),
        ({"layers": ["fc1"], "max_bytes": 1}, "over the"),
    ],
)
def test_refusals_say_why(kwargs: dict[str, Any], message: str) -> None:
    with paused_session(TinyNet()) as session:
        with pytest.raises(ExportError, match=message):
            build_export(session, **kwargs)


def test_nothing_captured_comes_back_as_an_error_reply() -> None:
    session = nansense.start(TinyNet(), epochs=1, phases={"train": 1})
    reply, data = _export_tool(session, {"layers": ["fc1"]})
    assert "No batch has been captured yet" in reply["error"] and data is None


def test_the_download_route_serves_the_same_file() -> None:
    with _trained_session() as session:
        app = FastAPI()
        add_export_route(app, session)
        with TestClient(app) as client:
            ok = client.get(export_href("fc1", activations=False, weights=True))
            refused = client.get(export_href("nope", activations=True, weights=True))
        assert ok.status_code == 200
        assert ok.headers["content-disposition"] == (
            'attachment; filename="nansense-fc1-weights-train-e0-b0.pt"'
        )
        assert set(_load(ok.content)["layers"]["fc1"]) == {"parameters"}
        assert refused.status_code == 409
