"""MCP image tools: the rendered views a coding agent can look at.

The pictures themselves are `nansense.ui.frames`' and are covered by the
recording tests; what matters here is everything between a `Session` and the
wire — that a picture comes back as PNG rather than the browser's BMP, that an
absent picture comes back as a *reason* instead of silence, and that the axis
defaults produce the view the page shows rather than a degenerate one.

As in `test_mcp`, the tool surface is driven through a real `mcp.client.Client`
on the in-memory transport, so registration and content-block encoding are
covered by the path an agent would take.
"""

from __future__ import annotations

import asyncio
import io
from collections.abc import Coroutine
from typing import Any, TypeVar

import pytest
import torch
from mcp.client import Client
from PIL import Image
from torch import Tensor, nn

import nansense
from nansense.input_config import InputDisplay
from nansense.mcp_images import (
    MAX_SIDE,
    RenderedImage,
    _encode,
    histogram_image,
    image_reply,
    input_image,
    layer_image,
    patches_image,
    weights_image,
)
from nansense.mcp_server import build_server
from nansense.session import Session
from nansense.ui.frames import WeightPanel

from .helpers import TinyNet, paused_session

_T = TypeVar("_T")

_DISPLAY = InputDisplay()


def _run(coro: Coroutine[Any, Any, _T]) -> _T:
    return asyncio.run(coro)


class TinyConvNet(nn.Module):
    """A conv net over a small image, so the strips and patch grids have
    something image-shaped to render."""

    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(1, 3, 3, padding=1)
        self.fc = nn.Linear(3 * 6 * 6, 4)

    def forward(self, x: Tensor) -> Tensor:
        return self.fc(torch.relu(self.conv(x)).flatten(1))


def _conv_step(model: TinyConvNet) -> None:
    model(torch.rand(2, 1, 6, 6)).square().mean().backward()


def _blocks(result: Any) -> tuple[list[str], list[Any]]:
    """A tool result split into its text lines and its image blocks."""
    texts = [b.text for b in result.content if getattr(b, "type", "") == "text"]
    images = [b for b in result.content if getattr(b, "type", "") == "image"]
    return texts, images


def _call(session: Session, name: str, arguments: dict[str, Any] | None = None) -> Any:
    async def go() -> Any:
        async with Client(build_server(session)) as client:
            return await client.call_tool(name, arguments or {})

    return _run(go())


# --- encoding ---------------------------------------------------------


def test_images_go_out_as_png_not_the_browsers_bmp() -> None:
    """`render.STRIP_FORMAT` is BMP for the localhost `<img>` path — near-memcpy
    encode at ~2x the bytes. On an MCP reply those bytes are base64'd inside
    JSON, so they are paid for twice."""
    png, caveat = _encode(Image.new("RGB", (10, 10), (1, 2, 3)))
    assert png is not None and caveat == ""
    assert png[:8] == b"\x89PNG\r\n\x1a\n"


def test_oversized_views_are_downscaled_and_say_so() -> None:
    """A wide layer composes into a picture thousands of pixels across. It is
    scaled down — but silently smoothing neighbouring channels together would
    let a reader mistake the averaging for data."""
    png, caveat = _encode(Image.new("RGB", (MAX_SIDE * 2, 40), (1, 2, 3)))
    assert png is not None
    assert Image.open(io.BytesIO(png)).width == MAX_SIDE
    assert "scaled down" in caveat and str(MAX_SIDE) in caveat


def test_encoding_nothing_yields_no_caveat() -> None:
    assert _encode(None) == (None, "")


def test_a_failed_render_answers_in_words_alone() -> None:
    """"No image" and "an image of nothing" are indistinguishable on the wire,
    so a failure returns the reason as text and no image block."""
    reply = image_reply(RenderedImage(None, "nothing captured yet"))
    assert reply == ["nothing captured yet"]


def test_a_successful_render_puts_the_note_before_the_picture() -> None:
    reply = image_reply(RenderedImage(b"\x89PNG", "at epoch 0"))
    assert reply[0] == "at epoch 0"
    assert getattr(reply[1], "_mime_type", None) == "image/png"


# --- input display resolution ----------------------------------------


def test_input_display_resolves_per_input_stats() -> None:
    """A multi-input model gets its stats as a dict keyed by input name; the
    tools must resolve the same way the pages do."""
    display = InputDisplay(
        mean={"image": (0.5,), "other": (0.1,)}, std={"image": (0.25,)}
    )
    assert display.stats("image") == ((0.5,), (0.25,))
    assert display.stats("other") == ((0.1,), None)


def test_input_display_applies_a_single_value_to_every_input() -> None:
    display = InputDisplay(mean=(0.5,), std=(0.25,))
    assert display.stats("anything") == ((0.5,), (0.25,))


# --- weight axis defaults --------------------------------------------


@pytest.mark.parametrize(
    ("ndim", "expected"),
    [
        (4, (3, 2, 1)),  # conv: kH x kW tiles across the input channels
        (2, (1, 0, None)),  # linear: one [out, in] image
        (1, (0, None, None)),  # bias: a heatmap row
    ],
)
def test_unconfigured_panel_takes_the_pages_default_layout(
    ndim: int, expected: tuple[int, int | None, int | None]
) -> None:
    """A panel with no axes chosen must mean "the default view", not "no Y axis"
    — the latter flattens a conv kernel into a meaningless row."""
    assert WeightPanel(name="w").layout(ndim) == expected


def test_an_explicit_axis_choice_is_honoured() -> None:
    """Choosing any axis switches to explicit mode: x falls back to the last
    axis, and without a Y axis there is nothing left to lay out as tiles."""
    assert WeightPanel(name="w", y_dim=0).layout(4) == (3, 0, None)
    assert WeightPanel(name="w", x_dim=0, y_dim=1, tile_dim=2).layout(4) == (0, 1, 2)
    assert WeightPanel(name="w", tile_dim=2).layout(4) == (3, None, None)


def test_conv_weights_render_as_kernels_not_a_flattened_row() -> None:
    """End to end for the axis default: the conv weight's picture must be
    taller than one heatmap row, which is what the flattened view produced."""
    with paused_session(TinyConvNet(), _conv_step) as session:
        rendered = weights_image(session, layer="conv", parameters=["conv.weight"])
        assert rendered.png is not None
        image = Image.open(io.BytesIO(rendered.png))
        # A 3x3 kernel tile is rendered at tile resolution; a 1-D row would be
        # a single short strip well under a hundred pixels tall.
        assert image.height > 100, rendered.note


# --- tools over a real client ----------------------------------------


@pytest.mark.parametrize(
    ("tool", "arguments"),
    [
        ("render_layer", {"layers": ["conv"], "include_input": True}),
        ("render_input", {}),
        ("render_weights", {"layer": "conv"}),
        ("render_histogram", {"layers": ["conv"]}),
        ("render_extreme_patches", {"layer": "conv"}),
    ],
)
def test_every_render_tool_returns_a_note_and_a_picture(
    tool: str, arguments: dict[str, Any]
) -> None:
    with paused_session(TinyConvNet(), _conv_step) as session:
        session.watch("conv")
        # One more batch so the watch accumulators have a bucket to draw from.
        before = session.pause_count
        session.step_batch()
        assert session.wait_until_paused(after_pauses=before, timeout=5.0)
        texts, images = _blocks(_call(session, tool, arguments))
        assert texts and texts[0], tool
        assert len(images) == 1, (tool, texts)
        assert images[0].mime_type == "image/png"


def test_render_layer_names_the_batch_it_drew() -> None:
    """The picture is of the last *captured* batch, which is not the live one
    while training runs — so the position travels with it."""
    with paused_session(TinyConvNet(), _conv_step) as session:
        rendered = layer_image(
            session, layers=["conv"], display=_DISPLAY, input_name="x"
        )
        assert rendered.png is not None
        assert "train batch 0" in rendered.note


def test_render_layer_reports_unknown_layers_rather_than_dropping_them() -> None:
    with paused_session(TinyConvNet(), _conv_step) as session:
        rendered = layer_image(
            session, layers=["conv", "nope"], display=_DISPLAY, input_name="x"
        )
        assert rendered.png is not None  # the known layer still drew
        assert "nope" in rendered.note


def test_rendering_before_the_first_capture_explains_itself() -> None:
    """An agent that asks too early gets told how to get a batch, not a blank."""
    session = nansense.start(TinyNet(), epochs=1, phases={"train": 1})
    rendered = layer_image(session, layers=["fc1"], display=_DISPLAY)
    assert rendered.png is None
    assert "pause()" in rendered.note and "refresh()" in rendered.note


def test_histograms_of_an_uncollected_layer_say_how_to_collect() -> None:
    """Histograms come from the watch accumulators, so an unwatched layer has
    nothing — the fix (watch it) belongs in the answer."""
    with paused_session(TinyNet()) as session:
        rendered = histogram_image(session, layers=["fc1"])
        assert rendered.png is None
        assert "watch_layers" in rendered.note


def test_patch_grids_of_an_unknown_layer_point_at_the_architecture() -> None:
    with paused_session(TinyNet()) as session:
        rendered = patches_image(session, layer="nope", display=_DISPLAY)
        assert rendered.png is None
        assert "get_architecture" in rendered.note


def test_weights_of_a_layer_without_parameters_say_so() -> None:
    """An fx intermediate (`relu`, `add`) is a layer with activations but no
    parameters; that is different from an unknown name."""
    with paused_session(TinyNet()) as session:
        rendered = weights_image(session, layer="relu")
        assert rendered.png is None
        assert "no parameters" in rendered.note
        assert "not a known layer" not in rendered.note


def test_weights_reject_a_parameter_of_another_layer() -> None:
    with paused_session(TinyNet()) as session:
        rendered = weights_image(session, layer="fc1", parameters=["fc2.weight"])
        assert rendered.png is None
        assert "fc1.weight" in rendered.note  # what it does have


def test_input_render_warns_when_no_normalization_was_given() -> None:
    """Without `input_mean`/`input_std` the renderer assumes `[0, 1]`, so a
    normalized input looks washed out; saying so beats a puzzling picture."""
    with paused_session(TinyConvNet(), _conv_step) as session:
        rendered = input_image(session, display=_DISPLAY, input_name="x")
        assert rendered.png is not None
        assert "input_mean" in rendered.note
