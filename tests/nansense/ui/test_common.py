"""Tests for strip-HTML assembly and pane resizing in nansense.ui.common."""

from __future__ import annotations

from typing import Literal

import pytest
import torch

from nansense.ui.common import _resizable_pane_props, _resize_handle, _strip_html
from nansense.ui.render import image_mime, render_strip
from nansense.ui.static import _PANEL_RESIZE_CSS, _PANEL_RESIZE_JS


def test_strip_html_scales_native_data_and_keeps_legend_crisp() -> None:
    strip = render_strip(torch.randn(1, 2, 8, 8), sample_idx=0)
    assert strip is not None
    html = _strip_html(strip)
    # One legend <img> (shown 1:1, no pixelated scaling) + one captioned <img>
    # per channel tile.
    assert html.count("<img") == 1 + len(strip.tiles)
    assert html.count("image-rendering:pixelated") == len(strip.tiles)
    assert html.count(f"data:{image_mime()};base64,") == 1 + len(strip.tiles)
    assert f"width:{strip.tiles[0].width}px" in html
    assert f"height:{strip.tiles[0].height}px" in html
    # Each tile carries its channel caption.
    assert "CHANNEL 0" in html and "CHANNEL 1" in html


def test_strip_html_carries_checkerboard_background() -> None:
    # Every data img sits over the GIMP-style checkerboard; opaque strips
    # cover it, transparent NaN/Inf cells reveal it.
    strip = render_strip(torch.randn(1, 2, 8, 8), sample_idx=0)
    assert strip is not None
    html = _strip_html(strip)
    assert "background-size:8px 8px" in html
    assert "background-image:" in html


def test_strip_html_uses_data_mime_for_nonfinite_strip() -> None:
    # A NaN/Inf strip is RGBA PNG; the data <img> data-URI must use the
    # strip's own png mime, while the legend keeps the global STRIP_FORMAT.
    tensor = torch.tensor([[float("nan"), float("inf"), 0.5, -0.5]])
    strip = render_strip(tensor, sample_idx=0)
    assert strip is not None
    assert strip.tiles[0].mime == "image/png"
    html = _strip_html(strip)
    assert "data:image/png;base64," in html  # tile img
    assert f"data:{image_mime()};base64," in html  # legend


def test_strip_html_empty_for_none() -> None:
    assert _strip_html(None) == ""


@pytest.mark.parametrize("side", ["left", "right"])
def test_resize_handle_marks_key_and_side(side: Literal["left", "right"]) -> None:
    handle = _resize_handle("main-input", side)
    assert "nansense-resize-handle" in handle.classes
    assert handle.props["data-resize-key"] == "main-input"
    assert handle.props["data-resize-side"] == side


def test_resize_script_matches_python_attribute_names() -> None:
    # The JS reads the attributes the Python side emits; if either name
    # drifts, dragging silently stops finding its pane.
    assert 'data-resize-pane="main-input"' in _resizable_pane_props("main-input")
    for attr in ("data-resize-pane", "data-resize-key", "data-resize-side"):
        assert attr in _PANEL_RESIZE_JS
    assert "nansense-resize-handle" in _PANEL_RESIZE_JS
    assert "nansense-resize-handle" in _PANEL_RESIZE_CSS
    # Width memory must be per browser session, not persistent.
    assert "sessionStorage" in _PANEL_RESIZE_JS
    assert "localStorage" not in _PANEL_RESIZE_JS
