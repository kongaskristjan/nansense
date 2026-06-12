"""Tests for strip-HTML assembly in nansense.ui.common."""

from __future__ import annotations

import torch

from nansense.ui.common import _strip_html
from nansense.ui.render import image_mime, render_strip


def test_strip_html_scales_native_data_and_keeps_legend_crisp() -> None:
    strip = render_strip(torch.randn(1, 2, 8, 8), sample_idx=0)
    assert strip is not None
    html = _strip_html(strip)
    # One legend <img> (shown 1:1, no pixelated scaling) + one data <img>.
    assert html.count("<img") == 2
    assert html.count("image-rendering:pixelated") == 1
    assert html.count(f"data:{image_mime()};base64,") == 2
    assert f"width:{strip.width}px" in html
    assert f"height:{strip.height}px" in html


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
    assert strip.data_mime == "image/png"
    html = _strip_html(strip)
    assert "data:image/png;base64," in html  # data img
    assert f"data:{image_mime()};base64," in html  # legend


def test_strip_html_empty_for_none() -> None:
    assert _strip_html(None) == ""
