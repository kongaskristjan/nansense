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


def test_strip_html_empty_for_none() -> None:
    assert _strip_html(None) == ""
