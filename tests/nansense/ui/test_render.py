"""Tests for tensor → image rendering."""

from __future__ import annotations

import io

import pytest
import torch
from PIL import Image

from nansense.patches import TypePatches
from nansense.ui import render
from nansense.ui.render import (
    HEAT_MAX_ALPHA,
    LEGEND_WIDTH,
    LINEAR_BIN_WIDTH,
    LINEAR_MAX_BINS,
    LINEAR_TILE_HEIGHT,
    PATCH_CELL_SIZE,
    TILE_SIZE,
    default_weight_dims,
    image_mime,
    render_image,
    render_patch_grid,
    render_strip,
    render_weight,
)


def _decode(image: bytes) -> Image.Image:
    return Image.open(io.BytesIO(image)).convert("RGB")


def _rgb_at(img: Image.Image, x: int, y: int) -> tuple[int, int, int]:
    pixel = img.getpixel((x, y))
    assert isinstance(pixel, tuple) and len(pixel) == 3
    r, g, b = pixel
    return int(r), int(g), int(b)


def test_chw_strip_is_single_native_resolution_image() -> None:
    # Small maps are encoded at their native H×W in one image — 8 tiles of
    # 12×16 with 1-px separators (max(1, 12 // 48)); the browser upscales the
    # whole strip to StripRender.width/height via CSS, so every tile spans
    # TILE_SIZE CSS px.
    tensor = torch.randn(4, 8, 16, 12)
    strip = render_strip(tensor, sample_idx=2)
    assert strip is not None
    assert _decode(strip.data_image).size == (8 * 12 + 7 * 1, 16)
    assert (strip.width, strip.height) == (
        round((8 * 12 + 7 * 1) * TILE_SIZE / 12),
        TILE_SIZE,
    )


def test_chw_strip_downsamples_large_maps_to_tile_size() -> None:
    # 2 tiles of 128×128 with a max(1, 128 // 48) = 2-px separator; native
    # and display size coincide since the tiles are already at TILE_SIZE.
    tensor = torch.randn(1, 2, 300, 200)
    strip = render_strip(tensor, sample_idx=0)
    assert strip is not None
    assert _decode(strip.data_image).size == (2 * TILE_SIZE + 2, TILE_SIZE)
    assert (strip.width, strip.height) == (2 * TILE_SIZE + 2, TILE_SIZE)


def test_chw_legend_is_display_resolution() -> None:
    strip = render_strip(torch.randn(1, 2, 8, 8), sample_idx=0)
    assert strip is not None
    assert _decode(strip.legend_image).size == (LEGEND_WIDTH, TILE_SIZE)


def test_1d_strip_dimensions() -> None:
    tensor = torch.randn(4, 10)
    strip = render_strip(tensor, sample_idx=0)
    assert strip is not None
    assert _decode(strip.data_image).size == (10, 1)
    assert (strip.width, strip.height) == (
        10 * LINEAR_BIN_WIDTH,
        LINEAR_TILE_HEIGHT,
    )
    assert _decode(strip.legend_image).size == (LEGEND_WIDTH, LINEAR_TILE_HEIGHT)


def test_1d_strip_caps_at_max_bins() -> None:
    tensor = torch.randn(4, LINEAR_MAX_BINS * 4)
    strip = render_strip(tensor, sample_idx=0)
    assert strip is not None
    assert _decode(strip.data_image).size == (LINEAR_MAX_BINS, 1)
    assert strip.width == LINEAR_MAX_BINS * LINEAR_BIN_WIDTH


def test_returns_none_for_none_tensor() -> None:
    assert render_strip(None, sample_idx=0) is None


def test_returns_none_for_out_of_range_sample() -> None:
    tensor = torch.randn(4, 8, 32, 32)
    assert render_strip(tensor, sample_idx=10) is None
    assert render_strip(tensor, sample_idx=-1) is None


def test_returns_none_for_unsupported_shape() -> None:
    # Per-sample shape would be [3, 4, 5, 6] — 4D, not supported.
    tensor = torch.randn(2, 3, 4, 5, 6)
    assert render_strip(tensor, sample_idx=0) is None


@pytest.mark.parametrize(
    ("n_tokens", "input_hw", "expected"),
    [
        (64, (32, 32), (0, 8, 8)),  # plain 8x8 grid, stride 4
        (64, (128, 128), (0, 8, 8)),  # same grid at stride 16
        (65, (32, 32), (1, 8, 8)),  # class token
        (66, (32, 32), (2, 8, 8)),  # class + distillation (DeiT)
        (68, (32, 32), (4, 8, 8)),  # four registers
        (48, (96, 128), (0, 6, 8)),  # non-square input, stride 16
        (1024, (32, 32), (0, 32, 32)),  # stride 1: a token per pixel
        (192, (32, 32), None),  # a ViT embedding dim — no integer stride
        (50, (32, 32), None),
    ],
)
def test_token_grid(
    n_tokens: int, input_hw: tuple[int, int], expected: tuple[int, int, int] | None
) -> None:
    assert render._token_grid(n_tokens, input_hw) == expected


def test_2d_tokens_unflatten_onto_input_grid() -> None:
    # [tokens, dim] with a fitting token axis renders exactly like the
    # explicitly unflattened [dim, h, w] conv-style view of the same data.
    tensor = torch.randn(3, 64, 24)
    strip = render_strip(tensor, sample_idx=1, input_hw=(32, 32))
    grid = tensor[1].T.reshape(24, 8, 8).unsqueeze(0)
    assert strip == render_strip(grid, sample_idx=0)


def test_2d_tokens_drop_leading_class_token() -> None:
    tensor = torch.randn(1, 65, 24)
    strip = render_strip(tensor, sample_idx=0, input_hw=(32, 32))
    grid = tensor[0, 1:].T.reshape(24, 8, 8).unsqueeze(0)
    assert strip == render_strip(grid, sample_idx=0)


def test_2d_tokens_fit_on_second_axis() -> None:
    # [dim, tokens]: only axis 1 matches the grid, so it is the token axis.
    tensor = torch.randn(1, 24, 64)
    strip = render_strip(tensor, sample_idx=0, input_hw=(32, 32))
    grid = tensor[0].reshape(24, 8, 8).unsqueeze(0)
    assert strip == render_strip(grid, sample_idx=0)


def test_2d_ambiguous_axes_prefer_tokens_first() -> None:
    # Both axes fit the grid (64 tokens, 64 dims); batch_first slicing makes
    # [tokens, dim] the common layout, so axis 0 wins.
    tensor = torch.randn(1, 64, 64)
    strip = render_strip(tensor, sample_idx=0, input_hw=(32, 32))
    grid = tensor[0].T.reshape(64, 8, 8).unsqueeze(0)
    assert strip == render_strip(grid, sample_idx=0)


def test_2d_without_grid_fit_renders_single_heatmap_tile() -> None:
    # Neither 7 nor 13 fits a 32x32 patch grid: a single [7, 13] tile.
    tensor = torch.randn(2, 7, 13)
    strip = render_strip(tensor, sample_idx=0, input_hw=(32, 32))
    assert strip == render_strip(tensor[0].reshape(1, 1, 7, 13), sample_idx=0)


def test_2d_without_input_hw_renders_single_heatmap_tile() -> None:
    tensor = torch.randn(2, 64, 24)
    strip = render_strip(tensor, sample_idx=0)
    assert strip == render_strip(tensor[0].reshape(1, 1, 64, 24), sample_idx=0)


def test_zero_variance_tensor_renders() -> None:
    # All-zero input must not crash the diverging colormap (abs_max=0 edge).
    tensor = torch.zeros(2, 4, 8, 8)
    strip = render_strip(tensor, sample_idx=0)
    assert strip is not None
    assert _decode(strip.data_image).size == (4 * 8 + 3 * 1, 8)


def test_strip_uses_diverging_colormap_with_white_separator() -> None:
    # Tile 0 is all +max → pure red; tile 1 all -max → pure blue; the 1-px
    # separator between them (x = 8) stays white.
    sample = torch.stack([torch.ones(8, 8), -torch.ones(8, 8)])
    strip = render_strip(sample.unsqueeze(0), sample_idx=0)
    assert strip is not None
    img = _decode(strip.data_image)
    assert _rgb_at(img, 4, 4) == (255, 0, 0)
    assert _rgb_at(img, 8, 4) == (255, 255, 255)
    assert _rgb_at(img, 8 + 1 + 4, 4) == (0, 0, 255)


@pytest.mark.parametrize("tile_w, gap", [(8, 1), (48, 1), (96, 2), (128, 2)])
def test_separator_width_scales_with_tile_width(tile_w: int, gap: int) -> None:
    # Separators are max(1, tile_width // TILE_GAP_DIVISOR) native pixels.
    tensor = torch.randn(1, 2, 8, tile_w)
    strip = render_strip(tensor, sample_idx=0)
    assert strip is not None
    assert _decode(strip.data_image).size == (2 * tile_w + gap, 8)


@pytest.mark.parametrize(
    "fmt, magic, mime",
    [("BMP", b"BM", "image/bmp"), ("PNG", b"\x89PNG", "image/png")],
)
def test_strip_format_switch(
    monkeypatch: pytest.MonkeyPatch, fmt: str, magic: bytes, mime: str
) -> None:
    # STRIP_FORMAT drives the encoding of every rendered image and the MIME
    # type the UI puts in data URIs; both formats stay decodable.
    monkeypatch.setattr(render, "STRIP_FORMAT", fmt)
    strip = render_strip(torch.randn(1, 2, 8, 8), sample_idx=0)
    assert strip is not None
    assert strip.data_image.startswith(magic)
    assert strip.legend_image.startswith(magic)
    assert _decode(strip.data_image).size == (2 * 8 + 1, 8)
    input_image = render_image(torch.rand(1, 3, 4, 4), sample_idx=0)
    assert input_image is not None
    assert input_image.startswith(magic)
    assert image_mime() == mime


def test_default_strip_format_is_bmp() -> None:
    strip = render_strip(torch.randn(1, 1, 4, 4), sample_idx=0)
    assert strip is not None
    assert strip.data_image.startswith(b"BM")


@pytest.mark.parametrize("channels", [1, 3])
def test_input_image_keeps_native_resolution(channels: int) -> None:
    # The UI scales the input image to its display size via CSS, so the PNG
    # stays at the sample's native H×W.
    tensor = torch.rand(2, channels, 16, 16)
    png = render_image(tensor, sample_idx=0)
    assert png is not None
    assert _decode(png).size == (16, 16)


def test_input_image_denormalizes_with_mean_std() -> None:
    # Tensor is `(value - mean) / std`; after denorm it should hit 0.5 ->
    # mid-gray (128) for every channel.
    mean = (0.5, 0.4, 0.6)
    std = (0.1, 0.2, 0.3)
    chans = [torch.full((4, 4), (0.5 - m) / s) for m, s in zip(mean, std, strict=True)]
    tensor = torch.stack(chans)[None]
    png = render_image(tensor, sample_idx=0, mean=mean, std=std)
    assert png is not None
    r, g, b = _rgb_at(_decode(png), 0, 0)
    assert abs(r - 128) <= 1
    assert abs(g - 128) <= 1
    assert abs(b - 128) <= 1


def test_input_image_default_assumes_unit_range() -> None:
    # An all-1.0 image should render as pure white without normalization.
    tensor = torch.ones(1, 3, 4, 4)
    png = render_image(tensor, sample_idx=0)
    assert png is not None
    assert _rgb_at(_decode(png), 0, 0) == (255, 255, 255)


def test_input_image_returns_none_for_unsupported_channels() -> None:
    tensor = torch.rand(1, 4, 8, 8)
    assert render_image(tensor, sample_idx=0) is None


def test_input_image_returns_none_for_unsupported_shape() -> None:
    assert render_image(torch.rand(3, 8, 8), sample_idx=0) is None


def test_input_image_returns_none_for_out_of_range_sample() -> None:
    tensor = torch.rand(2, 3, 8, 8)
    assert render_image(tensor, sample_idx=5) is None


def test_input_image_returns_none_when_mean_std_size_mismatched() -> None:
    tensor = torch.rand(1, 3, 8, 8)
    assert render_image(tensor, sample_idx=0, mean=(0.5,), std=(0.2,)) is None


@pytest.mark.parametrize(
    "ndim, x, y, tile, fixed",
    [
        (1, 0, None, None, ()),
        (2, 1, 0, None, ()),
        (3, 2, 1, 0, ()),
        (4, 3, 2, 1, (0,)),
    ],
)
def test_default_weight_dims(
    ndim: int, x: int, y: int | None, tile: int | None, fixed: tuple[int, ...]
) -> None:
    dims = default_weight_dims(ndim)
    assert (dims.x_dim, dims.y_dim, dims.tile_dim, dims.fixed_dims) == (
        x,
        y,
        tile,
        fixed,
    )


def test_render_weight_4d_default_lays_kernels_across_in_channels() -> None:
    # [out=8, in=3, kH=3, kW=3] default: kH×kW tiles across `in` (3 tiles),
    # `out` pinned by index.
    w = torch.randn(8, 3, 3, 3)
    d = default_weight_dims(4)
    strip = render_weight(
        w,
        x_dim=d.x_dim,
        y_dim=d.y_dim,
        tile_dim=d.tile_dim,
        fixed={f: 0 for f in d.fixed_dims},
    )
    assert strip is not None
    # 3 kernel tiles of 3×3 with 1-px separators, in a single image.
    assert _decode(strip.data_image).size == (3 * 3 + 2 * 1, 3)
    assert (strip.width, strip.height) == (
        round((3 * 3 + 2 * 1) * TILE_SIZE / 3),
        TILE_SIZE,
    )


def test_render_weight_2d_is_single_image_tile() -> None:
    w = torch.randn(10, 4)
    d = default_weight_dims(2)
    strip = render_weight(
        w, x_dim=d.x_dim, y_dim=d.y_dim, tile_dim=d.tile_dim, fixed={}
    )
    assert strip is not None
    assert _decode(strip.data_image).size == (4, 10)
    assert (strip.width, strip.height) == (TILE_SIZE, TILE_SIZE)


def test_render_weight_1d_is_single_row() -> None:
    w = torch.randn(16)
    strip = render_weight(w, x_dim=0, y_dim=None, tile_dim=None, fixed={})
    assert strip is not None
    assert _decode(strip.data_image).size == (16, 1)
    assert (strip.width, strip.height) == (
        16 * LINEAR_BIN_WIDTH,
        LINEAR_TILE_HEIGHT,
    )


def test_render_weight_custom_tile_axis_changes_tile_count() -> None:
    # Tile across `out`=8 instead of the default `in`, pinning `in` by index.
    w = torch.randn(8, 3, 3, 3)
    strip = render_weight(w, x_dim=3, y_dim=2, tile_dim=0, fixed={1: 1})
    assert strip is not None
    assert _decode(strip.data_image).size == (8 * 3 + 7 * 1, 3)


def test_render_weight_clamps_out_of_range_fixed_index() -> None:
    w = torch.randn(8, 3, 3, 3)
    d = default_weight_dims(4)
    strip = render_weight(
        w, x_dim=d.x_dim, y_dim=d.y_dim, tile_dim=d.tile_dim, fixed={0: 999}
    )
    assert strip is not None  # index clamped into range rather than crashing


def test_render_weight_returns_none_for_none_tensor() -> None:
    assert render_weight(None, x_dim=0, y_dim=None, tile_dim=None, fixed={}) is None


def test_render_weight_returns_none_for_duplicate_axes() -> None:
    w = torch.randn(8, 3, 3, 3)
    assert render_weight(w, x_dim=2, y_dim=2, tile_dim=0, fixed={1: 0}) is None


def test_render_weight_returns_none_for_out_of_range_axis() -> None:
    w = torch.randn(4, 4)
    assert render_weight(w, x_dim=5, y_dim=0, tile_dim=None, fixed={}) is None


def _type_patches(
    values: torch.Tensor,
    patches: torch.Tensor,
    *,
    heat: torch.Tensor | None = None,
    crop: bool = False,
    top: torch.Tensor | None = None,
    left: torch.Tensor | None = None,
    input_hw: tuple[int, int] = (8, 8),
) -> TypePatches:
    c, n = values.shape
    if heat is None:
        heat = torch.zeros(c, n, 1, 1)
    if top is None:
        top = torch.zeros(c, n, dtype=torch.int64)
    if left is None:
        left = torch.zeros(c, n, dtype=torch.int64)
    return TypePatches(
        values=values,
        patches=patches,
        heat=heat,
        top=top,
        left=left,
        input_hw=input_hw,
        crop=crop,
    )


def test_patch_grid_layout_and_css_size() -> None:
    tp = _type_patches(torch.zeros(2, 5), torch.rand(2, 5, 3, 4, 4))
    grid = render_patch_grid(tp)
    assert grid is not None
    # 2 channel columns × 5 sample rows of 4×4 cells with 1-px gaps.
    assert _decode(grid.image).size == (2 * 4 + 1, 5 * 4 + 4)
    assert (grid.width, grid.height) == (
        round((2 * 4 + 1) * PATCH_CELL_SIZE / 4),
        round((5 * 4 + 4) * PATCH_CELL_SIZE / 4),
    )


def test_patch_grid_is_png_regardless_of_strip_format() -> None:
    # Grids ignore STRIP_FORMAT: multi-MB BMP grid messages can pause the
    # websocket transport and crash its keepalive (see PatchGridRender).
    tp = _type_patches(torch.zeros(2, 5), torch.rand(2, 5, 3, 4, 4))
    grid = render_patch_grid(tp)
    assert grid is not None
    assert grid.image.startswith(b"\x89PNG")
    assert grid.mime == "image/png"


def test_patch_grid_denormalizes_with_mean_std() -> None:
    tp = _type_patches(torch.zeros(1, 5), torch.zeros(1, 5, 3, 4, 4))
    grid = render_patch_grid(tp, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    assert grid is not None
    assert _rgb_at(_decode(grid.image), 0, 0) == (127, 127, 127)


def test_patch_grid_marks_unfilled_slots_gray() -> None:
    values = torch.full((1, 5), float("-inf"))
    values[0, 0] = 1.0
    tp = _type_patches(values, torch.ones(1, 5, 3, 4, 4))
    grid = render_patch_grid(tp)
    assert grid is not None
    img = _decode(grid.image)
    assert _rgb_at(img, 0, 0) == (255, 255, 255)  # filled slot, white patch
    assert _rgb_at(img, 0, 5) == (235, 235, 235)  # row 1 never filled


def test_patch_grid_returns_none_until_first_fill() -> None:
    tp = _type_patches(torch.full((2, 5), float("inf")), torch.zeros(2, 5, 3, 4, 4))
    assert render_patch_grid(tp) is None


@pytest.mark.parametrize(("sign", "channel"), [(1.0, 0), (-1.0, 2)])
def test_patch_grid_heatmap_tints_red_or_blue(sign: float, channel: int) -> None:
    values = torch.full((1, 5), float("-inf"))
    values[0, 0] = sign
    heat = torch.zeros(1, 5, 2, 2)
    heat[0, 0] = sign  # uniform map at the grid-wide |max|
    tp = _type_patches(values, torch.ones(1, 5, 3, 4, 4), heat=heat)
    grid = render_patch_grid(tp, heatmap=True)
    assert grid is not None
    rgb = _rgb_at(_decode(grid.image), 0, 0)
    # White patch blended with a full-strength overlay at HEAT_MAX_ALPHA:
    # the tint channel stays 255, the others drop to 255 * (1 - alpha).
    faded = round(255 * (1 - HEAT_MAX_ALPHA))
    assert rgb[channel] == 255
    assert abs(rgb[1] - faded) <= 1
    assert abs(rgb[2 - channel] - faded) <= 1


def test_patch_grid_heatmap_crops_window_region() -> None:
    # Crop covers the input's top-left quadrant; only the matching
    # activation-map cell (positive) should drive the blend.
    values = torch.full((1, 5), float("-inf"))
    values[0, 0] = 1.0
    heat = torch.zeros(1, 5, 2, 2)
    heat[0, 0, 0, 0] = 1.0
    tp = _type_patches(
        values,
        torch.ones(1, 5, 3, 4, 4),
        heat=heat,
        crop=True,
        input_hw=(8, 8),
    )
    grid = render_patch_grid(tp, heatmap=True)
    assert grid is not None
    rgb = _rgb_at(_decode(grid.image), 0, 0)
    assert rgb[0] == 255 and rgb[1] < 255  # tinted red inside the window


def test_patch_grid_grayscale_input() -> None:
    tp = _type_patches(torch.zeros(1, 5), torch.full((1, 5, 1, 4, 4), 0.5))
    grid = render_patch_grid(tp)
    assert grid is not None
    assert _rgb_at(_decode(grid.image), 0, 0) == (127, 127, 127)


def test_patch_grid_heat_legend_only_when_heatmap_enabled() -> None:
    values = torch.full((1, 5), float("-inf"))
    values[0, 0] = 1.0
    heat = torch.zeros(1, 5, 2, 2)
    heat[0, 0] = 1.0
    tp = _type_patches(values, torch.ones(1, 5, 3, 4, 4), heat=heat)
    plain = render_patch_grid(tp)
    assert plain is not None and plain.heat_legend is None
    grid = render_patch_grid(tp, heatmap=True)
    assert grid is not None and grid.heat_legend is not None
    # Display-resolution colorbar matching the grid's CSS height.
    assert _decode(grid.heat_legend).size[1] == grid.height
    # All-zero heat has no scale to show, even with the heatmap enabled.
    flat = _type_patches(values, torch.ones(1, 5, 3, 4, 4))
    rendered = render_patch_grid(flat, heatmap=True)
    assert rendered is not None and rendered.heat_legend is None
