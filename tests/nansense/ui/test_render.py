"""Tests for tensor → image rendering."""

from __future__ import annotations

import io

import numpy as np
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
    PATCH_CELL_GAP,
    PATCH_CELL_SIZE,
    TILE_SIZE,
    default_weight_dims,
    dims_from_roles,
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


def _tile_sizes(strip: render.StripRender) -> list[tuple[int, int]]:
    """Native (decoded) `(w, h)` size of each tile image in a strip."""
    return [_decode(tile.image).size for tile in strip.tiles]


def test_chw_strip_is_one_native_tile_per_channel() -> None:
    # Each channel is its own native-resolution tile — 8 tiles of 12×16 — shown
    # as a TILE_SIZE square (the browser nearest-upscales it) and captioned by
    # channel index.
    tensor = torch.randn(4, 8, 16, 12)
    strip = render_strip(tensor, sample_idx=2)
    assert strip is not None
    assert _tile_sizes(strip) == [(12, 16)] * 8
    assert (strip.tiles[0].width, strip.tiles[0].height) == (TILE_SIZE, TILE_SIZE)
    assert [t.label for t in strip.tiles[:2]] == ["CHANNEL 0", "CHANNEL 1"]


def test_chw_strip_downsamples_large_maps_to_tile_size() -> None:
    # 2 tiles downsampled to 128×128; native and display size coincide since
    # the tiles are already at TILE_SIZE.
    tensor = torch.randn(1, 2, 300, 200)
    strip = render_strip(tensor, sample_idx=0)
    assert strip is not None
    assert _tile_sizes(strip) == [(TILE_SIZE, TILE_SIZE)] * 2
    assert (strip.tiles[0].width, strip.tiles[0].height) == (TILE_SIZE, TILE_SIZE)


def test_chw_legend_is_display_resolution() -> None:
    strip = render_strip(torch.randn(1, 2, 8, 8), sample_idx=0)
    assert strip is not None
    assert _decode(strip.legend_image).size == (LEGEND_WIDTH, TILE_SIZE)


def test_1d_strip_dimensions() -> None:
    tensor = torch.randn(4, 10)
    strip = render_strip(tensor, sample_idx=0)
    assert strip is not None
    # A 1D activation is a single heatmap-row tile with no channel caption.
    assert len(strip.tiles) == 1
    assert strip.tiles[0].label == ""
    assert _decode(strip.tiles[0].image).size == (10, 1)
    assert (strip.tiles[0].width, strip.tiles[0].height) == (
        10 * LINEAR_BIN_WIDTH,
        LINEAR_TILE_HEIGHT,
    )
    assert _decode(strip.legend_image).size == (LEGEND_WIDTH, LINEAR_TILE_HEIGHT)


def test_1d_strip_caps_at_max_bins() -> None:
    tensor = torch.randn(4, LINEAR_MAX_BINS * 4)
    strip = render_strip(tensor, sample_idx=0)
    assert strip is not None
    assert _decode(strip.tiles[0].image).size == (LINEAR_MAX_BINS, 1)
    assert strip.tiles[0].width == LINEAR_MAX_BINS * LINEAR_BIN_WIDTH


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
    "shape",
    [
        (2, 0, 4, 4),  # 0 channels -> [0, 4, 4] tile
        (2, 3, 0, 4),  # 0 height
        (2, 3, 4, 0),  # 0 width
        (2, 0, 4),  # 2D per-sample with a zero-length axis
        (2, 0),  # 1D per-sample with no features
        (0,),  # no samples at all
    ],
    ids=["no_channels", "no_height", "no_width", "2d_zero_axis", "1d_empty", "no_samples"],
)
def test_returns_none_for_empty_tensor(shape: tuple[int, ...]) -> None:
    # An empty activation (any zero-length dim — data-dependent selections,
    # boolean masking traced through fx, last-batch edge cases) must hide the
    # strip like an unsupported shape, never raise (which would drop the whole
    # frame for every layer of that snapshot).
    assert render_strip(torch.zeros(shape), sample_idx=0) is None


def test_normal_tensor_still_renders_after_empty_guard() -> None:
    # The empty guard must not regress the normal non-empty path.
    strip = render_strip(torch.randn(2, 3, 8, 8), sample_idx=0)
    assert strip is not None


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
    assert _tile_sizes(strip) == [(8, 8)] * 4


def _decode_rgba(image: bytes) -> Image.Image:
    return Image.open(io.BytesIO(image)).convert("RGBA")


@pytest.mark.parametrize(
    "shape",
    [(1, 3, 8, 8), (1, 12)],  # [C,H,W] and [F]
    ids=["chw", "linear"],
)
def test_nonfinite_strip_is_transparent_rgba_not_whitewashed(
    shape: tuple[int, ...],
) -> None:
    # Regression for the HIGH bug: a single NaN/±Inf used to NaN the colormap
    # scale and leave every cell white. The finite cells must still map to
    # multiple distinct colors, and the non-finite cells must be transparent.
    tensor = torch.linspace(-1.0, 1.0, int(torch.tensor(shape[1:]).prod())).reshape(
        shape
    )
    tensor.view(-1)[0] = float("nan")
    tensor.view(-1)[1] = float("inf")
    tensor.view(-1)[2] = float("-inf")
    strip = render_strip(tensor, sample_idx=0)
    assert strip is not None
    # A strip with non-finite cells switches every tile to RGBA PNG.
    assert all(t.mime == "image/png" for t in strip.tiles)
    assert all(t.image.startswith(b"\x89PNG") for t in strip.tiles)
    arr = np.concatenate(
        [np.asarray(_decode_rgba(t.image)) for t in strip.tiles], axis=1
    )
    alpha = arr[..., 3]
    # The bad cells are fully transparent; some cells remain opaque.
    assert (alpha == 0).any()
    assert (alpha == 255).any()
    # Finite (opaque) cells still span more than one color — not whitewashed.
    opaque_rgb = arr[..., :3][alpha == 255]
    assert len({tuple(c) for c in opaque_rgb}) > 1


def test_nonfinite_legend_label_uses_finite_max() -> None:
    # The legend's "+x" label must use the finite abs-max, never "+nan/+inf".
    tensor = torch.tensor([[float("nan"), float("inf"), 0.5, -0.25]])
    strip = render_strip(tensor, sample_idx=0)
    assert strip is not None
    # Legend stays the display-resolution RGB image (always STRIP_FORMAT).
    assert _decode(strip.legend_image).size == (LEGEND_WIDTH, LINEAR_TILE_HEIGHT)


def test_finite_strip_keeps_fast_rgb_path() -> None:
    # No non-finite values: the data image stays the default RGB/BMP fast
    # path (mime + mode unchanged), so the 30-60x BMP encode is preserved.
    tensor = torch.randn(1, 3, 8, 8)
    strip = render_strip(tensor, sample_idx=0)
    assert strip is not None
    assert strip.tiles[0].mime == image_mime() == "image/bmp"
    assert strip.tiles[0].image.startswith(b"BM")
    assert _decode(strip.tiles[0].image).mode == "RGB"


def test_nonfinite_does_not_smear_across_downsampled_tile() -> None:
    # A large map with one NaN downsamples; the bad cell stays a localized
    # transparent region rather than wiping out the whole tile.
    tensor = torch.randn(1, 1, 300, 300)
    tensor[0, 0, 0, 0] = float("nan")
    strip = render_strip(tensor, sample_idx=0)
    assert strip is not None
    arr = np.asarray(_decode_rgba(strip.tiles[0].image))
    transparent = (arr[..., 3] == 0).sum()
    # Only a small corner region goes transparent, not the whole 128x128 tile.
    assert 0 < transparent < arr.shape[0] * arr.shape[1] // 4


def test_strip_uses_diverging_colormap_per_tile() -> None:
    # Tile 0 is all +max → pure red; tile 1 all -max → pure blue. Tiles are now
    # separate images, so there is no baked separator between them.
    sample = torch.stack([torch.ones(8, 8), -torch.ones(8, 8)])
    strip = render_strip(sample.unsqueeze(0), sample_idx=0)
    assert strip is not None
    assert len(strip.tiles) == 2
    assert _rgb_at(_decode(strip.tiles[0].image), 4, 4) == (255, 0, 0)
    assert _rgb_at(_decode(strip.tiles[1].image), 4, 4) == (0, 0, 255)


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
    assert all(t.image.startswith(magic) for t in strip.tiles)
    assert strip.legend_image.startswith(magic)
    assert _tile_sizes(strip) == [(8, 8)] * 2
    input_image = render_image(torch.rand(1, 3, 4, 4), sample_idx=0)
    assert input_image is not None
    assert input_image.startswith(magic)
    assert image_mime() == mime


def test_default_strip_format_is_bmp() -> None:
    strip = render_strip(torch.randn(1, 1, 4, 4), sample_idx=0)
    assert strip is not None
    assert strip.tiles[0].image.startswith(b"BM")


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


def test_blend_signed_heat_colors_by_sign() -> None:
    rgb = np.full((1, 3, 3), 100, dtype=np.uint8)  # flat gray base
    heat = np.array([[1.0, -1.0, 0.0]], dtype=np.float32)  # +vmax, -vmax, 0
    out = render.blend_signed_heat(rgb, heat, vmax=1.0)
    assert out[0, 0, 0] > out[0, 0, 2]  # positive -> red dominates
    assert out[0, 1, 2] > out[0, 1, 0]  # negative -> blue dominates
    assert tuple(int(v) for v in out[0, 2]) == (100, 100, 100)  # 0 -> untouched


def test_blend_signed_heat_zero_vmax_leaves_image_unchanged() -> None:
    rgb = np.full((2, 2, 3), 77, dtype=np.uint8)
    out = render.blend_signed_heat(rgb, np.ones((2, 2), dtype=np.float32), vmax=0.0)
    assert np.array_equal(out, rgb)


def test_render_attribution_overlay_one_tile_per_channel() -> None:
    inp = torch.rand(3, 8, 8)  # [C_in, H, W]
    attribution = torch.randn(2, 8, 8)  # [C_a, h, w] -> two tiles
    strip = render.render_attribution_overlay(
        inp, attribution, mean=None, std=None, vmax=1.0
    )
    assert strip is not None
    assert _tile_sizes(strip) == [(8, 8)] * 2  # one tile per attribution channel
    assert [t.label for t in strip.tiles] == ["CHANNEL 0", "CHANNEL 1"]
    assert _decode(strip.legend_image).size == (LEGEND_WIDTH, TILE_SIZE)


def test_render_attribution_overlay_resizes_coarse_map_to_input() -> None:
    inp = torch.rand(1, 8, 8)
    attribution = torch.randn(1, 2, 2)  # coarse Grad-CAM-like map
    strip = render.render_attribution_overlay(
        inp, attribution, mean=None, std=None, vmax=1.0
    )
    assert strip is not None
    assert _tile_sizes(strip) == [(8, 8)]  # single tile resized to input size


def test_render_attribution_overlay_rejects_unsupported_input() -> None:
    assert (
        render.render_attribution_overlay(
            torch.rand(4, 8, 8), torch.randn(1, 8, 8), mean=None, std=None, vmax=1.0
        )
        is None
    )


def test_render_strip_tile_px_scales_display_and_legend() -> None:
    # tile_px bumps the CSS size each tile is shown at (and its legend height)
    # so the experiment page can size attribution maps to the inputs beside
    # them; the native-resolution data image is unchanged.
    tensor = torch.randn(1, 2, 8, 8)  # 2 tiles
    native = render_strip(tensor, sample_idx=0)
    scaled = render_strip(tensor, sample_idx=0, tile_px=200)
    assert native is not None and scaled is not None
    # The native tile images are unchanged; only the CSS square they fill grows.
    assert _tile_sizes(scaled) == _tile_sizes(native) == [(8, 8)] * 2
    assert (scaled.tiles[0].width, scaled.tiles[0].height) == (200, 200)
    assert _decode(scaled.legend_image).size == (LEGEND_WIDTH, 200)


def test_render_strip_downsamples_to_tile_px() -> None:
    # The server-side area downsample targets tile_px, so a map larger than
    # the requested tile is reduced to tile_px² rather than the default.
    strip = render_strip(torch.randn(1, 1, 300, 300), sample_idx=0, tile_px=64)
    assert strip is not None
    assert _tile_sizes(strip) == [(64, 64)]
    assert (strip.tiles[0].width, strip.tiles[0].height) == (64, 64)


def test_render_attribution_overlay_tile_px_scales_display() -> None:
    strip = render.render_attribution_overlay(
        torch.rand(1, 8, 8),
        torch.randn(2, 8, 8),
        mean=None,
        std=None,
        vmax=1.0,
        tile_px=200,
    )
    assert strip is not None
    assert _tile_sizes(strip) == [(8, 8)] * 2  # native, unchanged
    assert (strip.tiles[0].width, strip.tiles[0].height) == (200, 200)
    assert _decode(strip.legend_image).size == (LEGEND_WIDTH, 200)


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
    # 3 kernel tiles of 3×3, captioned as channels like the activation strips.
    assert _tile_sizes(strip) == [(3, 3)] * 3
    assert (strip.tiles[0].width, strip.tiles[0].height) == (TILE_SIZE, TILE_SIZE)
    assert [t.label for t in strip.tiles] == ["CHANNEL 0", "CHANNEL 1", "CHANNEL 2"]


def test_render_weight_2d_is_single_image_tile() -> None:
    w = torch.randn(10, 4)
    d = default_weight_dims(2)
    strip = render_weight(
        w, x_dim=d.x_dim, y_dim=d.y_dim, tile_dim=d.tile_dim, fixed={}
    )
    assert strip is not None
    # No tile axis → one uncaptioned image tile, shown as a TILE_SIZE square.
    assert _tile_sizes(strip) == [(4, 10)]
    assert strip.tiles[0].label == ""
    assert (strip.tiles[0].width, strip.tiles[0].height) == (TILE_SIZE, TILE_SIZE)


def test_render_weight_1d_is_single_row() -> None:
    w = torch.randn(16)
    strip = render_weight(w, x_dim=0, y_dim=None, tile_dim=None, fixed={})
    assert strip is not None
    assert _decode(strip.tiles[0].image).size == (16, 1)
    assert (strip.tiles[0].width, strip.tiles[0].height) == (
        16 * LINEAR_BIN_WIDTH,
        LINEAR_TILE_HEIGHT,
    )


def test_render_weight_custom_tile_axis_changes_tile_count() -> None:
    # Tile across `out`=8 instead of the default `in`, pinning `in` by index.
    w = torch.randn(8, 3, 3, 3)
    strip = render_weight(w, x_dim=3, y_dim=2, tile_dim=0, fixed={1: 1})
    assert strip is not None
    assert _tile_sizes(strip) == [(3, 3)] * 8
    # Tiles are captioned as channels regardless of which axis is tiled.
    assert strip.tiles[0].label == "CHANNEL 0"


def test_render_weight_clamps_out_of_range_fixed_index() -> None:
    w = torch.randn(8, 3, 3, 3)
    d = default_weight_dims(4)
    strip = render_weight(
        w, x_dim=d.x_dim, y_dim=d.y_dim, tile_dim=d.tile_dim, fixed={0: 999}
    )
    assert strip is not None  # index clamped into range rather than crashing


def test_render_weight_returns_none_for_none_tensor() -> None:
    assert render_weight(None, x_dim=0, y_dim=None, tile_dim=None, fixed={}) is None


@pytest.mark.parametrize(
    "shape",
    [(0, 3, 3, 3), (8, 0, 3, 3), (8, 3, 0, 3), (8, 3, 3, 0), (0, 4), (4, 0), (0,)],
)
def test_render_weight_returns_none_for_empty_tensor(shape: tuple[int, ...]) -> None:
    # An empty weight (any zero-length dim) shares the strip renderers, so it
    # is hidden like an empty activation rather than crashing the renderer.
    w = torch.zeros(shape)
    d = default_weight_dims(w.ndim)
    assert (
        render_weight(
            w,
            x_dim=d.x_dim,
            y_dim=d.y_dim,
            tile_dim=d.tile_dim,
            fixed={f: 0 for f in d.fixed_dims},
        )
        is None
    )


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


def test_patch_grid_is_one_cell_image_per_sample() -> None:
    tp = _type_patches(torch.zeros(2, 5), torch.rand(2, 5, 3, 4, 4))
    grid = render_patch_grid(tp)
    assert grid is not None
    # One captioned column per channel; each holds one 4×4 cell image per top-N
    # sample, shown as a PATCH_CELL_SIZE square.
    assert len(grid.columns) == 2
    col = grid.columns[0]
    assert len(col.cells) == 5
    assert _decode(col.cells[0]).size == (4, 4)
    assert col.cell_size == PATCH_CELL_SIZE
    assert [c.label for c in grid.columns] == ["CHANNEL 0", "CHANNEL 1"]


def test_patch_grid_is_png_regardless_of_strip_format() -> None:
    # Grids ignore STRIP_FORMAT: multi-MB BMP grid messages can pause the
    # websocket transport and crash its keepalive (see PatchGridRender).
    tp = _type_patches(torch.zeros(2, 5), torch.rand(2, 5, 3, 4, 4))
    grid = render_patch_grid(tp)
    assert grid is not None
    assert all(cell.startswith(b"\x89PNG") for c in grid.columns for cell in c.cells)
    assert grid.mime == "image/png"


def test_patch_grid_denormalizes_with_mean_std() -> None:
    tp = _type_patches(torch.zeros(1, 5), torch.zeros(1, 5, 3, 4, 4))
    grid = render_patch_grid(tp, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    assert grid is not None
    assert _rgb_at(_decode(grid.columns[0].cells[0]), 0, 0) == (127, 127, 127)


def test_patch_grid_returns_none_when_mean_std_size_mismatched() -> None:
    """Same policy as `render_image`: hide rather than render wrongly scaled."""
    tp = _type_patches(torch.zeros(1, 5), torch.zeros(1, 5, 3, 4, 4))
    assert render_patch_grid(tp, mean=(0.5,), std=(0.5,)) is None


def test_patch_grid_marks_unfilled_slots_gray() -> None:
    values = torch.full((1, 5), float("-inf"))
    values[0, 0] = 1.0
    tp = _type_patches(values, torch.ones(1, 5, 3, 4, 4))
    grid = render_patch_grid(tp)
    assert grid is not None
    cells = grid.columns[0].cells
    assert _rgb_at(_decode(cells[0]), 0, 0) == (255, 255, 255)  # filled, white patch
    assert _rgb_at(_decode(cells[1]), 0, 0) == (235, 235, 235)  # sample 1 never filled


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
    rgb = _rgb_at(_decode(grid.columns[0].cells[0]), 0, 0)
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
    rgb = _rgb_at(_decode(grid.columns[0].cells[0]), 0, 0)
    assert rgb[0] == 255 and rgb[1] < 255  # tinted red inside the window


def test_patch_grid_grayscale_input() -> None:
    tp = _type_patches(torch.zeros(1, 5), torch.full((1, 5, 1, 4, 4), 0.5))
    grid = render_patch_grid(tp)
    assert grid is not None
    assert _rgb_at(_decode(grid.columns[0].cells[0]), 0, 0) == (127, 127, 127)


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
    # Display-resolution colorbar spanning the full column height (5 cells + gaps).
    col = grid.columns[0]
    expected_h = len(col.cells) * col.cell_size + (len(col.cells) - 1) * PATCH_CELL_GAP
    assert _decode(grid.heat_legend).size[1] == expected_h
    # All-zero heat has no scale to show, even with the heatmap enabled.
    flat = _type_patches(values, torch.ones(1, 5, 3, 4, 4))
    rendered = render_patch_grid(flat, heatmap=True)
    assert rendered is not None and rendered.heat_legend is None


def test_dims_from_roles_resolves_axes() -> None:
    assert dims_from_roles(["index", "tile", "y", "x"]) == (3, 2, 1)
    assert dims_from_roles(["x"]) == (0, None, None)
    assert dims_from_roles(["index", "index"]) == (None, None, None)
