"""Tests for tensor → PNG rendering."""

from __future__ import annotations

import io

import pytest
import torch
from PIL import Image

from playgrad.ui.render import (
    INPUT_IMAGE_SIZE,
    LEGEND_WIDTH,
    LINEAR_BIN_WIDTH,
    LINEAR_MAX_BINS,
    LINEAR_TILE_HEIGHT,
    TILE_GAP,
    TILE_SIZE,
    default_weight_dims,
    render_image,
    render_strip,
    render_weight,
)


def _chw_strip_width(num_tiles: int) -> int:
    tiles = num_tiles * TILE_SIZE + max(0, num_tiles - 1) * TILE_GAP
    return LEGEND_WIDTH + tiles


def _1d_strip_width(num_bins: int) -> int:
    return LEGEND_WIDTH + num_bins * LINEAR_BIN_WIDTH


def _decode(png: bytes) -> Image.Image:
    return Image.open(io.BytesIO(png)).convert("RGB")


def test_chw_strip_dimensions() -> None:
    tensor = torch.randn(4, 8, 32, 32)
    png = render_strip(tensor, sample_idx=2)
    assert png is not None
    img = _decode(png)
    assert img.size == (_chw_strip_width(8), TILE_SIZE)


def test_1d_strip_dimensions() -> None:
    tensor = torch.randn(4, 10)
    png = render_strip(tensor, sample_idx=0)
    assert png is not None
    img = _decode(png)
    assert img.size == (_1d_strip_width(10), LINEAR_TILE_HEIGHT)


def test_1d_strip_caps_at_max_bins() -> None:
    tensor = torch.randn(4, LINEAR_MAX_BINS * 4)
    png = render_strip(tensor, sample_idx=0)
    assert png is not None
    img = _decode(png)
    assert img.size == (_1d_strip_width(LINEAR_MAX_BINS), LINEAR_TILE_HEIGHT)


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


def test_zero_variance_tensor_renders() -> None:
    # All-zero input must not crash the diverging colormap (abs_max=0 edge).
    tensor = torch.zeros(2, 4, 8, 8)
    png = render_strip(tensor, sample_idx=0)
    assert png is not None
    img = _decode(png)
    assert img.size == (_chw_strip_width(4), TILE_SIZE)


def test_strip_uses_diverging_colormap() -> None:
    # Tile 0 is all +max → pure red at its centre; tile 1 all -max → pure blue.
    sample = torch.stack([torch.ones(8, 8), -torch.ones(8, 8)])
    png = render_strip(sample.unsqueeze(0), sample_idx=0)
    assert png is not None
    img = _decode(png)
    y = TILE_SIZE // 2
    pos_x = LEGEND_WIDTH + TILE_SIZE // 2
    neg_x = LEGEND_WIDTH + TILE_SIZE + TILE_GAP + TILE_SIZE // 2
    assert _rgb_at(img, pos_x, y) == (255, 0, 0)
    assert _rgb_at(img, neg_x, y) == (0, 0, 255)


@pytest.mark.parametrize("channels", [1, 3])
def test_input_image_dimensions(channels: int) -> None:
    tensor = torch.rand(2, channels, 16, 16)
    png = render_image(tensor, sample_idx=0)
    assert png is not None
    img = _decode(png)
    assert img.size == (INPUT_IMAGE_SIZE, INPUT_IMAGE_SIZE)


def _rgb_at(img: Image.Image, x: int, y: int) -> tuple[int, int, int]:
    pixel = img.getpixel((x, y))
    assert isinstance(pixel, tuple) and len(pixel) == 3
    r, g, b = pixel
    return int(r), int(g), int(b)


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
    png = render_weight(
        w,
        x_dim=d.x_dim,
        y_dim=d.y_dim,
        tile_dim=d.tile_dim,
        fixed={f: 0 for f in d.fixed_dims},
    )
    assert png is not None
    assert _decode(png).size == (_chw_strip_width(3), TILE_SIZE)


def test_render_weight_2d_is_single_image_tile() -> None:
    w = torch.randn(10, 4)
    d = default_weight_dims(2)
    png = render_weight(w, x_dim=d.x_dim, y_dim=d.y_dim, tile_dim=d.tile_dim, fixed={})
    assert png is not None
    assert _decode(png).size == (_chw_strip_width(1), TILE_SIZE)


def test_render_weight_1d_is_single_row() -> None:
    w = torch.randn(16)
    png = render_weight(w, x_dim=0, y_dim=None, tile_dim=None, fixed={})
    assert png is not None
    assert _decode(png).size == (_1d_strip_width(16), LINEAR_TILE_HEIGHT)


def test_render_weight_custom_tile_axis_changes_tile_count() -> None:
    # Tile across `out`=8 instead of the default `in`, pinning `in` by index.
    w = torch.randn(8, 3, 3, 3)
    png = render_weight(w, x_dim=3, y_dim=2, tile_dim=0, fixed={1: 1})
    assert png is not None
    assert _decode(png).size == (_chw_strip_width(8), TILE_SIZE)


def test_render_weight_clamps_out_of_range_fixed_index() -> None:
    w = torch.randn(8, 3, 3, 3)
    d = default_weight_dims(4)
    png = render_weight(
        w, x_dim=d.x_dim, y_dim=d.y_dim, tile_dim=d.tile_dim, fixed={0: 999}
    )
    assert png is not None  # index clamped into range rather than crashing


def test_render_weight_returns_none_for_none_tensor() -> None:
    assert render_weight(None, x_dim=0, y_dim=None, tile_dim=None, fixed={}) is None


def test_render_weight_returns_none_for_duplicate_axes() -> None:
    w = torch.randn(8, 3, 3, 3)
    assert render_weight(w, x_dim=2, y_dim=2, tile_dim=0, fixed={1: 0}) is None


def test_render_weight_returns_none_for_out_of_range_axis() -> None:
    w = torch.randn(4, 4)
    assert render_weight(w, x_dim=5, y_dim=0, tile_dim=None, fixed={}) is None
