"""Render snapshot tensors to PNG bytes for the UI.

The library captures full per-batch tensors; the renderer takes a per-sample
slice and produces one horizontal strip per layer for the right pane of the
UI. Conv-style activations become a row of square channel tiles; 1D
activations become a single short heatmap row.

Every strip — activations, gradients, and weights alike — uses the same
diverging (blue-white-red) colormap on a symmetric `[-x, +x]` scale where
`x` is the per-strip absolute maximum; a vertical colorbar with `+x` / `0` /
`-x` labels is rendered into the left edge of each strip. PNGs are encoded
with `compress_level=1` to favour speed over size — bytes travel a local
WebSocket, so wire size is irrelevant.
"""

from __future__ import annotations

import io
from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from torch import Tensor
from torch.nn import functional as F

TILE_SIZE: int = 128
TILE_GAP: int = 2
LINEAR_TILE_HEIGHT: int = 32
LINEAR_MAX_BINS: int = 256
LINEAR_BIN_WIDTH: int = 16
INPUT_IMAGE_SIZE: int = 256
PNG_COMPRESS_LEVEL: int = 1
LEGEND_BAR_WIDTH: int = 12
LEGEND_LABEL_WIDTH: int = 52
LEGEND_GAP: int = 4
LEGEND_WIDTH: int = LEGEND_LABEL_WIDTH + LEGEND_GAP + LEGEND_BAR_WIDTH + LEGEND_GAP
LEGEND_MID_LABEL_MIN_HEIGHT: int = 64


def render_strip(tensor: Tensor | None, sample_idx: int) -> bytes | None:
    """Render a per-channel horizontal strip as PNG bytes.

    Returns `None` if the tensor is `None`, `sample_idx` is out of range, or
    the per-sample shape is unsupported (anything other than `[C, H, W]` or
    `[F]`).
    """
    if tensor is None or tensor.ndim == 0:
        return None
    if not 0 <= sample_idx < tensor.shape[0]:
        return None
    sample = tensor[sample_idx]
    if sample.ndim == 3:
        return _render_chw(sample)
    if sample.ndim == 1:
        return _render_1d(sample)
    return None


@dataclass(frozen=True)
class WeightDims:
    """How an N-D weight tensor maps onto the strip's visual axes.

    `x_dim` is the within-tile horizontal axis, `y_dim` the within-tile
    vertical axis, and `tile_dim` the axis laid out as separate side-by-side
    tiles. `fixed_dims` are every remaining axis — each pinned to a single
    index (chosen by number in the UI) so the result collapses to at most a
    3-D `[tile, y, x]` view.
    """

    x_dim: int
    y_dim: int | None
    tile_dim: int | None
    fixed_dims: tuple[int, ...]


def default_weight_dims(ndim: int) -> WeightDims:
    """Default axis assignment for a weight of rank `ndim`.

    4-D conv weights `[out, in, kH, kW]` become conv kernels (kH×kW tiles laid
    out across `in`, the leading `out` axis pinned by index); 2-D weights are a
    single `[out, in]` image; 1-D weights a single heatmap row. The last axis
    is always X, the second-to-last Y, the third-to-last the tile axis, and any
    further leading axes are fixed.
    """
    if ndim <= 0:
        raise ValueError(f"weight must have at least one dimension, got {ndim}")
    x_dim = ndim - 1
    y_dim = ndim - 2 if ndim >= 2 else None
    tile_dim = ndim - 3 if ndim >= 3 else None
    fixed_dims = tuple(range(ndim - 3)) if ndim >= 4 else ()
    return WeightDims(x_dim=x_dim, y_dim=y_dim, tile_dim=tile_dim, fixed_dims=fixed_dims)


def render_weight(
    tensor: Tensor | None,
    *,
    x_dim: int,
    y_dim: int | None,
    tile_dim: int | None,
    fixed: dict[int, int],
) -> bytes | None:
    """Render a weight tensor as a horizontal strip under a chosen axis layout.

    `x_dim` / `y_dim` / `tile_dim` pick the axes shown within and across tiles
    (`y_dim`/`tile_dim` may be `None` for lower-rank views). Every other axis
    is reduced to a single slice via `fixed` (axis -> index, clamped into
    range, defaulting to 0). Returns `None` for a `None`/scalar tensor or an
    invalid axis selection (out-of-range or duplicated axes).
    """
    if tensor is None or tensor.ndim == 0:
        return None
    ndim = tensor.ndim
    kept = [d for d in (tile_dim, y_dim, x_dim) if d is not None]
    if any(not 0 <= d < ndim for d in kept) or len(set(kept)) != len(kept):
        return None

    index: list[int | slice] = []
    for d in range(ndim):
        if d in kept:
            index.append(slice(None))
        else:
            i = fixed.get(d, 0)
            index.append(max(0, min(i, tensor.shape[d] - 1)))
    selected = tensor[tuple(index)]

    # After integer-indexing the fixed axes, the surviving axes keep their
    # original order; map each kept axis to its position so we can permute
    # into (tile, y, x) order.
    pos = {d: i for i, d in enumerate(sorted(kept))}
    if y_dim is None:
        return _render_1d(selected.reshape(-1))
    if tile_dim is None:
        yx = selected.permute(pos[y_dim], pos[x_dim]).contiguous()
        return _render_chw(yx.unsqueeze(0))
    chw = selected.permute(pos[tile_dim], pos[y_dim], pos[x_dim]).contiguous()
    return _render_chw(chw)


def _render_chw(tensor: Tensor) -> bytes:
    c, h, w = tensor.shape
    abs_max = float(tensor.detach().abs().max())
    mode = "nearest" if max(h, w) <= TILE_SIZE else "area"
    resized = F.interpolate(
        tensor.unsqueeze(0).float(),
        size=(TILE_SIZE, TILE_SIZE),
        mode=mode,
    )[0]
    rgb = _apply_colormap(resized.numpy(), abs_max=abs_max)
    strip = _concat_tiles_with_gaps(list(rgb), TILE_GAP)
    legend = _render_legend(TILE_SIZE, abs_max=abs_max)
    return _encode_png(np.concatenate([legend, strip], axis=1))


def _concat_tiles_with_gaps(tiles: list[np.ndarray], gap: int) -> np.ndarray:
    if gap <= 0 or len(tiles) <= 1:
        return np.concatenate(tiles, axis=1)
    h = tiles[0].shape[0]
    spacer = np.full((h, gap, 3), 255, dtype=np.uint8)
    pieces: list[np.ndarray] = []
    for i, tile in enumerate(tiles):
        if i > 0:
            pieces.append(spacer)
        pieces.append(tile)
    return np.concatenate(pieces, axis=1)


def render_image(
    tensor: Tensor | None,
    sample_idx: int,
    *,
    mean: tuple[float, ...] | None = None,
    std: tuple[float, ...] | None = None,
) -> bytes | None:
    """Render a per-sample input image as PNG bytes.

    Expects a `[B, C, H, W]` tensor with `C in (1, 3)`. Values are assumed
    to lie in `[0, 1]` unless both `mean` and `std` are provided, in which
    case the sample is denormalized as `x * std + mean` before being
    clamped and scaled to 8-bit. Returns `None` for unsupported shapes,
    out-of-range `sample_idx`, or a None tensor.
    """
    if tensor is None or tensor.ndim != 4:
        return None
    if not 0 <= sample_idx < tensor.shape[0]:
        return None
    sample = tensor[sample_idx]
    c, _, _ = sample.shape
    if c not in (1, 3):
        return None
    arr = sample.detach().float().cpu().numpy()
    if mean is not None and std is not None:
        if len(mean) != c or len(std) != c:
            return None
        m = np.asarray(mean, dtype=np.float32).reshape(c, 1, 1)
        s = np.asarray(std, dtype=np.float32).reshape(c, 1, 1)
        arr = arr * s + m
    arr = (arr.clip(0.0, 1.0) * 255).astype(np.uint8)
    hwc = np.transpose(arr, (1, 2, 0))
    if c == 1:
        pil = Image.fromarray(hwc[..., 0], mode="L")
    else:
        pil = Image.fromarray(hwc, mode="RGB")
    pil = pil.resize((INPUT_IMAGE_SIZE, INPUT_IMAGE_SIZE), Image.Resampling.NEAREST)
    return _pil_to_png(pil)


def _render_1d(tensor: Tensor) -> bytes:
    values = tensor.float()
    abs_max = float(values.abs().max())
    f = values.shape[0]
    if f > LINEAR_MAX_BINS:
        values = F.adaptive_avg_pool1d(
            values.reshape(1, 1, f), LINEAR_MAX_BINS
        ).reshape(-1)
        f = LINEAR_MAX_BINS
    rgb_row = _apply_colormap(values.numpy(), abs_max=abs_max)
    image = np.broadcast_to(rgb_row[None, :, :], (LINEAR_TILE_HEIGHT, f, 3)).copy()
    strip = np.asarray(
        Image.fromarray(image, mode="RGB").resize(
            (f * LINEAR_BIN_WIDTH, LINEAR_TILE_HEIGHT), Image.Resampling.NEAREST
        )
    )
    legend = _render_legend(LINEAR_TILE_HEIGHT, abs_max=abs_max)
    return _encode_png(np.concatenate([legend, strip], axis=1))


def _apply_colormap(values: np.ndarray, *, abs_max: float) -> np.ndarray:
    scale = max(abs_max, 1e-12)
    norm = (values / scale).clip(-1.0, 1.0)
    return _diverging_colormap(norm)


def _diverging_colormap(norm: np.ndarray) -> np.ndarray:
    rgb = np.full(norm.shape + (3,), 255, dtype=np.uint8)
    pos = norm > 0
    neg = norm < 0
    fade_pos = (255 * (1 - norm[pos])).astype(np.uint8)
    rgb[pos, 1] = fade_pos
    rgb[pos, 2] = fade_pos
    fade_neg = (255 * (1 + norm[neg])).astype(np.uint8)
    rgb[neg, 0] = fade_neg
    rgb[neg, 1] = fade_neg
    return rgb


def _render_legend(height: int, *, abs_max: float) -> np.ndarray:
    """Vertical colorbar with `+x` / `0` / `-x` labels.

    `+x` sits at the top of the bar, `-x` at the bottom; the middle `0`
    label is dropped on short strips where it would collide with the
    top/bottom labels.
    """
    values = np.linspace(abs_max, -abs_max, height, dtype=np.float32)
    bar_col = _apply_colormap(values, abs_max=abs_max)
    bar = np.broadcast_to(bar_col[:, None, :], (height, LEGEND_BAR_WIDTH, 3)).copy()

    labels_img = Image.new("RGB", (LEGEND_LABEL_WIDTH, height), (255, 255, 255))
    draw = ImageDraw.Draw(labels_img)
    font = ImageFont.load_default()
    x = LEGEND_LABEL_WIDTH - 2
    color = (0, 0, 0)
    draw.text((x, 0), f"+{abs_max:.2g}", fill=color, font=font, anchor="ra")
    draw.text((x, height - 1), f"-{abs_max:.2g}", fill=color, font=font, anchor="rd")
    if height >= LEGEND_MID_LABEL_MIN_HEIGHT:
        draw.text((x, height // 2), "0", fill=color, font=font, anchor="rm")
    labels = np.asarray(labels_img)

    gap = np.full((height, LEGEND_GAP, 3), 255, dtype=np.uint8)
    return np.concatenate([labels, gap, bar, gap], axis=1)


def _encode_png(rgb: np.ndarray) -> bytes:
    return _pil_to_png(Image.fromarray(rgb, mode="RGB"))


def _pil_to_png(pil: Image.Image) -> bytes:
    buf = io.BytesIO()
    pil.save(buf, format="PNG", compress_level=PNG_COMPRESS_LEVEL)
    return buf.getvalue()
