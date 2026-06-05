"""Render snapshot tensors to image bytes for the UI.

The library captures full per-batch tensors; the renderer takes a per-sample
slice and produces one horizontal strip per layer for the right pane of the
UI. Conv-style activations become a row of square channel tiles; 1D
activations become a single short heatmap row.

A strip is returned as a `StripRender`: one data image holding every tile at
the tensor's *native* resolution (downsampled server-side only when larger
than the display tile), upscaled by the browser via CSS sizing plus
`image-rendering: pixelated` — equivalent to nearest-neighbour, but without
inflating an 8×8 feature map to 128×128 pixels before colormapping and
encoding. Tiles are separated by white spacers `max(1, tile_width //
TILE_GAP_DIVISOR)` native pixels wide, so the gap scales with the tiles.
The legend (vertical colorbar with `+x` / `0` / `-x` labels) is the
exception to native-resolution encoding: it is rendered at display
resolution into its own image so its text stays crisp.

Every strip — activations, gradients, and weights alike — uses the same
diverging (blue-white-red) colormap on a symmetric `[-x, +x]` scale where
`x` is the per-strip absolute maximum.

Every image (strip data, legends, the input pane) is encoded in
`STRIP_FORMAT`. The default `"BMP"` is essentially a memcpy — 30–60× faster
to encode than PNG — at ~2× the payload, the right trade for a localhost
WebSocket. Flip to `"PNG"` (compressed at `PNG_COMPRESS_LEVEL`) when bytes
matter more than encode time, e.g. viewing the UI through an SSH port
forward.
"""

from __future__ import annotations

import io
from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from torch import Tensor
from torch.nn import functional as F

from playgrad.patches import TypePatches

TILE_SIZE: int = 128
TILE_GAP_DIVISOR: int = 48
LINEAR_TILE_HEIGHT: int = 32
LINEAR_MAX_BINS: int = 256
LINEAR_BIN_WIDTH: int = 16
INPUT_IMAGE_SIZE: int = 256
# Encoding for every rendered image: "BMP" (fastest encode, ~2x payload) or
# "PNG" (compressed; for byte-constrained links like an SSH port forward).
STRIP_FORMAT: str = "BMP"
PNG_COMPRESS_LEVEL: int = 1
_MIME_TYPES: dict[str, str] = {"BMP": "image/bmp", "PNG": "image/png"}
LEGEND_BAR_WIDTH: int = 12
LEGEND_LABEL_WIDTH: int = 52
LEGEND_GAP: int = 4
LEGEND_WIDTH: int = LEGEND_LABEL_WIDTH + LEGEND_GAP + LEGEND_BAR_WIDTH + LEGEND_GAP
LEGEND_MID_LABEL_MIN_HEIGHT: int = 64
# Extreme-patch grids: CSS pixel side of one cell, the strongest heatmap
# overlay opacity, the fill for never-filled slots, and the grids' fixed
# encoding (see `PatchGridRender` for why it isn't `STRIP_FORMAT`).
PATCH_CELL_SIZE: int = 44
HEAT_MAX_ALPHA: float = 0.55
_EMPTY_CELL_GRAY: int = 235
PATCH_GRID_FORMAT: str = "PNG"


def image_mime() -> str:
    """MIME type matching `STRIP_FORMAT`, for data-URI `<img>` sources."""
    return _MIME_TYPES[STRIP_FORMAT]


@dataclass(frozen=True)
class StripRender:
    """One rendered strip: a native-resolution data image plus a crisp legend.

    `data_image` holds every tile in a single image (encoded per
    `STRIP_FORMAT`) at native (or server-downsampled) resolution, with white
    separators `_tile_gap(w)` native pixels wide between tiles; the UI
    displays it at `width × height` CSS pixels with `image-rendering:
    pixelated`, so the separators scale together with the tiles.
    `legend_image` is already at display resolution and is shown 1:1.
    """

    legend_image: bytes
    data_image: bytes
    width: int
    height: int


def render_strip(tensor: Tensor | None, sample_idx: int) -> StripRender | None:
    """Render a per-channel horizontal strip.

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
) -> StripRender | None:
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


def _tile_gap(tile_width: int) -> int:
    """White separator width between tiles, in native pixels.

    Proportional to the tile so the gap stays ~2% of a tile's width after
    the browser upscales the strip; floors at one pixel."""
    return max(1, tile_width // TILE_GAP_DIVISOR)


def _render_chw(tensor: Tensor) -> StripRender:
    abs_max = float(tensor.detach().abs().max())
    data = tensor.detach().float()
    if max(data.shape[1], data.shape[2]) > TILE_SIZE:
        # Downsampling needs real averaging server-side; *up*scaling small
        # maps is left to the browser's nearest-neighbour (CSS `pixelated`).
        data = F.interpolate(
            data.unsqueeze(0), size=(TILE_SIZE, TILE_SIZE), mode="area"
        )[0]
    _, h, w = data.shape
    rgb = _apply_colormap(data.numpy(), abs_max=abs_max)
    strip = _concat_tiles_with_gaps(list(rgb), _tile_gap(w))
    # Each tile spans TILE_SIZE × TILE_SIZE CSS px, so the whole strip scales
    # by TILE_SIZE/w horizontally and TILE_SIZE/h vertically.
    return StripRender(
        legend_image=_encode_image(_render_legend(TILE_SIZE, abs_max=abs_max)),
        data_image=_encode_image(strip),
        width=round(strip.shape[1] * TILE_SIZE / w),
        height=TILE_SIZE,
    )


def _concat_tiles_with_gaps(tiles: list[np.ndarray], gap: int) -> np.ndarray:
    if len(tiles) <= 1:
        return tiles[0]
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
    """Render a per-sample input image as `STRIP_FORMAT` bytes.

    Expects a `[B, C, H, W]` tensor with `C in (1, 3)`. Values are assumed
    to lie in `[0, 1]` unless both `mean` and `std` are provided, in which
    case the sample is denormalized as `x * std + mean` before being
    clamped and scaled to 8-bit. The image keeps the sample's native
    `H × W`; the UI scales it to `INPUT_IMAGE_SIZE` with CSS
    nearest-neighbour. Returns `None` for unsupported shapes, out-of-range
    `sample_idx`, or a None tensor.
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
    return _pil_to_bytes(pil)


@dataclass(frozen=True)
class PatchGridRender:
    """One extreme-patch grid: channels across, top-N samples down.

    `image` is encoded at native patch resolution; the UI shows it at
    `width × height` CSS pixels (`PATCH_CELL_SIZE` per cell) with
    `image-rendering: pixelated`, like the activation strips. Unlike the
    strips, grids are always PNG (`PATCH_GRID_FORMAT`, mime in `mime`):
    a wide layer's BMP grids reach multiple MB per refresh message, enough
    to pause the websocket transport — concurrent drains then trip a known
    `websockets`-legacy keepalive assertion and kill the connection. PNG
    keeps grid messages ~10× under that regime for a few ms of (worker
    thread) encode time.
    """

    image: bytes
    width: int
    height: int
    mime: str


def render_patch_grid(
    tp: TypePatches,
    *,
    mean: tuple[float, ...] | None = None,
    std: tuple[float, ...] | None = None,
    heatmap: bool = False,
) -> PatchGridRender | None:
    """Render one patch type's double grid as `STRIP_FORMAT` bytes.

    Columns are activation channels, rows the per-channel top samples
    (best first). Patches are denormalized like `render_image`. With
    `heatmap`, the stored activation map is blended over each patch —
    transparent at 0, opacifying toward red (positive) / blue (negative)
    at the grid-wide absolute maximum. Slots never filled render as flat
    gray. Returns `None` when no slot is filled yet.
    """
    values = tp.values.numpy()
    valid = np.isfinite(values)
    if not valid.any():
        return None
    cells = _denormalized_cells(tp, mean=mean, std=std)
    c, n, ph, pw, _ = cells.shape
    if heatmap:
        cells = _blend_heat(cells, tp)
    cells[~valid] = _EMPTY_CELL_GRAY

    gap = 1
    grid = np.full(
        (n * ph + (n - 1) * gap, c * pw + (c - 1) * gap, 3), 255, dtype=np.uint8
    )
    for col in range(c):
        for row in range(n):
            y, x = row * (ph + gap), col * (pw + gap)
            grid[y : y + ph, x : x + pw] = cells[col, row]
    return PatchGridRender(
        image=_encode_image(grid, fmt=PATCH_GRID_FORMAT),
        width=round(grid.shape[1] * PATCH_CELL_SIZE / pw),
        height=round(grid.shape[0] * PATCH_CELL_SIZE / ph),
        mime=_MIME_TYPES[PATCH_GRID_FORMAT],
    )


def _denormalized_cells(
    tp: TypePatches,
    *,
    mean: tuple[float, ...] | None,
    std: tuple[float, ...] | None,
) -> np.ndarray:
    """Patches as `(C, N, ph, pw, 3)` uint8, denormalized like `render_image`."""
    arr = tp.patches.numpy()  # (C, N, Cin, ph, pw)
    cin = arr.shape[2]
    if mean is not None and std is not None and len(mean) == cin == len(std):
        m = np.asarray(mean, dtype=np.float32).reshape(1, 1, cin, 1, 1)
        s = np.asarray(std, dtype=np.float32).reshape(1, 1, cin, 1, 1)
        arr = arr * s + m
    arr = (arr.clip(0.0, 1.0) * 255).astype(np.uint8)
    rgb = np.moveaxis(arr, 2, -1)  # (C, N, ph, pw, Cin)
    if cin == 1:
        rgb = np.repeat(rgb, 3, axis=-1)
    return np.ascontiguousarray(rgb)


def _blend_heat(cells: np.ndarray, tp: TypePatches) -> np.ndarray:
    """Overlay each cell's activation map: 0 transparent, ±max red/blue."""
    heat = tp.heat.numpy()  # (C, N, Hh, Wh)
    valid = np.isfinite(tp.values.numpy())
    finite = np.nan_to_num(heat[valid], posinf=0.0, neginf=0.0)
    vmax = float(np.abs(finite).max()) if finite.size else 0.0
    if vmax <= 0.0:
        return cells
    c, n, ph, pw, _ = cells.shape
    out = cells.astype(np.float32)
    top = tp.top.numpy()
    left = tp.left.numpy()
    hin, win = tp.input_hw
    for col in range(c):
        for row in range(n):
            if not valid[col, row]:
                continue
            cell_heat = _cell_heat(
                heat[col, row],
                crop=tp.crop,
                window=(top[col, row], left[col, row], ph, pw),
                input_hw=(hin, win),
            )
            norm = (cell_heat / vmax).clip(-1.0, 1.0)
            alpha = (np.abs(norm) * HEAT_MAX_ALPHA)[..., None]
            color = np.zeros((ph, pw, 3), dtype=np.float32)
            color[..., 0] = np.where(norm > 0, 255.0, 0.0)
            color[..., 2] = np.where(norm < 0, 255.0, 0.0)
            out[col, row] = out[col, row] * (1 - alpha) + color * alpha
    return out.round().astype(np.uint8)


def _cell_heat(
    heat: np.ndarray,
    *,
    crop: bool,
    window: tuple[int, int, int, int],
    input_hw: tuple[int, int],
) -> np.ndarray:
    """The activation-map region under one cell, nearest-resized to it.

    For crops, the input-space window `[top, top+ph) × [left, left+pw)` is
    ratio-mapped back onto the activation map before resizing; whole-image
    patches use the full map.
    """
    top, left, ph, pw = window
    hh, wh = heat.shape
    if crop:
        hin, win = input_hw
        y0 = max(0, int(np.floor(top * hh / hin)))
        y1 = min(hh, max(y0 + 1, int(np.ceil((top + ph) * hh / hin))))
        x0 = max(0, int(np.floor(left * wh / win)))
        x1 = min(wh, max(x0 + 1, int(np.ceil((left + pw) * wh / win))))
        heat = heat[y0:y1, x0:x1]
    ys = (np.arange(ph) * heat.shape[0] / ph).astype(np.int64)
    xs = (np.arange(pw) * heat.shape[1] / pw).astype(np.int64)
    return heat[ys[:, None], xs[None, :]]


def _render_1d(tensor: Tensor) -> StripRender:
    values = tensor.detach().float()
    abs_max = float(values.abs().max())
    f = values.shape[0]
    if f > LINEAR_MAX_BINS:
        values = F.adaptive_avg_pool1d(
            values.reshape(1, 1, f), LINEAR_MAX_BINS
        ).reshape(-1)
        f = LINEAR_MAX_BINS
    rgb_row = _apply_colormap(values.numpy(), abs_max=abs_max)
    # A 1-px-tall row; the browser stretches it to the display height.
    return StripRender(
        legend_image=_encode_image(
            _render_legend(LINEAR_TILE_HEIGHT, abs_max=abs_max)
        ),
        data_image=_encode_image(rgb_row[None, :, :]),
        width=f * LINEAR_BIN_WIDTH,
        height=LINEAR_TILE_HEIGHT,
    )


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


def _encode_image(rgb: np.ndarray, *, fmt: str | None = None) -> bytes:
    return _pil_to_bytes(
        Image.fromarray(np.ascontiguousarray(rgb), mode="RGB"), fmt=fmt
    )


def _pil_to_bytes(pil: Image.Image, *, fmt: str | None = None) -> bytes:
    fmt = fmt or STRIP_FORMAT
    buf = io.BytesIO()
    if fmt == "PNG":
        pil.save(buf, format="PNG", compress_level=PNG_COMPRESS_LEVEL)
    else:
        pil.save(buf, format=fmt)
    return buf.getvalue()
