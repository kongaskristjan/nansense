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

NaN / ±Inf cells are the one exception to the fast RGB path. A single
non-finite value used to NaN/Inf the colormap scale and whitewash the whole
strip (indistinguishable from all-zero — the very divergence the tool exists
to surface). Now the symmetric scale is computed over finite values only,
and a strip that actually contains NaN/±Inf is encoded as RGBA PNG with
those cells fully transparent (alpha 0); the UI and recordings paint a fixed
display-resolution gray checkerboard behind the data image so the bad cells
read as "no value here", not a misleading color. All-finite strips keep the
byte-for-byte RGB `STRIP_FORMAT` path (`StripRender.data_mime` says which).
"""

from __future__ import annotations

import io
import math
from dataclasses import dataclass

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from torch import Tensor
from torch.nn import functional as F

from nansense.patches import TypePatches
from nansense.probe import ProbeResult

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
PATCH_CELL_SIZE: int = 132
HEAT_MAX_ALPHA: float = 0.825
_EMPTY_CELL_GRAY: int = 235
PATCH_GRID_FORMAT: str = "PNG"


def image_mime() -> str:
    """MIME type matching `STRIP_FORMAT`, for data-URI `<img>` sources."""
    return _MIME_TYPES[STRIP_FORMAT]


@dataclass(frozen=True)
class StripRender:
    """One rendered strip: a native-resolution data image plus a crisp legend.

    `data_image` holds every tile in a single image at native (or
    server-downsampled) resolution, with white separators `_tile_gap(w)`
    native pixels wide between tiles; the UI displays it at `width × height`
    CSS pixels with `image-rendering: pixelated`, so the separators scale
    together with the tiles. `legend_image` is already at display resolution
    and is shown 1:1.

    `data_mime` is the data image's MIME type: the common all-finite strip
    is RGB encoded per `STRIP_FORMAT` (the fast `image_mime()` path), but a
    strip containing NaN/±Inf cells is RGBA PNG instead — those cells are
    fully transparent so the UI / recording can show a checkerboard behind
    them. Consumers must use `data_mime` (not the global `image_mime()`) for
    the data image's data-URI; the `legend_image` is always `STRIP_FORMAT`.
    """

    legend_image: bytes
    data_image: bytes
    width: int
    height: int
    data_mime: str = _MIME_TYPES[STRIP_FORMAT]


def render_strip(
    tensor: Tensor | None,
    sample_idx: int,
    *,
    input_hw: tuple[int, int] | None = None,
) -> StripRender | None:
    """Render a per-channel horizontal strip.

    Per-sample `[C, H, W]` renders as channel tiles and `[F]` as a heatmap
    row. A 2D per-sample shape (e.g. flattened transformer tokens
    `[tokens, dim]`) renders as channel tiles too when one axis matches a
    token grid of the `input_hw` image (see `_render_tokens_2d`), and as a
    single 2D heatmap tile otherwise. Returns `None` if the tensor is
    `None`, `sample_idx` is out of range, or the per-sample shape is
    unsupported (4D and beyond).
    """
    if tensor is None or tensor.ndim == 0:
        return None
    if not 0 <= sample_idx < tensor.shape[0]:
        return None
    sample = tensor[sample_idx]
    if sample.ndim == 3:
        return _render_chw(sample)
    if sample.ndim == 2:
        return _render_tokens_2d(sample, input_hw)
    if sample.ndim == 1:
        return _render_1d(sample)
    return None


def tensor_hw(tensor: Tensor | None) -> tuple[int, int] | None:
    """Spatial size of a `[B, C, H, W]` input, or `None` when not image-like.

    Threaded into `render_strip` as `input_hw` so 2D (token-shaped)
    activations can be unflattened back onto the input's patch grid.
    """
    if tensor is None or tensor.ndim != 4:
        return None
    return int(tensor.shape[-2]), int(tensor.shape[-1])


def probe_act_tensor(probe: ProbeResult, name: str, *, compare: bool) -> Tensor | None:
    """The probe-sourced activation tensor one layer's strip shows.

    With `compare` the strip shows `perturbed − original` (zeros when no
    perturbed forward ran); otherwise the perturbed activations win when
    they exist. Shared by the main page and frame recording, so both render
    the same tensor for the same probe.
    """
    base = probe.activations.get(name)
    perturbed_acts = probe.perturbed_activations
    if compare:
        if base is None:
            return None
        if perturbed_acts is None:
            return torch.zeros_like(base)
        pert = perturbed_acts.get(name)
        if pert is None or pert.shape != base.shape:
            return None
        return pert - base
    if perturbed_acts is not None:
        return perturbed_acts.get(name)
    return base


# Leading non-spatial tokens a grid fit may skip: none, a class token, class
# + distillation tokens (DeiT), or four register tokens (ViT-with-registers).
_SPECIAL_TOKEN_COUNTS: tuple[int, ...] = (0, 1, 2, 4)


def _token_grid(n_tokens: int, input_hw: tuple[int, int]) -> tuple[int, int, int] | None:
    """Match a token count to a patch grid of the `input_hw` image.

    Returns `(extra, h, w)` where `h * w == n_tokens - extra` is the grid
    produced by an integer patch stride `s` over the input (`h = H/s`,
    `w = W/s` — preserving the input's aspect ratio) and `extra` is the
    smallest number of leading special tokens (`_SPECIAL_TOKEN_COUNTS`)
    that makes a stride fit. Returns `None` when no stride fits.
    """
    height, width = input_hw
    for extra in _SPECIAL_TOKEN_COUNTS:
        t = n_tokens - extra
        if t <= 0 or (height * width) % t:
            continue
        stride = math.isqrt(height * width // t)
        if stride == 0 or stride * stride != height * width // t:
            continue
        if height % stride or width % stride:
            continue
        return extra, height // stride, width // stride
    return None


def _render_tokens_2d(sample: Tensor, input_hw: tuple[int, int] | None) -> StripRender:
    """Render a 2D per-sample tensor, recovering a token grid when possible.

    When one axis matches a token grid of the input (`_token_grid`), the
    tokens unflatten to one `h x w` tile per embedding dim — the same view
    a conv layer's channels get. Special tokens detected by the fit (CLS /
    distillation / registers) are assumed to lead the sequence and are
    dropped from the strip. The token axis is assumed row-major over the
    grid (standard ViT raster flatten); tokens-first (`[tokens, dim]`) is
    preferred when both axes fit, matching the `batch_first` slicing
    convention. Without `input_hw` or a fitting axis, the sample renders
    as a single 2D heatmap tile.
    """
    if input_hw is not None:
        for token_axis in (0, 1):
            fit = _token_grid(int(sample.shape[token_axis]), input_hw)
            if fit is None:
                continue
            extra, h, w = fit
            tokens = sample if token_axis == 0 else sample.T
            return _render_chw(tokens[extra:].T.reshape(-1, h, w))
    return _render_chw(sample.unsqueeze(0))


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


def dims_from_roles(roles: list[str]) -> tuple[int | None, int | None, int | None]:
    """Resolve a per-dimension role list to (x_dim, y_dim, tile_dim) axes."""
    x = y = tile = None
    for d, role in enumerate(roles):
        if role == "x":
            x = d
        elif role == "y":
            y = d
        elif role == "tile":
            tile = d
    return x, y, tile


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
    data = tensor.detach().float()
    abs_max = _finite_abs_max(data)
    if max(data.shape[1], data.shape[2]) > TILE_SIZE:
        # Downsampling needs real averaging server-side; *up*scaling small
        # maps is left to the browser's nearest-neighbour (CSS `pixelated`).
        # `area` would spread a NaN/Inf cell across its neighbours; keep the
        # non-finite mask sharp by interpolating finite values and the mask
        # separately (the colormap re-derives the mask after downsampling).
        data = _interpolate_preserving_nonfinite(data)
    _, h, w = data.shape
    rgb, mime = _apply_colormap(data.numpy(), abs_max=abs_max)
    strip = _concat_tiles_with_gaps(list(rgb), _tile_gap(w))
    # Each tile spans TILE_SIZE × TILE_SIZE CSS px, so the whole strip scales
    # by TILE_SIZE/w horizontally and TILE_SIZE/h vertically.
    return StripRender(
        legend_image=_encode_image(_render_legend(TILE_SIZE, abs_max=abs_max)),
        data_image=_encode_strip_data(strip, mime),
        width=round(strip.shape[1] * TILE_SIZE / w),
        height=TILE_SIZE,
        data_mime=mime,
    )


def _interpolate_preserving_nonfinite(data: Tensor) -> Tensor:
    """Area-downsample to `TILE_SIZE²`, keeping NaN/±Inf cells non-finite.

    `F.interpolate(mode="area")` smears a single non-finite value across a
    whole tile. To keep the bad cells localized, finite values are averaged
    with their sentinel zeroed out, while a separate nearest-neighbour pass
    on the non-finite mask decides which output cells stay non-finite (they
    are then stamped back as NaN so the colormap renders them transparent).
    """
    finite = torch.isfinite(data)
    if finite.all():
        return F.interpolate(
            data.unsqueeze(0), size=(TILE_SIZE, TILE_SIZE), mode="area"
        )[0]
    cleaned = torch.where(finite, data, torch.zeros_like(data))
    down = F.interpolate(
        cleaned.unsqueeze(0), size=(TILE_SIZE, TILE_SIZE), mode="area"
    )[0]
    mask = F.interpolate(
        (~finite).float().unsqueeze(0), size=(TILE_SIZE, TILE_SIZE), mode="nearest"
    )[0]
    return torch.where(mask > 0, torch.full_like(down, float("nan")), down)


def _concat_tiles_with_gaps(tiles: list[np.ndarray], gap: int) -> np.ndarray:
    if len(tiles) <= 1:
        return tiles[0]
    h, channels = tiles[0].shape[0], tiles[0].shape[2]
    # Separators are opaque white in RGB strips and opaque-white in RGBA
    # ones too (alpha 255) — only the data's non-finite cells go transparent.
    spacer = np.full((h, gap, channels), 255, dtype=np.uint8)
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
    arr = _denormalize_uint8(
        sample.detach().float().cpu().numpy(), mean, std, channel_axis=0
    )
    if arr is None:
        return None
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
    # Display-resolution colorbar for the heatmap overlay (`±vmax` labels),
    # encoded in `STRIP_FORMAT`; `None` when the heatmap is off or flat.
    heat_legend: bytes | None = None


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
    at the grid-wide absolute maximum — and `heat_legend` carries a
    crisp display-resolution colorbar for that `±vmax` scale. Slots
    never filled render as flat gray. Returns `None` when no slot is
    filled yet.
    """
    values = tp.values.numpy()
    valid = np.isfinite(values)
    if not valid.any():
        return None
    cells = _denormalized_cells(tp, mean=mean, std=std)
    if cells is None:
        return None
    c, n, ph, pw, _ = cells.shape
    vmax = _heat_vmax(tp) if heatmap else 0.0
    if vmax > 0.0:
        cells = _blend_heat(cells, tp, vmax=vmax)
    cells[~valid] = _EMPTY_CELL_GRAY

    gap = 1
    grid = np.full(
        (n * ph + (n - 1) * gap, c * pw + (c - 1) * gap, 3), 255, dtype=np.uint8
    )
    for col in range(c):
        for row in range(n):
            y, x = row * (ph + gap), col * (pw + gap)
            grid[y : y + ph, x : x + pw] = cells[col, row]
    height = round(grid.shape[0] * PATCH_CELL_SIZE / ph)
    return PatchGridRender(
        image=_encode_image(grid, fmt=PATCH_GRID_FORMAT),
        width=round(grid.shape[1] * PATCH_CELL_SIZE / pw),
        height=height,
        mime=_MIME_TYPES[PATCH_GRID_FORMAT],
        heat_legend=(
            _encode_image(_render_legend(height, abs_max=vmax))
            if vmax > 0.0
            else None
        ),
    )


def _denormalize_uint8(
    arr: np.ndarray,
    mean: tuple[float, ...] | None,
    std: tuple[float, ...] | None,
    *,
    channel_axis: int,
) -> np.ndarray | None:
    """`x * std + mean` (when stats are given), clipped and scaled to uint8.

    Returns None when provided stats don't match the channel count —
    callers hide the render rather than show wrongly-scaled values.
    """
    if mean is not None and std is not None:
        c = arr.shape[channel_axis]
        if len(mean) != c or len(std) != c:
            return None
        shape = [1] * arr.ndim
        shape[channel_axis] = c
        m = np.asarray(mean, dtype=np.float32).reshape(shape)
        s = np.asarray(std, dtype=np.float32).reshape(shape)
        arr = arr * s + m
    return (arr.clip(0.0, 1.0) * 255).astype(np.uint8)


def _denormalized_cells(
    tp: TypePatches,
    *,
    mean: tuple[float, ...] | None,
    std: tuple[float, ...] | None,
) -> np.ndarray | None:
    """Patches as `(C, N, ph, pw, 3)` uint8, denormalized like `render_image`."""
    arr = _denormalize_uint8(tp.patches.numpy(), mean, std, channel_axis=2)
    if arr is None:
        return None
    cin = arr.shape[2]  # (C, N, Cin, ph, pw)
    rgb = np.moveaxis(arr, 2, -1)  # (C, N, ph, pw, Cin)
    if cin == 1:
        rgb = np.repeat(rgb, 3, axis=-1)
    return np.ascontiguousarray(rgb)


def _heat_vmax(tp: TypePatches) -> float:
    """Grid-wide absolute heat maximum — the overlay's (and legend's) scale."""
    heat = tp.heat.numpy()  # (C, N, Hh, Wh)
    valid = np.isfinite(tp.values.numpy())
    finite = np.nan_to_num(heat[valid], posinf=0.0, neginf=0.0)
    return float(np.abs(finite).max()) if finite.size else 0.0


def _blend_heat(cells: np.ndarray, tp: TypePatches, *, vmax: float) -> np.ndarray:
    """Overlay each cell's activation map: 0 transparent, ±vmax red/blue."""
    heat = tp.heat.numpy()  # (C, N, Hh, Wh)
    valid = np.isfinite(tp.values.numpy())
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
    abs_max = _finite_abs_max(values)
    f = values.shape[0]
    if f > LINEAR_MAX_BINS:
        # `nan_to_num` first so a single NaN can't poison a whole pooled bin;
        # the bin's non-finite cells are re-detected by the colormap below.
        finite = torch.isfinite(values)
        pooled = F.adaptive_avg_pool1d(
            torch.where(finite, values, torch.zeros_like(values)).reshape(1, 1, f),
            LINEAR_MAX_BINS,
        ).reshape(-1)
        if not finite.all():
            bad = F.adaptive_max_pool1d(
                (~finite).float().reshape(1, 1, f), LINEAR_MAX_BINS
            ).reshape(-1)
            pooled = torch.where(bad > 0, torch.full_like(pooled, float("nan")), pooled)
        values = pooled
        f = LINEAR_MAX_BINS
    rgb_row, mime = _apply_colormap(values.numpy(), abs_max=abs_max)
    # A 1-px-tall row; the browser stretches it to the display height.
    return StripRender(
        legend_image=_encode_image(
            _render_legend(LINEAR_TILE_HEIGHT, abs_max=abs_max)
        ),
        data_image=_encode_strip_data(rgb_row[None, :, :], mime),
        width=f * LINEAR_BIN_WIDTH,
        height=LINEAR_TILE_HEIGHT,
        data_mime=mime,
    )


def _finite_abs_max(values: Tensor) -> float:
    """Largest `|x|` over finite entries; a tiny epsilon when none are finite.

    Computing the symmetric colormap scale over finite values only is what
    keeps one NaN/±Inf from turning the whole strip's scale into NaN/Inf
    (which left every cell white — indistinguishable from all-zero). The
    epsilon fallback keeps an all-non-finite (or finite-but-all-zero) strip
    rendering without a divide-by-zero.
    """
    finite = values[torch.isfinite(values)]
    if finite.numel() == 0:
        return 1e-12
    return float(finite.abs().max())


def _apply_colormap(values: np.ndarray, *, abs_max: float) -> tuple[np.ndarray, str]:
    """Diverging colormap plus the MIME the data image must use.

    The all-finite common case returns an RGB array and the `STRIP_FORMAT`
    mime — byte-for-byte the previous fast path. When any value is NaN/±Inf
    the result is an RGBA array (PNG mime): finite cells keep the opaque
    blue-white-red color and the non-finite cells get alpha 0, so the UI /
    recording reveal a checkerboard behind them instead of a misleading
    color or white.
    """
    finite = np.isfinite(values)
    scale = max(abs_max, 1e-12)
    norm = np.where(finite, values, 0.0) / scale
    rgb = _diverging_colormap(np.clip(norm, -1.0, 1.0))
    if finite.all():
        return rgb, image_mime()
    rgba = np.concatenate(
        [rgb, np.full(rgb.shape[:-1] + (1,), 255, dtype=np.uint8)], axis=-1
    )
    rgba[~finite, 3] = 0
    return rgba, _MIME_TYPES["PNG"]


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
    bar_col, _ = _apply_colormap(values, abs_max=abs_max)  # finite: always RGB
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


def _encode_strip_data(pixels: np.ndarray, mime: str) -> bytes:
    """Encode a strip's data image, picking the format from its `mime`.

    RGB strips keep the `STRIP_FORMAT` fast path (BMP by default); the RGBA
    strips that carry transparent NaN/±Inf cells must be PNG, the only
    `_MIME_TYPES` format with an alpha channel.
    """
    if pixels.shape[-1] == 4:
        return _pil_to_bytes(
            Image.fromarray(np.ascontiguousarray(pixels), mode="RGBA"), fmt="PNG"
        )
    return _encode_image(pixels)


def _pil_to_bytes(pil: Image.Image, *, fmt: str | None = None) -> bytes:
    fmt = fmt or STRIP_FORMAT
    buf = io.BytesIO()
    if fmt == "PNG":
        pil.save(buf, format="PNG", compress_level=PNG_COMPRESS_LEVEL)
    else:
        pil.save(buf, format=fmt)
    return buf.getvalue()
