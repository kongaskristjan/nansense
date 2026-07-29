"""Composing rendered tensor pieces into single still images.

`nansense.ui.render` returns the *pieces* a view shows — one image per channel
tile, one per patch cell, plus a shared legend — because the browser lays them
out itself with CSS. Consumers that have to hand over one finished picture do
that layout themselves, and they all want the same one: a legend leading a row
of captioned columns, sections stacked under their labels.

There are two such consumers today — `nansense.recording`, which encodes each
composed image as a video frame, and `nansense.mcp_images`, which sends it to a
coding agent — and this module is the layout they share.

Import it lazily, the way both callers do. It imports `nansense.ui.render` at
module level, which pulls in `nansense.ui.__init__` and with it the NiceGUI
app; a library module that reaches for a composed image should not pay that at
import time.
"""

from __future__ import annotations

import io
from collections.abc import Sequence

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from torch import Tensor

from nansense.ui.render import (
    INPUT_IMAGE_SIZE,
    LABEL_HEIGHT,
    PATCH_CELL_GAP,
    PatchGridRender,
    StripRender,
    render_image,
)
from nansense.watch import N_BINS, LayerStatsSnapshot

# Hard cap on a composed image's width and height. A layer with thousands of
# channels would otherwise compose into a picture measured in gigapixels; the
# row is truncated instead. (`nansense.recording` caps its *video* dimensions
# at the same value for libx264's sake — a separate concern, same number.)
MAX_IMAGE_SIZE: int = 4096

_SECTION_GAP: int = 10
_FRAME_PAD: int = 10
_COLUMN_GAP: int = 2
_LABEL_COLOR: tuple[int, int, int] = (30, 41, 59)  # slate-800
_BACKGROUND: tuple[int, int, int] = (255, 255, 255)
_MIN_SECTION_WIDTH: int = 320

# GIMP-style transparency backdrop baked behind a strip's transparent NaN/±Inf
# cells, matching the live UI's CSS checkerboard
# (`static._STRIP_CHECKERBOARD_STYLE`): two slate grays in 4px boxes at display
# resolution. All-finite (opaque RGB) strips never see it.
_CHECKER_BOX: int = 4
_CHECKER_LIGHT: tuple[int, int, int] = (249, 250, 251)  # slate-50  (#f9fafb)
_CHECKER_DARK: tuple[int, int, int] = (229, 231, 235)  # slate-200 (#e5e7eb)

# Matplotlib histogram geometry: inches per subplot row at `_HIST_DPI`.
_HIST_DPI: int = 100
_HIST_ROW_INCHES: float = 2.4
_HIST_WIDTH_INCHES: float = 9.0


def checkerboard(width: int, height: int) -> Image.Image:
    """An opaque `_CHECKER_BOX`-square gray checkerboard, `width × height` RGBA.

    Mirrors the live UI's CSS backdrop so a composed NaN/±Inf cell shows the
    same two slate grays. Built vectorised: the box-parity of each pixel's
    `(row, col)` picks the light/dark color.
    """
    ys = (np.arange(height) // _CHECKER_BOX)[:, None]
    xs = (np.arange(width) // _CHECKER_BOX)[None, :]
    dark = (ys + xs) % 2 == 1
    rgb = np.empty((height, width, 3), dtype=np.uint8)
    rgb[...] = _CHECKER_LIGHT
    rgb[dark] = _CHECKER_DARK
    rgba = np.concatenate(
        [rgb, np.full((height, width, 1), 255, dtype=np.uint8)], axis=-1
    )
    return Image.fromarray(rgba, mode="RGBA")


def captioned_columns(
    legend: Image.Image | None, columns: Sequence[tuple[Image.Image, str]]
) -> Image.Image | None:
    """Lay out a legend plus captioned column images into one image.

    Shared by the activation strips (`strip_image`) and the MIN/MAX patch grids
    (`patch_grid_image`): the optional `legend` leads the row under a blank
    caption-height band, then each column image is placed left to right with its
    caption (already collapsed to fit) centered above it. Columns are
    accumulated until the row would exceed `MAX_IMAGE_SIZE`.
    """
    if not columns:
        return None
    x = legend.width + _COLUMN_GAP if legend is not None else 0
    body_height = legend.height if legend is not None else 0
    placements: list[tuple[Image.Image, str, int]] = []
    for img, label in columns:
        if x >= MAX_IMAGE_SIZE:
            break
        placements.append((img, label, x))
        body_height = max(body_height, img.height)
        x += img.width + _COLUMN_GAP
    total_width = min(x, MAX_IMAGE_SIZE)
    canvas = Image.new("RGB", (total_width, LABEL_HEIGHT + body_height), _BACKGROUND)
    if legend is not None:
        canvas.paste(legend, (0, LABEL_HEIGHT))
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    for img, label, col_x in placements:
        if label:
            text_w = draw.textlength(label, font=font)
            draw.text(
                (col_x + max(0, (img.width - text_w) / 2), 1),
                label,
                fill=_LABEL_COLOR,
                font=font,
            )
        canvas.paste(img, (col_x, LABEL_HEIGHT))
    return canvas


def strip_image(strip: StripRender | None) -> Image.Image | None:
    """Decode a `StripRender` to one display-resolution image.

    Each tile is nearest-upscaled to its CSS display size (matching the
    browser's `image-rendering: pixelated`) and laid out left to right after the
    crisp legend by `captioned_columns`, with its column caption drawn above it
    — reproducing the captioned columns the page shows.

    An RGBA tile carries transparent NaN/±Inf cells: it is composited over a
    baked gray `checkerboard` the same size as the upscaled tile, so the
    composed image shows the same GIMP-style backdrop the live UI paints with
    CSS. Opaque RGB tiles keep the plain path.
    """
    if strip is None or not strip.tiles:
        return None
    legend = Image.open(io.BytesIO(strip.legend_image)).convert("RGB")
    columns: list[tuple[Image.Image, str]] = []
    for tile in strip.tiles:
        decoded = Image.open(io.BytesIO(tile.image))
        if decoded.mode == "RGBA":
            up = decoded.resize((tile.width, tile.height), Image.Resampling.NEAREST)
            up = Image.alpha_composite(
                checkerboard(tile.width, tile.height), up
            ).convert("RGB")
        else:
            up = decoded.convert("RGB").resize(
                (tile.width, tile.height), Image.Resampling.NEAREST
            )
        columns.append((up, tile.label))
    return captioned_columns(legend, columns)


def patch_grid_image(grid: PatchGridRender | None) -> Image.Image | None:
    """Decode a `PatchGridRender` to one display-resolution image.

    Each channel's cells are nearest-upscaled to their CSS square and stacked
    with a `PATCH_CELL_GAP` gutter into a column image, then laid out after the
    optional heat legend by `captioned_columns` under a "CHANNEL N" caption —
    the still-image mirror of the MIN/MAX view's captioned cell grid.
    """
    if grid is None or not grid.columns:
        return None
    legend = (
        Image.open(io.BytesIO(grid.heat_legend)).convert("RGB")
        if grid.heat_legend is not None
        else None
    )
    columns: list[tuple[Image.Image, str]] = []
    for column in grid.columns:
        size = column.cell_size
        cell_imgs = [
            Image.open(io.BytesIO(cell))
            .convert("RGB")
            .resize((size, size), Image.Resampling.NEAREST)
            for cell in column.cells
        ]
        if not cell_imgs:
            continue
        height = len(cell_imgs) * size + (len(cell_imgs) - 1) * PATCH_CELL_GAP
        stack = Image.new("RGB", (size, height), _BACKGROUND)
        y = 0
        for cell_img in cell_imgs:
            stack.paste(cell_img, (0, y))
            y += size + PATCH_CELL_GAP
        columns.append((stack, column.label))
    return captioned_columns(legend, columns)


def upscaled_image(data: bytes | None) -> Image.Image | None:
    """Decode one rendered input image and scale it to `INPUT_IMAGE_SIZE` wide.

    The pane shows the raw pixels of a small input (a 28×28 digit, a 32×32
    CIFAR sample) blown up with nearest-neighbour; this reproduces that, so a
    composed frame is legible at the same size the page shows.
    """
    if data is None:
        return None
    img = Image.open(io.BytesIO(data)).convert("RGB")
    scale = INPUT_IMAGE_SIZE / max(img.width, 1)
    return img.resize(
        (INPUT_IMAGE_SIZE, max(1, round(img.height * scale))), Image.Resampling.NEAREST
    )


def batch_image_row(
    tensor: Tensor,
    *,
    mean: tuple[float, ...] | None,
    std: tuple[float, ...] | None,
) -> Image.Image | None:
    """Every sample of `tensor` as one horizontal row of upscaled images."""
    images: list[Image.Image] = []
    for i in range(int(tensor.shape[0])):
        img = upscaled_image(render_image(tensor, i, mean=mean, std=std))
        if img is not None:
            images.append(img)
    if not images:
        return None
    gap = 6
    width = sum(img.width for img in images) + gap * (len(images) - 1)
    height = max(img.height for img in images)
    canvas = Image.new("RGB", (min(width, MAX_IMAGE_SIZE), height), _BACKGROUND)
    x = 0
    for img in images:
        canvas.paste(img, (x, 0))
        x += img.width + gap
    return canvas


def stack_sections(
    sections: Sequence[tuple[str, Image.Image | None]],
) -> Image.Image | None:
    """Stack labelled images vertically onto one background.

    A section whose image is `None` contributes its label alone — that is how a
    view says "this part had nothing to draw" without collapsing the layout, and
    how a caller adds a bare text line (an experiment's progress, say).
    """
    if not sections:
        return None
    font = ImageFont.load_default()
    width = _FRAME_PAD * 2 + min(
        MAX_IMAGE_SIZE,
        max([img.width for _, img in sections if img is not None] + [_MIN_SECTION_WIDTH]),
    )
    height = _FRAME_PAD
    for label, img in sections:
        if label:
            height += LABEL_HEIGHT + 2
        if img is not None:
            height += min(img.height, MAX_IMAGE_SIZE) + _SECTION_GAP
        else:
            height += _SECTION_GAP
    height = min(height + _FRAME_PAD, MAX_IMAGE_SIZE)
    canvas = Image.new("RGB", (width, height), _BACKGROUND)
    draw = ImageDraw.Draw(canvas)
    y = _FRAME_PAD
    for label, img in sections:
        if label:
            draw.text((_FRAME_PAD, y), label, fill=_LABEL_COLOR, font=font)
            y += LABEL_HEIGHT + 2
        if img is not None:
            canvas.paste(img, (_FRAME_PAD, y))
            y += min(img.height, MAX_IMAGE_SIZE) + _SECTION_GAP
        else:
            y += _SECTION_GAP
    return canvas


def histogram_image(
    rows: Sequence[tuple[str, str, str, LayerStatsSnapshot]],
    *,
    log_x: bool = False,
    log_y: bool = False,
) -> Image.Image | None:
    """Matplotlib re-render of `(layer, kind, phase, stats)` histogram rows.

    The `/stats` page draws these with Plotly in the browser, which cannot
    produce a server-side picture; this redraws the same bars, ranges and
    overflow markers with the Agg backend so a video frame or an agent's image
    reply shows what the page shows. `kind` is `"activation"` or `"gradient"`.
    """
    if not rows:
        return None
    # Matplotlib is imported here rather than at module level: it is a heavy
    # import and only this one function needs it.
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    fig = Figure(
        figsize=(_HIST_WIDTH_INCHES, _HIST_ROW_INCHES * len(rows)), dpi=_HIST_DPI
    )
    axes = fig.subplots(len(rows), 1, squeeze=False)
    for ax_row, (layer, kind, phase, stats) in zip(axes, rows, strict=True):
        _draw_histogram_axes(
            ax_row[0], layer, kind, phase, stats, log_x=log_x, log_y=log_y
        )
    fig.tight_layout()
    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    return Image.fromarray(np.asarray(canvas.buffer_rgba())[..., :3].copy(), mode="RGB")


def _draw_histogram_axes(
    ax: object,
    layer: str,
    kind: str,
    phase: str,
    stats: LayerStatsSnapshot,
    *,
    log_x: bool,
    log_y: bool,
) -> None:
    """One subplot: the same bars/ranges the watch page draws with Plotly."""
    from matplotlib.axes import Axes

    from nansense.ui.histograms import (
        BIN_CENTERS,
        BIN_WIDTHS,
        OVERFLOW_MARKER_COLOR,
        axis_ranges,
        kind_stats,
        overflow_marks,
        phase_color,
        trace_heights,
        use_density,
        x_tick_layout,
    )

    assert isinstance(ax, Axes)
    tensor_stats = kind_stats(stats, kind)
    density = use_density(log_x)
    # A collapsed bucket (epoch-evicted bins) renders as an empty histogram.
    hist = tensor_stats.hist if tensor_stats.hist is not None else (0,) * N_BINS
    heights = trace_heights(hist, density)
    color = phase_color(phase, 0)
    x_values = list(range(N_BINS)) if log_x else list(BIN_CENTERS)
    if log_x:
        ax.bar(x_values, heights, width=1.0, color=color)
        tick_vals, tick_text = x_tick_layout()
        ax.set_xticks(tick_vals, tick_text, fontsize=6)
    else:
        ax.bar(x_values, heights, width=BIN_WIDTHS, color=color)
    per_phase = {phase: stats}
    x_range, y_range = axis_ranges(per_phase, kind, log_x=log_x, log_y=log_y)
    if x_range is not None:
        ax.set_xlim((x_range[0], x_range[1]))
    if log_y:
        ax.set_yscale("log")
    elif y_range is not None:
        ax.set_ylim((y_range[0], y_range[1]))
        # Flag bars clipped by the cap so they don't read as ending at the top
        # edge (mirrors the Plotly view's overflow markers).
        ((mark_xs, mark_ys),) = overflow_marks(
            [(phase, hist)], x_values, density, y_range[1]
        )
        if mark_xs:
            ax.scatter(
                mark_xs,
                mark_ys,
                marker="^",
                s=18,
                color=OVERFLOW_MARKER_COLOR,
                edgecolors="white",
                linewidths=0.5,
                zorder=3,
                clip_on=False,
            )
    title = (
        f"{layer} — {kind}s · {phase} (ep {stats.epoch}) · "
        f"n={tensor_stats.n:,} mean={tensor_stats.mean:.3g} "
        f"std={tensor_stats.std:.3g}"
    )
    ax.set_title(title, fontsize=8)
    ax.tick_params(labelsize=6)
