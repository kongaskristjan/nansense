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
import math
from collections.abc import Sequence
from typing import TYPE_CHECKING, NamedTuple

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
from nansense.ui.theme import (
    BAR_RADIUS,
    LABEL_FONT_SIZE,
    LABEL_GAP,
    LABEL_TRACKING,
    MARKER_FONT_SIZE,
    MARKER_GAP,
    MARKER_LABEL_MIN_HEIGHT,
    MARKER_TRACKING,
    MARKER_WIDTH,
    NEUTRAL_COLOR,
    Marker,
    mono_font,
    rgb,
)
from nansense.watch import N_BINS, LayerStatsSnapshot

if TYPE_CHECKING:
    # Annotations only — matplotlib stays a lazy import (see
    # `histogram_image`), and these cost nothing at runtime.
    from matplotlib.axis import Axis
    from matplotlib.ticker import Formatter

# Hard cap on a composed image's width and height. A layer with thousands of
# channels would otherwise compose into a picture measured in gigapixels; the
# row is truncated instead. (`nansense.recording` caps its *video* dimensions
# at the same value for libx264's sake — a separate concern, same number.)
MAX_IMAGE_SIZE: int = 4096

_SECTION_GAP: int = 10
_FRAME_PAD: int = 10
_COLUMN_GAP: int = 2
#: Horizontal space a strip's marker takes from the strip beside it.
_MARKER_COLUMN: int = MARKER_WIDTH + MARKER_GAP
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

# Matplotlib histogram geometry. A row's height is the page's own
# `histograms._PLOT_HEIGHT` at this DPI, so a composed row has the same aspect
# as the plot the page draws rather than a squatter one of its own.
_HIST_DPI: int = 100
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


_AnyFont = ImageFont.FreeTypeFont | ImageFont.ImageFont


def _tracked_width(text: str, font: _AnyFont, tracking: float) -> float:
    """Width of `text` including CSS-style `letter-spacing` after every glyph."""
    probe = ImageDraw.Draw(Image.new("RGB", (1, 1)))
    return sum(probe.textlength(ch, font=font) + tracking for ch in text)


def _draw_tracked(
    draw: ImageDraw.ImageDraw,
    xy: tuple[float, float],
    text: str,
    font: _AnyFont,
    fill: tuple[int, int, int],
    tracking: float,
) -> None:
    """Draw `text` glyph by glyph so `letter-spacing` survives into the image.

    PIL has no tracking of its own, and the page's label bars set it (0.04em on
    a caption, 0.12em on a marker's vertical label). At marker size that is
    ~1px a character — over a word like ACTIVATIONS it is the difference
    between matching the page and running a dozen pixels short.
    """
    x, y = xy
    for ch in text:
        draw.text((x, y), ch, fill=fill, font=font)
        x += draw.textlength(ch, font=font) + tracking


def _ellipsized(text: str, font: _AnyFont, limit: float, tracking: float) -> str:
    """`text` trimmed to `limit` px with an ellipsis, matching CSS overflow.

    The bars set `text-overflow: ellipsis`, so an over-long caption ends in "…"
    rather than being cut mid-glyph or overflowing its bar.
    """
    if _tracked_width(text, font, tracking) <= limit:
        return text
    for end in range(len(text) - 1, 0, -1):
        candidate = f"{text[:end]}…"
        if _tracked_width(candidate, font, tracking) <= limit:
            return candidate
    return ""


def label_bar(
    text: str, width: int, *, color: str = NEUTRAL_COLOR
) -> Image.Image:
    """A rounded filled caption bar — the PIL twin of `common._label_bar_html`.

    White bold monospace centered on a colored, `BAR_RADIUS`-rounded bar
    `LABEL_HEIGHT` tall: the shared look of the `CHANNEL n` column headers, the
    `SAMPLE n` row labels and the experiment cell captions. An empty `text`
    still returns the bar's worth of background, so a caption-less column keeps
    the same geometry as its neighbours.
    """
    bar = Image.new("RGB", (max(width, 1), LABEL_HEIGHT), _BACKGROUND)
    if not text:
        return bar
    draw = ImageDraw.Draw(bar)
    draw.rounded_rectangle(
        (0, 0, max(width, 1) - 1, LABEL_HEIGHT - 1), radius=BAR_RADIUS, fill=rgb(color)
    )
    font = mono_font(LABEL_FONT_SIZE)
    shown = _ellipsized(text, font, width - 4, LABEL_TRACKING)
    text_w = _tracked_width(shown, font, LABEL_TRACKING)
    _draw_tracked(
        draw,
        ((width - text_w) / 2, (LABEL_HEIGHT - LABEL_FONT_SIZE) / 2 - 1),
        shown,
        font,
        (255, 255, 255),
        LABEL_TRACKING,
    )
    return bar


def _vertical_bar(
    text: str,
    *,
    width: int,
    height: int,
    color: str,
    font_size: int,
    tracking: float,
) -> Image.Image:
    """A rounded colored bar with its label rotated to read bottom-up.

    The shared body of `row_label_bar` and `marker_bar`. The label is drawn
    horizontally into a scratch image and rotated a quarter turn
    counter-clockwise, which is what puts the first character at the bottom —
    the page reaches the same reading direction with `writing-mode: vertical-rl`
    plus a 180° turn.
    """
    bar = Image.new("RGB", (width, max(height, 1)), _BACKGROUND)
    if height <= 0:
        return bar
    draw = ImageDraw.Draw(bar)
    draw.rounded_rectangle(
        (0, 0, width - 1, height - 1), radius=BAR_RADIUS, fill=rgb(color)
    )
    if not text:
        return bar
    font = mono_font(font_size)
    shown = _ellipsized(text, font, height - 4, tracking)
    text_w = _tracked_width(shown, font, tracking)
    scratch = Image.new("RGB", (max(int(text_w) + 2, 1), font_size + 4), rgb(color))
    _draw_tracked(
        ImageDraw.Draw(scratch), (0, 0), shown, font, (255, 255, 255), tracking
    )
    rotated = scratch.rotate(90, expand=True)
    bar.paste(
        rotated, ((width - rotated.width) // 2, (height - rotated.height) // 2)
    )
    return bar


def row_label_bar(
    text: str, height: int, *, color: str = NEUTRAL_COLOR
) -> Image.Image:
    """A `SAMPLE n` row label — the PIL twin of `common._row_label_bar_html`.

    `LABEL_HEIGHT` wide (the bars are square-ish by design: a row label is a
    column header turned on its side) and as tall as the row it names.
    """
    return _vertical_bar(
        text,
        width=LABEL_HEIGHT,
        height=height,
        color=color,
        font_size=LABEL_FONT_SIZE,
        tracking=LABEL_TRACKING,
    )


def marker_bar(marker: Marker, label: str, height: int) -> Image.Image:
    """A strip's kind marker — the PIL twin of `common._strip_marker`.

    A `MARKER_WIDTH` colored bar as tall as the strip beside it, carrying the
    kind's name bottom-up. On a strip too short to read the label down
    (`MARKER_LABEL_MIN_HEIGHT`) the bar is drawn bare, exactly as the page's
    container query hides it — the color still says which kind this is.
    """
    return _vertical_bar(
        label if height >= MARKER_LABEL_MIN_HEIGHT else "",
        width=MARKER_WIDTH,
        height=height,
        color=marker.color,
        font_size=MARKER_FONT_SIZE,
        tracking=MARKER_TRACKING,
    )


def captioned_columns(
    legend: Image.Image | None,
    columns: Sequence[tuple[Image.Image, str]],
    *,
    show_labels: bool = True,
    label_gap: int = LABEL_GAP,
) -> Image.Image | None:
    """Lay out a legend plus captioned column images into one image.

    Shared by the activation strips (`strip_image`) and the MIN/MAX patch grids
    (`patch_grid_image`): the optional `legend` leads the row under a blank
    caption-height band, then each column image is placed left to right under
    its `label_bar` caption. Columns are accumulated until the row would exceed
    `MAX_IMAGE_SIZE`.

    `show_labels` mirrors `common._strip_html`'s flag of the same name: a card
    captions its *first* strip and stacks the rest below it uncaptioned, so the
    shared `CHANNEL n` headers sit once atop the table rather than repeating on
    every row. Without them the caption band and its gap collapse, and the row
    is only as tall as the images.

    `label_gap` is the gutter between a caption and what it captions: strips set
    `LABEL_GAP`, the patch grids `PATCH_CELL_GAP`, each matching the CSS `gap`
    its own page layout uses.
    """
    if not columns:
        return None
    band = LABEL_HEIGHT + label_gap if show_labels else 0
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
    canvas = Image.new("RGB", (total_width, band + body_height), _BACKGROUND)
    if legend is not None:
        canvas.paste(legend, (0, band))
    for img, label, col_x in placements:
        if show_labels and label:
            canvas.paste(label_bar(label, img.width), (col_x, 0))
        canvas.paste(img, (col_x, band))
    return canvas


def strip_image(
    strip: StripRender | None, *, show_labels: bool = True
) -> Image.Image | None:
    """Decode a `StripRender` to one display-resolution image.

    Each tile is nearest-upscaled to its CSS display size (matching the
    browser's `image-rendering: pixelated`) and laid out left to right after the
    crisp legend by `captioned_columns`, with its `CHANNEL n` caption bar drawn
    above it — reproducing the captioned columns the page shows.

    `show_labels` follows the page's card rule (see `captioned_columns`): the
    first strip of a card carries the headers, the rows stacked below it reuse
    them.

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
    return captioned_columns(legend, columns, show_labels=show_labels)


def patch_grid_image(grid: PatchGridRender | None) -> Image.Image | None:
    """Decode a `PatchGridRender` to one display-resolution image.

    Each channel's cells are nearest-upscaled to their CSS square and stacked
    with a `PATCH_CELL_GAP` gutter into a column image, then laid out after the
    optional heat legend by `captioned_columns` under a `CHANNEL n` caption. A
    `SAMPLE n` row-label column leads the row, so the composed grid reads as the
    same table the page draws: channel headers across, sample labels down.

    The caption gutter here is `PATCH_CELL_GAP`, not `LABEL_GAP` — the page's
    grid columns run on one vertical rhythm from the header through the cells
    (`stats_page._patch_column_html`), where a strip separates its header with
    the wider `LABEL_GAP`.
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
    body = captioned_columns(legend, columns, label_gap=PATCH_CELL_GAP)
    if body is None:
        return None
    return _hstack([_sample_column(grid), body])


def _sample_column(grid: PatchGridRender) -> Image.Image:
    """The `SAMPLE n` row labels running down the left of a patch grid.

    Leads with the same header-height spacer the channel columns do, so the
    labels line up with the cell rows rather than the headers above them.
    """
    cell = grid.columns[0].cell_size
    rows = len(grid.columns[0].cells)
    height = LABEL_HEIGHT + PATCH_CELL_GAP + rows * cell + max(rows - 1, 0) * PATCH_CELL_GAP
    column = Image.new("RGB", (LABEL_HEIGHT, max(height, 1)), _BACKGROUND)
    y = LABEL_HEIGHT + PATCH_CELL_GAP
    for i in range(rows):
        column.paste(row_label_bar(f"SAMPLE {i}", cell), (0, y))
        y += cell + PATCH_CELL_GAP
    return column


def _hstack(images: Sequence[Image.Image], gap: int = _COLUMN_GAP) -> Image.Image:
    """Place `images` left to right, top-aligned, `gap` px apart."""
    width = sum(img.width for img in images) + gap * max(len(images) - 1, 0)
    canvas = Image.new(
        "RGB",
        (min(width, MAX_IMAGE_SIZE), max(img.height for img in images)),
        _BACKGROUND,
    )
    x = 0
    for img in images:
        canvas.paste(img, (x, 0))
        x += img.width + gap
    return canvas


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


class Section(NamedTuple):
    """One row of a composed frame: a heading, a strip, or a marked strip.

    With no `marker` the row is a text line — a card's layer name, an
    experiment's status, a line of optimizer scalars — drawn the way the page
    draws its headings. With one, the row is a strip: the marker becomes the
    colored bar down its left carrying `label`, exactly as `common._strip_marker`
    puts it there, and `label` is not drawn as text at all.

    `header_gap` offsets that marker past the strip's `CHANNEL n` header band so
    it lines up with the tiles rather than standing taller than the markers on
    the header-less rows below it. Set it on the same row the caller renders
    with `strip_image(..., show_labels=True)` — they describe the one card rule
    from two sides.
    """

    label: str
    image: Image.Image | None
    marker: Marker | None = None
    header_gap: bool = False


def stack_sections(
    sections: Sequence[Section],
    *,
    require_image: bool = False,
) -> Image.Image | None:
    """Stack labelled images vertically onto one background.

    A section whose image is `None` contributes its label alone — that is how a
    view says "this part had nothing to draw" without collapsing the layout, and
    how a caller adds a bare text line (an experiment's progress, say).

    That is right for a recording, whose frame should keep its shape even on an
    empty update. It is wrong for a caller that has to decide between sending a
    picture and sending an explanation, because a canvas of nothing but captions
    *is* a valid image and would be sent as one. `require_image` returns `None`
    for that case instead, so the caller can say what was missing.
    """
    if not sections:
        return None
    if require_image and all(section.image is None for section in sections):
        return None
    font = mono_font(LABEL_FONT_SIZE)
    widths = [
        section.image.width + (_MARKER_COLUMN if section.marker is not None else 0)
        for section in sections
        if section.image is not None
    ]
    width = _FRAME_PAD * 2 + min(MAX_IMAGE_SIZE, max(widths + [_MIN_SECTION_WIDTH]))
    height = _FRAME_PAD
    for section in sections:
        if section.label and section.marker is None:
            height += LABEL_HEIGHT + 2
        if section.image is not None:
            height += min(section.image.height, MAX_IMAGE_SIZE) + _SECTION_GAP
        else:
            height += _SECTION_GAP
    height = min(height + _FRAME_PAD, MAX_IMAGE_SIZE)
    canvas = Image.new("RGB", (width, height), _BACKGROUND)
    draw = ImageDraw.Draw(canvas)
    y = _FRAME_PAD
    for section in sections:
        if section.label and section.marker is None:
            _draw_tracked(
                draw,
                (_FRAME_PAD, y),
                section.label,
                font,
                _LABEL_COLOR,
                LABEL_TRACKING,
            )
            y += LABEL_HEIGHT + 2
        if section.image is None:
            y += _SECTION_GAP
            continue
        drawn_height = min(section.image.height, MAX_IMAGE_SIZE)
        x = _FRAME_PAD
        if section.marker is not None:
            band = LABEL_HEIGHT + LABEL_GAP if section.header_gap else 0
            canvas.paste(
                marker_bar(section.marker, section.label, drawn_height - band),
                (x, y + band),
            )
            x += _MARKER_COLUMN
        canvas.paste(section.image, (x, y))
        y += drawn_height + _SECTION_GAP
    return canvas


def histogram_image(
    rows: Sequence[tuple[str, str, str, LayerStatsSnapshot]],
    *,
    log_x: bool = False,
    log_y: bool = False,
    channel: int | None = None,
) -> Image.Image | None:
    """Matplotlib re-render of `(layer, kind, phase, stats)` histogram rows.

    The `/stats` page draws these with Plotly in the browser, which cannot
    produce a server-side picture; this redraws the same bars, ranges and
    overflow markers with the Agg backend so a video frame or an agent's image
    reply shows what the page shows. `kind` is `"activation"` or `"gradient"`.

    Every styling value comes from `nansense.ui.histograms` — the same
    constants that build the Plotly figure — so the two agree on background,
    gridlines, bar color and opacity, fonts, and the power-of-ten tick format.
    Two things are deliberately *not* the page:

    - the page draws one figure per (layer, kind) with a row per phase, while a
      frame stacks rows from several layers and kinds, so each row's title
      carries its own layer and kind rather than relying on a figure title; and
    - the page puts the scalar stats in a table beside the plot
      (`_stats_table_html`), which a single stacked image has no room for, so
      they ride along in each row's right-hand corner instead. Dropping them
      would cost the frame information the page does show.

    `channel` only labels the subplot titles — the caller narrows the rows
    themselves, since a row without per-channel data keeps the universal
    histogram and the title has to say so rather than claim a channel.
    """
    if not rows:
        return None
    # Matplotlib is imported here rather than at module level: it is a heavy
    # import and only this one function needs it.
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    from nansense.ui.histograms import _PLOT_HEIGHT, PAPER_BG

    row_inches = _PLOT_HEIGHT / _HIST_DPI
    fig = Figure(
        figsize=(_HIST_WIDTH_INCHES, row_inches * len(rows)),
        dpi=_HIST_DPI,
        facecolor=PAPER_BG,
    )
    axes = fig.subplots(len(rows), 1, squeeze=False)
    for ax_row, (layer, kind, phase, stats) in zip(axes, rows, strict=True):
        _draw_histogram_axes(
            ax_row[0],
            layer,
            kind,
            phase,
            stats,
            log_x=log_x,
            log_y=log_y,
            channel=channel,
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
    channel: int | None = None,
) -> None:
    """One subplot: the same bars/ranges/chrome the watch page draws with Plotly."""
    from matplotlib.axes import Axes

    from nansense.ui.histograms import (
        AXIS_TITLE_FONT_SIZE,
        BAR_OPACITY,
        BIN_CENTERS,
        BIN_WIDTHS,
        OVERFLOW_MARKER_COLOR,
        SUBPLOT_TITLE_FONT_SIZE,
        TICK_COLOR,
        TICK_FONT_SIZE,
        TITLE_FONT_SIZE,
        Y_AXIS_TITLE,
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
    # `bargap=0` on the page: bars tile their bins with no edge line between.
    if log_x:
        ax.bar(
            x_values, heights, width=1.0, color=color, alpha=BAR_OPACITY, linewidth=0
        )
        tick_vals, tick_text = x_tick_layout()
        ax.set_xticks(tick_vals, tick_text)
    else:
        ax.bar(
            x_values,
            heights,
            width=BIN_WIDTHS,
            color=color,
            alpha=BAR_OPACITY,
            linewidth=0,
        )
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
    _style_histogram_axes(ax, log_x=log_x, density=density)
    # A row the caller could not narrow (no per-channel data) keeps the
    # universal histogram, so the title must not claim a channel it isn't
    # showing — the picture is the only place a reader can check that.
    scope = (
        f" · ch {min(channel, len(tensor_stats.channel_hists) - 1)}"
        if channel is not None and tensor_stats.channel_hists is not None
        else (" · all channels" if channel is not None else "")
    )
    # The page's three pieces of heading, in matplotlib's three title slots:
    # its figure title on the left, its phase-tinted subplot title centered,
    # and the stats table's scalars — which have nowhere else to go here — kept
    # small and gray on the right so they read as an annotation, not a label.
    ax.set_title(
        f"{layer} — {kind}s", loc="left", fontsize=_pt(TITLE_FONT_SIZE), color="#1e293b"
    )
    ax.set_title(
        f"{phase} (ep {stats.epoch}){scope}",
        loc="center",
        fontsize=_pt(SUBPLOT_TITLE_FONT_SIZE),
        color=color,
    )
    ax.set_title(
        f"n={tensor_stats.n:,}  mean={tensor_stats.mean:.3g}  "
        f"std={tensor_stats.std:.3g}",
        loc="right",
        fontsize=_pt(TICK_FONT_SIZE),
        color="#64748b",
    )
    ax.set_ylabel(
        Y_AXIS_TITLE[density], fontsize=_pt(AXIS_TITLE_FONT_SIZE), color=TICK_COLOR
    )


def _style_histogram_axes(ax: object, *, log_x: bool, density: bool) -> None:
    """The Plotly plot's chrome, on a matplotlib axes.

    Plotly draws no axis lines or tick marks — just a filled plotting area,
    horizontal gridlines, and (on the linear x-axis) a zero line. Matplotlib
    defaults to the opposite of all three: a black box, outward ticks, no fill.
    """
    from matplotlib.axes import Axes

    from nansense.ui.histograms import (
        GRID_COLOR,
        PLOT_BG,
        TICK_COLOR,
        TICK_FONT_SIZE,
        ZEROLINE_COLOR,
    )

    assert isinstance(ax, Axes)
    ax.set_facecolor(PLOT_BG)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(labelsize=_pt(TICK_FONT_SIZE), colors=TICK_COLOR, length=0)
    # y gridlines only, behind the bars.
    ax.grid(visible=True, axis="y", color=GRID_COLOR, linewidth=0.8)
    ax.grid(visible=False, axis="x")
    ax.set_axisbelow(True)
    ax.yaxis.set_major_formatter(_power_ticks(ax.yaxis))
    if not log_x:
        # The signed-log axis labels its own ticks (`x_tick_layout`); only the
        # linear one formats values, and only it carries a zero line.
        ax.xaxis.set_major_formatter(_power_ticks(ax.xaxis))
        ax.axvline(0.0, color=ZEROLINE_COLOR, linewidth=1, zorder=0)


def _pt(px: float) -> float:
    """CSS px as matplotlib points at `_HIST_DPI`.

    Plotly sizes its fonts in px, matplotlib in points — at 100 dpi a "size 12"
    asked of each differs by 1.39x, which is the whole heading hierarchy drawn
    a size too large. Every font size crossing over from `histograms` goes
    through here.
    """
    return px * 72.0 / _HIST_DPI


def _power_ticks(axis: Axis) -> Formatter:
    """A tick formatter in Plotly's `exponentformat="power"` style.

    Plotly writes 2e-8 as `2x10^-8`, and factors *one* exponent out across the
    whole axis — ticks up to 1.2e8 read `0.2x10^8`, `0.4x10^8`, not each with
    its own power. Matplotlib instead hoists a shared power into a corner
    offset box and labels the ticks `-2.0 ... 2.0`, which on a histogram of
    gradient magnitudes reads as if the axis ran to +/-2.

    The exponent is taken from the axis' view limits at draw time, so it tracks
    a zoomed or capped range the way Plotly's does. Ordinary magnitudes stay
    plain, as they do on the page.
    """
    from matplotlib.ticker import FuncFormatter

    def format_tick(value: float, _pos: object) -> str:
        if value == 0:
            return "0"
        lo, hi = axis.get_view_interval()
        largest = max(abs(lo), abs(hi)) or abs(value)
        exponent = math.floor(math.log10(largest))
        if -3 <= exponent < 4:
            return f"{value:g}"
        return f"${value / 10.0**exponent:g}{{\\times}}10^{{{exponent}}}$"

    return FuncFormatter(format_tick)
