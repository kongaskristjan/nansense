"""Tests for composing rendered pieces into one image (`nansense.ui.compose`)."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from PIL import Image

from nansense.ui.compose import (
    _CHECKER_DARK,
    _CHECKER_LIGHT,
    MAX_IMAGE_SIZE,
    Section,
    captioned_columns,
    checkerboard,
    label_bar,
    marker_bar,
    row_label_bar,
    stack_sections,
    strip_image,
    upscaled_image,
)
from nansense.ui.render import INPUT_IMAGE_SIZE, LABEL_HEIGHT, render_image, render_strip
from nansense.ui.theme import (
    ACTIVATIONS,
    BAR_RADIUS,
    GRADIENTS,
    LABEL_GAP,
    MARKER_LABEL_MIN_HEIGHT,
    MARKER_WIDTH,
    NEUTRAL_COLOR,
    mono_font,
    rgb,
)


def test_checkerboard_is_two_grays_in_4px_boxes() -> None:
    cb = np.asarray(checkerboard(8, 8))
    assert cb.shape == (8, 8, 4)
    assert (cb[..., 3] == 255).all()  # opaque backdrop
    assert tuple(cb[0, 0, :3]) == _CHECKER_LIGHT
    assert tuple(cb[0, 4, :3]) == _CHECKER_DARK  # box flips at 4px
    assert tuple(cb[4, 0, :3]) == _CHECKER_DARK
    assert tuple(cb[4, 4, :3]) == _CHECKER_LIGHT


def test_strip_image_bakes_checkerboard_behind_nan_cells() -> None:
    # An all-NaN strip is fully transparent RGBA; the composed image must show
    # the two checkerboard grays behind it, not white and not one gray.
    strip = render_strip(torch.full((1, 1, 8, 8), float("nan")), sample_idx=0)
    assert strip is not None
    composed = strip_image(strip)
    assert composed is not None
    arr = np.asarray(composed)
    colors = {tuple(c) for row in arr for c in row}
    assert _CHECKER_LIGHT in colors
    assert _CHECKER_DARK in colors
    # No fully-white data region (the old whitewash bug) over the NaN cells.
    nan_region = arr[:, arr.shape[1] // 2 :]  # right of the legend, in the tile
    assert not (nan_region == 255).all()


def test_strip_image_keeps_plain_path_for_finite_strip() -> None:
    # An all-finite strip is opaque RGB; no checkerboard gray leaks in.
    strip = render_strip(torch.zeros(1, 1, 8, 8), sample_idx=0)
    assert strip is not None
    composed = strip_image(strip)
    assert composed is not None
    colors = {tuple(c) for row in np.asarray(composed) for c in row}
    assert _CHECKER_DARK not in colors


@pytest.mark.parametrize("value", [None, "empty"])
def test_strip_image_of_nothing_is_none(value: str | None) -> None:
    """A `None` strip and a tile-less one both compose to nothing, so callers
    can pass a failed render straight through."""
    if value is None:
        assert strip_image(None) is None
    else:
        assert strip_image(render_strip(torch.zeros(1, 0, 4, 4), sample_idx=0)) is None


def _block(width: int, height: int, color: tuple[int, int, int]) -> Image.Image:
    return Image.new("RGB", (width, height), color)


def test_captioned_columns_places_every_column_left_to_right() -> None:
    columns = [(_block(10, 20, (255, 0, 0)), "CH 0"), (_block(10, 20, (0, 0, 255)), "CH 1")]
    composed = captioned_columns(None, columns)
    assert composed is not None
    arr = np.asarray(composed)
    # Both column colors survive, and the caption band sits above them.
    colors = {tuple(c) for row in arr for c in row}
    assert (255, 0, 0) in colors and (0, 0, 255) in colors
    # Caption bar plus the gutter the page puts between it and the image.
    assert composed.height == LABEL_HEIGHT + LABEL_GAP + 20


def test_captioned_columns_draws_the_pages_filled_label_bar() -> None:
    """The page captions a column with a filled slate bar, not bare text — the
    composed still has to show the same furniture (`common._label_bar_html`)."""
    composed = captioned_columns(None, [(_block(60, 20, (0, 0, 0)), "CHANNEL 0")])
    assert composed is not None
    band = np.asarray(composed)[:LABEL_HEIGHT]
    colors = {tuple(c) for row in band for c in row}
    assert rgb(NEUTRAL_COLOR) in colors  # the bar's fill
    assert (255, 255, 255) in colors  # white text on it


def test_captioned_columns_without_labels_drops_the_whole_band() -> None:
    """A card captions only its first strip; the rows below reuse those headers
    and must not reserve — or repeat — a band of their own."""
    columns = [(_block(10, 20, (255, 0, 0)), "CH 0")]
    headed = captioned_columns(None, columns)
    bare = captioned_columns(None, columns, show_labels=False)
    assert headed is not None and bare is not None
    assert bare.height == 20
    assert headed.height == bare.height + LABEL_HEIGHT + LABEL_GAP


def test_captioned_columns_stops_at_the_size_cap() -> None:
    """A layer with a great many channels truncates rather than composing a
    picture no encoder (and no agent) can take."""
    wide = [(_block(64, 8, (1, 2, 3)), "") for _ in range(200)]
    composed = captioned_columns(None, wide)
    assert composed is not None
    assert composed.width <= MAX_IMAGE_SIZE


def test_captioned_columns_of_nothing_is_none() -> None:
    assert captioned_columns(None, []) is None


def test_stack_sections_keeps_label_only_rows() -> None:
    """A section with no image is how a view says "nothing to draw here" — and
    how a caller adds a bare text line; it must not collapse the layout."""
    with_image = stack_sections([Section("a", _block(40, 30, (0, 0, 0)))])
    with_text_row = stack_sections(
        [Section("a", _block(40, 30, (0, 0, 0))), Section("just a line", None)]
    )
    assert with_image is not None and with_text_row is not None
    assert with_text_row.height > with_image.height


def test_stack_sections_of_nothing_is_none() -> None:
    assert stack_sections([]) is None


def test_stack_sections_marks_a_strip_with_its_kinds_color() -> None:
    """Every strip uses the same colormap, so the marker beside it is the only
    thing saying whether it is an activation or a gradient. A composed frame
    that drops it drops the distinction."""
    composed = stack_sections(
        [
            Section("ACTIVATIONS", _block(60, 128, (0, 0, 0)), ACTIVATIONS),
            Section("GRADIENTS", _block(60, 128, (0, 0, 0)), GRADIENTS),
        ]
    )
    assert composed is not None
    colors = {tuple(c) for row in np.asarray(composed) for c in row}
    assert rgb(ACTIVATIONS.color) in colors
    assert rgb(GRADIENTS.color) in colors


def test_stack_sections_offsets_a_marker_past_the_header_band() -> None:
    """The card's first strip carries the `CHANNEL n` headers, so its marker
    starts below them — otherwise it stands taller than every marker under it."""
    plain = stack_sections([Section("ACTIVATIONS", _block(60, 128, (0, 0, 0)), ACTIVATIONS)])
    headed = stack_sections(
        [Section("ACTIVATIONS", _block(60, 128, (0, 0, 0)), ACTIVATIONS, header_gap=True)]
    )
    assert plain is not None and headed is not None
    fill = rgb(ACTIVATIONS.color)

    def marker_rows(img: Image.Image) -> list[int]:
        arr = np.asarray(img)
        return [y for y in range(arr.shape[0]) if fill in {tuple(c) for c in arr[y]}]

    assert marker_rows(headed)[0] - marker_rows(plain)[0] == LABEL_HEIGHT + LABEL_GAP


def test_marker_label_is_dropped_on_a_strip_too_short_to_read_it() -> None:
    """Mirrors the page's container query: under `MARKER_LABEL_MIN_HEIGHT` the
    vertical label cannot be read, so the bar carries color alone."""
    tall = np.asarray(marker_bar(ACTIVATIONS, "ACTIVATIONS", MARKER_LABEL_MIN_HEIGHT))
    short = np.asarray(marker_bar(ACTIVATIONS, "ACTIVATIONS", MARKER_LABEL_MIN_HEIGHT - 1))
    # Inside the rounded corners, a bare bar is nothing but its fill; a labelled
    # one has white glyphs on it.
    inside = (slice(BAR_RADIUS, -BAR_RADIUS), slice(BAR_RADIUS, -BAR_RADIUS))
    assert (255, 255, 255) in {tuple(c) for row in tall[inside] for c in row}
    assert {tuple(c) for row in short[inside] for c in row} == {rgb(ACTIVATIONS.color)}
    assert tall.shape[1] == MARKER_WIDTH


def test_row_label_bar_reads_bottom_up() -> None:
    """`SAMPLE n` labels run down the grid's left edge, rotated — so the bar is
    `LABEL_HEIGHT` wide and as tall as the row it names."""
    bar = row_label_bar("SAMPLE 0", 64)
    assert (bar.width, bar.height) == (LABEL_HEIGHT, 64)
    colors = {tuple(c) for row in np.asarray(bar) for c in row}
    assert rgb(NEUTRAL_COLOR) in colors and (255, 255, 255) in colors


def test_label_bar_renders_the_em_dash_the_frame_labels_use() -> None:
    """PIL's built-in font has no em-dash glyph and drew a tofu box in every
    recorded frame; the composed bars must use a font that has one."""
    font = mono_font(10)
    assert font.getmask("—").getbbox() is not None
    bar = np.asarray(label_bar("a — b", 120))
    assert (255, 255, 255) in {tuple(c) for row in bar for c in row}


def test_label_bar_ellipsizes_rather_than_overflowing() -> None:
    """`text-overflow: ellipsis` on the page; the bar must not spill its text
    past the column it captions."""
    bar = label_bar("CHANNEL 1234567890 far too long", 40)
    assert bar.width == 40


def test_upscaled_image_blows_a_small_input_up_to_display_size() -> None:
    data = render_image(torch.rand(1, 3, 8, 8), 0)
    assert data is not None
    img = upscaled_image(data)
    assert img is not None
    assert img.width == INPUT_IMAGE_SIZE
    assert img.height == INPUT_IMAGE_SIZE


def test_upscaled_image_of_nothing_is_none() -> None:
    assert upscaled_image(None) is None
