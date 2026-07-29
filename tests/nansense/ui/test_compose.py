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
    captioned_columns,
    checkerboard,
    stack_sections,
    strip_image,
    upscaled_image,
)
from nansense.ui.render import INPUT_IMAGE_SIZE, render_image, render_strip


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
    assert composed.height > 20  # the caption row was added


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
    with_image = stack_sections([("a", _block(40, 30, (0, 0, 0)))])
    with_text_row = stack_sections(
        [("a", _block(40, 30, (0, 0, 0))), ("just a line", None)]
    )
    assert with_image is not None and with_text_row is not None
    assert with_text_row.height > with_image.height


def test_stack_sections_of_nothing_is_none() -> None:
    assert stack_sections([]) is None


def test_upscaled_image_blows_a_small_input_up_to_display_size() -> None:
    data = render_image(torch.rand(1, 3, 8, 8), 0)
    assert data is not None
    img = upscaled_image(data)
    assert img is not None
    assert img.width == INPUT_IMAGE_SIZE
    assert img.height == INPUT_IMAGE_SIZE


def test_upscaled_image_of_nothing_is_none() -> None:
    assert upscaled_image(None) is None
