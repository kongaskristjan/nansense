"""Visual tokens shared by the two front-ends that draw a view.

A view reaches the reader two ways: the browser lays out `nansense.ui.render`
pieces with CSS (`nansense.ui.common`), and `nansense.ui.compose` lays the same
pieces out with PIL for a recording frame or an agent's image. Both draw the
*same* furniture around the pixels — colored marker bars naming a strip's kind,
rounded `CHANNEL n` / `SAMPLE n` label bars, the gutter between a caption and
what it captions — so the sizes and colors live here once instead of as a CSS
string on one side and a magic number on the other.

Kept import-light on purpose: no NiceGUI, no `nansense.ui.render`. Both callers
already import `render` for `LABEL_HEIGHT` (a caption bar's height is also a
render-math input, since the legend reserves it), so it stays there and this
module deliberately does not restate it.

`Marker.css` is what the page puts on a `div`; `Marker.color` is the same color
as bytes PIL can fill with. Change one and you change both front-ends.
"""

from __future__ import annotations

import functools
from typing import NamedTuple

from PIL import ImageFont

#: Gap between a caption bar and the image beneath it. The legend column gets
#: the same one, so tiles still line up under their headers.
LABEL_GAP: int = 4
#: Corner radius on every filled label bar (`border-radius: 3px`).
BAR_RADIUS: int = 3
#: A strip marker's width and its gap to the strip (Tailwind `w-5`, `mr-2`).
MARKER_WIDTH: int = 20
MARKER_GAP: int = 8
#: Strips shorter than this hide the marker's vertical label — it cannot be read
#: in the space available. Mirrors the container query in
#: `static._STRIP_MARKER_CSS`, which hides it under the same height.
MARKER_LABEL_MIN_HEIGHT: int = 88

#: Neutral bar color (slate-500): `CHANNEL n` / `SAMPLE n` labels, and any
#: caption whose subject has no color of its own.
NEUTRAL_COLOR: str = "#64748b"


class Marker(NamedTuple):
    """A strip's kind, as a CSS class for the page and a color for PIL.

    Color carries the meaning here — every strip uses the same diverging
    colormap, so the marker beside it is what says whether it is an activation,
    a gradient or a weight (see `INTERNALS.md`, "Render conventions").
    """

    css: str
    color: str


ACTIVATIONS: Marker = Marker("bg-emerald-500", "#10b981")
GRADIENTS: Marker = Marker("bg-violet-500", "#8b5cf6")
WEIGHT: Marker = Marker("bg-sky-500", "#0ea5e9")
OPTIMIZER: Marker = Marker("bg-amber-600", "#d97706")
CUSTOM: Marker = Marker("bg-teal-600", "#0d9488")
NEUTRAL: Marker = Marker("bg-slate-500", NEUTRAL_COLOR)
#: A custom layer tensor on the main page (`Session.watch_layer_tensor`). It
#: shares the weights page's sky, not the activation green: what the strip holds
#: is the user's own tensor, not this layer's activations.
CUSTOM_TENSOR: Marker = WEIGHT

#: Experiment cell caption colors, keyed by the caption's first word. The input
#: is green and the attribution / overlay purple, echoing the main view's
#: activation / gradient markers; anything else (deep-dream channel images) is
#: neutral.
CAPTION_COLORS: dict[str, str] = {
    "input": ACTIVATIONS.color,
    "attribution": GRADIENTS.color,
    "overlay": GRADIENTS.color,
}


def caption_color(caption: str) -> str:
    """Bar color for an experiment cell caption (its first word keys the map)."""
    return CAPTION_COLORS.get(caption.split(" ", 1)[0].lower(), NEUTRAL_COLOR)


def rgb(color: str) -> tuple[int, int, int]:
    """`"#rrggbb"` as the tuple PIL fills with."""
    value = color.lstrip("#")
    return int(value[0:2], 16), int(value[2:4], 16), int(value[4:6], 16)


#: The page draws every label in `bold 10px monospace`; the marker labels a
#: point smaller. These are those two, for PIL.
LABEL_FONT_SIZE: int = 10
MARKER_FONT_SIZE: int = 9
#: `letter-spacing` on a label bar and on a marker's vertical label, in px at
#: the sizes above (0.04em and 0.12em).
LABEL_TRACKING: float = 0.4
MARKER_TRACKING: float = 1.08


@functools.lru_cache(maxsize=4)
def mono_font(size: int, *, bold: bool = True) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """The composed image's stand-in for the page's `monospace` font.

    PIL's built-in bitmap font is not monospace, is fixed at one size, and has
    no glyph for the em dash the frame labels use — it drew a tofu box in every
    recorded frame. DejaVu Sans Mono ships inside matplotlib, which is already a
    hard dependency (it draws the histogram frames), so this costs no new
    dependency and no vendored binary. The path is resolved without importing
    matplotlib itself, which is slow and needed nowhere else in this module.

    Falls back to the built-in font if that layout ever changes, so a composed
    image is never worse than it was before this existed.
    """
    import importlib.util
    from pathlib import Path

    spec = importlib.util.find_spec("matplotlib")
    if spec is not None and spec.origin is not None:
        name = "DejaVuSansMono-Bold.ttf" if bold else "DejaVuSansMono.ttf"
        path = Path(spec.origin).parent / "mpl-data" / "fonts" / "ttf" / name
        if path.is_file():
            return ImageFont.truetype(str(path), size)
    return ImageFont.load_default()
