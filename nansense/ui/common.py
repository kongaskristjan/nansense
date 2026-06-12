"""Small helpers shared by several UI pages."""

from __future__ import annotations

import base64
from collections.abc import Callable, Sequence
from typing import Literal

from nicegui import ui
from nicegui.elements.mixins.disableable_element import DisableableElement

from nansense.session import Session
from nansense.ui.render import StripRender, image_mime
from nansense.ui.static import (
    _PANEL_RESIZE_CSS,
    _PANEL_RESIZE_JS,
    _STRIP_CHECKERBOARD_STYLE,
)


def _page_scaffold(title: str = "") -> None:
    """Per-page setup boilerplate: the tab title plus the no-scroll viewport.

    `title` is the page-specific part of the tab title; the main page
    passes nothing and is titled plain "Nansense". Every page fills the
    viewport and scrolls inside its own panes, so page-level scrolling is
    disabled at every level.
    """
    ui.page_title(f"Nansense — {title}" if title else "Nansense")
    ui.query(".nicegui-content").classes("p-0 h-screen overflow-hidden")
    ui.query("body").classes("overflow-hidden")
    ui.query("html").classes("overflow-hidden")


def _install_panel_resize() -> None:
    """Ship the pane-resize CSS/JS; call once per page that uses handles."""
    ui.add_head_html(_PANEL_RESIZE_CSS)
    ui.add_body_html(_PANEL_RESIZE_JS)


def _resizable_pane_props(key: str) -> str:
    """Props marking a side pane as the resize target for `key`.

    The matching `_resize_handle(key, ...)` drags this pane's width; the
    pane keeps its Tailwind width class as the default size.
    """
    return f'data-resize-pane="{key}"'


def _resize_handle(key: str, side: Literal["left", "right"]) -> ui.element:
    """Drag handle resizing the side pane marked with `_resizable_pane_props`.

    Created between the pane and the center content, in DOM order (after a
    left pane, before a right pane). `side` is the edge of the view the
    pane sits on — it decides which drag direction grows the pane. Widths
    persist in sessionStorage for the rest of the browser session; a
    double-click resets to the default width. Requires
    `_install_panel_resize()` on the page.
    """
    return (
        ui.element("div")
        .classes("nansense-resize-handle")
        .props(f'data-resize-key="{key}" data-resize-side="{side}"')
        .tooltip("Drag to resize — double-click to reset")
    )


def _set_controls_enabled(
    controls: Sequence[DisableableElement], enabled: bool
) -> None:
    """Enable/disable a group of widgets (recording freeze helper)."""
    for control in controls:
        if enabled:
            control.enable()
        else:
            control.disable()


def _defer_value_write(write: Callable[[], object]) -> None:
    """Apply a widget `.value` write on the next event-loop iteration.

    NiceGUI suppresses value writes made from inside a value-change
    handler; scheduling the write for the next loop tick makes it actually
    reach the client.
    """
    ui.timer(0.0, write, once=True)


def _weights_placeholder(message: str) -> None:
    with ui.column().classes("items-center gap-2 py-12 w-full"):
        ui.icon("grid_off", size="lg").classes("text-slate-400")
        ui.label(message).classes("text-slate-600")


def _watch_views_recording(session: Session) -> bool:
    """Whether any watch-page view records (its layer set is then frozen).

    The histogram and MIN/MAX recordings render from the watch
    accumulators, and unwatching a layer *drops* its accumulated stats —
    so while either records, unwatch actions are refused.
    """
    recording = session.recording
    return recording.is_recording("watch_histogram") or recording.is_recording(
        "watch_minmax"
    )


def _refuse_unwatch_while_recording(session: Session) -> bool:
    """Notify and return True when unwatching must currently be refused.

    The watch-page recordings render from the watch accumulators, and
    unwatching a layer *drops* its accumulated stats (see
    `_watch_views_recording`) — so while one records, every unwatch action
    is refused with a warning toast instead.
    """
    if not _watch_views_recording(session):
        return False
    ui.notify(
        "Watched layers are frozen while a watch view is recording",
        type="warning",
    )
    return True


def _strip_marker(color_class: str, label: str) -> None:
    """A labelled colored bar marking which kind of strip sits next to it.

    Stretches to the strip's height via the flex row (and collapses to
    nothing when the strip is empty); `sticky left-0` keeps it pinned to the
    card's left edge while the strip scrolls underneath. A sticky element
    can only travel within its parent, so the caller must make the flex row
    span the full scrollable width (the `w-max min-w-full` wrapper around
    the rows) — otherwise the marker is dragged out of view once the strips
    are scrolled past the visible width. The label is drawn vertically,
    reading bottom-up, and is absolutely positioned so it adds no intrinsic
    height — otherwise a missing strip would leave a floating bar instead
    of an empty row. On strips too short to fit it the label is hidden via
    the container query in `_STRIP_MARKER_CSS`; the tooltip carries the
    full name regardless.
    """
    with ui.element("div").classes(
        f"nansense-marker w-5 shrink-0 rounded mr-2 sticky left-0 z-10 "
        f"overflow-hidden {color_class}"
    ).tooltip(label.capitalize()):
        ui.label(label).classes(
            "nansense-marker-label absolute text-white font-bold select-none"
        ).style(
            "writing-mode: vertical-rl; top: 50%; left: 50%; "
            "transform: translate(-50%, -50%) rotate(180deg); "
            "font-size: 9px; letter-spacing: 0.12em; line-height: 1; "
            "white-space: nowrap;"
        )


def _b64_img_src(image: bytes, *, mime: str | None = None) -> str:
    """Data-URI for an image, defaulting to the global `STRIP_FORMAT` mime."""
    encoded = base64.b64encode(image).decode("ascii")
    return f"data:{mime or image_mime()};base64,{encoded}"


def _strip_html(strip: StripRender | None) -> str:
    """HTML for one strip: a crisp legend `<img>` plus one native-res data `<img>`.

    The data image holds every tile (with native-resolution separators) at
    the tensor's native resolution; explicit CSS width/height plus
    `image-rendering: pixelated` make the browser do the nearest-neighbour
    upscale the renderer used to do server-side. The legend image is already
    at display resolution and renders 1:1, so its labels stay sharp.
    `flex:none` keeps the scroll container from squishing the images.

    The data img sits over a fixed display-resolution gray checkerboard
    (`_STRIP_CHECKERBOARD_STYLE`): an all-finite strip is fully opaque and
    hides it, while a strip carrying transparent NaN/±Inf cells (RGBA PNG,
    `strip.data_mime`) reveals the checkerboard through them — so the bad
    cells read as "no value here" instead of a misleading color or white.
    The data image uses `strip.data_mime` (RGB `STRIP_FORMAT` or RGBA PNG),
    not the global `image_mime()` the legend keeps.
    """
    if strip is None:
        return ""
    return (
        '<div style="display:flex; align-items:flex-start;">'
        f'<img src="{_b64_img_src(strip.legend_image)}" '
        'style="display:block; flex:none; max-width:none;" />'
        f'<img src="{_b64_img_src(strip.data_image, mime=strip.data_mime)}" '
        f'style="width:{strip.width}px; height:{strip.height}px; '
        f"image-rendering:pixelated; display:block; flex:none; max-width:none; "
        f'{_STRIP_CHECKERBOARD_STYLE}" />'
        "</div>"
    )
