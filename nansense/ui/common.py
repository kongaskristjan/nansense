"""Small helpers shared by several UI pages."""

from __future__ import annotations

import base64
import html
from collections.abc import Callable, Sequence
from typing import Literal

from nicegui import ui
from nicegui.elements.mixins.disableable_element import DisableableElement

from nansense.session import Session
from nansense.ui.render import LABEL_HEIGHT, StripRender, StripTile, image_mime
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


def _notice_banner(message: str, *, icon: str = "info") -> ui.element:
    """A red notice banner card for empty / no-data states.

    Returned so the caller can toggle its visibility as the underlying
    state changes. Intentionally a softer red than the full-strength
    numerical-error banner (`_add_error_banner`): this flags an empty
    state to act on, not an active error.
    """
    banner = ui.row().classes(
        "w-full items-center gap-2 no-wrap rounded border border-red-300 "
        "bg-red-50 text-red-700 px-3 py-2"
    )
    with banner:
        ui.icon(icon).classes("text-xl shrink-0")
        ui.label(message).classes("text-sm")
    return banner


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
        "Watched layers are frozen while a stats view is recording",
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


def _column_header_bar(label: str, width: int) -> str:
    """A `CHANNEL n` column header — a slate bar matching the row markers.

    Mirrors the `_strip_marker` row labels (rounded colored bar, white bold
    text) but laid horizontally above a tile column. An empty label (single-tile
    strips, or a header-less row) reserves the same height so the legend and
    images below still line up.
    """
    if not label:
        return f'<div style="height:{LABEL_HEIGHT}px;"></div>'
    return (
        f'<div title="{html.escape(label)}" style="width:{width}px; '
        f"height:{LABEL_HEIGHT}px; line-height:{LABEL_HEIGHT}px; "
        "background:#64748b; color:white; font:bold 10px monospace; "
        "letter-spacing:0.04em; text-align:center; border-radius:3px; "
        'overflow:hidden; white-space:nowrap; text-overflow:ellipsis;">'
        f"{html.escape(label)}</div>"
    )


def _strip_tile_html(tile: StripTile, *, show_label: bool) -> str:
    """One tile column: an optional `CHANNEL n` header bar above a native `<img>`.

    Explicit CSS width/height plus `image-rendering: pixelated` make the
    browser do the nearest-neighbour upscale the renderer used to do
    server-side; `flex:none` keeps the scroll container from squishing the
    image. The header bar is drawn only when `show_label` (the first strip of a
    card carries the shared column headers; the rows below it reuse them).

    The img sits over a fixed display-resolution gray checkerboard
    (`_STRIP_CHECKERBOARD_STYLE`): an all-finite tile is fully opaque and
    hides it, while a tile carrying transparent NaN/±Inf cells (RGBA PNG,
    `tile.mime`) reveals the checkerboard through them — so the bad cells read
    as "no value here" instead of a misleading color or white.
    """
    header = _column_header_bar(tile.label, tile.width) if show_label else ""
    return (
        f'<div style="display:flex; flex-direction:column; flex:none; '
        f'width:{tile.width}px;">{header}'
        f'<img src="{_b64_img_src(tile.image, mime=tile.mime)}" '
        f'style="width:{tile.width}px; height:{tile.height}px; '
        f"image-rendering:pixelated; display:block; flex:none; max-width:none; "
        f'{_STRIP_CHECKERBOARD_STYLE}" /></div>'
    )


def _strip_html(strip: StripRender | None, *, show_labels: bool = False) -> str:
    """HTML for one strip: a crisp legend `<img>` plus a row of tile columns.

    Each channel/tile is its own column (`_strip_tile_html`). With `show_labels`
    the strip carries a row of `CHANNEL n` header bars above its tiles (and a
    blank spacer above the legend so it lines up below the bars). A card renders
    its first strip with `show_labels=True` and the strips stacked below it
    without — their columns share the same widths and so sit under the same
    headers, reading as one table. A small `gap` spaces the columns in place of
    the white separators the renderer used to bake between tiles.
    """
    if strip is None or not strip.tiles:
        return ""
    legend_spacer = (
        f'<div style="height:{LABEL_HEIGHT}px;"></div>' if show_labels else ""
    )
    legend_col = (
        '<div style="display:flex; flex-direction:column; flex:none;">'
        f"{legend_spacer}"
        f'<img src="{_b64_img_src(strip.legend_image)}" '
        'style="display:block; flex:none; max-width:none;" /></div>'
    )
    tiles_html = "".join(
        _strip_tile_html(tile, show_label=show_labels) for tile in strip.tiles
    )
    return (
        '<div style="display:flex; align-items:flex-start; gap:2px;">'
        f"{legend_col}{tiles_html}</div>"
    )
