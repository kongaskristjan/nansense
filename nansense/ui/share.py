"""The Share dialog: the three things NaNsense hands out, and their previews.

Split out of `top_bar.py` (which opens it from a top-bar icon on every page)
because the dialog grew a life of its own: three share targets, a live preview
per target, and a server route that hands the demo video over as a file.

The dialog is captioned sections top to bottom: *what* to share (a segmented
toggle over `_SHARE_TARGETS`), a *preview* of the pick, the *link* itself (a
one-line pill with a copy button), and *where* to post it (one share-intent
button per platform). Everything below the toggle is rebuilt on every switch,
so each anchor's href — and the copy handler's URL — is baked in for the
current pick; the card itself is a fixed size (the playground section's), so
none of that rebuilding resizes the dialog under the pointer.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from contextlib import closing
from dataclasses import dataclass
from typing import IO, Literal
from urllib.parse import quote
from urllib.request import urlopen

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from nicegui import ui

from nansense.session import Session


#: The demo video, hosted as a GitHub asset — the same file the README and the
#: docs home page play as their hero (`tests/nansense/ui/test_share.py` keeps
#: the three in sync). 1280x720 webm, ~9 MB.
_VIDEO_URL: str = (
    "https://github.com/user-attachments/assets/d7ee7ecc-4828-4655-866d-a220174c2b44"
)
#: Where the app re-serves that video from its own origin (see
#: `add_video_download_route`), and the name the browser saves it under.
VIDEO_DOWNLOAD_PATH: str = "/nansense/share-video.webm"
_VIDEO_FILENAME: str = "nansense.webm"
_VIDEO_TIMEOUT_S: float = 30.0
_VIDEO_CHUNK: int = 256 * 1024


@dataclass(frozen=True)
class _ShareTarget:
    """One thing the share dialog can hand out: a link plus its share text.

    `preview` is what the dialog shows above the link: `"page"` frames the
    link's own page, `"video"` plays the file the link points at, and `None`
    shows nothing (the docs site is a page of prose — a thumbnail of it would
    say less than its title already does).
    """

    label: str
    url: str
    title: str
    preview: Literal["page", "video"] | None = None


# What the share dialog offers. The playground URL pins the `dev` version:
# unlike the docs site's own pages, the app can't derive the live docs
# version from its location (it runs on localhost or the HF Space), and
# `dev` is the only version the site currently publishes. The library link is
# the version-less site root, which redirects to the default published
# version — so it stays current when a release later takes over `latest`.
# The video is the raw asset rather than a page embedding it: a link that
# *is* the video plays anywhere, and the reupload path (the Download button)
# hands over that exact file.
_SHARE_TARGETS: dict[str, _ShareTarget] = {
    "playground": _ShareTarget(
        label="Playground",
        url="https://kongaskristjan.github.io/nansense/dev/playground/",
        title="NaNsense playground — a live PyTorch training run to poke around in",
        preview="page",
    ),
    "video": _ShareTarget(
        label="Video",
        url=_VIDEO_URL,
        title="NaNsense — a PyTorch debugger: pause training, look inside every layer",
        preview="video",
    ),
    "library": _ShareTarget(
        label="Library",
        url="https://kongaskristjan.github.io/nansense/",
        title="NaNsense — a PyTorch debugger: pause training, look inside every layer",
    ),
}


def _share_platform_links(url: str, title: str) -> list[tuple[str, str]]:
    """(platform label, share-intent href) pairs for one target link.

    Each href opens the platform's share/submit composer prefilled with the
    target URL (and title, where the platform takes one).
    """
    u = quote(url, safe="")
    t = quote(title, safe="")
    return [
        ("X", f"https://x.com/intent/post?text={t}&url={u}"),
        ("Facebook", f"https://www.facebook.com/sharer/sharer.php?u={u}"),
        ("LinkedIn", f"https://www.linkedin.com/sharing/share-offsite/?url={u}"),
        ("Reddit", f"https://www.reddit.com/submit?url={u}&title={t}"),
        ("Hacker News", f"https://news.ycombinator.com/submitlink?u={u}&t={t}"),
        ("Email", f"mailto:?subject={t}&body={u}"),
    ]


# Brand glyphs for the share-intent buttons, inlined as SVG path data so the
# app stays self-contained (NiceGUI bundles no brand-icon font; Email uses the
# bundled Material "mail" icon instead). X / Facebook / Reddit / Y Combinator
# come from Simple Icons (CC0 1.0 — public domain). LinkedIn comes from Font Awesome
# Free 6.7.2 by @fontawesome — https://fontawesome.com, License:
# https://fontawesome.com/license/free (Icons: CC BY 4.0), Copyright 2024
# Fonticons, Inc.; this comment carries the required attribution. The glyphs
# remain their owners' trademarks, used nominatively: each button does nothing
# but link to that platform's own share composer.
_PLATFORM_ICONS: dict[str, tuple[str, str]] = {  # label -> (viewBox, path d)
    "X": (
        "0 0 24 24",
        "M14.234 10.162 22.977 0h-2.072l-7.591 8.824L7.251 0H.258l9.168 "
        "13.343L.258 24H2.33l8.016-9.318L16.749 24h6.993zm-2.837 "
        "3.299-.929-1.329L3.076 1.56h3.182l5.965 8.532.929 1.329 7.754 "
        "11.09h-3.182z",
    ),
    "Facebook": (
        "0 0 24 24",
        "M9.101 23.691v-7.98H6.627v-3.667h2.474v-1.58c0-4.085 1.848-5.978 "
        "5.858-5.978.401 0 .955.042 1.468.103a8.68 8.68 0 0 1 1.141.195v3.325a8.623 "
        "8.623 0 0 0-.653-.036 26.805 26.805 0 0 0-.733-.009c-.707 0-1.259.096-1.675"
        ".309a1.686 1.686 0 0 0-.679.622c-.258.42-.374.995-.374 1.752v1.297h3.919l"
        "-.386 2.103-.287 1.564h-3.246v8.245C19.396 23.238 24 18.179 24 12.044c0"
        "-6.627-5.373-12-12-12s-12 5.373-12 12c0 5.628 3.874 10.35 9.101 11.647Z",
    ),
    "LinkedIn": (
        "0 0 448 512",
        "M100.28 448H7.4V148.9h92.88zM53.79 108.1C24.09 108.1 0 83.5 0 "
        "53.8a53.79 53.79 0 0 1 107.58 0c0 29.7-24.1 54.3-53.79 54.3zM447.9 "
        "448h-92.68V302.4c0-34.7-.7-79.2-48.29-79.2-48.29 0-55.69 37.7-55.69 "
        "76.7V448h-92.78V148.9h89.08v40.8h1.3c12.4-23.5 42.69-48.3 87.88-48.3 "
        "94 0 111.28 61.9 111.28 142.3V448z",
    ),
    "Reddit": (
        "0 0 24 24",
        "M12 0C5.373 0 0 5.373 0 12c0 3.314 1.343 6.314 3.515 8.485l-2.286 "
        "2.286C.775 23.225 1.097 24 1.738 24H12c6.627 0 12-5.373 "
        "12-12S18.627 0 12 0Zm4.388 3.199c1.104 0 1.999.895 1.999 1.999 0 "
        "1.105-.895 2-1.999 2-.946 0-1.739-.657-1.947-1.539v.002c-1.147.162"
        "-2.032 1.15-2.032 2.341v.007c1.776.067 3.4.567 4.686 1.363.473-.363 "
        "1.064-.58 1.707-.58 1.547 0 2.802 1.254 2.802 2.802 0 1.117-.655 "
        "2.081-1.601 2.531-.088 3.256-3.637 5.876-7.997 5.876-4.361 0-7.905"
        "-2.617-7.998-5.87-.954-.447-1.614-1.415-1.614-2.538 0-1.548 1.255"
        "-2.802 2.803-2.802.645 0 1.239.218 1.712.585 1.275-.79 2.881-1.291 "
        "4.64-1.365v-.01c0-1.663 1.263-3.034 2.88-3.207.188-.911.993-1.595 "
        "1.959-1.595Zm-8.085 8.376c-.784 0-1.459.78-1.506 1.797-.047 "
        "1.016.64 1.429 1.426 1.429.786 0 1.371-.369 1.418-1.385.047-1.017"
        "-.553-1.841-1.338-1.841Zm7.406 0c-.786 0-1.385.824-1.338 1.841.047 "
        "1.017.634 1.385 1.418 1.385.785 0 1.473-.413 1.426-1.429-.046-1.017"
        "-.721-1.797-1.506-1.797Zm-3.703 4.013c-.974 0-1.907.048-2.77.135"
        "-.147.015-.241.168-.183.305.483 1.154 1.622 1.964 2.953 1.964 1.33 "
        "0 2.47-.81 2.953-1.964.057-.137-.037-.29-.184-.305-.863-.087-1.795"
        "-.135-2.769-.135Z",
    ),
    "Hacker News": (  # the Y Combinator mark, HN's standard share glyph
        "0 0 24 24",
        "M0 24V0h24v24H0zM6.951 5.896l4.112 7.708v5.064h1.583v-4.972l4.148"
        "-7.799h-1.749l-2.457 4.875c-.372.745-.688 1.434-.688 1.434s-.297"
        "-.708-.651-1.434L8.831 5.896h-1.88z",
    ),
}


def _platform_icon_html(label: str) -> str | None:
    """The 18px `currentColor` SVG for a platform button, `None` for Email.

    Rendered as an inline SVG child of the button (not an `img:` data URI) so
    `fill="currentColor"` picks up the button's text color.
    """
    spec = _PLATFORM_ICONS.get(label)
    if spec is None:
        return None
    viewbox, path = spec
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="{viewbox}" '
        'width="18" height="18" fill="currentColor" aria-hidden="true">'
        f'<path d="{path}"/></svg>'
    )


def add_video_download_route(app: FastAPI) -> None:
    """Re-serve the demo video from our own origin, as a file download.

    The Video section's Download button exists so the video can be *reuploaded*
    — X, LinkedIn and the rest want the file, not a link to it — and neither
    browser route to the remote asset produces a file: `<a download>` is
    honoured for same-origin URLs only (cross-origin it degrades to a plain
    navigation, which just plays the video in a tab), and the asset host sends
    no CORS headers, so fetching it in JS and saving the blob is out too. Both
    limits are about origin, so the bytes are streamed through the app itself
    — same origin as the page, with the `Content-Disposition` that makes the
    browser save rather than play them.

    Registered by `ui.app.serve` ahead of NiceGUI's catch-all mount at `/`.
    The URL is a constant, so nothing user-supplied reaches `urlopen`; a
    failed fetch (offline, host down) surfaces as a 502 in the tab the button
    opened.
    """

    @app.get(VIDEO_DOWNLOAD_PATH, include_in_schema=False)
    def download_video() -> StreamingResponse:
        try:
            upstream = urlopen(_VIDEO_URL, timeout=_VIDEO_TIMEOUT_S)  # noqa: S310
        except OSError as exc:
            raise HTTPException(
                status_code=502, detail=f"could not fetch {_VIDEO_URL}: {exc}"
            ) from exc
        headers = {
            "Content-Disposition": f'attachment; filename="{_VIDEO_FILENAME}"'
        }
        length = upstream.headers.get("Content-Length")
        if length is not None:
            headers["Content-Length"] = length
        return StreamingResponse(
            _stream(upstream), media_type="video/webm", headers=headers
        )


def _stream(upstream: IO[bytes]) -> Iterator[bytes]:
    """Yield the upstream response in chunks, closing it when done.

    A plain sync generator: Starlette drives it (and the blocking route
    handler above) in a threadpool, so the ~9 MB never lands in memory whole
    and the event loop keeps serving the session it was opened from.
    """
    with closing(upstream):
        while chunk := upstream.read(_VIDEO_CHUNK):
            yield chunk


# The dialog card. Both dimensions are pinned, and pinned to what the
# *playground* section needs — the widest and the tallest of the three, since
# it carries a framed app. Switching sections rebuilds everything under the
# toggle, so a card sized to its contents would resize under the pointer on
# every switch (and the library, a link and six icons, would collapse it to a
# third of its height). The library simply leaves the slack empty.
#
# Width is what makes the previews worth having: the playground preview
# frames the real app, which pans instead of reflowing below `MIN_APP_WIDTH`
# (see `ui/static.py`), so the card's inner width divided by `_PREVIEW_ZOOM`
# has to clear that floor — `test_share.py` checks it does. `zoom` (as on the
# docs site's own embed) shrinks the rendering *and* widens the frame's
# viewport by the same factor; `transform: scale()` would only shrink what
# the app already drew at the preview's width, leaving it just as cramped.
_CARD_WIDTH_REM: int = 50
_CARD_PADDING_REM: float = 1.5  # `p-6`
_REM_PX: int = 16
#: Zoomed to 0.7 rather than nearer 1: at 0.8 the framed app's first layer
#: card is cut off mid-gradient-row, and a preview that crops the one thing
#: the playground is for isn't previewing much.
_PREVIEW_ZOOM: float = 0.7
#: 22rem of preview, plus the ~24rem the title, toggle, link, platform row
#: and Close take around it, plus a little slack so the video section — whose
#: caption row carries the Download button — clears it too. Fits a laptop
#: window with room to spare; `max-h-full` with a scroll is the fallback in
#: the short frame the docs home page embeds the app in.
_PREVIEW_HEIGHT_REM: int = 22
_CARD_HEIGHT_REM: float = 47
_CARD_CLASSES: str = (
    f"w-[{_CARD_WIDTH_REM}rem] max-w-full h-[{_CARD_HEIGHT_REM}rem] max-h-full "
    "overflow-y-auto p-6 gap-4"
)
_PREVIEW_BOX_CLASSES: str = (
    f"w-full h-[{_PREVIEW_HEIGHT_REM}rem] shrink-0 overflow-hidden rounded "
    "border border-slate-300"
)
_CAPTION_CLASSES: str = "text-xs uppercase tracking-wider text-slate-400"
_IN_PLAYGROUND_NOTE: str = "You are in the playground — this link opens this very app"


def _preview_viewport_px() -> float:
    """The viewport width the zoomed playground preview hands the framed app."""
    inner = (_CARD_WIDTH_REM - 2 * _CARD_PADDING_REM) * _REM_PX
    return inner / _PREVIEW_ZOOM


def _add_preview(target: _ShareTarget, *, in_playground: bool) -> None:
    """The preview section for one target — nothing at all for the library.

    The playground preview is a live frame of the page the link opens (the
    docs site's fullscreen playground, itself framing a hosted Space), so it
    boots a real demo session. That is worth it exactly once: from a local
    run, where it shows what the recipient will get. In the hosted playground
    the app *is* what the link opens, so the frame would load a second copy of
    the demo inside itself — the note says so instead.
    """
    if target.preview is None:
        return
    with ui.column().classes("w-full gap-1"):
        # Fixed-height caption row: the video's Download button is taller than
        # the bare caption beside the playground's frame, and without this the
        # link and platform rows below would shift by those few pixels every
        # time the two previewed sections are toggled between.
        with ui.row().classes("w-full h-7 items-center justify-between no-wrap"):
            ui.label("Preview").classes(_CAPTION_CLASSES)
            if target.preview == "video":
                _add_download_button()
        with ui.element("div").classes(_PREVIEW_BOX_CLASSES):
            if target.preview == "video":
                # `preload="metadata"`: opening the dialog fetches the first
                # frame and the duration, not 9 MB of video.
                ui.video(target.url, controls=True, muted=True).props(
                    'preload="metadata" playsinline'
                ).classes("w-full h-full bg-black object-contain")
            elif in_playground:
                with ui.column().classes(
                    "w-full h-full items-center justify-center gap-2 bg-slate-100"
                ):
                    ui.icon("open_in_browser", size="2rem").classes("text-slate-400")
                    ui.label(_IN_PLAYGROUND_NOTE).classes("text-sm text-slate-500")
            else:
                ui.element("iframe").props(
                    f'src="{target.url}" title="Playground preview" loading="lazy"'
                ).classes("w-full h-full border-0").style(f"zoom: {_PREVIEW_ZOOM}")


def _add_download_button() -> None:
    """The Video section's Download button: the file, for reuploading.

    A real anchor to this app's own copy of the video
    (`add_video_download_route`) — same origin, so the browser honours both
    the `download` attribute and the filename it suggests.
    """
    button = ui.button("Download", icon="download", color="slate-200").props(
        f'href="{VIDEO_DOWNLOAD_PATH}" download="{_VIDEO_FILENAME}" '
        'unelevated dense no-caps size=sm text-color=slate-700 '
        'aria-label="Download the video file"'
    )
    button.tooltip("Download the file — ready to upload straight into a post")


def _add_link_row(target: _ShareTarget) -> None:
    """The link pill with its inline copy button.

    The clipboard write runs client-side in the `js_handler` (then `emit`s so
    Python can toast), since a server round-trip would drop the user gesture
    the Clipboard API needs.
    """
    with ui.column().classes("w-full gap-1"):
        ui.label("Link").classes(_CAPTION_CLASSES)
        with ui.row().classes(
            "w-full items-center no-wrap gap-0.5 bg-slate-100 rounded pl-2 pr-1 py-1"
        ):
            # 11px mono lets the longest URL fit the card on one line;
            # `truncate` is the safety net.
            ui.label(target.url).classes(
                "grow min-w-0 truncate text-[11px] font-mono text-slate-600"
            )
            copy = ui.button(icon="content_copy", color="slate-500").props(
                'flat round dense size=sm aria-label="Copy link"'
            )
            copy.tooltip("Copy link")
            copy.on(
                "click",
                lambda: ui.notify("Link copied to clipboard"),
                js_handler=(
                    "(...args) => { if (navigator.clipboard) "
                    f"navigator.clipboard.writeText({json.dumps(target.url)}); "
                    "emit(...args); }"
                ),
            )


def _add_platform_row(target: _ShareTarget) -> None:
    """One share-intent button per platform — brand glyphs, hover for the name."""
    with ui.column().classes("w-full gap-1"):
        ui.label("Share on").classes(_CAPTION_CLASSES)
        with ui.row().classes("gap-2 items-center"):
            for label, href in _share_platform_links(target.url, target.title):
                # Real anchors opening a new tab, so the platform's composer
                # never replaces the app (nor the docs page embedding it).
                hint = "Share via email" if label == "Email" else f"Share on {label}"
                props = (
                    f'href="{href}" target="_blank" unelevated round '
                    f'dense size=md text-color=slate-700 aria-label="{hint}"'
                )
                icon_html = _platform_icon_html(label)
                if icon_html is None:
                    button = ui.button(icon="mail", color="slate-200").props(props)
                else:
                    button = ui.button(color="slate-200").props(props)
                    with button:
                        ui.html(icon_html)
                button.tooltip(hint)


def _add_share_button(session: Session) -> None:
    """The share icon just left of the logo, and the share dialog it opens.

    A flat icon-only button, quieter than the working controls beside it, on
    every page — the hosted playground included (handing out links is exactly
    what a public demo is for; the docs page delegates `clipboard-write` to
    the app iframe).

    The sections below the toggle are (re)built when the dialog opens and on
    every switch, and dropped again when it closes: the playground preview is
    a live frame of a hosted demo, so it exists only while someone is looking
    at it — never from the moment the page loads, and never after the dialog
    is dismissed.
    """
    with ui.dialog() as dialog, ui.card().classes(_CARD_CLASSES):
        ui.label("Share NaNsense").classes("text-lg font-bold")
        with ui.column().classes("w-full gap-1"):
            ui.label("What to share").classes(_CAPTION_CLASSES)
            toggle = (
                ui.toggle(
                    {key: t.label for key, t in _SHARE_TARGETS.items()},
                    value="playground",
                    on_change=lambda e: rebuild(str(e.value)),
                )
                .classes("w-full")
                .props('spread no-caps padding="xs lg"')
            )
        # `grow`: the card is a fixed height, so the sections' slack — all of
        # it, on the previewless library — collects here, which keeps Close in
        # the bottom corner instead of floating up under the last row.
        content = ui.column().classes("w-full grow gap-4")
        with ui.row().classes("w-full justify-end"):
            ui.button("Close", on_click=dialog.close).props("flat")

    def rebuild(key: str) -> None:
        target = _SHARE_TARGETS[key]
        content.clear()
        with content:
            _add_preview(target, in_playground=session.locked)
            _add_link_row(target)
            _add_platform_row(target)

    def open_dialog() -> None:
        rebuild(str(toggle.value))
        dialog.open()

    dialog.on("hide", lambda: content.clear())

    button = ui.button(icon="share", on_click=open_dialog, color="slate-500").props(
        "dense size=md flat"
    )
    button.tooltip("Share NaNsense")
