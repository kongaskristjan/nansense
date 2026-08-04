"""Tests for the Share dialog: its targets, previews, and video download."""

from __future__ import annotations

import io
from pathlib import Path
from urllib.parse import quote

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from nicegui import ui
from nicegui.element import Element

from nansense.session import Session
from nansense.ui import share
from nansense.ui.share import (
    _CARD_WIDTH_REM,
    _IN_PLAYGROUND_NOTE,
    _PLATFORM_ICONS,
    _PREVIEW_ZOOM,
    _SHARE_TARGETS,
    _VIDEO_FILENAME,
    _VIDEO_URL,
    VIDEO_DOWNLOAD_PATH,
    _add_share_button,
    _platform_icon_html,
    _preview_viewport_px,
    _share_platform_links,
)
from nansense.ui.static import MIN_APP_WIDTH
from tests.nansense.helpers import make_session

_REPO_ROOT = Path(__file__).parents[3]


def _descendants(element: Element) -> list[Element]:
    found: list[Element] = []
    for slot in element.slots.values():
        for child in slot.children:
            found.append(child)
            found += _descendants(child)
    return found


def _fire(element: Element, event: str, *args: object) -> None:
    """Invoke the element's handler for `event`, as the browser would."""
    handler = next(
        listener.handler
        for listener in element._event_listeners.values()
        if listener.type == event and listener.handler is not None
    )
    handler(*args)


class _ShareDialog:
    """A built share button plus the dialog it opens, for inspection."""

    def __init__(self, session: Session) -> None:
        before = set(ui.context.client.elements)
        with ui.card():
            _add_share_button(session)
        new = [
            element
            for id_, element in ui.context.client.elements.items()
            if id_ not in before
        ]
        self.dialog = next(e for e in new if isinstance(e, ui.dialog))
        self.button = next(
            e for e in new if isinstance(e, ui.button) and e._props.get("icon") == "share"
        )
        self.toggle = next(e for e in _descendants(self.dialog) if isinstance(e, ui.toggle))

    def open(self, key: str = "playground") -> None:
        self.toggle.value = key  # what picking a section does
        _fire(self.button, "click", None)

    def close(self) -> None:
        _fire(self.dialog, "hide")

    def elements(self) -> list[Element]:
        return _descendants(self.dialog)

    def tags(self) -> list[str]:
        return [element.tag for element in self.elements()]

    def texts(self) -> list[str]:
        return [
            element.text
            for element in self.elements()
            if isinstance(element, (ui.label, ui.button)) and element.text
        ]


class _FakeUpstream(io.BytesIO):
    """Stand-in for `urlopen`'s response: bytes plus the headers we forward."""

    def __init__(self, payload: bytes) -> None:
        super().__init__(payload)
        self.headers = {"Content-Length": str(len(payload))}


def test_share_targets_are_the_playground_the_video_and_the_library() -> None:
    """Three shareables, in the order the toggle offers them. The library link
    is the version-less site root (redirects to the default published
    version), not a pinned /dev/ page; the video link is the file itself."""
    assert list(_SHARE_TARGETS) == ["playground", "video", "library"]
    assert (
        _SHARE_TARGETS["playground"].url
        == "https://kongaskristjan.github.io/nansense/dev/playground/"
    )
    assert _SHARE_TARGETS["video"].url == _VIDEO_URL
    assert _SHARE_TARGETS["library"].url == "https://kongaskristjan.github.io/nansense/"


def test_only_the_playground_and_the_video_preview_themselves() -> None:
    """A thumbnail of the docs site would say less than its title already
    does; the two targets that *show* something get a preview."""
    previews = {key: target.preview for key, target in _SHARE_TARGETS.items()}
    assert previews == {"playground": "page", "video": "video", "library": None}


def test_the_shared_video_is_the_hero_video_of_the_readme_and_the_docs() -> None:
    """One demo video, three places that hand it out — a new recording has to
    replace all of them together."""
    for relative in ("README.md", "docs/index.md"):
        text = (_REPO_ROOT / relative).read_text(encoding="utf-8")
        assert _VIDEO_URL in text


@pytest.mark.parametrize("key", ["playground", "video", "library"])
def test_share_platform_links_embed_the_encoded_target(key: str) -> None:
    """Every platform's share-intent href carries the URL-encoded target link
    (and no raw spaces from the title), so the composer opens prefilled."""
    target = _SHARE_TARGETS[key]
    links = dict(_share_platform_links(target.url, target.title))
    assert set(links) == {"X", "Facebook", "LinkedIn", "Reddit", "Hacker News", "Email"}
    encoded = quote(target.url, safe="")
    for href in links.values():
        assert encoded in href
        assert " " not in href


def test_platform_icons_cover_every_platform_but_email() -> None:
    """Each share platform renders a brand glyph (Email uses the bundled
    Material `mail` icon, so it has no SVG entry); path data must be real SVG
    path commands so a truncated paste can't ship a blank button."""
    labels = {label for label, _ in _share_platform_links("https://e.com", "t")}
    assert set(_PLATFORM_ICONS) == labels - {"Email"}
    for viewbox, path in _PLATFORM_ICONS.values():
        assert viewbox.startswith("0 0 ")
        assert path.startswith("M") and len(path) > 50
    assert _platform_icon_html("Email") is None
    x_svg = _platform_icon_html("X")
    assert x_svg is not None
    assert 'fill="currentColor"' in x_svg


def test_share_button_is_a_quiet_icon_added_unconditionally() -> None:
    """One flat icon-only share button — no locked-session gate: the hosted
    playground shows it too (handing out links is what a demo is for)."""
    session, _model = make_session(epochs=1, phases={"train": 1})
    with ui.card() as card:
        _add_share_button(session)
    buttons = [c for c in card.default_slot.children if isinstance(c, ui.button)]
    assert len(buttons) == 1
    assert buttons[0]._props.get("icon") == "share"
    assert buttons[0].text == ""  # icon-only: quieter than the labelled controls
    assert "flat" in buttons[0]._props


def test_the_preview_hands_the_framed_app_a_viewport_it_can_lay_out_in() -> None:
    """The playground preview frames the real app, which pans instead of
    reflowing below `MIN_APP_WIDTH`. The card is wide enough that the zoomed
    frame still clears that floor — otherwise the preview would show a
    cropped app, which is exactly what a preview must not do."""
    assert _PREVIEW_ZOOM < 1  # zoomed out: more app per pixel of dialog
    assert _preview_viewport_px() >= MIN_APP_WIDTH
    assert _CARD_WIDTH_REM > 32  # the pre-preview dialog's width


def test_the_playground_preview_frames_the_page_the_link_opens() -> None:
    """Outside the playground the preview is a live frame of the shared page,
    zoomed out so the desktop-first app fits."""
    session, _model = make_session(epochs=1, phases={"train": 1})
    dialog = _ShareDialog(session)
    dialog.open("playground")
    frames = [e for e in dialog.elements() if e.tag == "iframe"]
    assert len(frames) == 1
    assert frames[0]._props["src"] == _SHARE_TARGETS["playground"].url
    assert frames[0]._style["zoom"] == str(_PREVIEW_ZOOM)
    assert _IN_PLAYGROUND_NOTE not in dialog.texts()


def test_the_playground_preview_is_a_note_when_the_app_is_the_playground() -> None:
    """A locked session *is* what the link opens, so framing it would boot a
    second copy of the hosted demo inside itself. The visitor is told they are
    already there instead."""
    session, _model = make_session(epochs=1, phases={"train": 1})
    session.lock()
    dialog = _ShareDialog(session)
    dialog.open("playground")
    assert [e for e in dialog.elements() if e.tag == "iframe"] == []
    assert _IN_PLAYGROUND_NOTE in dialog.texts()


def test_the_video_section_plays_the_video_and_offers_the_file() -> None:
    """The player is the preview; the Download button is the point of the
    section — social platforms want the file uploaded, not linked."""
    session, _model = make_session(epochs=1, phases={"train": 1})
    dialog = _ShareDialog(session)
    dialog.open("video")
    videos = [e for e in dialog.elements() if isinstance(e, ui.video)]
    assert len(videos) == 1
    assert videos[0]._props["src"] == _VIDEO_URL
    assert videos[0]._props["preload"] == "metadata"  # not 9 MB on dialog open
    download = next(
        e
        for e in dialog.elements()
        if isinstance(e, ui.button) and e.text == "Download"
    )
    # Same-origin, so the browser honours `download` (and its filename) —
    # pointed at the remote asset it would just play the video in a tab.
    assert download._props["href"] == VIDEO_DOWNLOAD_PATH
    assert download._props["download"] == _VIDEO_FILENAME


def test_the_library_section_is_link_and_platforms_only() -> None:
    session, _model = make_session(epochs=1, phases={"train": 1})
    dialog = _ShareDialog(session)
    dialog.open("library")
    assert [e for e in dialog.elements() if e.tag == "iframe"] == []
    assert [e for e in dialog.elements() if isinstance(e, ui.video)] == []
    assert _SHARE_TARGETS["library"].url in dialog.texts()  # the link pill


def test_the_preview_lives_only_while_the_dialog_is_open() -> None:
    """The playground preview boots a hosted demo session, so it is built when
    the dialog opens and dropped when it closes — never from page load."""
    session, _model = make_session(epochs=1, phases={"train": 1})
    dialog = _ShareDialog(session)
    assert [e for e in dialog.elements() if e.tag == "iframe"] == []
    dialog.open("playground")
    assert [e for e in dialog.elements() if e.tag == "iframe"] != []
    dialog.close()
    assert [e for e in dialog.elements() if e.tag == "iframe"] == []


def test_the_download_route_streams_the_video_as_an_attachment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The app re-serves the remote video from its own origin with the
    `Content-Disposition` that makes the browser save it."""
    payload = b"webm-bytes" * 100
    monkeypatch.setattr(
        share, "urlopen", lambda url, timeout: _FakeUpstream(payload)
    )
    app = FastAPI()
    share.add_video_download_route(app)
    with TestClient(app) as client:
        response = client.get(VIDEO_DOWNLOAD_PATH)
    assert response.status_code == 200
    assert response.headers["content-type"] == "video/webm"
    assert (
        response.headers["content-disposition"]
        == f'attachment; filename="{_VIDEO_FILENAME}"'
    )
    assert response.content == payload


def test_the_download_route_reports_an_unreachable_asset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Offline (or a dead host) is a bad gateway, not a traceback: the button
    opened a tab, and that tab should say what went wrong."""

    def _fail(url: str, timeout: float) -> _FakeUpstream:
        raise OSError("no route to host")

    monkeypatch.setattr(share, "urlopen", _fail)
    app = FastAPI()
    share.add_video_download_route(app)
    with TestClient(app) as client:
        response = client.get(VIDEO_DOWNLOAD_PATH)
    assert response.status_code == 502
    assert "no route to host" in response.json()["detail"]
