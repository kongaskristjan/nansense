"""Tests for the serve() entry point in nansense.ui.app."""

from __future__ import annotations

import warnings

import pytest

import torch
import torch.distributed as dist

import nansense
from nansense.ui import app
from nansense.ui.app import (
    _announce,
    _display_url,
    _format_box,
    _open_browser_when_ready,
    serve,
)


def test_reduce_op_future_warning_is_silenced() -> None:
    """`_silence_reduce_op_future_warning` mutes the `torch.distributed.reduce_op`
    FutureWarning that NiceGUI's `gc.get_objects()` server-discovery walk trips
    (its `isinstance` check reads the deprecated instance's `__class__`).

    The control branch confirms the walk really would warn without the filter,
    so a passing assertion can't be a false negative from warning de-dup."""
    reduce_op = dist.reduce_op  # the deprecated module-level instance

    with warnings.catch_warnings(record=True) as control:
        warnings.simplefilter("always")
        isinstance(reduce_op, int)  # what NiceGUI does to every live object
    assert any("reduce_op" in str(w.message) for w in control)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        app._silence_reduce_op_future_warning()
        isinstance(reduce_op, int)
    assert not any("reduce_op" in str(w.message) for w in caught)


def test_serve_on_disabled_session_is_noop() -> None:
    """`serve()` returns None without starting a server for a disabled session."""
    model = torch.nn.Linear(4, 2)
    session = nansense.start(model, epochs=1, phases={"train": 1}, enabled=False)
    assert serve(session) is None


@pytest.mark.parametrize(
    ("host", "expected"),
    [
        ("127.0.0.1", "http://127.0.0.1:8080"),
        ("localhost", "http://localhost:8080"),
        ("192.168.1.5", "http://192.168.1.5:8080"),
        # "bind every interface" addresses are not routable from a browser,
        # so they are shown (and opened) as loopback.
        ("0.0.0.0", "http://127.0.0.1:8080"),
        ("::", "http://127.0.0.1:8080"),
        ("", "http://127.0.0.1:8080"),
    ],
)
def test_display_url(host: str, expected: str) -> None:
    assert _display_url(host, 8080) == expected


def test_format_box_frames_every_line() -> None:
    box = _format_box(["nansense UI is running at:", "http://127.0.0.1:8080"], 100)
    lines = box.splitlines()
    # Top/bottom borders plus one row per input line.
    assert len(lines) == 4
    assert lines[0].startswith("┌") and lines[0].endswith("┐")
    assert lines[-1].startswith("└") and lines[-1].endswith("┘")
    # The URL is framed, and every framed row shares the box width.
    assert "http://127.0.0.1:8080" in box
    assert len({len(line) for line in lines}) == 1


def test_format_box_spans_requested_width() -> None:
    box = _format_box(["short"], 100)
    assert all(len(line) == 100 for line in box.splitlines())


def test_format_box_never_shrinks_below_content() -> None:
    """A width narrower than the text keeps the box readable rather than
    clipping the line."""
    long_line = "x" * 60
    box = _format_box([long_line], 10)
    assert all(len(line) == len(long_line) + 4 for line in box.splitlines())
    assert long_line in box


def test_announce_prints_boxed_url(capsys: pytest.CaptureFixture[str]) -> None:
    _announce("http://127.0.0.1:8080")
    out = capsys.readouterr().out
    assert "http://127.0.0.1:8080" in out
    assert "┌" in out and "└" in out


class _FakeServer:
    """Stands in for a uvicorn server that is already up."""

    started: bool = True


def test_open_browser_opens_focused_new_tab(monkeypatch: pytest.MonkeyPatch) -> None:
    """The browser is opened in a new tab (`new=2`) and raised (`autoraise=True`)."""
    calls: list[tuple[str, int, bool]] = []

    def fake_open(url: str, new: int = 0, autoraise: bool = True) -> bool:
        calls.append((url, new, autoraise))
        return True

    monkeypatch.setattr(app.webbrowser, "open", fake_open)
    _open_browser_when_ready(_FakeServer(), "http://127.0.0.1:8080")
    assert calls == [("http://127.0.0.1:8080", 2, True)]


def test_open_browser_swallows_backend_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    """A headless box (no browser) must never take down the run."""

    def boom(*args: object, **kwargs: object) -> bool:
        raise RuntimeError("no display")

    monkeypatch.setattr(app.webbrowser, "open", boom)
    # Should not raise.
    _open_browser_when_ready(_FakeServer(), "http://127.0.0.1:8080")
