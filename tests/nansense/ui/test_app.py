"""Tests for the serve() entry point in nansense.ui.app."""

from __future__ import annotations

import logging
import threading
import warnings

import pytest

import torch
import torch.distributed as dist
from fastapi import FastAPI
from starlette.routing import Mount

import nansense
from nansense.ui import app
from nansense.ui.app import (
    _DropBenignNiceguiNoise,
    _announce,
    _announce_when_ready,
    _display_url,
    _format_box,
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


def test_serve_registers_the_mcp_route_before_niceguis_catch_all(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Starlette matches routes in order and NiceGUI mounts itself at `/`, so a
    `/mcp` route registered after it would never be reached — and would fail
    silently, as a NiceGUI 404 rather than an error.

    Uvicorn is stubbed out so nothing binds a port: the assertion is about how
    `serve` assembles the app, not about serving it.
    """
    captured: dict[str, FastAPI] = {}

    class _StubServer:
        def __init__(self, config: object) -> None:
            captured["app"] = getattr(config, "app")
            self.started = False

        def run(self) -> None:
            return None

    monkeypatch.setattr(app.uvicorn, "Server", _StubServer)
    session = nansense.start(torch.nn.Linear(4, 2), epochs=1, phases={"train": 1})
    thread = serve(session, port=0, open_browser=False)
    assert thread is not None
    thread.join(timeout=10)

    routes = captured["app"].router.routes
    paths = [getattr(route, "path", None) for route in routes]
    assert "/mcp" in paths, paths
    # NiceGUI mounts at "/", which Starlette normalizes to an empty mount path;
    # it is the catch-all that anything registered after it disappears behind.
    catch_all = next(
        index for index, route in enumerate(routes) if isinstance(route, Mount)
    )
    assert paths.index("/mcp") < catch_all, paths


def test_serve_can_omit_the_mcp_endpoint(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, FastAPI] = {}

    class _StubServer:
        def __init__(self, config: object) -> None:
            captured["app"] = getattr(config, "app")
            self.started = False

        def run(self) -> None:
            return None

    monkeypatch.setattr(app.uvicorn, "Server", _StubServer)
    session = nansense.start(torch.nn.Linear(4, 2), epochs=1, phases={"train": 1})
    thread = serve(session, port=0, open_browser=False, mcp=False)
    assert thread is not None
    thread.join(timeout=10)

    served = captured["app"]
    assert "/mcp" not in [
        getattr(route, "path", None) for route in served.router.routes
    ]


@pytest.mark.parametrize(
    ("host", "expected"),
    [
        ("127.0.0.1", "http://127.0.0.1:8080"),
        ("localhost", "http://localhost:8080"),
        ("192.168.1.5", "http://192.168.1.5:8080"),
        # "bind every interface" addresses are not routable from a browser,
        # so they are shown as loopback.
        ("0.0.0.0", "http://127.0.0.1:8080"),
        ("::", "http://127.0.0.1:8080"),
        ("", "http://127.0.0.1:8080"),
    ],
)
def test_display_url(host: str, expected: str) -> None:
    assert _display_url(host, 8080) == expected


def test_format_box_frames_every_line() -> None:
    box = _format_box(["NaNsense UI is running at:", "http://127.0.0.1:8080"], 100)
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
    """Stands in for a uvicorn server; `started` reflects whether the bind
    succeeded (it never flips when a concurrent session holds the port)."""

    def __init__(self, started: bool) -> None:
        self.started = started


def _exited_thread() -> threading.Thread:
    """An unstarted thread whose `is_alive()` is False — stands in for a
    server thread that has already exited after a failed bind."""
    return threading.Thread(target=lambda: None)


def test_announce_when_ready_announces_and_opens_focused_tab(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A clean bind (`server.started`) prints the boxed banner and opens a
    focused new tab (`new=2`, `autoraise=True`)."""
    calls: list[tuple[str, int, bool]] = []

    def fake_open(url: str, new: int = 0, autoraise: bool = True) -> bool:
        calls.append((url, new, autoraise))
        return True

    monkeypatch.setattr(app.webbrowser, "open", fake_open)
    _announce_when_ready(
        _FakeServer(started=True), _exited_thread(), "http://127.0.0.1:8080", True,
        timeout=0.0,
    )
    out = capsys.readouterr().out
    assert "http://127.0.0.1:8080" in out and "┌" in out and "└" in out
    assert calls == [("http://127.0.0.1:8080", 2, True)]


def test_announce_when_ready_suppressed_under_concurrent_session(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A failed bind (concurrent session holds the port) means the server
    never starts: no banner is printed and no browser tab is opened. The
    already-exited server thread also short-circuits the wait, so the larger
    timeout never elapses."""
    calls: list[tuple[str, int, bool]] = []

    def fake_open(url: str, new: int = 0, autoraise: bool = True) -> bool:
        calls.append((url, new, autoraise))
        return True

    monkeypatch.setattr(app.webbrowser, "open", fake_open)
    _announce_when_ready(
        _FakeServer(started=False), _exited_thread(), "http://127.0.0.1:8080", True,
        timeout=5.0,
    )
    assert capsys.readouterr().out == ""
    assert calls == []


def test_announce_when_ready_respects_open_browser_false(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """`open_browser=False` still prints the banner but opens no tab."""
    calls: list[tuple[str, int, bool]] = []

    def fake_open(url: str, new: int = 0, autoraise: bool = True) -> bool:
        calls.append((url, new, autoraise))
        return True

    monkeypatch.setattr(app.webbrowser, "open", fake_open)
    _announce_when_ready(
        _FakeServer(started=True), _exited_thread(), "http://127.0.0.1:8080", False,
        timeout=0.0,
    )
    assert "┌" in capsys.readouterr().out
    assert calls == []


def test_announce_when_ready_swallows_browser_backend_errors(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A headless box (no browser) must never take down the run; the banner
    is still printed."""

    def boom(*args: object, **kwargs: object) -> bool:
        raise RuntimeError("no display")

    monkeypatch.setattr(app.webbrowser, "open", boom)
    _announce_when_ready(
        _FakeServer(started=True), _exited_thread(), "http://127.0.0.1:8080", True,
        timeout=0.0,
    )
    assert "┌" in capsys.readouterr().out  # banner printed despite the error


def _nicegui_record(message: object) -> logging.LogRecord:
    """A log record as `logging.Logger.exception`/`.warning` build it.

    `message` may be a string or an exception instance — NiceGUI's global
    handler (`app.handle_exception` → `log.exception`) logs the exception
    object itself, so the record's message text is `str(exc)`.
    """
    return logging.LogRecord(
        name="nicegui", level=logging.ERROR, pathname=__file__, lineno=0,
        msg=message, args=(), exc_info=None,
    )


@pytest.mark.parametrize(
    "message",
    [
        "Event listeners changed after initial definition",
        # The per-connection `ui.timer` disconnect race, logged as the bare
        # RuntimeError object exactly as NiceGUI's global handler emits it.
        RuntimeError("The parent slot of the element has been deleted."),
    ],
)
def test_benign_nicegui_noise_is_dropped(message: object) -> None:
    """Both known-benign NiceGUI lines are filtered out of the run log."""
    assert _DropBenignNiceguiNoise().filter(_nicegui_record(message)) is False


def test_unrelated_nicegui_errors_still_pass() -> None:
    """A genuine error must never be masked — only the listed lines are dropped."""
    record = _nicegui_record(RuntimeError("the model exploded"))
    assert _DropBenignNiceguiNoise().filter(record) is True


def test_noise_filter_installed_on_nicegui_logger() -> None:
    """Importing the module installs the filter on the `nicegui` logger — the
    same logger `app.handle_exception` → `log.exception` routes to — so the
    benign parent-slot teardown traceback is dropped end-to-end, while a real
    error still passes (`Logger.filter` consults the logger's own filters)."""
    logger = logging.getLogger("nicegui")
    assert any(isinstance(f, _DropBenignNiceguiNoise) for f in logger.filters)
    benign = _nicegui_record(
        RuntimeError("The parent slot of the element has been deleted.")
    )
    real = _nicegui_record(RuntimeError("the model exploded"))
    # `Logger.filter` returns a falsy value when a record is dropped and a
    # truthy one when it passes (a bool on Python <3.12, the record itself on
    # 3.12+), so assert truthiness rather than identity for version-robustness.
    assert not logger.filter(benign)
    assert logger.filter(real)
