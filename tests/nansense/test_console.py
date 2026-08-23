"""Tests for the encoding-degrading console printing in nansense.console."""

from __future__ import annotations

import io
import sys

import pytest

from nansense.console import console_print, encodable, stream_encoding


def _stdout(monkeypatch: pytest.MonkeyPatch, encoding: str) -> io.BytesIO:
    """Point `sys.stdout` at a real stream encoding like `encoding` would — an
    unencodable write raises there exactly as it does on a console. `newline`
    is pinned so the assertions don't have to care about Windows line endings."""
    raw = io.BytesIO()
    monkeypatch.setattr(
        sys, "stdout", io.TextIOWrapper(raw, encoding=encoding, newline="\n")
    )
    return raw


@pytest.mark.parametrize(
    ("encoding", "text", "expected"),
    [
        # UTF-8 carries everything: nothing to degrade.
        ("utf-8", "paused — ±Inf", "paused — ±Inf"),
        # cp1252 happens to carry both of those, so it degrades nothing either.
        ("cp1252", "paused — ±Inf", "paused — ±Inf"),
        # ASCII carries neither; the stand-ins keep the line readable.
        ("ascii", "paused — ±Inf", "paused - +/-Inf"),
        # cp1252 has no box-drawing characters, and there is no stand-in for
        # them — the codec's own replacement takes over.
        ("cp1252", "┌──┐", "????"),
        # A path under an accented directory survives cp1252 but not ASCII.
        ("cp1252", r"C:\Users\José\moment.pt", r"C:\Users\José\moment.pt"),
        ("ascii", r"C:\Users\José\moment.pt", r"C:\Users\Jos?\moment.pt"),
    ],
)
def test_encodable_degrades_only_what_the_stream_cannot_carry(
    monkeypatch: pytest.MonkeyPatch, encoding: str, text: str, expected: str
) -> None:
    _stdout(monkeypatch, encoding)
    assert encodable(text) == expected


def test_console_print_survives_a_stream_that_cannot_encode_the_message(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The regression: a raising `print` took down whichever thread ran it —
    the UI announcer, or a training loop reporting a numerical issue."""
    raw = _stdout(monkeypatch, "ascii")
    console_print("NaNsense: numerical issue detected (±Inf) — training paused.")
    sys.stdout.flush()
    assert raw.getvalue().decode("ascii") == (
        "NaNsense: numerical issue detected (+/-Inf) - training paused.\n"
    )


class _UnknownCodecStream:
    """A stdout replacement naming a codec this Python does not have."""

    encoding = "not-a-codec"


@pytest.mark.parametrize("stdout", [io.StringIO(), None, _UnknownCodecStream()])
def test_stream_encoding_assumes_ascii_when_stdout_names_no_usable_codec(
    monkeypatch: pytest.MonkeyPatch, stdout: object
) -> None:
    """A bare `StringIO` swapped in by a harness has no `encoding` at all, and
    `pythonw` leaves `sys.stdout` as `None`. Assuming the narrowest codec keeps
    the degrading conservative instead of trading one exception for another."""
    monkeypatch.setattr(sys, "stdout", stdout)
    assert stream_encoding() == "ascii"
