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
        # so they are shown as loopback.
        ("0.0.0.0", "http://127.0.0.1:8080"),
        ("::", "http://127.0.0.1:8080"),
        ("", "http://127.0.0.1:8080"),
    ],
)
def test_display_url(host: str, expected: str) -> None:
    assert _display_url(host, 8080) == expected


def test_announce_prints_plain_url(capsys: pytest.CaptureFixture[str]) -> None:
    """The address is one plain line (no box, no auto-open) so it never
    over-promises on a port the server thread may fail to bind."""
    _announce("http://127.0.0.1:8080")
    out = capsys.readouterr().out
    assert out == "nansense UI: http://127.0.0.1:8080\n"
    # No leftover banner box from the old prominent announcement.
    assert "┌" not in out and "└" not in out
