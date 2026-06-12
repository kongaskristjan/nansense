"""Tests for the serve() entry point in nansense.ui.app."""

from __future__ import annotations

import torch

import nansense
from nansense.ui.app import serve


def test_serve_on_disabled_session_is_noop() -> None:
    """`serve()` returns None without starting a server for a disabled session."""
    model = torch.nn.Linear(4, 2)
    session = nansense.start(model, epochs=1, phases={"train": 1}, enabled=False)
    assert serve(session) is None
