"""An enabled-but-unserved session must not deadlock on a pause.

A session with no UI served (and no separate thread driving it) used to hang
forever on the first STEP-mode pause, since nothing could ever resume it. It
now bounds the wait and detaches instead — see `Session._wait_for_proceed`.
"""

from __future__ import annotations

import pytest

import nansense
from nansense import session as session_module
from nansense.session import Mode
from tests.nansense.helpers import TinyNet, train_step


def test_unserved_session_detaches_instead_of_hanging(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Shrink the grace period so the (otherwise 30 s) timeout fires fast.
    monkeypatch.setattr(session_module, "_UNSERVED_PAUSE_TIMEOUT", 0.2)
    model = TinyNet()
    # Enabled, no port -> never served. Driven single-threaded (no resumer).
    session = nansense.start(model, epochs=1, phases={"train": 3})
    assert not session._served

    completed = 0
    with pytest.warns(RuntimeWarning, match="no UI is serving"):
        for _ in range(3):
            with session.batch(phase="train", epoch=0):
                train_step(model)
            completed += 1

    # The loop ran to completion (no deadlock) and the session auto-detached
    # after the first pause's grace period elapsed.
    assert completed == 3
    assert session.mode is Mode.DETACH
    session.close()


def test_served_session_is_not_auto_detached(monkeypatch: pytest.MonkeyPatch) -> None:
    """`mark_served()` opts a session back into waiting indefinitely, so the
    grace timeout never detaches it (the UI may legitimately pause for ages)."""
    monkeypatch.setattr(session_module, "_UNSERVED_PAUSE_TIMEOUT", 0.2)
    model = TinyNet()
    session = nansense.start(model, epochs=1, phases={"train": 1})
    session.mark_served()
    assert session._served
    # A served session keeps STEP mode (it would wait for the UI), so we only
    # assert the flag/mode here rather than entering a (now-blocking) batch.
    assert session.mode is Mode.STEP
    session.close()
