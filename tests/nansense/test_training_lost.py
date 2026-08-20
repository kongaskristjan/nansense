"""Tests for a training loop that ends without closing its session.

A script that raises leaves the served page up with nothing behind it: the
crash unwinds past the pause that would have cleared `_paused`, so
`is_running` stays True and the UI would gray out Run forever while every MCP
wait burned its full timeout on a thread that is not there.
`Session.training_lost` is the one fact the top bar, the run-control refusals
and the MCP `state` all read; `test_lock` is the same shape for the other
state that takes the controls away.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
from collections.abc import Coroutine
from pathlib import Path
from typing import Any, TypeVar

import pytest
import torch
from mcp.client import Client
from mcp.types import TextContent

import nansense
from nansense.mcp_server import build_server
from nansense.mcp_views import status_view
from nansense.restore import TimeTravelError
from nansense.session import Session, lost_loop_reason
from tests.nansense.helpers import (
    TinyNet,
    crashing_loop,
    make_session,
    optimizer_train_step,
    paused_session,
    run_to_death,
    train_step,
)

_T = TypeVar("_T")


def _detached_session() -> tuple[Session, TinyNet]:
    """A session whose batches never pause, so a worker can run one to its end."""
    session, model = make_session(epochs=1, phases={"train": 3})
    session.detach()
    return session, model


def _lost_session() -> Session:
    """A session whose loop raised `RuntimeError: boom` inside its first batch."""
    session, model = _detached_session()
    run_to_death(crashing_loop(session, model))
    return session


def _call(session: Session, name: str, arguments: dict[str, Any] | None = None) -> Any:
    """Call one tool through a real MCP client and parse its JSON payload."""

    async def go() -> Any:
        async with Client(build_server(session)) as client:
            result = await client.call_tool(name, arguments or {})
            assert result.content, f"{name} returned no content"
            block = result.content[0]
            assert isinstance(block, TextContent)
            return json.loads(block.text)

    return _run(go())


def _run(coro: Coroutine[Any, Any, _T]) -> _T:
    return asyncio.run(coro)


# --- Session state --------------------------------------------------------


def test_a_loop_that_has_not_started_or_is_still_running_is_not_lost() -> None:
    """Neither has died — one may not have reached its first batch yet."""
    session, _model = _detached_session()
    assert not session.training_lost
    with paused_session(TinyNet()) as live:
        assert not live.training_lost


def test_a_crashed_loop_is_lost_and_names_what_it_raised() -> None:
    session, model = _detached_session()
    run_to_death(crashing_loop(session, model, "shapes do not match\n(4x8, 16x3)"))

    assert session.training_lost
    # First line only: the rest belongs in the traceback the training script
    # already printed.
    assert session.training_error == "RuntimeError: shapes do not match"
    # The crash never reached a pause, so the run still looks like it is going.
    assert session.is_running


def test_a_closed_run_is_finished_rather_than_lost() -> None:
    """`close()` is the loop saying it meant to end; only silence is "lost"."""
    session, model = _detached_session()

    def loop() -> None:
        with session.batch(phase="train", epoch=0):
            train_step(model)
        session.close()

    run_to_death(loop)
    assert session.closed and not session.training_lost


@pytest.mark.parametrize("stop", ["return", "break", "recover"])
def test_a_loop_that_ends_without_failing_leaves_no_error(stop: str) -> None:
    """Returning, `break`ing out of `batches()` (which throws `GeneratorExit`
    through the batch context) and catching an error to keep training are all
    the loop's own control flow — the last one cleared by the batch after it."""
    session, model = _detached_session()

    def loop() -> None:
        if stop == "recover":
            with contextlib.suppress(RuntimeError):
                with session.batch(phase="train", epoch=0):
                    raise RuntimeError("recovered")
            assert session.training_error == "RuntimeError: recovered"
        for _ in session.batches([1, 2], phase="train", epoch=0):
            train_step(model)
            if stop == "break":
                break

    run_to_death(loop)
    assert session.training_lost
    assert session.training_error is None


# --- Time travel ----------------------------------------------------------


def test_time_travel_refuses_a_lost_loop(tmp_path: Path) -> None:
    """The jump is executed *by* the training thread, so a dead one can only
    arm a request nothing will ever pick up."""
    model = TinyNet()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    session = nansense.start(model, epochs=2, phases={"train": 2}, optimizer=optimizer)
    session.detach()

    def loop() -> None:
        for epoch in session.epochs(cache_dir=tmp_path / "cache"):
            with session.restore_point():
                with session.batch(phase="train", epoch=epoch):
                    optimizer_train_step(model, optimizer)
                    raise RuntimeError("boom")

    run_to_death(loop)
    status = session.time_travel_status()
    assert not status.available
    assert status.reason == lost_loop_reason("RuntimeError: boom")
    with pytest.raises(TimeTravelError, match="training loop is gone"):
        session.request_time_travel(0)
    # The agent's half of the same control, which arms the identical request.
    view = _call(session, "time_travel", {"epoch": 0})
    assert "training loop is gone" in view["error"]


# --- MCP parity -----------------------------------------------------------


def test_status_reports_the_lost_state_with_its_reason() -> None:
    session, model = _detached_session()
    assert "training_error" not in status_view(session)

    run_to_death(crashing_loop(session, model))
    view = status_view(session)
    assert view["state"] == "lost"
    assert view["training_error"] == "RuntimeError: boom"
    assert "RuntimeError: boom" in view["hint"]


@pytest.mark.parametrize(
    ("tool", "arguments"),
    [
        ("step", {"timeout_seconds": 0.1}),  # every run control shares a refusal
        ("pin_batch", {"timeout_seconds": 0.1}),
        (
            "run_experiment",
            {"kind": "deep_dream", "layer": "fc1", "timeout_seconds": 0.1},
        ),
    ],
)
def test_commands_that_need_the_training_thread_refuse(
    tool: str, arguments: dict[str, Any]
) -> None:
    """Every one of these is served by the training thread's pause loop, so a
    cheerful no-op would leave an agent polling a run that cannot answer."""
    view = _call(_lost_session(), tool, arguments)
    assert view["state"] == "lost"
    assert "RuntimeError: boom" in view["error"]


def test_reading_the_last_captured_state_still_works() -> None:
    """The point of leaving the page up: a crashed run is still inspectable."""
    assert "fc1" in json.dumps(_call(_lost_session(), "get_architecture"))
