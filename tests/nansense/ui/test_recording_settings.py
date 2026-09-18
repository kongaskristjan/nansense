"""Recording completion releases registrations independently of file and UI work."""

from __future__ import annotations

import asyncio
import threading

import pytest
from nicegui import ui

from nansense.contracts.recording import ExperimentView, RecordedView
from nansense.recording import ViewRecorder
from nansense.ui.settings.recording import RecordingSettings
from tests.nansense.helpers import make_session


@pytest.mark.parametrize("keep", [True, False], ids=["save", "delete"])
@pytest.mark.parametrize(
    ("fail", "replacement"),
    [(True, False), (False, True), (True, True)],
    ids=["failed-finalization", "active-replacement", "failed-with-replacement"],
)
def test_finishing_releases_only_unowned_experiments_and_refreshes_ui(
    monkeypatch: pytest.MonkeyPatch, keep: bool, fail: bool, replacement: bool
) -> None:
    session, _ = make_session(epochs=1, phases={"train": 1})
    auto_key = "page-experiment"
    seq = session.register_auto_experiment(
        auto_key, kind="deep_dream", layer="fc1", params={"steps": 1}
    )
    view = RecordedView(
        "experiment", "Experiment", ExperimentView(seq=seq, auto_key=auto_key)
    )
    assert session.pin_auto_experiment(auto_key)
    assert session.recording.start(view)
    released: list[str] = []
    unpin = session.unpin_auto_experiment

    def release(key: str) -> None:
        released.append(key)
        unpin(key)

    monkeypatch.setattr(session, "unpin_auto_experiment", release)
    notices: list[tuple[str, object]] = []

    def notify(message: str, **kwargs: object) -> None:
        notices.append((message, kwargs.get("type")))

    monkeypatch.setattr(ui, "notify", notify)
    refreshed: list[int] = []
    rebuilt: list[int] = []
    with ui.card():
        settings = RecordingSettings(
            session,
            lambda: view,
            lambda: refreshed.append(session.recording.count()),
            lambda update: update(),
        )
    monkeypatch.setattr(
        settings, "rebuild", lambda: rebuilt.append(session.recording.count())
    )

    async def finish() -> None:
        entered = asyncio.Event()
        proceed = threading.Event()
        loop = asyncio.get_running_loop()

        def finalize(recorder: ViewRecorder) -> None:
            loop.call_soon_threadsafe(entered.set)
            assert proceed.wait(timeout=5), "completion was not released"
            if fail:
                raise OSError("encoder failed")

        with monkeypatch.context() as scoped:
            scoped.setattr(ViewRecorder, "close" if keep else "delete", finalize)
            operation = settings.end_view if keep else settings.delete_view
            pending = asyncio.create_task(operation(view.key, view))
            try:
                await asyncio.wait_for(entered.wait(), timeout=5)
                assert session.recording.count() == 0
                if replacement:
                    # A distinct view may share the same auto registration.
                    newer = RecordedView("replacement", "Replacement", view.config)
                    assert session.recording.start(newer)
                    assert session.pin_auto_experiment(auto_key)
            finally:
                proceed.set()
                await pending

    try:
        asyncio.run(finish())
        assert released == ([] if replacement else [auto_key])
        assert refreshed == rebuilt == [int(replacement)]
        if fail:
            assert any(
                "encoder failed" in message and kind == "negative"
                for message, kind in notices
            )
        else:
            assert not any(kind == "negative" for _, kind in notices)
    finally:
        session.recording.delete_all()
        session.close()
