"""MCP recording cleanup reports file failures without orphaning experiments."""

from __future__ import annotations

import pytest

from nansense.contracts.recording import ExperimentView, RecordedView
from nansense.recording import ViewRecorder
from tests.nansense.helpers import make_session
from tests.nansense.test_mcp_control import _call


@pytest.mark.parametrize("tool", ["stop_recording", "discard_recording"])
def test_failed_finalization_continues_cleanup_and_preserves_active_replacement(
    monkeypatch: pytest.MonkeyPatch, tool: str
) -> None:
    session, _ = make_session(epochs=1, phases={"train": 1})
    for key in ("failed", "other"):
        seq = session.register_auto_experiment(
            key, kind="deep_dream", layer="fc1", params={"steps": 1}
        )
        view = RecordedView(key, key, ExperimentView(seq=seq, auto_key=key))
        session.pin_auto_experiment(key)
        session.recording.start(view)
    completed: list[str] = []

    def finalize(recorder: ViewRecorder) -> None:
        completed.append(recorder.view.key)
        if recorder.view.key == "failed":
            # Reproduce a second client starting a replacement while file work runs.
            replacement = RecordedView(
                "replacement", "Replacement", recorder.view.config
            )
            assert session.recording.start(replacement)
            raise OSError("encoder failed")

    with monkeypatch.context() as scoped:
        scoped.setattr(
            ViewRecorder, "close" if tool == "stop_recording" else "delete", finalize
        )
        result = _call(session, tool)
    try:
        assert "encoder failed" in result["error"]
        assert completed == ["failed", "other"]
        assert session.pin_auto_experiment("failed")
        assert not session.pin_auto_experiment("other")
        assert [status.view.key for status in session.recording.statuses()] == [
            "replacement"
        ]
    finally:
        session.recording.delete_all()
        session.close()
