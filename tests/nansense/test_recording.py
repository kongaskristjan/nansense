"""Tests for per-view MP4 recording (`nansense.recording`)."""

from __future__ import annotations

import threading
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import pytest
import torch
from torch import Tensor, nn

import nansense
import nansense.recording
from nansense.recording import (
    RecordedView,
    RecordingManager,
    _fit_frame,
    _sanitize,
)
from nansense.session import Session


class TinyConvNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(1, 3, 3, padding=1)
        self.fc = nn.Linear(3 * 6 * 6, 4)

    def forward(self, x: Tensor) -> Tensor:
        return self.fc(torch.relu(self.conv(x)).flatten(1))


def _make_session(
    tmp_path: Path, epochs: int = 2, phases: dict[str, int] | None = None
) -> tuple[Session, TinyConvNet, RecordingManager]:
    if phases is None:
        phases = {"train": 2}
    model = TinyConvNet()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    session = nansense.start(model, epochs=epochs, phases=phases, optimizer=optimizer)
    manager = RecordingManager(directory=tmp_path / "rec")
    session._recording_manager = manager
    return session, model, manager


def _run_epochs(
    session: Session, model: TinyConvNet, *, epochs: int, phases: dict[str, int]
) -> None:
    optimizer = session.optimizer
    assert optimizer is not None
    for epoch in range(epochs):
        for phase, n in phases.items():
            for _ in range(n):
                with session.batch(phase=phase, epoch=epoch):
                    x = torch.randn(2, 1, 6, 6)
                    optimizer.zero_grad()
                    model(x).sum().backward()
                    optimizer.step()


def _frame_count(path: Path) -> int:
    return len(imageio.mimread(str(path)))


def _main_view(layers: tuple[str, ...] = ("conv",)) -> RecordedView:
    return RecordedView(
        key="main",
        page="main",
        label="Main view",
        params={
            "layers": layers,
            "sample_idx": 0,
            "input_name": "x",
            "input_mean": None,
            "input_std": None,
        },
    )


def test_fit_frame_pads_and_crops() -> None:
    frame = np.zeros((4, 6, 3), dtype=np.uint8)
    padded = _fit_frame(frame, 6, 8)
    assert padded.shape == (6, 8, 3)
    assert (padded[4:] == 255).all()  # white padding
    cropped = _fit_frame(frame, 2, 3)
    assert cropped.shape == (2, 3, 3)


@pytest.mark.parametrize(
    ("key", "expected"),
    [("weights:conv.weight", "weights_conv.weight"), ("main", "main"), ("//", "view")],
)
def test_sanitize_keys(key: str, expected: str) -> None:
    assert _sanitize(key) == expected


def test_manager_start_end_delete(tmp_path: Path) -> None:
    manager = RecordingManager(directory=tmp_path)
    view = _main_view()
    assert manager.start(view)
    assert not manager.start(view)  # one recording per key
    assert manager.is_recording("main")
    assert manager.count() == 1
    statuses = manager.statuses()
    assert len(statuses) == 1 and statuses[0].view is view
    assert manager.end("main") == ()  # no frames captured, no files
    assert manager.count() == 0


def test_main_view_records_one_frame_per_update(tmp_path: Path) -> None:
    phases = {"train": 2}
    session, model, manager = _make_session(tmp_path, epochs=2, phases=phases)
    session.watch("conv")
    assert manager.start(_main_view())
    session.detach()
    _run_epochs(session, model, epochs=2, phases=phases)
    (path,) = manager.end("main")
    assert path.exists()
    # Default frequency: one update per epoch -> two frames.
    assert _frame_count(path) == 2


def test_minmax_records_pixel_and_average_to_separate_files(tmp_path: Path) -> None:
    phases = {"train": 2}
    session, model, manager = _make_session(tmp_path, epochs=2, phases=phases)
    session.watch("conv")
    assert manager.start(
        RecordedView(
            key="watch_minmax",
            page="watch_minmax",
            label="MIN/MAX",
            params={
                "layers": ("conv",),
                "phase": "train",
                "grids": ("max_pixel", "max_average"),
                "heatmap": True,
                "input_mean": None,
                "input_std": None,
            },
        )
    )
    session.detach()
    _run_epochs(session, model, epochs=2, phases=phases)
    paths = manager.end("watch_minmax")
    names = sorted(p.name for p in paths)
    assert names == ["watch_minmax_average.mp4", "watch_minmax_pixel.mp4"]
    for path in paths:
        assert _frame_count(path) == 2


def test_histogram_recording_renders_matplotlib_frames(tmp_path: Path) -> None:
    phases = {"train": 2}
    session, model, manager = _make_session(tmp_path, epochs=2, phases=phases)
    session.watch("conv")
    assert manager.start(
        RecordedView(
            key="watch_histogram",
            page="watch_histogram",
            label="Histograms",
            params={
                "layers": ("conv",),
                "phase": "train",
                "log_x": True,
                "log_y": False,
            },
        )
    )
    session.detach()
    _run_epochs(session, model, epochs=2, phases=phases)
    (path,) = manager.end("watch_histogram")
    assert _frame_count(path) == 2


def test_weights_recording_includes_optimizer_state(tmp_path: Path) -> None:
    phases = {"train": 2}
    session, model, manager = _make_session(tmp_path, epochs=1, phases=phases)
    assert manager.start(
        RecordedView(
            key="weights:conv",
            page="weights",
            label="Weights · conv",
            params={
                "layer": "conv",
                "panels": (
                    ("conv.weight", ("index", "tile", "y", "x"), ((0, 0),)),
                    ("conv.bias", ("x",), ()),
                ),
            },
        )
    )
    session.detach()
    _run_epochs(session, model, epochs=1, phases=phases)
    (path,) = manager.end("weights:conv")
    assert path.name == "weights_conv.mp4"
    assert _frame_count(path) == 1


def test_experiment_recording_tracks_auto_reruns(tmp_path: Path) -> None:
    phases = {"train": 1}
    session, model, manager = _make_session(tmp_path, epochs=2, phases=phases)
    seq = session.register_auto_experiment(
        "page-1",
        kind="deep_dream",
        layer="conv",
        params={"steps": 1, "batch": 1, "mean": None, "std": None},
    )
    assert session.pin_auto_experiment("page-1")
    assert manager.start(
        RecordedView(
            key="experiment:conv",
            page="experiment",
            label="Deep dream · conv",
            params={
                "layer": "conv",
                "seq": seq,
                "auto_key": "page-1",
                "input_mean": None,
                "input_std": None,
            },
        )
    )
    session.detach()
    _run_epochs(session, model, epochs=2, phases=phases)
    (path,) = manager.end("experiment:conv")
    # Both epoch-end updates re-ran the experiment and recorded its result.
    assert _frame_count(path) == 2


def test_delete_removes_files(tmp_path: Path) -> None:
    phases = {"train": 1}
    session, model, manager = _make_session(tmp_path, epochs=1, phases=phases)
    session.watch("conv")
    assert manager.start(_main_view())
    session.detach()
    _run_epochs(session, model, epochs=1, phases=phases)
    (status,) = manager.statuses()
    assert status.frames == 1
    assert status.paths  # the stream knows its target file
    manager.delete("main")
    assert all(not p.exists() for p in status.paths)
    assert not list((tmp_path / "rec").glob("*.mp4"))
    assert manager.count() == 0


def test_close_finalizes_recordings(tmp_path: Path) -> None:
    phases = {"train": 1}
    session, model, manager = _make_session(tmp_path, epochs=1, phases=phases)
    session.watch("conv")
    assert manager.start(_main_view())
    session.detach()
    _run_epochs(session, model, epochs=1, phases=phases)
    session.close()
    assert manager.count() == 0  # end_all ran
    (path,) = list((tmp_path / "rec").glob("*.mp4"))
    assert _frame_count(path) == 1


def _block_renderer(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[threading.Event, threading.Event]:
    """Stall `_render_view_frames` on an event, signalling when it started.

    Returns `(rendering, release)`: the fake renderer sets `rendering`,
    waits for `release`, then returns one tiny frame — letting a test hold
    a capture mid-render without sleeping.
    """
    rendering = threading.Event()
    release = threading.Event()

    def slow_render(
        view: RecordedView, session: Session
    ) -> dict[str, np.ndarray | None]:
        rendering.set()
        assert release.wait(timeout=30.0)
        return {"": np.zeros((4, 4, 3), dtype=np.uint8)}

    monkeypatch.setattr(nansense.recording, "_render_view_frames", slow_render)
    return rendering, release


def test_manager_queries_do_not_block_during_render(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """UI timers/handlers poll the manager on the asyncio event loop; they
    must never wait behind a frame render (a blocked loop starves NiceGUI's
    websocket keepalive and drops the connection)."""
    session, _, manager = _make_session(tmp_path)
    assert manager.start(_main_view())
    rendering, release = _block_renderer(monkeypatch)
    capture = threading.Thread(
        target=manager.capture_frames, args=(session,), daemon=True
    )
    capture.start()
    assert rendering.wait(timeout=30.0)

    # Probe from a helper thread so a regression hangs the probe, not pytest.
    probed: dict[str, object] = {}

    def probe() -> None:
        probed["count"] = manager.count()
        probed["recording"] = manager.is_recording("main")
        probed["statuses"] = len(manager.statuses())

    prober = threading.Thread(target=probe, daemon=True)
    prober.start()
    prober.join(timeout=10.0)
    assert not prober.is_alive(), "manager query blocked behind a frame render"
    assert probed == {"count": 1, "recording": True, "statuses": 1}

    release.set()
    capture.join(timeout=30.0)
    assert not capture.is_alive()
    assert manager.statuses()[0].frames == 1


def test_end_during_in_flight_render_drops_the_frame(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    session, _, manager = _make_session(tmp_path)
    assert manager.start(_main_view())
    rendering, release = _block_renderer(monkeypatch)
    capture = threading.Thread(
        target=manager.capture_frames, args=(session,), daemon=True
    )
    capture.start()
    assert rendering.wait(timeout=30.0)

    # Ending mid-render must not wait for the render; the recorder had no
    # frames yet, so there is nothing to finalize.
    assert manager.end("main") == ()
    assert manager.count() == 0

    # The raced render completes but its frame is dropped: no file appears.
    release.set()
    capture.join(timeout=30.0)
    assert not capture.is_alive()
    assert not list((tmp_path / "rec").glob("*.mp4"))


def test_renderer_error_is_stored_not_raised(tmp_path: Path) -> None:
    phases = {"train": 1}
    session, model, manager = _make_session(tmp_path, epochs=1, phases=phases)
    assert manager.start(
        RecordedView(key="broken", page="nonsense", label="broken", params={})
    )
    session.detach()
    _run_epochs(session, model, epochs=1, phases=phases)  # must not raise
    (status,) = manager.statuses()
    assert status.error is not None
    assert status.frames == 0
