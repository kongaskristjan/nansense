"""Tests for per-view MP4 recording (`nansense.recording`)."""

from __future__ import annotations

import os
import signal
import threading
from pathlib import Path
from typing import Any

import imageio.v2 as imageio
import numpy as np
import pytest
import torch
from torch import Tensor, nn

import nansense
import nansense.recording
from nansense.recording import (
    _CHECKER_DARK,
    _CHECKER_LIGHT,
    RecordedView,
    RecordingManager,
    _POSITION_BANNER_HEIGHT,
    _checkerboard,
    _fit_frame,
    _render_main_frame,
    _sanitize,
    _stamp_position,
    _strip_section,
    _VideoStream,
)
from nansense.session import Session
from nansense.ui.render import render_strip


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


def test_stamp_position_adds_banner() -> None:
    frame = np.zeros((20, 120, 3), dtype=np.uint8)
    stamped = _stamp_position(frame, "epoch 0 | train batch 0")
    assert stamped.shape == (20 + _POSITION_BANNER_HEIGHT, 120, 3)
    banner = stamped[:_POSITION_BANNER_HEIGHT]
    assert (banner == 255).any()  # white background
    assert (banner != 255).any()  # dark text pixels
    assert (stamped[_POSITION_BANNER_HEIGHT:] == 0).all()  # frame below, intact


def test_checkerboard_is_two_grays_in_4px_boxes() -> None:
    cb = np.asarray(_checkerboard(8, 8))
    assert cb.shape == (8, 8, 4)
    assert (cb[..., 3] == 255).all()  # opaque backdrop
    assert tuple(cb[0, 0, :3]) == _CHECKER_LIGHT
    assert tuple(cb[0, 4, :3]) == _CHECKER_DARK  # box flips at 4px
    assert tuple(cb[4, 0, :3]) == _CHECKER_DARK
    assert tuple(cb[4, 4, :3]) == _CHECKER_LIGHT


def test_strip_section_bakes_checkerboard_behind_nan_cells() -> None:
    # An all-NaN strip is fully transparent RGBA; the recorded section must
    # show the two checkerboard grays behind it, not white and not one gray.
    strip = render_strip(torch.full((1, 1, 8, 8), float("nan")), sample_idx=0)
    assert strip is not None
    section = _strip_section(strip)
    assert section is not None
    arr = np.asarray(section)
    colors = {tuple(c) for row in arr for c in row}
    assert _CHECKER_LIGHT in colors
    assert _CHECKER_DARK in colors
    # No fully-white data region (the old whitewash bug) over the NaN cells.
    nan_region = arr[:, strip.width // 2 :]  # right of the legend
    assert not (nan_region == 255).all()


def test_strip_section_keeps_plain_path_for_finite_strip() -> None:
    # An all-finite strip is opaque RGB; no checkerboard gray leaks in.
    strip = render_strip(torch.zeros(1, 1, 8, 8), sample_idx=0)
    assert strip is not None
    section = _strip_section(strip)
    assert section is not None
    colors = {tuple(c) for row in np.asarray(section) for c in row}
    assert _CHECKER_DARK not in colors


def test_main_frame_shows_checkerboard_for_nan_activation(tmp_path: Path) -> None:
    # End to end: a snapshot with a NaN activation produces a main frame whose
    # strip region shows both checkerboard grays (not a single gray, not white).
    from nansense.schedule import BatchPosition
    from nansense.session import BatchSnapshot

    session, model, _ = _make_session(tmp_path, epochs=1, phases={"train": 1})
    act = torch.randn(2, 3, 6, 6)
    act.view(-1)[0] = float("nan")
    session._snapshot = BatchSnapshot(
        position=BatchPosition(
            phase="train",
            epoch=0,
            batch_idx=0,
            is_last_in_phase=True,
            is_last_in_epoch=True,
            is_last_overall=True,
        ),
        activations={"conv": act, "x": torch.rand(2, 1, 6, 6)},
        activation_gradients={"conv": torch.randn(2, 3, 6, 6)},
        weights={},
        weight_gradients={},
    )
    frame = _render_main_frame(_main_view(), session)
    assert frame is not None
    colors = {tuple(c) for row in frame for c in row}
    assert _CHECKER_LIGHT in colors and _CHECKER_DARK in colors


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


@pytest.mark.skipif(os.name != "posix", reason="SIGSTOP stall sim is POSIX-only")
def test_close_does_not_hang_when_ffmpeg_stalls(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`_VideoStream.close()` must return even if ffmpeg never exits.

    `imageio_ffmpeg`'s writer waits forever for ffmpeg to quit, so a stalled
    encoder — here frozen with SIGSTOP, the same effect as starvation under
    memory pressure — would otherwise hang "Save & Finish" indefinitely. With a
    bounded `_FFMPEG_CLOSE_TIMEOUT` the subprocess is killed and close returns.
    """
    monkeypatch.setattr(nansense.recording, "_FFMPEG_CLOSE_TIMEOUT", 1.0)
    stream = _VideoStream(tmp_path / "stall.mp4", fps=10)
    for _ in range(3):
        stream.append(np.zeros((64, 96, 3), dtype=np.uint8))
    assert stream._writer is not None
    # The ffmpeg subprocess lives in the suspended generator's frame.
    gen: Any = stream._writer
    proc = gen.gi_frame.f_locals["p"]
    proc.send_signal(signal.SIGSTOP)  # freeze: ffmpeg can no longer exit
    try:
        done = threading.Event()
        error: list[BaseException] = []

        def _close() -> None:
            try:
                stream.close()
            except BaseException as e:  # noqa: BLE001 — surfaced via assert
                error.append(e)
            finally:
                done.set()

        threading.Thread(target=_close, daemon=True).start()
        # Comfortably above the 1s timeout but far below "forever".
        assert done.wait(timeout=10.0), "close() hung waiting for stalled ffmpeg"
        assert not error, f"close() raised: {error[0]!r}"
    finally:
        if proc.poll() is None:
            proc.kill()


def test_writer_passes_memory_bounded_encoder_settings(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The writer must request a bounded timeout and low-memory x264 settings.

    x264's default lookahead defers most encoding to the close() flush, where
    its RSS roughly triples — a save-time spike that can OOM training. The
    `ultrafast`/thread-capped `output_params` keep it flat; this guards them
    (and the close timeout) against regression without spawning real ffmpeg.
    """
    captured: dict[str, object] = {}

    def fake_write_frames(path: str, size: tuple[int, int], **kwargs: object):
        captured["path"] = path
        captured["size"] = size
        captured.update(kwargs)

        def gen():  # type: ignore[no-untyped-def]
            while True:
                yield

        return gen()

    monkeypatch.setattr("imageio_ffmpeg.write_frames", fake_write_frames)
    stream = _VideoStream(tmp_path / "x.mp4", fps=10)
    stream.append(np.zeros((64, 96, 3), dtype=np.uint8))

    assert captured["ffmpeg_timeout"] == nansense.recording._FFMPEG_CLOSE_TIMEOUT
    output_params = captured["output_params"]
    assert isinstance(output_params, list)
    assert "ultrafast" in output_params  # lookahead off -> no close-time spike
    assert "-threads" in output_params  # cap per-thread frame buffers


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
