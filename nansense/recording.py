"""Per-view MP4 recording — and single-frame PNG snapshots — of visualizations.

Each recorded view renders one frame per *visualization update* (the
frequency configured via `Session.set_update_frequency`, see
`UpdateFrequency`) into its own MP4 file under
`nansense_recordings/<run timestamp>/`. The training thread drives frame
capture (`Session._record_frames` → `RecordingManager.capture_frames`)
right after a frequency update published its snapshot, probe result, and
auto-experiment reruns — so every frame is consistent with one update.

`RecordingManager.snapshot` is the still counterpart: the same
`RecordedView`, the same renderers and the same position banner, written
once as PNG the moment it is asked for (from the caller's thread — the
UI's worker or an MCP tool), into the same run directory. It needs no
recording to be active, and leaves one that is running untouched.

A `RecordedView` freezes the view's parameters at record start (watched
layers, sample index, phase, axis scales, weight-axis layouts, the
experiment request's seq, ...). The UI disables the matching controls
while the view is recorded, and the renderers here only read the frozen
params — a recording is immune to later UI fiddling.

View pages map to videos as follows:

- ``main`` — input image plus every frozen layer's activation and
  activation-gradient strips, stacked into one frame. A pinned batch or
  perturbations are respected exactly like on the page: the probe result
  becomes the render source (with the perturbation diff view when edits
  exist), and probe frames carry no gradient strips.
- ``weights:<layer>`` — the layer's weight, weight-gradient, and
  optimizer-state strips under the page's frozen axis layout, plus the
  scalar optimizer values as a text line.
- ``watch_histogram`` — matplotlib re-renders of the frozen layers'
  activation/gradient histograms for the frozen phase (the live page uses
  Plotly client-side, which can't produce server-side frames).
- ``watch_minmax`` — the frozen patch grids, rendered exactly like the
  page. Pixel grids (crops) and average grids (whole inputs) have
  different cell sizes, so they record into *separate* files — the
  recorder returns one frame per group and the manager writes
  ``..._pixel.mp4`` / ``..._average.mp4``.
- ``experiment:<layer>`` — the freshest result of the view's pinned auto
  experiment (same seq on every rerun, so deep dream redraws the same
  seeded noise each update).

The images themselves are rendered by `nansense.ui.frames`, which the MCP
server's image tools also call — the functions here only unpack a view's
frozen params and hand the result to the encoder, so a recorded frame and
an agent's picture of the same view stay identical by construction.

MP4 files are fixed-size: the first frame decides each stream's
dimensions (rounded up to even, as libx264's yuv420p requires) and later
frames are padded/cropped to fit.
"""

from __future__ import annotations

import re
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, cast

import av
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from nansense.input_config import InputTransform
from nansense.params import float_tuple, int_param, str_tuple
from nansense.schedule import BatchPosition, format_position

if TYPE_CHECKING:
    from nansense.session import Session

# Where recordings land: one timestamped subdirectory per manager (i.e. per
# training run), inside the training process's working directory.
RECORDINGS_DIR: Path = Path("nansense_recordings")

# Every recording plays back at this rate: one visualization update per frame.
VIDEO_FPS: int = 10

# Hard cap on video width/height, matching the cap `nansense.ui.compose`
# already applies to the composed image: a stream wider than this is beyond
# what libx264 will encode, and a very wide layer (many channels) is cropped
# rather than producing a GB-sized file.
MAX_FRAME_SIZE: int = 4096

# libx264 settings (used in-process via PyAV — see `_VideoStream`). The default
# `medium` preset buffers ~40 frames of rc-lookahead and defers most encoding to
# the close() flush, where its working set roughly triples — a multi-GB spike
# *at save time* (proportional to frame area) that, stacked on the training
# process, can trip the OOM killer. `ultrafast` drops the lookahead and B-frames
# so frames encode as they stream in, and a small thread cap avoids per-thread
# frame-buffer duplication; together they keep the footprint flat (~10x lower)
# and the flush near-instant. The cost is weaker compression (larger files),
# fine for short clips. CRF 10 reproduces the previous visual quality (what
# imageio's `quality=8` mapped to). See INTERNALS.md.
_X264_PRESET: str = "ultrafast"
_X264_THREADS: int = 2
_X264_CRF: int = 10

# Only the position stamp is drawn here; the frame's own layout (padding,
# section gaps, the NaN checkerboard) belongs to `nansense.ui.compose`.
_FRAME_PAD: int = 10
_LABEL_COLOR: tuple[int, int, int] = (30, 41, 59)  # slate-800


@dataclass(frozen=True)
class RecordedView:
    """One view's frozen recording spec.

    `key` is the view's identity (one recording per key at a time), `page`
    selects the renderer, `label` is the human-readable description shown
    in the recording dialog, and `params` carries the page state frozen at
    record start (treated as immutable).
    """

    key: str
    page: str  # "main" | "weights" | "watch_histogram" | "watch_minmax" | "experiment"
    label: str
    params: dict[str, object]


@dataclass(frozen=True)
class RecordingStatus:
    """A recorder's state for the UI dialog."""

    view: RecordedView
    frames: int
    error: str | None
    paths: tuple[Path, ...]


class _VideoStream:
    """One MP4 file, lazily opened and locked to its first frame's size.

    Frames are encoded in-process with PyAV (the ffmpeg *libraries*, not a
    child process). That removes the failure modes of an ffmpeg subprocess:
    there are no stdin/stdout pipes to deadlock and no separate process that can
    stall the `close()` finalize (hanging the UI's "Save & Finish") or be
    OOM-killed out from under us — encoding and the flush are bounded in-process
    calls that raise on error instead.
    """

    def __init__(self, path: Path, fps: int) -> None:
        self.path = path
        self._fps = fps
        # PyAV output container and its single H.264 stream, opened lazily on
        # the first frame so the stream size locks to that frame.
        self._container: av.container.OutputContainer | None = None
        self._stream: av.video.stream.VideoStream | None = None
        self._size: tuple[int, int] | None = None  # (height, width)

    def append(self, frame: np.ndarray) -> None:
        if self._size is None:
            # libx264 with yuv420p needs even dimensions; pad with white.
            h = min(MAX_FRAME_SIZE, frame.shape[0] + frame.shape[0] % 2)
            w = min(MAX_FRAME_SIZE, frame.shape[1] + frame.shape[1] % 2)
            self._size = (h, w)
        stream = self._ensure_stream()
        assert self._container is not None
        # `from_ndarray` needs a C-contiguous rgb24 array; `_fit_frame` may
        # return a non-contiguous crop. PyAV converts rgb24 -> yuv420p on encode.
        fitted = np.ascontiguousarray(_fit_frame(frame, *self._size))
        video_frame = av.VideoFrame.from_ndarray(fitted, format="rgb24")
        for packet in stream.encode(video_frame):
            self._container.mux(packet)

    def _ensure_stream(self) -> av.video.stream.VideoStream:
        if self._stream is not None:
            return self._stream
        assert self._size is not None  # `append` sets it before calling this
        height, width = self._size
        self.path.parent.mkdir(parents=True, exist_ok=True)
        container = av.open(str(self.path), mode="w")
        stream = container.add_stream(
            "libx264",
            rate=self._fps,
            options={"preset": _X264_PRESET, "crf": str(_X264_CRF)},
        )
        stream.width = width
        stream.height = height
        stream.pix_fmt = "yuv420p"
        stream.thread_count = _X264_THREADS
        self._container = container
        self._stream = stream
        return stream

    def close(self) -> None:
        if self._container is None:
            return  # nothing was ever written (no frames captured)
        try:
            if self._stream is not None:
                for packet in self._stream.encode():  # flush the encoder
                    self._container.mux(packet)
        finally:
            self._container.close()
            self._container = None
            self._stream = None

    def delete(self) -> None:
        self.close()
        self.path.unlink(missing_ok=True)


def _fit_frame(frame: np.ndarray, height: int, width: int) -> np.ndarray:
    """Pad (white) or crop `frame` to exactly `height × width`."""
    frame = frame[:height, :width]
    if frame.shape[0] == height and frame.shape[1] == width:
        return frame
    fitted = np.full((height, width, 3), 255, dtype=np.uint8)
    fitted[: frame.shape[0], : frame.shape[1]] = frame
    return fitted


class ViewRecorder:
    """One recorded view: renders frames and feeds its video stream(s).

    Most views write a single stream (empty suffix). The MIN/MAX view
    writes one stream per enabled cell-size group ("pixel" / "average"),
    since their frames have different dimensions.

    Frame rendering runs outside every lock — it can take seconds for
    large views. Only the short append/close/delete sections are
    serialised by the recorder's own lock, so ending a recording never
    waits behind a render; a frame whose render raced a `close` is simply
    dropped (`_closed`).
    """

    def __init__(self, view: RecordedView, *, directory: Path, fps: int) -> None:
        self.view = view
        self.frames = 0
        self.error: str | None = None
        self._directory = directory
        self._fps = fps
        self._lock = threading.Lock()
        self._closed = False
        self._streams: dict[str, _VideoStream] = {}

    def capture(self, session: Session) -> None:
        # Rendering and position-stamping are unlocked: slow image work.
        frames = _render_view_frames(self.view, session)
        position = _capture_position(session)
        stamped: dict[str, np.ndarray] = {}
        for suffix, frame in frames.items():
            if frame is None:
                continue
            if position is not None:
                frame = _stamp_position(frame, format_position(position))
            stamped[suffix] = frame
        with self._lock:
            if self._closed:
                return
            for suffix, frame in stamped.items():
                self._stream_locked(suffix).append(frame)
            if stamped:
                self.frames += 1

    def _stream_locked(self, suffix: str) -> _VideoStream:
        stream = self._streams.get(suffix)
        if stream is None:
            name = _sanitize(self.view.key) + (f"_{suffix}" if suffix else "")
            stream = _VideoStream(self._directory / f"{name}.mp4", self._fps)
            self._streams[suffix] = stream
        return stream

    def paths(self) -> tuple[Path, ...]:
        with self._lock:
            return tuple(s.path for s in self._streams.values())

    def close(self) -> None:
        with self._lock:
            self._closed = True
            for stream in self._streams.values():
                stream.close()

    def delete(self) -> None:
        with self._lock:
            self._closed = True
            for stream in self._streams.values():
                stream.delete()


_POSITION_BANNER_HEIGHT: int = 18


def _capture_position(session: Session) -> BatchPosition | None:
    """The training position the captured frame corresponds to.

    Frames are captured at frequency updates on the training thread, where
    `live_position` is the batch being visualized; the snapshot covers the
    brief window before the first batch publishes a live position.
    """
    if session.live_position is not None:
        return session.live_position
    snapshot = session.snapshot
    return snapshot.position if snapshot is not None else None


def _stamp_position(frame: np.ndarray, text: str) -> np.ndarray:
    """Prepend a white banner with the training position to a frame."""
    image = Image.fromarray(frame)
    canvas = Image.new(
        "RGB", (image.width, image.height + _POSITION_BANNER_HEIGHT), (255, 255, 255)
    )
    draw = ImageDraw.Draw(canvas)
    draw.text(
        (_FRAME_PAD, 3), text, fill=_LABEL_COLOR, font=ImageFont.load_default()
    )
    canvas.paste(image, (0, _POSITION_BANNER_HEIGHT))
    return np.asarray(canvas)


def _sanitize(key: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", key).strip("_") or "view"


def _position_slug(position: BatchPosition) -> str:
    """A snapshot filename's position stamp — `ep3_train_b12`.

    The banner drawn *into* the image says the same thing in prose; this is
    the sortable, filename-safe form of it, so a directory listing of
    snapshots reads in training order.
    """
    return f"ep{position.epoch}_{position.phase}_b{position.batch_idx}"


def _unique_path(directory: Path, stem: str) -> Path:
    """`directory/stem.png`, suffixed `-2`, `-3`, … if it is already taken.

    Two snapshots of one view at one position are a normal thing to want
    (before and after a perturbation, say), and neither should overwrite
    the other.
    """
    path = directory / f"{stem}.png"
    n = 2
    while path.exists():
        path = directory / f"{stem}-{n}.png"
        n += 1
    return path


class RecordingManager:
    """All active recordings of one session, shared across UI connections.

    Start/end/delete come from the UI thread; `capture_frames` runs on the
    training thread at every frequency update. The manager's lock guards
    only the recorder dict, so the queries UI timers and handlers make on
    the asyncio event loop (`count`, `is_recording`, `statuses`) return
    immediately even while a frame renders — a blocked event loop starves
    NiceGUI's websocket keepalive (~6 s budget) and kills the connection.
    Rendering and writer finalization run outside the manager lock (see
    `ViewRecorder`). A renderer failure is stored on the recorder (and
    shown in the dialog) instead of propagating into the training loop.
    """

    def __init__(self, *, directory: Path | None = None, fps: int = VIDEO_FPS) -> None:
        self._lock = threading.Lock()
        # Snapshots take their own lock, held across "pick a free name, write
        # it" so two concurrent stills can't choose the same file. Keeping it
        # apart from `_lock` is what stops a PNG write from delaying the
        # `count` / `statuses` polls the event loop makes (see the class
        # docstring) — they contend for nothing here.
        self._snapshot_lock = threading.Lock()
        self._recorders: dict[str, ViewRecorder] = {}
        self.directory = (
            directory
            if directory is not None
            else RECORDINGS_DIR / time.strftime("%Y%m%d-%H%M%S")
        )
        self._fps = fps

    def start(self, view: RecordedView) -> bool:
        """Begin recording `view`. Returns False when its key already records."""
        with self._lock:
            if view.key in self._recorders:
                return False
            self._recorders[view.key] = ViewRecorder(
                view, directory=self.directory, fps=self._fps
            )
            return True

    def is_recording(self, key: str) -> bool:
        with self._lock:
            return key in self._recorders

    def count(self) -> int:
        with self._lock:
            return len(self._recorders)

    def statuses(self) -> list[RecordingStatus]:
        with self._lock:
            recorders = list(self._recorders.values())
        return [
            RecordingStatus(
                view=r.view, frames=r.frames, error=r.error, paths=r.paths()
            )
            for r in recorders
        ]

    def end(self, key: str) -> tuple[Path, ...]:
        """Finalize `key`'s video file(s) and stop recording it."""
        with self._lock:
            recorder = self._recorders.pop(key, None)
        if recorder is None:
            return ()
        recorder.close()
        return recorder.paths()

    def delete(self, key: str) -> None:
        """Discard `key`'s recording, removing its file(s) from disk."""
        with self._lock:
            recorder = self._recorders.pop(key, None)
        if recorder is not None:
            recorder.delete()

    def end_all(self) -> tuple[Path, ...]:
        with self._lock:
            recorders = list(self._recorders.values())
            self._recorders.clear()
        paths: list[Path] = []
        for recorder in recorders:
            recorder.close()
            paths.extend(recorder.paths())
        return tuple(paths)

    def delete_all(self) -> None:
        with self._lock:
            recorders = list(self._recorders.values())
            self._recorders.clear()
        for recorder in recorders:
            recorder.delete()

    def snapshot(self, view: RecordedView, session: Session) -> tuple[Path, ...]:
        """Render `view` once, right now, and write it as PNG file(s).

        A recording's single frame, taken on demand: same frozen
        `RecordedView`, same renderers, same position banner — but written
        immediately from the calling thread (the UI's worker thread or an
        MCP tool call) instead of once per update from the training thread.
        Nothing needs to be recording, and a recording of the same view is
        untouched — the recorder dict is never consulted.

        Returns the files written, newest name last: several when the view
        splits into groups (the MIN/MAX pixel/average grids), and `()` when
        the view has nothing to draw yet. Renderer failures propagate — a
        snapshot is a foreground action with a caller to tell, unlike a
        recording's frame, which must never reach the training loop.
        """
        frames = _render_view_frames(view, session)
        position = _capture_position(session)
        stamp = "" if position is None else _position_slug(position)
        paths: list[Path] = []
        for suffix, frame in frames.items():
            if frame is None:
                continue
            if position is not None:
                frame = _stamp_position(frame, format_position(position))
            stem = _sanitize(
                "_".join(part for part in (view.key, stamp, suffix) if part)
            )
            with self._snapshot_lock:
                self.directory.mkdir(parents=True, exist_ok=True)
                path = _unique_path(self.directory, stem)
                Image.fromarray(frame).save(path)
            paths.append(path)
        return tuple(paths)

    def capture_frames(self, session: Session) -> None:
        """Append one frame to every active recording (training thread).

        Renders outside the manager lock: a recorder that gets ended or
        deleted mid-render finishes the render and drops the frame (see
        `ViewRecorder.capture`).
        """
        with self._lock:
            recorders = list(self._recorders.values())
        for recorder in recorders:
            try:
                recorder.capture(session)
            except Exception as e:  # noqa: BLE001 — shown in the dialog
                recorder.error = f"{type(e).__name__}: {e}"


# ---------------------------------------------------------------------------
# Frame renderers — one per view page, all returning `suffix -> RGB array`.
# ---------------------------------------------------------------------------


def _render_view_frames(
    view: RecordedView, session: Session
) -> dict[str, np.ndarray | None]:
    """Every video stream this view writes, as `suffix -> RGB frame`.

    Each branch only unpacks the view's frozen params; the rendering itself
    lives in `nansense.ui.frames`, shared with the MCP server's image tools so
    a recorded frame and an agent's picture can't drift apart.
    """
    if view.page == "main":
        return {"": _array(_main_frame(view, session))}
    if view.page == "weights":
        return {"": _array(_weights_frame(view, session))}
    if view.page == "watch_histogram":
        return {"": _array(_histogram_frame(view, session))}
    if view.page == "watch_minmax":
        return _minmax_frames(view, session)
    if view.page == "experiment":
        return {"": _array(_experiment_frame(view, session))}
    raise ValueError(f"unknown recorded view page {view.page!r}")


def _array(image: Image.Image | None) -> np.ndarray | None:
    return None if image is None else np.asarray(image)


def _main_frame(view: RecordedView, session: Session) -> Image.Image | None:
    from nansense.ui.frames import main_frame

    return main_frame(
        session,
        layers=str_tuple(view.params.get("layers")),
        sample_idx=int_param(view.params, "sample_idx"),
        input_name=str(view.params.get("input_name") or "") or None,
        mean=float_tuple(view.params.get("input_mean")),
        std=float_tuple(view.params.get("input_std")),
        transform=cast(InputTransform | None, view.params.get("input_transform")),
    )


def _weights_frame(view: RecordedView, session: Session) -> Image.Image | None:
    from nansense.ui.frames import PanelAxes, WeightPanel, weights_frame
    from nansense.ui.render import dims_from_roles

    panels = view.params.get("panels")
    if not isinstance(panels, (list, tuple)):
        return None
    specs: list[WeightPanel] = []
    for spec in panels:
        # Each spec is `(name, roles, index pairs)` — see the weights page's
        # record-view factory.
        if not (isinstance(spec, (list, tuple)) and len(spec) == 3):
            continue
        x_dim, y_dim, tile_dim = dims_from_roles(
            [str(role) for role in str_tuple(spec[1])]
        )
        fixed: dict[int, int] = {}
        if isinstance(spec[2], (list, tuple)):
            for pair in spec[2]:
                if isinstance(pair, (list, tuple)) and len(pair) == 2:
                    dim, idx = pair
                    if isinstance(dim, int) and isinstance(idx, int):
                        fixed[dim] = idx
        specs.append(
            WeightPanel(
                name=str(spec[0]),
                # Always explicit: these came from the page's role selects, so
                # even an all-unassigned triple is a choice the user made.
                axes=PanelAxes(x_dim=x_dim, y_dim=y_dim, tile_dim=tile_dim),
                fixed=fixed,
            )
        )
    return weights_frame(session, panels=specs)


def _histogram_frame(view: RecordedView, session: Session) -> Image.Image | None:
    from nansense.ui.frames import histogram_frame

    return histogram_frame(
        session,
        layers=str_tuple(view.params.get("layers")),
        phase=str(view.params.get("phase") or ""),
        log_x=bool(view.params.get("log_x")),
        log_y=bool(view.params.get("log_y")),
    )


def _minmax_frames(
    view: RecordedView, session: Session
) -> dict[str, np.ndarray | None]:
    from nansense.ui.frames import patch_frames

    frames = patch_frames(
        session,
        layers=str_tuple(view.params.get("layers")),
        phase=str(view.params.get("phase") or ""),
        grids=str_tuple(view.params.get("grids")),
        heatmap=bool(view.params.get("heatmap")),
        mean=float_tuple(view.params.get("input_mean")),
        std=float_tuple(view.params.get("input_std")),
    )
    return {group: _array(image) for group, image in frames.items()}


def _experiment_frame(view: RecordedView, session: Session) -> Image.Image | None:
    from nansense.ui.frames import experiment_frame

    return experiment_frame(
        session,
        seq=int_param(view.params, "seq"),
        mean=float_tuple(view.params.get("input_mean")),
        std=float_tuple(view.params.get("input_std")),
    )
