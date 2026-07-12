"""Per-view MP4 recording of visualizations.

Each recorded view renders one frame per *visualization update* (the
frequency configured via `Session.set_update_frequency`, see
`UpdateFrequency`) into its own MP4 file under
`nansense_recordings/<run timestamp>/`. The training thread drives frame
capture (`Session._record_frames` → `RecordingManager.capture_frames`)
right after a frequency update published its snapshot, probe result, and
auto-experiment reruns — so every frame is consistent with one update.

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

MP4 files are fixed-size: the first frame decides each stream's
dimensions (rounded up to even, as libx264's yuv420p requires) and later
frames are padded/cropped to fit.
"""

from __future__ import annotations

import io
import re
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, cast

import av
import numpy as np
from matplotlib.axes import Axes
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from PIL import Image, ImageDraw, ImageFont
from torch import Tensor

from nansense.input_config import InputTransform
from nansense.params import float_tuple, int_param, str_tuple
from nansense.patches import PATCH_TYPES, PatchType
from nansense.schedule import BatchPosition, format_position
from nansense.watch import N_BINS, LayerStatsSnapshot, TensorStatsSnapshot

if TYPE_CHECKING:
    from nansense.session import Session

# Where recordings land: one timestamped subdirectory per manager (i.e. per
# training run), inside the training process's working directory.
RECORDINGS_DIR: Path = Path("nansense_recordings")

# Every recording plays back at this rate: one visualization update per frame.
VIDEO_FPS: int = 10

# Hard cap on frame width/height; very wide layers (many channels) are
# cropped rather than producing GB-sized videos.
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

_SECTION_GAP: int = 10
_FRAME_PAD: int = 10
_LABEL_COLOR: tuple[int, int, int] = (30, 41, 59)  # slate-800

# GIMP-style transparency backdrop baked behind a strip's transparent
# NaN/±Inf cells, matching the live UI's CSS checkerboard
# (`static._STRIP_CHECKERBOARD_STYLE`): two slate grays in 4px boxes at
# display resolution. All-finite (opaque RGB) strips never see it.
_CHECKER_BOX: int = 4
_CHECKER_LIGHT: tuple[int, int, int] = (249, 250, 251)  # slate-50  (#f9fafb)
_CHECKER_DARK: tuple[int, int, int] = (229, 231, 235)  # slate-200 (#e5e7eb)


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
    if view.page == "main":
        return {"": _render_main_frame(view, session)}
    if view.page == "weights":
        return {"": _render_weights_frame(view, session)}
    if view.page == "watch_histogram":
        return {"": _render_histogram_frame(view, session)}
    if view.page == "watch_minmax":
        return _render_minmax_frames(view, session)
    if view.page == "experiment":
        return {"": _render_experiment_frame(view, session)}
    raise ValueError(f"unknown recorded view page {view.page!r}")


def _compose_captioned_columns(
    legend: Image.Image | None, columns: list[tuple[Image.Image, str]]
) -> Image.Image | None:
    """Lay out a legend plus captioned column images into one PIL frame.

    Shared by the activation strips (`_strip_section`) and the MIN/MAX patch
    grids (`_patch_grid_section`): the optional `legend` leads the row under a
    blank caption-height band, then each column image is placed left to right
    with its caption (already collapsed to fit) centered above it. Columns are
    accumulated until the row would exceed `MAX_FRAME_SIZE`, the same width cap
    the old single-image path used.
    """
    from nansense.ui.render import LABEL_HEIGHT

    if not columns:
        return None
    gap = 2
    x = legend.width + gap if legend is not None else 0
    body_height = legend.height if legend is not None else 0
    placements: list[tuple[Image.Image, str, int]] = []
    for img, label in columns:
        if x >= MAX_FRAME_SIZE:
            break
        placements.append((img, label, x))
        body_height = max(body_height, img.height)
        x += img.width + gap
    total_width = min(x, MAX_FRAME_SIZE)
    canvas = Image.new("RGB", (total_width, LABEL_HEIGHT + body_height), (255, 255, 255))
    if legend is not None:
        canvas.paste(legend, (0, LABEL_HEIGHT))
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    for img, label, col_x in placements:
        if label:
            text_w = draw.textlength(label, font=font)
            draw.text(
                (col_x + max(0, (img.width - text_w) / 2), 1),
                label,
                fill=_LABEL_COLOR,
                font=font,
            )
        canvas.paste(img, (col_x, LABEL_HEIGHT))
    return canvas


def _strip_section(strip: object) -> Image.Image | None:
    """Decode a `StripRender` to one display-resolution PIL image.

    Each tile is nearest-upscaled to its CSS display size (matching the
    browser's `image-rendering: pixelated`) and laid out left to right after
    the crisp legend by `_compose_captioned_columns`, with its column caption
    drawn above it — reproducing the captioned columns the page shows.

    An RGBA tile carries transparent NaN/±Inf cells: it is composited over a
    baked gray checkerboard the same size as the upscaled tile (`_checkerboard`),
    so recorded frames show the same GIMP-style backdrop the live UI paints
    with CSS. Opaque RGB tiles keep the plain path.
    """
    from nansense.ui.render import StripRender

    if not isinstance(strip, StripRender) or not strip.tiles:
        return None
    legend = Image.open(io.BytesIO(strip.legend_image)).convert("RGB")
    columns: list[tuple[Image.Image, str]] = []
    for tile in strip.tiles:
        decoded = Image.open(io.BytesIO(tile.image))
        if decoded.mode == "RGBA":
            up = decoded.resize((tile.width, tile.height), Image.Resampling.NEAREST)
            up = Image.alpha_composite(
                _checkerboard(tile.width, tile.height), up
            ).convert("RGB")
        else:
            up = decoded.convert("RGB").resize(
                (tile.width, tile.height), Image.Resampling.NEAREST
            )
        columns.append((up, tile.label))
    return _compose_captioned_columns(legend, columns)


def _patch_grid_section(grid: object) -> Image.Image | None:
    """Decode a `PatchGridRender` to one display-resolution PIL image.

    Each channel's cells are nearest-upscaled to their CSS square and stacked
    with a `PATCH_CELL_GAP` gutter into a column image, then laid out after the
    optional heat legend by `_compose_captioned_columns` under a "CHANNEL N"
    caption — the recording mirror of the MIN/MAX view's captioned cell grid.
    """
    from nansense.ui.render import PATCH_CELL_GAP, PatchGridRender

    if not isinstance(grid, PatchGridRender) or not grid.columns:
        return None
    legend = (
        Image.open(io.BytesIO(grid.heat_legend)).convert("RGB")
        if grid.heat_legend is not None
        else None
    )
    columns: list[tuple[Image.Image, str]] = []
    for column in grid.columns:
        size = column.cell_size
        cell_imgs = [
            Image.open(io.BytesIO(cell))
            .convert("RGB")
            .resize((size, size), Image.Resampling.NEAREST)
            for cell in column.cells
        ]
        if not cell_imgs:
            continue
        height = len(cell_imgs) * size + (len(cell_imgs) - 1) * PATCH_CELL_GAP
        stack = Image.new("RGB", (size, height), (255, 255, 255))
        y = 0
        for cell_img in cell_imgs:
            stack.paste(cell_img, (0, y))
            y += size + PATCH_CELL_GAP
        columns.append((stack, column.label))
    return _compose_captioned_columns(legend, columns)


def _checkerboard(width: int, height: int) -> Image.Image:
    """An opaque `_CHECKER_BOX`-square gray checkerboard, `width × height` RGBA.

    Mirrors the live UI's CSS backdrop so a recorded NaN/±Inf cell shows the
    same two slate grays. Built vectorised: the box-parity of each pixel's
    `(row, col)` picks the light/dark color.
    """
    ys = (np.arange(height) // _CHECKER_BOX)[:, None]
    xs = (np.arange(width) // _CHECKER_BOX)[None, :]
    dark = (ys + xs) % 2 == 1
    rgb = np.empty((height, width, 3), dtype=np.uint8)
    rgb[...] = _CHECKER_LIGHT
    rgb[dark] = _CHECKER_DARK
    rgba = np.concatenate(
        [rgb, np.full((height, width, 1), 255, dtype=np.uint8)], axis=-1
    )
    return Image.fromarray(rgba, mode="RGBA")


def _compose_sections(
    sections: list[tuple[str, Image.Image | None]],
) -> np.ndarray | None:
    """Stack labelled images vertically onto one white frame."""
    if not sections:
        return None
    font = ImageFont.load_default()
    label_height = 14
    width = _FRAME_PAD * 2 + min(
        MAX_FRAME_SIZE,
        max(
            [img.width for _, img in sections if img is not None]
            + [320],
        ),
    )
    height = _FRAME_PAD
    for label, img in sections:
        if label:
            height += label_height + 2
        if img is not None:
            height += min(img.height, MAX_FRAME_SIZE) + _SECTION_GAP
        else:
            height += _SECTION_GAP
    height = min(height + _FRAME_PAD, MAX_FRAME_SIZE)
    canvas = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    y = _FRAME_PAD
    for label, img in sections:
        if label:
            draw.text((_FRAME_PAD, y), label, fill=_LABEL_COLOR, font=font)
            y += label_height + 2
        if img is not None:
            canvas.paste(img, (_FRAME_PAD, y))
            y += min(img.height, MAX_FRAME_SIZE) + _SECTION_GAP
        else:
            y += _SECTION_GAP
    return np.asarray(canvas)


def _render_main_frame(view: RecordedView, session: Session) -> np.ndarray | None:
    """Input image plus per-layer activation/gradient strips, one frame."""
    from nansense.ui.render import (
        probe_act_tensor,
        render_image,
        render_strip,
        tensor_hw,
    )

    snap = session.snapshot
    probe = session.probe_result
    if snap is None and probe is None:
        return None
    layers = str_tuple(view.params.get("layers"))
    sample_idx = int_param(view.params, "sample_idx")
    mean = float_tuple(view.params.get("input_mean"))
    std = float_tuple(view.params.get("input_std"))
    transform = cast(InputTransform | None, view.params.get("input_transform"))
    input_name = str(view.params.get("input_name") or "") or None
    compare = bool(session.perturbations)

    sections: list[tuple[str, Image.Image | None]] = []
    if probe is not None:
        shown_input = probe.shown_input(input_name)
        input_hw = tensor_hw(probe.base_input(input_name))
    else:
        assert snap is not None
        shown_input = (
            snap.activations.get(input_name) if input_name is not None else None
        )
        input_hw = tensor_hw(shown_input)
    input_img = render_image(
        shown_input, sample_idx, mean=mean, std=std, transform=transform
    )
    if input_img is not None:
        img = Image.open(io.BytesIO(input_img)).convert("RGB")
        from nansense.ui.render import INPUT_IMAGE_SIZE

        scale = INPUT_IMAGE_SIZE / max(img.width, 1)
        img = img.resize(
            (INPUT_IMAGE_SIZE, max(1, round(img.height * scale))), Image.Resampling.NEAREST
        )
        sections.append(("input", img))
    for name in layers:
        if probe is not None:
            act = probe_act_tensor(probe, name, compare=compare)
            act_img = _strip_section(
                render_strip(act, sample_idx, input_hw=input_hw)
            )
            sections.append((f"{name} — activations (probe)", act_img))
        else:
            assert snap is not None
            act_img = _strip_section(
                render_strip(
                    snap.activations.get(name), sample_idx, input_hw=input_hw
                )
            )
            grad_img = _strip_section(
                render_strip(
                    snap.activation_gradients.get(name),
                    sample_idx,
                    input_hw=input_hw,
                )
            )
            sections.append((f"{name} — activations", act_img))
            sections.append((f"{name} — gradients", grad_img))
    return _compose_sections(sections)


def _render_weights_frame(view: RecordedView, session: Session) -> np.ndarray | None:
    """One layer's weight / gradient / optimizer strips under frozen axes."""
    from nansense.ui.render import default_weight_dims, dims_from_roles, render_weight

    snap = session.snapshot
    if snap is None:
        return None
    panels = view.params.get("panels")
    if not isinstance(panels, (list, tuple)):
        return None
    sections: list[tuple[str, Image.Image | None]] = []
    for spec in panels:
        # Each spec is `(name, roles, index pairs)` — see the weights page's
        # record-view factory.
        if not (isinstance(spec, (list, tuple)) and len(spec) == 3):
            continue
        name = str(spec[0])
        roles = [str(r) for r in str_tuple(spec[1])]
        fixed: dict[int, int] = {}
        if isinstance(spec[2], (list, tuple)):
            for pair in spec[2]:
                if isinstance(pair, (list, tuple)) and len(pair) == 2:
                    dim, idx = pair
                    if isinstance(dim, int) and isinstance(idx, int):
                        fixed[dim] = idx
        tensor = snap.weights.get(name)
        if tensor is None:
            sections.append((f"{name} — no weights captured", None))
            continue
        x_dim, y_dim, tile_dim = dims_from_roles(roles)
        if x_dim is None:
            x_dim = tensor.ndim - 1
        tile = tile_dim if y_dim is not None else None
        sections.append(
            (
                f"{name} — weight",
                _strip_section(
                    render_weight(
                        tensor, x_dim=x_dim, y_dim=y_dim, tile_dim=tile, fixed=fixed
                    )
                ),
            )
        )
        grad = snap.weight_gradients.get(name)
        if grad is not None:
            sections.append(
                (
                    f"{name} — gradient",
                    _strip_section(
                        render_weight(
                            grad, x_dim=x_dim, y_dim=y_dim, tile_dim=tile, fixed=fixed
                        )
                    ),
                )
            )
        scalar_parts: list[str] = []
        for key, state in sorted(snap.optimizer_state.get(name, {}).items()):
            if state.ndim == 0:
                scalar_parts.append(f"{key} = {float(state):.4g}")
                continue
            if tuple(state.shape) == tuple(tensor.shape):
                strip = render_weight(
                    state, x_dim=x_dim, y_dim=y_dim, tile_dim=tile, fixed=fixed
                )
            else:
                dims = default_weight_dims(state.ndim)
                strip = render_weight(
                    state,
                    x_dim=dims.x_dim,
                    y_dim=dims.y_dim,
                    tile_dim=dims.tile_dim,
                    fixed={},
                )
            sections.append((f"{name} — {key}", _strip_section(strip)))
        scalar_parts += [
            f"{key} = {value:.4g}"
            for key, value in sorted(
                snap.optimizer_hyperparams.get(name, {}).items()
            )
        ]
        if scalar_parts:
            sections.append(("  ·  ".join(scalar_parts), None))
    sections.insert(0, (format_position(snap.position), None))
    return _compose_sections(sections)


# Matplotlib histogram geometry: inches per subplot row at `_HIST_DPI`.
_HIST_DPI: int = 100
_HIST_ROW_INCHES: float = 2.4
_HIST_WIDTH_INCHES: float = 9.0


def _render_histogram_frame(
    view: RecordedView, session: Session
) -> np.ndarray | None:
    """Matplotlib re-render of the frozen layers' histograms (one phase)."""
    layers = str_tuple(view.params.get("layers"))
    phase = str(view.params.get("phase") or "")
    log_x = bool(view.params.get("log_x"))
    log_y = bool(view.params.get("log_y"))
    if not layers:
        return None
    snap = session._watch_accumulator.snapshot(
        layers=layers, include_patches=False
    )
    rows: list[tuple[str, str, LayerStatsSnapshot]] = []
    for layer in layers:
        per_phase = snap.latest_per_phase(layer)
        stats = per_phase.get(phase)
        if stats is None:
            continue
        for kind in ("activation", "gradient"):
            rows.append((layer, kind, stats))
    if not rows:
        return None
    fig = Figure(
        figsize=(_HIST_WIDTH_INCHES, _HIST_ROW_INCHES * len(rows)), dpi=_HIST_DPI
    )
    axes = fig.subplots(len(rows), 1, squeeze=False)
    for ax_row, (layer, kind, stats) in zip(axes, rows, strict=True):
        _draw_histogram_axes(
            ax_row[0], layer, kind, phase, stats, log_x=log_x, log_y=log_y
        )
    fig.tight_layout()
    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    return np.asarray(canvas.buffer_rgba())[..., :3].copy()


def _draw_histogram_axes(
    ax: Axes,
    layer: str,
    kind: str,
    phase: str,
    stats: LayerStatsSnapshot,
    *,
    log_x: bool,
    log_y: bool,
) -> None:
    """One subplot: the same bars/ranges the watch page draws with Plotly."""
    from nansense.ui.histograms import (
        _OVERFLOW_MARKER_COLOR,
        BIN_CENTERS,
        BIN_WIDTHS,
        _overflow_marks,
        axis_ranges,
        kind_stats,
        phase_color,
        trace_heights,
        use_density,
        x_tick_layout,
    )

    tensor_stats: TensorStatsSnapshot = kind_stats(stats, kind)
    density = use_density(log_x)
    # A collapsed bucket (epoch-evicted bins) renders as an empty histogram.
    hist = tensor_stats.hist if tensor_stats.hist is not None else (0,) * N_BINS
    heights = trace_heights(hist, density)
    color = phase_color(phase, 0)
    x_values = list(range(N_BINS)) if log_x else list(BIN_CENTERS)
    if log_x:
        ax.bar(x_values, heights, width=1.0, color=color)
        tick_vals, tick_text = x_tick_layout()
        ax.set_xticks(tick_vals, tick_text, fontsize=6)
    else:
        ax.bar(x_values, heights, width=BIN_WIDTHS, color=color)
    per_phase = {phase: stats}
    x_range, y_range = axis_ranges(per_phase, kind, log_x=log_x, log_y=log_y)
    if x_range is not None:
        ax.set_xlim((x_range[0], x_range[1]))
    if log_y:
        ax.set_yscale("log")
    elif y_range is not None:
        ax.set_ylim((y_range[0], y_range[1]))
        # Flag bars clipped by the cap so they don't read as ending at the top
        # edge (mirrors the Plotly view's overflow markers).
        (mark_xs, mark_ys), = _overflow_marks(
            [(phase, hist)], x_values, density, y_range[1]
        )
        if mark_xs:
            ax.scatter(
                mark_xs, mark_ys, marker="^", s=18,
                color=_OVERFLOW_MARKER_COLOR, edgecolors="white",
                linewidths=0.5, zorder=3, clip_on=False,
            )
    title = (
        f"{layer} — {kind}s · {phase} (ep {stats.epoch}) · "
        f"n={tensor_stats.n:,} mean={tensor_stats.mean:.3g} "
        f"std={tensor_stats.std:.3g}"
    )
    ax.set_title(title, fontsize=8)
    ax.tick_params(labelsize=6)


# Which video file each patch grid records into: crops ("pixel" grids) and
# whole-input grids ("average") have different cell sizes.
_PATCH_GROUPS: dict[PatchType, str] = {
    "max_pixel": "pixel",
    "min_pixel": "pixel",
    "max_average": "average",
    "min_average": "average",
}


def _render_minmax_frames(
    view: RecordedView, session: Session
) -> dict[str, np.ndarray | None]:
    """The frozen patch grids, split into pixel/average video streams."""
    from nansense.ui.render import render_patch_grid

    layers = str_tuple(view.params.get("layers"))
    phase = str(view.params.get("phase") or "")
    enabled = [t for t in PATCH_TYPES if t in str_tuple(view.params.get("grids"))]
    heatmap = bool(view.params.get("heatmap"))
    mean = float_tuple(view.params.get("input_mean"))
    std = float_tuple(view.params.get("input_std"))
    snap = session._watch_accumulator.snapshot(layers=layers, include_patches=True)
    sections: dict[str, list[tuple[str, Image.Image | None]]] = {}
    for layer in layers:
        stats = snap.latest_per_phase(layer).get(phase)
        if stats is None or stats.patches is None:
            continue
        patches = stats.patches
        for ptype in enabled:
            tp = patches.by_type.get(ptype)
            if tp is None:
                continue
            grid = render_patch_grid(tp, mean=mean, std=std, heatmap=heatmap)
            if grid is None:
                continue
            img = _patch_grid_section(grid)
            if img is None:
                continue
            group = _PATCH_GROUPS[ptype]
            sections.setdefault(group, []).append(
                (f"{layer} — {ptype} · {phase} (ep {stats.epoch})", img)
            )
    return {
        group: _compose_sections(group_sections)
        for group, group_sections in sections.items()
    }


def _render_experiment_frame(
    view: RecordedView, session: Session
) -> np.ndarray | None:
    """The freshest result of the view's pinned auto experiment."""
    from nansense.ui.render import render_strip, tensor_hw

    seq = int_param(view.params, "seq")
    mean = float_tuple(view.params.get("input_mean"))
    std = float_tuple(view.params.get("input_std"))
    result = session.experiment_result_for(seq)
    if result is None:
        return None
    sections: list[tuple[str, Image.Image | None]] = []
    status = f"{result.kind} · {result.layer} · step {result.step}/{result.total_steps}"
    if result.objective is not None:
        status += f" · objective {result.objective:.4g}"
    if result.error is not None:
        status += f" · error: {result.error}"
    sections.append((status, None))
    if result.image is not None:
        sections.append(
            ("result", _batch_image_row(result.image, mean=mean, std=std))
        )
    if result.attribution is not None:
        sections.append(
            (
                "attribution",
                _strip_section(
                    render_strip(
                        result.attribution, 0, input_hw=tensor_hw(result.reference)
                    )
                ),
            )
        )
    if result.reference is not None:
        sections.append(
            ("input", _batch_image_row(result.reference, mean=mean, std=std))
        )
    return _compose_sections(sections)


def _batch_image_row(
    tensor: Tensor,
    *,
    mean: tuple[float, ...] | None,
    std: tuple[float, ...] | None,
) -> Image.Image | None:
    """Every sample of `tensor` as one horizontal row of upscaled images."""
    from nansense.ui.render import INPUT_IMAGE_SIZE, render_image

    images: list[Image.Image] = []
    for i in range(int(tensor.shape[0])):
        data = render_image(tensor, i, mean=mean, std=std)
        if data is None:
            continue
        img = Image.open(io.BytesIO(data)).convert("RGB")
        scale = INPUT_IMAGE_SIZE / max(img.width, 1)
        images.append(
            img.resize(
                (INPUT_IMAGE_SIZE, max(1, round(img.height * scale))),
                Image.Resampling.NEAREST,
            )
        )
    if not images:
        return None
    gap = 6
    width = sum(img.width for img in images) + gap * (len(images) - 1)
    height = max(img.height for img in images)
    canvas = Image.new("RGB", (min(width, MAX_FRAME_SIZE), height), (255, 255, 255))
    x = 0
    for img in images:
        canvas.paste(img, (x, 0))
        x += img.width + gap
    return canvas
