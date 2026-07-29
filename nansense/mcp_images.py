"""Rendered views as PNG bytes, for the MCP server (`nansense.mcp_server`).

The counterpart of `nansense.mcp_views`: where that module turns a `Session`
into numbers an agent reads, this one turns it into pictures an agent looks at.
Both answer the same question about the same session — "what is this layer
doing?" — and an agent generally wants both, because the statistics say *how
much* and the strip says *where*.

The pictures themselves come from `nansense.ui.frames`, the same renderer the
video recordings use, so what an agent sees is what the page shows. Everything
added here is about the wire:

- **PNG, always.** Strips are encoded as BMP for the browser (near-memcpy on
  localhost, at ~2× the bytes); an MCP reply is base64 inside JSON, where those
  bytes are paid for twice over.
- **Bounded size.** A wide layer composes into a picture thousands of pixels
  across. Past `MAX_SIDE` it is area-downscaled, and the note says so — an agent
  that is shown a shrunken strip without being told would read the smoothing as
  data.
- **A reason, never a blank.** Every function returns a `RenderedImage` whose
  `note` explains an absent picture (nothing captured yet, layer not watched,
  a 4-D activation with no 2-D view), because "no image" and "an image of
  nothing" are the same thing on the wire and very different to a reader.

Import `nansense.ui.*` lazily here, inside the functions: this module is
imported by `nansense.mcp_server`, which `nansense.ui.app` imports while
`nansense.ui.__init__` is still running.
"""

from __future__ import annotations

import io
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from nansense.input_config import InputTransform, MeanStd, resolve_per_input
from nansense.mcp_views import position_view
from nansense.patches import PATCH_TYPES
from nansense.session import Session

if TYPE_CHECKING:
    from PIL.Image import Image

#: Longest side an image is sent at. Larger pictures are area-downscaled: a
#: vision model resamples anything bigger anyway, and the bytes are the agent's
#: to pay for.
MAX_SIDE: int = 1568

# How many weight axes `render_weights`' `index` covers. Three are always
# assigned to x/y/tile, so this bounds the rank whose remaining axes can be
# pinned — comfortably past the 4-D conv weights that motivate the knob.
_MAX_WEIGHT_RANK: int = 8


@dataclass(frozen=True)
class RenderedImage:
    """One rendered view: the PNG bytes, or `None` plus a note saying why not.

    `note` is present either way — alongside a picture it carries the position
    and any caveat (a downscale, a truncated channel row), and in place of one
    it says what was missing.
    """

    png: bytes | None
    note: str


class InputDisplay:
    """How input tensors are turned into displayable images.

    `serve` receives `input_mean` / `input_std` / `input_transform` from the
    training script — each either one value for every input or a dict keyed by
    input name — and the pages resolve them per input they show. The MCP tools
    need the same resolution, so it lives here rather than being re-derived at
    every call site.
    """

    def __init__(
        self,
        *,
        mean: MeanStd | dict[str, MeanStd] | None = None,
        std: MeanStd | dict[str, MeanStd] | None = None,
        transform: InputTransform | dict[str, InputTransform] | None = None,
    ) -> None:
        self._mean = mean
        self._std = std
        self._transform = transform

    def stats(self, name: str | None) -> tuple[MeanStd | None, MeanStd | None]:
        """The `(mean, std)` denormalization pair for input `name`."""
        return resolve_per_input(self._mean, name), resolve_per_input(self._std, name)

    def transform(self, name: str | None) -> InputTransform | None:
        return resolve_per_input(self._transform, name)


def _encode(image: Image | None) -> tuple[bytes | None, str]:
    """PNG bytes for a composed image, downscaled past `MAX_SIDE`.

    The caveat comes back as text rather than being applied silently: a
    downscaled strip has had neighbouring channels averaged together, and a
    reader who thinks they are looking at raw pixels will misread it.
    """
    if image is None:
        return None, ""
    from PIL import Image as PILImage

    caveat = ""
    longest = max(image.width, image.height)
    if longest > MAX_SIDE:
        scale = MAX_SIDE / longest
        size = (max(1, round(image.width * scale)), max(1, round(image.height * scale)))
        # BOX averages the pixels it merges; NEAREST would drop whole channels
        # out of a strip rather than dimming them.
        image = image.resize(size, PILImage.Resampling.BOX)
        caveat = (
            f" The view is {longest}px on its longest side, so it was scaled "
            f"down to {MAX_SIDE}px — fine detail is averaged away; ask for "
            "fewer layers, or read the numbers with get_layer_stats, to see it."
        )
    buffer = io.BytesIO()
    image.save(buffer, format="PNG", compress_level=1)
    return buffer.getvalue(), caveat


def _position_note(session: Session) -> str:
    """Which batch the picture describes — never assume it is the live one."""
    snapshot = session.snapshot
    if snapshot is None:
        return "no batch captured"
    view = position_view(snapshot.position, schedule=session.schedule)
    assert view is not None
    return str(view["text"])


def _no_snapshot() -> RenderedImage:
    return RenderedImage(
        None,
        "No batch has been captured yet, so there is nothing to draw. A session "
        "pauses on its first batch; if training is running free, call pause() "
        "or refresh() first.",
    )


def _unknown_layers(session: Session, layers: Sequence[str]) -> list[str]:
    known = set(session.layer_names)
    return [name for name in layers if name not in known]


def layer_image(
    session: Session,
    *,
    layers: Sequence[str],
    sample: int = 0,
    display: InputDisplay,
    input_name: str | None = None,
    include_input: bool = False,
) -> RenderedImage:
    """The main page for `layers`: one activation and gradient strip each.

    Each strip is a row of per-channel tiles on a shared symmetric color scale
    (red positive, blue negative), with NaN/±Inf cells left transparent over a
    checkerboard — so a diverged channel is visible as a hole rather than as an
    extreme color.
    """
    from nansense.ui.frames import main_frame

    if session.snapshot is None and session.probe_result is None:
        return _no_snapshot()
    unknown = _unknown_layers(session, layers)
    resolved = [name for name in layers if name not in unknown]
    mean, std = display.stats(input_name)
    png, caveat = _encode(
        main_frame(
            session,
            layers=resolved,
            sample_idx=sample,
            input_name=input_name if include_input or not resolved else None,
            mean=mean,
            std=std,
            transform=display.transform(input_name),
        )
    )
    note = f"Sample {sample} of the batch at {_position_note(session)}."
    if session.probe_result is not None:
        note += (
            " Rendered from the probe (a pinned batch or perturbation is "
            "active), which is forward-only — so there are no gradient strips."
        )
    if unknown:
        note += f" Unknown layers, skipped: {unknown}."
    if png is None:
        return RenderedImage(
            None,
            note
            + " Nothing renderable: the layers captured no activation on this "
            "batch, or their per-sample shape has no 2-D view (4-D and beyond).",
        )
    return RenderedImage(png, note + caveat)


def input_image(
    session: Session,
    *,
    sample: int = 0,
    display: InputDisplay,
    input_name: str | None = None,
) -> RenderedImage:
    """The input pane: one sample of the model's input, as the pages show it.

    Denormalized with the `input_mean` / `input_std` the training script passed
    to `serve`; without them the values are taken to be in `[0, 1]` already, so
    a normalized input looks washed out.
    """
    from nansense.ui.frames import main_frame

    if session.snapshot is None and session.probe_result is None:
        return _no_snapshot()
    mean, std = display.stats(input_name)
    png, caveat = _encode(
        main_frame(
            session,
            layers=(),
            sample_idx=sample,
            input_name=input_name,
            mean=mean,
            std=std,
            transform=display.transform(input_name),
        )
    )
    if png is None:
        return RenderedImage(
            None,
            f"Input {input_name!r} has no displayable image for sample {sample}: "
            "either the sample index is out of range, or the input is neither "
            "1-/3-channel image-like nor covered by an `input_transform` "
            "(see nansense.serve).",
        )
    note = f"Input {input_name!r}, sample {sample}, at {_position_note(session)}."
    if mean is None or std is None:
        note += (
            " No input_mean/input_std was given to serve(), so the values are "
            "shown as if already in [0, 1]."
        )
    return RenderedImage(png, note + caveat)


def weights_image(
    session: Session,
    *,
    layer: str,
    parameters: Sequence[str] | None = None,
    x_dim: int | None = None,
    y_dim: int | None = None,
    tile_dim: int | None = None,
    index: int = 0,
) -> RenderedImage:
    """The weights page for one layer: weight, gradient and optimizer strips.

    By default each parameter takes the layout the page opens with: conv
    kernels as `kH×kW` tiles across the input-channel axis with the output
    channel pinned, a 2-D linear weight as one `[out, in]` image. `x_dim` /
    `y_dim` / `tile_dim` override that (the page's axis picker), and `index`
    pins every axis they leave over — for a 4-D conv weight that is the output
    channel, so `index` is how you page through filters.
    """
    from nansense.ui.frames import WeightPanel, weights_frame

    snapshot = session.snapshot
    if snapshot is None:
        return _no_snapshot()
    available = session.layer_weights.get(layer, [])
    if not available:
        return RenderedImage(
            None,
            f"Layer {layer!r} has no parameters to draw"
            + (
                "."
                if layer in set(session.layer_names)
                else " — in fact it is not a known layer; call get_architecture."
            ),
        )
    names = list(parameters) if parameters else available
    unknown = [name for name in names if name not in available]
    names = [name for name in names if name in available]
    if not names:
        return RenderedImage(
            None,
            f"None of {unknown} are parameters of {layer!r}; it has {available}.",
        )
    # `index` pins whatever axes the layout leaves over; `render_weight` clamps
    # an out-of-range one, so a too-large index shows the last slice rather
    # than failing. Every axis gets it — only the unassigned ones are read.
    fixed = dict.fromkeys(range(_MAX_WEIGHT_RANK), index)
    png, caveat = _encode(
        weights_frame(
            session,
            panels=[
                WeightPanel(
                    name=name,
                    x_dim=x_dim,
                    y_dim=y_dim,
                    tile_dim=tile_dim,
                    fixed=fixed,
                )
                for name in names
            ],
        )
    )
    note = f"{layer} — {', '.join(names)}, at {_position_note(session)}."
    if index:
        note += f" Axes not shown are pinned to index {index}."
    if unknown:
        note += f" Not parameters of this layer, skipped: {unknown}."
    if png is None:
        return RenderedImage(
            None, note + " The snapshot captured no weights for them."
        )
    return RenderedImage(png, note + caveat)


def histogram_image(
    session: Session,
    *,
    layers: Sequence[str],
    phase: str | None = None,
    log_x: bool = False,
    log_y: bool = False,
) -> RenderedImage:
    """The stats page's histograms: value distributions per layer and phase.

    Reads the running accumulators, so it only covers watched layers — the same
    restriction `get_stats_history` has, and the same fix (`watch_layers`).
    """
    from nansense.ui.frames import histogram_frame

    unknown = _unknown_layers(session, layers)
    resolved = [name for name in layers if name not in unknown]
    without_stats = [name for name in resolved if not session.stats_phases(name)]
    collected = [name for name in resolved if name not in without_stats]
    if not collected:
        return RenderedImage(
            None,
            f"No running statistics for {list(layers)}. Histograms come from the "
            "watch accumulators: call watch_layers (or set_stats_scope('all')) "
            "and let training advance at least one batch."
            + (f" Unknown layers: {unknown}." if unknown else ""),
        )
    chosen = phase
    if chosen is None:
        # Without a phase the newest one with data is the useful default; a
        # picture of an empty phase is the one answer nobody wants.
        available = sorted(session.stats_phases(collected[0]))
        chosen = available[-1] if available else ""
    png, caveat = _encode(
        histogram_frame(
            session, layers=collected, phase=chosen, log_x=log_x, log_y=log_y
        )
    )
    note = (
        f"Activation and gradient histograms for {collected}, phase {chosen!r}"
        f"{' (log x)' if log_x else ''}{' (log y)' if log_y else ''}."
    )
    if without_stats:
        note += f" Not collecting statistics, skipped: {without_stats}."
    if unknown:
        note += f" Unknown layers, skipped: {unknown}."
    if png is None:
        return RenderedImage(
            None,
            note + f" No bucket for phase {chosen!r}; phases with data: "
            f"{sorted(session.stats_phases(collected[0]))}.",
        )
    return RenderedImage(png, note + caveat)


def patches_image(
    session: Session,
    *,
    layer: str,
    phase: str | None = None,
    grids: Sequence[str] = PATCH_TYPES,
    heatmap: bool = False,
    display: InputDisplay,
    input_name: str | None = None,
) -> RenderedImage:
    """The stats page's MIN/MAX grids: the inputs that most excite each channel.

    Columns are channels, rows the top samples for that channel — the picture
    that answers "what has this unit learned to look for". `heatmap` blends the
    channel's activation map over each patch.
    """
    from nansense.ui.frames import patch_frames

    if layer not in set(session.layer_names):
        return RenderedImage(
            None,
            f"Unknown layer {layer!r}. Call get_architecture for the valid names.",
        )
    available = sorted(session.stats_phases(layer))
    if not available:
        return RenderedImage(
            None,
            f"No running statistics for {layer!r}. The patch grids come from the "
            f"watch accumulators: call watch_layers(['{layer}']) and let training "
            "advance at least one batch.",
        )
    chosen = phase if phase is not None else available[-1]
    mean, std = display.stats(input_name)
    frames = patch_frames(
        session,
        layers=[layer],
        phase=chosen,
        grids=grids,
        heatmap=heatmap,
        mean=mean,
        std=std,
    )
    # Crops and whole-input grids have different cell sizes, so they compose
    # separately; an agent asked for one picture, so the crops win when both
    # exist — they are the grids the page shows by default.
    image = frames.get("pixel") or frames.get("average")
    png, caveat = _encode(image)
    if png is None:
        return RenderedImage(
            None,
            f"No patch grids for {layer!r} in phase {chosen!r}. Grids need an "
            "image-like input to crop from, and the average grids also need "
            "set_watch_performance(average_patches=True). Phases with data: "
            f"{available}.",
        )
    note = (
        f"Extreme-input patch grids for {layer}, phase {chosen!r} — columns are "
        "channels, rows the top samples for that channel (best first)."
    )
    if heatmap:
        note += " The channel's activation map is blended over each patch."
    return RenderedImage(png, note + caveat)


def experiment_image(
    session: Session,
    *,
    seq: int,
    display: InputDisplay,
    input_name: str | None = None,
) -> RenderedImage:
    """The experiment page: the freshest result published for request `seq`."""
    from nansense.ui.frames import experiment_frame

    result = session.experiment_result_for(seq)
    if result is None:
        return RenderedImage(
            None,
            f"No result published for experiment {seq} — it may still be queued "
            "(experiments run on the paused training thread), or old enough to "
            "have been evicted.",
        )
    mean, std = display.stats(input_name)
    png, caveat = _encode(experiment_frame(session, seq=seq, mean=mean, std=std))
    note = (
        f"{result.kind} on {result.layer}, step {result.step}/{result.total_steps}"
        f"{'' if result.done else ' (still running)'}."
    )
    if result.error is not None:
        note += f" Error: {result.error}"
    if png is None:
        return RenderedImage(None, note + " Nothing renderable in this result.")
    return RenderedImage(png, note + caveat)


def image_reply(rendered: RenderedImage) -> list[Any]:
    """A `RenderedImage` as the content blocks an MCP tool returns.

    Text first, then the picture: the note says which batch is being shown and
    whether it was scaled, and both belong before the image rather than after
    it. A failed render is text alone — the tool still answers, it just answers
    in words.
    """
    from mcp.server.mcpserver import Image as McpImage

    if rendered.png is None:
        return [rendered.note]
    return [rendered.note, McpImage(data=rendered.png, format="png")]
