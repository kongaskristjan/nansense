"""One composed still image per view — the server-side render of a page.

Each function here answers "what does the `/`, `/weights`, `/stats` … view look
like right now?" with a single PIL image, assembled from `nansense.ui.render`
pieces by `nansense.ui.compose`. The browser never needs this (it lays the
pieces out itself), but two consumers do:

- `nansense.recording`, which encodes each image as a video frame; and
- `nansense.mcp_images`, which hands it to a coding agent.

They differ only in where the arguments come from — a recording's frozen
`RecordedView.params` versus an MCP tool call — so the rendering lives here once
and both pass plain values in. Keeping it that way is what stops the agent's
picture and the recording's frame from drifting apart from the page's.

Import lazily (see `nansense.ui.compose`): this module reaches the NiceGUI app
through `nansense.ui.__init__`.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from PIL import Image

from nansense.input_config import InputTransform
from nansense.patches import PATCH_TYPES, PatchType
from nansense.schedule import format_position
from nansense.ui.compose import (
    batch_image_row,
    histogram_image,
    patch_grid_image,
    stack_sections,
    strip_image,
    upscaled_image,
)
from nansense.ui.render import (
    default_weight_dims,
    probe_act_tensor,
    render_image,
    render_patch_grid,
    render_strip,
    render_weight,
    tensor_hw,
)
from nansense.watch import LayerStatsSnapshot

if TYPE_CHECKING:
    from nansense.session import Session

#: Which video file / group each patch grid belongs to. Crops ("pixel" grids)
#: and whole-input grids ("average") have different cell sizes, so they cannot
#: share one fixed-size frame.
PATCH_GROUPS: dict[PatchType, str] = {
    "max_pixel": "pixel",
    "min_pixel": "pixel",
    "max_average": "average",
    "min_average": "average",
}

_Section = tuple[str, Image.Image | None]


@dataclass(frozen=True)
class WeightPanel:
    """One parameter's strip on the weights view, under a chosen axis layout.

    `name` is a `named_parameters()` key and the axis roles follow
    `nansense.ui.render.render_weight`. Leaving *all three* axes `None` means
    "no layout chosen" and takes `default_weight_dims` for the tensor's rank —
    conv kernels as `kH×kW` tiles rather than a flattened row. Setting any of
    them selects a layout explicitly, and then a `None` `x_dim` falls back to
    the last axis and a `None` `y_dim` drops the tile axis with it, since a
    strip with no vertical axis is a single heatmap row with nothing to tile.
    """

    name: str
    x_dim: int | None = None
    y_dim: int | None = None
    tile_dim: int | None = None
    fixed: dict[int, int] = field(default_factory=dict)

    def layout(self, ndim: int) -> tuple[int, int | None, int | None]:
        """This panel's `(x_dim, y_dim, tile_dim)` for a rank-`ndim` tensor."""
        if self.x_dim is None and self.y_dim is None and self.tile_dim is None:
            dims = default_weight_dims(ndim)
            return dims.x_dim, dims.y_dim, dims.tile_dim
        x_dim = self.x_dim if self.x_dim is not None else ndim - 1
        return x_dim, self.y_dim, self.tile_dim if self.y_dim is not None else None


def main_frame(
    session: Session,
    *,
    layers: Sequence[str] = (),
    sample_idx: int = 0,
    input_name: str | None = None,
    mean: tuple[float, ...] | None = None,
    std: tuple[float, ...] | None = None,
    transform: InputTransform | None = None,
) -> Image.Image | None:
    """The main page: the input image plus each layer's strips.

    A pinned batch or active perturbations are respected exactly like on the
    page — the probe result becomes the render source (showing
    `perturbed − original` when edits exist), and probe frames carry no gradient
    strips, because probes are forward-only.
    """
    snap = session.snapshot
    probe = session.probe_result
    if snap is None and probe is None:
        return None
    compare = bool(session.perturbations)

    if probe is not None:
        shown_input = probe.shown_input(input_name)
        input_hw = tensor_hw(probe.base_input(input_name))
    else:
        assert snap is not None
        shown_input = (
            snap.activations.get(input_name) if input_name is not None else None
        )
        input_hw = tensor_hw(shown_input)

    sections: list[_Section] = []
    input_img = upscaled_image(
        render_image(shown_input, sample_idx, mean=mean, std=std, transform=transform)
    )
    if input_img is not None:
        sections.append(("input", input_img))
    for name in layers:
        if probe is not None:
            act = probe_act_tensor(probe, name, compare=compare)
            sections.append(
                (
                    f"{name} — activations (probe)",
                    strip_image(render_strip(act, sample_idx, input_hw=input_hw)),
                )
            )
        else:
            assert snap is not None
            sections.append(
                (
                    f"{name} — activations",
                    strip_image(
                        render_strip(
                            snap.activations.get(name), sample_idx, input_hw=input_hw
                        )
                    ),
                )
            )
            sections.append(
                (
                    f"{name} — gradients",
                    strip_image(
                        render_strip(
                            snap.activation_gradients.get(name),
                            sample_idx,
                            input_hw=input_hw,
                        )
                    ),
                )
            )
    return stack_sections(sections)


def weights_frame(
    session: Session, *, panels: Sequence[WeightPanel]
) -> Image.Image | None:
    """The weights page: each panel's weight, gradient and optimizer strips.

    Optimizer state whose shape matches the parameter is drawn under the same
    axis layout; state of a different shape falls back to its own default
    layout, and scalar state (Adam's `step`) joins the hyperparameters on a
    text line, exactly as the page shows them.
    """
    snap = session.snapshot
    if snap is None:
        return None
    sections: list[_Section] = []
    for panel in panels:
        tensor = snap.weights.get(panel.name)
        if tensor is None:
            sections.append((f"{panel.name} — no weights captured", None))
            continue
        x_dim, y_dim, tile_dim = panel.layout(tensor.ndim)
        layout = {
            "x_dim": x_dim,
            "y_dim": y_dim,
            "tile_dim": tile_dim,
            "fixed": panel.fixed,
        }
        sections.append(
            (f"{panel.name} — weight", strip_image(render_weight(tensor, **layout)))
        )
        grad = snap.weight_gradients.get(panel.name)
        if grad is not None:
            sections.append(
                (f"{panel.name} — gradient", strip_image(render_weight(grad, **layout)))
            )
        scalar_parts: list[str] = []
        for key, state in sorted(snap.optimizer_state.get(panel.name, {}).items()):
            if state.ndim == 0:
                scalar_parts.append(f"{key} = {float(state):.4g}")
                continue
            if tuple(state.shape) == tuple(tensor.shape):
                strip = render_weight(state, **layout)
            else:
                dims = default_weight_dims(state.ndim)
                strip = render_weight(
                    state,
                    x_dim=dims.x_dim,
                    y_dim=dims.y_dim,
                    tile_dim=dims.tile_dim,
                    fixed={},
                )
            sections.append((f"{panel.name} — {key}", strip_image(strip)))
        scalar_parts += [
            f"{key} = {value:.4g}"
            for key, value in sorted(snap.optimizer_hyperparams.get(panel.name, {}).items())
        ]
        if scalar_parts:
            sections.append(("  ·  ".join(scalar_parts), None))
    sections.insert(0, (format_position(snap.position), None))
    return stack_sections(sections)


def histogram_frame(
    session: Session,
    *,
    layers: Sequence[str],
    phase: str,
    log_x: bool = False,
    log_y: bool = False,
) -> Image.Image | None:
    """The stats page's HISTOGRAMS view for one phase, redrawn server-side."""
    if not layers:
        return None
    snap = session.watch_snapshot(layers=list(layers), include_patches=False)
    rows: list[tuple[str, str, str, LayerStatsSnapshot]] = []
    for layer in layers:
        stats = snap.latest_per_phase(layer).get(phase)
        if stats is None:
            continue
        for kind in ("activation", "gradient"):
            rows.append((layer, kind, phase, stats))
    return histogram_image(rows, log_x=log_x, log_y=log_y)


def patch_frames(
    session: Session,
    *,
    layers: Sequence[str],
    phase: str,
    grids: Sequence[str] = PATCH_TYPES,
    heatmap: bool = False,
    mean: tuple[float, ...] | None = None,
    std: tuple[float, ...] | None = None,
) -> dict[str, Image.Image | None]:
    """The stats page's MIN/MAX patch grids, keyed by `PATCH_GROUPS` group.

    Pixel grids (input crops) and average grids (whole inputs) have different
    cell sizes, so they compose into separate images rather than one.
    """
    enabled = [ptype for ptype in PATCH_TYPES if ptype in grids]
    snap = session.watch_snapshot(layers=list(layers), include_patches=True)
    sections: dict[str, list[_Section]] = {}
    for layer in layers:
        stats = snap.latest_per_phase(layer).get(phase)
        if stats is None or stats.patches is None:
            continue
        for ptype in enabled:
            tp = stats.patches.by_type.get(ptype)
            if tp is None:
                continue
            img = patch_grid_image(
                render_patch_grid(tp, mean=mean, std=std, heatmap=heatmap)
            )
            if img is None:
                continue
            sections.setdefault(PATCH_GROUPS[ptype], []).append(
                (f"{layer} — {ptype} · {phase} (ep {stats.epoch})", img)
            )
    return {group: stack_sections(rows) for group, rows in sections.items()}


def experiment_frame(
    session: Session,
    *,
    seq: int,
    mean: tuple[float, ...] | None = None,
    std: tuple[float, ...] | None = None,
) -> Image.Image | None:
    """The experiment page: request `seq`'s freshest result, headed by a status
    line (kind, layer, step, objective, error)."""
    result = session.experiment_result_for(seq)
    if result is None:
        return None
    status = f"{result.kind} · {result.layer} · step {result.step}/{result.total_steps}"
    if result.objective is not None:
        status += f" · objective {result.objective:.4g}"
    if result.error is not None:
        status += f" · error: {result.error}"
    sections: list[_Section] = [(status, None)]
    if result.image is not None:
        sections.append(("result", batch_image_row(result.image, mean=mean, std=std)))
    if result.attribution is not None:
        sections.append(
            (
                "attribution",
                strip_image(
                    render_strip(
                        result.attribution, 0, input_hw=tensor_hw(result.reference)
                    )
                ),
            )
        )
    if result.reference is not None:
        sections.append(("input", batch_image_row(result.reference, mean=mean, std=std)))
    return stack_sections(sections)
