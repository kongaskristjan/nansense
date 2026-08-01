"""The `/weights` page: per-layer weight viewer with remappable axes."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from urllib.parse import quote

from nicegui import ui
from torch import Tensor

from nansense.recording import RecordedView
from nansense.session import BatchSnapshot, Session
from nansense.ui.common import (
    _defer_value_write,
    _page_scaffold,
    _set_controls_enabled,
    _strip_html,
    _strip_marker,
    _weights_placeholder,
)
from nansense.ui.histograms import _format_stat
from nansense.ui.render import default_weight_dims, dims_from_roles, render_weight
from nansense.ui.static import _STRIP_MARKER_CSS
from nansense.ui.top_bar import (
    _add_error_banner,
    _add_repo_logo,
    _add_settings_button,
    _add_share_button,
    _add_step_controls,
    _add_tour_button,
    _back_button,
    _build_step_until_custom_dialog,
    _refresh_button,
    _top_bar_row,
)
from nansense.ui.theme import CUSTOM, GRADIENTS, OPTIMIZER, WEIGHT
from nansense.ui.tour import add_tour, weights_tour_steps


_ROLE_LABELS: dict[str, str] = {"x": "X", "y": "Y", "tile": "Tile", "index": "Index"}


def _role_options(ndim: int) -> dict[str, str]:
    """Role choices offered per dimension, scaled to the weight's rank.

    A rank-1 weight can only map an axis to X; rank-2 adds Y; rank-3+ adds the
    tiling axis. Every rank can pin an axis to a single index.
    """
    roles = ["x"]
    if ndim >= 2:
        roles.append("y")
    if ndim >= 3:
        roles.append("tile")
    roles.append("index")
    return {r: _ROLE_LABELS[r] for r in roles}


def _default_roles(ndim: int) -> list[str]:
    """Per-dimension role list matching `render.default_weight_dims`."""
    dims = default_weight_dims(ndim)
    roles = ["index"] * ndim
    roles[dims.x_dim] = "x"
    if dims.y_dim is not None:
        roles[dims.y_dim] = "y"
    if dims.tile_dim is not None:
        roles[dims.tile_dim] = "tile"
    return roles


def _axes_anchor_param(
    weight_names: list[str], shapes: dict[str, tuple[int, ...]]
) -> str | None:
    """The parameter whose panel anchors the tour's axis-mapping step.

    The first parameter with 2+ dimensions wins over e.g. a leading 1-D
    bias: the step explains mapping dimensions to X/Y/Tile/Index, and only
    a multi-dimensional weight has controls where those roles are
    non-trivial. Falls back to the first parameter; None when there are
    none.
    """
    for name in weight_names:
        if len(shapes.get(name, ())) >= 2:
            return name
    return weight_names[0] if weight_names else None


def _weight_graphs_href(layer: str) -> str:
    """Deep-link to `layer`'s GRAPHS view on the Stats page.

    The GRAPHS view renders the running watch aggregates; in the `watched`
    scope those only cover watched layers, so `watch=1` has the stats page
    watch the layer on open and the jump lands on data (collection starts
    with the next stepped batch — the other scopes ignore the flag).
    `scroll=weights` brings the card's Weights section into view once it
    renders.
    """
    return f"/stats?layer={quote(layer)}&view=graphs&scroll=weights&watch=1"


def _build_weights_page(session: Session, layer: str) -> None:
    """Per-layer weight viewer: kernel/image strips with selectable axes.

    Reuses the main page's stepping controls (minus the sample spinner — a
    weight has no batch axis) so the displayed weights track the same paused
    batch. One panel per parameter the layer owns; each panel lets the user
    remap which tensor axes become the X / Y / tiling axes and pins the rest
    by index.
    """
    title = f"Weights · {layer}" if layer else "Weights"
    _page_scaffold(title)
    ui.add_head_html(_STRIP_MARKER_CSS)
    add_tour("weights", weights_tour_steps(), locked=session.locked)

    weight_names = session.layer_weights.get(layer, [])
    wanted = set(weight_names)
    shapes = {
        name: tuple(p.shape)
        for name, p in session.model.named_parameters()
        if name in wanted
    }
    step_until_custom = _build_step_until_custom_dialog(session)
    panels: list[_WeightPanel] = []
    record_key = f"weights:{layer}"

    def record_view() -> RecordedView | None:
        if not panels:
            return None
        return RecordedView(
            key=record_key,
            page="weights",
            label=f"Weights · {layer}",
            params={
                "layer": layer,
                # One (name, roles, indices) spec per panel, frozen at
                # record start; `indices` travels as item pairs so the
                # params stay plain immutable-friendly structures.
                "panels": tuple(
                    (p.name, tuple(p.roles), tuple(p.indices.items()))
                    for p in panels
                ),
            },
        )

    with ui.column().classes("w-full h-screen no-wrap gap-0"):
        with _top_bar_row():
            # The page shows one fixed layer, so the locked Back deep link
            # never needs re-syncing.
            _back_button(layer if session.locked else None)
            _refresh_button(session)
            ui.label(title).classes(
                "font-mono text-base font-bold ml-2 truncate max-w-64"
            )
            _add_step_controls(session, step_until_custom)
            if layer in session.layer_names and weight_names:
                # A real anchor (href, not an `on_click` navigate) so
                # middle-click / ctrl-click open the graphs in a new tab.
                ui.button(
                    "Weight graphs",
                    icon="show_chart",
                    color="yellow-8",
                ).props(
                    f'dense no-caps size=sm href="{_weight_graphs_href(layer)}"'
                ).classes("ml-2").tooltip(
                    "This layer's weight statistics per epoch"
                )
            _add_settings_button(session, record_view).classes("ml-auto")
            _add_tour_button()
            _add_share_button()
            _add_repo_logo()

        _add_error_banner(session)

        with ui.column().classes(
            "w-full grow min-h-0 overflow-auto p-4 gap-4 bg-slate-200"
        ):
            if layer not in session.layer_names:
                _weights_placeholder(f"Unknown layer {layer!r}.")
            elif not weight_names:
                _weights_placeholder(
                    f"Layer {layer!r} has no weights to show."
                )
            else:
                anchor = _axes_anchor_param(weight_names, shapes)
                for name in weight_names:
                    panels.append(
                        _WeightPanel(
                            name=name,
                            shape=shapes[name],
                            session=session,
                            tour_anchor=name == anchor,
                        )
                    )

    async def tick() -> None:
        frozen = session.recording.is_recording(record_key)
        for panel in panels:
            panel.set_frozen(frozen)
        snap = session.snapshot
        if snap is None:
            return
        # Only the panels with a genuinely new snapshot need rendering; that
        # decision is cheap, but the rendering it gates is not, so it runs off
        # the loop. The Refresh button doesn't render here — it asks the
        # training thread to publish the next batch's snapshot, which this tick
        # then picks up like any other (see `Session.request_snapshot`).
        pending = [panel for panel in panels if panel.needs_render(snap)]
        if not pending:
            return
        renders = await asyncio.to_thread(_compute_snapshot_renders, pending, snap)
        for panel, render in zip(pending, renders, strict=True):
            panel.apply_render(render)

    ui.timer(0.2, tick)


def _compute_snapshot_renders(
    panels: list[_WeightPanel], snap: BatchSnapshot
) -> list[_PanelRender]:
    """Render every pending panel against the new snapshot (worker)."""
    return [
        panel.compute_render(
            snap.weights,
            snap.weight_gradients,
            optimizer_state=snap.optimizer_state,
            optimizer_hyperparams=snap.optimizer_hyperparams,
            custom_tensors=snap.custom_weight_tensors.get(panel.name),
        )
        for panel in panels
    ]


# Shown next to the GRADIENT marker before any backward pass has populated
# the parameter's gradient.
_NO_GRADIENT_HTML: str = (
    '<div class="text-xs text-slate-400 italic py-1">no gradient captured yet</div>'
)


@dataclass
class _PanelRender:
    """The result of rendering one panel, computed off the event loop.

    Holds only plain strings/markup — the heavy `render_weight` calls and
    `_strip_html` encoding already happened in the worker thread, so
    `_WeightPanel.apply_render` just writes these onto the UI elements back on
    the loop. `error` (when set) short-circuits to the error display; otherwise
    `weight_html` / `grad_html` / the optimizer fields drive the strips.
    """

    error: str | None = None
    weight_html: str = ""
    grad_html: str = ""
    # One (marker label, strip HTML) pair per tensor-valued optimizer entry.
    opt_strips: list[tuple[str, str]] = field(default_factory=list)
    opt_scalar_text: str = ""
    # One (marker label, strip HTML) pair per custom weight-tensor
    # instrument (`Session.watch_weight_tensor`).
    custom_strips: list[tuple[str, str]] = field(default_factory=list)


class _WeightPanel:
    """One card per parameter — an axis-remappable kernel/image strip.

    The weight's rank is fixed, so the per-dimension role selects and index
    spinners are built once. Changing a role auto-demotes whichever other axis
    held that role (roles X/Y/Tile stay unique), then re-renders against the
    last weights; new snapshots re-render via the page's `tick`, which only
    touches panels whose `needs_render` reports a fresh snapshot.
    """

    def __init__(
        self,
        *,
        name: str,
        shape: tuple[int, ...],
        session: Session,
        tour_anchor: bool = False,
    ) -> None:
        self.name = name
        self._shape = shape
        self._session = session
        self._ndim = len(shape)
        self._roles: list[str] = _default_roles(self._ndim)
        self._indices: dict[int, int] = {d: 0 for d in range(self._ndim)}
        self._last_snapshot: BatchSnapshot | None = None
        self._weights: dict[str, Tensor] | None = None
        self._gradients: dict[str, Tensor] | None = None
        self._opt_state: dict[str, dict[str, Tensor]] = {}
        self._opt_hparams: dict[str, dict[str, float]] = {}
        self._custom: dict[str, Tensor] = {}
        self._role_selects: list[ui.select] = []
        self._index_numbers: dict[int, ui.number] = {}
        self._frozen = False

        options = _role_options(self._ndim)
        with ui.card().classes("w-full p-4 gap-3"):
            with ui.row().classes("w-full items-baseline gap-3 no-wrap"):
                ui.label(name).classes("font-mono text-base font-bold")
                ui.label(f"shape {tuple(shape)}").classes(
                    "font-mono text-xs text-slate-500"
                )
            # `data-tour` marks the axis controls as the tour's arrow target
            # (`tour.weights_tour_steps`); only the anchor panel — the first
            # multi-dimensional parameter (`_axes_anchor_param`) — carries
            # it, so the arrow skips a leading bias's degenerate lone "X"
            # dropdown.
            dim_row = ui.row().classes("items-end gap-4 flex-wrap")
            if tour_anchor:
                dim_row.props('data-tour="axes"')
            with dim_row:
                for d in range(self._ndim):
                    with ui.column().classes("gap-1"):
                        ui.label(f"dim {d} · {shape[d]}").classes(
                            "text-xs text-slate-500 font-mono"
                        )
                        with ui.row().classes("items-center gap-1 no-wrap"):
                            select = ui.select(
                                options=options,
                                value=self._roles[d],
                                on_change=lambda e, d=d: self._on_role(
                                    d, getattr(e, "value", None)
                                ),
                            ).props("dense outlined").classes("w-24").tooltip(
                                "How to lay out this dimension"
                            )
                            self._role_selects.append(select)
                            number = ui.number(
                                value=0,
                                min=0,
                                max=shape[d] - 1,
                                step=1,
                                format="%d",
                                on_change=lambda e, d=d: self._on_index(
                                    d, getattr(e, "value", None)
                                ),
                            ).props("dense outlined").classes("w-20").tooltip(
                                "Which index of this dimension to show"
                            )
                            self._index_numbers[d] = number
            self._error = ui.label("").classes("text-amber-700 text-xs min-h-4")
            # Both strips share one horizontal scrollbar so they pan together,
            # and carry the same kind of labelled marker bars as the
            # activation/gradient pair on the main page's layer cards.
            with ui.element("div").classes("w-full overflow-x-auto"):
                # Same max-content wrapper as the layer cards: every row spans
                # the widest strip so the sticky markers stay in view across
                # the whole scroll range.
                with ui.element("div").classes("w-max min-w-full"):
                    # `data-tour`: the tour's strips arrow lands on the weight
                    # row; its message covers the gradient and optimizer
                    # strips right below (`tour.weights_tour_steps`).
                    with ui.element("div").classes(
                        "flex no-wrap items-stretch"
                    ).props('data-tour="weight-strips"'):
                        _strip_marker(
                            WEIGHT.css,
                            "WEIGHT",
                            header_gap=True,
                            tooltip="The parameter's current values",
                        )
                        self._img = ui.html("")
                    ui.element("div").classes("h-1")
                    with ui.element("div").classes("flex no-wrap items-stretch"):
                        _strip_marker(
                            GRADIENTS.css,
                            "GRADIENT",
                            tooltip="The parameter's gradient",
                        )
                        self._grad_img = ui.html("")
                    # One marker-barred strip per tensor-valued optimizer
                    # state entry (momentum_buffer, exp_avg, …); rebuilt on
                    # each render. Stays empty — invisible — when the session
                    # has no optimizer.
                    self._opt_container = ui.element("div").classes("w-full")
                    # Custom weight-tensor strips follow the optimizer's,
                    # under the same axis layout (the instrument contract
                    # pins their shape to the weight's).
                    self._custom_container = ui.element("div").classes(
                        "w-full"
                    )
            # Scalar optimizer values: 0-dim state entries (Adam's `step`) and
            # the param group's numeric hyperparameters (`lr`, …).
            self._opt_scalars = ui.label("").classes(
                "text-xs text-slate-500 font-mono"
            )
            self._opt_scalars.set_visibility(False)
        self._sync_index_visibility()

    @property
    def roles(self) -> list[str]:
        """The current per-dimension role assignment (for recordings)."""
        return list(self._roles)

    @property
    def indices(self) -> dict[int, int]:
        """The current per-dimension pinned indices (for recordings)."""
        return dict(self._indices)

    def set_frozen(self, frozen: bool) -> None:
        """Disable the axis controls while this layer's weights record."""
        if frozen == self._frozen:
            return
        self._frozen = frozen
        _set_controls_enabled(
            [*self._role_selects, *self._index_numbers.values()], not frozen
        )

    def _on_role(self, dim: int, value: object) -> None:
        role = str(value) if value is not None else "index"
        if role in ("x", "y", "tile"):
            for other in range(self._ndim):
                if other != dim and self._roles[other] == role:
                    self._roles[other] = "index"
        self._roles[dim] = role
        # Defer the select/visibility sync so demotions reach the client.
        _defer_value_write(self._apply_control_state)
        self._render_current()

    def _on_index(self, dim: int, value: float | None) -> None:
        idx = int(value) if value is not None else 0
        self._indices[dim] = max(0, min(idx, self._shape[dim] - 1))
        self._render_current()

    def _apply_control_state(self) -> None:
        for d, select in enumerate(self._role_selects):
            select.value = self._roles[d]
        self._sync_index_visibility()

    def _sync_index_visibility(self) -> None:
        for d, number in self._index_numbers.items():
            number.set_visibility(self._roles[d] == "index")

    def needs_render(self, snap: BatchSnapshot) -> bool:
        """Whether `snap` is a genuinely new snapshot worth rendering.

        Marks it consumed so the next tick skips it. A manual refresh
        (`compute_render` with live weights) leaves `_last_snapshot` untouched,
        so the live view persists until the next captured batch publishes a
        genuinely fresh snapshot.
        """
        if snap is self._last_snapshot:
            return False
        self._last_snapshot = snap
        return True

    def show_weights(
        self,
        weights: dict[str, Tensor],
        gradients: dict[str, Tensor],
        *,
        optimizer_state: dict[str, dict[str, Tensor]],
        optimizer_hyperparams: dict[str, dict[str, float]],
        custom_tensors: dict[str, Tensor] | None = None,
    ) -> None:
        """Display weight, gradient, and optimizer values (snapshot or live).

        Synchronous compute-and-apply, for the light paths (page build, tests).
        The 0.2s tick and the Refresh handler instead split `compute_render`
        (off the event loop) from `apply_render` (back on it).
        """
        self.apply_render(
            self.compute_render(
                weights,
                gradients,
                optimizer_state=optimizer_state,
                optimizer_hyperparams=optimizer_hyperparams,
                custom_tensors=custom_tensors,
            )
        )

    def compute_render(
        self,
        weights: dict[str, Tensor],
        gradients: dict[str, Tensor],
        *,
        optimizer_state: dict[str, dict[str, Tensor]],
        optimizer_hyperparams: dict[str, dict[str, float]],
        custom_tensors: dict[str, Tensor] | None = None,
    ) -> _PanelRender:
        """Render the panel's strips to HTML — pure, safe to run in a thread.

        Stashes the source dicts so `_on_role` / `_on_index` can re-render the
        same weights synchronously when the user remaps an axis, then returns a
        `_PanelRender` the loop applies via `apply_render`. Touches no NiceGUI
        element, so it runs off the event loop (`asyncio.to_thread`).
        `custom_tensors` is this parameter's custom weight-tensor instrument
        outputs (`BatchSnapshot.custom_weight_tensors[param]`).
        """
        self._weights = weights
        self._gradients = gradients
        self._opt_state = optimizer_state
        self._opt_hparams = optimizer_hyperparams
        self._custom = custom_tensors or {}
        return self._compute_render()

    def _compute_render(self) -> _PanelRender:
        tensor = self._weights.get(self.name) if self._weights is not None else None
        if tensor is None:
            return _PanelRender(error="no weights captured yet")
        x_dim, y_dim, tile_dim = dims_from_roles(self._roles)
        if x_dim is None:
            return _PanelRender(error="select an X dimension")
        # A tiling axis only makes sense once a Y axis exists.
        tile = tile_dim if y_dim is not None else None
        fixed = {
            d: self._indices.get(d, 0)
            for d in range(self._ndim)
            if self._roles[d] == "index"
        }
        strip = render_weight(
            tensor, x_dim=x_dim, y_dim=y_dim, tile_dim=tile, fixed=fixed
        )
        if strip is None:
            return _PanelRender(error="invalid axis selection")
        # The gradient shares the weight's shape, so the same axis layout
        # applies; it's simply absent before the first backward pass.
        grad = self._gradients.get(self.name) if self._gradients is not None else None
        grad_strip = (
            render_weight(grad, x_dim=x_dim, y_dim=y_dim, tile_dim=tile, fixed=fixed)
            if grad is not None
            else None
        )
        opt_strips, opt_scalar_text = self._compute_optimizer_values(
            x_dim=x_dim, y_dim=y_dim, tile=tile, fixed=fixed
        )
        return _PanelRender(
            weight_html=_strip_html(strip, show_labels=True),
            grad_html=(
                _strip_html(grad_strip)
                if grad_strip is not None
                else _NO_GRADIENT_HTML
            ),
            opt_strips=opt_strips,
            opt_scalar_text=opt_scalar_text,
            custom_strips=self._compute_custom_strips(
                x_dim=x_dim, y_dim=y_dim, tile=tile, fixed=fixed
            ),
        )

    def _compute_custom_strips(
        self,
        *,
        x_dim: int,
        y_dim: int | None,
        tile: int | None,
        fixed: dict[int, int],
    ) -> list[tuple[str, str]]:
        """Render the custom weight-tensor strips (pure, worker thread).

        The instrument contract pins each tensor to the weight's shape, so
        the panel's axis layout always applies — these render exactly like
        the shape-matched optimizer entries, labelled by instrument name.
        """
        strips: list[tuple[str, str]] = []
        for key, tensor in sorted(self._custom.items()):
            strip = render_weight(
                tensor, x_dim=x_dim, y_dim=y_dim, tile_dim=tile, fixed=fixed
            )
            if strip is None:
                continue
            strips.append((key.upper(), _strip_html(strip)))
        return strips

    def _compute_optimizer_values(
        self,
        *,
        x_dim: int,
        y_dim: int | None,
        tile: int | None,
        fixed: dict[int, int],
    ) -> tuple[list[tuple[str, str]], str]:
        """Render the optimizer strips + scalar line below the gradient (pure).

        Tensor state entries matching the weight's shape (momentum buffers,
        Adam moments) reuse the panel's axis layout; differently-shaped ones
        (e.g. factored second moments) fall back to their own rank's default
        axes. 0-dim entries (Adam's `step`) join the group hyperparameters
        (`lr`, …) on a single scalar line. With no optimizer attached both
        stay empty, leaving the panel exactly as before. Returns the
        `(marker label, strip HTML)` pairs and the scalar-line text; building
        the elements is `apply_render`'s job, back on the event loop.
        """
        entries = dict(sorted(self._opt_state.get(self.name, {}).items()))
        opt_strips: list[tuple[str, str]] = []
        scalar_parts: list[str] = []
        for key, tensor in entries.items():
            if tensor.ndim == 0:
                scalar_parts.append(f"{key} = {_format_stat(float(tensor))}")
                continue
            if tuple(tensor.shape) == self._shape:
                strip = render_weight(
                    tensor, x_dim=x_dim, y_dim=y_dim, tile_dim=tile, fixed=fixed
                )
            else:
                dims = default_weight_dims(tensor.ndim)
                strip = render_weight(
                    tensor,
                    x_dim=dims.x_dim,
                    y_dim=dims.y_dim,
                    tile_dim=dims.tile_dim,
                    fixed={},
                )
            if strip is None:
                continue
            opt_strips.append((key.upper(), _strip_html(strip)))
        scalar_parts += [
            f"{key} = {_format_stat(value)}"
            for key, value in sorted(self._opt_hparams.get(self.name, {}).items())
        ]
        return opt_strips, "  ·  ".join(scalar_parts)

    def apply_render(self, render: _PanelRender) -> None:
        """Push a computed `_PanelRender` onto the UI elements (event loop)."""
        if render.error is not None:
            self._show_error(render.error)
            return
        self._error.text = ""
        self._img.set_content(render.weight_html)
        self._grad_img.set_content(render.grad_html)
        self._opt_container.clear()
        with self._opt_container:
            for label, strip_html in render.opt_strips:
                ui.element("div").classes("h-1")
                with ui.element("div").classes("flex no-wrap items-stretch"):
                    # `label` is the optimizer state key upper-cased
                    # (`_compute_optimizer_values`); lower() restores the
                    # conventionally-lowercase key (exp_avg, momentum_buffer).
                    _strip_marker(
                        OPTIMIZER.css,
                        label,
                        tooltip="Optimizer state for this parameter",
                    )
                    ui.html(strip_html)
        self._custom_container.clear()
        with self._custom_container:
            for label, strip_html in render.custom_strips:
                ui.element("div").classes("h-1")
                with ui.element("div").classes("flex no-wrap items-stretch"):
                    # `label` is the instrument name upper-cased
                    # (`_compute_custom_strips`); lower() restores the
                    # conventionally-lowercase registration name.
                    _strip_marker(
                        CUSTOM.css,
                        label,
                        tooltip="Your custom instrument for this parameter",
                    )
                    ui.html(strip_html)
        self._opt_scalars.text = render.opt_scalar_text
        self._opt_scalars.set_visibility(bool(render.opt_scalar_text))

    def _render_current(self) -> None:
        if self._weights is not None:
            self.apply_render(self._compute_render())

    def _show_error(self, message: str) -> None:
        self._error.text = message
        self._img.set_content("")
        self._grad_img.set_content("")
        self._opt_container.clear()
        self._opt_scalars.set_visibility(False)
