"""The `/weights` page: per-layer weight viewer with remappable axes."""

from __future__ import annotations

from nicegui import ui
from torch import Tensor

from nansense.recording import RecordedView
from nansense.schedule import format_position
from nansense.session import BatchSnapshot, Session
from nansense.ui.common import (
    _set_controls_enabled,
    _strip_html,
    _strip_marker,
    _weights_placeholder,
)
from nansense.ui.histograms import _format_stat
from nansense.ui.render import default_weight_dims, dims_from_roles, render_weight
from nansense.ui.static import _STRIP_MARKER_CSS
from nansense.ui.top_bar import (
    _add_settings_button,
    _add_step_controls,
    _build_step_until_custom_dialog,
    _top_bar_row,
)


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


def _build_weights_page(session: Session, layer: str) -> None:
    """Per-layer weight viewer: kernel/image strips with selectable axes.

    Reuses the main page's stepping controls (minus the sample spinner — a
    weight has no batch axis) so the displayed weights track the same paused
    batch. One panel per parameter the layer owns; each panel lets the user
    remap which tensor axes become the X / Y / tiling axes and pins the rest
    by index.
    """
    title = f"Weights · {layer}" if layer else "Weights"
    ui.page_title(f"Nansense — {title}")
    ui.query(".nicegui-content").classes("p-0 h-screen overflow-hidden")
    ui.query("body").classes("overflow-hidden")
    ui.query("html").classes("overflow-hidden")
    ui.add_head_html(_STRIP_MARKER_CSS)

    weight_names = session.layer_weights.get(layer, [])
    shapes = {
        name: tuple(p.shape)
        for name, p in session.model.named_parameters()
        if name in set(weight_names)
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
            ui.button(
                icon="arrow_back",
                on_click=lambda: ui.navigate.to("/"),
                color="slate-500",
            ).props("dense size=md").tooltip("Back to the main page")
            ui.label(title).classes(
                "font-mono text-base font-bold ml-2 truncate max-w-64"
            )
            position_label = _add_step_controls(session, step_until_custom)
            _add_settings_button(session, record_view).classes("ml-auto")
            ui.button(
                icon="refresh",
                on_click=lambda: do_refresh(),
                color="slate-500",
            ).props("dense size=md flat").tooltip(
                "Show the model's current weights (works while training)"
            )

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
                for name in weight_names:
                    panels.append(
                        _WeightPanel(name=name, shape=shapes[name], session=session)
                    )

    def do_refresh() -> None:
        # Read the model's live parameters instead of the last snapshot, so the
        # weights update even mid-training (detach / run modes never publish a
        # snapshot). The live view then persists until the next captured batch.
        weights = session.current_weights()
        gradients = session.current_weight_gradients()
        optimizer_state = session.current_optimizer_state()
        optimizer_hyperparams = session.current_optimizer_hyperparams()
        for panel in panels:
            panel.show_weights(
                weights,
                gradients,
                optimizer_state=optimizer_state,
                optimizer_hyperparams=optimizer_hyperparams,
            )

    def tick() -> None:
        live = session.live_position
        if live is not None:
            position_label.text = format_position(live)
        frozen = session.recording.is_recording(record_key)
        for panel in panels:
            panel.set_frozen(frozen)
        snap = session.snapshot
        if snap is None:
            return
        for panel in panels:
            panel.maybe_render(snap)

    ui.timer(0.2, tick)


# Shown next to the GRADIENT marker before any backward pass has populated
# the parameter's gradient.
_NO_GRADIENT_HTML: str = (
    '<div class="text-xs text-slate-400 italic py-1">no gradient captured yet</div>'
)


class _WeightPanel:
    """One card per parameter — an axis-remappable kernel/image strip.

    The weight's rank is fixed, so the per-dimension role selects and index
    spinners are built once. Changing a role auto-demotes whichever other axis
    held that role (roles X/Y/Tile stay unique), then re-renders against the
    last snapshot; new snapshots re-render via `maybe_render`.
    """

    def __init__(self, *, name: str, shape: tuple[int, ...], session: Session) -> None:
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
            with ui.row().classes("items-end gap-4 flex-wrap"):
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
                            ).props("dense outlined").classes("w-24")
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
                            ).props("dense outlined").classes("w-20")
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
                    with ui.element("div").classes("flex no-wrap items-stretch"):
                        _strip_marker("bg-sky-500", "WEIGHT")
                        self._img = ui.html("")
                    ui.element("div").classes("h-1")
                    with ui.element("div").classes("flex no-wrap items-stretch"):
                        _strip_marker("bg-violet-500", "GRADIENT")
                        self._grad_img = ui.html("")
                    # One marker-barred strip per tensor-valued optimizer
                    # state entry (momentum_buffer, exp_avg, …); rebuilt on
                    # each render. Stays empty — invisible — when the session
                    # has no optimizer.
                    self._opt_container = ui.element("div").classes("w-full")
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
        # Writes to widget `.value` made from inside a value-change handler are
        # suppressed by NiceGUI; defer the select/visibility sync one loop tick
        # so demotions actually reach the client.
        ui.timer(0.0, self._apply_control_state, once=True)
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

    def maybe_render(self, snap: BatchSnapshot) -> None:
        """Render snapshot weights, but only when the snapshot is new.

        A manual refresh (`show_weights` with live weights) leaves
        `_last_snapshot` untouched, so the live view persists until the next
        captured batch publishes a genuinely fresh snapshot.
        """
        if snap is self._last_snapshot:
            return
        self._last_snapshot = snap
        self.show_weights(
            snap.weights,
            snap.weight_gradients,
            optimizer_state=snap.optimizer_state,
            optimizer_hyperparams=snap.optimizer_hyperparams,
        )

    def show_weights(
        self,
        weights: dict[str, Tensor],
        gradients: dict[str, Tensor],
        *,
        optimizer_state: dict[str, dict[str, Tensor]],
        optimizer_hyperparams: dict[str, dict[str, float]],
    ) -> None:
        """Display weight, gradient, and optimizer values (snapshot or live)."""
        self._weights = weights
        self._gradients = gradients
        self._opt_state = optimizer_state
        self._opt_hparams = optimizer_hyperparams
        self._render()

    def _render_current(self) -> None:
        if self._weights is not None:
            self._render()

    def _render(self) -> None:
        tensor = self._weights.get(self.name) if self._weights is not None else None
        if tensor is None:
            self._show_error("no weights captured yet")
            return
        x_dim, y_dim, tile_dim = dims_from_roles(self._roles)
        if x_dim is None:
            self._show_error("select an X dimension")
            return
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
            self._show_error("invalid axis selection")
            return
        self._error.text = ""
        self._img.set_content(_strip_html(strip))
        # The gradient shares the weight's shape, so the same axis layout
        # applies; it's simply absent before the first backward pass.
        grad = self._gradients.get(self.name) if self._gradients is not None else None
        grad_strip = (
            render_weight(grad, x_dim=x_dim, y_dim=y_dim, tile_dim=tile, fixed=fixed)
            if grad is not None
            else None
        )
        self._grad_img.set_content(
            _strip_html(grad_strip) if grad_strip is not None else _NO_GRADIENT_HTML
        )
        self._render_optimizer_values(x_dim=x_dim, y_dim=y_dim, tile=tile, fixed=fixed)

    def _render_optimizer_values(
        self,
        *,
        x_dim: int,
        y_dim: int | None,
        tile: int | None,
        fixed: dict[int, int],
    ) -> None:
        """Rebuild the optimizer strips + scalar line below the gradient.

        Tensor state entries matching the weight's shape (momentum buffers,
        Adam moments) reuse the panel's axis layout; differently-shaped ones
        (e.g. factored second moments) fall back to their own rank's default
        axes. 0-dim entries (Adam's `step`) join the group hyperparameters
        (`lr`, …) on a single scalar line. With no optimizer attached both
        stay empty, leaving the panel exactly as before.
        """
        entries = dict(sorted(self._opt_state.get(self.name, {}).items()))
        self._opt_container.clear()
        scalar_parts: list[str] = []
        with self._opt_container:
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
                        fixed={d: 0 for d in dims.fixed_dims},
                    )
                if strip is None:
                    continue
                ui.element("div").classes("h-1")
                with ui.element("div").classes("flex no-wrap items-stretch"):
                    _strip_marker("bg-amber-600", key.upper())
                    ui.html(_strip_html(strip))
        scalar_parts += [
            f"{key} = {_format_stat(value)}"
            for key, value in sorted(self._opt_hparams.get(self.name, {}).items())
        ]
        self._opt_scalars.text = "  ·  ".join(scalar_parts)
        self._opt_scalars.set_visibility(bool(scalar_parts))

    def _show_error(self, message: str) -> None:
        self._error.text = message
        self._img.set_content("")
        self._grad_img.set_content("")
        self._opt_container.clear()
        self._opt_scalars.set_visibility(False)
