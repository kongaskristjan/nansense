"""Stats selection, refresh cadence, and coalescing without widget ownership."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass

from nansense.patches import PatchType
from nansense.session import BatchSnapshot, Session

_VIEW_HISTOGRAM: str = "HISTOGRAM"


_VIEW_MINMAX: str = "MIN/MAX"


_VIEW_GRAPHS: str = "GRAPHS"


@dataclass
class _WatchPageState:
    """Mutable page state shared by the sidebar controls and layer panels."""

    axis_log_x: bool = False
    axis_log_y: bool = False
    retain_axes: bool = False
    show_bands: bool = False
    view: str = _VIEW_HISTOGRAM
    grid_type: PatchType = "max_pixel"
    heat_on: bool = False
    selected_phase: str = ""
    selected_layer: str = ""
    pending_scroll: str = ""
    # User choices during a tour take precedence over restoring its initial view.
    tour_saved_view: str | None = None
    tour_user_set_view: bool = False
    tour_view_write: bool = False
    refresh_running: bool = False
    refresh_dirty: bool = False
    frozen_hist: bool | None = None
    frozen_minmax: bool | None = None


@dataclass
class _RefreshGate:
    """Decides when the periodic tick re-renders the page's data."""

    last_snapshot: BatchSnapshot | None = None
    last_stats_layers: frozenset[str] = frozenset()
    last_phases: tuple[str, ...] = ()
    last_average_patches: bool | None = None
    last_running: bool | None = None

    def should_refresh(self, session: Session) -> bool:
        """Consume the session's current state; True if it changed."""
        snapshot = session.snapshot
        stats_layers = session.stats_layers
        phases = tuple(session.schedule.phase_order)
        average_patches = session.watch_performance.average_patches
        running = session.is_running
        changed = (
            snapshot is not self.last_snapshot
            or stats_layers != self.last_stats_layers
            or phases != self.last_phases
            or average_patches != self.last_average_patches
            or running != self.last_running
        )
        self.last_snapshot = snapshot
        self.last_stats_layers = stats_layers
        self.last_phases = phases
        self.last_average_patches = average_patches
        self.last_running = running
        return changed


def _tour_restore_view(
    saved: str | None, user_set_view: bool, current: str
) -> str | None:
    """The view to switch back to when a tour run ends, or `None`."""
    if saved is None or user_set_view or saved == current:
        return None
    return saved


_PHASE_CURRENT_BATCH: str = "::current-batch::"


_PHASE_CURRENT_BATCH_LABEL: str = "Current batch"


def _phase_select_options(view: str, phase_names: list[str]) -> dict[str, str]:
    """The Phase dropdown's value→label map for the current view."""
    options = {p: p for p in phase_names}
    if view != _VIEW_GRAPHS:
        options[_PHASE_CURRENT_BATCH] = _PHASE_CURRENT_BATCH_LABEL
    return options


def _reconcile_selected_phase(selected: str, view: str, phase_names: list[str]) -> str:
    """A valid Phase selection for the current view."""
    if view == _VIEW_GRAPHS:
        if selected in phase_names:
            return selected
        return phase_names[0] if phase_names else ""
    if selected == _PHASE_CURRENT_BATCH or selected in phase_names:
        return selected
    return _PHASE_CURRENT_BATCH


def _initial_phase(session: Session, layer: str) -> str:
    """The Phase selection the page opens on."""
    position = session.live_position
    if position is None:
        snapshot = session.snapshot
        position = snapshot.position if snapshot is not None else None
    if position is not None and position.phase in session.stats_phases(layer or None):
        return position.phase
    return _PHASE_CURRENT_BATCH


_LAYER_ALL: str = "\x00all"


_ALL_LAYERS_LABEL: str = "All watched layers"


_ALL_LAYERS_MAX: int = 10


def _watched_in_order(
    layer_names: list[str], stats_layers: frozenset[str]
) -> list[str]:
    """The stats-carrying layers in the page's stable graph order."""
    return [n for n in layer_names if n in stats_layers]


def _selectable_layers(
    selected_phase: str, layer_names: list[str], stats_layers: frozenset[str]
) -> list[str]:
    """The layers the Layer dropdown offers for the current phase selection."""
    if selected_phase == _PHASE_CURRENT_BATCH:
        return list(layer_names)
    return _watched_in_order(layer_names, stats_layers)


def _all_layers_available(watched_count: int) -> bool:
    """Whether the "all watched layers" entry is offered for this count."""
    return 0 < watched_count < _ALL_LAYERS_MAX


def _layer_select_options(ordered: list[str]) -> dict[str, str]:
    """The layer dropdown's value→label map for the watched layers."""
    options: dict[str, str] = {}
    if _all_layers_available(len(ordered)):
        options[_LAYER_ALL] = _ALL_LAYERS_LABEL
    for name in ordered:
        options[name] = name
    return options


def _reconcile_selected_layer(selected: str, ordered: list[str]) -> str:
    """A valid dropdown selection given the watched layers."""
    if selected == _LAYER_ALL and _all_layers_available(len(ordered)):
        return selected
    if selected in ordered:
        return selected
    return ordered[0] if ordered else ""


def _visible_layers(selected: str, ordered: list[str]) -> list[str]:
    """The watched layers whose cards should be rendered for `selected`."""
    if selected == _LAYER_ALL and _all_layers_available(len(ordered)):
        return ordered
    if selected in ordered:
        return [selected]
    return ordered[:1]


_PATCH_TYPE_LABELS: dict[PatchType, str] = {
    "max_pixel": "Max pixel",
    "min_pixel": "Min pixel",
    "max_average": "Max average",
    "min_average": "Min average",
}


_AVERAGE_PATCH_TYPES: frozenset[PatchType] = frozenset({"max_average", "min_average"})


def _grid_type_options(average_patches: bool) -> dict[PatchType, str]:
    """The MIN/MAX radio's value→label map for the Performance setting."""
    return {
        ptype: label
        for ptype, label in _PATCH_TYPE_LABELS.items()
        if average_patches or ptype not in _AVERAGE_PATCH_TYPES
    }


def _reconcile_grid_type(
    selected: PatchType, options: dict[PatchType, str]
) -> PatchType:
    """A valid radio selection: `selected` while offered, else the default."""
    return selected if selected in options else "max_pixel"


class StatsController:
    def __init__(
        self, session: Session, layers: list[str], state: _WatchPageState
    ) -> None:
        self.session = session
        self.layers = tuple(layers)
        self.state = state
        self.gate = _RefreshGate()

    def phase_options(self) -> dict[str, str]:
        names = list(self.session.schedule.phase_order)
        self.state.selected_phase = _reconcile_selected_phase(
            self.state.selected_phase, self.state.view, names
        )
        return _phase_select_options(self.state.view, names)

    def layer_options(self) -> dict[str, str]:
        ordered = _selectable_layers(
            self.state.selected_phase, list(self.layers), self.session.stats_layers
        )
        self.state.selected_layer = _reconcile_selected_layer(
            self.state.selected_layer, ordered
        )
        return _layer_select_options(ordered)

    def grid_options(self) -> dict[PatchType, str]:
        options = _grid_type_options(self.session.watch_performance.average_patches)
        self.state.grid_type = _reconcile_grid_type(self.state.grid_type, options)
        return options

    def select_view(self, view: str) -> None:
        if self.state.tour_view_write:
            self.state.tour_view_write = False
        else:
            self.state.tour_user_set_view = True
        self.state.view = view
        self.phase_options()
        self.layer_options()

    async def refresh(self, render: Callable[[], Awaitable[None]]) -> None:
        if self.state.refresh_running:
            self.state.refresh_dirty = True
            return
        self.state.refresh_running = True
        try:
            while True:
                self.state.refresh_dirty = False
                await render()
                if not self.state.refresh_dirty:
                    return
        finally:
            self.state.refresh_running = False
