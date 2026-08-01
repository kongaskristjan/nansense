"""The `/stats` page: per-layer histograms and extreme-patch grids."""

from __future__ import annotations

import asyncio
import html
import json
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field, replace
from urllib.parse import quote

import plotly.graph_objects as go
from nicegui import ui
from nicegui.elements.mixins.disableable_element import DisableableElement
from nicegui.events import GenericEventArguments, ValueChangeEventArguments

from nansense import debugger
from nansense.instruments import MetricSeries, MetricsSnapshot
from nansense.patches import PatchType
from nansense.recording import RecordedView
from nansense.session import BatchSnapshot, Session, StatsScope
from nansense.ui.bin_samples import sample_bin
from nansense.ui.common import (
    _b64_img_src,
    _column_header_bar,
    _defer_value_write,
    _install_panel_resize,
    _row_label_bar_html,
    _notice_banner,
    _page_scaffold,
    _refuse_unwatch_while_recording,
    _resizable_pane_props,
    _resize_handle,
    _set_controls_enabled,
)
from nansense.ui.epoch_stats import (
    epoch_axis_dtick,
    epoch_stat_series,
    make_epoch_stats_figure,
    make_metric_figure,
    metric_epochs,
    metric_trace_data,
    weight_stat_series,
)
from nansense.ui.histograms import (
    _BIN_VALUE_LABELS,
    _axis_ranges,
    _format_stat,
    _hover_customdata,
    kind_stats,
    _linear_bar_x,
    _make_histogram_figure,
    _min_positive_height,
    overflow_marks,
    phase_color,
    _phase_hists,
    _phases_with_data,
    _retained_y_range,
    _stats_table_html,
    _x_range_linear_to_log,
    _x_range_log_to_linear,
    trace_heights,
    use_density,
)
from nansense.ui.render import (
    LABEL_HEIGHT,
    PATCH_CELL_GAP,
    PatchColumn,
    PatchGridRender,
    render_image,
    render_patch_grid,
)
from nansense.ui.top_bar import (
    _add_error_banner,
    _add_repo_logo,
    _add_settings_button,
    _add_share_button,
    _add_step_controls,
    _add_tour_button,
    _back_button,
    _back_href,
    _build_step_until_custom_dialog,
    _top_bar_row,
)
from nansense.ui.tour import add_tour, stats_tour_steps
from nansense.watch import (
    N_BINS,
    LayerStatsSnapshot,
    WatchSnapshot,
    narrow_to_channel,
)


# The View dropdown's three entries; also the values `_WatchPageState.view`
# takes. HISTOGRAM is the default.
_VIEW_HISTOGRAM: str = "HISTOGRAM"
_VIEW_MINMAX: str = "MIN/MAX"
_VIEW_GRAPHS: str = "GRAPHS"


@dataclass
class _WatchPageState:
    """Mutable page state shared by the sidebar controls and layer panels."""

    # Whether the value (x) and probability (y) axes use a log-based scale.
    # Both default off — linear axes showing probability density (see
    # `use_density`); the sidebar checkboxes flip them and re-render every
    # plot immediately.
    axis_log_x: bool = False
    axis_log_y: bool = False
    # When set, a Log x / Log y / phase change keeps the current axis ranges
    # (re-expressed for the new scale) instead of auto-fitting to the data —
    # see `_HistPlot`. Lives with the histogram-view controls.
    retain_axes: bool = False
    # Whether to mark the dtype-aware subnormal/overflow band edges on each
    # histogram (the "Show subnormal/overflow" checkbox). Seeded on by
    # `_should_show_bands` when the page opens with an active subnormal/overflow
    # issue, regardless of how the page was reached.
    show_bands: bool = False
    # Which of the three views every card shows (a `_VIEW_*` value).
    view: str = _VIEW_HISTOGRAM
    # MIN/MAX view state: which grid is shown (a radio group defaulting to
    # "Max pixel"; the average entries are offered only while their
    # collection is on — see `_grid_type_options`) and whether the
    # activation heatmap is blended over the patches.
    grid_type: PatchType = "max_pixel"
    heat_on: bool = False
    # Every card shows one phase at a time, picked by a header dropdown
    # shared by both views; defaults to the phase training is currently in
    # when it has collected stats, else "Current batch" (`_initial_phase`).
    selected_phase: str = ""
    # Which watched layer's cards to render: a layer name, or `_LAYER_ALL`
    # for every watched layer at once. Defaults (via reconciliation) to the
    # first watched layer so the page stays fast with many layers watched.
    selected_layer: str = ""
    # One-shot scroll target from the `?scroll=` query param ("weights"
    # scrolls to the GRAPHS card's Weights section once it first renders);
    # cleared after the scroll fires.
    pending_scroll: str = ""
    # Tour view-restore bookkeeping (`_tour_start` / `_tour_end` in
    # `_build_stats_page`): the view showing when the current tour run
    # started, whether the visitor picked a view themselves during that run
    # (their choice then wins over the restore), and a one-shot marker
    # telling `set_mode` the pending View write is the tour's, not the
    # visitor's.
    tour_saved_view: str | None = None
    tour_user_set_view: bool = False
    tour_view_write: bool = False
    # Single-flight refresh flags (see `refresh` in `_build_stats_page`).
    refresh_running: bool = False
    refresh_dirty: bool = False
    # Last frozen flags pushed to the client, so the per-tick sync only
    # sends enable/disable when something actually changed.
    frozen_hist: bool | None = None
    frozen_minmax: bool | None = None
    # The sidebar's watched-layer count label, assigned during page build.
    count_label: ui.label = field(init=False)


@dataclass
class _RefreshGate:
    """Decides when the periodic tick re-renders the page's data.

    The page follows the visualization update cadence (the settings'
    "Update frequency"): a tick passes only once the session has published
    a new snapshot — at the configured frequency, on a capture/pause, or on
    a one-shot UI Refresh — never merely because the running watch
    aggregates advanced another batch. Watched-set and phase-list changes
    (e.g. from the main page in another tab) also pass, so the sidebar and
    cards stay in sync while training runs between updates or sits paused.
    So does a flip of the average-patches Performance setting: it flushes
    every aggregate bucket and decides which grid types the MIN/MAX radio
    offers, both of which the page must re-render. The sidebar controls and
    the "Refresh now" button (`_refresh_now`) call `refresh` directly,
    bypassing the gate.
    """

    last_snapshot: BatchSnapshot | None = None
    last_stats_layers: frozenset[str] = frozenset()
    last_phases: tuple[str, ...] = ()
    last_average_patches: bool | None = None

    def should_refresh(self, session: Session) -> bool:
        """Consume the session's current state; True if it changed."""
        snapshot = session.snapshot
        stats_layers = session.stats_layers
        phases = tuple(session.schedule.phase_order)
        average_patches = session.watch_performance.average_patches
        changed = (
            snapshot is not self.last_snapshot
            or stats_layers != self.last_stats_layers
            or phases != self.last_phases
            or average_patches != self.last_average_patches
        )
        self.last_snapshot = snapshot
        self.last_stats_layers = stats_layers
        self.last_phases = phases
        self.last_average_patches = average_patches
        return changed


async def _refresh_now(
    session: Session, refresh: Callable[[], Awaitable[None]]
) -> None:
    """Behind the top bar's "Refresh now" button.

    The immediate `refresh` re-renders from the data already at hand — the
    running aggregates and the last published snapshot. Fresh snapshot
    content (the "Current batch" phase) needs a publish, so this also arms
    `Session.request_snapshot` (like the main top bar's Refresh): the next
    free-running batch publishes without pausing, and the tick's
    `_RefreshGate` re-renders the page from it. A no-op when training isn't
    producing batches — the shown snapshot is then already current.
    """
    session.request_snapshot()
    await refresh()


def _should_show_bands(error: debugger.DebugError | None) -> bool:
    """Whether to pre-check the histogram under/overflow band on page open.

    True while a numerical issue whose under/overflow check *tripped* is
    active — so opening `/stats` from anywhere (a layer card's Stats button,
    the warning dialog's per-row link, a direct URL) surfaces the band that
    issue is about, without threading a query flag through every link.
    """
    return error is not None and debugger.UNDER_OVER in debugger.categories_present(
        error
    )


def _apply_watch_param(session: Session, layer: str, watch: str) -> None:
    """Honor a `?watch=1` deep link: start collecting stats for `layer`.

    Links that need the layer collecting by the time the page lands (the
    weights page's GRAPHS jump, the warning dialog's Stats-with-watch row)
    carry `watch=1` instead of calling `session.watch` in an `on_click` —
    that keeps them real anchors, so middle-click opens a new tab and still
    starts collection. Only the `watched` scope needs the watch: the other
    scopes either already collect every layer or are deliberately paused.
    Unknown layer names are refused by `Session.watch` itself; an already
    watched layer is left alone so reloading the link stays a no-op.
    """
    if (
        watch.strip()
        and session.stats_scope is StatsScope.WATCHED
        and layer not in session.watched_layers
    ):
        session.watch(layer)


def _tour_restore_view(
    saved: str | None, user_set_view: bool, current: str
) -> str | None:
    """The view to switch back to when a tour run ends, or `None`.

    Dismissing the tour — Skip, Done, or Escape — should land the visitor
    back on the view they started from, since the view-bound steps cycled
    the page through every view on their behalf. No restore when nothing
    was saved (no run started), when the run never left the saved view, or
    when the visitor picked a view themselves mid-run — an explicit choice
    the tour must not undo.
    """
    if saved is None or user_set_view or saved == current:
        return None
    return saved


def _build_stats_page(
    session: Session,
    layer_names: list[str],
    selected_layer: str = "",
    *,
    view: str = "",
    scroll: str = "",
    watch: str = "",
    input_mean: tuple[float, ...] | None = None,
    input_std: tuple[float, ...] | None = None,
) -> None:
    """The deep-dive page for watched layers.

    The top bar carries the shared stepping controls, like the main page.
    Left-sidebar dropdowns switch every layer card between three views, pick
    which phase (train / val / …) the cards show, and pick which watched
    layer to render — one named layer (the default, which keeps the page
    fast when many layers are watched) or, while fewer than `_ALL_LAYERS_MAX`
    are watched, every watched layer at once.

    The Phase dropdown's last entry, "Current batch", is special: it shows
    stats computed directly from the last captured `BatchSnapshot` rather than
    the running watch aggregates, so it works for *any* layer whether or not
    it is watched — and the Layer dropdown then offers every layer, not just
    the watched ones. The page opens on the phase training is currently in
    when that phase already holds collected stats (for the linked layer, if
    the URL named one); otherwise on "Current batch", which needs no watched
    aggregates, so a freshly opened Stats view always has something to show
    (see `_initial_phase`). The view and phase apply across views, one at a
    time:

    - HISTOGRAM (the default) — a merged activations/gradients stats table,
      then one plotly figure per tensor kind for the selected phase's latest
      epoch, with the "Log x" / "Log y" axis checkboxes and a per-histogram
      "Per channel" switch.
    - MIN/MAX — the extreme-activation patch grids
      (channels across, per-channel top samples down), one per patch
      type, picked one at a time by a radio group (defaulting to "Max
      pixel"), plus a heatmap checkbox
      that blends the stored activation maps over the patches.
    - GRAPHS — the phase's whole epoch series: per tensor kind,
      mean/std/median/min/max (and, for activations, dead channels)
      against epoch, stats toggled through the plotly legend (only the
      mean starts enabled). This view has no "Current batch" (a single
      batch has no epoch series): `sync_phase_select` drops the entry and
      swaps such a selection for the first schedule phase.
    Each control group is only visible while its view is selected. A
    `ui.timer` re-renders the visible view in place at the visualization
    update cadence: `_RefreshGate` passes a tick only once a new snapshot
    was published (the settings' "Update frequency" — per epoch by default —
    or a pause, a step, a one-shot Refresh), so the page updates in step
    with the main view instead of tracking the running aggregates live.
    Layers can also be unwatched directly from the card header here, which
    drops the corresponding accumulator entry — the change is reflected on
    the main page on next navigation.
    """
    _page_scaffold("Stats")
    _install_panel_resize()
    add_tour("stats", stats_tour_steps(), locked=session.locked)
    # A `?watch=1` link starts collecting the seeded layer before anything
    # renders, so the reconciliation below sees it as watched.
    _apply_watch_param(session, selected_layer, watch)

    layer_panels: dict[str, _WatchLayerPanel] = {}
    # element id -> the histogram plot to forward that element's hovers to.
    # Repopulated on every `rebuild_cards`, so a single page-level `ui.on`
    # replaces the per-element handlers that used to leak on rebuild.
    hover_registry: dict[int, _HistPlot] = {}
    body_container: ui.column
    phase_names = session.schedule.phase_order

    async def _dispatch_hover(e: GenericEventArguments) -> None:
        view = hover_registry.get(int(e.args.get("id", -1)))
        if view is not None:
            await view._on_hover(e)

    ui.on(_HOVER_EVENT, _dispatch_hover)
    # A `?view=` link opens straight on that view — "minmax" from the
    # experiment page's "Compare with MIN/MAX", "graphs" from the weights
    # page's per-layer jump — instead of the histograms.
    requested_view = view.strip().lower()
    initial_view = {
        "minmax": _VIEW_MINMAX,
        "graphs": _VIEW_GRAPHS,
    }.get(requested_view, _VIEW_HISTOGRAM)
    state = _WatchPageState(
        # Open on the phase training is currently in when its aggregates
        # already hold stats; else "Current batch", which reads the last
        # captured snapshot directly, so the page shows data immediately for
        # any layer without waiting for watch aggregates to fill.
        selected_phase=_initial_phase(session, selected_layer),
        view=initial_view,
        # Seed the layer picked by the caller (e.g. a `?layer=` link from the
        # main page's watch menu). Reconciliation drops it back to the first
        # watched layer if it isn't currently watched.
        selected_layer=selected_layer,
        # A `?scroll=weights` link (the weights page's jump) scrolls to the
        # GRAPHS card's Weights section once it first renders.
        pending_scroll=scroll.strip().lower(),
        # Pre-check the under/overflow band whenever the page opens with an
        # active under/overflow issue, regardless of how it was reached.
        show_bands=_should_show_bands(session.debug_error),
    )
    # A graphs deep-link can open with "Current batch" seeded (nothing
    # collected for the running phase yet) but has no such entry in its
    # Phase dropdown — resolve to a real phase before the dropdown is built
    # rather than flashing an empty selection until the first refresh
    # reconciles it.
    state.selected_phase = _reconcile_selected_phase(
        state.selected_phase, state.view, list(phase_names)
    )

    async def set_axis_log_x(value: bool) -> None:
        state.axis_log_x = value
        await refresh()

    async def set_axis_log_y(value: bool) -> None:
        state.axis_log_y = value
        await refresh()

    async def set_retain_axes(value: bool) -> None:
        # Just flips the flag; the refresh leaves a frozen view untouched and
        # re-fits on un-check (so the axes snap back to the data immediately).
        state.retain_axes = value
        await refresh()

    async def set_show_bands(value: bool) -> None:
        # Toggles the dtype-aware subnormal/overflow band lines on every
        # histogram; each plot rebuilds to add/remove its layout shapes.
        state.show_bands = value
        await refresh()

    async def set_mode(value: object) -> None:
        # Tour-driven View writes (`_tour_set_view`, `_tour_end`) announce
        # themselves via the one-shot marker; any other write is the
        # visitor's own choice and cancels the end-of-tour view restore.
        if state.tour_view_write:
            state.tour_view_write = False
        else:
            state.tour_user_set_view = True
        state.view = str(value)
        # The Phase dropdown's options depend on the view (the stats view
        # has no "Current batch"); reconcile before anything re-renders.
        sync_phase_select()
        hist_controls.set_visibility(state.view == _VIEW_HISTOGRAM)
        minmax_controls.set_visibility(state.view == _VIEW_MINMAX)
        compare_deep_dream.set_visibility(state.view == _VIEW_MINMAX)
        await refresh()

    async def set_phase(value: object) -> None:
        new = str(value)
        # Programmatic value writes from `sync_phase_select` re-enter here;
        # bailing when nothing changed avoids a redundant refresh pass.
        if new == state.selected_phase:
            return
        state.selected_phase = new
        await refresh()

    async def set_layer(value: object) -> None:
        new = str(value) if value is not None else ""
        # Programmatic value writes from `sync_layer_select` re-enter here;
        # bailing when nothing changed avoids a redundant refresh pass.
        if new == state.selected_layer:
            return
        state.selected_layer = new
        await refresh()

    async def set_grid(ptype: PatchType) -> None:
        # Programmatic value writes from `sync_grid_type_select` re-enter
        # here; bailing when nothing changed avoids a redundant refresh pass.
        if ptype == state.grid_type:
            return
        state.grid_type = ptype
        await refresh()

    async def set_heat(value: bool) -> None:
        state.heat_on = value
        await refresh()

    def sync_compare_href() -> None:
        # Keep the compare link on the currently shown layer. An `href` (not
        # an `on_click` navigate) renders the button as a real anchor, so
        # middle-click / ctrl-click open the experiment in a new tab. The
        # props write is a no-op unless the target actually changed.
        href = _deep_dream_href(
            state.selected_phase,
            layer_names,
            session.stats_layers,
            state.selected_layer,
        )
        compare_deep_dream.props(f'href="{href}"')

    def sync_back_href() -> None:
        # Locked playground only: Back carries the selected layer so the
        # main page opens with its card shown ("All watched layers" and an
        # empty selection carry none — plain "/"). The props write is a
        # no-op unless the target actually changed.
        if not session.locked:
            return
        target = "" if state.selected_layer == _LAYER_ALL else state.selected_layer
        back_button.props(f'href="{_back_href(target)}"')

    step_until_custom = _build_step_until_custom_dialog(session)

    def record_view() -> RecordedView | None:
        # The recorder renders from the running watch accumulators, which the
        # "Current batch" view doesn't use — so that view isn't recordable.
        # Neither is the epoch-stats view: it draws the whole epoch series,
        # not a per-step frame.
        if state.selected_phase == _PHASE_CURRENT_BATCH:
            return None
        if state.view == _VIEW_GRAPHS:
            return None
        # Record exactly the cards on screen — the selected layer, or every
        # stats-carrying layer while "all" is showing.
        ordered = _watched_in_order(layer_names, session.stats_layers)
        watched = _visible_layers(state.selected_layer, ordered)
        if not watched:
            return None
        phase = state.selected_phase
        if state.view == _VIEW_MINMAX:
            return RecordedView(
                key="watch_minmax",
                page="watch_minmax",
                label=f"Watch · MIN/MAX grids ({phase})",
                params={
                    "layers": tuple(watched),
                    "phase": phase,
                    "grids": (state.grid_type,),
                    "heatmap": state.heat_on,
                    "input_mean": input_mean,
                    "input_std": input_std,
                },
            )
        return RecordedView(
            key="watch_histogram",
            page="watch_histogram",
            label=f"Watch · histograms ({phase})",
            params={
                "layers": tuple(watched),
                "phase": phase,
                "log_x": state.axis_log_x,
                "log_y": state.axis_log_y,
            },
        )

    with ui.column().classes("w-full h-screen no-wrap gap-0"):
        with _top_bar_row():
            back_button = _back_button(
                state.selected_layer if session.locked else None
            )
            _add_step_controls(session, step_until_custom)
            _add_settings_button(session, record_view).classes("ml-auto")
            ui.button(
                icon="refresh",
                on_click=lambda: _refresh_now(session, refresh),
                color="slate-500",
            ).props("dense size=md flat").tooltip(
                "Refresh now — and from the next training batch while running"
            )
            _add_tour_button()
            _add_share_button()
            _add_repo_logo()

        _add_error_banner(session)

        with ui.row().classes("w-full grow min-h-0 no-wrap gap-0"):
            with ui.column().classes(
                "w-80 shrink-0 h-full overflow-auto p-4 gap-2 "
                "border-r-2 border-slate-300 bg-slate-50"
            ).props(_resizable_pane_props("watch-controls")):
                with ui.row().classes("items-baseline gap-2 no-wrap"):
                    ui.label("Stats").classes("font-mono text-base font-bold")
                    state.count_label = ui.label("").classes(
                        "text-sm text-slate-500"
                    )
                ui.separator()
                # `data-tour` marks the three dropdowns as the tour's arrow
                # targets (`tour.stats_tour_steps`).
                view_select = ui.select(
                    [_VIEW_HISTOGRAM, _VIEW_MINMAX, _VIEW_GRAPHS],
                    label="View",
                    value=state.view,
                    on_change=lambda e: set_mode(e.value),
                ).props('dense outlined options-dense data-tour="view"').classes(
                    "w-full text-sm"
                ).tooltip("What each layer card shows")
                # Phases, then "Current batch" as the last entry (dropped in
                # the epoch-stats view — see `sync_phase_select`). A scoped
                # `option` slot draws a divider above it (Quasar has no native
                # per-option separator) while keeping default selection via
                # `itemProps`.
                phase_select = ui.select(
                    _phase_select_options(state.view, phase_names),
                    label="Phase",
                    value=state.selected_phase,
                    on_change=lambda e: set_phase(e.value),
                ).props('dense outlined options-dense data-tour="phase"').classes(
                    "w-full text-sm"
                ).tooltip(
                    "Which phase the cards show — or the last captured batch's "
                    "stats for any layer (watched or not)"
                )
                # The divider keys off the label, not the value: NiceGUI sets
                # each option's `value` to its integer index, so only the label
                # carries our sentinel text.
                phase_select.add_slot(
                    "option",
                    "<q-separator v-if=\"props.opt.label === "
                    f"'{_PHASE_CURRENT_BATCH_LABEL}'\" />"
                    '<q-item v-bind="props.itemProps">'
                    "<q-item-section><q-item-label>"
                    "{{ props.opt.label }}"
                    "</q-item-label></q-item-section></q-item>",
                )
                layer_select = ui.select(
                    {},
                    label="Layer",
                    on_change=lambda e: set_layer(e.value),
                ).props('dense outlined options-dense data-tour="layer"').classes(
                    "w-full text-sm"
                ).tooltip(
                    "Which layer's cards to show — one keeps the page fast. In "
                    "a phase, the watched layers; in Current batch, any layer. "
                    f'"all" is offered with fewer than {_ALL_LAYERS_MAX} options'
                )
                hist_boxes: list[ui.checkbox] = []
                minmax_boxes: list[DisableableElement] = []
                with ui.column().classes("w-full gap-1") as hist_controls:
                    hist_boxes.append(
                        ui.checkbox(
                            "Log x",
                            value=state.axis_log_x,
                            on_change=lambda e: set_axis_log_x(bool(e.value)),
                        ).props("dense").classes("text-sm").tooltip(
                            "Log-based (signed-log) scale on the value axis"
                        )
                    )
                    hist_boxes.append(
                        ui.checkbox(
                            "Log y",
                            value=state.axis_log_y,
                            on_change=lambda e: set_axis_log_y(bool(e.value)),
                        ).props("dense").classes("text-sm").tooltip(
                            "Log scale on the probability axis"
                        )
                    )
                    hist_boxes.append(
                        ui.checkbox(
                            "Retain axes",
                            value=state.retain_axes,
                            on_change=lambda e: set_retain_axes(bool(e.value)),
                        ).props("dense").classes("text-sm").tooltip(
                            "Keep the current axis ranges when toggling Log x / "
                            "Log y or switching phase, instead of auto-fitting "
                            "to the data"
                        )
                    )
                    hist_boxes.append(
                        ui.checkbox(
                            "Show subnormal/overflow",
                            value=state.show_bands,
                            on_change=lambda e: set_show_bands(bool(e.value)),
                        ).props("dense").classes("text-sm").tooltip(
                            "Mark the dtype's subnormal and overflow "
                            "(near-saturation) magnitude bands with dotted lines "
                            "— in-range edges only (fp32's sit off the axis)"
                        )
                    )
                with ui.column().classes("w-full gap-1") as minmax_controls:
                    grid_radio = (
                        ui.radio(
                            _grid_type_options(
                                session.watch_performance.average_patches
                            ),
                            value=state.grid_type,
                            on_change=lambda e: set_grid(e.value),
                        ).props("dense").classes("text-sm").tooltip(
                            "Which extreme-activation patch grid to show"
                        )
                    )
                    minmax_boxes.append(grid_radio)
                    minmax_boxes.append(
                        ui.checkbox(
                            "Enable heatmap",
                            value=state.heat_on,
                            on_change=lambda e: set_heat(bool(e.value)),
                        ).props("dense").classes("text-sm").tooltip(
                            "Blend each channel's activation strength over the "
                            "patches (red positive, blue negative), with a scale "
                            "next to each grid"
                        )
                    )
                hist_controls.set_visibility(state.view == _VIEW_HISTOGRAM)
                minmax_controls.set_visibility(state.view == _VIEW_MINMAX)
                # Pinned to the very bottom of the sidebar (below a flexible
                # spacer): jump to the same layer's Deep Dream experiment, the
                # synthesized counterpart to these real-input extremes (point 3).
                # Shown only in the MIN/MAX view, like the controls above.
                ui.space()
                compare_deep_dream = (
                    ui.button(
                        "Compare with Deep Dream",
                        icon="science",
                        color="yellow-8",
                    )
                    .props("dense no-caps size=sm")
                    .classes("w-full")
                    .tooltip(
                        "Open this layer's Deep Dream experiment — the inputs "
                        "synthesized to excite the same channels"
                    )
                )
                compare_deep_dream.set_visibility(state.view == _VIEW_MINMAX)
                sync_compare_href()

            _resize_handle("watch-controls", "left")
            body_container = ui.column().classes(
                "grow min-w-0 h-full overflow-auto p-4 gap-3 bg-slate-200"
            )

    def _tour_set_view(e: GenericEventArguments) -> None:
        """Switch to the view a tour step describes (`TourStep.ensure_view`).

        Emitted by the tour driver whenever a view-bound step is shown; the
        write goes through the View dropdown so the widget and `set_mode`
        stay in sync, and a no-op (already on that view) is skipped — the
        driver re-emits on every re-show.
        """
        view = str(e.args)
        if view != state.view and view in (
            _VIEW_HISTOGRAM,
            _VIEW_MINMAX,
            _VIEW_GRAPHS,
        ):
            state.tour_view_write = True
            view_select.set_value(view)

    def _tour_start(_: GenericEventArguments) -> None:
        """Snapshot the view a fresh tour run starts from.

        The run's view-bound steps switch the page around on the visitor's
        behalf (`_tour_set_view`); dismissing the tour puts this view back
        (`_tour_end`) unless the visitor picked one themselves meanwhile.
        """
        state.tour_saved_view = state.view
        state.tour_user_set_view = False

    def _tour_end(_: GenericEventArguments) -> None:
        """Restore the pre-tour view when the ended run switched it away."""
        restore = _tour_restore_view(
            state.tour_saved_view, state.tour_user_set_view, state.view
        )
        state.tour_saved_view = None
        if restore is not None:
            state.tour_view_write = True
            view_select.set_value(restore)

    ui.on("nansense_tour_set_view", _tour_set_view)
    ui.on("nansense_tour_start", _tour_start)
    ui.on("nansense_tour_end", _tour_end)

    def sync_phase_select() -> None:
        """Refresh the Phase dropdown's options/value for the current view.

        The epoch-stats view has no "Current batch" entry (a single batch
        has no epoch series), and such a selection is swapped for the first
        schedule phase — the epoch-aggregating counterpart. Options are
        re-read from the schedule each pass so lazily discovered phases
        appear without reopening the page. Pushes to the widget only on an
        actual change (a no-op write would re-enter `set_phase`).
        """
        names = list(session.schedule.phase_order)
        state.selected_phase = _reconcile_selected_phase(
            state.selected_phase, state.view, names
        )
        options = _phase_select_options(state.view, names)
        if phase_select.options != options:
            phase_select.set_options(options, value=state.selected_phase)
        elif phase_select.value != state.selected_phase:
            phase_select.set_value(state.selected_phase)

    def sync_grid_type_select() -> None:
        """Refresh the MIN/MAX radio's options for the Performance setting.

        The average-extreme galleries are collected only while their
        Performance setting is on (`WatchPerformance.average_patches`), so
        the radio offers those entries only then — a dead option's body
        could only say "not collected". An average selection whose entry
        just disappeared falls back to the "Max pixel" default. Pushes to
        the widget only on an actual change (a no-op write would re-enter
        `set_grid`).
        """
        options = _grid_type_options(session.watch_performance.average_patches)
        state.grid_type = _reconcile_grid_type(state.grid_type, options)
        if grid_radio.options != options:
            grid_radio.set_options(options, value=state.grid_type)
        elif grid_radio.value != state.grid_type:
            grid_radio.set_value(state.grid_type)

    def sync_layer_select() -> None:
        """Refresh the layer dropdown's options/value from the stats layers.

        Reconciles the selection (drops "all" once too many layers carry
        stats, replaces a layer that no longer does), pushes the
        options/value to the widget only when they actually changed (a no-op
        write would re-enter `set_layer`), and disables the dropdown when no
        layer carries stats. Cheap enough to call every refresh tick.
        """
        ordered = _selectable_layers(
            state.selected_phase, layer_names, session.stats_layers
        )
        state.selected_layer = _reconcile_selected_layer(
            state.selected_layer, ordered
        )
        options = _layer_select_options(ordered)
        if layer_select.options != options:
            layer_select.set_options(options, value=state.selected_layer)
        elif layer_select.value != state.selected_layer:
            layer_select.set_value(state.selected_layer)
        _set_controls_enabled([layer_select], bool(ordered))

    def rebuild_cards() -> None:
        sync_layer_select()
        layer_panels.clear()
        # Drop the previous cards' hover routes; the rebuilt plots re-register
        # below. Without this the registry (and the dead plots it points at)
        # would grow on every rebuild.
        hover_registry.clear()
        body_container.clear()
        ordered = _selectable_layers(
            state.selected_phase, layer_names, session.stats_layers
        )
        with body_container:
            if not ordered:
                with ui.column().classes("items-center gap-2 py-12 w-full"):
                    ui.icon("visibility_off", size="lg").classes("text-slate-400")
                    ui.label("No layers selected.").classes("text-slate-600")
                    ui.label(
                        "Go back and click the eye icon on a layer card "
                        "to start watching."
                    ).classes("text-slate-500 text-sm")
                return
            for name in _visible_layers(state.selected_layer, ordered):
                layer_panels[name] = _WatchLayerPanel(
                    name=name,
                    session=session,
                    on_unwatched=rebuild_cards,
                    state=state,
                    hover_registry=hover_registry,
                    input_mean=input_mean,
                    input_std=input_std,
                )

    # Single-flight refresh: snapshotting and grid rendering run in a worker
    # thread so the event loop keeps serving websocket traffic (a blocked
    # loop starves keepalive pings and kills the connection). A toggle that
    # lands while a pass is in flight just marks it dirty — rapid Heatmap
    # clicks coalesce into one follow-up pass instead of queueing a full
    # re-render per click.
    async def refresh() -> None:
        if state.refresh_running:
            state.refresh_dirty = True
            return
        state.refresh_running = True
        try:
            while True:
                state.refresh_dirty = False
                stats_layers = session.stats_layers
                n = len(stats_layers)
                state.count_label.text = (
                    f"{n} layer{'' if n == 1 else 's'}"
                )
                sync_phase_select()
                sync_layer_select()
                sync_grid_type_select()
                sync_compare_href()
                sync_back_href()
                ordered = _selectable_layers(
                    state.selected_phase, layer_names, stats_layers
                )
                desired = _visible_layers(state.selected_layer, ordered)
                if list(layer_panels) != desired:
                    rebuild_cards()
                panels = dict(layer_panels)
                minmax = state.view == _VIEW_MINMAX
                graphs = state.view == _VIEW_GRAPHS
                current_batch = state.selected_phase == _PHASE_CURRENT_BATCH

                def compute(
                    panels: dict[str, _WatchLayerPanel] = panels,
                    minmax: bool = minmax,
                    graphs: bool = graphs,
                    current_batch: bool = current_batch,
                ) -> tuple[
                    WatchSnapshot,
                    dict[str, tuple[tuple[object, ...], str] | None],
                    MetricsSnapshot | None,
                ]:
                    # "Current batch" computes stats from the last snapshot for
                    # exactly the visible layers; a phase reads the running
                    # watch aggregates. Either way the patch GPU→CPU work is
                    # only paid when the MIN/MAX view will consume it, and the
                    # custom-metric copy only when the GRAPHS view shows it
                    # (Current batch never does — a batch has no epoch series).
                    metrics: MetricsSnapshot | None = None
                    if current_batch:
                        snap = session.current_batch_stats(
                            layers=list(panels), include_patches=minmax
                        )
                    else:
                        snap = session.watch_snapshot(include_patches=minmax)
                        if graphs:
                            metrics = session.watch_metrics_snapshot(
                                layers=list(panels)
                            )
                    grids: dict[str, tuple[tuple[object, ...], str] | None] = {}
                    if minmax:
                        for name, panel in panels.items():
                            grids[name] = panel.prepare_grids(snap)
                    return snap, grids, metrics

                snap, grids, metrics = await asyncio.to_thread(compute)
                for name, panel in panels.items():
                    if layer_panels.get(name) is panel:  # not rebuilt meanwhile
                        panel.update(snap, grids.get(name), metrics)
                if not state.refresh_dirty:
                    return
        finally:
            state.refresh_running = False

    def sync_frozen() -> None:
        hist = session.recording.is_recording("watch_histogram")
        minmax = session.recording.is_recording("watch_minmax")
        if hist != state.frozen_hist or minmax != state.frozen_minmax:
            state.frozen_hist = hist
            state.frozen_minmax = minmax
            # Phase and layer apply to both views and define the recorded
            # frame set, so either recording locks them.
            _set_controls_enabled(
                [phase_select, layer_select], not (hist or minmax)
            )
            _set_controls_enabled(hist_boxes, not hist)
            _set_controls_enabled(minmax_boxes, not minmax)

    # Build the initial cards (or the empty-state notice) up front so the
    # body isn't blank until the first refresh tick lands.
    rebuild_cards()
    # Seed the gate with the current session state: the once-timer below
    # renders exactly that state, so the first periodic tick must not pass
    # for it again.
    gate = _RefreshGate()
    gate.should_refresh(session)

    async def tick() -> None:
        sync_frozen()
        if gate.should_refresh(session):
            await refresh()

    ui.timer(0.0, refresh, once=True)
    ui.timer(0.2, tick)


def _plotly_restyle(
    plot: ui.plotly,
    update: dict[str, object],
    indices: list[int],
    layout: dict[str, object] | None = None,
) -> None:
    """Update trace attributes of an existing figure in place.

    Calls `Plotly.update` on the live graph div instead of replacing the
    figure, so client-side state (legend visibility toggles, zoom/pan) is left
    untouched. `update` maps each attribute to a list of per-trace values
    (e.g. `{"y": [y0, y1]}`) aligned with `indices`. `layout` optionally
    carries relayout-style updates (e.g. `{"yaxis.range": [0, 5]}`) applied in
    the same call; note an explicit axis-range write does reset any user zoom
    on that axis.

    The guard makes this a no-op until Plotly's module has loaded and drawn the
    graph (a just-connected client can fire a timer tick before then); the next
    refresh re-applies the data, so nothing is lost.
    """
    ui.run_javascript(
        f"const gd = getHtmlElement({plot.id}); "
        f"if (gd && gd.data && window.Plotly) "
        f"window.Plotly.update(gd, {json.dumps(update)}, "
        f"{json.dumps(layout or {})}, {json.dumps(indices)});"
    )


# Plotly config shared by all watch histograms. "Autoscale" would expand the
# axes to fit every bar — including the freak spikes and outlier tails the
# capped ranges deliberately clip — landing on a different scale than the
# initial render, so the button is removed; "Reset axes" and double-click
# restore the ranges the figure was built with instead.
_PLOTLY_CONFIG: dict[str, object] = {
    "modeBarButtonsToRemove": ["autoScale2d"],
    "doubleClick": "reset",
}


def _figure_payload(fig: go.Figure) -> dict[str, object]:
    """The data/layout/config dict NiceGUI hands to `Plotly.react`."""
    return {**fig.to_plotly_json(), "config": _PLOTLY_CONFIG}


# One shared global event for every histogram's hover; the emitted payload
# carries the source element id so the page-level handler can route it to the
# right plot. Using a single stable event (rather than one per element id)
# keeps a card rebuild from piling up dead `ui.on` handlers on the page layout.
_HOVER_EVENT: str = "nansense_hist_hover"


def _hover_attach_js(element_id: int) -> str:
    """JS that wires the figure's `plotly_hover` to a NiceGUI event.

    Plotly events fire on the graph div's own emitter, not as DOM events,
    so NiceGUI's `.on()` can't subscribe to them — the handler is attached
    with `gd.on` instead, with retries until Plotly has drawn the figure
    (which is when `gd.on` exists). Idempotent via the `_nansenseHover`
    flag, and throttled to one event per 200 ms so hovering across many
    bars doesn't flood the websocket. Handlers attached to the div survive
    `Plotly.react`/`update`, so one attach covers later figure rebuilds. The
    event is emitted on the shared `_HOVER_EVENT` channel with this element's
    id so the page's single handler can dispatch it to the right plot.
    """
    return (
        "(function attach(tries) {"
        f"const gd = getHtmlElement({element_id});"
        "if (!gd || !gd.on) {"
        "  if (tries > 0) setTimeout(() => attach(tries - 1), 300);"
        "  return;"
        "}"
        "if (gd._nansenseHover) return;"
        "gd._nansenseHover = true;"
        "gd.on('plotly_hover', (ev) => {"
        "  const p = ev.points && ev.points[0];"
        "  if (!p) return;"
        "  const now = Date.now();"
        "  if (gd._nansenseHoverAt && now - gd._nansenseHoverAt < 200) return;"
        "  gd._nansenseHoverAt = now;"
        f"  emitEvent('{_HOVER_EVENT}', {{id: {element_id}, bin: p.pointNumber}});"
        "});"
        "})(20);"
    )


def _reveal_samples_js(element_id: int) -> str:
    """JS that scrolls the just-filled bin-sample strip into view.

    The strip sits under a tall plot, so on a typical viewport it lands
    below the fold and the first hover looks like a no-op without this
    nudge. Scrolling only happens when the strip's bottom edge is actually
    off-screen, and `block: 'nearest'` keeps the scroll minimal so the
    hovered plot mostly stays put.
    """
    return (
        "(function() {"
        f"const el = getHtmlElement({element_id});"
        "if (!el) return;"
        "const bottom = el.getBoundingClientRect().bottom;"
        "if (bottom <= window.innerHeight) return;"
        "el.scrollIntoView({block: 'nearest', behavior: 'smooth'});"
        "})();"
    )


def _bin_samples_note(text: str) -> str:
    return f'<div class="text-xs text-slate-400 italic py-1">{text}</div>'


_HOVER_HINT_HTML: str = _bin_samples_note(
    "hover a bar to see a few random input samples from that value range "
    "(drawn from the last captured batch only)"
)


def _bin_samples_html(
    snapshot: BatchSnapshot | None,
    layer: str,
    kind: str,
    channel: int,
    bin_idx: int,
    input_name: str | None,
    mean: tuple[float, ...] | None,
    std: tuple[float, ...] | None,
    k: int = 4,
) -> str:
    """The hover strip for one (channel, bin) bar of a per-channel histogram.

    The histogram aggregates whole epochs, but its source values are
    discarded every batch — samples can only come from the last captured
    batch (`session.snapshot`), and every caption names that batch so the
    narrower population is explicit.
    """
    if snapshot is None:
        return _bin_samples_note(
            "no batch captured yet — sampling needs a captured batch"
        )
    pos = snapshot.position
    source = (
        "last captured batch only — "
        f"{html.escape(pos.phase)} ep {pos.epoch}, batch {pos.batch_idx}"
    )
    tensors = (
        snapshot.activations
        if kind == "activation"
        else snapshot.activation_gradients
    )
    tensor = tensors.get(layer)
    if tensor is None:
        return _bin_samples_note(
            f"no captured {kind}s for this layer in the {source}"
        )
    input_tensor = snapshot.activations.get(input_name) if input_name else None
    samples = sample_bin(
        tensor, input_tensor, channel=channel, bin_idx=bin_idx, k=k
    )
    header = (
        '<div class="text-xs text-slate-600">'
        f'<span class="font-bold">ch {channel}</span>, '
        f"value ≈ {_BIN_VALUE_LABELS[bin_idx]} — random samples, "
        f'<span class="font-bold">{source}</span></div>'
    )
    if not samples:
        return header + _bin_samples_note(
            "no values in this bar in the last captured batch "
            "(the bar may aggregate earlier batches)"
        )
    cells: list[str] = []
    for sample in samples:
        image = (
            render_image(
                sample.image.unsqueeze(0), 0, mean=mean, std=std
            )
            if sample.image is not None
            else None
        )
        img_html = (
            f'<img src="{_b64_img_src(image)}" '
            'style="width:64px;image-rendering:pixelated;display:block;" />'
            if image is not None
            else '<div class="w-16 h-16 bg-slate-200 rounded"></div>'
        )
        cells.append(
            '<div class="flex flex-col items-center gap-0.5">'
            + img_html
            + f'<div class="text-[10px] font-mono text-slate-600">'
            f"{_format_stat(sample.value)}</div>"
            f'<div class="text-[10px] text-slate-400">'
            f"sample {sample.sample_idx}</div></div>"
        )
    return header + '<div class="flex gap-3 py-1">' + "".join(cells) + "</div>"


# `data-tour` anchors for the tour's histograms step (`tour.stats_tour_steps`),
# one per tensor kind; spelled out literally so the anchor-wiring test can
# find them in the source. The first visible plot of each kind gets the arrow.
_HIST_TOUR_ANCHORS: dict[str, str] = {
    "activation": 'data-tour="hist-activation"',
    "gradient": 'data-tour="hist-gradient"',
}


class _HistPlot:
    """One Plotly histogram figure that refreshes its data in place.

    The figure (one subplot row per phase) is built once and rebuilt only
    when the set of phases or the axis scale
    changes. Routine per-tick updates go through `Plotly.update`, which
    leaves client-side state — zoom/pan — untouched.

    A "Per channel" switch narrows the plot from the universal histogram to
    a single channel's row of the per-channel histogram (dim 1 of the
    tensor), stepped through with an index spinner. While per-channel,
    hovering a bar fills the strip below the plot with a few random input
    samples whose values landed in that bar — drawn from the *last captured
    batch* only, since the running histogram's source values are discarded
    every batch; the strip's caption spells that out.
    """

    def __init__(
        self,
        kind: str,
        title: str,
        state: _WatchPageState,
        *,
        session: Session,
        layer: str,
        hover_registry: dict[int, _HistPlot],
        input_mean: tuple[float, ...] | None = None,
        input_std: tuple[float, ...] | None = None,
    ) -> None:
        self._kind = kind
        self._title = title
        self._state = state
        self._session = session
        self._layer = layer
        self._hover_registry = hover_registry
        self._input_mean = input_mean
        self._input_std = input_std
        self._per_channel = False
        self._channel = 0
        # Channel count of the latest data seen; `None` until per-channel
        # rows exist (no data yet, 1D tensors, or collapsed buffers).
        self._channel_count: int | None = None
        # Last stats handed to `update`, so control changes re-render
        # immediately instead of waiting for the next 2 s tick.
        self._last_per_phase: dict[str, LayerStatsSnapshot] = {}
        # Signature of what's currently drawn, so `update` can tell a plain
        # data refresh (restyle) from a structural change (rebuild).
        self._phases: list[str] = []
        self._axis = self._current_axis()
        # The subnormal/overflow band currently drawn (band-edge lines are
        # layout shapes that only a rebuild can add/remove/move), so a toggle of
        # the "Show subnormal/overflow" checkbox — or the dtype first becoming
        # known — forces a rebuild.
        self._band: tuple[float, float] | None = None
        # Last axis ranges applied (set by every figure build, including the
        # empty one below), so refreshes only push a relayout when a cap
        # actually moved (a range write resets zoom on that axis).
        self._y_range: list[float] | None = None
        self._x_range: list[float] | None = None
        # The retained linear y-cap (and the density mode it was measured in),
        # tracked while "Retain axes" is off so it's current the moment it
        # turns on — see `_retained_ranges` / `_capture_y_top`.
        self._y_top: float | None = None
        self._y_top_density: bool = use_density(self._axis[0])
        with ui.row().classes("items-center gap-x-3 no-wrap"):
            self._channel_switch = (
                ui.switch("Per channel", value=False, on_change=self._set_mode)
                .props("dense")
                .classes("text-sm")
            )
            self._channel_switch.tooltip(
                "Show one channel's histogram instead of all values pooled; "
                "hover a bar to sample inputs from that value range"
            )
            self._channel_spinner = (
                ui.number(
                    value=0,
                    min=0,
                    step=1,
                    format="%d",
                    on_change=self._set_channel,
                )
                .props("dense outlined")
                .classes("w-24")
                .tooltip("Which channel to show")
            )
            self._channel_total = ui.label("").classes("text-xs text-slate-500")
        fig, (self._x_range, self._y_range) = _make_histogram_figure(
            {}, kind, title, log_x=self._axis[0], log_y=self._axis[1]
        )
        self.element = (
            ui.plotly(_figure_payload(fig))
            .classes("w-full")
            .props(_HIST_TOUR_ANCHORS[kind])
        )
        self._samples = ui.html(_HOVER_HINT_HTML).classes("w-full")
        self._sync_control_visibility()
        # Route hovers through the page's single shared handler (see
        # `_HOVER_EVENT`); the registry is cleared on each card rebuild, so
        # this view is released instead of lingering in a global `ui.on`.
        self._hover_registry[self.element.id] = self

    def _current_axis(self) -> tuple[bool, bool]:
        return self._state.axis_log_x, self._state.axis_log_y

    def _set_mode(self, e: ValueChangeEventArguments) -> None:
        self._per_channel = bool(e.value)
        self._sync_control_visibility()
        self.update(self._last_per_phase)

    def _set_channel(self, e: ValueChangeEventArguments) -> None:
        value = e.value if isinstance(e.value, (int, float)) else 0
        self._channel = max(0, int(value))
        self.update(self._last_per_phase)

    def _sync_control_visibility(self) -> None:
        self._channel_spinner.set_visibility(self._per_channel)
        self._channel_total.set_visibility(self._per_channel)
        self._samples.set_visibility(self._per_channel)
        if self._per_channel:
            self._samples.set_content(_HOVER_HINT_HTML)
            # Attaching is idempotent client-side; (re-)sending it on every
            # mode flip covers clients that connected after page build.
            ui.run_javascript(_hover_attach_js(self.element.id))

    def _channel_rows(
        self, per_phase: dict[str, LayerStatsSnapshot]
    ) -> tuple[tuple[int, ...], ...] | None:
        """The drawn phase's per-channel rows, `None` when unavailable."""
        for snap in per_phase.values():
            rows = kind_stats(snap, self._kind).channel_hists
            if rows is not None:
                return rows
        return None

    def _sync_channel_controls(
        self, per_phase: dict[str, LayerStatsSnapshot]
    ) -> None:
        rows = self._channel_rows(per_phase)
        self._channel_count = len(rows) if rows is not None else None
        if self._channel_count is None:
            self._channel_total.text = "(no per-channel data)"
            return
        self._channel = min(self._channel, self._channel_count - 1)
        self._channel_total.text = f"of {self._channel_count} channels"
        self._channel_spinner.max = self._channel_count - 1
        if self._channel_spinner.value != self._channel:
            _defer_value_write(
                lambda: self._channel_spinner.set_value(self._channel)
            )

    def _view(
        self, per_phase: dict[str, LayerStatsSnapshot]
    ) -> dict[str, LayerStatsSnapshot]:
        """`per_phase` with each phase's histogram narrowed to the channel.

        Falls back to the universal histogram for phases without
        per-channel rows (1D tensors, collapsed older epochs).
        """
        if not self._per_channel:
            return per_phase
        field = "activations" if self._kind == "activation" else "gradients"
        return {
            # `narrow_to_channel` clamps the index and passes a stream without
            # per-channel rows through untouched — shared with the MCP server's
            # own per-channel view so the two cannot disagree on either rule.
            phase: replace(
                snap,
                **{field: narrow_to_channel(kind_stats(snap, self._kind), self._channel)},
            )
            for phase, snap in per_phase.items()
        }

    def _trace_names(self, view: dict[str, LayerStatsSnapshot]) -> list[str]:
        suffix = (
            f" — ch {self._channel}"
            if self._per_channel and self._channel_count is not None
            else ""
        )
        # In "Current batch" mode the single entry keeps its captured phase
        # key, so a "train" → "Current batch" flip can leave `update`'s
        # phase-set rebuild check unchanged — the labels still refresh
        # because the restyle path re-sends the trace names and subplot
        # titles on every update.
        current_batch = self._state.selected_phase == _PHASE_CURRENT_BATCH
        phases = _phases_with_data(view, self._kind)
        return [
            _phase_heading(p, view[p].epoch, current_batch=current_batch)
            + suffix
            for p in phases
        ]

    async def _on_hover(self, e: GenericEventArguments) -> None:
        if not self._per_channel:
            return
        bin_idx = int(e.args.get("bin", -1))
        if not 0 <= bin_idx < N_BINS:
            return
        snapshot = self._session.snapshot
        input_names = self._session.input_names
        content = await asyncio.to_thread(
            _bin_samples_html,
            snapshot,
            self._layer,
            self._kind,
            self._channel,
            bin_idx,
            input_names[0] if input_names else None,
            self._input_mean,
            self._input_std,
        )
        self._samples.set_content(content)
        ui.run_javascript(_reveal_samples_js(self._samples.id))

    def _under_over_band(
        self, per_phase: dict[str, LayerStatsSnapshot]
    ) -> tuple[float, float] | None:
        """The dtype-aware band edges for this stream, or `None`.

        `None` when the checkbox is off or no data dtype is known yet (the band
        is dtype-derived, so it can't be placed until a tensor has been seen).
        """
        if not self._state.show_bands:
            return None
        for snap in per_phase.values():
            dtype = kind_stats(snap, self._kind).dtype
            if dtype is not None:
                return debugger.dtype_band(dtype)
        return None

    def update(self, per_phase: dict[str, LayerStatsSnapshot]) -> None:
        self._last_per_phase = per_phase
        self._sync_channel_controls(per_phase)
        per_phase = self._view(per_phase)
        phases = _phases_with_data(per_phase, self._kind)
        axis = self._current_axis()
        log_x, log_y = axis
        density = use_density(log_x)
        retain = self._state.retain_axes
        phase_hists = _phase_hists(per_phase, self._kind)
        band = self._under_over_band(per_phase)
        if phases != self._phases or axis != self._axis or band != self._band:
            # A phase appeared/disappeared, an axis-scale checkbox flipped, or
            # the under/overflow band toggled — rebuild the whole figure (the
            # band-edge lines are layout shapes only a rebuild can change).
            # With "Retain axes" on, carry the current view across
            # (re-expressed for the new scale); else let the build fit the
            # ranges to the data and cache them.
            override = (
                self._retained_ranges(axis, phase_hists) if retain else None
            )
            fig, (self._x_range, self._y_range) = _make_histogram_figure(
                per_phase,
                self._kind,
                self._title,
                log_x=log_x,
                log_y=log_y,
                trace_names=self._trace_names(per_phase),
                override_ranges=override,
                under_over_band=band,
            )
            self.element.update_figure(_figure_payload(fig))
            self._phases = phases
            self._axis = axis
            self._band = band
            if not retain:
                self._capture_y_top(phase_hists, density, self._y_range)
        elif phases:
            # Same rows and axes — only counts (and the epoch label) moved.
            # Restyle in place so zoom/pan survives. A channel index change
            # lands here too: same structure, new bar heights.
            hists: list[tuple[int, ...]] = []
            for p in phases:
                hist = kind_stats(per_phase[p], self._kind).hist
                assert hist is not None  # `phases` excludes collapsed buckets
                hists.append(hist)
            names = self._trace_names(per_phase)
            update: dict[str, object] = {
                "name": names,
                "y": [trace_heights(h, density) for h in hists],
                "customdata": [_hover_customdata(h, density) for h in hists],
            }
            # The subplot titles carry the epoch, so refresh them with the
            # data (annotation order matches row order).
            layout: dict[str, object] = {
                f"annotations[{i}].text": name for i, name in enumerate(names)
            }
            # While retaining, the ranges are frozen — leave them untouched so
            # the kept view (and any client zoom) survives the refresh. Else
            # the caps follow the data, re-applied only when they moved so an
            # idle refresh doesn't keep snapping the user's zoom back.
            if not retain:
                x_range, y_range = _axis_ranges(
                    phase_hists, log_x=log_x, log_y=log_y
                )
                if y_range is not None and y_range != self._y_range:
                    # Every row gets the same range (row 1 is "yaxis", row n is
                    # "yaxis{n}") so the subplots stay comparable.
                    self._y_range = y_range
                    for i in range(len(phases)):
                        axis_name = "yaxis" if i == 0 else f"yaxis{i + 1}"
                        layout[f"{axis_name}.range"] = y_range
                if x_range != self._x_range:
                    # The rows' x-axes are matched, so one key updates every row.
                    self._x_range = x_range
                    layout["xaxis.range"] = x_range
                    # On the linear value axis the bars sit at bin centres with
                    # the off-view tail bins blanked (see `_linear_bar_x`); that
                    # blanking tracks the visible range, so a moved range
                    # re-blanks the bars.
                    if density:
                        bar_x = _linear_bar_x(x_range)
                        update["x"] = [bar_x for _ in phases]
                # `_axis_ranges` returns no y-range on a log-y axis, so this
                # is the linear cap when there is one and `None` otherwise.
                self._capture_y_top(phase_hists, density, y_range)
            n = len(phases)
            _plotly_restyle(self.element, update, list(range(n)), layout)
            # Refresh the overflow markers (trace n..2n-1) against the applied
            # cap and x-positions, so clipped bars stay flagged as the data and
            # ranges move. No cap on a log-y axis → no marks.
            x_values = (
                list(range(N_BINS)) if log_x else _linear_bar_x(self._x_range)
            )
            y_top = (
                self._y_range[1]
                if (self._y_range is not None and not log_y)
                else None
            )
            marks = overflow_marks(phase_hists, x_values, density, y_top)
            _plotly_restyle(
                self.element,
                {"x": [m[0] for m in marks], "y": [m[1] for m in marks]},
                list(range(n, 2 * n)),
            )

    def _capture_y_top(
        self,
        phase_hists: list[tuple[str, tuple[int, ...]]],
        density: bool,
        linear_y_range: list[float] | None,
    ) -> None:
        """Track the linear y-cap so it's ready when "Retain axes" turns on.

        `linear_y_range` is the freshly computed linear-y range when one is at
        hand (the cap is its top); on a log-y axis there isn't one, so the
        linear cap is computed separately.
        """
        if linear_y_range is not None:
            self._y_top = linear_y_range[1]
        else:
            _, lin = _axis_ranges(phase_hists, log_x=self._axis[0], log_y=False)
            self._y_top = lin[1] if lin is not None else None
        self._y_top_density = density

    def _retained_ranges(
        self,
        target_axis: tuple[bool, bool],
        phase_hists: list[tuple[str, tuple[int, ...]]],
    ) -> tuple[list[float] | None, list[float] | None]:
        """The `(x_range, y_range)` that keep the current view on a rebuild.

        The x-window is preserved, re-expressed between the linear value axis
        and the signed-log bin-index axis when Log x flipped. The linear y-cap
        is preserved too, re-expressed for the y-scale; only a Log x flip
        (which swaps the bar units between density and probability) re-fits it
        from the data. Falls back to a data fit before any view exists.
        """
        new_log_x, new_log_y = target_axis
        old_log_x = self._axis[0]
        if self._x_range is None:
            x_range, _ = _axis_ranges(
                phase_hists, log_x=new_log_x, log_y=new_log_y
            )
        elif new_log_x == old_log_x:
            x_range = list(self._x_range)
        elif new_log_x:
            x_range = _x_range_linear_to_log(self._x_range)
        else:
            x_range = _x_range_log_to_linear(self._x_range)
        new_density = use_density(new_log_x)
        if self._y_top is None or self._y_top_density != new_density:
            _, lin = _axis_ranges(phase_hists, log_x=new_log_x, log_y=False)
            self._y_top = lin[1] if lin is not None else None
            self._y_top_density = new_density
        floor = _min_positive_height(phase_hists, new_density)
        y_range = _retained_y_range(self._y_top, log_y=new_log_y, floor=floor)
        return x_range, y_range


class _EpochStatsPlot:
    """One value-vs-epoch Plotly figure that refreshes its data in place.

    The figure's trace set is fixed (see `make_epoch_stats_figure`), so
    refreshes are `Plotly.update`s of the trace arrays — client-side state
    (which stats the legend has deselected, zoom/pan) survives. The x-tick
    spacing rides along in the same call so epoch ticks stay integral as
    the run grows. The *first* data delivery instead replaces the whole
    figure through the element: a JS restyle silently no-ops while
    plotly.js is still loading, and with the refresh gate a paused session
    may not re-render until the next publish — so the initial data must
    live in the element's own payload, which the client draws whenever it
    is ready. There is no client state to lose at that point.
    """

    def __init__(self, kind: str, title: str) -> None:
        self._kind = kind
        self._title = title
        self._has_data = False
        fig = make_epoch_stats_figure(kind, title)
        # Every epoch plot carries the tour's GRAPHS anchor; the arrow lands
        # on the first visible one (`tour.stats_tour_steps`).
        self.element = (
            ui.plotly(_figure_payload(fig))
            .classes("w-full")
            .props('data-tour="epoch-graph"')
        )

    def update(self, history: list[LayerStatsSnapshot]) -> None:
        self.update_series(*epoch_stat_series(history, self._kind))

    def update_series(
        self, epochs: list[int], series: dict[str, list[float | None]]
    ) -> None:
        """Apply an already-extracted `stat -> values` map to the figure.

        `series` must be in the figure's trace order — `epoch_stat_series`
        and `weight_stat_series` both return it that way.
        """
        if not self._has_data:
            if not epochs:
                return
            self._has_data = True
            fig = make_epoch_stats_figure(self._kind, self._title)
            for trace, values in zip(
                fig.data, series.values(), strict=True
            ):
                trace.x = epochs
                trace.y = values
            fig.layout.xaxis.dtick = epoch_axis_dtick(epochs)
            self.element.update_figure(_figure_payload(fig))
            return
        update: dict[str, object] = {
            "x": [epochs for _ in series],
            "y": list(series.values()),
        }
        layout: dict[str, object] = {
            "xaxis.dtick": epoch_axis_dtick(epochs),
            "xaxis.tick0": 0,
        }
        _plotly_restyle(self.element, update, list(range(len(series))), layout)


class _MetricPlot:
    """One custom-metric figure that restyles in place between structure changes.

    The trace set is the metric's series names, which can grow mid-run (a
    dict-returning metric may add keys); a change rebuilds the figure through
    the element — also how the first data lands, exactly like
    `_EpochStatsPlot`'s initial delivery. In between, refreshes are in-place
    `Plotly.update`s so legend selections and zoom survive.
    """

    def __init__(self, metric: str) -> None:
        self._metric = metric
        self._traces: tuple[str, ...] = ()
        self.element = ui.plotly(
            _figure_payload(make_metric_figure(metric, {}))
        ).classes("w-full")

    def update(self, series_map: dict[str, MetricSeries]) -> None:
        names = tuple(series_map)
        if not names:
            return
        if names != self._traces:
            self._traces = names
            self.element.update_figure(
                _figure_payload(make_metric_figure(self._metric, series_map))
            )
            return
        xs: list[list[float]] = []
        ys: list[list[float | None]] = []
        custom: list[list[list[object]]] = []
        for series in series_map.values():
            x, y, c = metric_trace_data(series)
            xs.append(x)
            ys.append(y)
            custom.append(c)
        _plotly_restyle(
            self.element,
            {"x": xs, "y": ys, "customdata": custom},
            list(range(len(names))),
            {
                "xaxis.dtick": epoch_axis_dtick(metric_epochs(series_map)),
                "xaxis.tick0": 0,
            },
        )


class _WatchLayerPanel:
    """One card per watched layer — histograms or extreme-patch grids.

    Both views are built up front; `update` shows the one matching the
    header dropdown and only refreshes that view's content (hidden plotly
    figures and patch grids are left untouched until switched back). Patch
    grids re-render only when their cheap signature — toggles plus the
    stored extreme values — actually changes, so idle 2s refreshes don't
    re-encode images.
    """

    def __init__(
        self,
        *,
        name: str,
        session: Session,
        on_unwatched: Callable[[], None],
        state: _WatchPageState,
        hover_registry: dict[int, _HistPlot],
        input_mean: tuple[float, ...] | None,
        input_std: tuple[float, ...] | None,
    ) -> None:
        self.name = name
        self._session = session
        self._state = state
        self._input_mean = input_mean
        self._input_std = input_std
        self._grid_sig: tuple[object, ...] | None = None

        def unwatch() -> None:
            # Unwatching only exists in the coupled `watched` scope — in the
            # other scopes the watched set doesn't drive collection, so the
            # button is not built; this guard covers a stale card after a
            # scope switch.
            if session.stats_scope is not StatsScope.WATCHED:
                ui.notify(
                    "Layers are only unwatched while stats are collected "
                    "for watched layers",
                    type="warning",
                )
                return
            if _refuse_unwatch_while_recording(session):
                return
            session.unwatch(name)
            on_unwatched()

        with ui.card().classes("w-full p-4 gap-2"):
            with ui.row().classes("w-full items-center gap-2 no-wrap"):
                ui.label(name).classes("font-mono text-base font-bold grow")
                if session.stats_scope is StatsScope.WATCHED:
                    ui.button(
                        icon="visibility_off",
                        color="amber-600",
                        on_click=unwatch,
                    ).props("dense size=sm flat round").tooltip(
                        "Stop watching"
                    )
            # Shown (with both view sections hidden) until the layer has any
            # collected stats, so an unstepped layer is a single clear notice
            # rather than empty plots and "no data yet" tables.
            self._no_data = _notice_banner(
                _no_stats_message(session.locked), icon="bar_chart"
            )
            self._no_data.set_visibility(False)
            self._hist_section = ui.column().classes("w-full gap-3")
            with self._hist_section:
                ui.label("Statistics").classes(
                    "font-mono text-sm text-slate-600"
                )
                self._stats_table = ui.html(
                    _stats_table_html({})
                ).classes("font-mono text-sm")
                ui.label("Activations").classes(
                    "font-mono text-sm text-slate-600"
                )
                self._act = _HistPlot(
                    "activation",
                    "activations",
                    state,
                    session=session,
                    layer=name,
                    hover_registry=hover_registry,
                    input_mean=input_mean,
                    input_std=input_std,
                )
                ui.label("Gradients").classes(
                    "font-mono text-sm text-slate-600"
                )
                self._grad = _HistPlot(
                    "gradient",
                    "gradients",
                    state,
                    session=session,
                    layer=name,
                    hover_registry=hover_registry,
                    input_mean=input_mean,
                    input_std=input_std,
                )
            self._patch_section = ui.column().classes("w-full gap-2")
            with self._patch_section:
                self._grids = ui.html(_NO_PATCHES_HTML).classes("w-full")
            self._epochs_section = ui.column().classes("w-full gap-3")
            with self._epochs_section:
                ui.label("Activations").classes(
                    "font-mono text-sm text-slate-600"
                )
                self._act_epochs = _EpochStatsPlot(
                    "activation", "activation statistics by epoch"
                )
                ui.label("Gradients").classes(
                    "font-mono text-sm text-slate-600"
                )
                self._grad_epochs = _EpochStatsPlot(
                    "gradient", "gradient statistics by epoch"
                )
                # One plot per weight tensor, created lazily once the first
                # per-epoch sample lands (the parameter set is only known
                # from the data); hidden entirely for parameter-less layers
                # (fx intermediates, graph inputs).
                self._weights_label = ui.label("Weights").classes(
                    "font-mono text-sm text-slate-600"
                )
                self._weights_container = ui.column().classes(
                    "w-full gap-3"
                )
                self._weights_label.set_visibility(False)
                self._weight_plots: dict[str, _EpochStatsPlot] = {}
                # One plot per custom metric (`Session.watch_metric`),
                # created lazily like the weight plots — the metric set is
                # only known from the data.
                self._metrics_label = ui.label("Custom metrics").classes(
                    "font-mono text-sm text-slate-600"
                )
                self._metrics_container = ui.column().classes("w-full gap-3")
                self._metrics_label.set_visibility(False)
                self._metric_plots: dict[str, _MetricPlot] = {}
                # Instruments disabled by a raising callback are reported
                # here (they also print once to the console).
                self._metrics_error = ui.label("").classes(
                    "text-xs text-red-600"
                )
                self._metrics_error.set_visibility(False)
            self._hist_section.set_visibility(state.view == _VIEW_HISTOGRAM)
            self._patch_section.set_visibility(state.view == _VIEW_MINMAX)
            self._epochs_section.set_visibility(state.view == _VIEW_GRAPHS)

    def update(
        self,
        snap: WatchSnapshot,
        grids: tuple[tuple[object, ...], str] | None = None,
        metrics: MetricsSnapshot | None = None,
    ) -> None:
        """Refresh the visible view. Runs on the UI event loop.

        `grids` is the output of `prepare_grids` (computed off the event
        loop by the page's refresh); `None` means the grids are unchanged.
        `metrics` is the custom-metric series for the GRAPHS view (`None`
        outside it — the other views never render them).
        """
        view = self._state.view
        # The epoch-stats view draws the phase's whole epoch series; the
        # other two read the latest epoch per phase (`_phase_view`).
        history = (
            snap.phase_history(self.name, self._state.selected_phase)
            if view == _VIEW_GRAPHS
            else []
        )
        per_phase = {} if view == _VIEW_GRAPHS else self._phase_view(snap)
        # No stats accumulated for this layer/phase yet — show the notice and
        # hide every view (their empty plots/grids are pure clutter here).
        has_data = bool(history) if view == _VIEW_GRAPHS else bool(per_phase)
        self._no_data.set_visibility(not has_data)
        self._hist_section.set_visibility(
            has_data and view == _VIEW_HISTOGRAM
        )
        self._patch_section.set_visibility(has_data and view == _VIEW_MINMAX)
        self._epochs_section.set_visibility(has_data and view == _VIEW_GRAPHS)
        if not has_data:
            return
        if view == _VIEW_GRAPHS:
            self._act_epochs.update(history)
            self._grad_epochs.update(history)
            self._update_weight_plots(snap)
            self._update_metric_plots(metrics)
            return
        if view == _VIEW_MINMAX:
            if grids is not None:
                self._grid_sig, html = grids
                self._grids.set_content(html)
            return
        self._stats_table.set_content(
            _stats_table_content(
                per_phase,
                current_batch=self._state.selected_phase
                == _PHASE_CURRENT_BATCH,
            )
        )
        self._act.update(per_phase)
        self._grad.update(per_phase)

    def _update_weight_plots(self, snap: WatchSnapshot) -> None:
        """Refresh the GRAPHS view's per-weight-tensor plots.

        A layer's parameter set is fixed, so plots are keyed by the full
        parameter name and only ever added — creation happens on the first
        refresh that has a sample for that parameter. Titles drop the
        `layer.` prefix (a leaf module's params read as plain "weight" /
        "bias"); functionally-used params keep their own full name.
        """
        per_param = snap.weight_history(self.name)
        self._weights_label.set_visibility(bool(per_param))
        for param, history in per_param.items():
            plot = self._weight_plots.get(param)
            if plot is None:
                short = param.removeprefix(f"{self.name}.")
                with self._weights_container:
                    plot = _EpochStatsPlot(
                        "weight", f"{short} statistics by epoch"
                    )
                self._weight_plots[param] = plot
            plot.update_series(*weight_stat_series(history))
        if per_param and self._state.pending_scroll == "weights":
            # One-shot `?scroll=weights` deep-link (the weights page's
            # jump): fires on the first refresh that has weight data, so
            # the section exists before the viewport moves to it. The
            # scroll re-runs a couple of times because the plots above are
            # near-zero-height divs until plotly.js draws them (each then
            # inflates to its fixed figure height, shifting the target
            # down); the last pass runs after that layout has settled.
            self._state.pending_scroll = ""
            ui.run_javascript(
                f"const el = getHtmlElement({self._weights_label.id});"
                "if (el) {"
                "const go = (b) => el.scrollIntoView("
                "{behavior: b, block: 'start'});"
                "go('auto');"
                "setTimeout(() => go('auto'), 700);"
                "setTimeout(() => go('smooth'), 1600);"
                "}"
            )

    def _update_metric_plots(self, metrics: MetricsSnapshot | None) -> None:
        """Refresh the GRAPHS view's custom-metric plots.

        Plots are created lazily on the first refresh with data for their
        metric (mirroring `_update_weight_plots`); a metric without data in
        the selected phase hides its plot rather than showing another
        phase's curves. Disabled instruments are reported below the plots.
        """
        if metrics is None:
            return
        plots = metrics.plots(self.name, self._state.selected_phase)
        self._metrics_label.set_visibility(bool(plots))
        for metric, series_map in plots.items():
            plot = self._metric_plots.get(metric)
            if plot is None:
                with self._metrics_container:
                    plot = _MetricPlot(metric)
                self._metric_plots[metric] = plot
            plot.update(series_map)
        for metric, plot in self._metric_plots.items():
            plot.element.set_visibility(metric in plots)
        errors = self._session.instrument_errors
        text = "; ".join(
            f"{name}: {error}" for name, error in sorted(errors.items())
        )
        self._metrics_error.set_text(
            f"instrument disabled after an error — {text}" if text else ""
        )
        self._metrics_error.set_visibility(bool(text))

    def _phase_view(self, snap: WatchSnapshot) -> dict[str, LayerStatsSnapshot]:
        """The layer's stats for the current selection.

        For a phase, the latest-epoch stats narrowed to it. For "Current
        batch", `snap` already holds exactly one entry (keyed by the captured
        batch's own phase/epoch), so it's returned unfiltered.
        """
        per_phase = snap.latest_per_phase(self.name)
        if self._state.selected_phase == _PHASE_CURRENT_BATCH:
            return per_phase
        return _filter_phase(per_phase, self._state.selected_phase)

    def prepare_grids(
        self, snap: WatchSnapshot
    ) -> tuple[tuple[object, ...], str] | None:
        """Render this panel's patch grids if their signature changed.

        Pure compute — no UI element access — so the page's refresh can run
        it in a worker thread: blending heatmaps and encoding a card's worth
        of grid images would otherwise block the event loop long enough to
        starve websocket keepalive pings when toggles re-render every card.
        Returns `(signature, html)` for `update` to apply, or `None` when
        the current content is already up to date.
        """
        per_phase = self._phase_view(snap)
        enabled = [self._state.grid_type]
        heatmap = self._state.heat_on
        current_batch = self._state.selected_phase == _PHASE_CURRENT_BATCH
        sig = _patch_grids_signature(
            per_phase, enabled, heatmap, current_batch=current_batch
        )
        if sig == self._grid_sig:
            return None
        html = _patch_grids_html(
            per_phase,
            enabled=enabled,
            heatmap=heatmap,
            current_batch=current_batch,
            mean=self._input_mean,
            std=self._input_std,
        )
        return sig, html


def _filter_phase(
    per_phase: dict[str, LayerStatsSnapshot], phase: str
) -> dict[str, LayerStatsSnapshot]:
    """Narrow a `phase -> stats` mapping to the dropdown-selected phase."""
    return {p: s for p, s in per_phase.items() if p == phase}


# Phase dropdown: the sentinel value (and label) of the "Current batch" entry,
# which shows the last captured batch's stats for any layer instead of a
# phase's running aggregate. The value is a plain-but-unlikely string (not a
# control char) so it can be compared in the option slot's Vue template.
_PHASE_CURRENT_BATCH: str = "::current-batch::"
_PHASE_CURRENT_BATCH_LABEL: str = "Current batch"


def _phase_heading(phase: str, epoch: int, *, current_batch: bool) -> str:
    """The heading of one rendered phase block (histogram subplot titles,
    MIN/MAX grid headers, stats-table corner headers).

    In "Current batch" mode the single entry is keyed by the captured
    batch's own phase/epoch, so the default heading would read exactly like
    the phase's whole-run aggregate and make the Phase dropdown look
    ignored — that mode leads with "current batch" and keeps the batch's
    position as detail.
    """
    if current_batch:
        return f"current batch — {phase} ep {epoch}"
    return f"{phase} (ep {epoch})"


def _stats_table_content(
    per_phase: dict[str, LayerStatsSnapshot], *, current_batch: bool
) -> str:
    """The stats tables' HTML with mode-aware corner headers.

    In "Current batch" mode each corner header is threaded to
    `_stats_table_html` (nansense.ui.histograms) as `_phase_heading`'s
    current-batch form; the default mode keeps the table's own
    "{phase} ep {epoch}" headers. Either way the header tint stays keyed
    on the real phase name (see `_stats_table_html`), so a relabeled
    header still matches the histogram traces below.
    """
    headings = (
        {
            phase: _phase_heading(phase, snap.epoch, current_batch=True)
            for phase, snap in per_phase.items()
        }
        if current_batch
        else None
    )
    return _stats_table_html(per_phase, headings=headings)


def _phase_select_options(view: str, phase_names: list[str]) -> dict[str, str]:
    """The Phase dropdown's value→label map for the current view.

    The schedule's phases always; the "Current batch" entry only where the
    view can render it — the epoch-stats view plots per-epoch aggregates,
    which a single batch doesn't have.
    """
    options = {p: p for p in phase_names}
    if view != _VIEW_GRAPHS:
        options[_PHASE_CURRENT_BATCH] = _PHASE_CURRENT_BATCH_LABEL
    return options


def _reconcile_selected_phase(
    selected: str, view: str, phase_names: list[str]
) -> str:
    """A valid Phase selection for the current view.

    The epoch-stats view swaps "Current batch" (or any stale selection) for
    the first schedule phase, its epoch-aggregating counterpart ("" while
    no phase has been declared or observed yet). The other views fall back
    to "Current batch", which always has something to show.
    """
    if view == _VIEW_GRAPHS:
        if selected in phase_names:
            return selected
        return phase_names[0] if phase_names else ""
    if selected == _PHASE_CURRENT_BATCH or selected in phase_names:
        return selected
    return _PHASE_CURRENT_BATCH


def _initial_phase(session: Session, layer: str) -> str:
    """The Phase selection the page opens on.

    The phase training is currently in — the live batch position's, falling
    back to the published snapshot's — when the running aggregates already
    hold stats for it. `layer` (the `?layer=` deep link, "" when the URL
    named none) scopes that check: a Stats link on an unwatched layer must
    land on "Current batch", the only selection whose Layer dropdown offers
    that layer, rather than bounce to a phase that would swap the layer out.
    Before any batch, or while the phase has nothing collected, "Current
    batch" — it always has something to show.
    """
    position = session.live_position
    if position is None:
        snapshot = session.snapshot
        position = snapshot.position if snapshot is not None else None
    if position is not None and position.phase in session.stats_phases(
        layer or None
    ):
        return position.phase
    return _PHASE_CURRENT_BATCH


# Layer dropdown: the sentinel value of the "all watched layers" entry (a NUL
# prefix keeps it distinct from any real layer name) and its display label.
_LAYER_ALL: str = "\x00all"
_ALL_LAYERS_LABEL: str = "All watched layers"
# Rendering every watched layer's cards at once is what makes the page slow,
# so the "all" entry is only offered while fewer than this many layers are
# watched; at or above it a single layer must be picked.
_ALL_LAYERS_MAX: int = 10


def _watched_in_order(
    layer_names: list[str], stats_layers: frozenset[str]
) -> list[str]:
    """The stats-carrying layers in the page's stable graph order.

    `stats_layers` is an unordered set (`Session.stats_layers` — the watched
    set in the `watched` scope, every layer in `all`, the frozen buckets in
    `none`), so order comes from `layer_names` (the architecture order the
    cards have always rendered in).
    """
    return [n for n in layer_names if n in stats_layers]


def _selectable_layers(
    selected_phase: str, layer_names: list[str], stats_layers: frozenset[str]
) -> list[str]:
    """The layers the Layer dropdown offers for the current phase selection.

    "Current batch" stats come from the snapshot, which covers every layer, so
    any layer is selectable there; a real phase only has running stats for
    the layers in `Session.stats_layers`. Either way the order is the stable
    graph order.
    """
    if selected_phase == _PHASE_CURRENT_BATCH:
        return list(layer_names)
    return _watched_in_order(layer_names, stats_layers)


def _all_layers_available(watched_count: int) -> bool:
    """Whether the "all watched layers" entry is offered for this count.

    Gated below `_ALL_LAYERS_MAX` because rendering every card at once is the
    slow path the layer dropdown exists to avoid.
    """
    return 0 < watched_count < _ALL_LAYERS_MAX


def _layer_select_options(ordered: list[str]) -> dict[str, str]:
    """The layer dropdown's value→label map for the watched layers.

    Each watched layer maps to itself; the "all" entry is prepended only
    while few enough layers are watched (see `_all_layers_available`).
    """
    options: dict[str, str] = {}
    if _all_layers_available(len(ordered)):
        options[_LAYER_ALL] = _ALL_LAYERS_LABEL
    for name in ordered:
        options[name] = name
    return options


def _reconcile_selected_layer(selected: str, ordered: list[str]) -> str:
    """A valid dropdown selection given the watched layers.

    Keeps `selected` when it's still offered ("all" while few enough layers
    are watched, or a still-watched layer); otherwise falls back to the first
    watched layer — never "all", which the spec reserves for an explicit pick
    so the default shows a single fast card. Returns "" when nothing is
    watched.
    """
    if selected == _LAYER_ALL and _all_layers_available(len(ordered)):
        return selected
    if selected in ordered:
        return selected
    return ordered[0] if ordered else ""


def _visible_layers(selected: str, ordered: list[str]) -> list[str]:
    """The watched layers whose cards should be rendered for `selected`.

    "all" (while available) → every watched layer; a layer name → just that
    layer; anything else (stale or empty) → the first watched layer, if any.
    """
    if selected == _LAYER_ALL and _all_layers_available(len(ordered)):
        return ordered
    if selected in ordered:
        return [selected]
    return ordered[:1]


def _deep_dream_href(
    selected_phase: str,
    layer_names: list[str],
    stats_layers: frozenset[str],
    selected_layer: str,
) -> str:
    """Deep-link to the currently shown layer's Deep Dream experiment.

    With "all" selected (or nothing watched) falls back to the first visible
    layer so the link always lands somewhere sensible.
    """
    ordered = _selectable_layers(selected_phase, layer_names, stats_layers)
    visible = _visible_layers(selected_layer, ordered)
    target = visible[0] if visible else selected_layer
    return f"/experiment?layer={quote(target)}"


_PATCH_TYPE_LABELS: dict[PatchType, str] = {
    "max_pixel": "Max pixel",
    "min_pixel": "Min pixel",
    "max_average": "Max average",
    "min_average": "Min average",
}

# The MIN/MAX radio entries gated behind the average-patches Performance
# setting (`WatchPerformance.average_patches`, off by default).
_AVERAGE_PATCH_TYPES: frozenset[PatchType] = frozenset(
    {"max_average", "min_average"}
)


def _grid_type_options(average_patches: bool) -> dict[PatchType, str]:
    """The MIN/MAX radio's value→label map for the Performance setting.

    The average-extreme entries are offered only while their collection is
    on: a setting change flushes every aggregate bucket
    (`WatchAccumulator.configure`) and the "Current batch" stats follow the
    live setting too, so while it's off those grids are nowhere to be had.
    """
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

_NO_PATCHES_HTML: str = (
    '<div class="text-xs text-slate-400 italic py-1">no patches gathered '
    "yet — patches need an image-like (4D) model input</div>"
)

# Patches exist for this layer, just not the selected grid type — the
# average-extreme galleries are a Performance setting, off by default.
_TYPE_NOT_COLLECTED_HTML: str = (
    '<div class="text-xs text-slate-400 italic py-1">this grid type was '
    "not collected — the average-extreme galleries are off in the "
    "Performance settings</div>"
)

def _no_stats_message(locked: bool) -> str:
    """The layer-card notice shown until any stats exist for the phase.

    Unlocked, it stresses that only batches stepped after the layer is
    watched feed the running aggregate (it grows rather than overwriting
    with the last batch). A locked session (the shared hosted demo) can't
    step at all, and its per-epoch stats usually do exist and are merely
    still in flight — so that variant must not claim nothing was collected
    or advise stepping, and points at "Current batch" (the one phase that
    works for any layer) as the fallback.
    """
    if locked:
        return (
            "Stats for this layer haven't loaded yet — they can take a "
            "moment to arrive on the shared demo. If nothing appears, the "
            "Current batch phase works for any layer."
        )
    return (
        "No stats collected for this layer yet — step at least one batch to "
        "start collecting. Each batch you step after watching the layer adds "
        "to the running statistics."
    )


def _patch_grids_signature(
    per_phase: dict[str, LayerStatsSnapshot],
    enabled: list[PatchType],
    heatmap: bool,
    *,
    current_batch: bool = False,
) -> tuple[object, ...]:
    """Cheap change-detection key for a panel's patch grids.

    The stored extreme values identify the buffer contents: any accepted
    candidate changes its channel's value row, so unchanged values ⇒
    unchanged patches (within one epoch bucket, which the phase/epoch part
    of the key pins down). `current_batch` is part of the key because the
    block headings render differently in that mode: a Phase-dropdown flip
    between a phase and "Current batch" can leave the phase key and patch
    bytes identical, which would otherwise skip the re-render and keep a
    stale heading.
    """
    parts: list[object] = [tuple(enabled), heatmap, current_batch]
    for phase, layer_snap in per_phase.items():
        patches = layer_snap.patches
        if patches is None:
            parts.append((phase, layer_snap.epoch, None))
            continue
        for ptype in enabled:
            tp = patches.by_type.get(ptype)
            values = tp.values.numpy().tobytes() if tp is not None else None
            parts.append((phase, layer_snap.epoch, ptype, values))
    return tuple(parts)


def _patch_grids_html(
    per_phase: dict[str, LayerStatsSnapshot],
    *,
    enabled: list[PatchType],
    heatmap: bool,
    current_batch: bool = False,
    mean: tuple[float, ...] | None,
    std: tuple[float, ...] | None,
) -> str:
    """The MIN/MAX view body for one layer: per-phase blocks of grids.

    `current_batch` marks the block headings as "Current batch" content
    (see `_phase_heading`) — the snapshot entry itself is keyed and tinted
    by the captured batch's own phase either way.
    """
    blocks: list[str] = []
    for i, (phase, layer_snap) in enumerate(per_phase.items()):
        patches = layer_snap.patches
        if patches is None:
            continue
        rows: list[str] = []
        for ptype in enabled:
            tp = patches.by_type.get(ptype)
            if tp is None:
                continue
            grid = render_patch_grid(tp, mean=mean, std=std, heatmap=heatmap)
            if grid is None:
                continue
            rows.append(_patch_grid_row_html(_PATCH_TYPE_LABELS[ptype], grid))
        if not rows:
            continue
        color = phase_color(phase, i)
        heading = _phase_heading(
            phase, layer_snap.epoch, current_batch=current_batch
        )
        blocks.append(
            '<div class="flex flex-col gap-2 w-full">'
            f'<div class="font-mono text-xs font-bold" style="color:{color}">'
            f"{heading}</div>" + "".join(rows) + "</div>"
        )
    if not blocks:
        collected = any(
            snap.patches is not None and snap.patches.by_type
            for snap in per_phase.values()
        )
        return _TYPE_NOT_COLLECTED_HTML if collected else _NO_PATCHES_HTML
    return '<div class="flex flex-col gap-4 w-full">' + "".join(blocks) + "</div>"


def _patch_column_html(
    column: PatchColumn, mime: str, label: str, *, tour_anchor: bool = False
) -> str:
    """One channel column of a patch grid: a `CHANNEL n` header over its cells.

    Mirrors the activation strips' table layout — a slate header bar
    (`_column_header_bar`) over the channel's top-N sample cells, each a
    separate `cell_size` square stacked with a `PATCH_CELL_GAP` gutter so the
    grid reads as discrete cells rather than one merged column. `tour_anchor`
    (the grid's first column) tags the div as the tour's MIN/MAX arrow target
    (`tour.stats_tour_steps`).
    """
    size = column.cell_size
    header = _column_header_bar(column.label, size)
    anchor = ' data-tour="patch-column"' if tour_anchor else ""
    cells = "".join(
        f'<img src="{_b64_img_src(cell, mime=mime)}" '
        f"style=\"width:{size}px; height:{size}px; image-rendering:pixelated; "
        f'display:block; max-width:none;" title="{html.escape(label)} — '
        f'{html.escape(column.label)}, sample {i} (best first)" />'
        for i, cell in enumerate(column.cells)
    )
    return (
        f'<div{anchor} style="display:flex; flex-direction:column; flex:none; '
        f'gap:{PATCH_CELL_GAP}px; width:{size}px;">{header}{cells}</div>'
    )


def _patch_grid_row_html(label: str, grid: PatchGridRender) -> str:
    """One labeled grid: `CHANNEL n` columns across, `SAMPLE n` rows down.

    A `SAMPLE n` row-label column (vertical bars, `_row_label_bar_html`) and —
    with the heatmap enabled — the crisp display-resolution colorbar (the
    overlay's `±vmax` scale) sit fixed to the left of the channel columns, which
    scroll horizontally on their own. The row labels and legend each lead with a
    header-height spacer so they line up below the columns' `CHANNEL n` headers,
    and share the cells' `PATCH_CELL_GAP` vertical rhythm. `max-width:none` opts
    the images out of the preflight `max-width:100%` so wide grids scroll
    horizontally instead of being squashed.
    """
    cell = grid.columns[0].cell_size
    n_samples = len(grid.columns[0].cells)
    sample_labels = "".join(
        _row_label_bar_html(f"SAMPLE {i}", height=cell) for i in range(n_samples)
    )
    sample_col = (
        f'<div style="display:flex; flex-direction:column; flex:none; '
        f'gap:{PATCH_CELL_GAP}px;"><div style="height:{LABEL_HEIGHT}px;"></div>'
        f"{sample_labels}</div>"
    )
    legend = (
        '<div style="display:flex; flex-direction:column; flex:none; '
        f'gap:{PATCH_CELL_GAP}px;"><div style="height:{LABEL_HEIGHT}px;"></div>'
        f'<img src="{_b64_img_src(grid.heat_legend)}" '
        'style="display:block; max-width:none;" /></div>'
        if grid.heat_legend is not None
        else ""
    )
    columns = "".join(
        _patch_column_html(column, grid.mime, label, tour_anchor=i == 0)
        for i, column in enumerate(grid.columns)
    )
    return (
        '<div class="flex flex-col gap-0.5 w-full">'
        '<div class="text-base font-bold uppercase tracking-widest '
        f'text-slate-800 font-mono">{label}</div>'
        # `width:100%; max-width:100%` pins this row to the card width: as a
        # nested flex row it otherwise sizes to its content, so the scroll
        # child below never gets a bounded width to scroll within.
        '<div style="display:flex; gap:2px; align-items:flex-start; '
        'width:100%; max-width:100%; min-width:0;">'
        f"{sample_col}{legend}"
        # `min-width:0; flex:1 1 0` lets this flex child shrink below the
        # channel columns' intrinsic width so `overflow-x-auto` yields a
        # scrollbar instead of the columns spilling out of the layer card.
        '<div class="overflow-x-auto" '
        'style="display:flex; gap:2px; min-width:0; flex:1 1 0;">'
        f"{columns}"
        "</div></div></div>"
    )
