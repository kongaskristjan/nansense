"""The `/stats` page: per-layer histograms and extreme-patch grids."""

from __future__ import annotations

import asyncio
import html
import json
from collections.abc import Awaitable, Callable
from dataclasses import replace
from urllib.parse import quote

import plotly.graph_objects as go
from nicegui import ui
from nicegui.events import GenericEventArguments, ValueChangeEventArguments

from nansense import debugger
from nansense.contracts.recording import HistogramView, PatchView, patch_types
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
    _notice_banner,
    _page_scaffold,
    _refuse_unwatch_while_recording,
    _resize_handle,
    _row_label_bar_html,
    _set_controls_enabled,
    _StatusChip,
    _StatusPill,
)
from nansense.ui.components.stats_sidebar import StatsActions, StatsSidebar
from nansense.ui.controllers.stats import (
    _LAYER_ALL,
    _PATCH_TYPE_LABELS,
    _PHASE_CURRENT_BATCH,
    _VIEW_GRAPHS,
    _VIEW_HISTOGRAM,
    _VIEW_MINMAX,
    StatsController,
    _initial_phase,
    _reconcile_selected_phase,
    _selectable_layers,
    _tour_restore_view,
    _visible_layers,
    _watched_in_order,
    _WatchPageState,
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
    _linear_bar_x,
    _make_histogram_figure,
    _min_positive_height,
    _phase_hists,
    _phases_with_data,
    _retained_y_range,
    _stats_table_html,
    _x_range_linear_to_log,
    _x_range_log_to_linear,
    kind_stats,
    overflow_marks,
    phase_color,
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
from nansense.ui.share import _add_share_button
from nansense.ui.top_bar import (
    _add_error_banner,
    _add_repo_logo,
    _add_settings_button,
    _add_step_controls,
    _add_tour_button,
    _back_button,
    _back_href,
    _build_step_until_custom_dialog,
    _top_bar_row,
)
from nansense.ui.tour import STATS_HOWTO, add_tour, stats_tour_steps
from nansense.watch import (
    N_BINS,
    LayerStatsSnapshot,
    WatchSnapshot,
    narrow_to_channel,
)

# The View dropdown's three entries; also the values `_WatchPageState.view`
# takes. HISTOGRAM is the default.


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
    starts collection. Only a `watched` collecting scope needs the watch
    (paused or not — the watched set is what a resume collects); under `all`
    every layer already collects. Unknown layer names are refused by
    `Session.watch` itself; an already watched layer is left alone so
    reloading the link stays a no-op.
    """
    if (
        watch.strip()
        and session.collecting_scope is StatsScope.WATCHED
        and layer not in session.watched_layers
    ):
        session.watch(layer)


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
    _StatsPage(
        session,
        layer_names,
        selected_layer,
        view=view,
        scroll=scroll,
        watch=watch,
        input_mean=input_mean,
        input_std=input_std,
    )


class _StatsPage:
    """NiceGUI adapter: builds components and synchronizes them with its controller."""

    def __init__(
        self,
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
        self.input_mean = input_mean
        self.input_std = input_std
        self.layer_names = layer_names
        self.session = session
        _page_scaffold("Stats")
        _install_panel_resize()
        add_tour("stats", stats_tour_steps(), locked=self.session.locked)
        _apply_watch_param(self.session, selected_layer, watch)
        self.layer_panels: dict[str, _WatchLayerPanel] = {}
        self.hover_registry: dict[int, _HistPlot] = {}
        self.body_container: ui.column
        self.phase_names = self.session.schedule.phase_order
        ui.on(_HOVER_EVENT, self._dispatch_hover)
        requested_view = view.strip().lower()
        initial_view = {"minmax": _VIEW_MINMAX, "graphs": _VIEW_GRAPHS}.get(
            requested_view, _VIEW_HISTOGRAM
        )
        self.state = _WatchPageState(
            selected_phase=_initial_phase(self.session, selected_layer),
            view=initial_view,
            selected_layer=selected_layer,
            pending_scroll=scroll.strip().lower(),
            show_bands=_should_show_bands(self.session.debug_error),
        )
        self.state.selected_phase = _reconcile_selected_phase(
            self.state.selected_phase, self.state.view, list(self.phase_names)
        )
        self.controller = StatsController(self.session, self.layer_names, self.state)
        self.step_until_custom = _build_step_until_custom_dialog(self.session)
        self._build_layout()
        ui.on("nansense_tour_set_view", self._tour_set_view)
        ui.on("nansense_tour_start", self._tour_start)
        ui.on("nansense_tour_end", self._tour_end)
        self.rebuild_cards()
        self.controller.gate.should_refresh(self.session)
        ui.timer(0.0, self.refresh, once=True)
        ui.timer(0.2, self.tick)

    async def _dispatch_hover(self, e: GenericEventArguments) -> None:
        view = self.hover_registry.get(int(e.args.get("id", -1)))
        if view is not None:
            await view._on_hover(e)

    async def set_axis_log_x(self, value: bool) -> None:
        self.state.axis_log_x = value
        await self.refresh()

    async def set_axis_log_y(self, value: bool) -> None:
        self.state.axis_log_y = value
        await self.refresh()

    async def set_retain_axes(self, value: bool) -> None:
        self.state.retain_axes = value
        await self.refresh()

    async def set_show_bands(self, value: bool) -> None:
        self.state.show_bands = value
        await self.refresh()

    async def set_mode(self, value: object) -> None:
        self.controller.select_view(str(value))
        self.sync_phase_select()
        self.sidebar.hist_controls.set_visibility(self.state.view == _VIEW_HISTOGRAM)
        self.sidebar.minmax_controls.set_visibility(self.state.view == _VIEW_MINMAX)
        self.sidebar.compare_deep_dream.set_visibility(self.state.view == _VIEW_MINMAX)
        await self.refresh()

    async def set_phase(self, value: object) -> None:
        new = str(value)
        if new == self.state.selected_phase:
            return
        self.state.selected_phase = new
        await self.refresh()

    async def set_layer(self, value: object) -> None:
        new = str(value) if value is not None else ""
        if new == self.state.selected_layer:
            return
        self.state.selected_layer = new
        await self.refresh()

    async def set_grid(self, ptype: PatchType) -> None:
        if ptype == self.state.grid_type:
            return
        self.state.grid_type = ptype
        await self.refresh()

    async def set_heat(self, value: bool) -> None:
        self.state.heat_on = value
        await self.refresh()

    def sync_compare_href(self) -> None:
        href = _deep_dream_href(
            self.state.selected_phase,
            self.layer_names,
            self.session.stats_layers,
            self.state.selected_layer,
        )
        self.sidebar.compare_deep_dream.props(f'href="{href}"')

    def sync_back_href(self) -> None:
        if not self.session.locked:
            return
        target = (
            "" if self.state.selected_layer == _LAYER_ALL else self.state.selected_layer
        )
        self.back_button.props(f'href="{_back_href(target)}"')

    def record_view(self) -> RecordedView | None:
        if self.state.selected_phase == _PHASE_CURRENT_BATCH:
            return None
        if self.state.view == _VIEW_GRAPHS:
            return None
        ordered = _watched_in_order(self.layer_names, self.session.stats_layers)
        watched = _visible_layers(self.state.selected_layer, ordered)
        if not watched:
            return None
        phase = self.state.selected_phase
        if self.state.view == _VIEW_MINMAX:
            return RecordedView(
                key="watch_minmax",
                label=f"Watch · MIN/MAX grids ({phase})",
                config=PatchView(
                    layers=tuple(watched),
                    phase=phase,
                    grids=patch_types((self.state.grid_type,)),
                    heatmap=self.state.heat_on,
                    input_mean=self.input_mean,
                    input_std=self.input_std,
                ),
            )
        return RecordedView(
            key="watch_histogram",
            label=f"Watch · histograms ({phase})",
            config=HistogramView(
                layers=tuple(watched),
                phase=phase,
                log_x=self.state.axis_log_x,
                log_y=self.state.axis_log_y,
            ),
        )

    def _tour_set_view(self, e: GenericEventArguments) -> None:
        """Switch to the view a tour step describes (`TourStep.ensure_view`).

        Emitted by the tour driver whenever a view-bound step is shown; the
        write goes through the View dropdown so the widget and `set_mode`
        stay in sync, and a no-op (already on that view) is skipped — the
        driver re-emits on every re-show.
        """
        view = str(e.args)
        if view != self.state.view and view in (
            _VIEW_HISTOGRAM,
            _VIEW_MINMAX,
            _VIEW_GRAPHS,
        ):
            self.state.tour_view_write = True
            self.sidebar.view_select.set_value(view)

    def _tour_start(self, _: GenericEventArguments) -> None:
        """Snapshot the view a fresh tour run starts from.

        The run's view-bound steps switch the page around on the visitor's
        behalf (`_tour_set_view`); dismissing the tour puts this view back
        (`_tour_end`) unless the visitor picked one themselves meanwhile.
        """
        self.state.tour_saved_view = self.state.view
        self.state.tour_user_set_view = False

    def _tour_end(self, _: GenericEventArguments) -> None:
        """Restore the pre-tour view when the ended run switched it away."""
        restore = _tour_restore_view(
            self.state.tour_saved_view, self.state.tour_user_set_view, self.state.view
        )
        self.state.tour_saved_view = None
        if restore is not None:
            self.state.tour_view_write = True
            self.sidebar.view_select.set_value(restore)

    def sync_phase_select(self) -> None:
        """Refresh the Phase dropdown's options/value for the current view.

        The epoch-stats view has no "Current batch" entry (a single batch
        has no epoch series), and such a selection is swapped for the first
        schedule phase — the epoch-aggregating counterpart. Options are
        re-read from the schedule each pass so lazily discovered phases
        appear without reopening the page. Pushes to the widget only on an
        actual change (a no-op write would re-enter `set_phase`).
        """
        options = self.controller.phase_options()
        if self.sidebar.phase_select.options != options:
            self.sidebar.phase_select.set_options(
                options, value=self.state.selected_phase
            )
        elif self.sidebar.phase_select.value != self.state.selected_phase:
            self.sidebar.phase_select.set_value(self.state.selected_phase)

    def sync_grid_type_select(self) -> None:
        """Refresh the MIN/MAX radio's options for the Performance setting.

        The average-extreme galleries are collected only while their
        Performance setting is on (`WatchPerformance.average_patches`), so
        the radio offers those entries only then — a dead option's body
        could only say "not collected". An average selection whose entry
        just disappeared falls back to the "Max pixel" default. Pushes to
        the widget only on an actual change (a no-op write would re-enter
        `set_grid`).
        """
        options = self.controller.grid_options()
        if self.sidebar.grid_radio.options != options:
            self.sidebar.grid_radio.set_options(options, value=self.state.grid_type)
        elif self.sidebar.grid_radio.value != self.state.grid_type:
            self.sidebar.grid_radio.set_value(self.state.grid_type)

    def sync_layer_select(self) -> None:
        """Refresh the layer dropdown's options/value from the stats layers.

        Reconciles the selection (drops "all" once too many layers carry
        stats, replaces a layer that no longer does), pushes the
        options/value to the widget only when they actually changed (a no-op
        write would re-enter `set_layer`), and disables the dropdown when no
        layer carries stats. Cheap enough to call every refresh tick.
        """
        options = self.controller.layer_options()
        if self.sidebar.layer_select.options != options:
            self.sidebar.layer_select.set_options(
                options, value=self.state.selected_layer
            )
        elif self.sidebar.layer_select.value != self.state.selected_layer:
            self.sidebar.layer_select.set_value(self.state.selected_layer)
        _set_controls_enabled([self.sidebar.layer_select], bool(options))

    def rebuild_cards(self) -> None:
        self.sync_layer_select()
        self.layer_panels.clear()
        self.hover_registry.clear()
        self.body_container.clear()
        ordered = _selectable_layers(
            self.state.selected_phase, self.layer_names, self.session.stats_layers
        )
        with self.body_container:
            if not ordered:
                with ui.column().classes("items-center gap-2 py-12 w-full"):
                    ui.icon("visibility_off", size="lg").classes("text-slate-400")
                    ui.label("No layers watched.").classes("text-slate-600")
                    ui.label(
                        "Go back and click a node in the architecture diagram "
                        "to start watching a layer."
                    ).classes("text-slate-500 text-sm")
                return
            for name in _visible_layers(self.state.selected_layer, ordered):
                self.layer_panels[name] = _WatchLayerPanel(
                    name=name,
                    session=self.session,
                    on_unwatched=self.rebuild_cards,
                    state=self.state,
                    hover_registry=self.hover_registry,
                    input_mean=self.input_mean,
                    input_std=self.input_std,
                )

    async def refresh_pass(self) -> None:
        stats_layers = self.session.stats_layers
        n = len(stats_layers)
        self.sidebar.count_label.text = f"{n} layer{('' if n == 1 else 's')}"
        self.sync_phase_select()
        self.sync_layer_select()
        self.sync_grid_type_select()
        self.sync_compare_href()
        self.sync_back_href()
        ordered = _selectable_layers(
            self.state.selected_phase, self.layer_names, stats_layers
        )
        desired = _visible_layers(self.state.selected_layer, ordered)
        if list(self.layer_panels) != desired:
            self.rebuild_cards()
        panels = dict(self.layer_panels)
        minmax = self.state.view == _VIEW_MINMAX
        graphs = self.state.view == _VIEW_GRAPHS
        current_batch = self.state.selected_phase == _PHASE_CURRENT_BATCH

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
            metrics: MetricsSnapshot | None = None
            if current_batch:
                snap = self.session.current_batch_stats(
                    layers=list(panels), include_patches=minmax
                )
            else:
                snap = self.session.watch_snapshot(include_patches=minmax)
                if graphs:
                    metrics = self.session.watch_metrics_snapshot(layers=list(panels))
            grids: dict[str, tuple[tuple[object, ...], str] | None] = {}
            if minmax:
                for name, panel in panels.items():
                    grids[name] = panel.prepare_grids(snap)
            return (snap, grids, metrics)

        snap, grids, metrics = await asyncio.to_thread(compute)
        for name, panel in panels.items():
            # A control change may have rebuilt cards while the worker rendered.
            if self.layer_panels.get(name) is panel:
                panel.update(snap, grids.get(name), metrics)

    async def refresh(self) -> None:
        await self.controller.refresh(self.refresh_pass)

    def sync_frozen(self) -> None:
        hist = self.session.recording.is_recording("watch_histogram")
        minmax = self.session.recording.is_recording("watch_minmax")
        if hist != self.state.frozen_hist or minmax != self.state.frozen_minmax:
            self.state.frozen_hist = hist
            self.state.frozen_minmax = minmax
            _set_controls_enabled(
                [self.sidebar.phase_select, self.sidebar.layer_select],
                not (hist or minmax),
            )
            _set_controls_enabled(self.sidebar.hist_boxes, not hist)
            _set_controls_enabled(self.sidebar.minmax_boxes, not minmax)

    async def tick(self) -> None:
        self.sync_frozen()
        if self.controller.gate.should_refresh(self.session):
            await self.refresh()

    def _build_header(self) -> None:
        with _top_bar_row():
            self.back_button = _back_button(
                self.state.selected_layer if self.session.locked else None
            )
            _add_step_controls(self.session, self.step_until_custom)
            _add_settings_button(self.session, self.record_view).classes("ml-auto")
            ui.button(
                icon="refresh",
                on_click=lambda: _refresh_now(self.session, self.refresh),
                color="slate-500",
            ).props("dense size=md flat").tooltip(
                "Refresh now, and from the next training batch"
            )
            _add_tour_button()
            _add_share_button(self.session)
            _add_repo_logo()

    def _build_content(self) -> None:
        with ui.row().classes("w-full grow min-h-0 no-wrap gap-0"):
            self.sidebar = StatsSidebar(
                self.state,
                list(self.phase_names),
                self.session.watch_performance.average_patches,
                StatsActions(
                    self.set_mode,
                    self.set_phase,
                    self.set_layer,
                    self.set_axis_log_x,
                    self.set_axis_log_y,
                    self.set_retain_axes,
                    self.set_show_bands,
                    self.set_grid,
                    self.set_heat,
                ),
            )
            self.sync_compare_href()
            _resize_handle("watch-controls", "left")
            self.body_container = ui.column().classes(
                "grow min-w-0 h-full overflow-auto p-4 gap-3 bg-slate-200"
            )

    def _build_layout(self) -> None:
        with ui.column().classes("w-full h-screen no-wrap gap-0"):
            self._build_header()
            _add_error_banner(self.session)
            self._build_content()


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
                "One channel's histogram — hover a bar to sample inputs "
                "from that range"
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
            # Unwatching only exists while the watched set drives collection
            # (`watched`, paused or not) — under `all` it doesn't, so the
            # button is not built; this guard covers a stale card after a
            # scope switch.
            if session.collecting_scope is not StatsScope.WATCHED:
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
                if session.collecting_scope is StatsScope.WATCHED:
                    ui.button(
                        icon="visibility_off",
                        color="amber-600",
                        on_click=unwatch,
                    ).props("dense size=sm flat round").tooltip(
                        "Stop watching"
                    )
            # A card with nothing to draw shows one of two things, never
            # empty plots and "no data yet" tables: a spinning pill while
            # the numbers are on their way (the first refresh computes them
            # off the event loop, and a running session keeps adding to
            # them), or the notice below once waiting can't help.
            self._status = _StatusPill(_LOADING_CHIP)
            self._no_data = _notice_banner(
                _no_stats_message(session.locked),
                icon="bar_chart",
                action=None if session.locked else _show_me_how_action(name),
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
            # Every view starts hidden behind the loading pill; the first
            # `update` reveals the one the dropdown selects.
            self._hist_section.set_visibility(False)
            self._patch_section.set_visibility(False)
            self._epochs_section.set_visibility(False)

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
        # No stats accumulated for this layer/phase yet — show the pill or
        # the notice and hide every view (their empty plots/grids are pure
        # clutter here). While training advances, the numbers are on their
        # way, so that wait spins rather than advising anything.
        has_data = bool(history) if view == _VIEW_GRAPHS else bool(per_phase)
        collecting = not has_data and self._session.is_running
        self._status.show(_COLLECTING_CHIP)
        self._status.set_visibility(collecting)
        self._no_data.set_visibility(not has_data and not collecting)
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


# Layer dropdown: the sentinel value of the "all watched layers" entry (a NUL
# prefix keeps it distinct from any real layer name) and its display label.
# Rendering every watched layer's cards at once is what makes the page slow,
# so the "all" entry is only offered while fewer than this many layers are
# watched; at or above it a single layer must be picked.


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


# The MIN/MAX radio entries gated behind the average-patches Performance
# setting (`WatchPerformance.average_patches`, off by default).


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

# The layer-card pills, both of them waits that resolve on their own: the
# first refresh computes a card's numbers off the event loop, and a running
# session keeps feeding them. Spinner (`icon=None`) rather than advice —
# there is nothing for the user to do but watch.
_LOADING_CHIP = _StatusChip("waiting", None, "Loading this layer's statistics…")
_COLLECTING_CHIP = _StatusChip(
    "waiting", None, "Collecting — statistics arrive as training advances."
)


def _no_stats_message(locked: bool) -> str:
    """The layer-card notice shown when waiting can no longer help.

    Only reached with training stopped — while it advances the card spins
    (`_COLLECTING_CHIP`) instead. Unlocked, the notice is a how-to rather
    than a diagnosis: watch the layer, turn collection on (off by default),
    step — and only batches stepped from then on feed the running
    aggregate (it grows rather than overwriting with the last batch); the
    SHOW ME HOW button below it (`_show_me_how_action`) walks through the
    same three things on the main view. A locked session (the shared hosted
    demo) can't step at all and never will collect more, so that variant
    must not advise stepping: what's missing is missing for this phase, and
    it points at "Current batch" (the one phase that works for any layer)
    as the fallback.
    """
    if locked:
        return (
            "No stats for this layer in this phase — nothing further will "
            "arrive on the parked demo. The Current batch phase works for "
            "any layer."
        )
    return (
        "To get stats here, watch this layer, turn on stats collection with "
        "the stats button in the main view's top bar, and step at least one "
        "batch. Every batch from then on adds to the running statistics."
    )


def _show_me_how_action(layer: str) -> tuple[str, str]:
    """The notice's button: the main view, playing the stats how-to tour.

    A `?tour=` deep link (`main_page._requested_extra_tour`) rather than a
    click handler, so the button stays a real anchor; `layer` makes the
    tour's first arrow point at the node of the layer the visitor came from.
    """
    return ("SHOW ME HOW", f"/?layer={quote(layer)}&tour={STATS_HOWTO}")


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
