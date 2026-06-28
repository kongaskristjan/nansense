"""The `/stats` page: per-layer histograms and extreme-patch grids."""

from __future__ import annotations

import asyncio
import html
import json
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from urllib.parse import quote

import plotly.graph_objects as go
from nicegui import ui
from nicegui.elements.mixins.disableable_element import DisableableElement
from nicegui.events import GenericEventArguments, ValueChangeEventArguments

from nansense import debugger
from nansense.patches import PatchType
from nansense.recording import RecordedView
from nansense.session import BatchSnapshot, Session
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
from nansense.ui.histograms import (
    _BIN_VALUE_LABELS,
    _axis_ranges,
    _format_stat,
    _hover_customdata,
    kind_stats,
    _linear_bar_x,
    _make_histogram_figure,
    _min_positive_height,
    _overflow_marks,
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
    _add_step_controls,
    _back_button,
    _build_step_until_custom_dialog,
    _top_bar_row,
)
from nansense.watch import N_BINS, LayerStatsSnapshot, WatchSnapshot


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
    # MIN/MAX view state: which one of the four grids is shown (a radio
    # group defaulting to "Max pixel") and whether the activation heatmap
    # is blended over the patches. HISTOGRAM is the default view.
    view_minmax: bool = False
    grid_type: PatchType = "max_pixel"
    heat_on: bool = False
    # Every card shows one phase at a time, picked by a header dropdown
    # shared by both views; defaults to the schedule's first phase.
    selected_phase: str = ""
    # Which watched layer's cards to render: a layer name, or `_LAYER_ALL`
    # for every watched layer at once. Defaults (via reconciliation) to the
    # first watched layer so the page stays fast with many layers watched.
    selected_layer: str = ""
    # Single-flight refresh flags (see `refresh` in `_build_stats_page`).
    refresh_running: bool = False
    refresh_dirty: bool = False
    # Last frozen flags pushed to the client, so the per-tick sync only
    # sends enable/disable when something actually changed.
    frozen_hist: bool | None = None
    frozen_minmax: bool | None = None
    # The sidebar's watched-layer count label, assigned during page build.
    count_label: ui.label = field(init=False)


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


def _build_stats_page(
    session: Session,
    layer_names: list[str],
    selected_layer: str = "",
    *,
    view: str = "",
    input_mean: tuple[float, ...] | None = None,
    input_std: tuple[float, ...] | None = None,
) -> None:
    """The deep-dive page for watched layers.

    The top bar carries the shared stepping controls, like the main page.
    Left-sidebar dropdowns switch every layer card between two views, pick
    which phase (train / val / …) the cards show, and pick which watched
    layer to render — one named layer (the default, which keeps the page
    fast when many layers are watched) or, while fewer than `_ALL_LAYERS_MAX`
    are watched, every watched layer at once.

    The Phase dropdown's last entry, "Current batch", is special: it shows
    stats computed directly from the last captured `BatchSnapshot` rather than
    the running watch aggregates, so it works for *any* layer whether or not
    it is watched — and the Layer dropdown then offers every layer, not just
    the watched ones. The page opens on "Current batch" by default (it needs no
    watched aggregates, so a freshly opened Stats view always has something to
    show); the user can switch to a recorded phase from the dropdown. The view
    and phase apply in both views, one at a time:

    - HISTOGRAM (the default) — one plotly figure per tensor kind for the
      selected phase's latest epoch, with the "Log x" / "Log y" axis
      checkboxes and a per-histogram "Per channel" switch.
    - MIN/MAX — the extreme-activation patch grids
      (channels across, per-channel top samples down), one per patch
      type, picked one at a time by a radio group (defaulting to "Max
      pixel"), plus a heatmap checkbox
      that blends the stored activation maps over the patches.
    Each control group is only visible while its view is selected. A
    `ui.timer` polls `session.watch_snapshot()` and refreshes the visible
    view in place. Layers can also be unwatched directly from the card
    header here, which drops the corresponding accumulator entry — the
    change is reflected on the main page on next navigation.
    """
    _page_scaffold("Stats")
    _install_panel_resize()

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
    # A `?view=minmax` link (e.g. "Compare with MIN/MAX" on the experiment
    # page) opens straight on the MIN/MAX grids instead of the histograms.
    initial_minmax = view.strip().lower() == "minmax"
    state = _WatchPageState(
        # Default to "Current batch": it reads the last captured snapshot
        # directly, so the page shows data immediately for any layer without
        # waiting for watch aggregates to fill.
        selected_phase=_PHASE_CURRENT_BATCH,
        view_minmax=initial_minmax,
        # Seed the layer picked by the caller (e.g. a `?layer=` link from the
        # main page's watch menu). Reconciliation drops it back to the first
        # watched layer if it isn't currently watched.
        selected_layer=selected_layer,
        # Pre-check the under/overflow band whenever the page opens with an
        # active under/overflow issue, regardless of how it was reached.
        show_bands=_should_show_bands(session.debug_error),
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
        state.view_minmax = value == _VIEW_MINMAX
        hist_controls.set_visibility(not state.view_minmax)
        minmax_controls.set_visibility(state.view_minmax)
        compare_deep_dream.set_visibility(state.view_minmax)
        await refresh()

    async def set_phase(value: object) -> None:
        state.selected_phase = str(value)
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
        state.grid_type = ptype
        await refresh()

    async def set_heat(value: bool) -> None:
        state.heat_on = value
        await refresh()

    def open_deep_dream() -> None:
        # Carry the currently shown layer to its Deep Dream experiment; with
        # "all" selected (or nothing watched) fall back to the first visible
        # layer so the link always lands somewhere sensible.
        ordered = _selectable_layers(
            state.selected_phase, layer_names, session.watched_layers
        )
        visible = _visible_layers(state.selected_layer, ordered)
        target = visible[0] if visible else state.selected_layer
        ui.navigate.to(f"/experiment?layer={quote(target)}")

    step_until_custom = _build_step_until_custom_dialog(session)

    def record_view() -> RecordedView | None:
        # The recorder renders from the running watch accumulators, which the
        # "Current batch" view doesn't use — so that view isn't recordable.
        if state.selected_phase == _PHASE_CURRENT_BATCH:
            return None
        # Record exactly the cards on screen — the selected layer, or every
        # watched layer while "all" is showing.
        ordered = _watched_in_order(layer_names, session.watched_layers)
        watched = _visible_layers(state.selected_layer, ordered)
        if not watched:
            return None
        phase = state.selected_phase
        if state.view_minmax:
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
            _back_button()
            _add_step_controls(session, step_until_custom)
            _add_settings_button(session, record_view).classes("ml-auto")
            ui.button(
                icon="refresh",
                on_click=lambda: refresh(),
                color="slate-500",
            ).props("dense size=md flat").tooltip("Refresh now")
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
                ui.select(
                    [_VIEW_HISTOGRAM, _VIEW_MINMAX],
                    label="View",
                    value=_VIEW_MINMAX if state.view_minmax else _VIEW_HISTOGRAM,
                    on_change=lambda e: set_mode(e.value),
                ).props("dense outlined options-dense").classes(
                    "w-full text-sm"
                ).tooltip("What each layer card shows")
                # Phases, then "Current batch" as the last entry. A scoped
                # `option` slot draws a divider above it (Quasar has no native
                # per-option separator) while keeping default selection via
                # `itemProps`.
                phase_options = {p: p for p in phase_names}
                phase_options[_PHASE_CURRENT_BATCH] = _PHASE_CURRENT_BATCH_LABEL
                phase_select = ui.select(
                    phase_options,
                    label="Phase",
                    value=state.selected_phase,
                    on_change=lambda e: set_phase(e.value),
                ).props("dense outlined options-dense").classes(
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
                ).props("dense outlined options-dense").classes(
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
                    minmax_boxes.append(
                        ui.radio(
                            _PATCH_TYPE_LABELS,
                            value=state.grid_type,
                            on_change=lambda e: set_grid(e.value),
                        ).props("dense").classes("text-sm").tooltip(
                            "Which extreme-activation patch grid to show"
                        )
                    )
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
                hist_controls.set_visibility(not state.view_minmax)
                minmax_controls.set_visibility(state.view_minmax)
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
                        on_click=open_deep_dream,
                    )
                    .props("dense no-caps size=sm")
                    .classes("w-full")
                    .tooltip(
                        "Open this layer's Deep Dream experiment — the inputs "
                        "synthesized to excite the same channels"
                    )
                )
                compare_deep_dream.set_visibility(state.view_minmax)

            _resize_handle("watch-controls", "left")
            body_container = ui.column().classes(
                "grow min-w-0 h-full overflow-auto p-4 gap-3 bg-slate-200"
            )

    def sync_layer_select() -> None:
        """Refresh the layer dropdown's options/value from the watched set.

        Reconciles the selection (drops "all" once too many layers are
        watched, replaces an unwatched layer), pushes the options/value to
        the widget only when they actually changed (a no-op write would
        re-enter `set_layer`), and disables the dropdown when nothing is
        watched. Cheap enough to call every refresh tick.
        """
        ordered = _selectable_layers(
            state.selected_phase, layer_names, session.watched_layers
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
            state.selected_phase, layer_names, session.watched_layers
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
                watched = session.watched_layers
                n = len(watched)
                state.count_label.text = (
                    f"{n} layer{'' if n == 1 else 's'}"
                )
                sync_layer_select()
                ordered = _selectable_layers(
                    state.selected_phase, layer_names, watched
                )
                desired = _visible_layers(state.selected_layer, ordered)
                if list(layer_panels) != desired:
                    rebuild_cards()
                panels = dict(layer_panels)
                minmax = state.view_minmax
                current_batch = state.selected_phase == _PHASE_CURRENT_BATCH

                def compute(
                    panels: dict[str, _WatchLayerPanel] = panels,
                    minmax: bool = minmax,
                    current_batch: bool = current_batch,
                ) -> tuple[
                    WatchSnapshot,
                    dict[str, tuple[tuple[object, ...], str] | None],
                ]:
                    # "Current batch" computes stats from the last snapshot for
                    # exactly the visible layers; a phase reads the running
                    # watch aggregates. Either way the patch GPU→CPU work is
                    # only paid when the MIN/MAX view will consume it.
                    if current_batch:
                        snap = session.current_batch_stats(
                            layers=list(panels), include_patches=minmax
                        )
                    else:
                        snap = session.watch_snapshot(include_patches=minmax)
                    grids: dict[str, tuple[tuple[object, ...], str] | None] = {}
                    if minmax:
                        for name, panel in panels.items():
                            grids[name] = panel.prepare_grids(snap)
                    return snap, grids

                snap, grids = await asyncio.to_thread(compute)
                for name, panel in panels.items():
                    if layer_panels.get(name) is panel:  # not rebuilt meanwhile
                        panel.update(snap, grids.get(name))
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
    ui.timer(0.2, sync_frozen)
    ui.timer(0.0, refresh, once=True)
    ui.timer(2.0, refresh)


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
        self.element = ui.plotly(_figure_payload(fig)).classes("w-full")
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
        out: dict[str, LayerStatsSnapshot] = {}
        for phase, snap in per_phase.items():
            stats = kind_stats(snap, self._kind)
            if stats.channel_hists is None:
                out[phase] = snap
                continue
            channel = min(self._channel, len(stats.channel_hists) - 1)
            narrowed = replace(stats, hist=stats.channel_hists[channel])
            field = "activations" if self._kind == "activation" else "gradients"
            out[phase] = replace(snap, **{field: narrowed})
        return out

    def _trace_names(self, view: dict[str, LayerStatsSnapshot]) -> list[str]:
        suffix = (
            f" — ch {self._channel}"
            if self._per_channel and self._channel_count is not None
            else ""
        )
        phases = _phases_with_data(view, self._kind)
        return [f"{p} (ep {view[p].epoch}){suffix}" for p in phases]

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
            hists = [kind_stats(per_phase[p], self._kind).hist for p in phases]
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
            marks = _overflow_marks(phase_hists, x_values, density, y_top)
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
            if _refuse_unwatch_while_recording(session):
                return
            session.unwatch(name)
            on_unwatched()

        with ui.card().classes("w-full p-4 gap-2"):
            with ui.row().classes("w-full items-center gap-2 no-wrap"):
                ui.label(name).classes("font-mono text-base font-bold grow")
                ui.button(
                    icon="visibility_off",
                    color="amber-600",
                    on_click=unwatch,
                ).props("dense size=sm flat round").tooltip("Stop watching")
            # Shown (with both view sections hidden) until the layer has any
            # collected stats, so an unstepped layer is a single clear notice
            # rather than empty plots and "no data yet" tables.
            self._no_data = _notice_banner(_NO_STATS_MESSAGE, icon="bar_chart")
            self._no_data.set_visibility(False)
            self._hist_section = ui.column().classes("w-full gap-3")
            with self._hist_section:
                ui.label("Activations").classes(
                    "font-mono text-sm text-slate-600"
                )
                self._act_stats = ui.html(
                    _stats_table_html({}, "activation")
                ).classes("font-mono text-sm")
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
                self._grad_stats = ui.html(
                    _stats_table_html({}, "gradient")
                ).classes("font-mono text-sm")
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
            self._hist_section.set_visibility(not state.view_minmax)
            self._patch_section.set_visibility(state.view_minmax)

    def update(
        self,
        snap: WatchSnapshot,
        grids: tuple[tuple[object, ...], str] | None = None,
    ) -> None:
        """Refresh the visible view. Runs on the UI event loop.

        `grids` is the output of `prepare_grids` (computed off the event
        loop by the page's refresh); `None` means the grids are unchanged.
        """
        per_phase = self._phase_view(snap)
        # No stats accumulated for this layer/phase yet — show the notice and
        # hide both views (their empty plots/grids are pure clutter here).
        has_data = bool(per_phase)
        self._no_data.set_visibility(not has_data)
        minmax = self._state.view_minmax
        self._hist_section.set_visibility(has_data and not minmax)
        self._patch_section.set_visibility(has_data and minmax)
        if not has_data:
            return
        if minmax:
            if grids is not None:
                self._grid_sig, html = grids
                self._grids.set_content(html)
            return
        self._act_stats.set_content(_stats_table_html(per_phase, "activation"))
        self._grad_stats.set_content(_stats_table_html(per_phase, "gradient"))
        self._act.update(per_phase)
        self._grad.update(per_phase)

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
        sig = _patch_grids_signature(per_phase, enabled, heatmap)
        if sig == self._grid_sig:
            return None
        html = _patch_grids_html(
            per_phase,
            enabled=enabled,
            heatmap=heatmap,
            mean=self._input_mean,
            std=self._input_std,
        )
        return sig, html


def _filter_phase(
    per_phase: dict[str, LayerStatsSnapshot], phase: str
) -> dict[str, LayerStatsSnapshot]:
    """Narrow a `phase -> stats` mapping to the dropdown-selected phase."""
    return {p: s for p, s in per_phase.items() if p == phase}


_VIEW_HISTOGRAM: str = "HISTOGRAM"
_VIEW_MINMAX: str = "MIN/MAX"

# Phase dropdown: the sentinel value (and label) of the "Current batch" entry,
# which shows the last captured batch's stats for any layer instead of a
# phase's running aggregate. The value is a plain-but-unlikely string (not a
# control char) so it can be compared in the option slot's Vue template.
_PHASE_CURRENT_BATCH: str = "::current-batch::"
_PHASE_CURRENT_BATCH_LABEL: str = "Current batch"

# Layer dropdown: the sentinel value of the "all watched layers" entry (a NUL
# prefix keeps it distinct from any real layer name) and its display label.
_LAYER_ALL: str = "\x00all"
_ALL_LAYERS_LABEL: str = "All watched layers"
# Rendering every watched layer's cards at once is what makes the page slow,
# so the "all" entry is only offered while fewer than this many layers are
# watched; at or above it a single layer must be picked.
_ALL_LAYERS_MAX: int = 10


def _watched_in_order(
    layer_names: list[str], watched: frozenset[str]
) -> list[str]:
    """Currently-watched layers in the page's stable graph order.

    `watched` is an unordered set, so order comes from `layer_names` (the
    architecture order the cards have always rendered in).
    """
    return [n for n in layer_names if n in watched]


def _selectable_layers(
    selected_phase: str, layer_names: list[str], watched: frozenset[str]
) -> list[str]:
    """The layers the Layer dropdown offers for the current phase selection.

    "Current batch" stats come from the snapshot, which covers every layer, so
    any layer is selectable there; a real phase only has running stats for the
    watched layers. Either way the order is the stable graph order.
    """
    if selected_phase == _PHASE_CURRENT_BATCH:
        return list(layer_names)
    return _watched_in_order(layer_names, watched)


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

_PATCH_TYPE_LABELS: dict[PatchType, str] = {
    "max_pixel": "Max pixel",
    "min_pixel": "Min pixel",
    "max_average": "Max average",
    "min_average": "Min average",
}

_NO_PATCHES_HTML: str = (
    '<div class="text-xs text-slate-400 italic py-1">no patches gathered '
    "yet — patches need an image-like (4D) model input</div>"
)

# Shown in a layer card (both views) until the layer has accumulated any
# stats for the selected phase. Stresses that only batches stepped after
# the layer is watched feed the running aggregate (it grows rather than
# overwriting with the last batch).
_NO_STATS_MESSAGE: str = (
    "No stats collected for this layer yet — step at least one batch to "
    "start collecting. Each batch you step after watching the layer adds "
    "to the running statistics."
)


def _patch_grids_signature(
    per_phase: dict[str, LayerStatsSnapshot],
    enabled: list[PatchType],
    heatmap: bool,
) -> tuple[object, ...]:
    """Cheap change-detection key for a panel's patch grids.

    The stored extreme values identify the buffer contents: any accepted
    candidate changes its channel's value row, so unchanged values ⇒
    unchanged patches (within one epoch bucket, which the phase/epoch part
    of the key pins down).
    """
    parts: list[object] = [tuple(enabled), heatmap]
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
    mean: tuple[float, ...] | None,
    std: tuple[float, ...] | None,
) -> str:
    """The MIN/MAX view body for one layer: per-phase blocks of grids."""
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
        blocks.append(
            '<div class="flex flex-col gap-2 w-full">'
            f'<div class="font-mono text-xs font-bold" style="color:{color}">'
            f"{phase} (ep {layer_snap.epoch})</div>" + "".join(rows) + "</div>"
        )
    if not blocks:
        return _NO_PATCHES_HTML
    return '<div class="flex flex-col gap-4 w-full">' + "".join(blocks) + "</div>"


def _patch_column_html(column: PatchColumn, mime: str, label: str) -> str:
    """One channel column of a patch grid: a `CHANNEL n` header over its cells.

    Mirrors the activation strips' table layout — a slate header bar
    (`_column_header_bar`) over the channel's top-N sample cells, each a
    separate `cell_size` square stacked with a `PATCH_CELL_GAP` gutter so the
    grid reads as discrete cells rather than one merged column.
    """
    size = column.cell_size
    header = _column_header_bar(column.label, size)
    cells = "".join(
        f'<img src="{_b64_img_src(cell, mime=mime)}" '
        f"style=\"width:{size}px; height:{size}px; image-rendering:pixelated; "
        f'display:block; max-width:none;" title="{html.escape(label)} — '
        f'{html.escape(column.label)}, sample {i} (best first)" />'
        for i, cell in enumerate(column.cells)
    )
    return (
        f'<div style="display:flex; flex-direction:column; flex:none; '
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
        _patch_column_html(column, grid.mime, label) for column in grid.columns
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
