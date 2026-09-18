"""Stats sidebar layout and controls; actions are supplied by the page."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass

from nicegui import ui
from nicegui.elements.mixins.disableable_element import DisableableElement

from nansense.patches import PatchType
from nansense.ui.common import _resizable_pane_props
from nansense.ui.controllers.stats import (
    _PHASE_CURRENT_BATCH_LABEL,
    _VIEW_GRAPHS,
    _VIEW_HISTOGRAM,
    _VIEW_MINMAX,
    _grid_type_options,
    _phase_select_options,
    _WatchPageState,
)


@dataclass(frozen=True)
class StatsActions:
    set_mode: Callable[[object], Awaitable[None]]
    set_phase: Callable[[object], Awaitable[None]]
    set_layer: Callable[[object], Awaitable[None]]
    set_axis_log_x: Callable[[bool], Awaitable[None]]
    set_axis_log_y: Callable[[bool], Awaitable[None]]
    set_retain_axes: Callable[[bool], Awaitable[None]]
    set_show_bands: Callable[[bool], Awaitable[None]]
    set_grid: Callable[[PatchType], Awaitable[None]]
    set_heat: Callable[[bool], Awaitable[None]]


class StatsSidebar:
    def __init__(
        self,
        state: _WatchPageState,
        phase_names: list[str],
        average_patches: bool,
        actions: StatsActions,
    ) -> None:
        with (
            ui.column()
            .classes(
                "w-80 shrink-0 h-full overflow-auto p-4 gap-2 "
                "border-r-2 border-slate-300 bg-slate-50"
            )
            .props(_resizable_pane_props("watch-controls"))
        ):
            with ui.row().classes("items-baseline gap-2 no-wrap"):
                ui.label("Stats").classes("font-mono text-base font-bold")
                self.count_label = ui.label("").classes("text-sm text-slate-500")
            ui.separator()
            # `data-tour` marks the three dropdowns as the tour's arrow
            # targets (`tour.stats_tour_steps`).
            self.view_select = (
                ui.select(
                    [_VIEW_HISTOGRAM, _VIEW_MINMAX, _VIEW_GRAPHS],
                    label="View",
                    value=state.view,
                    on_change=lambda e: actions.set_mode(e.value),
                )
                .props('dense outlined options-dense data-tour="view"')
                .classes("w-full text-sm")
                .tooltip("What each layer card shows")
            )
            # Phases, then "Current batch" as the last entry (dropped in
            # the epoch-stats view — see `sync_phase_select`). A scoped
            # `option` slot draws a divider above it (Quasar has no native
            # per-option separator) while keeping default selection via
            # `itemProps`.
            self.phase_select = (
                ui.select(
                    _phase_select_options(state.view, phase_names),
                    label="Phase",
                    value=state.selected_phase,
                    on_change=lambda e: actions.set_phase(e.value),
                )
                .props('dense outlined options-dense data-tour="phase"')
                .classes("w-full text-sm")
                .tooltip("Which phase the cards show")
            )
            # The divider keys off the label, not the value: NiceGUI sets
            # each option's `value` to its integer index, so only the label
            # carries our sentinel text.
            self.phase_select.add_slot(
                "option",
                '<q-separator v-if="props.opt.label === '
                f"'{_PHASE_CURRENT_BATCH_LABEL}'\" />"
                '<q-item v-bind="props.itemProps">'
                "<q-item-section><q-item-label>"
                "{{ props.opt.label }}"
                "</q-item-label></q-item-section></q-item>",
            )
            self.layer_select = (
                ui.select(
                    {},
                    label="Layer",
                    on_change=lambda e: actions.set_layer(e.value),
                )
                .props('dense outlined options-dense data-tour="layer"')
                .classes("w-full text-sm")
                .tooltip("Which layer's cards to show — one keeps the page fast")
            )
            self.hist_boxes: list[ui.checkbox] = []
            self.minmax_boxes: list[DisableableElement] = []
            with ui.column().classes("w-full gap-1") as self.hist_controls:
                self.hist_boxes.append(
                    ui.checkbox(
                        "Log x",
                        value=state.axis_log_x,
                        on_change=lambda e: actions.set_axis_log_x(bool(e.value)),
                    )
                    .props("dense")
                    .classes("text-sm")
                    .tooltip("Log scale on the value axis")
                )
                self.hist_boxes.append(
                    ui.checkbox(
                        "Log y",
                        value=state.axis_log_y,
                        on_change=lambda e: actions.set_axis_log_y(bool(e.value)),
                    )
                    .props("dense")
                    .classes("text-sm")
                    .tooltip("Log scale on the probability axis")
                )
                self.hist_boxes.append(
                    ui.checkbox(
                        "Retain axes",
                        value=state.retain_axes,
                        on_change=lambda e: actions.set_retain_axes(bool(e.value)),
                    )
                    .props("dense")
                    .classes("text-sm")
                    .tooltip(
                        "Keep the current axis ranges instead of "
                        "auto-fitting to the data"
                    )
                )
                self.hist_boxes.append(
                    ui.checkbox(
                        "Show subnormal/overflow",
                        value=state.show_bands,
                        on_change=lambda e: actions.set_show_bands(bool(e.value)),
                    )
                    .props("dense")
                    .classes("text-sm")
                    .tooltip("Mark the subnormal and overflow magnitude bands")
                )
            with ui.column().classes("w-full gap-1") as self.minmax_controls:
                self.grid_radio = (
                    ui.radio(
                        _grid_type_options(average_patches),
                        value=state.grid_type,
                        on_change=lambda e: actions.set_grid(e.value),
                    )
                    .props("dense")
                    .classes("text-sm")
                    .tooltip("Which extreme-activation patch grid to show")
                )
                self.minmax_boxes.append(self.grid_radio)
                self.minmax_boxes.append(
                    ui.checkbox(
                        "Enable heatmap",
                        value=state.heat_on,
                        on_change=lambda e: actions.set_heat(bool(e.value)),
                    )
                    .props("dense")
                    .classes("text-sm")
                    .tooltip(
                        "Blend each channel's activation strength over the patches"
                    )
                )
            self.hist_controls.set_visibility(state.view == _VIEW_HISTOGRAM)
            self.minmax_controls.set_visibility(state.view == _VIEW_MINMAX)
            # Pinned to the very bottom of the sidebar (below a flexible
            # spacer): jump to the same layer's Deep Dream experiment, the
            # synthesized counterpart to these real-input extremes.
            # Shown only in the MIN/MAX view, like the controls above.
            ui.space()
            self.compare_deep_dream = (
                ui.button(
                    "Compare with Deep Dream",
                    icon="science",
                    color="yellow-8",
                )
                .props("dense no-caps size=sm")
                .classes("w-full")
                .tooltip("Open this layer's Deep Dream experiment")
            )
            self.compare_deep_dream.set_visibility(state.view == _VIEW_MINMAX)
