"""Settings sections with explicit widget ownership and guarded state loading."""

from __future__ import annotations

from nicegui import ui

from nansense.patches import DEFAULT_SAMPLES_PER_CHANNEL
from nansense.session import Session, StatsScope
from nansense.watch import DEFAULT_CHANNEL_LIMIT

_FREQUENCY_UNIT_OPTIONS = {"epoch": "Every nth epoch", "batch": "Every nth batch"}
_STATS_SCOPE_OPTIONS = {
    str(StatsScope.NONE): "No layers (paused)",
    str(StatsScope.WATCHED): "Watched layers",
    str(StatsScope.ALL): "All layers",
}
_ANY_PHASE = "(any phase)"


class CollectionSettings:
    def __init__(self, session: Session) -> None:
        self.session = session
        self.loading = True
        ui.label("Experiments").classes("text-lg font-bold")
        self.auto_run_switch = (
            ui.switch(
                "Auto-run experiments",
                on_change=lambda: self.apply_auto_run(),
            )
            .props("dense")
            .tooltip(
                "Re-run the experiment on every parameter change, without pressing Run"
            )
        )
        ui.separator()
        ui.label("Statistics collection").classes("text-lg font-bold")
        ui.label(
            "Choose which layers contribute to histograms, input galleries, "
            "and per-epoch graphs."
        ).classes("text-sm text-slate-600")
        self.scope_select = (
            ui.select(
                _STATS_SCOPE_OPTIONS,
                label="Collect stats for",
                value=str(StatsScope.WATCHED),
                on_change=lambda: self.apply_stats_scope(),
            )
            .props("dense outlined")
            .classes("w-64")
            .tooltip(
                'Which layers collect statistics — "No layers" keeps what is '
                "already collected"
            )
        )

    def load(self) -> None:
        self.loading = True
        try:
            self.auto_run_switch.value = self.session.auto_run_experiments
            self.scope_select.value = str(self.session.stats_scope)
        finally:
            self.loading = False

    def apply_stats_scope(self) -> None:
        """Push the stats-scope select to the session (auto-applied)."""
        if self.loading:
            return
        self.session.set_stats_scope(str(self.scope_select.value))

    def apply_auto_run(self) -> None:
        if not self.loading:
            self.session.set_auto_run_experiments(bool(self.auto_run_switch.value))


class WatchSettings:
    def __init__(self, session: Session) -> None:
        self.session = session
        self.loading = True
        ui.label("Performance").classes("text-lg font-bold")
        ui.label(
            "Reduce the detail below if NaNsense uses too much memory or slows training."
        ).classes("text-sm text-slate-600")
        ui.label("Watched-layer memory").classes("text-sm font-medium mt-1")
        ui.label("Channel galleries are usually the largest memory cost.").classes(
            "text-xs text-slate-500"
        )
        self.channel_limit_switch = (
            ui.switch(
                "Limit recorded channels",
                on_change=lambda: self.apply_watch_performance(),
            )
            .props("dense")
            .tooltip(
                "Limit per-channel data to the first N channels; off keeps every "
                "channel (highest VRAM)"
            )
        )
        with ui.row().classes("w-full gap-2 no-wrap items-start"):
            self.channel_limit_input = (
                ui.number(
                    label="Channels",
                    value=DEFAULT_CHANNEL_LIMIT,
                    min=1,
                    step=1,
                    format="%d",
                    on_change=lambda: self.apply_watch_performance(),
                )
                .props("dense outlined")
                .classes("flex-1")
                .tooltip("How many channels keep per-channel data")
            )
            self.samples_input = (
                ui.number(
                    label="Samples per channel",
                    value=DEFAULT_SAMPLES_PER_CHANNEL,
                    min=1,
                    step=1,
                    format="%d",
                    on_change=lambda: self.apply_watch_performance(),
                )
                .props("dense outlined")
                .classes("flex-1")
                .tooltip("How many extreme inputs to keep per channel")
            )
        self.average_patches_switch = (
            ui.switch(
                "Average-extreme patch galleries",
                on_change=lambda: self.apply_watch_performance(),
            )
            .props("dense")
            .tooltip("Also collect the max/min-average galleries (extra VRAM)")
        )
        ui.label("Changing these options clears collected statistics.").classes(
            "text-xs text-red-500"
        )

    def load(self) -> None:
        self.loading = True
        try:
            perf = self.session.watch_performance
            self.channel_limit_switch.value = perf.channel_limit_enabled
            self.channel_limit_input.value = perf.channel_limit
            self.channel_limit_input.set_enabled(perf.channel_limit_enabled)
            self.samples_input.value = perf.samples_per_channel
            self.average_patches_switch.value = perf.average_patches
        finally:
            self.loading = False

    def apply_watch_performance(self) -> None:
        """Push the per-channel watch caps to the session (auto-applied)."""
        if self.loading:
            return
        enabled = bool(self.channel_limit_switch.value)
        # The channel count is moot when the cap is off.
        self.channel_limit_input.set_enabled(enabled)
        try:
            limit = (
                int(self.channel_limit_input.value)
                if self.channel_limit_input.value is not None
                else DEFAULT_CHANNEL_LIMIT
            )
            samples = (
                int(self.samples_input.value)
                if self.samples_input.value is not None
                else DEFAULT_SAMPLES_PER_CHANNEL
            )
        except (TypeError, ValueError):
            return
        flushed = self.session.set_watch_performance(
            channel_limit_enabled=enabled,
            channel_limit=limit,
            samples_per_channel=samples,
            average_patches=bool(self.average_patches_switch.value),
        )
        if flushed:
            ui.notify("Watch statistics flushed", type="info")


class FrequencySettings:
    def __init__(self, session: Session) -> None:
        self.session = session
        self.loading = True
        ui.label("Update frequency").classes("text-sm font-medium mt-1")
        ui.label(
            "How often views refresh while training runs. They also refresh when training stops."
        ).classes("text-xs text-slate-500")
        with ui.row().classes("w-full gap-2 no-wrap items-start"):
            self.unit_select = (
                ui.select(
                    _FREQUENCY_UNIT_OPTIONS,
                    label="Update every",
                    value="epoch",
                    on_change=lambda: self.on_unit_change(),
                )
                .props("dense outlined")
                .classes("flex-1")
            )
            self.n_input = (
                ui.number(
                    label="n",
                    value=1,
                    min=1,
                    step=1,
                    format="%d",
                    on_change=lambda: self.apply_frequency(),
                )
                .props("dense outlined")
                .classes("w-20")
                .tooltip("Update on every nth epoch/batch")
            )
            self.phase_select = (
                ui.select(
                    [_ANY_PHASE] + self.session.schedule.phase_order,
                    label="Phase",
                    value=_ANY_PHASE,
                    on_change=lambda: self.apply_frequency(),
                )
                .props("dense outlined")
                .classes("flex-1")
                .tooltip("Count only this phase's batches (batch unit only)")
            )
        self.error_label = ui.label("").classes("text-red-500 text-sm min-h-4")
        self.lock_note = ui.label(
            "The frequency is locked while recordings are active — frames "
            "are recorded at this cadence."
        ).classes("text-xs text-amber-700")

    def load(self) -> None:
        self.loading = True
        try:
            freq = self.session.update_frequency
            self.unit_select.value = freq.unit
            self.n_input.value = freq.n
            self.phase_select.set_options(
                [_ANY_PHASE] + self.session.schedule.phase_order,
                value=freq.phase if freq.phase is not None else _ANY_PHASE,
            )
            self.sync_phase_visibility()
            self.error_label.text = ""
            self.refresh_recording_lock()
        finally:
            self.loading = False

    def sync_phase_visibility(self) -> None:
        self.phase_select.set_visibility(self.unit_select.value == "batch")

    def apply_frequency(self) -> None:
        """Push the controls' current values to the session (auto-applied)."""
        if self.loading:
            return
        unit = str(self.unit_select.value)
        phase = str(self.phase_select.value)
        try:
            n = int(self.n_input.value) if self.n_input.value is not None else 1
            self.session.set_update_frequency(
                unit=unit,
                n=n,
                phase=phase if unit == "batch" and phase != _ANY_PHASE else None,
            )
        except (TypeError, ValueError) as e:
            self.error_label.text = str(e)
            return
        self.error_label.text = ""

    def on_unit_change(self) -> None:
        # The phase select only applies to the batch unit; show/hide it before
        # re-applying so a stale phase doesn't leak into an epoch-unit setting.
        self.sync_phase_visibility()
        self.apply_frequency()

    def refresh_recording_lock(self) -> None:
        locked = self.session.recording.count() > 0
        self.lock_note.set_visibility(locked)
        for control in (self.unit_select, self.n_input, self.phase_select):
            control.set_enabled(not locked)


class DebugSettings:
    def __init__(self, session: Session) -> None:
        self.session = session
        self.loading = True
        ui.label("Error checks").classes("text-lg font-bold")
        ui.label(
            "Pause when NaN or infinite values appear, or when gradients approach "
            "the limits of their numeric format."
        ).classes("text-sm text-slate-600")
        self.debug_enable = ui.switch(
            "Enable error checks",
            on_change=lambda: self.apply_debug(),
        ).props("dense")
        with ui.row().classes("w-full gap-2 no-wrap items-start"):
            self.debug_interval = (
                ui.number(
                    label="Check every (batches)",
                    value=100,
                    min=1,
                    step=1,
                    format="%d",
                    on_change=lambda: self.apply_debug(),
                )
                .props("dense outlined")
                .classes("flex-1")
            )
            self.debug_threshold = (
                ui.number(
                    label="Under/overflow %",
                    value=10,
                    min=0,
                    max=100,
                    step=1,
                    format="%g",
                    on_change=lambda: self.apply_debug(),
                )
                .props("dense outlined")
                .classes("flex-1")
                .tooltip(
                    "Trip when this share of a layer's |gradient| falls in the "
                    "under/overflow band"
                )
            )
        with ui.row().classes("w-full gap-6 no-wrap"):
            self.debug_nan_inf = (
                ui.switch("NaN / Inf", on_change=lambda: self.apply_debug())
                .props("dense")
                .tooltip("Flag any NaN or ±Inf value")
            )
            self.debug_under_over = (
                ui.switch("Underflow / overflow", on_change=lambda: self.apply_debug())
                .props("dense")
                .tooltip("Flag gradients in the dtype's subnormal or saturation range")
            )

    def load(self) -> None:
        self.loading = True
        try:
            debug = self.session.debug_settings
            self.debug_enable.value = debug.enabled
            self.debug_interval.value = debug.interval
            self.debug_threshold.value = round(debug.threshold * 100, 4)
            self.debug_nan_inf.value = debug.check_nan_inf
            self.debug_under_over.value = debug.check_under_over
        finally:
            self.loading = False

    def apply_debug(self) -> None:
        """Push the error-check controls to the session (auto-applied)."""
        if self.loading:
            return
        try:
            interval = (
                int(self.debug_interval.value)
                if self.debug_interval.value is not None
                else 100
            )
            percent = (
                float(self.debug_threshold.value)
                if self.debug_threshold.value is not None
                else 10.0
            )
        except (TypeError, ValueError):
            return
        self.session.set_debug_settings(
            enabled=bool(self.debug_enable.value),
            interval=interval,
            check_nan_inf=bool(self.debug_nan_inf.value),
            check_under_over=bool(self.debug_under_over.value),
            threshold=percent / 100.0,
        )
