"""The shared top-bar/step controls and the dialogs they open.

The Share dialog the top bar's share icon opens is its own module
(`nansense.ui.share`) — it carries three share targets, their previews, and
the route that hands the demo video over as a file.
"""

from __future__ import annotations

import asyncio
import base64
from bisect import bisect_right
from collections.abc import Callable
from functools import lru_cache
from typing import Literal
from urllib.parse import quote

import torch
from nicegui import ui

from nansense import debugger
from nansense.assets import logo_small_bytes
from nansense.debugger import DebugError, LayerReport
from nansense.patches import DEFAULT_SAMPLES_PER_CHANNEL
from nansense.recording import RecordedView
from nansense.restore import TimeTravelError
from nansense.schedule import BatchPosition, format_position
from nansense.session import (
    BatchSnapshot,
    Session,
    StatsScope,
    lost_loop_reason,
)
from nansense.watch import DEFAULT_CHANNEL_LIMIT


_TOP_BAR_CLASSES: str = (
    "w-full items-center gap-x-3 gap-y-0 px-3 py-2 shrink-0 "
    "border-b-2 border-slate-300 bg-slate-100 shadow-sm z-10"
)

_REPO_URL: str = "https://github.com/kongaskristjan/nansense"
_STAR_TOOLTIP: str = "View NaNsense on GitHub"
# The wordmark beside the brand mark (`_add_repo_logo`). The tagline is the
# docs site's own one-liner, cut to what fits a top bar.
_BRAND_NAME: str = "NaNsense"
_BRAND_TAGLINE: str = "PyTorch debugger"
# The dialog's own explanation of what tripped. The banner links to the
# dialog rather than repeating this on hover — see `_DEBUG_BANNER_TOOLTIP`.
_DEBUG_DESCRIPTION: str = (
    "NaNsense found NaN or infinite values, or gradients near the limits of "
    "their numeric format. Training paused at the first issue."
)
_DEBUG_BANNER_TOOLTIP: str = "Click for the affected layers"
_DEBUG_UNDER_OVER_INTRO: str = (
    "Subnormal / overflow is dtype-aware and measured on gradients. A value "
    "counts as subnormal when its magnitude is nonzero but below the dtype's "
    "smallest normal value — still representable, but with fewer and fewer "
    "significant bits as it shrinks toward zero (and which some hardware "
    "flushes straight to zero). It counts as overflow when its magnitude "
    "climbs to within a small factor of the dtype's largest finite value — "
    "close enough to be about to saturate to ±inf (actual saturation itself "
    "shows up under ±Inf). A layer trips when that band holds at least the "
    "threshold share of its summed |gradient|."
)
_DEBUG_UNDER_OVER_TIP: str = (
    "Tip: fp16 gradients slip into the subnormal range (and then to zero) "
    "easily. Keep them in range with loss scaling (torch.amp.GradScaler), or "
    "switch to bfloat16 (torch.autocast(device_type, dtype=torch.bfloat16)) — "
    "it shares fp32's exponent range, so it rarely goes subnormal or overflows "
    "(trading a little precision). For overflow, lower the loss scale or clip "
    "gradients (torch.nn.utils.clip_grad_norm_)."
)
_DEBUG_WATCH_NOTE: str = (
    "A layer's gradient histogram only becomes available once it has been "
    "watched for a few batches — click Watch, let training step a few times, "
    "then open the histogram view."
)


def _debug_banner_summary(error: DebugError) -> str:
    """The one-line warning-banner message for a detected issue."""
    return (
        f"Numerical issue detected — {debugger.reasons_text(error)} — at "
        f"{format_position(error.position)}"
    )


def _under_over_band_lines(error: DebugError) -> list[str]:
    """Per-dtype subnormal/overflow band magnitudes for the affected layers.

    One line per distinct gradient dtype seen across the error's layers,
    spelling out the exact magnitudes counted as subnormal and (near-)overflow
    for that dtype — so the dialog states the band in real numbers rather than
    abstractly.
    """
    dtypes: list[torch.dtype] = []
    for report in error.layers:
        if report.dtype is not None and report.dtype not in dtypes:
            dtypes.append(report.dtype)
    lines: list[str] = []
    for dtype in dtypes:
        tiny, maxv = debugger.dtype_band(dtype)
        name = str(dtype).removeprefix("torch.")
        lines.append(
            f"{name}: subnormal when 0 < |grad| < {tiny:.2e}; "
            f"overflow when |grad| ≥ {maxv:.2e}"
        )
    return lines


def _debug_pct(frac: float) -> str:
    """Format a fraction in [0, 1] as a table-cell percentage."""
    if frac <= 0.0:
        return "—"
    if frac < 0.001:
        return "<0.1%"
    return f"{frac * 100:.1f}%"


def _top_bar_row() -> ui.row:
    """The shared top-bar row container used by every page."""
    return ui.row().classes(_TOP_BAR_CLASSES)


@lru_cache(maxsize=1)
def _logo_data_uri() -> str:
    """The NaNsense mark as a base64 data URI (read once, then cached).

    Inlined rather than served from a static route — the mark is ~3 KB and this
    matches the data-URI pattern the strip images already use (see
    `common._b64_img_src`), so it needs no extra media route.
    """
    encoded = base64.b64encode(logo_small_bytes()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def _add_repo_logo() -> ui.link:
    """The NaNsense brand mark and wordmark at the far-right of the top bar.

    Sits last in every page's top bar — after the right-aligned controls — as a
    quiet star call-to-action. Rendered as a native link opening in a new tab
    (so middle/ctrl-click works, like the nav buttons) with a hover tooltip
    nudging a repo star. Returned so a top bar with no `ml-auto` control of its
    own can right-align it directly.

    The mark is followed by the name and a two-word descriptor, because the
    app is mostly met where nothing around it says what it is: inside an
    iframe on the docs pages, or on a bare Space URL reached from a shared
    link. A wordless mark leaves a visitor no way to learn that the demo they
    are driving is a library they can run themselves — this is the one place
    the app names itself. `data-tour="brand"` anchors the closing tour step.
    """
    link = (
        ui.link(target=_REPO_URL, new_tab=True)
        .classes("shrink-0 flex items-center gap-2 no-underline")
        .props('data-tour="brand"')
    )
    with link:
        ui.image(_logo_data_uri()).classes("h-7 w-7").props("no-spinner")
        # A plain flex div rather than `ui.column`, whose default gap would
        # have to be fought off; the two lines stack to the mark's height.
        with ui.element("div").classes("flex flex-col leading-none"):
            ui.label(_BRAND_NAME).classes("text-sm font-bold text-slate-700")
            ui.label(_BRAND_TAGLINE).classes("text-[10px] text-slate-500")
        # Right-anchored so the tooltip grows leftward and stays on-screen at
        # the top bar's right edge.
        ui.tooltip(_STAR_TOOLTIP).props('anchor="bottom right" self="top right"')
    return link


def _add_tour_button() -> None:
    """The `?` left of the share button (every page): (re)runs the page's tour.

    Purely client-side — the driver installed by `tour.add_tour` owns all
    the state, so the click never round-trips to the server, and it replays
    the tour even after it was dismissed. Available on any session; only
    the locked playground *auto*-starts a tour (see `tour.add_tour`), so a
    local runner sees it exactly when they ask.
    """
    button = ui.button(icon="help_outline", color="slate-500").props(
        "dense size=md flat"
    )
    button.on("click", js_handler="() => window.nansenseStartTour()")
    button.tooltip("Quick tour")


def _back_href(layer: str | None) -> str:
    """The main-page URL a subpage's Back button targets.

    With a layer, a `?layer=` deep link — the locked playground's subpages
    carry the layer they show, so going back opens the main page with that
    layer's card visible and scrolled into view (`main_page._build_page`).
    """
    return f"/?layer={quote(layer)}" if layer else "/"


def _back_button(layer: str | None = None) -> ui.button:
    """The arrow-back button to the main page (every subpage's top bar).

    Rendered as a native link (`href` prop) rather than an `on_click`
    navigation so middle/ctrl-click opens the main page in a new tab —
    same pattern as the layer cards' Weights/Experiment buttons. Returned
    so pages that switch layers in place can keep the `?layer=` deep link
    current (see `_back_href`).
    """
    return (
        ui.button(
            icon="arrow_back",
            color="slate-500",
        )
        .props(f'dense size=md href="{_back_href(layer)}"')
        .tooltip("Back to the main page")
    )


def _refresh_button(session: Session) -> None:
    """The Refresh button shared by the main and weights top bars.

    Arms `Session.request_snapshot` so the next training batch publishes a
    snapshot, updating the activations, gradients, weights, and probe at once.
    The visualizations are already current when training is paused or idle;
    this only matters in `detach` / `step_run`, where training runs freely and
    the views would otherwise stay frozen between frequency-cadence updates.
    Styled like the adjacent nav button so it reads as the second left-cluster
    control on every page that shows it. Skipped on a locked session — parked
    training produces no batches to refresh from.
    """
    if session.locked:
        return
    ui.button(
        icon="refresh",
        on_click=session.request_snapshot,
        color="slate-500",
    ).props("dense size=md").tooltip("Update from the next training batch")


def _add_step_controls(
    session: Session,
    step_until_custom: ui.dialog,
) -> None:
    """Add the stepping buttons + a live-position label to the open row.

    Shared by every page's top bar so they all drive the session
    identically. The settings (gear) button lives in the top bar's
    right-side cluster instead — each page adds it via
    `_add_settings_button`. The label tracks the *live* training position
    via a timer registered here (see `format_position`); the 0.2s period is
    the throttle — rapid batches in step_epoch / step_run / detach coalesce
    into at most ~5 cheap label updates per second, and NiceGUI skips the
    write when the text is unchanged.

    On a locked session the whole control cluster is replaced by a demo
    notice — the session refuses the calls anyway (`Session.lock`), so the
    buttons would only mislead. The position label goes with them: parked
    training never advances, so it would read as a fixed, unexplained
    "epoch 49/50 | train batch 590/591" forever.
    """
    if session.locked:
        with ui.element("div").classes(
            "flex items-center gap-1 px-2 py-1 rounded bg-amber-100 "
            "text-amber-800 text-sm font-medium"
        ).tooltip(
            "Paused at a trained checkpoint. Layers, stats, and experiments "
            "all work."
        ):
            ui.icon("lock").classes("text-base")
            ui.label("playground — training controls disabled")
        return
    last_batch_confirm = _build_last_batch_confirm_dialog(session)

    def run() -> None:
        # Run normally pauses *at* the final batch (UNTIL_END); stepping past
        # it ends the run, so that only happens through the confirmation.
        if _at_last_batch(session):
            last_batch_confirm.open()
        else:
            session.step_run()

    def step_batch() -> None:
        if _at_last_batch(session):
            last_batch_confirm.open()
        else:
            session.step_batch()

    # The wrapper mirrors the top-bar row's flex/gap so it renders
    # invisibly; it exists as the tour's anchor for the whole stepping
    # cluster (one ring around Run / Step Batch / Stop / time travel).
    with ui.element("div").classes("flex items-center gap-x-3").props(
        'data-tour="step-controls"'
    ):
        run_button = (
            ui.button("Run", on_click=run, color="green")
            .props("dense size=md")
            .tooltip("Run to the end of training")
        )
        step_button = (
            ui.dropdown_button(
                "Step Batch",
                on_click=step_batch,
                split=True,
                auto_close=True,
                color="orange",
            )
            .props("dense size=md")
            .tooltip("Advance one batch")
        )
        with step_button:
            _step_menu_item(
                "Step epoch",
                "Run to the start of the next epoch",
                session.step_epoch,
            )
            _step_menu_item(
                "Step custom…",
                "Pick a phase/epoch/batch to pause at",
                step_until_custom.open,
            )
        stop_button = (
            ui.button("Stop", on_click=session.stop, color="red")
            .props("dense size=md")
            .tooltip("Pause at the next batch")
        )
        _add_time_travel_button(session)
    _add_position_label(session)
    ended_chip, ended_tooltip = _add_ended_chip()
    # Run is grayed while training advances (started by Run or any Step), Stop
    # while it sits paused — a 0.2s timer toggles both off
    # `session.is_running`. Once the run is over, every control is grayed: a
    # finished run has nothing left to advance, and a loop that died would
    # leave Run grayed forever anyway (`is_running` never clears without the
    # pause the thread never reached). `None` forces the first apply.
    last_state: tuple[bool, bool, tuple[str, str] | None] | None = None

    def refresh_buttons() -> None:
        nonlocal last_state
        over = session.closed or session.training_lost
        running = session.is_running
        finished = _ended_note(session)
        state = (running, over, finished)
        if state == last_state:
            return
        last_state = state
        run_button.set_enabled(not running and not over)
        step_button.set_enabled(not over)
        stop_button.set_enabled(running and not over)
        if finished is not None:
            ended_chip.text, ended_tooltip.text = finished
        ended_chip.set_visibility(finished is not None)

    refresh_buttons()
    ui.timer(0.2, refresh_buttons)


def _ended_note(session: Session) -> tuple[str, str] | None:
    """Chip text and tooltip for a run that ended cleanly, else `None`.

    A finished run leaves controls that cannot do anything, and a grayed-out
    Run with no explanation is the thing this chip exists to prevent. The
    other way a run ends — the loop dying — grays the same controls but says
    so in the red banner under the bar (`_add_lost_loop_banner`), which has
    room for the exception; repeating it here would be noise.
    """
    if not session.closed:
        return None
    return (
        "training finished",
        "The training loop ran to the end and closed the session. Stepping "
        "and running do nothing; everything already captured stays "
        "inspectable.",
    )


def _add_ended_chip() -> tuple[ui.label, ui.tooltip]:
    """The (initially hidden) amber chip and tooltip `refresh_buttons` fills in."""
    chip = ui.label().classes(
        "ml-3 px-2 py-1 rounded bg-amber-100 text-amber-800 text-sm "
        "font-medium max-w-xl truncate"
    )
    with chip:
        tooltip = ui.tooltip("")
    chip.set_visibility(False)
    return chip, tooltip


def _add_position_label(session: Session) -> None:
    """The live epoch/phase/batch label, updated by its own 0.2 s timer."""
    position_label = ui.label("(waiting for first batch)").classes(
        "ml-3 font-mono text-sm"
    )

    def refresh() -> None:
        live = session.live_position
        if live is not None:
            # Append the run's totals when known: total epochs from the
            # schedule, total batches from the live phase's learned count
            # (both stay None — and so render bare — until known).
            schedule = session.schedule
            position_label.text = format_position(
                live,
                total_epochs=schedule.epochs,
                total_batches=schedule.phase_count(live.phase),
            )

    ui.timer(0.2, refresh)


def _step_menu_item(
    title: str, caption: str, on_click: Callable[[], object]
) -> None:
    """One entry of the Step dropdown: an action title + explanatory caption."""
    with ui.item(on_click=on_click), ui.item_section():
        ui.item_label(title)
        ui.item_label(caption).props("caption")


def _at_last_batch(session: Session) -> bool:
    """Whether training is paused on the final overall batch.

    Resuming from here in any mode runs the loop off its end, so Run and Step
    Batch confirm before stepping past it (see `_build_last_batch_confirm_dialog`).
    """
    pos = session.live_position
    return pos is not None and pos.is_last_overall


def _build_last_batch_confirm_dialog(session: Session) -> ui.dialog:
    """The confirm-before-stepping-past-the-final-batch dialog.

    At the last overall batch there is nowhere left to advance to but the end
    of the training loop: stepping past it discards the collected stats and
    lets the training script exit, closing the session. Both Run and Step Batch
    route here when `_at_last_batch`, so the step only happens on confirmation.
    """
    with ui.dialog() as dialog, ui.card().classes("min-w-96 p-6 gap-3"):
        ui.label("Step over the last batch?").classes("text-lg font-bold")
        ui.label(
            "Training is paused at the final batch. Stepping past it ends the "
            "run — the collected stats are lost and the session will likely "
            "close."
        ).classes("text-sm text-slate-700")
        with ui.row().classes("w-full justify-end gap-2"):
            ui.button("Cancel", on_click=dialog.close, color="slate-500")

            def step_over() -> None:
                dialog.close()
                session.step_batch()

            ui.button("Step over", on_click=step_over, color="red")
    return dialog


def _add_time_travel_button(session: Session) -> None:
    """The blue Time Travel button (right of Detach) and its dialogs.

    The button is grayed out — with the reason as a hover tooltip — when the
    session has no training restorer, i.e. the training loop did not opt
    into time travel. The picker dialog rebuilds its content on every open,
    so the cached-epoch choices track the checkpoints written so far; a
    request the session rejects (e.g. a cache written by a different model)
    surfaces in a separate error dialog and no jump happens.
    """
    with ui.dialog() as error_dialog, ui.card().classes("min-w-96 p-6 gap-3"):
        ui.label("Time travel failed").classes("text-lg font-bold text-red-600")
        error_label = ui.label("").classes("text-sm text-slate-700")
        with ui.row():
            ui.button("Close", on_click=error_dialog.close)

    with ui.dialog() as dialog, ui.card().classes("min-w-96 p-6 gap-4"):
        ui.label("Time travel").classes("text-lg font-bold")
        content = ui.column().classes("w-full gap-3")

    def show_error(message: str) -> None:
        error_label.text = message
        error_dialog.open()

    def submit(value: object) -> None:
        if not isinstance(value, (int, float)):
            show_error("Pick an epoch to jump to.")
            return
        try:
            session.request_time_travel(int(value))
        except TimeTravelError as e:
            show_error(str(e))
            return
        dialog.close()

    def cache_run() -> None:
        # Runs training to the very end; with a restorer active, every
        # epoch start gets checkpointed along the way, filling the cache.
        session.step_run()
        dialog.close()

    def rebuild() -> None:
        content.clear()
        status = session.time_travel_status()
        cached = status.cached_epochs
        missing = sorted(set(range(status.total_epochs)) - set(cached))
        with content:
            if not status.available:
                ui.label(status.reason or "Time travel is unavailable.").classes(
                    "text-sm text-slate-600"
                )
                with ui.row():
                    ui.button("Close", on_click=dialog.close)
                return
            # The slider runs over *indices into the cached-epoch list*, so
            # epochs without a checkpoint are unselectable even when the
            # cached set has gaps; the label shows the mapped epoch number.
            epoch_slider: ui.slider | None = None
            if cached:
                position = _current_position(session.live_position, session.snapshot)
                initial = _time_travel_default_index(
                    cached, position.epoch if position is not None else None
                )
                ui.label("Jump to the start of a cached epoch:").classes(
                    "text-sm"
                )
                with ui.row().classes("w-full items-center gap-4 no-wrap"):
                    epoch_label = ui.label(f"epoch {cached[initial]}").classes(
                        "font-mono text-sm w-20 shrink-0"
                    )
                    epoch_slider = ui.slider(
                        min=0,
                        max=len(cached) - 1,
                        step=1,
                        value=initial,
                        on_change=lambda e: epoch_label.set_text(
                            f"epoch {cached[int(e.value)]}"
                        ),
                    ).classes("grow")
            else:
                ui.label("No epochs have a cached model yet.").classes(
                    "text-sm text-slate-600"
                )
            ui.label(
                "Cached epochs: "
                + (_summarize_epoch_ranges(cached) if cached else "none")
            ).classes("text-xs text-slate-500")
            if missing:
                ui.label(
                    f"Uncached epochs: {_summarize_epoch_ranges(missing)}. "
                    "An epoch becomes available once training has passed "
                    "its start."
                ).classes("text-xs text-slate-500")
            with ui.row():
                ui.button("Cancel", on_click=dialog.close)
                if missing:
                    ui.button(
                        "Cache full training run", on_click=cache_run
                    ).tooltip("Run to the end, checkpointing every epoch")
                if epoch_slider is not None:
                    slider = epoch_slider
                    ui.button(
                        "Time travel",
                        on_click=lambda: submit(
                            cached[int(slider.value)]
                            if slider.value is not None
                            else None
                        ),
                        color="blue",
                    )

    def open_dialog() -> None:
        rebuild()
        dialog.open()

    status = session.time_travel_status()
    # Quasar suppresses pointer events on disabled buttons, so the tooltip
    # (which must explain *why* the button is off) lives on a wrapper div.
    tooltip = (
        "Jump to the start of a cached epoch"
        if status.available
        else (status.reason or "Time travel is unavailable.")
    )
    with ui.element("div").tooltip(tooltip):
        button = ui.button("Time Travel", on_click=open_dialog, color="blue").props(
            "dense size=md"
        )
        if not status.available:
            button.props("disable")


_FREQUENCY_UNIT_OPTIONS: dict[str, str] = {
    "epoch": "Every nth epoch",
    "batch": "Every nth batch",
}

# Settings-dialog labels for the stats scope (`Session.set_stats_scope`).
_STATS_SCOPE_OPTIONS: dict[str, str] = {
    str(StatsScope.NONE): "No layers (paused)",
    str(StatsScope.WATCHED): "Watched layers",
    str(StatsScope.ALL): "All layers",
}

# Phase-select sentinel for "count batches of every phase".
_ANY_PHASE: str = "(any phase)"


def _add_settings_button(
    session: Session,
    record_view: Callable[[], RecordedView | None] | None,
) -> ui.button:
    """The gear button (every top bar's right-side cluster) + settings dialog.

    Returned so the page can right-align it (`ml-auto`) when it is the
    first element of the cluster.

    "Experiments" carries the shared, session-wide auto-run toggle
    (`Session.set_auto_run_experiments`): when on, experiment pages run on open
    and on every parameter change instead of waiting for a manual Run.

    "Performance" groups the knobs that trade visualization detail for GPU
    VRAM and overhead. The watched-layer caps (`Session.set_watch_performance`)
    bound per-channel memory: a channel limit (the per-channel histograms and
    extreme-input patch galleries are kept only for the first N channels),
    the samples-per-channel kept by the patch galleries, and whether the
    average-extreme galleries (a whole input image per slot) are collected at
    all — changing any of them flushes all collected watch statistics, since
    the buffer shapes change.
    "Update frequency" (`Session.set_update_frequency`) sets how often all
    visualizations recompute: every nth epoch (the default, n=1) or every nth
    batch, optionally counting only one phase's batches. The frequency is
    locked while recordings are active — recording frames advance at this
    frequency, so changing it mid-recording would change the videos' time
    base.

    "Recording" offers "Snapshot" and "Record" buttons for the page's own
    view (built by `record_view` with the page's *current* parameters —
    frozen for the recording's lifetime, or for the one instant a snapshot
    takes; None when the page's current state can't be captured yet) plus
    the list of all active recordings, each one save-&-finishable (finalize
    the MP4) or deletable (discard it). Snapshot writes one PNG of the view
    as it stands and is always available; Record is replaced by the view's
    entry in that list (marked "this view") while it records, rather than
    offered twice. A red badge on the gear carries the active-recording
    count.

    On a locked session every setting in here is shared, mutable state, so
    the gear opens a short notice instead of the dialog.
    """
    if session.locked:
        with ui.dialog() as locked_dialog, ui.card().classes("max-w-md p-6 gap-3"):
            ui.label("Settings are locked").classes("text-lg font-bold")
            ui.label(
                "This hosted playground is shared by everyone viewing it, so "
                "the session-wide settings (stats collection, update "
                "frequency, performance caps, error checks, recording) and "
                "the shared probe state (pinning, perturbations, forward "
                "mode) are fixed. Everything per-tab — shown layers, "
                "experiments — works normally."
            ).classes("text-sm text-slate-600")
            with ui.row().classes("w-full justify-end"):
                ui.button("Close", on_click=locked_dialog.close).props("flat")
        button = ui.button(
            icon="settings", on_click=locked_dialog.open, color="slate-500"
        ).props("dense size=md")
        button.tooltip("Settings (locked in this demo)")
        return button

    # Phases seen so far; refreshed each time the dialog opens (lazy schedules
    # learn them as training runs).
    phase_names = session.schedule.phase_order

    with ui.dialog() as dialog, ui.card().classes("min-w-[30rem] p-6 gap-3"):
        ui.label("Experiments").classes("text-lg font-bold")
        auto_run_switch = (
            ui.switch(
                "Auto-run experiments",
                on_change=lambda e: session.set_auto_run_experiments(bool(e.value)),
            )
            .props("dense")
            .tooltip(
                "Re-run the experiment on every parameter change, without "
                "pressing Run"
            )
        )
        ui.separator()
        ui.label("Statistics collection").classes("text-lg font-bold")
        ui.label(
            "Choose which layers contribute to histograms, input galleries, "
            "and per-epoch graphs."
        ).classes("text-sm text-slate-600")
        scope_select = ui.select(
            _STATS_SCOPE_OPTIONS,
            label="Collect stats for",
            value=str(StatsScope.WATCHED),
            on_change=lambda: apply_stats_scope(),
        ).props("dense outlined").classes("w-64").tooltip(
            "Which layers collect statistics — \"No layers\" keeps what is "
            "already collected"
        )
        ui.separator()
        ui.label("Performance").classes("text-lg font-bold")
        ui.label(
            "Reduce the detail below if NaNsense uses too much memory or slows training."
        ).classes("text-sm text-slate-600")
        ui.label("Watched-layer memory").classes("text-sm font-medium mt-1")
        ui.label(
            "Channel galleries are usually the largest memory cost."
        ).classes("text-xs text-slate-500")
        channel_limit_switch = ui.switch(
            "Limit recorded channels",
            on_change=lambda: apply_watch_performance(),
        ).props("dense").tooltip(
            "Limit per-channel data to the first N channels; off keeps every "
            "channel (highest VRAM)"
        )
        with ui.row().classes("w-full gap-2 no-wrap items-start"):
            channel_limit_input = ui.number(
                label="Channels",
                value=DEFAULT_CHANNEL_LIMIT,
                min=1,
                step=1,
                format="%d",
                on_change=lambda: apply_watch_performance(),
            ).props("dense outlined").classes("flex-1").tooltip(
                "How many channels keep per-channel data"
            )
            samples_input = ui.number(
                label="Samples per channel",
                value=DEFAULT_SAMPLES_PER_CHANNEL,
                min=1,
                step=1,
                format="%d",
                on_change=lambda: apply_watch_performance(),
            ).props("dense outlined").classes("flex-1").tooltip(
                "How many extreme inputs to keep per channel"
            )
        average_patches_switch = ui.switch(
            "Average-extreme patch galleries",
            on_change=lambda: apply_watch_performance(),
        ).props("dense").tooltip(
            "Also collect the max/min-average galleries (extra VRAM)"
        )
        ui.label(
            "Changing these options clears collected statistics."
        ).classes("text-xs text-red-500")
        ui.label("Update frequency").classes("text-sm font-medium mt-1")
        ui.label(
            "How often views refresh while training runs. They also refresh when training stops."
        ).classes("text-xs text-slate-500")
        with ui.row().classes("w-full gap-2 no-wrap items-start"):
            unit_select = ui.select(
                _FREQUENCY_UNIT_OPTIONS,
                label="Update every",
                value="epoch",
                on_change=lambda: on_unit_change(),
            ).props("dense outlined").classes("flex-1")
            n_input = ui.number(
                label="n",
                value=1,
                min=1,
                step=1,
                format="%d",
                on_change=lambda: apply_frequency(),
            ).props("dense outlined").classes("w-20").tooltip(
                "Update on every nth epoch/batch"
            )
            phase_select = ui.select(
                [_ANY_PHASE] + phase_names,
                label="Phase",
                value=_ANY_PHASE,
                on_change=lambda: apply_frequency(),
            ).props("dense outlined").classes("flex-1").tooltip(
                "Count only this phase's batches (batch unit only)"
            )
        error_label = ui.label("").classes("text-red-500 text-sm min-h-4")
        lock_note = ui.label(
            "The frequency is locked while recordings are active — frames "
            "are recorded at this cadence."
        ).classes("text-xs text-amber-700")
        ui.separator()
        ui.label("Error checks").classes("text-lg font-bold")
        ui.label(
            "Pause when NaN or infinite values appear, or when gradients approach "
            "the limits of their numeric format."
        ).classes("text-sm text-slate-600")
        debug_enable = ui.switch(
            "Enable error checks",
            on_change=lambda: apply_debug(),
        ).props("dense")
        with ui.row().classes("w-full gap-2 no-wrap items-start"):
            debug_interval = ui.number(
                label="Check every (batches)",
                value=100,
                min=1,
                step=1,
                format="%d",
                on_change=lambda: apply_debug(),
            ).props("dense outlined").classes("flex-1")
            debug_threshold = ui.number(
                label="Under/overflow %",
                value=10,
                min=0,
                max=100,
                step=1,
                format="%g",
                on_change=lambda: apply_debug(),
            ).props("dense outlined").classes("flex-1").tooltip(
                "Trip when this share of a layer's |gradient| falls in the "
                "under/overflow band"
            )
        with ui.row().classes("w-full gap-6 no-wrap"):
            debug_nan_inf = ui.switch(
                "NaN / Inf", on_change=lambda: apply_debug()
            ).props("dense").tooltip("Flag any NaN or ±Inf value")
            debug_under_over = ui.switch(
                "Underflow / overflow", on_change=lambda: apply_debug()
            ).props("dense").tooltip(
                "Flag gradients in the dtype's subnormal or saturation range"
            )
        ui.separator()
        ui.label("Recording").classes("text-lg font-bold")
        ui.label(
            "Record the current view as an MP4, or save one frame as a PNG."
        ).classes("text-sm text-slate-600")
        recording_section = ui.column().classes("w-full gap-3")
        ui.separator()
        with ui.row().classes("w-full justify-end"):
            ui.button("Close", on_click=dialog.close).props("flat")

    # Guards the auto-apply handlers while `open_dialog` programmatically
    # loads the session's current values into the controls — otherwise those
    # writes would fire `apply_frequency` mid-load with a half-set combination.
    loading = False

    def sync_phase_visibility() -> None:
        phase_select.set_visibility(unit_select.value == "batch")

    def apply_stats_scope() -> None:
        """Push the stats-scope select to the session (auto-applied)."""
        if loading:
            return
        session.set_stats_scope(str(scope_select.value))

    def apply_frequency() -> None:
        """Push the controls' current values to the session (auto-applied)."""
        if loading:
            return
        unit = str(unit_select.value)
        phase = str(phase_select.value)
        try:
            n = int(n_input.value) if n_input.value is not None else 1
            session.set_update_frequency(
                unit=unit,
                n=n,
                phase=phase if unit == "batch" and phase != _ANY_PHASE else None,
            )
        except (TypeError, ValueError) as e:
            error_label.text = str(e)
            return
        error_label.text = ""

    def on_unit_change() -> None:
        # The phase select only applies to the batch unit; show/hide it before
        # re-applying so a stale phase doesn't leak into an epoch-unit setting.
        sync_phase_visibility()
        apply_frequency()

    def apply_watch_performance() -> None:
        """Push the per-channel watch caps to the session (auto-applied)."""
        if loading:
            return
        enabled = bool(channel_limit_switch.value)
        # The channel count is moot when the cap is off.
        channel_limit_input.set_enabled(enabled)
        try:
            limit = (
                int(channel_limit_input.value)
                if channel_limit_input.value is not None
                else DEFAULT_CHANNEL_LIMIT
            )
            samples = (
                int(samples_input.value)
                if samples_input.value is not None
                else DEFAULT_SAMPLES_PER_CHANNEL
            )
        except (TypeError, ValueError):
            return
        flushed = session.set_watch_performance(
            channel_limit_enabled=enabled,
            channel_limit=limit,
            samples_per_channel=samples,
            average_patches=bool(average_patches_switch.value),
        )
        if flushed:
            ui.notify("Watch statistics flushed", type="info")

    def apply_debug() -> None:
        """Push the error-check controls to the session (auto-applied)."""
        if loading:
            return
        try:
            interval = (
                int(debug_interval.value) if debug_interval.value is not None else 100
            )
            percent = (
                float(debug_threshold.value)
                if debug_threshold.value is not None
                else 10.0
            )
        except (TypeError, ValueError):
            return
        session.set_debug_settings(
            enabled=bool(debug_enable.value),
            interval=interval,
            check_nan_inf=bool(debug_nan_inf.value),
            check_under_over=bool(debug_under_over.value),
            threshold=percent / 100.0,
        )

    def unpin(view: RecordedView) -> None:
        # An experiment recording pins the page's auto experiment so it
        # keeps re-running even after the page closes; ending the recording
        # puts it back on the page-heartbeat clock.
        if view.page == "experiment":
            session.unpin_auto_experiment(str(view.params.get("auto_key", "")))

    def add_view() -> None:
        view = record_view() if record_view is not None else None
        if view is None:
            ui.notify(
                "Nothing to record on this page yet", type="warning"
            )
            return
        if not session.recording.start(view):
            ui.notify("This view is already being recorded", type="warning")
            return
        if view.page == "experiment":
            session.pin_auto_experiment(str(view.params.get("auto_key", "")))
        ui.notify(f"Recording into {session.recording.directory}/")
        refresh_recording_lock()
        rebuild()

    # Rendering a frame can take seconds for a large view, so the snapshot
    # runs off the event loop like end/delete below — and its notify goes
    # through `_best_effort_ui_update`, since the dialog may be gone by then.
    async def take_snapshot() -> None:
        view = record_view() if record_view is not None else None
        if view is None:
            ui.notify("Nothing to capture on this page yet", type="warning")
            return
        message: str
        kind: Literal["positive", "negative", "warning"]
        try:
            paths = await asyncio.to_thread(
                session.recording.snapshot, view, session
            )
        except Exception as e:  # noqa: BLE001 — reported, like a frame error
            message, kind = f"Snapshot failed: {type(e).__name__}: {e}", "negative"
        else:
            message, kind = (
                ("Saved " + ", ".join(str(p) for p in paths), "positive")
                if paths
                else ("Nothing to capture in this view yet", "warning")
            )
        _best_effort_ui_update(lambda: ui.notify(message, type=kind))

    # End/delete run via `asyncio.to_thread`: finalizing the MP4 writers
    # (ffmpeg) takes a moment and may briefly wait for an in-flight frame
    # append — blocking the event loop here would starve the websocket
    # keepalive and drop the connection. The post-finalize UI refresh is
    # wrapped in `_best_effort_ui_update`: by the time the await returns the
    # recording is already saved/deleted, but the user may have closed the
    # dialog or navigated away, deleting this client's elements.
    async def end_view(key: str, view: RecordedView) -> None:
        paths = await asyncio.to_thread(session.recording.end, key)
        unpin(view)
        message = (
            "Saved " + ", ".join(str(p) for p in paths)
            if paths
            else "Recording ended before any frame was captured"
        )

        def apply() -> None:
            ui.notify(message)
            refresh_recording_lock()
            rebuild()

        _best_effort_ui_update(apply)

    async def delete_view(key: str, view: RecordedView) -> None:
        await asyncio.to_thread(session.recording.delete, key)
        unpin(view)

        def apply() -> None:
            refresh_recording_lock()
            rebuild()

        _best_effort_ui_update(apply)

    def rebuild() -> None:
        recording_section.clear()
        statuses = session.recording.statuses()
        current = record_view() if record_view is not None else None
        # Snapshot stays offered whatever the view is doing — a still costs
        # nothing and is just as useful mid-recording. "Record" is the one
        # that disappears once the view records, since it then carries a
        # "this view" entry in the list below; never both.
        current_recording = (
            current is not None and session.recording.is_recording(current.key)
        )
        with recording_section:
            if current is None:
                ui.label(
                    "Nothing capturable on this page yet — watch a layer or "
                    "run an experiment first."
                ).classes("text-sm text-slate-500 italic")
            else:
                with ui.row().classes("w-full items-center gap-2 no-wrap"):
                    ui.label(current.label).classes(
                        "text-sm font-medium truncate grow min-w-0"
                    )
                    ui.button(
                        "Snapshot",
                        icon="photo_camera",
                        on_click=take_snapshot,
                        color="grey-8",
                    ).props("dense size=sm no-caps").tooltip(
                        "Save this view as a PNG"
                    )
                    if not current_recording:
                        ui.button(
                            "Record",
                            icon="fiber_manual_record",
                            on_click=add_view,
                            color="red",
                        ).props("dense size=sm no-caps").tooltip(
                            "Record this view — one frame per update"
                        )
            if statuses:
                ui.label("Currently recording").classes(
                    "text-xs uppercase tracking-wider text-slate-400"
                )
                for status in statuses:
                    is_current = (
                        current is not None and status.view.key == current.key
                    )
                    with ui.row().classes("w-full items-center gap-2 no-wrap"):
                        with ui.column().classes("grow min-w-0 gap-0"):
                            label = status.view.label + (
                                "  (this view)" if is_current else ""
                            )
                            ui.label(label).classes(
                                "text-sm font-medium truncate"
                            )
                            note = f"{status.frames} frame" + (
                                "" if status.frames == 1 else "s"
                            )
                            if status.error is not None:
                                note += f" · {status.error}"
                            ui.label(note).classes(
                                "text-xs "
                                + (
                                    "text-red-600"
                                    if status.error is not None
                                    else "text-slate-500"
                                )
                            )
                        ui.button(
                            "Save & Finish",
                            on_click=lambda s=status: end_view(s.view.key, s.view),
                            color="grey-8",
                        ).props("dense size=sm no-caps").tooltip(
                            "Finish this view's MP4 file(s)"
                        )
                        ui.button(
                            "Delete",
                            on_click=lambda s=status: delete_view(
                                s.view.key, s.view
                            ),
                            color="red",
                        ).props("dense size=sm no-caps flat").tooltip(
                            "Discard this view's recording"
                        )
                ui.label(f"Files: {session.recording.directory}/").classes(
                    "text-xs text-slate-500 font-mono"
                )

    def refresh_recording_lock() -> None:
        locked = session.recording.count() > 0
        lock_note.set_visibility(locked)
        for control in (unit_select, n_input, phase_select):
            control.set_enabled(not locked)

    def open_dialog() -> None:
        nonlocal loading
        loading = True
        auto_run_switch.value = session.auto_run_experiments
        scope_select.value = str(session.stats_scope)
        perf = session.watch_performance
        channel_limit_switch.value = perf.channel_limit_enabled
        channel_limit_input.value = perf.channel_limit
        channel_limit_input.set_enabled(perf.channel_limit_enabled)
        samples_input.value = perf.samples_per_channel
        average_patches_switch.value = perf.average_patches
        freq = session.update_frequency
        unit_select.value = freq.unit
        n_input.value = freq.n
        # Re-read the phases seen so far before mapping the saved filter onto them.
        phase_select.set_options(
            [_ANY_PHASE] + session.schedule.phase_order,
            value=freq.phase if freq.phase is not None else _ANY_PHASE,
        )
        debug = session.debug_settings
        debug_enable.value = debug.enabled
        debug_interval.value = debug.interval
        debug_threshold.value = round(debug.threshold * 100, 4)
        debug_nan_inf.value = debug.check_nan_inf
        debug_under_over.value = debug.check_under_over
        loading = False
        sync_phase_visibility()
        error_label.text = ""
        refresh_recording_lock()
        rebuild()
        dialog.open()

    button = ui.button(icon="settings", on_click=open_dialog, color="slate-500").props(
        "dense size=md"
    )
    button.tooltip("Settings")
    with button:
        badge = ui.badge("").props("color=red floating")

    def refresh_badge() -> None:
        n = session.recording.count()
        badge.set_text(str(n))
        badge.set_visibility(n > 0)

    refresh_badge()
    ui.timer(0.5, refresh_badge)
    return button


def _add_error_banner(session: Session) -> None:
    """The full-width banners every page places directly under its top bar.

    Two of them, stacked: a red *error* strip once the training loop has died
    (`_add_lost_loop_banner`) above the yellow numerical *warning* below, so a
    run that tripped a check and then crashed shows both, worst first.
    """
    _add_lost_loop_banner(session)
    _add_debug_warning_banner(session)


def _lost_loop_summary(session: Session) -> str | None:
    """The one-line error-banner message for a dead training loop, or `None`.

    Shaped like `_debug_banner_summary`: what happened, what it was, where —
    with the position naming the batch the loop died on, since the traceback
    went to a terminal the user may no longer be looking at.
    """
    if not session.training_lost:
        return None
    error = session.training_error
    what = error if error is not None else "it returned without calling close()"
    position = session.live_position
    where = "" if position is None else f" — at {format_position(position)}"
    return f"Training loop exited — {what}{where}"


def _add_lost_loop_banner(session: Session) -> None:
    """Full-width red banner shown once the training loop has died.

    Red rather than the numerical check's amber, because this one is not a
    warning to weigh: the thread is gone, every run control is grayed out
    behind it, and no amount of resuming brings it back. It offers no
    "Silence" — there is no check to turn off, only a run to restart — and
    hides again only if a fresh loop starts driving the same session.
    """
    container = ui.element("div").classes("w-full shrink-0")
    container.set_visibility(False)
    shown: dict[str, str | None] = {"key": None}

    def rebuild(summary: str) -> None:
        container.clear()
        with container:
            with ui.row().classes(
                "w-full bg-red-600 text-white items-center gap-3 px-4 py-2 "
                "no-wrap shadow-md"
            ):
                ui.icon("error").classes("text-2xl shrink-0")
                ui.label(summary).classes(
                    "text-sm font-medium grow min-w-0 truncate"
                ).tooltip(
                    f"{lost_loop_reason(session.training_error)} Everything "
                    "already captured stays inspectable."
                )

    def refresh() -> None:
        summary = _lost_loop_summary(session)
        if summary == shown["key"]:
            return
        shown["key"] = summary
        if summary is not None:
            rebuild(summary)
        container.set_visibility(summary is not None)

    refresh()
    ui.timer(0.2, refresh)


def _add_debug_warning_banner(session: Session) -> None:
    """Full-width yellow banner shown while a numerical error is active.

    A 0.2 s timer polls `session.debug_error`: the banner rebuilds when the
    error identity changes (every detection / merge makes a fresh frozen
    record) and hides when it clears. It is a *warning* — training paused at
    the first issue, but resuming keeps the banner standing while later issues
    fold into it. Clicking the message opens the details dialog; "Silence
    warning" turns off the active checks and clears the banner.
    """
    container = ui.element("div").classes("w-full shrink-0")
    container.set_visibility(False)
    # `id(error)` of the currently-shown record, so the timer only rebuilds on
    # a genuine change (every detection / merge makes a fresh frozen record).
    shown: dict[str, int | None] = {"key": None}

    def silence(error: DebugError) -> None:
        _silence_warning(session, error)
        refresh()

    def rebuild(error: DebugError) -> None:
        container.clear()
        with container:
            with ui.row().classes(
                "w-full bg-amber-400 text-amber-950 items-center gap-3 px-4 "
                "py-2 no-wrap shadow-md"
            ):
                ui.icon("warning").classes("text-2xl shrink-0")
                message = ui.label(_debug_banner_summary(error)).classes(
                    "text-sm font-medium grow min-w-0 truncate cursor-pointer"
                )
                message.tooltip(_DEBUG_BANNER_TOOLTIP)
                message.on("click", lambda e=error: _open_debug_dialog(session, e))
                ui.button(
                    "Details", on_click=lambda e=error: _open_debug_dialog(session, e)
                ).props("dense size=sm flat color=grey-10 no-caps")
                ui.button(
                    "Silence warning",
                    on_click=lambda e=error: silence(e),
                ).props(
                    "dense size=sm outline color=grey-10 no-caps"
                ).tooltip(
                    "Turn off the numerical checks and dismiss this warning"
                )

    def refresh() -> None:
        error = session.debug_error
        if error is None:
            if shown["key"] is not None:
                shown["key"] = None
                container.set_visibility(False)
            return
        if id(error) != shown["key"]:
            shown["key"] = id(error)
            rebuild(error)
            container.set_visibility(True)

    refresh()
    ui.timer(0.2, refresh)


def _silence_warning(session: Session, error: DebugError) -> None:
    """Turn off every check category present in `error`, clearing the banner.

    Both "Silence warning" buttons (banner and dialog) route here: each active
    category is disabled via `Session.disable_debug_check`, which trims the
    matching reasons from the standing error until nothing remains and the
    banner disappears. The checks can be turned back on from the settings gear.
    """
    for category in debugger.categories_present(error):
        session.disable_debug_check(category)


def _open_debug_dialog(session: Session, error: DebugError) -> None:
    """The details dialog: explanation + per-layer table + Silence button.

    Built fresh on each open so the per-layer Watch/Stats actions reflect the
    current watched set (a Watch click reopens it). When the under/overflow
    check ran, the dialog also spells out the dtype-aware bands in real
    magnitudes (`_under_over_band_lines`).
    """
    cols = debugger.columns(error)
    # Watch actions only exist in the `watched` stats scope; in `all` every
    # layer already collects, and in `none` collection is paused, so both
    # get a plain Stats link.
    offer_watch = session.stats_scope is StatsScope.WATCHED
    watched = session.watched_layers
    with ui.dialog() as dialog, ui.card().classes(
        "min-w-[36rem] max-w-[56rem] p-6 gap-3"
    ):
        ui.label("Numerical issue detected").classes(
            "text-lg font-bold text-amber-700"
        )
        ui.label(_DEBUG_DESCRIPTION).classes("text-sm text-slate-600")
        ui.label(
            f"Reasons: {debugger.reasons_text(error)}  ·  "
            f"{format_position(error.position)}"
        ).classes("text-sm font-mono")

        if debugger.UNDER_OVER in error.checks_used:
            with ui.column().classes(
                "w-full gap-1 bg-amber-50 border border-amber-200 rounded "
                "px-3 py-2"
            ):
                ui.label(_DEBUG_UNDER_OVER_INTRO).classes(
                    "text-xs text-slate-600"
                )
                for line in _under_over_band_lines(error):
                    ui.label(line).classes("text-xs font-mono text-amber-800")
                ui.label(_DEBUG_UNDER_OVER_TIP).classes(
                    "text-xs text-slate-700 font-medium mt-1"
                )

        with ui.element("div").classes(
            "w-full overflow-auto max-h-[24rem] border rounded"
        ):
            with ui.row().classes(
                "w-full items-center gap-x-6 px-3 py-1 no-wrap bg-slate-100 "
                "text-xs font-semibold uppercase tracking-wider text-slate-500"
            ):
                ui.label("Layer").classes("grow min-w-0")
                for col in cols:
                    ui.label(debugger.REASON_LABELS[col]).classes(
                        "w-24 text-right"
                    )
                ui.label("").classes("w-40 shrink-0")
            for report in error.layers:
                with ui.row().classes(
                    "w-full items-center gap-x-6 px-3 py-1 no-wrap border-t"
                ):
                    ui.label(report.layer).classes(
                        "grow min-w-0 font-mono text-sm truncate"
                    )
                    for col in cols:
                        ui.label(_debug_pct(getattr(report, col))).classes(
                            "w-24 text-right font-mono text-sm"
                        )
                    with ui.element("div").classes(
                        "w-40 shrink-0 flex justify-end"
                    ):
                        _debug_action_button(
                            session,
                            dialog,
                            error,
                            report,
                            watched,
                            offer_watch=offer_watch,
                        )

        ui.label(_DEBUG_WATCH_NOTE).classes("text-xs text-slate-500")
        with ui.row().classes("w-full justify-end gap-2"):

            def silence() -> None:
                _silence_warning(session, error)
                dialog.close()

            ui.button(
                "Silence warning", on_click=silence, color="amber-7"
            ).props("flat no-caps").tooltip(
                "Turn off the numerical checks and dismiss this warning"
            )
            ui.button("Close", on_click=dialog.close).props("flat")
    dialog.open()


def _debug_action_button(
    session: Session,
    dialog: ui.dialog,
    error: DebugError,
    report: LayerReport,
    watched: frozenset[str],
    *,
    offer_watch: bool,
) -> None:
    """Per-row actions: a Stats link, plus a Watch button when not yet watched.

    The stats histograms need a layer that collects (the watch accumulators
    feed them). A layer already collecting just gets a Stats link; with
    `offer_watch` (the `watched` stats scope), an unwatched layer gets both:
    Watch (start collecting and stay in the dialog) and Stats (start
    collecting *and* jump to the stats view — via the `watch=1` param the
    stats page honors on open, which keeps the button a real anchor so
    middle-click opens a new tab). The stats page pre-checks its
    "Show subnormal/overflow" band from the active issue itself
    (`stats_page._should_show_bands`), so either route opens with the band
    marked; the histogram fills in once a few batches have stepped.
    """
    href = f"/stats?layer={quote(report.layer)}"
    with ui.row().classes("gap-1 no-wrap items-center"):
        if offer_watch and report.layer not in watched:

            def watch_layer() -> None:
                session.watch(report.layer)
                ui.notify(
                    f"Watching {report.layer} — let training step a few "
                    "batches, then open the stats view.",
                    type="positive",
                )
                dialog.close()
                _open_debug_dialog(session, error)

            ui.button("Watch", on_click=watch_layer).props(
                "dense size=sm flat no-caps color=primary"
            ).tooltip("Start collecting this layer's stats and stay here")

            ui.button("Stats").props(
                f'href="{href}&watch=1" dense size=sm flat no-caps color=primary'
            ).tooltip("Start collecting and open this layer's stats")
        else:
            ui.button("Stats").props(
                f'href="{href}" dense size=sm flat no-caps color=primary'
            ).tooltip("Open this layer's stats")


def _best_effort_ui_update(update: Callable[[], None]) -> None:
    """Run a deferred UI update, tolerating a client torn down mid-`await`.

    `end_view` / `delete_view` await ffmpeg finalization off-thread; if the
    settings dialog's page is closed or navigated away during that wait, this
    client's elements are gone and NiceGUI raises `RuntimeError` the moment the
    update touches them (`ui.notify`, the controls, the recording list). The
    recording is already saved/deleted by then, so the toast and refresh are
    moot — swallow that teardown error rather than surface it as an unhandled
    background-task exception (which then also trips NiceGUI's own handler).
    """
    try:
        update()
    except RuntimeError:
        pass


def _summarize_epoch_ranges(epochs: list[int]) -> str:
    """Compact "0–2, 5, 7–49" rendering of a sorted epoch list."""
    parts: list[str] = []
    start = prev = epochs[0]
    for e in epochs[1:]:
        if e == prev + 1:
            prev = e
            continue
        parts.append(f"{start}–{prev}" if prev > start else f"{start}")
        start = prev = e
    parts.append(f"{start}–{prev}" if prev > start else f"{start}")
    return ", ".join(parts)


def _time_travel_default_index(cached: list[int], current_epoch: int | None) -> int:
    """Slider index (into the sorted `cached` list) to preselect on dialog open.

    The preselected epoch is where training currently is — the closest
    cached epoch at or before `current_epoch` — not simply the last cached
    one: after a backwards jump the checkpoints of later, abandoned epochs
    stay on disk, so the last cached epoch can lie in the future. Falls back
    to the last cached epoch before the first batch publishes a position.
    """
    if current_epoch is None:
        return len(cached) - 1
    return max(0, bisect_right(cached, current_epoch) - 1)


def _int_value(element: ui.number, default: int = 0) -> int:
    """A `ui.number`'s value as a non-negative int (it stores floats / None)."""
    try:
        return max(0, int(element.value)) if element.value is not None else default
    except (TypeError, ValueError):
        return default


def _build_step_until_custom_dialog(session: Session) -> ui.dialog:
    """Run-until-position dialog over a possibly-still-unknown schedule.

    The schedule is discovered lazily, so the target is addressed by *phase
    index* rather than name: pick an epoch (slider over the known total), a
    phase number (its name is shown once observed), and a batch number. Phases
    or batch counts beyond what has been seen are allowed with a warning —
    training matches the target if it ever reaches it, and runs to the end
    otherwise.
    """
    with ui.dialog() as dialog, ui.card().classes("min-w-96 p-6 gap-3"):
        ui.label("Step until custom").classes("text-lg font-bold")

        with ui.row().classes("w-full items-center gap-4 no-wrap"):
            epoch_label = ui.label("epoch 0").classes("font-mono text-sm w-20 shrink-0")
            epoch_slider = ui.slider(
                min=0,
                max=0,
                step=1,
                value=0,
                on_change=lambda e: epoch_label.set_text(f"epoch {int(e.value)}"),
            ).classes("grow")

        with ui.row().classes("w-full gap-4 items-start no-wrap"):
            phase_input = ui.number(
                label="Phase #", value=0, min=0, step=1, format="%d"
            ).classes("flex-1")
            batch_input = ui.number(
                label="Batch", value=0, min=0, step=1, format="%d"
            ).classes("flex-1")
        phase_hint = ui.label("").classes("text-xs min-h-4")
        batch_hint = ui.label("").classes("text-xs min-h-4")
        error_label = ui.label("").classes("text-red-500 text-sm min-h-4")

        def refresh_hints() -> None:
            order = session.schedule.phase_order
            pidx = _int_value(phase_input)
            if pidx < len(order):
                name = order[pidx]
                phase_hint.text = f"phase {pidx}: {name!r}"
                phase_hint.classes(replace="text-xs min-h-4 text-slate-600")
                count = session.schedule.phase_count(name)
            else:
                phase_hint.text = (
                    f"phase {pidx} not observed yet — matched once training "
                    "reaches it"
                )
                phase_hint.classes(replace="text-xs min-h-4 text-amber-700")
                count = None
            bidx = _int_value(batch_input)
            if count is None:
                batch_hint.text = ""
            elif bidx >= count:
                batch_hint.text = (
                    f"phase has {count} batches so far — the target may differ "
                    "if the dataset size changes"
                )
                batch_hint.classes(replace="text-xs min-h-4 text-amber-700")
            else:
                batch_hint.text = f"{count} batches in this phase"
                batch_hint.classes(replace="text-xs min-h-4 text-slate-600")

        phase_input.on_value_change(lambda: refresh_hints())
        batch_input.on_value_change(lambda: refresh_hints())

        def submit() -> None:
            epoch = int(epoch_slider.value) if epoch_slider.value is not None else 0
            phase_index = _int_value(phase_input)
            batch_idx = _int_value(batch_input)
            error = _validate_step_until_target(
                live_position=session.live_position,
                snapshot=session.snapshot,
                phase_order=session.schedule.phase_order,
                phase_index=phase_index,
                epoch=epoch,
                batch_idx=batch_idx,
            )
            if error is not None:
                error_label.text = error
                return
            error_label.text = ""
            session.step_until_position(
                phase_index=phase_index, epoch=epoch, batch_idx=batch_idx
            )
            dialog.close()

        with ui.row():
            ui.button("Cancel", on_click=dialog.close)
            ui.button("Step", on_click=submit)

        def on_show() -> None:
            total = session.schedule.epochs
            epoch_slider._props["max"] = (total - 1) if total else 0
            epoch_slider.update()
            order = session.schedule.phase_order
            position = _current_position(session.live_position, session.snapshot)
            if position is not None:
                epoch_slider.value = position.epoch
                phase_input.value = (
                    order.index(position.phase) if position.phase in order else 0
                )
                batch_input.value = position.batch_idx
            epoch_label.set_text(f"epoch {int(epoch_slider.value or 0)}")
            error_label.text = ""
            refresh_hints()

        dialog.on("before-show", on_show)

    return dialog


def _current_position(
    live_position: BatchPosition | None, snapshot: BatchSnapshot | None
) -> BatchPosition | None:
    """Where training currently is, for dialogs that prefill from it.

    `live_position` tracks every batch even in modes where
    `snapshot.position` is frozen between boundaries, so it wins; the
    snapshot covers the brief window before the first batch publishes a
    live position.
    """
    if live_position is not None:
        return live_position
    return snapshot.position if snapshot is not None else None


def _validate_step_until_target(
    *,
    live_position: BatchPosition | None,
    snapshot: BatchSnapshot | None,
    phase_order: list[str],
    phase_index: int,
    epoch: int,
    batch_idx: int,
) -> str | None:
    """The one hard requirement on a custom-step target: it must lie ahead.

    Unknown phases / over-large batch indices are *not* rejected here (the
    dialog surfaces those as warnings) — they simply match if training reaches
    them. But the target must be after the current position, since stepping
    only moves forward (going back is time travel). The rank compares against
    the live position the dialog prefills from; `snapshot.position` is stale
    mid-step_epoch/step_run/detach, so a target between it and the live
    position would pass yet never be hit.
    """
    current = live_position if live_position is not None else (
        snapshot.position if snapshot is not None else None
    )
    if current is None:
        return None
    cur_index = (
        phase_order.index(current.phase) if current.phase in phase_order else 0
    )
    target_rank = (epoch, phase_index, batch_idx)
    current_rank = (current.epoch, cur_index, current.batch_idx)
    if target_rank <= current_rank:
        return "Target must be after the current position"
    return None
