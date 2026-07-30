"""The shared top-bar/step controls and the dialogs they open."""

from __future__ import annotations

import asyncio
import base64
import json
from bisect import bisect_right
from collections.abc import Callable
from dataclasses import dataclass
from functools import lru_cache
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
from nansense.session import BatchSnapshot, Session, StatsScope
from nansense.watch import DEFAULT_CHANNEL_LIMIT


_TOP_BAR_CLASSES: str = (
    "w-full items-center gap-x-3 gap-y-0 px-3 py-2 shrink-0 "
    "border-b-2 border-slate-300 bg-slate-100 shadow-sm z-10"
)

_REPO_URL: str = "https://github.com/kongaskristjan/nansense"
_STAR_TOOLTIP: str = (
    "Like NaNsense? A GitHub ★ star ★ means a lot and keeps me hacking."
)
_DEBUG_DESCRIPTION: str = (
    "A numerical issue was detected: NaN/±Inf values, or gradients whose "
    "magnitude collapsed into a precision-losing under/overflow range. "
    "Training paused at the first issue; resuming keeps running and folds any "
    "further issues into this warning. Click for the affected layers."
)
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
    """The NaNsense brand mark at the far-right end of the top bar.

    Sits last in every page's top bar — after the right-aligned controls — as a
    quiet star call-to-action. Rendered as a native link opening in a new tab
    (so middle/ctrl-click works, like the nav buttons) with a hover tooltip
    nudging a repo star. Returned so a top bar with no `ml-auto` control of its
    own can right-align it directly.
    """
    link = ui.link(target=_REPO_URL, new_tab=True).classes(
        "shrink-0 flex items-center"
    )
    with link:
        ui.image(_logo_data_uri()).classes("h-7 w-7").props("no-spinner")
        # Right-anchored so the tooltip grows leftward and stays on-screen at
        # the top bar's right edge.
        ui.tooltip(_STAR_TOOLTIP).props('anchor="bottom right" self="top right"')
    return link


@dataclass(frozen=True)
class _ShareTarget:
    """One thing the share dialog can hand out: a link plus its share text."""

    label: str
    url: str
    title: str


# What the share dialog offers. The playground URL pins the `dev` version:
# unlike the docs site's own pages, the app can't derive the live docs
# version from its location (it runs on localhost or the HF Space), and
# `dev` is the only version the site currently publishes. The library link is
# the version-less site root, which redirects to the default published
# version — so it stays current when a release later takes over `latest`.
_SHARE_TARGETS: dict[str, _ShareTarget] = {
    "playground": _ShareTarget(
        label="Playground",
        url="https://kongaskristjan.github.io/nansense/dev/playground/",
        title="NaNsense playground — a live PyTorch training run to poke around in",
    ),
    "library": _ShareTarget(
        label="Library",
        url="https://kongaskristjan.github.io/nansense/",
        title="NaNsense — a PyTorch debugger: pause training, look inside every layer",
    ),
}


def _share_platform_links(url: str, title: str) -> list[tuple[str, str]]:
    """(platform label, share-intent href) pairs for one target link.

    Each href opens the platform's share/submit composer prefilled with the
    target URL (and title, where the platform takes one).
    """
    u = quote(url, safe="")
    t = quote(title, safe="")
    return [
        ("X", f"https://x.com/intent/post?text={t}&url={u}"),
        ("Facebook", f"https://www.facebook.com/sharer/sharer.php?u={u}"),
        ("LinkedIn", f"https://www.linkedin.com/sharing/share-offsite/?url={u}"),
        ("Reddit", f"https://www.reddit.com/submit?url={u}&title={t}"),
        ("Hacker News", f"https://news.ycombinator.com/submitlink?u={u}&t={t}"),
        ("Email", f"mailto:?subject={t}&body={u}"),
    ]


# Brand glyphs for the share-intent buttons, inlined as SVG path data so the
# app stays self-contained (NiceGUI bundles no brand-icon font; Email uses the
# bundled Material "mail" icon instead). X / Facebook / Reddit / Y Combinator
# come from Simple Icons (CC0 1.0 — public domain). LinkedIn comes from Font Awesome
# Free 6.7.2 by @fontawesome — https://fontawesome.com, License:
# https://fontawesome.com/license/free (Icons: CC BY 4.0), Copyright 2024
# Fonticons, Inc.; this comment carries the required attribution. The glyphs
# remain their owners' trademarks, used nominatively: each button does nothing
# but link to that platform's own share composer.
_PLATFORM_ICONS: dict[str, tuple[str, str]] = {  # label -> (viewBox, path d)
    "X": (
        "0 0 24 24",
        "M14.234 10.162 22.977 0h-2.072l-7.591 8.824L7.251 0H.258l9.168 "
        "13.343L.258 24H2.33l8.016-9.318L16.749 24h6.993zm-2.837 "
        "3.299-.929-1.329L3.076 1.56h3.182l5.965 8.532.929 1.329 7.754 "
        "11.09h-3.182z",
    ),
    "Facebook": (
        "0 0 24 24",
        "M9.101 23.691v-7.98H6.627v-3.667h2.474v-1.58c0-4.085 1.848-5.978 "
        "5.858-5.978.401 0 .955.042 1.468.103a8.68 8.68 0 0 1 1.141.195v3.325a8.623 "
        "8.623 0 0 0-.653-.036 26.805 26.805 0 0 0-.733-.009c-.707 0-1.259.096-1.675"
        ".309a1.686 1.686 0 0 0-.679.622c-.258.42-.374.995-.374 1.752v1.297h3.919l"
        "-.386 2.103-.287 1.564h-3.246v8.245C19.396 23.238 24 18.179 24 12.044c0"
        "-6.627-5.373-12-12-12s-12 5.373-12 12c0 5.628 3.874 10.35 9.101 11.647Z",
    ),
    "LinkedIn": (
        "0 0 448 512",
        "M100.28 448H7.4V148.9h92.88zM53.79 108.1C24.09 108.1 0 83.5 0 "
        "53.8a53.79 53.79 0 0 1 107.58 0c0 29.7-24.1 54.3-53.79 54.3zM447.9 "
        "448h-92.68V302.4c0-34.7-.7-79.2-48.29-79.2-48.29 0-55.69 37.7-55.69 "
        "76.7V448h-92.78V148.9h89.08v40.8h1.3c12.4-23.5 42.69-48.3 87.88-48.3 "
        "94 0 111.28 61.9 111.28 142.3V448z",
    ),
    "Reddit": (
        "0 0 24 24",
        "M12 0C5.373 0 0 5.373 0 12c0 3.314 1.343 6.314 3.515 8.485l-2.286 "
        "2.286C.775 23.225 1.097 24 1.738 24H12c6.627 0 12-5.373 "
        "12-12S18.627 0 12 0Zm4.388 3.199c1.104 0 1.999.895 1.999 1.999 0 "
        "1.105-.895 2-1.999 2-.946 0-1.739-.657-1.947-1.539v.002c-1.147.162"
        "-2.032 1.15-2.032 2.341v.007c1.776.067 3.4.567 4.686 1.363.473-.363 "
        "1.064-.58 1.707-.58 1.547 0 2.802 1.254 2.802 2.802 0 1.117-.655 "
        "2.081-1.601 2.531-.088 3.256-3.637 5.876-7.997 5.876-4.361 0-7.905"
        "-2.617-7.998-5.87-.954-.447-1.614-1.415-1.614-2.538 0-1.548 1.255"
        "-2.802 2.803-2.802.645 0 1.239.218 1.712.585 1.275-.79 2.881-1.291 "
        "4.64-1.365v-.01c0-1.663 1.263-3.034 2.88-3.207.188-.911.993-1.595 "
        "1.959-1.595Zm-8.085 8.376c-.784 0-1.459.78-1.506 1.797-.047 "
        "1.016.64 1.429 1.426 1.429.786 0 1.371-.369 1.418-1.385.047-1.017"
        "-.553-1.841-1.338-1.841Zm7.406 0c-.786 0-1.385.824-1.338 1.841.047 "
        "1.017.634 1.385 1.418 1.385.785 0 1.473-.413 1.426-1.429-.046-1.017"
        "-.721-1.797-1.506-1.797Zm-3.703 4.013c-.974 0-1.907.048-2.77.135"
        "-.147.015-.241.168-.183.305.483 1.154 1.622 1.964 2.953 1.964 1.33 "
        "0 2.47-.81 2.953-1.964.057-.137-.037-.29-.184-.305-.863-.087-1.795"
        "-.135-2.769-.135Z",
    ),
    "Hacker News": (  # the Y Combinator mark, HN's standard share glyph
        "0 0 24 24",
        "M0 24V0h24v24H0zM6.951 5.896l4.112 7.708v5.064h1.583v-4.972l4.148"
        "-7.799h-1.749l-2.457 4.875c-.372.745-.688 1.434-.688 1.434s-.297"
        "-.708-.651-1.434L8.831 5.896h-1.88z",
    ),
}


def _platform_icon_html(label: str) -> str | None:
    """The 18px `currentColor` SVG for a platform button, `None` for Email.

    Rendered as an inline SVG child of the button (not an `img:` data URI) so
    `fill="currentColor"` picks up the button's text color.
    """
    spec = _PLATFORM_ICONS.get(label)
    if spec is None:
        return None
    viewbox, path = spec
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="{viewbox}" '
        'width="18" height="18" fill="currentColor" aria-hidden="true">'
        f'<path d="{path}"/></svg>'
    )


def _add_share_button() -> None:
    """The share icon just left of the logo, and the share dialog it opens.

    A flat icon-only button, quieter than the working controls beside it, on
    every page — the hosted playground included (handing out links is exactly
    what a public demo is for; the docs page delegates `clipboard-write` to
    the app iframe). The dialog is three captioned sections: *what* to share
    (a full-width segmented toggle over `_SHARE_TARGETS`), the link itself (a
    one-line pill with an inline copy button), and *where* (one share-intent
    icon per platform — brand glyphs from `_PLATFORM_ICONS`, hover for the
    name). The card is fixed-width and the pill single-line, so switching
    targets never resizes the dialog. The link/platform sections are rebuilt
    on every target switch so each anchor's href — and the copy handler's URL
    — is baked in for the current pick; the clipboard write runs client-side
    in the `js_handler` (then `emit`s so Python can toast), since a server
    round-trip would drop the user gesture the Clipboard API needs.
    """
    caption_classes = "text-xs uppercase tracking-wider text-slate-400"
    with ui.dialog() as dialog, ui.card().classes("w-[32rem] max-w-full p-6 gap-4"):
        ui.label("Share NaNsense").classes("text-lg font-bold")
        with ui.column().classes("w-full gap-1"):
            ui.label("What to share").classes(caption_classes)
            ui.toggle(
                {key: t.label for key, t in _SHARE_TARGETS.items()},
                value="playground",
                on_change=lambda e: rebuild(str(e.value)),
            ).classes("w-full").props('spread no-caps padding="xs lg"')
        content = ui.column().classes("w-full gap-4")
        with ui.row().classes("w-full justify-end"):
            ui.button("Close", on_click=dialog.close).props("flat")

    def rebuild(key: str) -> None:
        target = _SHARE_TARGETS[key]
        content.clear()
        with content:
            with ui.column().classes("w-full gap-1"):
                ui.label("Link").classes(caption_classes)
                with ui.row().classes(
                    "w-full items-center no-wrap gap-0.5 bg-slate-100 rounded "
                    "pl-2 pr-1 py-1"
                ):
                    # 11px mono lets the longest URL (the playground's) fit the
                    # 32rem card on one line; `truncate` is the safety net.
                    ui.label(target.url).classes(
                        "grow min-w-0 truncate text-[11px] font-mono text-slate-600"
                    )
                    copy = ui.button(icon="content_copy", color="slate-500").props(
                        'flat round dense size=sm aria-label="Copy link"'
                    )
                    copy.tooltip("Copy link")
                    copy.on(
                        "click",
                        lambda: ui.notify("Link copied to clipboard"),
                        js_handler=(
                            "(...args) => { if (navigator.clipboard) "
                            f"navigator.clipboard.writeText({json.dumps(target.url)}); "
                            "emit(...args); }"
                        ),
                    )
            with ui.column().classes("w-full gap-1"):
                ui.label("Share on").classes(caption_classes)
                with ui.row().classes("gap-2 items-center"):
                    for label, href in _share_platform_links(
                        target.url, target.title
                    ):
                        # Real anchors opening a new tab, so the platform's
                        # composer never replaces the app (nor the docs page
                        # embedding it).
                        hint = (
                            "Share via email"
                            if label == "Email"
                            else f"Share on {label}"
                        )
                        props = (
                            f'href="{href}" target="_blank" unelevated round '
                            f'dense size=md text-color=slate-700 '
                            f'aria-label="{hint}"'
                        )
                        icon_html = _platform_icon_html(label)
                        if icon_html is None:
                            button = ui.button(icon="mail", color="slate-200").props(
                                props
                            )
                        else:
                            button = ui.button(color="slate-200").props(props)
                            with button:
                                ui.html(icon_html)
                        button.tooltip(hint)

    rebuild("playground")

    button = ui.button(icon="share", on_click=dialog.open, color="slate-500").props(
        "dense size=md flat"
    )
    button.tooltip("Share NaNsense — the playground or the library")


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
    button.tooltip("Show a quick tour of this view")


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
    ).props("dense size=md").tooltip(
        "Update the views from the next training batch (use while training)"
    )


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
            "This is a shared, hosted playground: training sits paused at a "
            "trained checkpoint and stepping, time travel, and input "
            "pinning/perturbation are disabled. Show layers, browse stats, "
            "and run experiments freely."
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
            .tooltip("Run to the last batch of training, then pause")
        )
        with ui.dropdown_button(
            "Step Batch",
            on_click=step_batch,
            split=True,
            auto_close=True,
            color="orange",
        ).props("dense size=md").tooltip("Advance one batch"):
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
            .tooltip("Pause at next batch")
        )
        _add_time_travel_button(session)
    _add_position_label(session)
    # Run is grayed while training advances (started by Run or any Step), Stop
    # while it sits paused — a 0.2s timer toggles both off
    # `session.is_running`. `None` forces the first apply.
    last_running: bool | None = None

    def refresh_buttons() -> None:
        nonlocal last_running
        running = session.is_running
        if running != last_running:
            last_running = running
            run_button.set_enabled(not running)
            stop_button.set_enabled(running)

    ui.timer(0.2, refresh_buttons)


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
                    ).tooltip(
                        "Run training to the end, checkpointing every epoch "
                        "start along the way"
                    )
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

    "Recording" offers a "Record" button for the page's own view (built by
    `record_view` with the page's *current* parameters, frozen for the
    recording's lifetime; None when the page's current state can't be
    recorded yet) plus the list of all active recordings, each one
    save-&-finishable (finalize the MP4) or deletable (discard it). The
    current page's view, while it records, appears only in that list (marked
    "this view") rather than twice. A red badge on the gear carries the
    active-recording count.

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
        button.tooltip("Settings (locked in this shared demo)")
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
                "Run experiments on the experiment page automatically — on "
                "open and on every parameter change — instead of clicking Run "
                "(Run is grayed out while on). Shared across all tabs."
            )
        )
        ui.separator()
        ui.label("Statistics collection").classes("text-lg font-bold")
        ui.label(
            "Which layers fold training batches into the running statistics "
            "(histograms, extreme-input patches, per-epoch graphs). With "
            "anything other than \"Watched layers\", showing or hiding a "
            "card on the main page is per-tab and never affects collection."
        ).classes("text-sm text-slate-600")
        scope_select = ui.select(
            _STATS_SCOPE_OPTIONS,
            label="Collect stats for",
            value=str(StatsScope.WATCHED),
            on_change=lambda: apply_stats_scope(),
        ).props("dense outlined").classes("w-64").tooltip(
            "\"No layers\" pauses collection but keeps everything collected "
            "so far; \"All layers\" collects for every layer on every batch "
            "— bound the memory with the channel limit below"
        )
        ui.separator()
        ui.label("Performance").classes("text-lg font-bold")
        ui.label(
            "How much NaNsense computes and stores while training runs — "
            "these trade visualization detail for GPU VRAM and overhead."
        ).classes("text-sm text-slate-600")
        ui.label("Watched-layer memory").classes("text-sm font-medium mt-1")
        ui.label(
            "Watched layers keep, per channel, a histogram and a gallery of "
            "extreme input patches. The patches store an input image per "
            "channel, so this is the dominant GPU VRAM cost; the layer-wide "
            "histogram always covers every channel."
        ).classes("text-xs text-slate-500")
        channel_limit_switch = ui.switch(
            "Limit recorded channels",
            on_change=lambda: apply_watch_performance(),
        ).props("dense").tooltip(
            "Keep per-channel data for only the first N channels of each "
            "watched layer. Turn off to record every channel (highest VRAM)."
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
                "Per-channel histograms and patch galleries are kept for "
                "this many channels"
            )
            samples_input = ui.number(
                label="Samples per channel",
                value=DEFAULT_SAMPLES_PER_CHANNEL,
                min=1,
                step=1,
                format="%d",
                on_change=lambda: apply_watch_performance(),
            ).props("dense outlined").classes("flex-1").tooltip(
                "Extreme input samples kept per channel, per ranking"
            )
        average_patches_switch = ui.switch(
            "Average-extreme patch galleries",
            on_change=lambda: apply_watch_performance(),
        ).props("dense").tooltip(
            "Also collect the max/min-average galleries, which keep a whole "
            "input image per slot (extra VRAM). The pixel-extreme galleries "
            "are always collected."
        )
        ui.label(
            "Changing the channel limit, samples per channel, or the average "
            "galleries flushes all collected statistics."
        ).classes("text-xs text-red-500")
        ui.label("Update frequency").classes("text-sm font-medium mt-1")
        ui.label(
            "How often all visualizations refresh while training runs. "
            "They additionally refresh whenever training stops."
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
            "Pause training automatically on numerical trouble: NaN/±Inf "
            "values, or gradients collapsing into a precision-losing "
            "under/overflow range. Checks run every nth batch on the compute "
            "device."
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
                "Trip when at least this share of a layer's summed |gradient| "
                "falls in the under/overflow band"
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
            "Each recorded view becomes an MP4 file — one frame per "
            "visualization update."
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
        # The current page's view shows up either as the "Record" button (not
        # yet recording) or — once recording — as a "this view" entry in the
        # list below, never both.
        current_recording = (
            current is not None and session.recording.is_recording(current.key)
        )
        with recording_section:
            if current is None:
                ui.label(
                    "Nothing recordable on this page yet — watch a layer or "
                    "run an experiment first."
                ).classes("text-sm text-slate-500 italic")
            elif not current_recording:
                with ui.row().classes("w-full items-center gap-2 no-wrap"):
                    ui.label(current.label).classes(
                        "text-sm font-medium truncate grow min-w-0"
                    )
                    ui.button(
                        "Record",
                        icon="fiber_manual_record",
                        on_click=add_view,
                        color="red",
                    ).props("dense size=sm no-caps").tooltip(
                        "Add this view to recording — one frame per "
                        "visualization update"
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
                            "Finalize this view's MP4 file(s)"
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
    button.tooltip("Settings — auto-run, performance, error checks, recording")
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
    """Full-width red banner shown while a numerical error is active.

    Placed by every page directly under its top bar. A 0.2 s timer polls
    `session.debug_error`: the banner rebuilds when the error identity changes
    (every detection / merge makes a fresh frozen record) and hides when it
    clears. It is a yellow *warning* — training paused at the first issue, but
    resuming keeps the banner standing while later issues fold into it.
    Clicking the message opens the details dialog; "Silence warning" turns off
    the active checks and clears the banner.
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
                message.tooltip(_DEBUG_DESCRIPTION)
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
                    "Turn off the active numerical checks and dismiss this "
                    "warning (re-enable them under the settings gear)"
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
                "Turn off the active numerical checks and dismiss this warning"
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
            ).tooltip("Collect this layer's gradient stats (stay here)")

            ui.button("Stats").props(
                f'href="{href}&watch=1" dense size=sm flat no-caps color=primary'
            ).tooltip("Watch this layer and open its stats view")
        else:
            ui.button("Stats").props(
                f'href="{href}" dense size=sm flat no-caps color=primary'
            ).tooltip("Open this layer's stats view (gradient histograms)")


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
