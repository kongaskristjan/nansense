"""JSON-shaped views of a `Session`, for the MCP server (`nansense.mcp_server`).

Pure translation: each function takes a live `Session` (or one of its frozen
snapshot dataclasses) and returns plain dicts / lists / strings a coding agent
can read. Nothing here imports `mcp`, performs I/O, or mutates the session — so
the shapes are unit-testable without the MCP SDK installed, and the server
module stays a thin registration layer over them.

Three conventions the agent-facing output depends on:

- **Non-finite floats survive as strings.** JSON has no NaN/Infinity literal,
  but "this gradient is Inf" is precisely the signal a debugging agent is
  looking for, so `_num` renders them `"nan"` / `"inf"` / `"-inf"` rather than
  `null` — which would be indistinguishable from "not measured".
- **Two positions, always both.** `live_position` is where training is *now*;
  `snapshot_position` is the batch the statistics actually describe. They
  diverge during `run` / `detach`, and an agent that conflates them reads stale
  numbers as current.
- **Empty and diverged are different.** The accumulators' scalars cover the
  finite population only, so an all-NaN layer arrives here looking identical
  to an unused one. `tensor_stats_view` separates the two — the distinction is
  the whole reason the debugger exists.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable
from typing import Any, Literal

import torch

from nansense import debugger, experiments, hyperparams, instruments
from nansense.schedule import BatchPosition, Schedule, format_position
from nansense.session import Session, StatsScope, lost_loop_reason
from nansense.watch import (
    LayerStatsSnapshot,
    TensorStatsSnapshot,
    bin_midpoint,
    dead_channel_indices,
    narrow_to_channel,
)

# Derived statistics are rounded to this many significant digits. The tail of
# a float64 repr is noise for a reader deciding whether a gradient collapsed,
# and it is noise the agent pays for by the token.
_SIGNIFICANT_DIGITS = 6


def _finite_num(value: float) -> float:
    """Round a known-finite number to a token-cheap precision."""
    return float(f"{value:.{_SIGNIFICANT_DIGITS}g}")


def _num(value: float | None) -> float | str | None:
    """A JSON-safe rendering of `value` (see the module docstring's contract)."""
    if value is None:
        return None
    number = float(value)
    if math.isnan(number):
        return "nan"
    if math.isinf(number):
        return "inf" if number > 0 else "-inf"
    return _finite_num(number)


def _median_num(value: float) -> float | str | None:
    """Round a histogram-derived median to two significant digits.

    Regular nonzero bins span a seventh of a decade, so extra digits reflect
    the bin geometry rather than additional measurement precision.
    """
    number = float(value)
    return _num(number) if not math.isfinite(number) else float(f"{number:.2g}")


def _dtype_name(dtype: torch.dtype | None) -> str | None:
    return None if dtype is None else str(dtype).removeprefix("torch.")


def position_view(
    position: BatchPosition | None, *, schedule: Schedule | None = None
) -> dict[str, Any] | None:
    """A position as both the UI's label text and its structured parts.

    `schedule` supplies the run totals, so `text` reads "epoch 0/2 | train
    batch 1/4" once they are known and stays bare while they are not (a lazy
    schedule learns a phase's batch count only at the end of its first epoch).
    """
    if position is None:
        return None
    total_epochs = None if schedule is None else schedule.epochs
    total_batches = None if schedule is None else schedule.phase_count(position.phase)
    return {
        "text": format_position(
            position, total_epochs=total_epochs, total_batches=total_batches
        ),
        "phase": position.phase,
        "epoch": position.epoch,
        "batch": position.batch_idx,
        "total_epochs": total_epochs,
        "total_batches_in_phase": total_batches,
    }


def _run_state(session: Session) -> str:
    """The state an agent branches on.

    `"not_started"` is called out separately from `"running"`: a session whose
    loop has not reached its first batch is *technically* not paused, but an
    agent told "running" would wait for a pause that nothing is driving toward.
    `"lost"` is the same trap after the fact — a training loop that raised
    still reads as running (the thread never reached the pause that clears
    it), so an agent told "running" would wait out every timeout it has.
    """
    if session.closed:
        return "finished"
    if session.training_lost:
        return "lost"
    if session.live_position is None:
        return "not_started"
    return "running" if session.is_running else "paused"


def _state_hint(session: Session) -> str | None:
    """One line on what this state means for the caller, or `None` when the
    state speaks for itself (a plain pause is the expected working state)."""
    if session.closed:
        return (
            "Training has finished. The last captured batch stays inspectable, "
            "but stepping and time travel no longer do anything."
        )
    if session.training_lost:
        return (
            f"{lost_loop_reason(session.training_error)} Nothing can be "
            "stepped, resumed or probed; the last captured batch stays "
            "inspectable. Fix the training script and start the run again."
        )
    if session.locked:
        return (
            "This session is locked (a shared demo): run controls and settings "
            "are disabled, inspection works normally."
        )
    if session.live_position is None:
        return (
            "The training loop has not reached its first batch yet. It pauses "
            "there on its own — no command needed."
        )
    if session.is_running:
        return (
            "Training is advancing, so the statistics below describe the last "
            "captured batch, not the live one. Call `pause` to stop on the next "
            "batch, or `refresh` to publish a fresh snapshot without pausing."
        )
    return None


def _warning_summary(error: debugger.DebugError | None) -> dict[str, Any] | None:
    """The numerical-error banner in brief; `get_debug_report` has the detail."""
    if error is None:
        return None
    return {
        "reasons": debugger.reasons_text(error),
        "first_detected_at": format_position(error.position),
        "affected_layers": [report.layer for report in error.layers],
        "detail": "Call get_debug_report for per-layer fractions and settings.",
    }


def status_view(session: Session) -> dict[str, Any]:
    """Where the run sits and what is being collected — the orientation call."""
    snapshot = session.snapshot
    schedule = session.schedule
    view: dict[str, Any] = {
        "state": _run_state(session),
        "mode": str(session.mode),
        "live_position": position_view(session.live_position, schedule=schedule),
        "snapshot_position": position_view(
            None if snapshot is None else snapshot.position, schedule=schedule
        ),
        "total_epochs": schedule.epochs,
        "phases": schedule.phase_order,
        # Bounds the `sample` argument every per-sample tool takes. Without it
        # an agent has no way to know how far it can page through a batch, and
        # an out-of-range index renders as "nothing here" rather than as a
        # mistake. `None` before the first capture, and it genuinely varies —
        # the last batch of an epoch is usually short.
        "batch_size": session.input_batch_size,
        "locked": session.locked,
        "layer_count": len(session.layer_names),
        "watched_layers": sorted(session.watched_layers),
        "stats_scope": str(session.stats_scope),
        "stats_collecting": session.stats_collecting,
        "numerical_warning": _warning_summary(session.debug_error),
    }
    # Only when there is one: a run that is still going has nothing to say
    # here, and the field would read as an error every caller has to dismiss.
    if session.training_lost and session.training_error is not None:
        view["training_error"] = session.training_error
    hint = _state_hint(session)
    if hint is not None:
        view["hint"] = hint
    return view


def brief_status_view(session: Session) -> dict[str, Any]:
    """Brief run status returned by control commands.

    Reports where the run is and any current numerical warning.
    `get_status` has the full view, including watched layers, phases and
    collection settings.
    """
    full = status_view(session)
    snapshot = full["snapshot_position"]
    view: dict[str, Any] = {
        "state": full["state"],
        "mode": full["mode"],
        "live_position": full["live_position"],
        "snapshot_position": None if snapshot is None else snapshot["text"],
    }
    warning = full["numerical_warning"]
    if warning is not None:
        view["numerical_warning"] = (
            f"{warning['reasons']} — see get_debug_report"
        )
    for key in ("training_error", "hint"):
        if key in full:
            view[key] = full[key]
    return view


def architecture_view(
    session: Session, *, mermaid: str | None = None
) -> dict[str, Any]:
    """The layer table, plus the Mermaid compute graph when `mermaid` is given.

    The caller passes the graph source rather than having it rebuilt here: it
    is fixed for the session's lifetime and `serve` already builds it once for
    the UI, so re-running `torch.fx.symbolic_trace` per tool call would be
    pure waste.
    """
    info = session.layer_info
    weights = session.layer_weights
    layers: list[dict[str, Any]] = []
    for name in session.layer_names:
        entry: dict[str, Any] = {"name": name}
        if info.get(name):
            entry["hyperparameters"] = info[name]
        if weights.get(name):
            entry["parameters"] = weights[name]
        layers.append(entry)
    view: dict[str, Any] = {
        "inputs": session.input_names,
        "fx_traced": session.fx_traced,
        "layers": layers,
    }
    if mermaid is not None:
        view["mermaid"] = mermaid
    return view


def histogram_view(hist: tuple[int, ...] | None) -> list[tuple[float, int]] | None:
    """Non-empty histogram bins as `[value, count]` pairs, or `None`.

    The underlying histogram is 211 fixed symmetric-log bins spanning ±1e-9 to
    ±1e6 with a zero band in the middle, and on a real layer nearly all of them
    are empty — so the pairs (bin midpoint, count) carry the same information
    at a fraction of the tokens. `None` when the bins were released, which
    happens to an older epoch once a newer one starts for its phase.
    """
    if hist is None:
        return None
    # Bin midpoints are always finite (the open-ended end bins report their
    # closed edge), so no non-finite rendering is needed here.
    return [
        (_finite_num(bin_midpoint(index)), count)
        for index, count in enumerate(hist)
        if count > 0
    ]


def channel_cap_note(tracked: int, total: int | None) -> str | None:
    """Why `tracked` per-channel rows may cover fewer than the `total` channels.

    With 16 tracked channels on a 64-channel layer, a dead-channel count of 14
    means 14 of those first 16; the remaining 48 channels are untracked.
    """
    if total is None or tracked >= total:
        return None
    return (
        f"Per-channel statistics (dead channels, channel histograms) cover only "
        f"the first {tracked} of {total} channels; raise `channel_limit` with "
        "set_watch_performance to track more (this clears collected statistics)."
    )


def _channel_view(
    stats: TensorStatsSnapshot, channel: int | None, total: int | None
) -> dict[str, Any]:
    """The per-channel keys: how many channels, which are dead, one's histogram.

    The accumulator keeps a histogram row per channel (capped at
    `channel_limit`), which is what the page's "Per channel" switch draws. Two
    things here are not derivable from the layer-wide view. `dead_channels`
    alone names a count, and a count cannot be drilled into — the *indices* are
    what `render_bin_samples(channel=...)` and `render_layer` need to go from
    "twelve channels are dead" to seeing one of them. And a channel's own
    distribution can be bimodal, saturated or collapsed while the layer-wide
    histogram it sums into looks unremarkable.

    `total` is the tensor's real channel count (its dim 1), when known, so a
    capped row count is never mistaken for the whole layer.
    """
    rows = stats.channel_hists
    if rows is None:
        return {}
    view: dict[str, Any] = {"channels_tracked": len(rows)}
    if total is not None:
        view["channel_count"] = total
    cap = channel_cap_note(len(rows), total)
    if cap is not None:
        view["channels_note"] = cap
    dead = dead_channel_indices(rows)
    if dead:
        # Bounded by `channel_limit` (16 by default), so the whole list ships
        # rather than a truncated sample the caller cannot act on.
        view["dead_channel_indices"] = dead
    if channel is not None:
        narrowed = narrow_to_channel(stats, channel)
        index = min(max(channel, 0), len(rows) - 1)
        view["channel"] = index
        view["channel_histogram"] = histogram_view(narrowed.hist)
        if index != channel:
            view["channel_note"] = (
                f"Channel {channel} is out of range; clamped to {index} "
                f"(this tensor tracks {len(rows)} channels)."
            )
    return view


def tensor_stats_view(
    stats: TensorStatsSnapshot,
    *,
    include_histogram: bool = False,
    shape: torch.Size | None = None,
    channel: int | None = None,
) -> dict[str, Any]:
    """One tensor stream's scalars (and optionally its histogram).

    The accumulator's scalars deliberately cover the **finite** population only
    — one NaN would otherwise poison `min`/`max` permanently, and one Inf would
    take `mean` with it — while the histogram counts every value. That split is
    invisible in the raw snapshot and dangerous to a reader: a fully diverged
    layer has `n == 0`, which reads as "nothing here" when the truth is
    "everything here is NaN or Inf". So the counts are reported separately, and
    a partly-diverged layer says outright which population its mean describes.

    `channel` narrows the *histogram* to one channel's row, the page's "Per
    channel" switch. The scalars stay tensor-wide either way — see
    `watch.narrow_to_channel` for why — so they are not silently re-scoped
    under the caller.
    """
    finite = stats.n
    channels = None if shape is None or len(shape) < 2 else int(shape[1])
    # `hist` counts every value, including the non-finite ones (NaN in the zero
    # band, ±Inf in the end bins); it is `None` only for a collapsed older
    # epoch, where the total is genuinely unknown.
    total = None if stats.hist is None else sum(stats.hist)
    view: dict[str, Any] = {}
    if shape is not None:
        view["shape"] = list(shape)
    view["count"] = finite if total is None else total
    non_finite = 0 if total is None else total - finite
    if non_finite > 0:
        view["finite_count"] = finite
        view["non_finite_count"] = non_finite

    if finite == 0:
        view["note"] = (
            f"every captured value is non-finite (NaN or ±Inf) — {non_finite} of them"
            if non_finite > 0
            else "no values captured for this tensor on this batch"
        )
        if include_histogram:
            view["histogram"] = histogram_view(stats.hist)
        view.update(_channel_view(stats, channel, channels))
        return view

    view["mean"] = _num(stats.mean)
    view["std"] = _num(stats.std)
    view["min"] = _num(stats.min)
    view["max"] = _num(stats.max)
    view["median"] = _median_num(stats.median)
    if non_finite > 0:
        view["note"] = (
            "mean, std, min and max describe only the finite values; the "
            "median comes from the histogram, which includes the non-finite ones"
        )
    dead = stats.dead_channel_count
    if dead is not None:
        view["dead_channels"] = dead
    dtype = _dtype_name(stats.dtype)
    if dtype is not None:
        view["dtype"] = dtype
    if include_histogram:
        view["histogram"] = histogram_view(stats.hist)
    view.update(_channel_view(stats, channel, channels))
    return view


def layer_stats_view(
    session: Session,
    *,
    layers: Iterable[str],
    include_histogram: bool = False,
    channel: int | None = None,
) -> dict[str, Any]:
    """Activation and gradient statistics for the last captured batch.

    Reads the published `BatchSnapshot`, so any layer works whether or not it
    is watched — watching only matters for the epoch-over-epoch series that
    `stats_history_view` reads.

    `channel` adds that channel's own histogram to every stream, alongside the
    dead-channel indices each one always reports.
    """
    snapshot = session.snapshot
    if snapshot is None:
        return {
            "error": "No batch has been captured yet.",
            "hint": (
                "A session pauses on its first batch; if training is running "
                "free, call `pause` or `refresh` first."
            ),
        }
    known = set(session.layer_names)
    requested = list(dict.fromkeys(layers))
    resolved = [name for name in requested if name in known]
    unknown = [name for name in requested if name not in known]

    stats = session.current_batch_stats(layers=resolved, include_patches=False)
    position = snapshot.position
    entries: list[dict[str, Any]] = []
    no_data: list[str] = []
    for name in resolved:
        layer_stats = stats.stats.get((name, position.phase, position.epoch))
        if layer_stats is None:
            no_data.append(name)
            continue
        activation = snapshot.activations.get(name)
        gradient = snapshot.activation_gradients.get(name)
        entry: dict[str, Any] = {
            "layer": name,
            "activations": tensor_stats_view(
                layer_stats.activations,
                include_histogram=include_histogram,
                shape=None if activation is None else activation.shape,
                channel=channel,
            ),
            "gradients": tensor_stats_view(
                layer_stats.gradients,
                include_histogram=include_histogram,
                shape=None if gradient is None else gradient.shape,
                channel=channel,
            ),
        }
        custom = _custom_summaries(snapshot.custom_activations.get(name))
        if custom:
            entry["custom"] = custom
        entries.append(entry)

    view: dict[str, Any] = {
        "position": position_view(position, schedule=session.schedule),
        "layers": entries,
    }
    if include_histogram or channel is not None:
        view["histogram_format"] = (
            "[value, count] pairs over signed-log bins; value is the bin midpoint"
        )
    if unknown:
        view["unknown_layers"] = unknown
        view["hint"] = "Call get_architecture for the valid layer names."
    if no_data:
        view["layers_without_data"] = no_data
    return view


def _abs_max(stats: TensorStatsSnapshot) -> float | None:
    return max(abs(stats.min), abs(stats.max)) if stats.n > 0 else None


def _dead_fraction(stats: TensorStatsSnapshot) -> float | None:
    dead, rows = stats.dead_channel_count, stats.channel_hists
    return None if dead is None or not rows else dead / len(rows)


def _non_finite_fraction(stats: TensorStatsSnapshot) -> float | None:
    total = None if stats.hist is None else sum(stats.hist)
    return (total - stats.n) / total if total else None


RankMetric = Literal[
    "gradient_std",
    "gradient_abs_max",
    "gradient_non_finite_fraction",
    "activation_std",
    "activation_abs_max",
    "activation_non_finite_fraction",
    "dead_channel_fraction",
]

# Each ranking metric: the stream it reads, and the scalar it ranks by
# (`None` when the stream has nothing to measure, e.g. no backward pass).
_RANK_METRICS: dict[str, tuple[str, Callable[[TensorStatsSnapshot], float | None]]] = {
    "gradient_std": ("gradients", lambda s: s.std if s.n > 0 else None),
    "gradient_abs_max": ("gradients", _abs_max),
    "gradient_non_finite_fraction": ("gradients", _non_finite_fraction),
    "activation_std": ("activations", lambda s: s.std if s.n > 0 else None),
    "activation_abs_max": ("activations", _abs_max),
    "activation_non_finite_fraction": ("activations", _non_finite_fraction),
    "dead_channel_fraction": ("activations", _dead_fraction),
}


def layer_ranking_view(
    session: Session, *, by: RankMetric, top: int = 10, ascending: bool = False
) -> dict[str, Any]:
    """Top layers of the last captured batch, ranked by one statistic.

    Avoids fetching and sorting detailed `get_layer_stats` results for a
    whole model. Each row carries the ranked value plus the few scalars
    needed to read it.
    """
    snapshot = session.snapshot
    if snapshot is None:
        return {
            "error": "No batch has been captured yet.",
            "hint": (
                "A session pauses on its first batch; if training is running "
                "free, call pause or refresh first."
            ),
        }
    inputs = set(session.input_names)
    layers = [name for name in session.layer_names if name not in inputs]
    stats = session.current_batch_stats(layers=layers, include_patches=False)
    position = snapshot.position
    stream, measure = _RANK_METRICS[by]
    ranked: list[tuple[float, dict[str, Any]]] = []
    unmeasured: list[str] = []
    for name in layers:
        layer_stats = stats.stats.get((name, position.phase, position.epoch))
        value = None
        if layer_stats is not None:
            value = measure(getattr(layer_stats, stream))
        if value is None or not math.isfinite(value):
            unmeasured.append(name)
            continue
        assert layer_stats is not None
        activations, gradients = layer_stats.activations, layer_stats.gradients
        row: dict[str, Any] = {"layer": name, by: _num(value)}
        row["activation_std"] = _num(activations.std) if activations.n else None
        row["gradient_std"] = _num(gradients.std) if gradients.n else None
        row["gradient_abs_max"] = _num(_abs_max(gradients))
        if activations.channel_hists is not None:
            row["dead_channels"] = activations.dead_channel_count
            row["channels_tracked"] = len(activations.channel_hists)
        ranked.append((value, row))
    ranked.sort(key=lambda pair: pair[0], reverse=not ascending)
    top = max(top, 1)
    view: dict[str, Any] = {
        "position": position_view(position, schedule=session.schedule),
        "ranked_by": by,
        "order": "ascending" if ascending else "descending",
        "layers": [row for _, row in ranked[:top]],
        "ranked_count": len(ranked),
    }
    if len(ranked) > top:
        view["omitted"] = len(ranked) - top
    if unmeasured:
        view["layers_without_value"] = unmeasured
    if not ranked and stream == "gradients":
        view["hint"] = (
            "No layer has a usable value for this gradient metric. If the "
            "phase has no backward pass (validation, say), step to a training "
            "batch first."
        )
    return view


def phases_with_data(session: Session, layers: Iterable[str]) -> list[str]:
    """Every phase any of `layers` has collected statistics for, sorted."""
    return sorted({phase for layer in layers for phase in session.stats_phases(layer)})


def default_phase(session: Session, layers: Iterable[str]) -> str | None:
    """The phase to show when the caller named none, or `None` if there is no
    data at all.

    The phase training is *currently* in, when it has data — the same choice
    the `/stats` page opens on, and the one an agent means by "the histogram
    for this layer". Falling back to the alphabetically last phase would answer
    a question about a paused `eval` run with the `train` numbers.
    """
    available = phases_with_data(session, layers)
    if not available:
        return None
    position = session.live_position
    if position is None:
        snapshot = session.snapshot
        position = snapshot.position if snapshot is not None else None
    if position is not None and position.phase in available:
        return position.phase
    return available[-1]


def _empty_point_note(stats: TensorStatsSnapshot, kind: str) -> str:
    """Why an epoch's bucket has no finite values: none seen, or all non-finite.

    The histogram counts every value, finite or not, so it tells the two
    apart; a collapsed older epoch has lost it and can only say either.
    """
    seen = None if stats.hist is None else sum(stats.hist)
    if seen == 0:
        return (
            "no gradients captured in this epoch (e.g. this phase ran no "
            "backward pass)"
            if kind == "gradients"
            else "no values captured in this epoch"
        )
    if seen is None:
        return "no finite values in this epoch (none captured, or all NaN or ±Inf)"
    return "no finite values in this epoch (all NaN or ±Inf)"


def _history_point(
    epoch: int,
    stats: TensorStatsSnapshot,
    *,
    kind: str = "activations",
    partial: bool = False,
) -> dict[str, Any]:
    point: dict[str, Any] = {"epoch": epoch, "count": stats.n}
    if partial:
        point["partial"] = True
    if stats.n == 0:
        # With nothing finite the accumulator's scalars are its ±inf
        # placeholders, which would come out as `min` above `max` and a
        # fabricated `std` of 0 — for exactly the epoch a reader most needs to
        # understand. Say what happened instead.
        point["note"] = _empty_point_note(stats, kind)
        return point
    point.update(
        {
            "mean": _num(stats.mean),
            "std": _num(stats.std),
            "min": _num(stats.min),
            "max": _num(stats.max),
            "median": _median_num(stats.median),
        }
    )
    dead = stats.dead_channel_count
    if dead is not None:
        point["dead_channels"] = dead
    return point


def stats_gap_hint(session: Session, layers: Iterable[str]) -> str:
    """Why `layers` have no running statistics, and the call that fixes it.

    Three causes produce the same empty series — the layers sit outside the
    collecting scope, collection is off, or nothing has been folded in yet —
    and each is a different tool away, so the reason is named rather than left
    to be guessed.
    """
    names = list(layers)
    subject = repr(names[0]) if len(names) == 1 else str(names)
    fixes: list[str] = []
    if session.stats_scope is not StatsScope.ALL:
        unwatched = [name for name in names if name not in session.watched_layers]
        if unwatched:
            fixes.append(f"call `watch_layers --layers {' '.join(unwatched)}`")
    if not session.stats_collecting:
        fixes.append(
            "turn collection on with `set_stats_scope --scope watched` (it is "
            "'none', "
            "which collects nothing; 'all' covers every layer)"
        )
    if not fixes:
        return (
            f"Statistics are being collected for {subject}, but no batch has "
            "folded into them yet — advance training with step or run."
        )
    return (
        f"Nothing is collecting statistics for {subject}: "
        + " and ".join(fixes)
        + "."
    )


def _in_progress(session: Session, phase: str, epoch: int) -> bool:
    """Whether `phase` of `epoch` is still being folded into its bucket.

    True at the live position in an open session, unless training is paused
    on that phase's last batch — a pause comes after the batch's statistics
    were folded in.
    """
    live = session.live_position
    if session.closed or live is None or (live.phase, live.epoch) != (phase, epoch):
        return False
    return session.is_running or not live.is_last_in_phase


def _history_channel_cap(
    session: Session, layer: str, buckets: Iterable[LayerStatsSnapshot]
) -> str | None:
    """`channel_cap_note` for a trend, whose buckets carry no tensor shape.

    The tracked row count comes from any bucket still holding its rows, the
    real channel count from the layer's activation in the published snapshot.
    """
    tracked = next(
        (
            len(stats.channel_hists)
            for bucket in buckets
            for stats in (bucket.activations, bucket.gradients)
            if stats.channel_hists is not None
        ),
        None,
    )
    snapshot = session.snapshot
    activation = None if snapshot is None else snapshot.activations.get(layer)
    if tracked is None or activation is None or activation.ndim < 2:
        return None
    return channel_cap_note(tracked, int(activation.shape[1]))


def stats_history_view(
    session: Session, *, layer: str, phase: str | None = None
) -> dict[str, Any]:
    """A watched layer's per-epoch statistics — the trend across the run.

    These come from the running accumulators, which only cover layers the
    current stats scope collects; a layer with no buckets yet gets told how to
    start collecting rather than an empty series.
    """
    if layer not in set(session.layer_names):
        return {
            "error": f"Unknown layer {layer!r}.",
            "hint": "Call get_architecture for the valid layer names.",
        }
    available = sorted(session.stats_phases(layer))
    snapshot = session.watch_snapshot(layers=[layer], include_patches=False)
    # Weight samples are kept per epoch with no phase, and `stats_phases`
    # deliberately does not count them — so a layer whose activation was never
    # captured (a module returning a tuple, say) can still have a weight trend,
    # and bailing on the phase check alone would hide it.
    weight_points = snapshot.weight_history(layer)
    if not available and not weight_points:
        return {
            "layer": layer,
            "history": {},
            "hint": stats_gap_hint(session, [layer]),
        }
    phases = [phase] if phase is not None else available
    unknown_phase = [name for name in phases if name not in available]
    history: dict[str, Any] = {}
    for name in phases:
        buckets = snapshot.phase_history(layer, name)
        if not buckets:
            continue
        history[name] = {
            kind: [
                _history_point(
                    bucket.epoch,
                    getattr(bucket, kind),
                    kind=kind,
                    partial=_in_progress(session, name, bucket.epoch),
                )
                for bucket in buckets
            ]
            for kind in ("activations", "gradients")
        }
    view: dict[str, Any] = {
        "layer": layer,
        "phases_with_data": available,
        "history": history,
    }
    buckets = [b for name in phases for b in snapshot.phase_history(layer, name)]
    cap = _history_channel_cap(session, layer, buckets)
    if cap is not None:
        view["channels_note"] = cap
    # Weight samples are taken once per epoch, at its first watched batch, and
    # carry no phase — the same parameter is in play across all of them.
    weights = {
        name: [_history_point(epoch, stats) for epoch, stats in points]
        for name, points in weight_points.items()
    }
    if weights:
        view["weight_history"] = weights
        view["weight_note"] = (
            "One sample per epoch, taken at that epoch's first watched batch, "
            "so these track the parameters drifting rather than any one batch."
        )
        if not available:
            view["hint"] = (
                f"{layer!r} has a weight trend but no activation statistics — "
                "its output was never captured, so watching it further will not "
                "add any."
            )
    if unknown_phase:
        view["unknown_phases"] = unknown_phase
    return view


def _tensor_summary(tensor: torch.Tensor) -> dict[str, Any]:
    """Scalars for a tensor read straight off a snapshot, not an accumulator.

    Weights and optimizer state never go through the watch accumulators — the
    pages draw them as strips and read the extremes off the colorbar — so their
    numbers have to be computed here. Non-finite values are separated out and
    the scalars describe the finite population, matching what
    `tensor_stats_view` reports for activations so the two read alike — down to
    the *population* standard deviation the accumulators use, so a parameter's
    `get_weight_stats` number and its `get_stats_history` trend agree.
    """
    # float64, not float32: a double-precision run whose gradient has reached
    # ~1e39 is finite, and casting down would report it as an Inf — turning the
    # very measurement that shows the explosion into a phantom divergence.
    values = tensor.detach().to(torch.float64).flatten()
    total = int(values.numel())
    finite_mask = torch.isfinite(values)
    finite = values[finite_mask]
    count = int(finite.numel())
    view: dict[str, Any] = {"shape": list(tensor.shape), "count": total}
    if count < total:
        view["finite_count"] = count
        view["non_finite_count"] = total - count
    if count == 0:
        view["note"] = (
            "every value is non-finite (NaN or ±Inf)"
            if total
            else "the tensor is empty"
        )
        return view
    view["mean"] = _num(finite.mean().item())
    view["std"] = _num(finite.std(unbiased=False).item())
    view["min"] = _num(finite.min().item())
    view["max"] = _num(finite.max().item())
    view["abs_max"] = _num(finite.abs().max().item())
    dtype = _dtype_name(tensor.dtype)
    if dtype is not None:
        view["dtype"] = dtype
    return view


# How many of a param group's parameter names the hyperparameters view lists;
# get_weight_stats names each parameter's group, for the rest.
_GROUP_PARAMETER_NAMES = 5


def _edited_keys(session: Session) -> dict[int, dict[str, Any]]:
    """Per param group, each edited key's value before its first edit."""
    edited: dict[int, dict[str, Any]] = {}
    for edit in session.hyperparameter_edits:
        edited.setdefault(edit.group, {})[edit.key] = {"original": _num(edit.original)}
    return edited


def hyperparameters_view(session: Session) -> dict[str, Any]:
    """The optimizer's param groups and their live numeric hyperparameters.

    Marks every value changed from NaNsense with what it was before its first
    edit, and names an attached scheduler, which keeps driving `lr` after an
    edit (see `Session.set_hyperparameters`).
    """
    optimizer = session.optimizer
    if optimizer is None:
        return {
            "error": "No optimizer is attached to this session.",
            "hint": "Pass optimizer= to nansense.start() to expose optimizer hyperparameters.",
        }
    group_of = session.optimizer_group_indices()
    edited_keys = _edited_keys(session)
    groups: list[dict[str, Any]] = []
    for index, group in enumerate(optimizer.param_groups):
        names = [name for name, g in group_of.items() if g == index]
        entry: dict[str, Any] = {
            "group": index,
            "parameter_count": len(names),
            "parameters": names[:_GROUP_PARAMETER_NAMES],
            "hyperparameters": {
                key: _num(value)
                for key, value in sorted(hyperparams.numeric_hyperparams(group).items())
            },
        }
        if index in edited_keys:
            entry["edited"] = edited_keys[index]
        groups.append(entry)
    view: dict[str, Any] = {"optimizer": type(optimizer).__name__, "groups": groups}
    if session.scheduler is not None:
        view["scheduler"] = type(session.scheduler).__name__
        view["scheduler_note"] = (
            "The scheduler keeps adjusting lr after an edit. StepLR, "
            "ExponentialLR, CosineAnnealingLR and ReduceLROnPlateau continue "
            "from the edited value; LambdaLR, OneCycleLR and CyclicLR recompute "
            "lr from their own schedule and replace it at their next step."
        )
    return view


def _custom_summaries(
    tensors: dict[str, torch.Tensor] | None,
) -> dict[str, dict[str, Any]]:
    """`_tensor_summary` per custom tensor instrument (`watch_*_tensor`)."""
    return {key: _tensor_summary(value) for key, value in sorted((tensors or {}).items())}


def weight_stats_view(
    session: Session, *, layer: str, parameters: Iterable[str] | None = None
) -> dict[str, Any]:
    """A layer's parameters as numbers: values, gradients, optimizer state.

    The numeric counterpart of the `/weights` page, which draws the same
    tensors as strips. Optimizer state is whatever the optimizer keeps per
    parameter (SGD's `momentum_buffer`, Adam's `exp_avg` / `exp_avg_sq` /
    `step`) and the hyperparameters are read live, so a scheduler-mutated
    learning rate shows its current value rather than the one it started with.
    """
    snapshot = session.snapshot
    if snapshot is None:
        return {
            "error": "No batch has been captured yet.",
            "hint": (
                "A session pauses on its first batch; if training is running "
                "free, call `pause` or `refresh` first."
            ),
        }
    available = session.layer_weights.get(layer, [])
    if not available:
        known = layer in set(session.layer_names)
        return {
            "error": (
                f"Layer {layer!r} has no parameters."
                if known
                else f"Unknown layer {layer!r}."
            ),
            "hint": (
                "Intermediates like `relu` or `add` carry activations but no "
                "weights."
                if known
                else "Call get_architecture for the valid layer names."
            ),
        }
    requested = list(dict.fromkeys(parameters)) if parameters else list(available)
    unknown = [name for name in requested if name not in available]
    group_of = session.optimizer_group_indices()
    edited_keys = _edited_keys(session)
    # A frozen parameter has no gradient and no optimizer state for a reason
    # that has nothing to do with the training loop's timing, and "nothing has
    # run backward yet" would send a reader looking for a bug in their loop.
    trainable = {
        name: param.requires_grad
        for name, param in session.model.named_parameters()
    }
    entries: list[dict[str, Any]] = []
    for name in requested:
        if name in unknown:
            continue
        frozen = trainable.get(name) is False
        entry: dict[str, Any] = {"parameter": name}
        if frozen:
            entry["requires_grad"] = False
        weight = snapshot.weights.get(name)
        if weight is not None:
            entry["weight"] = _tensor_summary(weight)
        gradient = snapshot.weight_gradients.get(name)
        if gradient is not None:
            entry["gradient"] = _tensor_summary(gradient)
        else:
            entry["gradient"] = None
            entry["gradient_note"] = (
                "no gradient: this parameter is frozen (requires_grad=False)"
                if frozen
                else (
                    "no gradient captured — nothing has run backward yet, or "
                    "zero_grad(set_to_none=True) cleared it before the capture"
                )
            )
        state = snapshot.optimizer_state.get(name, {})
        if state:
            entry["optimizer_state"] = {
                key: (
                    _num(float(value))
                    if value.ndim == 0
                    else _tensor_summary(value)
                )
                for key, value in sorted(state.items())
            }
        group_hparams = snapshot.optimizer_hyperparams.get(name, {})
        if group_hparams:
            group = group_of.get(name)
            entry["optimizer_param_group"] = group
            entry["optimizer_hyperparameters"] = {
                key: _num(value) for key, value in sorted(group_hparams.items())
            }
            edited = edited_keys.get(group, {}) if group is not None else {}
            if edited:
                entry["edited_hyperparameters"] = edited
        custom = _custom_summaries(snapshot.custom_weight_tensors.get(name))
        if custom:
            entry["custom"] = custom
        entries.append(entry)
    view: dict[str, Any] = {
        "layer": layer,
        "position": position_view(snapshot.position, schedule=session.schedule),
        "parameters": entries,
    }
    if unknown:
        view["unknown_parameters"] = unknown
        view["hint"] = f"{layer!r} has {available}."
    if entries and not any(entry.get("optimizer_state") for entry in entries):
        # Say which of the three it is where we can tell: a frozen parameter or
        # one outside every param group will never gain state, and telling the
        # reader to wait for the first step() would be a dead end.
        view["optimizer_note"] = (
            "No optimizer state — every parameter here is frozen "
            "(requires_grad=False), so no optimizer tracks it."
            if all(entry.get("requires_grad") is False for entry in entries)
            else (
                "No optimizer state. Either no optimizer was passed to "
                "nansense.start(), this layer is not in any of its param "
                "groups, or it has not stepped yet — torch.optim initialises "
                "per-parameter state lazily on the first step()."
            )
        )
    return view


def time_travel_view(session: Session) -> dict[str, Any]:
    """Whether the run can jump back, and to which epochs."""
    status = session.time_travel_status()
    view: dict[str, Any] = {
        "available": status.available,
        "cached_epochs": status.cached_epochs,
        "total_epochs": status.total_epochs,
    }
    if status.reason is not None:
        view["reason"] = status.reason
    if status.available:
        view["hint"] = (
            "Jumping restores the model, optimizer and scheduler state saved "
            "at the start of that epoch and pauses on its first batch. "
            "Training re-runs from there, so anything after it is discarded."
        )
    return view


def probe_view(session: Session) -> dict[str, Any]:
    """The probe: what fixed input the model is being re-run on, and how.

    A probe re-runs the model on one held input at every capture, so stepping
    shows the network's changing response to a *constant* stimulus instead of
    to a new batch each time. Perturbations edit that input in place.
    """
    perturbations = session.perturbations
    view: dict[str, Any] = {
        "pinned": session.is_pinned,
        "pinned_position": position_view(
            session.pinned_position, schedule=session.schedule
        ),
        "mode": session.probe_mode,
        "runs_completed": session.probe_count,
        "perturbations": [
            {
                "input": input_name,
                "sample": sample,
                "index": list(index),
                "values": [_num(value) for value in values],
            }
            for (input_name, sample, index), values in sorted(perturbations.items())
        ],
        "active": session.probe_result is not None,
    }
    result = session.probe_result
    if perturbations:
        # An entry that doesn't fit the base — out of range, wrong value count,
        # an input that isn't there — is skipped when the probe applies it, and
        # stays in the map regardless. So the map alone cannot tell an agent
        # whether its edit took; the probe's second forward can.
        view["perturbations_applied"] = (
            result is not None and result.perturbed_inputs is not None
        )
    if session.probe_error is not None:
        view["error"] = session.probe_error
    if not view["active"]:
        view["hint"] = (
            "The last probe failed; see `error`. The pin/perturbations/mode "
            "below are still in force."
            if session.probe_error is not None
            else (
                "Probing is set up but has produced no result yet — probes run "
                "on the training thread, so `pause` or `step` to get one."
                if session.is_pinned
                or perturbations
                or session.probe_mode != "unchanged"
                else (
                    "No probe is running. `pin_batch` holds the current input "
                    "and re-runs the model on it at every capture; "
                    "`set_probe_mode --mode eval` activates one without pinning."
                )
            )
        )
    elif perturbations:
        view["hint"] = (
            "With perturbations active the layer strips show "
            "perturbed − original, so an unchanged layer reads as flat zero."
        )
    return view


def _param_spec_view(
    spec: experiments.ExperimentParam, defaults: dict[str, object]
) -> dict[str, Any]:
    view: dict[str, Any] = {
        "key": spec.key,
        "type": spec.kind,
        # The value actually in force: a hosted playground seeds cheaper
        # defaults, and reporting the built-in one would have an agent reason
        # about a cost the run will not pay.
        "default": defaults.get(spec.key, spec.default),
        "description": spec.tooltip or spec.label,
    }
    if spec.options:
        view["options"] = sorted(spec.options)
    if spec.minimum is not None:
        view["minimum"] = _num(spec.minimum)
    return view


def experiment_catalog_view(session: Session) -> dict[str, Any]:
    """Every experiment kind, what it does, and the knobs it takes."""
    # The graph's own inputs are layer names but not experiment targets —
    # dreaming an input against itself is a no-op, and the page filters them
    # out of its selector for that reason.
    inputs = set(session.input_names)
    layers = [name for name in session.layer_names if name not in inputs]
    # `layer_available` rebuilds `named_modules()` on every call; hoisting it
    # turns a kinds x layers x modules sweep — seconds on a large model, on the
    # event loop — into one pass.
    modules = set(dict(session.model.named_modules()))
    # Every kind accepts one of two layer lists, so each list ships once under
    # `layer_sets` and a kind names its own — repeating it per kind made the
    # catalog tens of kilobytes on a large model.
    layer_sets = {
        "any_layer": layers,
        "modules_only": [name for name in layers if name in modules],
    }
    defaults = session.experiment_defaults
    kinds: list[dict[str, Any]] = []
    for spec in session.experiment_specs.values():
        needs_module = spec.needs_module or not session.fx_traced
        kinds.append(
            {
                "kind": spec.name,
                "title": spec.label,
                "summary": spec.summary,
                "description": spec.description,
                "accepts": "modules_only" if needs_module else "any_layer",
                "params": [_param_spec_view(param, defaults) for param in spec.params],
            }
        )
    used = {kind["accepts"] for kind in kinds}
    return {
        "kinds": kinds,
        "layer_sets": {key: names for key, names in layer_sets.items() if key in used},
        "hint": (
            "Experiments run on the paused training thread, so pause first. "
            "Each kind's `accepts` names the `layer_sets` list it can target "
            "— Grad-CAM and the neuron methods need a real nn.Module, so fx "
            "intermediates are excluded from `modules_only`."
        ),
        "time_limit_seconds": experiments._EXPERIMENT_TIME_LIMIT,
    }


# Rows of a result table returned inline; a longer table is cut, and says so.
_TABLE_ROWS_SHOWN = 200


def _table_view(table: experiments.Table) -> dict[str, Any]:
    rows = [
        {
            column: _num(cell) if isinstance(cell, float) else cell
            for column, cell in zip(table.columns, row, strict=True)
        }
        for row in table.rows[:_TABLE_ROWS_SHOWN]
    ]
    view: dict[str, Any] = {"columns": list(table.columns), "rows": rows}
    if len(table.rows) > _TABLE_ROWS_SHOWN:
        view["truncated"] = (
            f"First {_TABLE_ROWS_SHOWN} of {len(table.rows)} rows; the page "
            "shows them all."
        )
    return view


def experiment_result_view(
    session: Session, *, seq: int
) -> dict[str, Any]:
    """One experiment request's latest published progress or outcome."""
    result = session.experiment_result_for(seq)
    if result is None:
        # Nothing published yet is ambiguous on its own — the queue says
        # whether the request is under way, waiting, or gone (the UI draws
        # the same distinction as a spinner or a static pill).
        queue = session.experiment_queue_state(seq)
        pending: dict[str, Any] = {"seq": seq, "stage": queue.stage}
        if queue.stage == "running":
            pending["note"] = (
                "Running on the training thread now; it has published no "
                "progress yet (the Captum methods publish once, at the end)."
            )
        elif queue.stage == "queued":
            pending["queued_ahead"] = queue.ahead
            pending["note"] = (
                "Queued — experiments run on the paused training thread, so "
                "this waits for the pause and for the "
                f"{queue.ahead} request(s) in front of it."
            )
        else:
            pending["error"] = (
                "No result published for this request, and it is not queued — "
                "cancelled, superseded, or old enough to have been evicted."
            )
        return pending
    view: dict[str, Any] = {
        "seq": result.seq,
        "kind": result.kind,
        "layer": result.layer,
        "step": result.step,
        "total_steps": result.total_steps,
        "done": result.done,
        "produced": result.payload,
    }
    if result.objective is not None:
        view["objective"] = _num(result.objective)
    if result.error is not None:
        view["error"] = result.error
    if result.done and result.step < result.total_steps and result.error is None:
        view["note"] = (
            "Stopped before its last step — cancelled, superseded, or past the "
            "wall-clock limit. The result so far is still valid."
        )
    if result.attribution is not None:
        view["attribution"] = _tensor_summary(result.attribution)
    if result.image is not None:
        view["image"] = _tensor_summary(result.image)
    if result.scalars:
        view["scalars"] = {key: _num(value) for key, value in result.scalars.items()}
    if result.table is not None:
        view["table"] = _table_view(result.table)
    if result.note:
        view["note"] = result.note
    if result.image is not None or result.attribution is not None:
        view["hint"] = f"`render_experiment --seq {result.seq}` draws this result."
    if result.video is not None:
        # Imported here, not at module scope: `nansense.recording` pulls in the
        # video encoder for a constant this view only sometimes needs.
        from nansense.recording import VIDEO_FPS

        view["video"] = result.video
        view["video_hint"] = (
            f"The whole run as an MP4 ({VIDEO_FPS} fps, one frame per "
            "published step), written beside the session's recordings."
        )
    return view


def _series_view(series: instruments.MetricSeries, last: int | None) -> dict[str, Any]:
    points = list(zip(series.epochs, series.batches, series.values, strict=True))
    view: dict[str, Any] = {"cadence": series.on}
    if last is not None and len(points) > max(last, 0):
        view["points_total"] = len(points)
        points = points[len(points) - max(last, 0) :]
    view["points"] = [
        {"epoch": epoch, "batch": batch, "value": _num(value)}
        for epoch, batch, value in points
    ]
    return view


def metrics_view(
    session: Session,
    *,
    layers: Iterable[str] | None = None,
    metrics: Iterable[str] | None = None,
    phase: str | None = None,
    last: int | None = None,
) -> dict[str, Any]:
    """Custom scalar metrics registered by the training script.

    These come from `session.watch_metric(...)` in the user's own code, so what
    is here is entirely up to that script — and the fact that a metric *exists*
    is itself a signal about what its author was worried about.

    `metrics` and `phase` keep only matching series, and `last` keeps each
    series' newest points. When points are omitted, `points_total` gives the
    count before truncation. Per-batch metrics can produce large responses.
    """
    snapshot = session.watch_metrics_snapshot(layers=layers)
    wanted = None if metrics is None else set(metrics)
    series: list[dict[str, Any]] = []
    available: set[str] = set()
    for (layer, series_phase, metric, name), values in sorted(snapshot.series.items()):
        available.add(metric)
        if (wanted is not None and metric not in wanted) or (
            phase is not None and series_phase != phase
        ):
            continue
        entry: dict[str, Any] = {
            "layer": layer,
            "phase": series_phase,
            "metric": metric,
        }
        if name:
            entry["series"] = name
        entry.update(_series_view(values, last))
        series.append(entry)
    view: dict[str, Any] = {"series": series}
    errors = session.instrument_errors
    if errors:
        view["failed_instruments"] = errors
        view["hint"] = (
            "A raising instrument is disabled rather than taking the run down; "
            "these stopped collecting at the error shown."
        )
    elif not series and available:
        view["available_metrics"] = sorted(available)
        view["hint"] = "No series matches the filters; these metrics exist."
    elif not series:
        view["hint"] = (
            "No custom metrics. The training script registers them with "
            "session.watch_metric(name); they are only collected for layers "
            "the stats scope covers."
        )
    return view


def _json_safe(value: object) -> Any:
    """`value` as something JSON can carry, with non-finite floats as strings."""
    if isinstance(value, bool) or value is None or isinstance(value, (int, str)):
        return value
    if isinstance(value, float):
        return _num(value)
    return repr(value)


def settings_view(session: Session) -> dict[str, Any]:
    """The knobs behind the UI's settings dialog."""
    frequency = session.update_frequency
    performance = session.watch_performance
    return {
        "update_frequency": {
            "unit": frequency.unit,
            "n": frequency.n,
            "phase": frequency.phase,
            "description": (
                "How often views refresh while training runs — without pausing. "
                "Each update publishes a snapshot, re-runs the probe and any "
                "auto experiments, and appends a recording frame."
            ),
        },
        "watch_performance": {
            "channel_limit_enabled": performance.channel_limit_enabled,
            "channel_limit": performance.channel_limit,
            "samples_per_channel": performance.samples_per_channel,
            "average_patches": performance.average_patches,
            "description": (
                "Per-channel caps on watch memory. Changing any of them "
                "reshapes the buffers, which flushes every collected statistic."
            ),
        },
        "stats_scope": str(session.stats_scope),
        "stats_collecting": session.stats_collecting,
        "auto_run_experiments": session.auto_run_experiments,
        # `set_experiment_defaults` takes whatever the hosting script passes,
        # so this is arbitrary Python; anything the wire cannot carry is
        # rendered rather than handed to the serializer, which would take the
        # whole tool down with it.
        "experiment_defaults": {
            key: _json_safe(value)
            for key, value in sorted(session.experiment_defaults.items())
        },
    }


def recordings_view(session: Session) -> dict[str, Any]:
    """Every recording in progress, with its frame count and files."""
    statuses = session.recording.statuses()
    view: dict[str, Any] = {
        "directory": str(session.recording.directory),
        "recordings": [
            {
                "key": status.view.key,
                "label": status.view.label,
                "frames": status.frames,
                "files": [str(path) for path in status.paths],
                **({"error": status.error} if status.error is not None else {}),
            }
            for status in statuses
        ],
    }
    if not statuses:
        view["hint"] = (
            "Nothing recording. A recording appends one frame per "
            "visualization update, so set_update_frequency controls its frame "
            "rate — and a paused run produces no frames at all."
        )
    else:
        frequency = session.update_frequency
        view["frame_cadence"] = (
            f"one frame per {frequency.n} {frequency.unit}"
            + (f" of phase {frequency.phase!r}" if frequency.phase else "")
        )
    return view


def _layer_report_view(
    report: debugger.LayerReport, columns: list[str]
) -> dict[str, Any]:
    view: dict[str, Any] = {"layer": report.layer}
    for reason in columns:
        # Fractions are reported as percentages: the dialog's own framing, and
        # the scale a reader judges "how bad is this" on.
        view[f"{debugger.REASON_LABELS[reason]}_percent"] = _num(
            getattr(report, reason) * 100.0
        )
    if report.dtype is not None:
        tiny, overflow = debugger.dtype_band(report.dtype)
        view["gradient_dtype"] = _dtype_name(report.dtype)
        view["subnormal_below"] = _num(tiny)
        view["overflow_above"] = _num(overflow)
    return view


def debug_view(session: Session) -> dict[str, Any]:
    """The numerical-error debugger: its settings and any standing detection."""
    settings = session.debug_settings
    view: dict[str, Any] = {
        "settings": {
            "enabled": settings.enabled,
            "interval_batches": settings.interval,
            "check_nan_inf": settings.check_nan_inf,
            "check_under_over": settings.check_under_over,
            "threshold_fraction": _num(settings.threshold),
        }
    }
    error = session.debug_error
    if error is None:
        view["detected"] = None
        if not settings.any_check():
            view["hint"] = (
                "No checks are running — enable them with "
                "`configure_debug_checks --enabled`."
            )
        return view
    columns = debugger.columns(error)
    view["detected"] = {
        "reasons": debugger.reasons_text(error),
        "first_detected_at": position_view(error.position, schedule=session.schedule),
        "checks_that_ran": list(error.checks_used),
        "layers": [_layer_report_view(report, columns) for report in error.layers],
    }
    view["hint"] = (
        "NaN/Inf percentages are of element count; subnormal/overflow "
        "percentages are of the layer's summed |gradient|."
    )
    return view
