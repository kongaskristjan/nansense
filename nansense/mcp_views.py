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
from collections.abc import Iterable
from typing import Any

import torch

from nansense import debugger
from nansense.schedule import BatchPosition, Schedule, format_position
from nansense.session import Session
from nansense.watch import TensorStatsSnapshot, bin_midpoint

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
    """
    if session.closed:
        return "finished"
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
            "captured batch, not the live one. Call pause() to stop on the next "
            "batch, or refresh() to publish a fresh snapshot without pausing."
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
        "locked": session.locked,
        "layer_count": len(session.layer_names),
        "watched_layers": sorted(session.watched_layers),
        "stats_scope": str(session.stats_scope),
        "stats_collecting": session.stats_collecting,
        "numerical_warning": _warning_summary(session.debug_error),
    }
    hint = _state_hint(session)
    if hint is not None:
        view["hint"] = hint
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


def tensor_stats_view(
    stats: TensorStatsSnapshot,
    *,
    include_histogram: bool = False,
    shape: torch.Size | None = None,
) -> dict[str, Any]:
    """One tensor stream's scalars (and optionally its histogram).

    The accumulator's scalars deliberately cover the **finite** population only
    — one NaN would otherwise poison `min`/`max` permanently, and one Inf would
    take `mean` with it — while the histogram counts every value. That split is
    invisible in the raw snapshot and dangerous to a reader: a fully diverged
    layer has `n == 0`, which reads as "nothing here" when the truth is
    "everything here is NaN or Inf". So the counts are reported separately, and
    a partly-diverged layer says outright which population its mean describes.
    """
    finite = stats.n
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
        return view

    view["mean"] = _num(stats.mean)
    view["std"] = _num(stats.std)
    view["min"] = _num(stats.min)
    view["max"] = _num(stats.max)
    view["median"] = _num(stats.median)
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
    return view


def layer_stats_view(
    session: Session,
    *,
    layers: Iterable[str],
    include_histogram: bool = False,
) -> dict[str, Any]:
    """Activation and gradient statistics for the last captured batch.

    Reads the published `BatchSnapshot`, so any layer works whether or not it
    is watched — watching only matters for the epoch-over-epoch series that
    `stats_history_view` reads.
    """
    snapshot = session.snapshot
    if snapshot is None:
        return {
            "error": "No batch has been captured yet.",
            "hint": (
                "A session pauses on its first batch; if training is running "
                "free, call pause() or refresh() first."
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
        entries.append(
            {
                "layer": name,
                "activations": tensor_stats_view(
                    layer_stats.activations,
                    include_histogram=include_histogram,
                    shape=None if activation is None else activation.shape,
                ),
                "gradients": tensor_stats_view(
                    layer_stats.gradients,
                    include_histogram=include_histogram,
                    shape=None if gradient is None else gradient.shape,
                ),
            }
        )

    view: dict[str, Any] = {
        "position": position_view(position, schedule=session.schedule),
        "layers": entries,
    }
    if include_histogram:
        view["histogram_format"] = (
            "[value, count] pairs over signed-log bins; value is the bin midpoint"
        )
    if unknown:
        view["unknown_layers"] = unknown
        view["hint"] = "Call get_architecture for the valid layer names."
    if no_data:
        view["layers_without_data"] = no_data
    return view


def _history_point(
    epoch: int, stats: TensorStatsSnapshot
) -> dict[str, Any]:
    point: dict[str, Any] = {
        "epoch": epoch,
        "count": stats.n,
        "mean": _num(stats.mean),
        "std": _num(stats.std),
        "min": _num(stats.min),
        "max": _num(stats.max),
        "median": _num(stats.median),
    }
    dead = stats.dead_channel_count
    if dead is not None:
        point["dead_channels"] = dead
    return point


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
    if not available:
        return {
            "layer": layer,
            "history": {},
            "hint": (
                f"No statistics collected for {layer!r} yet. Call "
                f"watch_layers(['{layer}']) (or set_stats_scope('all')) and let "
                "training advance at least one batch."
            ),
        }
    phases = [phase] if phase is not None else available
    unknown_phase = [name for name in phases if name not in available]
    snapshot = session.watch_snapshot(layers=[layer], include_patches=False)
    history: dict[str, Any] = {}
    for name in phases:
        buckets = snapshot.phase_history(layer, name)
        if not buckets:
            continue
        history[name] = {
            "activations": [
                _history_point(bucket.epoch, bucket.activations) for bucket in buckets
            ],
            "gradients": [
                _history_point(bucket.epoch, bucket.gradients) for bucket in buckets
            ],
        }
    view: dict[str, Any] = {
        "layer": layer,
        "phases_with_data": available,
        "history": history,
    }
    if unknown_phase:
        view["unknown_phases"] = unknown_phase
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
                "set_debug_settings(enabled=True)."
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
