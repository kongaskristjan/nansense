"""Frozen debugger moments: save a paused view to disk, reload it to showcase.

A *moment* is everything the UI shows for one batch: the published
`BatchSnapshot` (activations, gradients, weights, optimizer state — the main,
weights, and "Current batch" views), the running watch statistics behind the
HISTOGRAM / MIN-MAX / GRAPHS views, the watched-layer set, and the schedule
shape behind the position label. `Session.freeze_moment` arms a one-shot
write at an exact batch position during training; `load_moment` rebuilds the
frozen pause around a freshly constructed model of the same architecture.

The intended use is the locked showcase (`examples/playground`): a prepare
run trains once and freezes its last train batch; the serving process then
needs no dataset, optimizer, or training loop —

    session = nansense.load_moment(model, "moment.pt", port=7860)
    session.lock()
    session.park()

`load_moment` also loads the frozen weights *and buffers* into `model`, so
experiments (deep dream, attribution) run against exactly the frozen network,
on inputs taken from the frozen snapshot.

Deliberately not part of a moment: probe pins/perturbations, experiment
results, recordings, and the numerical-warning banner — per-visitor or
transient state. In a distributed run the leader freezes its own shard's
statistics (patches are rank-local anyway); freeze a single-process run when
the demo needs exact global numbers.

The file is one `torch.save` payload of plain dicts/lists/tensors — loadable
with `weights_only=True`, nothing pickled by reference — written through a
temp file + atomic rename like the epoch cache.
"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch

from nansense.input_config import InputTransform, MeanStd
from nansense.restore import validate_model_state
from nansense.schedule import BatchPosition, format_position

if TYPE_CHECKING:
    from torch import nn

    from nansense.session import Session

_MOMENT_KIND = "nansense_moment"
_FORMAT_VERSION = 1
# The snapshot's tensor-dict fields, stored (and rebuilt) by name.
_SNAPSHOT_FIELDS = (
    "activations",
    "activation_gradients",
    "weights",
    "weight_gradients",
    "optimizer_state",
    "optimizer_hyperparams",
)


class MomentError(RuntimeError):
    """A moment file could not be read or does not fit the given model."""


def write_moment(session: Session, path: Path) -> None:
    """Serialize the just-published moment to `path` (training thread).

    Called from the batch `__exit__` of the position `Session.freeze_moment`
    armed, right after `_publish_snapshot` and the watch-stats fold — so the
    snapshot *is* the frozen batch and the statistics include it.
    """
    snapshot = session.snapshot
    if snapshot is None:  # unreachable from the armed path; defensive
        raise MomentError("no published snapshot to freeze")
    payload: dict[str, Any] = {
        "kind": _MOMENT_KIND,
        "version": _FORMAT_VERSION,
        # Layer/input names double-check that `load_moment`'s model discovers
        # the same graph the freezing run did (the snapshot and watch buckets
        # are keyed by them); the full state dict below is the exact
        # name/shape fingerprint `validate_model_state` checks first.
        "fingerprint": {
            "layer_names": list(session.layer_names),
            "input_names": list(session.input_names),
        },
        # Parameters *and buffers* (BatchNorm running stats, ...), so the
        # loading side can run experiments against the exact frozen network —
        # the snapshot's `weights` dict alone carries no buffers.
        "model": {
            name: value.detach().to("cpu", copy=True)
            if isinstance(value, torch.Tensor)
            else value
            for name, value in session.model.state_dict().items()
        },
        "position": asdict(snapshot.position),
        "snapshot": {
            name: getattr(snapshot, name) for name in _SNAPSHOT_FIELDS
        },
        "watched_layers": sorted(session.watched_layers),
        "watch": session._watch_accumulator.state_dict(),
        "schedule": session.schedule.state_dict(),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    torch.save(payload, tmp)
    tmp.replace(path)
    print(
        f"nansense: moment frozen at {format_position(snapshot.position)}"
        f" -> {path}",
        flush=True,
    )


def _load_payload(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise MomentError(f"no moment file at {path}")
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as e:  # corrupt file, unpicklable content, ...
        raise MomentError(f"failed to load {path}: {e}") from e
    if not isinstance(payload, dict) or payload.get("kind") != _MOMENT_KIND:
        raise MomentError(f"{path} is not a nansense moment file")
    version = payload.get("version")
    if version != _FORMAT_VERSION:
        raise MomentError(
            f"{path} has moment format version {version}; this nansense "
            f"reads version {_FORMAT_VERSION}"
        )
    return payload


def _validate(payload: dict[str, Any], session: Session, path: Path) -> None:
    problem = validate_model_state(payload, session.model)
    if problem is not None:
        raise MomentError(f"{path}: {problem}")
    fingerprint = payload["fingerprint"]
    for key, current in (
        ("layer_names", session.layer_names),
        ("input_names", session.input_names),
    ):
        if [str(n) for n in fingerprint[key]] != list(current):
            raise MomentError(
                f"{path}: the frozen run discovered different {key} than "
                "this model — was it built by the same code?"
            )


def load_moment(
    model: nn.Module,
    path: Path | str,
    *,
    port: int | None = None,
    host: str = "127.0.0.1",
    open_browser: bool = True,
    input_mean: MeanStd | dict[str, MeanStd] | None = None,
    input_std: MeanStd | dict[str, MeanStd] | None = None,
    input_transform: InputTransform | dict[str, InputTransform] | None = None,
) -> Session:
    """Rebuild a frozen moment around `model` for browsing or a locked demo.

    `model` must be a fresh instance of the architecture the moment was
    frozen from (validated against the file: parameter names/shapes and the
    discovered layer names must match); its device is kept, and the frozen
    parameters and buffers are loaded into it. The returned session shows
    exactly the frozen pause — snapshot, watch statistics, watched set, and
    schedule totals — with stats collection off (scope `"none"`), so the
    numbers sit frozen while every view, and experiments, keep working.
    Nothing trains: the session is a viewer. Raises `MomentError` when the
    file is unreadable or does not fit `model`.

    `port` / `host` / `open_browser` / `input_*` mirror `nansense.start` and
    serve the UI immediately when `port` is given. For a shared deployment,
    follow with `session.lock()` and `session.park()` — see
    `examples/playground` and `Session.freeze_moment` for the saving side.
    """
    # Imported lazily to keep the import graph acyclic (the session module
    # calls back into this one when a freeze triggers).
    from nansense.session import BatchSnapshot, Session, StatsScope

    resolved = Path(path)
    payload = _load_payload(resolved)
    session = Session(model)
    _validate(payload, session, resolved)
    model.load_state_dict(payload["model"])
    position = BatchPosition(**payload["position"])
    stored = payload["snapshot"]
    session._snapshot = BatchSnapshot(
        position=position,
        **{name: stored[name] for name in _SNAPSHOT_FIELDS},
    )
    session._live_position = position
    session._watched_layers = {
        str(layer)
        for layer in payload["watched_layers"]
        if str(layer) in session.layer_names
    }
    session._watch_performance = _watch_performance(payload["watch"])
    session._watch_accumulator.load_state_dict(payload["watch"])
    session._schedule.load_state_dict(payload["schedule"])
    # Frozen buckets stay browsable under scope "none" while nothing
    # accumulates; `lock()` later forces "all", equivalent here since no
    # batches ever run.
    session._stats_scope = StatsScope.NONE
    session._prev_stats_scope = StatsScope.NONE
    if port is not None:
        from nansense.ui import serve

        serve(
            session,
            port=port,
            host=host,
            open_browser=open_browser,
            input_mean=input_mean,
            input_std=input_std,
            input_transform=input_transform,
        )
    return session


def _watch_performance(watch_state: dict[str, Any]) -> Any:
    """The `WatchPerformance` matching the file's accumulator caps.

    Mirroring them into the session is what makes the pair consistent — a
    later `set_watch_performance` to the same values must not flush the
    restored buckets (and on a locked showcase it refuses anyway).
    """
    from nansense.session import WatchPerformance

    limit = watch_state["channel_limit"]
    if limit is None:
        return WatchPerformance(
            channel_limit_enabled=False,
            samples_per_channel=int(watch_state["samples_per_channel"]),
        )
    return WatchPerformance(
        channel_limit_enabled=True,
        channel_limit=int(limit),
        samples_per_channel=int(watch_state["samples_per_channel"]),
    )
