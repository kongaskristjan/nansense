"""Frozen debugger moments: save a paused view to disk, reload it to showcase.

A *moment* is everything the UI shows for one batch, stored as the minimal
recipe rather than the rendered tensors: the frozen batch itself (the loader's
`(inputs, targets)` item), the model's parameters and buffers, the optimizer
state, the running watch statistics behind the HISTOGRAM / MIN-MAX / GRAPHS
views, the watched-layer set, and the schedule shape behind the position
label. `Session.freeze_moment` arms a one-shot write at an exact batch
position during training; `load_moment` rebuilds the frozen pause around a
freshly constructed model of the same architecture by *replaying* the batch —
one deterministic forward/backward under the capture hooks regenerates every
activation and gradient the "Current batch" views show.

Replaying instead of storing is what keeps the file small: a deep model at
real image sizes carries gigabytes of per-layer activations and gradients for
a single batch, all reproducible from a few megabytes of inputs. Not
reproducible — and therefore stored — are the optimizer state and the watch
statistics (they aggregate training history). The extreme-patch buffers are
history too (per-channel extremes over a whole epoch's data); bound their
size at the source with `Session.set_patch_layers`.

The intended use is the locked showcase (`examples/playground`): a prepare
run trains once and freezes its last train batch; the serving process then
needs no dataset or training loop —

    session = nansense.load_moment(
        model, "moment.pt", replay=lambda m, batch: criterion(m(batch[0]), batch[1]),
        port=7860,
    )
    session.lock()
    session.park()

`replay` receives the fresh model and the stored batch item and returns the
loss to backpropagate; it must mirror the training step's forward (the same
criterion, no optimizer step). Replay runs the model in train mode — the mode
the frozen batch ran in — and restores the stored parameters and buffers
afterwards, so BatchNorm running stats are not perturbed. The replayed batch
reproduces the frozen forward against the *stored* weights: the training run
published its snapshot after the optimizer step, so the served activations
are the frozen weights' own (self-consistent with every weight view), not the
pre-step tensors the live run displayed. Determinism assumes an
inference-deterministic architecture (no dropout at train time) and no
autocast during the freeze.

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

from dataclasses import asdict, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

import torch

from nansense.console import console_print
from nansense.input_config import InputTransform, MeanStd
from nansense.restore import validate_model_state
from nansense.schedule import BatchPosition, format_position

if TYPE_CHECKING:
    from torch import nn

    from nansense.session import Session

_MOMENT_KIND = "nansense_moment"
# 4: patch/heat payloads stored quantized (uint8 + per-slot [offset, scale],
# byte 255 = non-finite sentinel); average patch grids gated behind the
# `average_patches` performance flag stored with the caps.
_FORMAT_VERSION = 4


class MomentError(RuntimeError):
    """A moment file could not be read or does not fit the given model."""


def _cpu_copy_item(item: Any) -> Any:
    """A CPU deep copy of a loader batch item (tensors in nested containers)."""
    if isinstance(item, torch.Tensor):
        return item.detach().to("cpu", copy=True)
    if isinstance(item, (list, tuple)):
        return type(item)(_cpu_copy_item(v) for v in item)
    if isinstance(item, dict):
        return {k: _cpu_copy_item(v) for k, v in item.items()}
    return item


def _to_device_item(item: Any, device: torch.device) -> Any:
    """The loader item with every tensor on `device` (`_cpu_copy_item`'s inverse)."""
    if isinstance(item, torch.Tensor):
        return item.to(device)
    if isinstance(item, (list, tuple)):
        return type(item)(_to_device_item(v, device) for v in item)
    if isinstance(item, dict):
        return {k: _to_device_item(v, device) for k, v in item.items()}
    return item


def _model_device(model: nn.Module) -> torch.device:
    """The device the model computes on (first parameter, then buffer)."""
    for tensor in model.parameters():
        return tensor.device
    for tensor in model.buffers():
        return tensor.device
    return torch.device("cpu")


def write_moment(session: Session, path: Path, *, batch_item: Any) -> None:
    """Serialize the just-published moment to `path` (training thread).

    Called from the batch `__exit__` of the position `Session.freeze_moment`
    armed, right after `_publish_snapshot` and the watch-stats fold — so the
    statistics include the frozen batch. `batch_item` is that batch's loader
    item, the replay seed `load_moment` regenerates the snapshot from.
    """
    snapshot = session.snapshot
    if snapshot is None:  # unreachable from the armed path; defensive
        raise MomentError("no published snapshot to freeze")
    if batch_item is None:
        raise MomentError(
            "the frozen batch carried no loader item to store — drive the "
            "loop with session.batches(...) or pass item= to session.batch()"
        )
    payload: dict[str, Any] = {
        "kind": _MOMENT_KIND,
        "version": _FORMAT_VERSION,
        # Layer/input names double-check that `load_moment`'s model discovers
        # the same graph the freezing run did (the replayed snapshot and the
        # watch buckets are keyed by them); the full state dict below is the
        # exact name/shape fingerprint `validate_model_state` checks first.
        "fingerprint": {
            "layer_names": list(session.layer_names),
            "input_names": list(session.input_names),
        },
        # Parameters *and* buffers (BatchNorm running stats, ...), so the
        # loading side replays and experiments against the exact frozen
        # network.
        "model": {
            name: value.detach().to("cpu", copy=True)
            if isinstance(value, torch.Tensor)
            else value
            for name, value in session.model.state_dict().items()
        },
        "position": asdict(snapshot.position),
        # The replay seed: the frozen batch's loader item, as yielded.
        "batch_item": _cpu_copy_item(batch_item),
        # Training history the replay cannot regenerate.
        "optimizer": {
            "state": snapshot.optimizer_state,
            "hyperparams": snapshot.optimizer_hyperparams,
        },
        "watched_layers": sorted(session.watched_layers),
        "watch": session._watch_accumulator.state_dict(),
        # Custom-instrument outputs the replay cannot regenerate (the
        # callbacks are code, not data): the scalar-metric series as
        # already-reduced points, and the snapshot's custom tensors.
        "instruments": session._instruments.state_dict(),
        "custom_tensors": {
            "activations": snapshot.custom_activations,
            "weights": snapshot.custom_weight_tensors,
        },
        "schedule": session.schedule.state_dict(),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    torch.save(payload, tmp)
    tmp.replace(path)
    console_print(
        f"NaNsense: moment frozen at {format_position(snapshot.position)}"
        f" -> {path}"
    )


def _load_payload(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise MomentError(f"no moment file at {path}")
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as e:  # corrupt file, unpicklable content, ...
        raise MomentError(f"failed to load {path}: {e}") from e
    if not isinstance(payload, dict) or payload.get("kind") != _MOMENT_KIND:
        raise MomentError(f"{path} is not a NaNsense moment file")
    version = payload.get("version")
    if version != _FORMAT_VERSION:
        raise MomentError(
            f"{path} has moment format version {version}; this NaNsense "
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


def _replay_batch(
    session: Session,
    model: nn.Module,
    payload: dict[str, Any],
    replay: Callable[[nn.Module, Any], torch.Tensor],
    position: BatchPosition,
) -> None:
    """Regenerate the frozen snapshot: one forward/backward under capture.

    The same hook machinery a publishing training batch uses captures every
    layer's activation (with `retain_grad`), the `replay` loss backpropagates
    through them, and `_publish_snapshot` clones the lot to CPU. The stored
    optimizer state — which the replay has no optimizer to regenerate — is
    then spliced into the published snapshot. Model mode, parameters, and
    buffers are restored afterwards (a train-mode forward advances BatchNorm
    running stats; the frozen ones must win).

    The batch item is stored on CPU (`_cpu_copy_item`), so it is moved to
    the model's device here — `load_moment` keeps the model's device, and
    the replay must run wherever the model lives (e.g. a CUDA-served
    playground).
    """
    from nansense import capture

    batch_item = _to_device_item(payload["batch_item"], _model_device(model))
    was_training = model.training
    model.train()
    try:
        capture.install_hooks(session)
        try:
            loss = replay(model, batch_item)
            if not isinstance(loss, torch.Tensor):
                raise MomentError(
                    "replay must return the loss tensor to backpropagate"
                )
            loss.backward()
        finally:
            capture.remove_hooks(session)
        session._publish_snapshot(position)
        snapshot = session._snapshot
        assert snapshot is not None
        optimizer = payload["optimizer"]
        # Stored custom-instrument tensors are spliced in like the optimizer
        # state: the loading session has no registered callbacks to
        # regenerate them. `.get` tolerates files frozen before they existed.
        custom = payload.get("custom_tensors", {})
        session._snapshot = replace(
            snapshot,
            optimizer_state=optimizer["state"],
            optimizer_hyperparams=optimizer["hyperparams"],
            custom_activations=custom.get("activations", {}),
            custom_weight_tensors=custom.get("weights", {}),
        )
    finally:
        session._activations.clear()
        model.zero_grad(set_to_none=True)
        # Undo the replay's BatchNorm running-stat updates; parameters are
        # untouched by a step-less replay, so this is a buffer restore.
        model.load_state_dict(payload["model"])
        model.train(was_training)


def load_moment(
    model: nn.Module,
    path: Path | str,
    *,
    replay: Callable[[nn.Module, Any], torch.Tensor],
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
    parameters and buffers are loaded into it. `replay` is the training
    step's forward in miniature — given the model and the stored batch item
    it returns the loss, e.g. ``lambda m, batch: criterion(m(batch[0]),
    batch[1])`` — and is run once, here, to regenerate the frozen batch's
    activations and gradients (see the module docstring for why they are not
    stored). The returned session shows exactly the frozen pause — snapshot,
    watch statistics, watched set, and schedule totals — with stats
    collection off (scope `"none"`), so the numbers sit frozen while every
    view, and experiments, keep working. Nothing trains: the session is a
    viewer. Raises `MomentError` when the file is unreadable or does not fit
    `model`.

    `port` / `host` / `open_browser` / `input_*` mirror `nansense.start` and
    serve the UI immediately when `port` is given. For a shared deployment,
    follow with `session.lock()` and `session.park()` — see
    `examples/playground` and `Session.freeze_moment` for the saving side.
    """
    # Imported lazily to keep the import graph acyclic (the session module
    # calls back into this one when a freeze triggers).
    from nansense.session import Session, StatsScope

    resolved = Path(path)
    payload = _load_payload(resolved)
    session = Session(model)
    _validate(payload, session, resolved)
    model.load_state_dict(payload["model"])
    position = BatchPosition(**payload["position"])
    _replay_batch(session, model, payload, replay, position)
    session._live_position = position
    session._watched_layers = {
        str(layer)
        for layer in payload["watched_layers"]
        if str(layer) in session.layer_names
    }
    session._watch_performance = _watch_performance(payload["watch"])
    session._watch_accumulator.load_state_dict(payload["watch"])
    instruments_state = payload.get("instruments")
    if instruments_state is not None:
        session._instruments.load_state_dict(instruments_state)
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
    samples = int(watch_state["samples_per_channel"])
    average = bool(watch_state.get("average_patches", False))
    if limit is None:
        return WatchPerformance(
            channel_limit_enabled=False,
            samples_per_channel=samples,
            average_patches=average,
        )
    return WatchPerformance(
        channel_limit_enabled=True,
        channel_limit=int(limit),
        samples_per_channel=samples,
        average_patches=average,
    )
