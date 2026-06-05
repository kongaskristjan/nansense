"""Playgrad session: state machine, hook installation, snapshot publishing.

A `Session` is created once per training run via `playgrad.start(...)`.
The user wraps each batch with `with session.batch(phase=..., epoch=...)`:

    session = playgrad.start(model, epochs=50, phases={"train": 196, "val": 40})
    for epoch in range(50):
        for batch in train_loader:
            with session.batch(phase="train", epoch=epoch):
                optimizer.zero_grad()
                loss = ...
                loss.backward()
                optimizer.step()

Because `optimizer.zero_grad()` lives at the start of the user's batch body,
parameter `.grad` is still populated when the context manager exits, so the
session reads it straight off the model — no backward hooks needed.

The UI (added later) drives the session by calling `stop`, `step_batch`,
`step_phase`, `step_epoch`, `step_run`, `step_until_position`, `detach`,
and finally `close`.
Whether the session captures activations/gradients for a given batch is
decided up-front at `__enter__` from the schedule + current mode, so
forward hooks are only installed for batches that will actually be
inspected.
"""

from __future__ import annotations

import inspect
import threading
from collections.abc import Callable, Iterable, Iterator
from contextlib import AbstractContextManager
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from types import TracebackType
from typing import Self

import torch
from torch import Tensor, fx, nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.hooks import RemovableHandle

from playgrad.fx_names import friendly_names
from playgrad.probe import (
    PROBE_MODES,
    PerturbationMap,
    ProbeResult,
    apply_perturbations,
)
from playgrad.restore import (
    DEFAULT_CACHE_DIR,
    TimeTravelError,
    TimeTravelJump,
    TimeTravelStatus,
    TrainingRestorer,
    validate_model_state,
)
from playgrad.schedule import BatchPosition, Schedule
from playgrad.watch import WatchAccumulator, WatchSnapshot


class Mode(StrEnum):
    STEP = "step"
    UNTIL_PHASE_CHANGE = "until_phase_change"
    UNTIL_EPOCH_CHANGE = "until_epoch_change"
    UNTIL_END = "until_end"
    UNTIL_POSITION = "until_position"
    DETACH = "detach"


@dataclass(frozen=True)
class BatchSnapshot:
    """Immutable per-batch view, fully resident on CPU.

    All tensor dicts are independent CPU clones taken at snapshot time,
    so the snapshot survives subsequent batches freeing the live tensors and
    can be safely read from any thread.

    `optimizer_state` / `optimizer_hyperparams` are populated only when the
    session was given an optimizer at `start()`; otherwise they stay empty.
    State entries are keyed `param name -> state key -> tensor` (scalar
    entries like Adam's `step` become 0-dim tensors); hyperparams are the
    numeric knobs of the parameter's group (`lr`, `momentum`, ...), read at
    the same instant — so a scheduler-driven `lr` is the batch's actual one.
    """

    position: BatchPosition
    activations: dict[str, Tensor]
    activation_gradients: dict[str, Tensor]
    weights: dict[str, Tensor]
    weight_gradients: dict[str, Tensor]
    optimizer_state: dict[str, dict[str, Tensor]] = field(default_factory=dict)
    optimizer_hyperparams: dict[str, dict[str, float]] = field(default_factory=dict)


class Session:
    def __init__(
        self,
        model: nn.Module,
        *,
        epochs: int,
        phases: dict[str, int],
        enabled: bool = True,
        optimizer: Optimizer | None = None,
        scheduler: LRScheduler | None = None,
    ) -> None:
        self.model = model
        self._enabled = enabled
        self._optimizer = optimizer
        self._scheduler = scheduler
        self._schedule = Schedule(epochs=epochs, phases=phases)
        self._mode: Mode = Mode.STEP
        self._target_position: tuple[str, int, int] | None = None
        self._restorer: TrainingRestorer | None = None
        self._pending_jump: int | None = None
        self._cv = threading.Condition()
        self._resume_token = 0
        self._pause_count = 0
        self._closed = False
        self._activations: dict[str, Tensor] = {}
        self._hook_handles: list[RemovableHandle] = []
        self._snapshot: BatchSnapshot | None = None
        self._live_position: BatchPosition | None = None
        # When disabled, skip the up-front fx trace and all name/weight
        # discovery entirely — symbolic_trace runs a proxy forward pass and is
        # the only expensive part of construction. A disabled session never
        # installs hooks, so these stay empty and every per-batch path
        # short-circuits.
        self._fx_graph: fx.GraphModule | None = _try_trace(model) if enabled else None
        self._input_names: list[str] = self._compute_input_names() if enabled else []
        self._layer_names: list[str] = self._compute_layer_names() if enabled else []
        self._layer_weights: dict[str, list[str]] = (
            self._compute_layer_weights() if enabled else {}
        )
        self._original_forward: object | None = None
        self._had_instance_forward: bool = False
        self._watched_layers: set[str] = set()
        self._watch_accumulator = WatchAccumulator()
        # Probe state (see playgrad.probe). Config fields are mutated by the
        # UI thread under `_cv`; `_probe_result` is published by the training
        # thread (also under `_cv`, so a stale in-flight run can be detected
        # via `_probe_version` and dropped instead of overwriting newer
        # config's result).
        self._pinned_input: Tensor | None = None
        self._pinned_position: BatchPosition | None = None
        self._perturbations: PerturbationMap = {}
        self._probe_mode: str = "eval"
        self._probe_request = False
        self._probe_version = 0
        self._probe_count = 0
        self._probe_result: ProbeResult | None = None
        self._probe_error: str | None = None

    @property
    def schedule(self) -> Schedule:
        return self._schedule

    @property
    def optimizer(self) -> Optimizer | None:
        return self._optimizer

    @property
    def scheduler(self) -> LRScheduler | None:
        return self._scheduler

    @property
    def mode(self) -> Mode:
        with self._cv:
            return self._mode

    @property
    def snapshot(self) -> BatchSnapshot | None:
        return self._snapshot

    @property
    def live_position(self) -> BatchPosition | None:
        """Position of the batch the training thread is currently on.

        Updated on *every* batch's `__enter__` regardless of capture mode, so
        the UI can show live epoch/batch progress during `step_epoch`,
        `step_until_position`, `step_run`, and `detach` — modes that publish a
        snapshot only at boundaries (or never), leaving `snapshot.position`
        frozen in between. Written by the training thread, read point-in-time
        by the UI thread: a single atomic reference assignment under the GIL,
        no lock needed (same contract as `snapshot`).
        """
        return self._live_position

    @property
    def closed(self) -> bool:
        with self._cv:
            return self._closed

    @property
    def enabled(self) -> bool:
        """Whether this session captures anything. Set once at `start()`.

        A disabled session is fully inert: `batch()` is a no-op context
        manager, `serve()` does nothing, and no model hooks are ever
        installed — the intended near-zero-overhead off switch for leaving
        playgrad wiring in place in a training script.
        """
        return self._enabled

    @property
    def input_names(self) -> list[str]:
        return list(self._input_names)

    @property
    def layer_names(self) -> list[str]:
        """Ordered list of every per-batch tensor key the snapshot may carry.

        In fx mode, this is the friendly name of every non-output node in the
        traced graph: inputs (`x`), module outputs (`stage1.0.bn1`), and
        scope-qualified function/method results (`stage1.0.relu1`). In the
        hook fallback, it's `input_names + named_modules`.
        """
        return list(self._layer_names)

    @property
    def layer_weights(self) -> dict[str, list[str]]:
        """Map each layer name to the parameter names it uses.

        Keys match `layer_names`; every layer has an entry, with an empty
        list for layers that consume no weights (graph inputs, `relu`, `add`,
        …). Values are qualified parameter names that index into a
        `BatchSnapshot.weights` / `.weight_gradients` dict.

        In fx mode the mapping is exact: a `call_module` node owns the
        parameters under its dotted target, and any node that references a
        parameter functionally pulls it in via a `get_attr` input. In the
        hook fallback a module maps to every parameter in its subtree (the
        weights that contributed to the output the hook captured).
        """
        return {name: list(params) for name, params in self._layer_weights.items()}

    @property
    def fx_traced(self) -> bool:
        return self._fx_graph is not None

    @property
    def pause_count(self) -> int:
        with self._cv:
            return self._pause_count

    def batch(self, *, phase: str, epoch: int) -> _BatchContext:
        return _BatchContext(self, phase=phase, epoch=epoch)

    def batches[T](
        self, loader: Iterable[T], *, phase: str, epoch: int
    ) -> Iterator[T]:
        """Iterate `loader` with each item wrapped in a `batch()` context.

        Sugar over `batch()` for the common loop shape: the user's batch
        body runs while the generator is suspended at `yield`, i.e. inside
        the batch context — hooks are installed before the forward pass and
        the capture/pause happens when the loop asks for the next item.
        A `TimeTravelJump` raised at a batch boundary therefore surfaces
        from the `for` statement itself, not from inside the user's body.

            for inputs, targets in session.batches(loader, phase="train", epoch=e):
                ...  # forward / backward / step
        """
        for item in loader:
            with self.batch(phase=phase, epoch=epoch):
                yield item

    @property
    def watched_layers(self) -> frozenset[str]:
        """Immutable snapshot of the currently-watched layer names."""
        with self._cv:
            return frozenset(self._watched_layers)

    def watch(self, layer: str) -> bool:
        """Start collecting stats for `layer`. Returns False for unknown names.

        Any name that appears in `Session.layer_names` is watchable: named
        modules, graph inputs (e.g. `x`), and fx-traced intermediate ops
        (`relu`, `add`, `mean`). Watching activates the full capture
        machinery on every batch — fx interpreter when traceable, root
        pre-hook + per-module hooks otherwise — so the visualisation runs
        at capture-mode speed regardless of pause behaviour. Real training
        runs should not enable the UI.
        """
        if layer not in self._layer_names:
            return False
        with self._cv:
            self._watched_layers.add(layer)
        return True

    def unwatch(self, layer: str) -> None:
        """Stop watching `layer` and drop any stats already collected for it."""
        with self._cv:
            self._watched_layers.discard(layer)
        self._watch_accumulator.forget_layer(layer)

    def watch_snapshot(self) -> WatchSnapshot:
        """Snapshot of all currently-watched layers' stats."""
        with self._cv:
            layers = list(self._watched_layers)
        return self._watch_accumulator.snapshot(layers=layers)

    @property
    def probe_result(self) -> ProbeResult | None:
        """The last probe run's outputs, or `None` when nothing is pinned.

        Same lock-free read contract as `snapshot`: an atomic reference to a
        frozen dataclass of CPU tensors, safe to hold from any thread.
        """
        return self._probe_result

    @property
    def probe_error(self) -> str | None:
        """Why the last probe run failed, or `None` when it succeeded."""
        return self._probe_error

    @property
    def probe_count(self) -> int:
        """Monotonic count of completed probe runs (including failed ones)."""
        with self._cv:
            return self._probe_count

    @property
    def probe_mode(self) -> str:
        """Train/eval handling for probe forwards: "unchanged", "eval", or "train"."""
        with self._cv:
            return self._probe_mode

    @property
    def is_pinned(self) -> bool:
        with self._cv:
            return self._pinned_input is not None

    @property
    def pinned_position(self) -> BatchPosition | None:
        """Where the pinned input was captured, or `None` when not pinned."""
        return self._pinned_position

    def pin_current_batch(self) -> bool:
        """Pin the last captured batch's input as the probe input.

        Returns `False` when there is nothing to pin (disabled session, no
        snapshot yet, or a snapshot without an input tensor). While pinned,
        every capture re-runs the model on this input (a "probe") and
        publishes a `ProbeResult` alongside the snapshot, so the UI can show
        the network's evolving response to one fixed batch across stepping
        and time travel. Pinning while paused runs the probe immediately on
        the paused training thread (see `_wait_for_proceed`).
        """
        if not self._enabled:
            return False
        snap = self._snapshot
        input_name = self._input_names[0] if self._input_names else None
        if snap is None or input_name is None:
            return False
        pinned = snap.activations.get(input_name)
        if pinned is None:
            return False
        with self._cv:
            self._pinned_input = pinned
            self._pinned_position = snap.position
            self._request_probe_locked()
        return True

    def unpin_batch(self) -> None:
        """Drop the pinned input (and the probe result, absent perturbations)."""
        with self._cv:
            if self._pinned_input is None:
                return
            self._pinned_input = None
            self._pinned_position = None
            if self._perturbations:
                # Perturbations keep probing, now against the snapshot input.
                self._request_probe_locked()
                return
            self._clear_probe_result_locked()

    @property
    def perturbations(self) -> PerturbationMap:
        """Copy of the active perturbations: (sample, y, x) -> channel values."""
        with self._cv:
            return dict(self._perturbations)

    def add_perturbation(
        self, *, sample: int, y: int, x: int, values: tuple[float, ...]
    ) -> None:
        """Pin pixel `(y, x)` of `sample` to per-channel `values` on probe inputs.

        `values` are in the model's input space (i.e. already normalized by
        the caller — the UI back-transforms the picked display color with the
        `input_mean` / `input_std` it was given). Perturbations apply to the
        probe's base input — the pinned batch, or the current snapshot's
        input when nothing is pinned — and trigger a probe re-run that also
        captures the perturbed activations. Entries that don't fit the base
        (out of range, wrong channel count) are skipped at apply time.
        """
        if not self._enabled:
            return
        with self._cv:
            self._perturbations[(sample, y, x)] = tuple(values)
            self._request_probe_locked()

    def clear_perturbations(self) -> None:
        """Drop all perturbations (and the probe result, when not pinned)."""
        with self._cv:
            if not self._perturbations:
                return
            self._perturbations.clear()
            if self._pinned_input is not None:
                self._request_probe_locked()
                return
            self._clear_probe_result_locked()

    def _clear_probe_result_locked(self) -> None:
        """Deactivate probing and drop the published result (caller holds `_cv`)."""
        self._probe_version += 1
        self._probe_request = False
        self._probe_result = None
        self._probe_error = None
        self._cv.notify_all()

    def set_probe_mode(self, mode: str) -> None:
        """Set train/eval handling for probe forwards.

        - `"eval"` (default): the whole model is switched to eval — BatchNorm
          uses running stats, dropout is off — and restored afterwards.
        - `"train"`: the whole model is switched to train and restored.
        - `"unchanged"`: modules run with whatever `training` flags the
          training loop left on them.

        Regardless of mode, probes never mutate training state: per-module
        flags and all buffers are restored after the run, and the RNG is
        forked around it. A mode change while pinned re-runs the probe.
        """
        if mode not in PROBE_MODES:
            raise ValueError(
                f"unknown probe mode {mode!r}; expected one of {PROBE_MODES}"
            )
        with self._cv:
            if mode == self._probe_mode:
                return
            self._probe_mode = mode
            if self._probe_active_locked():
                self._request_probe_locked()
            else:
                self._probe_version += 1

    def wait_for_probe(
        self, *, after_count: int = 0, timeout: float | None = None
    ) -> bool:
        """Block until more than `after_count` probe runs have completed.

        The probe-side counterpart of `wait_until_paused`, used by tests and
        the UI to synchronize with asynchronously-requested probe runs
        without polling. Counts failed runs too — check `probe_error`.
        """
        with self._cv:
            return self._cv.wait_for(
                lambda: self._probe_count > after_count or self._closed,
                timeout=timeout,
            )

    def current_weights(self) -> dict[str, Tensor]:
        """CPU clones of the model's parameters, read live at call time.

        Unlike `snapshot.weights` (captured at a pause), this reads the
        parameters whenever it's called — so the UI can show current weights
        mid-training, including in `detach` / `step_run` where no snapshot is
        published. The read races with the training thread's in-place updates;
        for a visualisation that's benign — a torn read at worst, never a
        crash. Keys match `named_parameters()`, the same keys
        `layer_weights` indexes into.
        """
        return {n: self._cpu_clone(p) for n, p in self.model.named_parameters()}

    def current_weight_gradients(self) -> dict[str, Tensor]:
        """CPU clones of the parameters' current `.grad`, read live at call time.

        The live counterpart of `BatchSnapshot.weight_gradients`, with the
        same contract: parameters whose gradient is `None` (nothing has run
        backward yet, or `zero_grad(set_to_none=True)` just cleared them) are
        omitted. Same benign-race caveat as `current_weights`.
        """
        return {
            n: self._cpu_clone(p.grad)
            for n, p in self.model.named_parameters()
            if p.grad is not None
        }

    def current_optimizer_state(self) -> dict[str, dict[str, Tensor]]:
        """Per-parameter optimizer state, read live at call time.

        `optimizer.state` is keyed by the parameter object itself, so the
        mapping back to parameter names is an identity lookup — it works for
        any optimizer following the `torch.optim` convention with no
        per-optimizer code (SGD's `momentum_buffer`, Adam's `step` /
        `exp_avg` / `exp_avg_sq`, custom optimizers alike). Tensor entries
        are CPU-cloned; plain int/float entries become 0-dim tensors so the
        value type stays uniform. Empty when no optimizer was passed to
        `start()`, or before the first `optimizer.step()` (state is lazily
        initialised). Same benign-race caveat as `current_weights`.
        """
        if self._optimizer is None:
            return {}
        names = {id(p): n for n, p in self.model.named_parameters()}
        result: dict[str, dict[str, Tensor]] = {}
        for param, state in self._optimizer.state.items():
            name = names.get(id(param))
            if name is None:
                continue  # parameter from some other model
            entries: dict[str, Tensor] = {}
            for key, value in state.items():
                if isinstance(value, Tensor):
                    entries[key] = self._cpu_clone(value)
                elif isinstance(value, (int, float)):
                    entries[key] = torch.tensor(float(value))
            if entries:
                result[name] = entries
        return result

    def current_optimizer_hyperparams(self) -> dict[str, dict[str, float]]:
        """Per-parameter numeric hyperparameters of the optimizer group.

        Maps each parameter name to the plain-numeric knobs of the param
        group it belongs to (`lr`, `momentum`, `weight_decay`, ...). Read
        live, so a scheduler-mutated `lr` shows its current value. Unlike
        `current_optimizer_state` this is populated as soon as the optimizer
        exists — groups are not lazily initialised. Empty when no optimizer
        was passed to `start()`.
        """
        if self._optimizer is None:
            return {}
        names = {id(p): n for n, p in self.model.named_parameters()}
        result: dict[str, dict[str, float]] = {}
        for group in self._optimizer.param_groups:
            numeric = {
                key: float(value)
                for key, value in group.items()
                if key != "params"
                and isinstance(value, (int, float))
                and not isinstance(value, bool)
            }
            for param in group["params"]:
                name = names.get(id(param))
                if name is not None:
                    result[name] = dict(numeric)
        return result

    def set_schedule(
        self,
        *,
        epochs: int | None = None,
        phases: dict[str, int] | None = None,
    ) -> None:
        with self._cv:
            self._schedule.update(epochs=epochs, phases=phases)

    def stop(self) -> None:
        self._set_mode(Mode.STEP, resume=False)

    def step_batch(self) -> None:
        self._set_mode(Mode.STEP, resume=True)

    def step_phase(self) -> None:
        self._set_mode(Mode.UNTIL_PHASE_CHANGE, resume=True)

    def step_epoch(self) -> None:
        self._set_mode(Mode.UNTIL_EPOCH_CHANGE, resume=True)

    def step_run(self) -> None:
        self._set_mode(Mode.UNTIL_END, resume=True)

    def step_until_position(self, *, phase: str, epoch: int, batch_idx: int) -> None:
        with self._cv:
            self._target_position = (phase, epoch, batch_idx)
        self._set_mode(Mode.UNTIL_POSITION, resume=True)

    def detach(self) -> None:
        self._set_mode(Mode.DETACH, resume=True)

    def training_restorer(
        self, *, cache_dir: Path = DEFAULT_CACHE_DIR
    ) -> TrainingRestorer:
        """Create the restorer that opts this session into time travel.

        Wrapping the epoch loop in the returned object (see
        `TrainingRestorer`) enables both epoch-start checkpointing to
        `cache_dir` and UI-driven jumps back to any cached epoch. A session
        without a restorer never writes checkpoints and never raises
        `TimeTravelJump`. On a disabled session the restorer is inert: the
        loop runs exactly once and nothing touches the disk.
        """
        if self._restorer is not None:
            raise RuntimeError("this session already has a training restorer")
        restorer = TrainingRestorer(self, cache_dir=cache_dir)
        if self._enabled:
            self._restorer = restorer
        return restorer

    def request_time_travel(self, epoch: int) -> None:
        """Ask the training thread to restart at the start of `epoch`.

        Validates the request up-front on the calling (UI) thread — the
        checkpoint must exist, load, and match the live model's parameter
        shapes — and raises `TimeTravelError` with a displayable message
        otherwise, so an incompatible cache (e.g. written by a previous run
        with a different model) is rejected before anything unwinds. On
        success the jump is armed: the training thread raises
        `TimeTravelJump` at its next batch boundary (immediately, when
        paused), the restorer rolls the state back, and the session enters
        `STEP` mode so the first batch of `epoch` pauses for inspection.
        """
        restorer = self._restorer
        if restorer is None:
            raise TimeTravelError(
                "time travel requires the training loop to be wrapped in a "
                "training restorer (session.training_restorer())"
            )
        if restorer.finished:
            raise TimeTravelError("the training run has already completed")
        if not 0 <= epoch < self._schedule.epochs:
            raise TimeTravelError(
                f"epoch {epoch} out of range [0, {self._schedule.epochs})"
            )
        payload = restorer.cache.load(epoch)
        error = validate_model_state(payload, self.model)
        if error is not None:
            raise TimeTravelError(error)
        with self._cv:
            self._pending_jump = epoch
            self._mode = Mode.STEP
            self._resume_token += 1
            self._cv.notify_all()

    def time_travel_status(self) -> TimeTravelStatus:
        """What the UI needs to render the time-travel button and dialog."""
        total = self._schedule.epochs
        restorer = self._restorer
        if restorer is None:
            return TimeTravelStatus(
                available=False,
                reason=(
                    "Time travel is off: the training loop is not wrapped in "
                    "a training restorer (`while restorer.pending(): with "
                    "restorer: ...`), so no epoch checkpoints are saved."
                ),
                cached_epochs=[],
                total_epochs=total,
            )
        cached = [e for e in restorer.cache.cached_epochs() if 0 <= e < total]
        if restorer.finished:
            return TimeTravelStatus(
                available=False,
                reason="The training run has completed; time travel is no longer available.",
                cached_epochs=cached,
                total_epochs=total,
            )
        return TimeTravelStatus(
            available=True, reason=None, cached_epochs=cached, total_epochs=total
        )

    def close(self) -> None:
        with self._cv:
            self._closed = True
            self._cv.notify_all()

    def wait_until_paused(
        self,
        *,
        after_pauses: int = 0,
        timeout: float | None = None,
    ) -> bool:
        with self._cv:
            return self._cv.wait_for(
                lambda: self._pause_count > after_pauses or self._closed,
                timeout=timeout,
            )

    def _set_mode(self, mode: Mode, *, resume: bool) -> None:
        with self._cv:
            self._mode = mode
            if resume:
                self._resume_token += 1
                self._cv.notify_all()

    def _should_capture(self, pos: BatchPosition) -> bool:
        with self._cv:
            if self._closed:
                return False
            mode = self._mode
            target = self._target_position
        match mode:
            case Mode.STEP:
                return True
            case Mode.UNTIL_PHASE_CHANGE:
                return pos.is_last_in_phase
            case Mode.UNTIL_EPOCH_CHANGE:
                return pos.is_last_in_epoch
            case Mode.UNTIL_END:
                return pos.is_last_overall
            case Mode.UNTIL_POSITION:
                if target is None:
                    return False
                return (pos.phase, pos.epoch, pos.batch_idx) == target
            case Mode.DETACH:
                return False

    def _install_hooks(self) -> None:
        self._activations.clear()
        if self._fx_graph is not None:
            self._patch_forward()
            return
        pre = self.model.register_forward_pre_hook(
            self._make_pre_hook(self._activations)
        )
        self._hook_handles.append(pre)
        for name, module in self.model.named_modules():
            if module is self.model:
                continue
            handle = module.register_forward_hook(
                self._make_hook(name, self._activations)
            )
            self._hook_handles.append(handle)

    def _remove_hooks(self) -> None:
        if self._original_forward is not None:
            self._unpatch_forward()
        for h in self._hook_handles:
            h.remove()
        self._hook_handles.clear()

    def _update_watch_stats(self, pos: BatchPosition) -> None:
        for name in self._watched_layers:
            tensor = self._activations.get(name)
            if tensor is None:
                continue
            self._watch_accumulator.update(
                layer=name,
                phase=pos.phase,
                epoch=pos.epoch,
                kind="activation",
                x=tensor,
            )
            grad = tensor.grad
            if grad is not None:
                self._watch_accumulator.update(
                    layer=name,
                    phase=pos.phase,
                    epoch=pos.epoch,
                    kind="gradient",
                    x=grad,
                )

    def _patch_forward(self) -> None:
        # Stash whatever .forward currently resolves to so we can put it back,
        # remembering whether it was an instance attribute or a class method.
        self._had_instance_forward = "forward" in self.model.__dict__
        self._original_forward = self.model.forward
        graph = self._fx_graph
        capture = self._activations
        assert graph is not None

        def fx_forward(*args: Tensor) -> object:
            # fx.Interpreter.run takes positional args matched to placeholder
            # order; kwargs aren't passed through.
            return _CaptureInterpreter(graph, capture).run(*args)

        object.__setattr__(self.model, "forward", fx_forward)

    def _unpatch_forward(self) -> None:
        if self._had_instance_forward and self._original_forward is not None:
            object.__setattr__(self.model, "forward", self._original_forward)
        elif "forward" in self.model.__dict__:
            object.__delattr__(self.model, "forward")
        self._original_forward = None
        self._had_instance_forward = False

    def _make_hook(
        self, name: str, capture: dict[str, Tensor]
    ) -> Callable[[nn.Module, object, object], None]:
        def hook(_module: nn.Module, _inputs: object, output: object) -> None:
            if not isinstance(output, Tensor):
                return
            if output.requires_grad:
                output.retain_grad()
            capture[name] = output

        return hook

    def _make_pre_hook(
        self, capture: dict[str, Tensor]
    ) -> Callable[[nn.Module, tuple[object, ...]], None]:
        def hook(_module: nn.Module, inputs: tuple[object, ...]) -> None:
            for i, inp in enumerate(inputs):
                if not isinstance(inp, Tensor):
                    continue
                name = (
                    self._input_names[i]
                    if i < len(self._input_names)
                    else f"arg_{i}"
                )
                if inp.requires_grad:
                    inp.retain_grad()
                capture[name] = inp

        return hook

    def _compute_input_names(self) -> list[str]:
        if self._fx_graph is not None:
            return [
                n.name for n in self._fx_graph.graph.nodes if n.op == "placeholder"
            ]
        return _infer_input_names(self.model)

    def _compute_layer_names(self) -> list[str]:
        if self._fx_graph is not None:
            names = friendly_names(self._fx_graph.graph)
            return [
                names[n]
                for n in self._fx_graph.graph.nodes
                if n.op != "output"
            ]
        return self._input_names + [
            name for name, m in self.model.named_modules() if m is not self.model
        ]

    def _compute_layer_weights(self) -> dict[str, list[str]]:
        param_names = [name for name, _ in self.model.named_parameters()]
        if self._fx_graph is not None:
            return self._fx_layer_weights(param_names)
        return self._hook_layer_weights(param_names)

    def _fx_layer_weights(self, param_names: list[str]) -> dict[str, list[str]]:
        assert self._fx_graph is not None
        param_set = set(param_names)
        names = friendly_names(self._fx_graph.graph)
        result: dict[str, list[str]] = {}
        for node in self._fx_graph.graph.nodes:
            if node.op == "output":
                continue
            used: set[str] = set()
            if node.op == "call_module":
                used.update(_params_under(param_names, str(node.target)))
            # Parameters used functionally (e.g. F.conv2d(x, self.weight)) reach
            # the node through a get_attr input whose target is the param name.
            for inp in node.all_input_nodes:
                if inp.op == "get_attr" and inp.target in param_set:
                    used.add(str(inp.target))
            result[names[node]] = sorted(used)
        return result

    def _hook_layer_weights(self, param_names: list[str]) -> dict[str, list[str]]:
        result: dict[str, list[str]] = {name: [] for name in self._input_names}
        for name, module in self.model.named_modules():
            if module is self.model:
                continue
            result[name] = _params_under(param_names, name)
        return result

    @staticmethod
    def _cpu_clone(t: Tensor) -> Tensor:
        return t.detach().to("cpu", copy=True)

    def _publish_snapshot(self, pos: BatchPosition) -> None:
        activations = {n: self._cpu_clone(a) for n, a in self._activations.items()}
        activation_gradients = {
            n: self._cpu_clone(a.grad)
            for n, a in self._activations.items()
            if a.grad is not None
        }
        weights = {
            n: self._cpu_clone(p) for n, p in self.model.named_parameters()
        }
        weight_gradients = {
            n: self._cpu_clone(p.grad)
            for n, p in self.model.named_parameters()
            if p.grad is not None
        }
        self._snapshot = BatchSnapshot(
            position=pos,
            activations=activations,
            activation_gradients=activation_gradients,
            weights=weights,
            weight_gradients=weight_gradients,
            # Runs on the training thread at __exit__, so these reads are
            # consistent with the weights above ({} when no optimizer given).
            optimizer_state=self.current_optimizer_state(),
            optimizer_hyperparams=self.current_optimizer_hyperparams(),
        )

    def _take_pending_jump(self) -> int | None:
        """Consume the armed time-travel target, if any (training thread)."""
        # Lock-free fast path: this runs on every batch boundary, and the
        # GIL-atomic read is None for the entire life of most sessions. A
        # request racing the read is simply consumed at the next boundary.
        if self._pending_jump is None:
            return None
        with self._cv:
            jump = self._pending_jump
            self._pending_jump = None
            return jump

    def _rewind_to_epoch(self, epoch: int) -> None:
        """Reset per-epoch bookkeeping after a time-travel restore.

        The schedule's batch counters for `epoch` and later are dropped so
        the re-run epochs advance from batch 0, and the watch accumulators
        forget the abandoned timeline's buckets — they're additive, so the
        re-run samples must start from empty ones.
        """
        with self._cv:
            self._schedule.rewind_to_epoch(epoch)
        self._watch_accumulator.forget_epochs_from(epoch)

    def _wait_for_proceed(self) -> None:
        # A pending time-travel jump also ends the wait: its request already
        # bumped the resume token, so a pause that began *after* the request
        # would otherwise sit waiting for a second UI command.
        with self._cv:
            seen = self._resume_token
            self._pause_count += 1
            self._cv.notify_all()
        # Pause-time job loop: probe requests (pin / mode changes from the
        # UI) also wake the paused training thread, which runs the probe
        # forward *here* — the model is only ever touched from the training
        # thread — and re-enters the wait. The forward runs outside the lock
        # so UI reads (mode, pause_count, ...) stay responsive meanwhile.
        while True:
            with self._cv:
                self._cv.wait_for(
                    lambda: self._resume_token != seen
                    or self._closed
                    or self._pending_jump is not None
                    or self._probe_request
                )
                run_probe = self._probe_request
                self._probe_request = False
                done = (
                    self._resume_token != seen
                    or self._closed
                    or self._pending_jump is not None
                )
            if done:
                # A coalesced probe request is dropped: resuming into a
                # capture re-runs the probe anyway (_maybe_run_probe_at_capture).
                return
            if run_probe:
                self._run_probe_guarded()

    def _probe_active_locked(self) -> bool:
        """Whether probe runs should happen at all (caller holds `_cv`)."""
        return self._pinned_input is not None or bool(self._perturbations)

    def _request_probe_locked(self) -> None:
        """Arm a probe run and wake a paused training thread (caller holds `_cv`)."""
        self._probe_version += 1
        self._probe_request = True
        self._cv.notify_all()

    def _maybe_run_probe_at_capture(self) -> None:
        """Run a probe right after a capture published its snapshot.

        Called by `_BatchContext.__exit__` before the pause, so every pause
        shows a probe result consistent with the just-captured weights. Any
        UI request armed in the meantime is consumed here — the run below
        uses the current config either way.
        """
        with self._cv:
            self._probe_request = False
            active = self._probe_active_locked()
        if active:
            self._run_probe_guarded()

    def _run_probe_guarded(self) -> None:
        # A failing probe (bad input, OOM, model quirk) must not kill the
        # training thread or wedge the pause loop; the error is published
        # for the UI to display instead.
        try:
            self._run_probe()
        except Exception as e:  # noqa: BLE001 — surfaced via probe_error
            with self._cv:
                self._probe_error = f"{type(e).__name__}: {e}"
                self._probe_count += 1
                self._cv.notify_all()

    def _run_probe(self) -> None:
        """One probe run: isolated forwards on the base (and perturbed) input.

        Training-thread only. Reads the probe config under `_cv`, runs the
        forwards without the lock, and publishes the result only if the
        config is still current — a config change mid-run (re-pin, mode flip,
        new perturbation) wins and its own request re-runs the probe. The
        base input is the pinned batch, or the snapshot's input when only
        perturbations are active.
        """
        with self._cv:
            version = self._probe_version
            pinned = self._pinned_input
            mode = self._probe_mode
            perturbations = dict(self._perturbations)
        if pinned is None and not perturbations:
            return
        base = pinned if pinned is not None else self._snapshot_input()
        if base is None:
            return
        perturbed = apply_perturbations(base, perturbations)
        inputs = [base] if perturbed is None else [base, perturbed]
        captures = self._probe_forwards(inputs, mode=mode)
        result = ProbeResult(
            input=base,
            activations=captures[0],
            mode=mode,
            perturbed_input=perturbed,
            perturbed_activations=captures[1] if perturbed is not None else None,
        )
        with self._cv:
            if self._probe_version != version:
                return
            self._probe_result = result
            self._probe_error = None
            self._probe_count += 1
            self._cv.notify_all()

    def _snapshot_input(self) -> Tensor | None:
        """The last snapshot's input tensor (the probe base when unpinned)."""
        snap = self._snapshot
        input_name = self._input_names[0] if self._input_names else None
        if snap is None or input_name is None:
            return None
        return snap.activations.get(input_name)

    def _probe_forwards(
        self, inputs: list[Tensor], *, mode: str
    ) -> list[dict[str, Tensor]]:
        """Run isolated no-grad forwards, capturing every layer's output.

        Isolation contract — probes never mutate training state:

        - Per-module `training` flags are saved and restored ("eval"/"train"
          probes flip them; "unchanged" runs with whatever the loop set).
        - Every buffer is restored afterwards (a train-mode BatchNorm forward
          updates running stats in place).
        - The RNG is forked, so e.g. train-mode dropout doesn't perturb the
          global stream that time-travel replays depend on.
        - `torch.no_grad()` keeps parameters' `.grad` and autograd state
          untouched.
        """
        device = self._model_device()
        saved_flags = [(m, m.training) for m in self.model.modules()]
        saved_buffers = [(b, b.detach().clone()) for _, b in self.model.named_buffers()]
        try:
            if mode == "eval":
                self.model.eval()
            elif mode == "train":
                self.model.train()
            with self._fork_rng(device), torch.no_grad():
                return [self._capture_forward(inp.to(device)) for inp in inputs]
        finally:
            for module, flag in saved_flags:
                module.training = flag
            with torch.no_grad():
                for buffer, saved in saved_buffers:
                    buffer.copy_(saved)

    def _capture_forward(self, inp: Tensor) -> dict[str, Tensor]:
        """One forward pass capturing every layer output into a fresh dict.

        Never touches the batch path's state (`_activations`,
        `_hook_handles`, the patched forward): in fx mode the interpreter
        writes straight into a local dict, and in the hook fallback temporary
        hooks are registered and removed around the call. Safe because probes
        only run between batches, when the batch path's hooks are
        uninstalled.
        """
        capture: dict[str, Tensor] = {}
        if self._fx_graph is not None:
            _CaptureInterpreter(self._fx_graph, capture).run(inp)
        else:
            handles = [
                self.model.register_forward_pre_hook(self._make_pre_hook(capture))
            ]
            handles += [
                module.register_forward_hook(self._make_hook(name, capture))
                for name, module in self.model.named_modules()
                if module is not self.model
            ]
            try:
                self.model(inp)
            finally:
                for handle in handles:
                    handle.remove()
        return {name: self._cpu_clone(tensor) for name, tensor in capture.items()}

    def _model_device(self) -> torch.device:
        param = next(self.model.parameters(), None)
        if param is not None:
            return param.device
        buffer = next(self.model.buffers(), None)
        if buffer is not None:
            return buffer.device
        return torch.device("cpu")

    @staticmethod
    def _fork_rng(device: torch.device) -> AbstractContextManager[None]:
        if device.type in ("cuda", "mps"):
            return torch.random.fork_rng(devices=[device], device_type=device.type)
        return torch.random.fork_rng(devices=[])


class _BatchContext:
    def __init__(self, session: Session, *, phase: str, epoch: int) -> None:
        self._session = session
        self._phase = phase
        self._epoch = epoch
        self._position: BatchPosition | None = None
        self._captured = False
        self._stats_only = False

    def __enter__(self) -> Self:
        # `_enabled` is a plain attribute read (no lock), checked first so a
        # disabled session pays nothing per batch: no schedule advance, no
        # capture decision, no hook install. `_position` stays None, so
        # `__exit__` also returns immediately.
        if not self._session._enabled or self._session.closed:
            return self
        self._position = self._session._schedule.advance(self._phase, self._epoch)
        # Publish the live position before the forward pass so the UI top bar
        # tracks progress on every batch, even in modes that don't capture
        # snapshots here (step_epoch, step_until_position, step_run, detach).
        self._session._live_position = self._position
        # A jump that arrived while training was running (not paused) is
        # consumed before this batch does any work. Raising from __enter__
        # skips __exit__, but nothing has been installed yet.
        jump = self._session._take_pending_jump()
        if jump is not None:
            raise TimeTravelJump(jump)
        # With a restorer attached, the first batch of each epoch checkpoints
        # the epoch-start state (model/optimizer/scheduler/RNG) to disk —
        # before any forward pass, so a later jump back to this epoch
        # restores exactly this moment.
        restorer = self._session._restorer
        if (
            restorer is not None
            and self._position.batch_idx == 0
            and self._position.phase == self._session._schedule.first_phase_name
        ):
            restorer.save_epoch_start(self._epoch)
        self._captured = self._session._should_capture(self._position)
        self._stats_only = (
            not self._captured and bool(self._session._watched_layers)
        )
        # Capture and stats-only use the same hook installation: full fx
        # interpreter (or full per-module hooks + root pre-hook in
        # hook-mode). That way any name in `layer_names` — inputs, fx
        # intermediates, modules — can be watched. The only difference
        # is whether we publish a snapshot and pause at __exit__.
        if self._captured or self._stats_only:
            self._session._install_hooks()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        if self._position is None:
            return
        if self._captured or self._stats_only:
            if exc is None and self._session._watched_layers:
                self._session._update_watch_stats(self._position)
            self._session._remove_hooks()
            if self._captured and exc is None and not self._session.closed:
                self._session._publish_snapshot(self._position)
                self._session._maybe_run_probe_at_capture()
                self._session._wait_for_proceed()
            self._session._activations.clear()
        # Every batch boundary — captured, stats-only, or plain (detach) —
        # consumes an armed time-travel jump. `_wait_for_proceed` above
        # returns immediately when a jump is pending, so a paused batch
        # reacts to the request without a second UI command.
        if exc is None:
            jump = self._session._take_pending_jump()
            if jump is not None:
                raise TimeTravelJump(jump)

    @property
    def position(self) -> BatchPosition | None:
        return self._position

    @property
    def captured(self) -> bool:
        return self._captured


def start(
    model: nn.Module,
    *,
    epochs: int,
    phases: dict[str, int],
    enabled: bool = True,
    optimizer: Optimizer | None = None,
    scheduler: LRScheduler | None = None,
    port: int | None = None,
    host: str = "127.0.0.1",
    input_mean: tuple[float, ...] | None = None,
    input_std: tuple[float, ...] | None = None,
) -> Session:
    """Create a `Session` for `model` (and optionally serve the UI).

    With `enabled=False` the session is a near-zero-overhead no-op: no fx
    trace at construction, `batch()` does nothing, and the UI is skipped.
    This lets a training script keep its playgrad wiring in place and turn
    the whole UI off with a single flag.

    `optimizer` is optional: when given, snapshots (and the weights page)
    additionally carry each parameter's optimizer state — momentum buffers,
    Adam moments, step counts — plus its param group's numeric
    hyperparameters. Without it, everything behaves exactly as before.

    `scheduler` is optional: when given, time-travel checkpoints include the
    LR scheduler's state, so a jump restores the learning-rate schedule
    automatically along with the model and optimizer.

    `port` is optional: when given, the UI is served immediately on that
    port (equivalent to a separate `playgrad.serve(session, port=...)`
    call, which remains available for finer control). `host`, `input_mean`,
    and `input_std` are forwarded to `serve`.
    """
    session = Session(
        model,
        epochs=epochs,
        phases=phases,
        enabled=enabled,
        optimizer=optimizer,
        scheduler=scheduler,
    )
    if port is not None:
        # Imported lazily: playgrad.ui imports this module at the top level.
        from playgrad.ui import serve

        serve(
            session, port=port, host=host, input_mean=input_mean, input_std=input_std
        )
    return session


def _try_trace(model: nn.Module) -> fx.GraphModule | None:
    try:
        return fx.symbolic_trace(model)
    except Exception:
        return None


def _params_under(param_names: list[str], target: str) -> list[str]:
    """Qualified parameter names owned by the module at dotted path `target`.

    Matches `target.*` (the params the module and its descendants hold). The
    bare `target` is included too for the degenerate case of a parameter
    registered directly under that name.
    """
    prefix = f"{target}."
    return sorted(p for p in param_names if p == target or p.startswith(prefix))


class _CaptureInterpreter(fx.Interpreter):
    """fx interpreter that snapshots every node's tensor output.

    The interpreter runs the traced graph one node at a time and lets us
    intercept after each run. We retain_grad on every non-leaf tensor so
    the user's subsequent loss.backward() populates `.grad`, and store the
    live tensor under its friendly name in `capture`.
    """

    def __init__(self, gm: fx.GraphModule, capture: dict[str, Tensor]) -> None:
        super().__init__(gm)
        self._capture = capture
        self._names = friendly_names(gm.graph)

    def run_node(self, n: fx.Node) -> object:
        result = super().run_node(n)
        if n.op == "output":
            return result
        if isinstance(result, Tensor):
            if result.requires_grad:
                result.retain_grad()
            self._capture[self._names[n]] = result
        return result


def _infer_input_names(model: nn.Module) -> list[str]:
    """Positional parameter names of model.forward (excluding self/*args/**kwargs)."""
    try:
        params = inspect.signature(model.forward).parameters
    except (TypeError, ValueError):
        return ["x"]
    names = [
        name
        for name, p in params.items()
        if p.kind
        in (
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
        )
    ]
    return names or ["x"]
