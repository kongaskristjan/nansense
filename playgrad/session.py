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
from dataclasses import dataclass
from enum import StrEnum
from types import TracebackType
from typing import Self

from torch import Tensor, fx, nn
from torch.utils.hooks import RemovableHandle

from playgrad.fx_names import friendly_names
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

    All four tensor dicts are independent CPU clones taken at snapshot time,
    so the snapshot survives subsequent batches freeing the live tensors and
    can be safely read from any thread.
    """

    position: BatchPosition
    activations: dict[str, Tensor]
    activation_gradients: dict[str, Tensor]
    weights: dict[str, Tensor]
    weight_gradients: dict[str, Tensor]


class Session:
    def __init__(
        self,
        model: nn.Module,
        *,
        epochs: int,
        phases: dict[str, int],
        enabled: bool = True,
    ) -> None:
        self.model = model
        self._enabled = enabled
        self._schedule = Schedule(epochs=epochs, phases=phases)
        self._mode: Mode = Mode.STEP
        self._target_position: tuple[str, int, int] | None = None
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

    @property
    def schedule(self) -> Schedule:
        return self._schedule

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
        pre = self.model.register_forward_pre_hook(self._make_pre_hook())
        self._hook_handles.append(pre)
        for name, module in self.model.named_modules():
            if module is self.model:
                continue
            handle = module.register_forward_hook(self._make_hook(name))
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

    def _make_hook(self, name: str):
        def hook(_module: nn.Module, _inputs: object, output: object) -> None:
            if not isinstance(output, Tensor):
                return
            if output.requires_grad:
                output.retain_grad()
            self._activations[name] = output

        return hook

    def _make_pre_hook(self):
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
                self._activations[name] = inp

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
        )

    def _wait_for_proceed(self) -> None:
        with self._cv:
            seen = self._resume_token
            self._pause_count += 1
            self._cv.notify_all()
            while self._resume_token == seen and not self._closed:
                self._cv.wait()


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
        if self._position is None or not (self._captured or self._stats_only):
            return
        if exc is None and self._session._watched_layers:
            self._session._update_watch_stats(self._position)
        self._session._remove_hooks()
        if self._captured and exc is None and not self._session.closed:
            self._session._publish_snapshot(self._position)
            self._session._wait_for_proceed()
        self._session._activations.clear()

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
) -> Session:
    """Create a `Session` for `model`.

    With `enabled=False` the session is a near-zero-overhead no-op: no fx
    trace at construction, `batch()` does nothing, and `serve()` is skipped.
    This lets a training script keep its playgrad wiring in place and turn
    the whole UI off with a single flag.
    """
    return Session(model, epochs=epochs, phases=phases, enabled=enabled)


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
