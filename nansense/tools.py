"""User-registered tools (`Session.tool`).

A tool is a plain function of a `ToolContext` plus typed keyword arguments.
It shows up as an MCP tool (so `nansense-cli` lists it, with `--help` built
from the signature and docstring) and on the UI's Tools page, whose form is
built from the same schema. Both front-ends go through `ToolManager`:
`request` queues a call and returns its seq, `call_state(seq)` reports where
it is and what it returned.

By default a call runs on the training thread while training is paused —
the pause loop in `Session._wait_for_proceed` drains the queue next to probes
and experiments — inside the same isolation as custom experiments, so the
model is restored afterwards. `on_training_thread=False` tools run at once on
a worker thread instead and must only read `ctx.snapshot` / `ctx.inputs`.
Registration only adds a registry entry, so it costs nothing per batch.
"""

from __future__ import annotations

import inspect
import math
import threading
import time
from collections import OrderedDict, deque
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import pydantic
import torch
from torch import Tensor

from nansense.experiments import Table
from nansense.extensions import (
    ModelMode,
    PauseContext,
    describe_error,
    restored_parameters,
    snapshot_inputs,
)
from nansense.probe import isolated_model

if TYPE_CHECKING:
    from nansense.session import BatchSnapshot, Session

ToolFn = Callable[..., Any]

# How many finished calls stay retrievable by seq (oldest evicted first).
_RESULTS_KEPT = 16
# A tool result tensor larger than this is refused rather than listed in JSON.
_TENSOR_ELEMENTS_MAX = 10_000


@dataclass(frozen=True)
class ToolContext(PauseContext):
    """Context for a custom tool, with the model, captured inputs and snapshot.

    Inherits `PauseContext`'s fields and methods. `snapshot` is the latest
    published `BatchSnapshot` when execution starts, with captured tensors
    on CPU, or `None` before the first capture.

    Training-thread tools receive inputs on the model's device and the
    current `batch_item`, when available. Worker-thread tools receive CPU
    inputs and `batch_item=None`; read only `snapshot` and `inputs`, without
    accessing the live model or calling `forward()` or `activation()`.
    """

    snapshot: BatchSnapshot | None = None


@dataclass(frozen=True)
class ToolSpec:
    """One registered tool. `signature` is the function's own, minus the
    context parameter, with annotations evaluated — what the MCP schema and
    the Tools page form are built from."""

    name: str
    description: str
    fn: ToolFn
    signature: inspect.Signature
    on_training_thread: bool
    mode: ModelMode
    timeout: float
    public: bool

    def json_schema(self) -> dict[str, Any]:
        """The arguments' JSON schema, as MCP describes them."""
        return _arguments_model(self.name, self.signature).model_json_schema()


@dataclass
class ToolCall:
    """Where one call is (`stage`: queued, running, done or absent) and, once
    done, its JSON-ready `value` or its `error`."""

    seq: int
    tool: str
    stage: str
    value: object = None
    error: str | None = None


@dataclass
class _Job:
    seq: int
    spec: ToolSpec
    kwargs: dict[str, object] = field(default_factory=dict)


def _arguments_model(name: str, signature: inspect.Signature) -> type[pydantic.BaseModel]:
    fields: dict[str, Any] = {}
    for param in signature.parameters.values():
        annotation = Any if param.annotation is inspect.Parameter.empty else param.annotation
        if param.default is inspect.Parameter.empty:
            fields[param.name] = (annotation, ...)
        else:
            fields[param.name] = (annotation, param.default)
    return pydantic.create_model(f"{name}_arguments", **fields)


def tool_spec(
    name: str,
    fn: ToolFn,
    *,
    description: str | None,
    on_training_thread: bool,
    mode: ModelMode,
    timeout: float,
    public: bool,
) -> ToolSpec:
    """Validate `fn` and build its registry entry (see `Session.tool`)."""
    if mode not in ("eval", "train", "unchanged"):
        raise ValueError("mode must be 'eval', 'train' or 'unchanged'")
    if not (math.isfinite(timeout) and timeout > 0):
        raise ValueError("timeout must be a finite, positive number of seconds")
    signature = inspect.signature(fn, eval_str=True)
    params = list(signature.parameters.values())
    if not params or params[0].kind not in (
        inspect.Parameter.POSITIONAL_ONLY,
        inspect.Parameter.POSITIONAL_OR_KEYWORD,
    ):
        raise ValueError(f"tool {name!r} must take a ToolContext as its first argument")
    arguments = params[1:]
    for param in arguments:
        if param.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
            raise ValueError(f"tool {name!r}: *args and **kwargs are not supported")
        if param.name.startswith("_"):
            raise ValueError(f"tool {name!r}: argument {param.name!r} must not start with '_'")
    signature = signature.replace(
        parameters=[p.replace(kind=inspect.Parameter.KEYWORD_ONLY) for p in arguments],
        return_annotation=dict[str, Any],
    )
    try:
        _arguments_model(name, signature).model_json_schema()
    except (pydantic.PydanticSchemaGenerationError, TypeError) as error:
        raise ValueError(
            f"tool {name!r}: argument types must have a JSON schema "
            f"(for example, int, float, bool, str, Literal or lists): {error}"
        ) from None
    return ToolSpec(
        name=name,
        description=description or inspect.getdoc(fn) or "",
        fn=fn,
        signature=signature,
        on_training_thread=on_training_thread,
        mode=mode,
        timeout=float(timeout),
        public=public,
    )


def jsonable(value: object) -> object:
    """A tool's return value as JSON-ready data (`TypeError` if it isn't)."""
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        if math.isnan(value):
            return "nan"
        if math.isinf(value):
            return "inf" if value > 0 else "-inf"
        return value
    if isinstance(value, Table):
        return {
            "columns": list(value.columns),
            "rows": [
                dict(zip(value.columns, (jsonable(c) for c in row), strict=True))
                for row in value.rows
            ],
        }
    if isinstance(value, Tensor):
        if value.numel() > _TENSOR_ELEMENTS_MAX:
            raise TypeError(
                f"Tool result tensor has {value.numel()} elements; the limit is "
                "10,000. Return a smaller tensor or use "
                "export_tensors for captured tensors."
            )
        return jsonable(value.detach().cpu().tolist())
    if isinstance(value, Mapping):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    raise TypeError(
        f"Tool returned unsupported type {type(value).__name__}. Return "
        "JSON-compatible values, tensors or a nansense.Table."
    )


class ToolManager:
    """Own registered tools, the call queue, and recent results."""

    def __init__(
        self,
        cv: threading.Condition,
        *,
        locked: Callable[[], bool],
        control_state: Callable[[], tuple[int, bool, bool]],
    ) -> None:
        self._cv = cv
        self._locked = locked
        self._control_state = control_state
        self._specs: dict[str, ToolSpec] = {}
        self._listeners: list[Callable[[ToolSpec], None]] = []
        self._queue: deque[_Job] = deque()
        self._running: _Job | None = None
        self._results: OrderedDict[int, ToolCall] = OrderedDict()
        self._seq = 0

    @property
    def specs(self) -> dict[str, ToolSpec]:
        """The tools callable here: all of them, or the `public` ones when
        the session is locked."""
        with self._cv:
            locked = self._locked()
            return {k: s for k, s in self._specs.items() if s.public or not locked}

    @property
    def all_names(self) -> frozenset[str]:
        """Every registered name, hidden ones included (names stay unique)."""
        with self._cv:
            return frozenset(self._specs)

    def register(self, spec: ToolSpec) -> None:
        # Listeners (a live MCP server) add the tool, and may reject a
        # built-in's name, before the registry commits to it. All under the
        # lock, so a server attaching mid-registration misses nothing.
        with self._cv:
            if spec.name in self._specs:
                raise ValueError(f"tool {spec.name!r} is already registered")
            for listener in self._listeners:
                listener(spec)
            self._specs[spec.name] = spec

    def add_listener(self, listener: Callable[[ToolSpec], None]) -> None:
        """Call `listener` for every tool, registered already or from now on."""
        with self._cv:
            for spec in self._specs.values():
                listener(spec)
            self._listeners.append(listener)

    @property
    def pending(self) -> bool:
        with self._cv:
            return bool(self._queue)

    def request(self, session: Session, name: str, kwargs: Mapping[str, object]) -> int:
        """Queue one call (or start it, for an off-thread tool); its seq."""
        spec = self.specs.get(name)
        if spec is None:
            raise ValueError(f"tool {name!r} is not registered or is unavailable in this session")
        validated = _arguments_model(name, spec.signature).model_validate(dict(kwargs))
        job = _Job(0, spec, {k: getattr(validated, k) for k in spec.signature.parameters})
        with self._cv:
            self._seq += 1
            job.seq = self._seq
            if spec.on_training_thread:
                self._queue.append(job)
                self._cv.notify_all()
                return job.seq
            self._store(ToolCall(job.seq, name, "running"))
        threading.Thread(
            target=self._run, args=(session, job), daemon=True, name=f"tool-{name}"
        ).start()
        return job.seq

    def take_pending(self) -> _Job | None:
        with self._cv:
            if not self._queue:
                return None
            job = self._queue.popleft()
            self._running = job
            return job

    def call_state(self, seq: int) -> ToolCall:
        with self._cv:
            if seq in self._results:
                return self._results[seq]
            if self._running is not None and self._running.seq == seq:
                return ToolCall(seq, self._running.spec.name, "running")
            for job in self._queue:
                if job.seq == seq:
                    return ToolCall(seq, job.spec.name, "queued")
            return ToolCall(seq, "", "absent")

    def run_guarded(self, session: Session, job: _Job) -> None:
        """Run one queued call on the (paused) training thread."""
        try:
            self._run(session, job)
        finally:
            with self._cv:
                self._running = None

    def _run(self, session: Session, job: _Job) -> None:
        resume_seen = self._control_state()[0]
        deadline = time.monotonic() + job.spec.timeout

        def should_abort() -> bool:
            resume, closed, jumping = self._control_state()
            return (
                time.monotonic() >= deadline
                or closed
                or jumping
                or (job.spec.on_training_thread and resume != resume_seen)
            )

        try:
            value = jsonable(self._call(session, job, should_abort))
            outcome = ToolCall(job.seq, job.spec.name, "done", value=value)
        except Exception as error:  # noqa: BLE001 — reported as the call's error
            outcome = ToolCall(job.seq, job.spec.name, "done", error=describe_error(error))
        with self._cv:
            self._store(outcome)
            self._cv.notify_all()

    def _call(
        self, session: Session, job: _Job, should_abort: Callable[[], bool]
    ) -> object:
        snapshot = session.snapshot
        if not job.spec.on_training_thread:
            ctx = ToolContext(
                session=session,
                model=session.model,
                device=torch.device("cpu"),
                inputs=snapshot_inputs(session, torch.device("cpu")),
                batch_item=None,
                _abort=should_abort,
                snapshot=snapshot,
            )
            return job.spec.fn(ctx, **job.kwargs)
        with isolated_model(session, job.spec.mode) as device, restored_parameters(
            session.model
        ):
            ctx = ToolContext(
                session=session,
                model=session.model,
                device=device,
                inputs=snapshot_inputs(session, device),
                batch_item=session._batch_item,
                _abort=should_abort,
                snapshot=snapshot,
            )
            return job.spec.fn(ctx, **job.kwargs)

    def _store(self, call: ToolCall) -> None:
        self._results[call.seq] = call
        self._results.move_to_end(call.seq)
        while len(self._results) > _RESULTS_KEPT:
            self._results.popitem(last=False)
