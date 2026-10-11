"""User-registered experiment kinds (`Session.experiment`).

A custom kind is a function of an `ExperimentContext` that returns an
`ExperimentOutput` (or yields several, to stream progress). Registering one
builds an `ExperimentSpec` and adds it to the session's experiment registry,
after which it is an experiment like the built-ins: the experiment page offers
it with a form built from its `ExperimentParam` knobs, `list_experiments` /
`run_experiment` serve it to agents, and requests queue, cancel and publish
through the same `ExperimentManager`. Registration only adds a registry entry,
so it costs nothing per training batch.

User code runs on the training thread between batches, while paused or at a
snapshot capture for Live experiments. Training flags, existing buffers,
PyTorch RNG state, existing parameter values and their `.grad` are restored
afterwards. Keep the model structure intact; optimizer state and other side
effects are not restored.
"""

from __future__ import annotations

import inspect
import re
import traceback
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import TYPE_CHECKING, Literal

import torch
from torch import Tensor, fx, nn

from nansense.contracts.experiments import CustomParams, finite_number, normalization
from nansense.experiments import (
    ExperimentParam,
    ExperimentRequest,
    ExperimentResult,
    ExperimentSpec,
    Table,
    _error,
)
from nansense.fx_names import friendly_names
from nansense.probe import isolated_model

if TYPE_CHECKING:
    from nansense.session import Session

ModelMode = Literal["eval", "train", "unchanged"]
LayerEdit = Callable[[str, Tensor], Tensor | None]
ExperimentFn = Callable[
    ["ExperimentContext"], "ExperimentOutput | Iterator[ExperimentOutput]"
]

_NAME = re.compile(r"[a-z][a-z0-9_]*\Z")
_KNOB_KINDS = ("int", "float", "bool", "select")
# Rows a published result table keeps; the rest are cut, and the note says so.
_TABLE_ROWS_KEPT = 2000


@dataclass(frozen=True)
class PauseContext:
    """The model and captured inputs handed to user code between batches.

    `inputs` are the last snapshot's model inputs, moved to `device`, in the
    model's argument order. `batch_item` is the current loader item (targets
    included), available while its batch context publishes or pauses. It is
    `None` outside that context or when no loader item was supplied.
    """

    session: Session
    model: nn.Module
    device: torch.device
    inputs: tuple[Tensor, ...]
    batch_item: object | None
    _abort: Callable[[], bool] = field(repr=False)

    def should_abort(self) -> bool:
        """Return whether cancellation, the time limit, or a resume, rewind or
        close request requires the run to stop. Check it inside long loops;
        stopping is cooperative."""
        return self._abort()

    def forward(self, *inputs: Tensor, edit: LayerEdit | None = None) -> object:
        """Run the model on `inputs` (default: `self.inputs`).

        `edit(layer, activation)` receives tensor outputs as they are produced:
        graph nodes when the model is traced, named modules otherwise. Return
        a tensor to replace that output for the rest of the forward (ablation,
        activation patching), or `None` to keep it.
        """
        args = inputs or self.inputs
        if edit is None:
            return self.model(*args)
        return _edited_forward(self.session, args, edit)

    def activation(self, layer: str, *inputs: Tensor) -> Tensor:
        """Rerun the model and return `layer`'s tensor output for `inputs`
        (default: `self.inputs`). Its autograd graph is preserved when
        gradients are enabled."""
        captured: dict[str, Tensor] = {}

        def record(name: str, act: Tensor) -> None:
            if name == layer:
                captured[name] = act

        self.forward(*inputs, edit=record)
        if layer not in captured:
            raise ValueError(
                f"layer {layer!r} produced no tensor output; choose a layer "
                "name from the architecture view"
            )
        return captured[layer]


@dataclass(frozen=True)
class ExperimentContext(PauseContext):
    """A `PauseContext` plus the request: the chosen `layer` and the validated
    knob values in `params`, keyed by `ExperimentParam.key`."""

    layer: str = ""
    params: Mapping[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class ExperimentOutput:
    """What a custom experiment returns (or yields, to stream progress).

    Set at least one payload. `image` is an input-space batch shown like the
    model's input (denormalized); `attribution` is a signed per-sample map
    shown with the diverging colormap, beside `reference` (the inputs it
    explains). `table`, `scalars` (headline numbers) and `note` (free text)
    show as numbers and text. `step` / `total_steps` label streamed progress;
    by default each yield counts as one step. Each streamed output publishes
    when the next arrives; the final one publishes after model restoration.
    """

    image: Tensor | None = None
    attribution: Tensor | None = None
    reference: Tensor | None = None
    table: Table | None = None
    scalars: Mapping[str, float] = field(default_factory=dict)
    note: str | None = None
    step: int | None = None
    total_steps: int | None = None


def _edited_forward(
    session: Session, inputs: Sequence[Tensor], edit: LayerEdit
) -> object:
    gm = session._fx_graph
    if gm is not None:
        return _EditingInterpreter(gm, edit).run(*inputs)
    handles = []
    for name, module in session.model.named_modules():
        if name:
            handles.append(module.register_forward_hook(_edit_hook(name, edit)))
    try:
        return session.model(*inputs)
    finally:
        for handle in handles:
            handle.remove()


def _edit_hook(name: str, edit: LayerEdit) -> Callable[..., Tensor | None]:
    def hook(_module: nn.Module, _inputs: object, output: object) -> Tensor | None:
        return edit(name, output) if isinstance(output, Tensor) else None

    return hook


class _EditingInterpreter(fx.Interpreter):
    """Runs the traced graph, letting `edit` replace any node's tensor output."""

    def __init__(self, gm: fx.GraphModule, edit: LayerEdit) -> None:
        super().__init__(gm)
        self._edit = edit
        self._names = friendly_names(gm.graph)

    def run_node(self, n: fx.Node) -> object:
        result = super().run_node(n)
        if n.op == "output" or not isinstance(result, Tensor):
            return result
        replaced = self._edit(self._names[n], result)
        return result if replaced is None else replaced


@contextmanager
def _restored_parameters(model: nn.Module) -> Iterator[None]:
    """Put every parameter's value and `.grad` back on exit.

    Copies in place when the shape still matches, so parameters keep their
    storage (DDP gradient buckets, optimizer references); a replaced tensor
    is reassigned instead.
    """
    saved = [
        (p, p.detach().clone(), None if p.grad is None else p.grad.detach().clone())
        for p in model.parameters()
    ]
    try:
        yield
    finally:
        with torch.no_grad():
            for param, data, grad in saved:
                if param.shape == data.shape:
                    param.copy_(data)
                else:
                    param.data = data
                if grad is None or param.grad is None or param.grad.shape != grad.shape:
                    param.grad = grad
                else:
                    param.grad.copy_(grad)


def check_name(name: str, taken: Iterable[str], what: str) -> None:
    """Raise `ValueError` unless `name` is a fresh `snake_case` identifier."""
    if not _NAME.match(name):
        raise ValueError(
            f"{what} name {name!r} must be lowercase snake_case "
            "(letters, digits and underscores, starting with a letter)"
        )
    if name in taken:
        raise ValueError(f"{what} {name!r} already exists")


def _check_knobs(params: Sequence[ExperimentParam]) -> None:
    keys: set[str] = set()
    for param in params:
        if not isinstance(param, ExperimentParam):
            raise TypeError(f"params must be ExperimentParam, got {param!r}")
        if param.key in keys or param.key in ("mean", "std"):
            raise ValueError(f"knob key {param.key!r} is taken")
        keys.add(param.key)
        if param.kind not in _KNOB_KINDS:
            raise ValueError(f"knob {param.key!r}: kind must be one of {_KNOB_KINDS}")
        default = param.default
        if param.kind == "bool":
            ok = isinstance(default, bool)
        elif param.kind == "select":
            ok = bool(param.options) and default in (param.options or {})
        else:
            ok = isinstance(default, (int, float)) and not isinstance(default, bool)
        if not ok:
            raise ValueError(
                f"knob {param.key!r} needs a default matching kind "
                f"{param.kind!r}; for 'select', use one of the option values"
            )


def custom_experiment_spec(
    name: str,
    fn: ExperimentFn,
    *,
    label: str | None,
    summary: str,
    description: str,
    params: Sequence[ExperimentParam],
    layers: Literal["any", "modules"],
    mode: ModelMode,
    locked_limits: Mapping[str, float] | None,
) -> ExperimentSpec:
    """The registry entry for `Session.experiment` (validated here)."""
    knobs = tuple(params)
    _check_knobs(knobs)
    if layers not in ("any", "modules"):
        raise ValueError("layers must be 'any' or 'modules'")
    if mode not in ("eval", "train", "unchanged"):
        raise ValueError("mode must be 'eval', 'train' or 'unchanged'")
    doc = inspect.getdoc(fn) or ""
    limits = None if locked_limits is None else MappingProxyType(dict(locked_limits))
    return ExperimentSpec(
        name=name,
        label=label or name,
        summary=summary or doc.split("\n", 1)[0],
        description=description or doc,
        params=knobs,
        runner=lambda session, request, abort: _run_custom(
            session, request, abort, fn, mode
        ),
        parse=lambda values, locked: _parse(name, knobs, limits, values, locked),
        needs_module=layers == "modules",
        custom=True,
        locked_limits=limits,
    )


def _parse(
    kind: str,
    knobs: tuple[ExperimentParam, ...],
    limits: Mapping[str, float] | None,
    values: Mapping[str, object],
    locked: bool,
) -> CustomParams:
    unknown = values.keys() - {k.key for k in knobs} - {"mean", "std"}
    if unknown:
        raise ValueError(f"unknown parameters for {kind}: {sorted(unknown)}")
    parsed: dict[str, object] = {}
    for knob in knobs:
        value = values.get(knob.key, knob.default)
        if knob.kind == "bool":
            if not isinstance(value, bool):
                raise ValueError(f"{knob.key} must be a boolean")
        elif knob.kind == "select":
            if value not in (knob.options or {}):
                raise ValueError(f"{knob.key} must be one of {sorted(knob.options or {})}")
        else:
            number = finite_number(value, knob.key)
            if knob.minimum is not None:
                number = max(knob.minimum, number)
            if locked and limits is not None and knob.key in limits:
                number = min(limits[knob.key], number)
            value = int(number) if knob.kind == "int" else float(number)
        parsed[knob.key] = value
    return CustomParams(
        kind=kind,
        values=MappingProxyType(parsed),
        mean=normalization(values.get("mean"), "mean"),
        std=normalization(values.get("std"), "std"),
    )


def snapshot_inputs(session: Session, device: torch.device) -> tuple[Tensor, ...]:
    """The last snapshot's model inputs on `device`, in argument order."""
    inputs = session._snapshot_inputs()
    return tuple(inputs[n].to(device) for n in session.input_names if n in inputs)


def describe_error(error: Exception) -> str:
    """`Type: message`, plus where in the user's code it was raised."""
    text = f"{type(error).__name__}: {error}"
    frames = traceback.extract_tb(error.__traceback__)
    if frames:
        last = frames[-1]
        text += f" (at {last.filename}:{last.lineno} in {last.name})"
    return text


def _run_custom(
    session: Session,
    request: ExperimentRequest,
    should_abort: Callable[[], bool],
    fn: ExperimentFn,
    mode: ModelMode,
) -> Iterator[ExperimentResult]:
    """Drive one custom experiment, streaming all but its last output.

    The last output publishes only after the isolation scope has restored the
    model — waiters wake on the final result and must not find the model
    still edited (see `probe.isolated_model`).
    """
    params = request.params
    assert isinstance(params, CustomParams)
    last: ExperimentOutput | None = None
    step = 0
    try:
        with isolated_model(session, mode) as device, _restored_parameters(
            session.model
        ):
            ctx = ExperimentContext(
                session=session,
                model=session.model,
                device=device,
                inputs=snapshot_inputs(session, device),
                batch_item=session._batch_item,
                _abort=should_abort,
                layer=request.layer,
                params=params.values,
            )
            produced = fn(ctx)
            outputs = produced if isinstance(produced, Iterator) else iter([produced])
            for output in outputs:
                if last is not None:
                    yield _result(request, last, step, done=False)
                step += 1
                last = _checked(output)
                if should_abort():
                    break
    except Exception as error:  # noqa: BLE001 — published as the run's error
        yield _error(request, describe_error(error))
        return
    if last is None:
        yield _error(
            request, "the experiment yielded no ExperimentOutput; yield at least one result"
        )
        return
    yield _result(request, last, step, done=True)


def _checked(output: object) -> ExperimentOutput:
    if not isinstance(output, ExperimentOutput):
        raise TypeError(
            f"an experiment must return or yield nansense.ExperimentOutput, got "
            f"{type(output).__name__}"
        )
    if (
        output.image is None
        and output.attribution is None
        and output.table is None
        and not output.scalars
        and not output.note
    ):
        raise ValueError(
            "the ExperimentOutput is empty; set image, attribution, table, scalars or note"
        )
    return output


def _cpu(tensor: Tensor | None) -> Tensor | None:
    return None if tensor is None else tensor.detach().cpu().float()


def _result(
    request: ExperimentRequest, output: ExperimentOutput, step: int, *, done: bool
) -> ExperimentResult:
    table = output.table
    note = output.note
    if table is not None and len(table.rows) > _TABLE_ROWS_KEPT:
        cut = f"(table cut to its first {_TABLE_ROWS_KEPT} of {len(table.rows)} rows)"
        note = cut if not note else f"{note} {cut}"
        table = Table(table.columns, table.rows[:_TABLE_ROWS_KEPT])
    shown = output.step if output.step is not None else step
    return ExperimentResult(
        seq=request.seq,
        kind=request.kind,
        layer=request.layer,
        step=shown,
        total_steps=output.total_steps if output.total_steps is not None else shown,
        done=done,
        image=_cpu(output.image),
        attribution=_cpu(output.attribution),
        reference=_cpu(output.reference),
        table=table,
        scalars={key: float(value) for key, value in output.scalars.items()},
        note=note,
    )
