"""Reading tensors off a model: activation capture and live introspection.

This module holds the machinery a `Session` uses to get tensors out of the
user's model, in three groups:

- **Construction-time discovery.** `try_trace` attempts the up-front
  `fx.symbolic_trace`; `compute_input_names` / `compute_layer_names` /
  `compute_layer_weights` / `compute_layer_info` derive the stable name
  universe the UI indexes into (graph inputs, module outputs, fx
  intermediates, the layer -> parameter-names map, and the per-layer
  hyperparameter strings).

- **Per-batch capture.** `install_hooks` / `remove_hooks` arm one batch's
  capture into `session._activations`: in fx mode by monkey-patching
  `model.forward` with a `_CaptureInterpreter` run, in the hook fallback
  with a root pre-hook plus per-module forward hooks. Captured tensors stay
  live and get `retain_grad()` so the user's `loss.backward()` populates
  `.grad` — no backward hooks needed.

- **Isolated forwards and live reads.** `capture_forward` runs one forward
  that clones every layer output to CPU as it is produced (the probe and
  experiment path; see `nansense.probe`), and the `current_*` functions
  read weights / gradients / optimizer state straight off the live model
  whenever they're called.

Functions taking a `session` parameter mutate or read `Session` capture
state; everything runs on the training thread (the model is never touched
from the UI thread).
"""

from __future__ import annotations

import inspect
from collections.abc import Callable
from contextlib import AbstractContextManager
from typing import TYPE_CHECKING

import torch
from torch import Tensor, fx, nn
from torch.optim import Optimizer

from nansense.fx_names import _op_base, friendly_names

if TYPE_CHECKING:
    from nansense.session import Session


def cpu_clone(t: Tensor) -> Tensor:
    return t.detach().to("cpu", copy=True)


def try_trace(model: nn.Module) -> fx.GraphModule | None:
    try:
        return fx.symbolic_trace(model)
    except Exception:
        return None


def compute_input_names(
    fx_graph: fx.GraphModule | None, model: nn.Module
) -> list[str]:
    if fx_graph is not None:
        return [n.name for n in fx_graph.graph.nodes if n.op == "placeholder"]
    return _infer_input_names(model)


def compute_layer_names(
    fx_graph: fx.GraphModule | None, model: nn.Module, input_names: list[str]
) -> list[str]:
    if fx_graph is not None:
        names = friendly_names(fx_graph.graph)
        return [names[n] for n in fx_graph.graph.nodes if n.op != "output"]
    return input_names + [
        name for name, m in model.named_modules() if m is not model
    ]


def compute_layer_weights(
    fx_graph: fx.GraphModule | None, model: nn.Module, input_names: list[str]
) -> dict[str, list[str]]:
    param_names = [name for name, _ in model.named_parameters()]
    if fx_graph is not None:
        return _fx_layer_weights(fx_graph, param_names)
    return _hook_layer_weights(model, param_names, input_names)


def _fx_layer_weights(
    fx_graph: fx.GraphModule, param_names: list[str]
) -> dict[str, list[str]]:
    param_set = set(param_names)
    names = friendly_names(fx_graph.graph)
    result: dict[str, list[str]] = {}
    for node in fx_graph.graph.nodes:
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


def compute_layer_info(
    fx_graph: fx.GraphModule | None, model: nn.Module, input_names: list[str]
) -> dict[str, str]:
    """Human-readable hyperparameter string per layer name ("" when none).

    Module layers report their `print(model)`-style signature —
    `ClassName(extra_repr())`, e.g. `Conv2d(3, 64, kernel_size=(3, 3), ...)`.
    `extra_repr` is PyTorch's universal hyperparameter surface: every
    built-in layer implements it and custom modules override it to join in.
    fx function/method ops report their literal (non-tensor) call arguments
    instead — `max_pool2d(2, stride=None, ...)`; ops whose inputs are all
    tensors (`relu`, `add`) and graph inputs have nothing to show and map
    to "". Keys match `compute_layer_names`.
    """
    if fx_graph is not None:
        return _fx_layer_info(fx_graph, model)
    result = {name: "" for name in input_names}
    for name, module in model.named_modules():
        if module is not model:
            result[name] = _module_info(module)
    return result


def _module_info(module: nn.Module) -> str:
    # Containers and custom modules may produce multi-line extra_reprs;
    # collapse the whitespace so the info string stays a single line.
    extra = " ".join(module.extra_repr().split())
    return f"{type(module).__name__}({extra})"


def _fx_layer_info(fx_graph: fx.GraphModule, model: nn.Module) -> dict[str, str]:
    names = friendly_names(fx_graph.graph)
    result: dict[str, str] = {}
    for node in fx_graph.graph.nodes:
        if node.op == "output":
            continue
        if node.op == "call_module":
            result[names[node]] = _module_info(
                model.get_submodule(str(node.target))
            )
        elif node.op in ("call_function", "call_method"):
            result[names[node]] = _fx_call_info(node)
        else:
            result[names[node]] = ""
    return result


def _fx_call_info(node: fx.Node) -> str:
    """Literal-argument signature of a function/method node ("" when none).

    Tensor inputs are data flow, not hyperparameters, so any argument that
    contains an `fx.Node` is dropped; what remains are the call's literal
    knobs (`kernel_size`, `stride`, a flatten dim, ...).
    """
    parts = [repr(a) for a in node.args if not _contains_fx_node(a)]
    parts += [
        f"{key}={value!r}"
        for key, value in node.kwargs.items()
        if not _contains_fx_node(value)
    ]
    if not parts:
        return ""
    return f"{_op_base(node)}({', '.join(parts)})"


def _contains_fx_node(value: object) -> bool:
    if isinstance(value, fx.Node):
        return True
    if isinstance(value, (list, tuple)):
        return any(_contains_fx_node(v) for v in value)
    if isinstance(value, dict):
        return any(_contains_fx_node(v) for v in value.values())
    return False


def _hook_layer_weights(
    model: nn.Module, param_names: list[str], input_names: list[str]
) -> dict[str, list[str]]:
    result: dict[str, list[str]] = {name: [] for name in input_names}
    for name, module in model.named_modules():
        if module is model:
            continue
        result[name] = _params_under(param_names, name)
    return result


def _params_under(param_names: list[str], target: str) -> list[str]:
    """Qualified parameter names owned by the module at dotted path `target`.

    Matches `target.*` (the params the module and its descendants hold). The
    bare `target` is included too for the degenerate case of a parameter
    registered directly under that name.
    """
    prefix = f"{target}."
    return sorted(p for p in param_names if p == target or p.startswith(prefix))


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


def install_hooks(session: Session) -> None:
    session._activations.clear()
    if session._fx_graph is not None:
        _patch_forward(session)
        return
    pre = session.model.register_forward_pre_hook(
        _make_pre_hook(session._input_names, session._activations)
    )
    session._hook_handles.append(pre)
    for name, module in session.model.named_modules():
        if module is session.model:
            continue
        handle = module.register_forward_hook(
            _make_hook(name, session._activations)
        )
        session._hook_handles.append(handle)


def remove_hooks(session: Session) -> None:
    if session._original_forward is not None:
        _unpatch_forward(session)
    for h in session._hook_handles:
        h.remove()
    session._hook_handles.clear()


def _patch_forward(session: Session) -> None:
    # Defense-in-depth: never overwrite an already-stashed original. If a
    # previous batch leaked its patch (e.g. an exception skipped
    # `remove_hooks`), `model.forward` is the stale fx_forward — capturing it
    # as the new "original" would permanently lose the real forward. The
    # normal install/remove cycle always clears `_original_forward` back to
    # None in `_unpatch_forward`, so this guard never trips in healthy runs.
    assert session._original_forward is None, (
        "forward already patched — a previous batch's hook removal leaked"
    )
    # Stash whatever .forward currently resolves to so we can put it back,
    # remembering whether it was an instance attribute or a class method.
    session._had_instance_forward = "forward" in session.model.__dict__
    session._original_forward = session.model.forward
    graph = session._fx_graph
    capture = session._activations
    assert graph is not None

    def fx_forward(*args: Tensor) -> object:
        # fx.Interpreter.run takes positional args matched to placeholder
        # order; kwargs aren't passed through.
        return _CaptureInterpreter(graph, capture).run(*args)

    object.__setattr__(session.model, "forward", fx_forward)


def _unpatch_forward(session: Session) -> None:
    if session._had_instance_forward and session._original_forward is not None:
        object.__setattr__(session.model, "forward", session._original_forward)
    elif "forward" in session.model.__dict__:
        object.__delattr__(session.model, "forward")
    session._original_forward = None
    session._had_instance_forward = False


def _make_hook(
    name: str, capture: dict[str, Tensor], *, to_cpu: bool = False
) -> Callable[[nn.Module, object, object], None]:
    # `to_cpu` mirrors `_CaptureInterpreter`: probe forwards clone each
    # output to CPU as it is produced instead of retaining live tensors.
    def hook(_module: nn.Module, _inputs: object, output: object) -> None:
        if not isinstance(output, Tensor):
            return
        if to_cpu:
            capture[name] = cpu_clone(output)
            return
        if output.requires_grad:
            output.retain_grad()
        capture[name] = output

    return hook


def _make_pre_hook(
    input_names: list[str], capture: dict[str, Tensor], *, to_cpu: bool = False
) -> Callable[[nn.Module, tuple[object, ...]], None]:
    def hook(_module: nn.Module, inputs: tuple[object, ...]) -> None:
        for i, inp in enumerate(inputs):
            if not isinstance(inp, Tensor):
                continue
            name = input_names[i] if i < len(input_names) else f"arg_{i}"
            if to_cpu:
                capture[name] = cpu_clone(inp)
                continue
            if inp.requires_grad:
                inp.retain_grad()
            capture[name] = inp

    return hook


class _CaptureInterpreter(fx.Interpreter):
    """fx interpreter that snapshots every node's tensor output.

    The interpreter runs the traced graph one node at a time and lets us
    intercept after each run. We retain_grad on every non-leaf tensor so
    the user's subsequent loss.backward() populates `.grad`, and store the
    live tensor under its friendly name in `capture`.

    With `to_cpu=True` (probe forwards, which never backward), each output
    is cloned to CPU as it is produced instead — so a whole model's worth
    of layer outputs never accumulates on the GPU during the run.
    """

    def __init__(
        self, gm: fx.GraphModule, capture: dict[str, Tensor], *, to_cpu: bool = False
    ) -> None:
        super().__init__(gm)
        self._capture = capture
        self._names = friendly_names(gm.graph)
        self._to_cpu = to_cpu

    def run_node(self, n: fx.Node) -> object:
        result = super().run_node(n)
        if n.op == "output":
            return result
        if isinstance(result, Tensor):
            if self._to_cpu:
                self._capture[self._names[n]] = cpu_clone(result)
            else:
                if result.requires_grad:
                    result.retain_grad()
                self._capture[self._names[n]] = result
        return result


def capture_forward(session: Session, inp: Tensor) -> dict[str, Tensor]:
    """One forward pass capturing every layer output as a fresh CPU clone.

    Never touches the batch path's state (`session._activations`,
    `session._hook_handles`, the patched forward): in fx mode the
    interpreter writes straight into a local dict, and in the hook fallback
    temporary hooks are registered and removed around the call. Safe because
    probes only run between batches, when the batch path's hooks are
    uninstalled.

    Captures clone to CPU eagerly (`to_cpu=True`): holding live outputs
    until the end of the forward would keep every layer's activation
    resident on the GPU at once — roughly a training forward's worth of
    memory on top of the training step's own.
    """
    capture: dict[str, Tensor] = {}
    if session._fx_graph is not None:
        _CaptureInterpreter(session._fx_graph, capture, to_cpu=True).run(inp)
    else:
        handles = [
            session.model.register_forward_pre_hook(
                _make_pre_hook(session._input_names, capture, to_cpu=True)
            )
        ]
        handles += [
            module.register_forward_hook(_make_hook(name, capture, to_cpu=True))
            for name, module in session.model.named_modules()
            if module is not session.model
        ]
        try:
            session.model(inp)
        finally:
            for handle in handles:
                handle.remove()
    return capture


def model_device(model: nn.Module) -> torch.device:
    param = next(model.parameters(), None)
    if param is not None:
        return param.device
    buffer = next(model.buffers(), None)
    if buffer is not None:
        return buffer.device
    return torch.device("cpu")


def fork_rng(device: torch.device) -> AbstractContextManager[None]:
    if device.type in ("cuda", "mps"):
        return torch.random.fork_rng(devices=[device], device_type=device.type)
    return torch.random.fork_rng(devices=[])


def current_weights(model: nn.Module) -> dict[str, Tensor]:
    """CPU clones of `model`'s parameters, read live at call time.

    Implementation of `Session.current_weights` (see its docstring for the
    read-race contract); also the snapshot path's weight copy.
    """
    return {n: cpu_clone(p) for n, p in model.named_parameters()}


def current_weight_gradients(model: nn.Module) -> dict[str, Tensor]:
    """CPU clones of the parameters' current `.grad` (`None` grads omitted).

    Implementation of `Session.current_weight_gradients`; also the snapshot
    path's gradient copy.
    """
    return {
        n: cpu_clone(p.grad)
        for n, p in model.named_parameters()
        if p.grad is not None
    }


def current_optimizer_state(
    model: nn.Module, optimizer: Optimizer | None
) -> dict[str, dict[str, Tensor]]:
    """Per-parameter optimizer state, read live at call time.

    Implementation of `Session.current_optimizer_state`. `optimizer.state`
    is keyed by the parameter object itself, so the mapping back to
    parameter names is an identity lookup — it works for any optimizer
    following the `torch.optim` convention with no per-optimizer code.
    Tensor entries are CPU-cloned; plain int/float entries become 0-dim
    tensors so the value type stays uniform.
    """
    if optimizer is None:
        return {}
    names = {id(p): n for n, p in model.named_parameters()}
    result: dict[str, dict[str, Tensor]] = {}
    for param, state in optimizer.state.items():
        name = names.get(id(param))
        if name is None:
            continue  # parameter from some other model
        entries: dict[str, Tensor] = {}
        for key, value in state.items():
            if isinstance(value, Tensor):
                entries[key] = cpu_clone(value)
            elif isinstance(value, (int, float)):
                entries[key] = torch.tensor(float(value))
        if entries:
            result[name] = entries
    return result


def current_optimizer_hyperparams(
    model: nn.Module, optimizer: Optimizer | None
) -> dict[str, dict[str, float]]:
    """Per-parameter numeric hyperparameters of the optimizer group.

    Implementation of `Session.current_optimizer_hyperparams`: maps each
    parameter name to the plain-numeric knobs of the param group it belongs
    to (`lr`, `momentum`, `weight_decay`, ...), read live.
    """
    if optimizer is None:
        return {}
    names = {id(p): n for n, p in model.named_parameters()}
    result: dict[str, dict[str, float]] = {}
    for group in optimizer.param_groups:
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
