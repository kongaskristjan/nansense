"""Build a Mermaid TD diagram of the compute graph.

`torch.fx.symbolic_trace` produces a true data-flow graph — for a ResNet
that means vertical chains with branches at each residual block — which is
what users expect when they ask for "the compute graph". If symbolic tracing
fails (data-dependent control flow, untraceable ops, etc.), we fall back to
the simpler parent->child module hierarchy.
"""

from __future__ import annotations

import re
from collections.abc import Iterator

import torch
from torch import fx, nn

from playgrad.fx_names import friendly_names

CONFIG_HEADER: str = """---
config:
  layout: elk
  theme: neutral
  look: neo
  elk:
    mergeEdges: true
---
"""

ROOT_ID: str = "root"


def build_mermaid(model: nn.Module, *, root_label: str = "model") -> str:
    """Return Mermaid source for the model's compute graph.

    Tries `torch.fx.symbolic_trace` first; on any tracing failure (a model
    with dynamic control flow, custom ops, etc.) falls back to the static
    module hierarchy tree rooted at a synthetic "root" node.
    """
    try:
        traced = fx.symbolic_trace(model)
    except Exception:
        return _build_from_hierarchy(model, root_label=root_label)
    return _build_from_fx(model, traced)


def _build_from_fx(model: nn.Module, traced: fx.GraphModule) -> str:
    # Friendly names key the node ids so each Mermaid node lines up with the
    # matching layer card (`data-layer=slug(layer_name)`); scope-qualified
    # function names also give relus etc. a unique, locatable label.
    names = friendly_names(traced.graph)
    lines: list[str] = [CONFIG_HEADER, "flowchart TD"]
    for node in traced.graph.nodes:
        lines.append(_node_def(node, model, names))
    for node in traced.graph.nodes:
        for arg in _node_inputs(node):
            lines.append(f"  {slug(names[arg])} --> {slug(names[node])}")
    return "\n".join(lines)


def _node_def(node: fx.Node, model: nn.Module, names: dict[fx.Node, str]) -> str:
    # Shape encodes node kind: circles for graph in/out, ovals (stadiums) for
    # weightless function/method calls (relu, add, ...), rectangles for modules.
    node_id = slug(names[node])
    if node.op == "placeholder":
        return f'  {node_id}(("in: {node.name}"))'
    if node.op == "output":
        return f'  {node_id}(("out"))'
    if node.op == "call_module":
        sub = model.get_submodule(str(node.target))
        label = f"{node.target}<br/>{type(sub).__name__}"
        return f'  {node_id}["{label}"]'
    if node.op == "call_function":
        return f'  {node_id}(["{names[node]}"])'
    if node.op == "call_method":
        return f'  {node_id}(["{names[node]}"])'
    return f'  {node_id}["{names[node]}"]'


def _node_inputs(node: fx.Node) -> Iterator[fx.Node]:
    for a in node.args:
        yield from _walk_fx_nodes(a)
    for v in node.kwargs.values():
        yield from _walk_fx_nodes(v)


def _walk_fx_nodes(value: object) -> Iterator[fx.Node]:
    if isinstance(value, fx.Node):
        yield value
        return
    if isinstance(value, (list, tuple)):
        for v in value:
            yield from _walk_fx_nodes(v)
        return
    if isinstance(value, dict):
        for v in value.values():
            yield from _walk_fx_nodes(v)


def _build_from_hierarchy(model: nn.Module, *, root_label: str) -> str:
    lines: list[str] = [CONFIG_HEADER, "flowchart TD", f'  {ROOT_ID}["{root_label}"]']
    for full_name, module in model.named_modules():
        if module is model:
            continue
        label = f"{full_name}<br/>{type(module).__name__}"
        lines.append(f'  {slug(full_name)}["{label}"]')
    for full_name, module in model.named_modules():
        parent_id = ROOT_ID if module is model else slug(full_name)
        for child_name, _ in module.named_children():
            child_full = f"{full_name}.{child_name}" if full_name else child_name
            lines.append(f"  {parent_id} --> {slug(child_full)}")
    return "\n".join(lines)


def slug(name: str) -> str:
    """Map a node/layer name to the id used in Mermaid sources and the DOM.

    Mermaid only accepts `[A-Za-z0-9_]` in node ids, so dots in dotted
    module paths (`stem.0`) become underscores (`stem_0`). The same slug
    is used on the per-layer card via `data-layer=...` so clicks on a
    Mermaid node can find their matching card.
    """
    return re.sub(r"[^A-Za-z0-9]", "_", name) or ROOT_ID
