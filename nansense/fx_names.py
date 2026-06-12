"""Friendly, scope-qualified names for fx graph nodes.

`torch.fx.symbolic_trace` flattens a model into a single graph, so functional
ops written inside submodules — `torch.relu(...)`, `out + shortcut`,
`x.flatten(1)` — surface as `call_function` / `call_method` nodes whose fx
auto-names (`relu`, `relu_1`, `add`) say nothing about which submodule they
came from. A CIFAR ResNet with one `torch.relu` per block then produces a
pile of indistinguishable `relu`, `relu_1`, ... keys.

We recover the missing context from `node.meta["nn_module_stack"]` — the
module call stack fx records on every node — and prefix non-module ops with
the innermost submodule path, yielding readable keys like `stage1.0.relu1`.
These names are used both as snapshot/watch keys (`Session.layer_names`) and
as labels in the Mermaid graph, so the two always agree.

Naming rules (computed in one pass over the graph so per-scope numbering is
deterministic):

- `placeholder` / `output`     -> the fx node name (`x`, `output`)
- `call_module`                -> the dotted target path (`stage1.0.bn1`)
- `call_function` / `call_method` -> `<scope>.<op>`, where `<scope>` is the
  innermost submodule the op ran inside (`stage1.0.relu`). A 1-based index is
  appended only when a scope holds more than one op of that name, so the two
  relus of a block become `stage1.0.relu1` / `stage1.0.relu2` while a lone
  `add` stays `stage1.0.add`. Root-scope ops carry no prefix (`relu`,
  `flatten`).
- anything else (`get_attr`)   -> the fx node name (already unique)

A final pass forces the map to be *globally* unique: per-scope numbering only
disambiguates scoped ops within a scope, so a `call_module` `relu` can still
collide with a root-scope `torch.relu(...)` (both `relu`). The first node in
graph order keeps the bare name; later colliders get a stable `_N` suffix.
"""

from __future__ import annotations

from torch import fx

# Ops whose fx auto-name is uninformative and which we scope-qualify.
_SCOPED_OPS: frozenset[str] = frozenset({"call_function", "call_method"})


def _node_scope(node: fx.Node) -> str:
    """Dotted path of the innermost submodule the node ran inside ('' if root).

    `nn_module_stack` is an insertion-ordered mapping from a scope key to a
    `(module_path, module_type)` tuple, outermost first; the last entry is the
    innermost module. Older fx versions stored just the type as the value, so
    we fall back to the mapping key (which is the path) in that case.
    """
    stack = node.meta.get("nn_module_stack")
    if not stack:
        return ""
    key, value = next(reversed(stack.items()))
    if isinstance(value, (tuple, list)) and value:
        return str(value[0])
    return str(key)


def _op_base(node: fx.Node) -> str:
    """Readable base name for a function/method node ('relu', 'add', 'flatten')."""
    if node.op == "call_method":
        return str(node.target)
    name = getattr(node.target, "__name__", None)
    return name if name else node.name


def friendly_names(graph: fx.Graph) -> dict[fx.Node, str]:
    """Map every node in `graph` to its friendly, scope-qualified name.

    Keyed by node identity, so callers iterating the same graph (the session's
    capture interpreter, the Mermaid builder) get matching names.
    """
    # First pass: group scoped ops by (scope, base) so we know which names
    # collide within a scope and therefore need a numeric suffix.
    groups: dict[tuple[str, str], list[fx.Node]] = {}
    keys: dict[fx.Node, tuple[str, str]] = {}
    for node in graph.nodes:
        if node.op in _SCOPED_OPS:
            key = (_node_scope(node), _op_base(node))
            groups.setdefault(key, []).append(node)
            keys[node] = key

    names: dict[fx.Node, str] = {}
    counters: dict[tuple[str, str], int] = {}
    for node in graph.nodes:
        if node.op == "call_module":
            names[node] = str(node.target)
        elif node.op in _SCOPED_OPS:
            scope, base = keys[node]
            stem = f"{scope}.{base}" if scope else base
            if len(groups[(scope, base)]) > 1:
                counters[(scope, base)] = counters.get((scope, base), 0) + 1
                names[node] = f"{stem}{counters[(scope, base)]}"
            else:
                names[node] = stem
        else:
            names[node] = node.name
    return _dedupe(names)


def _dedupe(names: dict[fx.Node, str]) -> dict[fx.Node, str]:
    """Force global uniqueness across the whole map, deterministically.

    Per-scope numbering already disambiguates scoped ops *within* a scope, but
    a name can still collide across kinds — a `self.relu = nn.ReLU()`
    `call_module` named `relu` versus a root-scope `torch.relu(...)` named
    `relu`, or `self.flatten` versus `x.flatten(1)`. Those collisions silently
    merge layers (one snapshot/weight key clobbers the other), so we resolve
    them here. The first node (in graph iteration order) to claim a name keeps
    it unchanged; each later collider gets a stable `_N` suffix bumped until it
    is free. Non-colliding names — the common case (ResNet etc.) — are
    untouched, and the result is deterministic across runs.
    """
    used: set[str] = set()
    result: dict[fx.Node, str] = {}
    for node, name in names.items():
        unique = name
        suffix = 2
        while unique in used:
            unique = f"{name}_{suffix}"
            suffix += 1
        used.add(unique)
        result[node] = unique
    return result
