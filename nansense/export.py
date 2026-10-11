"""On-demand tensor export: a slice of the published snapshot as one `.pt` file.

The raw-numbers counterpart of the `render_*` pictures, for analysis the
debugger does not anticipate: an agent (`export_tensors`) or a person (the
Download buttons on a layer card and the weights page) asks for some layers,
and gets their tensors exactly as the last published `BatchSnapshot` holds
them — full batch, every channel, original dtype.

Nothing here runs on the training thread or touches disk: the selection is
built from the snapshot's existing CPU clones, and the bytes go to the caller
(an MCP blob resource, an HTTP download), so a run that never exports pays
nothing.

The payload is plain dicts of tensors, strings and numbers, loadable with
`torch.load(path, weights_only=True)`:

    {"format": 1,
     "position": {"phase": "train", "epoch": 1, "batch_idx": 12},
     "layers": {"conv1": {
         "activation": Tensor, "gradient": Tensor,
         "custom": {"<instrument>": Tensor},
         "parameters": {"conv1.weight": {
             "value": Tensor, "gradient": Tensor,
             "optimizer_state": {"exp_avg": Tensor, ...},
             "hyperparameters": {"lr": 0.001, ...},
             "custom": {"<instrument>": Tensor}}}}}}

A missing entry means the snapshot has none (no backward in an eval phase, a
frozen parameter, no optimizer passed to `start`). A parameter shared by two
exported layers is the same tensor object in both places, which `torch.save`
stores once.
"""

from __future__ import annotations

import io
import re
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch
from torch import Tensor

if TYPE_CHECKING:
    from nansense.session import BatchSnapshot, Session

#: Version of the payload layout above.
EXPORT_FORMAT = 1
#: Largest export served in one file, counted in raw tensor bytes. An MCP
#: reply carries the file base64-encoded inside JSON, so this bounds both
#: replies and downloads well below what would stall the server.
MAX_EXPORT_BYTES = 256 * 1024**2


class ExportError(ValueError):
    """Why an export could not be built; the message is shown to the user."""


@dataclass(frozen=True)
class SnapshotExport:
    """A built export: the payload, its manifest, and the file name to use."""

    filename: str
    payload: dict[str, Any]
    manifest: dict[str, Any]
    nbytes: int

    def to_bytes(self) -> bytes:
        buffer = io.BytesIO()
        torch.save(self.payload, buffer)
        return buffer.getvalue()


def build_export(
    session: Session,
    *,
    layers: Sequence[str],
    activations: bool = True,
    weights: bool = True,
    max_bytes: int = MAX_EXPORT_BYTES,
) -> SnapshotExport:
    """Select `layers`' tensors from the published snapshot.

    `activations` takes each layer's activation, its gradient and its custom
    layer tensors; `weights` takes the parameters the layer uses, with their
    gradients, optimizer state, hyperparameters and custom weight tensors.
    Unknown names are reported in the manifest's `unknown_layers`. Raises
    `ExportError` when there is nothing to export or it exceeds `max_bytes`.
    """
    if not (activations or weights):
        raise ExportError("Nothing selected: enable activations, weights, or both.")
    snapshot = session.snapshot
    if snapshot is None:
        raise ExportError(
            "No batch has been captured yet, or a snapshot is being updated. "
            "Pause or refresh the run, then try again."
        )
    known = set(session.layer_names)
    requested = list(dict.fromkeys(layers))
    selected = [name for name in requested if name in known]
    unknown = [name for name in requested if name not in known]
    if not selected:
        raise ExportError(
            f"No known layer among {requested}. Choose layer names from "
            "the architecture."
            if requested
            else "No layers given. Select at least one layer."
        )
    owned = session.layer_weights
    entries = {
        name: _layer_entry(
            snapshot,
            name,
            owned.get(name, []) if weights else [],
            activations=activations,
        )
        for name in selected
    }
    nbytes = _unique_nbytes(entries)
    if nbytes > max_bytes:
        raise ExportError(
            f"Selected tensor data totals {_mib(nbytes)}, over the "
            f"{_mib(max_bytes)} limit. Export fewer layers, or select only "
            "activations or weights."
        )
    position = {
        "phase": snapshot.position.phase,
        "epoch": snapshot.position.epoch,
        "batch_idx": snapshot.position.batch_idx,
    }
    payload = {"format": EXPORT_FORMAT, "position": position, "layers": entries}
    side = "" if activations and weights else (
        "-activations" if activations else "-weights"
    )
    filename = _filename(selected, side, position)
    manifest: dict[str, Any] = {
        "file": filename,
        "bytes": nbytes,
        "position": position,
        "load": "torch.load(path, weights_only=True)",
        "layers": _describe(entries),
    }
    if unknown:
        manifest["unknown_layers"] = unknown
    return SnapshotExport(
        filename=filename, payload=payload, manifest=manifest, nbytes=nbytes
    )


def _layer_entry(
    snapshot: BatchSnapshot,
    layer: str,
    parameters: list[str],
    *,
    activations: bool,
) -> dict[str, Any]:
    entry: dict[str, Any] = {}
    if activations:
        _put(entry, "activation", snapshot.activations.get(layer))
        _put(entry, "gradient", snapshot.activation_gradients.get(layer))
        custom = _tensors(snapshot.custom_activations.get(layer, {}))
        if custom:
            entry["custom"] = custom
    params = {name: _parameter_entry(snapshot, name) for name in parameters}
    if params:
        entry["parameters"] = params
    return entry


def _parameter_entry(snapshot: BatchSnapshot, name: str) -> dict[str, Any]:
    entry: dict[str, Any] = {}
    _put(entry, "value", snapshot.weights.get(name))
    _put(entry, "gradient", snapshot.weight_gradients.get(name))
    state = _tensors(snapshot.optimizer_state.get(name, {}))
    if state:
        entry["optimizer_state"] = state
    hyper = snapshot.optimizer_hyperparams.get(name)
    if hyper:
        entry["hyperparameters"] = dict(hyper)
    custom = _tensors(snapshot.custom_weight_tensors.get(name, {}))
    if custom:
        entry["custom"] = custom
    return entry


def _put(entry: dict[str, Any], key: str, value: object) -> None:
    # fx nodes can yield tuples or other non-tensor outputs; only tensors go.
    if isinstance(value, Tensor):
        entry[key] = value


def _tensors(values: dict[str, Tensor]) -> dict[str, Tensor]:
    return {k: v for k, v in values.items() if isinstance(v, Tensor)}


def _walk(tree: object) -> Iterator[Tensor]:
    if isinstance(tree, Tensor):
        yield tree
    elif isinstance(tree, dict):
        for value in tree.values():
            yield from _walk(value)


def _unique_nbytes(tree: object) -> int:
    seen: dict[int, int] = {
        id(t): t.numel() * t.element_size() for t in _walk(tree)
    }
    return sum(seen.values())


def _describe(tree: object) -> object:
    """The payload with every tensor replaced by its shape and dtype."""
    if isinstance(tree, Tensor):
        dtype = str(tree.dtype).removeprefix("torch.")
        return {"shape": list(tree.shape), "dtype": dtype}
    if isinstance(tree, dict):
        return {k: _describe(v) for k, v in tree.items()}
    return tree


def _filename(layers: list[str], side: str, position: dict[str, Any]) -> str:
    subject = layers[0] if len(layers) == 1 else f"{len(layers)}-layers"
    stem = (
        f"nansense-{subject}{side}-{position['phase']}"
        f"-e{position['epoch']}-b{position['batch_idx']}"
    )
    return re.sub(r"[^A-Za-z0-9._-]+", "_", stem) + ".pt"


def _mib(nbytes: int) -> str:
    return f"{nbytes / 1024**2:.1f} MiB"
