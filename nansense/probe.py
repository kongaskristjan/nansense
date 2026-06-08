"""Probe runs: nansense-internal forward passes between batches.

A probe re-runs the model on a fixed ("pinned") input batch so the UI can
show how the network's response to the *same* input evolves across stepping
and time-travel jumps — during normal training the displayed batch changes
every step because the loader reshuffles, which makes comparisons hard.

Probes can also carry *perturbations*: per-pixel edits applied to a copy of
the base input ("click to perturb" in the UI). A perturbed probe runs two
forwards — base and perturbed — so the UI can show the perturbed
activations or their diff against the original, e.g. to inspect how far a
single-pixel change propagates through the network (receptive field).

Probes execute on the training thread only (the model is never touched from
the UI thread): either right after a capture publishes its snapshot, or — for
requests arriving while training is paused — inside the pause loop in
`Session._wait_for_proceed`. A probe never mutates training state: per-module
`training` flags are saved and restored, buffers (BatchNorm running stats)
are restored, gradients are never produced (`torch.no_grad`), and the RNG is
forked so time-travel replays stay deterministic.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

PROBE_MODES: tuple[str, ...] = ("unchanged", "eval", "train")

# (sample, y, x) -> per-channel values in model-input (normalized) space.
PerturbationMap = dict[tuple[int, int, int], tuple[float, ...]]


@dataclass(frozen=True)
class ProbeResult:
    """One probe run's outputs, fully resident on CPU.

    Same thread contract as `BatchSnapshot`: all tensors are independent CPU
    clones, so the UI can hold the result for as long as it wants. `input` is
    the batch the probe ran on (the pinned input, or the snapshot's input
    when only perturbations are active), `activations` carries every layer
    output keyed like `Session.layer_names`, and `mode` records which
    train/eval mode the forward ran under ("unchanged", "eval", or "train").

    With perturbations applied, `perturbed_input` is the edited copy of
    `input` and `perturbed_activations` its layer outputs from a second
    forward in the same isolation scope; both stay `None` otherwise. Probes
    are forward-only: there are no activation gradients.
    """

    input: Tensor
    activations: dict[str, Tensor]
    mode: str
    perturbed_input: Tensor | None = None
    perturbed_activations: dict[str, Tensor] | None = None


def apply_perturbations(
    base: Tensor, perturbations: PerturbationMap
) -> Tensor | None:
    """Clone `base` and write per-channel values at each (sample, y, x) pixel.

    Returns `None` when there is nothing to apply: no perturbations, a
    non-image base (not `[B, C, H, W]`), or no entry in range. Entries out
    of bounds or with a mismatched channel count are skipped individually —
    the base batch may have changed shape since the click was recorded.
    """
    if not perturbations or base.ndim != 4:
        return None
    b, c, h, w = base.shape
    perturbed = base.clone()
    applied = False
    for (sample, y, x), values in perturbations.items():
        if not (0 <= sample < b and 0 <= y < h and 0 <= x < w):
            continue
        if len(values) != c:
            continue
        perturbed[sample, :, y, x] = torch.tensor(values, dtype=perturbed.dtype)
        applied = True
    return perturbed if applied else None
