"""Probe runs: playgrad-internal forward passes between batches.

A probe re-runs the model on a fixed ("pinned") input batch so the UI can
show how the network's response to the *same* input evolves across stepping
and time-travel jumps — during normal training the displayed batch changes
every step because the loader reshuffles, which makes comparisons hard.

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

from torch import Tensor

PROBE_MODES: tuple[str, ...] = ("unchanged", "eval", "train")


@dataclass(frozen=True)
class ProbeResult:
    """One probe run's outputs, fully resident on CPU.

    Same thread contract as `BatchSnapshot`: all tensors are independent CPU
    clones, so the UI can hold the result for as long as it wants. `input` is
    the batch the probe ran on (the pinned input), `activations` carries every
    layer output keyed like `Session.layer_names`, and `mode` records which
    train/eval mode the forward ran under ("unchanged", "eval", or "train").
    Probes are forward-only: there are no activation gradients.
    """

    input: Tensor
    activations: dict[str, Tensor]
    mode: str
