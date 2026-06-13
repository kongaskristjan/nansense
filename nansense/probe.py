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

The probe config lives on the `Session` (`_pinned_input`, `_perturbations`,
`_probe_mode`, the version/request/count fields, all under `Session._cv`);
this module implements every transition of that state — the pin /
perturbation / mode setters behind the public `Session` methods — plus the
probe runs themselves. `isolated_model` is the isolation contract shared
with `nansense.experiments`.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from torch import Tensor

from nansense.capture import fork_rng, model_device

if TYPE_CHECKING:
    from nansense.session import Session

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


def pin_current_batch(session: Session) -> bool:
    """Implementation of `Session.pin_current_batch`."""
    if not session._enabled:
        return False
    snap = session._snapshot
    input_name = session._input_names[0] if session._input_names else None
    if snap is None or input_name is None:
        return False
    pinned = snap.activations.get(input_name)
    if pinned is None:
        return False
    with session._cv:
        session._pinned_input = pinned
        session._pinned_position = snap.position
        request_probe_locked(session)
    return True


def unpin_batch(session: Session) -> None:
    """Implementation of `Session.unpin_batch`."""
    with session._cv:
        if session._pinned_input is None:
            return
        session._pinned_input = None
        session._pinned_position = None
        if session._perturbations:
            # Perturbations keep probing, now against the snapshot input.
            request_probe_locked(session)
            return
        _clear_probe_result_locked(session)


def add_perturbation(
    session: Session, *, sample: int, y: int, x: int, values: tuple[float, ...]
) -> None:
    """Implementation of `Session.add_perturbation`."""
    if not session._enabled:
        return
    with session._cv:
        session._perturbations[(sample, y, x)] = tuple(values)
        request_probe_locked(session)


def clear_perturbations(session: Session) -> None:
    """Implementation of `Session.clear_perturbations`."""
    with session._cv:
        if not session._perturbations:
            return
        session._perturbations.clear()
        if session._pinned_input is not None:
            request_probe_locked(session)
            return
        _clear_probe_result_locked(session)


def set_probe_mode(session: Session, mode: str) -> None:
    """Implementation of `Session.set_probe_mode`."""
    if mode not in PROBE_MODES:
        raise ValueError(
            f"unknown probe mode {mode!r}; expected one of {PROBE_MODES}"
        )
    with session._cv:
        if mode == session._probe_mode:
            return
        session._probe_mode = mode
        if _probe_active_locked(session):
            request_probe_locked(session)
        else:
            session._probe_version += 1


def _clear_probe_result_locked(session: Session) -> None:
    """Deactivate probing and drop the published result (caller holds `_cv`)."""
    session._probe_version += 1
    session._probe_request = False
    session._probe_result = None
    session._probe_error = None
    session._cv.notify_all()


def _probe_active_locked(session: Session) -> bool:
    """Whether probe runs should happen at all (caller holds `_cv`)."""
    return session._pinned_input is not None or bool(session._perturbations)


def request_probe_locked(session: Session) -> None:
    """Arm a probe run and wake a paused training thread (caller holds `_cv`)."""
    session._probe_version += 1
    session._probe_request = True
    session._cv.notify_all()


def maybe_run_probe_at_capture(session: Session) -> None:
    """Run a probe right after a capture published its snapshot.

    Called by `_BatchContext.__exit__` before the pause, so every pause
    shows a probe result consistent with the just-captured weights. Any
    UI request armed in the meantime is consumed here — the run below
    uses the current config either way.
    """
    with session._cv:
        session._probe_request = False
        active = _probe_active_locked(session)
    if active:
        run_probe_guarded(session)


def run_probe_guarded(session: Session) -> None:
    # A failing probe (bad input, OOM, model quirk) must not kill the
    # training thread or wedge the pause loop; the error is published
    # for the UI to display instead.
    with session._cv:
        version = session._probe_version
    try:
        _run_probe(session)
    except Exception as e:  # noqa: BLE001 — surfaced via probe_error
        with session._cv:
            # Mirror the success path's staleness guard: a config change
            # mid-run (re-pin, mode flip, un-pin) bumps the version and arms
            # its own probe, so a superseded run must not leave a stuck error
            # behind that newer config — especially when the new config makes
            # probing inactive and nothing else clears it.
            if session._probe_version != version:
                return
            session._probe_error = f"{type(e).__name__}: {e}"
            session._probe_count += 1
            session._cv.notify_all()


def _run_probe(session: Session) -> None:
    """One probe run: isolated forwards on the base (and perturbed) input.

    Training-thread only. Reads the probe config under `_cv`, runs the
    forwards without the lock, and publishes the result only if the
    config is still current — a config change mid-run (re-pin, mode flip,
    new perturbation) wins and its own request re-runs the probe. The
    base input is the pinned batch, or the snapshot's input when only
    perturbations are active.
    """
    with session._cv:
        version = session._probe_version
        pinned = session._pinned_input
        mode = session._probe_mode
        perturbations = dict(session._perturbations)
    if pinned is None and not perturbations:
        return
    base = pinned if pinned is not None else session._snapshot_input()
    if base is None:
        return
    perturbed = apply_perturbations(base, perturbations)
    inputs = [base] if perturbed is None else [base, perturbed]
    captures = _probe_forwards(session, inputs, mode=mode)
    result = ProbeResult(
        input=base,
        activations=captures[0],
        mode=mode,
        perturbed_input=perturbed,
        perturbed_activations=captures[1] if perturbed is not None else None,
    )
    with session._cv:
        if session._probe_version != version:
            return
        session._probe_result = result
        session._probe_error = None
        session._probe_count += 1
        session._cv.notify_all()


@contextmanager
def isolated_model(session: Session, mode: str) -> Iterator[torch.device]:
    """Run model inference without mutating training state.

    The shared isolation contract of probes and experiments:

    - Per-module `training` flags are saved and restored ("eval"/"train"
      flip the whole model; "unchanged" runs with whatever the loop set).
    - Every buffer is restored afterwards (a train-mode BatchNorm forward
      updates running stats in place).
    - The RNG is forked, so e.g. train-mode dropout doesn't perturb the
      global stream that time-travel replays depend on.

    Callers add their own gradient policy: probes wrap the body in
    `torch.no_grad()`; experiments take input gradients via
    `torch.autograd.grad`, which leaves parameter `.grad` untouched.
    Yields the model's device.
    """
    device = model_device(session.model)
    saved_flags = [(m, m.training) for m in session.model.modules()]
    saved_buffers = [
        (b, b.detach().clone()) for _, b in session.model.named_buffers()
    ]
    try:
        if mode == "eval":
            session.model.eval()
        elif mode == "train":
            session.model.train()
        with fork_rng(device):
            yield device
    finally:
        for module, flag in saved_flags:
            module.training = flag
        with torch.no_grad():
            for buffer, saved in saved_buffers:
                buffer.copy_(saved)


def _probe_forwards(
    session: Session, inputs: list[Tensor], *, mode: str
) -> list[dict[str, Tensor]]:
    """Run isolated no-grad forwards, capturing every layer's output."""
    with isolated_model(session, mode) as device, torch.no_grad():
        return [session._capture_forward(inp.to(device)) for inp in inputs]
