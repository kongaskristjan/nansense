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

# (input_name, sample, index) -> values in model-input (normalized) space.
# `index` is (y, x) for an image input `[B, C, H, W]` and `values` has length
# C (the whole channel vector of that pixel); `index` is (channel,) for a flat
# input `[B, C]` and `values` is a single scalar.
PerturbationKey = tuple[str, int, tuple[int, ...]]
PerturbationMap = dict[PerturbationKey, tuple[float, ...]]


@dataclass(frozen=True)
class ProbeResult:
    """One probe run's outputs, fully resident on CPU.

    Same thread contract as `BatchSnapshot`: all tensors are independent CPU
    clones, so the UI can hold the result for as long as it wants. `inputs`
    maps every model input name to the tensor the probe ran on (the pinned
    batch, or the snapshot's inputs when only perturbations are active);
    `activations` carries every layer output keyed like `Session.layer_names`,
    and `mode` records which train/eval mode the forward ran under
    ("unchanged", "eval", or "train").

    With perturbations applied, `perturbed_inputs` is the per-input mapping
    with the edited input(s) substituted (unperturbed inputs share the base
    tensor), and `perturbed_activations` its layer outputs from a second
    forward in the same isolation scope; both stay `None` otherwise. Probes
    are forward-only: there are no activation gradients.
    """

    inputs: dict[str, Tensor]
    activations: dict[str, Tensor]
    mode: str
    perturbed_inputs: dict[str, Tensor] | None = None
    perturbed_activations: dict[str, Tensor] | None = None

    def shown_input(self, name: str | None) -> Tensor | None:
        """The (perturbed if edited, else base) tensor for input `name`."""
        if name is None:
            return None
        if self.perturbed_inputs is not None and name in self.perturbed_inputs:
            return self.perturbed_inputs[name]
        return self.inputs.get(name)

    def base_input(self, name: str | None) -> Tensor | None:
        """The unperturbed base tensor for input `name`."""
        return self.inputs.get(name) if name is not None else None

    def batch_size(self) -> int | None:
        """Batch size `B`, read off whichever input has a batch axis."""
        for tensor in self.inputs.values():
            if tensor.ndim > 0:
                return int(tensor.shape[0])
        return None


def apply_perturbations(
    bases: dict[str, Tensor], perturbations: PerturbationMap
) -> dict[str, Tensor] | None:
    """Substitute edited copies of the perturbed inputs into a new mapping.

    Every input carrying at least one in-range perturbation is cloned and
    edited; the rest reuse their base tensor (the returned dict shares those
    objects). Returns `None` when nothing applies: no perturbations, or no
    entry lands in any base. Per input shape:

    - image `[B, C, H, W]`: index `(y, x)` writes the length-`C` `values`
      across the channel axis of pixel `(y, x)`.
    - flat `[B, C]`: index `(channel,)` writes the single `values[0]` scalar.

    Entries out of bounds, naming an absent input, or with a value/channel
    count that doesn't fit the base are skipped individually — the base batch
    may have changed shape since the click was recorded.
    """
    if not perturbations:
        return None
    result: dict[str, Tensor] = dict(bases)
    applied = False
    for (name, sample, index), values in perturbations.items():
        base = bases.get(name)
        if base is None:
            continue
        # Clone lazily on the first hit for an input; later hits edit in place.
        current = result[name]
        target = current if current is not base else base.clone()
        if _write_perturbation(target, sample, index, values):
            result[name] = target
            applied = True
    return result if applied else None


def _write_perturbation(
    target: Tensor, sample: int, index: tuple[int, ...], values: tuple[float, ...]
) -> bool:
    """Write one perturbation into `target` in place; `False` if it doesn't fit."""
    if target.ndim == 4:
        b, c, h, w = target.shape
        if len(index) != 2 or len(values) != c:
            return False
        y, x = index
        if not (0 <= sample < b and 0 <= y < h and 0 <= x < w):
            return False
        target[sample, :, y, x] = torch.tensor(values, dtype=target.dtype)
        return True
    if target.ndim == 2:
        b, c = target.shape
        if len(index) != 1 or len(values) != 1:
            return False
        (channel,) = index
        if not (0 <= sample < b and 0 <= channel < c):
            return False
        target[sample, channel] = values[0]
        return True
    return False


def pin_current_batch(session: Session) -> bool:
    """Implementation of `Session.pin_current_batch`."""
    if not session._enabled:
        return False
    snap = session._snapshot
    if snap is None:
        return False
    pinned = session._snapshot_inputs()
    if not pinned:
        return False
    with session._cv:
        session._pinned_inputs = pinned
        session._pinned_position = snap.position
        request_probe_locked(session)
    return True


def unpin_batch(session: Session) -> None:
    """Implementation of `Session.unpin_batch`."""
    with session._cv:
        if session._pinned_inputs is None:
            return
        session._pinned_inputs = None
        session._pinned_position = None
        if _probe_active_locked(session):
            # Perturbations or an "eval"/"train" mode keep probing, now
            # against the snapshot input.
            request_probe_locked(session)
            return
        _clear_probe_result_locked(session)


def add_perturbation(
    session: Session,
    *,
    input_name: str,
    sample: int,
    index: tuple[int, ...],
    values: tuple[float, ...],
) -> None:
    """Implementation of `Session.add_perturbation`."""
    if not session._enabled:
        return
    with session._cv:
        session._perturbations[(input_name, sample, tuple(index))] = tuple(values)
        request_probe_locked(session)


def clear_perturbations(session: Session) -> None:
    """Implementation of `Session.clear_perturbations`."""
    with session._cv:
        if not session._perturbations:
            return
        session._perturbations.clear()
        if _probe_active_locked(session):
            # A pin or an "eval"/"train" mode keeps probing without the
            # cleared perturbations.
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
            # Selecting "eval"/"train" (or changing mode while pinned /
            # perturbed) re-runs the probe under the new mode.
            request_probe_locked(session)
        else:
            # Back to "unchanged" with nothing else probing: drop the stale
            # eval/train result so the UI reverts to the live snapshot.
            _clear_probe_result_locked(session)


def _clear_probe_result_locked(session: Session) -> None:
    """Deactivate probing and drop the published result (caller holds `_cv`)."""
    session._probe_version += 1
    session._probe_request = False
    session._probe_result = None
    session._probe_error = None
    session._cv.notify_all()


def _probe_active_locked(session: Session) -> bool:
    """Whether probe runs should happen at all (caller holds `_cv`).

    A pinned batch or any perturbation activates probing, and so does a
    non-"unchanged" forward mode on its own: "eval"/"train" re-run the model
    on the current snapshot's batch so the UI shows that batch's activations
    under the chosen mode — no pin required. "unchanged" only probes when a
    pin or perturbation gives it something to re-run.
    """
    return (
        session._pinned_inputs is not None
        or bool(session._perturbations)
        or session._probe_mode != "unchanged"
    )


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
    """One probe run: isolated forwards on the base (and perturbed) inputs.

    Training-thread only. Reads the probe config under `_cv`, runs the
    forwards without the lock, and publishes the result only if the
    config is still current — a config change mid-run (re-pin, mode flip,
    new perturbation) wins and its own request re-runs the probe. The
    base inputs are the pinned batch, or the snapshot's inputs when only
    perturbations are active; a perturbed forward re-runs the *whole* model
    with the edited input(s) substituted, so multi-input models work.
    """
    with session._cv:
        version = session._probe_version
        pinned = session._pinned_inputs
        mode = session._probe_mode
        perturbations = dict(session._perturbations)
    if pinned is None and not perturbations and mode == "unchanged":
        return
    bases = pinned if pinned is not None else session._snapshot_inputs()
    if not bases:
        return
    perturbed = apply_perturbations(bases, perturbations)
    base_caps = _probe_forward(session, bases, mode=mode)
    pert_caps = (
        _probe_forward(session, perturbed, mode=mode)
        if perturbed is not None
        else None
    )
    result = ProbeResult(
        inputs=bases,
        activations=base_caps,
        mode=mode,
        perturbed_inputs=perturbed,
        perturbed_activations=pert_caps,
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


def _probe_forward(
    session: Session, inputs: dict[str, Tensor], *, mode: str
) -> dict[str, Tensor]:
    """Run one isolated no-grad forward of the full model, capturing outputs.

    `inputs` is keyed by model input name; the tensors are passed positionally
    in `Session.input_names` (= forward / fx-placeholder) order, so a model
    with several inputs — positional or keyword — is re-run with all of them.
    """
    ordered = [inputs[n] for n in session._input_names if n in inputs]
    with isolated_model(session, mode) as device, torch.no_grad():
        return session._capture_forward([t.to(device) for t in ordered])
