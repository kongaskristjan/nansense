"""Probe runs: NaNsense-internal forward passes between batches.

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

import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
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

# Per-client perturbation state (see `_ProbeClient` and the `*_for` Session
# methods). In a locked (shared demo) session the pin, forward mode, and the
# base input stay shared and frozen, but each visitor perturbs their own copy:
# the edits and the resulting perturbed activations are held per connection so
# one visitor's clicks never change what another sees. `_MAX_PROBE_CLIENTS`
# caps how many such per-visitor containers are retained at once (oldest
# evicted first — a hard memory ceiling), and `_PROBE_CLIENT_TTL` is how long a
# container survives without a page heartbeat before it is reaped (a closed tab
# stops ticking and is dropped ~this many seconds later, like an auto
# experiment). The base activations are computed once and shared across every
# client's diff (see `_shared_base_caps`), so a client stores only its own
# perturbed forward — roughly one image's activations per visitor.
_MAX_PROBE_CLIENTS: int = 16
_PROBE_CLIENT_TTL: float = 5.0


@dataclass
class _ProbeClient:
    """One browser connection's private perturbation state (locked sessions).

    Mirrors the shared probe fields (`_perturbations`, `_probe_request`,
    `_probe_version`, `_probe_count`, `_probe_result`, `_probe_error`) but per
    connection. `version` guards a stale in-flight run from overwriting newer
    edits (same contract as the shared `_probe_version`); `request` arms a
    re-run the pause loop drains; `expires_at` is a `time.monotonic` deadline
    refreshed by the page heartbeat, after which the container is reaped.
    """

    perturbations: PerturbationMap = field(default_factory=dict)
    request: bool = False
    version: int = 0
    count: int = 0
    result: ProbeResult | None = None
    error: str | None = None
    expires_at: float | None = None


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


def perturbation_fits(
    target: Tensor, sample: int, index: tuple[int, ...], values: tuple[float, ...]
) -> bool:
    """Whether this perturbation addresses a real position of `target`.

    Split out of `_write_perturbation` so a caller can ask *before* recording an
    entry: the writer skips a misfit silently (the batch may have changed shape
    since a click was registered), which is right for replaying stored edits and
    useless to anyone who wants to know whether the edit they just made landed.
    """
    if target.ndim == 4:
        b, c, h, w = target.shape
        if len(index) != 2 or len(values) != c:
            return False
        y, x = index
        return 0 <= sample < b and 0 <= y < h and 0 <= x < w
    if target.ndim == 2:
        b, c = target.shape
        if len(index) != 1 or len(values) != 1:
            return False
        (channel,) = index
        return 0 <= sample < b and 0 <= channel < c
    return False


def _write_perturbation(
    target: Tensor, sample: int, index: tuple[int, ...], values: tuple[float, ...]
) -> bool:
    """Write one perturbation into `target` in place; `False` if it doesn't fit."""
    if not perturbation_fits(target, sample, index, values):
        return False
    if target.ndim == 4:
        y, x = index
        target[sample, :, y, x] = torch.tensor(values, dtype=target.dtype)
    else:
        (channel,) = index
        target[sample, channel] = values[0]
    return True


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

    A generator body restores nothing while it is suspended on a `yield`, so
    an experiment must yield its *final* result after the `with` block: that
    result is what waiters wake on, and they must not find the model still
    flipped to eval.
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


# --- Per-client perturbation state (locked / shared demo sessions) ---------
#
# The shared setters above refuse to run on a locked session because they
# mutate state every visitor sees. These `*_for(key)` entry points give each
# connection its own perturbation set and probe result instead, so perturbation
# works in a shared demo without one visitor's clicks leaking into another's
# view. They run on the same pause-loop / training thread as the shared probe
# (the model is only touched there); the base input and forward mode stay
# shared, and the base activations are computed once and reused by every
# client (see `_shared_base_caps`).


def _client_locked(session: Session, key: str) -> _ProbeClient:
    """Get or create `key`'s container and mark it most-recently used (`_cv`)."""
    client = session._probe_clients.get(key)
    if client is None:
        client = _ProbeClient()
        session._probe_clients[key] = client
    session._probe_clients.move_to_end(key)
    return client


def _evict_probe_clients_locked(session: Session) -> None:
    """Drop the least-recently-used containers past the cap (caller holds `_cv`)."""
    while len(session._probe_clients) > _MAX_PROBE_CLIENTS:
        session._probe_clients.popitem(last=False)


def register_probe_client(session: Session, key: str) -> None:
    """Implementation of `Session.register_probe_client`."""
    with session._cv:
        client = _client_locked(session, key)
        client.expires_at = time.monotonic() + _PROBE_CLIENT_TTL
        _evict_probe_clients_locked(session)


def touch_probe_client(session: Session, key: str) -> None:
    """Implementation of `Session.touch_probe_client` (heartbeat)."""
    with session._cv:
        client = session._probe_clients.get(key)
        if client is not None:
            client.expires_at = time.monotonic() + _PROBE_CLIENT_TTL
            session._probe_clients.move_to_end(key)


def unregister_probe_client(session: Session, key: str) -> None:
    """Implementation of `Session.unregister_probe_client`."""
    with session._cv:
        session._probe_clients.pop(key, None)


def gc_probe_clients(session: Session) -> None:
    """Reap containers whose heartbeat lapsed (training thread, pause loop).

    The per-client counterpart of `experiments.run_auto_experiments`' expiry
    sweep: a closed tab stops heartbeating and its container is dropped once
    `expires_at` passes. Called on pause-loop activity — a parked demo has no
    snapshot publishes to hang the sweep off, and the LRU cap bounds memory
    even when the loop is idle.
    """
    now = time.monotonic()
    with session._cv:
        expired = [
            key
            for key, client in session._probe_clients.items()
            if client.expires_at is not None and client.expires_at < now
        ]
        for key in expired:
            del session._probe_clients[key]


def add_perturbation_for(
    session: Session,
    key: str,
    *,
    input_name: str,
    sample: int,
    index: tuple[int, ...],
    values: tuple[float, ...],
) -> None:
    """Add a perturbation to `key`'s private set and arm its probe re-run."""
    if not session._enabled:
        return
    with session._cv:
        client = _client_locked(session, key)
        client.perturbations[(input_name, sample, tuple(index))] = tuple(values)
        client.request = True
        client.version += 1
        client.expires_at = time.monotonic() + _PROBE_CLIENT_TTL
        _evict_probe_clients_locked(session)
        session._cv.notify_all()


def clear_perturbations_for(session: Session, key: str) -> None:
    """Drop `key`'s perturbations and its probe result (nothing left to show)."""
    with session._cv:
        client = session._probe_clients.get(key)
        if client is None or not client.perturbations:
            return
        client.perturbations.clear()
        client.version += 1
        client.request = False
        client.result = None
        client.error = None
        session._cv.notify_all()


def client_probe_result(session: Session, key: str) -> ProbeResult | None:
    """The latest probe result for `key`, or `None`."""
    with session._cv:
        client = session._probe_clients.get(key)
        return client.result if client is not None else None


def client_probe_error(session: Session, key: str) -> str | None:
    """Why `key`'s last probe failed, or `None`."""
    with session._cv:
        client = session._probe_clients.get(key)
        return client.error if client is not None else None


def client_perturbations(session: Session, key: str) -> PerturbationMap:
    """Copy of `key`'s active perturbations."""
    with session._cv:
        client = session._probe_clients.get(key)
        return dict(client.perturbations) if client is not None else {}


def pending_probe_client_keys_locked(session: Session) -> list[str]:
    """Keys of clients with an armed probe re-run (caller holds `_cv`)."""
    return [k for k, c in session._probe_clients.items() if c.request]


def _shared_base_caps(
    session: Session, bases: dict[str, Tensor], mode: str
) -> dict[str, Tensor]:
    """Base activations for the frozen input under `mode`, computed once.

    Every client's perturbed probe diffs against the *same* unperturbed base,
    which in a locked demo never changes — so this caches the single base
    forward and hands the same capture dict to each client. The cache key is
    the identity of the snapshot / pinned inputs the base came from plus the
    mode, so a new snapshot or a re-pin (unlocked sessions) recomputes it.
    Runs on the training thread only, serialized with every other probe, so no
    two clients race to fill the cache.
    """
    sig: tuple[int, int, str] = (
        id(session._snapshot),
        id(session._pinned_inputs),
        mode,
    )
    cached = session._shared_base_cache
    if cached is not None and cached[0] == sig:
        return cached[1]
    caps = _probe_forward(session, bases, mode=mode)
    session._shared_base_cache = (sig, caps)
    return caps


def run_client_probe_guarded(session: Session, key: str) -> None:
    """Run `key`'s perturbed probe, publishing an error instead of crashing."""
    with session._cv:
        client = session._probe_clients.get(key)
        if client is None:
            return
        version = client.version
    try:
        _run_client_probe(session, key, version)
    except Exception as e:  # noqa: BLE001 — surfaced via the client's error
        with session._cv:
            client = session._probe_clients.get(key)
            if client is None or client.version != version:
                return
            client.error = f"{type(e).__name__}: {e}"
            client.count += 1
            session._cv.notify_all()


def _run_client_probe(session: Session, key: str, version: int) -> None:
    """One client's probe: the shared base plus this client's perturbed forward.

    Reads the client's edits and the shared base/mode under `_cv`, runs the
    forwards without the lock, and publishes only if the client's edits haven't
    changed since (its `version` still matches). With no perturbations left the
    result is dropped so the view reverts to the shared snapshot.
    """
    with session._cv:
        client = session._probe_clients.get(key)
        if client is None:
            return
        perturbations = dict(client.perturbations)
        mode = session._probe_mode
        pinned = session._pinned_inputs
    if not perturbations:
        with session._cv:
            client = session._probe_clients.get(key)
            if client is None or client.version != version:
                return
            client.result = None
            client.error = None
            client.count += 1
            session._cv.notify_all()
        return
    bases = pinned if pinned is not None else session._snapshot_inputs()
    if not bases:
        return
    perturbed = apply_perturbations(bases, perturbations)
    base_caps = _shared_base_caps(session, bases, mode)
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
        client = session._probe_clients.get(key)
        if client is None or client.version != version:
            return
        client.result = result
        client.error = None
        client.count += 1
        session._cv.notify_all()
