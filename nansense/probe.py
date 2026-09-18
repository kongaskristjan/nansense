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

`ProbeManager` owns configuration, client lifetimes, and request/result state
under the condition shared with Session. Its injected forward callback runs
on the training thread. `isolated_model` is the isolation contract shared
with `nansense.experiments`.
"""

from __future__ import annotations

import time
import threading
from collections import OrderedDict
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch
from torch import Tensor

from nansense.capture import fork_rng, model_device

if TYPE_CHECKING:
    from nansense.session import BatchSnapshot, Session
    from nansense.schedule import BatchPosition

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
# client's diff (see `_shared_base_caps`), and a client retains only the rows
# it actually edited (see `ProbeResult`) — so a visitor costs one *sample's*
# activations, not one batch's. That distinction is the difference between
# megabytes and gigabytes at the cap: a capture holds every layer the model
# has, so on a 128px ResNet at batch 16 the full batch is ~800 MB per visitor
# against ~50 MB for the one row anyone is looking at.
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

    The perturbed forward runs the *whole* batch — batch-dependent layers
    (BatchNorm in train mode) would otherwise see different statistics — but
    only the edited samples' rows are retained, in `perturbed_samples` order.
    Every other sample's perturbed activation is by definition its base
    activation, so keeping the full batch would store `B` copies of a result
    that differs in one row; `perturbed_act` resolves a sample against that
    layout. `perturbed_inputs` stays full-batch: it is one input tensor
    rather than every layer's output, so slicing it saves little and would
    complicate `shown_input`.
    """

    inputs: dict[str, Tensor]
    activations: dict[str, Tensor]
    mode: str
    perturbed_inputs: dict[str, Tensor] | None = None
    perturbed_activations: dict[str, Tensor] | None = None
    perturbed_samples: tuple[int, ...] = ()

    def perturbed_act(self, name: str, sample_idx: int) -> Tensor | None:
        """Layer `name`'s perturbed activation for one sample, shaped `[1, ...]`.

        `None` when no perturbed forward ran, when `sample_idx` carries no
        edit (its perturbed activation is the base one), or when the layer is
        absent from the perturbed capture.
        """
        acts = self.perturbed_activations
        if acts is None or sample_idx not in self.perturbed_samples:
            return None
        tensor = acts.get(name)
        if tensor is None or tensor.ndim == 0:
            return None
        row = self.perturbed_samples.index(sample_idx)
        return tensor[row : row + 1] if row < tensor.shape[0] else None

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


def perturbed_samples(perturbations: PerturbationMap) -> tuple[int, ...]:
    """The sample indices carrying at least one edit, ascending."""
    return tuple(sorted({sample for _, sample, _ in perturbations}))


def _batch_size(bases: dict[str, Tensor]) -> int | None:
    """Batch size of the base inputs, read off whichever input has a batch axis."""
    for tensor in bases.values():
        if tensor.ndim > 0:
            return int(tensor.shape[0])
    return None


def _select_samples(
    caps: dict[str, Tensor], samples: tuple[int, ...], batch: int | None
) -> dict[str, Tensor]:
    """Keep only `samples`' rows of every batched capture in `caps`.

    A capture whose leading axis isn't the batch (a 0-dim scalar, or a tensor
    the model produced at some other size) is passed through untouched.
    `index_select` copies, so the full-batch tensors are freed once the caller
    drops `caps` — that release is the whole point (see `ProbeResult`).
    """
    if batch is None or not samples or samples[-1] >= batch:
        return caps
    index = torch.tensor(samples)
    return {
        name: (
            tensor.index_select(0, index)
            if tensor.ndim > 0 and tensor.shape[0] == batch
            else tensor
        )
        for name, tensor in caps.items()
    }


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
    saved_buffers = [(b, b.detach().clone()) for _, b in session.model.named_buffers()]
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


def _same_bases(cached: dict[str, Tensor], bases: dict[str, Tensor]) -> bool:
    """Whether both mappings name the same input tensor objects."""
    return cached.keys() == bases.keys() and all(
        cached[name] is tensor for name, tensor in bases.items()
    )


class ProbeManager:
    """Own probe configuration, client lifetimes, requests, and published results.

    The shared condition coordinates with Session's pause loop. Forward work
    runs outside it, on the training thread, through the supplied callback.
    """

    def __init__(
        self,
        cv: threading.Condition,
        *,
        enabled: bool,
        input_names: tuple[str, ...],
        snapshot: Callable[[], BatchSnapshot | None],
        forward: Callable[[dict[str, Tensor], str], dict[str, Tensor]],
        closed: Callable[[], bool],
    ) -> None:
        self._cv = cv
        self._enabled = enabled
        self._input_names = input_names
        self._snapshot = snapshot
        self._forward = forward
        self._closed = closed
        self._pinned_inputs: dict[str, Tensor] | None = None
        self._pinned_position: BatchPosition | None = None
        self._perturbations: PerturbationMap = {}
        self._mode = "unchanged"
        self._request = False
        self._version = 0
        self._count = 0
        self._result: ProbeResult | None = None
        self._error: str | None = None
        self._clients: OrderedDict[str, _ProbeClient] = OrderedDict()
        self._shared_base_cache: (
            tuple[dict[str, Tensor], str, dict[str, Tensor]] | None
        ) = None

    def invalidate_base(self) -> None:
        """Training-thread boundary: weights, buffers, or mode may have changed."""
        self._shared_base_cache = None

    def _snapshot_inputs(self) -> dict[str, Tensor]:
        snap = self._snapshot()
        return (
            {}
            if snap is None
            else {
                name: snap.activations[name]
                for name in self._input_names
                if name in snap.activations
            }
        )

    @property
    def result(self) -> ProbeResult | None:
        return self._result

    @property
    def error(self) -> str | None:
        return self._error

    @property
    def count(self) -> int:
        with self._cv:
            return self._count

    @property
    def mode(self) -> str:
        with self._cv:
            return self._mode

    @property
    def is_pinned(self) -> bool:
        with self._cv:
            return self._pinned_inputs is not None

    @property
    def pinned_position(self) -> BatchPosition | None:
        return self._pinned_position

    @property
    def perturbations(self) -> PerturbationMap:
        with self._cv:
            return dict(self._perturbations)

    @property
    def pending(self) -> bool:
        with self._cv:
            return self._request or any(c.request for c in self._clients.values())

    def take_pending(self) -> tuple[bool, list[str]]:
        """Consume pending flags atomically; execute their work outside the lock."""
        with self._cv:
            shared = self._request
            self._request = False
            clients = self.pending_probe_client_keys_locked()
            for key in clients:
                self._clients[key].request = False
            return shared, clients

    def wait(
        self, *, after_count: int, timeout: float | None, client: str | None
    ) -> bool:
        def completed() -> bool:
            if client is None:
                return self._count > after_count
            entry = self._clients.get(client)
            return entry is not None and entry.count > after_count

        with self._cv:
            return self._cv.wait_for(
                lambda: completed() or self._closed(), timeout=timeout
            )

    def pin_current_batch(self) -> bool:
        """Implementation of `Session.pin_current_batch`."""
        if not self._enabled:
            return False
        snap = self._snapshot()
        if snap is None:
            return False
        pinned = {
            name: snap.activations[name]
            for name in self._input_names
            if name in snap.activations
        }
        if not pinned:
            return False
        with self._cv:
            self._pinned_inputs = pinned
            self._pinned_position = snap.position
            self.request_probe_locked()
        return True

    def unpin_batch(self) -> None:
        """Implementation of `Session.unpin_batch`."""
        with self._cv:
            if self._pinned_inputs is None:
                return
            self._pinned_inputs = None
            self._pinned_position = None
            if self._probe_active_locked():
                # Perturbations or an "eval"/"train" mode keep probing, now
                # against the snapshot input.
                self.request_probe_locked()
                return
            self._clear_probe_result_locked()

    def add_perturbation(
        self,
        *,
        input_name: str,
        sample: int,
        index: tuple[int, ...],
        values: tuple[float, ...],
    ) -> None:
        """Implementation of `Session.add_perturbation`."""
        if not self._enabled:
            return
        with self._cv:
            self._perturbations[(input_name, sample, tuple(index))] = tuple(values)
            self.request_probe_locked()

    def clear_perturbations(self) -> None:
        """Implementation of `Session.clear_perturbations`."""
        with self._cv:
            if not self._perturbations:
                return
            self._perturbations.clear()
            if self._probe_active_locked():
                # A pin or an "eval"/"train" mode keeps probing without the
                # cleared perturbations.
                self.request_probe_locked()
                return
            self._clear_probe_result_locked()

    def set_probe_mode(self, mode: str) -> None:
        """Implementation of `Session.set_probe_mode`."""
        if mode not in PROBE_MODES:
            raise ValueError(
                f"unknown probe mode {mode!r}; expected one of {PROBE_MODES}"
            )
        with self._cv:
            if mode == self._mode:
                return
            self._mode = mode
            if self._probe_active_locked():
                # Selecting "eval"/"train" (or changing mode while pinned /
                # perturbed) re-runs the probe under the new mode.
                self.request_probe_locked()
            else:
                # Back to "unchanged" with nothing else probing: drop the stale
                # eval/train result so the UI reverts to the live snapshot.
                self._clear_probe_result_locked()

    def _clear_probe_result_locked(self) -> None:
        """Deactivate probing and drop the published result (caller holds `_cv`)."""
        self._version += 1
        self._request = False
        self._result = None
        self._error = None
        self._cv.notify_all()

    def _probe_active_locked(self) -> bool:
        """Whether probe runs should happen at all (caller holds `_cv`).

        A pinned batch or any perturbation activates probing, and so does a
        non-"unchanged" forward mode on its own: "eval"/"train" re-run the model
        on the current snapshot's batch so the UI shows that batch's activations
        under the chosen mode — no pin required. "unchanged" only probes when a
        pin or perturbation gives it something to re-run.
        """
        return (
            self._pinned_inputs is not None
            or bool(self._perturbations)
            or self._mode != "unchanged"
        )

    def request_probe_locked(self) -> None:
        """Arm a probe run and wake a paused training thread (caller holds `_cv`)."""
        self._version += 1
        self._request = True
        self._cv.notify_all()

    def maybe_run_probe_at_capture(self) -> None:
        """Run a probe right after a capture published its snapshot.

        Called by `_BatchContext.__exit__` before the pause, so every pause
        shows a probe result consistent with the just-captured weights. Any
        UI request armed in the meantime is consumed here — the run below
        uses the current config either way.
        """
        with self._cv:
            self._request = False
            active = self._probe_active_locked()
        if active:
            self.run_probe_guarded()

    def run_probe_guarded(self) -> None:
        # A failing probe (bad input, OOM, model quirk) must not kill the
        # training thread or wedge the pause loop; the error is published
        # for the UI to display instead.
        with self._cv:
            version = self._version
        try:
            self._run_probe()
        except Exception as e:  # noqa: BLE001 — surfaced via probe_error
            with self._cv:
                # Mirror the success path's staleness guard: a config change
                # mid-run (re-pin, mode flip, un-pin) bumps the version and arms
                # its own probe, so a superseded run must not leave a stuck error
                # behind that newer config — especially when the new config makes
                # probing inactive and nothing else clears it.
                if self._version != version:
                    return
                self._error = f"{type(e).__name__}: {e}"
                self._count += 1
                self._cv.notify_all()

    def _run_probe(self) -> None:
        """One probe run: isolated forwards on the base (and perturbed) inputs.

        Training-thread only. Reads the probe config under `_cv`, runs the
        forwards without the lock, and publishes the result only if the
        config is still current — a config change mid-run (re-pin, mode flip,
        new perturbation) wins and its own request re-runs the probe. The
        base inputs are the pinned batch, or the snapshot's inputs when only
        perturbations are active; a perturbed forward re-runs the *whole* model
        with the edited input(s) substituted, so multi-input models work.
        """
        with self._cv:
            version = self._version
            pinned = self._pinned_inputs
            mode = self._mode
            perturbations = dict(self._perturbations)
        if pinned is None and not perturbations and mode == "unchanged":
            return
        bases = pinned if pinned is not None else self._snapshot_inputs()
        if not bases:
            return
        perturbed = apply_perturbations(bases, perturbations)
        base_caps = self._forward(bases, mode)
        samples = perturbed_samples(perturbations)
        pert_caps = (
            _select_samples(self._forward(perturbed, mode), samples, _batch_size(bases))
            if perturbed is not None
            else None
        )
        result = ProbeResult(
            inputs=bases,
            activations=base_caps,
            mode=mode,
            perturbed_inputs=perturbed,
            perturbed_activations=pert_caps,
            perturbed_samples=samples,
        )
        with self._cv:
            if self._version != version:
                return
            self._result = result
            self._error = None
            self._count += 1
            self._cv.notify_all()

    def _client_locked(self, key: str) -> _ProbeClient:
        """Get or create `key`'s container and mark it most-recently used (`_cv`)."""
        client = self._clients.get(key)
        if client is None:
            client = _ProbeClient()
            self._clients[key] = client
        self._clients.move_to_end(key)
        return client

    def _evict_probe_clients_locked(self) -> None:
        """Drop the least-recently-used containers past the cap (caller holds `_cv`)."""
        while len(self._clients) > _MAX_PROBE_CLIENTS:
            self._clients.popitem(last=False)

    def register_probe_client(self, key: str) -> None:
        """Implementation of `Session.register_probe_client`."""
        with self._cv:
            client = self._client_locked(key)
            client.expires_at = time.monotonic() + _PROBE_CLIENT_TTL
            self._evict_probe_clients_locked()

    def touch_probe_client(self, key: str) -> None:
        """Implementation of `Session.touch_probe_client` (heartbeat)."""
        with self._cv:
            client = self._clients.get(key)
            if client is not None:
                client.expires_at = time.monotonic() + _PROBE_CLIENT_TTL
                self._clients.move_to_end(key)

    def unregister_probe_client(self, key: str) -> None:
        """Implementation of `Session.unregister_probe_client`."""
        with self._cv:
            self._clients.pop(key, None)

    def gc_probe_clients(self) -> None:
        """Reap containers whose heartbeat lapsed (training thread, pause loop).

        The per-client counterpart of `experiments.run_auto_experiments`' expiry
        sweep: a closed tab stops heartbeating and its container is dropped once
        `expires_at` passes. Called on pause-loop activity — a parked demo has no
        snapshot publishes to hang the sweep off, and the LRU cap bounds memory
        even when the loop is idle.
        """
        now = time.monotonic()
        with self._cv:
            expired = [
                key
                for key, client in self._clients.items()
                if client.expires_at is not None and client.expires_at < now
            ]
            for key in expired:
                del self._clients[key]

    def add_perturbation_for(
        self,
        key: str,
        *,
        input_name: str,
        sample: int,
        index: tuple[int, ...],
        values: tuple[float, ...],
    ) -> None:
        """Add a perturbation to `key`'s private set and arm its probe re-run."""
        if not self._enabled:
            return
        with self._cv:
            client = self._client_locked(key)
            client.perturbations[(input_name, sample, tuple(index))] = tuple(values)
            client.request = True
            client.version += 1
            client.expires_at = time.monotonic() + _PROBE_CLIENT_TTL
            self._evict_probe_clients_locked()
            self._cv.notify_all()

    def clear_perturbations_for(self, key: str) -> None:
        """Drop `key`'s perturbations and its probe result (nothing left to show)."""
        with self._cv:
            client = self._clients.get(key)
            if client is None or not client.perturbations:
                return
            client.perturbations.clear()
            client.version += 1
            client.request = False
            client.result = None
            client.error = None
            self._cv.notify_all()

    def client_probe_result(self, key: str) -> ProbeResult | None:
        """The latest probe result for `key`, or `None`."""
        with self._cv:
            client = self._clients.get(key)
            return client.result if client is not None else None

    def client_probe_error(self, key: str) -> str | None:
        """Why `key`'s last probe failed, or `None`."""
        with self._cv:
            client = self._clients.get(key)
            return client.error if client is not None else None

    def client_perturbations(self, key: str) -> PerturbationMap:
        """Copy of `key`'s active perturbations."""
        with self._cv:
            client = self._clients.get(key)
            return dict(client.perturbations) if client is not None else {}

    def pending_probe_client_keys_locked(self) -> list[str]:
        """Keys of clients with an armed probe re-run (caller holds `_cv`)."""
        return [k for k, c in self._clients.items() if c.request]

    def _shared_base_caps(
        self, bases: dict[str, Tensor], mode: str
    ) -> dict[str, Tensor]:
        """Base activations for the frozen input under `mode`, computed once.

        Every client's perturbed probe diffs against the *same* unperturbed base,
        which in a locked demo never changes — so this caches the single base
        forward and hands the same capture dict to each client. Batch entry and
        time-travel restoration clear the cache even when inputs remain pinned;
        a new input or mode also recomputes it. Runs on the training thread
        only, serialized with every other probe, so no two clients race to fill
        the cache.

        The cache holds the base tensors it was built from and validates by
        identity. Keying on `id()` alone would be a correctness bug: CPython
        reuses an address once the old object is freed, so a fresh snapshot
        landing where its predecessor sat would match a stale entry and every
        client would silently diff against the wrong base. Holding the references
        makes the comparison meaningful — and costs only the model's inputs, not
        the snapshot they came from.
        """
        cached = self._shared_base_cache
        if cached is not None:
            cached_bases, cached_mode, caps = cached
            if cached_mode == mode and _same_bases(cached_bases, bases):
                return caps
        caps = self._forward(bases, mode)
        self._shared_base_cache = (dict(bases), mode, caps)
        return caps

    def run_client_probe_guarded(self, key: str) -> None:
        """Run `key`'s perturbed probe, publishing an error instead of crashing."""
        with self._cv:
            client = self._clients.get(key)
            if client is None:
                return
            version = client.version
        try:
            self._run_client_probe(key, version)
        except Exception as e:  # noqa: BLE001 — surfaced via the client's error
            with self._cv:
                client = self._clients.get(key)
                if client is None or client.version != version:
                    return
                client.error = f"{type(e).__name__}: {e}"
                client.count += 1
                self._cv.notify_all()

    def _run_client_probe(self, key: str, version: int) -> None:
        """One client's probe: the shared base plus this client's perturbed forward.

        Reads the client's edits and the shared base/mode under `_cv`, runs the
        forwards without the lock, and publishes only if the client's edits haven't
        changed since (its `version` still matches). With no perturbations left the
        result is dropped so the view reverts to the shared snapshot.
        """
        with self._cv:
            client = self._clients.get(key)
            if client is None:
                return
            perturbations = dict(client.perturbations)
            mode = self._mode
            pinned = self._pinned_inputs
        if not perturbations:
            with self._cv:
                client = self._clients.get(key)
                if client is None or client.version != version:
                    return
                client.result = None
                client.error = None
                client.count += 1
                self._cv.notify_all()
            return
        bases = pinned if pinned is not None else self._snapshot_inputs()
        if not bases:
            return
        perturbed = apply_perturbations(bases, perturbations)
        base_caps = self._shared_base_caps(bases, mode)
        samples = perturbed_samples(perturbations)
        pert_caps = (
            _select_samples(self._forward(perturbed, mode), samples, _batch_size(bases))
            if perturbed is not None
            else None
        )
        result = ProbeResult(
            inputs=bases,
            activations=base_caps,
            mode=mode,
            perturbed_inputs=perturbed,
            perturbed_activations=pert_caps,
            perturbed_samples=samples,
        )
        with self._cv:
            client = self._clients.get(key)
            if client is None or client.version != version:
                return
            client.result = result
            client.error = None
            client.count += 1
            self._cv.notify_all()
