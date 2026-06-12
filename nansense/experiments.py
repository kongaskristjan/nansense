"""Per-layer experiments: deep dream and a selection of Captum attributions.

An experiment is a long-running, cancellable job executed by the *training
thread* while it is paused — the heavyweight cousin of a probe run (see
`nansense.probe`). The UI arms an `ExperimentRequest` on the session; the
pause loop in `Session._wait_for_proceed` consumes it and drives `run()`,
a generator that yields `ExperimentResult` progress snapshots (the session
publishes each one, so the UI can stream e.g. the evolving deep-dream
image). Yielding a generator instead of returning once is what makes
cancellation cheap: the runner checks `should_abort()` between steps, and
the session aborts on cancel, on a newer request, and on any resume
command.

Experiments reuse the probe isolation contract (`nansense.probe.isolated_model`):
eval-mode forwards with flags/buffers restored, a forked RNG, and gradients
taken via `torch.autograd.grad` w.r.t. the *input* only — parameter `.grad`
is never touched, so the training loop's gradient pickup stays intact.

Besides the experiment kinds, this module owns the request plumbing the
`Session` methods delegate to: the request queue (`request_experiment` /
`cancel_experiment`), the auto-experiment registry (re-run on every
visualization update, kept alive by page heartbeats or a recording's pin),
and the guarded runner that streams results back onto the session.

Captum method selection (deliberately small):

- **Grad-CAM** (`LayerGradCam`) — the classic, cheap localization of a
  class score onto the selected layer.
- **Neuron Gradient** (`NeuronGradient`) — input-saliency of one channel of
  the selected layer; the gradient view of its receptive field.
- **Neuron Integrated Gradients** (`NeuronIntegratedGradients`) — the
  higher-quality path-integrated version of the same.
- **Occlusion** (`Occlusion`) — perturbation-based: slide a patch over the
  input and measure the class-score drop.

Layer-targeted Captum methods need an `nn.Module`; fx intermediates
(`relu`, `add`, …) are rejected with a pointer to the producing module.
Deep dream works on *any* captured layer via the fx interpreter.
"""

from __future__ import annotations

import importlib.util
import time
from collections import deque
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from nansense.capture import _CaptureInterpreter
from nansense.params import bool_param, float_param, float_tuple, int_param
from nansense.probe import isolated_model

if TYPE_CHECKING:
    from nansense.session import Session

EXPERIMENT_KINDS: dict[str, str] = {
    "deep_dream": "Deep Dream",
    "gradcam": "Grad-CAM (Captum)",
    "neuron_gradient": "Neuron Gradient (Captum)",
    "neuron_ig": "Neuron Integrated Gradients (Captum)",
    "occlusion": "Occlusion (Captum)",
}

_CAPTUM_KINDS = frozenset(EXPERIMENT_KINDS) - {"deep_dream"}


def available_experiment_kinds() -> dict[str, str]:
    """Experiment kinds the UI should offer.

    captum is an optional dependency the user installs themselves; the
    captum-based kinds are simply not offered when it is absent.
    """
    if importlib.util.find_spec("captum") is not None:
        return dict(EXPERIMENT_KINDS)
    return {k: v for k, v in EXPERIMENT_KINDS.items() if k not in _CAPTUM_KINDS}

# How many intermediate publishes a deep-dream run spreads over its steps.
_PUBLISH_COUNT: int = 20

# Default cap on how many inputs a deep dream covers (the UI mirrors it).
_DEFAULT_DREAM_BATCH: int = 8


@dataclass(frozen=True)
class ExperimentRequest:
    """One armed experiment: what to run, on which layer, with what knobs.

    `params` values come straight from the UI form (numbers, bools, strings)
    plus the display normalization stats (`mean` / `std` tuples or `None`),
    which the clamp option and result rendering both need.
    """

    kind: str
    layer: str
    params: dict[str, object]
    seq: int


@dataclass(frozen=True)
class ExperimentResult:
    """A progress snapshot or final outcome, fully resident on CPU.

    `done=False` results stream progress (deep dream publishes its evolving
    image); the final yield has `done=True` — also when aborted early, in
    which case `step < total_steps`. Exactly one of `image` (input-space,
    shown denormalized) or `attribution` (signed, shown with the diverging
    colormap) is set on success; `reference` carries the input batch the
    experiment started from.
    """

    seq: int
    kind: str
    layer: str
    step: int
    total_steps: int
    done: bool
    error: str | None = None
    image: Tensor | None = None
    attribution: Tensor | None = None
    reference: Tensor | None = None
    objective: float | None = None


def run(
    session: Session,
    request: ExperimentRequest,
    should_abort: Callable[[], bool],
) -> Iterator[ExperimentResult]:
    """Drive one experiment, yielding progress until done (training thread)."""
    if request.kind == "deep_dream":
        yield from _run_deep_dream(session, request, should_abort)
    elif request.kind in ("gradcam", "neuron_gradient", "neuron_ig", "occlusion"):
        yield from _run_captum(session, request, should_abort)
    else:
        yield _error(request, f"unknown experiment kind {request.kind!r}")


# --- Request queue and lifecycle (driven by the Session) -------------------

# How many per-seq experiment results stay retrievable (oldest evicted
# first); generous enough for every open client plus a little history.
_EXPERIMENT_RESULTS_KEPT: int = 8

# How long an auto-experiment registration survives without a heartbeat
# (`touch_auto_experiment`). UI pages tick every ~0.2 s, so anything beyond
# a few seconds means the page is gone.
_AUTO_EXPERIMENT_TTL: float = 5.0


@dataclass
class _AutoExperiment:
    """One experiment re-run on every visualization update.

    The request keeps its seq for the registration's whole lifetime, so
    `experiment_result_for(seq)` always returns the freshest rerun and a
    seeded experiment (deep dream derives its noise from the seq) draws the
    same start on every update. `expires_at` is a `time.monotonic` deadline
    refreshed by page heartbeats; `None` pins the entry (an active
    recording holds the view)."""

    request: ExperimentRequest
    expires_at: float | None


def request_experiment(
    session: Session, *, kind: str, layer: str, params: dict[str, object]
) -> int:
    """Implementation of `Session.request_experiment`."""
    if kind not in EXPERIMENT_KINDS:
        raise ValueError(
            f"unknown experiment kind {kind!r}; "
            f"expected one of {list(EXPERIMENT_KINDS)}"
        )
    with session._cv:
        session._experiment_seq += 1
        session._experiment_queue.append(
            ExperimentRequest(
                kind=kind,
                layer=layer,
                params=dict(params),
                seq=session._experiment_seq,
            )
        )
        session._cv.notify_all()
        return session._experiment_seq


def cancel_experiment(session: Session, seq: int | None = None) -> None:
    """Implementation of `Session.cancel_experiment`."""
    with session._cv:
        if seq is None:
            session._experiment_queue.clear()
            if session._experiment_running is not None:
                session._experiment_cancelled.add(session._experiment_running)
        else:
            queued = [r for r in session._experiment_queue if r.seq != seq]
            if len(queued) != len(session._experiment_queue):
                session._experiment_queue = deque(queued)
            elif session._experiment_running == seq:
                session._experiment_cancelled.add(seq)
        session._cv.notify_all()


def register_auto_experiment(
    session: Session, key: str, *, kind: str, layer: str, params: dict[str, object]
) -> int:
    """Implementation of `Session.register_auto_experiment`."""
    if kind not in EXPERIMENT_KINDS:
        raise ValueError(
            f"unknown experiment kind {kind!r}; "
            f"expected one of {list(EXPERIMENT_KINDS)}"
        )
    with session._cv:
        session._experiment_seq += 1
        request = ExperimentRequest(
            kind=kind, layer=layer, params=dict(params), seq=session._experiment_seq
        )
        session._auto_experiments[key] = _AutoExperiment(
            request=request,
            expires_at=time.monotonic() + _AUTO_EXPERIMENT_TTL,
        )
        session._experiment_queue.append(request)
        session._cv.notify_all()
        return request.seq


def touch_auto_experiment(session: Session, key: str) -> None:
    """Implementation of `Session.touch_auto_experiment`."""
    with session._cv:
        entry = session._auto_experiments.get(key)
        if entry is not None and entry.expires_at is not None:
            entry.expires_at = time.monotonic() + _AUTO_EXPERIMENT_TTL


def pin_auto_experiment(session: Session, key: str) -> bool:
    """Implementation of `Session.pin_auto_experiment`."""
    with session._cv:
        entry = session._auto_experiments.get(key)
        if entry is None:
            return False
        entry.expires_at = None
        return True


def unpin_auto_experiment(session: Session, key: str) -> None:
    """Implementation of `Session.unpin_auto_experiment`."""
    with session._cv:
        entry = session._auto_experiments.get(key)
        if entry is not None and entry.expires_at is None:
            entry.expires_at = time.monotonic() + _AUTO_EXPERIMENT_TTL


def unregister_auto_experiment(session: Session, key: str) -> None:
    """Implementation of `Session.unregister_auto_experiment`."""
    with session._cv:
        session._auto_experiments.pop(key, None)


def run_auto_experiments(session: Session) -> None:
    """Re-run every live auto experiment (training thread, post-publish).

    Runs at every snapshot publish — frequency updates and mode
    captures alike — so open experiment pages and recordings track the
    evolving weights. Expired registrations (no page heartbeat, not
    pinned by a recording) are dropped first. A registration whose
    initial request is still queued is taken over here: the queued
    duplicate is removed so the request runs exactly once per update.
    """
    now = time.monotonic()
    with session._cv:
        for key in [
            k
            for k, e in session._auto_experiments.items()
            if e.expires_at is not None and e.expires_at < now
        ]:
            del session._auto_experiments[key]
        requests = [e.request for e in session._auto_experiments.values()]
        seqs = {r.seq for r in requests}
        if seqs:
            session._experiment_queue = deque(
                r for r in session._experiment_queue if r.seq not in seqs
            )
    for request in requests:
        run_experiment_guarded(session, request)


def run_experiment_guarded(session: Session, request: ExperimentRequest) -> None:
    """Drive one experiment to completion on the training thread.

    Streams every yielded progress result through `_publish_experiment`.
    The abort predicate stops the run on `cancel_experiment` (for this
    seq) and on anything that ends the pause — resume commands, a
    pending time-travel jump, `close()` — so the pause loop regains
    control promptly; queued requests from other clients wait their
    turn instead of aborting the run. A failing experiment publishes an
    error result instead of killing the training thread.
    """
    with session._cv:
        resume_seen = session._resume_token
        session._experiment_running = request.seq

    def should_abort() -> bool:
        with session._cv:
            return (
                request.seq in session._experiment_cancelled
                or session._closed
                or session._pending_jump is not None
                or session._resume_token != resume_seen
            )

    try:
        for partial in run(session, request, should_abort):
            _publish_experiment(session, partial)
    except Exception as e:  # noqa: BLE001 — surfaced via the result
        _publish_experiment(session, _error(request, f"{type(e).__name__}: {e}"))
    finally:
        with session._cv:
            session._experiment_running = None
            session._experiment_cancelled.discard(request.seq)


def _publish_experiment(session: Session, result: ExperimentResult) -> None:
    with session._cv:
        session._experiment_results[result.seq] = result
        session._experiment_results.move_to_end(result.seq)
        while len(session._experiment_results) > _EXPERIMENT_RESULTS_KEPT:
            session._experiment_results.popitem(last=False)
        session._experiment_result = result
        session._cv.notify_all()


# --- Experiment-kind implementations ---------------------------------------


def _error(request: ExperimentRequest, message: str) -> ExperimentResult:
    return ExperimentResult(
        seq=request.seq,
        kind=request.kind,
        layer=request.layer,
        step=0,
        total_steps=0,
        done=True,
        error=message,
    )


def _sample_input(
    session: Session, request: ExperimentRequest
) -> Tensor | ExperimentResult:
    """The `[1, C, H, W]` input sample the experiment works on, or an error."""
    base = session._snapshot_input()
    if base is None:
        return _error(
            request, "no input available yet — run at least one batch first"
        )
    if base.ndim != 4:
        return _error(request, "experiments need an image input [B, C, H, W]")
    sample = min(max(0, int_param(request.params, "sample", 0)), int(base.shape[0]) - 1)
    return base[sample : sample + 1].detach().clone().float()


def _dream_start(
    session: Session, request: ExperimentRequest, rng: torch.Generator
) -> Tensor | ExperimentResult:
    """The `[batch, ...]` starting batch for deep dream, or an error.

    Built from the network's *real* input (the snapshot's input-node
    tensor), so non-image inputs work too. `start="noise"` draws `batch`
    fresh samples matching the real input's per-sample shape and overall
    mean/std from `rng` — seeded per request, so successive runs explore
    different noise; `start="sample"` takes the first `batch` samples of
    the real input batch. `batch` defaults to the real batch size, capped
    at `_DEFAULT_DREAM_BATCH`.
    """
    base = session._snapshot_input()
    if base is None:
        return _error(
            request, "no input available yet — run at least one batch first"
        )
    if base.ndim < 2:
        return _error(request, "deep dream needs a batched input [B, ...]")
    base = base.detach().float()
    default = min(_DEFAULT_DREAM_BATCH, int(base.shape[0]))
    batch = max(1, int_param(request.params, "batch", default))
    if str(request.params.get("start", "noise")) != "noise":
        return base[:batch].clone()
    noise = torch.randn((batch, *base.shape[1:]), generator=rng)
    return float(base.mean()) + float(base.std()) * noise


def _value_bounds(
    channels: int, mean: object, std: object
) -> tuple[Tensor, Tensor]:
    """Per-channel input-space bounds of the displayable pixel range.

    The UI denormalizes with `x * std + mean` and clamps to `[0, 1]`, so the
    inverse image of that range is `[(0 - mean) / std, (1 - mean) / std]`.
    Without stats the input is assumed to already live in `[0, 1]`.
    """
    means = float_tuple(mean, channels)
    stds = float_tuple(std, channels)
    if means is not None and stds is not None:
        m = torch.tensor(means).view(1, channels, 1, 1)
        s = torch.tensor(stds).view(1, channels, 1, 1)
        return (0.0 - m) / s, (1.0 - m) / s
    return (
        torch.zeros(1, channels, 1, 1),
        torch.ones(1, channels, 1, 1),
    )


def _target_activation(session: Session, x: Tensor, layer: str) -> Tensor:
    """The live (grad-connected) activation of `layer` for input `x`.

    fx mode runs the capture interpreter against a local dict — gradients
    flow through it like through any tensor ops. The hook fallback registers
    a single temporary hook on the target module.
    """
    if session._fx_graph is not None:
        capture: dict[str, Tensor] = {}
        _CaptureInterpreter(session._fx_graph, capture).run(x)
        act = capture.get(layer)
    else:
        module = dict(session.model.named_modules()).get(layer)
        if module is None:
            raise ValueError(f"unknown layer {layer!r}")
        captured: list[Tensor] = []

        def hook(_module: object, _inputs: object, output: object) -> None:
            if isinstance(output, Tensor):
                captured.append(output)

        handle = module.register_forward_hook(hook)
        try:
            session.model(x)
        finally:
            handle.remove()
        act = captured[-1] if captured else None
    if act is None:
        raise ValueError(f"layer {layer!r} did not produce a tensor activation")
    return act


def _channel_objective(act: Tensor, channel: int) -> Tensor:
    """Mean activation of one channel (or of the whole layer for -1)."""
    if act.ndim < 2 or channel < 0:
        return act.mean()
    if channel >= act.shape[1]:
        raise ValueError(
            f"channel {channel} out of range for activation shape "
            f"{tuple(act.shape)}"
        )
    return act[:, channel].mean()


def _zoom_in(x: Tensor, zoom: float) -> Tensor:
    """Zoom into the image center by a multiplier `zoom` (no-op when the
    upscale rounds below one pixel — relevant for small inputs and factors
    close to one)."""
    h, w = int(x.shape[2]), int(x.shape[3])
    zh, zw = int(round(h * zoom)), int(round(w * zoom))
    if zh <= h or zw <= w:
        return x
    big = F.interpolate(x, size=(zh, zw), mode="bilinear", align_corners=False)
    top, left = (zh - h) // 2, (zw - w) // 2
    return big[:, :, top : top + h, left : left + w]


def _run_deep_dream(
    session: Session,
    request: ExperimentRequest,
    should_abort: Callable[[], bool],
) -> Iterator[ExperimentResult]:
    """Gradient ascent on a channel's mean activation w.r.t. a batch of inputs.

    The starting batch comes from the network's real input (`_dream_start`):
    fresh per-request noise by default, or the current input batch. The
    classic bag of regularizers, each optional and image-only (applied when
    the input is `[B, C, H, W]`): per-step jitter (random roll, undone after
    the update — drawn from the request-seeded generator), "diffusion"
    (blend with a 3×3 box blur, damping high-frequency noise), center zoom
    (a per-step multiplier), and clamping to the displayable value range.
    Gradients are normalized per sample by their mean magnitude so `lr`
    behaves comparably across layers and batch sizes.
    """
    p = request.params
    steps = max(1, int_param(p, "steps", 100))
    lr = float_param(p, "lr", 0.05)
    diffusion = min(1.0, max(0.0, float_param(p, "diffusion", 0.05)))
    jitter = max(0, int_param(p, "jitter", 2))
    zoom = max(1.0, float_param(p, "zoom", 1.0))
    channel = int_param(p, "channel", 0)
    clamp = bool_param(p, "clamp", True)

    rng = torch.Generator().manual_seed(request.seq)
    x0 = _dream_start(session, request, rng)
    if isinstance(x0, ExperimentResult):
        yield x0
        return
    reference = x0.clone()
    spatial = x0.ndim == 4  # the regularizers below act on image axes only
    lo, hi = _value_bounds(int(x0.shape[1]), p.get("mean"), p.get("std"))
    publish_every = max(1, steps // _PUBLISH_COUNT)

    def partial(
        x: Tensor, step: int, objective: float, *, done: bool
    ) -> ExperimentResult:
        return ExperimentResult(
            seq=request.seq,
            kind=request.kind,
            layer=request.layer,
            step=step,
            total_steps=steps,
            done=done,
            image=x.detach().cpu(),
            reference=reference,
            objective=objective,
        )

    with isolated_model(session, "eval") as device:
        lo, hi = lo.to(device), hi.to(device)
        x = x0.to(device)
        objective_value = 0.0
        step_done = 0
        for step in range(steps):
            if should_abort():
                break
            dy = dx = 0
            x_step = x
            if spatial and jitter > 0:
                dy = int(torch.randint(-jitter, jitter + 1, (1,), generator=rng).item())
                dx = int(torch.randint(-jitter, jitter + 1, (1,), generator=rng).item())
                x_step = torch.roll(x, shifts=(dy, dx), dims=(2, 3))
            x_step = x_step.detach().requires_grad_(True)
            with torch.enable_grad():
                act = _target_activation(session, x_step, request.layer)
                objective = _channel_objective(act, channel)
                (grad,) = torch.autograd.grad(objective, x_step)
            objective_value = float(objective.detach())
            sample_dims = tuple(range(1, grad.ndim))
            norm = grad.abs().mean(dim=sample_dims, keepdim=True)
            x = x_step.detach() + lr * grad / (norm + 1e-8)
            if spatial and jitter > 0:
                x = torch.roll(x, shifts=(-dy, -dx), dims=(2, 3))
            if spatial and diffusion > 0:
                x = (1 - diffusion) * x + diffusion * F.avg_pool2d(
                    x, 3, stride=1, padding=1
                )
            if spatial:
                x = _zoom_in(x, zoom)
            if spatial and clamp:
                x = torch.min(torch.max(x, lo), hi)
            step_done = step + 1
            if step_done % publish_every == 0 and step_done < steps:
                yield partial(x, step_done, objective_value, done=False)
        yield partial(x, step_done, objective_value, done=True)


def _resolve_target(model: nn.Module, x: Tensor, params: dict[str, object]) -> int:
    """The class index for output-targeted methods (-1 means the argmax)."""
    target = int_param(params, "target", -1)
    if target >= 0:
        return target
    with torch.no_grad():
        out = model(x)
    if not isinstance(out, Tensor) or out.ndim != 2:
        raise ValueError(
            "target-based methods need a [batch, classes] model output; "
            "set an explicit target class for other output shapes"
        )
    return int(out.argmax(dim=1).item())


def _neuron_selector(channel: int) -> Callable[[Tensor], Tensor]:
    """Captum neuron selector: per-example mean of one channel (-1 = all)."""

    def selector(out: Tensor) -> Tensor:
        if channel >= 0:
            if out.ndim < 2 or channel >= out.shape[1]:
                raise ValueError(
                    f"channel {channel} out of range for activation shape "
                    f"{tuple(out.shape)}"
                )
            out = out[:, channel]
        return out.reshape(out.shape[0], -1).mean(dim=1)

    return selector


def _run_captum(
    session: Session,
    request: ExperimentRequest,
    should_abort: Callable[[], bool],
) -> Iterator[ExperimentResult]:
    """One-shot Captum attribution on the selected sample.

    All methods run the *unpatched* model (probes/experiments only execute
    between batches) inside the isolation scope. Gradient-based methods use
    `torch.autograd.grad` internally, so parameter `.grad` survives.
    """
    try:
        from captum import attr as captum_attr
    except ImportError:
        yield _error(request, "captum is not installed (`pip install captum`)")
        return

    p = request.params
    x0 = _sample_input(session, request)
    if isinstance(x0, ExperimentResult):
        yield x0
        return
    module: nn.Module | None = dict(session.model.named_modules()).get(
        request.layer
    )
    if request.kind in ("gradcam", "neuron_gradient", "neuron_ig") and module is None:
        yield _error(
            request,
            f"{EXPERIMENT_KINDS[request.kind]} needs an nn.Module layer; "
            f"{request.layer!r} is an fx intermediate — pick the producing "
            "module instead",
        )
        return
    if should_abort():
        yield _error(request, "cancelled")
        return

    with isolated_model(session, "eval") as device:
        x = x0.to(device)
        if request.kind == "gradcam":
            assert module is not None  # checked above
            target = _resolve_target(session.model, x, p)
            attribution = captum_attr.LayerGradCam(session.model, module).attribute(
                x, target=target
            )
        elif request.kind == "neuron_gradient":
            assert module is not None  # checked above
            attribution = captum_attr.NeuronGradient(
                session.model, module
            ).attribute(x, neuron_selector=_neuron_selector(int_param(p, "channel", 0)))
        elif request.kind == "neuron_ig":
            assert module is not None  # checked above
            attribution = captum_attr.NeuronIntegratedGradients(
                session.model, module
            ).attribute(
                x,
                neuron_selector=_neuron_selector(int_param(p, "channel", 0)),
                n_steps=max(2, int_param(p, "steps", 32)),
            )
        else:  # occlusion
            target = _resolve_target(session.model, x, p)
            channels = int(x.shape[1])
            window = max(1, int_param(p, "window", 4))
            stride = max(1, int_param(p, "stride", 2))
            attribution = captum_attr.Occlusion(session.model).attribute(
                x,
                target=target,
                sliding_window_shapes=(channels, window, window),
                strides=(channels, stride, stride),
                baselines=0.0,
                perturbations_per_eval=8,
            )

    assert isinstance(attribution, Tensor)
    yield ExperimentResult(
        seq=request.seq,
        kind=request.kind,
        layer=request.layer,
        step=1,
        total_steps=1,
        done=True,
        attribution=attribution.detach().cpu().float(),
        reference=x0,
    )
