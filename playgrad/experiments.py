"""Per-layer experiments: deep dream and a selection of Captum attributions.

An experiment is a long-running, cancellable job executed by the *training
thread* while it is paused — the heavyweight cousin of a probe run (see
`playgrad.probe`). The UI arms an `ExperimentRequest` on the session; the
pause loop in `Session._wait_for_proceed` consumes it and drives `run()`,
a generator that yields `ExperimentResult` progress snapshots (the session
publishes each one, so the UI can stream e.g. the evolving deep-dream
image). Yielding a generator instead of returning once is what makes
cancellation cheap: the runner checks `should_abort()` between steps, and
the session aborts on cancel, on a newer request, and on any resume
command.

Experiments reuse the probe isolation contract (`Session._isolated_model`):
eval-mode forwards with flags/buffers restored, a forked RNG, and gradients
taken via `torch.autograd.grad` w.r.t. the *input* only — parameter `.grad`
is never touched, so the training loop's gradient pickup stays intact.

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

from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from torch import Tensor, nn
from torch.nn import functional as F

if TYPE_CHECKING:
    from playgrad.session import Session

EXPERIMENT_KINDS: dict[str, str] = {
    "deep_dream": "Deep Dream",
    "gradcam": "Grad-CAM (Captum)",
    "neuron_gradient": "Neuron Gradient (Captum)",
    "neuron_ig": "Neuron Integrated Gradients (Captum)",
    "occlusion": "Occlusion (Captum)",
}

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


def _f(params: dict[str, object], key: str, default: float) -> float:
    value = params.get(key, default)
    return float(value) if isinstance(value, (int, float)) else default


def _i(params: dict[str, object], key: str, default: int) -> int:
    value = params.get(key, default)
    return int(value) if isinstance(value, (int, float)) else default


def _b(params: dict[str, object], key: str, default: bool) -> bool:
    value = params.get(key, default)
    return bool(value)


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
    sample = min(max(0, _i(request.params, "sample", 0)), int(base.shape[0]) - 1)
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
    batch = max(1, _i(request.params, "batch", default))
    if str(request.params.get("start", "noise")) != "noise":
        return base[:batch].clone()
    noise = torch.randn((batch, *base.shape[1:]), generator=rng)
    return float(base.mean()) + float(base.std()) * noise


def _float_tuple(value: object, length: int) -> tuple[float, ...] | None:
    if not isinstance(value, (tuple, list)) or len(value) != length:
        return None
    result: list[float] = []
    for v in value:
        if not isinstance(v, (int, float)):
            return None
        result.append(float(v))
    return tuple(result)


def _value_bounds(
    channels: int, mean: object, std: object
) -> tuple[Tensor, Tensor]:
    """Per-channel input-space bounds of the displayable pixel range.

    The UI denormalizes with `x * std + mean` and clamps to `[0, 1]`, so the
    inverse image of that range is `[(0 - mean) / std, (1 - mean) / std]`.
    Without stats the input is assumed to already live in `[0, 1]`.
    """
    means = _float_tuple(mean, channels)
    stds = _float_tuple(std, channels)
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
    # Imported lazily: playgrad.session imports this module at the top level.
    from playgrad.session import _CaptureInterpreter

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
    steps = max(1, _i(p, "steps", 100))
    lr = _f(p, "lr", 0.05)
    diffusion = min(1.0, max(0.0, _f(p, "diffusion", 0.05)))
    jitter = max(0, _i(p, "jitter", 2))
    zoom = max(1.0, _f(p, "zoom", 1.0))
    channel = _i(p, "channel", 0)
    clamp = _b(p, "clamp", True)

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

    with session._isolated_model("eval") as device:
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
    target = _i(params, "target", -1)
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
        yield _error(request, "captum is not installed (`uv add captum`)")
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

    with session._isolated_model("eval") as device:
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
            ).attribute(x, neuron_selector=_neuron_selector(_i(p, "channel", 0)))
        elif request.kind == "neuron_ig":
            assert module is not None  # checked above
            attribution = captum_attr.NeuronIntegratedGradients(
                session.model, module
            ).attribute(
                x,
                neuron_selector=_neuron_selector(_i(p, "channel", 0)),
                n_steps=max(2, _i(p, "steps", 32)),
            )
        else:  # occlusion
            target = _resolve_target(session.model, x, p)
            channels = int(x.shape[1])
            window = max(1, _i(p, "window", 4))
            stride = max(1, _i(p, "stride", 2))
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
