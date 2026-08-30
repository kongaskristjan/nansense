"""Per-layer experiments: deep dream and a selection of Captum attributions.

An experiment is a long-running, cancellable job executed by the *training
thread* while it is paused — the heavyweight cousin of a probe run (see
`nansense.probe`). The UI arms an `ExperimentRequest` on the session; the
pause loop in `Session._wait_for_proceed` consumes it and drives `run()`,
a generator that yields `ExperimentResult` progress snapshots (the session
publishes each one, so the UI can stream e.g. the evolving deep-dream
image). Yielding a generator instead of returning once is what makes
cancellation cheap: the runner checks `should_abort()` between steps, and
the session aborts on cancel, on a newer request, on any resume command,
and once a run outlives the `_EXPERIMENT_TIME_LIMIT` wall-clock ceiling.

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
  input and measure the drop in the selected layer-channel's mean activation
  (a thin wrapper exposes that channel as the model output to attribute).

The Captum methods run on a *batch* of inputs (like deep dream) and publish
one attribution per sample.

Grad-CAM and the neuron methods need an `nn.Module`; fx intermediates
(`relu`, `add`, …) are rejected with a pointer to the producing module.
Deep dream and occlusion read the activation through the fx interpreter, so
they work on *any* captured layer.
"""

from __future__ import annotations

import time
from collections import deque
from collections.abc import Callable, Iterator
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

from captum import attr as captum_attr
import torch
from torch import Tensor, nn
from torch.nn import functional as F

from nansense.capture import _CaptureInterpreter
from nansense.params import bool_param, float_param, float_tuple, int_param
from nansense.probe import isolated_model

if TYPE_CHECKING:
    from nansense.recording import ExperimentClip
    from nansense.session import Session

EXPERIMENT_KINDS: dict[str, str] = {
    "deep_dream": "Deep Dream",
    "gradcam": "Grad-CAM (Captum)",
    "neuron_gradient": "Neuron Gradient (Captum)",
    "neuron_ig": "Neuron Integrated Gradients (Captum)",
    "occlusion": "Occlusion (Captum)",
}

# Kinds whose Captum attribution targets the layer's `nn.Module` object
# directly (Grad-CAM's localization layer, the neuron methods' layer). An fx
# intermediate (relu/add/…) has no module to hand Captum, so these are
# unavailable there. Deep dream and the retargeted occlusion read the
# activation through the fx interpreter, so they accept any captured layer.
_MODULE_KINDS = frozenset({"gradcam", "neuron_gradient", "neuron_ig"})

# How many intermediate publishes a deep-dream run spreads over its steps.
# The starting image publishes as step 0 on top of these, so a run streams
# `_PUBLISH_COUNT` + 1 frames in all: the picture before the ascent, and the
# ascent.
_PUBLISH_COUNT: int = 20

# Default count for deep dream channels and the Captum input batch — a cap a
# layer with fewer channels (or a smaller input batch) shrinks to; it is also
# the default `EXPERIMENT_PARAMS` shows for both knobs.
_DEFAULT_DREAM_BATCH: int = 8


@dataclass(frozen=True)
class ExperimentParam:
    """One configurable knob of an experiment.

    Both front-ends read these: the experiment page renders one form widget per
    spec, and the MCP server describes and validates a tool call's `params`
    against them. `default` is the value `run` falls back to when the key is
    absent, so the two stay in step — a knob is described in exactly one place.
    """

    key: str
    label: str
    kind: str  # "int" | "float" | "bool" | "select"
    default: object
    options: dict[str, str] | None = None
    minimum: float | None = None
    step: float | None = None
    tooltip: str = ""


# Shared knobs reused across kinds. A param is shared *by key*, so its value
# survives switching experiment type (point 1): set "channel" for Neuron
# Gradient and it carries over to Occlusion, "Inputs" carries everywhere, …
_CHANNEL_PARAM = ExperimentParam(
    "channel",
    "Channel (-1 = whole layer)",
    "int",
    0,
    minimum=-1,
    tooltip="Which channel of the layer to target",
)
_CHANNELS_PARAM = ExperimentParam(
    "channels",
    "Channels",
    "int",
    _DEFAULT_DREAM_BATCH,
    minimum=1,
    tooltip=(
        "How many of the layer's channels to dream on — one sample each"
    ),
)
_MINIMIZE_PARAM = ExperimentParam(
    "minimize",
    "Minimize activations",
    "bool",
    False,
    tooltip="Synthesize inputs that suppress each channel instead",
)
_SAMPLE_PARAM = ExperimentParam(
    "sample",
    "Sample",
    "int",
    0,
    minimum=0,
    tooltip="Which batch sample every dream starts from",
)
_TARGET_PARAM = ExperimentParam(
    "target",
    "Target class (-1 = argmax)",
    "int",
    -1,
    minimum=-1,
    tooltip="Which class to explain",
)
_BATCH_PARAM = ExperimentParam(
    "batch",
    "Inputs",
    "int",
    _DEFAULT_DREAM_BATCH,
    minimum=1,
    tooltip="How many inputs to run on",
)
_START_PARAM = ExperimentParam(
    "start",
    "Start from",
    "select",
    "noise",
    options={"noise": "Noise", "sample": "Current batch"},
    tooltip="What the synthesized inputs start from",
)
_CLAMP_PARAM = ExperimentParam(
    "clamp",
    "Clamp to displayable range",
    "bool",
    True,
    tooltip="Keep pixels inside the displayable range",
)
_DIFFUSION_PARAM = ExperimentParam(
    "diffusion",
    "Diffusion",
    "float",
    0.05,
    minimum=0,
    step=0.01,
    tooltip="Blur a little each step; damps high-frequency noise",
)
_JITTER_PARAM = ExperimentParam(
    "jitter",
    "Jitter (px)",
    "int",
    2,
    minimum=0,
    tooltip="Random shift each step; reduces pixel-grid artifacts",
)
_ZOOM_PARAM = ExperimentParam(
    "zoom",
    "Zoom multiplier per step",
    "float",
    1.0,
    minimum=1,
    step=0.01,
    tooltip="Zoom into the centre a little each step (1 = no zoom)",
)
# Deep dream publishes the starting image and then `_PUBLISH_COUNT` evenly
# spaced snapshots of a run by default — enough for a page that only ever
# draws the freshest one, and far fewer CPU copies than one per step. This
# knob publishes every step instead, which is what a recorded run
# (`ExperimentRequest.video`) needs to replay the ascent frame by frame
# rather than in ~20 jumps.
_ALL_STEPS_PARAM = ExperimentParam(
    "all_steps",
    "Publish every step",
    "bool",
    False,
    tooltip="Stream one result per step instead of ~20 evenly spaced ones",
)

# Ordered per kind: the targeting knob first (deep dream's Channels, Captum's
# Channel/Target), then Inputs (Captum) or Start from + Sample (deep dream),
# then the method-specific knobs (point 1). The Layer selector is rendered
# above this list (point 2). Deep dream's Sample knob only shows when starting
# from the current batch (toggled in `rebuild_params`).
EXPERIMENT_PARAMS: dict[str, list[ExperimentParam]] = {
    "deep_dream": [
        _CHANNELS_PARAM,
        _START_PARAM,
        _SAMPLE_PARAM,
        ExperimentParam("steps", "Steps", "int", 300, minimum=1),
        _ALL_STEPS_PARAM,
        ExperimentParam("lr", "Learning rate", "float", 0.05, minimum=0, step=0.01),
        _DIFFUSION_PARAM,
        _JITTER_PARAM,
        _ZOOM_PARAM,
        # The objective-direction toggle sits with the value-range knob below it.
        _MINIMIZE_PARAM,
        _CLAMP_PARAM,
    ],
    "gradcam": [_TARGET_PARAM, _BATCH_PARAM],
    "neuron_gradient": [_CHANNEL_PARAM, _BATCH_PARAM],
    "neuron_ig": [
        _CHANNEL_PARAM,
        _BATCH_PARAM,
        ExperimentParam("ig_steps", "Integration steps", "int", 32, minimum=2),
    ],
    "occlusion": [
        _CHANNEL_PARAM,
        _BATCH_PARAM,
        ExperimentParam(
            "window",
            "Window (px)",
            "int",
            4,
            minimum=1,
            tooltip="Side length of the occluding patch",
        ),
        ExperimentParam("stride", "Stride (px)", "int", 2, minimum=1),
    ],
}

def default_param_values(overrides: dict[str, object]) -> dict[str, object]:
    """Every kind's per-key defaults, with session overrides applied.

    `overrides` is `Session.experiment_defaults` — e.g. a hosted playground
    seeds cheaper deep-dream defaults; anything not overridden keeps its
    `ExperimentParam.default`.
    """
    values: dict[str, object] = {}
    for specs in EXPERIMENT_PARAMS.values():
        for spec in specs:
            values.setdefault(spec.key, overrides.get(spec.key, spec.default))
    return values




# Per kind: (one-line summary, full description). The page shows the first
# as the dropdown's tooltip and the second at the bottom of its left pane;
# the MCP server returns both from `list_experiments`, which is the only
# way an agent learns what these methods actually do.
EXPERIMENT_DESCRIPTIONS: dict[str, tuple[str, str]] = {
    "deep_dream": (
        "Synthesize one input per channel that maximally excites it.",
        "Deep Dream runs gradient ascent on the input to maximize each of the "
        "layer's first channels' mean activation — one synthesized sample per "
        "channel, a picture of what each unit 'wants' to see.",
    ),
    "gradcam": (
        "Coarse class heatmap localized onto the selected layer.",
        "Grad-CAM weights the selected layer's feature maps by the gradient "
        "of the target class score, giving a coarse heatmap of where that "
        "layer supports the class.",
    ),
    "neuron_gradient": (
        "Raw input gradient of one channel — tends to look grainy.",
        "Neuron Gradient is the raw input-space gradient of one channel's "
        "activation — the gradient view of its receptive field. It tends to "
        "produce noisy, high-frequency (grainy) attribution maps; Neuron "
        "Integrated Gradients is the smoother alternative.",
    ),
    "neuron_ig": (
        "Path-integrated input attribution of one channel (cleaner).",
        "Neuron Integrated Gradients integrates one channel's input gradient "
        "along a path from a zero baseline, giving a cleaner, less noisy "
        "version of Neuron Gradient.",
    ),
    "occlusion": (
        "Drop in a channel's activation as input regions are occluded.",
        "Occlusion slides a patch over the input and measures how much the "
        "selected layer-channel's mean activation drops — a perturbation "
        "view of that channel's receptive field.",
    ),
}


def available_experiment_kinds() -> dict[str, str]:
    """Experiment kinds the UI should offer.

    captum is a standard dependency, so every kind (deep dream and the four
    Captum attributions) is always offered.
    """
    return dict(EXPERIMENT_KINDS)


def layer_available(session: Session, layer: str, kind: str) -> bool:
    """Whether experiment `kind` can run on `layer` (UI grays out the rest).

    Grad-CAM and the neuron methods need the layer's `nn.Module` object, so
    an fx intermediate is off-limits. Deep dream and occlusion read the
    activation through the fx interpreter and accept any captured layer — but
    in the hook fallback (no fx graph) the only thing to hook is a named
    module, so there every kind needs one.
    """
    if kind in _MODULE_KINDS or not session.fx_traced:
        return layer in dict(session.model.named_modules())
    return True


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
    # Record the run's published progress to an MP4 (`ExperimentClip`). Not a
    # knob of the experiment — the run is identical either way — but of how
    # its progress is delivered, which is why it is a field here rather than
    # an `EXPERIMENT_PARAMS` entry: the page streams the steps live and needs
    # nothing, while an MCP client only ever sees the result it polled.
    video: bool = False


@dataclass(frozen=True)
class ExperimentResult:
    """A progress snapshot or final outcome, fully resident on CPU.

    `done=False` results stream progress (deep dream publishes its evolving
    image); the final yield has `done=True` — also when aborted early, in
    which case `step < total_steps`. Exactly one of `image` (input-space,
    shown denormalized) or `attribution` (signed, shown with the diverging
    colormap) is set on success; `reference` carries the input batch the
    experiment started from.

    `video` is set on the final result of a run that asked for one
    (`ExperimentRequest.video`): the MP4 of every progress snapshot above,
    written by `run_experiment_guarded`, so the path arrives with the same
    result that says the run is done.
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
    video: str | None = None


@dataclass(frozen=True)
class ExperimentQueueState:
    """Where one request sits before its first result (`experiment_queue_state`).

    A request publishes nothing until it produces progress — Captum methods
    only publish once, at the end — so "no result yet" alone can't tell a
    request that is executing from one still waiting its turn. `stage` makes
    that difference visible: `"running"` (the training thread is on it now),
    `"queued"` (waiting behind `ahead` other requests, the running one
    included), or `"absent"` (never queued, cancelled, superseded, or long
    since finished).
    """

    stage: str
    ahead: int = 0


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

# Wall-clock ceiling on a single experiment run. The training thread is held
# for the whole run (and on a locked demo every queued visitor waits behind
# it), so a request that turns out heavier than its parameters suggested is
# cancelled instead of holding the pause loop hostage. Checked between steps
# (`should_abort`), so one long step can still overrun the deadline; the
# progress streamed so far publishes as the final result.
_EXPERIMENT_TIME_LIMIT: float = 90.0

# On a locked session (a shared demo, `Session.lock`), the queue is the one
# resource every visitor contends on — the pause loop runs experiments one at
# a time — so request parameters are clamped to these ceilings and the queue
# depth is capped. Values are chosen so the heaviest allowed request stays in
# the seconds range on the demo-scale models a locked session hosts.
_LOCKED_PARAM_LIMITS: dict[str, int] = {
    "steps": 300,  # deep-dream ascent steps
    "channels": 8,  # deep-dream channels (one sample each)
    "batch": 8,  # Captum input batch
    "ig_steps": 64,  # integrated-gradients interpolation steps
}
_LOCKED_MAX_QUEUE: int = 8


def _locked_params(params: dict[str, object]) -> dict[str, object]:
    """`params` with every capped numeric knob clamped to its ceiling."""
    out = dict(params)
    for key, limit in _LOCKED_PARAM_LIMITS.items():
        value = out.get(key)
        if isinstance(value, (int, float)):
            out[key] = min(int(value), limit)
    return out

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
    session: Session,
    *,
    kind: str,
    layer: str,
    params: dict[str, object],
    video: bool = False,
) -> int:
    """Implementation of `Session.request_experiment`."""
    if kind not in EXPERIMENT_KINDS:
        raise ValueError(
            f"unknown experiment kind {kind!r}; "
            f"expected one of {list(EXPERIMENT_KINDS)}"
        )
    with session._cv:
        session._experiment_seq += 1
        request = ExperimentRequest(
            kind=kind,
            layer=layer,
            params=(
                _locked_params(params) if session._locked else dict(params)
            ),
            seq=session._experiment_seq,
            # A locked demo shares one training thread between anonymous
            # visitors and caps every heavy knob for it; writing a video file
            # per request is exactly the kind of unbounded work that cap is
            # there to prevent, so recording is off there.
            video=video and not session._locked,
        )
        if session._locked and len(session._experiment_queue) >= _LOCKED_MAX_QUEUE:
            # Shared-demo backstop: publish a queue-full error for this seq
            # (the requesting page polls it like any result) instead of
            # letting one visitor pile up unbounded work. `_cv` is an RLock,
            # so publishing under the held lock is fine.
            _publish_experiment(
                session,
                _error(
                    request,
                    "the experiment queue is full — try again in a moment",
                ),
            )
            return request.seq
        session._experiment_queue.append(request)
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


def experiment_queue_state(session: Session, seq: int) -> ExperimentQueueState:
    """Implementation of `Session.experiment_queue_state`."""
    with session._cv:
        running = session._experiment_running
        if running == seq:
            return ExperimentQueueState("running")
        for position, request in enumerate(session._experiment_queue):
            if request.seq == seq:
                # A run already in flight is one more wait in front of it.
                return ExperimentQueueState(
                    "queued", position + (1 if running is not None else 0)
                )
        return ExperimentQueueState("absent")


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
        # A re-registration (e.g. auto-run on a parameter change) supersedes
        # this key's previous request: drop the old one if it is still queued
        # so a burst of edits never floods the pause loop with stale runs —
        # only the latest queued request for the key ever executes. A request
        # already mid-flight stays running (it cannot be un-run), but the
        # superseding one is queued behind it, so the view still ends on the
        # up-to-date parameters.
        prev = session._auto_experiments.get(key)
        if prev is not None:
            session._experiment_queue = deque(
                r for r in session._experiment_queue if r.seq != prev.request.seq
            )
        session._experiment_seq += 1
        request = ExperimentRequest(
            kind=kind,
            layer=layer,
            params=(
                _locked_params(params) if session._locked else dict(params)
            ),
            seq=session._experiment_seq,
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
    duplicate is dropped so the request runs exactly once per update.

    The batch's requests then go back on the queue in the order they run,
    ahead of anything the pause loop still holds (this publish path owns
    the training thread until the last of them finishes), and each is
    popped as it starts. So a request waiting its turn keeps reading as
    queued to `experiment_queue_state` instead of vanishing for the
    duration, and `cancel_experiment` on it still bites.
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
                requests + [r for r in session._experiment_queue if r.seq not in seqs]
            )
    for request in requests:
        with session._cv:
            # Pop and mark running under one lock, exactly as the pause loop
            # hands a request over: a cancel landing in between would
            # otherwise find the seq neither queued nor running and be a
            # silent no-op, letting a cancelled experiment run. A request
            # already gone from the queue *was* cancelled while it waited
            # its turn, so it is skipped rather than run.
            queued = deque(
                r for r in session._experiment_queue if r.seq != request.seq
            )
            if len(queued) == len(session._experiment_queue):
                continue
            session._experiment_queue = queued
            session._experiment_running = request.seq
        run_experiment_guarded(session, request)


def run_experiment_guarded(session: Session, request: ExperimentRequest) -> None:
    """Drive one experiment to completion on the training thread.

    Streams every yielded progress result through `_publish_experiment`.
    The abort predicate stops the run on `cancel_experiment` (for this
    seq), on the `_EXPERIMENT_TIME_LIMIT` wall-clock deadline expiring,
    and on anything that ends the pause — resume commands, a pending
    time-travel jump, `close()` — so the pause loop regains control
    promptly; queued requests from other clients wait their turn
    instead of aborting the run. A failing experiment publishes an
    error result instead of killing the training thread.

    A `video` request additionally draws every one of those results into an
    `ExperimentClip` as it goes, and the *final* result carries the finished
    file's path — so whoever is polling learns about the video from the same
    result that tells them the run is over, with nothing left to encode.
    """
    with session._cv:
        resume_seen = session._resume_token
        session._experiment_running = request.seq
    deadline = time.monotonic() + _EXPERIMENT_TIME_LIMIT

    def should_abort() -> bool:
        if time.monotonic() >= deadline:
            return True
        with session._cv:
            return (
                request.seq in session._experiment_cancelled
                or session._closed
                or session._pending_jump is not None
                or session._resume_token != resume_seen
            )

    clip = _experiment_clip(session, request)
    try:
        for partial in run(session, request, should_abort):
            if clip is not None:
                clip.append(partial)
                if partial.done:
                    partial = _with_video(partial, clip)
            _publish_experiment(session, partial)
    except Exception as e:  # noqa: BLE001 — surfaced via the result
        failure = _error(request, f"{type(e).__name__}: {e}")
        _publish_experiment(
            session, failure if clip is None else _with_video(failure, clip)
        )
    finally:
        if clip is not None:
            clip.finish()  # a no-op once `_with_video` has closed it
        with session._cv:
            session._experiment_running = None
            session._experiment_cancelled.discard(request.seq)


def _experiment_clip(
    session: Session, request: ExperimentRequest
) -> ExperimentClip | None:
    """The video recorder for `request`, or `None` when it wants no video.

    Imported lazily like `Session.recording` itself: `nansense.recording`
    pulls in the UI rendering stack, which a headless run has no reason to
    load until something actually records.
    """
    if not request.video:
        return None
    from nansense.recording import ExperimentClip

    return ExperimentClip.start(session, request)


def _with_video(result: ExperimentResult, clip: ExperimentClip) -> ExperimentResult:
    """`result` with the finished clip's path (or its failure) attached."""
    path = clip.finish()
    if path is not None:
        return replace(result, video=str(path))
    if clip.error is None:
        return result  # nothing was drawable, so there is no video to mention
    note = f"video recording failed: {clip.error}"
    return replace(
        result, error=note if result.error is None else f"{result.error}; {note}"
    )


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


def _captum_input(
    session: Session, request: ExperimentRequest
) -> Tensor | ExperimentResult:
    """The `[batch, C, H, W]` input batch a Captum run attributes, or an error.

    The first `batch` samples of the live snapshot input, so the page shows
    the whole batch like deep dream.
    """
    base = session._snapshot_input()
    if base is None:
        return _error(
            request, "no input available yet — run at least one batch first"
        )
    if base.ndim != 4:
        return _error(request, "experiments need an image input [B, C, H, W]")
    batch = max(1, int_param(request.params, "batch", _DEFAULT_DREAM_BATCH))
    count = min(batch, int(base.shape[0]))
    return base[:count].detach().clone().float()


def _dream_start(
    session: Session, request: ExperimentRequest, rng: torch.Generator, n: int
) -> Tensor | ExperimentResult:
    """The `[n, ...]` starting batch for deep dream, or an error.

    Deep dream runs one sample per channel over the layer's first `n`
    channels, so the batch size is the channel count. Built from the
    network's *real* input (the snapshot's input-node tensor), so non-image
    inputs work too: `start="noise"` draws `n` fresh samples matching the real
    input's per-sample shape and overall mean/std from `rng` — seeded per
    request, so successive runs explore different noise; `start="sample"`
    replicates the chosen input-batch sample `n` times, so every channel's
    dream starts from the same real image.
    """
    base = session._snapshot_input()
    if base is None:
        return _error(
            request, "no input available yet — run at least one batch first"
        )
    if base.ndim < 2:
        return _error(request, "deep dream needs a batched input [B, ...]")
    base = base.detach().float()
    if str(request.params.get("start", "noise")) != "noise":
        sample = int_param(request.params, "sample", 0)
        sample = max(0, min(sample, int(base.shape[0]) - 1))
        chosen = base[sample : sample + 1]
        return chosen.repeat(n, *([1] * (base.ndim - 1)))
    noise = torch.randn((n, *base.shape[1:]), generator=rng)
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


def _channels_objective(act: Tensor) -> Tensor:
    """Sum of each sample's own channel mean — sample i targets channel i.

    Deep dream runs one sample per channel over the layer's first channels, so
    the batch and channel axes are matched on the diagonal: sample i maximizes
    channel i's mean activation, and the per-sample gradient ascent drives each
    sample toward its channel. Falls back to the whole-tensor mean for an
    activation with no channel axis (a flat `[B]` shape).
    """
    if act.ndim < 2:
        return act.mean()
    n = min(int(act.shape[0]), int(act.shape[1]))
    idx = torch.arange(n, device=act.device)
    diag = act[idx, idx]  # [n, *spatial] — sample i's channel-i map
    return diag.reshape(n, -1).mean(dim=1).sum()


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
    """Gradient ascent over the layer's first channels — one sample per channel.

    The batch is sized to the `channels` knob and clipped to the layer's
    channel count, so sample i maximizes channel i's mean activation
    (`_channels_objective`) — or minimizes it when the `minimize` knob is on,
    descending the same objective to synthesize an input that suppresses each
    channel. The starting batch comes from the network's real
    input (`_dream_start`): fresh per-request noise by default, or the chosen
    sample of the current input batch replicated across the channels. The
    classic bag of regularizers, each optional and image-only (applied when
    the input is `[B, C, H, W]`): per-step jitter (random roll, undone after
    the update — drawn from the request-seeded generator), "diffusion"
    (blend with a 3×3 box blur, damping high-frequency noise), center zoom
    (a per-step multiplier), and clamping to the displayable value range.
    Gradients are normalized per sample by their mean magnitude so `lr`
    behaves comparably across layers and channels. `reference` (the shown
    input) is carried only for the current-batch start; noise has none.

    Progress publishes at step 0 — the untouched starting image — and then at
    `_PUBLISH_COUNT` evenly spaced steps, or at every step under the
    `all_steps` knob: the difference between a recorded run
    (`ExperimentRequest.video`) that skips through the ascent and one that
    replays all of it. Either way the first frame is the one nothing has
    happened to yet.
    """
    p = request.params
    steps = max(1, int_param(p, "steps", 300))
    lr = float_param(p, "lr", 0.05)
    diffusion = min(1.0, max(0.0, float_param(p, "diffusion", 0.05)))
    jitter = max(0, int_param(p, "jitter", 2))
    zoom = max(1.0, float_param(p, "zoom", 1.0))
    n_channels = max(1, int_param(p, "channels", _DEFAULT_DREAM_BATCH))
    clamp = bool_param(p, "clamp", True)
    # Minimize descends the same objective instead of ascending it, so the step
    # direction simply flips sign (the reported objective stays the signed
    # channel mean, which then falls over the run).
    direction = -1.0 if bool_param(p, "minimize", False) else 1.0
    from_sample = str(p.get("start", "noise")) != "noise"

    rng = torch.Generator().manual_seed(request.seq)
    x0 = _dream_start(session, request, rng, n_channels)
    if isinstance(x0, ExperimentResult):
        yield x0
        return
    spatial = x0.ndim == 4  # the regularizers below act on image axes only
    lo, hi = _value_bounds(int(x0.shape[1]), p.get("mean"), p.get("std"))
    publish_every = 1 if bool_param(p, "all_steps", False) else max(
        1, steps // _PUBLISH_COUNT
    )
    reference: Tensor | None = None

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
        # One sample per channel: clip the batch to the layer's channel count
        # so sample i targets channel i with no empty trailing samples.
        with torch.no_grad():
            probe = _target_activation(session, x[:1], request.layer)
        if probe.ndim >= 2:
            x = x[: min(int(x.shape[0]), int(probe.shape[1]))]
        # The starting input is shown only when dreaming from the current
        # batch (noise has nothing meaningful to show); all channels share it.
        if from_sample:
            reference = x[:1].detach().cpu()
        # Step 0 is the image the ascent has not touched yet — the noise it
        # starts from, or the sample it starts from. It is published like any
        # other progress frame because it is the run's own baseline: without
        # it the earliest frame anyone can see is already `publish_every`
        # steps up the climb, and a viewer has to take the starting point on
        # trust. Its objective is measured rather than assumed, so the series
        # a caller reads back begins at the value the run actually set out
        # from and "did this climb?" is a comparison against the start.
        with torch.no_grad():
            objective_value = float(
                _channels_objective(_target_activation(session, x, request.layer))
            )
        yield partial(x, 0, objective_value, done=False)
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
                objective = _channels_objective(act)
                (grad,) = torch.autograd.grad(objective, x_step)
            objective_value = float(objective.detach())
            sample_dims = tuple(range(1, grad.ndim))
            norm = grad.abs().mean(dim=sample_dims, keepdim=True)
            x = x_step.detach() + direction * lr * grad / (norm + 1e-8)
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
    # Outside the isolation context on purpose: the final result is what
    # `wait_for_experiment` and every polling client wake on, and a generator
    # suspended on a `yield` has not run `isolated_model`'s restore yet. Yield
    # it from inside and whoever wakes sees the model still flipped to eval.
    yield partial(x, step_done, objective_value, done=True)


def _resolve_target(
    model: nn.Module, x: Tensor, params: dict[str, object]
) -> int | list[int]:
    """The Grad-CAM target class (-1 means each sample's own argmax).

    An explicit class applies to the whole batch; the argmax default resolves
    per sample (a length-`batch` list), since the batch may span predictions.
    """
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
    return out.argmax(dim=1).tolist()


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


class _LayerChannelModel(nn.Module):
    """Exposes one layer-channel's per-sample mean activation as the output.

    Lets Occlusion attribute input regions against an *intermediate* channel
    (instead of an output class), so it respects the layer + channel
    selection like the neuron methods. The output is `[B, 1]`, so Captum
    takes target 0; forwards go through `_target_activation`, which handles
    fx intermediates and the hook fallback alike.
    """

    def __init__(self, session: Session, layer: str, channel: int) -> None:
        super().__init__()
        self._session = session
        self._layer = layer
        self._channel = channel

    def forward(self, x: Tensor) -> Tensor:
        act = _target_activation(self._session, x, self._layer)
        return _neuron_selector(self._channel)(act).reshape(-1, 1)


def _run_captum(
    session: Session,
    request: ExperimentRequest,
    should_abort: Callable[[], bool],
) -> Iterator[ExperimentResult]:
    """Captum attribution over a batch, one attribution per sample.

    All methods run the *unpatched* model (probes/experiments only execute
    between batches) inside the isolation scope. Gradient-based methods use
    `torch.autograd.grad` internally, so parameter `.grad` survives.
    """
    p = request.params
    x0 = _captum_input(session, request)
    if isinstance(x0, ExperimentResult):
        yield x0
        return
    module: nn.Module | None = dict(session.model.named_modules()).get(request.layer)
    if request.kind in _MODULE_KINDS and module is None:
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

    def attribute(x: Tensor) -> Tensor:
        kind = request.kind
        if kind == "gradcam":
            assert module is not None  # checked above
            target = _resolve_target(session.model, x, p)
            out = captum_attr.LayerGradCam(session.model, module).attribute(
                x, target=target
            )
        elif kind == "neuron_gradient":
            assert module is not None  # checked above
            out = captum_attr.NeuronGradient(session.model, module).attribute(
                x, neuron_selector=_neuron_selector(int_param(p, "channel", 0))
            )
        elif kind == "neuron_ig":
            assert module is not None  # checked above
            out = captum_attr.NeuronIntegratedGradients(
                session.model, module
            ).attribute(
                x,
                neuron_selector=_neuron_selector(int_param(p, "channel", 0)),
                n_steps=max(2, int_param(p, "ig_steps", 32)),
            )
        else:  # occlusion, retargeted to the selected layer-channel
            channels = int(x.shape[1])
            window = max(1, int_param(p, "window", 4))
            stride = max(1, int_param(p, "stride", 2))
            target_model = _LayerChannelModel(
                session, request.layer, int_param(p, "channel", 0)
            )
            out = captum_attr.Occlusion(target_model).attribute(
                x,
                target=0,
                sliding_window_shapes=(channels, window, window),
                strides=(channels, stride, stride),
                baselines=0.0,
                perturbations_per_eval=8,
            )
        assert isinstance(out, Tensor)
        return out

    with isolated_model(session, "eval") as device:
        attribution = attribute(x0.to(device))

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
