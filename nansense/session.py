"""NaNsense session: state machine, batch lifecycle, snapshot publishing.

A `Session` is created once per training run via `nansense.start(...)`.
The user wraps each batch with `with session.batch(phase=..., epoch=...)`:

    session = nansense.start(model, epochs=50, phases={"train": 196, "val": 40})
    for epoch in range(50):
        for batch in train_loader:
            with session.batch(phase="train", epoch=epoch):
                optimizer.zero_grad()
                loss = ...
                loss.backward()
                optimizer.step()

Because `optimizer.zero_grad()` lives at the start of the user's batch body,
parameter `.grad` is still populated when the context manager exits, so the
session reads it straight off the model — no backward hooks needed.

The UI (added later) drives the session by calling `stop`, `step_batch`,
`step_phase`, `step_epoch`, `step_run`, `step_until_position`, `detach`,
and finally `close`.
Whether the session captures activations/gradients for a given batch is
decided up-front at `__enter__` from the schedule + current mode, so
forward hooks are only installed for batches that will actually be
inspected.

The heavy lifting lives in sibling modules that this one delegates to:
`nansense.capture` (hook installation, fx interpretation, live weight /
optimizer reads), `nansense.probe` (pinned-input probe state and runs),
and `nansense.experiments` (the experiment queue and runners).
"""

from __future__ import annotations

import contextlib
import sys
import threading
import warnings
from collections import OrderedDict, deque
from collections.abc import Callable, Iterable, Iterator, Mapping
from dataclasses import dataclass, field, replace
from pathlib import Path
from types import TracebackType
from typing import TYPE_CHECKING, TypeVar

if sys.version_info >= (3, 11):
    from enum import StrEnum
    from typing import Self
else:  # Python 3.10: enum.StrEnum and typing.Self both landed in 3.11.
    from enum import Enum

    from typing_extensions import Self

    class StrEnum(str, Enum):
        """Minimal backport of `enum.StrEnum` for Python 3.10."""

        __str__ = str.__str__

from torch import Tensor, fx, nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.hooks import RemovableHandle

from nansense import capture, debugger, distributed, experiments, probe
from nansense.debugger import DebugError, DebugSettings
from nansense.input_config import InputTransform, MeanStd
from nansense.experiments import (
    ExperimentRequest,
    ExperimentResult,
    _AutoExperiment,
)
from nansense.instruments import (
    InstrumentManager,
    LayerContext,
    MetricReduce,
    MetricsSnapshot,
    WeightContext,
)
from nansense.probe import PerturbationMap, ProbeResult
from nansense.restore import (
    DEFAULT_CACHE_DIR,
    TimeTravelError,
    TimeTravelJump,
    TimeTravelStatus,
    TrainingRestorer,
    validate_model_state,
    validate_optimizer_state,
    validate_scheduler_state,
)
from nansense.schedule import BatchPosition, Schedule, format_position
from nansense.watch import (
    DEFAULT_AVERAGE_PATCHES,
    DEFAULT_CHANNEL_LIMIT,
    DEFAULT_SAMPLES_PER_CHANNEL,
    LayerStatsSnapshot,
    WatchAccumulator,
    WatchSnapshot,
    single_batch_stats,
)

if TYPE_CHECKING:
    from nansense.recording import RecordingManager

# Grace period an *unserved* enabled session waits at a pause before giving up
# and detaching (see `Session._wait_for_proceed`). A served session — or one
# driven from another thread, like the test harness — resumes long before this;
# it only fires for a single-threaded script that paused with no way to resume.
_UNSERVED_PAUSE_TIMEOUT = 30.0


def _warn_unserved_detach() -> None:
    warnings.warn(
        "NaNsense: the session paused on a batch but no UI is serving it and "
        f"nothing resumed it within {_UNSERVED_PAUSE_TIMEOUT:.0f}s. Continuing "
        "without pausing (detached). Pass `port=` to nansense.start() (or call "
        "nansense.serve()) to drive the UI, or use enabled=False for plain "
        "training.",
        RuntimeWarning,
        stacklevel=3,
    )


class Mode(StrEnum):
    """How training proceeds after a resume, exposed as `Session.mode`.

    Each mode maps to a top-bar control: `STEP` pauses on every batch (Step
    Batch; also the initial mode, so the first batch always pauses),
    `UNTIL_PHASE_CHANGE` / `UNTIL_EPOCH_CHANGE` run to the first batch of the
    next phase/epoch (Step Phase / Step Epoch), `UNTIL_POSITION` runs to an
    exact position (the step-until dialog), `UNTIL_END` runs to the run's
    last batch (Run), and `DETACH` never pauses again — capture overhead
    drops to near zero until the UI re-engages.
    """

    STEP = "step"
    UNTIL_PHASE_CHANGE = "until_phase_change"
    UNTIL_EPOCH_CHANGE = "until_epoch_change"
    UNTIL_END = "until_end"
    UNTIL_POSITION = "until_position"
    DETACH = "detach"


class StatsScope(StrEnum):
    """Which layers fold their batches into the running statistics.

    - `NONE`: nothing is collected; already-collected stats are kept frozen
      (the pause the top bar's stats toggle uses).
    - `WATCHED` (default): the watched layers collect, and watching a layer is
      also what shows its cards — the classic coupled behaviour.
    - `ALL`: every layer in `layer_names` collects, independent of the watched
      set.

    Outside `WATCHED`, the watched set no longer drives collection, so the UI
    treats showing/hiding a layer's cards as per-tab state that never touches
    the session (see `main_page`).
    """

    NONE = "none"
    WATCHED = "watched"
    ALL = "all"


FREQUENCY_UNITS: tuple[str, ...] = ("batch", "epoch")

# Element type of a loader passed to `Session.batches` (PEP 695 `def batches[T]`
# would require Python 3.12; this keeps the floor at 3.10).
_BatchItem = TypeVar("_BatchItem")

# The `watch_metric` / `watch_layer_tensor` / `watch_weight_tensor`
# decorators return the callable unchanged, so stacking and direct calls
# (`session.watch_metric("x")(fn)`) both keep the original type.
_InstrumentFn = TypeVar("_InstrumentFn", bound="Callable[..., object]")

# Live (uncloned) views for instrument contexts: named parameters, each
# parameter's optimizer state, and its group's numeric hyperparameters.
_InstrumentSources = tuple[
    dict[str, Tensor],
    dict[str, dict[str, Tensor | float]],
    dict[str, dict[str, float]],
]


@dataclass(frozen=True)
class UpdateFrequency:
    """How often visualizations refresh while training runs.

    `unit="epoch"` updates on the first batch of every `n`-th epoch
    (0, n, 2n, …), detecting the boundary from the epoch number the same way
    Step Epoch does; `unit="batch"` updates on every `n`-th batch — counting
    only `phase`'s batches when one is given. A frequency update publishes a
    snapshot, re-runs the probe and any live auto experiments, and feeds
    recording frames — all *without pausing*, in addition to the mode-driven
    captures that pause training. The default is one update per epoch.
    """

    unit: str = "epoch"
    n: int = 1
    phase: str | None = None


@dataclass(frozen=True)
class WatchPerformance:
    """Per-channel watch caps that bound GPU VRAM use (see `nansense.watch`).

    Watched layers keep, per channel, a histogram and a gallery of extreme
    input patches; the patches store a per-channel input image, so their cost
    scales with the channel count. `channel_limit_enabled` caps per-channel
    data to the first `channel_limit` channels (the layer-wide histogram and
    scalars always cover all channels); `samples_per_channel` is how many
    extreme samples are kept per channel per ranking; `average_patches`
    enables the max/min-average patch grids, which store a whole input image
    per slot (off by default — the pixel grids are the ones usually
    consulted). Changing any field flushes all watch statistics, since the
    buffer shapes change.
    """

    channel_limit_enabled: bool = True
    channel_limit: int = DEFAULT_CHANNEL_LIMIT
    samples_per_channel: int = DEFAULT_SAMPLES_PER_CHANNEL
    average_patches: bool = DEFAULT_AVERAGE_PATCHES


@dataclass(frozen=True)
class BatchSnapshot:
    """Immutable per-batch view, fully resident on CPU.

    All tensor dicts are independent CPU clones taken at snapshot time,
    so the snapshot survives subsequent batches freeing the live tensors and
    can be safely read from any thread.

    `position` is where the batch sat in the run. `activations` /
    `activation_gradients` are keyed by watched-layer name (the names shown
    in the architecture graph); `weights` / `weight_gradients` by parameter
    name, matching `model.named_parameters()`.

    `optimizer_state` / `optimizer_hyperparams` are populated only when the
    session was given an optimizer at `start()`; otherwise they stay empty.
    State entries are keyed `param name -> state key -> tensor` (scalar
    entries like Adam's `step` become 0-dim tensors); hyperparams are the
    numeric knobs of the parameter's group (`lr`, `momentum`, ...), read at
    the same instant — so a scheduler-driven `lr` is the batch's actual one.

    `custom_activations` / `custom_weight_tensors` carry the tensor
    instruments' outputs (see `Session.watch_layer_tensor` /
    `watch_weight_tensor`): activation-shaped tensors keyed
    `layer -> instrument name`, weight-shaped ones `param -> instrument
    name`. Empty without registered instruments (or when nothing collects).
    """

    position: BatchPosition
    activations: dict[str, Tensor]
    activation_gradients: dict[str, Tensor]
    weights: dict[str, Tensor]
    weight_gradients: dict[str, Tensor]
    optimizer_state: dict[str, dict[str, Tensor]] = field(default_factory=dict)
    optimizer_hyperparams: dict[str, dict[str, float]] = field(default_factory=dict)
    custom_activations: dict[str, dict[str, Tensor]] = field(default_factory=dict)
    custom_weight_tensors: dict[str, dict[str, Tensor]] = field(
        default_factory=dict
    )


class Session:
    """The bridge between a live training loop and the NaNsense UI.

    Create one with `nansense.start` (the intended entry point) rather than
    directly. The training loop drives the session through `batches` (wrap
    each phase's dataloader), `epochs` + `restore_point` (the time-travel
    epoch loop), and `close` when training finishes; the served UI drives
    pausing, stepping, layer watching and experiments through the rest of the
    surface. With `enabled=False` every method is a near-zero-overhead no-op.
    """

    def __init__(
        self,
        model: nn.Module,
        *,
        epochs: int | None = None,
        phases: dict[str, int] | None = None,
        enabled: bool = True,
        optimizer: Optimizer | None = None,
        scheduler: LRScheduler | None = None,
    ) -> None:
        # A DDP-wrapped model is unwrapped up front: hooks on the inner
        # module still fire through the wrapper's forward, while names and
        # the fx trace stay clean (no `module.` prefix, traceable graph).
        model = distributed.unwrap_ddp(model)
        self.model = model
        self._enabled = enabled
        # Set by `nansense.serve` once a UI is actually served. An enabled but
        # unserved session has no way to be resumed from a pause, so it bounds
        # each pause with a grace timeout instead of deadlocking (see
        # `_wait_for_proceed`); a served session waits for the UI indefinitely.
        self._served = False
        # Multi-rank coordination (None outside distributed runs): rank 0
        # leads (UI, snapshots, pauses), other ranks follow — they sync the
        # watched set per batch and join the watch-stats reductions, but
        # never serve, publish, or pause. See `nansense.distributed`.
        self._dist: distributed.DistContext | None = (
            distributed.context() if enabled else None
        )
        self._watch_version = 0
        # Last cross-rank-reduced watch stats (leader only); overlaid on the
        # local view by `watch_snapshot`. Atomic-reference read contract.
        self._dist_watch_stats: distributed.ReducedStats | None = None
        self._optimizer = optimizer
        self._scheduler = scheduler
        self._schedule = Schedule(epochs=epochs, phases=phases)
        self._mode: Mode = Mode.STEP
        # UNTIL_POSITION target as (phase_index, epoch, batch_idx) — phase by
        # index into the (possibly still-growing) phase order, so a target in a
        # not-yet-observed phase/epoch is matched once training reaches it.
        self._target_position: tuple[int, int, int] | None = None
        # The (phase, epoch) the live run sat at when "step phase"/"step epoch"
        # was issued. The step modes compare against it to land on the *first*
        # batch of the next phase/epoch — a position comparison that needs no
        # prospective is_last_* flags, so it works on the first (unlearned) epoch.
        self._step_origin: tuple[str, int] | None = None
        self._restorer: TrainingRestorer | None = None
        # The restorer that backs the flat `session.epochs()` / `restore_point()`
        # loop, created lazily on the first `epochs()` call. Kept distinct from
        # `_restorer` (which stays None on a disabled session) so `restore_point()`
        # can reach it regardless, while time travel itself only arms when enabled.
        self._loop_restorer: TrainingRestorer | None = None
        # The epoch `session.epochs()` last yielded, so `session.batches()` can
        # default its `epoch=` to the current epoch instead of taking it again.
        self._current_epoch = 0
        # Tracks which epoch's start checkpoint has already been written for
        # the current attempt, so the pre-iter save (in `batches`) and the
        # fallback save (in `_BatchContext.__enter__`) don't double-save — and
        # so the post-iter `__enter__` save can't clobber the good pre-iter one
        # with a now-stale RNG state. Reset on a time-travel rewind.
        self._epoch_start_saved_for: int | None = None
        self._pending_jump: int | None = None
        # One-shot UI request to publish the next batch's snapshot (consumed at
        # the batch boundary, like `_pending_jump`). The views only refresh on
        # a published snapshot, so in detach / step_run — which run freely and
        # publish only on the frequency cadence — this lets the Refresh button
        # force the next batch to publish without pausing or recomputing.
        self._snapshot_request = False
        # One-shot moment freeze, armed by `freeze_moment` as
        # (path, phase, epoch, batch_idx) and consumed by the exactly-matching
        # batch, which publishes and writes the file (see `nansense.moments`).
        self._freeze_request: tuple[Path, str, int, int] | None = None
        self._cv = threading.Condition()
        self._resume_token = 0
        self._pause_count = 0
        # True while the training thread sits in `_wait_for_proceed` (paused at
        # a batch), False while it is actively advancing batches. The top bar
        # reads it through `is_running` to gray out Run while running and Stop
        # while stopped.
        self._paused = False
        self._closed = False
        self._activations: dict[str, Tensor] = {}
        self._hook_handles: list[RemovableHandle] = []
        self._snapshot: BatchSnapshot | None = None
        self._live_position: BatchPosition | None = None
        # When disabled, skip the up-front fx trace and all name/weight
        # discovery entirely — symbolic_trace runs a proxy forward pass and is
        # the only expensive part of construction. A disabled session never
        # installs hooks, so these stay empty and every per-batch path
        # short-circuits.
        self._fx_graph: fx.GraphModule | None = (
            capture.try_trace(model) if enabled else None
        )
        self._input_names: list[str] = (
            capture.compute_input_names(self._fx_graph, model) if enabled else []
        )
        self._layer_names: list[str] = (
            capture.compute_layer_names(self._fx_graph, model, self._input_names)
            if enabled
            else []
        )
        self._layer_weights: dict[str, list[str]] = (
            capture.compute_layer_weights(self._fx_graph, model, self._input_names)
            if enabled
            else {}
        )
        self._layer_info: dict[str, str] = (
            capture.compute_layer_info(self._fx_graph, model, self._input_names)
            if enabled
            else {}
        )
        self._original_forward: object | None = None
        self._had_instance_forward: bool = False
        self._watched_layers: set[str] = set()
        # Which layers fold their batches into the running stats (see
        # `StatsScope`). `NONE` collects nothing but keeps existing buckets
        # frozen (and non-publishing batches skip the capture-mode hook
        # install they'd otherwise pay for stats); `_prev_stats_scope` is the
        # last collecting scope, restored by the top bar's stats toggle.
        self._stats_scope = StatsScope.WATCHED
        self._prev_stats_scope = StatsScope.WATCHED
        # One-way switch for shared (demo) deployments: run control and every
        # global setting refuse to change once locked (see `lock`).
        self._locked = False
        self._watch_accumulator = WatchAccumulator()
        # User-registered instruments (custom metrics / tensors) and their
        # scalar series store — see `nansense.instruments`.
        self._instruments = InstrumentManager()
        # Per-channel watch caps (GPU VRAM); the accumulator defaults already
        # match, so no initial `configure` flush is needed.
        self._watch_performance = WatchPerformance()
        self._patch_layers: frozenset[str] | None = None
        # Probe state (see nansense.probe). Config fields are mutated by the
        # UI thread under `_cv`; `_probe_result` is published by the training
        # thread (also under `_cv`, so a stale in-flight run can be detected
        # via `_probe_version` and dropped instead of overwriting newer
        # config's result).
        self._pinned_inputs: dict[str, Tensor] | None = None
        self._pinned_position: BatchPosition | None = None
        self._perturbations: PerturbationMap = {}
        self._probe_mode: str = "unchanged"
        self._probe_request = False
        self._probe_version = 0
        self._probe_count = 0
        self._probe_result: ProbeResult | None = None
        self._probe_error: str | None = None
        # Experiment state (see nansense.experiments): requests queue up and
        # the pause loop drains them in order, so concurrent clients (browser
        # tabs) don't supersede each other. Results are kept per request seq
        # (bounded; each client polls its own via `experiment_result_for`)
        # alongside the latest one; cancellation is per seq too.
        self._experiment_queue: deque[ExperimentRequest] = deque()
        self._experiment_seq = 0
        self._experiment_results: OrderedDict[int, ExperimentResult] = OrderedDict()
        self._experiment_result: ExperimentResult | None = None
        self._experiment_cancelled: set[int] = set()
        self._experiment_running: int | None = None
        # Visualization update frequency (see `UpdateFrequency`): mutated by
        # the UI under `_cv`; `_freq_counter` (batch unit) and `_freq_epoch`
        # (epoch unit, the last epoch the detector saw) are touched by the
        # training thread only.
        self._update_frequency = UpdateFrequency()
        self._freq_counter = 0
        self._freq_epoch: int | None = None
        # Experiments re-run on every update, keyed by the registering
        # client (a UI page or a recording). Mutated under `_cv`.
        self._auto_experiments: dict[str, _AutoExperiment] = {}
        # Session-wide "auto-run experiments" preference (shared across tabs):
        # when set, experiment pages re-run on every parameter change instead
        # of waiting for a manual Run (the init run self-starts either way).
        # Default on.
        self._auto_run_experiments = True
        # Per-key overrides for the experiment form's default parameter
        # values (see `set_experiment_defaults`). Mutated under `_cv`.
        self._experiment_defaults: dict[str, object] = {}
        # Per-view video recording (see `nansense.recording`); created
        # lazily on first UI access so headless sessions never import the
        # rendering stack.
        self._recording_manager: RecordingManager | None = None
        # Numerical-error debugger (see `nansense.debugger`): settings are
        # mutated by the UI under `_cv`; `_debug_counter` counts batches on
        # the training thread to throttle checks to every nth batch; the
        # published `_debug_error` is read lock-free by the UI (atomic ref).
        self._debug_settings = DebugSettings()
        self._debug_counter = 0
        self._debug_error: DebugError | None = None

    @property
    def schedule(self) -> Schedule:
        return self._schedule

    @property
    def optimizer(self) -> Optimizer | None:
        return self._optimizer

    @property
    def scheduler(self) -> LRScheduler | None:
        return self._scheduler

    @property
    def mode(self) -> Mode:
        with self._cv:
            return self._mode

    @property
    def snapshot(self) -> BatchSnapshot | None:
        # A plain lock-free read (assignment is atomic under the GIL). `None`
        # means either nothing has published yet or a publish is mid-flight:
        # `_publish_snapshot` drops the old snapshot before building the new
        # one so the two never stack in CPU memory. Readers never block on it —
        # the render loop treats a `None` like a frame with no new data and
        # keeps the prior frame, re-rendering once the next read sees the new
        # snapshot. Blocking here instead would stall the UI's asyncio event
        # loop (the render tick reads this synchronously before offloading).
        return self._snapshot

    @property
    def live_position(self) -> BatchPosition | None:
        """Position of the batch the training thread is currently on.

        Updated on *every* batch's `__enter__` regardless of capture mode, so
        the UI can show live epoch/batch progress during `step_epoch`,
        `step_until_position`, `step_run`, and `detach` — modes that publish a
        snapshot only at boundaries (or never), leaving `snapshot.position`
        frozen in between. Written by the training thread, read point-in-time
        by the UI thread: a single atomic reference assignment under the GIL,
        no lock needed (same contract as `snapshot`).
        """
        return self._live_position

    @property
    def closed(self) -> bool:
        with self._cv:
            return self._closed

    @property
    def enabled(self) -> bool:
        """Whether this session captures anything. Set once at `start()`.

        A disabled session is fully inert: `batch()` is a no-op context
        manager, `serve()` does nothing, and no model hooks are ever
        installed — the intended near-zero-overhead off switch for leaving
        NaNsense wiring in place in a training script.
        """
        return self._enabled

    def mark_served(self) -> None:
        """Record that a UI is now serving this session.

        Called by `nansense.serve`. While served, a pause waits for the UI
        indefinitely; while unserved, `_wait_for_proceed` bounds the wait so an
        enabled-but-undriven session can't deadlock (it detaches instead).
        """
        self._served = True

    @property
    def is_leader(self) -> bool:
        """Whether this session drives the run's UI and snapshots.

        Always True outside distributed training. In a multi-rank run only
        rank 0 leads; follower sessions never serve, publish snapshots, or
        pause — they contribute their shard's watch stats to the leader's
        global view (see `nansense.distributed`).
        """
        return self._dist is None or self._dist.is_leader

    @property
    def input_names(self) -> list[str]:
        return list(self._input_names)

    @property
    def input_batch_size(self) -> int | None:
        """Batch size of the last snapshot's input (None before any batch)."""
        base = self._snapshot_input()
        if base is None or base.ndim < 1:
            return None
        return int(base.shape[0])

    @property
    def layer_names(self) -> list[str]:
        """Ordered list of every per-batch tensor key the snapshot may carry.

        In fx mode, this is the friendly name of every non-output node in the
        traced graph: inputs (`x`), module outputs (`stage1.0.bn1`), and
        scope-qualified function/method results (`stage1.0.relu1`). In the
        hook fallback, it's `input_names + named_modules`.
        """
        return list(self._layer_names)

    @property
    def layer_weights(self) -> dict[str, list[str]]:
        """Map each layer name to the parameter names it uses.

        Keys match `layer_names`; every layer has an entry, with an empty
        list for layers that consume no weights (graph inputs, `relu`, `add`,
        …). Values are qualified parameter names that index into a
        `BatchSnapshot.weights` / `.weight_gradients` dict.

        In fx mode the mapping is exact: a `call_module` node owns the
        parameters under its dotted target, and any node that references a
        parameter functionally pulls it in via a `get_attr` input. In the
        hook fallback a module maps to every parameter in its subtree (the
        weights that contributed to the output the hook captured).
        """
        return {name: list(params) for name, params in self._layer_weights.items()}

    @property
    def layer_info(self) -> dict[str, str]:
        """Map each layer name to a human-readable hyperparameter string.

        Keys match `layer_names`. Module layers carry their
        `print(model)`-style signature (`Conv2d(3, 64, kernel_size=(3, 3),
        ...)` — built from `extra_repr()`, which custom modules can override
        to surface their own knobs); fx function/method ops carry their
        literal call arguments (`max_pool2d(2, stride=None, ...)`); layers
        with nothing to report (graph inputs, `relu`, `add`, …) map to "".
        The UI shows these as hover tooltips on diagram nodes and layer
        cards.
        """
        return dict(self._layer_info)

    @property
    def fx_traced(self) -> bool:
        return self._fx_graph is not None

    @property
    def pause_count(self) -> int:
        with self._cv:
            return self._pause_count

    @property
    def is_running(self) -> bool:
        """Whether the training thread is actively advancing batches.

        False while paused at a batch (waiting for a UI Step/Run/Stop command)
        or once the session has closed; True while a step/run/detach is in
        flight. The top bar grays out Run while running and Stop while stopped.
        """
        with self._cv:
            return not self._paused and not self._closed

    def batch(
        self, *, phase: str, epoch: int, item: object = None
    ) -> _BatchContext:
        """One batch boundary; see `batches()` for the loop sugar.

        `item` is the loader's yielded batch (typically `(inputs, targets)`),
        needed only when an armed `freeze_moment` may trigger on this batch —
        the moment stores it so `load_moment` can replay the forward/backward
        instead of storing every activation tensor. `batches()` passes it
        automatically.
        """
        return _BatchContext(self, phase=phase, epoch=epoch, item=item)

    def batches(
        self, loader: Iterable[_BatchItem], *, phase: str, epoch: int | None = None
    ) -> Iterator[_BatchItem]:
        """Iterate `loader` with each item wrapped in a `batch()` context.

        Sugar over `batch()` for the common loop shape: the user's batch
        body runs while the generator is suspended at `yield`, i.e. inside
        the batch context — hooks are installed before the forward pass and
        the capture/pause happens when the loop asks for the next item.
        A `TimeTravelJump` raised at a batch boundary therefore surfaces
        from the `for` statement itself, not from inside the user's body.

            for inputs, targets in session.batches(loader, phase="train"):
                ...  # forward / backward / step

        `epoch` defaults to the epoch `session.epochs()` last yielded, so the
        flat loop need not repeat it; pass it explicitly when driving the
        phases outside an `epochs()` loop.
        """
        if epoch is None:
            epoch = self._current_epoch
        # Checkpoint the epoch-start state BEFORE `iter(loader)` draws the
        # DataLoader's shuffle seed from the global RNG. The `__enter__` save
        # (a fallback for users who drive `batch()` manually) would capture the
        # RNG *after* the draw, so restoring it on replay produces a different
        # shuffle order. Saving here, pre-iter, makes the replay deterministic.
        self._maybe_save_epoch_start(epoch)
        count = 0
        for item in loader:
            with self.batch(phase=phase, epoch=epoch, item=item):
                yield item
            count += 1
        # The loop ran to completion (no early break / time-travel jump): teach
        # a lazily-discovered schedule this phase's batch count, so the next
        # epoch's is_last_* flags are exact. No-op when phases were declared.
        self._schedule.record_phase_length(phase, epoch, count)

    @property
    def watched_layers(self) -> frozenset[str]:
        """Immutable snapshot of the currently-watched layer names."""
        with self._cv:
            return frozenset(self._watched_layers)

    def watch(self, layer: str) -> bool:
        """Start collecting stats for `layer`. Returns False for unknown names.

        Any name that appears in `Session.layer_names` is watchable: named
        modules, graph inputs (e.g. `x`), and fx-traced intermediate ops
        (`relu`, `add`, `mean`). Watching activates the full capture
        machinery on every batch — fx interpreter when traceable, root
        pre-hook + per-module hooks otherwise — so the visualisation runs
        at capture-mode speed regardless of pause behaviour. Real training
        runs should not enable the UI.
        """
        if self._locked or layer not in self._layer_names:
            return False
        with self._cv:
            self._watched_layers.add(layer)
            self._watch_version += 1
        return True

    def unwatch(self, layer: str) -> None:
        """Stop watching `layer` and drop any stats already collected for it."""
        if self._locked:
            return
        with self._cv:
            self._watched_layers.discard(layer)
            self._watch_version += 1
        self._watch_accumulator.forget_layer(layer)
        self._instruments.forget_layer(layer)

    def watch_metric(
        self,
        name: str,
        *,
        on: str = "batch",
        reduce: MetricReduce | None = None,
    ) -> Callable[[_InstrumentFn], _InstrumentFn]:
        """Register a custom scalar metric, evaluated per collected layer.

        Use as a decorator (or call the returned registrar directly with any
        callable — a stateful object works too):

            @session.watch_metric("sparsity")
            def sparsity(ctx: nansense.LayerContext) -> float:
                return (ctx.activation > 0).float().mean().item()

        The callback runs on the training thread, under `torch.no_grad()`,
        once per stats batch for every layer the stats scope collects (the
        watched set by default) — it receives a `LayerContext` with the
        batch's live activation, its gradient, and the layer's weights /
        optimizer state, and must not mutate them. It may return a number
        (or 1-element tensor), a mapping of named scalars (one plot trace
        per key), or `None` to skip the layer.

        `on="batch"` plots every batch's value; `on="epoch"` folds each
        epoch's values through `reduce` — `"mean"` (default), `"sum"`,
        `"min"`, `"max"`, `"last"`, or any `values -> float` callable — into
        one point. The series appear in the `/stats` GRAPHS view, one plot
        per metric. A raising callback is disabled (training continues) and
        reported via `instrument_errors`. Raises `ValueError` on invalid
        arguments and `RuntimeError` on a locked session.
        """
        return self._register_instrument(
            name, kind="metric", on=on, reduce=reduce
        )

    def watch_layer_tensor(
        self, name: str
    ) -> Callable[[_InstrumentFn], _InstrumentFn]:
        """Register a custom activation-shaped tensor per collected layer.

            @session.watch_layer_tensor("zscore")
            def zscore(ctx: nansense.LayerContext) -> Tensor:
                a = ctx.activation
                return (a - a.mean()) / (a.std() + 1e-6)

        The callback follows `watch_metric`'s contract (training thread,
        `torch.no_grad()`, live tensors, `None` skips) but runs on publish
        batches only and must return a tensor of the activation's shape —
        it lands on `BatchSnapshot.custom_activations` and renders as an
        extra strip under the layer card's activation/gradient strips.
        """
        return self._register_instrument(name, kind="layer_tensor")

    def watch_weight_tensor(
        self, name: str
    ) -> Callable[[_InstrumentFn], _InstrumentFn]:
        """Register a custom weight-shaped tensor per collected parameter.

            @session.watch_weight_tensor("adam_update")
            def adam_update(ctx: nansense.WeightContext) -> Tensor | None:
                if "exp_avg" not in ctx.optimizer_state:
                    return None
                m = ctx.optimizer_state["exp_avg"]
                v = ctx.optimizer_state["exp_avg_sq"]
                return m / (v.sqrt() + 1e-8)

        The callback follows `watch_metric`'s contract but runs on publish
        batches only, once per (collected layer, parameter), receiving a
        `WeightContext`; it must return a tensor of the parameter's shape —
        it lands on `BatchSnapshot.custom_weight_tensors` and renders on the
        `/weights` page alongside the weight/gradient/optimizer strips.
        """
        return self._register_instrument(name, kind="weight_tensor")

    def _register_instrument(
        self,
        name: str,
        *,
        kind: str,
        on: str = "batch",
        reduce: MetricReduce | None = None,
    ) -> Callable[[_InstrumentFn], _InstrumentFn]:
        if self._locked:
            # Unlike the UI-driven setters (which no-op quietly), registering
            # from the hosting script after `lock()` is a programming error —
            # fail loudly instead of silently never evaluating.
            raise RuntimeError(
                "cannot register instruments on a locked session"
            )

        def register(fn: _InstrumentFn) -> _InstrumentFn:
            self._instruments.register(
                name, kind=kind, fn=fn, on=on, reduce=reduce
            )
            return fn

        return register

    @property
    def instrument_errors(self) -> dict[str, str]:
        """`instrument name -> error` for instruments disabled by a failure."""
        return self._instruments.errors()

    def watch_metrics_snapshot(
        self, *, layers: Iterable[str] | None = None
    ) -> MetricsSnapshot:
        """Frozen view of the custom scalar-metric series (see `watch_metric`).

        The GRAPHS view's data source for the custom-metric plots; safe to
        call from any thread. Pass `layers` to restrict the copy.
        """
        return self._instruments.metrics_snapshot(layers=layers)

    @property
    def stats_scope(self) -> StatsScope:
        """Which layers fold their batches into the running stats."""
        with self._cv:
            return self._stats_scope

    @property
    def stats_collecting(self) -> bool:
        """Whether any running stats are being collected (scope ≠ `NONE`)."""
        with self._cv:
            return self._stats_scope is not StatsScope.NONE

    def set_stats_scope(self, scope: StatsScope | str) -> None:
        """Set which layers fold their batches into the running stats.

        - `"none"` collects nothing but keeps every already-collected bucket
          frozen — non-publishing batches also skip the capture-mode hook
          install they'd otherwise pay just for stats, and switching back to a
          collecting scope resumes adding to the existing buckets.
        - `"watched"` (the default) collects for the watched layers only.
          Entering it drops the buckets of layers outside the watched set —
          the same semantics as unwatching them.
        - `"all"` collects for every layer regardless of the watched set; the
          per-channel caps (`set_watch_performance`) bound the memory.

        Raises `ValueError` for an unknown scope. A locked session pins the
        scope to `ALL` and ignores this call.
        """
        scope = StatsScope(scope)
        if self._locked:
            return
        with self._cv:
            self._stats_scope = scope
            if scope is not StatsScope.NONE:
                self._prev_stats_scope = scope
            watched = list(self._watched_layers)
        if scope is StatsScope.WATCHED:
            # Buckets outside the watched set (collected under a wider scope)
            # are dropped, mirroring `unwatch`; the training thread's own
            # `retain_layers` pass covers an update racing this call.
            self._watch_accumulator.retain_layers(watched)
            self._instruments.retain_layers(watched)

    def toggle_stats_collecting(self) -> bool:
        """Flip between `NONE` and the last collecting scope; True = collecting.

        The top bar's stats toggle: pausing keeps every collected bucket (and
        the shown cards) intact; resuming restores the previous scope
        (`WATCHED` or `ALL`) and continues adding to the existing buckets.
        """
        with self._cv:
            if self._locked:
                return self._stats_scope is not StatsScope.NONE
            if self._stats_scope is StatsScope.NONE:
                self._stats_scope = self._prev_stats_scope
            else:
                self._prev_stats_scope = self._stats_scope
                self._stats_scope = StatsScope.NONE
            return self._stats_scope is not StatsScope.NONE

    def _stats_collection_layers_locked(self) -> list[str]:
        """Layers whose batches currently fold into the stats (`_cv` held)."""
        match self._stats_scope:
            case StatsScope.NONE:
                return []
            case StatsScope.WATCHED:
                return list(self._watched_layers)
            case StatsScope.ALL:
                return list(self._layer_names)

    def _stats_active(self) -> bool:
        """Whether batches currently fold stats (some layer is collecting)."""
        with self._cv:
            match self._stats_scope:
                case StatsScope.NONE:
                    return False
                case StatsScope.WATCHED:
                    return bool(self._watched_layers)
                case StatsScope.ALL:
                    return bool(self._layer_names)

    @property
    def stats_layers(self) -> frozenset[str]:
        """Layers with running stats available or being collected.

        The `/stats` page's selectable universe: the current scope's
        collecting layers plus any layer whose buckets are still retained —
        under scope `NONE` collection stops but nothing is dropped, so
        previously collected layers stay browsable while paused.
        """
        with self._cv:
            layers = set(self._stats_collection_layers_locked())
        return frozenset(layers | self._watch_accumulator.layers_with_stats())

    def stats_phases(self, layer: str | None = None) -> frozenset[str]:
        """Phases with retained running stats — for `layer`, or any layer.

        Backs the `/stats` page's opening Phase selection: the phase training
        is currently in is only a useful default while some retained bucket
        can actually render it.
        """
        return frozenset(self._watch_accumulator.phases_with_stats(layer))

    def watch_snapshot(
        self,
        *,
        layers: Iterable[str] | None = None,
        include_patches: bool = True,
    ) -> WatchSnapshot:
        """Snapshot of the retained running stats (all buckets by default).

        The accumulator holds buckets for exactly the layers the stats scope
        retains — the watched set in `WATCHED` scope, every layer in `ALL`,
        frozen as-is in `NONE` — so the default covers everything browsable;
        pass `layers` to restrict the (GPU→CPU) copy to a subset.
        `include_patches=False` skips the extreme-patch copies for callers
        that only need the scalar/histogram stats.

        In a distributed run the leader overlays the last cross-rank
        reduction on its local view: histogram/scalar stats become global
        (refreshed at every snapshot publish), while patches — and any
        bucket the reduction hasn't covered yet — stay rank-local.
        """
        snap = self._watch_accumulator.snapshot(
            layers=layers, include_patches=include_patches
        )
        reduced = self._dist_watch_stats
        if reduced is None:
            return snap
        stats = {
            key: (
                # The reduced stats don't carry the source dtype; keep the
                # local accumulator's so the histogram's under/overflow band
                # still shows on the leader.
                replace(
                    layer_snap,
                    activations=replace(
                        r[0], dtype=layer_snap.activations.dtype
                    ),
                    gradients=replace(r[1], dtype=layer_snap.gradients.dtype),
                )
                if (r := reduced.get(key)) is not None
                else layer_snap
            )
            for key, layer_snap in snap.stats.items()
        }
        # Weight samples need no reduction — DDP replicas hold identical
        # weights, so the leader's local samples are already global.
        return WatchSnapshot(stats=stats, weights=snap.weights)

    def current_batch_stats(
        self, *, layers: Iterable[str], include_patches: bool = True
    ) -> WatchSnapshot:
        """Stats computed directly from the last published batch snapshot.

        Backs the `/stats` page's "Current batch" view. Unlike
        `watch_snapshot` — running aggregates over watched layers only — this
        reads the published `BatchSnapshot`, so it covers *any* requested
        layer (watched or not) for the one most recently captured batch. The
        result is keyed by the snapshot's own `(phase, epoch)` and shaped like
        a `WatchSnapshot` so the UI renders it through the same path. Returns
        an empty snapshot before any batch has been captured.
        """
        snap = self._snapshot
        if snap is None:
            return WatchSnapshot()
        perf = self._watch_performance
        channel_limit = perf.channel_limit if perf.channel_limit_enabled else None
        patch_source = (
            self._image_like_input(snap.activations) if include_patches else None
        )
        pos = snap.position
        out: dict[tuple[str, str, int], LayerStatsSnapshot] = {}
        for layer in layers:
            activation = snap.activations.get(layer)
            gradient = snap.activation_gradients.get(layer)
            if activation is None and gradient is None:
                continue
            out[(layer, pos.phase, pos.epoch)] = single_batch_stats(
                layer=layer,
                phase=pos.phase,
                epoch=pos.epoch,
                activation=activation,
                gradient=gradient,
                patch_source=patch_source,
                channel_limit=channel_limit,
                samples_per_channel=perf.samples_per_channel,
                average_patches=perf.average_patches,
                include_patches=include_patches,
            )
        return WatchSnapshot(stats=out)

    @property
    def probe_result(self) -> ProbeResult | None:
        """The last probe run's outputs, or `None` when nothing is pinned.

        Same lock-free read contract as `snapshot`: an atomic reference to a
        frozen dataclass of CPU tensors, safe to hold from any thread.
        """
        return self._probe_result

    @property
    def probe_error(self) -> str | None:
        """Why the last probe run failed, or `None` when it succeeded."""
        return self._probe_error

    @property
    def probe_count(self) -> int:
        """Monotonic count of completed probe runs (including failed ones)."""
        with self._cv:
            return self._probe_count

    @property
    def probe_mode(self) -> str:
        """Train/eval handling for probe forwards: "unchanged", "eval", or "train"."""
        with self._cv:
            return self._probe_mode

    @property
    def is_pinned(self) -> bool:
        with self._cv:
            return self._pinned_inputs is not None

    @property
    def pinned_position(self) -> BatchPosition | None:
        """Where the pinned input was captured, or `None` when not pinned."""
        return self._pinned_position

    def pin_current_batch(self) -> bool:
        """Pin the last captured batch's input as the probe input.

        Returns `False` when there is nothing to pin (disabled session, no
        snapshot yet, or a snapshot without an input tensor). While pinned,
        every capture re-runs the model on this input (a "probe") and
        publishes a `ProbeResult` alongside the snapshot, so the UI can show
        the network's evolving response to one fixed batch across stepping
        and time travel. Pinning while paused runs the probe immediately on
        the paused training thread (see `_wait_for_proceed`).

        Refused (returning `False`) on a locked session — the pin is shared
        state every visitor sees. Pin *before* locking to give a demo a
        fixed probe input.
        """
        if self._locked:
            return False
        return probe.pin_current_batch(self)

    def unpin_batch(self) -> None:
        """Drop the pinned input (and the probe result, absent perturbations).

        No-op on a locked session, so a pin made before locking sticks.
        """
        if self._locked:
            return
        probe.unpin_batch(self)

    @property
    def perturbations(self) -> PerturbationMap:
        """Copy of the active perturbations: (input, sample, index) -> values."""
        with self._cv:
            return dict(self._perturbations)

    def add_perturbation(
        self,
        *,
        input_name: str,
        sample: int,
        index: tuple[int, ...],
        values: tuple[float, ...],
    ) -> None:
        """Pin `index` of `sample` in input `input_name` to `values` on probes.

        `index` is `(y, x)` for an image input (`values` is its length-`C`
        channel vector) or `(channel,)` for a flat input (`values` is a single
        scalar). `values` are in the model's input space (already normalized by
        the caller — the UI back-transforms the picked display color with the
        `input_mean` / `input_std` it was given). Perturbations apply to the
        probe's base inputs — the pinned batch, or the current snapshot's
        inputs when nothing is pinned — and trigger a probe re-run that also
        captures the perturbed activations. Entries that don't fit the base
        (out of range, wrong count, absent input) are skipped at apply time.
        No-op on a locked session — perturbations are shared state.
        """
        if self._locked:
            return
        probe.add_perturbation(
            self, input_name=input_name, sample=sample, index=index, values=values
        )

    def clear_perturbations(self) -> None:
        """Drop all perturbations (and the probe result, when not pinned).

        No-op on a locked session.
        """
        if self._locked:
            return
        probe.clear_perturbations(self)

    def set_probe_mode(self, mode: str) -> None:
        """Set train/eval handling for probe forwards.

        - `"unchanged"` (default): modules run with whatever `training`
          flags the training loop left on them.
        - `"eval"`: the whole model is switched to eval — BatchNorm uses
          running stats, dropout is off — and restored afterwards.
        - `"train"`: the whole model is switched to train and restored.

        Regardless of mode, probes never mutate training state: per-module
        flags and all buffers are restored after the run, and the RNG is
        forked around it. Selecting `"eval"` or `"train"` itself activates
        probing — the model is re-run on the current snapshot's batch under
        that mode, no pin required — and switching back to `"unchanged"`
        (with nothing pinned or perturbed) drops the result. No-op on a
        locked session — the probe mode is shared state.
        """
        if self._locked:
            return
        probe.set_probe_mode(self, mode)

    def wait_for_probe(
        self, *, after_count: int = 0, timeout: float | None = None
    ) -> bool:
        """Block until more than `after_count` probe runs have completed.

        The probe-side counterpart of `wait_until_paused`, used by tests and
        the UI to synchronize with asynchronously-requested probe runs
        without polling. Counts failed runs too — check `probe_error`.
        """
        with self._cv:
            return self._cv.wait_for(
                lambda: self._probe_count > after_count or self._closed,
                timeout=timeout,
            )

    @property
    def experiment_result(self) -> ExperimentResult | None:
        """The latest experiment progress/outcome (lock-free read, like
        `snapshot`): a frozen dataclass of CPU tensors, or `None` before the
        first run."""
        return self._experiment_result

    def experiment_result_for(self, seq: int) -> ExperimentResult | None:
        """The latest progress/outcome of one request, by its seq.

        Each client (browser tab) polls its own request this way, so
        concurrent experiments don't overwrite each other's view. Returns
        `None` before the request's first publish — and again once the
        result has been evicted (only the `_EXPERIMENT_RESULTS_KEPT` most
        recently updated seqs are retained).
        """
        with self._cv:
            return self._experiment_results.get(seq)

    @property
    def experiment_pending(self) -> bool:
        """Whether any request is queued but not yet picked up."""
        with self._cv:
            return bool(self._experiment_queue)

    def request_experiment(
        self, *, kind: str, layer: str, params: dict[str, object]
    ) -> int:
        """Queue an experiment for the paused training thread to run.

        Like probe requests, experiments execute only on the training thread
        — immediately when paused, otherwise at the next pause. Requests
        from concurrent clients queue up and run in order; none of them
        supersedes a running one (use `cancel_experiment(seq)` to replace
        your own). Returns the request's seq.
        """
        return experiments.request_experiment(
            self, kind=kind, layer=layer, params=params
        )

    def cancel_experiment(self, seq: int | None = None) -> None:
        """Cancel one request by seq, or every request when `seq` is None.

        A queued request is dropped; a running one notices the cancel flag
        at its next abort check and stops. Other clients' requests are
        untouched when a seq is given.
        """
        experiments.cancel_experiment(self, seq)

    def wait_for_experiment(self, *, timeout: float | None = None) -> bool:
        """Block until the latest request publishes its final result.

        The experiment counterpart of `wait_until_paused` / `wait_for_probe`,
        used by tests to synchronize without polling.
        """
        with self._cv:
            return self._cv.wait_for(
                lambda: (
                    self._experiment_result is not None
                    and self._experiment_result.done
                    and self._experiment_result.seq == self._experiment_seq
                )
                or self._closed,
                timeout=timeout,
            )

    @property
    def update_frequency(self) -> UpdateFrequency:
        """The current visualization update frequency setting."""
        with self._cv:
            return self._update_frequency

    def set_update_frequency(
        self, *, unit: str, n: int = 1, phase: str | None = None
    ) -> None:
        """Set how often visualizations refresh while training runs.

        `unit="epoch"` updates on the first batch of every `n`-th epoch (the
        default, with `n=1`); `unit="batch"` updates every `n`-th batch,
        counting only `phase`'s batches when one is given. Raises
        `ValueError` for an unknown unit/phase, `n < 1`, or a phase
        combined with the epoch unit. Changing the setting restarts the
        batch counter. No-op on a locked session.
        """
        if self._locked:
            return
        if unit not in FREQUENCY_UNITS:
            raise ValueError(
                f"unknown frequency unit {unit!r}; expected one of "
                f"{FREQUENCY_UNITS}"
            )
        if n < 1:
            raise ValueError(f"n must be at least 1, got {n}")
        if phase is not None:
            if unit != "batch":
                raise ValueError("a phase filter only applies to unit='batch'")
            if phase not in self._schedule.phase_order:
                raise ValueError(
                    f"unknown phase {phase!r}; seen so far: "
                    f"{self._schedule.phase_order}"
                )
        with self._cv:
            self._update_frequency = UpdateFrequency(unit=unit, n=n, phase=phase)
            self._freq_counter = 0
            self._freq_epoch = None

    @property
    def auto_run_experiments(self) -> bool:
        """Whether experiment pages re-run automatically on every change.

        A session-wide preference shared across browser tabs (toggled from
        the settings dialog). When set, an experiment page runs on init and
        on any parameter change without a manual Run; when clear, the page
        still runs once on init but re-runs only on a manual Run. Default
        `True`.
        """
        with self._cv:
            return self._auto_run_experiments

    def set_auto_run_experiments(self, enabled: bool) -> None:
        """Set the shared auto-run-experiments preference (see the getter).

        No-op on a locked session.
        """
        if self._locked:
            return
        with self._cv:
            self._auto_run_experiments = bool(enabled)
            self._cv.notify_all()

    @property
    def experiment_defaults(self) -> dict[str, object]:
        """Per-key overrides for the experiment form's default values.

        Keys are experiment parameter names (e.g. ``steps``, ``channels``);
        an experiment page seeds its form with these instead of the built-in
        defaults, and the user can still change them freely (up to any locked
        ceiling). Empty unless `set_experiment_defaults` was called.
        """
        with self._cv:
            return dict(self._experiment_defaults)

    def set_experiment_defaults(self, **defaults: object) -> None:
        """Override the experiment form's default parameter values.

        Only seeds what a fresh experiment page shows — it neither clamps
        requests (a locked session's ceilings do that) nor touches pages that
        are already open. Arm before `lock`: like the other global settings,
        this is a no-op on a locked session.
        """
        if self._locked:
            return
        with self._cv:
            self._experiment_defaults.update(defaults)
            self._cv.notify_all()

    @property
    def debug_settings(self) -> DebugSettings:
        """Current numerical-error debugger configuration (see `debugger`)."""
        with self._cv:
            return self._debug_settings

    def set_debug_settings(
        self,
        *,
        enabled: bool | None = None,
        interval: int | None = None,
        check_nan_inf: bool | None = None,
        check_under_over: bool | None = None,
        threshold: float | None = None,
    ) -> None:
        """Update the debugger settings (only the given fields change).

        Resets the per-batch check counter so a changed interval takes effect
        from the next batch rather than carrying a stale count. No-op on a
        locked session.
        """
        if self._locked:
            return
        with self._cv:
            current = self._debug_settings
            self._debug_settings = replace(
                current,
                enabled=current.enabled if enabled is None else bool(enabled),
                interval=(
                    current.interval if interval is None else max(1, int(interval))
                ),
                check_nan_inf=(
                    current.check_nan_inf
                    if check_nan_inf is None
                    else bool(check_nan_inf)
                ),
                check_under_over=(
                    current.check_under_over
                    if check_under_over is None
                    else bool(check_under_over)
                ),
                threshold=(
                    current.threshold if threshold is None else float(threshold)
                ),
            )
            self._debug_counter = 0
            self._cv.notify_all()

    @property
    def watch_performance(self) -> WatchPerformance:
        """Current per-channel watch caps (see `WatchPerformance`)."""
        with self._cv:
            return self._watch_performance

    def set_patch_layers(self, layers: Iterable[str] | None) -> None:
        """Restrict extreme-patch collection to `layers` (`None` = every layer).

        Histogram/min-max/graph statistics are unaffected — this only gates
        the per-channel extreme-input patch buffers, by far the largest watch
        state (input crops, whole-image samples, and activation heatmaps per
        channel per layer). A scope-`all` run on a deep model at real image
        sizes spends gigabytes there; restricting patches to the layers a
        demo actually showcases keeps memory (and a frozen moment's file
        size) proportional to the shortlist while every layer keeps its
        statistics views.

        Unknown layer names raise `ValueError` on an enabled session (they
        would silently collect nothing). No-op when locked; layers already
        holding patch buffers keep them (this gates new accumulation only).
        """
        if self._locked:
            return
        if layers is None:
            self._patch_layers = None
            return
        requested = frozenset(str(name) for name in layers)
        if self._enabled:
            unknown = requested - set(self._layer_names)
            if unknown:
                raise ValueError(
                    f"unknown layer names for patches: {sorted(unknown)}"
                )
        self._patch_layers = requested

    def set_watch_performance(
        self,
        *,
        channel_limit_enabled: bool | None = None,
        channel_limit: int | None = None,
        samples_per_channel: int | None = None,
        average_patches: bool | None = None,
    ) -> bool:
        """Update the per-channel watch caps (only the given fields change).

        Returns whether the change flushed the watch statistics: the channel
        limit, samples-per-channel, and average-patches toggle fix the
        per-channel buffer shapes, so any change to them drops every bucket
        and rebuilds it under the new config (the UI warns about this).
        `channel_limit` must be ≥ 1 and `samples_per_channel` ≥ 1. No-op
        (returning False) on a locked session.
        """
        if self._locked:
            return False
        with self._cv:
            current = self._watch_performance
            updated = replace(
                current,
                channel_limit_enabled=(
                    current.channel_limit_enabled
                    if channel_limit_enabled is None
                    else bool(channel_limit_enabled)
                ),
                channel_limit=(
                    current.channel_limit
                    if channel_limit is None
                    else max(1, int(channel_limit))
                ),
                samples_per_channel=(
                    current.samples_per_channel
                    if samples_per_channel is None
                    else max(1, int(samples_per_channel))
                ),
                average_patches=(
                    current.average_patches
                    if average_patches is None
                    else bool(average_patches)
                ),
            )
            self._watch_performance = updated
        # Push to the accumulator outside `_cv` (it takes its own lock); it
        # flushes iff the effective caps changed and reports that back.
        return self._watch_accumulator.configure(
            channel_limit=(
                updated.channel_limit if updated.channel_limit_enabled else None
            ),
            samples_per_channel=updated.samples_per_channel,
            average_patches=updated.average_patches,
        )

    @property
    def debug_error(self) -> DebugError | None:
        """The current detected numerical error, or `None`.

        Read lock-free by the UI banner timer; assignment is a single atomic
        attribute write (the record is a frozen dataclass).
        """
        return self._debug_error

    def disable_debug_check(self, category: str) -> None:
        """Turn off one check category and drop its part of the banner.

        `category` is `"nan_inf"` or `"under_over"`. The matching reasons and
        table columns are removed from any active error; the banner clears
        entirely if nothing remains. No-op on a locked session.
        """
        if self._locked:
            return
        with self._cv:
            if category == debugger.NAN_INF:
                self._debug_settings = replace(
                    self._debug_settings, check_nan_inf=False
                )
            elif category == debugger.UNDER_OVER:
                self._debug_settings = replace(
                    self._debug_settings, check_under_over=False
                )
            else:
                return
            error = self._debug_error
        if error is not None:
            self._debug_error = debugger.without_category(error, category)

    @property
    def recording(self) -> RecordingManager:
        """The session's per-view video recording manager (lazily created)."""
        if self._recording_manager is None:
            # Imported lazily: nansense.recording pulls in the UI rendering
            # stack, which imports this module at the top level.
            from nansense.recording import RecordingManager

            self._recording_manager = RecordingManager()
        return self._recording_manager

    def register_auto_experiment(
        self, key: str, *, kind: str, layer: str, params: dict[str, object]
    ) -> int:
        """Queue an experiment and re-run it on every visualization update.

        Like `request_experiment`, but the request is also remembered under
        `key` and re-executed (with the *same* seq, so deep dream redraws
        the same seeded noise and `experiment_result_for(seq)` keeps
        returning the freshest rerun) at every frequency update and capture.
        The registration expires a few seconds after the last
        `touch_auto_experiment(key)` heartbeat unless pinned by an active
        recording (`pin_auto_experiment`). Re-registering a key replaces its
        request. Returns the request's seq.
        """
        return experiments.register_auto_experiment(
            self, key, kind=kind, layer=layer, params=params
        )

    def touch_auto_experiment(self, key: str) -> None:
        """Heartbeat: keep `key`'s auto experiment alive (no-op when pinned)."""
        experiments.touch_auto_experiment(self, key)

    def pin_auto_experiment(self, key: str) -> bool:
        """Keep `key`'s auto experiment alive indefinitely (recordings).

        Returns `False` when no such registration exists."""
        return experiments.pin_auto_experiment(self, key)

    def unpin_auto_experiment(self, key: str) -> None:
        """Put `key`'s auto experiment back on the heartbeat clock."""
        experiments.unpin_auto_experiment(self, key)

    def unregister_auto_experiment(self, key: str) -> None:
        """Drop `key`'s auto experiment (already-published results remain)."""
        experiments.unregister_auto_experiment(self, key)

    def current_weights(self) -> dict[str, Tensor]:
        """CPU clones of the model's parameters, read live at call time.

        Unlike `snapshot.weights` (captured at a pause), this reads the
        parameters whenever it's called — so the UI can show current weights
        mid-training, including in `detach` / `step_run` where no snapshot is
        published. The read races with the training thread's in-place updates;
        for a visualisation that's benign — a torn read at worst, never a
        crash. Keys match `named_parameters()`, the same keys
        `layer_weights` indexes into.
        """
        return capture.current_weights(self.model)

    def current_weight_gradients(self) -> dict[str, Tensor]:
        """CPU clones of the parameters' current `.grad`, read live at call time.

        The live counterpart of `BatchSnapshot.weight_gradients`, with the
        same contract: parameters whose gradient is `None` (nothing has run
        backward yet, or `zero_grad(set_to_none=True)` just cleared them) are
        omitted. Same benign-race caveat as `current_weights`.
        """
        return capture.current_weight_gradients(self.model)

    def current_optimizer_state(self) -> dict[str, dict[str, Tensor]]:
        """Per-parameter optimizer state, read live at call time.

        `optimizer.state` is keyed by the parameter object itself, so the
        mapping back to parameter names is an identity lookup — it works for
        any optimizer following the `torch.optim` convention with no
        per-optimizer code (SGD's `momentum_buffer`, Adam's `step` /
        `exp_avg` / `exp_avg_sq`, custom optimizers alike). Tensor entries
        are CPU-cloned; plain int/float entries become 0-dim tensors so the
        value type stays uniform. Empty when no optimizer was passed to
        `start()`, or before the first `optimizer.step()` (state is lazily
        initialised). Same benign-race caveat as `current_weights`.
        """
        return capture.current_optimizer_state(self.model, self._optimizer)

    def current_optimizer_hyperparams(self) -> dict[str, dict[str, float]]:
        """Per-parameter numeric hyperparameters of the optimizer group.

        Maps each parameter name to the plain-numeric knobs of the param
        group it belongs to (`lr`, `momentum`, `weight_decay`, ...). Read
        live, so a scheduler-mutated `lr` shows its current value. Unlike
        `current_optimizer_state` this is populated as soon as the optimizer
        exists — groups are not lazily initialised. Empty when no optimizer
        was passed to `start()`.
        """
        return capture.current_optimizer_hyperparams(self.model, self._optimizer)

    def request_snapshot(self) -> None:
        """Ask the next batch to publish a snapshot (the UI Refresh button).

        A snapshot is what refreshes the views (activations, gradients,
        weights, and the pinned-input probe), but it is only published on
        captures and on the frequency cadence — so in `detach` / `step_run`,
        which run freely between cadence ticks, the views freeze mid-training.
        This arms a one-shot request that makes the next batch publish without
        pausing (like a frequency update) and without recomputing anything:
        the already-running forward/backward is simply captured and sent to the
        views. A no-op when training isn't producing batches (the shown
        snapshot is then already current) and idempotent until consumed.
        """
        with self._cv:
            self._snapshot_request = True

    def set_schedule(
        self,
        *,
        epochs: int | None = None,
        phases: dict[str, int] | None = None,
    ) -> None:
        """Re-declare the schedule mid-run; batch counters are kept.

        `phases` switches the schedule to declared mode and replaces the phase
        set and their batch counts, so it is the way to express a shape that
        varies per epoch — validating every N epochs declares a train-only
        schedule on the epochs that skip it (what the Lightning integration
        does on each `on_train_epoch_start`). A schedule fixed for the whole
        run is better passed to `start(phases=...)` once.

        Either argument may be omitted to leave that part unchanged.
        """
        with self._cv:
            self._schedule.update(epochs=epochs, phases=phases)

    def freeze_moment(
        self, path: Path | str, *, phase: str, epoch: int, batch_idx: int
    ) -> None:
        """Arm a one-shot moment freeze at an exact batch position.

        When training reaches `(phase, epoch, batch_idx)`, that batch installs
        hooks and publishes a snapshot like a capture — whatever the mode, so
        a detached prepare run works — and, after folding its watch stats,
        writes the complete debugger moment to `path` without pausing: the
        batch's loader item (the replay seed the snapshot is regenerated
        from — drive the loop with `batches()`, or pass `item=` to a manual
        `batch()`), running statistics, watched set, model weights and
        buffers, and the schedule shape. `nansense.load_moment` later rebuilds
        the view around a fresh model of the same architecture, the backing
        for a locked showcase (see `nansense.moments` and
        `examples/playground`).

        One request at a time — arming again replaces an unconsumed target; a
        target the run never reaches is reported at `close()`. No-op on a
        disabled or locked session; leader-only under DDP (the frozen
        statistics are the leader's shard).
        """
        if not self._enabled or self._locked:
            return
        with self._cv:
            self._freeze_request = (Path(path), phase, int(epoch), int(batch_idx))

    @property
    def locked(self) -> bool:
        """Whether run control and global settings are locked (see `lock`)."""
        return self._locked

    def lock(self) -> None:
        """Lock run control and global settings for a shared demo deployment.

        Intended for a publicly hosted playground where many anonymous
        visitors share one session. Once locked:

        - the run-control methods (`stop`, every `step_*`, `detach`),
          `request_time_travel`, `watch`/`unwatch`, the probe surface
          (`pin_current_batch`, `unpin_batch`, `add_perturbation`,
          `clear_perturbations`, `set_probe_mode` — all shared state every
          visitor sees), and the global settings (`set_stats_scope`,
          `set_update_frequency`, `set_watch_performance`,
          `set_debug_settings`, `disable_debug_check`,
          `set_auto_run_experiments`, `set_experiment_defaults`) become
          no-ops (`request_time_travel` raises `TimeTravelError`, so the UI
          can show why);
        - the stats scope is forced to `ALL` — every layer collects, so the
          UI's per-tab show/hide never touches shared state;
        - experiment requests still run, with their parameters clamped and
          the queue depth capped (see `nansense.experiments`);
        - registering instruments (`watch_metric`, `watch_layer_tensor`,
          `watch_weight_tensor`) raises — register them, like every other
          host-script setting, *before* locking.

        Everything read-only stays available, as do experiments. The lock is
        one-way by design — arm the wanted mode (e.g. `step_run()`), the
        settings, and any demo pin *before* locking. `close()` is not
        locked; it belongs to the hosting script.
        """
        with self._cv:
            self._stats_scope = StatsScope.ALL
            self._prev_stats_scope = StatsScope.ALL
            self._locked = True

    def stop(self) -> None:
        self._set_mode(Mode.STEP, resume=False)

    def step_batch(self) -> None:
        self._set_mode(Mode.STEP, resume=True)

    def step_phase(self) -> None:
        self._record_step_origin()
        self._set_mode(Mode.UNTIL_PHASE_CHANGE, resume=True)

    def step_epoch(self) -> None:
        self._record_step_origin()
        self._set_mode(Mode.UNTIL_EPOCH_CHANGE, resume=True)

    def step_run(self) -> None:
        self._set_mode(Mode.UNTIL_END, resume=True)

    def _record_step_origin(self) -> None:
        """Snapshot where the run sits now, so a step-phase/epoch lands on the
        first batch of the *next* phase/epoch (not the rest of the current one)."""
        pos = self._live_position
        with self._cv:
            self._step_origin = None if pos is None else (pos.phase, pos.epoch)

    def step_until_position(
        self, *, phase_index: int, epoch: int, batch_idx: int
    ) -> None:
        """Run until the batch at `(phase_index, epoch, batch_idx)` is reached.

        `phase_index` is the position of the phase in the epoch's phase order
        (0 = first phase). A target in a phase/epoch not yet observed is simply
        matched once training arrives there; one that never arrives runs to the
        end (acceptable — there is nothing to stop on)."""
        if self._locked:
            return
        with self._cv:
            self._target_position = (phase_index, epoch, batch_idx)
        self._set_mode(Mode.UNTIL_POSITION, resume=True)

    def detach(self) -> None:
        self._set_mode(Mode.DETACH, resume=True)

    def park(self) -> None:
        """Hold the calling thread at a pause, serving UI requests, until
        `close()`.

        The showcase counterpart of a training loop: a script that restored a
        frozen moment (`nansense.load_moment`) has no batches to drive, but
        experiments and probes still execute on the pause loop of whatever
        thread owns the model — this provides that loop. Call it from the
        thread that built the model, typically right after `lock()`; on an
        unlocked session a Run/Step click simply re-enters the park. Returns
        once the session is closed.
        """
        if not self._enabled:
            return
        # Parking is an explicit "wait for the UI indefinitely" — suppress the
        # unserved-pause grace, which would otherwise detach out of the wait.
        self.mark_served()
        while not self.closed:
            self._wait_for_proceed()

    def epochs(
        self,
        n: int | None = None,
        *,
        cache_dir: Path = DEFAULT_CACHE_DIR,
        start_epoch: int = 0,
    ) -> Iterator[int]:
        """Time-travel-aware epoch loop: `for epoch in session.epochs(50): ...`.

        `n` is the total number of epochs — the canonical place to declare it
        (it sets the schedule's epoch count); pass it here rather than to
        `nansense.start`. If omitted, the count set on `start(epochs=…)` is
        used, and it is an error if neither was given.

        Yields `0 … n - 1` like `range`, but opts the session into time travel:
        each epoch start is checkpointed to `cache_dir` and a UI-requested jump
        re-enters the loop at the chosen epoch with the model / optimizer /
        scheduler / RNG restored. Wrap each iteration's body — every phase of
        the epoch — in `with session.restore_point():`, the block that catches
        the jump; the generator then re-yields the target epoch::

            for epoch in session.epochs(50, cache_dir="models/latest"):
                with session.restore_point():
                    for inputs, targets in session.batches(train_dl, phase="train"):
                        ...
                    for inputs, targets in session.batches(val_dl, phase="val"):
                        ...

        `start_epoch` resumes the run from `cache_dir`'s checkpoint of that
        epoch instead of training from scratch: the cache directory is
        adopted as-is (it may come from an earlier process — e.g. one baked
        into a deployment image), the checkpoint is validated against the
        live model / optimizer / scheduler up-front (`TimeTravelError` on a
        mismatch), and the loop yields `start_epoch … n - 1` with the state
        restored on the training thread before the first batch. Time travel
        then covers every epoch the directory holds a checkpoint for.

        Not iterating this leaves the run a straight pass with the UI's Time
        Travel button disabled; on a disabled session it is inert and nothing
        touches the disk (`start_epoch` is then ignored). Under DDP call it
        on every rank — a leader jump is broadcast so all ranks re-yield the
        same epoch in lockstep.
        """
        if n is not None:
            self._schedule.set_epochs(n)
        elif self._schedule.epochs is None:
            raise ValueError(
                "epoch count is unknown — pass it to session.epochs(n) "
                "(or, as a fallback, nansense.start(epochs=n))"
            )
        if self._loop_restorer is None:
            self._loop_restorer = self.training_restorer(cache_dir=cache_dir)
        if start_epoch and self._enabled:
            self._loop_restorer.resume_from(start_epoch)
        return self._loop_restorer.iter_epochs()

    def restore_point(self) -> contextlib.AbstractContextManager[TrainingRestorer]:
        """Per-epoch restore boundary for the `session.epochs()` loop.

        Enter it around each epoch's body (both train and val phases): on a
        time-travel jump the `TimeTravelJump` unwinds to here, is suppressed,
        and `session.epochs()` re-yields the target epoch with state restored.
        Loop state that depends on history (a running `best_acc`, metric
        curves) belongs inside it, so a jump rewinds it naturally. Must be
        used inside a `for epoch in session.epochs():` loop.
        """
        if self._loop_restorer is None:
            raise RuntimeError(
                "session.restore_point() must be used inside a "
                "`for epoch in session.epochs():` loop"
            )
        return self._loop_restorer.epoch_guard()

    def training_restorer(
        self, *, cache_dir: Path = DEFAULT_CACHE_DIR
    ) -> TrainingRestorer:
        """Create the restorer that opts this session into time travel.

        Wrapping the epoch loop in the returned object (see
        `TrainingRestorer`) enables both epoch-start checkpointing to
        `cache_dir` and UI-driven jumps back to any cached epoch. A session
        without a restorer never writes checkpoints and never raises
        `TimeTravelJump`. On a disabled session the restorer is inert: the
        loop runs exactly once and nothing touches the disk.

        Under DDP every rank wraps its epoch loop in a restorer the same way:
        a UI-requested jump on the leader is broadcast to all ranks at the
        next batch-start barrier, where every rank raises `TimeTravelJump` in
        lockstep and restores from its own per-rank checkpoint (see
        `nansense.distributed`).
        """
        restorer = TrainingRestorer(self, cache_dir=cache_dir)
        self.attach_restorer(restorer)
        return restorer

    def attach_restorer(self, restorer: TrainingRestorer) -> None:
        """Register an externally-created restorer with this session.

        Binds the restorer to the session and enables time travel, exactly
        like `training_restorer()` — but for restorer objects that must be
        created before the session exists (e.g. the Lightning integration's
        `LightningRestorer`, built by `fit_with_time_travel` and attached
        once the first `trainer.fit` constructs the session). On a disabled
        session the restorer is bound but stays inert.
        """
        if self._restorer is not None:
            raise RuntimeError("this session already has a training restorer")
        restorer._session = self
        if self._enabled:
            self._restorer = restorer

    def request_time_travel(self, epoch: int) -> None:
        """Ask the training thread to restart at the start of `epoch`.

        Validates the request up-front on the calling (UI) thread — the
        checkpoint must exist, load, and match the live model's parameter
        shapes *and* the live optimizer's / scheduler's state layout — and
        raises `TimeTravelError` with a displayable message otherwise, so an
        incompatible cache (e.g. written by a previous run with a different
        model, or a different optimizer config) is rejected before anything
        unwinds or any state is mutated. On success the jump is armed: the
        training thread raises `TimeTravelJump` at its next batch boundary
        (immediately, when paused), the restorer rolls the state back, and the
        session enters `STEP` mode so the first batch of `epoch` pauses for
        inspection.

        Under DDP the armed jump is broadcast to every rank at the next
        batch-start barrier (`sync_batch_control`), where all ranks raise
        `TimeTravelJump` together — never mid-batch, so no collective is left
        half-issued. The leader validates only its own (rank-0) checkpoint
        here; the followers' replicated state is restored from their own files.
        """
        if self._locked:
            raise TimeTravelError(
                "Time travel is disabled in this shared demo."
            )
        restorer = self._restorer
        if restorer is None:
            raise TimeTravelError(
                "time travel requires the training loop to be wrapped in a "
                "training restorer (session.training_restorer())"
            )
        if restorer.finished:
            raise TimeTravelError("the training run has already completed")
        total = self._schedule.epochs
        if total is None or not 0 <= epoch < total:
            raise TimeTravelError(f"epoch {epoch} out of range [0, {total or 0})")
        # Memory-mapped: validation only reads keys and shapes, so the full
        # model + optimizer state never materializes in CPU memory on the UI
        # thread (the training thread's `_restore` loads the values for real).
        payload = restorer.cache.load(epoch, mmap=True)
        # Validate every piece of the checkpoint the training thread will load
        # back into live state — model, optimizer, scheduler — here on the UI
        # thread, before the jump is armed. The training thread's `_restore`
        # mutates the model first, so an unvalidated optimizer/scheduler
        # mismatch would otherwise crash mid-restore, after the model was
        # already overwritten and after this method already reported success.
        for error in (
            validate_model_state(payload, self.model),
            validate_optimizer_state(payload, self._optimizer),
            validate_scheduler_state(payload, self._scheduler),
        ):
            if error is not None:
                raise TimeTravelError(error)
        with self._cv:
            self._pending_jump = epoch
            self._mode = Mode.STEP
            self._resume_token += 1
            self._cv.notify_all()

    def time_travel_status(self) -> TimeTravelStatus:
        """What the UI needs to render the time-travel button and dialog."""
        total = self._schedule.epochs
        if self._locked:
            return TimeTravelStatus(
                available=False,
                reason="Time travel is disabled in this shared demo.",
                cached_epochs=[],
                total_epochs=total or 0,
            )
        restorer = self._restorer
        if restorer is None or total is None:
            return TimeTravelStatus(
                available=False,
                reason=(
                    "Time travel is off: the training loop is not driven by "
                    "`for epoch in session.epochs(): with session.restore_point(): "
                    "...`, so no epoch checkpoints are saved."
                ),
                cached_epochs=[],
                total_epochs=total or 0,
            )
        cached = [e for e in restorer.cache.cached_epochs() if 0 <= e < total]
        if restorer.finished:
            return TimeTravelStatus(
                available=False,
                reason="The training run has completed; time travel is no longer available.",
                cached_epochs=cached,
                total_epochs=total,
            )
        return TimeTravelStatus(
            available=True, reason=None, cached_epochs=cached, total_epochs=total
        )

    def close(self) -> None:
        """Mark training finished once the loop completes.

        Releases anything waiting on the session and finalizes in-flight
        recordings so their MP4 files are playable. The served page stays up
        for post-mortem browsing of the last captured state — `close` ends
        the *training* side of the session, not the UI.
        """
        with self._cv:
            self._closed = True
            self._cv.notify_all()
            freeze_request = self._freeze_request
            self._freeze_request = None
        if freeze_request is not None:
            # The run ended without reaching the armed freeze position — say
            # so instead of leaving a prepare script silently moment-less.
            path, phase, epoch, batch_idx = freeze_request
            print(
                f"NaNsense: freeze_moment target (epoch {epoch} | {phase} "
                f"batch {batch_idx}) was never reached; {path} was not "
                "written.",
                flush=True,
            )
        # Finalize any in-flight recordings so their MP4 files are playable
        # even when the training script simply runs to completion.
        if self._recording_manager is not None:
            self._recording_manager.end_all()

    def wait_until_paused(
        self,
        *,
        after_pauses: int = 0,
        timeout: float | None = None,
    ) -> bool:
        with self._cv:
            return self._cv.wait_for(
                lambda: self._pause_count > after_pauses or self._closed,
                timeout=timeout,
            )

    def _set_mode(self, mode: Mode, *, resume: bool) -> None:
        # The single choke point for every run-control method (`stop`,
        # `step_*`, `detach`), so a locked session refuses them all here.
        if self._locked:
            return
        with self._cv:
            self._mode = mode
            if resume:
                # Resuming no longer clears the numerical-error banner: once
                # training has stopped on an issue, proceeding keeps the banner
                # standing and later detections accumulate into it without
                # stopping again (see `_run_debug_checks`). It is cleared only
                # by silencing the checks (`disable_debug_check`) or a
                # time-travel rewind (`_rewind_to_epoch`).
                self._resume_token += 1
                self._cv.notify_all()

    def _should_capture(self, pos: BatchPosition) -> bool:
        with self._cv:
            if self._closed:
                return False
            mode = self._mode
            target = self._target_position
            origin = self._step_origin
        match mode:
            case Mode.STEP:
                return True
            case Mode.UNTIL_PHASE_CHANGE:
                # First batch of the next phase (the position left `origin`),
                # or the run's last batch when that is detectable (so the last
                # phase, which has no successor, still stops instead of running
                # off the end). Falls back to the flag if no origin was recorded.
                if origin is None:
                    return pos.is_last_in_phase
                return (pos.phase, pos.epoch) != origin or pos.is_last_overall
            case Mode.UNTIL_EPOCH_CHANGE:
                # First batch of the next epoch, or the run's last batch when
                # detectable (last epoch has no successor to land on).
                if origin is None:
                    return pos.is_last_in_epoch
                return pos.epoch > origin[1] or pos.is_last_overall
            case Mode.UNTIL_END:
                return pos.is_last_overall
            case Mode.UNTIL_POSITION:
                if target is None:
                    return False
                phases = self._schedule.phase_order
                phase_index = phases.index(pos.phase) if pos.phase in phases else -1
                return (phase_index, pos.epoch, pos.batch_idx) == target
            case Mode.DETACH:
                return False

    def _should_freq_update(self, pos: BatchPosition) -> bool:
        """Whether this batch publishes a non-pausing frequency update.

        Training thread only (it advances `_freq_counter`/`_freq_epoch`).
        Frequency updates fire in every mode — including detach and the
        run-until modes — so the visualizations keep refreshing at the
        configured cadence while training runs freely.
        """
        with self._cv:
            if self._closed:
                return False
            freq = self._update_frequency
            if freq.unit == "epoch":
                # Mirror Step Epoch: detect the epoch boundary from the epoch
                # number advancing, not the `is_last_in_epoch` flag (which
                # needs the batch count and so never fires during the first
                # lazy epoch). Publishes on the first batch of every n-th
                # epoch (0, n, 2n, …); `_freq_epoch` is the last epoch seen.
                is_new_epoch = pos.epoch != self._freq_epoch
                self._freq_epoch = pos.epoch
                return is_new_epoch and pos.epoch % freq.n == 0
            if freq.phase is not None and pos.phase != freq.phase:
                return False
            # Mutated under the lock so `set_update_frequency`'s reset to 0
            # can't be lost to a racing unlocked read-modify-write here.
            self._freq_counter += 1
            return self._freq_counter % freq.n == 0

    def _should_debug_check(self, pos: BatchPosition) -> bool:
        """Whether this batch runs the numerical-error checks (training thread).

        Independent of capture/frequency: the debugger throttles itself with
        its own `_debug_counter` so checks run every nth batch in *every*
        mode, including detach and the run-until modes. Leader-only in
        distributed runs (followers never publish or pause).
        """
        if not self.is_leader:
            return False
        with self._cv:
            if self._closed:
                return False
            settings = self._debug_settings
            if not settings.any_check():
                return False
            do_check = self._debug_counter % max(1, settings.interval) == 0
            self._debug_counter += 1
            return do_check

    def _run_debug_checks(self, pos: BatchPosition) -> bool:
        """Run the debugger over this batch's live tensors (training thread).

        Called at `__exit__` while the captured activations (and their
        retained `.grad`) are still resident, so it sees the same gradients the
        snapshot will. Returns whether this batch should force a publish+pause.

        The *first* detection of an episode records the error and stops
        training (STEP mode) so the batch pauses for inspection — returning
        `True`. Once a banner is standing, resuming does not clear it (see
        `_set_mode`); further detections merge into it (`debugger.merged`)
        *without* stopping again — returning `False` — so a user who chose to
        proceed past the first issue keeps running while the banner grows.
        """
        with self._cv:
            settings = self._debug_settings
        activations = {
            n: t for n, t in self._activations.items() if isinstance(t, Tensor)
        }
        activation_grads = {
            n: t.grad for n, t in activations.items() if t.grad is not None
        }
        weight_grads = {
            name: p.grad
            for name, p in self.model.named_parameters()
            if p.grad is not None
        }
        error = debugger.run_checks(
            settings,
            position=pos,
            activations=activations,
            activation_grads=activation_grads,
            weight_grads=weight_grads,
            layer_weights=self._layer_weights,
        )
        if error is None:
            return False
        existing = self._debug_error
        if existing is not None:
            # An episode is already on screen — accumulate and keep running.
            self._debug_error = debugger.merged(existing, error)
            return False
        self._debug_error = error
        # Surface the first detection on the console too, so a headless run
        # (no browser) still sees it. Later merges stay quiet — only the
        # episode's onset prints.
        print(
            f"NaNsense: numerical issue detected ({debugger.reasons_text(error)}) "
            f"at {format_position(error.position)} — training paused. See the "
            "UI banner for affected layers and fixes (e.g. loss scaling or "
            "bfloat16 for fp16 subnormal gradients).",
            flush=True,
        )
        # Re-check the very next batch so a Step immediately re-evaluates,
        # rather than waiting out the rest of the interval.
        self._debug_counter = 0
        self.stop()
        return True

    def _record_frames(self) -> None:
        """Append one frame to every active recording (training thread).

        Called at frequency updates only — recordings advance at the
        configured show frequency, not on every user step. The manager
        guards per-recorder failures internally, so a broken view never
        kills the training thread.
        """
        manager = self._recording_manager
        if manager is not None:
            manager.capture_frames(self)

    def _update_watch_stats(self, pos: BatchPosition) -> None:
        # Iterate a snapshot of the stats-eligible set taken under the lock —
        # the watched set in WATCHED scope, every layer in ALL. The UI thread
        # mutates `_watched_layers` via watch()/unwatch() (and the "Show all"
        # / "Hide all" loops), so iterating the live set here on the training
        # thread would race into a "set changed size during iteration"
        # RuntimeError.
        with self._cv:
            watched = self._stats_collection_layers_locked()
        # Reap buckets for layers no longer in scope before updating.
        # `unwatch` (and a scope narrowed to WATCHED) forgets on the UI
        # thread, which can race this batch and let an `update` below
        # resurrect a just-forgotten bucket; doing it here, on the only
        # `update` caller, makes that leak impossible.
        self._watch_accumulator.retain_layers(watched)
        self._instruments.retain_layers(watched)
        # Custom scalar metrics run here too — same collection set, same
        # cadence as the built-in accumulators. Leader-only under DDP (they
        # stay rank-local, like the patch buffers), and their live
        # parameter/optimizer views are built at most once per batch.
        run_metrics = self.is_leader and self._instruments.has_metrics()
        instrument_sources = self._instrument_sources() if run_metrics else None
        # Extreme-input patches stay rank-local in distributed runs: only
        # the leader renders them, so followers skip the buffers entirely.
        source = self._patch_source_input() if self.is_leader else None
        patch_layers = self._patch_layers
        # Live parameter tensors, resolved at most once per batch — and only
        # when some watched layer still needs this epoch's weight sample
        # (every batch after the epoch's first is just the cheap
        # `weights_pending` check).
        live_params: dict[str, Tensor] | None = None
        for name in watched:
            # Weight-tensor stats for the GRAPHS view: sampled at the
            # epoch's first watched batch. Leader-only, like the patches —
            # DDP replicas hold identical weights, so one rank's sample is
            # already global.
            param_names = self._layer_weights.get(name)
            if (
                self.is_leader
                and param_names
                and self._watch_accumulator.weights_pending(name, pos.epoch)
            ):
                if live_params is None:
                    live_params = dict(self.model.named_parameters())
                self._watch_accumulator.update_weights(
                    layer=name,
                    epoch=pos.epoch,
                    params=[
                        (p, live_params[p])
                        for p in param_names
                        if p in live_params
                    ],
                )
            tensor = self._activations.get(name)
            if tensor is None:
                continue
            self._watch_accumulator.update(
                layer=name,
                phase=pos.phase,
                epoch=pos.epoch,
                kind="activation",
                x=tensor,
            )
            grad = tensor.grad
            if grad is not None:
                self._watch_accumulator.update(
                    layer=name,
                    phase=pos.phase,
                    epoch=pos.epoch,
                    kind="gradient",
                    x=grad,
                )
            if source is not None and (patch_layers is None or name in patch_layers):
                self._watch_accumulator.update_patches(
                    layer=name,
                    phase=pos.phase,
                    epoch=pos.epoch,
                    act=tensor,
                    x=source,
                )
            if instrument_sources is not None:
                self._instruments.run_metrics(
                    self._layer_context(name, pos, tensor, instrument_sources)
                )

    def _instrument_sources(self) -> _InstrumentSources:
        """Build the per-batch live tensor views instrument contexts slice.

        Unlike the `current_*` snapshot readers these never copy — instrument
        callbacks read the live device tensors. Training thread, at most once
        per batch (and only while some instrument is registered).
        """
        params: dict[str, Tensor] = dict(self.model.named_parameters())
        state: dict[str, dict[str, Tensor | float]] = {}
        hyperparams: dict[str, dict[str, float]] = {}
        if self._optimizer is not None:
            names = {id(p): n for n, p in params.items()}
            for param, entries in self._optimizer.state.items():
                name = names.get(id(param))
                if name is not None:
                    state[name] = dict(entries)
            hyperparams = capture.current_optimizer_hyperparams(
                self.model, self._optimizer
            )
        return params, state, hyperparams

    def _layer_module(self, name: str) -> nn.Module | None:
        """The `nn.Module` behind a layer name; `None` for fx intermediates
        (`relu`, `add`, ...) and graph inputs, which no module owns."""
        try:
            return self.model.get_submodule(name)
        except AttributeError:
            return None

    def _layer_context(
        self,
        name: str,
        pos: BatchPosition,
        activation: Tensor,
        sources: _InstrumentSources,
    ) -> LayerContext:
        """Assemble one layer's live `LayerContext` (training thread)."""
        params, state, _ = sources
        param_names = self._layer_weights.get(name, [])
        weights = {p: params[p] for p in param_names if p in params}
        return LayerContext(
            layer=name,
            phase=pos.phase,
            epoch=pos.epoch,
            batch_idx=pos.batch_idx,
            module=self._layer_module(name),
            activation=activation,
            gradient=activation.grad,
            weights=weights,
            weight_gradients={
                p: t.grad for p, t in weights.items() if t.grad is not None
            },
            optimizer_state={p: state[p] for p in param_names if p in state},
        )

    def _run_tensor_instruments(
        self, pos: BatchPosition
    ) -> tuple[dict[str, dict[str, Tensor]], dict[str, dict[str, Tensor]]]:
        """Evaluate the tensor instruments for this publish (training thread).

        Covers the stats scope's collection set, like the scalar metrics —
        "only watched layers are logged" under the default scope. Publish
        batches only: the results are pure snapshot cargo (rendered strips),
        so computing them on a non-publishing batch would be wasted work.
        Returns the two `BatchSnapshot` dicts (empty without instruments).
        """
        run_layer = self._instruments.has_layer_tensors()
        run_weight = self._instruments.has_weight_tensors()
        if not (run_layer or run_weight):
            return {}, {}
        with self._cv:
            layers = self._stats_collection_layers_locked()
        if not layers:
            return {}, {}
        sources = self._instrument_sources()
        params, state, hyperparams = sources
        acts: dict[str, dict[str, Tensor]] = {}
        weights: dict[str, dict[str, Tensor]] = {}
        for name in layers:
            activation = self._activations.get(name)
            if run_layer and activation is not None:
                out = self._instruments.run_layer_tensors(
                    self._layer_context(name, pos, activation, sources)
                )
                if out:
                    acts[name] = out
            if not run_weight:
                continue
            for param_name in self._layer_weights.get(name, []):
                tensor = params.get(param_name)
                if tensor is None:
                    continue
                out = self._instruments.run_weight_tensors(
                    WeightContext(
                        layer=name,
                        param=param_name,
                        phase=pos.phase,
                        epoch=pos.epoch,
                        batch_idx=pos.batch_idx,
                        module=self._layer_module(name),
                        weight=tensor,
                        gradient=tensor.grad,
                        optimizer_state=state.get(param_name, {}),
                        hyperparams=hyperparams.get(param_name, {}),
                    )
                )
                if out:
                    weights[param_name] = out
        return acts, weights

    def _patch_source_input(self) -> Tensor | None:
        """The live forward input to crop extreme-activation patches from."""
        return self._image_like_input(self._activations)

    def _image_like_input(
        self, activations: Mapping[str, Tensor]
    ) -> Tensor | None:
        """The input to crop extreme-activation patches from, from `activations`.

        Prefers the first image-like input (4D with 1 or 3 channels); falls
        back to any 4D input so `PatchAccumulator.update` can apply its own
        guards. `None` when the model takes no 4D input. Shared by the live
        watch path (`_activations`) and the "Current batch" view (a snapshot's
        activation dict).
        """
        fallback: Tensor | None = None
        for name in self._input_names:
            t = activations.get(name)
            if not isinstance(t, Tensor) or t.ndim != 4:
                continue
            if t.shape[1] in (1, 3):
                return t
            if fallback is None:
                fallback = t
        return fallback

    def _publish_snapshot(self, pos: BatchPosition) -> None:
        # Release the previous snapshot's CPU clones *before* allocating the
        # new ones: a snapshot clones every activation, gradient, weight and
        # optimizer-state tensor, so holding the old one while the new one is
        # built would stack two full snapshots in CPU memory. Dropping the
        # session's reference first lets the allocator reuse those pages, so
        # the publish peak is one snapshot, not two. A reader that already
        # grabbed the old snapshot keeps it alive (no torn read); one that
        # reads `snapshot` during the build sees `None` and skips a render tick
        # (see the `snapshot` property). Both writes are atomic under the GIL.
        self._snapshot = None
        # Tensor instruments read the live activations/weights, so they run
        # here — after the old snapshot is released, before the new clones.
        custom_acts, custom_weights = self._run_tensor_instruments(pos)
        self._snapshot = BatchSnapshot(
            position=pos,
            activations={
                n: capture.cpu_clone(a) for n, a in self._activations.items()
            },
            activation_gradients={
                n: capture.cpu_clone(a.grad)
                for n, a in self._activations.items()
                if a.grad is not None
            },
            weights=capture.current_weights(self.model),
            weight_gradients=capture.current_weight_gradients(self.model),
            # Runs on the training thread at __exit__, so these reads are
            # consistent with the weights above ({} when no optimizer given).
            optimizer_state=self.current_optimizer_state(),
            optimizer_hyperparams=self.current_optimizer_hyperparams(),
            custom_activations=custom_acts,
            custom_weight_tensors=custom_weights,
        )

    def _take_pending_jump(self) -> int | None:
        """Consume the armed time-travel target, if any (training thread)."""
        # Lock-free fast path: this runs on every batch boundary, and the
        # GIL-atomic read is None for the entire life of most sessions. A
        # request racing the read is simply consumed at the next boundary.
        if self._pending_jump is None:
            return None
        with self._cv:
            jump = self._pending_jump
            self._pending_jump = None
            return jump

    def _take_snapshot_request(self) -> bool:
        """Consume the one-shot Refresh request, if armed (training thread)."""
        # Same lock-free fast path as `_take_pending_jump`: read False on every
        # batch in the common case, only locking to clear an actual request.
        if not self._snapshot_request:
            return False
        with self._cv:
            self._snapshot_request = False
            return True

    def _take_freeze_request(self, pos: BatchPosition) -> Path | None:
        """Consume the armed moment freeze if `pos` is its exact target.

        Training thread, at batch start. Same lock-free fast path as
        `_take_pending_jump`: the reference read is None for the entire life
        of most sessions, and only a position match takes the lock.
        """
        request = self._freeze_request
        if request is None:
            return None
        path, phase, epoch, batch_idx = request
        if (pos.phase, pos.epoch, pos.batch_idx) != (phase, epoch, batch_idx):
            return None
        with self._cv:
            self._freeze_request = None
        return path

    def _peek_pending_jump(self) -> int:
        """The armed time-travel target without consuming it, -1 when none.

        Used by the distributed path: the leader broadcasts this value to all
        ranks at the batch-start barrier, and only *then* — once every rank
        has agreed to jump — does it consume the request via `_take_pending_jump`
        and raise. Peeking (rather than consuming) here keeps the leader's
        consume atomic with the lockstep raise, so a follower can never be told
        to jump while the leader's own request has already been cleared.
        """
        # GIL-atomic read; mirrors `_take_pending_jump`'s lock-free fast path.
        jump = self._pending_jump
        return jump if jump is not None else -1

    def _maybe_save_epoch_start(self, epoch: int) -> None:
        """Checkpoint the epoch-start state, once per epoch attempt.

        Called from `batches()` before `iter(loader)` draws the shuffle seed
        (the deterministic-replay anchor) and, as a fallback, from
        `_BatchContext.__enter__` for users who drive `batch()` manually. The
        `_epoch_start_saved_for` guard makes the epoch's *first* phase win — a
        later phase of the same epoch finds it already saved and skips — and
        also lets whichever entry point runs first win, so the fallback's
        post-iter save can't overwrite the good pre-iter RNG. Anchoring on "the
        first call this epoch" (rather than a declared first-phase name) is what
        keeps it correct when the schedule is discovered lazily.
        """
        restorer = self._restorer
        if restorer is not None and self._epoch_start_saved_for != epoch:
            restorer.save_epoch_start(epoch)
            self._epoch_start_saved_for = epoch

    def _rewind_to_epoch(self, epoch: int) -> None:
        """Reset per-epoch bookkeeping after a time-travel restore.

        The schedule's batch counters for `epoch` and later are dropped so
        the re-run epochs advance from batch 0, and the watch accumulators
        forget the abandoned timeline's buckets — they're additive, so the
        re-run samples must start from empty ones.
        """
        with self._cv:
            self._schedule.rewind_to_epoch(epoch)
            # Restart the update cadence so post-jump frames fire on a clean
            # phase/epoch instead of wherever the abandoned timeline left it.
            self._freq_counter = 0
            self._freq_epoch = None
            # The standing numerical-error banner belonged to the abandoned
            # timeline; drop it so the re-run starts clean and can stop afresh.
            self._debug_error = None
        self._watch_accumulator.forget_epochs_from(epoch)
        # Custom-metric series from the abandoned timeline are dropped the
        # same way, and stateful instrument callables get their optional
        # `on_rewind` hook so cross-batch state doesn't leak into the replay.
        self._instruments.forget_epochs_from(epoch)
        self._instruments.notify_rewind(epoch)
        # Re-running this (or any later) epoch must re-save fresh RNG, so clear
        # the "already saved" marker — otherwise the re-run's pre-iter save in
        # `batches()` would be skipped and the replay would reuse stale state.
        self._epoch_start_saved_for = None
        # Note the jump on the console too (covers both the plain-loop and
        # Lightning restorers, which funnel through here).
        print(
            f"NaNsense: time-traveled to the start of epoch {epoch}.", flush=True
        )

    def _wait_for_proceed(self) -> None:
        # A pending time-travel jump also ends the wait: its request already
        # bumped the resume token, so a pause that began *after* the request
        # would otherwise sit waiting for a second UI command.
        with self._cv:
            seen = self._resume_token
            self._pause_count += 1
            self._paused = True
            self._cv.notify_all()
        # An unserved enabled session has no UI to resume a pause. A separate
        # driver thread (e.g. the test harness) still works — it resumes within
        # milliseconds — but a single-threaded script that forgot `port=` would
        # otherwise deadlock here forever. So when unserved, bound each wait:
        # if nothing resumes within the grace period, warn once and detach so
        # training runs to completion instead of hanging. A served session
        # waits for the UI indefinitely (grace is None).
        grace = None if self._served else _UNSERVED_PAUSE_TIMEOUT
        # Pause-time job loop: probe and experiment requests from the UI also
        # wake the paused training thread, which runs the work *here* — the
        # model is only ever touched from the training thread — and re-enters
        # the wait. Jobs run outside the lock so UI reads (mode, pause_count,
        # ...) stay responsive meanwhile.
        while True:
            with self._cv:
                resumed = self._cv.wait_for(
                    lambda: self._resume_token != seen
                    or self._closed
                    or self._pending_jump is not None
                    or self._probe_request
                    or bool(self._experiment_queue),
                    timeout=grace,
                )
                if not resumed:
                    # Unserved and nothing resumed us within the grace period:
                    # detach so this and every later batch run without pausing.
                    self._mode = Mode.DETACH
                    self._resume_token += 1
                    self._cv.notify_all()
                done = (
                    self._resume_token != seen
                    or self._closed
                    or self._pending_jump is not None
                )
                if done:
                    self._paused = False
                run_probe = False
                experiment: ExperimentRequest | None = None
                if not done:
                    run_probe = self._probe_request
                    self._probe_request = False
                    if self._experiment_queue:
                        experiment = self._experiment_queue.popleft()
                        # Mark it running while still holding the lock and
                        # atomically with the dequeue: otherwise a
                        # `cancel_experiment(seq)` landing between here and
                        # `run_experiment_guarded` acquiring the lock would
                        # find the seq neither queued nor running and be a
                        # silent no-op, letting a cancelled experiment run.
                        self._experiment_running = experiment.seq
            if not resumed:
                # The detach above already satisfied `done`; warn outside the
                # lock and return so the batch proceeds.
                _warn_unserved_detach()
                return
            if done:
                # A coalesced probe request is dropped (resuming into a
                # capture re-runs the probe anyway); queued experiments
                # stay queued for the next pause.
                return
            if run_probe:
                probe.run_probe_guarded(self)
            if experiment is not None:
                experiments.run_experiment_guarded(self, experiment)

    def _snapshot_input(self) -> Tensor | None:
        """The last snapshot's primary input tensor (the first model input).

        The experiment/attribution path works on a single image input; the
        probe instead re-forwards every input via `_snapshot_inputs`.
        """
        snap = self._snapshot
        input_name = self._input_names[0] if self._input_names else None
        if snap is None or input_name is None:
            return None
        return snap.activations.get(input_name)

    def _snapshot_inputs(self) -> dict[str, Tensor]:
        """Every input tensor of the last snapshot, keyed by input name.

        The probe base: a mapping of all of the model's inputs (each captured
        under its placeholder / forward-parameter name) so a probe can re-run
        the whole forward. Empty before the first snapshot.
        """
        snap = self._snapshot
        if snap is None:
            return {}
        inputs: dict[str, Tensor] = {}
        for name in self._input_names:
            tensor = snap.activations.get(name)
            if isinstance(tensor, Tensor):
                inputs[name] = tensor
        return inputs

    def _capture_forward(self, inputs: list[Tensor]) -> dict[str, Tensor]:
        """One isolated forward, every layer output as a fresh CPU clone.

        Thin wrapper over `capture.capture_forward` so probe runs go
        through the session (tests intercept probe forwards here). `inputs`
        are passed positionally in `input_names` order.
        """
        return capture.capture_forward(self, inputs)


class _BatchContext:
    def __init__(
        self, session: Session, *, phase: str, epoch: int, item: object = None
    ) -> None:
        self._session = session
        self._phase = phase
        self._epoch = epoch
        self._item = item
        self._position: BatchPosition | None = None
        self._captured = False
        self._freq_update = False
        self._snapshot_requested = False
        self._freeze_path: Path | None = None
        self._stats_only = False
        self._dist_reduce = False
        self._debug_check = False

    @property
    def _publishes(self) -> bool:
        """Whether this batch publishes a snapshot at `__exit__`.

        A mode capture, a frequency-cadence update, a one-shot UI Refresh
        request, or an armed moment freeze all publish; only a capture
        additionally pauses, and only a frequency update records a frame.
        """
        return (
            self._captured
            or self._freq_update
            or self._snapshot_requested
            or self._freeze_path is not None
        )

    def __enter__(self) -> Self:
        # `_enabled` is a plain attribute read (no lock), checked first so a
        # disabled session pays nothing per batch: no schedule advance, no
        # capture decision, no hook install. `_position` stays None, so
        # `__exit__` also returns immediately.
        if not self._session._enabled or self._session.closed:
            return self
        self._position = self._session._schedule.advance(self._phase, self._epoch)
        # Publish the live position before the forward pass so the UI top bar
        # tracks progress on every batch, even in modes that don't capture
        # snapshots here (step_epoch, step_until_position, step_run, detach).
        self._session._live_position = self._position
        dist_ctx = self._session._dist
        # A jump that arrived while training was running (not paused) is
        # applied before this batch does any work. Raising from __enter__
        # skips __exit__, but nothing has been installed yet.
        #
        # Single-process: consume the leader's pending jump and raise straight
        # away (byte-identical to the pre-DDP path). Distributed: the jump
        # can't be raised before `sync_batch_control`, or the followers — held
        # in that broadcast at their next batch start — would deadlock. So the
        # leader peeks its target and carries it through the control broadcast;
        # every rank then raises together *after* the barrier (see below).
        if dist_ctx is None:
            jump = self._session._take_pending_jump()
            if jump is not None:
                raise TimeTravelJump(jump)
        # With a restorer attached, the first batch of each epoch checkpoints
        # the epoch-start state (model/optimizer/scheduler/RNG) to disk —
        # before any forward pass, so a later jump back to this epoch restores
        # exactly this moment. This is a fallback for users who drive `batch()`
        # manually; the canonical `batches()` path saves *before* `iter(loader)`
        # draws the shuffle seed, and the `_epoch_start_saved_for` guard inside
        # `_maybe_save_epoch_start` keeps this post-draw save from clobbering it.
        if self._position.batch_idx == 0:
            self._session._maybe_save_epoch_start(self._epoch)
        if dist_ctx is None or dist_ctx.is_leader:
            self._captured = self._session._should_capture(self._position)
            # A frequency update publishes like a capture but never pauses;
            # it is decided independently of the mode, so visualizations
            # keep refreshing during step-epoch / run / detach.
            self._freq_update = self._session._should_freq_update(self._position)
            # A one-shot UI Refresh request also publishes (no pause), so a
            # free-running mode shows the live model on demand. Consumed here
            # so exactly the next batch publishes.
            self._snapshot_requested = self._session._take_snapshot_request()
            # The numerical-error debugger checks every nth batch, independent
            # of mode — so it needs hooks installed here even on a plain
            # (detach) batch to see the activation gradients at __exit__.
            self._debug_check = self._session._should_debug_check(self._position)
            # An armed moment freeze matches exactly one position; the
            # matching batch publishes (hooks install below) and `__exit__`
            # writes the file.
            self._freeze_path = self._session._take_freeze_request(self._position)
        if dist_ctx is not None:
            # Per-batch control sync (every rank): the leader announces
            # whether this batch's watch stats get globally reduced at
            # __exit__, shares watched-set changes, and broadcasts its armed
            # time-travel target; followers apply them (and never capture or
            # publish themselves). This is also the pacing point — a leader
            # paused in the UI holds every other rank right here, at its next
            # batch start.
            self._dist_reduce, jump_epoch = distributed.sync_batch_control(
                self._session,
                publish=self._publishes,
                jump_epoch=(
                    self._session._peek_pending_jump() if dist_ctx.is_leader else -1
                ),
            )
            # All ranks raise at this identical point — after the barrier,
            # before any forward/backward/collective — so no collective is
            # left half-issued and the ranks unwind in lockstep. The leader
            # now consumes its own request (atomically clearing it so the next
            # batch doesn't re-broadcast a stale jump).
            if jump_epoch >= 0:
                if dist_ctx.is_leader:
                    self._session._take_pending_jump()
                raise TimeTravelJump(jump_epoch)
        self._stats_only = not self._publishes and self._session._stats_active()
        # Publishing, frequency-update, stats-only, and debug-check batches use
        # the same hook installation: full fx interpreter (or full per-module
        # hooks + root pre-hook in hook-mode). That way any name in
        # `layer_names` — inputs, fx intermediates, modules — can be watched or
        # checked. The only difference is what happens at __exit__ (publish /
        # pause / stats / numerical check).
        if self._publishes or self._stats_only or self._debug_check:
            capture.install_hooks(self._session)
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        if self._position is None:
            return
        if self._publishes or self._stats_only or self._debug_check:
            # Hook removal MUST run once hooks were installed, even if the
            # watch-stats update raises — otherwise the fx-patched forward
            # leaks past this batch and the next install captures it as the
            # "original", permanently losing the real forward. So the whole
            # block runs under try/finally with `remove_hooks` in the finally.
            # In the normal flow we still remove hooks *before* publishing and
            # probing: the snapshot reads the already-captured `_activations`,
            # and the probe runs its own forward (in hook-fallback mode that
            # would otherwise re-fire the live per-module hooks). The finally's
            # `remove_hooks` is then a no-op (it is idempotent) — its job is to
            # cover the path where `_update_watch_stats` raised before it ran.
            try:
                if exc is None and self._session._stats_active():
                    self._session._update_watch_stats(self._position)
                if exc is None and self._dist_reduce:
                    # Collective: every rank folds its shard's accumulated
                    # watch stats into the leader's global view, right
                    # before the leader publishes (and possibly pauses), so
                    # the UI never shows a pause with stale global stats.
                    distributed.reduce_watch_stats(self._session)
                # Numerical-error checks read the live activations and their
                # retained gradients, so they run before the hooks come off.
                # The *first* detected error stops training and forces this
                # batch to publish + pause (even in a free-running mode) so the
                # UI can show the affected layers behind the banner. Once a
                # banner is up, later detections accumulate without stopping
                # (no force), so a resumed run keeps going.
                debug_force = (
                    self._session._run_debug_checks(self._position)
                    if exc is None and self._debug_check
                    else False
                )
                capture.remove_hooks(self._session)
                if exc is None and not self._session.closed:
                    force = debug_force
                    if self._publishes or force:
                        self._session._publish_snapshot(self._position)
                        # The snapshot holds CPU clones of everything; drop the
                        # live GPU activations (and their retained grads) now so
                        # the probe's forward below doesn't stack a second
                        # batch's worth of memory on top of them.
                        self._session._activations.clear()
                        if self._freeze_path is not None:
                            # Imported lazily: nansense.moments imports this
                            # module at the top level.
                            from nansense import moments

                            moments.write_moment(
                                self._session,
                                self._freeze_path,
                                batch_item=self._item,
                            )
                        probe.maybe_run_probe_at_capture(self._session)
                        # Auto experiments re-run on every publish, so a pause
                        # shows fresh results and a free-running frequency
                        # update keeps open pages / recordings current.
                        experiments.run_auto_experiments(self._session)
                    if self._freq_update:
                        self._session._record_frames()
                    if self._captured or force:
                        self._session._wait_for_proceed()
            finally:
                capture.remove_hooks(self._session)
                self._session._activations.clear()
        # Single-process: every batch boundary — captured, stats-only, or
        # plain (detach) — consumes an armed time-travel jump and raises here.
        # `_wait_for_proceed` above returns immediately when a jump is pending,
        # so a paused batch reacts to the request without a second UI command.
        #
        # Distributed: the jump is NOT raised here. The followers are held in
        # the *next* batch's `sync_batch_control` broadcast, so a leader that
        # raised at __exit__ would deadlock them. Instead `_pending_jump` stays
        # armed; `_wait_for_proceed` woke the (possibly paused) leader, this
        # __exit__ completes normally, and the jump is broadcast and applied in
        # lockstep at the next __enter__ barrier — never mid-batch, so
        # `reduce_watch_stats` runs on all-or-no ranks.
        if exc is None and self._session._dist is None:
            jump = self._session._take_pending_jump()
            if jump is not None:
                raise TimeTravelJump(jump)

    @property
    def position(self) -> BatchPosition | None:
        return self._position

    @property
    def captured(self) -> bool:
        return self._captured


def start(
    model: nn.Module,
    *,
    epochs: int | None = None,
    phases: dict[str, int] | None = None,
    enabled: bool = True,
    optimizer: Optimizer | None = None,
    scheduler: LRScheduler | None = None,
    port: int | None = None,
    host: str = "127.0.0.1",
    open_browser: bool = True,
    input_mean: MeanStd | dict[str, MeanStd] | None = None,
    input_std: MeanStd | dict[str, MeanStd] | None = None,
    input_transform: InputTransform | dict[str, InputTransform] | None = None,
) -> Session:
    """Create a `Session` for `model` (and optionally serve the UI).

    The training schedule is discovered as you go: declare the epoch count at
    the loop (`for epoch in session.epochs(N)`), and phase names + per-phase
    batch counts are learned as `session.batches(loader, phase=…)` runs (the
    full shape is known after the first epoch). `epochs` here is an optional
    fallback for the count; `phases={"train": a, "val": b}` is an optional
    up-front declaration that restores full first-epoch fidelity (exact
    progress and boundary stops from batch 0) — the Lightning integration uses
    it, and it is the right choice when you need the UI fully precise on epoch 0.

    With `enabled=False` the session is a near-zero-overhead no-op: no fx
    trace at construction, `batch()` does nothing, and the UI is skipped.
    This lets a training script keep its NaNsense wiring in place and turn
    the whole UI off with a single flag.

    `optimizer` is optional: when given, snapshots (and the weights page)
    additionally carry each parameter's optimizer state — momentum buffers,
    Adam moments, step counts — plus its param group's numeric
    hyperparameters. Without it, everything behaves exactly as before.

    `scheduler` is optional: when given, time-travel checkpoints include the
    LR scheduler's state, so a jump restores the learning-rate schedule
    automatically along with the model and optimizer.

    `port` is optional: when given, the UI is served immediately on that
    port (equivalent to a separate `nansense.serve(session, port=...)`
    call, which remains available for finer control). `host`,
    `open_browser`, `input_mean`, `input_std`, and `input_transform` are
    forwarded to `serve`; once the port binds, the address is printed in a box
    and (unless `open_browser=False`) opened in a focused browser tab. If a
    concurrent session already holds the port, both are suppressed.
    `input_mean` / `input_std` / `input_transform` each take a single value
    applied to every input, or a dict keyed by input name for multi-input
    models (see `nansense.input_config`); `input_transform` maps a non-RGB
    input to a displayable 1-/3-channel image.

    Distributed (DDP) runs need no special wiring: call `start()` on every
    rank (a `DistributedDataParallel`-wrapped model is unwrapped
    automatically). Rank 0 serves the UI and drives pausing/stepping;
    the other ranks skip the UI, follow rank 0's pace, and contribute
    their data shard to the watch page's statistics, which become global
    across ranks. Time travel is supported under DDP: drive every rank's epoch
    loop with `session.epochs()`; a jump rewinds all ranks in lockstep from
    their own per-rank checkpoints.
    """
    session = Session(
        model,
        epochs=epochs,
        phases=phases,
        enabled=enabled,
        optimizer=optimizer,
        scheduler=scheduler,
    )
    if port is not None:
        # Imported lazily: nansense.ui imports this module at the top level.
        from nansense.ui import serve

        serve(
            session,
            port=port,
            host=host,
            open_browser=open_browser,
            input_mean=input_mean,
            input_std=input_std,
            input_transform=input_transform,
        )
    return session
