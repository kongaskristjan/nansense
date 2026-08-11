# API reference

The public surface of the `nansense` package. Everything here is importable as `nansense.<name>`, except the Lightning integration, which lives in `nansense.lightning` (available when `lightning` is installed).

## Entry points

## nansense.start

```
start(
    model: Module,
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
    input_transform: InputTransform
    | dict[str, InputTransform]
    | None = None,
) -> Session
```

Create a `Session` for `model` (and optionally serve the UI).

The training schedule is discovered as you go: declare the epoch count at the loop (`for epoch in session.epochs(N)`), and phase names + per-phase batch counts are learned as `session.batches(loader, phase=…)` runs (the full shape is known after the first epoch). `epochs` here is an optional fallback for the count; `phases={"train": a, "val": b}` is an optional up-front declaration that restores full first-epoch fidelity (exact progress and boundary stops from batch 0) — the Lightning integration uses it, and it is the right choice when you need the UI fully precise on epoch 0.

With `enabled=False` the session is a near-zero-overhead no-op: no fx trace at construction, `batch()` does nothing, and the UI is skipped. This lets a training script keep its NaNsense wiring in place and turn the whole UI off with a single flag.

`optimizer` is optional: when given, snapshots (and the weights page) additionally carry each parameter's optimizer state — momentum buffers, Adam moments, step counts — plus its param group's numeric hyperparameters. Without it, everything behaves exactly as before.

`scheduler` is optional: when given, time-travel checkpoints include the LR scheduler's state, so a jump restores the learning-rate schedule automatically along with the model and optimizer.

`port` is optional: when given, the UI is served immediately on that port (equivalent to a separate `nansense.serve(session, port=...)` call, which remains available for finer control). `host`, `open_browser`, `input_mean`, `input_std`, and `input_transform` are forwarded to `serve`; once the port binds, the address is printed in a box and (unless `open_browser=False`) opened in a focused browser tab. If a concurrent session already holds the port, both are suppressed. `input_mean` / `input_std` / `input_transform` each take a single value applied to every input, or a dict keyed by input name for multi-input models (see `nansense.input_config`); `input_transform` maps a non-RGB input to a displayable 1-/3-channel image.

Distributed (DDP) runs need no special wiring: call `start()` on every rank (a `DistributedDataParallel`-wrapped model is unwrapped automatically). Rank 0 serves the UI and drives pausing/stepping; the other ranks skip the UI, follow rank 0's pace, and contribute their data shard to the watch page's statistics, which become global across ranks. Time travel is supported under DDP: drive every rank's epoch loop with `session.epochs()`; a jump rewinds all ranks in lockstep from their own per-rank checkpoints.

## nansense.serve

```
serve(
    session: Session,
    *,
    port: int = 8080,
    host: str = "127.0.0.1",
    log_level: str = "warning",
    open_browser: bool = True,
    mcp: bool = True,
    input_mean: MeanStd | dict[str, MeanStd] | None = None,
    input_std: MeanStd | dict[str, MeanStd] | None = None,
    input_transform: InputTransform
    | dict[str, InputTransform]
    | None = None,
) -> Thread | None
```

Start the NiceGUI app on a background thread and return that thread.

`port` / `host` pick the bind address (default `127.0.0.1:8080`). `log_level` is uvicorn's log level — `"warning"` by default, so routine request logging stays out of the training console.

Returns `None` without starting anything when `session` is disabled (`nansense.start(..., enabled=False)`), so a training script can call `serve()` unconditionally and pay nothing when the UI is turned off — and likewise on the non-zero ranks of a distributed run, where the UI lives on rank 0.

NiceGUI is mounted onto a bare FastAPI app via `ui.run_with`; the app is then served by uvicorn from a non-main thread, with signal handlers disabled so uvicorn doesn't try to wire SIGINT/SIGTERM from a thread that isn't the main one.

Once the server thread is launched, a daemon thread waits for the port to bind and then prints the UI address inside a box (so it stands out in the training log) and, unless `open_browser` is `False`, opens it in a focused browser tab. If a concurrent session already holds the port the bind fails, so the banner and the browser tab are both suppressed — only uvicorn's own `address already in use` error is shown. On a headless machine the bind still succeeds, so the banner prints and the browser open is a harmless no-op.

`mcp` (default `True`) also serves the MCP endpoint at `/mcp` on the same port, so a coding agent can drive the debugger through the same session the browser shows (`nansense.mcp_server`). Its route is registered *before* NiceGUI's catch-all mount at `/` — Starlette matches routes in order — and its lifespan is passed to the app at construction, since NiceGUI wraps whatever lifespan it finds and a mounted sub-app never receives one.

`input_mean` / `input_std` are passed to the input-image pane so the sample is denormalized (`x * std + mean`) before display. When either is `None`, the renderer assumes the input is already in `[0, 1]`. `input_transform` maps a non-RGB input to a displayable 1-/3-channel image. Each of the three is either a single value applied to every input, or a `dict` keyed by input name for a multi-input model (see `nansense.input_config`); the stats and experiment panes use the primary input's resolved values, and so does the MCP server, whose image tools render the same views.

## The session

## nansense.Session

The bridge between a live training loop and the NaNsense UI.

Create one with `nansense.start` (the intended entry point) rather than directly. The training loop drives the session through `batches` (wrap each phase's dataloader), `epochs` + `restore_point` (the time-travel epoch loop), and `close` when training finishes; the served UI drives pausing, stepping, layer watching and experiments through the rest of the surface. With `enabled=False` every method is a near-zero-overhead no-op.

### enabled

```
enabled: bool
```

Whether this session captures anything. Set once at `start()`.

A disabled session is fully inert: `batch()` is a no-op context manager, `serve()` does nothing, and no model hooks are ever installed — the intended near-zero-overhead off switch for leaving NaNsense wiring in place in a training script.

### snapshot

```
snapshot: BatchSnapshot | None
```

### live_position

```
live_position: BatchPosition | None
```

Position of the batch the training thread is currently on.

Updated on *every* batch's `__enter__` regardless of capture mode, so the UI can show live epoch/batch progress during `step_epoch`, `step_until_position`, `step_run`, and `detach` — modes that publish a snapshot only at boundaries (or never), leaving `snapshot.position` frozen in between. Written by the training thread, read point-in-time by the UI thread: a single atomic reference assignment under the GIL, no lock needed (same contract as `snapshot`).

### mode

```
mode: Mode
```

### instrument_errors

```
instrument_errors: dict[str, str]
```

`instrument name -> error` for instruments disabled by a failure.

### batches

```
batches(
    loader: Iterable[_BatchItem],
    *,
    phase: str,
    epoch: int | None = None,
) -> Iterator[_BatchItem]
```

Iterate `loader` with each item wrapped in a `batch()` context.

Sugar over `batch()` for the common loop shape: the user's batch body runs while the generator is suspended at `yield`, i.e. inside the batch context — hooks are installed before the forward pass and the capture/pause happens when the loop asks for the next item. A `TimeTravelJump` raised at a batch boundary therefore surfaces from the `for` statement itself, not from inside the user's body.

```
for inputs, targets in session.batches(loader, phase="train"):
    ...  # forward / backward / step
```

`epoch` defaults to the epoch `session.epochs()` last yielded, so the flat loop need not repeat it; pass it explicitly when driving the phases outside an `epochs()` loop.

### epochs

```
epochs(
    n: int | None = None,
    *,
    cache_dir: Path = DEFAULT_CACHE_DIR,
    start_epoch: int = 0,
) -> Iterator[int]
```

Time-travel-aware epoch loop: `for epoch in session.epochs(50): ...`.

`n` is the total number of epochs — the canonical place to declare it (it sets the schedule's epoch count); pass it here rather than to `nansense.start`. If omitted, the count set on `start(epochs=…)` is used, and it is an error if neither was given.

Yields `0 … n - 1` like `range`, but opts the session into time travel: each epoch start is checkpointed to `cache_dir` and a UI-requested jump re-enters the loop at the chosen epoch with the model / optimizer / scheduler / RNG restored. Wrap each iteration's body — every phase of the epoch — in `with session.restore_point():`, the block that catches the jump; the generator then re-yields the target epoch::

```
for epoch in session.epochs(50, cache_dir="models/latest"):
    with session.restore_point():
        for inputs, targets in session.batches(train_dl, phase="train"):
            ...
        for inputs, targets in session.batches(val_dl, phase="val"):
            ...
```

`start_epoch` resumes the run from `cache_dir`'s checkpoint of that epoch instead of training from scratch: the cache directory is adopted as-is (it may come from an earlier process — e.g. one baked into a deployment image), the checkpoint is validated against the live model / optimizer / scheduler up-front (`TimeTravelError` on a mismatch), and the loop yields `start_epoch … n - 1` with the state restored on the training thread before the first batch. Time travel then covers every epoch the directory holds a checkpoint for.

Not iterating this leaves the run a straight pass with the UI's Time Travel button disabled; on a disabled session it is inert and nothing touches the disk (`start_epoch` is then ignored). Under DDP call it on every rank — a leader jump is broadcast so all ranks re-yield the same epoch in lockstep.

### set_schedule

```
set_schedule(
    *,
    epochs: int | None = None,
    phases: dict[str, int] | None = None,
) -> None
```

Re-declare the schedule mid-run; batch counters are kept.

`phases` switches the schedule to declared mode and replaces the phase set and their batch counts, so it is the way to express a shape that varies per epoch — validating every N epochs declares a train-only schedule on the epochs that skip it (what the Lightning integration does on each `on_train_epoch_start`). A schedule fixed for the whole run is better passed to `start(phases=...)` once.

Either argument may be omitted to leave that part unchanged.

### restore_point

```
restore_point() -> AbstractContextManager[TrainingRestorer]
```

Per-epoch restore boundary for the `session.epochs()` loop.

Enter it around each epoch's body (both train and val phases): on a time-travel jump the `TimeTravelJump` unwinds to here, is suppressed, and `session.epochs()` re-yields the target epoch with state restored. Loop state that depends on history (a running `best_acc`, metric curves) belongs inside it, so a jump rewinds it naturally. Must be used inside a `for epoch in session.epochs():` loop.

### training_restorer

```
training_restorer(
    *, cache_dir: Path = DEFAULT_CACHE_DIR
) -> TrainingRestorer
```

Create the restorer that opts this session into time travel.

Wrapping the epoch loop in the returned object (see `TrainingRestorer`) enables both epoch-start checkpointing to `cache_dir` and UI-driven jumps back to any cached epoch. A session without a restorer never writes checkpoints and never raises `TimeTravelJump`. On a disabled session the restorer is inert: the loop runs exactly once and nothing touches the disk.

Under DDP every rank wraps its epoch loop in a restorer the same way: a UI-requested jump on the leader is broadcast to all ranks at the next batch-start barrier, where every rank raises `TimeTravelJump` in lockstep and restores from its own per-rank checkpoint (see `nansense.distributed`).

### close

```
close() -> None
```

Mark training finished once the loop completes.

Releases anything waiting on the session and finalizes in-flight recordings so their MP4 files are playable. The served page stays up for post-mortem browsing of the last captured state — `close` ends the *training* side of the session, not the UI.

### time_travel_status

```
time_travel_status() -> TimeTravelStatus
```

What the UI needs to render the time-travel button and dialog.

### lock

```
lock() -> None
```

Lock run control and global settings for a shared demo deployment.

Intended for a publicly hosted playground where many anonymous visitors share one session. Once locked:

- the run-control methods (`stop`, every `step_*`, `detach`), `request_time_travel`, `watch`/`unwatch`, the probe surface (`pin_current_batch`, `unpin_batch`, `add_perturbation`, `clear_perturbations`, `set_probe_mode` — all shared state every visitor sees), and the global settings (`set_stats_scope`, `set_update_frequency`, `set_watch_performance`, `set_debug_settings`, `disable_debug_check`, `set_auto_run_experiments`, `set_experiment_defaults`) become no-ops (`request_time_travel` raises `TimeTravelError`, so the UI can show why);
- the stats scope is forced to `ALL` — every layer collects, so the UI's per-tab show/hide never touches shared state;
- experiment requests still run, with their parameters clamped and the queue depth capped (see `nansense.experiments`);
- registering instruments (`watch_metric`, `watch_layer_tensor`, `watch_weight_tensor`) raises — register them, like every other host-script setting, *before* locking.

Everything read-only stays available, as do experiments. The lock is one-way by design — arm the wanted mode (e.g. `step_run()`), the settings, and any demo pin *before* locking. `close()` is not locked; it belongs to the hosting script.

### freeze_moment

```
freeze_moment(
    path: Path | str,
    *,
    phase: str,
    epoch: int,
    batch_idx: int,
) -> None
```

Arm a one-shot moment freeze at an exact batch position.

When training reaches `(phase, epoch, batch_idx)`, that batch installs hooks and publishes a snapshot like a capture — whatever the mode, so a detached prepare run works — and, after folding its watch stats, writes the complete debugger moment to `path` without pausing: the batch's loader item (the replay seed the snapshot is regenerated from — drive the loop with `batches()`, or pass `item=` to a manual `batch()`), running statistics, watched set, model weights and buffers, and the schedule shape. `nansense.load_moment` later rebuilds the view around a fresh model of the same architecture, the backing for a locked showcase (see `nansense.moments` and `examples/playground`).

One request at a time — arming again replaces an unconsumed target; a target the run never reaches is reported at `close()`. No-op on a disabled or locked session; leader-only under DDP (the frozen statistics are the leader's shard).

### set_patch_layers

```
set_patch_layers(layers: Iterable[str] | None) -> None
```

Restrict extreme-patch collection to `layers` (`None` = every layer).

Histogram/min-max/graph statistics are unaffected — this only gates the per-channel extreme-input patch buffers, by far the largest watch state (input crops, whole-image samples, and activation heatmaps per channel per layer). A scope-`all` run on a deep model at real image sizes spends gigabytes there; restricting patches to the layers a demo actually showcases keeps memory (and a frozen moment's file size) proportional to the shortlist while every layer keeps its statistics views.

Unknown layer names raise `ValueError` on an enabled session (they would silently collect nothing). No-op when locked; layers already holding patch buffers keep them (this gates new accumulation only).

### park

```
park() -> None
```

Hold the calling thread at a pause, serving UI requests, until `close()`.

The showcase counterpart of a training loop: a script that restored a frozen moment (`nansense.load_moment`) has no batches to drive, but experiments and probes still execute on the pause loop of whatever thread owns the model — this provides that loop. Call it from the thread that built the model, typically right after `lock()`; on an unlocked session a Run/Step click simply re-enters the park. Returns once the session is closed.

### watch_metric

```
watch_metric(
    name: str,
    *,
    on: str = "batch",
    reduce: MetricReduce | None = None,
) -> Callable[[_InstrumentFn], _InstrumentFn]
```

Register a custom scalar metric, evaluated per collected layer.

Use as a decorator (or call the returned registrar directly with any callable — a stateful object works too):

```
@session.watch_metric("sparsity")
def sparsity(ctx: nansense.LayerContext) -> float:
    return (ctx.activation > 0).float().mean().item()
```

The callback runs on the training thread, under `torch.no_grad()`, once per stats batch for every layer the stats scope collects (the watched set by default) — it receives a `LayerContext` with the batch's live activation, its gradient, and the layer's weights / optimizer state, and must not mutate them. It may return a number (or 1-element tensor), a mapping of named scalars (one plot trace per key), or `None` to skip the layer.

`on="batch"` plots every batch's value; `on="epoch"` folds each epoch's values through `reduce` — `"mean"` (default), `"sum"`, `"min"`, `"max"`, `"last"`, or any `values -> float` callable — into one point. The series appear in the `/stats` GRAPHS view, one plot per metric. A raising callback is disabled (training continues) and reported via `instrument_errors`. Raises `ValueError` on invalid arguments and `RuntimeError` on a locked session.

### watch_layer_tensor

```
watch_layer_tensor(
    name: str,
) -> Callable[[_InstrumentFn], _InstrumentFn]
```

Register a custom activation-shaped tensor per collected layer.

```
@session.watch_layer_tensor("zscore")
def zscore(ctx: nansense.LayerContext) -> Tensor:
    a = ctx.activation
    return (a - a.mean()) / (a.std() + 1e-6)
```

The callback follows `watch_metric`'s contract (training thread, `torch.no_grad()`, live tensors, `None` skips) but runs on publish batches only and must return a tensor of the activation's shape — it lands on `BatchSnapshot.custom_activations` and renders as an extra strip under the layer card's activation/gradient strips.

### watch_weight_tensor

```
watch_weight_tensor(
    name: str,
) -> Callable[[_InstrumentFn], _InstrumentFn]
```

Register a custom weight-shaped tensor per collected parameter.

```
@session.watch_weight_tensor("adam_update")
def adam_update(ctx: nansense.WeightContext) -> Tensor | None:
    if "exp_avg" not in ctx.optimizer_state:
        return None
    m = ctx.optimizer_state["exp_avg"]
    v = ctx.optimizer_state["exp_avg_sq"]
    return m / (v.sqrt() + 1e-8)
```

The callback follows `watch_metric`'s contract but runs on publish batches only, once per (collected layer, parameter), receiving a `WeightContext`; it must return a tensor of the parameter's shape — it lands on `BatchSnapshot.custom_weight_tensors` and renders on the `/weights` page alongside the weight/gradient/optimizer strips.

### watch_metrics_snapshot

```
watch_metrics_snapshot(
    *, layers: Iterable[str] | None = None
) -> MetricsSnapshot
```

Frozen view of the custom scalar-metric series (see `watch_metric`).

The GRAPHS view's data source for the custom-metric plots; safe to call from any thread. Pass `layers` to restrict the copy.

## PyTorch Lightning integration

## nansense.lightning.NansenseCallback

Bases: `Callback`

Drive a NaNsense session from a Lightning `Trainer`.

The session is created when `fit` starts (optimizers exist by then) and closed when it ends — the UI stays up for post-mortem browsing, exactly like a hand-written loop. Pass `model="net"` (an attribute path inside the LightningModule) to point NaNsense at the actual network; this is recommended whenever the module wraps its layers in a submodule, both for fx tracing and so input capture sees the real forward signature.

`port` / `host` / `open_browser` / `enabled` / `input_mean` / `input_std` / `input_transform` mean the same as on `nansense.start`; the UI comes up once `fit` begins.

Supported out of the box: automatic optimization, epoch-boundary validation (including `check_val_every_n_epoch > 1`), sanity-check skipping, and `enabled=False` as the zero-overhead off switch. Mid-epoch validation (`val_check_interval < 1.0` or step-based) is rejected with a clear error, and iterable-style dataloaders without a length are not supported — NaNsense declares the schedule up-front.

### session

```
session: Session | None
```

The live session, or None before `fit` has started.

## nansense.lightning.fit_with_time_travel

```
fit_with_time_travel(
    make_trainer: Callable[[], Trainer],
    model: LightningModule,
    *,
    callback: NansenseCallback,
    train_dataloaders: _Dataloaders | None = None,
    val_dataloaders: _Dataloaders | None = None,
    datamodule: LightningDataModule | None = None,
    cache_dir: Path = DEFAULT_CACHE_DIR,
) -> None
```

Run `trainer.fit` under NaNsense time travel.

Equivalent to `make_trainer().fit(model, ...)` with `callback` attached, plus the UI's Time Travel button: every epoch boundary is checkpointed to `cache_dir`, and a jump re-enters training at the chosen epoch with model / optimizer / scheduler / RNG state restored. `model`, `train_dataloaders`, `val_dataloaders` and `datamodule` are forwarded to `trainer.fit` unchanged.

`make_trainer` is a factory because each jump needs a fresh `Trainer`: Lightning treats a trainer as single-use for `fit`, and after the `TimeTravelJump` unwinds, the old one has already torn itself down. The callback (and its session, UI included) lives across attempts.

Note that metric loggers cannot time-travel — after a jump, an attached logger sees the replayed epochs again (overlapping curves or a fresh run per attempt, depending on the logger).

## Positions and snapshots

## nansense.BatchPosition

Where a batch sits in the run: the `phase` name plus 0-indexed `epoch` and `batch_idx`. This is the position record carried by `BatchSnapshot.position`, `DebugError.position` and `Session.live_position`.

The `is_last_*` flags mark boundaries — the last batch of the phase, of the epoch, and of the whole run. They are best-effort: with a lazy schedule they stay `False` until the phase's batch count has been learned at the end of the first epoch.

## nansense.Mode

Bases: `StrEnum`

How training proceeds after a resume, exposed as `Session.mode`.

Each mode maps to a top-bar control: `STEP` pauses on every batch (Step Batch; also the initial mode, so the first batch always pauses), `UNTIL_PHASE_CHANGE` / `UNTIL_EPOCH_CHANGE` run to the first batch of the next phase/epoch (Step Phase / Step Epoch), `UNTIL_POSITION` runs to an exact position (the step-until dialog), `UNTIL_END` runs to the run's last batch (Run), and `DETACH` never pauses again — capture overhead drops to near zero until the UI re-engages.

## nansense.Schedule

Batch-position bookkeeping, safe to touch from both session threads.

`advance` and `record_phase_length` run on the training thread (every batch / each phase end) while the UI thread reads `epochs`/`phases`/ `phase_order` and may `update`/`set_epochs`/`rewind_to_epoch`; a lock guards the shared counters and the phase/epoch fields so a mid-`advance` schedule swap can't yield an inconsistent `BatchPosition` or corrupt the counters. The lock is never held while touching the session's condition variable, so it only ever nests *inside* it — no lock-ordering cycle.

`phases` may be `None` (lazy mode): phase names and counts are then learned by observation. A declared `phases` dict pins both up front and keeps the stricter validation (unknown phase / more batches than declared raise).

### epochs

```
epochs: int | None
```

Total epochs, or `None` until `set_epochs`/`session.epochs(n)` runs.

### phases

```
phases: dict[str, int]
```

Known phases mapped to their batch counts, in first-seen order.

Declared mode returns the full dict up front; lazy mode grows it as each phase's count is learned (so it can be empty during the first epoch).

### phase_order

```
phase_order: list[str]
```

All phase names seen so far, in order — including ones whose count is not yet known (unlike `phases`, which only lists counted phases).

### phase_count

```
phase_count(phase: str) -> int | None
```

The known batch count for `phase`, or `None` if not yet learned.

## nansense.BatchSnapshot

Immutable per-batch view, fully resident on CPU.

All tensor dicts are independent CPU clones taken at snapshot time, so the snapshot survives subsequent batches freeing the live tensors and can be safely read from any thread.

`position` is where the batch sat in the run. `activations` / `activation_gradients` are keyed by watched-layer name (the names shown in the architecture graph); `weights` / `weight_gradients` by parameter name, matching `model.named_parameters()`.

`optimizer_state` / `optimizer_hyperparams` are populated only when the session was given an optimizer at `start()`; otherwise they stay empty. State entries are keyed `param name -> state key -> tensor` (scalar entries like Adam's `step` become 0-dim tensors); hyperparams are the numeric knobs of the parameter's group (`lr`, `momentum`, ...), read at the same instant — so a scheduler-driven `lr` is the batch's actual one.

`custom_activations` / `custom_weight_tensors` carry the tensor instruments' outputs (see `Session.watch_layer_tensor` / `watch_weight_tensor`): activation-shaped tensors keyed `layer -> instrument name`, weight-shaped ones `param -> instrument name`. Empty without registered instruments (or when nothing collects).

## Watch statistics

## nansense.StatsScope

Bases: `StrEnum`

Which layers fold their batches into the running statistics.

- `NONE`: nothing is collected; already-collected stats are kept frozen (the pause the top bar's stats toggle uses).
- `WATCHED` (default): the watched layers collect, and watching a layer is also what shows its cards — the classic coupled behaviour.
- `ALL`: every layer in `layer_names` collects, independent of the watched set.

Outside `WATCHED`, the watched set no longer drives collection, so the UI treats showing/hiding a layer's cards as per-tab state that never touches the session (see `main_page`).

## nansense.WatchSnapshot

Immutable view of all accumulated stats at a point in time.

Keyed by `(layer, phase, epoch)`. The UI is expected to filter to the layers it wants to display (typically the latest epoch for each phase).

### latest_per_phase

```
latest_per_phase(
    layer: str,
) -> dict[str, LayerStatsSnapshot]
```

For `layer`, return `phase -> stats` for the most recent epoch seen.

Returns an empty dict if the layer has no entries yet.

### phase_history

```
phase_history(
    layer: str, phase: str
) -> list[LayerStatsSnapshot]
```

`layer`'s buckets for `phase`, ordered by epoch.

The epoch-by-epoch series behind the value-vs-epoch stats view. Older epochs carry universal-histogram stats only (their per-channel buffers collapsed when a newer epoch started).

### weight_history

```
weight_history(
    layer: str,
) -> dict[str, list[tuple[int, TensorStatsSnapshot]]]
```

`layer`'s per-epoch weight samples, `param -> [(epoch, stats)]`.

Each list is ordered by epoch — the series behind the GRAPHS view's weight plots. Empty for layers without parameters (fx intermediates, graph inputs).

## nansense.LayerStatsSnapshot

One watched layer's statistics for a (phase, epoch) bucket.

Bundles the layer's activation and gradient `TensorStatsSnapshot`s with the extreme-input patch gallery; `layer` / `phase` / `epoch` locate the bucket within `WatchSnapshot.stats`.

## nansense.TensorStatsSnapshot

Immutable CPU-side view of a single (layer, phase, epoch, kind) accumulator.

`n` / `sum` / `sum_sq` / `min` / `max` are running scalars over every tensor element seen so far, feeding the `mean` / `variance` / `std` properties. `hist` is the layer-wide histogram: one count per bin over the fixed symmetric-log bin edges of `histogram_edges()`; `None` when an older epoch's bins were collapsed by a newer one starting for its phase (only the latest epoch per phase renders bins — the per-epoch GRAPHS curves read the scalars and `median`, which survives the collapse via `collapsed_median`).

### mean

```
mean: float
```

Arithmetic mean of all elements seen; `nan` before any data.

### variance

```
variance: float
```

Population variance; `nan` with fewer than two elements.

### std

```
std: float
```

Population standard deviation derived from `variance`.

### median

```
median: float
```

Histogram-derived median: midpoint of the bin that holds the median.

For a collapsed bucket (`hist is None`) this is the value cached at collapse time (`collapsed_median`).

### dead_channel_count

```
dead_channel_count: int | None
```

How many channels only ever hit the zero bin, `None` when unknown.

Live from `channel_hists` while the per-channel histogram exists; the count stored at collapse time for an evicted older epoch.

## Custom instruments

See the [Custom metrics & tensors guide](https://kongaskristjan.github.io/nansense/0.3/instruments/index.md) for usage; the registration decorators live on the session (`Session.watch_metric`, `Session.watch_layer_tensor`, `Session.watch_weight_tensor` above).

## nansense.LayerContext

One watched layer's live tensors for a batch, passed to instruments.

`activation` / `gradient` are the layer's captured output and its retained grad (`None` in forward-only phases); `weights`, `weight_gradients`, and `optimizer_state` cover the layer's own parameters, keyed by qualified parameter name. All tensors are the live training-device tensors — treat them as read-only. `module` is the owning `nn.Module`, or `None` for fx intermediates (`relu`, `add`) and graph inputs.

## nansense.WeightContext

One (layer, parameter) pair's live tensors, passed to weight-tensor instruments once per parameter on publish batches.

`weight` / `gradient` are the live parameter and its `.grad` (`None` before the first backward); `optimizer_state` is this parameter's state entries (empty without an optimizer, or before its lazy init) and `hyperparams` its param group's numeric knobs (`lr`, `momentum`, ...). All tensors are live training-device tensors — treat them as read-only.

## nansense.MetricsSnapshot

Immutable view of every custom-metric series at a point in time.

Keyed by `(layer, phase, metric, series)` — `series` is the mapping key for dict-returning metrics and `""` for plain scalar returns.

### plots

```
plots(
    layer: str, phase: str
) -> dict[str, dict[str, MetricSeries]]
```

`metric -> series name -> series` for one layer and phase.

The per-figure view behind the GRAPHS custom-metric plots, in sorted (metric, series) order so trace order is stable across refreshes.

## nansense.MetricSeries

One plot trace of a custom metric: points ordered by (epoch, batch).

`xs` is the shared epoch-fraction axis: an `on="epoch"` point sits at its integer epoch, an `on="batch"` point at `epoch + batch_idx / n` where `n` spans the epoch's observed batches — so batch series line up under the per-epoch GRAPHS figures. `batches[i]` is `None` for reduced epoch points. Values are stored as-is; render paths gap non-finite ones.

## Time travel

## nansense.TrainingRestorer

Restores the training loop at a cached epoch after a time-travel jump.

The session drives this through two equivalent loop shapes. The flat one (`iter_epochs` + `epoch_guard`, surfaced as `session.epochs()` and `session.restore_point()`) is what hand-written loops use::

```
for epoch in session.epochs(cache_dir=...):
    with session.restore_point():
        ...
```

The nested one (`pending` + the restorer itself as a context manager) is what the Lightning integration transplants around `trainer.fit`::

```
while restorer.pending():
    with restorer:
        for epoch in restorer.epochs():
            ...
```

Both re-enter at the start of an epoch: entering after a jump loads the cached epoch state back into the model / optimizer / scheduler and RNG, rewinds the session's schedule and watch statistics, and sets `start_epoch` to the jump target (`_apply_pending_jump`). The exit suppresses `TimeTravelJump` (and only that), so any other exception still propagates normally.

### start_epoch

```
start_epoch: int
```

First epoch the current attempt should train (0, or a jump target).

### finished

```
finished: bool
```

Whether a `with` block completed without a time-travel jump.

### epochs

```
epochs() -> range
```

Epochs the current attempt should run: `start_epoch` to the end.

### pending

```
pending() -> bool
```

Whether another `with restorer:` attempt should run.

### iter_epochs

```
iter_epochs() -> Iterator[int]
```

Yield epoch indices for the flat `session.epochs()` loop.

Drives the whole run as a generator: it yields `start_epoch … schedule.epochs - 1`, but after a time-travel jump (recorded by the `with session.restore_point():` block's exit) it re-yields the jump target and continues from there instead of advancing. The run is finished once the last epoch's body completes without a jump.

Each body must be wrapped in `with session.restore_point():` — that context manager is what catches the `TimeTravelJump` (a `for` loop never throws its body's exception back into the iterator, so the generator cannot catch it itself), and a missing wrapper is reported rather than left to crash training or loop forever.

### epoch_guard

```
epoch_guard() -> Iterator[TrainingRestorer]
```

Per-epoch context for the flat loop (`session.restore_point()`).

On entry it restores the cached state when the previous epoch's exit armed a jump; on exit it catches `TimeTravelJump` and re-arms it, suppressing the exception so `iter_epochs` re-yields the target. Unlike the restorer's own `__exit__`, completing one epoch does not mark the run finished — `iter_epochs` owns completion.

### save_epoch_start

```
save_epoch_start(epoch: int) -> None
```

Checkpoint the training state at the start of `epoch`.

Called by the session on the first batch of each epoch, before any forward pass — so the file holds exactly the state a jump to this epoch should restore (including any between-epoch user code such as `scheduler.step()` that already ran).

## nansense.TimeTravelStatus

UI-facing view of whether/where time travel can jump.

`available` is False when no restorer wraps the training loop (or the run already completed); `reason` then carries the human-readable explanation for the disabled button's tooltip. `cached_epochs` are the epochs with a loadable checkpoint on disk, restricted to the current schedule's range. `total_epochs` is the run's known epoch count (`0` while a lazy schedule hasn't learned it), bounding the jump target picker.

## nansense.TimeTravelJump

Bases: `BaseException`

Raised by the session inside a batch to unwind to the restorer.

A `BaseException` on purpose: the jump must travel through the user's training code (which may contain broad `except Exception` handlers) all the way to the `with restorer:` block that suppresses it.

## nansense.TimeTravelError

Bases: `RuntimeError`

A time-travel request that cannot be honored (shown in the UI).

## Frozen moments

## nansense.load_moment

```
load_moment(
    model: Module,
    path: Path | str,
    *,
    replay: Callable[[Module, Any], Tensor],
    port: int | None = None,
    host: str = "127.0.0.1",
    open_browser: bool = True,
    input_mean: MeanStd | dict[str, MeanStd] | None = None,
    input_std: MeanStd | dict[str, MeanStd] | None = None,
    input_transform: InputTransform
    | dict[str, InputTransform]
    | None = None,
) -> Session
```

Rebuild a frozen moment around `model` for browsing or a locked demo.

`model` must be a fresh instance of the architecture the moment was frozen from (validated against the file: parameter names/shapes and the discovered layer names must match); its device is kept, and the frozen parameters and buffers are loaded into it. `replay` is the training step's forward in miniature — given the model and the stored batch item it returns the loss, e.g. `lambda m, batch: criterion(m(batch[0]), batch[1])` — and is run once, here, to regenerate the frozen batch's activations and gradients (see the module docstring for why they are not stored). The returned session shows exactly the frozen pause — snapshot, watch statistics, watched set, and schedule totals — with stats collection off (scope `"none"`), so the numbers sit frozen while every view, and experiments, keep working. Nothing trains: the session is a viewer. Raises `MomentError` when the file is unreadable or does not fit `model`.

`port` / `host` / `open_browser` / `input_*` mirror `nansense.start` and serve the UI immediately when `port` is given. For a shared deployment, follow with `session.lock()` and `session.park()` — see `examples/playground` and `Session.freeze_moment` for the saving side.

## nansense.MomentError

Bases: `RuntimeError`

A moment file could not be read or does not fit the given model.

## Numerical debugging

## nansense.DebugSettings

User-facing configuration of the debugger (mutated via the gear menu).

`interval` is the batch cadence (1 = every batch). `check_nan_inf` scans activations and gradients for NaN/Inf; `check_under_over` watches for gradient underflow/overflow, tripping when more than `threshold` (a fraction of summed `|grad|`) lands in the dtype's subnormal/overflow band. The two checks toggle independently; `enabled` is the master switch.

### any_check

```
any_check() -> bool
```

Whether at least one check would run (master + one sub-toggle).

## nansense.DebugError

An immutable record of one detected numerical error.

`position` is the batch the error was detected on. `reasons` is the subset of `REASONS` that tripped; `checks_used` is the subset of categories that actually ran (so the UI shows only the relevant table columns even for an error that tripped just one of them); `layers` are the affected layers, in `layer_names` order.

## nansense.LayerReport

Per-layer fractions for one detected error (a row in the dialog table).

`layer` names the affected layer, as shown in the architecture graph. `nan` / `inf` are fractions of *element count*; `underflow` / `overflow` are fractions of the layer's summed `|grad|`. All in `[0, 1]`. `dtype` is the gradient dtype the under/overflow band was measured against (its `finfo.tiny` / `finfo.max` are the band edges), or `None` when no floating-point gradient was scanned for the layer.
