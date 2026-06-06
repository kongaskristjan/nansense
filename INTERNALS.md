# Playgrad internals

This document explains how the `playgrad` library is structured under the
hood. For *using* the library, see `README.md`. For agent-facing guidelines,
see `AGENTS.md`.

## Threading model

A playgrad session lives across two threads:

- **Training thread.** The user's training loop. Forward / backward / step
  run here, and `with session.batch(phase=..., epoch=...)` is entered here.
- **UI thread.** Driven by NiceGUI (not yet implemented). Reads session
  state, calls control methods (`stop`, `step_batch`, …, `detach`, `close`).

Synchronization is a single `threading.Condition` (`Session._cv`) protecting:

- `_mode` — current `Mode` enum value.
- `_resume_token` — monotonic counter; bumped by every "go" command.
- `_pause_count` — monotonic counter; bumped each time the training thread
  enters `_wait_for_proceed`.
- `_closed` — flips once on `close()`.
- `_schedule` — mutated by `set_schedule()` and time-travel rewinds.
- `_pending_jump` — armed by `request_time_travel()`, consumed at batch
  boundaries (with a lock-free `is None` fast path on the consuming side).

`_snapshot`, `_activations`, and `_hook_handles` are written only by the
training thread and read by the UI thread; reads are point-in-time and
don't need a lock because Python attribute assignment is atomic under the
GIL.

## Schedule

A `Schedule` is constructed once at `playgrad.start(model, epochs=...,
phases=...)`. The `phases` dict is order-preserving (the last key in
insertion order is treated as the final phase of each epoch).

`Schedule.advance(phase, epoch)` is called inside `_BatchContext.__enter__`
and returns a `BatchPosition` with:

- `batch_idx` (0-based within `(phase, epoch)`)
- `is_last_in_phase`
- `is_last_in_epoch`
- `is_last_overall`

Because the run length is declared up-front, these flags are *predictive*:
we know on a batch's `__enter__` whether it is a boundary, before any
forward pass runs. That's what lets the session decide whether to install
hooks before the forward pass — there is no reactive "phase just changed"
detection at `__exit__`.

For non-deterministic workloads, `Session.set_schedule()` re-declares
`epochs` / `phases` mid-run.

## Modes and capture decisions

| Mode | Public method | Captures + pauses at |
|---|---|---|
| `STEP` | `step_batch()` (also `stop()`, no resume) | every batch |
| `UNTIL_PHASE_CHANGE` | `step_phase()` | `is_last_in_phase` |
| `UNTIL_EPOCH_CHANGE` | `step_epoch()` | `is_last_in_epoch` |
| `UNTIL_END` | `step_run()` | `is_last_overall` |
| `UNTIL_POSITION` | `step_until_position(phase, epoch, batch_idx)` | exactly that `(phase, epoch, batch_idx)` |
| `DETACH` | `detach()` | never |

A session starts in `STEP` mode — the first batch always pauses so the UI
can show its initial state.

`_should_capture(pos)` is the single decision function for both "install
hooks?" and "pause after this batch?". Capture and pause are intentionally
the same predicate — there is no implicit pause that the user did not ask
for, and there is no orphan capture without a pause to consume it.

## Hook lifecycle, gradient pickup, snapshot copy

There are two capture paths, picked once at session construction by
trying `torch.fx.symbolic_trace(model)`.

**fx path (preferred).** When the trace succeeds, the session holds the
resulting `fx.GraphModule` and `_install_hooks` monkey-patches
`model.forward` with a function that runs a custom `fx.Interpreter`
subclass against that graph. The interpreter overrides `run_node` so that
after every node executes — placeholders, `call_module`, `call_function`,
`call_method` — it stores the returned tensor in `Session._activations`
under a friendly key built by `playgrad.fx_names.friendly_names`. Module
calls use the dotted target (`stage1.0.bn1`); function/method ops are
prefixed with the innermost submodule scope fx records in
`node.meta["nn_module_stack"]`, so `torch.relu(...)` inside a block becomes
`stage1.0.relu1` (numbered only when a scope holds more than one op of that
name) rather than the uninformative `relu`, `relu_1`, ... fx auto-names.
Root-scope ops keep a bare name (`relu`, `flatten`), and inputs stay `x`.
Each captured tensor gets `retain_grad()` so the user's `loss.backward()`
populates `.grad`. This is what lets `torch.relu(...)`, `out + shortcut(x)`,
and similar non-module operations show up in the UI on equal footing with
named modules — and the Mermaid graph builder reuses the same map, so each
node's id and label line up with its layer card. `_remove_hooks` restores
the original `forward`.

**Hook fallback.** When `fx.symbolic_trace` raises (data-dependent
control flow, tracing-unfriendly ops, etc.), the session falls back to:

1. A forward **pre-hook** on the root model. It captures each positional
   tensor input under its parameter name (derived from
   `inspect.signature(model.forward)`), e.g. `x` for a model whose
   forward signature is `forward(self, x)`.
2. A forward hook on every submodule (skipping the root model itself).
   The hook stores `output` in `Session._activations` as a live
   reference *and* calls `output.retain_grad()` on it (when
   `requires_grad`) so that PyTorch populates `output.grad` after
   `loss.backward()`. No backward hook is needed.

In both paths `Session.layer_names` is computed once at construction and
exposes the same key set the UI later reads from each `BatchSnapshot`.

`Session.layer_weights` is computed alongside it — a `layer name ->
parameter names` map keyed identically to `layer_names`, with an empty
list for weightless layers (graph inputs, `relu`, `add`, …). Its values
are qualified parameter names that index straight into a snapshot's
`weights` / `weight_gradients` dicts. In fx mode the mapping is exact: a
`call_module` node owns every parameter under its dotted target, and any
node that uses a parameter functionally (e.g. `F.conv2d(x, self.weight)`)
picks it up through its `get_attr` input nodes — so on a traced model a
parameter maps to nothing only if the forward pass genuinely never uses
it. In the hook fallback a module maps to every parameter in its subtree
(prefix match on the dotted name), the set of weights that contributed to
the output the forward hook captured. For ResNet-20 every one of the 61
parameters is covered by a `call_module` node.

At `__exit__`:

1. `_remove_hooks()` removes every registered hook.
2. If the batch ran without an exception and the session isn't closed,
   `_publish_snapshot()` builds a `BatchSnapshot` containing **CPU clones**
   of four tensor dicts:
   - `activations`: hook-stored module outputs, cloned to CPU.
   - `activation_gradients`: `activation.grad` for each captured output
     that has one, cloned to CPU.
   - `weights`: every `param` from `named_parameters()`, cloned to CPU.
   - `weight_gradients`: `param.grad` where non-`None`, cloned to CPU.
   When the session was given an optimizer at `start()`, two more fields
   are filled in (otherwise they default to empty dicts):
   - `optimizer_state`: per-parameter optimizer variables, resolved by
     identity — `optimizer.state` is keyed by the parameter object, so
     `{id(param): name}` from `named_parameters()` maps every entry back
     to its qualified name with no per-optimizer code (SGD's
     `momentum_buffer`, Adam's `step`/`exp_avg`/`exp_avg_sq`, custom
     optimizers alike). Tensor entries are CPU-cloned; plain int/float
     entries become 0-dim tensors. Empty until the first
     `optimizer.step()` — state is lazily initialised.
   - `optimizer_hyperparams`: each parameter's group's plain-numeric
     knobs (`lr`, `momentum`, `weight_decay`, …) as floats, captured at
     the same instant — so a scheduler-mutated `lr` is the batch's
     actual one. Available from batch 0 (groups are not lazy).
   Every clone goes through `tensor.detach().to("cpu", copy=True)`, so the
   snapshot is fully independent of the live computation graph — the next
   batch can free / overwrite all of its source tensors without affecting
   the snapshot.
3. The training thread calls `_wait_for_proceed()` (described below).
4. After resume, `_activations` is cleared so the next batch starts clean.

Weight gradients are read straight off `param.grad` rather than via
backward hooks. This works because the user's training loop calls
`optimizer.zero_grad()` at the *start* of their batch body, before
`loss.backward()`. By `__exit__`, the gradients from this batch are still
on the parameters; `optimizer.step()` does not touch `.grad`.

Memory profile: the eager CPU clone costs O(activations + parameters)
bytes of host memory per captured batch. For small models (ResNet-20)
this is tens of MB; for large models, a `watch=` filter to opt out of
some modules is the future escape hatch. In exchange, the snapshot is
thread-safe to read from the UI without holding any session lock and
survives arbitrarily long after the training thread has moved on.

## Resume mechanism

`_wait_for_proceed()` uses a token-counter pattern instead of an
`Event`:

```python
def _wait_for_proceed(self) -> None:
    with self._cv:
        seen = self._resume_token
        self._pause_count += 1
        self._cv.notify_all()
        while self._resume_token == seen and not self._closed:
            self._cv.wait()
```

Each `step_*` / `detach()` call bumps `_resume_token` under the lock and
`notify_all`s. The waiting thread loops until the token has advanced past
the value it captured on entry.

Why not a plain `Event`? `Event.set()` is idempotent: if the UI sends two
rapid commands (e.g. `step_batch()` followed immediately by `detach()`)
while the worker is in flight, only one set survives `wait/clear`, so the
next pause would deadlock. The token pattern is robust to coalesced
commands — each pause only requires *any* resume command issued after the
pause began.

`_pause_count` is the symmetric counter for the UI side:
`wait_until_paused(after_pauses=N, timeout=...)` blocks until the worker
has paused more than `N` times. Tests and the UI use this to synchronize
without polling.

## Snapshot lifecycle

`Session.snapshot` is the last `BatchSnapshot` published, or `None` if no
batch has been captured yet. It persists after `close()`, so the UI can
stay open and present a post-mortem view.

The snapshot is a frozen dataclass of CPU tensors. Assignment is a single
attribute write; readers in the UI thread observe either the previous
snapshot or the new one — never a torn half-written state. The UI can
hold references to a snapshot for as long as it wants without preventing
the next batch from running.

Rendering (image strips, histograms, summary stats) happens on the UI
thread against the published snapshot; the eager copy in
`_publish_snapshot` only moves raw tensor data, not anything pixel-shaped.
A per-snapshot render cache in the UI (`_RenderCache`) keeps repeat
renders free.

## Watch accumulators

Independently of the snapshot path, the session can collect running
statistics for any subset of layer names — driven by the eye-icon
toggle on the main page, surfaced on the `/watch` deep-dive page.

`Session.watch(name)` / `unwatch(name)` mutate the `_watched_layers`
set under `_cv`. The `Session.watched_layers` snapshot is a
`frozenset`, safe to read from the UI thread. `watch()` accepts any
name in `session.layer_names`: named modules, fx-traced intermediates
(`relu`, `add`, `mean`), and graph inputs (`x`).

`_BatchContext` extends the capture machinery to cover watching too.
If any layer is watched but the batch is *not* a capture batch
(`detach`, mid-`step_run`, etc.), `_install_hooks()` still runs — the
same fx interpreter or pre-hook+per-module-hook installation that the
snapshot path uses — so every name in `layer_names` lands in
`_activations` with `retain_grad()` applied. The only difference
between capture and stats-only batches is that capture also publishes
a snapshot and pauses; stats-only batches just compute stats and let
the training loop continue.

This is a deliberate trade: when watching is active, the user is
already paying capture-mode forward cost on every batch (fx
interpretation Python overhead, doubled activation memory from
`retain_grad`). That's the price of an accurate visualisation;
production runs should not enable the UI in the first place. The
benefit is that any node in the graph is reachable for stats without
a separate code path.

At `__exit__`, before snapshot publishing, `_update_watch_stats(pos)`
walks `_watched_layers` and feeds each captured tensor's activation
and gradient into `WatchAccumulator.update(layer, phase, epoch, kind, x)`.
The accumulator is keyed by `(layer, phase, epoch)` so each
epoch/phase gets its own bucket and history accumulates rather than
overwriting. Unwatching a layer drops every key for it via
`forget_layer(name)`.

Inside `TensorAccumulator.update(x)`:

1. `x.detach().to(torch.float32).reshape(-1)` — the fp32 cast is the
   bf16/fp16-safety knob. bf16 sum-of-squares saturates after a few
   hundred unit-magnitude samples; fp32 keeps the running sum precise
   for typical epoch sizes.
2. Reductions stay on the input's device: `sum()`, `square().sum()`,
   `min()`, `max()`, plus a `torch.bincount(_bin_indices(x))` over the
   211-bin signed-log histogram.
3. All running state — `_n`, `_sum`, `_sum_sq`, `_min`, `_max`,
   `_hist` — lives on that same device. No GPU→CPU sync happens
   during training.

Histogram bin assignment (`_bin_indices`) is a vectorised log10:

- `|x| < 1e-9` → the zero band (bin 105).
- Otherwise `floor((log10|x| - LOG10_MIN) * BINS_PER_DECADE)` gives the
  per-sign offset, clamped to `[0, N_POS-1]` so values beyond `±1e6`
  saturate into the two end bins (which the UI marks as overflow).
- `torch.where(x >= 0, ZERO_BIN+1+pos, ZERO_BIN-1-pos)` packs both
  signs into the 211-bin space with the zero band in the middle.

`WatchAccumulator.snapshot(layers=...)` is the UI-thread reader. It
holds the accumulator lock only briefly to copy the dict of stat
references, then computes each `TensorStatsSnapshot` outside the lock
— that's the one GPU→CPU sync per call, batched into one `torch.stack`
for the scalars plus a single `.cpu()` for the histogram. The result
is a frozen dataclass tree (`WatchSnapshot → LayerStatsSnapshot →
TensorStatsSnapshot`) that the UI can render without holding any
session state.

The watch path adds zero overhead when nothing is watched. With at
least one layer watched, every batch pays capture-mode cost — that's
the cost of exposing fx intermediates and inputs to the stats
collector without a parallel implementation. Snapshot timeline and
pause behaviour are unaffected on non-watching sessions.

### Extreme input patches (`playgrad.patches`)

Alongside the histogram stats, each watch bucket owns a
`PatchAccumulator` that keeps, per activation channel, the
`N_PER_CHANNEL` input samples producing the most extreme activations
under four rankings:

- `max_pixel` / `min_pixel` — the channel's single largest/smallest
  spatial value. The stored patch is an input crop around that pixel's
  ratio-mapped location (side ≈ `PATCH_FACTOR ×` the activation→input
  downsampling ratio, floored at `MIN_PATCH` — an approximation of the
  receptive field, not an exact computation).
- `max_average` / `min_average` — the channel's spatial mean; the
  stored patch is the whole input image (there is no single location).

Each entry also stores the channel's full activation map so the UI can
blend a heatmap over the patch. 2D `(B, F)` activations degrade
gracefully: features act as channels, pixel ≡ average, whole-image
patches, single-cell heat.

`Session._update_watch_stats` feeds the accumulator via
`WatchAccumulator.update_patches(layer, phase, epoch, act, x)`, where
`x` is `_patch_source_input()` — the first image-like (4D, 1- or
3-channel) forward input captured by the pre-hook. Non-image models
simply gather no patches.

Like `TensorAccumulator`, everything stays on the training device with
no syncs and no data-dependent branching: per batch and per type, one
reduction over the activation produces per-sample-per-channel scores
`(B, C)`, a per-channel `topk` over the batch axis picks
`min(N_PER_CHANNEL, B)` candidates, one vectorised fancy-index gathers
their patches, and a `cat`+`topk` merge folds them into the `(C, N)`
running buffers. Ranking per channel over the batch axis doubles as
deduplication — a sample appears in one batch per epoch, so a channel
row can never hold the same image twice. NaN scores are demoted to the
placeholder `∓inf` so diverged batches never enter the buffers.

One memory caveat drives the eviction rule in `update_patches`:
histogram buckets are ~2 KB and live forever, but a patch bucket holds
`4 × C × N` image crops on the GPU. When a newer epoch starts for the
same `(layer, phase)`, older epochs' patch buffers are released — the
`/watch` page only renders the latest epoch per phase, so nothing
visible is lost. `forget_layer` / `forget_epochs_from` (unwatch, time
travel) drop patches together with the rest of the bucket.

`PatchAccumulator.snapshot()` copies the buffers to CPU as a frozen
`PatchSnapshot → TypePatches` tree carried on
`LayerStatsSnapshot.patches`; slots never filled keep their `∓inf`
values and are masked by the renderer.

## Probe runs (`playgrad.probe`)

A probe is a playgrad-internal forward pass on a *pinned* input batch, run
between batches so the UI can show the network's response to one fixed
input across stepping and time travel. `Session.pin_current_batch()` pins
the last snapshot's input tensor (already a CPU clone); from then on every
capture re-runs the model on it right after `_publish_snapshot` and
publishes a `ProbeResult` — CPU clones of the input and of every layer
output, keyed like `layer_names`. Probes are forward-only: no gradients.

**Execution stays on the training thread.** The model is never touched from
the UI thread (the invariant the snapshot path already relies on). Probes
run at two points:

1. `_BatchContext.__exit__`, between `_publish_snapshot` and
   `_wait_for_proceed` — so every pause shows a probe consistent with the
   just-captured weights.
2. Inside `_wait_for_proceed`'s wait loop. UI requests (`pin_current_batch`,
   `set_probe_mode`) arm `_probe_request` under `_cv` and notify; the paused
   training thread wakes, runs the probe *outside* the lock (so UI reads
   stay responsive), and re-enters the wait — the same "armed request
   consumed at a safe point" pattern as `_pending_jump`, generalized to
   running work. A request arriving mid-`detach` stays armed until the next
   capture.

**Isolation contract** (`_probe_forwards`): probes never mutate training
state. Per-module `training` flags are saved and restored ("eval" / "train"
probe modes flip the whole model; "unchanged" runs as-is); every buffer is
restored afterwards (a train-mode BatchNorm forward updates running stats
in place); the RNG is forked (`torch.random.fork_rng`, CUDA/MPS-aware) so
e.g. train-mode dropout doesn't perturb the global stream that time-travel
replays depend on; and the whole run sits under `torch.no_grad()`.

**Capture reuse without interference** (`_capture_forward`): in fx mode the
probe runs `_CaptureInterpreter` against a fresh local dict — the original
`model.forward` is never patched, since the interpreter is invoked
directly. In the hook fallback, temporary pre/forward hooks write into the
local dict and are removed in a `finally`. Neither path touches
`_activations` or `_hook_handles`; both are safe because probes only run
between batches, when the batch path's hooks are uninstalled.

**Perturbations.** `Session.add_perturbation(sample=, y=, x=, values=)`
records per-pixel edits (`(sample, y, x) -> per-channel values` in
model-input space; the UI back-transforms the picked display color via
`input_panel.normalized_color`). When any exist, `apply_perturbations`
clones the base input, writes the in-range entries (out-of-range or
channel-mismatched ones are skipped — the base may have changed shape since
the click), and the probe runs a *second* forward on the copy inside the
same isolation scope. `ProbeResult` then carries `perturbed_input` /
`perturbed_activations` next to the base pair, and the UI renders either
the perturbed activations or the per-layer diff ("Compare with original").
Perturbations alone keep probing active without a pin — the base falls
back to the snapshot's input (`_snapshot_input`), so edits track the
current training batch.

**Publishing and races.** Probe config (pinned input, mode, perturbations)
is mutated by the UI thread under `_cv`, bumping `_probe_version`.
`_run_probe` snapshots the config under the lock, computes without it, then
publishes under the lock only if the version is unchanged — a config change
mid-run wins and its own request re-runs the probe. `_probe_count` is the
monotonic completion counter (`wait_for_probe` mirrors `wait_until_paused`
for tests and the UI). A probe that raises publishes `probe_error` instead
of killing the training thread (`_run_probe_guarded`); deactivating the
probe (`unpin_batch` / `clear_perturbations` with nothing else active)
clears the published result so the UI falls back to the snapshot.

## Experiments (`playgrad.experiments`)

Experiments — deep dream and a small Captum selection (Grad-CAM, Neuron
Gradient, Neuron Integrated Gradients, Occlusion) — are the long-running,
cancellable extension of the probe job queue. Each client (browser tab)
queues an `ExperimentRequest` (`Session.request_experiment`, returning a
seq); the pause loop in `_wait_for_proceed` drains the queue in order and
calls `experiments.run(...)`, a generator yielding `ExperimentResult`
progress snapshots that the session publishes one by one
(`_publish_experiment`) — that's what streams the evolving deep-dream
image to the page. Results are kept per seq in a bounded map (the
`_EXPERIMENT_RESULTS_KEPT` most recently updated seqs;
`experiment_result_for(seq)`) plus a latest-result slot
(`experiment_result`), so concurrent tabs each poll their own run without
overwriting each other.

**Cancellation.** The runner checks a `should_abort()` predicate between
steps; it fires on `cancel_experiment(seq)` for the running seq (no seq
cancels everything; a queued seq is just dropped) and on anything that
ends the pause — resume commands, a pending time-travel jump, `close()` —
so the pause loop regains control within one step. Another client's new
request does *not* abort a running experiment; it waits its turn in the
queue. Requests queued while training is running stay queued until the
next pause (`experiment_pending` lets the UI say so). A raising experiment
publishes an error result instead of killing the training thread.

**Isolation.** Experiments share the probe contract via
`Session._isolated_model` (the refactored common core): eval-mode forwards
with per-module flags and all buffers restored, forked RNG. Gradients are
taken w.r.t. the *input* only (`torch.autograd.grad`), so parameter
`.grad` — which the snapshot path reads at `__exit__` — is never touched.

**Deep dream** (`_run_deep_dream`) is per-sample-normalized gradient
ascent on a channel's mean activation (`_target_activation`: the fx
interpreter against a local dict, so any name in `layer_names` is a valid
target; a single temporary hook in the fallback). It runs on a *batch*
built from the network's real input (`_dream_start` — any input shape, not
just images): by default `batch` fresh noise samples matching the real
input's per-sample shape and overall mean/std, drawn from a generator
seeded by the request seq so successive Runs explore different noise;
`start="sample"` takes the first `batch` samples of the real input batch
instead. Regularizers per step, applied only to `[B, C, H, W]` inputs:
jitter (random roll, undone after the update — drawn from the same
request-seeded generator), diffusion (blend with a 3×3 box blur), center
zoom (a per-step multiplier, ≥ 1), and clamping to `_value_bounds` — the
displayable `[0, 1]` range mapped through the input mean/std.

**Captum** (`_run_captum`) imports lazily and runs the *unpatched* model
inside the isolation scope (experiments only execute between batches).
Layer-targeted methods require an `nn.Module`; fx intermediates are
rejected with a pointer to the producing module. Output-targeted methods
(Grad-CAM, Occlusion) resolve target class −1 to the model's argmax and
require a `[batch, classes]` output. Neuron methods address a channel via
a selector callable (per-example mean of that channel).

**UI** (`/experiment?layer=...`, one yellow "Experiment" button per layer
card): the kind dropdown defaults to deep dream; each kind's knobs are
declared in `_EXPERIMENT_PARAMS` (`_ExperimentParam` specs rendered as
number/switch/select widgets and collected on Run; the deep-dream "Inputs"
count defaults to the live `session.input_batch_size`, capped at
`_DEFAULT_DREAM_BATCH` = 8, and the channel knob is bounded by the layer's
captured channel count, `_layer_channel_count`). A 200 ms timer streams
the page's *own* request via `session.experiment_result_for(seq)` — Run
replaces and Cancel aborts only this page's request, so tabs don't clobber
each other: deep-dream result and start batches render denormalized via
`render_image` as wrapping grids (non-image inputs fall back to a "not
renderable" note), attributions via the shared diverging-colormap
`render_strip`, next to the input sample.

## Time travel (`playgrad.restore`)

Time travel jumps training back to the start of any epoch whose state was
checkpointed to disk. It is opt-in at the training-loop level: the user
creates a `TrainingRestorer` (`session.training_restorer(cache_dir=...)`,
default `models/latest/`) and wraps the epoch loop in it:

```python
while restorer.pending():
    with restorer:
        for epoch in restorer.epochs():   # range(start_epoch, schedule.epochs)
            ...
```

Without a restorer, nothing is written to disk, the session never raises a
jump, and the UI's Time Travel button is disabled (its tooltip explains
why). On a disabled session the restorer is inert.

**Epoch cache.** With a restorer attached, `_BatchContext.__enter__` on the
first batch of each epoch (batch 0 of the schedule's first phase) writes
`epoch_<n>.pt` into the cache directory before any forward pass:
`model.state_dict()`, `optimizer.state_dict()`, `scheduler.state_dict()`
(when passed to `start()`), and the torch/CUDA RNG states. Writes go
through a temp file + atomic rename; an existing file for the same epoch is
overwritten when training passes it again, so after a jump the older
timeline's entries persist until the re-run reaches them. Restoring the
global RNG state is what makes the replay deterministic — the DataLoader
draws its shuffle seed from the global generator at `iter()` time.

**Jump flow.** `Session.request_time_travel(epoch)` runs on the UI thread
and validates everything up-front: the restorer exists and isn't finished,
the epoch is in range, the checkpoint loads, and its model state matches
the live model's parameter names and shapes (`validate_model_state`). Any
failure raises `TimeTravelError` with a displayable message and nothing
unwinds — this is what catches a cache directory left behind by a previous
run of a different model. On success the request arms `_pending_jump`
under `_cv`, switches the mode to `STEP`, and bumps the resume token.

The training thread consumes the jump at batch boundaries by raising
`TimeTravelJump(epoch)` — a `BaseException` subclass, so a user's broad
`except Exception` around the batch body can't swallow it. There are two
consumption points: `_BatchContext.__enter__` (right after the schedule
advance, covering requests that arrive mid-run between batches) and
`__exit__` after `_wait_for_proceed` (covering the common paused case).
`_wait_for_proceed`'s predicate also wakes on a pending jump: the request
already bumped the token, so a pause that began after the request would
otherwise wait for a second UI command.

The exception unwinds through the user's loaders and loops to the
`with restorer:` block, whose `__exit__` suppresses exactly this type and
records the target. The next `__enter__` (training thread, between
attempts, so nothing races a forward pass) loads the checkpoint back into
the model / optimizer / scheduler / RNG and calls
`Session._rewind_to_epoch(epoch)`, which drops the schedule's batch
counters for `epoch` onward (`Schedule.rewind_to_epoch`) and the watch
accumulators' buckets for those epochs (`forget_epochs_from` — they're
additive, so the re-run samples must start from empty ones). Because the
mode was set to `STEP`, the first batch of the target epoch captures and
pauses for inspection — the same behaviour as session start.

`restorer.pending()` returns `False` once a `with` block completes without
a jump; it also raises if called twice without the block ever being
entered, catching a `while` loop that forgot the `with` (which would
otherwise re-run training forever). Loop state that depends on history
(`best_acc`, metric curves) belongs inside the `with` block, where a jump
naturally resets it.

**UI.** The blue Time Travel button (right of Detach, built by
`_add_time_travel_button`) opens a dialog whose content is rebuilt on
every open from `session.time_travel_status()`: a slider over the cached
epochs within the current schedule (it runs over indices into the
cached-epoch list, so uncached epochs are unselectable even when the set
has gaps, with a label showing the mapped epoch number), text stating the
cached and uncached epoch ranges, and — while any are missing — a "Cache
full training run" button that simply arms `step_run()` (running to the
end checkpoints every epoch start along the way). A rejected request (`TimeTravelError`) opens a
separate error dialog and no jump happens. When the session reports time
travel unavailable at page build (no restorer), the button is rendered
disabled with the reason as a tooltip on a wrapper div — Quasar suppresses
pointer events on disabled buttons, so the tooltip can't live on the
button itself.

## UI layer

`playgrad.ui` is a thin NiceGUI app that reads `Session.snapshot` and
drives `Session` via the five control methods plus `detach` and `close`.
It does not touch tensors directly until they need to be rendered.

- `playgrad.ui.graph.build_mermaid(model)` produces the Mermaid TD source
  for the architecture view. It tries `torch.fx.symbolic_trace(model)`
  first, which yields a real data-flow graph — vertical chains, with
  branches and merges at residual blocks. For models that aren't
  fx-traceable (dynamic control flow, custom ops), it falls back to a
  static module-hierarchy tree rooted at a synthetic `root` node. Nodes
  use different Mermaid shapes per fx op (rectangles for `call_module`,
  ovals/stadiums for `call_function` / `call_method`, circles for
  `placeholder` / `output`).
- `playgrad.ui.render.render_strip(tensor, sample_idx)` turns per-layer
  CPU tensors into a `StripRender`: one data image at the tensor's
  *native* resolution plus a legend image at display resolution.
  - For per-sample shape `[C, H, W]` every channel tile lands in a single
    image, downsampled server-side (`area`) only when larger than
    `TILE_SIZE × TILE_SIZE`, with white separators between tiles
    `max(1, tile_width // TILE_GAP_DIVISOR)` native pixels wide. The
    browser upscales the whole strip to `StripRender.width × height` via
    CSS sizing plus `image-rendering: pixelated` — equivalent to the old
    server-side nearest-neighbour interpolation, but an 8×8 feature map
    travels as 64 pixels instead of 16k; the separators scale together
    with the tiles (`_strip_html` in `app.py` builds the two-`<img>` row).
  - For `[F]` the data image is a single 1-px-tall heatmap row, downsampled
    to at most `LINEAR_MAX_BINS` bins when `F` is large, stretched
    client-side to `LINEAR_BIN_WIDTH` per bin × `LINEAR_TILE_HEIGHT`.
  - The legend (vertical colorbar with `+x` / `0` / `-x` labels) is the
    exception to native-resolution encoding: it is rendered at display
    resolution into its own image so its text stays crisp.
  - Every strip — activations, gradients, and weights alike — uses the
    same diverging blue-white-red colormap; strips are told apart by a
    labelled colored marker bar on each one's left edge (emerald
    ACTIVATIONS / violet GRADIENTS on the layer cards, sky WEIGHT /
    violet GRADIENT on the weights page). Images are encoded per
    `STRIP_FORMAT`: the default `BMP` is essentially a memcpy (30–60×
    faster to encode than PNG) at ~2× the payload — the right trade for a
    localhost WebSocket. Flip to `PNG` (`compress_level=1`) when bytes
    matter more than encode time, e.g. an SSH-port-forwarded UI.
  - Other per-sample shapes return `None`; the UI hides those images.
- `playgrad.ui.render.render_image(tensor, sample_idx, mean=..., std=...)`
  renders the model input as a natural RGB or grayscale image at the
  sample's native resolution; the UI scales it to `INPUT_IMAGE_SIZE` with
  CSS nearest-neighbour. Channels are assumed to lie
  in `[0, 1]` unless both `mean` and `std` are passed, in which case the
  sample is denormalized (`x * std + mean`) before being clamped and
  scaled to 8-bit. Anything other than `C == 1` or `C == 3` returns
  `None`.
- `playgrad.ui.render.render_weight(tensor, x_dim=, y_dim=, tile_dim=,
  fixed=)` renders an arbitrary-rank weight under a chosen axis layout.
  Unlike `render_strip`, a weight has no batch axis, so instead of slicing
  a sample it pins every axis not assigned to X/Y/tile to a single index
  (`fixed`, clamped into range), permutes the survivors into
  `[tile, y, x]`, then funnels through the same `_render_chw` /
  `_render_1d` tile machinery and shared diverging colormap.
  `default_weight_dims(ndim)` gives the default assignment —
  last axis X, second-to-last Y, third-to-last the tile axis, the rest
  fixed — which renders 4D conv weights as kernels, 2D as one image, 1D as
  one row. Duplicate or out-of-range axes return `None`.
- The top-bar control row is shared via `_top_bar_row()` (the row
  container) and `_add_step_controls(session, dialog)` (the five stepping
  buttons + a live-position label, returned for the page's timer to
  refresh through `_format_live_position`). The main page, `/watch`, and
  `/weights` all build their bars from these, differing only in the
  leading/trailing widgets they add around the shared controls.
- `playgrad.ui.app.serve(session, port=..., host=...)` runs the NiceGUI
  app on a background thread. NiceGUI is mounted onto a bare FastAPI
  app via `ui.run_with`, which is then served by `uvicorn.Server` from
  the thread. `install_signal_handlers` is patched to a no-op because
  uvicorn would otherwise try to register SIGINT/SIGTERM handlers from
  a non-main thread. The thread is non-daemon, so the UI stays alive
  even after the training script's main thread returns — the user
  closes the browser / Ctrl-Cs when they're done browsing post-mortem.
- The page handler creates one `_LayerView` per submodule (a card with
  two strips inside a shared horizontal scroll container, each flanked by
  a sticky marker bar with a vertical label — emerald ACTIVATIONS, violet
  GRADIENTS) and a `ui.timer` that, every 200 ms, compares
  `session.snapshot` and `session.probe_result` against the last rendered
  pair by identity. When either changed, every layer view re-renders,
  slicing each tensor at the current `sample_idx`. A probe result (pinned
  batch) takes precedence over the snapshot as the render source; its
  gradient strips show a placeholder note (probes are forward-only). The
  `_RenderCache` is keyed by render-source identity — snapshot or probe
  result — so both share one cache.
- The right sidebar is `playgrad.ui.input_panel.InputPanel`, laid out
  top-down as: the input image (what every control below acts on), the
  "Viewing sample" `ui.number` (moved out of the top bar), a "Probe"
  section — the "Pin batch" switch, the probe-mode toggle (unchanged /
  eval / train, shown only while pinned via `bind_visibility_from`,
  `spread` so it fills the pane width) and a pinned-position caption —
  and a "Perturb" section: the "Click to perturb" switch with a compact
  color-swatch button beside it (the button's background *is* the current
  color; clicking opens a nested `ui.color_picker`, forced to
  `format-model=hex` since `normalized_color` expects `#rrggbb`), the
  "Compare with original" switch, and a perturbed-pixel count with a
  "Clear" button on its right. Compare is visible whenever "Click to
  perturb" is on (`bind_visibility_from` the perturb switch) — with
  nothing perturbed its diff is identically zero and renders as white
  strips, in both the probe and the plain-snapshot paths — while the
  count/clear row shows only when perturbations exist (`refresh_status`
  keeps it in sync). Switching "Click to perturb" off clears all
  perturbations and resets compare — leaving editing mode discards the
  edits and the diff view.
  Pin / mode / perturbation changes call straight
  into the session; the session reacts by publishing a new `ProbeResult`,
  which the tick loop picks up like a new snapshot. The compare switch is
  pure view state (`panel.compare`, threaded into `_compute_frame`) — both
  forwards are already in the result, so flipping it just re-renders. A
  failed pin (no snapshot yet) reverts the switch with the usual
  one-tick-deferred value write.
- The input image is a `ui.interactive_image` sized by CSS to
  `INPUT_IMAGE_SIZE` with `image-rendering: pixelated` (the per-frame
  payload is the native-resolution data URI from `_input_img_src`).
  NiceGUI delivers click coordinates in the image's *native* pixel space,
  so the handler clamps `image_x`/`image_y` into the input's `H × W`,
  converts the picked color with `normalized_color` (grayscale inputs use
  the RGB mean; `mean`/`std` back-transform into normalized space), and
  calls `session.add_perturbation` for the viewed sample. The "Click to
  perturb" switch only gates clicking (and toggles a crosshair cursor);
  recorded edits persist until cleared.
- The same timer also refreshes the top-bar position label from
  `session.live_position` — the position recorded on *every* batch's
  `__enter__`, independent of capture. This is what keeps the displayed
  epoch/batch advancing during `step_epoch`, `step_until_position`,
  `step_run`, and `detach`, where `snapshot.position` would otherwise stay
  frozen until the next boundary capture (or never, under `detach`). It is
  a single label write per tick, decoupled from the strip rendering; the
  200 ms timer is the natural throttle for the rapid batch advances those
  modes produce.
- Rendering is intentionally eager when a new snapshot lands — and during
  `RUN` / `DETACH` modes no snapshots are produced so the UI is idle.
  Per-layer renders are independent, so a frame fans out over a shared
  `ThreadPoolExecutor` (`_RENDER_POOL`); the heavy parts (torch
  interpolate, numpy colormap, PIL image encode) release the GIL, so
  layers render in parallel across cores. Rendered strip HTML is cached
  in a `_RenderCache` shared by every
  connection: entries are keyed `(name, kind, sample_idx)` and the whole
  cache is invalidated by snapshot identity (snapshots are frozen; every
  pause publishes a new object). Flipping the sample spinner back to a
  value already seen, or a second browser tab on the same session, is a
  dict lookup instead of a re-render. For larger models, viewport-aware
  lazy rendering is the natural next step, but the current code path
  keeps the wiring simple.
- The `/watch` page is its own NiceGUI page handler keyed to the same
  `Session`. A header dropdown switches every layer card between the
  HISTOGRAM view and the MIN/MAX extreme-patch view; each view's
  checkbox group (**Log x** / **Log y** vs. the four grid toggles +
  **Heatmap**) is only visible while its view is selected. The page
  builds one `_WatchLayerPanel` per watched module; each holds both
  views and refreshes only the visible one. The histogram view holds
  two `_HistPlot`s (activations and gradients) and a stats table
  above each. A 2-second `ui.timer` calls
  `session.watch_snapshot()` and hands the per-phase stats to every
  `_HistPlot.update`. Routine ticks only change the bar counts (and the
  epoch label), so the plot **restyles the existing figure in place** —
  `Plotly.update(getHtmlElement(id), {y, name}, layout, indices)` run via
  `ui.run_javascript` — rather than replacing it. Restyle leaves the rest
  of the client-side state alone, so any zoom/pan survives the refresh
  for free. The figure is only rebuilt
  (`plot.update_figure(_figure_payload(new_fig))`) when the *structure* changes —
  a phase appears/disappears, or a **Log x** / **Log y** checkbox flips
  an axis scale — which `_HistPlot` detects by comparing the current
  `(phases, axis)` signature against the last render. Each figure is a
  `make_subplots` column with one stacked row per phase (no overlay), so
  one phase never obscures another: the rows share the x-axis, each row
  is `_PLOT_HEIGHT` tall, and the subplot titles — phase + epoch, tinted
  with the trace color — double as the legend (refreshed via
  `annotations[i].text` relayouts on every tick). Both axes default to
  linear (the **Log x** / **Log y** checkboxes start unchecked): bars
  sit at their true linear bin centres (`_BIN_CENTERS`) with per-bin
  widths (`_BIN_WIDTHS`). The x-axis zooms to the trimmed bin span
  (`_trimmed_bin_bounds`): pooled across traces, the outermost bins are
  dropped greedily — lighter end first — while the dropped bins together
  hold under the clip budget's share of the points, so a lone outlier
  value can't stretch the axis; `_linear_x_range` maps the span to value
  space (the full ±1e6 span is mostly empty) and `_log_x_range` to
  bin-index space for the signed-log view. The y-axis is always
  normalized per phase so train and val are comparable regardless of
  sample counts: with a linear value axis (`_use_density`, governed by
  **Log x** alone), bar heights are probability densities
  (`count / (n * bin width)`, `_probability_densities`) so bar area is
  the share of values; checking **Log x** redraws the bars at evenly
  spaced bin indices (signed-log tick positions computed once by
  `_x_tick_layout`, powers of 10 labelled) with plain per-bin
  probabilities (`count / n`, `_probabilities`). On a linear y-axis the
  range is capped by `_linear_y_range`, with the same range applied to
  every row so the subplots stay comparable. Two clipping rules keep
  freak spikes from flattening the rest of the distribution: per phase,
  a single drastically dominant bar — more than `_DOMINANCE_RATIO` (5x)
  taller than the runner-up, e.g. ReLU's exact zeros piling into the
  2e-9-wide zero band — never anchors the scale (`_scale_bars`); among
  the rest, bars clip tallest-first only while the clipped ones together
  hold under the clip budget's share of the pooled data points. Both
  axes draw their budget from `_axis_ranges`: it starts at
  `_BASE_CLIP_SHARE` (0.5%) and, while the bars would cover less than
  `_MIN_FILL_FRACTION` (5%) of the plot area (`_fill_fraction` — bar
  area over x-span x y-top, averaged across rows), raises the share in
  `_CLIP_SHARE_STEP` increments up to `_MAX_CLIP_SHARE` (5%), so a tall
  near-zero peak next to a long thin tail can't leave the plot nearly
  empty. The **Log x** checkbox alone decides the x-mode: multi-decade
  distributions (real gradients spread mass across ~6 decades) render
  on the linear axis as trimmed as the budget allows until the user
  checks it. Refresh ticks
  restyle `y` + `customdata` (the raw counts shown on
  hover) and push per-row `yaxis{n}.range` (plus a single matched
  `xaxis.range`) relayouts only when a cap actually moved. Checking
  **Log y** swaps the y-axis `type` to log so distribution tails stay
  visible (dropping the y-cap in favour of autorange — a linear-space
  range would be misread as log10 units) without changing what the bars
  measure. Every figure ships with `_PLOTLY_CONFIG` (via
  `_figure_payload`): the "Autoscale" modebar button is removed because
  it would autorange onto the clipped spikes — a different scale than
  the initial render — and double-click is set to `reset`, restoring
  the built ranges. The scalar stats above each figure render as an
  HTML table (`_stats_table_html`) in a light framed box: one column per
  phase with data (header tinted with the trace color), one row per stat
  (`n`, `mean`, `std`, `median`, `min`, `max`).
- The `/watch` MIN/MAX view renders each enabled patch type as one
  composite image per phase (`render_patch_grid`): channels as columns,
  the per-channel top samples as rows (best first), denormalized with
  the session's `input_mean` / `input_std` like the input pane, encoded
  at native patch resolution and CSS-upscaled to `PATCH_CELL_SIZE` per
  cell with `image-rendering: pixelated` (`max-width:none` opts out of
  the preflight clamp so wide grids scroll horizontally). The
  **Heatmap** checkbox blends each cell's stored activation map over
  the patch — transparent at zero, opacifying toward red/blue at
  `HEAT_MAX_ALPHA` for the grid-wide absolute extreme, with crop
  windows ratio-mapped back onto the activation map. Never-filled
  slots render flat gray. Grids re-render only when a cheap signature
  (enabled toggles + heatmap flag + the stored extreme values) changes
  (`_patch_grids_signature`), so routine 2-second ticks cost nothing
  once an epoch's buffers settle. The page's refresh is single-flight:
  snapshotting and grid rendering run in a worker thread
  (`asyncio.to_thread`) so the event loop keeps serving websocket
  traffic, and a toggle landing mid-render marks the pass dirty —
  rapid Heatmap clicks coalesce into one follow-up render instead of
  queueing one per click. Two further guards against websocket
  overload: grids are always PNG (`PATCH_GRID_FORMAT`) because a wide
  layer's BMP grids reach multiple MB per message — enough to pause
  the transport, where concurrent drains trip a known
  `websockets`-legacy keepalive assertion that kills the connection —
  and `watch_snapshot(include_patches=False)` skips the patch GPU→CPU
  copies entirely while the HISTOGRAM view is selected.
- The `/weights` page (one `?layer=` query param) is the per-layer weight
  viewer. It reuses the shared stepping controls (no sample spinner) and
  builds one `_WeightPanel` per name in `session.layer_weights[layer]`,
  reading the parameter shape from `model.named_parameters()` so the
  controls exist before any snapshot has been captured. Each panel holds a
  per-dimension role select (X / Y / Tile / Index, scaled to the weight's
  rank) plus an index spinner per axis; picking a role auto-demotes
  whichever other axis held it, keeping X/Y/Tile unique, then re-renders
  against the last snapshot via `render_weight`. Each panel stacks its
  strips in one horizontal scroll container, each flanked by a labelled
  marker bar (`_strip_marker`) — the weight (sky WEIGHT), then its
  gradient (violet GRADIENT; same shape, so the same axis layout applies;
  sourced from `snapshot.weight_gradients`, a placeholder note before the
  first backward), then one amber-marked strip per tensor-valued
  optimizer state entry when the session has an optimizer (labelled with
  the state key). State entries matching the weight's
  shape reuse the panel's axis controls; differently-shaped ones (e.g.
  factored second moments) fall back to their own rank's defaults.
  0-dim entries (Adam's `step`) join the group hyperparameters on a
  scalar line below the strips. Without an optimizer the container and
  line stay empty, leaving the page exactly as before. New snapshots
  re-render
  through the page's `ui.timer` (`maybe_render`). Because NiceGUI
  suppresses `.value` writes made from inside a value-change handler, the
  select/visibility sync after a demotion is deferred one event-loop tick
  with `ui.timer(0.0, …, once=True)` — the same workaround the main page's
  sample spinner uses. A top-bar Refresh button calls
  `session.current_weights()` / `current_weight_gradients()` /
  `current_optimizer_state()` / `current_optimizer_hyperparams()`
  (live CPU clones read at call time rather than at a pause) and pushes
  them through each panel's `show_weights`, so every strip updates
  mid-training even in `detach` / `step_run` where no snapshot is
  published. Because `maybe_render` only redraws on a *new* snapshot,
  the manually-refreshed live view persists until the next captured
  batch.

## Enabled flag (zero-overhead off switch)

`playgrad.start(model, ..., enabled=False)` returns a fully inert session,
the intended way to leave playgrad wiring in a training script and turn it
off with one flag:

- **Construction** skips `_try_trace(model)` (the proxy forward pass is the
  only expensive part) and leaves `_input_names` / `_layer_names` empty.
- **`_BatchContext.__enter__`** checks `self._session._enabled` *first* — a
  plain attribute read, no lock. When false it returns immediately, so a
  disabled batch does no schedule advance, no capture decision, and no hook
  install. `_position` stays `None`, so `__exit__` also returns at once and
  the user's training body is the only thing that runs.
- **`serve()`** returns `None` without building the page or starting uvicorn.
- Knock-on effects: `watch()` returns `False` (nothing is in `layer_names`),
  `fx_traced` is `False`, and the declared batch counts are never enforced
  because `Schedule.advance` is never called.

## Lifecycle summary

```text
playgrad.start(model, epochs, phases, enabled=True)
        │
        ▼
   Session (mode=STEP)
        │
        ├── with session.batch(phase, epoch):
        │       ▼
        │   schedule.advance() → BatchPosition
        │   _should_capture()? ── no ──┐
        │       │ yes                   │
        │       ▼                       │
        │   _install_hooks()            │
        │       │                       │
        │       (user code:             │
        │        zero_grad, forward,    │
        │        backward, step)        │
        │       │                       │
        │       ▼                       │
        │   _remove_hooks()             │
        │   _publish_snapshot()         │
        │   _wait_for_proceed() ────────┤
        │       │                       │
        │       ▼                       ▼
        │   _activations.clear()    (no capture, no pause)
        │
        ▼ (UI thread, anytime)
   stop / step_batch / step_phase / step_epoch /
   step_run / step_until_position / detach / close
```
