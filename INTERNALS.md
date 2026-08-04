# NaNsense internals

This document explains how the `nansense` library is structured under the
hood. For *using* the library, see `README.md`. For agent-facing guidelines,
see `AGENTS.md`.

## Threading model

A NaNsense session lives across two threads:

- **Training thread.** The user's training loop. Forward / backward / step
  run here, and `with session.batch(phase=..., epoch=...)` is entered here.
- **UI thread.** Driven by NiceGUI (not yet implemented). Reads session
  state, calls control methods (`stop`, `step_batch`, …, `detach`, `close`).

Synchronization is a single `threading.Condition` (`Session._cv`) protecting:

- `_mode` — current `Mode` enum value.
- `_resume_token` — monotonic counter; bumped by every "go" command.
- `_pause_count` — monotonic counter; bumped each time the training thread
  enters `_wait_for_proceed`.
- `_paused` — True while the training thread sits in `_wait_for_proceed`,
  False while it advances batches; surfaced point-in-time as the `is_running`
  property the top bar grays Run/Stop from.
- `_closed` — flips once on `close()`.
- `_schedule` — mutated by `set_schedule()` and time-travel rewinds.
- `_pending_jump` — armed by `request_time_travel()`, consumed at batch
  boundaries (with a lock-free `is None` fast path on the consuming side).

`_snapshot`, `_activations`, and `_hook_handles` are written only by the
training thread and read by the UI thread; reads are point-in-time and
don't need a lock because Python attribute assignment is atomic under the
GIL.

## Schedule

A `Schedule` can be **declared** up front (`Schedule(epochs=N,
phases={...})`) or **discovered lazily** (the default — `start()` with no
`phases`). The epoch count comes from `session.epochs(N)` (or, as a fallback,
`start(epochs=N)`); phase names appear in `advance` first-seen order, and a
phase's batch count is learned by observation: `Session.batches` reports the
count to `Schedule.record_phase_length` when its loop runs to completion.

`Schedule.advance(phase, epoch)` is called inside `_BatchContext.__enter__`
and returns a `BatchPosition` with `batch_idx` and three boundary flags:
`is_last_in_phase`, `is_last_in_epoch`, `is_last_overall`. The flags are
*predictive* — known on a batch's `__enter__`, before its forward pass — but
only when the relevant count is known: that is from batch 0 when `phases` was
declared, and from the **second** epoch on when discovered (the first epoch is
the blind window, since a count isn't known until that phase first completes).
A declared schedule additionally validates (unknown phase / over-count raise);
a lazy one tolerates both (an over-count just re-learns the length). The step
modes that must work on the first epoch do not rely on these flags — see below.

For non-deterministic workloads (and the Lightning integration, which knows its
counts up front), `Session.set_schedule()` re-declares `epochs` / `phases`
mid-run, which also pins them for full first-epoch fidelity.

## Modes and capture decisions

| Mode | Public method | Captures + pauses at |
|---|---|---|
| `STEP` | `step_batch()` (also `stop()`, no resume) | every batch |
| `UNTIL_PHASE_CHANGE` | `step_phase()` | first batch of the next phase (position left the origin), or the run's last batch |
| `UNTIL_EPOCH_CHANGE` | `step_epoch()` | first batch of the next epoch (`epoch > origin`), or the run's last batch |
| `UNTIL_END` | `step_run()` | `is_last_overall` |
| `UNTIL_POSITION` | `step_until_position(phase_index, epoch, batch_idx)` | exactly that `(phase_index, epoch, batch_idx)` |
| `DETACH` | `detach()` | never |

A session starts in `STEP` mode — the first batch always pauses so the UI
can show its initial state.

`step_phase` / `step_epoch` snapshot the live position as `_step_origin` when
issued and pause on the first batch that *leaves* it (a position comparison, so
no prospective `is_last_*` flag is needed — these work on the first, unlearned
epoch). "Step epoch" therefore lands on **batch 0 of the next epoch**, not the
last batch of the current one. The last epoch has no successor to land on, so
both fall back to `is_last_overall` to stop on the run's final batch *when that
is detectable* (a multi-epoch run has learned its shape by then); when it is not
(e.g. a single-epoch lazy run), stepping simply runs off the end — acceptable,
the post-mortem page keeps the last frame. `step_until_position` addresses the
phase by **index** into the phase order, so a target in a not-yet-observed
phase/epoch is matched once training reaches it.

`_should_capture(pos)` is the single decision function for both "install
hooks?" and "pause after this batch?". Capture and pause are intentionally
the same predicate — there is no implicit pause that the user did not ask
for, and there is no orphan capture without a pause to consume it.

### Frequency updates

Orthogonally to the mode, `_should_freq_update(pos)` decides whether the
batch publishes a non-pausing *frequency update* — the cadence configured
via `Session.set_update_frequency` (an `UpdateFrequency` of `unit`
`"epoch"` or `"batch"`, multiplier `n`, and an optional phase filter for
the batch unit; the default is every epoch). A frequency-update batch
installs the same hooks as a capture and, at `__exit__`, publishes a
snapshot, re-runs the probe, re-runs every live auto experiment, and feeds
one frame to each active recording — but never calls `_wait_for_proceed`.
This is what keeps the UI (and recordings) refreshing during `step_epoch`,
`step_run`, and `detach`. The batch-unit counter (`_freq_counter`) lives on
the training thread and resets when the setting changes.

### On-demand refresh

`Session.request_snapshot()` (the UI's Refresh button) arms a one-shot
`_snapshot_request` flag, consumed at the next batch start by
`_take_snapshot_request()` — the same lock-free fast path as `_take_pending_jump`.
A batch that consumed it publishes exactly like a frequency update **minus the
recording frame**: it installs hooks, and at `__exit__` publishes a snapshot,
re-runs the probe, and re-runs auto experiments, but never records a frame and
never pauses. `_BatchContext._publishes` (capture, frequency update, *or*
snapshot request) is the single predicate gating hook install and the publish
block; only `_freq_update` still gates `_record_frames()` and only `_captured`
gates `_wait_for_proceed()`. The point is that the views only refresh on a
published snapshot, so in `detach` / `step_run` between cadence ticks they sit
frozen on the live model — Refresh nudges the next batch to publish without
recomputing anything off-batch (the running forward/backward is simply
captured). It is a no-op when no batch follows (the shown snapshot is already
current). In distributed runs only the leader holds the flag; folding it into
the `publish=` argument of `sync_batch_control` keeps followers in lockstep for
the watch-stats reduce.

## Hook lifecycle, gradient pickup, snapshot copy

The capture machinery — hook installation, the fx interpreter,
construction-time name/weight discovery, and the live `current_*` weight
and optimizer reads — lives in `nansense.capture`; `_BatchContext` and the
thin `Session` methods call into it.

There are two capture paths, picked once at session construction by
trying `torch.fx.symbolic_trace(model)` (`capture.try_trace`).

**fx path (preferred).** When the trace succeeds, the session holds the
resulting `fx.GraphModule` and `capture.install_hooks` monkey-patches
`model.forward` with a function that runs a custom `fx.Interpreter`
subclass against that graph. The interpreter overrides `run_node` so that
after every node executes — placeholders, `call_module`, `call_function`,
`call_method` — it stores the returned tensor in `Session._activations`
under a friendly key built by `nansense.fx_names.friendly_names`. Module
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
node's id and label line up with its layer card. `capture.remove_hooks`
restores the original `forward`.

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

`Session.layer_info` is the third construction-time map — `layer name ->
hyperparameter string`, keyed identically. Module layers carry their
`print(model)`-style signature built from `extra_repr()` (PyTorch's
universal hyperparameter surface — `Conv2d(3, 64, kernel_size=(3, 3),
...)`; custom modules override `extra_repr` to join in), fx
function/method ops carry their literal non-tensor call arguments
(`max_pool2d(2, stride=None, ...)`), and layers with nothing to report
(graph inputs, `relu`, `add`, …) map to `""`. The main page publishes the
non-empty entries to the browser as a slug-keyed JS map
(`_layer_info_script`) and a cursor-following tooltip div in the shared
static JS shows them while hovering a diagram node or a card header.

At `__exit__`:

1. `capture.remove_hooks()` removes every registered hook.
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
   Every clone goes through `tensor.detach().to("cpu", copy=True)`
   (`capture.cpu_clone`), so the snapshot is fully independent of the live
   computation graph — the next batch can free / overwrite all of its
   source tensors without affecting the snapshot. The weight and optimizer
   dicts come from `nansense.capture`'s `current_*` functions — the same
   live-read functions behind `Session.current_weights` & co., called here
   on the training thread so they're consistent with the just-run batch.
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
`_publish_snapshot` releases the previous snapshot *before* allocating the
new clones (see the Snapshot lifecycle section), so the peak is one
snapshot rather than two stacked while the new one is built.

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
without polling. The `is_running` property (`not _paused and not _closed`)
is the point-in-time companion the top bar polls to gray out Run while
training advances and Stop while it is paused.

## Snapshot lifecycle

`Session.snapshot` is the last `BatchSnapshot` published, or `None` if no
batch has been captured yet. It persists after `close()`, so the UI can
stay open and present a post-mortem view.

The snapshot is a frozen dataclass of CPU tensors. To avoid stacking two
full snapshots in host memory, `_publish_snapshot` first drops the session's
reference (`_snapshot = None`), builds the new clones, then installs them —
both writes atomic under the GIL. A reader that already holds the previous
snapshot keeps it alive (no torn read); a reader that calls the `snapshot`
property during the build sees the transient `None`. The property never
blocks on it: the render tick reads `snapshot` synchronously on the UI's
asyncio event loop before offloading the heavy render to a thread, so
blocking would stall every client. Instead a `None` is treated like a tick
with no new frame — the render loop keeps the prior frame and re-renders once
the next read (≈200 ms later) sees the published snapshot. The UI can hold
references to a snapshot for as long as it wants without preventing the next
batch from running.

Rendering (image strips, histograms, summary stats) happens on the UI
thread against the published snapshot; the eager copy in
`_publish_snapshot` only moves raw tensor data, not anything pixel-shaped.
A per-snapshot render cache in the UI (`_RenderCache`) keeps repeat
renders free.

## Watch accumulators

Independently of the snapshot path, the session can collect running
statistics for any subset of layer names — driven by clicking nodes in
the main page's architecture diagram, surfaced on the `/stats` deep-dive
page.

`Session.watch(name)` / `unwatch(name)` mutate the `_watched_layers`
set under `_cv`. The `Session.watched_layers` snapshot is a
`frozenset`, safe to read from the UI thread. `watch()` accepts any
name in `session.layer_names`: named modules, fx-traced intermediates
(`relu`, `add`, `mean`), and graph inputs (`x`).

`_BatchContext` extends the capture machinery to cover watching too.
If any layer is watched but the batch is *not* a capture batch
(`detach`, mid-`step_run`, etc.), `capture.install_hooks()` still runs — the
same fx interpreter or pre-hook+per-module-hook installation that the
snapshot path uses — so every name in `layer_names` lands in
`_activations` with `retain_grad()` applied. The only difference
between capture and stats-only batches is that capture also publishes
a snapshot and pauses; stats-only batches just compute stats and let
the training loop continue.

Which layers feed the accumulators is set by the three-way **stats scope**
(`Session.set_stats_scope`, a select in the settings dialog): `"watched"`
(the default) collects for the watched layers, `"all"` for every name in
`layer_names` regardless of the watched set, and `"none"` collects nothing
while keeping every already-collected bucket frozen — the pause the top
bar's stats toggle uses (`toggle_stats_collecting` flips between `"none"`
and the last collecting scope). Scope `"none"` gates both `_stats_only` (so
a non-publishing batch no longer installs hooks just for stats — the
capture-mode cost above disappears) and the `_update_watch_stats` call at
`__exit__` (so even a capture batch, which installs hooks for its snapshot,
folds nothing in); cards stay visible and keep rendering from the published
snapshot while the running aggregates freeze. Narrowing to `"watched"`
drops the buckets of layers outside the watched set (the same semantics as
unwatching them); `Session.stats_layers` — the `/stats` page's selectable
universe — is the collecting set plus every layer with retained buckets, so
`"none"` keeps previously collected layers browsable. In distributed runs
the scope rides the per-batch control broadcast (followers mirror it and
accumulate the same buckets), and `sync_batch_control` only sets the
leader's reduce flag when `publish` and some layer is collecting.

Outside the `"watched"` scope, the watched set no longer drives collection,
so the main page decouples card visibility from it: each browser tab keeps
its own "shown" set (seeded from the watched set when a decoupled scope is
entered) and diagram/card clicks never touch the session — two tabs can
show different layers, and hiding a card cannot drop anyone's stats.

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
   `min()`, `max()`, plus a `torch.bincount` over
   `channel * N_BINS + _bin_indices(x)` that yields a `(C, 211)`
   per-channel signed-log histogram. With no channel cap this is one
   fused pass and the universal 211-bin histogram is its channel sum, so
   per-channel tracking adds only a cheap reduction over what the
   universal histogram already paid. Under a `channel_limit` (see
   *Performance settings* below) only the first N channels keep
   per-channel rows; the universal histogram and the scalar reductions
   then run over *all* channels separately, so the layer-wide view stays
   accurate regardless of the cap. Channels are dim 1 of the batch-first
   tensor (features act as channels for 2D activations, matching the patch
   accumulator); 1D tensors and accumulators whose (capped) dim-1 size
   changes mid-stream (variable token counts) fall back to a plain
   universal bincount for good (`collapse_channels`).
3. All running state — `_n`, `_sum`, `_sum_sq`, `_min`, `_max`,
   `_hist`, `_channel_hist` — lives on that same device. No GPU→CPU sync
   happens during training.

Bin data lives only in the most recent epoch per `(layer, phase)` — the
UI never renders older bins (HISTOGRAM and MIN-MAX read the latest epoch;
GRAPHS reads per-epoch scalars), and keeping them would grow memory with
run length (a per-channel buffer is `C × 211` int64, ≈ 0.9 MB at 512
channels; the universal histogram ~2 KB, ×2 kinds ×layers ×phases
×epochs). When `WatchAccumulator.update` creates the bucket for a new
epoch of a phase, the *same* phase's older epochs collapse — the
phase-scoped release rule the patch buffers already use (a new train
epoch releases only older train buffers; val keeps its own until the
next val epoch starts). The eviction keeps the scalar aggregates and
caches the two bin-derived stats the views still need as plain scalars,
one small GPU→CPU sync per epoch boundary each: the buffer's final
dead-channel count (`collapse_channels(keep_dead_count=True)`) for the
GRAPHS dead-neurons series, and the universal histogram's median
(`collapse_hist`) for the GRAPHS median curve.
`TensorStatsSnapshot.dead_channel_count` / `.median` read the live
buffers when present and fall back to the stored values (`hist` is then
`None`; render paths treat such a bucket as having nothing drawable —
reachable as a phase's latest only through a time-travel rewind). The
mid-stream channel collapses (1D tensors, a dim-1 size change) store no
dead count — a partial epoch's count would lie. Under DDP the reduced
overlay stands in the leader's cached medians for collapsed buckets
(rank-local, like patches and dead counts).

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

**Current-batch stats.** The same `WatchSnapshot` shape can be produced for
*one* batch without any running accumulator: `watch.single_batch_stats` folds
a single tensor through throwaway `TensorAccumulator` / `PatchAccumulator`
instances, and `Session.current_batch_stats(layers=...)` calls it for each
requested layer using the published `BatchSnapshot`'s own activations,
gradients, and image input (`_image_like_input`, the snapshot-dict twin of
`_patch_source_input`). Because the snapshot CPU-clones *every* layer (not
just the watched ones), this works for any layer regardless of the watched
set — it backs the `/stats` page's "Current batch" phase. The caps come from
`watch_performance`, so per-channel rows match the running path; the result
keys under the snapshot's own `(phase, epoch)`.

### Extreme input patches (`nansense.patches`)

Alongside the histogram stats, each watch bucket owns a
`PatchAccumulator` that keeps, per activation channel, the
`n_per_channel` input samples producing the most extreme activations
under up to four rankings:

- `max_pixel` / `min_pixel` — the channel's single largest/smallest
  spatial value. The stored patch is an input crop around that pixel's
  ratio-mapped location (side ≈ `PATCH_FACTOR ×` the activation→input
  downsampling ratio, floored at `MIN_PATCH` — an approximation of the
  receptive field, not an exact computation).
- `max_average` / `min_average` — the channel's spatial mean; the
  stored patch is the whole input image (there is no single location).
  Collected only when `WatchPerformance.average_patches` is on — off by
  default, since the whole-image payloads roughly double the buffer cost
  for the least-consulted grids.

Each entry also stores the channel's full activation map so the UI can
blend a heatmap over the patch. 2D `(B, F)` activations degrade
gracefully: features act as channels, pixel ≡ average, whole-image
patches, single-cell heat.

Patch and heat payloads are stored quantized: uint8 levels `0..254` plus
a per-`(channel, sample)` fp32 `[offset, scale]` pair computed over that
slot's whole slice, with byte 255 reserved as a non-finite sentinel that
dequantizes to NaN (a diverged activation stays visible in the heat).
Rendering is 8-bit anyway, and the payloads are pure cargo — quantized
once at gather time, then only permuted (bytes and scale rows selected by
the same merge indices) — so this quarters both the GPU footprint and a
frozen moment's gallery bytes. The `vals` ranking scores stay fp32:
ranking arithmetic depends on them, and their `∓inf` placeholder slots
double as the renderer's mask.

`Session._update_watch_stats` feeds the accumulator via
`WatchAccumulator.update_patches(layer, phase, epoch, act, x)`, where
`x` is `_patch_source_input()` — the first image-like (4D, 1- or
3-channel) forward input captured by the pre-hook. Non-image models
simply gather no patches.

Like `TensorAccumulator`, everything stays on the training device with
no syncs and no data-dependent branching: per batch and per type, one
reduction over the activation produces per-sample-per-channel scores
`(B, C)`, a per-channel `topk` over the batch axis picks
`min(n_per_channel, B)` candidates, one vectorised fancy-index gathers
their patches, and a `cat`+`topk` merge folds them into the `(C, N)`
running buffers. `act` is sliced to the first `channel_limit` channels
on entry, so `C` here is the capped count. Ranking per channel over the
batch axis doubles as
deduplication — a sample appears in one batch per epoch, so a channel
row can never hold the same image twice. NaN scores are demoted to the
placeholder `∓inf` so diverged batches never enter the buffers.

One memory caveat drives the eviction rule in `update_patches`:
histogram buckets are ~2 KB and live forever, but a patch bucket holds
up to `4 × C × N` image crops on the GPU — and the average-type patches store
a *whole input image per channel*, so the cost scales with `C` and the
input resolution (a 512-channel layer at 192×256 is multiple GB). When
a newer epoch starts for the same `(layer, phase)`, older epochs' patch
buffers are released — the `/stats` page only renders the latest epoch
per phase, so nothing visible is lost. `forget_layer` /
`forget_epochs_from` (unwatch, time travel) drop patches together with
the rest of the bucket.

#### Performance settings (`Session.set_watch_performance`)

Because the per-channel patches dominate GPU VRAM, three knobs are
user-tunable from the settings dialog's "Performance" section and held on
the `WatchAccumulator`:

- `channel_limit` — keep per-channel data (both the per-channel
  histograms and the patch galleries) for only the first N channels of
  each watched layer (default 16, toggleable off for "all channels").
- `samples_per_channel` — how many extreme samples each ranking keeps per
  channel (the `N` in the `(C, N)` buffers; default 5).
- `average_patches` — whether the whole-image average grids are collected
  at all (default off; `PatchSnapshot.by_type` then simply lacks the
  average keys, the patch-grid renderers skip absent types, and the
  `/stats` MIN/MAX radio drops the two average entries —
  `_grid_type_options`).

All three fix the per-channel buffer shapes, so `WatchAccumulator.configure`
drops every bucket when any changes (the next update rebuilds under the
new config) — the UI warns that statistics are flushed. The accumulator
reads the config under its own lock during `update`/`update_patches`, so a
flush can never interleave with a half-built bucket.

`PatchAccumulator.snapshot()` copies the buffers to CPU — the payloads
cross the bus as uint8 and are dequantized CPU-side, so the frozen
`PatchSnapshot → TypePatches` tree carried on `LayerStatsSnapshot.patches`
stays fp32 for every render path; slots never filled keep their `∓inf`
values and are masked by the renderer.

## Custom instruments (`nansense.instruments`)

Instruments are user callbacks evaluated per collected layer against the
live device tensors — the general-purpose extension of the watch path.
Three kinds, registered through decorators on the session (`watch_metric`,
`watch_layer_tensor`, `watch_weight_tensor`, all thin wrappers over
`InstrumentManager.register`; names are unique across kinds):

- **Scalar metrics** run in `_update_watch_stats`, right after the built-in
  accumulator folds, for the same stats-collection set — so scope gating,
  the `retain_layers` reaper, and "only watched layers are logged" come for
  free. Each callback gets a `LayerContext` (live activation, its retained
  grad, the layer's parameters/optimizer state sliced from per-batch
  `_instrument_sources` views — no clones) and returns a number, a mapping
  of named scalars (one trace per key), or `None` to skip. Samples land in
  the manager's series store keyed `(layer, phase, epoch, metric, series)`
  as `array("q")`/`array("d")` pairs — raw per-batch floats, cheap enough
  to keep for every epoch (no eviction analog to the histogram collapse).
- **Tensor instruments** run in `_publish_snapshot` only (their outputs are
  render cargo, so non-publishing batches skip them): activation-shaped
  results land on `BatchSnapshot.custom_activations` (extra strips under
  the layer card's act/grad rows; probe frames show none — a probe forward
  never runs instruments), weight-shaped ones on
  `BatchSnapshot.custom_weight_tensors` (extra `/weights` strips under the
  panel's axis controls, after the optimizer entries). Shapes are enforced
  against the activation/parameter; results are CPU-cloned like all
  snapshot cargo.

The GRAPHS view reads `Session.watch_metrics_snapshot()` (fetched by the
stats page's refresh worker only while GRAPHS is showing): the manager
copies the raw arrays under its lock, then assembles frozen `MetricSeries`
outside it — `on="epoch"` series fold each epoch's values through the
registered reduce (`mean`/`sum`/`min`/`max`/`last` or a callable) into one
point, `on="batch"` series place each sample at `epoch + batch/n` so batch
curves share the per-epoch x-axis. One lazily-created plot per metric
(`_MetricPlot`), restyled in place like the built-in figures and rebuilt
only when the trace set changes.

Failure isolation is the load-bearing property: a raising callback, an
uncoercible return, or a shape mismatch disables that instrument (it drops
out of the lock-free enabled caches the per-batch checks read), records the
error (`Session.instrument_errors`, shown under the GRAPHS plots, one
console line), and never touches the training thread's health. Collected
data outlives the disable.

Lifecycle mirrors the accumulator: `unwatch`/`retain_layers` drop a layer's
series, time travel drops `epoch >=` buckets and additionally calls the
optional `on_rewind(epoch)` hook on stateful callables (state from the
abandoned timeline must not leak into the replay). Frozen moments store the
*reduced* snapshot (`InstrumentManager.state_dict` — a callable reduce
can't be serialized, its reduced points can) plus the snapshot's custom
tensors, spliced back like the optimizer state on load; a restored manager
serves that overlay read-only. Under DDP instruments are leader-only
(rank-local, like patches). `lock()` makes registration raise — a locked
demo registers before locking. Recordings don't include the custom strips
(the recorders render through their own frame builders).

## Numerical-error debugger (`nansense.debugger`)

The debugger watches for two numerical failures and pauses training when it
first finds one, surfacing a yellow *warning* banner in the UI. It runs every
*n*th batch (default 100, configurable/toggleable) so a clean run pays almost
nothing.

- **NaN / ±Inf** — trips if a single non-finite value appears in any checked
  tensor (forward activations, activation gradients, and weight gradients).
  One bad value poisons the run, so there is no fraction threshold.
- **Subnormal / overflow** — trips when a layer's *gradient* magnitude
  collapses or saturates into a precision-losing band. The band is
  **dtype-aware**: the subnormal edge is nonzero `|x|` below `finfo.tiny`
  (precision degrading toward zero); the overflow edge is `|x| >= finfo.max /
  OVERFLOW_HEADROOM` — an early warning a factor below the ceiling, because a
  value that genuinely overflows rounds to `±inf` (caught by the NaN/Inf
  check), so flagging only the exact max would almost never fire. A layer
  trips when the summed `|x|` inside the band is at least `threshold` (default
  0.1) of the layer's total summed `|x|`; non-finite values are excluded from
  those sums (they belong to the NaN/Inf check). The summed-`|x|` metric makes
  detection deliberately conservative — a handful of subnormals next to
  normal-magnitude values barely moves the ratio; it fires when the gradient
  has *broadly* collapsed. The internal reason key stays `"underflow"` but the
  UI labels it **"subnormal"** (`REASON_LABELS`), its precise meaning.

**Everything runs on the compute device.** `debugger.run_checks` builds one
`[nan_count, inf_count, total_count, underflow_abssum, overflow_abssum,
finite_abssum]` vector per layer (weight gradients are mapped to layers via
`Session.layer_weights`, exactly like the snapshot), stacks them, and pulls
the whole batch's counters to the CPU in a single transfer. It returns a
frozen `DebugError` (the tripped `reasons`, the `checks_used` categories, and
a `LayerReport` per affected layer) or `None`. Each `LayerReport` also carries
the scanned gradient's `dtype`, so the UI can name the band edges (the
subnormal `finfo.tiny` and the overflow `finfo.max / OVERFLOW_HEADROOM`, via
`debugger.dtype_band`) in real magnitudes.

**Lifecycle integration.** `_should_debug_check(pos)` mirrors
`_should_freq_update`: a training-thread `_debug_counter` throttles checks to
every *n*th batch, independent of mode (so detach / run-until are covered),
leader-only under DDP. A check batch installs hooks like a capture/stats
batch (it needs the activation gradients), and `_run_debug_checks` runs at
`__exit__` *before* `remove_hooks`, while the activations and their retained
`.grad` are still live. The *first* hit of an episode records the error, resets
`_debug_counter` to 0 (so the next Step re-checks immediately rather than
waiting out the interval), and calls `stop()` — forcing this batch to publish a
snapshot (so the affected layers have data behind the banner) and pause, even
in a free-running mode (it returns `True` from `_run_debug_checks`, the
`debug_force` that gates the publish+pause). Once a banner is standing, later
hits **merge** into it (`debugger.merged`: union the reasons/`checks_used`,
keep the *first* error's position, keep each layer's worst observed fraction)
and return `False` — so a user who chose to proceed past the first issue keeps
running while the warning accumulates rather than stopping every *n*th batch.
The onset also prints one console line (so a headless run with no browser still
sees it); merges stay quiet. A time-travel rewind prints its own line
(`_rewind_to_epoch`, covering both the plain-loop and Lightning restorers).

`Session.debug_error` is published as an atomic reference and read lock-free
by the UI. Resuming no longer clears it (`_set_mode(resume=True)` leaves it
alone) — the warning stands across Run/Step so detections can accumulate. It
is cleared only by silencing the active checks (the banner/dialog "Silence
warning" button → `disable_debug_check` for every present category) or by a
time-travel rewind (`_rewind_to_epoch`, the abandoned timeline's banner).
`disable_debug_check(category)` turns off one check (`"nan_inf"` or
`"under_over"`) and trims that category's reasons/columns from the active
error via `without_category` (the banner clears entirely if nothing remains).

**UI** (`top_bar._add_error_banner`, added under every page's top bar): a
0.2 s timer polls `session.debug_error`, rebuilding the full-width yellow
warning banner (amber background, ⚠ icon) when the record identity changes
(every detection / merge makes a fresh frozen record) and hiding it when it
clears. The banner shows "Numerical issue detected", the reasons, and the
*first* error's `epoch | phase batch`, a hover description, a Details button,
and a single **Silence warning** button (turns off the active checks and
dismisses the warning). The details dialog explains the problem, spells out the
dtype-aware under/overflow band in real magnitudes when that check ran
(`_under_over_band_lines`) followed by a remediation tip (loss scaling /
bfloat16 / gradient clipping), and lists the affected layers in a table — one
column per reason whose check *ran* (so under/overflow columns show even when
only NaN tripped), percentages per layer, and per-row actions: a **Stats** link
to `/stats` when the layer is already watched, else a **Watch** button (start
collecting, stay) plus a **Stats** link carrying `watch=1` (the stats page
starts the watch on open, so the jump stays a middle-clickable anchor). The
gradient histogram fills in once a few watched batches have stepped. The gear
settings dialog's "Error checks" section edits the `DebugSettings` (enable,
interval, per-check toggles, threshold %) via `Session.set_debug_settings`.

## Distributed training (`nansense.distributed`)

In a multi-rank `torch.distributed` run, every rank constructs a session
(`Session.__init__` picks up an initialized process group via
`distributed.context()`; a world size of 1 gets `None` and the
single-process behaviour). A `DistributedDataParallel`-wrapped model is
unwrapped up front — hooks on the inner module still fire through the
wrapper's forward, while names and the fx trace stay clean (the wrapper
adds a `module.` prefix everywhere and is not fx-traceable).

Rank 0 **leads**: it alone serves the UI (`serve()` no-ops elsewhere,
which also keeps the ranks from fighting over one port), publishes
snapshots, pauses on captures, and runs probes/experiments — all
single-rank work on the inner model, with no collectives. The other
ranks **follow**: they never capture, publish, or pause, but accumulate
watch stats over their own data shard. Two collective touch points, both
on the training thread:

1. **Per-batch control broadcast** (`sync_batch_control`, at
   `_BatchContext.__enter__`): a 4-int tensor — "does this batch's
   `__exit__` reduce watch stats" (true on the leader's mode captures and
   frequency updates, when anything is collecting), the watched-set
   version, the leader's stats scope (mirrored by followers so every rank
   accumulates the same buckets), and its armed time-travel target. On a
   version change, a follow-up object broadcast carries the
   watched-layer list; followers apply it, dropping buckets of unwatched
   layers like `Session.unwatch` does. The version advances in lockstep
   on every rank, so "changed since the last batch" is a decision each
   rank makes identically and the object broadcast stays collective.
   This is also the pacing point: a paused leader holds every follower
   at its next batch start (the same place DDP's own gradient all-reduce
   would block them — but also covering forward-only phases like `val`).
2. **Stats reduction** (`reduce_watch_stats`, at `__exit__` right after
   `_update_watch_stats`, before the leader publishes — so a pause never
   shows stale global stats): the leader broadcasts its sorted bucket
   list with per-stream channel counts (`WatchAccumulator.reduce_meta`),
   each rank packs those buckets into four flat tensors
   (`TensorAccumulator.reduce_payload`; missing buckets contribute
   neutral values), and four all-reduces combine them — SUM for
   counts/sums/histograms, MIN/MAX for the extremes, plus a MIN-reduced
   per-stream "channel-ok" flag that drops the combined per-channel rows
   when any rank's buffer is missing or shaped differently (the universal
   histogram stays exact). The reduction never mutates local
   accumulators, so repeated reductions can't double-count. The leader
   unpacks the result (`_unpack_reduced`) into per-bucket
   `TensorStatsSnapshot`s stored on `Session._dist_watch_stats` (atomic
   reference), and `watch_snapshot()` overlays them on its local view —
   patches, and buckets no reduction has covered yet, stay rank-local.

Collectives run on the process group's natural device (CUDA for NCCL,
CPU otherwise) over the default group; orderings stay consistent because
every rank issues the same sequence (control broadcast → forward/backward
→ optional reduction) per batch. That also defines the contract: every
rank must drive the same `session.batch` structure, which
`DistributedSampler`-sharded loaders give naturally. Time travel works in
distributed mode: the per-batch control broadcast carries the leader's armed
jump epoch (a fourth int, `-1` when none), so a UI-requested jump makes every
rank raise `TimeTravelJump` at the same batch-start barrier — before any
forward/backward or reduction, so no collective is left half-issued (a leader
woken from a pause keeps the jump armed and applies it at the next barrier
rather than mid-`__exit__`). Each rank then restores from its own checkpoint:
model/optimizer/scheduler are replicated (loaded from the rank's
self-sufficient file, `epoch_<n>.pt` on the leader and `epoch_<n>.rank<r>.pt`
on followers), RNG is captured/restored per rank, and `DistributedSampler.set_epoch`
reproduces shard order — so the replay is deterministic on every rank. Every
rank drives the same `session.epochs()` loop. `close()` should run after the
loop on all ranks; a leader closed mid-loop stops broadcasting and would leave
followers blocked at their next batch start until the collective timeout.

## Probe runs (`nansense.probe`)

A probe is a NaNsense-internal forward pass on a *pinned* input batch, run
between batches so the UI can show the network's response to one fixed
input across stepping and time travel. `Session.pin_current_batch()` pins
*every* input tensor of the last snapshot (already CPU clones, keyed by input
name); from then on each capture re-runs the whole model on them right after
`_publish_snapshot` and publishes a `ProbeResult` — CPU clones of the inputs
(`inputs`) and of every layer output, keyed like `layer_names`. Re-forwarding
all inputs (positional or keyword, ordered by `input_names`) is what makes
multi-input models work; the UI's input pane picks which input to view.
Probes are forward-only: no gradients.
The probe config lives on the `Session` (under `_cv`), but every state
transition and the runs themselves are module functions in
`nansense.probe` that the thin `Session` methods delegate to.

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

**Isolation contract** (`isolated_model`): probes never mutate training
state. Per-module `training` flags are saved and restored ("eval" / "train"
probe modes flip the whole model; "unchanged" runs as-is); every buffer is
restored afterwards (a train-mode BatchNorm forward updates running stats
in place); the RNG is forked (`torch.random.fork_rng`, CUDA/MPS-aware) so
e.g. train-mode dropout doesn't perturb the global stream that time-travel
replays depend on; and the whole run sits under `torch.no_grad()`.

**Capture reuse without interference** (`capture.capture_forward`): takes the
inputs as an ordered list and runs `interpreter.run(*inputs)` / `model(*inputs)`.
In fx mode the probe runs `_CaptureInterpreter` against a fresh local dict — the
original `model.forward` is never patched, since the interpreter is
invoked directly. In the hook fallback, temporary pre/forward hooks write
into the local dict and are removed in a `finally`. Neither path touches
`_activations` or `_hook_handles`; both are safe because probes only run
between batches, when the batch path's hooks are uninstalled.

**GPU memory.** Probe captures clone every layer output to CPU *as it is
produced* (`to_cpu=True` in both capture paths) rather than holding live
tensors until the forward ends — otherwise a probe would keep a full
training-forward's worth of activations resident at once. For the same
reason `_BatchContext.__exit__` clears `_activations` (the training batch's
live GPU activations and retained grads) right after `_publish_snapshot`
CPU-clones them, before any probe runs — so a pinned probe never stacks a
second batch's activations on top of the training step's own.

**Perturbations.** `Session.add_perturbation(input_name=, sample=, index=, values=)`
records edits keyed by `(input_name, sample, index) -> values` in model-input
space — `index` is `(y, x)` for an image input (`values` is its `C`-vector; the
UI back-transforms a picked color via `input_panel.normalized_color`, or reads
per-channel fields directly) and `(channel,)` for a flat `[B, C]` input
(`values` is one scalar). When any exist, `apply_perturbations` clones only the
edited inputs (others are shared by reference) and writes the in-range entries
(out-of-range, count-mismatched, or absent-input ones are skipped — the base
may have changed shape since the click), and the probe runs a *second* full
forward on the substituted inputs inside the same isolation scope. `ProbeResult`
then carries `perturbed_inputs` / `perturbed_activations` next to the base pair,
and the UI renders the per-layer diff against the original whenever any
perturbation exists. Perturbations alone keep probing active without a pin — the
bases fall back to the snapshot's inputs (`_snapshot_inputs`), so edits track
the current training batch.

**Mode activates probing too.** `_probe_active_locked` treats a
non-`"unchanged"` forward mode as active on its own, alongside a pin or any
perturbation. So selecting `"eval"`/`"train"` re-runs the model on the
current snapshot's batch under that mode (no pin needed), and switching back
to `"unchanged"` with nothing pinned or perturbed clears the result so the
UI reverts to the live snapshot.

**Publishing and races.** Probe config (pinned input, mode, perturbations)
is mutated by the UI thread under `_cv`, bumping `_probe_version`.
`_run_probe` snapshots the config under the lock, computes without it, then
publishes under the lock only if the version is unchanged — a config change
mid-run wins and its own request re-runs the probe. `_probe_count` is the
monotonic completion counter (`wait_for_probe` mirrors `wait_until_paused`
for tests and the UI). A probe that raises publishes `probe_error` instead
of killing the training thread (`run_probe_guarded`); deactivating the
probe (`unpin_batch` / `clear_perturbations` with nothing else active)
clears the published result so the UI falls back to the snapshot.

## Experiments (`nansense.experiments`)

Experiments — deep dream and a small Captum selection (Grad-CAM, Neuron
Gradient, Neuron Integrated Gradients, Occlusion) — are the long-running,
cancellable extension of the probe job queue. Each client (browser tab)
queues an `ExperimentRequest` (`Session.request_experiment`, returning a
seq); the pause loop in `_wait_for_proceed` drains the queue in order and
calls `experiments.run(...)`, a generator yielding `ExperimentResult`
progress snapshots that are published one by one (`_publish_experiment`)
— that's what streams the evolving deep-dream image to the page. The
queue state lives on the `Session`, but the plumbing — `request_experiment`
/ `cancel_experiment`, the auto-experiment registry, and the guarded
runner (`run_experiment_guarded`) — are module functions in
`nansense.experiments` behind thin `Session` delegators. Results are kept per seq in a bounded map (the
`_EXPERIMENT_RESULTS_KEPT` most recently updated seqs;
`experiment_result_for(seq)`) plus a latest-result slot
(`experiment_result`), so concurrent tabs each poll their own run without
overwriting each other.

**Cancellation.** The runner checks a `should_abort()` predicate between
steps; it fires on `cancel_experiment(seq)` for the running seq (no seq
cancels everything; a queued seq is just dropped), once the run outlives
the `_EXPERIMENT_TIME_LIMIT` wall-clock ceiling (90 s — the training
thread, and on a locked demo every queued visitor, is held for the whole
run), and on anything that ends the pause — resume commands, a pending
time-travel jump, `close()` — so the pause loop regains control within
one step. Another client's new
request does *not* abort a running experiment; it waits its turn in the
queue. Requests queued while training is running wait for the next
snapshot publish or pause, whichever comes first (`experiment_pending`
lets the UI say so). A raising experiment publishes an error result
instead of killing the training thread.

**Where a request sits.** Nothing is published until a run has progress to
show — and the Captum methods publish once, at the end — so "no result
yet" alone can't tell a run in flight from one still waiting.
`Session.experiment_queue_state(seq)` resolves it into an
`ExperimentQueueState`: `"running"`, `"queued"` (with how many requests
are `ahead`), or `"absent"` (never queued, cancelled, superseded, or long
finished). Both front-ends use it — the experiment page picks its status
pill from it (spinner vs. static glyph), `get_experiment_result` returns
`stage` / `queued_ahead` for the same distinction. Both hand-offs from
queue to runner therefore pop the request and mark it running under one
lock, so no poll ever catches a live request as `"absent"` and a cancel
in that window still bites.

**Isolation.** Experiments share the probe contract via
`probe.isolated_model` (the refactored common core): eval-mode forwards
with per-module flags and all buffers restored, forked RNG. Gradients are
taken w.r.t. the *input* only (`torch.autograd.grad`), so parameter
`.grad` — which the snapshot path reads at `__exit__` — is never touched.

**Deep dream** (`_run_deep_dream`) is per-sample-normalized gradient ascent
that runs **one sample per channel** over the layer's first `channels`
channels: the batch and channel axes are matched on the diagonal, so sample i
maximizes channel i's mean activation (`_channels_objective`), read via
`_target_activation` (the fx interpreter against a local dict, so any name in
`layer_names` is a valid target; a single temporary hook in the fallback). The
`minimize` knob flips the step's sign, descending the same objective to
synthesize an input that *suppresses* each channel instead of exciting it. The
batch is sized to the `channels` knob and clipped to the layer's channel count
by a one-shot probe forward, so there are never empty trailing samples. The
starting batch is built from the network's real input (`_dream_start` — any
input shape, not just images): `start="noise"` draws one fresh noise sample
per channel matching the real input's per-sample shape and overall mean/std,
from a generator seeded by the request seq so successive Runs explore
different noise; `start="sample"` replicates one chosen input-batch sample
across the channels, so every channel's dream starts from the same real image.
The shown input (`reference`) is carried only for the current-batch start —
noise has none. Regularizers per step, applied only to `[B, C, H, W]` inputs:
jitter (random roll, undone after the update — drawn from the same
request-seeded generator), diffusion (blend with a 3×3 box blur), center
zoom (a per-step multiplier, ≥ 1), and clamping to `_value_bounds` — the
displayable `[0, 1]` range mapped through the input mean/std.

**Captum** (`_run_captum`) runs the *unpatched* model inside the isolation
scope (experiments only execute between batches). captum is a standard
dependency, so `captum.attr` is imported at module load and
`available_experiment_kinds()` always offers all four attribution methods.
Like deep dream, the Captum methods run on a *batch* (`_captum_input`: the
first `batch` samples of the live input) and publish one attribution per
sample. Grad-CAM and the neuron methods need the layer's `nn.Module`; fx
intermediates are rejected with a pointer to the producing module
(`layer_available` mirrors this so the UI can gray the layer out). Grad-CAM
explains a target class (−1 resolves to each sample's argmax, requiring a
`[batch, classes]` output). The neuron methods and Occlusion target the
selected layer-channel via the per-example mean selector
(`_neuron_selector`); Occlusion wraps the model in `_LayerChannelModel`,
exposing that channel's activation as the output so the slid patch
attributes against an *intermediate* channel instead of a class.

**UI** (`/experiment?layer=...`, one yellow "Experiment" button per layer
card): the left pane stacks the kind dropdown (with a hover tooltip and a
description at the pane's foot), Run / Cancel, then the parameter form. The
form is headed by a **layer selector** — the first knob — whose options
carry a `disable` flag (`_patched_update_options` reassigns
`_props['options']` so the flag survives NiceGUI's option regeneration),
graying out layers the current kind can't run; switching to a shorter layer
clips the channel selectors (deep dream's Channels count to the channel
count, Captum's Channel index to one less). The rest of each kind's knobs are
declared in `experiments.EXPERIMENT_PARAMS` (`ExperimentParam` specs rendered as
number/switch/select widgets) ordered, for deep dream, Channels → Start from →
Sample → method knobs (Minimize sits just above Clamp; Sample shows only for the
current-batch start) and, for Captum, Channel/Target → Inputs → method knobs; values persist
across kind switches via a shared `state.values`, seeded from each knob's
default with `session.experiment_defaults` overrides applied
(`experiments.default_param_values`) — how a hosted playground serves cheaper
deep-dream defaults without lowering the locked ceilings. Beneath the description, a
deep-dream-only **"Compare with MIN/MAX"** button jumps to the same layer's
`/stats?view=minmax` grids (the MIN/MAX view carries a symmetric "Compare with
Deep Dream" button at the foot of its controls). Like every cross-page jump
button, both are real anchors — an `href` prop kept in sync with the shown
layer, never an `on_click` navigate — so middle-click opens a new tab. A 200 ms timer streams the page's *own* request via
`session.experiment_result_for(seq)`, drives **auto-run** (re-register on
init and on any parameter / layer change, buffered to one run per tick;
`register_auto_experiment` drops a superseded queued request so the pause
loop is never flooded) gated on the shared `session.auto_run_experiments`
setting — except the init run, which self-starts even with auto-run off so
the page opens onto a result — and toggles Run/Cancel enablement (Run off while auto-run is on or
a run is in flight; Cancel off while idle). Run replaces and Cancel aborts
only this page's request, so tabs don't clobber each other.

The results pane leads with a **status pill** (`_StatusPill`, shared with
the stats cards) rather than a line of text: a tone plus either a spinner
or a glyph. Idle before the page has a request of its own, then — while
nothing is published — whatever `experiment_queue_state` says: running,
queued behind *n*, starting, or (cancelled/superseded) stopped before it
ran; and finally the streamed result, green on done, slate on stopped
early, red on failed. Every wait that resolves on its own spins, which is
all of them but the last: an auto experiment runs at the next
visualization update as well as at a pause, so advancing training only
*offers* Stop / Step Batch as a shortcut — and offers it solely when
unlocked, a locked demo having no such controls to name.

Results render through `render_result`. Deep dream uses a **single** card
(`_render_image_row`) holding one horizontal row of captioned cells: the
shared starting input first (only for the current-batch start), then one
dreamed image per channel captioned `channel i` (`render_image`; non-image
inputs fall back to a "not renderable" note). Captum keeps one card per sample
(`_sample_card`, the same card look as the watch / weights pages) with
captioned cells: the attribution strip *first*, then its input (`render_strip`,
shared diverging colormap).
A Captum-only **Overlay** toggle instead blends each attribution channel over
the input (`render_attribution_overlay` → `blend_signed_heat`, the same
signed red/blue alpha overlay the MIN/MAX patch grid uses) on a shared
`±vmax` scale — a pure display toggle that re-renders the last result without
a backend re-run. Both the attribution strips and the overlays are rendered
with `tile_px=INPUT_IMAGE_SIZE` so each map (a coarse Grad-CAM map upscaled
by the browser's nearest-neighbour, an input-resolution gradient map as-is)
is shown at the same size as the input image beside it.

**Auto experiments.** The experiment page's Run goes through
`session.register_auto_experiment(key, ...)` rather than the plain
`request_experiment`: besides queueing the request, the session remembers
it under the page's key (`_AutoExperiment`) and re-runs it on *every*
snapshot publish — mode captures and frequency updates alike — keeping the
request's seq, so `experiment_result_for(seq)` keeps returning the
freshest rerun and the deep-dream noise (seeded by the seq) is identical
on every update. A registration expires `_AUTO_EXPERIMENT_TTL` (5 s) after
the page's last `touch_auto_experiment` heartbeat (its 200 ms tick), so
closing the page stops the reruns; an active experiment recording pins the
entry (`pin_auto_experiment`) for its own lifetime. At each update,
`run_auto_experiments` first removes any still-queued duplicate of a
registered request so it runs exactly once per update, then puts the
batch's requests back on the queue in the order it will run them and pops
each as it starts — so a registration waiting its turn still reads as
`"queued"` (not `"absent"`) to `experiment_queue_state`, and cancelling one
mid-batch skips it instead of being a silent no-op.

## Recording (`nansense.recording`)

Each recorded view writes one MP4 (10 fps) per visualization update under
`nansense_recordings/<run timestamp>/`. `Session.recording` lazily creates
the per-session `RecordingManager`; the UI's settings dialog starts a
`ViewRecorder` from a `RecordedView` — the view's identity key (`"main"`,
`"weights:<layer>"`, `"watch_histogram"`, `"watch_minmax"`,
`"experiment:<layer>"`; one recording per key), its renderer page, and the
page parameters frozen at record start (watched layers, sample index,
phase, axis toggles, weight-axis layouts, the experiment seq). While a
view records, the matching page controls are disabled each tick (e.g.
`InputPanel.set_frozen`), unwatch actions are refused for the watch-page
views (their frames render from the accumulators that `unwatch` drops),
and the update-frequency dialog locks Apply — recordings advance at that
frequency, so changing it would change the videos' time base.

Frames are produced on the training thread: `_BatchContext.__exit__` calls
`Session._record_frames()` on frequency-update batches only (not on plain
mode captures — recordings advance at the show frequency, not per user
step), after the snapshot/probe/auto-experiment refresh, so each frame is
consistent with one update. The renderers reuse the UI's server-side
machinery (`render_strip`, `render_weight`, `render_patch_grid`,
`render_image`), nearest-upscaling the native-resolution data images to
their CSS display size like the browser does; histograms — Plotly on the
live page, hence client-side — are re-rendered with matplotlib (Agg)
using the same bar heights and axis-range helpers the page uses, and the
same *chrome*: `histograms.py` exports the plot's background, gridline,
zero-line and tick colors, its bar opacity, its font sizes and its y-axis
label, and both renderers read them from there. Three things matplotlib
has to be told explicitly, because its defaults are the opposite of
Plotly's: no spines or tick marks (Plotly draws a filled plotting area and
gridlines, no axis lines), font sizes converted px→pt (`_pt`; at 100 dpi a
"size 12" differs by 1.39× between the two), and `_power_ticks`, which
factors one exponent across an axis the way `exponentformat="power"` does
— matplotlib would otherwise park a shared power in a corner offset box
and label a ±2e-8 axis `-2.0 … 2.0`, which reads as if it ran to ±2. A row
is `_PLOT_HEIGHT` tall, so it has the page's aspect. The
MIN/MAX view writes pixel-grid (crops) and average-grid (whole inputs)
frames to *separate* `*_pixel.mp4` / `*_average.mp4` streams because
their frame sizes differ. Every stream is locked to its first frame's
dimensions (rounded up to even, as libx264's yuv420p requires; capped at
`MAX_FRAME_SIZE`) and later frames are white-padded/cropped to fit. A
renderer failure is stored on the recorder and shown in the dialog rather
than propagating into the training loop, and `Session.close()` calls
`end_all()` so files are playable when a run simply finishes.

Locking is deliberately fine-grained: frame rendering (seconds for large
views) runs outside every lock. The manager's lock guards only the
recorder dict, so the `count` / `is_recording` / `statuses` queries that
UI timers and click handlers make on the asyncio event loop return
immediately even mid-render — NiceGUI's websocket keepalive budget is
only ~6 s (ping interval 4 s + timeout 2 s at the default
`reconnect_timeout`), so an event loop blocked behind a render drops the
connection and loses the in-flight click. Each `ViewRecorder` serialises
just its short stream append/close sections with its own lock; a
recording ended or deleted mid-render finishes the render and drops that
frame (`_closed`). The dialog's end/delete actions additionally run via
`asyncio.to_thread`, since finalizing the file can take a moment.

Encoding is done **in-process with PyAV** (`av`, the ffmpeg *libraries* —
no child process). Each `_VideoStream` opens an `av` container with one
libx264 stream, converts each `rgb24` frame to `yuv420p`, and flushes the
encoder on `close()`. This deliberately avoids an ffmpeg subprocess: a
subprocess writer (the previous `imageio_ffmpeg` approach) communicated
over pipes and `close()` *waited on a separate process to exit*, which
could stall indefinitely (hanging "Save & Finish") or be OOM-killed — all
outside our control. In-process, encode/flush are bounded calls that raise
on error instead. `imageio` / `imageio-ffmpeg` left the dependency list with
that switch — nothing imports them, and `imageio-ffmpeg` alone vendored an
80 MB ffmpeg binary into every install. The stream uses `_X264_PRESET = "ultrafast"` with a
`_X264_THREADS` cap: libx264's default `medium` lookahead buffers ~40
frames and defers most encoding to the flush, where its working set
roughly triples (a multi-GB spike *at save time* for large frames that can
OOM the training process); `ultrafast` encodes frames as they arrive so
the footprint stays flat and the flush is near-instant. `_X264_CRF = 10`
reproduces the previous visual quality; the cost is weaker compression
(larger files), fine for short clips. Even so, the post-finalize UI
refresh runs through `_best_effort_ui_update` (in `top_bar`) so a page
closed during the await can't surface a teardown error.

**Snapshots** (`RecordingManager.snapshot`) are the same frame, taken once.
The dialog's Snapshot button — and the MCP `save_snapshot` tool — hand it
the very `RecordedView` a recording would freeze and get back PNG file(s)
in the same run directory, named `<key>_ep<E>_<phase>_b<B>[_<group>].png`
(`_unique_path` suffixes `-2`, `-3` rather than overwrite, since two
stills of one position — before and after a perturbation — are a normal
thing to want). It goes through `_render_view_frames` and
`_stamp_position` exactly like `ViewRecorder.capture`, so a still and a
video frame of one view are the same picture; only the sink differs, and
the MIN/MAX split into pixel/average files for the same reason. The
recorder dict is never consulted: nothing needs to be recording, and a
running recording neither gains a frame nor loses one. It runs on the
caller's thread (the UI's `asyncio.to_thread` worker, or an MCP tool
call) rather than the training thread, so a renderer failure *raises* to
that caller instead of being stored on a recorder — there is someone to
tell. Naming and writing take a separate `_snapshot_lock`, so a PNG write
never delays the `count` / `statuses` polls the event loop makes.
`save_snapshot` shares `_recorded_view` for every page but the
experiment: a recording registers a rerunning request so each frame is
fresh, which a one-shot still has no use for, so `_snapshot_view` draws
an already-published `seq` and registers nothing to leak.

## Time travel (`nansense.restore`)

Time travel jumps training back to the start of any epoch whose state was
checkpointed to disk. It is opt-in at the training-loop level. A hand-written
loop drives the flat API — `session.epochs(cache_dir=...)` (default
`.nansense_cache/`) yields the epoch indices, and `session.restore_point()`
wraps each iteration's body:

```python
for epoch in session.epochs(cache_dir=...):
    with session.restore_point():
        ...
```

Both sit on a single session-owned `TrainingRestorer` (created lazily by the
first `epochs()` call). The same object also exposes the older nested shape —
`while restorer.pending(): with restorer: for epoch in restorer.epochs():` —
which the Lightning integration transplants around `trainer.fit` (it cannot
hand Lightning a generator to drive). `restore.py` keeps the restore/jump
logic in one place: `_apply_pending_jump` (shared by both loops' entry) rolls
state back to a jump target, and `__exit__` / `epoch_guard` both suppress only
`TimeTravelJump`.

Why a context manager is mandatory rather than folding the catch into
`epochs()`: a `for` loop never throws its body's exception back into the
iterator (it closes it with `GeneratorExit`), so the generator cannot catch
the jump itself — `restore_point()` is what catches it, after which
`iter_epochs` re-yields the target epoch. Omitting it is detected (the
generator checks an "entered since the last yield" flag) rather than left to
crash training or loop forever.

Without this loop, nothing is written to disk, the session never raises a
jump, and the UI's Time Travel button is disabled (its tooltip explains
why). On a disabled session the loop is inert.

**Epoch cache.** With a restorer attached, the first `session.batches` call of
each epoch (its first phase, before `iter(loader)`) — or `_BatchContext.__enter__`
on batch 0, as the manual-`batch()` fallback — writes
`epoch_<n>.pt` into the cache directory before any forward pass:
`model.state_dict()`, `optimizer.state_dict()`, `scheduler.state_dict()`
(when passed to `start()`), and the torch/CUDA RNG states. Writes go
through a temp file + atomic rename; an existing file for the same epoch is
overwritten when training passes it again, so after a jump the older
timeline's entries persist until the re-run reaches them. Restoring the
global RNG state is what makes the replay deterministic — the DataLoader
draws its shuffle seed from the global generator at `iter()` time.

**Resume from a baked cache.** `session.epochs(n, cache_dir=...,
start_epoch=K)` starts the run at epoch `K` instead of 0, restoring
`epoch_K.pt` from a cache directory that may have been written by an
*earlier process* — `EpochCache.cached_epochs` scans the disk, so a
pre-existing directory (e.g. baked into a deployment image) is adopted
as-is and immediately feeds the time-travel dialog. The checkpoint is
validated up-front exactly like a jump request (`TrainingRestorer.
resume_from`: mmap load + the three `validate_*_state` checks, raising
`TimeTravelError` on a mismatch) and then armed as a pending jump, so the
first `restore_point()` entry loads the state on the training thread
through the same `_apply_pending_jump` → `_restore` machinery. This is
what lets a run resume straight into its final epoch from a cache baked
into a deployment image.

**Jump flow.** `Session.request_time_travel(epoch)` runs on the UI thread
and validates everything up-front: the restorer exists and isn't finished,
the epoch is in range, the checkpoint loads, and its model state matches
the live model's parameter names and shapes (`validate_model_state`). The
validation load is memory-mapped (`EpochCache.load(mmap=True)`): it only
reads keys and shapes, so the full model + optimizer state never
materializes in RAM on the UI thread. Any
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

The exception unwinds through the user's loaders and loops to the enclosing
context manager — `session.restore_point()` per epoch in the flat loop, or
`with restorer:` around the whole epoch range in the nested one — which
suppresses exactly this type and records the target. The next entry (training
thread, between epochs, so nothing races a forward pass) loads the checkpoint
back into the model / optimizer / scheduler / RNG and calls
`Session._rewind_to_epoch(epoch)`, which drops the schedule's batch
counters for `epoch` onward (`Schedule.rewind_to_epoch`) and the watch
accumulators' buckets for those epochs (`forget_epochs_from` — they're
additive, so the re-run samples must start from empty ones). The restore
load is the real (non-mmap) one; once its tensors are copied into the live
model/optimizer/scheduler, `_restore` drops the payload and calls
`release_cpu_memory()` (`gc.collect()` + glibc `malloc_trim`) so the
checkpoint's load peak — model params plus the optimizer's moment tensors —
is handed back to the OS at the jump instead of sitting resident. Because the
mode was set to `STEP`, the first batch of the target epoch captures and
pauses for inspection — the same behaviour as session start.

Completion differs by loop. In the flat loop `iter_epochs` owns it: after the
last epoch's body completes without a jump it marks the run finished and stops
(`restore_point`'s exit deliberately does *not* mark completion, since it runs
every epoch). In the nested loop `restorer.pending()` returns `False` once the
single `with` block completes without a jump, and raises if called twice
without the block being entered. Either way `finished` is what flips the UI to
"run completed".

One ergonomic difference: the nested loop's `with` spans the whole epoch
range, so history-dependent state reset inside it (`best_acc`, metric curves)
is rewound by a jump for free. The flat loop's `restore_point()` spans a single
epoch, so it can't auto-reset such accumulators — a script that cares must
reset them itself (e.g. when the yielded epoch is not the previous one + 1).
The examples just keep `best_acc` across the run, which is fine for a demo.

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

## PyTorch Lightning integration (`nansense.lightning`)

`NansenseCallback` maps the batch context onto Lightning's hook pairs: the
context returned by `session.batch(...)` is entered in
`on_train_batch_start` / `on_validation_batch_start` and exited in the
matching `*_batch_end` (where the capture, pause, and a possible
`TimeTravelJump` happen — on the training thread, inside Lightning's
hook). The gradient contract holds unchanged because Lightning's automatic
optimization zero-grads before backward and `optimizer.step()` doesn't
clear `.grad`. Sanity-check val batches are skipped
(`trainer.sanity_checking`); `on_exception` closes an open context without
publishing. The session is created once in `on_fit_start` (optimizers are
configured by then; a time-travel re-fit reuses it) with a placeholder
schedule; the real per-phase batch counts are re-declared at every
`on_train_epoch_start` (and `on_validation_epoch_start`, for counts that
were unknown at epoch start) via `set_schedule` — per-epoch re-declaration
is also what models `check_val_every_n_epoch > 1` runs, where epochs
without validation declare a train-only schedule. Mid-epoch validation and
unsized dataloaders are rejected: the schedule must be known up-front.

Time travel cannot live in the callback — it needs to own the retry loop —
so `fit_with_time_travel` transplants the `while restorer.pending(): with
restorer:` shape around `trainer.fit`. Its `LightningRestorer` (a
`TrainingRestorer` subclass created before the session exists and bound
via `Session.attach_restorer`) delegates all state restoration to
Lightning: epoch boundaries are checkpointed with
`trainer.save_checkpoint` into `epoch_<n>.ckpt`, and `_restore` just
records the checkpoint path and rewinds NaNsense's schedule/watch
bookkeeping — the next attempt's `trainer.fit(ckpt_path=...)` restores
model, optimizers, schedulers, and loop counters itself. Each attempt
needs a fresh trainer (hence the factory argument): Lightning trainers are
single-use for `fit`, and the jump's teardown has already run. The
session's first-batch `save_epoch_start` call is a no-op here; saves
happen at Lightning's own boundaries, where resume semantics are exact.

Lightning checkpoints don't include global RNG state, so the callback
stashes torch/CUDA states in `on_save_checkpoint` and restores them in
`on_load_checkpoint`. The anchor positions are deliberate: creating a
dataloader iterator draws a seed from the global stream, and a resumed fit
creates its first train iterator eagerly (before `on_train_start`) while a
running fit creates each next epoch's lazily (after
`on_train_epoch_start`). Anchoring both the epoch-0 save (`on_fit_start`)
and the restore (`on_load_checkpoint`) *before any dataloader setup* — and
the other epochs' saves at `on_train_epoch_end`, after which nothing draws
until the next epoch's iterator — keeps the save→draw sequence identical
between an epoch and its replay, which is what makes the replayed
DataLoader shuffling exact. (Lightning's sanity check runs under
`isolate_rng()`, so it never shifts the stream.)

## UI layer

`nansense.ui` is a thin NiceGUI app that reads `Session.snapshot` (plus
`probe_result`, watch, and debug state) and drives the session through its
control methods. It never touches tensors except to render them, and never
touches the model — that invariant belongs to the training thread.

One module per page plus shared support: `app.py` (`serve` + page routes),
`main_page.py`, `stats_page.py`, `weights_page.py`, `experiment_page.py`,
`top_bar.py` (the shared top-bar/step controls and the time-travel,
settings/recording, and step-until dialogs), `share.py` (the Share dialog:
the playground / video / library targets, and the previews the first two
carry — a live frame of the page the playground link opens, zoomed out so
the desktop-first app fits, and a player for the demo video; the frame is
replaced by a note on a locked session, which *is* what that link opens, so
the hosted playground never loads a second copy of itself. The card is a
fixed size — the playground section's, the largest of the three — so picking
a section never resizes the dialog under the pointer; the previewless library
leaves the slack empty, and a window too short for the card scrolls it
(`max-h-full`) rather than clipping it. The video's
Download button — the point of that section, since social platforms want the
file uploaded rather than linked — is answered by a route the module
registers on the app: `<a download>` is honoured same-origin only and the
asset host sends no CORS headers, so the app streams the bytes through
itself with a `Content-Disposition`. Everything below the toggle is built
when the dialog opens and dropped when it closes, so a page load never boots
a hosted demo), `input_panel.py` (the main page's right sidebar), `render.py` + `histograms.py` (pure render/plot
math), `graph.py` (the Mermaid architecture graph), `bin_samples.py`,
`common.py` (small cross-page helpers), `theme.py` (the sizes and colors the
page and the composed still both draw), `static.py` (the CSS/JS blobs), and
`tour.py` (the per-page guided tours: Python step data for every page — a
long tour for the main view, one-to-three-step tours for the subpages —
plus an overlay-JS driver that draws arrows to `data-tour`-tagged
elements; every step is one sentence under 100 characters, naming only
what its arrows ring and using the labels the UI actually prints, since
the reader landed seconds ago (`test_tour.py` enforces both); two of the
main view's steps are the layer card's only written key — which row is
which, and that the diverging colormap runs red-positive to blue-negative,
neither of which the card itself spells out; the main view's tour rides on the layer whose card the page
already shows (`main_page._pick_tour_layer`), scoping its card anchors by
`data-layer` so the node it points at and the card it talks about are the
same layer; on locked sessions each page's tour auto-starts once per browser
— a per-page seen flag set on dismissal, held by the embedding page when it
offers to (`docs/javascripts/playground-embed.js` does, so the hosted
playground's two Spaces — two origins in one docs frame — stop replaying
every tour when the visitor switches demos) and in the app's own
localStorage otherwise — and every page's top-bar
`?` button replays its tour anywhere; the driver brackets each run with
start/end events, which the stats page — whose view-bound steps cycle the
View dropdown — uses to restore the pre-tour view on dismissal unless the
visitor picked a view themselves mid-run; a locked main tour closes on the
one step whose subject is the library rather than the UI, ringing the top
bar's brand mark and — via `host_anchor`, the same postMessage channel as
the seen flags — sending a second arrow out of the frame to the docs
header's "one prompt" call to action, which the embedder locates in the
frame's own coordinates, answers `null` for when it has no such button
(the home page's embed), and leaves to a viewport-width fallback when
nobody answers at all).
Page modules import from the shared modules; `app.py` imports the pages —
the graph is acyclic. The favicon and the top-bar logo come from
`nansense/assets` (the packaged PNG plus `importlib.resources` accessors):
an installed wheel ships only the `nansense` package, so the UI must not
resolve paths relative to the repo checkout. The mark is followed by the
wordmark and a two-word descriptor, and the pair links to the repo: the app
is usually met inside the docs iframe or on a bare Space URL, where nothing
else on screen names it or says it is a library.

**Render contract shared with recording.** `nansense.recording` renders the
same content the pages show, so the render model lives in the pure modules:
recording imports only the public names of `render.py` (`render_strip`,
`render_image`, `render_weight`, `render_patch_grid`, `probe_act_tensor`, …)
and `histograms.py`, never page modules. Those signatures are the contract
between the UI and recording — keep them stable.

`render.py` returns the *pieces*; the layout around them is drawn twice, by the
browser (`common.py`, CSS) and by PIL (`compose.py`), so the sizes and colors
that layout needs live once in **`nansense/ui/theme.py`**: the gutter between a
caption and its image, the label-bar radius, a marker's width and the height
below which its label is unreadable, and a `Marker` per strip kind carrying both
a Tailwind class for the page and a hex color for PIL. A page module that
hardcodes `bg-emerald-500`, or a composer that hardcodes `#10b981`, is how the
two front-ends drift apart — take both from `theme.py` instead. `LABEL_HEIGHT`
stays in `render.py`, since a caption bar's height is also a render-math input
(the legend reserves it).

Composed stills therefore carry the same furniture the page draws: filled
`CHANNEL n` header bars (once per card — `strip_image(..., show_labels=False)`
on the rows below the first, mirroring `_strip_html`), `SAMPLE n` row labels
down a patch grid's left edge, and the colored kind marker beside every strip.
That marker is not decoration: every strip uses the same diverging colormap, so
a frame without it cannot say which row is the gradient. Text is drawn in
DejaVu Sans Mono, resolved out of matplotlib's package data — PIL's built-in
bitmap font is not monospace and has no em dash, which it drew as a tofu box in
every recorded frame.

Render conventions worth knowing before editing `render.py`:

- A `StripRender` is a row of `StripTile`s (one image per channel) plus a
  shared legend — **not** one concatenated picture. Each tile image is encoded
  at the tensor's **native** resolution and upscaled client-side with
  `image-rendering: pixelated` (an 8×8 feature map ships as 64 px, not a
  server-upscaled blob), shown as a `tile_px` square. Legends are the exception
  (rendered at display resolution so their text stays crisp).
- Each tile carries a `CHANNEL n` column caption (`StripTile.label`, weight
  strips included — the tiled axis reads as a channel). `_tile_labels` picks the
  longest form that fits the tile width (full → short `CH n` → bare index),
  uniformly across the strip. The captions render as slate header bars
  (`_column_header_bar`, styled to match the row markers) but only when
  `_strip_html(..., show_labels=True)`: a card draws its *first* strip with the
  headers and stacks the rows below it without (`show_labels=False`), so the
  shared column headers sit once atop the table rather than repeating per row.
- Every strip (activations, gradients, weights) uses one diverging
  blue-white-red colormap; strips are told apart by a labelled colored marker
  bar, never by palette.
- `render_strip` handles `[C,H,W]`, `[F]`, and 2D token shapes
  (`[tokens, dim]`, unflattened onto the input patch grid when `input_hw` is
  threaded in, assuming row-major ViT token order); 4D-and-beyond per-sample
  shapes return `None` and the UI hides them. `[F]` and single 2D heatmaps are
  one uncaptioned tile (bins/pixels aren't channels).
- `render_weight` has no batch axis: it pins every axis not assigned to
  X/Y/tile, then funnels through the same tile machinery; `default_weight_dims`
  gives the conv-kernel / matrix / row defaults.
- `render_patch_grid` (the MIN/MAX galleries) is a 2-D grid of separate cell
  images: a `PatchColumn` per channel, each holding one image per top-N sample
  (`cells`), under a `CHANNEL n` header bar — the UI/recording stack the cells
  with a `PATCH_CELL_GAP` gutter so the grid reads as discrete cells. In the UI
  the grid is a table: `CHANNEL n` column headers across, `SAMPLE n` row labels
  (vertical bars, `_row_label_bar_html`) down the left.
- Every filled label bar (`CHANNEL n` headers, `SAMPLE n` row labels, the
  experiment cell captions) shares `_label_bar_html` / `_row_label_bar_html` —
  white bold mono on a rounded bar, the look of the `_strip_marker` row markers.
  Color carries meaning where it can: experiment `INPUT` is green and
  `ATTRIBUTION` / `OVERLAY` purple, echoing the activation/gradient markers.
- Image encoding is governed by `STRIP_FORMAT` — BMP by default (near-memcpy,
  the right trade for a localhost socket; flip to PNG for an SSH-forwarded UI).

`serve(session, port=, host=)` mounts NiceGUI onto a bare FastAPI app and runs
uvicorn on a **non-daemon background thread**, so the UI outlives the training
script's main thread for post-mortem browsing. `install_signal_handlers` is
patched to a no-op because uvicorn can't register signal handlers off the main
thread. `serve()` no-ops on non-leader ranks and on a disabled session. Once the
server thread is launched, a second **daemon** thread (`_announce_when_ready`)
waits for `server.started` before doing anything user-facing: on a clean bind it
prints the address in a Unicode box spanning the terminal width
(`shutil.get_terminal_size`, so it stands out in the training log) and, unless
`open_browser=False`, opens a focused new browser tab (`new=2`, `autoraise=True`,
a no-op on a headless box). The wait is what makes a concurrent session safe: if
another session already holds the port, uvicorn's bind raises, it logs the
`address already in use` error and `sys.exit`s its own thread, so `started` never
flips — the announcer sees the dead thread (or times out), prints nothing and
opens nothing. No banner promises a URL we don't own, and no tab races to a page
served by the *other* session. `0.0.0.0`/`::` are shown (and opened) as loopback.

**Main page.** One `_LayerView` card per layer, but a card is visible only
while its layer is in the page's **shown** set. In the default `watched`
stats scope, shown ≡ watched (`session.watch`) — a global set, so toggles
propagate across tabs; in the decoupled scopes (`none` / `all`) the shown
set is per-connection state that never touches the session, so each tab
shows its own cards (seeded from the watched set on entering the scope) and
the per-card button reads "Hide" instead of "Unwatch". Either way the
center pane starts empty and points at the diagram. The diagram is
`graph.build_mermaid`, which tries `torch.fx.symbolic_trace` for a real
data-flow graph and falls back to a static module-hierarchy tree when the
model isn't traceable. Clicking a node toggles the shown state;
`sync_watch_ui` diffs the shown set against the connection's last-known set
and pushes only the changes. A 200 ms timer re-renders shown views when
`session.snapshot` or `session.probe_result` changes by identity (a probe
result takes precedence as the render source; its gradient strips are
placeholders, since probes are forward-only). Hidden layers are never
rendered or shipped — that is what keeps large models responsive. Renders fan
out over a shared `ThreadPoolExecutor` (the torch/numpy/PIL work releases the
GIL) into a `_RenderCache` keyed `(name, kind, sample_idx)` and invalidated by
render-source identity, so re-showing a card or a second tab is a dict hit.

The top-bar watch/stats chip shows the shown-layer count behind an eye icon
whose glyph and colour reflect `session.stats_collecting`: green `visibility`
when collecting (scope `watched` or `all`), red `visibility_off` (the slashed
eye of the per-card hide button) when paused (scope `none`).
`sync_stats_icon` runs on init, on toggle, and on the 200 ms tick so a toggle
in one tab shows in every other — but it rewrites the icon only when the
state flips (a guard that also avoids re-adding a tooltip every tick). Its
menu carries *Show all layers* (behind the perf-warning dialog), *Hide all
layers*, *Toggle collecting stats* (`session.toggle_stats_collecting`, the
`none` ↔ previous-scope flip), a *Current batch* submenu listing every
layer (each a `/stats?layer=…&phase=current` anchor), and the shown-layer
list (each a plain-phase `/stats?layer=…` anchor).

The right sidebar (`InputPanel`) shows the selected input plus the Pin /
probe-mode / Perturb controls, and (for a multi-input model) a dropdown
choosing which input to view and perturb — its per-input `mean` / `std` /
`transform` are resolved from the possibly-per-input config via
`resolve_per_input`. The pane renders the selected input through
`render_input_image`: a `C in (1, 3)` image directly, any other `(N, C, H, W)`
through `input_transform` (else a hint names the missing transform), and a flat
`(N, C)` input as a one-row `C`-wide colormapped strip with a scale legend. The
Perturb value control rebuilds to match (`_sync_perturb_control`): a color
picker, one numeric field per channel (with a transform-preview swatch), or a
single value for a flat input. Three non-obvious points: NiceGUI delivers click
coordinates in the image's **native** pixel space (so a flat strip's `image_x`
*is* the channel), perturbations are keyed by `(input_name, sample, index)`, and
"Comparing with original" is **not** a user toggle — `panel.compare` derives
from whether any perturbation exists, so the diff view is active exactly while
they do and the all-zero no-edit diff is unreachable. Pin / mode / perturb
changes call into the session, which republishes a `ProbeResult` the tick loop
picks up like a new snapshot.

The top-bar position label has its own 200 ms timer reading
`session.live_position` (recorded on every batch `__enter__`, independent of
capture), so the epoch/batch counter advances during `step_run` / `detach`
where `snapshot.position` would stay frozen. It passes the schedule's known
totals to `format_position` — `schedule.epochs` and the live phase's
`schedule.phase_count(phase)` — which append an "epoch 0/50 | train batch
0/196" suffix to whichever is known (a fully-lazy schedule shows no epoch
total until `session.epochs(n)`, and no batch total until the phase's count is
learned at the end of the first epoch). A locked session drops the label along
with the step controls: parked training never advances, so it would read as a
fixed position forever — the demo chip carries that meaning instead.

**`/stats` page.** One `_WatchLayerPanel` per watched layer, switchable
between a HISTOGRAM view, a MIN/MAX extreme-patch view, and a GRAPHS
view; a fast timer feeds `session.watch_snapshot()` to the visible view,
gated by `_RefreshGate` to the visualization update cadence — a tick
re-renders only after a new snapshot publish (the settings' "Update
frequency", a pause/step, a one-shot Refresh), a watched-set/phase-list
change, an average-patches Performance flip (it flushes the aggregates
and re-gates the MIN/MAX radio's entries), or training starting/stopping,
so the page updates in step with the main view. A card with nothing to
draw hides every view behind one of two things, never empty plots: the
`_StatusPill` spinner while the numbers are on their way — before the
first refresh lands (it computes off the event loop), or while batches
advance and keep feeding them — and, once waiting can't help, the red
`_no_stats_message` notice, whose unlocked variant is the only one that
advises stepping. That is why the run/stop flip is a gate input: the two
read differently and the next publish can be an epoch away. The HISTOGRAM card leads with a **Statistics** section — one
framed table per phase with activations and gradients as the two value
columns (dead channels activation-only) — above the two histogram plots.
GRAPHS plots each stat (mean/std/median/min/max, plus dead
channels on a secondary count axis for activations) against epoch from
`WatchSnapshot.phase_history`, one Plotly line figure per tensor kind with
legend toggling as the stat selector; its fixed trace set means refreshes
are always in-place restyles, so legend selections and zoom survive. The
view has no "Current batch" phase (a single batch has no epoch series):
`sync_phase_select` drops the entry and swaps such a selection for the
first schedule phase, which also narrows the Layer dropdown back to the
watched layers. Like Current batch, the view is not recordable. Below
the two tensor-kind figures, a **Weights** section adds one figure per
weight tensor from `WatchSnapshot.weight_history` — per-epoch samples the
accumulator captures at each epoch's first watched batch
(`weights_pending` / `update_weights`, keyed `(layer, epoch)` since
weights don't vary by phase; leader-only under DDP, where replicas hold
identical weights). Plots are created lazily as parameters first report
data and hidden for parameter-less layers. A
`/stats?layer=…&view=graphs&scroll=weights&watch=1` deep-link (the weights
page's "Weight graphs" button) opens the view directly and scrolls to that
section once it renders; `watch=1` (`_apply_watch_param`) starts watching the
layer on page open — under the `watched` scope only — so the jump lands on
data without the link needing an `on_click` side effect. The constraint shaping this
page is the **websocket keepalive budget** (~6 s): snapshotting and rendering
run in a worker thread (`asyncio.to_thread`) so the event loop keeps answering
pings, refreshes are single-flight (a toggle landing mid-render marks the pass
dirty rather than queueing one render per click), patch grids are always PNG (a
wide layer's BMP grid is multi-MB and stalls the transport), and
`watch_snapshot(include_patches=False)` skips the patch GPU→CPU copy while the
histogram view is showing.

The Phase dropdown's last entry, **Current batch** (a sentinel value with a
divider drawn above it by an `option` scoped slot — Quasar has no native
per-option separator, and the slot keys off the *label* because NiceGUI sets
each option's `value` to its integer index), switches the data source: the
refresh's worker thread calls `session.current_batch_stats(layers=...)`
instead of `watch_snapshot`, and `_WatchLayerPanel._phase_view` returns its
single snapshot-keyed entry unfiltered. `_selectable_layers` then offers
*every* layer in the Layer dropdown (not just the watched ones), since the
snapshot covers them all. The page opens on the phase training is currently
in when the running aggregates already hold stats for it — scoped to the
`?layer=` link's layer, so a link naming an unwatched layer isn't bounced
to a phase whose Layer dropdown would swap the layer out — and on Current
batch otherwise (`_initial_phase`, backed by `Session.stats_phases`).
`record_view` returns `None` in this mode — the recorders render from the
running accumulators, which it doesn't use.

The histogram view's one load-bearing design choice: routine ticks **restyle
the Plotly figure in place** (`Plotly.update`) — only bar counts change, and an
in-place restyle preserves client-side zoom/pan for free — and the figure is
rebuilt only when the *structure* changes (a phase appears/disappears, a
log-axis toggle flips, or the under/overflow band toggles). A **Per channel**
switch swaps each phase's universal histogram for one channel row, falling back
to the universal histogram where rows are absent (1D layers, collapsed older
epochs); hovering a bar while per-channel samples that `(channel, bin)` cell
from the **last captured snapshot** — the running histogram discards its source
values each batch, so the snapshot is the only population available. The
axis-scaling and bar-clipping heuristics live in `histograms.py`; their
tunables are constants at the top of that module.

A **Show subnormal/overflow** checkbox marks the dtype-aware band edges
(`±finfo.tiny` and `±finfo.max / OVERFLOW_HEADROOM`, matching `debugger`'s
detection) as dotted vertical lines spanning every phase row
(`histograms.under_over_line_positions` / `_add_under_over_lines`). The band is
per-stream: each `TensorStatsSnapshot` carries the source `dtype` (recorded by
`TensorAccumulator` before its fp32 reduction cast, preserved across the
distributed reduce overlay), and the activation/gradient histograms each use
their own. Only edges within the histogram's `1e-9 .. 1e6` span are drawn — so
fp32's bands, which sit off both ends, draw nothing (correctly reading as "no
subnormal/overflow risk at this scale"), while fp16's land in view. The lines are
layout shapes, so a toggle (or the dtype first becoming known) forces a figure
rebuild rather than a restyle. The box is pre-checked whenever the page opens
while an under/overflow issue is active (`_should_show_bands` reads
`session.debug_error`), so any route to `/stats` — a layer card's Stats button,
the numerical-warning dialog's per-row link, or a direct URL — surfaces the
band that issue is about.

**`/weights` page** (`?layer=`). One `_WeightPanel` per name in
`session.layer_weights[layer]`, reading shapes from `model.named_parameters()`
so the controls exist before any snapshot. A top-bar "Weight graphs" link
jumps to the layer's GRAPHS view per-epoch weight series (the
`scroll=weights&watch=1` deep-link above). Each panel renders the weight, its
gradient (same shape → same axis layout), and one strip per tensor-valued
optimizer-state entry when the session has an optimizer (shape-matched entries
reuse the panel's axis controls; 0-dim entries like Adam's `step` join a scalar
line below). Per-axis role selects (X/Y/Tile/Index) auto-demote whichever axis
previously held a role, keeping X/Y/Tile unique. The shared top-bar **Refresh**
button renders nothing itself: it calls `session.request_snapshot()` to arm the
one-shot publish flag (see *On-demand refresh*), and the page's existing timer
renders the resulting snapshot — so there is no separate live-read path to keep
consistent with `_publish_snapshot`.

## MCP server (`nansense.mcp_server`, `nansense.mcp_views`, `nansense.mcp_images`)

The agent-facing front end, served on the UI's own port at `/mcp` over MCP's
streamable-HTTP transport. It is a second reader of the same `Session` — the
threading contract the UI already relies on (lock-free reads of frozen
snapshot dataclasses, control methods synchronized on `_cv`) is exactly what
makes a second controller safe, so the core library needed no changes.

The split mirrors the UI's render/page split. `mcp_views` is pure translation
— `Session` → plain dicts — with no `mcp` import, so the output shapes are
unit-testable without the SDK; `mcp_images` is the same for pictures,
`Session` → PNG bytes; `mcp_server` is the tool registration over both plus
the transport wiring. None of them imports a page module.

**Pictures** (`mcp_images`) come from `nansense.ui.frames`, the shared
per-view renderer the recordings also use, so what an agent sees is what the
page shows. Three things are added on top for the wire. They are encoded as
PNG rather than the browser's BMP: BMP is the right localhost trade
(near-memcpy encode at ~2× the bytes) but an MCP reply base64s those bytes
inside JSON, paying for them twice. They are downscaled past `MAX_SIDE`
(1568px, what a vision model resamples to anyway) with the caveat returned as
*text*, since silently averaging neighbouring channels together would let a
reader mistake the smoothing for data. And a render that produces nothing
returns a reason instead — "no image" and "an image of nothing" are the same
thing on the wire and very different to a reader. `mcp_images` imports
`nansense.ui.*` lazily inside its functions: `nansense.ui.app` imports
`mcp_server` while `nansense.ui.__init__` is still executing.

**Two invariants shape every tool.** First, *never block the event loop*:
uvicorn serves NiceGUI's websockets from the same loop on a ~6 s keepalive
budget, so waiting on the training thread (`wait_until_paused`) and copying
tensors (`current_batch_stats`, `watch_snapshot`) both go through
`asyncio.to_thread`, the discipline the `/stats` page already follows. Second,
*a refusal is an answer*: a locked or closed session silently no-ops its
control methods, so the tools check for that up front — an agent that reads a
no-op as success loops forever.

**Serialization decisions that carry meaning.** JSON has no NaN literal, so
non-finite floats render as the strings `"nan"` / `"inf"` / `"-inf"` rather
than `null`, which would be indistinguishable from "not measured". The
accumulators' scalars deliberately describe the *finite* population only (one
NaN would poison `min`/`max` for good), which means a fully diverged tensor
arrives with `n == 0` — identical to an unused one. `tensor_stats_view`
separates the two by reading the histogram's total, since the histogram counts
every value: it reports `finite_count` / `non_finite_count` and suppresses the
derived scalars when nothing finite was seen. Histograms ship as
`[value, count]` pairs over the populated bins only (211 fixed bins are nearly
all empty on a real layer).

**Per-channel data.** `_channel_view` carries the accumulator's `channel_hists`
onto the wire in the two shapes a reader can act on: the *indices* of the dead
channels (a count cannot be drilled into — `render_bin_samples(channel=…)`
needs the number) and, with `channel=`, one channel's own histogram, which can
be saturated or collapsed while the layer-wide one it sums into looks
unremarkable. The scalars stay tensor-wide in both cases; the accumulator keeps
no per-channel sums, so re-scoping them would invent numbers. The narrowing
itself is `watch.narrow_to_channel`, shared with the `/stats` page's "Per
channel" switch so the clamping and the no-rows fallback cannot drift between
the two front-ends — and the picture's subplot title says `all channels` when a
row could not be narrowed, since the fallback is otherwise invisible. Both `live_position` and the snapshot's position
are always reported: they diverge under `run`/`detach`, and conflating them
reads stale numbers as current. `status_view` also carries `batch_size`, which
bounds the `sample` argument the per-sample tools take — the page has a spinner
whose max tracks it, and a caller without it can only find the end of the batch
by rendering past it. An out-of-range index is called out by name
(`_sample_note`) rather than left as the blank picture it would otherwise
share with "this layer captured nothing"; the size moves during a run, since
the last batch of an epoch is usually short.

**Tools that act, not just read.** Probes, experiments and time travel all
execute on the *training* thread — probes and experiments inside
`_wait_for_proceed`, a jump at the next batch boundary — so a tool that fires
one and returns immediately would report the state before it happened. Each
waits off-loop for its own completion signal: `wait_for_probe` for probes, a
per-seq poll of `experiment_result_for` for experiments (not
`wait_for_experiment`, which waits on the *latest* request and would be
satisfied by a concurrent browser tab's), `wait_until_paused` for a jump. Each
also decides whether waiting is meaningful at all — `_probe_will_run` checks
that something is actually pinned, perturbed or mode-forced *and* that training
is paused, since a free-running session lands its probe at the next capture
rather than now.

Several failure modes are invisible unless deliberately surfaced, and each has
cost a bug:

- A perturbation that doesn't fit its base input is skipped at apply time and
  *stays in the map*, so afterwards nothing distinguishes "your edit was
  dropped" from "someone else's edit landed". The tool validates against the
  probe's own base tensor (`probe.perturbation_fits`, shared with the writer so
  the rules cannot drift) and refuses before recording anything.
- A closed session no-ops differently from a locked one: the probe setters
  still record state, but the pause loop that serves them is gone — and
  `wait_for_probe` returns *immediately* on a closed session, so an unguarded
  wait reads as a completed run. Hence `_probe_refusal` checks both.
- A setter given the value already in force returns early and arms nothing;
  no amount of state inspection afterwards distinguishes that from a probe
  still pending, so `_probe_result` takes the caller's own `expected` flag.
- A recording that captured no frames finalizes to no files, which is
  indistinguishable afterwards from a key that was never recording. Stopping
  therefore reads `statuses()` *before* ending — which is also where the
  `auto_key` lives. Releasing the pinned auto experiment by the *recording*
  key would miss a browser-started recording entirely, since the page keys its
  registration by a per-tab uuid.
- Registering an auto experiment replaces the entry under its key with a *new*
  seq, while a running recording holds the old one in its frozen params. So a
  duplicate `start_recording` is rejected before that mutation, not after.
- Ending and discarding a recording differ only in whether the frames become a
  file, so `stop_recording` and `discard_recording` share
  `_release_recordings`: reading `statuses()` before the manager forgets them
  and releasing the `auto_key`s those views pinned are identical either way,
  and a discard that skipped the release would leave the experiment rerunning
  for the rest of the training run.

**The attribution overlay** (`experiment_frame(overlay=…)`) is the page's
"Overlay on input" switch, reached by `render_experiment` and by an
`"experiment"` recording alike — for a spatial method like Grad-CAM the blend
*is* the readable view, and a heat strip beside the image is one the reader has
to align by eye. Two fallbacks keep it honest: deep dream has no attribution to
blend and says so in the note rather than returning a silently identical
picture, and an input that will not denormalize to an image (`C` outside
`(1, 3)`) falls back to the plain attribution strip — losing the overlay must
not also lose the attribution it was meant to make legible. The ±scale comes
from `render.attribution_vmax` over the *whole* attribution tensor, shared with
the page so one frame of a recording means the same thing as the next.

Two shapes of "empty" also need care. `stack_sections` composes label-only
sections into a perfectly valid picture of nothing but captions — right for a
video frame, wrong for a caller choosing between a picture and an explanation,
so `require_image` lets the MCP side ask for `None` instead. And an all-non-
finite epoch leaves the accumulator's ±inf placeholders in place, which would
report `min` above `max` and a fabricated `std` of 0 for exactly the epoch a
reader most needs to understand.

Statistics computed here rather than by an accumulator (`_tensor_summary`, for
weights and optimizer state) match the accumulators' conventions deliberately:
float64 throughout, so a double-precision gradient near 1e39 is not truncated
into a phantom Inf, and the *population* standard deviation, so a parameter's
`get_weight_stats` number agrees with its `get_stats_history` trend.

**Mounting** (`build_mount`, consumed by `serve`). The transport's Starlette
app is built only to harvest its route — the route is then registered on the
UI's own FastAPI app, and its session-manager lifespan is passed to that app at
construction. Both halves are load-bearing and fail *silently* otherwise: a
mounted sub-app never receives lifespan events (so the session manager would
never start), and Starlette matches routes in order, so a `/mcp` route
registered after `ui.run_with`'s catch-all mount at `/` would surface as a
NiceGUI 404. Sub-mounting instead of lifting the route would also put a
Starlette trailing-slash redirect in front of a POST. NiceGUI wraps whatever
lifespan it finds (`ui_run_with` captures `app.router.lifespan_context` and
composes), so passing ours at construction is enough. On a loopback bind the
transport enables DNS-rebinding protection, so a page on another origin cannot
drive the run.

## Enabled flag (zero-overhead off switch)

`nansense.start(model, ..., enabled=False)` returns a fully inert session,
the intended way to leave NaNsense wiring in a training script and turn it
off with one flag:

- **Construction** skips `capture.try_trace(model)` (the proxy forward pass is the
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

## Locked sessions (shared demos)

`Session.lock()` is the one-way switch behind a publicly hosted playground,
where many anonymous visitors share one session. It forces the stats scope
to `all` (so per-tab show/hide never touches shared state) and makes every
run-control and global-settings method a no-op: `_set_mode` — the choke
point for `stop` / `step_*` / `detach` — returns early, `request_time_travel`
raises `TimeTravelError` (and `time_travel_status` reports why, so the
button renders disabled with the reason), and `watch`/`unwatch`,
`set_stats_scope`, `set_update_frequency`, `set_watch_performance`,
`set_debug_settings`, `disable_debug_check`, `set_auto_run_experiments`,
and `set_experiment_defaults` all refuse. `close()` stays available — it belongs to the hosting script,
which arms its wanted mode (typically `step_run()`) and settings *before*
locking.

Everything per-visitor keeps working: browsing, per-tab shown layers,
and experiments — the latter with their numeric knobs
clamped (`experiments._LOCKED_PARAM_LIMITS`) and the shared queue capped
(`_LOCKED_MAX_QUEUE`; an over-cap request gets a queue-full error result
published under its own seq, which the requesting page polls like any
outcome). The probe surface (`pin_current_batch`, `add_perturbation`,
`set_probe_mode`, and their clears) is refused too — the pinned batch,
perturbations, and forward mode are shared state every visitor sees; a
hosting script can still pin a demo input *before* locking, and the lock
then keeps it pinned. The UI reads `session.locked` to swap the step
controls for a demo notice, hide the Refresh button, the stats pause
toggle, and the input pane's pin/forward-mode/perturb sections, and turn
the settings gear into a "settings are locked" note; enforcement lives in
the `Session` methods, so the UI state is cosmetic.
`render.set_strip_format("PNG")` is the companion knob for internet-facing
deployments — BMP strips are the localhost trade. `examples/playground`
hosts the reference deployments (one spec per demo dataset): each freezes a
moment with `--prepare` and serves it via `load_moment` + `lock` + `park`
(see *Frozen moments* below), arming the demo preferences before the
lock — experiment auto-run off (a page's first experiment still
self-starts, but re-runs take a manual Run, which the tour points out) and any spec-level deep-dream form defaults
(`set_experiment_defaults`, e.g. imagenette's 150 steps / 4 channels).

## Frozen moments (`nansense.moments`)

A *moment* is everything the debugger shows for one batch, saved to one
file as the minimal recipe — the frozen batch's loader item, the model
state, the optimizer state, the watch accumulators behind the HISTOGRAM /
MIN-MAX / GRAPHS views, the watched set, and the schedule totals — and
later rebuilt around a fresh model by *replaying* that batch. It exists for
the locked showcase: train once, freeze, then serve the frozen pause with
no dataset or training loop in the serving process.

**Freezing** is a one-shot armed request in the `_snapshot_request` family:
`Session.freeze_moment(path, phase=, epoch=, batch_idx=)` arms a target
position, `_take_freeze_request` (the same lock-free fast path as
`_take_pending_jump`) consumes it in `_BatchContext.__enter__` on the exact
match, and the matching batch counts as publishing (`_publishes`) — hooks
install and a snapshot publishes whatever the mode, so a detached prepare
run freezes without pausing. `moments.write_moment` runs at `__exit__`
right after `_publish_snapshot`, so the accumulators include the target
batch. It stores the batch's loader item as the replay seed, which the
batch context carries only when given one — `session.batches(...)` passes
each yielded item through automatically; a manual `session.batch(...)`
must pass `item=`, and freezing without one raises `MomentError`. A target
the run never reaches is reported at `close()`. Leader-only under DDP —
the frozen statistics are the leader's shard; freeze a single-process run
when the demo needs exact global numbers.

**The file** is a single `torch.save` payload of plain dicts/lists/tensors —
loadable with `weights_only=True`, nothing pickled by reference — written
via temp file + atomic rename like the epoch cache. It carries the full
`model.state_dict()` (parameters *and* buffers, so the replay and later
experiments run against the exact frozen network), the frozen position and
batch item, the snapshot's optimizer state and hyperparameters (training
history a replay cannot regenerate), the `WatchAccumulator` state
(`state_dict`/`load_state_dict` pairs on `TensorAccumulator` and
`PatchAccumulator`; bucket maps stored as record lists, dtypes by name),
the watched set, the per-channel caps, and `Schedule.state_dict()` (epochs
/ phase order / learned counts — the position label's totals).
Deliberately *not* stored: the snapshot's activation and gradient tensors.
For a deep model at real image sizes those run to gigabytes per batch,
all reproducible from a few megabytes of inputs — storing the recipe
instead of the render is what keeps moment files deployable. Also excluded:
probe pins, perturbations, experiment results, recordings, and the debug
banner — per-visitor or transient state.

**Loading** (`nansense.load_moment(model, path, replay=..., port=...)`)
constructs a normal `Session` around the fresh model (fx trace and name
discovery run as usual), validates the file against it —
`validate_model_state` on the stored state dict plus layer-/input-name
equality, `MomentError` on any mismatch — loads the weights+buffers into
the model, and regenerates the snapshot: capture hooks install, the
caller's `replay(model, batch_item)` runs the training step's forward and
returns the loss, one `backward()` populates the retained gradients, and
`_publish_snapshot` clones the lot exactly as a live publishing batch
would; the stored optimizer state is then spliced into the published
snapshot. The replay runs in train mode (the mode the batch was frozen in;
BatchNorm normalizes by batch statistics either way) and the stored state
dict is re-loaded afterwards, so the transient running-stat updates never
leak into the served buffers. Because the training run stepped the
optimizer before the freeze wrote the weights, the replayed activations
are the stored weights' own — self-consistent with every weight view,
not bit-identical to the pre-step forward the live run displayed.
Determinism assumes no train-time stochastic layers (dropout) and no
autocast at the freeze. The rest installs as plain state: `live_position`,
accumulators (tensors stay on CPU; a restored moment is browse-only,
nothing ever accumulates into it), watched set, a `WatchPerformance`
mirrored from the file's caps (so an equal `configure` can never flush the
restored buckets), and the schedule. The stats scope is set to `none`:
buckets stay browsable while nothing collects. Every view then works
unchanged, because the UI only ever reads `session.snapshot` /
`watch_snapshot()`.

**Patch shortlists.** The extreme-patch buffers are the one watch state
that dwarfs everything else on deep models (per-channel input crops,
whole-image samples, and activation heatmaps, per layer and phase — they
are also training history, so the moment must store them; the file keeps
the raw uint8 payloads + scale tensors, a quarter of the fp32 bytes).
`Session.set_patch_layers([...])` gates patch accumulation to a shortlist
(`None` = every layer, the default) while histogram/min-max/graph
statistics keep covering the full stats scope; `_update_watch_stats`
checks the set at its single `update_patches` call site. The hosted
playgrounds no longer shortlist — uint8 payloads, average grids off, and
the per-spec channel cap keep full-model galleries small — but the knob
remains for models that outgrow even that.

**Parking.** Experiments execute on the pause loop, which normally lives
inside a batch's `_wait_for_proceed`. A moment session drives no batches,
so `Session.park()` provides the loop: it marks the session served (an
explicit "wait indefinitely") and re-enters `_wait_for_proceed` until
`close()`. The hosting script's whole serve path is `load_moment` →
`lock()` → `park()`.

In `examples/playground`, `--prepare` trains under scope `all` (so the
frozen stats cover every layer across the whole run), arms `freeze_moment`
at the last train batch of the last epoch, and skips the final epoch's
validation so the frozen batch is the run's last gradient-carrying one.
Galleries cover the full model; uint8 payloads and the default-off
average grids make that affordable, and the per-spec channel cap and
frozen-batch size are the remaining sizing levers. Serving replays that
batch once at boot (seconds for LeNet, under a minute for the imagenette
ResNet on the free Space's CPUs); the container ships no dataset and no
epoch cache (time travel is disabled under lock anyway).

## Lifecycle summary

```text
nansense.start(model, epochs, phases, enabled=True)
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
        │   capture.install_hooks()     │
        │       │                       │
        │       (user code:             │
        │        zero_grad, forward,    │
        │        backward, step)        │
        │       │                       │
        │       ▼                       │
        │   capture.remove_hooks()      │
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
