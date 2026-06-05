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
- `_schedule` — mutated by `set_schedule()` only.

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

Rendering (PNG strips, histograms, summary stats) happens on the UI
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
  CPU tensors into a `StripRender`: per-tile data PNGs at the tensor's
  *native* resolution plus a legend PNG at display resolution.
  - For per-sample shape `[C, H, W]` each channel becomes one tile PNG,
    downsampled server-side (`area`) only when larger than
    `TILE_SIZE × TILE_SIZE`. The browser upscales each tile to the display
    size via CSS sizing plus `image-rendering: pixelated` — equivalent to
    the old server-side nearest-neighbour interpolation, but an 8×8
    feature map travels as 64 pixels instead of 16k (`_strip_html` in
    `app.py` builds the flex row; tile gaps are `TILE_GAP`-px CSS gaps).
  - For `[F]` the data PNG is a single 1-px-tall heatmap row, downsampled
    to at most `LINEAR_MAX_BINS` bins when `F` is large, stretched
    client-side to `LINEAR_BIN_WIDTH` per bin × `LINEAR_TILE_HEIGHT`.
  - The legend (vertical colorbar with `+x` / `0` / `-x` labels) is the
    exception to native-resolution encoding: it is rendered at display
    resolution into its own PNG so its text stays crisp.
  - Every strip — activations, gradients, and weights alike — uses the
    same diverging blue-white-red colormap; strips are told apart by a
    labelled colored marker bar on each one's left edge (emerald
    ACTIVATIONS / violet GRADIENTS on the layer cards, sky WEIGHT /
    violet GRADIENT on the weights page). PNG `compress_level=1`
    — wire size doesn't matter, encode speed does.
  - Other per-sample shapes return `None`; the UI hides those images.
- `playgrad.ui.render.render_image(tensor, sample_idx, mean=..., std=...)`
  renders the model input as a natural RGB or grayscale PNG at the
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
  GRADIENTS) and a `ui.timer` that, every 200 ms, checks
  `session.pause_count`. If
  it has advanced since the last render, every layer view re-renders
  against the new snapshot, slicing each tensor at the current
  `sample_idx` (driven by a single `ui.number` input in the top bar).
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
  interpolate, numpy colormap, PIL PNG encode) release the GIL, so
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
  `Session`. It builds one `_WatchLayerPanel` per watched module; each
  holds two `_HistPlot`s (activations and gradients) and a one-line stats
  summary above each. A 2-second `ui.timer` calls
  `session.watch_snapshot()` and hands the per-phase stats to every
  `_HistPlot.update`. Routine ticks only change the bar counts (and the
  epoch label), so the plot **restyles the existing figure in place** —
  `Plotly.update(getHtmlElement(id), {y, name}, layout, indices)` run via
  `ui.run_javascript` — rather than replacing it. Restyle leaves the rest
  of the client-side state alone, so legend toggles (a series you clicked
  off) and any zoom/pan survive the refresh for free. The figure is only
  rebuilt (`plot.figure = new_fig; plot.update()`) when the *structure*
  changes — a phase appears/disappears, or a **Log x** / **Log y**
  checkbox flips an axis scale — which `_HistPlot` detects by comparing the
  current `(phases, axis)` signature against the last render. The figures
  use `barmode="overlay"` with per-phase opacity so train/val sit on the
  same axes. Both axes default to linear (the **Log x** / **Log y**
  checkboxes start unchecked): bars sit at their true linear bin centres
  (`_BIN_CENTERS`) with per-bin widths (`_BIN_WIDTHS`), auto-zoomed by
  `_linear_x_range` to the populated bins since the full ±1e6 span is
  mostly empty. With both axes linear (`_use_density`), bar heights are
  densities (`count / bin width`, `_density_heights`) so bar area stays
  proportional to count, and the y-axis is capped by `_density_y_range`
  at the 20th-tallest bar (`_DENSITY_TOP_BINS`) — the near-zero bins are
  so narrow that a few stray values would otherwise stretch the scale by
  orders of magnitude. Refresh ticks in density mode restyle `y` +
  `customdata` (the raw counts shown on hover) and push a `yaxis.range`
  relayout only when the cap actually moved. Checking **Log y** swaps
  the count axis `type` to log so distribution tails stay visible;
  checking **Log x** redraws the bars at evenly spaced bin indices with
  signed-log tick positions computed once by `_x_tick_layout` (powers of
  10 labelled, intermediate edges unlabelled). Either checkbox leaves
  density mode, returning the bars to raw counts.
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
