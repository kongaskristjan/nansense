# playgrad

A visualization library for deep learning experiments (work in progress) and a
playground for hand-rolled PyTorch models.

## Layout

- `playgrad/` — the visualization library. Intended to be `pip`-installable;
  contains no training logic. Currently a stub.
- `examples/` — runnable Python examples, each in its own subdirectory and
  fully containing its training logic.
- `tests/` — tests for both the `playgrad` library and the examples; the
  layout mirrors the source tree.

## Setup

```bash
uv sync
```

## CIFAR10 example

The first example is a small CIFAR-style residual convolutional network
trained on CIFAR10 (`examples/cifar10/`).

```bash
uv run python -m examples.cifar10.main --epochs 50
```

Useful flags:

- `--batch-size` (default `256`).
- `--blocks-per-stage` — depth knob; total depth is `6n + 2` (default `3` gives ResNet-20).
- `--lr`, `--momentum`, `--weight-decay` — SGD hyperparameters.
- `--device` — `cpu`, `cuda`, or `mps`. Auto-detected when omitted.
- `--bf16` — wrap forward/loss in `torch.autocast` with `bfloat16` (no `GradScaler` needed).
- `--checkpoint path/to/file.pt` — save best-by-test-accuracy weights.
- `--cache-dir` — directory for time-travel epoch checkpoints (default
  `models/latest`). Every epoch start is checkpointed there, and the UI's
  Time Travel button can jump training back to any of them.
- `--playgrad-port 8080` — launch the playgrad UI on this port. Training
  pauses on the first batch; open the URL to drive it with the step / detach
  controls.
- `--disable-playgrad` — turn playgrad off entirely (`enabled=False`). The
  session becomes a near-zero-overhead no-op, so the loop runs as plain
  training with no UI and no capture machinery.

The script uses SGD with Nesterov momentum, cosine LR annealing, and the
standard CIFAR10 augmentations (random crop with 4-pixel padding + horizontal
flip + normalisation).

### Architecture

`examples/cifar10/resnet.py` defines a pre-activation CIFAR ResNet (ResNet v2,
He et al. 2016) with ResNet-D-style downsampling shortcuts (He et al. 2018):

- 3x3 stem conv into 16 channels.
- Three stages of `PreActBlock`s at widths `(16, 32, 64)`. Each block is
  `BN -> ReLU -> 3x3 conv -> BN -> ReLU -> 3x3 conv`, then adds the shortcut,
  keeping the identity path free of nonlinearities and batch-norms.
- Stage transitions downsample with `stride=2` and use the ResNet-D shortcut:
  a 2x2 average pool followed by a 1x1 conv (no info-losing strided 1x1, no
  extra BN on the shortcut path).
- A final BN + ReLU before global average pool, then a linear classifier.

`resnet20()` is a convenience constructor (`blocks_per_stage=3`,
~270k parameters).

## Using the `playgrad` library

```python
import playgrad

# enabled=False (default True) makes the session a near-zero-overhead no-op:
# no fx trace, batch() does nothing, and the UI is skipped. This lets you
# leave the wiring below in place and toggle the whole UI off with one flag.
session = playgrad.start(
    model,
    epochs=50,
    phases={"train": 196, "val": 40},
    enabled=True,
    # Optional: with an optimizer attached, the weights page also shows each
    # parameter's optimizer state (momentum buffers, Adam moments, ...) and
    # its param group's numeric hyperparameters (live lr, ...). Omit it and
    # the UI looks exactly as without this feature.
    optimizer=optimizer,
    # Optional: time-travel checkpoints then include the scheduler's state,
    # so a jump restores the learning-rate schedule automatically.
    scheduler=scheduler,
    # Optional: serve the UI immediately. Omit `port` (or call
    # playgrad.serve(session, port=...) separately) to stay headless.
    port=8080,
    # Optional: denormalize input images for display (e.g., CIFAR10 stats).
    input_mean=(0.4914, 0.4822, 0.4465),
    input_std=(0.2470, 0.2435, 0.2616),
)

# Optional: opt into time travel. The restorer checkpoints the training
# state (model / optimizer / scheduler / RNG) to `cache_dir` at the start of
# every epoch, and re-enters the loop at `restorer.start_epoch` when the UI
# jumps back. Skip the wrapper (use a plain `for epoch in range(50):`) and
# the UI works as before, just with time travel and caching disabled.
restorer = session.training_restorer(cache_dir=Path("models/latest"))
while restorer.pending():
    with restorer:
        for epoch in restorer.epochs():  # range(start_epoch, 50)
            # session.batches wraps each item in a `session.batch(...)`
            # context; `with session.batch(phase=..., epoch=...):` around the
            # body is the equivalent long form.
            for batch in session.batches(train_loader, phase="train", epoch=epoch):
                optimizer.zero_grad()
                loss = ...
                loss.backward()
                optimizer.step()
            for batch in session.batches(val_loader, phase="val", epoch=epoch):
                ...
            scheduler.step()

session.close()  # UI keeps running so you can browse the last snapshot
```

Loop state that depends on history (`best_acc`, metric curves, ...) should
be initialised inside the `with restorer:` block — a time-travel jump
restarts the block, which resets them along with the rewound timeline.

Open `http://localhost:8080` while training is running. The top bar drives
the session with five "go" buttons — `stop`, `step batch`, `step epoch`,
`step until end`, `step until custom` (opens a dialog where you pick the
target phase / epoch / batch) — `detach` (run unattended without
further pauses), and a blue `time travel` button. Time travel opens a
dialog with a slider over the epochs that have a checkpoint on disk
(epochs without one are unselectable); picking one jumps training back to
that epoch's start (model, optimizer, scheduler, and RNG state restored,
so the replay is deterministic). Text below the slider states the cached
and uncached epoch ranges, and while any epochs are uncached a "Cache
full training run" button runs training to the end, checkpointing every
epoch along the way.
A checkpoint that no longer matches the model (e.g. left behind by a
previous run with a different architecture) is rejected with an error
dialog and no jump happens. Without the `training_restorer` wrapper the
button is grayed out; hovering it explains why. The leading icon button
toggles the architecture pane;
a trailing icon button toggles the "Input Selection" pane. The left pane
shows the module hierarchy as a Mermaid diagram; hovering either a Mermaid
node or a layer card highlights both ends of the pair, and clicking
either side scrolls the *other* pane so the matching element lands at
the top. The centre pane shows one card per
submodule with horizontally-scrollable activation and activation-gradient
strips for the selected sample, both drawn with the same diverging
red/blue colormap and told apart by the labelled marker bar on each
strip's left edge (emerald ACTIVATIONS, violet GRADIENTS); the right
"Input Selection" pane shows the input image for the selected sample at
the top (RGB or grayscale, denormalized with the `input_mean` /
`input_std` passed to `serve()` if any), the "Viewing sample" spinner
right beneath it, and then two compact control sections — "Probe" and
"Perturb" — described below.

The Input Selection pane's "Pin batch" toggle pins the currently displayed
batch as a fixed *probe* input: from then on, every pause re-runs the model
on that same batch (a forward-only "probe run") and the layer cards show
its activations instead of the changing training batch's. Because the
input is held fixed, activation changes across `step batch` / `step epoch`
/ time-travel jumps are attributable to training alone — normally the
displayed batch changes every step with the loader's shuffling, which
makes such comparisons hard. Gradient strips show a placeholder note while
pinned (probes never run backward). A mode toggle — shown only while a
batch is pinned — picks how probe forwards treat train/eval state:
**Eval** (default) switches the whole model to eval so BatchNorm uses its
running stats and dropout is off, **Train** switches it to train, and
**Unchanged** runs with whatever modes the training loop left. In every
mode the probe is side-effect-free: per-module `training` flags and all
buffers (BatchNorm running stats) are restored afterwards, the RNG stream
is forked around the run, and no gradients are produced — training and
time-travel determinism are unaffected. Pinning while paused publishes the
probe immediately; pinning mid-`detach` takes effect at the next pause.
Unpinning returns the page to the live training batch.

The "Perturb" section sits below the pinning controls. With the "Click
to perturb" toggle on, clicking the input image paints the current color —
shown on the swatch button next to the toggle, which opens a color picker —
into the clicked pixel of the viewed sample,
back-transformed into model-input space with the `input_mean` /
`input_std` given to `serve()` — on a *copy* of the probe input. Each
probe then runs twice (original and perturbed, in the same isolation
scope) and the layer cards show the perturbed forward's activations; the
input pane shows the edited image. Turning on "Compare with original"
switches every card to the per-layer activation diff (perturbed −
original) instead — the spatial extent of the nonzero diff at each layer
traces how far the edit propagates through the network, i.e. the
receptive field. Clicks accumulate; the "Compare with original" switch
and a "{n} perturbed pixels" / "Clear" row appear once at least one pixel
is perturbed, and turning the "Click to perturb" toggle off discards all
edits (the image and strips revert to the unperturbed input). Perturbing
works with or without a pinned batch — unpinned, the edits apply to the
current training batch's input at each pause. Edits to pixels outside a
later batch's bounds are skipped rather than erroring.

Each layer card has a "Watch" toggle in its header that marks the
layer as "watched". Watched cards (and the matching architecture
node) get a stronger amber outline that persists across hover. The
top bar carries a small watch chip showing how many layers are
currently selected; clicking it opens a menu with a link to the
deep-dive `/watch` page and shortcuts that scroll both the architecture
pane and the centre pane to each watched layer.

Cards for layers that own parameters also carry a "Weights" button
that opens a per-layer weight viewer at `/weights?layer=...`. The
weight page reuses the main page's stepping controls and epoch/batch
readout — minus the "Viewing sample" spinner, since a weight has no
batch axis — so the displayed weights track the currently paused
batch. It renders one panel per parameter the layer uses — the weight
strip with its gradient strip directly below, both drawn with the same
diverging colormap, marked by the same kind of labelled bars as the
main page (sky WEIGHT, violet GRADIENT), and sharing the panel's axis
controls (the gradient strip shows a placeholder note until a backward
pass has run). When an `optimizer=` was passed to `start()`, each panel
additionally shows one amber-marked strip per tensor-valued optimizer
state entry for that parameter (SGD's `momentum_buffer`, Adam's
`exp_avg` / `exp_avg_sq`, …) using the same axis controls, plus a scalar
line combining 0-dim state entries (Adam's `step`) with the param
group's numeric hyperparameters — `lr` there is the live,
scheduler-driven value. Any optimizer following the `torch.optim` state
convention works; no per-optimizer code is involved. By default a 4D
conv weight
`[out, in, kH, kW]` is shown as conv kernels (kH×kW tiles laid out
across the input channels, with the output channel pinned by index), a
2D weight as a single image, and a 1D weight as a single heatmap row.
Per-dimension selects let you remap which axes become the X, Y, and
tiling (third horizontal) axes; every remaining axis is pinned to a
single index chosen by number. A Refresh button in the top bar reads
the model's current weights and gradients on demand — so it updates
even mid-training in `detach` / `step until end`, where no snapshot is
being published.
`session.layer_weights` exposes the underlying
`layer name -> parameter names` map.

The `/watch` page renders one card per watched layer with two plotly
figures (activations + activation gradients), each showing one stacked
subplot row per phase (train / val) for the most recent epoch — the
rows share the value axis and y-range so the distributions compare
directly without obscuring each other. Each histogram has 211
signed-log bins covering `(-1e6, 1e6)` with bin edges on powers of 10
and at six log-spaced points between them. Two checkboxes in the top
bar — **Log x** and **Log y** — switch the value and probability axes
from the linear default to a log-based scale. With **Log x** unchecked
(the default), bars show probability density (`count / (n * bin
width)`) — on a linear value axis that makes bar area proportional to
the share of values despite the wildly different linear bin widths.
Checking **Log x** switches the bars to plain per-bin probabilities
(`count / n`); per-phase normalization keeps train and val comparable
regardless of how many batches each has seen. While **Log y** is
unchecked, the y-axis is capped so that bars holding 99.5% of the
values stay fully in range, and a single drastically dominant bar per
phase — more than 5x taller than the runner-up, e.g. the exact-zero
spike of a ReLU — is excluded from the scale entirely so it can't
flatten the rest of the distribution. The x-axis is trimmed the same
way: it zooms to the bins holding 99.5% of the values, so a lone
outlier can't stretch the value axis. When a tall near-zero peak plus
a long thin tail still leaves the bars covering less than 5% of the
plot area, the 0.5% clip budget is raised step by step (up to 5%)
until the plot is at least 5% full — and if even the most aggressive
trims can't get there (the distribution spans too many decades for
any linear window, as gradients routinely do), the figure falls back
to the signed-log view on its own, labeled "signed-log scale (auto)"
on the x-axis. Because Plotly's "Autoscale"
would undo these caps (landing on a different scale than the initial
render), the button is removed — "Reset axes" and double-click return
to the intended ranges. Above each figure a table shows one column per
phase (`train ep N`, `val ep N`) and one row per stat: `n`, `mean`,
`std`, histogram-derived `median`, and `min`/`max`. Any layer in `session.layer_names` is
watchable — named modules, fx-traced intermediates (scope-qualified by
their submodule, e.g. `stage1.0.relu1`, `stage1.0.add`), and the graph
input itself (`x`). While at least one layer
is watched, every batch runs through the full capture machinery (fx
interpreter when traceable, root pre-hook plus per-module hooks
otherwise), which roughly doubles activation memory and adds Python
overhead — visualisation is for diagnostics, not for production
training runs.

See `INTERNALS.md` for the architecture overview.

## Tests

```bash
uv run pytest
uv run ty check
```
