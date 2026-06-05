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
# no fx trace, batch() does nothing, and serve() is skipped. This lets you
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
)
playgrad.serve(
    session,
    port=8080,
    # Optional: denormalize input images for display (e.g., CIFAR10 stats).
    input_mean=(0.4914, 0.4822, 0.4465),
    input_std=(0.2470, 0.2435, 0.2616),
)

for epoch in range(50):
    for batch in train_loader:
        with session.batch(phase="train", epoch=epoch):
            optimizer.zero_grad()
            loss = ...
            loss.backward()
            optimizer.step()
    for batch in val_loader:
        with session.batch(phase="val", epoch=epoch):
            ...

session.close()  # UI keeps running so you can browse the last snapshot
```

Open `http://localhost:8080` while training is running. The top bar drives
the session with five "go" buttons — `stop`, `step batch`, `step epoch`,
`step until end`, `step until custom` (opens a dialog where you pick the
target phase / epoch / batch) — and `detach` (run unattended without
further pauses). The leading icon button toggles the architecture pane;
a trailing icon button toggles the input-image pane. The left pane shows
the module hierarchy as a Mermaid diagram; hovering either a Mermaid
node or a layer card highlights both ends of the pair, and clicking
either side scrolls the *other* pane so the matching element lands at
the top. The centre pane shows one card per
submodule with horizontally-scrollable activation and activation-gradient
strips for the selected sample, both drawn with the same diverging
red/blue colormap and told apart by the labelled marker bar on each
strip's left edge (emerald ACTIVATIONS, violet GRADIENTS); the right
pane shows the input image for that sample (RGB or grayscale),
denormalized with the `input_mean` / `input_std` passed to `serve()` if
any.

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
histograms (activations + activation gradients) overlaid by phase
(train / val) for the most recent epoch. Each histogram has 211
signed-log bins covering `(-1e6, 1e6)` with bin edges on powers of 10
and at six log-spaced points between them. Two checkboxes in the top
bar — **Log x** and **Log y** — switch the value and count axes from
the linear default to a log-based scale. With both unchecked (the
default), bars show density (count / bin width) instead of raw counts —
on linear axes that makes bar area proportional to count despite the
wildly different linear bin widths — and the y-axis is capped at the
20th-tallest bar so the ultra-narrow near-zero bins can't blow out the
scale. Above each histogram a
one-line summary shows `n`, `mean`, `std`, histogram-derived
`median`, and `min`/`max`. Any layer in `session.layer_names` is
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
