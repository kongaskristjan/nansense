# Complete nansense!

A visualization library for deep learning experiments: hook a `Session` into
your PyTorch training loop and inspect activations, gradients, weights, and
more from a web UI — pausing, stepping, and time-traveling the loop as it
runs. See `INTERNALS.md` for how it works under the hood.

## Installation

```bash
pip install nansense
```

nansense deliberately does not depend on torch: install PyTorch separately
(see [pytorch.org](https://pytorch.org/get-started/locally/)) so your
hardware-specific build — CUDA, ROCm, or CPU — is preserved. The same goes
for the optional integrations: with `pip install captum` the experiment page
offers the Captum attribution methods (they are hidden otherwise), and with
`pip install lightning` the `nansense.lightning` module becomes importable.

## Running the examples (this repository)

```bash
uv sync --group cpu    # CPU-only machines (smallest download)
uv sync --group cu130  # NVIDIA GPU with CUDA 13

uv run python -m examples.vision.main --nansense-port 8080
```

PyTorch is installed through one of several mutually exclusive dependency
groups, so pick the one matching your hardware: `cpu` (works everywhere),
`cu126` / `cu130` / `cu132` (NVIDIA CUDA, Linux/Windows), or `rocm7-2`
(AMD ROCm, Linux). Groups are local to this repository and never published,
which is what keeps the PyPI package torch-free.

Always pass a group — a plain `uv sync` installs torch only transitively
(via captum/lightning in the dev group) from the default PyPI wheels. All
variants are pinned in the same `uv.lock`, so switching groups is
reproducible and doesn't re-lock.

Open `http://localhost:8080`. Training pauses on the first batch; drive it
from the top bar (stop, the Step button, time travel).

Available examples:

- `examples.mnist_linear.main` — a single linear layer on MNIST with the
  minimal nansense wiring (no scheduler, no time travel).
- `examples.lightning_mnist.main` — a tiny convnet on MNIST trained with
  PyTorch Lightning: `NansenseCallback` + `fit_with_time_travel`.
- `examples.vision.main` — a small pre-activation ResNet (default), a
  deeper five-stage variant (`--model resnet_deep`), a simple ViT
  (`--model vit`), or LeNet-5 (`--model lenet`) on CIFAR10 (default),
  MNIST (`--dataset mnist`, scaled to 32x32), or Imagenette
  (`--dataset imagenette`), trained with AdamW + a cosine schedule and
  the full wiring (scheduler, time travel, checkpoints).
- `examples.ddp_mnist.main` — a small MLP on MNIST trained with
  DistributedDataParallel across two (or more) ranks; launch with
  `uv run torchrun --nproc_per_node=2 -m examples.ddp_mnist.main`
  (falls back from NCCL to CPU/gloo when there are fewer GPUs than
  ranks, so it runs anywhere).

## Minimal example

```python
import nansense

session = nansense.start(
    model,
    epochs=10,
    phases={"train": len(train_loader)},
    port=8080,
)

for epoch in range(10):
    for batch in session.batches(train_loader, phase="train", epoch=epoch):
        ...  # forward / backward / optimizer step

session.close()  # UI keeps serving the last snapshot
```

## Full example

With an optimizer (weights page shows optimizer state and the live learning
rate), a scheduler (time-travel jumps restore the LR schedule), input
denormalization for display, and time travel:

```python
from pathlib import Path

import nansense

session = nansense.start(
    model,
    epochs=50,
    phases={"train": len(train_loader), "val": len(val_loader)},
    optimizer=optimizer,
    scheduler=scheduler,
    port=8080,
    input_mean=(0.4914, 0.4822, 0.4465),
    input_std=(0.2470, 0.2435, 0.2616),
)

# Time travel: every epoch start is checkpointed to cache_dir. A jump from
# the UI re-enters the loop at the chosen epoch with model / optimizer /
# scheduler / RNG state restored, so the replay is deterministic.
restorer = session.training_restorer(cache_dir=Path("models/latest"))
while restorer.pending():
    with restorer:
        best_acc = 0.0  # history-dependent state goes inside: a jump resets it
        for epoch in restorer.epochs():
            for batch in session.batches(train_loader, phase="train", epoch=epoch):
                optimizer.zero_grad()
                loss = criterion(model(batch[0]), batch[1])
                loss.backward()
                optimizer.step()
            for batch in session.batches(val_loader, phase="val", epoch=epoch):
                ...  # evaluation
            scheduler.step()

session.close()
```

`enabled=False` on `nansense.start()` turns the whole thing into a
near-zero-overhead no-op, so the wiring can stay in place for plain training
runs. The runnable version of this loop is `examples/vision/main.py`.

## Distributed training (DDP)

Multi-rank `DistributedDataParallel` runs need no special wiring: call
`nansense.start()` on every rank — passing the DDP-wrapped model is fine,
it is unwrapped automatically — and wrap batches as usual:

```python
model = DistributedDataParallel(build_model().to(device))
session = nansense.start(
    model,
    epochs=10,
    phases={"train": len(train_loader)},  # per-rank batch counts
    port=8080,
)
```

The ranks then play different roles:

- **Rank 0** serves the UI and behaves like a single-process session:
  snapshots, pausing/stepping, probes, experiments, recordings. Pausing
  pauses the whole run — the other ranks block at their next batch start
  until rank 0 resumes. (Pause longer than the process group's collective
  timeout — 10–30 min by default — and the other ranks time out; raise
  `init_process_group(timeout=...)` for long interactive sessions.)
- **Every other rank** skips the UI and never pauses on its own, but
  follows rank 0's watched-layer set and folds its data shard into the
  watch page's statistics at every visualization update — histograms and
  the stats table cover the global batch, not just rank 0's shard.

Per-rank views (snapshot strips, the input pane, extreme-input patches,
histogram bar samples) show rank 0's shard. Every rank must run the same
batch structure — `DistributedSampler` on each phase's loader gives equal
per-rank batch counts. Time travel is not supported in distributed mode.
The runnable version is `examples/ddp_mnist/main.py`:

```bash
uv run torchrun --nproc_per_node=2 -m examples.ddp_mnist.main --nansense-port 8080
```

## PyTorch Lightning

With the `lightning` package installed (`pip install lightning`), a stock
`Trainer` gets the full experience through a callback — no changes to the
training code:

```python
import lightning as L

from nansense.lightning import NansenseCallback

callback = NansenseCallback(
    port=8080,
    model="net",  # attribute path to the network inside the LightningModule
    input_mean=(0.4914, 0.4822, 0.4465),
    input_std=(0.2470, 0.2435, 0.2616),
)
trainer = L.Trainer(max_epochs=50, callbacks=[callback])
trainer.fit(module, datamodule)

callback.session  # the live Session (None until fit starts)
```

`model=` is recommended whenever the LightningModule wraps its layers in a
submodule: nansense then traces and probes the actual network instead of
the module wrapper. `enabled=False` is the same zero-overhead off switch as
on `nansense.start()`.

For time travel, the retry loop around `trainer.fit` must live outside the
callback, so it ships as a wrapper. Pass a trainer *factory* — each jump
re-resumes from a Lightning checkpoint on a fresh trainer:

```python
from nansense.lightning import fit_with_time_travel

fit_with_time_travel(
    lambda: L.Trainer(max_epochs=50),
    module,
    callback=callback,
    datamodule=datamodule,
    cache_dir=Path("models/latest"),
)
```

Epoch boundaries are checkpointed via `trainer.save_checkpoint` (with RNG
states stashed alongside), and a jump re-invokes
`trainer.fit(ckpt_path=...)`, so the replay is exactly as deterministic as
the hand-written loop's. The runnable version of this wiring is
`examples/lightning_mnist/main.py`. Supported: automatic optimization and
epoch-boundary validation, including `check_val_every_n_epoch > 1`.
Rejected with a clear error: mid-epoch validation
(`val_check_interval < 1.0` or step-driven) and unsized dataloaders — the
schedule is declared up-front. Metric loggers cannot time-travel: after a
jump they see the replayed epochs again.

## Views

### Main view

The landing page. The top bar drives the training loop: stop, a split Step
Batch button (clicking it steps one batch; its dropdown offers step epoch,
step until end, and step custom — pick a phase/epoch/batch to pause at),
and time travel (jump back to any checkpointed epoch); on the bar's right
side, a gear button opens the settings dialog (update frequency and MP4
recording — see below). The left pane shows the architecture as a diagram;
clicking a node toggles that layer's card in the center pane — visible is
synonymous with watched, so each shown card carries activation and gradient
strips for the selected sample plus an "Unwatch" button that hides it
again. The center pane starts empty and only visible layers are rendered
and sent to the browser, which keeps large models responsive. Hovering a
diagram node or a card header shows the layer's hyperparameters in a
tooltip — `Conv2d(3, 64, kernel_size=(3, 3), stride=(2, 2), ...)` — built
from each module's `extra_repr()` (custom modules can override it to
surface their own knobs) or, for functional ops, the call's literal
arguments. The top-bar eye menu jumps to watched layers, watches all
layers at once (behind a performance warning), or clears every watch.

The right "Input Selection" pane shows the input image and sample picker.
"Pin batch" freezes the current batch as a probe input that is re-run on
every pause, so activation changes are attributable to training rather than
to the batch changing. "Click to perturb" paints pixels onto the input; as
soon as at least one pixel is perturbed, the layer strips switch to per-layer
activation diffs against the original input (a note below the controls says
so), tracing how far the edit propagates (the receptive field).

The side panes — architecture and input here, the controls panes on the
watch and experiment pages — are resizable: drag the thin handle between a
pane and the center content, or double-click it to restore the default
width. Pane sizes are remembered for the rest of the browser session.

![Main view](https://raw.githubusercontent.com/kongaskristjan/nansense/main/assets/view-main.png)

### Watch

Layers watched on the main page (diagram clicks or the eye menu) also feed
the deep-dive `/watch` page, which renders one card per watched layer.

The HISTOGRAM view (the default) shows activation and activation-gradient
distributions over the most recent epoch as signed-log histograms with a
stats table (`n`, `mean`, `std`, `median`, `min`/`max`); a phase dropdown
switches between train/val, and Log x / Log y checkboxes handle
distributions spanning many decades. A "Retain axes" checkbox keeps the
current axis ranges when toggling Log x / Log y or switching phase (instead
of auto-fitting to the data) — handy for comparing phases or scales on a
fixed frame. Each histogram has a "Per channel"
switch that narrows it to a single channel (stepped with an index spinner);
while per-channel, hovering a bar shows a few random input samples whose
values fell in that bar — drawn from the last captured batch only, since
the running histogram's source values are discarded every batch (the strip
names the batch it sampled from).

![Watch page, histogram view](https://raw.githubusercontent.com/kongaskristjan/nansense/main/assets/view-watch-histogram.png)

The MIN/MAX view shows the input patches that drove the layer's most
extreme activations: per channel, the top input crops around the
largest/smallest spatial activation and whole inputs ranked by spatial
mean, with an optional activation-heatmap overlay. Only the "Max pixel"
grid starts enabled; the other three have their own checkboxes.

![Watch page, min/max view](https://raw.githubusercontent.com/kongaskristjan/nansense/main/assets/view-watch-minmax.png)

### Weights

Each parameterized layer card has a "Weights" button opening
`/weights?layer=...`. It renders one panel per parameter: the weight strip
with its gradient strip below, plus — when an `optimizer=` was passed to
`start()` — one strip per tensor-valued optimizer state entry (momentum
buffer, Adam moments, …) and the param group's live hyperparameters.
Per-dimension selects remap which tensor axes become X, Y, and tiling (a 4D
conv weight defaults to kernel tiles); a Refresh button re-reads weights on
demand, even mid-training.

![Weights page](https://raw.githubusercontent.com/kongaskristjan/nansense/main/assets/view-weights.png)

### Experiment

Each layer card's "Experiment" button opens `/experiment?layer=...`, which
runs per-layer experiments on the paused training thread without side
effects on training or time-travel determinism. Deep Dream runs gradient
ascent on a channel's mean activation over a batch of inputs — by default
fresh noise shaped like the network's real input, different on every Run —
with configurable regularizers, streaming the evolving images live. Four
Captum attribution methods — Grad-CAM, Neuron Gradient, Neuron Integrated
Gradients, and Occlusion — render attributions next to the input sample
they explain.

Run also registers the experiment for automatic re-runs: while the page
stays open (or its view is being recorded), the experiment re-executes at
every visualization update with the *same* random seed, so the result
tracks the evolving weights instead of the changing noise.

![Experiment page](https://raw.githubusercontent.com/kongaskristjan/nansense/main/assets/view-experiment.png)

### Update frequency

The "Update frequency" section of the settings dialog (the gear button in
every page's top bar) sets how often
all visualizations refresh while training runs freely: every nth epoch (the
default, n=1) or every nth batch, optionally counting only one phase's
batches. A frequency update publishes a fresh snapshot, re-runs the probe
(pinned batch / perturbations) and any registered experiments — without
pausing training, in every mode including detach. Stopping or stepping
still refreshes everything, exactly as before.

### Recording

The "Recording" section of the settings dialog (the gear button in every
page's top bar; a red badge on the gear carries the count of active
recordings) records visualizations to MP4 — one file per view, one frame
per visualization update (the frequency above, not per user step), written
to `nansense_recordings/<timestamp>/` in the training process's working
directory at 10 fps. Each frame carries a banner with the training
position it was captured at (e.g. `epoch 0 | train batch 0`).

The section offers a red "Record this view" button for the page you're on,
the list of currently recorded views — each can be ended (finalize the
MP4) or deleted (discard it) individually — and end-all / delete-all
buttons. Recordable views:

- **Main view** — the input image plus every watched layer's activation and
  gradient strips packed into a single video; a pinned batch and
  perturbations are respected exactly like on the page.
- **Weights** — one video per recorded layer: weight, gradient, and
  optimizer-state strips under the page's current axis layout.
- **Watch · histograms** — server-side matplotlib re-renders of the
  activation/gradient histograms for the selected phase.
- **Watch · MIN/MAX** — the enabled patch grids; pixel (crop) and average
  (whole input) grids have different image sizes, so they record into
  separate `*_pixel.mp4` / `*_average.mp4` files.
- **Experiment** — the page's experiment result, re-run automatically at
  every update with a fixed random seed.

A view's parameters are frozen while it records: the matching page controls
are disabled (and unwatching layers is refused while a watch view records),
so the video stays consistent from the first frame to the last. The update
frequency itself is likewise locked while any recording is active.

## Tests

```bash
uv run pytest
uv run ty check
```
