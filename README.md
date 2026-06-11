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
hardware-specific build — CUDA, ROCm, or CPU — is preserved. The attribution
experiments on the experiment page additionally need
`pip install nansense[captum]`, and the Lightning integration
`pip install nansense[lightning]`.

## Running the examples (this repository)

```bash
uv sync --extra cpu    # CPU-only machines (smallest download)
uv sync --extra cu130  # NVIDIA GPU with CUDA 13

uv run python -m examples.vision.main --nansense-port 8080
```

PyTorch is installed through one of several mutually exclusive extras, so
pick the one matching your hardware: `cpu` (works everywhere), `cu126` /
`cu130` / `cu132` (NVIDIA CUDA, Linux/Windows), or `rocm7-2` (AMD ROCm,
Linux).

A plain `uv sync` (no extra) installs no torch via the extras and falls back
to the default PyPI wheels through transitive dependencies — always pass an
extra. All variants are pinned in the same `uv.lock`, so switching extras is
reproducible and doesn't re-lock.

Open `http://localhost:8080`. Training pauses on the first batch; drive it
from the top bar (step batch / epoch / custom, detach, time travel).

Available examples:

- `examples.mnist_linear.main` — a single linear layer on MNIST with the
  minimal nansense wiring (no scheduler, no time travel).
- `examples.mnist_lenet.main` — LeNet-5 on MNIST: SGD + momentum, basic
  augmentation, and the full wiring (scheduler, time travel, checkpoints).
- `examples.vision.main` — a small pre-activation ResNet (default), a
  deeper five-stage variant (`--model resnet_deep`), or a simple ViT
  (`--model vit`) on CIFAR10 (default) or Imagenette
  (`--dataset imagenette`), trained with AdamW + a cosine schedule.

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

## PyTorch Lightning

With the `lightning` package installed (`uv add lightning` or
`pip install nansense[lightning]`), a stock `Trainer` gets the full
experience through a callback — no changes to the training code:

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
the hand-written loop's. Supported: automatic optimization and
epoch-boundary validation, including `check_val_every_n_epoch > 1`.
Rejected with a clear error: mid-epoch validation
(`val_check_interval < 1.0` or step-driven) and unsized dataloaders — the
schedule is declared up-front. Metric loggers cannot time-travel: after a
jump they see the replayed epochs again.

## Views

### Main view

The landing page. The top bar drives the training loop: stop, step batch /
epoch / custom, detach (run without pauses), and time travel (jump back to
any checkpointed epoch). The left pane shows the architecture as a diagram;
clicking a node toggles that layer's card in the center pane — visible is
synonymous with watched, so each shown card carries activation and gradient
strips for the selected sample plus an "Unwatch" button that hides it
again. The center pane starts empty and only visible layers are rendered
and sent to the browser, which keeps large models responsive. The top-bar
eye menu jumps to watched layers, watches all layers at once (behind a
performance warning), or clears every watch.

The right "Input Selection" pane shows the input image and sample picker.
"Pin batch" freezes the current batch as a probe input that is re-run on
every pause, so activation changes are attributable to training rather than
to the batch changing. "Click to perturb" paints pixels onto the input;
"Compare with original" then shows per-layer activation diffs, tracing how
far the edit propagates (the receptive field).

![Main view](https://raw.githubusercontent.com/kongaskristjan/nansense/main/assets/view-main.png)

### Watch

Layers watched on the main page (diagram clicks or the eye menu) also feed
the deep-dive `/watch` page, which renders one card per watched layer. The MIN/MAX view (the
default) shows the input patches that drove the layer's most extreme
activations: per channel, the top input crops around the largest/smallest
spatial activation and whole inputs ranked by spatial mean, with an optional
activation-heatmap overlay.

![Watch page, min/max view](https://raw.githubusercontent.com/kongaskristjan/nansense/main/assets/view-watch-minmax.png)

The HISTOGRAM view shows activation and activation-gradient distributions
over the most recent epoch as signed-log histograms with a stats table (`n`,
`mean`, `std`, `median`, `min`/`max`); a phase dropdown switches between
train/val, and Log x / Log y checkboxes handle distributions spanning many
decades.

![Watch page, histogram view](https://raw.githubusercontent.com/kongaskristjan/nansense/main/assets/view-watch-histogram.png)

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

![Experiment page](https://raw.githubusercontent.com/kongaskristjan/nansense/main/assets/view-experiment.png)

## Tests

```bash
uv run pytest
uv run ty check
```
