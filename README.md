<p align="center">
  <img src="https://raw.githubusercontent.com/kongaskristjan/nansense/main/assets/logo_large.png" alt="nansense" width="220">
</p>

<h1 align="center">nansense</h1>

<p align="center"><em>See inside your neural net while it trains.</em></p>

<!-- TODO: replace with a real showcase GIF (assets/showcase.gif): perturb a pixel,
     watch the diff ripple through the layers, then time-travel back an epoch. -->
<p align="center">
  <img src="https://raw.githubusercontent.com/kongaskristjan/nansense/main/assets/showcase.gif" alt="nansense showcase" width="720">
</p>

Hook one `Session` into your PyTorch loop and a web UI opens onto the running
model — activations, gradients, weights, and optimizer state, **live as it
trains**. Pause, step batch-by-batch, and see exactly what every layer is doing.

- **Trace the receptive field** — paint a pixel onto the input and watch the
  change ripple outward as per-layer activation diffs.
- **Spot dead units and vanishing gradients** — per-layer activation and
  gradient distributions with full stats, down to a single channel.
- **See what each neuron learned** — the input patches that drove a channel's
  most extreme activations.
- **Ask a neuron what it wants to see** — deep-dream synthesis plus four Captum
  attribution methods, both re-run as the weights evolve.
- **Watch weights and optimizer state move** — weight and gradient strips, every
  optimizer-state tensor, and the param group's live, scheduler-driven LR.
- **Time-travel** the loop back to any epoch and replay it deterministically;
  **record** any view to MP4 for a timelapse.
- Works with **raw PyTorch, PyTorch Lightning, and multi-GPU DDP**, and turns
  off to a near-zero-overhead no-op for production runs.
- **[A few lines of code](#wire-it-into-your-loop)** wire it into your existing
  training loop.

## What you get

### Trace the receptive field

Paint a pixel onto the input and every layer's strip switches to the **diff**
against the original — watch the change ripple outward layer by layer and see
exactly how far each neuron actually looks. Pin a batch to attribute drift to
training rather than to a changing input.

<!-- TODO: value-focused screenshot — input perturbation + per-layer activation diffs -->
![Trace the receptive field](https://raw.githubusercontent.com/kongaskristjan/nansense/main/assets/view-main.png)

### Spot dead units and vanishing gradients

Per-layer activation and gradient distributions over the most recent epoch, as
signed-log histograms with full stats — drill into a single channel to find the
dying ReLU or the gradient that's quietly collapsing.

<!-- TODO: value-focused screenshot — activation/gradient histograms -->
![Spot dead units and vanishing gradients](https://raw.githubusercontent.com/kongaskristjan/nansense/main/assets/view-watch-histogram.png)

### See what each neuron learned to detect

For any channel, the input patches that drove its most extreme activations —
the strongest evidence of what a feature has specialized in, with an optional
activation heatmap overlaid.

<!-- TODO: value-focused screenshot — top input patches per channel -->
![See what each neuron learned to detect](https://raw.githubusercontent.com/kongaskristjan/nansense/main/assets/view-watch-minmax.png)

### Ask a neuron what it wants to see

Deep-dream gradient ascent synthesizes the input that maximally excites a
channel, streaming live as it forms; alongside it four Captum attribution
methods (Grad-CAM, Occlusion, Neuron Gradient / Integrated Gradients) explain
real samples. Both re-run automatically as the weights evolve.

<!-- TODO: value-focused screenshot — deep dream / attribution -->
![Ask a neuron what it wants to see](https://raw.githubusercontent.com/kongaskristjan/nansense/main/assets/view-experiment.png)

### Watch weights and optimizer state move

Per parameter: the weight strip, its gradient, every optimizer-state tensor
(momentum, Adam moments), and the param group's live, scheduler-driven
hyperparameters.

<!-- TODO: value-focused screenshot — weights + optimizer state strips -->
![Watch weights and optimizer state move](https://raw.githubusercontent.com/kongaskristjan/nansense/main/assets/view-weights.png)

### Rewind, and keep a record

**Time travel** jumps the loop back to any checkpointed epoch and replays it
deterministically (model / optimizer / scheduler / RNG restored) — re-run a
moment that looked off. **Record** any view to MP4, one frame per update, to
keep a timelapse of training.

## Run examples

The examples run with [`uv`](https://docs.astral.sh/uv/), a fast Python package
manager. Install it once with `curl -LsSf https://astral.sh/uv/install.sh | sh`
(or see the [install docs](https://docs.astral.sh/uv/getting-started/installation/)).
It builds an isolated, project-local environment for this repo and won't touch
your system Python or any conda/venv you already use.

Sync the dependency group that matches your hardware:

```bash
uv sync --group cpu        # CPU only
uv sync --group cu126      # NVIDIA CUDA 12.6
uv sync --group cu130      # NVIDIA CUDA 13.0
uv sync --group cu132      # NVIDIA CUDA 13.2
uv sync --group rocm7-2    # AMD ROCm 7.2
```

Then launch any example (each serves the UI on `--nansense-port`):

```bash
uv run examples/standard/main.py --nansense-port 8080
uv run examples/pytorch_lightning/main.py --nansense-port 8080
uv run examples/game_of_life/main.py --nansense-port 8080
uv run examples/audio_keywords/main.py --nansense-port 8080
uv run examples/depth_make3d/main.py --nansense-port 8080
uv run torchrun --nproc_per_node=2 examples/standard/main.py --distributed --nansense-port 8080  # multi-rank DDP
```

Each example is self-contained and downloads its dataset on first run. Open the
printed URL; training pauses on the first batch — drive it from the top bar.

If you hit out-of-memory, lower `--batch-size` (or pass `--dtype bf16`; every
example also takes `--dtype fp16`, which autocasts in fp32-weight mode with no
grad scaling so you can watch underflow happen — see the flag's `--help`).

- **`standard`** — ResNet / ViT / LeNet on CIFAR-10 / MNIST / Imagenette with
  the full wiring (scheduler, time travel, checkpoints). Add `--distributed`
  and launch under `torchrun` for multi-rank DDP.
- **`pytorch_lightning`** — a tiny convnet on MNIST via `NansenseCallback` +
  `fit_with_time_travel`.
- **`game_of_life`** — predict Conway's Game of Life; board-shaped activations
  make perturbation light-cones and deep-dream motifs especially legible.
- **`audio_keywords`** — spoken-keyword ResNet over log-mel spectrograms.
- **`depth_make3d`** — monocular depth (pretrained ResNet encoder + U-Net
  decoder) by transfer learning.

## Use the library

```bash
pip install nansense
```

Install your PyTorch build first (see
[pytorch.org](https://pytorch.org/get-started/locally/)) so your CUDA / ROCm /
CPU choice is preserved: nansense bundles `captum` for the experiment page's
attribution methods, and captum needs torch ≥ 2.3, so a pre-existing torch
keeps `pip` from pulling a default CPU build. `pip install lightning`
additionally enables `nansense.lightning`. Runs on Python 3.10–3.14.

### Wire it into your loop

It's a handful of lines either way — `start()`, wrap your loader, and wrap the
epoch loop in a restorer to opt into time travel.

<!-- These diff images are generated from the snippets in assets/code-examples/
     by `uv run assets/code-examples/render_diffs.py`. They are theme-aware SVGs
     (a single file adapts to light/dark); regenerate them after editing the
     snippets. -->

**Raw PyTorch**

![Raw PyTorch — wiring nansense (and time travel) into a training loop](https://raw.githubusercontent.com/kongaskristjan/nansense/main/assets/code-examples/pytorch_raw.svg)

**PyTorch Lightning** — your `LightningModule` is untouched; a callback drives
the UI and `fit_with_time_travel` wraps a stock `Trainer` so the Time Travel
button works (a factory, because each jump needs a fresh `Trainer`):

![PyTorch Lightning — wiring nansense via NansenseCallback and fit_with_time_travel](https://raw.githubusercontent.com/kongaskristjan/nansense/main/assets/code-examples/pytorch_lightning.svg)

The full `start()` surface:

```python
session = nansense.start(
    model,
    epochs=50, phases={"train": N, "val": M},  # phases: {name: batch_count}
    optimizer=None, scheduler=None, # optional: optimizer-state view; scheduler restored on time travel
    input_mean=None, input_std=None,# optional: denormalize inputs for display
    port=8080,                      # serve the UI here (None = build session, don't serve)
    enabled=True,                   # enabled=False → near-zero-overhead no-op
)

for batch in session.batches(loader, phase="train", epoch=epoch): ...
session.close()
```

**Time travel** is opt-in: wrap the epoch loop in a restorer and the UI's
jump button comes alive.

```python
restorer = session.training_restorer(cache_dir="models/latest")
while restorer.pending():
    with restorer:                      # a UI jump unwinds here and replays
        for epoch in restorer.epochs():
            train_one_epoch(...); validate(...); scheduler.step()
```

**DDP** needs no special wiring: call `nansense.start()` on every rank (pass
the DDP-wrapped model — it's unwrapped automatically). Rank 0 serves the UI and
drives pausing; the other ranks fold their data shard into the watch-page
statistics, and a time-travel jump rewinds every rank in lockstep.

See [`INTERNALS.md`](INTERNALS.md) for how it works under the hood.
