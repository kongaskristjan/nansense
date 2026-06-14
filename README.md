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
model: activations, gradients, weights, and optimizer state — live. Pause,
step batch-by-batch, and **time-travel** the loop back to any epoch and replay
it deterministically. Find maximum activation patches for neurons,
run deep dream experiments and measure the receptive field of layers.
It works with raw PyTorch, PyTorch Lightning, and multi-GPU DDP, and turns
off to a near-zero-overhead no-op for production runs.

## Install

```bash
pip install nansense
```

nansense deliberately **does not depend on torch** — install PyTorch yourself
(see [pytorch.org](https://pytorch.org/get-started/locally/)) so your CUDA /
ROCm / CPU build is preserved. Optional extras light up on import: `pip install
captum` adds attribution methods to the experiment page, `pip install
lightning` enables `nansense.lightning`. Runs on Python 3.10–3.14.

## Wire it into your loop

It's a handful of lines either way — one `start()` and wrapping your loader.

**Raw PyTorch**

```diff
  import torch
+ import nansense

  model = MyNet().to(device)
  optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

+ # One call serves a live UI at http://localhost:8080.
+ session = nansense.start(
+     model,
+     epochs=50,
+     phases={"train": len(train_loader), "val": len(val_loader)},
+     optimizer=optimizer,   # optional: weights page shows optimizer state + live LR
+     port=8080,
+ )

  for epoch in range(50):
-     for inputs, targets in train_loader:
+     for inputs, targets in session.batches(train_loader, phase="train", epoch=epoch):
          optimizer.zero_grad()
          loss = criterion(model(inputs), targets)
          loss.backward()
          optimizer.step()

+ session.close()   # UI keeps serving the final snapshot
```

**PyTorch Lightning** — a stock `Trainer`, no changes to your training code:

```diff
  import lightning as L
+ from nansense.lightning import NansenseCallback

+ # `model=` is the attribute path to the network inside your LightningModule.
+ callback = NansenseCallback(port=8080, model="net")
- trainer = L.Trainer(max_epochs=50)
+ trainer = L.Trainer(max_epochs=50, callbacks=[callback])
  trainer.fit(module, datamodule)
```

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
channel, streaming live as it forms; with `captum` installed, four attribution
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

## API in a nutshell

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

## Run the examples (this repo)

```bash
uv sync --group cu130        # NVIDIA CUDA 13 — or: cpu / cu126 / cu132 / rocm7-2
uv run examples/standard/main.py --nansense-port 8080
```

Each example is self-contained and downloads its dataset on first run. Open the
printed URL; training pauses on the first batch — drive it from the top bar.

If you hit out-of-memory, lower `--batch-size` (or add `--bf16` where supported).

- **`standard`** — ResNet / ViT / LeNet on CIFAR-10 / MNIST / Imagenette with
  the full wiring (scheduler, time travel, checkpoints). Add `--distributed`
  and launch under `torchrun` for multi-rank DDP.
- **`pytorch_lightning`** — a tiny convnet on MNIST via `NansenseCallback` +
  `fit_with_time_travel`.
- **`game_of_life`** — predict Conway's Game of Life; board-shaped activations
  make perturbation light-cones and deep-dream motifs especially legible.
- **`audio_keywords`** — spoken-keyword CNN over log-mel spectrograms.
- **`depth_make3d`** — monocular depth (pretrained ResNet encoder + U-Net
  decoder) by transfer learning.

```bash
# multi-rank DDP
uv run torchrun --nproc_per_node=2 examples/standard/main.py --distributed --nansense-port 8080
```

## Development

```bash
uv run pytest && uv run ty check
```

See [`INTERNALS.md`](INTERNALS.md) for how it works under the hood and
[`AGENTS.md`](AGENTS.md) for contributor guidelines.
