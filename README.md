<h1 align="center">
  <img src="assets/logo/logo_small.png" alt="nansense logo" height="36" align="middle"> nansense
</h1>

<p align="center"><em>Don't guess why your neural network fails to learn. Instead, have a look inside.</em></p>

https://github.com/user-attachments/assets/d7ee7ecc-4828-4655-866d-a220174c2b44

<p align="center"><em>Video 1. The main nansense UI. Clicking the layers in the architecture shows activation/gradient maps. The size of the receptive field can be measured by perturbing the input image and measuring the diff. Watched layers collect histograms and min/max activating pixel statistics for interpretability. You can even run deep dream at any point during the training run to visualize what exactly each neuron is looking for.</em></p>

*Nansense* is a PyTorch debugger that visualizes activations, gradients, weights, optimizer state and various statistics. You can **pause, step batch-by-batch, and time-travel to a different epoch while training**, and see exactly what every layer is doing.

Here's how *nansense* can help:

- **See what is actually going on**. Visualize activations and gradients, find image patches with minimal or maximal activation for a given channel, and simulate what each neuron is searching for (deep dream)
- **Spot optimization bottlenecks**. Discover insufficient receptive fields, measure neuron death, discover padding artifacts and spot gradient underflow

**📚 Documentation: [kongaskristjan.github.io/nansense](https://kongaskristjan.github.io/nansense/)** — a visual showcase, a guide to every UI page, and the full Python API. Every docs page is also served as plain Markdown for AI assistants ([llms.txt](https://kongaskristjan.github.io/nansense/llms.txt)).

## How is this different from wandb or TensorBoard?

Loggers like Weights & Biases and TensorBoard record scalar curves of loss and accuracy that you scroll through after the run. Nansense works inside the live training loop instead: it pauses so you can step batch-by-batch and time-travel while inspecting the activations, gradients, weights and optimizer state of every layer. You can even run experiments like deep dream or Grad-CAM on the paused model to probe what a given neuron has learned.

Persisting all this data on disk is infeasible, as a single batch of activations and gradients can easily be several gigabytes. Nansense sidesteps that by pausing and inspecting the tensors on demand, instead of writing everything to disk.

## Run examples

The examples run with [uv](https://docs.astral.sh/uv/getting-started/installation), a fast Python package manager. `uv` does not pollute your other Python environments, and automatically installs the necessary packages when running a script.

```bash
# Install uv:
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Pick the dependency group that matches your hardware and pass it as `--group`:

| Group | Hardware |
| --- | --- |
| `cpu` | No GPU, CPU-only, any platform |
| `cuda-legacy` | Older NVIDIA GPUs: Maxwell, Pascal, Volta (CUDA 12.6) |
| `cuda` | Current NVIDIA GPUs: Turing through Blackwell (CUDA 13.0) |
| `rocm` | AMD GPUs (ROCm 7.2) |

Then launch any example; the requirements, datasets and any pretrained networks are downloaded automatically, and the UI serves on `--nansense-port`.

```bash
# `examples/standard/main.py` is a good starting point for mnist, cifar10 and imagenette. Use `--dataset` and `--model` for different combinations.
uv run --group [group] examples/standard/main.py --nansense-port 8080

# More exotic, but harder to interpret tasks:
uv run --group [group] examples/game_of_life/main.py --nansense-port 8080
uv run --group [group] examples/audio_keywords/main.py --nansense-port 8080
uv run --group [group] examples/depth_make3d/main.py --nansense-port 8080

# Multi-input demo: a 5-channel image + a flat stats vector. Shows the input
# pane's input picker, the `input_transform` for non-RGB images, and the
# flat-input strip.
uv run --group [group] examples/multimodal/main.py --nansense-port 8080
```

A focused browser tab opens automatically at the boxed URL it prints (open it yourself if your environment has no browser); training pauses on the first batch. Drive it from the top bar. See the [UI guide](https://kongaskristjan.github.io/nansense/ui/) for more info.

If you hit out-of-memory errors, lower `--batch-size`. If training is slow and you have GPU VRAM left, increase `--batch-size`. Both memory and training speed can be improved with `--dtype bf16` (older GPUs don't support it).

## Use the library

```bash
pip install nansense
```

> **Note:** Install your PyTorch build first (see
> [pytorch.org](https://pytorch.org/get-started/locally/)) so your CUDA / ROCm /
> CPU choice is preserved: nansense bundles `captum` for the experiment page's
> attribution methods, and captum needs torch ≥ 2.3, so a pre-existing torch
> keeps `pip` from pulling a default CPU wheel. `pip install lightning`
> additionally enables `nansense.lightning`. Runs on Python 3.10–3.14.

Wire it into a raw PyTorch loop:

```python
import torch
import nansense

# Init model, optimizer, criterion, dataloaders
model = ...
optimizer = ...
criterion = ...
train_dl, val_dl = ...

# Setup UI. The schedule is discovered as you train (phase names and batch
# counts are learned from the loop below); no need to declare them up front.
session = nansense.start(model, optimizer=optimizer, port=8080, enabled=True)

# Time travel needs an epoch cache. `session.epochs(50)` iterates like
# `range(50)` but checkpoints each epoch start; wrap each iteration's body in
# `with session.restore_point():` so a UI-requested jump can unwind it and
# re-enter at a different epoch. Without this loop, training runs once through
# and the Time Travel button is disabled.
for epoch in session.epochs(50, cache_dir=".nansense_cache"):
    with session.restore_point():
        # Training batch iteration
        for inputs, targets in session.batches(train_dl, phase="train"):
            optimizer.zero_grad()  # keep zero_grad at the beginning of the batch
            loss = criterion(model(inputs), targets)  # as nansense reads .grad when
            loss.backward()  # the batch exits, so zeroing after step() would
            optimizer.step()  # leave the weight-gradient views empty.
        # Validation batch iteration ...

# Close the UI (the served page stays up for post-mortem browsing)
session.close()
```

The [Wiring guide](https://kongaskristjan.github.io/nansense/wiring/) covers the rest: the PyTorch Lightning integration, displaying non-RGB and multi-input models, distributed training (DDP), and the full [API reference](https://kongaskristjan.github.io/nansense/api/).

See [`INTERNALS.md`](INTERNALS.md) for how it works under the hood (it's long).
