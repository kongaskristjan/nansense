<p align="center">
  <img src="assets/logo_large.png" alt="nansense" width="220">
</p>

<h1 align="center">nansense</h1>

<p align="center"><em>Don't guess why your neural network fails to learn. Instead, have a look inside.</em></p>

<!-- TODO: replace with a real showcase GIF (assets/showcase.gif): perturb a pixel,
     watch the diff ripple through the layers, then time-travel back an epoch. -->
<p align="center">
  <img src="assets/showcase.gif" alt="nansense showcase" width="720">
</p>

Hook a `Session` into your PyTorch loop and a web UI opens onto the running
model with activations, gradients, weights, and optimizer state, live as it
trains. **Pause, step batch-by-batch, and time-travel to a different epoch**, and see exactly what every layer is doing. Here's what you can do:

- **Deepen your intuition** — [investigate activations and gradients](), [find min/max activation patches]() and [simulate what each neuron is searching for]()
- **Spot optimization bottlenecks** — [discover insufficient receptive fields](), [measure neuron death]() and [fix augmentation padding artifacts]()
- **Investigate failure modes** — [spot gradient underflow]() and [record weight and optimizer dynamics to understand training instability]()

[Try out the pre-made examples]() or wire it into your own training loop. You're just a `pip install nansense` and a few lines of code away. Here's an example integration in [raw PyTorch]() and in [Lightning]().

## Showcase

### Activations and gradients throughout training

TBD

### Min/max activation patches

TBD

### Simulate what a neuron is searching for (deep dream)

TBD

### Measure receptive field of a neuron

TBD

### Investigate dead neurons

TBD

### Augmentation padding artifacts

TBD

### Gradient underflow

TBD

### Training instability

TBD

## Run examples

The examples run with [uv](https://docs.astral.sh/uv/getting-started/installation), a fast Python package manager. `uv` does not pollute your other Python environments, and automatically installs the necessary packages when running a script.

Sync the dependency group that matches your hardware:

```bash
# Install uv:
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Then launch any example. The requirements, datasets and any pretrained networks are downloaded automatically. UI servs on `--nansense-port`.

```bash
# The `examples/standard/main.py` script is a good starting point for mnist, cifar10 and imagenette. Use `--dataset` and `--model` for different combinations.
uv run examples/standard/main.py --nansense-port 8080

# More exotic, but harder to interpret tasks:
uv run examples/game_of_life/main.py --nansense-port 8080
uv run examples/audio_keywords/main.py --nansense-port 8080
uv run examples/depth_make3d/main.py --nansense-port 8080
```

A browser tab opens automatically at the boxed URL it prints (open it yourself if your environment has no browser); training pauses on the first batch. Drive it from the top bar. See [UI Tutorial]() for more info.

If you hit out-of-memory errors, lower `--batch-size` (or pass `--dtype bf16`).

## UI tutorial

TBD intro (1 paragraph)

### Watching layers and viewing stats

TBD paragraph explaining watching layers and how it's related to collecting stats. A mention about update frequency. One paragraph for looking at histograms and max patches.

### Perturbing and pinning inputs

One short paragraph for perturbing inputs (1 pixel changed means diff is shown). One paragraph for pinning inputs.

### Running experiments

One short paragraph how to open and configure experiments.

### Recording videos

One paragraph on recording videos.

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

### Wire it into your loop: raw PyTorch

```python
import torch
import nansense

# Init model, optimizer, dataloaders
model = ...
optimizer = ...
train_dl, val_dl = ...

# Nansense needs to know the total number of batches in each phase
phases = {"train": len(train_dl), "val": len(val_dl)}

# Setup UI
session = nansense.start(model, epochs=50, phases=phases, optimizer=optimizer, port=8080, enabled=True)

# Restorer while/with loop: wrap the epoch loop so the UI can back off and restart training at a different epoch for time-travel
restorer = session.training_restorer(cache_dir="models/latest")
while restorer.pending():
    with restorer:
        # time-travel aware epochs iteration: use just like `for epoch in range(50)`
        for epoch in restorer.epochs():
            # Training batch iteration (matched phase="train")
            for input, targets in session.batches(train_dl, phase="train", epoch=epoch):
                optimizer.zero_grad()
                loss = criterion(model(inputs), targets)
                loss.backward()
                optimizer.step()
            # Validation batch iteration ...

# Close the UI
session.close()
```

See [Python API]() for more information.

### Wire it into your loop: PyTorch Lightning

```python
import lightning as L
from nansense.lightning import NansenseCallback, fit_with_time_travel

# Pytorch Lightning modules
module = ...
datamodule = ...

# `model="net"` is the attribute path to the attribute path to the network inside your LightningModule. Eg. module.net
callback = NansenseCallback(port=8080, model="net", enabled=True)

# Time-travel: trainer factor enables restarting the training at different epochs
trainer_factor = lambda: L.Trainer(max_epochs=50)
fit_with_time_travel(trainer_factory, module, datamodule=datamodule, callback=callback)
```

See [Python API]() for more information.

### Python API

TBD. Fill it in, especially the `start`, `training_restorer` and `NansenseCallback`

The full `start()` surface, NansenseCallback, etc: TBD.

**DDP** needs no special wiring: call `nansense.start()` on every rank (pass
the DDP-wrapped model — it's unwrapped automatically). Rank 0 serves the UI and
drives pausing, everything else should just work. DDP support is currently experimental.

See [`INTERNALS.md`](INTERNALS.md) for how it works under the hood (it's long).
