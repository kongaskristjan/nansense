# Getting started

The fastest way to try nansense is to run one of the bundled examples — they download their datasets and pretrained networks automatically. To add nansense to your own training loop instead, install the library and see the [Wiring guide](wiring.md).

## Run the examples

The examples run with [uv](https://docs.astral.sh/uv/getting-started/installation), a fast Python package manager. `uv` does not pollute your other Python environments, and automatically installs the necessary packages when running a script.

```bash
# Install uv:
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Clone the repository, then pick the dependency group that matches your hardware and pass it as `--group`:

| Group | Hardware |
| --- | --- |
| `cpu` | No GPU, CPU-only, any platform |
| `cuda-legacy` | Older NVIDIA GPUs: Maxwell, Pascal, Volta (CUDA 12.6) |
| `cuda` | Current NVIDIA GPUs: Turing through Blackwell (CUDA 13.0) |
| `rocm` | AMD GPUs (ROCm 7.2) |

Then launch any example; the requirements, datasets and any pretrained networks are downloaded automatically, and the UI serves on `--nansense-port`.

```bash
# `examples/standard/main.py` is a good starting point for mnist, cifar10 and
# imagenette. Use `--dataset` and `--model` for different combinations.
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

A focused browser tab opens automatically at the boxed URL it prints (open it yourself if your environment has no browser); training pauses on the first batch. Drive it from the top bar — see the [UI guide](ui.md).

!!! tip "Memory and speed"
    If you hit out-of-memory errors, lower `--batch-size`. If training is slow and you have GPU VRAM left, increase `--batch-size`. Both memory and training speed can be improved with `--dtype bf16` (older GPUs don't support it).

## Install the library

```bash
pip install nansense
```

!!! note "Install torch first"
    Install your PyTorch build first (see [pytorch.org](https://pytorch.org/get-started/locally/)) so your CUDA / ROCm / CPU choice is preserved: nansense bundles `captum` for the experiment page's attribution methods, and captum needs torch ≥ 2.3, so a pre-existing torch keeps `pip` from pulling a default CPU build. `pip install lightning` additionally enables `nansense.lightning`. Runs on Python 3.10–3.14.

Wiring nansense into a training loop is a few lines:

```python
import nansense

session = nansense.start(model, optimizer=optimizer, port=8080)
for epoch in session.epochs(50):
    with session.restore_point():
        for inputs, targets in session.batches(train_dl, phase="train"):
            ...  # your usual training step
session.close()
```

The [Wiring guide](wiring.md) walks through this for raw PyTorch and PyTorch Lightning, including time travel and distributed training.
