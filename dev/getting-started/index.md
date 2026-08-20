# Getting started

The fastest way to try NaNsense is to run one of the bundled examples — they download their datasets and pretrained networks automatically. To add NaNsense to your own training loop instead, paste [one prompt](https://kongaskristjan.github.io/nansense/dev/integrate/index.md) into your coding agent, or install the library and follow the [Wiring guide](https://kongaskristjan.github.io/nansense/dev/wiring/index.md) yourself.

## Run the examples

The examples run with [uv](https://docs.astral.sh/uv/getting-started/installation), a fast Python package manager. `uv` does not pollute your other Python environments, and automatically installs Python and the necessary packages when running a script. Datasets and any pretrained networks are downloaded automatically too, and the UI serves on `--nansense-port`.

```
# Install uv (Windows: https://docs.astral.sh/uv/getting-started/installation):
curl -LsSf https://astral.sh/uv/install.sh | sh

git clone https://github.com/kongaskristjan/nansense
cd nansense

# --group: cpu | cuda (NVIDIA) | cuda-legacy (pre-Turing NVIDIA) | rocm (AMD)
# `examples/standard/main.py` is a good starting point; `--dataset` and
# `--model` switch between mnist, cifar10 and imagenette.
uv run --group cpu examples/standard/main.py --nansense-port 8080
```

`--group cpu` is the torch build that works everywhere. If you have a GPU, swap it for the group matching your hardware — nothing else about the commands changes:

| Group         | Hardware                                                  |
| ------------- | --------------------------------------------------------- |
| `cpu`         | No GPU, CPU-only, any platform                            |
| `cuda-legacy` | Older NVIDIA GPUs: Maxwell, Pascal, Volta (CUDA 12.6)     |
| `cuda`        | Current NVIDIA GPUs: Turing through Blackwell (CUDA 13.0) |
| `rocm`        | AMD GPUs (ROCm 7.2)                                       |

The other bundled examples run the same way:

```
# More exotic, but harder to interpret tasks:
uv run --group cpu examples/game_of_life/main.py --nansense-port 8080
uv run --group cpu examples/audio_keywords/main.py --nansense-port 8080
uv run --group cpu examples/depth_make3d/main.py --nansense-port 8080

# Multi-input demo: a 5-channel image + a flat stats vector. Shows the input
# pane's input picker, the `input_transform` for non-RGB images, and the
# flat-input strip.
uv run --group cpu examples/multimodal/main.py --nansense-port 8080
```

A focused browser tab opens automatically at the boxed URL it prints (open it yourself if your environment has no browser); training pauses on the first batch. Drive it from the top bar — see the [UI guide](https://kongaskristjan.github.io/nansense/dev/ui/index.md).

The first run is the slow one

A cold start installs torch and downloads the example's dataset before the UI can come up — a few minutes, and the example says so as it starts. Everything is cached under `--data-dir` (`./data` by default), so later runs skip straight to training. `examples/depth_make3d/main.py` is the outlier: Make3D is 914 MB from a slow host.

Memory and speed

If you hit out-of-memory errors, lower `--batch-size`. If training is slow and you have GPU VRAM left, increase `--batch-size`. Both memory and training speed can be improved with `--dtype bf16` (older GPUs don't support it).

## Install the library

```
pip install nansense
```

Install torch first

Install your PyTorch build first (see [pytorch.org](https://pytorch.org/get-started/locally/)) so your CUDA / ROCm / CPU choice is preserved: NaNsense bundles `captum` for the experiment page's attribution methods, and captum needs torch ≥ 2.3, so a pre-existing torch keeps `pip` from pulling a default CPU build. `pip install lightning` additionally enables `nansense.lightning`. Runs on Python 3.10–3.14.

Wiring NaNsense into a training loop is a few lines:

```
import nansense

session = nansense.start(model, optimizer=optimizer, port=8080)
for epoch in session.epochs(50):
    with session.restore_point():
        for inputs, targets in session.batches(train_dl, phase="train"):
            ...  # your usual training step
session.close()
```

The [Wiring guide](https://kongaskristjan.github.io/nansense/dev/wiring/index.md) walks through this for raw PyTorch and PyTorch Lightning, including time travel and distributed training — or let a coding agent do the wiring via [Integrate with one prompt](https://kongaskristjan.github.io/nansense/dev/integrate/index.md).
