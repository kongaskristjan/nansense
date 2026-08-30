# Getting started

The fastest way to try NaNsense is the standard example. To add it to your own training loop, use the [integration prompt](integrate.md) or follow the [Wiring guide](wiring.md).

## Run the examples

The examples use [uv](https://docs.astral.sh/uv/getting-started/installation), which installs Python and the required packages for you. Datasets and pretrained models are downloaded when needed.

```bash
# Install uv (Windows: https://docs.astral.sh/uv/getting-started/installation):
curl -LsSf https://astral.sh/uv/install.sh | sh

git clone https://github.com/kongaskristjan/nansense
cd nansense

# --group: cpu | cuda (NVIDIA) | cuda-legacy (pre-Turing NVIDIA) | rocm (AMD)
uv run --group cpu examples/standard/main.py --nansense-port 8080
```

`--group cpu` works on any machine. For a GPU build, use the matching group:

| Group | Hardware |
| --- | --- |
| `cpu` | No GPU, CPU-only, any platform |
| `cuda-legacy` | Older NVIDIA GPUs: Maxwell, Pascal, Volta (CUDA 12.6) |
| `cuda` | Current NVIDIA GPUs: Turing through Blackwell (CUDA 13.0) |
| `rocm` | AMD GPUs (ROCm 7.2) |

The other bundled examples run the same way:

```bash
# Other examples
uv run --group cpu examples/game_of_life/main.py --nansense-port 8080
uv run --group cpu examples/audio_keywords/main.py --nansense-port 8080
uv run --group cpu examples/depth_make3d/main.py --nansense-port 8080

# Multiple model inputs
uv run --group cpu examples/multimodal/main.py --nansense-port 8080
```

A browser tab opens and training pauses on the first batch. Use the top bar to run or step through training; see the [UI guide](ui.md).

!!! note "The first run is the slow one"
    The first run installs packages and downloads data, so it can take a few minutes. Later runs use the cache in `--data-dir` (`./data` by default).

!!! tip "Memory and speed"
    If you hit out-of-memory errors, lower `--batch-size`. If training is slow and you have GPU VRAM left, increase `--batch-size`. Both memory and training speed can be improved with `--dtype bf16` (older GPUs don't support it).

## Install the library

```bash
pip install nansense
```

!!! note "Install torch first"
    Install the [PyTorch build for your hardware](https://pytorch.org/get-started/locally/) first. NaNsense supports Python 3.10–3.14 and PyTorch 2.3 or newer. Install `lightning` as well to use the PyTorch Lightning integration.

Wiring NaNsense into a training loop is a few lines:

```python
import nansense

session = nansense.start(model, optimizer=optimizer, port=8080)
for epoch in session.epochs(50):
    with session.restore_point():
        for inputs, targets in session.batches(train_dl, phase="train"):
            ...  # your usual training step
session.close()
```

The [Wiring guide](wiring.md) covers raw PyTorch, PyTorch Lightning, time travel, and distributed training.
