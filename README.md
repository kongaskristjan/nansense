<h1 align="center">
  <img src="nansense/assets/logo_small.png" alt="NaNsense logo" height="36" align="middle"> NaNsense
</h1>

<p align="center"><em>Don't guess why your neural network fails to learn. Instead, have a look inside.</em></p>

https://github.com/user-attachments/assets/d7ee7ecc-4828-4655-866d-a220174c2b44

<p align="center"><em>The main NaNsense UI: click layers to see activations and gradients, measure receptive fields, collect per-channel statistics, and run deep dream mid-training.</em></p>

<p align="center">
  <a href="https://kongaskristjan.github.io/nansense/dev/playground/"><b>🕹️ Try the Playground</b></a> — inspect a trained network in your browser, no install<br>
  <a href="https://kongaskristjan.github.io/nansense/dev/integrate/"><b>✨ Integrate with one prompt</b></a> — a coding agent wires it into your training loop<br>
  <a href="https://kongaskristjan.github.io/nansense/dev/mcp/"><b>🤖 Debug with a coding agent</b></a> — it speaks MCP, so the agent drives the debugger<br>
  <a href="https://kongaskristjan.github.io/nansense/"><b>📚 Documentation</b></a> — showcase, guides and the full Python API (also as <a href="https://kongaskristjan.github.io/nansense/llms.txt">llms.txt</a>)
</p>

*NaNsense* is a PyTorch debugger that visualizes activations, gradients, weights, optimizer state and various statistics. You can **pause, step batch-by-batch, and time-travel to a different epoch while training**, and see exactly what every layer is doing.

Here's how *NaNsense* can help:

- **See what is actually going on**. Visualize activations and gradients, find image patches with minimal or maximal activation for a given channel, and simulate what each neuron is searching for (deep dream)
- **Spot optimization bottlenecks**. Discover insufficient receptive fields, measure neuron death, discover padding artifacts and spot gradient underflow

Unlike wandb or TensorBoard, which log external metrics (loss, accuracy) to scroll through after the run, NaNsense is about understanding the internals of the network. A logger tells you *that* the loss stopped falling; NaNsense shows you *why* — say, a layer's channels dying or fp16 gradients underflowing.

## Try it

Clone the repository and run a bundled example with [uv](https://docs.astral.sh/uv/getting-started/installation) — datasets and pretrained networks download automatically, and a browser tab opens with the UI:

```bash
# --group: cpu | cuda (NVIDIA) | cuda-legacy (pre-Turing NVIDIA) | rocm (AMD)
uv run --group cuda examples/standard/main.py --nansense-port 8080
```

[Getting started](https://kongaskristjan.github.io/nansense/dev/getting-started/) lists all the examples and explains the hardware groups.

## Use it in your project

```bash
pip install nansense
```

Wiring it into a training loop is a few lines of code: paste [one prompt](https://kongaskristjan.github.io/nansense/dev/integrate/) into your coding agent, or follow the [Wiring guide](https://kongaskristjan.github.io/nansense/dev/wiring/) yourself.

See [`INTERNALS.md`](INTERNALS.md) for how it works under the hood (it's long).
