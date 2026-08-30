<h1 align="center">
  <img src="nansense/assets/logo_small.png" alt="NaNsense logo" height="36" align="middle"> NaNsense
</h1>

<p align="center"><em>See what your neural network is doing while it trains.</em></p>

https://github.com/user-attachments/assets/b15f7dc5-1a6f-44d2-81fa-a97daafa89c6

<p align="center"><em>Follow a channel through training, see what it responds to, and trace how one changed pixel moves through the network.</em></p>

<p align="center">
  <img src="docs/images/readme_hero.png" alt="The NaNsense main view: architecture graph on the left, layer cards showing activations and gradients in the centre, input panel on the right">
</p>

<p align="center"><em>Click a layer to inspect its activations and gradients. Step through batches, revisit earlier epochs, or edit the input and see what changes.</em></p>

- 🕹️ **[Try the Playground](https://kongaskristjan.github.io/nansense/dev/playground/)** — explore a trained network in your browser
- ✨ **[Integrate with one prompt](https://kongaskristjan.github.io/nansense/dev/integrate/)** — let a coding agent add NaNsense to your training loop
- 🤖 **[Debug with a coding agent](https://kongaskristjan.github.io/nansense/dev/mcp/)** — let an agent inspect the same run through MCP
- 📚 **[Documentation](https://kongaskristjan.github.io/nansense/)** — guides, examples, and the Python API

*NaNsense* is a visual debugger for PyTorch. Pause training, step through batches, revisit earlier epochs, and inspect activations, gradients, weights, and optimizer state.

- **Understand learned features.** See activations and gradients, find the inputs that excite a channel, and use deep dream to reveal what it has learned.
- **Find training problems.** Spot dead channels, padding artifacts, small receptive fields, and gradient underflow.

Experiment trackers show that a run went wrong. NaNsense helps you find where it went wrong inside the network—for example, a layer with dead channels or underflowing gradients.

## Try it

Clone the repository and run the standard example with [uv](https://docs.astral.sh/uv/getting-started/installation). The first run may take a few minutes while it downloads Python, packages, and data.

```bash
# Install uv (Windows: https://docs.astral.sh/uv/getting-started/installation):
curl -LsSf https://astral.sh/uv/install.sh | sh

git clone https://github.com/kongaskristjan/nansense
cd nansense

uv run --group cpu examples/standard/main.py --nansense-port 8080
```

[Getting started](https://kongaskristjan.github.io/nansense/dev/getting-started/) lists all the examples and explains the hardware groups.

## Use it in your project

```bash
pip install nansense
```

Paste [this prompt](https://kongaskristjan.github.io/nansense/dev/integrate/) into your coding agent, or follow the [Wiring guide](https://kongaskristjan.github.io/nansense/dev/wiring/).

See [`INTERNALS.md`](INTERNALS.md) for the architecture.
