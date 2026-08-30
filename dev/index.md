# NaNsense

*See what your neural network is doing while it trains.*

Live playground Easy: MNIST · LeNet-5 Advanced: Imagenette · ResNet

The playground may take a minute or two to start. The video below gives you a quick tour while it loads.

[](https://github.com/user-attachments/assets/b15f7dc5-1a6f-44d2-81fa-a97daafa89c6)

*NaNsense* is a visual debugger for PyTorch. Pause training, step through batches, revisit earlier epochs, and inspect activations, gradients, weights, and optimizer state.

[🕹️ Try the Playground](https://kongaskristjan.github.io/nansense/dev/playground/index.md) [✨ Integrate with one prompt](https://kongaskristjan.github.io/nansense/dev/integrate/index.md) [🤖 Debug with a coding agent](https://kongaskristjan.github.io/nansense/dev/mcp/index.md)

- **Understand learned features.** [See activations and gradients](https://kongaskristjan.github.io/nansense/dev/showcase/#visualize-activations-and-gradients-throughout-training), [find the inputs that excite a channel](https://kongaskristjan.github.io/nansense/dev/showcase/#minmax-activation-patches), and [use deep dream to reveal what it has learned](https://kongaskristjan.github.io/nansense/dev/showcase/#simulate-what-a-neuron-is-searching-for-deep-dream).
- **Find training problems.** Spot [small receptive fields](https://kongaskristjan.github.io/nansense/dev/showcase/#measure-the-receptive-field-of-a-neuron), [dead channels](https://kongaskristjan.github.io/nansense/dev/showcase/#investigate-dead-neurons), [padding artifacts](https://kongaskristjan.github.io/nansense/dev/showcase/#padding-artifacts), and [gradient underflow](https://kongaskristjan.github.io/nansense/dev/showcase/#spot-gradient-underflow).

See the [Showcase](https://kongaskristjan.github.io/nansense/dev/showcase/index.md) for screenshots, [Getting started](https://kongaskristjan.github.io/nansense/dev/getting-started/index.md) to run an example, and the [Wiring guide](https://kongaskristjan.github.io/nansense/dev/wiring/index.md) to add NaNsense to your project. A coding agent can also inspect the same session through [MCP](https://kongaskristjan.github.io/nansense/dev/mcp/index.md).

## How is this different from wandb or TensorBoard?

Experiment trackers show that a run went wrong. NaNsense helps you find where it went wrong inside the network—for example, a layer with dead channels or underflowing gradients.
