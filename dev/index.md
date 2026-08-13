# NaNsense

*Don't guess why your neural network fails to learn. Instead, have a look inside.*

Live playground Easy: MNIST · LeNet-5 Advanced: Imagenette · ResNet

The playground starts a real training run, so it can take 1–2 minutes to boot — watch the video below while it loads.

[](https://github.com/user-attachments/assets/07063057-7891-4c52-89be-ced10012c6f4)

*NaNsense* is a PyTorch debugger that visualizes activations, gradients, weights, optimizer state and various statistics. You can **pause, step batch-by-batch, and time-travel to a different epoch while training**, and see exactly what every layer is doing.

[🕹️ Try the Playground](https://kongaskristjan.github.io/nansense/dev/playground/index.md) [✨ Integrate with one prompt](https://kongaskristjan.github.io/nansense/dev/integrate/index.md) [🤖 Debug with a coding agent](https://kongaskristjan.github.io/nansense/dev/mcp/index.md)

Here's how *NaNsense* can help:

- **See what is actually going on.** [Visualize activations and gradients](https://kongaskristjan.github.io/nansense/dev/showcase/#visualize-activations-and-gradients-throughout-training), [find image patches with minimal or maximal activation for a given channel](https://kongaskristjan.github.io/nansense/dev/showcase/#minmax-activation-patches) and [simulate what each neuron is searching for (deep dream)](https://kongaskristjan.github.io/nansense/dev/showcase/#simulate-what-a-neuron-is-searching-for-deep-dream).
- **Spot optimization bottlenecks.** [Discover insufficient receptive fields](https://kongaskristjan.github.io/nansense/dev/showcase/#measure-the-receptive-field-of-a-neuron), [measure neuron death](https://kongaskristjan.github.io/nansense/dev/showcase/#investigate-dead-neurons), [discover padding artifacts](https://kongaskristjan.github.io/nansense/dev/showcase/#padding-artifacts) and [spot gradient underflow](https://kongaskristjan.github.io/nansense/dev/showcase/#spot-gradient-underflow).

The [Showcase](https://kongaskristjan.github.io/nansense/dev/showcase/index.md) walks through all of these with real screenshots. [Getting started](https://kongaskristjan.github.io/nansense/dev/getting-started/index.md) runs an example in minutes, the [UI guide](https://kongaskristjan.github.io/nansense/dev/ui/index.md) tours every page, and the [Wiring guide](https://kongaskristjan.github.io/nansense/dev/wiring/index.md) adds NaNsense to your own training loop — it's just a few lines of code. It also speaks [MCP](https://kongaskristjan.github.io/nansense/dev/mcp/index.md), so a coding agent can drive the debugger itself and look at the same views you do.

## How is this different from wandb or TensorBoard?

Loggers record external metrics — the loss and accuracy curves you scroll through after the run. NaNsense is focused on understanding the internals of the network. A logger tells you *that* the loss stopped falling; NaNsense shows you *why* — say, half a layer's channels died after epoch three, or fp16 gradients are underflowing.
