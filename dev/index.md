# nansense

*Don't guess why your neural network fails to learn. Instead, have a look inside.*

[](https://github.com/user-attachments/assets/d7ee7ecc-4828-4655-866d-a220174c2b44)

*Nansense* is a PyTorch debugger that visualizes activations, gradients, weights, optimizer state and various statistics. You can **pause, step batch-by-batch, and time-travel to a different epoch while training**, and see exactly what every layer is doing.

[🕹️ Try the Playground](https://kongaskristjan.github.io/nansense/dev/playground/index.md) [✨ Integrate with one prompt](https://kongaskristjan.github.io/nansense/dev/integrate/index.md)

Here's how *nansense* can help:

- **See what is actually going on.** [Visualize activations and gradients](https://kongaskristjan.github.io/nansense/dev/showcase/#visualize-activations-and-gradients-throughout-training), [find image patches with minimal or maximal activation for a given channel](https://kongaskristjan.github.io/nansense/dev/showcase/#minmax-activation-patches) and [simulate what each neuron is searching for (deep dream)](https://kongaskristjan.github.io/nansense/dev/showcase/#simulate-what-a-neuron-is-searching-for-deep-dream).
- **Spot optimization bottlenecks.** [Discover insufficient receptive fields](https://kongaskristjan.github.io/nansense/dev/showcase/#measure-the-receptive-field-of-a-neuron), [measure neuron death](https://kongaskristjan.github.io/nansense/dev/showcase/#investigate-dead-neurons), [discover padding artifacts](https://kongaskristjan.github.io/nansense/dev/showcase/#padding-artifacts) and [spot gradient underflow](https://kongaskristjan.github.io/nansense/dev/showcase/#spot-gradient-underflow).

The [Showcase](https://kongaskristjan.github.io/nansense/dev/showcase/index.md) walks through all of these with real screenshots. [Getting started](https://kongaskristjan.github.io/nansense/dev/getting-started/index.md) runs an example in minutes, the [UI guide](https://kongaskristjan.github.io/nansense/dev/ui/index.md) tours every page, and the [Wiring guide](https://kongaskristjan.github.io/nansense/dev/wiring/index.md) adds nansense to your own training loop — it's just a few lines of code.

## How is this different from wandb or TensorBoard?

Loggers like Weights & Biases and TensorBoard record scalar curves of loss and accuracy that you scroll through after the run. Nansense works inside the live training loop instead: it pauses so you can step batch-by-batch and time-travel while inspecting the activations, gradients, weights and optimizer state of every layer. You can even run experiments like deep dream or Grad-CAM on the paused model to probe what a given neuron has learned.

Persisting all this data on disk is infeasible, as a single batch of activations and gradients can easily be several gigabytes. Nansense sidesteps that by pausing and inspecting the tensors on demand, instead of writing everything to disk.
