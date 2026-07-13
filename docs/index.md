# nansense

*Don't guess why your neural network fails to learn. Instead, have a look inside.*

<video controls muted style="max-width: 100%;" src="https://github.com/user-attachments/assets/d7ee7ecc-4828-4655-866d-a220174c2b44"></video>

*Nansense* is a PyTorch debugger that visualizes activations, gradients, weights, optimizer state and various statistics. You can **pause, step batch-by-batch, and time-travel to a different epoch while training**, and see exactly what every layer is doing.

[🕹️ Try the Playground](playground.md){ .md-button .md-button--primary }
[✨ Integrate with one prompt](integrate.md){ .md-button .md-button--primary }

Here's how *nansense* can help:

- **See what is actually going on.** [Visualize activations and gradients](showcase.md#visualize-activations-and-gradients-throughout-training), [find image patches with minimal or maximal activation for a given channel](showcase.md#minmax-activation-patches) and [simulate what each neuron is searching for (deep dream)](showcase.md#simulate-what-a-neuron-is-searching-for-deep-dream).
- **Spot optimization bottlenecks.** [Discover insufficient receptive fields](showcase.md#measure-the-receptive-field-of-a-neuron), [measure neuron death](showcase.md#investigate-dead-neurons), [discover padding artifacts](showcase.md#padding-artifacts) and [spot gradient underflow](showcase.md#spot-gradient-underflow).

The [Showcase](showcase.md) walks through all of these with real screenshots. [Getting started](getting-started.md) runs an example in minutes, the [UI guide](ui.md) tours every page, and the [Wiring guide](wiring.md) adds nansense to your own training loop — it's just a few lines of code.

## How is this different from wandb or TensorBoard?

Loggers like Weights & Biases and TensorBoard record scalar curves of loss and accuracy that you scroll through after the run. Nansense works inside the live training loop instead: it pauses so you can step batch-by-batch and time-travel while inspecting the activations, gradients, weights and optimizer state of every layer. You can even run experiments like deep dream or Grad-CAM on the paused model to probe what a given neuron has learned.

Persisting all this data on disk is infeasible, as a single batch of activations and gradients can easily be several gigabytes. Nansense sidesteps that by pausing and inspecting the tensors on demand, instead of writing everything to disk.
