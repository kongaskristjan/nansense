---
hide:
  - toc
---

# NaNsense

*Don't guess why your neural network fails to learn. Instead, have a look inside.*

<div class="pg-embed">
  <div class="pg-embed__bar" role="group" aria-label="Playground variant">
    <span class="pg-embed__label">Live playground</span>
    <button class="pg-embed__btn" type="button" data-variant="imagenette" aria-pressed="true">Imagenette · ResNet</button>
    <button class="pg-embed__btn" type="button" data-variant="mnist" aria-pressed="false">MNIST · LeNet-5</button>
  </div>
  <!-- The app is desktop-first (`body { min-width: 800px }`) and pans instead of
       reflowing, so the frame is given a fixed logical viewport and scaled to fit
       the column — cropping it would cut the input pane off mid-word.
       `clipboard-write` delegates the Clipboard API into the cross-origin app
       frame, so the app's own Share dialog can copy links. -->
  <div class="pg-embed__screen">
    <iframe class="pg-embed__frame" title="NaNsense playground" src="https://kongaskristjan-nansense-playground.hf.space" allow="fullscreen; clipboard-write"></iframe>
  </div>
  <p class="pg-embed__note">The playground starts a real training run, so it can take 1–2 minutes to boot — watch the video below while it loads.</p>
</div>

<video controls muted style="max-width: 100%;" src="https://github.com/user-attachments/assets/d7ee7ecc-4828-4655-866d-a220174c2b44"></video>

*NaNsense* is a PyTorch debugger that visualizes activations, gradients, weights, optimizer state and various statistics. You can **pause, step batch-by-batch, and time-travel to a different epoch while training**, and see exactly what every layer is doing.

[🕹️ Try the Playground](playground.md){ .md-button .md-button--primary }
[✨ Integrate with one prompt](integrate.md){ .md-button .md-button--primary }
[🤖 Debug with a coding agent](mcp.md){ .md-button .md-button--primary }

Here's how *NaNsense* can help:

- **See what is actually going on.** [Visualize activations and gradients](showcase.md#visualize-activations-and-gradients-throughout-training), [find image patches with minimal or maximal activation for a given channel](showcase.md#minmax-activation-patches) and [simulate what each neuron is searching for (deep dream)](showcase.md#simulate-what-a-neuron-is-searching-for-deep-dream).
- **Spot optimization bottlenecks.** [Discover insufficient receptive fields](showcase.md#measure-the-receptive-field-of-a-neuron), [measure neuron death](showcase.md#investigate-dead-neurons), [discover padding artifacts](showcase.md#padding-artifacts) and [spot gradient underflow](showcase.md#spot-gradient-underflow).

The [Showcase](showcase.md) walks through all of these with real screenshots. [Getting started](getting-started.md) runs an example in minutes, the [UI guide](ui.md) tours every page, and the [Wiring guide](wiring.md) adds NaNsense to your own training loop — it's just a few lines of code. It also speaks [MCP](mcp.md), so a coding agent can drive the debugger itself and look at the same views you do.

## How is this different from wandb or TensorBoard?

Loggers record external metrics — the loss and accuracy curves you scroll through after the run. NaNsense is focused on understanding the internals of the network. A logger tells you *that* the loss stopped falling; NaNsense shows you *why* — say, half a layer's channels died after epoch three, or fp16 gradients are underflowing.
