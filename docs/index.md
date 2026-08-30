---
hide:
  - toc
---

# NaNsense

*See what your neural network is doing while it trains.*

<div class="pg-embed">
  <div class="pg-embed__bar" role="group" aria-label="Playground variant">
    <span class="pg-embed__label">Live playground</span>
    <button class="pg-embed__btn" type="button" data-variant="mnist" aria-pressed="true">Easy: MNIST · LeNet-5</button>
    <button class="pg-embed__btn" type="button" data-variant="imagenette" aria-pressed="false">Advanced: Imagenette · ResNet</button>
  </div>
  <!-- The app is desktop-first (`body { min-width: 800px }`) and pans instead of
       reflowing, so the frame is given a fixed logical viewport and scaled to fit
       the column — cropping it would cut the input pane off mid-word.
       `clipboard-write` delegates the Clipboard API into the cross-origin app
       frame, so the app's own Share dialog can copy links. -->
  <div class="pg-embed__screen">
    <iframe class="pg-embed__frame" title="NaNsense playground" src="https://kongaskristjan-nansense-playground-mnist.hf.space" allow="fullscreen; clipboard-write"></iframe>
  </div>
  <p class="pg-embed__note">The playground may take a minute or two to start. The video below gives you a quick tour while it loads.</p>
</div>

<video controls muted style="max-width: 100%;" src="https://github.com/user-attachments/assets/b15f7dc5-1a6f-44d2-81fa-a97daafa89c6"></video>

*NaNsense* is a visual debugger for PyTorch. Pause training, step through batches, revisit earlier epochs, and inspect activations, gradients, weights, and optimizer state.

[🕹️ Try the Playground](playground.md){ .md-button .md-button--primary }
[✨ Integrate with one prompt](integrate.md){ .md-button .md-button--primary }
[🤖 Debug with a coding agent](mcp.md){ .md-button .md-button--primary }

- **Understand learned features.** [See activations and gradients](showcase.md#visualize-activations-and-gradients-throughout-training), [find the inputs that excite a channel](showcase.md#minmax-activation-patches), and [use deep dream to reveal what it has learned](showcase.md#simulate-what-a-neuron-is-searching-for-deep-dream).
- **Find training problems.** Spot [small receptive fields](showcase.md#measure-the-receptive-field-of-a-neuron), [dead channels](showcase.md#investigate-dead-neurons), [padding artifacts](showcase.md#padding-artifacts), and [gradient underflow](showcase.md#spot-gradient-underflow).

See the [Showcase](showcase.md) for screenshots, [Getting started](getting-started.md) to run an example, and the [Wiring guide](wiring.md) to add NaNsense to your project. A coding agent can also inspect the same session through [MCP](mcp.md).

## How is this different from wandb or TensorBoard?

Experiment trackers show that a run went wrong. NaNsense helps you find where it went wrong inside the network—for example, a layer with dead channels or underflowing gradients.
