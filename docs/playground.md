---
hide:
  - navigation
  - toc
---

# Playground

Explore a trained network without installing anything. Each playground is a nansense session paused on its final training batch: show and hide layers, browse activations, gradients, weights and epoch statistics, and run experiments such as deep dream. Stepping, time travel and the global settings are disabled in this shared, locked session.

Pick a network below — Imagenette (a PreActResNet trained for 50 epochs) or MNIST (a LeNet-5 trained for 20 epochs). The demos run on free Hugging Face Spaces, so the first load after a quiet period can take a minute to wake up.

<div class="playground-switch" role="group" aria-label="Playground variant">
  <button class="md-button md-button--primary" data-variant="imagenette">Imagenette · ResNet</button>
  <button class="md-button" data-variant="mnist">MNIST · LeNet-5</button>
  <a class="md-button" id="playground-open" href="https://kongaskristjan-nansense-playground.hf.space" target="_blank" rel="noopener">Open full screen</a>
</div>

<iframe id="playground-frame" title="Nansense playground" src="https://kongaskristjan-nansense-playground.hf.space" allow="fullscreen"></iframe>

<style>
  /* Full-width page: the embedded app needs more room than prose does. */
  .md-grid {
    max-width: 100%;
  }
  .playground-switch {
    display: flex;
    flex-wrap: wrap;
    gap: 0.5rem;
    align-items: center;
  }
  .playground-switch .md-button {
    margin: 0;
    cursor: pointer;
  }
  #playground-frame {
    width: 100%;
    height: max(30rem, calc(100vh - 15rem));
    margin-top: 1rem;
    border: 1px solid var(--md-default-fg-color--lightest);
    border-radius: 0.2rem;
  }
</style>

<script>
  (function () {
    var spaces = {
      imagenette: "https://kongaskristjan-nansense-playground.hf.space",
      mnist: "https://kongaskristjan-nansense-playground-mnist.hf.space",
    };
    var frame = document.getElementById("playground-frame");
    var open = document.getElementById("playground-open");
    var buttons = document.querySelectorAll(".playground-switch button[data-variant]");
    var current = "imagenette"; // matches the iframe's HTML src attribute
    function select(variant) {
      if (variant === current) return;
      current = variant;
      frame.src = spaces[variant];
      open.href = spaces[variant];
      buttons.forEach(function (button) {
        button.classList.toggle("md-button--primary", button.dataset.variant === variant);
      });
      history.replaceState(null, "", variant === "imagenette" ? location.pathname : "#" + variant);
    }
    buttons.forEach(function (button) {
      button.addEventListener("click", function () {
        select(button.dataset.variant);
      });
    });
    function selectFromHash() {
      select(location.hash === "#mnist" ? "mnist" : "imagenette");
    }
    window.addEventListener("hashchange", selectFromHash);
    if (location.hash === "#mnist") selectFromHash();
  })();
</script>
