---
hide:
  - navigation
  - toc
  - footer
---

<div class="playground-switch" role="group" aria-label="Playground variant">
  <button class="pg-btn" type="button" data-variant="imagenette" aria-pressed="true">Imagenette · ResNet</button>
  <button class="pg-btn" type="button" data-variant="mnist" aria-pressed="false">MNIST · LeNet-5</button>
</div>

<button class="pg-share-btn" type="button">Share playground</button>

<iframe id="playground-frame" title="Nansense playground" src="https://kongaskristjan-nansense-playground.hf.space" allow="fullscreen"></iframe>

<style>
  /* Fullscreen app page: the header collapses to a slim toolbar (logo +
     variant switch + repo link) and the iframe fills the rest. Scoped to
     this page — the inline <style> only ships on /playground/. */
  html,
  body {
    overflow: hidden;
  }

  /* The header row is a centered .md-grid (max-width 61rem) by default,
     which floats everything in a column with a gap on either side. Let it
     span the full width so the logo sits flush left and the repo link right. */
  .md-header__inner {
    max-width: none;
  }

  /* Strip the header down to what we keep: hide the wordmark, the version
     selector, the light/dark toggle and search — leave the logo (home link)
     and the repo link. */
  .md-header__title,
  .md-version,
  .md-header__option,
  .md-search,
  .md-header [for="__search"] {
    display: none !important;
  }
  /* The variant switch, relocated into the header toolbar by the script.
     Its auto right margin eats the free space, pushing the repo link that
     follows it to the far right. */
  .playground-switch {
    display: flex;
    align-items: center;
    gap: 0.3rem;
    margin-right: auto;
  }
  .pg-btn {
    appearance: none;
    border: 0;
    border-radius: 0.4rem;
    background: transparent;
    color: var(--md-primary-bg-color);
    font: inherit;
    font-size: 0.68rem;
    font-weight: 700;
    letter-spacing: 0.03em;
    line-height: 1.2;
    padding: 0.3rem 0.6rem;
    white-space: nowrap;
    cursor: pointer;
    opacity: 0.7;
    transition: opacity 0.2s, background-color 0.2s;
  }
  .pg-btn:hover {
    opacity: 1;
  }
  .pg-btn[aria-pressed="true"] {
    opacity: 1;
    background: rgba(255, 255, 255, 0.18);
  }
  /* "Share playground" action button, relocated next to the repo link by the
     script. Outlined rather than flat like .pg-btn so it reads as an action
     (copy the link), not a third variant toggle. */
  .pg-share-btn {
    appearance: none;
    border: 1px solid rgba(255, 255, 255, 0.4);
    border-radius: 0.4rem;
    background: transparent;
    color: var(--md-primary-bg-color);
    font: inherit;
    font-size: 0.68rem;
    font-weight: 700;
    letter-spacing: 0.03em;
    line-height: 1.2;
    padding: 0.3rem 0.6rem;
    margin-right: 0.4rem;
    white-space: nowrap;
    cursor: pointer;
    opacity: 0.85;
    transition: opacity 0.2s, background-color 0.2s;
  }
  .pg-share-btn:hover {
    opacity: 1;
    background: rgba(255, 255, 255, 0.18);
  }

  /* Iframe fills everything below the header; its top is set from the live
     header height by the script (robust to zoom and mobile header sizing). */
  .md-main,
  .md-content,
  .md-content__inner {
    margin: 0;
    padding: 0;
  }
  .md-content__inner::before {
    display: none;
  }
  /* MkDocs auto-adds an <h1> from the page title when the markdown has none;
     the app shell doesn't want it (the tab title still names the page). */
  .md-content__inner > h1 {
    display: none;
  }
  .md-footer {
    display: none;
  }
  #playground-frame {
    position: fixed;
    top: var(--pg-header-h, 2.4rem);
    left: 0;
    right: 0;
    /* Replaced elements keep their intrinsic height when top/bottom are both
       set, so size it explicitly from the live header height. */
    height: calc(100vh - var(--pg-header-h, 2.4rem));
    width: 100%;
    border: 0;
  }
</style>

<script>
  (function () {
    var spaces = {
      imagenette: "https://kongaskristjan-nansense-playground.hf.space",
      mnist: "https://kongaskristjan-nansense-playground-mnist.hf.space",
    };
    var frame = document.getElementById("playground-frame");
    var switchEl = document.querySelector(".playground-switch");
    var buttons = switchEl.querySelectorAll(".pg-btn[data-variant]");
    var current = "imagenette"; // matches the iframe's HTML src attribute

    // Lift the variant switch into the header, just past the (hidden) title,
    // so the repo link keeps the far right.
    var title = document.querySelector(".md-header__title");
    if (title) {
      title.insertAdjacentElement("afterend", switchEl);
    }

    // Add a "Share playground" button to the header, just before the repo
    // link, copying this page's URL (Web Share sheet where available, clipboard
    // otherwise). Kept out of the variant switch so it reads as an action, not
    // a third variant.
    var shareBtn = document.querySelector(".pg-share-btn");
    var shareUrl = "https://kongaskristjan.github.io/nansense/dev/playground/";
    if (shareBtn) {
      var source = document.querySelector(".md-header__source");
      if (source) {
        source.insertAdjacentElement("beforebegin", shareBtn);
      } else {
        (document.querySelector(".md-header__inner") || document.body).appendChild(shareBtn);
      }
      var shareLabel = shareBtn.textContent;
      var shareReset = null;
      shareBtn.addEventListener("click", function () {
        if (navigator.share) {
          navigator.share({ title: "Nansense playground", url: shareUrl }).catch(function () {});
          return;
        }
        if (navigator.clipboard) {
          navigator.clipboard.writeText(shareUrl).then(function () {
            shareBtn.textContent = "Link copied";
            if (shareReset) clearTimeout(shareReset);
            shareReset = setTimeout(function () { shareBtn.textContent = shareLabel; }, 1500);
          }).catch(function () {});
        }
      });
    }

    // Keep the iframe sized to the live header height.
    var header = document.querySelector(".md-header");
    function sizeFrame() {
      document.documentElement.style.setProperty("--pg-header-h", header.offsetHeight + "px");
    }
    sizeFrame();
    window.addEventListener("resize", sizeFrame);

    function select(variant) {
      if (variant === current) return;
      current = variant;
      frame.src = spaces[variant];
      buttons.forEach(function (button) {
        button.setAttribute("aria-pressed", String(button.dataset.variant === variant));
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
