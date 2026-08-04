---
hide:
  - navigation
  - toc
  - footer
---

<div class="playground-switch" role="group" aria-label="Playground variant">
  <button class="pg-btn" type="button" data-variant="mnist" aria-pressed="true">Easy: MNIST · LeNet-5</button>
  <button class="pg-btn" type="button" data-variant="imagenette" aria-pressed="false">Advanced: Imagenette · ResNet</button>
</div>

<!-- `clipboard-write` delegates the Clipboard API into the cross-origin app
     frame: the app's own Share dialog (top bar) copies links, and without the
     delegation the browser rejects the write inside the iframe. -->
<iframe id="playground-frame" title="NaNsense playground" src="https://kongaskristjan-nansense-playground-mnist.hf.space" allow="fullscreen; clipboard-write"></iframe>

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
  /* Below Material's 76.25em breakpoint the theme already collapses the nav
     into a hamburger drawer, and this page keeps that. Above it the theme
     would lay the nav out as a column beside the content — which this page
     has no room for, so it used to drop the nav entirely (`hide: navigation`
     in the front matter). Instead, re-create the drawer above the breakpoint
     too: same hamburger, same slide-out panel, at every width. The rules
     mirror the theme's own `max-width: 76.234375em` drawer block, which
     simply doesn't reach up here. */
  @media screen and (min-width: 76.25em) {
    /* The theme hides the hamburger once the nav has a column of its own. */
    .md-header__button[for="__drawer"] {
      display: inline-block;
    }
    /* `hide: navigation` leaves the sidebar in the page but marked `hidden`;
       display it again as the off-canvas panel (the same trick the theme
       plays below the breakpoint). Its sticky-column offsets are written
       inline by the theme's script, hence the !important. */
    .md-sidebar--primary {
      display: block;
      position: fixed;
      top: 0 !important;
      left: -12.1rem;
      width: 12.1rem;
      height: 100%;
      padding: 0;
      background-color: var(--md-default-bg-color);
      transform: translateX(0);
      transition: transform 0.25s cubic-bezier(0.4, 0, 0.2, 1), box-shadow 0.25s;
      z-index: 5;
    }
    [data-md-toggle="drawer"]:checked ~ .md-container .md-sidebar--primary {
      transform: translateX(12.1rem);
      box-shadow: var(--md-shadow-z3);
    }
    .md-sidebar--primary .md-sidebar__scrollwrap {
      position: absolute;
      inset: 0;
      margin: 0;
      height: auto !important;
      overflow: hidden;
    }
    /* The nav itself: laid out as the drawer's full-height panel (title bar
       on top, scrolling list below) rather than as a column of the page.
       Every selector is anchored on .md-sidebar--primary, which is both true
       (these only make sense inside the panel) and enough specificity to beat
       the theme's own `[dir] .md-nav--primary …` column rules. */
    .md-sidebar--primary .md-nav--primary {
      position: absolute;
      top: 0;
      left: 0;
      right: 0;
      height: 100%;
      display: flex;
      flex-direction: column;
      background-color: var(--md-default-bg-color);
      z-index: 1;
    }
    .md-sidebar--primary .md-nav--primary .md-nav__title {
      position: relative;
      height: 5.6rem;
      padding: 3rem 0.8rem 0.2rem;
      background-color: var(--md-primary-fg-color);
      box-shadow: none;
      color: var(--md-primary-bg-color);
      cursor: pointer;
      font-size: 0.8rem;
      line-height: 2.4rem;
      white-space: nowrap;
    }
    /* The theme hides the logo in the nav title outside the drawer. */
    .md-sidebar--primary .md-nav--primary .md-nav__title .md-logo {
      display: block;
      position: absolute;
      top: 0.2rem;
      left: 0.2rem;
      right: 0.2rem;
      margin: 0.2rem;
      padding: 0.4rem;
    }
    .md-sidebar--primary .md-nav--primary .md-nav__list {
      flex: 1;
      padding: 0 0 0.4rem;
      overflow-y: auto;
      box-shadow: 0 0.05rem 0 var(--md-default-fg-color--lightest) inset;
    }
    .md-sidebar--primary .md-nav--primary .md-nav__item {
      border-top: 0.05rem solid var(--md-default-fg-color--lightest);
      font-size: 0.8rem;
      line-height: 1.5;
    }
    /* The list's inset shadow already draws the line under the title bar. */
    .md-sidebar--primary .md-nav--primary .md-nav__list > :first-child {
      border-top: 0;
    }
    .md-sidebar--primary .md-nav--primary .md-nav__item > .md-nav__link {
      margin: 0;
      padding: 0.6rem 0.8rem;
    }

    /* The scrim that dims the app and closes the drawer when clicked. */
    .md-overlay {
      position: fixed;
      top: 0;
      width: 0;
      height: 0;
      opacity: 0;
      background-color: rgba(0, 0, 0, 0.54);
      transition: width 0ms 0.25s, height 0ms 0.25s, opacity 0.25s;
      z-index: 5;
    }
    [data-md-toggle="drawer"]:checked ~ .md-overlay {
      width: 100%;
      height: 100%;
      opacity: 1;
      transition: width 0ms, height 0ms, opacity 0.25s;
    }
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
      mnist: "https://kongaskristjan-nansense-playground-mnist.hf.space",
      imagenette: "https://kongaskristjan-nansense-playground.hf.space",
    };
    var frame = document.getElementById("playground-frame");
    var switchEl = document.querySelector(".playground-switch");
    var buttons = switchEl.querySelectorAll(".pg-btn[data-variant]");
    var current = "mnist"; // matches the iframe's HTML src attribute

    // Lift the variant switch into the header, just past the (hidden) title,
    // so the repo link keeps the far right.
    var title = document.querySelector(".md-header__title");
    if (title) {
      title.insertAdjacentElement("afterend", switchEl);
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
      history.replaceState(null, "", variant === "mnist" ? location.pathname : "#" + variant);
    }
    buttons.forEach(function (button) {
      button.addEventListener("click", function () {
        select(button.dataset.variant);
      });
    });
    function selectFromHash() {
      select(location.hash === "#imagenette" ? "imagenette" : "mnist");
    }
    window.addEventListener("hashchange", selectFromHash);
    if (location.hash === "#imagenette") selectFromHash();
  })();
</script>
