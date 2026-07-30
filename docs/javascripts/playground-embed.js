/* The playground embedded on the home page: variant switch, plus fitting the
   app into the content column. The full-page playground (/playground/) carries
   its own copy of the switch inline, since there it also has to be relocated
   into the header toolbar; it needs no scaling because it owns the viewport. */

(function () {
  var SPACES = {
    imagenette: "https://kongaskristjan-nansense-playground.hf.space",
    mnist: "https://kongaskristjan-nansense-playground-mnist.hf.space",
  };

  /* The app's own floor is `body { min-width: 800px }` (nansense/ui/static.py);
     below that it pans horizontally instead of reflowing. Laying the frame out a
     little above the floor keeps the three panes clear of each other without
     shrinking the text more than necessary. */
  var LOGICAL_WIDTH = 960;

  function init() {
    var root = document.querySelector(".pg-embed");
    if (!root) return;
    var screen = root.querySelector(".pg-embed__screen");
    var frame = root.querySelector(".pg-embed__frame");
    var buttons = root.querySelectorAll(".pg-embed__btn[data-variant]");
    var current = "imagenette"; // matches the iframe's HTML src attribute

    // Never scale up: on a wide enough column the app renders at native size.
    function fit() {
      var width = screen.clientWidth;
      if (!width) return;
      root.style.setProperty("--pg-scale", Math.min(1, width / LOGICAL_WIDTH));
    }
    fit();
    if (window.ResizeObserver) new ResizeObserver(fit).observe(screen);
    else window.addEventListener("resize", fit);

    buttons.forEach(function (button) {
      button.addEventListener("click", function () {
        var variant = button.dataset.variant;
        if (variant === current) return;
        current = variant;
        frame.src = SPACES[variant];
        buttons.forEach(function (other) {
          other.setAttribute("aria-pressed", String(other.dataset.variant === variant));
        });
      });
    });
  }

  if (document.readyState !== "loading") init();
  else document.addEventListener("DOMContentLoaded", init);
})();
