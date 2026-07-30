/* Variant switch for the playground embedded on the home page. The full-page
   playground (/playground/) carries its own copy of this logic inline, since
   there the switch also has to be relocated into the header toolbar. */

(function () {
  var SPACES = {
    imagenette: "https://kongaskristjan-nansense-playground.hf.space",
    mnist: "https://kongaskristjan-nansense-playground-mnist.hf.space",
  };

  function init() {
    var root = document.querySelector(".pg-embed");
    if (!root) return;
    var frame = root.querySelector(".pg-embed__frame");
    var buttons = root.querySelectorAll(".pg-embed__btn[data-variant]");
    var current = "imagenette"; // matches the iframe's HTML src attribute

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
