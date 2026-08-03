/* Variant switch for the playground embedded on the home page, plus the tour
   bookkeeping every embed shares. The full-page playground (/playground/)
   carries its own copy of the switch logic inline, since there the switch
   also has to be relocated into the header toolbar — but it loads this file
   too (mkdocs `extra_javascript` ships it on every page), so the tour store
   below serves both embeds from one place.

   The embed's zoom is pure CSS (`--pg-scale` in stylesheets/extra.css) — the
   frame is laid out wide and scaled down, so it needs no JS to stay fitted. */

(function () {
  var SPACES = {
    mnist: "https://kongaskristjan-nansense-playground-mnist.hf.space",
    imagenette: "https://kongaskristjan-nansense-playground.hf.space",
  };

  /* The playground's guided tour auto-starts once per browser and then
     remembers that it was dismissed. The frame used to keep that flag in its
     own localStorage — scoped to the Space origin, and the two demos are two
     Spaces, so switching demos replayed the tour from step 1 (on the home
     page and on /playground/ alike). The frames now ask their embedder to
     hold the flags: this page is a single first-party origin behind both
     demos and both embeds, so a tour dismissed anywhere stays dismissed
     everywhere — and being first-party, it also survives the third-party
     storage blocking that would eventually drop the frame's own copy.

     `nansense/ui/tour.py` is the other half of the protocol; it falls back to
     the frame's own storage whenever this listener isn't there to answer. */
  var TOUR_ORIGINS = Object.keys(SPACES).map(function (name) {
    return SPACES[name];
  });
  // The frames may only touch their own flags (`tour.seen_key`).
  var TOUR_KEY = /^nansense-tour-seen(-[a-z]+)?$/;

  function tourSeen(key) {
    try {
      return !!localStorage.getItem(key);
    } catch (e) {
      return false;
    }
  }

  function onTourMessage(event) {
    var data = event.data;
    if (TOUR_ORIGINS.indexOf(event.origin) < 0 || !data) return;
    if (!TOUR_KEY.test(String(data.key))) return;
    if (data.nansenseTour === "set") {
      try {
        localStorage.setItem(data.key, "1");
      } catch (e) {}
    } else if (data.nansenseTour === "get") {
      event.source.postMessage(
        { nansenseTour: "is", key: data.key, seen: tourSeen(data.key) },
        event.origin
      );
    }
  }

  // Listening before anything else runs: the frame asks as soon as its own
  // scripts do, and an ask nobody answers costs the visitor a replayed tour.
  window.addEventListener("message", onTourMessage);

  function init() {
    var root = document.querySelector(".pg-embed");
    if (!root) return;
    var frame = root.querySelector(".pg-embed__frame");
    var buttons = root.querySelectorAll(".pg-embed__btn[data-variant]");
    var current = "mnist"; // matches the iframe's HTML src attribute

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
