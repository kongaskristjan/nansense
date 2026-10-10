/* Rewrites the "Integrate with one prompt" copy-paste prompt when the reader
   toggles the "--debugger-port only" checkbox. The prompt lives in the page as
   a normal code block (so it also shows up in llms.txt / with JS disabled); we
   only swap the one line that differs between the two modes. */

(function () {
  var PORT_STEP =
    "3. Add a `--debugger-port PORT` CLI flag and pass `port=that_port, enabled=that_port is not None`, so NaNsense stays a zero-overhead no-op unless a port is given.";
  var PLAIN_STEP =
    "3. Turn the debugger on directly with `port=8080, enabled=True`.";

  function apply(code, portOnly) {
    var text = code.textContent;
    code.textContent = portOnly
      ? text.replace(PLAIN_STEP, PORT_STEP)
      : text.replace(PORT_STEP, PLAIN_STEP);
  }

  function init() {
    document.querySelectorAll(".nansense-prompt").forEach(function (root) {
      var box = root.querySelector('input[type="checkbox"]');
      var code = root.querySelector("pre code");
      if (!box || !code) return;
      apply(code, box.checked);
      box.addEventListener("change", function () {
        apply(code, box.checked);
      });
    });
  }

  if (document.readyState !== "loading") init();
  else document.addEventListener("DOMContentLoaded", init);
})();
