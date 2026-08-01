"""The per-page guided tours: a few one-sentence steps with pointing arrows.

A floating message box (bottom-center, above the page) walks through the
current page's least obvious controls; each step draws an amber arrow —
plus an outline ring — from the box to the live element(s) it talks about.
Every page has its own short tour (`*_tour_steps` below); the main view's
is the long one, the subpages get one to three steps covering only what a
first look wouldn't reveal.

The main tour's framing, strips and buttons steps need the right layer
card on screen, so showing any of them auto-shows the weights-owning layer
(via the
show-only `nansense_tour_show_layer` event, which never hides a card the
visitor already opened) whenever one of the step's own anchors has no
visible target — with no card the strips anchor is missing, and a card
without weights (e.g. a ReLU) has no Weights button for the buttons step's
arrow. On a locked session that only touches the tab's own `shown` set, so
playground visitors never affect each other. The sample step similarly
re-opens the input pane (`nansense_tour_show_input`) if the top bar's
image button had hidden it.

The driver is a self-contained JS blob in the `static.py` style: targets
are plain CSS selectors resolved to their first *visible* match on a 200 ms
interval, which transparently rides out Mermaid's async render, the
auto-shown card's server round-trip, pane scrolling, and window resizes.
Steps whose target hasn't appeared (yet) simply show no arrow. The driver
brackets each fresh run with `nansense_tour_start` / `nansense_tour_end`
events: the stats page — whose view-bound steps switch what every card
shows — uses them to restore the view the visitor was on before the tour,
unless they picked a view themselves while it ran.

Auto-start policy (see `add_tour`): only a locked session — the hosted
playground, whose visitors are exactly the people who have never seen the
UI — starts a tour on load, and each page's tour only once per browser
(a per-page localStorage flag, set the moment the tour is dismissed —
skipped, stepped through, or escaped; the playground's public origin is
stable, so the flag survives visits). Local runs never auto-start: the top
bar's `?` button (`top_bar._add_tour_button`) is the explicit way in on
every page, and it replays that page's tour anywhere, seen or not.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

from nicegui import ui

# The localStorage flags marking a page's tour as seen. Origin-scoped, which
# is fine for the fixed-origin playground (the only auto-start case); local
# runs change ports (origins) freely because they never auto-start. The main
# page keeps the original unsuffixed key so playground visitors who already
# dismissed its tour aren't replayed by the rename.
SEEN_KEY_PREFIX = "nansense-tour-seen"


def seen_key(page: str) -> str:
    """The localStorage seen-flag for one page's tour."""
    return SEEN_KEY_PREFIX if page == "main" else f"{SEEN_KEY_PREFIX}-{page}"


@dataclass(frozen=True)
class TourStep:
    """One tour message plus the element(s) its arrows point at.

    Each selector contributes one arrow, drawn to its first visible match
    (hidden cards' elements have zero-size rects and are skipped).
    `ensure_card` marks steps that need the auto-watch layer's card on
    screen (main view only); `ensure_input` marks the step whose target
    lives in the input pane, which the top bar's image button can hide;
    `ensure_view` names the Stats view the step talks about — showing the
    step switches the page to it (via the `nansense_tour_set_view` event
    the stats page listens for), so the arrows land on a live example of
    what the message describes.
    """

    text: str
    selectors: tuple[str, ...]
    ensure_card: bool = False
    ensure_input: bool = False
    ensure_view: str | None = None


def _mermaid_node_selector(slug: str) -> str:
    """The diagram node for a layer slug (same scheme as `findMermaidNode`)."""
    return f'g.node[id*="-flowchart-{slug}-"]'


def main_tour_steps(layer_slug: str | None, *, locked: bool) -> list[TourStep]:
    """The main view's steps: a framing step naming the three panes, then
    the mechanics, with the first two steps pointing at `layer_slug`'s
    diagram node.

    Falls back to the first diagram node when no layer is known (a model
    with no captured layers) — the arrow still lands on something sensible.
    A locked session (the playground) parks training, so the closing step
    about the Run / Step Batch / Stop cluster only exists on live runs —
    the cluster itself is replaced by the demo notice there anyway.
    """
    node = _mermaid_node_selector(layer_slug) if layer_slug else "g.node"
    steps = [
        TourStep(
            "Click any layer in the neural network.",
            (node,),
        ),
        TourStep(
            "A card opens with the it's activations and gradients.",
            ('[data-tour="strips"]',),
            ensure_card=True,
        ),
        TourStep(
            "Weights, Experiments, and Stats go deeper: its "
            "parameters and optimizer state, deep-dream, and "
            "training statistics.",
            (
                '[data-tour="weights"]',
                '[data-tour="experiment"]',
                '[data-tour="stats"]',
            ),
            # The Weights anchor only exists on a card whose layer owns
            # weights — a shown ReLU card leaves it missing, so this step
            # must be able to bring the weights-owning card on screen too.
            ensure_card=True,
        ),
        TourStep(
            "Select the input to inspect.",
            ('[data-tour="sample"]',),
            ensure_input=True,
        ),
    ]
    if not locked:
        steps.append(
            TourStep(
                "Step or run the paused training from here.",
                ('[data-tour="step-controls"]',),
            )
        )
    return steps


def stats_tour_steps() -> list[TourStep]:
    """The Stats page's steps: the sidebar dropdowns, then one per view.

    Step 1 covers the three dropdowns in one message; each following step
    describes one view and forces it (`ensure_view`), so its arrows point
    at a live example: the activation/gradient histograms (with the
    hover-a-bar sampler, the page's least discoverable feature), a MIN/MAX
    grid's per-channel column, and a GRAPHS epoch series.
    """
    return [
        TourStep(
            "Select the investigation view, train or validation phase, "
            "and the layer.",
            (
                '[data-tour="view"]',
                '[data-tour="phase"]',
                '[data-tour="layer"]',
            ),
        ),
        TourStep(
            "In HISTOGRAM view, you can see the distributions of each layer's "
            "activations and gradients.",
            ('[data-tour="hist-activation"]', '[data-tour="hist-gradient"]'),
            ensure_view="HISTOGRAM",
        ),
        TourStep(
            "In MIN/MAX view, each column is one channel: the inputs that activates "
            "it most.",
            ('[data-tour="patch-column"]',),
            ensure_view="MIN/MAX",
        ),
        TourStep(
            "In GRAPHS view, layer's activation statistics are plot against time. ",
            ('[data-tour="epoch-graph"]',),
            ensure_view="GRAPHS",
        ),
    ]


def weights_tour_steps() -> list[TourStep]:
    """The Weights page's steps: what the strips are, then axis remapping.

    The strips come first — the arrow lands on the weight strip, and the
    message covers the gradient and optimizer-state strips below it (they
    only appear once training has stepped, so they shouldn't be mistaken
    for more weights). The dimension-role controls follow as the page's
    one mechanic that isn't self-evident.
    """
    return [
        TourStep(
            "Each parameter's values, with its gradient and the "
            "optimizer's state (momentum, Adam moments) in the strips "
            "below.",
            ('[data-tour="weight-strips"]',),
        ),
    ]


def experiment_tour_steps(*, locked: bool) -> list[TourStep]:
    """The Experiment page's steps.

    Step 1 points at the kind and layer selectors — mainly for the
    description that appears at the bottom of the pane, which is easy to
    miss. Step 2 is per-flavor: the playground (locked) turns auto-run off
    — the page's first experiment still self-starts, but re-runs take a
    manual Run — so its visitors get the Run / Cancel pair pointed out;
    live runs auto-run by default and instead get the page's one real
    gotcha —
    experiments execute on the training thread, so nothing runs until
    training pauses (the playground's training is always parked, and its
    step cluster is replaced by the demo notice anyway).
    """
    steps = [
        TourStep(
            "Pick the experiment and the target layer.",
            ('[data-tour="kind"]', '[data-tour="layer"]'),
        ),
    ]
    if locked:
        steps.append(
            TourStep(
                "Don't forget to rerun when changing parameters.",
                ('[data-tour="run"]',),
            )
        )
    else:
        steps.append(
            TourStep(
                "Experiments run while training is paused — if results "
                "stay queued, pause with Stop or Step Batch.",
                ('[data-tour="step-controls"]',),
            )
        )
    return steps


def tour_config(
    steps: list[TourStep],
    *,
    page: str,
    auto_start: bool,
    auto_watch_slug: str | None = None,
) -> dict[str, object]:
    """The driver's config object (`window.nansenseTourConfig`)."""
    return {
        "steps": [
            {
                "text": step.text,
                "selectors": list(step.selectors),
                "ensureCard": step.ensure_card,
                "ensureInput": step.ensure_input,
                "ensureView": step.ensure_view,
            }
            for step in steps
        ],
        "autoWatchSlug": auto_watch_slug,
        "autoStart": auto_start,
        "seenKey": seen_key(page),
    }


def add_tour(
    page: str,
    steps: list[TourStep],
    *,
    locked: bool,
    auto_watch_slug: str | None = None,
) -> None:
    """Install `page`'s tour (config + CSS + driver) into the current page.

    `auto_watch_slug` (main page only) is the layer auto-shown for the
    card-needing steps (strips and buttons). Only a locked session (the
    shared playground) auto-starts, and only when the browser hasn't
    dismissed this page's tour before
    (per-page `seen_key`); everywhere else the tour waits for the top bar's
    `?` button, which replays it regardless of the seen flag.
    """
    # Same `</`-escape as `main_page._layer_info_script`: layer names (via
    # slugs) are user data and must not terminate the script tag early.
    payload = json.dumps(
        tour_config(
            steps, page=page, auto_start=locked, auto_watch_slug=auto_watch_slug
        )
    )
    payload = payload.replace("</", "<\\/")
    ui.add_head_html(_TOUR_CSS)
    ui.add_body_html(f"<script>window.nansenseTourConfig = {payload};</script>")
    ui.add_body_html(_TOUR_JS)


# The box is dark like the layer-info tooltip; arrows and rings are amber to
# match the watched-layer treatment. The overlay ignores the pointer except
# on the box itself, so the page stays fully usable mid-tour (step 1 invites
# a diagram click).
_TOUR_CSS: str = """
<style>
  .nansense-tour-overlay {
    position: fixed;
    inset: 0;
    z-index: 7000;
    pointer-events: none;
  }
  .nansense-tour-overlay svg {
    position: absolute;
    inset: 0;
    width: 100%;
    height: 100%;
  }
  .nansense-tour-box {
    position: absolute;
    left: 50%;
    bottom: 12vh;
    transform: translateX(-50%);
    max-width: 26rem;
    padding: 12px 16px;
    background: rgb(15 23 42 / 0.95);
    color: rgb(241 245 249);
    font-size: 14px;
    line-height: 1.5;
    border-radius: 8px;
    box-shadow: 0 8px 24px rgb(0 0 0 / 0.35);
    pointer-events: auto;
  }
  .nansense-tour-row {
    display: flex;
    align-items: center;
    gap: 12px;
    margin-top: 10px;
  }
  .nansense-tour-count {
    color: rgb(148 163 184);
    font-size: 12px;
    margin-right: auto;
  }
  .nansense-tour-skip {
    background: none;
    border: none;
    color: rgb(148 163 184);
    font-size: 13px;
    cursor: pointer;
    padding: 4px 6px;
  }
  .nansense-tour-skip:hover { color: rgb(226 232 240); }
  .nansense-tour-next {
    background: rgb(245 158 11);
    border: none;
    color: rgb(15 23 42);
    font-weight: 600;
    font-size: 13px;
    padding: 5px 14px;
    border-radius: 6px;
    cursor: pointer;
  }
  .nansense-tour-next:hover { background: rgb(251 191 36); }
  .nansense-tour-ring {
    position: absolute;
    border: 2px solid rgb(245 158 11);
    border-radius: 6px;
    box-shadow: 0 0 0 4px rgb(245 158 11 / 0.25);
    pointer-events: none;
  }
</style>
"""


_TOUR_JS: str = """
<script>
(function() {
  const cfg = window.nansenseTourConfig;
  if (!cfg) return;

  let stepIdx = -1;
  let scrolledStep = -1;
  let overlay = null, svg = null, box = null;
  let textEl = null, countEl = null, nextBtn = null;
  let timer = null;

  function seen() {
    try { return !!localStorage.getItem(cfg.seenKey); }
    catch (e) { return false; }
  }
  function markSeen() {
    try { localStorage.setItem(cfg.seenKey, '1'); } catch (e) {}
  }

  // First *visible* match: hidden layer cards keep their DOM (display:none),
  // so a zero-size rect distinguishes shown cards from hidden ones. Also
  // works for SVG nodes, which have no offsetParent. Quasar form fields
  // forward unrecognized attributes (our data-tour anchors) to their inner
  // native control — a bare input line — so a match inside a q-field is
  // widened to the whole field, keeping the ring aligned with the visible
  // widget (label, border and all).
  function findTarget(sel) {
    for (const el of document.querySelectorAll(sel)) {
      const r = el.getBoundingClientRect();
      if (r.width > 0 && r.height > 0) return el.closest('.q-field') || el;
    }
    return null;
  }

  // Where an arrow tip lands: the point on the target rect's (padded)
  // border along the line from the rect center towards the box.
  function edgePoint(rect, from) {
    const cx = rect.left + rect.width / 2;
    const cy = rect.top + rect.height / 2;
    const dx = from.x - cx, dy = from.y - cy;
    if (dx === 0 && dy === 0) return { x: cx, y: cy };
    const sx = dx !== 0 ? (rect.width / 2 + 10) / Math.abs(dx) : Infinity;
    const sy = dy !== 0 ? (rect.height / 2 + 10) / Math.abs(dy) : Infinity;
    const t = Math.min(sx, sy, 1);
    return { x: cx + dx * t, y: cy + dy * t };
  }

  // A gentle quadratic bend so parallel arrows (step 3) stay distinct.
  function arrowPath(a, b) {
    const mx = (a.x + b.x) / 2, my = (a.y + b.y) / 2;
    const dx = b.x - a.x, dy = b.y - a.y;
    const len = Math.hypot(dx, dy) || 1;
    const off = Math.min(40, len * 0.2);
    return 'M ' + a.x + ' ' + a.y +
      ' Q ' + (mx - dy / len * off) + ' ' + (my + dx / len * off) +
      ' ' + b.x + ' ' + b.y;
  }

  const SVG_NS = 'http://www.w3.org/2000/svg';

  function buildOverlay() {
    overlay = document.createElement('div');
    overlay.className = 'nansense-tour-overlay';

    svg = document.createElementNS(SVG_NS, 'svg');
    const defs = document.createElementNS(SVG_NS, 'defs');
    const marker = document.createElementNS(SVG_NS, 'marker');
    marker.setAttribute('id', 'nansense-tour-head');
    marker.setAttribute('viewBox', '0 0 10 10');
    marker.setAttribute('refX', '8');
    marker.setAttribute('refY', '5');
    marker.setAttribute('markerWidth', '7');
    marker.setAttribute('markerHeight', '7');
    marker.setAttribute('orient', 'auto-start-reverse');
    const head = document.createElementNS(SVG_NS, 'path');
    head.setAttribute('d', 'M 0 0 L 10 5 L 0 10 z');
    head.setAttribute('fill', 'rgb(245 158 11)');
    marker.appendChild(head);
    defs.appendChild(marker);
    svg.appendChild(defs);
    overlay.appendChild(svg);

    box = document.createElement('div');
    box.className = 'nansense-tour-box';
    textEl = document.createElement('div');
    box.appendChild(textEl);
    const row = document.createElement('div');
    row.className = 'nansense-tour-row';
    countEl = document.createElement('span');
    countEl.className = 'nansense-tour-count';
    const skipBtn = document.createElement('button');
    skipBtn.className = 'nansense-tour-skip';
    skipBtn.textContent = 'Skip';
    skipBtn.addEventListener('click', stop);
    nextBtn = document.createElement('button');
    nextBtn.className = 'nansense-tour-next';
    nextBtn.addEventListener('click', next);
    row.appendChild(countEl);
    row.appendChild(skipBtn);
    row.appendChild(nextBtn);
    box.appendChild(row);
    overlay.appendChild(box);

    document.body.appendChild(overlay);
  }

  // Redrawn on a 200 ms tick (plus resize): recomputing from live rects is
  // what lets arrows appear as Mermaid / the auto-shown card render, and
  // follow the panes as they scroll. A handful of nodes at 5 Hz is cheap.
  function reposition() {
    if (!overlay || stepIdx < 0) return;
    const step = cfg.steps[stepIdx];
    for (const el of overlay.querySelectorAll('.nansense-tour-ring')) {
      el.remove();
    }
    for (const el of svg.querySelectorAll('path[data-arrow]')) el.remove();
    const boxRect = box.getBoundingClientRect();
    const start = { x: boxRect.left + boxRect.width / 2, y: boxRect.top };
    let first = null;
    for (const sel of step.selectors) {
      const el = findTarget(sel);
      if (!el) continue;
      if (!first) first = el;
      const r = el.getBoundingClientRect();
      const ring = document.createElement('div');
      ring.className = 'nansense-tour-ring';
      ring.style.left = (r.left - 4) + 'px';
      ring.style.top = (r.top - 4) + 'px';
      ring.style.width = (r.width + 8) + 'px';
      ring.style.height = (r.height + 8) + 'px';
      overlay.appendChild(ring);
      const path = document.createElementNS(SVG_NS, 'path');
      path.setAttribute('data-arrow', '1');
      path.setAttribute('d', arrowPath(start, edgePoint(r, start)));
      path.setAttribute('fill', 'none');
      path.setAttribute('stroke', 'rgb(245 158 11)');
      path.setAttribute('stroke-width', '2.5');
      path.setAttribute('marker-end', 'url(#nansense-tour-head)');
      svg.appendChild(path);
    }
    // Once per step: bring an off-screen target into view (e.g. the shown
    // card when many cards precede it in the centre pane).
    if (first && scrolledStep !== stepIdx) {
      scrolledStep = stepIdx;
      const r = first.getBoundingClientRect();
      if (r.top < 0 || r.bottom > window.innerHeight ||
          r.left < 0 || r.right > window.innerWidth) {
        first.scrollIntoView({ block: 'center', behavior: 'smooth' });
      }
    }
  }

  function showStep(i) {
    stepIdx = i;
    const step = cfg.steps[i];
    // Card-needing steps: any of this step's own anchors missing — no card
    // at all, or a shown card whose layer has no Weights button — means the
    // weights-owning card must come on screen. The event is show-only (a
    // toggle would hide a card the visitor already opened); on a locked
    // session the show is per-tab, like a diagram click.
    if (step.ensureCard && cfg.autoWatchSlug &&
        step.selectors.some(sel => !findTarget(sel))) {
      emitEvent('nansense_tour_show_layer', cfg.autoWatchSlug);
    }
    // The sample spinner lives in the input pane, which the top bar's image
    // button can hide; re-open it when the step's target is invisible (the
    // page's handler is a no-op if the pane is already showing).
    if (step.ensureInput &&
        step.selectors.every(sel => !findTarget(sel))) {
      emitEvent('nansense_tour_show_input');
    }
    // A view-bound step switches the Stats page to the view it describes;
    // the page's handler no-ops when that view is already showing.
    if (step.ensureView) {
      emitEvent('nansense_tour_set_view', step.ensureView);
    }
    textEl.textContent = step.text;
    countEl.textContent = (i + 1) + ' / ' + cfg.steps.length;
    nextBtn.textContent = i === cfg.steps.length - 1 ? 'Done' : 'Next';
    reposition();
  }

  function next() {
    if (stepIdx + 1 >= cfg.steps.length) { stop(); return; }
    showStep(stepIdx + 1);
  }

  function stop() {
    markSeen();
    if (timer) { clearInterval(timer); timer = null; }
    window.removeEventListener('resize', reposition);
    document.removeEventListener('keydown', onKey);
    // Every way out — Skip, Done, Escape — announces the end, so a page
    // whose view the tour switched can restore what was showing before.
    if (overlay) emitEvent('nansense_tour_end');
    if (overlay) overlay.remove();
    overlay = null;
    stepIdx = -1;
    scrolledStep = -1;
  }

  function onKey(e) {
    if (e.key === 'Escape') stop();
  }

  window.nansenseStartTour = function() {
    if (overlay) { scrolledStep = -1; showStep(0); return; }
    buildOverlay();
    // Fresh runs only (a mid-run restart keeps the original tour context):
    // pages with view-bound steps snapshot their state on this event so
    // dismissing the tour can put things back (see the stats page).
    emitEvent('nansense_tour_start');
    showStep(0);
    timer = setInterval(reposition, 200);
    window.addEventListener('resize', reposition);
    document.addEventListener('keydown', onKey);
  };

  if (cfg.autoStart && !seen()) {
    // A beat after load so the first arrow lands on a rendered diagram in
    // the common case (the 200 ms tick catches it regardless).
    setTimeout(function() { window.nansenseStartTour(); }, 800);
  }
})();
</script>
"""
