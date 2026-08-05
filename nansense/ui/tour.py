"""The per-page guided tours: a few one-sentence steps with pointing arrows.

A floating message box (bottom-center, above the page) walks through the
current page's least obvious controls; each step draws an amber arrow —
plus an outline ring — from the box to the live element(s) it talks about.
Every page has its own short tour (`*_tour_steps` below); the main view's
is the long one, the subpages get one to three steps covering only what a
first look wouldn't reveal.

The main tour opens on one layer — the one the page already shows
(`main_page._pick_tour_layer`), whose diagram node the first arrow points
at — but its strips, colors and buttons steps follow the visitor instead of
that layer: they scope their anchors to a card that is really open,
preferring the tour's own and taking any other over none (`openCardSlug` in
the driver). The first step invites a diagram click, so by the time those
steps run the visitor may have opened another layer and closed the one the
tour started on; pointing back at it would read as the tour undoing their
click. Only with no card open at all is the page asked to open the tour's
layer, via the show-only `nansense_tour_show_layer` event (a toggle would
hide a card the visitor already opened). On a locked session that only
touches the tab's own `shown` set, so playground visitors never affect each
other. The sample step similarly re-opens the input pane
(`nansense_tour_show_input`) if the top bar's image button had hidden it —
both of that step's arrows live in there.

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
(a per-page seen flag, set the moment the tour is dismissed — skipped,
stepped through, or escaped). Local runs never auto-start: the top bar's
`?` button (`top_bar._add_tour_button`) is the explicit way in on every
page, and it replays that page's tour anywhere, seen or not.

Those flags live in whoever embeds the app, when they offer to hold them
(`docs/javascripts/playground-embed.js`), and in the app's own
localStorage otherwise. The hosted playground is the reason: its two
demos are two Hugging Face Spaces — two origins — swapped into one frame
on the docs pages, so an origin-scoped flag set under Imagenette is
invisible to MNIST and switching demos replayed every tour from step 1.
The docs origin is a single first-party one behind both demos and both
embeds (the home page and `/playground/`), so a tour dismissed anywhere
stays dismissed everywhere; `resolveSeen` in the driver below falls back
to this origin whenever nobody answers.

The same channel places the playground tour's closing arrow. That step
points at the docs header's "one prompt" call to action, which is the
embedder's own DOM and cross-origin to us, so the host is asked where it
is and answers in our coordinates (`askHostAnchor` / `anchorX` in
`playground-embed.js`). The arrow tip lands on our top edge under it;
where nobody answers it falls back to a share of the viewport width, and
where the embedder says it has no such button the step simply keeps its
in-app arrow alone.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

from nicegui import ui

# The flags marking a page's tour as seen — held by the embedding page when
# there is one, in this origin's localStorage otherwise (see the driver's
# `resolveSeen`). `docs/javascripts/playground-embed.js` only accepts keys
# under this prefix, so the two halves must keep agreeing on it
# (`test_tour.py`). The main page keeps the original unsuffixed key so
# playground visitors who already dismissed its tour aren't replayed by the
# rename.
SEEN_KEY_PREFIX = "nansense-tour-seen"


def seen_key(page: str) -> str:
    """The localStorage seen-flag for one page's tour."""
    return SEEN_KEY_PREFIX if page == "main" else f"{SEEN_KEY_PREFIX}-{page}"


@dataclass(frozen=True)
class TourStep:
    """One tour message plus the element(s) its arrows point at.

    Each selector contributes one arrow, drawn to its first visible match
    (hidden cards' elements have zero-size rects and are skipped).
    `ensure_card` marks the steps that talk about a layer card (main view
    only): their selectors ship as bare card anchors and the driver scopes
    them to whichever card is open, opening the tour's own layer when none
    is. `ensure_input` marks the step whose targets live in the input pane,
    which the top bar's image button can hide; `ensure_view` names the Stats
    view the step talks about — showing the step switches the page to it
    (via the `nansense_tour_set_view` event the stats page listens for), so
    the arrows land on a live example of what the message describes.

    `alt_text` is the message for a card that hasn't got the step's first
    anchor — the buttons step's Weights button, which only a layer owning
    parameters has. Which card a step lands on is the visitor's to decide,
    so both messages ship and the driver picks once it can see the card
    (`stepText`); until then `text` stands.

    `host_anchor` adds one more arrow that leaves the app entirely, aimed
    at the embedding docs page's "one prompt" call to action — the only
    tour target that isn't ours to select (see the driver's
    `hostAnchorPoint`). It draws no ring: there is nothing inside this
    document to ring.
    """

    text: str
    selectors: tuple[str, ...]
    ensure_card: bool = False
    ensure_input: bool = False
    ensure_view: str | None = None
    host_anchor: bool = False
    alt_text: str | None = None


def _mermaid_node_selector(slug: str) -> str:
    """The diagram node for a layer slug (same scheme as `findMermaidNode`)."""
    return f'g.node[id*="-flowchart-{slug}-"]'


def _card_anchor(anchor: str) -> str:
    """One anchor inside a layer card, left for the driver to scope.

    Cards carry `data-layer="<slug>"` (`main_page._LayerView`) and the
    driver prefixes that scope at show time (`scopedSelectors`), because
    which card a step is about is only settled in the browser: it is
    whichever one the visitor has open. Unscoped, the anchor would resolve
    to whichever card comes first in the pane instead.
    """
    return f'[data-tour="{anchor}"]'


def _buttons_step() -> TourStep:
    """The step covering the card's deep-dive buttons.

    The message names the buttons its arrows ring, and nothing more — what
    each page holds is the page's own business. Weights only appears on a
    card whose layer owns parameters, and the card this step lands on is
    the visitor's choice, so both messages ship: the driver drops the arrow
    to a Weights button that isn't there, and `alt_text` drops its name
    along with it rather than pointing the visitor at another card.
    """
    return TourStep(
        "Weights, Experiment, and Stats go deeper.",
        (
            _card_anchor("weights"),
            _card_anchor("experiment"),
            _card_anchor("stats"),
        ),
        ensure_card=True,
        alt_text="Experiment and Stats go deeper.",
    )


def main_tour_steps(layer_slug: str | None, *, locked: bool) -> list[TourStep]:
    """The main view's steps: click a layer, read its card, go deeper.

    Two of those steps are the card's only written key: the strips step
    names its two rows in the order they sit in, and the step after rings
    the colorbar to say what its diverging colors mean. Nothing else in the
    UI spells either out — the row markers just read ACTIVATIONS /
    GRADIENTS and the colorbar just prints `+x` / `0` / `-x` — and the
    playground's embedded visitors never see the docs that do.

    The tour opens on `layer_slug`, the layer whose card the page already
    shows (`main_page._pick_tour_layer`): the first arrow points at its
    diagram node, and the card steps fall back to it when the visitor has
    no card open. They don't insist on it — their anchors are bare, scoped
    in the browser to whichever card is open — so a visitor who takes step
    1's invitation on some other layer keeps the card they chose. Falls
    back to the first diagram node when no layer is known (a model with no
    captured layers) — the arrow still lands on something sensible.

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
            "A card opens: activations above, gradients below.",
            (_card_anchor("strips"),),
            ensure_card=True,
        ),
        TourStep(
            "Red is positive, blue negative, white zero.",
            (_card_anchor("legend"),),
            ensure_card=True,
        ),
        _buttons_step(),
        TourStep(
            "Select the input to inspect.",
            ('[data-tour="input-image"]', '[data-tour="sample"]'),
            ensure_input=True,
        ),
    ]
    if locked:
        steps.append(_playground_closing_step())
    else:
        steps.append(
            TourStep(
                "Run or step training from here.",
                ('[data-tour="step-controls"]',),
            )
        )
    return steps


def _playground_closing_step() -> TourStep:
    """The locked playground's last step: what this is, and how to get it.

    Everything before it teaches the UI; nothing before it says that the
    thing being driven is a library. The playground is where that gap
    hurts — its visitors arrive inside a docs iframe or on a bare Space
    URL, having never seen the repo — and the tour's last panel is the one
    moment their attention is already held.

    Two arrows: the in-app one rings the brand mark, which is the top bar's
    link to the repo, and `host_anchor` sends a second one out of the frame
    to the docs header's "one prompt" call to action — the shortest path
    from here to a wired-up training loop of their own.
    """
    return TourStep(
        "This is a hosted playground — run NaNsense on your own net, "
        "fully unlocked.",
        ('[data-tour="brand"]',),
        host_anchor=True,
    )


def stats_tour_steps() -> list[TourStep]:
    """The Stats page's steps: the sidebar dropdowns, then one per view.

    Step 1 covers the three dropdowns in one message; each following step
    describes one view and forces it (`ensure_view`), so its arrows point
    at a live example: the activation/gradient histograms, a MIN/MAX grid's
    per-channel column, and a GRAPHS epoch series. Those three lead with
    the view's own name because the step switched the page to it under the
    visitor — the name ties the message to the View dropdown they just saw.
    """
    return [
        TourStep(
            "Pick the view, phase, and layer.",
            (
                '[data-tour="view"]',
                '[data-tour="phase"]',
                '[data-tour="layer"]',
            ),
        ),
        TourStep(
            "HISTOGRAM: activation and gradient distributions.",
            ('[data-tour="hist-activation"]', '[data-tour="hist-gradient"]'),
            ensure_view="HISTOGRAM",
        ),
        TourStep(
            "MIN/MAX: each column is one channel — the inputs that "
            "activate it most.",
            ('[data-tour="patch-column"]',),
            ensure_view="MIN/MAX",
        ),
        TourStep(
            "GRAPHS: activation and gradient statistics per epoch.",
            ('[data-tour="epoch-graph"]',),
            ensure_view="GRAPHS",
        ),
    ]


def weights_tour_steps() -> list[TourStep]:
    """The Weights page's one step: what the strips are.

    The arrow lands on the weight strip, and the message covers the
    gradient and optimizer-state strips below it (they only appear once
    training has stepped, so they shouldn't be mistaken for more weights).
    """
    return [
        TourStep(
            "The parameter's values, with its gradient and optimizer "
            "state below.",
            ('[data-tour="weight-strips"]',),
        ),
    ]


def experiment_tour_steps(*, locked: bool) -> list[TourStep]:
    """The Experiment page's steps.

    Step 1 points at the kind and layer selectors. Step 2 is per-flavor:
    the playground (locked) turns auto-run off — the page's first
    experiment still self-starts, but re-runs take a manual Run — so its
    visitors get the Run / Cancel pair pointed out; live runs auto-run by
    default and instead get the page's one real gotcha — experiments
    execute on the training thread, so nothing runs until training pauses.
    That step's arrow rings the step controls, which is where the visitor
    pauses, so the message doesn't have to name the buttons (the
    playground's training is always parked, and its step cluster is
    replaced by the demo notice anyway).
    """
    steps = [
        TourStep(
            "Pick the experiment and the layer.",
            ('[data-tour="kind"]', '[data-tour="layer"]'),
        ),
    ]
    if locked:
        steps.append(
            TourStep(
                "Press Run after changing a parameter.",
                ('[data-tour="run"]',),
            )
        )
    else:
        steps.append(
            TourStep(
                "Experiments only run while training is paused.",
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
                "hostAnchor": step.host_anchor,
                "altText": step.alt_text,
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

    `auto_watch_slug` (main page only) is the card steps' layer: the one
    they prefer among the open cards, and the one they ask the page to open
    when the visitor has none. Only a locked session (the
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

  // Whoever embeds us keeps the seen flags when they offer to (the docs
  // pages do, via `docs/javascripts/playground-embed.js`): the hosted
  // playground is two Spaces — two origins — swapped into one frame, so an
  // origin-scoped flag set under one demo is invisible to the other and
  // switching demos replayed every tour from step 1. This origin's own
  // localStorage stays the fallback, which is what every unembedded case
  // uses: a local run, a direct Space visit, huggingface.co's own Space
  // wrapper (an embedder that doesn't answer), a browser that blocks
  // third-party storage.
  const HOST_RETRY_MS = 200;
  const HOST_WAIT_MS = 700;

  // The closing step's second arrow points at the embedding docs page's
  // "one prompt" call to action, which sits in the parent header — above
  // our viewport, in a document we cannot read across origins. So we ask:
  // the host answers with that button's centre x expressed in *our*
  // coordinates, and the tip lands on our own top edge beneath it, which
  // reads as pointing up and out of the frame. `null` is the host saying
  // it has no such button (the home page's embed), and the step then keeps
  // only its in-app arrow. An embedder that never answers falls back to
  // the share below — where the header's flex layout puts the button, with
  // its own auto margins and the variant switch's competing for the free
  // space (`docs/stylesheets/extra.css`).
  const HOST_ANCHOR_FALLBACK_X = 0.66;
  const HOST_ANCHOR_TIP_Y = 8;
  let anchorX;                 // undefined until the host answers
  let anchorSilent = false;    // nobody answered in time — use the fallback
  let anchorAskedWidth = -1;   // the width the standing answer was asked at

  function localSeen() {
    try { return !!localStorage.getItem(cfg.seenKey); }
    catch (e) { return false; }
  }
  function tellHost(action) {
    if (window.parent === window) return;
    try {
      window.parent.postMessage(
        { nansenseTour: action, key: cfg.seenKey }, '*');
    } catch (e) {}
  }
  function markSeen() {
    try { localStorage.setItem(cfg.seenKey, '1'); } catch (e) {}
    tellHost('set');
  }

  // The host's flag if one answers, this origin's own otherwise. The ask
  // repeats while we wait: the frame can be ready before the page holding
  // it, and a dropped first ask costs the visitor a replayed tour.
  function resolveSeen(cb) {
    if (window.parent === window) { cb(localSeen()); return; }
    let done = false, retry = null;
    function finish(value) {
      if (done) return;
      done = true;
      if (retry) clearInterval(retry);
      window.removeEventListener('message', onHostReply);
      cb(value);
    }
    function onHostReply(e) {
      const data = e.data;
      if (e.source !== window.parent || !data) return;
      if (data.nansenseTour !== 'is' || data.key !== cfg.seenKey) return;
      // A flag from this origin still counts — it's what visitors who
      // dismissed a tour before the host kept the flags have — and is
      // handed over so the host can answer for it from now on.
      if (!data.seen && localSeen()) { tellHost('set'); finish(true); return; }
      finish(!!data.seen);
    }
    window.addEventListener('message', onHostReply);
    tellHost('get');
    retry = setInterval(function() { tellHost('get'); }, HOST_RETRY_MS);
    setTimeout(function() { finish(localSeen()); }, HOST_WAIT_MS);
  }

  function onHostAnchor(e) {
    const data = e.data;
    if (e.source !== window.parent || !data) return;
    if (data.nansenseTour !== 'anchorAt') return;
    anchorX = typeof data.x === 'number' ? data.x : null;
  }

  // Asked once per distinct viewport width: the button's x moves when the
  // header reflows, and nothing else moves it. A dropped ask costs one
  // arrow's precision, not the arrow — the wait below hands over to the
  // fallback share, and any resize asks again.
  function askHostAnchor() {
    if (window.parent === window) return;
    if (anchorAskedWidth === window.innerWidth) return;
    if (anchorAskedWidth < 0) {
      window.addEventListener('message', onHostAnchor);
      setTimeout(function() {
        anchorSilent = anchorX === undefined;
      }, HOST_WAIT_MS);
    }
    anchorAskedWidth = window.innerWidth;
    try {
      window.parent.postMessage({ nansenseTour: 'anchor' }, '*');
    } catch (e) {}
  }

  // Where the out-of-frame arrow lands, or null when there is nothing above
  // us to point at: an unembedded run (a local session, a direct Space
  // visit) or a host that answered that it has no call to action.
  function hostAnchorPoint() {
    if (window.parent === window || anchorX === null) return null;
    const x = typeof anchorX === 'number'
      ? anchorX
      : (anchorSilent ? window.innerWidth * HOST_ANCHOR_FALLBACK_X : null);
    return x === null ? null : { x: x, y: HOST_ANCHOR_TIP_Y };
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

  // Which layer card the main view's card-bound steps are about. The tour
  // opens on one layer, but its first step invites a diagram click: by the
  // time these steps run the visitor may have opened another card and closed
  // that one, and pointing back at it (worse, re-opening it) reads as the
  // tour undoing their click. So the tour's own layer wins whenever its card
  // is on screen, any other open card beats none, and only with nothing open
  // does `showStep` ask the page for the tour's layer. Hidden cards keep
  // their DOM, so a zero-size rect is what "closed" looks like.
  function openCardSlug() {
    let other = null;
    for (const card of document.querySelectorAll('[data-layer]')) {
      const r = card.getBoundingClientRect();
      if (r.width === 0 || r.height === 0) continue;
      const slug = card.getAttribute('data-layer');
      if (slug === cfg.autoWatchSlug) return slug;
      if (other === null) other = slug;
    }
    return other;
  }

  // Card anchors ship bare (`[data-tour="strips"]`) and are scoped to that
  // card here; unscoped they would resolve to whichever card comes first in
  // the pane, which is rarely the one the step is about.
  function scopedSelectors(step, slug) {
    if (!step.ensureCard || !slug) return step.selectors;
    const scope = '[data-layer="' + slug.replace(/"/g, '') + '"] ';
    return step.selectors.map(function(sel) { return scope + sel; });
  }

  // The message, for a step that carries one per card flavor: the buttons
  // step names the buttons it rings, and a layer owning no parameters has no
  // Weights button on its card. Which card the step landed on is only known
  // here, so `altText` takes over when that first anchor has no target —
  // but only once there is a card to check it against, since a step still
  // waiting for its auto-shown card would otherwise read as the short one.
  function stepText(step, selectors, cardOpen) {
    if (!step.altText || !cardOpen) return step.text;
    return findTarget(selectors[0]) ? step.text : step.altText;
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

  function drawArrow(from, to) {
    const path = document.createElementNS(SVG_NS, 'path');
    path.setAttribute('data-arrow', '1');
    path.setAttribute('d', arrowPath(from, to));
    path.setAttribute('fill', 'none');
    path.setAttribute('stroke', 'rgb(245 158 11)');
    path.setAttribute('stroke-width', '2.5');
    path.setAttribute('marker-end', 'url(#nansense-tour-head)');
    svg.appendChild(path);
  }

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
    // Re-resolved every tick, like the rects: the auto-shown card arrives a
    // round-trip late, and the visitor stays free to open and close cards
    // mid-step — the arrows and the message follow what is on screen.
    const slug = step.ensureCard ? openCardSlug() : null;
    const selectors = scopedSelectors(step, slug);
    const text = stepText(step, selectors, slug !== null);
    if (textEl.textContent !== text) textEl.textContent = text;
    for (const el of overlay.querySelectorAll('.nansense-tour-ring')) {
      el.remove();
    }
    for (const el of svg.querySelectorAll('path[data-arrow]')) el.remove();
    const boxRect = box.getBoundingClientRect();
    const start = { x: boxRect.left + boxRect.width / 2, y: boxRect.top };
    let first = null;
    for (const sel of selectors) {
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
      drawArrow(start, edgePoint(r, start));
    }
    // The one target that isn't in this document: no ring, just an arrow
    // to our top edge under the embedder's call to action.
    if (step.hostAnchor) {
      askHostAnchor();
      const tip = hostAnchorPoint();
      if (tip) drawArrow(start, tip);
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
    // Card-bound steps talk about a card the visitor actually has open
    // (`openCardSlug`); only when there is none at all does the page get
    // asked for the tour's own layer. The event is show-only (a toggle would
    // hide a card the visitor already opened); on a locked session the show
    // is per-tab, like a diagram click.
    if (step.ensureCard && cfg.autoWatchSlug && openCardSlug() === null) {
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
    // The message itself is `reposition`'s: it depends on the card the step
    // lands on, which may still be on its way here.
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

  // The flag lookup and the settle beat run together, so asking the host
  // costs nothing when it answers promptly.
  function autoStart() {
    let waited = false, flag = null;
    function ready() {
      if (waited && flag === false) window.nansenseStartTour();
    }
    resolveSeen(function(value) { flag = value; ready(); });
    // A beat after load so the first arrow lands on a rendered diagram in
    // the common case (the 200 ms tick catches it regardless).
    setTimeout(function() { waited = true; ready(); }, 800);
  }

  if (cfg.autoStart) autoStart();
})();
</script>
"""
