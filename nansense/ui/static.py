"""Static CSS/JS string constants shipped with the UI pages."""


_ARCHITECTURE_CLICK_CSS: str = """
<style>
  g.node { cursor: pointer; }
  [data-layer] > :first-child { cursor: pointer; }
  [data-layer].nansense-highlight {
    box-shadow: 0 0 0 3px rgb(96 165 250);
  }
  /* SVG nodes don't honour `box-shadow`, so the matching highlight uses
     an SVG filter that glows around the node's shape. */
  g.node.nansense-highlight {
    filter: drop-shadow(0 0 4px rgb(96 165 250));
  }
  /* Watched: stronger, amber-tinted treatment that persists across hover.
     Distinct from the blue hover highlight so the two signals don't
     blur into one. */
  [data-layer].nansense-watched {
    box-shadow:
      0 0 0 3px rgb(245 158 11),
      0 0 12px rgba(245, 158, 11, 0.55);
  }
  g.node.nansense-watched {
    filter:
      drop-shadow(0 0 6px rgb(245 158 11))
      drop-shadow(0 0 3px rgb(245 158 11));
  }
  /* Watched + hovered: amber ring stays, blue layered around it. */
  [data-layer].nansense-watched.nansense-highlight {
    box-shadow:
      0 0 0 3px rgb(245 158 11),
      0 0 0 6px rgba(96, 165, 250, 0.6);
  }
  g.node.nansense-watched.nansense-highlight {
    filter:
      drop-shadow(0 0 6px rgb(245 158 11))
      drop-shadow(0 0 4px rgb(96 165 250));
  }
  /* Hyperparameter tooltip shown while hovering a diagram node or a card
     header (see the layer-info block in the JS blob). `pointer-events:
     none` keeps the cursor-chasing div from stealing the hover it serves. */
  .nansense-layer-tooltip {
    position: fixed;
    z-index: 10000;
    max-width: 28rem;
    padding: 4px 8px;
    background: rgb(15 23 42 / 0.92);
    color: rgb(241 245 249);
    font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
    font-size: 12px;
    line-height: 1.4;
    border-radius: 4px;
    pointer-events: none;
    overflow-wrap: anywhere;
    display: none;
  }
</style>
"""


# Mermaid SVG node ids look like "<element>-flowchart-<slug>-<counter>"; the
# matching layer card carries `data-layer="<slug>"` so we can cross-link
# the two. Hovering either side adds `.nansense-highlight` to both ends
# of the pair. Clicking a diagram node emits `nansense_toggle_layer` to
# the server, which toggles the layer's watched state (and with it the
# card's visibility); clicking a card header scrolls the diagram to the
# matching node. Scroll positions are computed directly instead of via
# `scrollIntoView`, because the latter leaves the target several dozen
# pixels below the column's top edge here (the previous item's tail stays
# visible), even with `block: 'start'`.
_ARCHITECTURE_CLICK_JS: str = """
<script>
(function() {
  const watchedSlugs = new Set();

  function slugFromMermaidId(id) {
    const m = /-flowchart-(.+)-\\d+$/.exec(id || '');
    return m ? m[1] : null;
  }
  function findMermaidNode(slug) {
    return document.querySelector(
      'g.node[id*="-flowchart-' + slug.replace(/"/g, '') + '-"]'
    );
  }
  function findCard(slug) {
    return document.querySelector(
      '[data-layer="' + slug.replace(/"/g, '') + '"]'
    );
  }
  function matchPair(el) {
    if (!el || !el.closest) return null;
    const node = el.closest('g.node');
    if (node) {
      const slug = slugFromMermaidId(node.id);
      if (!slug) return null;
      const card = findCard(slug);
      if (!card) return null;
      return { node: node, card: card };
    }
    const card = el.closest('[data-layer]');
    if (card) {
      const slug = card.getAttribute('data-layer');
      const node = findMermaidNode(slug);
      if (!node) return null;
      return { node: node, card: card };
    }
    return null;
  }
  function scrollableParent(el) {
    let p = el.parentElement;
    while (p) {
      const oy = getComputedStyle(p).overflowY;
      if ((oy === 'auto' || oy === 'scroll') && p.scrollHeight > p.clientHeight) {
        return p;
      }
      p = p.parentElement;
    }
    return null;
  }
  function scrollTargetToTop(target) {
    const container = scrollableParent(target);
    if (!container) return;
    const cRect = container.getBoundingClientRect();
    const tRect = target.getBoundingClientRect();
    const topPadding = 12;
    container.scrollTo({
      top: container.scrollTop + (tRect.top - cRect.top) - topPadding,
      behavior: 'smooth',
    });
  }

  let highlighted = null;
  function setHighlight(pair) {
    if (highlighted && pair && highlighted.node === pair.node) return;
    if (highlighted) {
      highlighted.node.classList.remove('nansense-highlight');
      highlighted.card.classList.remove('nansense-highlight');
    }
    highlighted = pair;
    if (pair) {
      pair.node.classList.add('nansense-highlight');
      pair.card.classList.add('nansense-highlight');
    }
  }
  document.addEventListener('mouseover', function(e) {
    setHighlight(matchPair(e.target));
  });
  document.addEventListener('mouseleave', function() {
    setHighlight(null);
  });

  // Layer-info tooltip: the page publishes `window.nansenseLayerInfo`
  // (slug -> hyperparameter string, see `_layer_info_script`); hovering a
  // diagram node or a card *header* shows the hovered layer's entry next
  // to the cursor. Layers without an entry (relu, add, inputs) show none.
  const infoTooltip = document.createElement('div');
  infoTooltip.className = 'nansense-layer-tooltip';
  document.body.appendChild(infoTooltip);

  function infoSlug(el) {
    if (!el || !el.closest) return null;
    const node = el.closest('g.node');
    if (node) return slugFromMermaidId(node.id);
    const card = el.closest('[data-layer]');
    // Only the header names the layer; the strip area below is data.
    if (card && card.firstElementChild && card.firstElementChild.contains(el)) {
      return card.getAttribute('data-layer');
    }
    return null;
  }
  function moveInfoTooltip(e) {
    const pad = 14;
    const rect = infoTooltip.getBoundingClientRect();
    let x = e.clientX + pad;
    let y = e.clientY + pad;
    if (x + rect.width > window.innerWidth - 4) x = e.clientX - rect.width - pad;
    if (y + rect.height > window.innerHeight - 4) y = e.clientY - rect.height - pad;
    infoTooltip.style.left = Math.max(x, 4) + 'px';
    infoTooltip.style.top = Math.max(y, 4) + 'px';
  }
  document.addEventListener('mousemove', function(e) {
    const slug = infoSlug(e.target);
    const info = slug && window.nansenseLayerInfo
      ? window.nansenseLayerInfo[slug] : null;
    if (!info) {
      infoTooltip.style.display = 'none';
      return;
    }
    infoTooltip.textContent = info;
    infoTooltip.style.display = 'block';
    moveInfoTooltip(e);
  });
  document.addEventListener('mouseleave', function() {
    infoTooltip.style.display = 'none';
  });

  document.addEventListener('click', function(e) {
    if (!e.target.closest) return;
    // Header action buttons (Watch, Weights) handle their own click; the
    // document-level navigation must not fire on top of them.
    if (e.target.closest('[data-card-action]')) return;
    const node = e.target.closest('g.node');
    if (node) {
      const slug = slugFromMermaidId(node.id);
      if (!slug) return;
      // Toggling watched state lives server-side (session.watch); the
      // server answers by updating card visibility and amber classes.
      emitEvent('nansense_toggle_layer', slug);
      return;
    }
    const card = e.target.closest('[data-layer]');
    if (!card) return;
    // Only the card header (the first child) navigates back to the diagram;
    // clicks inside the strip area shouldn't trigger a jump.
    const header = card.firstElementChild;
    if (!header || !header.contains(e.target)) return;
    const slug = card.getAttribute('data-layer');
    const mNode = findMermaidNode(slug);
    if (!mNode) return;
    scrollTargetToTop(mNode);
  });

  // Toggle the `nansense-watched` class on both the card and the matching
  // mermaid node. Mermaid renders the SVG asynchronously, so the node may
  // not exist yet when this runs; the MutationObserver below catches it.
  window.nansenseSetWatched = function(slug, on) {
    if (on) { watchedSlugs.add(slug); } else { watchedSlugs.delete(slug); }
    const card = findCard(slug);
    if (card) card.classList.toggle('nansense-watched', on);
    const node = findMermaidNode(slug);
    if (node) node.classList.toggle('nansense-watched', on);
  };
  // Jump both panes to a layer: the right pane's card and the architecture
  // pane's mermaid node each scroll within their own container.
  window.nansenseScrollToLayer = function(slug) {
    const card = findCard(slug);
    if (card) scrollTargetToTop(card);
    const node = findMermaidNode(slug);
    if (node) scrollTargetToTop(node);
  };
  // Card-only variant: used right after a diagram click reveals a card,
  // where also scrolling the diagram would yank the just-clicked node away
  // from under the cursor.
  window.nansenseScrollToCard = function(slug) {
    const card = findCard(slug);
    if (card) scrollTargetToTop(card);
  };

  // Re-apply watched classes to any matching mermaid node / card that
  // appears after the initial render. Skips work when nothing is watched.
  const observer = new MutationObserver(function() {
    if (watchedSlugs.size === 0) return;
    for (const slug of watchedSlugs) {
      const card = findCard(slug);
      if (card && !card.classList.contains('nansense-watched')) {
        card.classList.add('nansense-watched');
      }
      const node = findMermaidNode(slug);
      if (node && !node.classList.contains('nansense-watched')) {
        node.classList.add('nansense-watched');
      }
    }
  });
  observer.observe(document.body, { childList: true, subtree: true });
})();
</script>
"""


# GIMP-style transparency backdrop for strip data images: a 4px-box gray
# checkerboard, fixed at *display* resolution (independent of the CSS
# `image-rendering: pixelated` upscale of the data image in front of it).
# Every strip data img carries this background; opaque (all-finite) strips
# fully cover it, and only the transparent NaN/±Inf cells of a divergent
# strip reveal it — reading as "no value here" rather than a misleading
# color or white. The two slate grays match `recording.CHECKER_*` so live
# UI and recorded MP4 frames look identical.
_STRIP_CHECKERBOARD_STYLE: str = (
    "background-color:#f9fafb;"
    "background-image:"
    "linear-gradient(45deg,#e5e7eb 25%,transparent 25%,transparent 75%,#e5e7eb 75%),"
    "linear-gradient(45deg,#e5e7eb 25%,transparent 25%,transparent 75%,#e5e7eb 75%);"
    "background-size:8px 8px;"
    "background-position:0 0,4px 4px;"
)


# Side panes are resizable by dragging the thin handle sitting between
# the pane and the center content (`_resize_handle` in `common.py`). The
# handle's resting color matches the center panes' `bg-slate-200` so it
# reads as part of the center padding until hovered. `touch-action: none`
# lets the pointer-capture drag work on touch screens too.
_PANEL_RESIZE_CSS: str = """
<style>
  .nansense-resize-handle {
    flex: 0 0 6px;
    align-self: stretch;
    cursor: col-resize;
    background: rgb(226 232 240);
    touch-action: none;
  }
  .nansense-resize-handle:hover,
  .nansense-resize-handle.nansense-resizing {
    background: rgb(96 165 250 / 0.7);
  }
</style>
"""


# Dragging a `.nansense-resize-handle` sets an inline `width` on the pane
# carrying the matching `data-resize-pane` key (inline style beats the
# pane's Tailwind width class, which thereby stays the default). The width
# is saved to sessionStorage on release, so it survives navigation between
# pages but lasts only for the browser session, and is re-applied on every
# page load. Vue mounts the elements after this script runs, so restoring
# happens via a MutationObserver as panes appear rather than once at load.
_PANEL_RESIZE_JS: str = """
<script>
(function() {
  const STORAGE_PREFIX = 'nansense-panel-width:';
  const MIN_WIDTH = 150;

  function clampWidth(px) {
    const max = Math.max(MIN_WIDTH, window.innerWidth * 0.6);
    return Math.min(Math.max(px, MIN_WIDTH), max);
  }
  function paneFor(handle) {
    const key = handle.getAttribute('data-resize-key');
    if (!key) return null;
    return document.querySelector(
      '[data-resize-pane="' + key.replace(/"/g, '') + '"]'
    );
  }

  const restored = new WeakSet();
  function restore() {
    document.querySelectorAll('[data-resize-pane]').forEach(function(pane) {
      if (restored.has(pane)) return;
      restored.add(pane);
      const key = pane.getAttribute('data-resize-pane');
      const stored = parseFloat(sessionStorage.getItem(STORAGE_PREFIX + key));
      if (isFinite(stored)) pane.style.width = clampWidth(stored) + 'px';
    });
  }
  new MutationObserver(restore).observe(
    document.documentElement, { childList: true, subtree: true }
  );
  restore();

  document.addEventListener('pointerdown', function(e) {
    if (!e.target.closest) return;
    const handle = e.target.closest('.nansense-resize-handle');
    if (!handle) return;
    const pane = paneFor(handle);
    if (!pane) return;
    e.preventDefault();
    // Capture keeps move/up events on the handle even when the pointer
    // outruns the 6px strip; not supported for some synthetic pointers.
    try { handle.setPointerCapture(e.pointerId); } catch (err) {}
    // 'right' marks a pane on the right edge of the view: its handle sits
    // on the pane's left side, so dragging left grows the pane.
    const grow = handle.getAttribute('data-resize-side') === 'right' ? -1 : 1;
    const startX = e.clientX;
    const startWidth = pane.getBoundingClientRect().width;
    handle.classList.add('nansense-resizing');
    document.body.style.userSelect = 'none';
    document.body.style.cursor = 'col-resize';
    function onMove(ev) {
      pane.style.width =
        clampWidth(startWidth + grow * (ev.clientX - startX)) + 'px';
    }
    function onUp() {
      handle.removeEventListener('pointermove', onMove);
      handle.removeEventListener('pointerup', onUp);
      handle.removeEventListener('pointercancel', onUp);
      handle.classList.remove('nansense-resizing');
      document.body.style.userSelect = '';
      document.body.style.cursor = '';
      const key = pane.getAttribute('data-resize-pane');
      sessionStorage.setItem(
        STORAGE_PREFIX + key, String(pane.getBoundingClientRect().width)
      );
    }
    handle.addEventListener('pointermove', onMove);
    handle.addEventListener('pointerup', onUp);
    handle.addEventListener('pointercancel', onUp);
  });

  // Double-click restores the pane's default (class-derived) width.
  document.addEventListener('dblclick', function(e) {
    if (!e.target.closest) return;
    const handle = e.target.closest('.nansense-resize-handle');
    if (!handle) return;
    const pane = paneFor(handle);
    if (!pane) return;
    pane.style.width = '';
    sessionStorage.removeItem(
      STORAGE_PREFIX + pane.getAttribute('data-resize-pane')
    );
  });
})();
</script>
"""


# The marker's vertical label is hidden on strips too short to fit it
# (1D heatmap rows, the no-gradient placeholder); the tallest label is
# ~75px, so anything under 88px can't show it cleanly. The 128px conv
# tiles clear the threshold comfortably.
_STRIP_MARKER_CSS: str = """
<style>
  .nansense-marker { container-type: size; }
  @container (max-height: 88px) {
    .nansense-marker-label { display: none; }
  }
</style>
"""
