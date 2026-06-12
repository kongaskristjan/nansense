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
