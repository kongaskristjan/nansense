"""Render the README's "wire it into your loop" snippets as side-by-side SVG diffs.

    uv run assets/code-examples/render_diffs.py

For each `*_before.py` / `*_after.py` pair in this directory it writes one SVG
showing the two versions side by side, tinting added / removed lines and
highlighting the differing *characters* within lines that were modified. Each
SVG embeds a `prefers-color-scheme` media query, so it adapts to GitHub's light
and dark themes from a single file (falling back to light where the query is
unsupported).

Alignment is done in two passes: a line-level `difflib` diff, then a
similarity-scored alignment within each replace block so only genuinely
corresponding lines are paired (and char-diffed) while the rest read as clean
additions / removals — the standard "split view" look.
"""

from __future__ import annotations

import difflib
from dataclasses import dataclass, field
from html import escape
from pathlib import Path

# --- Geometry (px). A forced `textLength` makes every glyph occupy exactly
# CHAR_W, so character-highlight rectangles line up regardless of the viewer's
# monospace font. ---
FONT_SIZE = 13.0
CHAR_W = 7.8
LINE_H = 19.0
HEADER_H = 34.0
BODY_PAD_Y = 8.0
PAD_L = 10.0
PAD_R = 12.0
GUTTER_GAP = 6.0  # between gutter and sign
SIGN_GAP = 6.0  # between sign and code
FONT_FAMILY = "ui-monospace, SFMono-Regular, Menlo, Consolas, 'Liberation Mono', monospace"

# Pair lines inside a replace block as a modification (with char highlights)
# only when they are at least this similar; below it they read as a plain
# remove + add.
MATCH_THRESHOLD = 0.4

Span = tuple[int, int]


@dataclass(frozen=True)
class Side:
    """One column's view of a row."""

    lineno: int | None  # None when this side has no line here
    text: str
    spans: tuple[Span, ...] = ()  # char ranges to highlight (modifications only)


@dataclass(frozen=True)
class Row:
    left: Side
    right: Side
    kind: str  # "equal" | "del" | "add" | "change"


def _char_spans(a: str, b: str) -> tuple[tuple[Span, ...], tuple[Span, ...]]:
    """Char ranges that differ between two lines, for the left and right sides."""
    left: list[Span] = []
    right: list[Span] = []
    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(None, a, b, autojunk=False).get_opcodes():
        if tag == "equal":
            continue
        if i2 > i1:
            left.append((i1, i2))
        if j2 > j1:
            right.append((j1, j2))
    return tuple(left), tuple(right)


def _align_replace(left: list[str], right: list[str]) -> list[tuple[str, int, int]]:
    """Order-preserving alignment of two blocks, maximizing total similarity of
    matched line pairs (only pairs at/above MATCH_THRESHOLD may match).

    Returns ops in display order: ("match", i, j), ("del", i, -1), ("add", -1, j).
    """
    m, n = len(left), len(right)
    ratios: list[list[float]] = [
        [difflib.SequenceMatcher(None, left[i], right[j], autojunk=False).ratio() for j in range(n)]
        for i in range(m)
    ]
    # dp[i][j]: best score aligning left[:i] with right[:j]; choice records the move.
    dp: list[list[float]] = [[0.0] * (n + 1) for _ in range(m + 1)]
    choice: list[list[str]] = [[""] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if dp[i - 1][j] >= dp[i][j - 1]:
                dp[i][j], choice[i][j] = dp[i - 1][j], "del"
            else:
                dp[i][j], choice[i][j] = dp[i][j - 1], "add"
            r = ratios[i - 1][j - 1]
            if r >= MATCH_THRESHOLD and dp[i - 1][j - 1] + r > dp[i][j]:
                dp[i][j], choice[i][j] = dp[i - 1][j - 1] + r, "match"

    ops: list[tuple[str, int, int]] = []
    i, j = m, n
    while i > 0 and j > 0:
        move = choice[i][j]
        if move == "match":
            ops.append(("match", i - 1, j - 1))
            i, j = i - 1, j - 1
        elif move == "del":
            ops.append(("del", i - 1, -1))
            i -= 1
        else:
            ops.append(("add", -1, j - 1))
            j -= 1
    while i > 0:
        ops.append(("del", i - 1, -1))
        i -= 1
    while j > 0:
        ops.append(("add", -1, j - 1))
        j -= 1
    ops.reverse()
    return ops


@dataclass
class _Pending:
    """Buffers unmatched removals/additions so a run of them can be laid out as
    compact side-by-side rows (the surplus side spilling to one-sided rows)."""

    dels: list[Side] = field(default_factory=list)
    adds: list[Side] = field(default_factory=list)

    def flush(self, out: list[Row]) -> None:
        for k in range(max(len(self.dels), len(self.adds))):
            left = self.dels[k] if k < len(self.dels) else Side(None, "")
            right = self.adds[k] if k < len(self.adds) else Side(None, "")
            kind = "change" if k < len(self.dels) and k < len(self.adds) else (
                "del" if k < len(self.dels) else "add"
            )
            out.append(Row(left, right, kind))
        self.dels.clear()
        self.adds.clear()


def build_rows(before: list[str], after: list[str]) -> list[Row]:
    """Side-by-side rows aligning `before` against `after`."""
    rows: list[Row] = []
    pending = _Pending()
    ln_l = ln_r = 0
    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(None, before, after).get_opcodes():
        if tag == "equal":
            pending.flush(rows)
            for k in range(i2 - i1):
                ln_l += 1
                ln_r += 1
                rows.append(Row(Side(ln_l, before[i1 + k]), Side(ln_r, after[j1 + k]), "equal"))
        elif tag == "delete":
            pending.flush(rows)
            for k in range(i1, i2):
                ln_l += 1
                rows.append(Row(Side(ln_l, before[k]), Side(None, ""), "del"))
        elif tag == "insert":
            pending.flush(rows)
            for k in range(j1, j2):
                ln_r += 1
                rows.append(Row(Side(None, ""), Side(ln_r, after[k]), "add"))
        else:  # replace: align by similarity, char-diffing only matched pairs
            for move, i, j in _align_replace(before[i1:i2], after[j1:j2]):
                if move == "del":
                    ln_l += 1
                    pending.dels.append(Side(ln_l, before[i1 + i]))
                elif move == "add":
                    ln_r += 1
                    pending.adds.append(Side(ln_r, after[j1 + j]))
                else:
                    pending.flush(rows)
                    ln_l += 1
                    ln_r += 1
                    a, b = before[i1 + i], after[j1 + j]
                    lspans, rspans = _char_spans(a, b)
                    rows.append(Row(Side(ln_l, a, lspans), Side(ln_r, b, rspans), "change"))
    pending.flush(rows)
    return rows


def _svg_text(x: float, y: float, text: str, cls: str) -> str:
    """A left-anchored monospace run pinned to an exact width so glyphs sit on
    the CHAR_W grid the highlight rectangles use."""
    if not text:
        return ""
    return (
        f'<text x="{x:.1f}" y="{y:.2f}" class="{cls}" xml:space="preserve" '
        f'textLength="{len(text) * CHAR_W:.1f}" lengthAdjust="spacingAndGlyphs">'
        f"{escape(text)}</text>"
    )


def render_svg(rows: list[Row], left_title: str, right_title: str) -> str:
    """Render aligned rows to a self-contained, theme-aware SVG string."""
    max_chars = max((len(r.left.text) for r in rows), default=0)
    max_chars = max(max_chars, *(len(r.right.text) for r in rows)) if rows else 0
    max_lineno = max(
        (s.lineno or 0 for r in rows for s in (r.left, r.right)), default=1
    )
    gutter_w = len(str(max_lineno)) * CHAR_W
    sign_x = PAD_L + gutter_w + GUTTER_GAP
    code_x = sign_x + CHAR_W + SIGN_GAP
    col_w = code_x + max_chars * CHAR_W + PAD_R
    width = col_w * 2
    height = HEADER_H + BODY_PAD_Y * 2 + len(rows) * LINE_H

    bg_rects: list[str] = []  # line/char tints (drawn under text)
    fg: list[str] = []  # gutters, signs, code, headers (drawn over tints)

    def emit_side(side: Side, origin: float, tint: str, sign: str) -> None:
        if side.lineno is None:
            # No line on this side: a faint "absent" band spanning the column.
            bg_rects.append(
                f'<rect x="{origin:.1f}" y="{y:.2f}" width="{col_w:.1f}" '
                f'height="{LINE_H:.1f}" class="empty"/>'
            )
            return
        if tint:
            bg_rects.append(
                f'<rect x="{origin:.1f}" y="{y:.2f}" width="{col_w:.1f}" '
                f'height="{LINE_H:.1f}" class="{tint}-line"/>'
            )
        for s0, s1 in side.spans:
            bg_rects.append(
                f'<rect x="{origin + code_x + s0 * CHAR_W:.1f}" y="{y:.2f}" '
                f'width="{(s1 - s0) * CHAR_W:.1f}" height="{LINE_H:.1f}" class="{tint}-char"/>'
            )
        fg.append(
            f'<text x="{origin + sign_x - GUTTER_GAP:.1f}" y="{ty:.2f}" '
            f'class="gutter" text-anchor="end">{side.lineno}</text>'
        )
        if sign:
            fg.append(
                f'<text x="{origin + sign_x:.1f}" y="{ty:.2f}" class="sign-{tint}">{sign}</text>'
            )
        fg.append(_svg_text(origin + code_x, ty, side.text, "code"))

    for idx, row in enumerate(rows):
        y = HEADER_H + BODY_PAD_Y + idx * LINE_H
        ty = y + LINE_H * 0.72  # text baseline within the row
        left_tint = "del" if row.kind in ("del", "change") else ""
        right_tint = "add" if row.kind in ("add", "change") else ""
        emit_side(row.left, 0.0, left_tint, "-" if left_tint else "")
        emit_side(row.right, col_w, right_tint, "+" if right_tint else "")

    # Header band + column titles + dividers.
    hdr_baseline = HEADER_H * 0.66
    fg.append(
        f'<text x="{code_x:.1f}" y="{hdr_baseline:.2f}" class="hdr">{escape(left_title)}</text>'
    )
    fg.append(
        f'<text x="{col_w + code_x:.1f}" y="{hdr_baseline:.2f}" class="hdr">'
        f"{escape(right_title)}</text>"
    )
    dividers = (
        f'<rect x="0" y="0" width="{width:.1f}" height="{HEADER_H:.1f}" class="header"/>'
        f'<line x1="0" y1="{HEADER_H:.1f}" x2="{width:.1f}" y2="{HEADER_H:.1f}" class="rule"/>'
        f'<line x1="{col_w:.1f}" y1="0" x2="{col_w:.1f}" y2="{height:.1f}" class="rule"/>'
    )

    return _SVG_TEMPLATE.format(
        width=f"{width:.1f}",
        height=f"{height:.1f}",
        font_family=FONT_FAMILY,
        font_size=FONT_SIZE,
        radius=6,
        bg="".join(bg_rects),
        dividers=dividers,
        fg="".join(fg),
    )


_SVG_TEMPLATE = """\
<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" \
viewBox="0 0 {width} {height}" font-family="{font_family}" font-size="{font_size}">
  <style>
    .card {{ fill: #ffffff; stroke: #d0d7de; }}
    .header {{ fill: #f6f8fa; }}
    .rule {{ stroke: #d8dee4; stroke-width: 1; }}
    .hdr {{ fill: #57606a; font-weight: 600; }}
    .code {{ fill: #1f2328; }}
    .gutter {{ fill: #8c959f; }}
    .empty {{ fill: #f6f8fa; }}
    .del-line {{ fill: #ffebe9; }}
    .del-char {{ fill: #ffc1bf; }}
    .add-line {{ fill: #e6ffec; }}
    .add-char {{ fill: #a0e7af; }}
    .sign-del {{ fill: #cf222e; }}
    .sign-add {{ fill: #1a7f37; }}
    @media (prefers-color-scheme: dark) {{
      .card {{ fill: #0d1117; stroke: #30363d; }}
      .header {{ fill: #161b22; }}
      .rule {{ stroke: #21262d; }}
      .hdr {{ fill: #8b949e; }}
      .code {{ fill: #e6edf3; }}
      .gutter {{ fill: #6e7681; }}
      .empty {{ fill: #161b22; }}
      .del-line {{ fill: #3c181b; }}
      .del-char {{ fill: #6a2b2f; }}
      .add-line {{ fill: #122a1d; }}
      .add-char {{ fill: #1f5c33; }}
      .sign-del {{ fill: #f85149; }}
      .sign-add {{ fill: #3fb950; }}
    }}
  </style>
  <rect x="0.5" y="0.5" width="{width}" height="{height}" rx="{radius}" class="card"/>
  {bg}
  {dividers}
  {fg}
</svg>
"""

# (before, after, output) triples, resolved relative to this file's directory.
DIFFS: list[tuple[str, str, str, str, str]] = [
    (
        "pytorch_raw_before.py",
        "pytorch_raw_after.py",
        "pytorch_raw.svg",
        "Raw PyTorch",
        "Raw PyTorch + nansense",
    ),
    (
        "pytorch_lightning_before.py",
        "pytorch_lightning_after.py",
        "pytorch_lightning.svg",
        "PyTorch Lightning",
        "PyTorch Lightning + nansense",
    ),
]


def main() -> None:
    here = Path(__file__).resolve().parent
    for before_name, after_name, out_name, left_title, right_title in DIFFS:
        before = (here / before_name).read_text().splitlines()
        after = (here / after_name).read_text().splitlines()
        rows = build_rows(before, after)
        svg = render_svg(rows, left_title, right_title)
        (here / out_name).write_text(svg)
        print(f"wrote {out_name} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
