"""Console printing that survives whatever codec stdout happens to use.

Python only gives stdout UTF-8 on Windows when it is a *real* console. The
moment it is a pipe or a file — an agent's tool call, `> train.log`, CI — it
falls back to the legacy ANSI codepage (cp1252 in Western locales), and every
non-ASCII character NaNsense prints becomes a `UnicodeEncodeError` waiting to
happen: the em dash in a paused-run notice, `±Inf` out of the debugger, an
accented directory in a frozen-moment path.

That exception surfaces wherever the print ran. On the UI announcer thread it
killed the banner outright; from the training loop it would abort a run over a
log line. So NaNsense prints through `console_print`, which degrades the text to
what the stream can carry instead of raising: known characters get a readable
ASCII stand-in, and anything else falls through to the codec's own `?`.
"""

from __future__ import annotations

import codecs
import sys
from typing import TextIO

# Readable stand-ins for the non-ASCII characters NaNsense itself emits, used
# only once the stream has been found unable to carry the original.
_STAND_INS = str.maketrans(
    {
        "—": "-",
        "–": "-",
        "±": "+/-",
        "·": "-",
        "…": "...",
        "→": "->",
        "≈": "~",
    }
)


def stream_encoding(stream: TextIO | None = None) -> str:
    """The codec `stream` (default `sys.stdout`) writes with.

    Falls back to `ascii` — the conservative assumption — when the stream has
    no `encoding` at all (a bare `StringIO` swapped in for stdout), when it is
    `None`, or when it names a codec this Python does not have.
    """
    encoding = getattr(stream or sys.stdout, "encoding", None) or "ascii"
    try:
        codecs.lookup(encoding)
    except LookupError:
        return "ascii"
    return encoding


def encodable(text: str, stream: TextIO | None = None) -> str:
    """`text` reduced to what `stream` can encode, returned unchanged when it
    already fits.

    Tries the stand-ins first so a degraded line still reads as prose
    (`+/-Inf`, not `?Inf`); whatever has no stand-in is left to the codec's
    `replace` handler.
    """
    encoding = stream_encoding(stream)
    if _encodes(text, encoding):
        return text
    swapped = text.translate(_STAND_INS)
    if _encodes(swapped, encoding):
        return swapped
    return swapped.encode(encoding, "replace").decode(encoding, "replace")


def console_print(text: str) -> None:
    """`print(text, flush=True)`, degraded to what stdout can encode.

    Flushed because NaNsense's lines are progress notes interleaved with a
    training script's own output, and a buffered one arrives too late to mean
    anything.
    """
    print(encodable(text), flush=True)


def _encodes(text: str, encoding: str) -> bool:
    """Whether `text` survives `encoding` intact."""
    try:
        text.encode(encoding)
    except UnicodeEncodeError:
        return False
    return True
