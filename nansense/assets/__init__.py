"""Binary assets shipped inside the package, and their accessors.

The UI needs the NaNsense mark both as a filesystem path (NiceGUI's per-page
``favicon`` argument) and as raw bytes (the top bar inlines it as a data
URI). Both accessors resolve the file with ``importlib.resources`` instead of
walking up from ``__file__`` to the repo root: an installed wheel ships only
the ``nansense`` package, so a repo-relative path exists in a source checkout
but not in site-packages.
"""

from __future__ import annotations

import atexit
from contextlib import ExitStack
from functools import lru_cache
from importlib.resources import as_file, files
from pathlib import Path

# Keeps the `as_file` context (a temp-file extraction when the package is
# imported from a zip; a passthrough for a normal directory install) alive
# for the process lifetime — the favicon path is handed to NiceGUI once at
# startup and must stay valid for as long as pages are served.
_extractions = ExitStack()
atexit.register(_extractions.close)


@lru_cache(maxsize=1)
def logo_small_path() -> Path:
    """Filesystem path of the NaNsense mark (the UI pages' favicon)."""
    return _extractions.enter_context(as_file(files(__name__) / "logo_small.png"))


def logo_small_bytes() -> bytes:
    """Raw PNG bytes of the NaNsense mark (inlined as the top-bar logo)."""
    return (files(__name__) / "logo_small.png").read_bytes()
