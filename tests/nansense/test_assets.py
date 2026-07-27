"""Tests for the packaged binary assets in nansense.assets."""

from __future__ import annotations

from pathlib import Path

import nansense
from nansense.assets import logo_small_bytes, logo_small_path

_PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


def test_logo_small_path_resolves_inside_the_package() -> None:
    # The favicon must come from inside the installed package: a wheel ships
    # no repo-level `assets/` directory, so resolving relative to the source
    # tree 500s every page render on an installed copy (the original bug).
    path = logo_small_path()
    assert path.is_file()
    package_dir = Path(nansense.__file__).resolve().parent
    assert path.resolve().is_relative_to(package_dir)


def test_logo_small_path_is_cached() -> None:
    assert logo_small_path() is logo_small_path()


def test_logo_small_bytes_is_a_png() -> None:
    assert logo_small_bytes().startswith(_PNG_MAGIC)
