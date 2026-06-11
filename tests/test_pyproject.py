"""Consistency checks for the PyTorch variant extras in pyproject.toml.

PyTorch is installed through mutually exclusive extras (cpu / cu126 / cu130 /
cu132 / rocm7-2), each pinned to its PyTorch wheel index. These tests guard
against the extras, conflict declaration, sources, and index definitions
drifting apart when dependencies are edited.
"""

import tomllib
from pathlib import Path
from typing import Any

import pytest

TORCH_EXTRAS = ["cpu", "cu126", "cu130", "cu132", "rocm7-2"]


def _load_pyproject() -> dict[str, Any]:
    path = Path(__file__).parent.parent / "pyproject.toml"
    with path.open("rb") as f:
        return tomllib.load(f)


def _requirement_names(requirements: list[str]) -> set[str]:
    return {r.split(";")[0].strip().split(">=")[0].split("==")[0].strip() for r in requirements}


def test_torch_extras_exist() -> None:
    extras = _load_pyproject()["project"]["optional-dependencies"]
    assert set(extras) == set(TORCH_EXTRAS)


@pytest.mark.parametrize("extra", TORCH_EXTRAS)
def test_extra_contains_torch_and_torchvision(extra: str) -> None:
    extras = _load_pyproject()["project"]["optional-dependencies"]
    assert {"torch", "torchvision"} <= _requirement_names(extras[extra])


def test_torch_not_in_base_dependencies() -> None:
    data = _load_pyproject()
    base = _requirement_names(data["project"]["dependencies"])
    dev = _requirement_names(data["dependency-groups"]["dev"])
    assert {"torch", "torchvision"}.isdisjoint(base | dev)


def test_extras_declared_mutually_exclusive() -> None:
    conflicts = _load_pyproject()["tool"]["uv"]["conflicts"]
    conflict_sets = [{entry["extra"] for entry in group} for group in conflicts]
    assert set(TORCH_EXTRAS) in conflict_sets


@pytest.mark.parametrize("package", ["torch", "torchvision"])
def test_sources_cover_all_extras(package: str) -> None:
    data = _load_pyproject()
    sources = data["tool"]["uv"]["sources"][package]
    assert {s["extra"] for s in sources} == set(TORCH_EXTRAS)
    index_names = {idx["name"] for idx in data["tool"]["uv"]["index"]}
    assert {s["index"] for s in sources} <= index_names


def test_indexes_are_explicit_pytorch_urls() -> None:
    for idx in _load_pyproject()["tool"]["uv"]["index"]:
        assert idx["explicit"] is True
        assert idx["url"].startswith("https://download.pytorch.org/whl/")
