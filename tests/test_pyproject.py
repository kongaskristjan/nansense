"""Consistency checks for the PyTorch variant dependency groups in pyproject.toml.

PyTorch is installed through mutually exclusive dependency groups (cpu /
cu126 / cu130 / cu132 / rocm7-2), each pinned to its PyTorch wheel index.
Groups are never published, which keeps the PyPI package torch-free. These
tests guard against the groups, conflict declaration, sources, and index
definitions drifting apart when dependencies are edited.
"""

import tomllib
from pathlib import Path
from typing import Any

import pytest

TORCH_GROUPS = ["cpu", "cu126", "cu130", "cu132", "rocm7-2"]


def _load_pyproject() -> dict[str, Any]:
    path = Path(__file__).parent.parent / "pyproject.toml"
    with path.open("rb") as f:
        return tomllib.load(f)


def _requirement_names(requirements: list[str]) -> set[str]:
    return {r.split(";")[0].strip().split(">=")[0].split("==")[0].strip() for r in requirements}


def test_torch_groups_exist() -> None:
    groups = _load_pyproject()["dependency-groups"]
    assert set(TORCH_GROUPS) <= set(groups)


@pytest.mark.parametrize("group", TORCH_GROUPS)
def test_group_contains_torch_and_torchvision(group: str) -> None:
    groups = _load_pyproject()["dependency-groups"]
    assert {"torch", "torchvision"} <= _requirement_names(groups[group])


def test_published_metadata_is_torch_free() -> None:
    """`pip install nansense` (with or without extras) must never pull torch
    or torchvision directly. captum and lightning depend on torch, so they
    may only appear in opt-in extras and the dev group, never in the base
    dependencies.
    """
    data = _load_pyproject()
    base = _requirement_names(data["project"]["dependencies"])
    assert {"torch", "torchvision", "captum", "lightning"}.isdisjoint(base)
    for extra, requirements in data["project"]["optional-dependencies"].items():
        assert {"torch", "torchvision"}.isdisjoint(_requirement_names(requirements)), extra


def test_dev_group_has_no_direct_torch() -> None:
    dev = _requirement_names(_load_pyproject()["dependency-groups"]["dev"])
    assert {"torch", "torchvision"}.isdisjoint(dev)


def test_groups_declared_mutually_exclusive() -> None:
    conflicts = _load_pyproject()["tool"]["uv"]["conflicts"]
    conflict_sets = [{entry["group"] for entry in group} for group in conflicts]
    assert set(TORCH_GROUPS) in conflict_sets


@pytest.mark.parametrize("package", ["torch", "torchvision"])
def test_sources_cover_all_groups(package: str) -> None:
    data = _load_pyproject()
    sources = data["tool"]["uv"]["sources"][package]
    assert {s["group"] for s in sources} == set(TORCH_GROUPS)
    index_names = {idx["name"] for idx in data["tool"]["uv"]["index"]}
    assert {s["index"] for s in sources} <= index_names


def test_indexes_are_explicit_pytorch_urls() -> None:
    for idx in _load_pyproject()["tool"]["uv"]["index"]:
        assert idx["explicit"] is True
        assert idx["url"].startswith("https://download.pytorch.org/whl/")
