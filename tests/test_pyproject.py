"""Consistency checks for the PyTorch variant dependency groups in pyproject.toml.

PyTorch is installed through mutually exclusive dependency groups (cpu /
cuda-legacy / cuda / rocm), each pinned to its PyTorch wheel index, so
the user picks their own hardware build. nansense declares no direct torch
dependency; the one torch-bearing standard dependency is captum (for the
experiment page's attributions), which pulls torch transitively. These tests
guard against the groups, conflict declaration, sources, and index
definitions drifting apart when dependencies are edited.
"""

import sys

if sys.version_info >= (3, 11):
    import tomllib
else:  # Python < 3.11
    import tomli as tomllib

from pathlib import Path
from typing import Any

import pytest

TORCH_GROUPS = ["cpu", "cuda-legacy", "cuda", "rocm"]


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


@pytest.mark.parametrize("group", TORCH_GROUPS)
def test_torchaudio_in_every_group(group: str) -> None:
    groups = _load_pyproject()["dependency-groups"]
    assert "torchaudio" in _requirement_names(groups[group])


def test_published_metadata_declares_no_direct_torch() -> None:
    """nansense never *declares* torch (or its example-only siblings) directly.

    The hardware-specific torch / torchvision / torchaudio builds are left to
    the user (installed via the unpublished dependency groups) and lightning is
    an optional integration, so none of them may appear in the published
    metadata. captum is the one deliberate exception: a standard dependency
    (for the experiment page's attribution methods) that pulls torch
    transitively, which is why installing your own torch build first is advised.
    """
    data = _load_pyproject()
    deps = _requirement_names(data["project"]["dependencies"])
    assert "captum" in deps  # intentionally a standard dependency
    forbidden = {"torch", "torchvision", "torchaudio", "lightning"}
    assert forbidden.isdisjoint(deps)
    for extra, requirements in data["project"].get("optional-dependencies", {}).items():
        assert forbidden.isdisjoint(_requirement_names(requirements)), extra


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


def test_torchaudio_sources_cover_all_groups() -> None:
    data = _load_pyproject()
    sources = data["tool"]["uv"]["sources"]["torchaudio"]
    assert {s["group"] for s in sources} == set(TORCH_GROUPS)
    index_names = {idx["name"] for idx in data["tool"]["uv"]["index"]}
    assert {s["index"] for s in sources} <= index_names


def test_indexes_are_explicit_pytorch_urls() -> None:
    for idx in _load_pyproject()["tool"]["uv"]["index"]:
        assert idx["explicit"] is True
        assert idx["url"].startswith("https://download.pytorch.org/whl/")
