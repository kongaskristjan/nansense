"""Consistency checks for the PyTorch variant dependency groups in pyproject.toml.

PyTorch is installed through mutually exclusive dependency groups (cpu /
cu126 / cu130 / cu132 / rocm7-2), each pinned to its PyTorch wheel index.
Groups are never published, which keeps the PyPI package torch-free. These
tests guard against the groups, conflict declaration, sources, and index
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

TORCH_GROUPS = ["cpu", "cu126", "cu130", "cu132", "rocm7-2"]
# torchaudio (audio example only) ships no linux-x86_64 wheel on the CUDA 13.2
# index, so it lives in every torch group except cu132.
TORCHAUDIO_GROUPS = ["cpu", "cu126", "cu130", "rocm7-2"]


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
def test_torchaudio_in_every_group_except_cu132(group: str) -> None:
    groups = _load_pyproject()["dependency-groups"]
    has_torchaudio = "torchaudio" in _requirement_names(groups[group])
    assert has_torchaudio == (group in TORCHAUDIO_GROUPS)


def test_published_metadata_is_torch_free() -> None:
    """`pip install nansense` (with or without extras) must never pull torch.

    torch, torchvision, captum, and lightning (the latter two depend on
    torch) are all delegated to the user: they may only appear in the
    unpublished dependency groups, never in the published metadata.
    """
    data = _load_pyproject()
    forbidden = {"torch", "torchvision", "torchaudio", "captum", "lightning"}
    assert forbidden.isdisjoint(_requirement_names(data["project"]["dependencies"]))
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


def test_torchaudio_sources_cover_audio_groups() -> None:
    data = _load_pyproject()
    sources = data["tool"]["uv"]["sources"]["torchaudio"]
    assert {s["group"] for s in sources} == set(TORCHAUDIO_GROUPS)
    index_names = {idx["name"] for idx in data["tool"]["uv"]["index"]}
    assert {s["index"] for s in sources} <= index_names


def test_indexes_are_explicit_pytorch_urls() -> None:
    for idx in _load_pyproject()["tool"]["uv"]["index"]:
        assert idx["explicit"] is True
        assert idx["url"].startswith("https://download.pytorch.org/whl/")
