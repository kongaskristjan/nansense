"""Tests for deploy/push_space.py's pure pieces.

The git/push orchestration is exercised by hand with --dry-run; these pin
the parts that can silently drift: the per-Space front matter, the
Dockerfile stamping, the moment paths, and the prepare invocation.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]


def _load() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "push_space", _REPO_ROOT / "deploy" / "push_space.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    # Registered before exec: @dataclass resolves its module through
    # sys.modules when the script uses `from __future__ import annotations`.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


push_space = _load()


def test_spaces_cover_every_playground() -> None:
    from examples.playground.main import PLAYGROUNDS

    assert set(push_space.SPACES) == set(PLAYGROUNDS)


@pytest.mark.parametrize("name", ["imagenette", "mnist"])
def test_front_matter_is_valid_space_config(name: str) -> None:
    space = push_space.SPACES[name]
    fm = space.front_matter()
    assert fm.startswith("---\n") and fm.endswith("---\n\n")
    for key in ("title:", "emoji:", "sdk: docker", "app_port: 7860", "license:"):
        assert key in fm, key
    assert space.moment == Path(f".nansense_cache/playground/{name}/moment.pt")
    assert space.remote == f"space-{name}"


def test_stamp_dockerfile_rewrites_the_arg_default() -> None:
    text = "FROM x\nARG PLAYGROUND=imagenette\nENV PLAYGROUND=${PLAYGROUND}\n"
    stamped = push_space.stamp_dockerfile(text, "mnist")
    assert "ARG PLAYGROUND=mnist\n" in stamped
    assert "ARG PLAYGROUND=imagenette" not in stamped


def test_stamp_dockerfile_stamps_the_real_dockerfile() -> None:
    text = (_REPO_ROOT / "Dockerfile").read_text()
    stamped = push_space.stamp_dockerfile(text, "mnist")
    assert "ARG PLAYGROUND=mnist" in stamped


@pytest.mark.parametrize("bad", ["FROM x\n", "ARG PLAYGROUND=a\nARG PLAYGROUND=b\n"])
def test_stamp_dockerfile_requires_exactly_one_arg_line(bad: str) -> None:
    with pytest.raises(SystemExit, match="exactly one"):
        push_space.stamp_dockerfile(bad, "mnist")


def test_prepare_command_matches_the_documented_invocation() -> None:
    cmd = push_space.prepare_command(push_space.SPACES["imagenette"])
    assert cmd[:4] == ["uv", "run", "--group", "cuda"]
    assert "examples/playground/main.py" in cmd
    assert "--prepare" in cmd
    assert cmd[cmd.index("--playground") + 1] == "imagenette"
