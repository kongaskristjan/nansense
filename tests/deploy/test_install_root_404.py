"""Tests for deploy/install_root_404.py.

Rendering is pinned directly; the git orchestration runs for real against a
throwaway clone with a bare origin (a few tiny commits, no network).
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path
from types import ModuleType

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SCRIPT = _REPO_ROOT / "deploy" / "install_root_404.py"


def _load() -> ModuleType:
    spec = importlib.util.spec_from_file_location("install_root_404", _SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


install_root_404 = _load()


def test_render_bakes_versions_base_and_target() -> None:
    html = install_root_404.render_404(["0.3", "dev", "latest"], "latest")
    assert 'var versions = ["0.3", "dev", "latest"];' in html
    assert '"/nansense/".length' in html  # strips the site prefix
    assert '"/nansense/latest/" + rest' in html  # redirects into the default
    assert '<a href="/nansense/">' in html


def _git(cwd: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args], cwd=cwd, check=True, capture_output=True, text=True
    )
    return result.stdout


def _make_site(tmp_path: Path) -> tuple[Path, Path]:
    """A clone + bare origin whose gh-pages holds a mike-style `dev` deploy."""
    origin = tmp_path / "origin.git"
    origin.mkdir()
    _git(origin, "init", "--bare", "--initial-branch=main", ".")
    repo = tmp_path / "repo"
    _git(tmp_path, "clone", str(origin), "repo")
    _git(repo, "config", "user.name", "test")
    _git(repo, "config", "user.email", "test@example.com")
    _git(repo, "checkout", "--orphan", "gh-pages")
    _deploy_version(repo, "dev")
    (repo / "index.html").write_text("root redirect", encoding="utf-8")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-m", "deploy dev")
    _git(repo, "push", "origin", "gh-pages")
    # The script refuses a branch that is checked out somewhere, like mike.
    _git(repo, "checkout", "--detach")
    return repo, origin


def _deploy_version(repo: Path, version: str) -> None:
    (repo / version).mkdir()
    (repo / version / "index.html").write_text(f"{version} home", encoding="utf-8")


def _run_script(repo: Path) -> None:
    subprocess.run(
        [sys.executable, str(_SCRIPT)], cwd=repo, check=True, capture_output=True
    )


def test_installs_dev_redirect_and_skips_unchanged_reruns(tmp_path: Path) -> None:
    repo, origin = _make_site(tmp_path)
    _run_script(repo)
    html = _git(origin, "show", "gh-pages:404.html")
    assert 'var versions = ["dev"];' in html
    assert '"/nansense/dev/" + rest' in html
    before = _git(origin, "rev-parse", "gh-pages")
    _run_script(repo)
    assert _git(origin, "rev-parse", "gh-pages") == before


def test_targets_latest_once_a_release_exists(tmp_path: Path) -> None:
    repo, origin = _make_site(tmp_path)
    _git(repo, "checkout", "gh-pages")
    _deploy_version(repo, "0.3")
    _deploy_version(repo, "latest")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-m", "deploy 0.3 + latest")
    _git(repo, "checkout", "--detach")
    _run_script(repo)
    html = _git(origin, "show", "gh-pages:404.html")
    assert 'var versions = ["0.3", "dev", "latest"];' in html
    assert '"/nansense/latest/" + rest' in html
