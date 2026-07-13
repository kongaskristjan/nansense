#!/usr/bin/env python3
"""Deploy the playgrounds to their Hugging Face Spaces.

    deploy/push_space.py <imagenette|mnist|all> [--prepare-cache] [--dry-run]

For each requested playground the script rebuilds the Space's single-commit
`space-snapshot-<playground>` branch from `main` — the Space front matter
prepended to the README, the Dockerfile's PLAYGROUND default stamped, the
playground's locally trained moment file force-added, and every PNG/GIF/PT
renormalized to a git-LFS pointer (the Hub rejects plain-git binaries and
inspects the whole pushed history, hence the orphan commit) — and
force-pushes it to the Space remote. See deploy/README.md for the
background.

`--prepare-cache` first (re)trains the playground's frozen moment on the
local GPU (`examples/playground/main.py --prepare` — expect minutes for
mnist, hours for imagenette); without it, a missing moment is an error.
`--dry-run` does everything except the push, including any requested
prepare.

Requires git-lfs (`git lfs install` once) and a clean working tree; run it
from any branch, it returns you there. Stdlib-only on purpose: it must run
before any venv exists.
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Space:
    """One playground's Hugging Face Space and its README front matter."""

    playground: str
    url: str
    title: str
    emoji: str
    short_description: str

    @property
    def remote(self) -> str:
        return f"space-{self.playground}"

    @property
    def snapshot_branch(self) -> str:
        return f"space-snapshot-{self.playground}"

    @property
    def moment(self) -> Path:
        # Mirrors examples/playground/main.py's default_moment_path.
        return Path(".nansense_cache/playground") / self.playground / "moment.pt"

    def front_matter(self) -> str:
        return (
            "---\n"
            f"title: {self.title}\n"
            f"emoji: {self.emoji}\n"
            "sdk: docker\n"
            "app_port: 7860\n"
            "pinned: false\n"
            "license: mit\n"
            f"short_description: {self.short_description}\n"
            "---\n\n"
        )


SPACES: dict[str, Space] = {
    "imagenette": Space(
        playground="imagenette",
        url="ssh://git@hf.co/spaces/kongaskristjan/nansense-playground",
        title="Nansense Playground",
        emoji="👁",
        short_description="A PyTorch debugger playground on an Imagenette ResNet",
    ),
    "mnist": Space(
        playground="mnist",
        url="ssh://git@hf.co/spaces/kongaskristjan/nansense-playground-mnist",
        title="Nansense Playground (MNIST)",
        emoji="🔢",
        short_description="A PyTorch debugger playground on an MNIST LeNet",
    ),
}


def git(*args: str, check: bool = True, capture: bool = True) -> str:
    """Run a git command at the CWD, returning stripped stdout."""
    result = subprocess.run(
        ["git", *args], check=check, text=True, capture_output=capture
    )
    return (result.stdout or "").strip() if capture else ""


def stamp_dockerfile(text: str, playground: str) -> str:
    """Rewrite the root Dockerfile's PLAYGROUND build-arg default.

    Spaces pass no Docker build args, so each snapshot ships the Dockerfile
    with its own playground as the default.
    """
    stamped, n = re.subn(
        r"^ARG PLAYGROUND=.*$",
        f"ARG PLAYGROUND={playground}",
        text,
        flags=re.MULTILINE,
    )
    if n != 1:
        raise SystemExit(
            f"expected exactly one 'ARG PLAYGROUND=' line in Dockerfile, found {n}"
        )
    return stamped


def prepare_command(space: Space) -> list[str]:
    """The moment-training invocation `--prepare-cache` runs."""
    return [
        "uv", "run", "--group", "cuda",
        "examples/playground/main.py",
        "--playground", space.playground,
        "--prepare", "--device", "cuda",
    ]


def assert_lfs_pointer(staged_path: str) -> None:
    """Fail unless the staged file is an LFS pointer, not raw bytes."""
    content = git("show", f":{staged_path}")
    if "git-lfs" not in content.splitlines()[0]:
        raise SystemExit(f"renormalize failed: {staged_path} is not an LFS pointer")


def prepare_cache(space: Space) -> None:
    print(f"preparing the {space.playground} moment (this trains the model)...")
    subprocess.run(prepare_command(space), check=True, capture_output=False)


def push_space(space: Space, *, dry_run: bool) -> None:
    """Build and push one Space's snapshot; always returns to the start ref.

    The snapshot tracks the (gitignored) moment while the original ref does
    not, so git would delete the trained file on the way back — it is parked
    aside across the final checkout, on every path out.
    """
    if not space.moment.is_file():
        raise SystemExit(
            f"missing {space.moment} — train it first:\n"
            f"  {' '.join(prepare_command(space))}\n"
            "(or pass --prepare-cache)"
        )
    # Add the remote, or repoint one left behind by an older URL scheme.
    current_url = git("remote", "get-url", space.remote, check=False)
    if not current_url:
        git("remote", "add", space.remote, space.url)
    elif current_url != space.url:
        git("remote", "set-url", space.remote, space.url)

    orig_ref = git("symbolic-ref", "--quiet", "--short", "HEAD", check=False) or git(
        "rev-parse", "HEAD"
    )
    # A fresh orphan branch holding main's tree: parentless, so the push
    # carries no history for the Hub's binary-file check to trip over.
    git("branch", "-D", space.snapshot_branch, check=False)
    git("checkout", "-q", "--orphan", space.snapshot_branch, "main")
    try:
        readme = Path("README.md")
        readme.write_text(space.front_matter() + readme.read_text())
        dockerfile = Path("Dockerfile")
        dockerfile.write_text(
            stamp_dockerfile(dockerfile.read_text(), space.playground)
        )
        # `lfs track` may fail to touch container-owned files (chtimes);
        # harmless — the renormalize below re-stages everything regardless.
        git("lfs", "track", "*.png", "*.gif", "*.pt", check=False)
        if "filter=lfs" not in Path(".gitattributes").read_text():
            raise SystemExit(".gitattributes carries no LFS filter rules")
        git("add", "README.md", "Dockerfile", ".gitattributes")
        git("add", "-f", str(space.moment))
        git("add", "--renormalize", ".")
        git(
            "commit", "-q", "-m",
            f"nansense {space.playground} playground Space snapshot",
        )

        sample_png = git("ls-files", "*.png").splitlines()[0]
        assert_lfs_pointer(sample_png)
        assert_lfs_pointer(str(space.moment))

        push = ["push", "--force", space.remote, f"{space.snapshot_branch}:main"]
        if dry_run:
            print(f"dry run — skipping: git {' '.join(push)}")
        else:
            git(*push, capture=False)
        print(
            f"done (deployed snapshot: {git('rev-parse', '--short', space.snapshot_branch)})"
        )
    finally:
        parked = space.moment.with_name(space.moment.name + ".keep")
        if space.moment.is_file():
            space.moment.rename(parked)
        git("checkout", "-qf", orig_ref, check=False)
        if parked.is_file():
            parked.rename(space.moment)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Deploy a playground (or all of them) to Hugging Face Spaces."
    )
    parser.add_argument("playground", choices=[*sorted(SPACES), "all"])
    parser.add_argument(
        "--prepare-cache",
        action="store_true",
        help="(Re)train the frozen moment on the local GPU before pushing.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Build and verify the snapshot(s) without pushing.",
    )
    args = parser.parse_args()

    os.chdir(git("rev-parse", "--show-toplevel"))
    if not git("config", "--get", "filter.lfs.clean", check=False):
        raise SystemExit(
            "git-lfs filters are not configured — install git-lfs and run: "
            "git lfs install"
        )
    if git("status", "--porcelain", "-uno"):
        raise SystemExit("working tree not clean — commit or stash first")

    spaces = list(SPACES.values()) if args.playground == "all" else [
        SPACES[args.playground]
    ]
    for space in spaces:
        if args.prepare_cache:
            prepare_cache(space)
        push_space(space, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
