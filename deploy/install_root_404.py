#!/usr/bin/env python3
"""Install the root 404.html redirect on the gh-pages branch.

GitHub Pages serves the site root's 404.html for every missing path, and
mike only redirects the site root itself — so a versionless deep link like
/nansense/playground would 404 even though /nansense/latest/playground
exists. This script renders a 404.html that rewrites such paths to the
default version on the client and commits it to the gh-pages root, where
mike leaves unmanaged files alone. A path whose first segment already names
a deployed version is a genuinely missing page and keeps the 404.

Run by .github/workflows/docs.yml after every mike deploy so the baked-in
version list and default track what was just published. Requires a local
gh-pages branch that is not checked out anywhere; commits to it and pushes
to origin, skipping both when the rendered file is unchanged. Stdlib-only
on purpose: CI runs it without installing the project.
"""

from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path
from string import Template

# Must match site_url in mkdocs.yml: the project site lives under this prefix.
BASE = "/nansense/"

_TEMPLATE = Template("""\
<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <title>Page not found</title>
  <script>
    var versions = [${versions}];
    var rest = window.location.pathname.slice("${base}".length);
    if (versions.indexOf(rest.split("/")[0]) === -1) {
      window.location.replace(
        "${base}${target}/" + rest + window.location.search + window.location.hash
      );
    }
  </script>
</head>
<body>
  Page not found. Try the <a href="${base}">documentation home</a>.
</body>
</html>
""")


def render_404(versions: list[str], target: str) -> str:
    return _TEMPLATE.substitute(
        base=BASE,
        versions=", ".join(json.dumps(v) for v in versions),
        target=target,
    )


def _git(*args: str, cwd: Path | None = None) -> str:
    result = subprocess.run(
        ["git", *args], cwd=cwd, check=True, capture_output=True, text=True
    )
    return result.stdout


def deployed_versions() -> list[str]:
    """Top-level directories on gh-pages: version dirs plus aliases."""
    return _git("ls-tree", "-d", "--name-only", "gh-pages").splitlines()


def default_version() -> str:
    """Mirror the workflow's set-default logic: `latest` once a release exists."""
    latest = subprocess.run(
        ["git", "cat-file", "-e", "gh-pages:latest/index.html"], capture_output=True
    )
    return "latest" if latest.returncode == 0 else "dev"


def main() -> None:
    html = render_404(deployed_versions(), default_version())
    with tempfile.TemporaryDirectory() as tmp:
        worktree = Path(tmp) / "gh-pages"
        _git("worktree", "add", str(worktree), "gh-pages")
        try:
            (worktree / "404.html").write_text(html, encoding="utf-8")
            _git("add", "404.html", cwd=worktree)
            staged = subprocess.run(["git", "diff", "--cached", "--quiet"], cwd=worktree)
            if staged.returncode == 0:
                print("Root 404.html already up to date.")
                return
            _git("commit", "-m", "Install the root 404 redirect", cwd=worktree)
            _git("push", "origin", "gh-pages", cwd=worktree)
            print("Installed the root 404 redirect.")
        finally:
            _git("worktree", "remove", "--force", str(worktree))


if __name__ == "__main__":
    main()
