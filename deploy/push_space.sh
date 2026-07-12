#!/usr/bin/env bash
# Deploy the playground to its Hugging Face Space.
#
#     deploy/push_space.sh [--dry-run]
#
# Rebuilds the Space's single-commit `space-snapshot` branch from `main` —
# the Space front matter prepended to the README, every PNG/GIF renormalized
# to a git-LFS pointer (the Hub rejects plain-git binaries and inspects the
# whole pushed history, hence the orphan commit) — and force-pushes it to
# the `space` remote. See deploy/README.md for the background.
#
# Requires git-lfs (`git lfs install` once) and a clean working tree; run it
# from any branch, it returns you there. `--dry-run` does everything except
# the push.

set -euo pipefail

SPACE_URL="https://huggingface.co/spaces/kongaskristjan/nansense-playground"
FRONT_MATTER='---
title: Nansense Playground
emoji: 👁
sdk: docker
app_port: 7860
pinned: false
license: mit
short_description: A Pytorch debugger playground on a pretrained network
---

'

dry_run=false
[[ "${1:-}" == "--dry-run" ]] && dry_run=true

cd "$(git rev-parse --show-toplevel)"
if ! git config --get filter.lfs.clean >/dev/null; then
    echo "git-lfs filters are not configured — install git-lfs and run: git lfs install" >&2
    exit 1
fi
if [[ -n "$(git status --porcelain)" ]]; then
    echo "working tree not clean — commit or stash first" >&2
    exit 1
fi
git remote get-url space >/dev/null 2>&1 || git remote add space "$SPACE_URL"

orig_ref=$(git symbolic-ref --quiet --short HEAD || git rev-parse HEAD)

# A fresh orphan branch holding main's tree: parentless, so the push carries
# no history for the Hub's binary-file check to trip over.
git branch -D space-snapshot 2>/dev/null || true
git checkout -q --orphan space-snapshot main

printf '%s' "$FRONT_MATTER" | cat - README.md > README.md.tmp
mv README.md.tmp README.md
# `lfs track` may fail to touch container-owned files (chtimes); harmless —
# the renormalize below re-stages everything regardless of timestamps.
git lfs track "*.png" "*.gif" >/dev/null 2>&1 || true
grep -q "filter=lfs" .gitattributes
git add README.md .gitattributes
git add --renormalize .
git commit -q -m "nansense playground Space snapshot"

# Staged images must be LFS pointers now, not raw bytes.
sample=$(git ls-files "*.png" | head -1)
if ! git show ":$sample" | head -1 | grep -q "git-lfs"; then
    echo "renormalize failed: $sample is not an LFS pointer" >&2
    exit 1
fi

if $dry_run; then
    echo "dry run — skipping: git push --force space space-snapshot:main"
else
    git push --force space space-snapshot:main
fi
git checkout -qf "$orig_ref"
echo "done (deployed snapshot: $(git rev-parse --short space-snapshot))"
