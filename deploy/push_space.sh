#!/usr/bin/env bash
# Deploy a playground to its Hugging Face Space.
#
#     deploy/push_space.sh <imagenette|mnist> [--dry-run]
#
# Rebuilds the Space's single-commit `space-snapshot-<playground>` branch
# from `main` — the Space front matter prepended to the README, the
# Dockerfile's PLAYGROUND default stamped, the playground's locally trained
# moment file force-added, and every PNG/GIF/PT renormalized to a git-LFS
# pointer (the Hub rejects plain-git binaries and inspects the whole pushed
# history, hence the orphan commit) — and force-pushes it to the Space
# remote. See deploy/README.md for the background.
#
# Requires git-lfs (`git lfs install` once), a clean working tree, and the
# playground's moment file (train it first — see deploy/README.md); run it
# from any branch, it returns you there. `--dry-run` does everything except
# the push.

set -euo pipefail

usage() {
    echo "usage: deploy/push_space.sh <imagenette|mnist> [--dry-run]" >&2
    exit 1
}

playground="${1:-}"
dry_run=false
[[ "${2:-}" == "--dry-run" ]] && dry_run=true

case "$playground" in
imagenette)
    space_url="https://huggingface.co/spaces/kongaskristjan/nansense-playground"
    title="Nansense Playground"
    emoji="👁"
    short_description="A PyTorch debugger playground on an Imagenette ResNet"
    ;;
mnist)
    space_url="https://huggingface.co/spaces/kongaskristjan/nansense-playground-mnist"
    title="Nansense Playground (MNIST)"
    emoji="🔢"
    short_description="A PyTorch debugger playground on an MNIST LeNet"
    ;;
*)
    usage
    ;;
esac

front_matter="---
title: $title
emoji: $emoji
sdk: docker
app_port: 7860
pinned: false
license: mit
short_description: $short_description
---

"

cd "$(git rev-parse --show-toplevel)"
if ! git config --get filter.lfs.clean >/dev/null; then
    echo "git-lfs filters are not configured — install git-lfs and run: git lfs install" >&2
    exit 1
fi
# Untracked files can't leak into the snapshot (only tracked files are
# renormalized and committed), so only tracked modifications block.
if [[ -n "$(git status --porcelain -uno)" ]]; then
    echo "working tree not clean — commit or stash first" >&2
    exit 1
fi
moment=".nansense_cache/playground/$playground/moment.pt"
if [[ ! -f "$moment" ]]; then
    echo "missing $moment — train it first:" >&2
    echo "  uv run --group cuda examples/playground/main.py --playground $playground --prepare --device cuda" >&2
    exit 1
fi

remote="space-$playground"
git remote get-url "$remote" >/dev/null 2>&1 || git remote add "$remote" "$space_url"

orig_ref=$(git symbolic-ref --quiet --short HEAD || git rev-parse HEAD)

# Whatever happens past this point — success or a failed check — return to
# the original branch. The snapshot tracks the (gitignored) moment while
# $orig_ref doesn't, so git would delete the trained file on the way back;
# park it aside across the checkout.
finish() {
    status=$?
    [[ -f "$moment" ]] && mv "$moment" "$moment.keep"
    git checkout -qf "$orig_ref" 2>/dev/null || true
    [[ -f "$moment.keep" ]] && mv "$moment.keep" "$moment"
    exit "$status"
}
trap finish EXIT

# A fresh orphan branch holding main's tree: parentless, so the push carries
# no history for the Hub's binary-file check to trip over.
snapshot="space-snapshot-$playground"
git branch -D "$snapshot" 2>/dev/null || true
git checkout -q --orphan "$snapshot" main

printf '%s' "$front_matter" | cat - README.md > README.md.tmp
mv README.md.tmp README.md
# The root Dockerfile serves whichever playground its PLAYGROUND arg
# defaults to; Spaces pass no build args, so the snapshot stamps it.
sed -i "s/^ARG PLAYGROUND=.*/ARG PLAYGROUND=$playground/" Dockerfile
grep -q "^ARG PLAYGROUND=$playground$" Dockerfile
# `lfs track` may fail to touch container-owned files (chtimes); harmless —
# the renormalize below re-stages everything regardless of timestamps.
git lfs track "*.png" "*.gif" "*.pt" >/dev/null 2>&1 || true
grep -q "filter=lfs" .gitattributes
git add README.md Dockerfile .gitattributes
git add -f "$moment"
git add --renormalize .
git commit -q -m "nansense $playground playground Space snapshot"

# Staged binaries must be LFS pointers now, not raw bytes.
sample=$(git ls-files "*.png" | head -1)
for staged in "$sample" "$moment"; do
    if ! git show ":$staged" | head -1 | grep -q "git-lfs"; then
        echo "renormalize failed: $staged is not an LFS pointer" >&2
        exit 1
    fi
done

if $dry_run; then
    echo "dry run — skipping: git push --force $remote $snapshot:main"
else
    git push --force "$remote" "$snapshot":main
fi
echo "done (deployed snapshot: $(git rev-parse --short "$snapshot"))"
