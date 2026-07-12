# Deploying the nansense playground

The playground is a publicly hosted, shared nansense session: LeNet trained
on MNIST, parked paused on its final training batch with the session locked
(`Session.lock`). Visitors can show/hide layers per tab, browse statistics,
and run experiments; stepping, time travel, input pinning/perturbation, and
the global settings are disabled. The entrypoint is
[`examples/playground/main.py`](examples/playground/main.py); the mechanics
are described in [`INTERNALS.md`](INTERNALS.md#locked-sessions-shared-demos).

Everything that belongs in this repository already lives here: the
entrypoint, the root [`Dockerfile`](Dockerfile) (root-level because Docker
Spaces accept no other location), the root `.dockerignore`, and the deploy
script [`deploy/push_space.sh`](deploy/push_space.sh). This document covers running it and the pieces
that must live *outside* the repository (the Hugging Face Space, scheduled
restarts).

## Run it locally

```bash
# One-time: train and write the epoch-checkpoint cache (~1 min/epoch on CPU).
uv run --group cpu examples/playground/main.py --prepare

# Serve: resumes at the final epoch, replays it to fill the statistics,
# then parks locked. Open http://127.0.0.1:7860/.
uv run --group cpu examples/playground/main.py

# Or the container, which bakes both steps into the image:
docker build -t nansense-playground .
docker run --rm -p 7860:7860 nansense-playground
```

The two invocations must agree on `--epochs`, `--cache-dir`, and the model
hyperparameters — serving validates the cache against the freshly built
model and refuses a mismatch.

Local `docker build` needs BuildKit (the `docker-buildx-plugin`): Docker's
deprecated legacy builder, especially under a rootless daemon, writes layers
the image's non-root `USER` cannot read — the build then dies at the first
`RUN` after the user switch with `libc.so.6 … Permission denied`. Hugging
Face builds with BuildKit on its own infrastructure, so this is a
local-build concern only.

## Hugging Face Spaces (recommended host)

A [Docker-SDK Space](https://huggingface.co/docs/hub/spaces-sdks-docker) is
the least-ops option: HF builds the Dockerfile, fronts it with TLS and
websocket support (the UI is socket.io-based — this matters), and the free
"CPU basic" tier (2 vCPU / 16 GB) is plenty for the LeNet demo.

Create a Space with the Docker SDK, then deploy — and later re-deploy —
with one command from a clean tree on any branch:

```bash
git lfs install                # one-time
deploy/push_space.sh           # --dry-run builds the snapshot without pushing
```

The script rebuilds a single-commit `space-snapshot` branch from `main` and
force-pushes it to the `space` remote (added automatically; git prompts for
your HF username and a write token). Two Hub rules shape what it does. The
Space's configuration is YAML front matter at the top of the README — the
script prepends the block below, and since the [accepted
parameters](https://huggingface.co/docs/hub/spaces-config-reference) include
no key for a custom Dockerfile location, the Dockerfile sits at the repo
root. And the Hub rejects binary files committed as plain git blobs while
inspecting the whole pushed history — hence a parentless snapshot commit
whose PNGs/GIFs are renormalized to git-LFS pointers (HF materializes LFS
files both in the Docker build, e.g. the runtime logo asset, and when
rendering the Space README).

```yaml
---
title: Nansense Playground
emoji: 👁
sdk: docker
app_port: 7860
pinned: false
license: mit
short_description: A Pytorch debugger playground on a pretrained network
---
```

The same can be automated with a GitHub Action on push (HF documents the
[sync-to-hub pattern](https://huggingface.co/docs/hub/spaces-github-actions))
by running the script from the workflow with an `HF_TOKEN` secret supplying
the push credential.

One piece lives outside the repository entirely — **a scheduled restart**
(optional). The container filesystem is ephemeral and a free Space also
sleeps after ~48 h idle, so every wake is a fresh boot — which doubles as
the reset for whatever experiment results and queue state visitors
accumulated. A busy Space never sleeps, so for a guaranteed nightly reset
add a cron workflow (GitHub Actions or anywhere) calling:

```python
from huggingface_hub import HfApi
HfApi(token=...).restart_space("YOUR_USER/nansense-playground")
```

Notes:

- Build time is dominated by the `--prepare` step (a few CPU minutes for
  5 MNIST epochs) — well inside HF's build limits; the trained checkpoints
  live only inside the image, never in git.
- No secrets, persistent storage, or GPU are needed. GPU tiers work but are
  wasted on LeNet.
- The direct app URL is `https://YOUR-USER-nansense-playground.hf.space`;
  the Space page embeds it in an iframe and doubles as the landing page.

## Any other Docker host

The image is self-contained, so a small VM with `docker compose` works the
same way. Put a websocket-capable reverse proxy with TLS in front (Caddy's
`reverse_proxy 127.0.0.1:7860` handles websockets and certificates with one
line), set `restart: always`, and schedule a nightly `docker restart` as the
shared-state reset. Give the container a memory limit (the demo stays well
under 2 GB) and skip persistent volumes — statelessness is the feature.
