# Deploying to Hugging Face Spaces

The playgrounds ([`examples/playground/main.py`](../examples/playground/main.py))
are hosted as [Docker-SDK Spaces](https://huggingface.co/docs/hub/spaces-sdks-docker),
one Space per demo:

- `imagenette` → <https://kongaskristjan-nansense-playground.hf.space>
- `mnist` → <https://kongaskristjan-nansense-playground-mnist.hf.space>

Deploy from a clean tree on any branch — one playground or both:

```bash
git lfs install                       # one-time
deploy/push_space.py imagenette       # or: mnist, or: all
```

`--prepare-cache` first (re)trains the playground's frozen moment on the
local GPU (minutes for mnist, hours for imagenette); without it the script
expects the moment to exist and prints the training command if not.
`--dry-run` builds and verifies the snapshot(s) without pushing.

Pushes go over SSH (`ssh://git@hf.co/spaces/...`), so your SSH key must be
registered with your Hugging Face account
(<https://huggingface.co/settings/keys>). The Space then builds
the root [`Dockerfile`](../Dockerfile) in a few minutes — no training
happens there; the moment file ships in the push as a git-LFS object and the
image serves it directly. Retraining is only needed when the moment itself
should change; a code-only redeploy re-uses the already-uploaded LFS object.

The script exists because of three Hub rules. A Space is configured by YAML
front matter at the top of its README, with no config key for a custom
Dockerfile path — so the script prepends the block (kept inside
`push_space.py`) and the Dockerfile lives at the repo root. Spaces pass no
Docker build args — so the script stamps the Dockerfile's `PLAYGROUND`
default per Space. And the Hub rejects binary files committed as plain git
blobs, inspecting the whole pushed history — so the script builds a
parentless `space-snapshot-<playground>` commit from `main` with every
PNG/GIF/PT renormalized to a git-LFS pointer (the gitignored moment file
force-added), and force-pushes that single commit to the Space.

The free CPU Space sleeps after ~48 h idle and wakes on the next visit with
a fresh boot, which also resets the visitors' shared state;
`huggingface_hub.HfApi().restart_space(...)` forces that anytime. To
automate deploys, run the script from a GitHub Action on pushes to `main`
(HF documents the
[sync-to-hub pattern](https://huggingface.co/docs/hub/spaces-github-actions));
an SSH deploy key stored as a secret supplies the credential (or point the
script's URLs back at `https://huggingface.co/...` and use an `HF_TOKEN`).
