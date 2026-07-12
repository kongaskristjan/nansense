# Deploying to Hugging Face Spaces

The playgrounds ([`examples/playground/main.py`](../examples/playground/main.py))
are hosted as [Docker-SDK Spaces](https://huggingface.co/docs/hub/spaces-sdks-docker),
one Space per demo:

- `imagenette` → <https://kongaskristjan-nansense-playground.hf.space>
- `mnist` → <https://kongaskristjan-nansense-playground-mnist.hf.space>

Deploying is two commands: train the frozen moment locally (a GPU makes the
imagenette run reasonable), then push the snapshot from a clean tree on any
branch:

```bash
git lfs install         # one-time
uv run --group cuda examples/playground/main.py --playground imagenette --prepare --device cuda
deploy/push_space.sh imagenette    # --dry-run builds the snapshot without pushing
```

git prompts for your HF username and a write token. The Space then builds
the root [`Dockerfile`](../Dockerfile) in a few minutes — no training
happens there; the moment file ships in the push as a git-LFS object and the
image serves it directly. Retraining is only needed when the moment itself
should change; a code-only redeploy re-uses the already-uploaded LFS object.

The script exists because of three Hub rules. A Space is configured by YAML
front matter at the top of its README, with no config key for a custom
Dockerfile path — so the script prepends the block (kept inside
`push_space.sh`) and the Dockerfile lives at the repo root. Spaces pass no
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
an `HF_TOKEN` secret supplies the credential.
