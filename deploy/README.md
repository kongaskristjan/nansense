# Deploying to Hugging Face Spaces

The playground ([`examples/playground/main.py`](../examples/playground/main.py))
is hosted as a [Docker-SDK Space](https://huggingface.co/docs/hub/spaces-sdks-docker).
Deploy — and re-deploy — with one command, from a clean tree on any branch:

```bash
git lfs install         # one-time
deploy/push_space.sh    # --dry-run builds the snapshot without pushing
```

git prompts for your HF username and a write token. The Space then builds
the root [`Dockerfile`](../Dockerfile) (~15 min — the MNIST training is
baked into the image) and serves at
<https://kongaskristjan-nansense-playground.hf.space>.

The script exists because of two Hub rules. A Space is configured by YAML
front matter at the top of its README, with no config key for a custom
Dockerfile path — so the script prepends the block (kept inside
`push_space.sh`) and the Dockerfile lives at the repo root. And the Hub
rejects binary files committed as plain git blobs, inspecting the whole
pushed history — so the script builds a parentless `space-snapshot` commit
from `main` with every PNG/GIF renormalized to a git-LFS pointer, and
force-pushes that single commit to the Space.

The free CPU Space sleeps after ~48 h idle and wakes on the next visit with
a fresh boot, which also resets the visitors' shared state;
`huggingface_hub.HfApi().restart_space(...)` forces that anytime. To
automate deploys, run the script from a GitHub Action on pushes to `main`
(HF documents the
[sync-to-hub pattern](https://huggingface.co/docs/hub/spaces-github-actions));
an `HF_TOKEN` secret supplies the credential.
