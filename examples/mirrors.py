"""Fast mirrors for the dataset archives the examples download.

torchvision hard-codes one download URL per dataset, and CIFAR-10's
(cs.toronto.edu) serves its 170 MB archive at well under 1 Mbit/s — some 40
minutes on a 100 Mbit link, far longer than the training run it precedes. This
module points CIFAR-10 at a CDN copy of the very same archive and keeps the
canonical URL as a fallback.

The other archives the examples fetch are fine as they ship and are downloaded
straight from torchvision: MNIST (11 MB, torchvision's S3 mirror), Imagenette
160px (99 MB, fast.ai's S3 bucket) and mini_speech_commands (182 MB, Google's
CDN) all arrive at tens of Mbit/s. Make3D (914 MB across four archives, from
cs.stanford.edu and cs.cornell.edu) is genuinely slow and has no public mirror
to switch to, so `depth_make3d/data.py` just retries it.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

from torchvision import datasets

# A byte-identical copy of torchvision's `cifar-10-python.tar.gz`, served from
# the Hugging Face CDN at ~85 Mbit/s where cs.toronto.edu manages ~0.6. Trusting
# a third-party copy is safe because torchvision verifies every download against
# `CIFAR10.tgz_md5`: a substituted or truncated archive fails that check.
CIFAR10_MIRROR_URL = (
    "https://huggingface.co/datasets/liangnanying/cifar-10-python/resolve/main/"
    "cifar-10-python.tar.gz"
)

# torchvision's own URL, kept as the fallback below.
CIFAR10_CANONICAL_URL = datasets.CIFAR10.url


class _MirroredCIFAR10(datasets.CIFAR10):
    """`datasets.CIFAR10`, fetched from `CIFAR10_MIRROR_URL`."""

    url = CIFAR10_MIRROR_URL


def cifar10(
    root: Path,
    *,
    train: bool,
    transform: Callable[[Any], Any] | None = None,
    download: bool = True,
) -> datasets.CIFAR10:
    """CIFAR-10 from the fast mirror, falling back to torchvision's own URL.

    The mirror is a third party, so it must not become a single point of
    failure: any download error (repo gone, CDN hiccup, md5 mismatch) is retried
    once against cs.toronto.edu, which is slow but canonical. A partial archive
    from the failed attempt is re-fetched rather than reused, because
    torchvision skips a present file only when its md5 checks out.

    Already-downloaded data short-circuits both paths, so this costs nothing
    after the first run.
    """
    try:
        return _MirroredCIFAR10(str(root), train=train, download=download, transform=transform)
    except (RuntimeError, OSError) as error:
        if not download:
            raise  # nothing to fall back to: the caller asked for cached data only
        print(f"CIFAR-10 mirror failed ({error}); retrying from {CIFAR10_CANONICAL_URL}")
        return datasets.CIFAR10(str(root), train=train, download=True, transform=transform)
