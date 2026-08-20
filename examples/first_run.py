"""The cold-start notice the examples print before their own imports.

A first run looks a lot like a hang: `uv` may still be installing torch, the
torch and nansense imports cost a few seconds cold, and whatever dataset the
example needs is downloaded before the first batch (CIFAR-10 is 170 MB; Make3D
is 914 MB from a slow host). So each example calls `note_first_run()` *above*
its own imports, and this module imports nothing but the standard library —
a notice that arrives after the wait is no notice at all.

The check is deliberately shallow: a dataset counts as present when the
directory it extracts to exists. Being wrong only misplaces a hint.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

DEFAULT_DATA_DIR = Path("./data")

# The directories each dataset extracts to under `--data-dir` (torchvision picks
# most of these names). Only ever used to tell a cold start from a warm one.
_CACHE_DIRS: dict[str, tuple[str, ...]] = {
    "mnist": ("MNIST",),
    "cifar10": ("cifar-10-batches-py",),
    "imagenette": ("imagenette2-160",),
    "keywords": ("mini_speech_commands",),
    "make3d": ("Train400Img", "Train400Depth", "Test134", "Gridlaserdata"),
}

_NOTICE = (
    "First run: fetching datasets into {data_dir} and warming up cold torch / "
    "nansense imports.\nThis can take a few minutes before the UI comes up; "
    "later runs start in seconds."
)


def argv_value(flag: str, default: str) -> str:
    """Read `--flag value` or `--flag=value` straight out of `sys.argv`.

    The notice runs before `argparse` — which lives past the slow imports — so
    the couple of flags it depends on are read by hand. Anything malformed falls
    through to `default`; argparse reports it properly moments later.
    """
    args = sys.argv[1:]
    for index, arg in enumerate(args):
        if arg == flag and index + 1 < len(args):
            return args[index + 1]
        if arg.startswith(f"{flag}="):
            return arg.split("=", 1)[1]
    return default


def note_first_run(*datasets: str) -> None:
    """Warn about the cold-start wait unless every dataset is already cached.

    `datasets` are `_CACHE_DIRS` keys naming what the example downloads before
    it can train. Names that aren't keys (an unrecognized `--dataset` read from
    argv, say) are ignored — argparse rejects those a moment later anyway.

    Stays quiet for `--help` (which downloads nothing) and for the non-zero
    ranks of a torchrun launch, which would otherwise repeat the notice per
    process.
    """
    if {"-h", "--help"} & set(sys.argv[1:]) or os.environ.get("RANK", "0") != "0":
        return
    data_dir = Path(argv_value("--data-dir", str(DEFAULT_DATA_DIR)))
    expected = [data_dir / name for key in datasets for name in _CACHE_DIRS.get(key, ())]
    if all(path.exists() for path in expected):
        return
    print(_NOTICE.format(data_dir=data_dir), flush=True)
