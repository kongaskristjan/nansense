"""Make3D monocular-depth data loading.

Make3D pairs an outdoor RGB photo with a laser-scanned depth map. The four
canonical archives (~0.9 GB total) are fetched on first use with torchvision's
`download_and_extract_archive`; the host can be slow and occasionally flaky, so
each archive is downloaded independently and skipped when already extracted.

Each example is an image `img-<id>.jpg` paired with a MATLAB depth file
`depth_sph_corr-<id>.mat` (key `Position3DGrid`, shape ~ (55, 305, 4); depth in
metres is channel 3). The image is resized to 192x256 and ImageNet-normalised;
the depth is resized to a 1/4 grid (48x64), clipped to [0, 70] m, and invalid
(non-positive / non-finite) pixels are written as `0` — the sentinel the loss
and metric recompute their validity mask from.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import scipy.io
import torch
from PIL import Image, ImageFile

# A few Make3D JPEGs are slightly truncated (a known quirk of the dataset); PIL
# refuses them by default. Load what's there — the depth target comes from the
# .mat file, so a few missing pixels at an image's edge are harmless.
ImageFile.LOAD_TRUNCATED_IMAGES = True  # ty: ignore[invalid-assignment]  # PIL stub types it Literal[False]
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.datasets.utils import download_and_extract_archive

# ImageNet statistics — the encoder is pretrained on ImageNet, so inputs are
# normalised to match (and passed to nansense as input_mean / input_std).
IMAGENET_MEAN: tuple[float, float, float] = (0.485, 0.456, 0.406)
IMAGENET_STD: tuple[float, float, float] = (0.229, 0.224, 0.225)

# Each archive: (url, the directory name it extracts to under data_dir).
_ARCHIVES: dict[str, tuple[str, str]] = {
    "train_images": (
        "https://cs.stanford.edu/group/reconstruction3d/Train400Img.tar.gz",
        "Train400Img",
    ),
    "train_depths": (
        "https://cs.stanford.edu/group/reconstruction3d/Train400Depth.tgz",
        "Train400Depth",
    ),
    "test_images": (
        "https://www.cs.cornell.edu/~asaxena/learningdepth/Test134.tar.gz",
        "Test134",
    ),
    "test_depths": (
        "https://www.cs.cornell.edu/~asaxena/learningdepth/Test134Depth.tar.gz",
        "Gridlaserdata",
    ),
}

_DEPTH_KEY: str = "Position3DGrid"
_DEPTH_PREFIX: str = "depth_sph_corr-"
_IMAGE_PREFIX: str = "img-"


@dataclass(frozen=True)
class DatasetConfig:
    """Static description of the Make3D depth task."""

    name: str = "make3d"
    image_size: tuple[int, int] = (192, 256)  # (H, W) fed to the encoder
    depth_size: tuple[int, int] = (48, 64)  # (h, w) of the predicted grid (1/4)
    max_depth: float = 70.0  # metres; far depths beyond this are clipped
    mean: tuple[float, float, float] = IMAGENET_MEAN
    std: tuple[float, float, float] = field(default=IMAGENET_STD)


def _archive_dirs(split: str) -> tuple[tuple[str, str], tuple[str, str]]:
    """The (image, depth) archive specs for a split ("train" or "test")."""
    if split == "train":
        return _ARCHIVES["train_images"], _ARCHIVES["train_depths"]
    if split == "test":
        return _ARCHIVES["test_images"], _ARCHIVES["test_depths"]
    raise ValueError(f"split must be 'train' or 'test', got {split!r}")


_DOWNLOAD_ATTEMPTS = 4


def _maybe_download(url: str, extract_dir: str, data_dir: Path, expected_glob: str) -> Path:
    """Fetch + extract `url` into `data_dir/extract_dir` unless already present.

    The archive is extracted into its own `extract_dir` (via `extract_root`) so
    that the train and test image sets — whose files share the `img-*.jpg`
    naming — never land in the same directory. The Make3D archives are
    inconsistent about internal layout (the image tarball is flat, the depth
    one nests a folder), so callers locate files with `rglob`, not a fixed
    depth. No published checksum + a slow host means "a matching file already
    exists under the target" is the skip signal.

    The Make3D host is slow and occasionally truncates a transfer. A partial
    archive would otherwise be silently re-used (torchvision skips re-download
    when the file exists and there is no checksum) and fail extraction with a
    cryptic `EOFError` forever, so each attempt first removes any stale archive
    and the whole thing is retried a few times before giving up.
    """
    target = data_dir / extract_dir
    if target.is_dir() and next(target.rglob(expected_glob), None) is not None:
        return target
    target.mkdir(parents=True, exist_ok=True)
    archive = data_dir / Path(url).name  # where torchvision saves the download
    last_error: Exception | None = None
    for _attempt in range(_DOWNLOAD_ATTEMPTS):
        archive.unlink(missing_ok=True)  # never extract a stale/partial download
        try:
            download_and_extract_archive(
                url, download_root=str(data_dir), extract_root=str(target),
                remove_finished=True,
            )
        except Exception as error:  # truncated transfer, corrupt gzip/tar, network
            last_error = error
            continue
        if next(target.rglob(expected_glob), None) is not None:
            return target
    raise RuntimeError(
        f"failed to download and extract {url} after {_DOWNLOAD_ATTEMPTS} attempts "
        f"(the Make3D host is slow and sometimes truncates transfers); last error: "
        f"{last_error}. Re-run to retry."
    )


def _build_index(image_root: Path, depth_root: Path) -> list[tuple[Path, Path]]:
    """Pair every depth `.mat` with its image by id, searching recursively.

    The id is the filename with its prefix (`depth_sph_corr-` / `img-`) and
    suffix stripped; an image `img-<id>.jpg` matches depth
    `depth_sph_corr-<id>.mat`. `rglob` tolerates whichever nesting each archive
    happened to extract into.
    """
    images = {
        p.stem[len(_IMAGE_PREFIX) :]: p
        for p in image_root.rglob(f"{_IMAGE_PREFIX}*.jpg")
        if p.name.startswith(_IMAGE_PREFIX)  # skip __MACOSX '._img-*' resource forks
    }
    pairs: list[tuple[Path, Path]] = []
    for depth_path in sorted(depth_root.rglob(f"{_DEPTH_PREFIX}*.mat")):
        if not depth_path.name.startswith(_DEPTH_PREFIX):
            continue
        image_path = images.get(depth_path.stem[len(_DEPTH_PREFIX) :])
        if image_path is not None:
            pairs.append((image_path, depth_path))
    return pairs


def _load_depth(depth_path: Path, size: tuple[int, int], max_depth: float) -> Tensor:
    """Load a Make3D depth map as a `[1, h, w]` tensor in metres.

    Resizes (area-style via bilinear) to `size`, clips to `[0, max_depth]`, and
    writes `0` for any non-finite or non-positive pixel — the invalid sentinel.
    """
    grid = scipy.io.loadmat(str(depth_path))[_DEPTH_KEY]
    depth = np.asarray(grid[:, :, 3], dtype=np.float32)
    depth = np.where(np.isfinite(depth), depth, 0.0)

    depth_t = torch.from_numpy(depth)[None, None]  # [1, 1, H0, W0]
    depth_t = torch.nn.functional.interpolate(
        depth_t, size=size, mode="bilinear", align_corners=False
    )[0]  # [1, h, w]
    depth_t = depth_t.clamp(0.0, max_depth)
    depth_t[~torch.isfinite(depth_t) | (depth_t <= 0.0)] = 0.0
    return depth_t


class Make3DDataset(Dataset):
    """Map-style Make3D dataset returning `(image[3,H,W], depth[1,h,w])`.

    `image` is float, ImageNet-normalised; `depth` is metres with `0` for
    invalid pixels (the loss / metric mask is `depth > 0`).
    """

    def __init__(self, config: DatasetConfig, data_dir: Path, train: bool, download: bool) -> None:
        split = "train" if train else "test"
        (img_url, img_dir), (depth_url, depth_dir) = _archive_dirs(split)
        image_glob = f"{_IMAGE_PREFIX}*.jpg"
        depth_glob = f"{_DEPTH_PREFIX}*.mat"
        if download:
            image_root = _maybe_download(img_url, img_dir, data_dir, image_glob)
            depth_root = _maybe_download(depth_url, depth_dir, data_dir, depth_glob)
        else:
            image_root = data_dir / img_dir
            depth_root = data_dir / depth_dir

        self.config = config
        self.pairs = _build_index(image_root, depth_root)
        if not self.pairs:
            raise RuntimeError(
                f"no image/depth pairs found under {data_dir} for split {split!r}; "
                "pass download=True (or run main.py, which downloads by default)"
            )
        self.image_transform = transforms.Compose(
            [
                transforms.Resize(config.image_size),
                transforms.ToTensor(),
                transforms.Normalize(config.mean, config.std),
            ]
        )

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        image_path, depth_path = self.pairs[index]
        with Image.open(image_path) as raw:
            image = self.image_transform(raw.convert("RGB"))
        depth = _load_depth(depth_path, self.config.depth_size, self.config.max_depth)
        return image, depth


def build_dataloaders(
    config: DatasetConfig,
    data_dir: Path,
    batch_size: int = 16,
    num_workers: int = 2,
    download: bool = True,
) -> tuple[DataLoader, DataLoader]:
    train_set = Make3DDataset(config, data_dir, train=True, download=download)
    test_set = Make3DDataset(config, data_dir, train=False, download=download)

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    return train_loader, test_loader
