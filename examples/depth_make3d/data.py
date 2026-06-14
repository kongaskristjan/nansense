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
from PIL import Image
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


def _maybe_download(url: str, extract_dir: str, data_dir: Path) -> Path:
    """Fetch + extract `url` under `data_dir/extract_dir` unless already present.

    Make3D archives carry no published checksum and the host is occasionally
    slow, so existence of the target directory (with files) is the skip signal.
    """
    target = data_dir / extract_dir
    if target.is_dir() and any(target.iterdir()):
        return target
    data_dir.mkdir(parents=True, exist_ok=True)
    download_and_extract_archive(url, download_root=str(data_dir), remove_finished=True)
    return target


def _build_index(image_dir: Path, depth_dir: Path) -> list[tuple[Path, Path]]:
    """Pair every depth `.mat` whose image `img-<id>.jpg` exists on disk.

    The id is the depth filename with the `depth_sph_corr-` prefix and `.mat`
    suffix stripped; the matching image is `img-<id>.jpg` (images live flat in
    their archive directory).
    """
    pairs: list[tuple[Path, Path]] = []
    for depth_path in sorted(depth_dir.glob(f"{_DEPTH_PREFIX}*.mat")):
        example_id = depth_path.stem[len(_DEPTH_PREFIX) :]
        image_path = image_dir / f"{_IMAGE_PREFIX}{example_id}.jpg"
        if image_path.is_file():
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
        if download:
            image_root = _maybe_download(img_url, img_dir, data_dir)
            depth_root = _maybe_download(depth_url, depth_dir, data_dir)
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
