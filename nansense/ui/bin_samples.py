"""Random samples from the last captured batch for a histogram bin.

The watch histograms are running aggregates — the values and inputs that
filled them are discarded every batch. Hovering a per-channel bar therefore
samples from the *last captured batch* (`Session.snapshot`), the only batch
whose full activations and input still exist: elements of the hovered
channel that fall in the hovered bin are drawn uniformly at random, and
each is rendered as an input crop around the element's ratio-mapped
location — the same receptive-field approximation the extreme-patch grids
use. The UI labels the strip with the snapshot's batch position to make
the narrower population explicit (the bar may aggregate a whole epoch).

Bin membership reuses `nansense.watch._bin_indices`, so a sampled element
is exactly one the accumulator would have counted in that bar.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from nansense.patches import crop_side
from nansense.watch import _bin_indices


@dataclass(frozen=True)
class BinSample:
    """One sampled element of a (channel, bin) cell from the last batch."""

    sample_idx: int
    value: float
    # Input crop `(Cin, ph, pw)` around the element's location (the whole
    # input image when the activation has no spatial axes); `None` when the
    # model input isn't image-like.
    image: Tensor | None


def _input_image(input_tensor: Tensor | None) -> Tensor | None:
    """The input as `(B, Cin, H, W)` with a renderable channel count."""
    if input_tensor is None or input_tensor.ndim != 4:
        return None
    if input_tensor.shape[1] not in (1, 3):
        return None
    return input_tensor


def _crop_for(
    image: Tensor,
    sample_idx: int,
    loc: tuple[int, int] | None,
    act_hw: tuple[int, int] | None,
) -> Tensor:
    """Crop `image[sample_idx]` around the ratio-mapped activation location.

    Mirrors `PatchAccumulator._window_origins`: the activation pixel's
    center maps to input space by the downsampling ratio, and the window is
    `crop_side` per axis, clamped inside the image. Without a spatial
    location (2D activations) the whole image is returned.
    """
    sample = image[sample_idx].detach().to(torch.float32)
    if loc is None or act_hw is None:
        return sample
    hin, win = int(image.shape[2]), int(image.shape[3])
    ha, wa = act_hw
    ph, pw = crop_side(ha, hin), crop_side(wa, win)
    cy = int((loc[0] + 0.5) * (hin / ha))
    cx = int((loc[1] + 0.5) * (win / wa))
    top = min(max(cy - ph // 2, 0), hin - ph)
    left = min(max(cx - pw // 2, 0), win - pw)
    return sample[:, top : top + ph, left : left + pw]


def sample_bin(
    tensor: Tensor,
    input_tensor: Tensor | None,
    *,
    channel: int,
    bin_idx: int,
    k: int = 4,
    generator: torch.Generator | None = None,
) -> list[BinSample]:
    """Up to `k` random elements of `tensor[:, channel]` that land in `bin_idx`.

    `tensor` is a snapshot activation or gradient `(B, C, ...)` on CPU;
    `input_tensor` is the snapshot's model input, used for the crops (pass
    `None` for non-image models — samples then carry no image). Returns an
    empty list when the channel is out of range or no element of the last
    batch falls in the bin.
    """
    if tensor.ndim < 2 or not 0 <= channel < tensor.shape[1]:
        return []
    per_channel = tensor[:, channel].detach().to(torch.float32)
    batch = per_channel.shape[0]
    flat = per_channel.reshape(batch, -1)
    hits = (_bin_indices(flat.reshape(-1)) == bin_idx).nonzero().reshape(-1)
    if hits.numel() == 0:
        return []
    picks = hits[torch.randperm(hits.numel(), generator=generator)[:k]]
    image = _input_image(input_tensor)
    act_hw = (
        (int(per_channel.shape[1]), int(per_channel.shape[2]))
        if per_channel.ndim == 3
        else None
    )
    per_sample = flat.shape[1]
    out: list[BinSample] = []
    for flat_idx in picks.tolist():
        sample_idx, rest = divmod(flat_idx, per_sample)
        loc = divmod(rest, act_hw[1]) if act_hw is not None else None
        crop = None
        if image is not None and sample_idx < image.shape[0]:
            crop = _crop_for(image, sample_idx, loc, act_hw)
        out.append(
            BinSample(
                sample_idx=sample_idx,
                value=float(flat[sample_idx, rest]),
                image=crop,
            )
        )
    return out
