"""Per-channel extreme-activation input patches for watched layers.

For each watched layer we keep, per `(phase, epoch)` and per activation
channel, the `N_PER_CHANNEL` input samples that produced the most extreme
activations, under four rankings:

- ``max_pixel`` / ``min_pixel`` — the channel's single largest / smallest
  activation value anywhere in its spatial map. The stored patch is a crop
  of the input image around the receptive-field location of that pixel
  (ratio-mapped, not an exact receptive field).
- ``max_average`` / ``min_average`` — the channel's spatial mean. There is
  no single location to crop around, so the stored patch is the whole
  input image.

Each stored entry also keeps the channel's full activation map for that
sample, so the UI can blend an activation heatmap over the patch.

Everything runs on the training device with no GPU→CPU syncs and no
data-dependent branching: per-batch work is a handful of reductions over
the activation, a per-channel ``topk`` over the batch axis, one vectorised
gather for the candidate patches, and a ``cat``+``topk`` merge into the
running buffers. Within one epoch a sample appears in exactly one batch,
so per-channel ranking over the batch axis can never store duplicate
samples in one channel's row.

Like `TensorAccumulator`, the training thread reassigns whole buffer
tensors while `snapshot()` (UI thread, every ~2s) copies them to CPU; a
snapshot may pair values from one merge with patches from the next, which
is harmless for display purposes.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import torch
from torch import Tensor

N_PER_CHANNEL: int = 5
# Crop side ≈ PATCH_FACTOR × the activation→input downsampling ratio,
# floored at MIN_PATCH input pixels and capped at the image side.
PATCH_FACTOR: int = 4
MIN_PATCH: int = 10

PatchType = Literal["max_pixel", "min_pixel", "max_average", "min_average"]
PATCH_TYPES: tuple[PatchType, ...] = (
    "max_pixel",
    "min_pixel",
    "max_average",
    "min_average",
)
# Whether the ranking keeps the largest scores (True) or the smallest.
_LARGEST: dict[PatchType, bool] = {
    "max_pixel": True,
    "min_pixel": False,
    "max_average": True,
    "min_average": False,
}
_PIXEL_TYPES: frozenset[PatchType] = frozenset({"max_pixel", "min_pixel"})


def crop_side(act_side: int, input_side: int) -> int:
    """Input-pixel side of the crop stored for pixel-extreme patches."""
    ratio = math.ceil(input_side / act_side)
    return max(min(ratio * PATCH_FACTOR, input_side), min(MIN_PATCH, input_side))


@dataclass(frozen=True)
class TypePatches:
    """CPU view of one grid: per-channel top-`N_PER_CHANNEL` entries.

    Rows along dim 1 are sorted best-first (descending for max types,
    ascending for min types). Slots never filled hold non-finite `values`;
    their patches/heat are zeros and must be masked by the renderer.

    `heat` is the channel's full activation map for the stored sample.
    When `crop` is true, `patches` are input crops whose window origin is
    `(top, left)` in input pixels; otherwise they are whole input images
    and `top`/`left` are zero.
    """

    values: Tensor  # (C, N) fp32
    patches: Tensor  # (C, N, Cin, ph, pw) fp32, input value range
    heat: Tensor  # (C, N, Hh, Wh) fp32
    top: Tensor  # (C, N) int64
    left: Tensor  # (C, N) int64
    input_hw: tuple[int, int]
    crop: bool


@dataclass(frozen=True)
class PatchSnapshot:
    """Immutable CPU view of all four grids of one accumulator."""

    by_type: dict[PatchType, TypePatches]


@dataclass
class _TypeBuffer:
    """Running per-channel top-N state for one patch type (GPU tensors)."""

    largest: bool
    crop: bool
    vals: Tensor  # (C, N) fp32, ∓inf placeholders
    patches: Tensor  # (C, N, Cin, ph, pw) fp32
    heat: Tensor  # (C, N, Hh, Wh) fp32
    top: Tensor  # (C, N) int64
    left: Tensor  # (C, N) int64


@dataclass(frozen=True)
class _Config:
    """Shapes frozen on first update so buffers stay uniform tensors."""

    act_ndim: int
    channels: int
    act_hw: tuple[int, int]  # (1, 1) for 2D activations
    in_channels: int
    input_hw: tuple[int, int]
    crop_hw: tuple[int, int]


class PatchAccumulator:
    """Per-channel extreme patches for one (layer, phase, epoch) bucket."""

    def __init__(self) -> None:
        self._config: _Config | None = None
        self._buffers: dict[PatchType, _TypeBuffer] = {}

    def clear(self) -> None:
        """Drop all GPU buffers (e.g. when a newer epoch supersedes this one)."""
        self._config = None
        self._buffers = {}

    @property
    def empty(self) -> bool:
        return self._config is None

    def update(self, *, act: Tensor, x: Tensor) -> None:
        """Fold one batch's activations into the running per-channel top-N.

        `act` is the watched layer's output `(B, C, H, W)` or `(B, F)`;
        `x` is the model's image input `(B, Cin, Hin, Win)`, `Cin in (1, 3)`.
        Silently skips unsupported shapes so exotic layers just leave the
        galleries empty instead of breaking training.
        """
        if act.ndim not in (2, 4) or not act.is_floating_point():
            return
        if x.ndim != 4 or x.shape[1] not in (1, 3):
            return
        if act.shape[0] != x.shape[0] or act.shape[0] == 0:
            return
        config = self._make_config(act, x)
        if self._config is None:
            self._config = config
            self._init_buffers(x.device)
        elif config != self._config:
            return

        if act.ndim == 4:
            self._update_4d(act, x)
        else:
            self._update_2d(act, x)

    def snapshot(self) -> PatchSnapshot | None:
        """Copy the running state to CPU. `None` until the first update."""
        config = self._config
        if config is None:
            return None
        out: dict[PatchType, TypePatches] = {}
        for ptype, buf in self._buffers.items():
            out[ptype] = TypePatches(
                values=buf.vals.cpu(),
                patches=buf.patches.cpu(),
                heat=buf.heat.cpu(),
                top=buf.top.cpu(),
                left=buf.left.cpu(),
                input_hw=config.input_hw,
                crop=buf.crop,
            )
        return PatchSnapshot(by_type=out)

    def _make_config(self, act: Tensor, x: Tensor) -> _Config:
        hin, win = int(x.shape[2]), int(x.shape[3])
        if act.ndim == 4:
            ha, wa = int(act.shape[2]), int(act.shape[3])
            crop_hw = (crop_side(ha, hin), crop_side(wa, win))
            return _Config(
                act_ndim=4,
                channels=int(act.shape[1]),
                act_hw=(ha, wa),
                in_channels=int(x.shape[1]),
                input_hw=(hin, win),
                crop_hw=crop_hw,
            )
        return _Config(
            act_ndim=2,
            channels=int(act.shape[1]),
            act_hw=(1, 1),
            in_channels=int(x.shape[1]),
            input_hw=(hin, win),
            crop_hw=(hin, win),
        )

    def _init_buffers(self, device: torch.device) -> None:
        config = self._config
        assert config is not None
        c, n = config.channels, N_PER_CHANNEL
        cin = config.in_channels
        hh, wh = config.act_hw
        for ptype in PATCH_TYPES:
            largest = _LARGEST[ptype]
            crop = ptype in _PIXEL_TYPES and config.act_ndim == 4
            ph, pw = config.crop_hw if crop else config.input_hw
            fill = float("-inf") if largest else float("inf")
            self._buffers[ptype] = _TypeBuffer(
                largest=largest,
                crop=crop,
                vals=torch.full((c, n), fill, dtype=torch.float32, device=device),
                patches=torch.zeros(
                    (c, n, cin, ph, pw), dtype=torch.float32, device=device
                ),
                heat=torch.zeros((c, n, hh, wh), dtype=torch.float32, device=device),
                top=torch.zeros((c, n), dtype=torch.int64, device=device),
                left=torch.zeros((c, n), dtype=torch.int64, device=device),
            )

    def _update_4d(self, act: Tensor, x: Tensor) -> None:
        config = self._config
        assert config is not None
        af = act.detach().to(torch.float32)
        b, c = af.shape[0], af.shape[1]
        ha, wa = config.act_hw
        flat = af.reshape(b, c, ha * wa)
        pix_max, pix_max_idx = flat.max(dim=2)  # (B, C)
        pix_min, pix_min_idx = flat.min(dim=2)
        avg = flat.mean(dim=2)
        per_type: dict[PatchType, tuple[Tensor, Tensor | None]] = {
            "max_pixel": (pix_max, pix_max_idx),
            "min_pixel": (pix_min, pix_min_idx),
            "max_average": (avg, None),
            "min_average": (avg, None),
        }
        for ptype, (scores, flat_idx) in per_type.items():
            buf = self._buffers[ptype]
            cand_vals, cand_samples = self._rank_candidates(scores, buf.largest)
            heat = self._gather_heat(af, cand_samples)
            if buf.crop:
                assert flat_idx is not None
                pos = flat_idx.transpose(0, 1).gather(1, cand_samples)  # (C, k)
                top, left = self._window_origins(pos)
                patches = self._gather_windows(x, cand_samples, top, left)
            else:
                patches = self._gather_images(x, cand_samples)
                top = torch.zeros_like(cand_samples)
                left = torch.zeros_like(cand_samples)
            self._merge(buf, cand_vals, patches, heat, top, left)

    def _update_2d(self, act: Tensor, x: Tensor) -> None:
        # (B, F): a feature is both its own "pixel" and its own spatial
        # mean, so the pixel and average grids coincide; the patch is the
        # whole image and the heatmap a single uniform cell.
        af = act.detach().to(torch.float32)
        for ptype in PATCH_TYPES:
            buf = self._buffers[ptype]
            cand_vals, cand_samples = self._rank_candidates(af, buf.largest)
            patches = self._gather_images(x, cand_samples)
            heat = cand_vals[:, :, None, None]
            zeros = torch.zeros_like(cand_samples)
            self._merge(buf, cand_vals, patches, heat, zeros, zeros)

    def _rank_candidates(
        self, scores: Tensor, largest: bool
    ) -> tuple[Tensor, Tensor]:
        """Per-channel best `min(N, B)` batch rows: `(C, k)` values + samples.

        NaN scores (diverged training) are demoted to the placeholder
        extreme so they never enter the buffers.
        """
        guard = float("-inf") if largest else float("inf")
        per_channel = scores.transpose(0, 1).nan_to_num(nan=guard)  # (C, B)
        k = min(N_PER_CHANNEL, per_channel.shape[1])
        vals, samples = per_channel.topk(k, dim=1, largest=largest)
        return vals, samples

    def _window_origins(self, pos: Tensor) -> tuple[Tensor, Tensor]:
        """Map activation-space flat positions `(C, k)` to crop origins."""
        config = self._config
        assert config is not None
        ha, wa = config.act_hw
        hin, win = config.input_hw
        ph, pw = config.crop_hw
        y, xpos = pos // wa, pos % wa
        cy = ((y.to(torch.float32) + 0.5) * (hin / ha)).to(torch.int64)
        cx = ((xpos.to(torch.float32) + 0.5) * (win / wa)).to(torch.int64)
        top = (cy - ph // 2).clamp_(0, hin - ph)
        left = (cx - pw // 2).clamp_(0, win - pw)
        return top, left

    def _gather_windows(
        self, x: Tensor, samples: Tensor, top: Tensor, left: Tensor
    ) -> Tensor:
        """Crop `(C, k)` fixed-size windows from `x` in one fancy-index."""
        config = self._config
        assert config is not None
        ph, pw = config.crop_hw
        c, k = samples.shape
        flat_samples = samples.reshape(-1)  # (M,)
        rows = top.reshape(-1)[:, None, None] + torch.arange(
            ph, device=x.device
        ).reshape(1, ph, 1)
        cols = left.reshape(-1)[:, None, None] + torch.arange(
            pw, device=x.device
        ).reshape(1, 1, pw)
        # Advanced indices split by the channel slice put the broadcast
        # (M, ph, pw) dims first: result is (M, ph, pw, Cin).
        windows = x.detach()[flat_samples[:, None, None], :, rows, cols]
        return (
            windows.permute(0, 3, 1, 2)
            .to(torch.float32)
            .reshape(c, k, config.in_channels, ph, pw)
        )

    def _gather_images(self, x: Tensor, samples: Tensor) -> Tensor:
        """Whole input images for `(C, k)` sample indices."""
        config = self._config
        assert config is not None
        c, k = samples.shape
        imgs = x.detach()[samples.reshape(-1)].to(torch.float32)
        return imgs.reshape(c, k, config.in_channels, *config.input_hw)

    def _gather_heat(self, af: Tensor, samples: Tensor) -> Tensor:
        """Per-channel activation maps for `(C, k)` sample indices."""
        c, k = samples.shape
        chans = torch.arange(c, device=af.device).repeat_interleave(k)
        maps = af[samples.reshape(-1), chans]  # (M, Ha, Wa)
        return maps.reshape(c, k, *af.shape[2:])

    def _merge(
        self,
        buf: _TypeBuffer,
        cand_vals: Tensor,
        patches: Tensor,
        heat: Tensor,
        top: Tensor,
        left: Tensor,
    ) -> None:
        """Keep the per-channel best N of buffer ∪ candidates (sorted)."""
        n = N_PER_CHANNEL
        all_vals = torch.cat([buf.vals, cand_vals], dim=1)
        buf.vals, sel = all_vals.topk(n, dim=1, largest=buf.largest)
        c = sel.shape[0]
        patch_idx = sel.reshape(c, n, 1, 1, 1).expand(-1, -1, *patches.shape[2:])
        buf.patches = torch.cat([buf.patches, patches], dim=1).gather(1, patch_idx)
        heat_idx = sel.reshape(c, n, 1, 1).expand(-1, -1, *heat.shape[2:])
        buf.heat = torch.cat([buf.heat, heat], dim=1).gather(1, heat_idx)
        buf.top = torch.cat([buf.top, top], dim=1).gather(1, sel)
        buf.left = torch.cat([buf.left, left], dim=1).gather(1, sel)
