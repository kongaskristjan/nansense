"""Per-channel extreme-activation input patches for watched layers.

For each watched layer we keep, per `(phase, epoch)` and per activation
channel, the `n_per_channel` input samples that produced the most extreme
activations, under up to four rankings:

- ``max_pixel`` / ``min_pixel`` — the channel's single largest / smallest
  activation value anywhere in its spatial map. The stored patch is a crop
  of the input image around the receptive-field location of that pixel
  (ratio-mapped, not an exact receptive field).
- ``max_average`` / ``min_average`` — the channel's spatial mean. There is
  no single location to crop around, so the stored patch is the whole
  input image. Collected only when `average_patches` is on (a performance
  setting, off by default — whole-image payloads double the buffer cost).

Each stored entry also keeps the channel's full activation map for that
sample, so the UI can blend an activation heatmap over the patch.

Patch and heat payloads are stored quantized: uint8 levels `0..254` plus a
per-(channel, sample) fp32 `[offset, scale]` pair, with byte 255 reserved
as a "non-finite value here" sentinel that dequantizes to NaN. Rendering
is 8-bit anyway, and the payloads are pure cargo — candidates are
quantized once at gather time and the merge only permutes bytes (and
their scale rows) — so this quarters both GPU footprint and frozen-moment
size. The `vals` ranking scores stay fp32: ranking arithmetic depends on
them.

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
from typing import Any, Literal, cast

import torch
from torch import Tensor


def _cpu_copy(t: Tensor) -> Tensor:
    """An independent CPU copy (never an alias, unlike `Tensor.cpu()`)."""
    return t.detach().to("cpu", copy=True)

# Default number of extreme samples kept per channel per ranking. The live
# value is a per-accumulator setting (`PatchAccumulator(n_per_channel=...)`),
# driven by the "Performance" settings section — both it and the channel limit
# scale GPU VRAM, so they are user-tunable.
DEFAULT_SAMPLES_PER_CHANNEL: int = 5
# Whether the average-type grids are collected. Off by default: they store a
# whole input image per (channel, sample) slot, roughly doubling the patch
# buffers' GPU and moment-file cost for the least-consulted grids.
DEFAULT_AVERAGE_PATCHES: bool = False
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

# Payload quantization: finite values map to levels 0..254 of a per-slot
# affine grid; byte 255 is reserved as the "non-finite value here" sentinel
# (dequantizes to NaN, so a diverged activation stays visible in the heat).
_QUANT_LEVELS: int = 254
_SENTINEL_BYTE: int = 255


def crop_side(act_side: int, input_side: int) -> int:
    """Input-pixel side of the crop stored for pixel-extreme patches."""
    ratio = math.ceil(input_side / act_side)
    return max(min(ratio * PATCH_FACTOR, input_side), min(MIN_PATCH, input_side))


def _quantize(t: Tensor) -> tuple[Tensor, Tensor]:
    """Quantize a `(C, K, ...)` fp32 payload to uint8 + per-slot scales.

    One `[offset, scale]` pair per `(channel, sample)` slot, computed over
    that slot's whole trailing slice: `offset` is the slice minimum over its
    finite values, `scale` is `(max - min) / 254`. Finite values encode as
    levels `0..254`; non-finite elements as the byte-255 sentinel. A
    constant slice gets scale 0 (its finite bytes all 0); a slot with no
    finite values gets `[0, 0]` and all bytes 255.

    Returns `(bytes, scales)`: `bytes` uint8 with `t`'s shape, `scales`
    fp32 `(C, K, 2)`.
    """
    c, k = t.shape[0], t.shape[1]
    flat = t.reshape(c, k, -1)
    finite = torch.isfinite(flat)
    inf = torch.full_like(flat, float("inf"))
    lo = torch.where(finite, flat, inf).amin(dim=2)
    hi = torch.where(finite, flat, -inf).amax(dim=2)
    # Slots with no finite values keep [0, 0] instead of the ±inf extremes.
    no_finite = ~finite.any(dim=2)
    zero = torch.zeros_like(lo)
    lo = torch.where(no_finite, zero, lo)
    scale = torch.where(no_finite, zero, (hi - lo) / _QUANT_LEVELS)
    # Constant slices (max == min) get scale 0 — guard the division; their
    # finite bytes become 0 and dequantize back to the offset exactly.
    denom = torch.where(scale > 0, scale, torch.ones_like(scale))
    levels = (flat - lo[:, :, None]).div_(denom[:, :, None])
    levels = levels.round_().clamp_(0, _QUANT_LEVELS)
    sentinel = torch.full_like(levels, float(_SENTINEL_BYTE))
    levels = torch.where(finite, levels, sentinel)
    return levels.to(torch.uint8).reshape(t.shape), torch.stack([lo, scale], dim=2)


def _dequantize(q: Tensor, scales: Tensor) -> Tensor:
    """Invert `_quantize`: `offset + byte * scale`, sentinel bytes to NaN."""
    c, n = q.shape[0], q.shape[1]
    levels = q.reshape(c, n, -1).to(torch.float32)
    values = scales[:, :, 0:1] + levels * scales[:, :, 1:2]
    nan = torch.full_like(values, float("nan"))
    return torch.where(levels == float(_SENTINEL_BYTE), nan, values).reshape(q.shape)


@dataclass(frozen=True)
class TypePatches:
    """CPU view of one grid: per-channel top-`n_per_channel` entries.

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
    """Immutable CPU view of one accumulator's grids.

    Keys follow `PATCH_TYPES` order; the average types are present only
    when the accumulator collects them (`average_patches`).
    """

    by_type: dict[PatchType, TypePatches]


@dataclass
class _TypeBuffer:
    """Running per-channel top-N state for one patch type (GPU tensors).

    `patches` and `heat` are quantized payloads (see `_quantize`); their
    `*_scales` carry the per-slot `[offset, scale]` pairs that dequantize
    them. Only `vals` takes part in ranking arithmetic.
    """

    largest: bool
    crop: bool
    vals: Tensor  # (C, N) fp32, ∓inf placeholders
    patches: Tensor  # (C, N, Cin, ph, pw) uint8
    patch_scales: Tensor  # (C, N, 2) fp32 [offset, scale]
    heat: Tensor  # (C, N, Hh, Wh) uint8
    heat_scales: Tensor  # (C, N, 2) fp32 [offset, scale]
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
        # Samples kept per channel per ranking, and whether the average-type
        # grids are collected; frozen on first update (the buffers are
        # flushed when either setting changes, so they never vary within one
        # accumulator's lifetime).
        self._n_per_channel: int = DEFAULT_SAMPLES_PER_CHANNEL
        self._average_patches: bool = DEFAULT_AVERAGE_PATCHES

    def clear(self) -> None:
        """Drop all GPU buffers (e.g. when a newer epoch supersedes this one)."""
        self._config = None
        self._buffers = {}

    @property
    def empty(self) -> bool:
        return self._config is None

    def update(
        self,
        *,
        act: Tensor,
        x: Tensor,
        channel_limit: int | None = None,
        n_per_channel: int = DEFAULT_SAMPLES_PER_CHANNEL,
        average_patches: bool = DEFAULT_AVERAGE_PATCHES,
    ) -> None:
        """Fold one batch's activations into the running per-channel top-N.

        `act` is the watched layer's output `(B, C, H, W)` or `(B, F)`;
        `x` is the model's image input `(B, Cin, Hin, Win)`, `Cin in (1, 3)`.
        Silently skips unsupported shapes so exotic layers just leave the
        galleries empty instead of breaking training.

        `channel_limit` caps the work (and buffer size) to the first that many
        channels — the per-channel image patches are the dominant GPU cost, so
        this is the main VRAM knob; `None` keeps every channel. `n_per_channel`
        sets how many extreme samples are kept per channel per ranking;
        `average_patches` whether the whole-image average grids are collected
        at all. All three are frozen for the buffers' lifetime; callers flush
        the accumulator when any of them changes.
        """
        if act.ndim not in (2, 4) or not act.is_floating_point():
            return
        if x.ndim != 4 or x.shape[1] not in (1, 3):
            return
        if act.shape[0] != x.shape[0] or act.shape[0] == 0:
            return
        # dim 1 is the channel axis for both (B, C, H, W) and (B, F).
        if channel_limit is not None and act.shape[1] > channel_limit:
            act = act[:, :channel_limit]
        self._n_per_channel = n_per_channel
        self._average_patches = average_patches
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
        """Copy the running state to CPU. `None` until the first update.

        The payloads cross the bus as uint8 (a quarter of the fp32 bytes)
        and are dequantized on the CPU side, so `TypePatches` and every
        render path keep seeing fp32 values.
        """
        config = self._config
        if config is None:
            return None
        out: dict[PatchType, TypePatches] = {}
        for ptype, buf in self._buffers.items():
            out[ptype] = TypePatches(
                values=buf.vals.cpu(),
                patches=_dequantize(buf.patches.cpu(), buf.patch_scales.cpu()),
                heat=_dequantize(buf.heat.cpu(), buf.heat_scales.cpu()),
                top=buf.top.cpu(),
                left=buf.left.cpu(),
                input_hw=config.input_hw,
                crop=buf.crop,
            )
        return PatchSnapshot(by_type=out)

    def state_dict(self) -> dict[str, Any]:
        """Independent CPU copy of the buffers + frozen shapes, for a frozen
        moment. Plain tensors/primitives only (`weights_only`-loadable); an
        accumulator that never saw data stores `config: None`. Payloads are
        stored raw — uint8 bytes plus their scale tensors — which is what
        keeps moment files ~4x smaller than fp32 galleries."""
        config = self._config
        return {
            "n_per_channel": self._n_per_channel,
            "average_patches": self._average_patches,
            "config": None
            if config is None
            else {
                "act_ndim": config.act_ndim,
                "channels": config.channels,
                "act_hw": list(config.act_hw),
                "in_channels": config.in_channels,
                "input_hw": list(config.input_hw),
                "crop_hw": list(config.crop_hw),
            },
            "buffers": {
                ptype: {
                    "largest": buf.largest,
                    "crop": buf.crop,
                    "vals": _cpu_copy(buf.vals),
                    "patches": _cpu_copy(buf.patches),
                    "patch_scales": _cpu_copy(buf.patch_scales),
                    "heat": _cpu_copy(buf.heat),
                    "heat_scales": _cpu_copy(buf.heat_scales),
                    "top": _cpu_copy(buf.top),
                    "left": _cpu_copy(buf.left),
                }
                for ptype, buf in self._buffers.items()
            },
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Restore a `state_dict()`. Buffers stay on CPU — a restored
        accumulator backs a frozen (browse-only) moment; the never-filled
        `∓inf` placeholder slots round-trip as-is for the renderer's mask."""
        self._n_per_channel = int(state["n_per_channel"])
        self._average_patches = bool(state["average_patches"])
        config = state["config"]
        if config is None:
            self._config = None
            self._buffers = {}
            return
        hin, win = (int(v) for v in config["input_hw"])
        ha, wa = (int(v) for v in config["act_hw"])
        ph, pw = (int(v) for v in config["crop_hw"])
        self._config = _Config(
            act_ndim=int(config["act_ndim"]),
            channels=int(config["channels"]),
            act_hw=(ha, wa),
            in_channels=int(config["in_channels"]),
            input_hw=(hin, win),
            crop_hw=(ph, pw),
        )
        self._buffers = {
            ptype: _TypeBuffer(
                largest=bool(buf["largest"]),
                crop=bool(buf["crop"]),
                vals=buf["vals"].clone(),
                patches=buf["patches"].clone(),
                patch_scales=buf["patch_scales"].clone(),
                heat=buf["heat"].clone(),
                heat_scales=buf["heat_scales"].clone(),
                top=buf["top"].clone(),
                left=buf["left"].clone(),
            )
            for ptype, buf in cast(
                "dict[PatchType, dict[str, Any]]", state["buffers"]
            ).items()
            if ptype in PATCH_TYPES
        }

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

    def _active_types(self) -> tuple[PatchType, ...]:
        """The rankings this accumulator collects, in `PATCH_TYPES` order."""
        if self._average_patches:
            return PATCH_TYPES
        return tuple(t for t in PATCH_TYPES if t in _PIXEL_TYPES)

    def _init_buffers(self, device: torch.device) -> None:
        config = self._config
        assert config is not None
        c, n = config.channels, self._n_per_channel
        cin = config.in_channels
        hh, wh = config.act_hw
        for ptype in self._active_types():
            largest = _LARGEST[ptype]
            crop = ptype in _PIXEL_TYPES and config.act_ndim == 4
            ph, pw = config.crop_hw if crop else config.input_hw
            fill = float("-inf") if largest else float("inf")
            # Unfilled slots dequantize to zeros (byte 0, offset 0, scale 0),
            # matching the pre-quantization zero-filled buffers.
            self._buffers[ptype] = _TypeBuffer(
                largest=largest,
                crop=crop,
                vals=torch.full((c, n), fill, dtype=torch.float32, device=device),
                patches=torch.zeros(
                    (c, n, cin, ph, pw), dtype=torch.uint8, device=device
                ),
                patch_scales=torch.zeros(
                    (c, n, 2), dtype=torch.float32, device=device
                ),
                heat=torch.zeros((c, n, hh, wh), dtype=torch.uint8, device=device),
                heat_scales=torch.zeros(
                    (c, n, 2), dtype=torch.float32, device=device
                ),
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
        per_type: dict[PatchType, tuple[Tensor, Tensor | None]] = {
            "max_pixel": (pix_max, pix_max_idx),
            "min_pixel": (pix_min, pix_min_idx),
        }
        if "max_average" in self._buffers:
            avg = flat.mean(dim=2)
            per_type["max_average"] = (avg, None)
            per_type["min_average"] = (avg, None)
        for ptype, (scores, flat_idx) in per_type.items():
            buf = self._buffers[ptype]
            cand_vals, cand_samples = self._rank_candidates(scores, buf.largest)
            heat, heat_scales = _quantize(self._gather_heat(af, cand_samples))
            if buf.crop:
                assert flat_idx is not None
                pos = flat_idx.transpose(0, 1).gather(1, cand_samples)  # (C, k)
                top, left = self._window_origins(pos)
                windows = self._gather_windows(x, cand_samples, top, left)
            else:
                windows = self._gather_images(x, cand_samples)
                top = torch.zeros_like(cand_samples)
                left = torch.zeros_like(cand_samples)
            patches, patch_scales = _quantize(windows)
            self._merge(
                buf, cand_vals, patches, patch_scales, heat, heat_scales, top, left
            )

    def _update_2d(self, act: Tensor, x: Tensor) -> None:
        # (B, F): a feature is both its own "pixel" and its own spatial
        # mean, so the pixel and average grids coincide; the patch is the
        # whole image and the heatmap a single uniform cell.
        af = act.detach().to(torch.float32)
        for buf in self._buffers.values():
            cand_vals, cand_samples = self._rank_candidates(af, buf.largest)
            patches, patch_scales = _quantize(self._gather_images(x, cand_samples))
            heat, heat_scales = _quantize(cand_vals[:, :, None, None])
            zeros = torch.zeros_like(cand_samples)
            self._merge(
                buf, cand_vals, patches, patch_scales, heat, heat_scales, zeros, zeros
            )

    def _rank_candidates(
        self, scores: Tensor, largest: bool
    ) -> tuple[Tensor, Tensor]:
        """Per-channel best `min(N, B)` batch rows: `(C, k)` values + samples.

        NaN scores (diverged training) are demoted to the placeholder
        extreme so they never enter the buffers.
        """
        guard = float("-inf") if largest else float("inf")
        per_channel = scores.transpose(0, 1).nan_to_num(nan=guard)  # (C, B)
        k = min(self._n_per_channel, per_channel.shape[1])
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
        patch_scales: Tensor,
        heat: Tensor,
        heat_scales: Tensor,
        top: Tensor,
        left: Tensor,
    ) -> None:
        """Keep the per-channel best N of buffer ∪ candidates (sorted).

        Ranking reads only `vals`; the uint8 payloads AND their scale rows
        are row-selected with the same `sel` indices, so a byte can never be
        paired with another slot's dequantization parameters.
        """
        n = self._n_per_channel
        all_vals = torch.cat([buf.vals, cand_vals], dim=1)
        buf.vals, sel = all_vals.topk(n, dim=1, largest=buf.largest)
        c = sel.shape[0]
        patch_idx = sel.reshape(c, n, 1, 1, 1).expand(-1, -1, *patches.shape[2:])
        buf.patches = torch.cat([buf.patches, patches], dim=1).gather(1, patch_idx)
        heat_idx = sel.reshape(c, n, 1, 1).expand(-1, -1, *heat.shape[2:])
        buf.heat = torch.cat([buf.heat, heat], dim=1).gather(1, heat_idx)
        scale_idx = sel.reshape(c, n, 1).expand(-1, -1, 2)
        buf.patch_scales = torch.cat([buf.patch_scales, patch_scales], dim=1).gather(
            1, scale_idx
        )
        buf.heat_scales = torch.cat([buf.heat_scales, heat_scales], dim=1).gather(
            1, scale_idx
        )
        buf.top = torch.cat([buf.top, top], dim=1).gather(1, sel)
        buf.left = torch.cat([buf.left, left], dim=1).gather(1, sel)
