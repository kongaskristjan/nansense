"""Tests for per-channel extreme-activation patch gathering."""

from __future__ import annotations

import pytest
import torch

from playgrad.patches import (
    N_PER_CHANNEL,
    PATCH_TYPES,
    PatchAccumulator,
    PatchType,
    crop_side,
)


def _image_batch(b: int, cin: int = 3, side: int = 8) -> torch.Tensor:
    """Images whose every pixel encodes (sample, channel, y, x) uniquely."""
    return torch.arange(b * cin * side * side, dtype=torch.float32).reshape(
        b, cin, side, side
    )


def test_crop_side_clamps_to_min_and_image() -> None:
    assert crop_side(8, 32) == 16  # ceil(4) * 4
    assert crop_side(32, 32) == 10  # 4 -> floored at MIN_PATCH
    assert crop_side(2, 32) == 32  # 64 -> capped at the image side
    assert crop_side(4, 4) == 4  # MIN_PATCH capped at tiny images


def test_max_pixel_crop_geometry() -> None:
    acc = PatchAccumulator()
    x = _image_batch(3, side=32)
    act = torch.zeros(3, 2, 8, 8)
    act[1, 0, 2, 7] = 10.0
    acc.update(act=act, x=x)
    snap = acc.snapshot()
    assert snap is not None
    tp = snap.by_type["max_pixel"]
    assert tp.crop
    assert tp.values[0, 0].item() == 10.0
    # act 8x8 -> input 32x32: center (10, 30), 16x16 window clamped to
    # top=2, left=16 (left would be 22 but 32-16=16 caps it).
    assert tp.top[0, 0].item() == 2
    assert tp.left[0, 0].item() == 16
    assert torch.equal(tp.patches[0, 0], x[1, :, 2:18, 16:32])


@pytest.mark.parametrize(
    ("ptype", "sign"),
    [
        ("max_pixel", 1.0),
        ("min_pixel", -1.0),
        ("max_average", 1.0),
        ("min_average", -1.0),
    ],
)
def test_extreme_sample_wins_first_slot(ptype: PatchType, sign: float) -> None:
    acc = PatchAccumulator()
    x = _image_batch(4, side=8)
    act = torch.zeros(4, 2, 4, 4)
    act[2, 1] = sign  # uniform map: extreme pixel AND extreme mean
    acc.update(act=act, x=x)
    snap = acc.snapshot()
    assert snap is not None
    tp = snap.by_type[ptype]
    assert tp.values[1, 0].item() == sign
    if not tp.crop:  # average types store the whole image
        assert torch.equal(tp.patches[1, 0], x[2])


def test_average_diverges_from_pixel() -> None:
    acc = PatchAccumulator()
    x = _image_batch(2, side=8)
    act = torch.zeros(2, 1, 4, 4)
    act[0, 0, 1, 1] = 16.0  # huge single pixel, mean = 1.0
    act[1, 0] = 2.0  # flat map, mean = 2.0
    acc.update(act=act, x=x)
    snap = acc.snapshot()
    assert snap is not None
    assert snap.by_type["max_pixel"].values[0, 0].item() == 16.0
    assert snap.by_type["max_average"].values[0, 0].item() == 2.0
    assert torch.equal(snap.by_type["max_average"].patches[0, 0], x[1])


def _flat_act(values: list[float]) -> torch.Tensor:
    """(B, 1, 2, 2) activations with a uniform map per sample."""
    b = len(values)
    return torch.tensor(values).reshape(b, 1, 1, 1).expand(b, 1, 2, 2).clone()


def test_merge_across_updates_keeps_global_top_n() -> None:
    acc = PatchAccumulator()
    acc.update(act=_flat_act([1.0, 2.0, 3.0]), x=_image_batch(3))
    acc.update(act=_flat_act([5.0, 0.5, 4.0]), x=_image_batch(3))
    snap = acc.snapshot()
    assert snap is not None
    assert snap.by_type["max_average"].values[0].tolist() == [
        5.0,
        4.0,
        3.0,
        2.0,
        1.0,
    ]
    assert snap.by_type["min_average"].values[0].tolist() == [
        0.5,
        1.0,
        2.0,
        3.0,
        4.0,
    ]


def test_eviction_beyond_n_per_channel() -> None:
    acc = PatchAccumulator()
    scores = [float(i) for i in range(8)]
    acc.update(act=_flat_act(scores), x=_image_batch(8))
    snap = acc.snapshot()
    assert snap is not None
    assert snap.by_type["max_average"].values[0].tolist() == [
        7.0,
        6.0,
        5.0,
        4.0,
        3.0,
    ]


def test_patch_follows_value_through_merge() -> None:
    acc = PatchAccumulator()
    x1 = _image_batch(2)
    x2 = _image_batch(2) + 1000.0
    acc.update(act=_flat_act([1.0, 2.0]), x=x1)
    acc.update(act=_flat_act([3.0, 0.5]), x=x2)
    snap = acc.snapshot()
    assert snap is not None
    tp = snap.by_type["max_average"]
    assert tp.values[0, :3].tolist() == [3.0, 2.0, 1.0]
    assert torch.equal(tp.patches[0, 0], x2[0])
    assert torch.equal(tp.patches[0, 1], x1[1])
    assert torch.equal(tp.patches[0, 2], x1[0])


@pytest.mark.parametrize("corner", [(0, 0), (7, 7)])
def test_border_extremes_stay_in_bounds(corner: tuple[int, int]) -> None:
    acc = PatchAccumulator()
    x = _image_batch(1, side=32)
    act = torch.zeros(1, 1, 8, 8)
    act[0, 0, corner[0], corner[1]] = 1.0
    acc.update(act=act, x=x)
    snap = acc.snapshot()
    assert snap is not None
    tp = snap.by_type["max_pixel"]
    top, left = tp.top[0, 0].item(), tp.left[0, 0].item()
    assert 0 <= top <= 32 - 16
    assert 0 <= left <= 32 - 16
    assert tp.patches.shape[-2:] == (16, 16)


def test_heat_is_winning_samples_channel_map() -> None:
    acc = PatchAccumulator()
    x = _image_batch(3, side=8)
    act = torch.randn(3, 2, 4, 4)
    act[2, 1, 0, 0] = 99.0
    acc.update(act=act, x=x)
    snap = acc.snapshot()
    assert snap is not None
    assert torch.equal(snap.by_type["max_pixel"].heat[1, 0], act[2, 1])


def test_2d_activation_uses_whole_images_and_uniform_heat() -> None:
    acc = PatchAccumulator()
    x = _image_batch(3, side=8)
    act = torch.tensor([[1.0, -1.0], [3.0, 0.0], [2.0, 5.0]])
    acc.update(act=act, x=x)
    snap = acc.snapshot()
    assert snap is not None
    for ptype in ("max_pixel", "max_average"):
        tp = snap.by_type[ptype]
        assert not tp.crop
        assert tp.values[0, :3].tolist() == [3.0, 2.0, 1.0]
        assert tp.values[1, :3].tolist() == [5.0, 0.0, -1.0]
        assert torch.equal(tp.patches[0, 0], x[1])
        assert tp.heat[0, 0].shape == (1, 1)
        assert tp.heat[0, 0].item() == 3.0


def test_unfilled_slots_are_non_finite() -> None:
    acc = PatchAccumulator()
    acc.update(act=_flat_act([1.0, 2.0]), x=_image_batch(2))
    snap = acc.snapshot()
    assert snap is not None
    vals = snap.by_type["max_average"].values[0]
    assert torch.isfinite(vals[:2]).all()
    assert not torch.isfinite(vals[2:]).any()


@pytest.mark.parametrize(
    ("act", "x"),
    [
        (torch.zeros(2, 3, 4, 4, dtype=torch.int64), _image_batch(2)),  # int act
        (torch.zeros(2, 3, 4, 4), torch.zeros(2, 10)),  # non-image input
        (torch.zeros(2, 3, 4, 4), torch.zeros(2, 4, 8, 8)),  # 4-channel input
        (torch.zeros(3, 3, 4, 4), _image_batch(2)),  # batch mismatch
        (torch.zeros(2, 3, 4), _image_batch(2)),  # 3D activation
        (torch.zeros(0, 3, 4, 4), _image_batch(0)),  # empty batch
    ],
)
def test_unsupported_shapes_are_skipped(act: torch.Tensor, x: torch.Tensor) -> None:
    acc = PatchAccumulator()
    acc.update(act=act, x=x)
    assert acc.snapshot() is None
    assert acc.empty


def test_shape_change_after_init_is_skipped() -> None:
    acc = PatchAccumulator()
    acc.update(act=_flat_act([1.0]), x=_image_batch(1))
    acc.update(act=torch.zeros(1, 1, 3, 3), x=_image_batch(1))  # new act shape
    snap = acc.snapshot()
    assert snap is not None
    assert snap.by_type["max_average"].values[0, 0].item() == 1.0


def test_nan_scores_never_enter_buffers() -> None:
    acc = PatchAccumulator()
    act = _flat_act([1.0, 2.0])
    act[1] = float("nan")
    acc.update(act=act, x=_image_batch(2))
    snap = acc.snapshot()
    assert snap is not None
    for ptype in PATCH_TYPES:
        vals = snap.by_type[ptype].values[0]
        finite = vals[torch.isfinite(vals)]
        assert finite.tolist() == [1.0]


def test_grayscale_input() -> None:
    acc = PatchAccumulator()
    x = _image_batch(2, cin=1, side=8)
    acc.update(act=_flat_act([1.0, 2.0]), x=x)
    snap = acc.snapshot()
    assert snap is not None
    assert snap.by_type["max_average"].patches.shape[2] == 1


def test_clear_resets_accumulator() -> None:
    acc = PatchAccumulator()
    acc.update(act=_flat_act([1.0]), x=_image_batch(1))
    assert not acc.empty
    acc.clear()
    assert acc.empty
    assert acc.snapshot() is None


def test_buffer_shapes() -> None:
    acc = PatchAccumulator()
    x = _image_batch(2, side=32)
    acc.update(act=torch.randn(2, 3, 8, 8), x=x)
    snap = acc.snapshot()
    assert snap is not None
    pixel = snap.by_type["max_pixel"]
    assert pixel.patches.shape == (3, N_PER_CHANNEL, 3, 16, 16)  # ceil(4)*4
    assert pixel.heat.shape == (3, N_PER_CHANNEL, 8, 8)
    avg = snap.by_type["min_average"]
    assert avg.patches.shape == (3, N_PER_CHANNEL, 3, 32, 32)
    assert avg.input_hw == (32, 32)
