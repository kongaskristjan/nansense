"""Tests for per-channel extreme-activation patch gathering."""

from __future__ import annotations

import pytest
import torch

from nansense.patches import (
    DEFAULT_SAMPLES_PER_CHANNEL,
    PATCH_TYPES,
    PatchAccumulator,
    PatchType,
    _dequantize,
    _quantize,
    crop_side,
)


def _image_batch(b: int, cin: int = 3, side: int = 8) -> torch.Tensor:
    """Images whose every pixel encodes (sample, channel, y, x) uniquely."""
    return torch.arange(b * cin * side * side, dtype=torch.float32).reshape(
        b, cin, side, side
    )


def _round_trip(t: torch.Tensor) -> torch.Tensor:
    """The quantize→dequantize image of one slot's payload.

    Stored patches/heat are quantized per (channel, sample) slot at gather
    time, so a stored payload must equal this round trip of its source
    exactly (same math, deterministic).
    """
    return _dequantize(*_quantize(t[None, None]))[0, 0]


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
    assert torch.equal(tp.patches[0, 0], _round_trip(x[1, :, 2:18, 16:32]))


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
    acc.update(act=act, x=x, average_patches=True)
    snap = acc.snapshot()
    assert snap is not None
    tp = snap.by_type[ptype]
    assert tp.values[1, 0].item() == sign
    if not tp.crop:  # average types store the whole image
        assert torch.equal(tp.patches[1, 0], _round_trip(x[2]))


def test_average_diverges_from_pixel() -> None:
    acc = PatchAccumulator()
    x = _image_batch(2, side=8)
    act = torch.zeros(2, 1, 4, 4)
    act[0, 0, 1, 1] = 16.0  # huge single pixel, mean = 1.0
    act[1, 0] = 2.0  # flat map, mean = 2.0
    acc.update(act=act, x=x, average_patches=True)
    snap = acc.snapshot()
    assert snap is not None
    assert snap.by_type["max_pixel"].values[0, 0].item() == 16.0
    assert snap.by_type["max_average"].values[0, 0].item() == 2.0
    assert torch.equal(snap.by_type["max_average"].patches[0, 0], _round_trip(x[1]))


def _flat_act(values: list[float]) -> torch.Tensor:
    """(B, 1, 2, 2) activations with a uniform map per sample."""
    b = len(values)
    return torch.tensor(values).reshape(b, 1, 1, 1).expand(b, 1, 2, 2).clone()


def test_merge_across_updates_keeps_global_top_n() -> None:
    acc = PatchAccumulator()
    acc.update(act=_flat_act([1.0, 2.0, 3.0]), x=_image_batch(3), average_patches=True)
    acc.update(act=_flat_act([5.0, 0.5, 4.0]), x=_image_batch(3), average_patches=True)
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
    acc.update(act=_flat_act(scores), x=_image_batch(8), average_patches=True)
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
    acc.update(act=_flat_act([1.0, 2.0]), x=x1, average_patches=True)
    acc.update(act=_flat_act([3.0, 0.5]), x=x2, average_patches=True)
    snap = acc.snapshot()
    assert snap is not None
    tp = snap.by_type["max_average"]
    assert tp.values[0, :3].tolist() == [3.0, 2.0, 1.0]
    assert torch.equal(tp.patches[0, 0], _round_trip(x2[0]))
    assert torch.equal(tp.patches[0, 1], _round_trip(x1[1]))
    assert torch.equal(tp.patches[0, 2], _round_trip(x1[0]))


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
    assert torch.equal(snap.by_type["max_pixel"].heat[1, 0], _round_trip(act[2, 1]))


def test_2d_activation_uses_whole_images_and_uniform_heat() -> None:
    acc = PatchAccumulator()
    x = _image_batch(3, side=8)
    act = torch.tensor([[1.0, -1.0], [3.0, 0.0], [2.0, 5.0]])
    acc.update(act=act, x=x, average_patches=True)
    snap = acc.snapshot()
    assert snap is not None
    for ptype in ("max_pixel", "max_average"):
        tp = snap.by_type[ptype]
        assert not tp.crop
        assert tp.values[0, :3].tolist() == [3.0, 2.0, 1.0]
        assert tp.values[1, :3].tolist() == [5.0, 0.0, -1.0]
        assert torch.equal(tp.patches[0, 0], _round_trip(x[1]))
        assert tp.heat[0, 0].shape == (1, 1)
        # A single-cell heat slice is constant, so it round-trips exactly.
        assert tp.heat[0, 0].item() == 3.0


def test_unfilled_slots_are_non_finite() -> None:
    acc = PatchAccumulator()
    acc.update(act=_flat_act([1.0, 2.0]), x=_image_batch(2))
    snap = acc.snapshot()
    assert snap is not None
    tp = snap.by_type["max_pixel"]
    vals = tp.values[0]
    assert torch.isfinite(vals[:2]).all()
    assert not torch.isfinite(vals[2:]).any()
    # Unfilled payload slots still dequantize to zeros (the renderer's mask
    # reads the values; the zeros just keep the buffers NaN-free).
    assert (tp.patches[0, 2:] == 0).all()
    assert (tp.heat[0, 2:] == 0).all()


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
    assert snap.by_type["max_pixel"].values[0, 0].item() == 1.0


def test_nan_scores_never_enter_buffers() -> None:
    acc = PatchAccumulator()
    act = _flat_act([1.0, 2.0])
    act[1] = float("nan")
    acc.update(act=act, x=_image_batch(2), average_patches=True)
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
    assert snap.by_type["max_pixel"].patches.shape[2] == 1


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
    acc.update(act=torch.randn(2, 3, 8, 8), x=x, average_patches=True)
    snap = acc.snapshot()
    assert snap is not None
    pixel = snap.by_type["max_pixel"]
    n = DEFAULT_SAMPLES_PER_CHANNEL
    assert pixel.patches.shape == (3, n, 3, 16, 16)  # ceil(4)*4
    assert pixel.heat.shape == (3, n, 8, 8)
    avg = snap.by_type["min_average"]
    assert avg.patches.shape == (3, n, 3, 32, 32)
    assert avg.input_hw == (32, 32)


def test_channel_limit_caps_recorded_channels() -> None:
    """`channel_limit` keeps only the first that-many channels' patches."""
    acc = PatchAccumulator()
    x = _image_batch(2, side=8)
    acc.update(act=torch.randn(2, 8, 4, 4), x=x, channel_limit=3, average_patches=True)
    snap = acc.snapshot()
    assert snap is not None
    for ptype in PATCH_TYPES:
        assert snap.by_type[ptype].patches.shape[0] == 3


def test_channel_limit_above_count_keeps_all() -> None:
    acc = PatchAccumulator()
    acc.update(act=torch.randn(2, 4, 4, 4), x=_image_batch(2, side=8), channel_limit=16)
    snap = acc.snapshot()
    assert snap is not None
    assert snap.by_type["max_pixel"].patches.shape[0] == 4


def test_samples_per_channel_sets_buffer_depth() -> None:
    acc = PatchAccumulator()
    acc.update(
        act=_flat_act([float(i) for i in range(6)]),
        x=_image_batch(6),
        n_per_channel=2,
        average_patches=True,
    )
    snap = acc.snapshot()
    assert snap is not None
    tp = snap.by_type["max_average"]
    assert tp.values.shape[1] == 2
    assert tp.values[0].tolist() == [5.0, 4.0]  # only the top 2 kept


@pytest.mark.parametrize(
    ("average_patches", "expected"),
    [
        (False, ("max_pixel", "min_pixel")),
        (True, PATCH_TYPES),
    ],
    ids=["default-off", "on"],
)
def test_average_patches_selects_collected_types(
    average_patches: bool, expected: tuple[PatchType, ...]
) -> None:
    acc = PatchAccumulator()
    acc.update(
        act=torch.randn(2, 2, 4, 4),
        x=_image_batch(2),
        average_patches=average_patches,
    )
    snap = acc.snapshot()
    assert snap is not None
    assert tuple(snap.by_type) == expected


def test_non_finite_heat_dequantizes_to_nan() -> None:
    """±inf in a winning activation map become NaN in the snapshot's heat;
    finite positions keep their (quantized) values."""
    acc = PatchAccumulator()
    x = _image_batch(2, side=8)
    act = torch.full((2, 1, 2, 2), 0.5)
    act[1, 0] = torch.tensor([[float("inf"), 2.0], [float("-inf"), 1.0]])
    acc.update(act=act, x=x)
    snap = acc.snapshot()
    assert snap is not None
    heat = snap.by_type["max_pixel"].heat[0, 0]  # sample 1 wins on +inf
    non_finite = torch.tensor([[True, False], [True, False]])
    assert torch.equal(torch.isnan(heat), non_finite)
    assert torch.equal(heat[~non_finite], _round_trip(act[1, 0])[~non_finite])


def test_quantize_marks_non_finite_and_round_trips_finite() -> None:
    t = torch.tensor(
        [[[1.0, float("nan"), 5.0, float("inf"), float("-inf"), 3.0]]]
    )
    q, scales = _quantize(t)
    assert q.dtype == torch.uint8
    # Offset/scale come from the finite values only.
    assert scales[0, 0, 0].item() == 1.0
    assert scales[0, 0, 1].item() == pytest.approx((5.0 - 1.0) / 254)
    # Non-finite elements carry the reserved sentinel byte.
    assert q[0, 0].tolist()[1] == 255
    assert q[0, 0].tolist()[3] == 255
    assert q[0, 0].tolist()[4] == 255
    out = _dequantize(q, scales)
    finite = torch.tensor([True, False, True, False, False, True])
    assert torch.equal(torch.isnan(out[0, 0]), ~finite)
    atol = (5.0 - 1.0) / 254 / 2 + 1e-6
    assert torch.allclose(out[0, 0][finite], t[0, 0][finite], atol=atol)


def test_quantize_all_non_finite_slot() -> None:
    t = torch.tensor([[[float("nan"), float("inf"), float("-inf")]]])
    q, scales = _quantize(t)
    assert (q == 255).all()
    assert scales[0, 0].tolist() == [0.0, 0.0]
    assert torch.isnan(_dequantize(q, scales)).all()


def test_constant_payload_round_trips_exactly() -> None:
    """A constant slice gets scale 0 and dequantizes to the offset exactly."""
    acc = PatchAccumulator()
    x = torch.full((1, 3, 8, 8), 7.25)
    act = torch.full((1, 1, 2, 2), 2.5)
    acc.update(act=act, x=x)
    snap = acc.snapshot()
    assert snap is not None
    tp = snap.by_type["max_pixel"]
    assert (tp.patches[0, 0] == 7.25).all()
    assert (tp.heat[0, 0] == 2.5).all()


def test_state_dict_round_trip_preserves_quantized_galleries() -> None:
    acc = PatchAccumulator()
    acc.update(act=torch.randn(3, 2, 4, 4), x=_image_batch(3), average_patches=True)
    state = acc.state_dict()
    # Serialized payloads are the raw uint8 buffers plus their scale rows.
    for buf in state["buffers"].values():
        assert buf["patches"].dtype == torch.uint8
        assert buf["heat"].dtype == torch.uint8
        assert buf["patch_scales"].shape[-1] == 2
        assert buf["heat_scales"].shape[-1] == 2
    restored = PatchAccumulator()
    restored.load_state_dict(state)
    left, right = acc.snapshot(), restored.snapshot()
    assert left is not None and right is not None
    assert tuple(left.by_type) == tuple(right.by_type)
    for ptype, lp in left.by_type.items():
        rp = right.by_type[ptype]
        for field in ("values", "patches", "heat", "top", "left"):
            assert torch.equal(getattr(lp, field), getattr(rp, field)), (ptype, field)
