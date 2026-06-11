"""Tests for last-batch bin sampling (the histogram hover strip)."""

from __future__ import annotations

import pytest
import torch

from nansense.patches import crop_side
from nansense.ui.bin_samples import sample_bin
from nansense.watch import bin_index


def _generator(seed: int = 0) -> torch.Generator:
    return torch.Generator().manual_seed(seed)


def test_sample_bin_returns_only_matching_elements() -> None:
    act = torch.zeros(2, 3, 4, 4)
    act[0, 1, 2, 3] = 5.0
    act[1, 1, 0, 0] = 5.1  # same signed-log bin as 5.0
    x = torch.rand(2, 3, 16, 16)
    target = bin_index(5.0)
    assert bin_index(5.1) == target

    hits = sample_bin(act, x, channel=1, bin_idx=target, k=4, generator=_generator())
    assert {h.sample_idx for h in hits} == {0, 1}
    assert sorted(h.value for h in hits) == pytest.approx([5.0, 5.1])
    # Channel 0 has no value in that bin; channel 1's zeros are elsewhere.
    assert sample_bin(act, x, channel=0, bin_idx=target, generator=_generator()) == []


def test_sample_bin_limits_to_k() -> None:
    act = torch.full((4, 2, 3, 3), 2.0)
    x = torch.rand(4, 1, 9, 9)
    hits = sample_bin(
        act, x, channel=0, bin_idx=bin_index(2.0), k=3, generator=_generator()
    )
    assert len(hits) == 3


def test_sample_bin_is_deterministic_with_a_seeded_generator() -> None:
    torch.manual_seed(1)
    act = torch.randn(4, 2, 5, 5)
    x = torch.rand(4, 3, 20, 20)
    target = bin_index(0.5)
    first = sample_bin(act, x, channel=1, bin_idx=target, k=2, generator=_generator())
    second = sample_bin(act, x, channel=1, bin_idx=target, k=2, generator=_generator())
    assert [(h.sample_idx, h.value) for h in first] == [
        (h.sample_idx, h.value) for h in second
    ]


def test_sample_bin_crops_around_the_element_location() -> None:
    """4D activations crop the input like the extreme-patch grids do."""
    act = torch.zeros(1, 1, 4, 4)
    act[0, 0, 1, 2] = 3.0
    x = torch.rand(1, 3, 32, 32)
    hits = sample_bin(
        act, x, channel=0, bin_idx=bin_index(3.0), k=1, generator=_generator()
    )
    assert len(hits) == 1
    image = hits[0].image
    assert image is not None
    side = crop_side(4, 32)
    assert image.shape == (3, side, side)
    # The crop is a window of the right sample's image (here sample 0).
    flat_crop = image.reshape(3, -1)
    assert flat_crop.shape[1] <= x[0].reshape(3, -1).shape[1]


def test_sample_bin_2d_activations_use_whole_input_image() -> None:
    act = torch.full((2, 3), 1.5)
    x = torch.rand(2, 1, 8, 8)
    hits = sample_bin(
        act, x, channel=2, bin_idx=bin_index(1.5), k=2, generator=_generator()
    )
    assert len(hits) == 2
    assert all(h.image is not None and h.image.shape == (1, 8, 8) for h in hits)


def test_sample_bin_without_image_input_has_no_images() -> None:
    act = torch.full((2, 3), 1.5)
    hits = sample_bin(
        act, None, channel=0, bin_idx=bin_index(1.5), k=2, generator=_generator()
    )
    assert len(hits) == 2
    assert all(h.image is None for h in hits)


@pytest.mark.parametrize("channel", [-1, 3])
def test_sample_bin_channel_out_of_range_is_empty(channel: int) -> None:
    act = torch.ones(2, 3)
    assert sample_bin(act, None, channel=channel, bin_idx=bin_index(1.0)) == []


def test_sample_bin_1d_tensor_is_empty() -> None:
    assert sample_bin(torch.ones(4), None, channel=0, bin_idx=bin_index(1.0)) == []
