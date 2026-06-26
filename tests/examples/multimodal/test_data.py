"""Tests for the multimodal example's per-image feature derivation."""

from __future__ import annotations

import torch

from examples.multimodal.data import (
    IMAGE_CHANNELS,
    RGB_MEAN,
    RGB_STD,
    STATS_DIM,
    to_multimodal,
)


def test_to_multimodal_shapes() -> None:
    image, stats = to_multimodal(torch.rand(3, 32, 32))
    assert image.shape == (IMAGE_CHANNELS, 32, 32)
    assert stats.shape == (STATS_DIM,)


def test_rgb_channels_are_normalized_and_recoverable() -> None:
    rgb01 = torch.rand(3, 16, 16)
    image, _ = to_multimodal(rgb01)
    mean = torch.tensor(RGB_MEAN).view(3, 1, 1)
    std = torch.tensor(RGB_STD).view(3, 1, 1)
    # The first three channels are the normalized RGB; un-normalizing recovers it.
    torch.testing.assert_close(image[:3] * std + mean, rgb01)


def test_stats_first_three_are_rgb_means() -> None:
    rgb01 = torch.rand(3, 16, 16)
    _, stats = to_multimodal(rgb01)
    torch.testing.assert_close(stats[:3], rgb01.mean(dim=(1, 2)))


def test_edge_channel_responds_to_edges() -> None:
    # A flat image has no edges; a sharp left/right split has a strong one.
    flat = torch.full((3, 16, 16), 0.5)
    split = torch.zeros(3, 16, 16)
    split[:, :, 8:] = 1.0
    flat_edges = to_multimodal(flat)[0][4]
    split_edges = to_multimodal(split)[0][4]
    assert float(flat_edges.abs().max()) < 1e-5
    assert float(split_edges.max()) > 1.0
