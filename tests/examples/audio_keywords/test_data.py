"""Tests for the audio_keywords log-mel front end and dataset (no network)."""

from __future__ import annotations

import math

import pytest
import torch

from examples.audio_keywords.data import (
    KEYWORDS,
    AudioConfig,
    LogMelTransform,
    fix_length,
)


def test_keyword_classes() -> None:
    assert KEYWORDS == ("down", "go", "left", "no", "right", "stop", "up", "yes")
    assert AudioConfig().num_classes == len(KEYWORDS) == 8


@pytest.mark.parametrize("n_mels", [20, 40, 64])
def test_log_mel_transform_shape_and_finite(n_mels: int) -> None:
    """Run the torchaudio transform on a synthetic waveform (no disk read) and
    check the `[1, n_mels, n_frames]` shape and that every value is finite."""
    config = AudioConfig(n_mels=n_mels)
    transform = LogMelTransform(config)

    t = torch.arange(config.clip_length, dtype=torch.float32) / config.sample_rate
    waveform = 0.5 * torch.sin(2 * math.pi * 440.0 * t)  # a 440 Hz tone

    spectrogram = transform(waveform)

    expected_frames = config.clip_length // config.hop_length + 1  # center=True padding
    assert spectrogram.shape == (1, n_mels, expected_frames)
    assert torch.isfinite(spectrogram).all()


def test_log_mel_silence_is_log_eps_floor() -> None:
    """Silence must map to the log floor everywhere (no NaNs/-inf from log(0))."""
    config = AudioConfig()
    transform = LogMelTransform(config)
    spectrogram = transform(torch.zeros(config.clip_length))
    assert torch.isfinite(spectrogram).all()
    assert torch.allclose(spectrogram, torch.full_like(spectrogram, math.log(config.log_eps)))


@pytest.mark.parametrize(
    ("length", "target", "expected"),
    [(8, 16, 16), (32, 16, 16), (16, 16, 16)],
)
def test_fix_length_pads_or_truncates(length: int, target: int, expected: int) -> None:
    out = fix_length(torch.ones(length), target)
    assert out.numel() == expected
    if length < target:  # padding is zeros on the right
        assert torch.all(out[length:] == 0.0)
