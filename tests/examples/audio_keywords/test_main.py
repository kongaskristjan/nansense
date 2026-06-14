"""Tests for the audio_keywords entrypoint helpers."""

from __future__ import annotations

import argparse

import torch

from examples.audio_keywords import main as main_module
from examples.audio_keywords.data import AudioConfig
from examples.audio_keywords.model import KeywordCNN


def test_build_model_output_shape() -> None:
    config = AudioConfig()
    model = main_module.build_model(config)
    assert isinstance(model, KeywordCNN)
    x = torch.randn(2, config.in_channels, config.n_mels, 101)
    assert model(x).shape == (2, config.num_classes)


def test_build_optimizer_and_scheduler() -> None:
    model = torch.nn.Linear(4, 2)
    args = argparse.Namespace(lr=1e-3, weight_decay=0.05, epochs=10)
    optimizer, scheduler = main_module.build_optimizer_and_scheduler(model, args)
    assert isinstance(optimizer, torch.optim.AdamW)
    assert isinstance(scheduler, torch.optim.lr_scheduler.CosineAnnealingLR)
