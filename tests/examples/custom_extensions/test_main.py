"""Tests for the custom-extensions example: its experiment on a paused run."""

from __future__ import annotations

import sys

import pytest
import torch

import nansense
from examples.custom_extensions import main as main_module
from tests.nansense.helpers import paused_worker


def test_default_port(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "argv", ["main.py"])
    assert main_module.parse_args().nansense_port == 8080


def test_channel_ablation_tabulates_every_channel_of_the_layer() -> None:
    model = main_module.BlobNet(channels=4)
    session = nansense.start(model, epochs=1, phases={"train": 2})
    main_module.register_extensions(session)
    images, labels = main_module.make_blob_dataset(8, seed=0).tensors
    items = [(images[:4], labels[:4]), (images[4:], labels[4:])]

    def loop() -> None:
        for x, y in session.batches(items, phase="train", epoch=0):
            torch.nn.functional.cross_entropy(model(x), y).backward()

    with paused_worker(session, loop):
        seq = session.request_experiment(
            kind="ablate_channels", layer="conv1", params={"top": 3}
        )
        assert session.wait_for_experiment(timeout=10.0)
        result = session.experiment_result_for(seq)
    assert result is not None and result.error is None
    assert result.table is not None and len(result.table.rows) == 3
    assert result.scalars["channels"] == 4
    drops = [row[1] for row in result.table.rows]
    assert drops == sorted(drops, reverse=True)
