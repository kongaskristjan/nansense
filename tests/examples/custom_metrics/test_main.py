"""Tests for the custom-instruments example: data, model, and instrument wiring."""

from __future__ import annotations

import sys

import pytest
import torch
import torch.fx

import nansense
from examples.custom_metrics import main as main_module


def test_blob_dataset_is_deterministic_and_labelled_by_quadrant() -> None:
    a = main_module.make_blob_dataset(16, seed=3)
    b = main_module.make_blob_dataset(16, seed=3)
    images, labels = a.tensors
    assert images.shape == (16, 1, 16, 16)
    assert labels.shape == (16,)
    assert torch.equal(images, b.tensors[0])
    # The brightest pixel sits in the labelled quadrant (the noise floor is
    # far below the blob peak).
    flat = images.view(16, -1).argmax(dim=1)
    rows, cols = flat // 16, flat % 16
    assert torch.equal((rows >= 8).long() * 2 + (cols >= 8).long(), labels)


def test_model_shape_and_fx_traceable() -> None:
    model = main_module.BlobNet(channels=4)
    out = model(torch.randn(2, 1, 16, 16))
    assert out.shape == (2, 4)
    # nansense traces the graph to name layers; tracing must succeed.
    torch.fx.symbolic_trace(model)


def test_default_port(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "argv", ["main.py"])
    assert main_module.parse_args().nansense_port == 8080


def test_instruments_record_through_a_training_step() -> None:
    """The example's instruments produce data on a real (unserved) session."""
    model = main_module.BlobNet(channels=4)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    session = nansense.start(
        model, epochs=1, phases={"train": 2}, optimizer=optimizer
    )
    main_module.register_instruments(session)
    session.watch("conv1")
    session.detach()

    images, labels = main_module.make_blob_dataset(8, seed=0).tensors
    for start in (0, 4):
        # Publish the last batch so the tensor instruments run too.
        if start == 4:
            session.request_snapshot()
        with session.batch(phase="train", epoch=0):
            optimizer.zero_grad(set_to_none=True)
            loss = torch.nn.functional.cross_entropy(
                model(images[start : start + 4]), labels[start : start + 4]
            )
            loss.backward()
            optimizer.step()

    plots = session.watch_metrics_snapshot().plots("conv1", "train")
    assert len(plots["sparsity"][""].values) == 2  # one point per batch
    assert plots["grad_rms"][""].batches == (None,)  # reduced per epoch
    snap = session.snapshot
    assert snap is not None
    assert snap.custom_activations["conv1"]["zscore"].shape == (
        snap.activations["conv1"].shape
    )
    # Adam state exists after the first step, so the weight tensor is there.
    assert (
        snap.custom_weight_tensors["conv1.weight"]["adam_dir"].shape
        == snap.weights["conv1.weight"].shape
    )
    assert session.instrument_errors == {}
    session.close()
