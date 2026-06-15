"""Tests for watch-stats accumulation alongside the capture machinery."""

from __future__ import annotations

import pytest
import torch
from torch import Tensor, nn

import nansense
from tests.nansense.helpers import make_session, paused_worker, train_step


def test_watch_accepts_any_layer_name_and_rejects_unknown() -> None:
    session, model = make_session()
    # Modules, fx intermediates, and the input are all in layer_names.
    assert session.watch("fc1") is True
    assert session.watch("relu") is True  # fx intermediate
    assert session.watch("x") is True  # graph input
    assert session.watch("bogus") is False
    assert session.watched_layers == frozenset({"fc1", "relu", "x"})


def test_watch_accumulates_stats_while_detached() -> None:
    """Stats accumulate on every batch even when detach() means no captures."""
    session, model = make_session(epochs=1, phases={"train": 3})
    session.watch("fc1")
    session.detach()

    for _ in range(3):
        with session.batch(phase="train", epoch=0) as ctx:
            train_step(model)
            assert ctx.captured is False  # detach mode

    snap = session.watch_snapshot()
    assert ("fc1", "train", 0) in snap.stats
    layer_stats = snap.stats[("fc1", "train", 0)]
    # Three forward passes × 2 samples × 8 output features = 48 elements.
    assert layer_stats.activations.n == 48
    # Three backward passes' gradients aggregated too.
    assert layer_stats.gradients.n == 48


def test_watch_accumulates_stats_alongside_capture() -> None:
    """Stats also accumulate when the batch is being captured for the snapshot."""
    session, model = make_session(epochs=1, phases={"train": 1})
    session.watch("fc1")

    def loop() -> None:
        with session.batch(phase="train", epoch=0):
            train_step(model)

    with paused_worker(session, loop):
        snap = session.watch_snapshot()
        assert ("fc1", "train", 0) in snap.stats
        assert snap.stats[("fc1", "train", 0)].activations.n == 16


def test_unwatch_drops_collected_stats() -> None:
    session, model = make_session(epochs=1, phases={"train": 1})
    session.watch("fc1")
    session.detach()
    with session.batch(phase="train", epoch=0):
        train_step(model)
    assert ("fc1", "train", 0) in session.watch_snapshot().stats

    session.unwatch("fc1")
    assert session.watched_layers == frozenset()
    assert session.watch_snapshot().stats == {}


class TinyConvNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(3, 2, kernel_size=3, padding=1)
        self.fc = nn.Linear(2 * 8 * 8, 3)

    def forward(self, x: Tensor) -> Tensor:
        return self.fc(torch.relu(self.conv(x)).flatten(1))


def test_watch_gathers_patches_for_image_inputs() -> None:
    model = TinyConvNet()
    session = nansense.start(model, epochs=1, phases={"train": 2})
    session.watch("conv")
    session.detach()

    for _ in range(2):
        with session.batch(phase="train", epoch=0):
            x = torch.randn(2, 3, 8, 8)
            y = torch.randint(0, 3, (2,))
            model.zero_grad(set_to_none=True)
            nn.functional.cross_entropy(model(x), y).backward()

    patches = session.watch_snapshot().stats[("conv", "train", 0)].patches
    assert patches is not None
    tp = patches.by_type["max_pixel"]
    assert tp.values.shape[0] == 2  # one row per conv channel
    # Two batches × 2 samples = 4 candidates per channel made it in.
    assert torch.isfinite(tp.values[:, :4]).all()


def test_watch_skips_patches_without_image_input() -> None:
    session, model = make_session(epochs=1, phases={"train": 1})
    session.watch("fc1")
    session.detach()
    with session.batch(phase="train", epoch=0):
        train_step(model)
    # TinyNet's input is 2D — stats accumulate but no patches are gathered.
    layer_stats = session.watch_snapshot().stats[("fc1", "train", 0)]
    assert layer_stats.activations.n > 0
    assert layer_stats.patches is None


def test_watching_uses_full_capture_machinery_under_detach() -> None:
    """Watching engages the same hook path as capture, so fx intermediates work."""
    session, model = make_session(epochs=1, phases={"train": 1})
    session.watch("fc1")
    session.detach()

    with session.batch(phase="train", epoch=0):
        # The exact handle count depends on fx-vs-hook mode; what matters
        # is that *something* got installed (capture machinery is live).
        # TinyNet (no Module-only ops between modules) traces cleanly, so
        # fx mode patches forward without registering RemovableHandles
        # — `_original_forward` is the signal there.
        installed = (
            len(session._hook_handles) > 0  # type: ignore[reportPrivateUsage]
            or session._original_forward is not None  # type: ignore[reportPrivateUsage]
        )
        assert installed
        train_step(model)
    assert session._hook_handles == []  # type: ignore[reportPrivateUsage]
    assert session._original_forward is None  # type: ignore[reportPrivateUsage]


def test_watch_fx_intermediate_accumulates_stats() -> None:
    """Watching an fx-traced intermediate op (`relu`) produces stats."""
    session, model = make_session(epochs=1, phases={"train": 1})
    assert session.fx_traced
    assert "relu" in session.layer_names
    session.watch("relu")
    session.detach()

    with session.batch(phase="train", epoch=0):
        train_step(model)

    snap = session.watch_snapshot()
    assert ("relu", "train", 0) in snap.stats
    relu_stats = snap.stats[("relu", "train", 0)].activations
    # ReLU output is non-negative — the histogram's negative half is empty.
    from nansense.watch import ZERO_BIN
    neg_count = sum(relu_stats.hist[:ZERO_BIN])
    assert neg_count == 0
    assert relu_stats.n == 16  # batch 2 × 8 hidden features


def test_watch_input_x_accumulates_stats() -> None:
    """Watching the graph input `x` produces stats."""
    session, model = make_session(epochs=1, phases={"train": 1})
    assert "x" in session.layer_names
    session.watch("x")
    session.detach()

    with session.batch(phase="train", epoch=0):
        train_step(model)

    snap = session.watch_snapshot()
    assert ("x", "train", 0) in snap.stats
    x_stats = snap.stats[("x", "train", 0)].activations
    assert x_stats.n == 8  # batch 2 × 4 input features


def test_watch_stats_failure_still_removes_hooks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A raise inside the watch-stats update must not leak the patched forward.

    `_update_watch_stats` runs at `__exit__` before hook removal; if it raises
    and removal is skipped, the next `install_hooks` would capture the leaked
    fx_forward as the "original" and permanently lose the real forward. The
    try/finally in `__exit__` keeps `remove_hooks` running regardless.
    """
    session, model = make_session(epochs=1, phases={"train": 2})
    assert session.fx_traced  # the leak only corrupts the fx patch path
    session.watch("fc1")
    session.detach()
    # The traced model patches its forward via an instance attribute; the real
    # forward has no entry in the instance __dict__.
    assert "forward" not in model.__dict__
    baseline = model(torch.zeros(1, 4)).detach().clone()

    # Make the accumulator's update raise on the first batch's stats pass.
    def boom(**_kwargs: object) -> None:
        raise RuntimeError("stats blew up")

    monkeypatch.setattr(session._watch_accumulator, "update", boom)

    with pytest.raises(RuntimeError, match="stats blew up"):
        with session.batch(phase="train", epoch=0):
            train_step(model)

    # Hooks were torn down despite the raise: the fx patch is gone and forward
    # resolves to the real one (no leaked patch left on the instance).
    assert session._hook_handles == []  # type: ignore[reportPrivateUsage]
    assert session._original_forward is None  # type: ignore[reportPrivateUsage]
    assert "forward" not in model.__dict__
    torch.testing.assert_close(model(torch.zeros(1, 4)), baseline)

    # The next batch reinstalls cleanly — no stale patch became the "original".
    monkeypatch.undo()
    with session.batch(phase="train", epoch=0):
        train_step(model)
    assert session._hook_handles == []  # type: ignore[reportPrivateUsage]
    assert session._original_forward is None  # type: ignore[reportPrivateUsage]
    assert "forward" not in model.__dict__
    torch.testing.assert_close(model(torch.zeros(1, 4)), baseline)
    snap = session.watch_snapshot()
    # Only the second batch's stats landed (the first raised before updating):
    # one batch × 2 samples × 8 fc1 features.
    assert snap.stats[("fc1", "train", 0)].activations.n == 16
