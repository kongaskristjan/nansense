"""Tests for the "Current batch" stats path and the stats-collection toggle.

`single_batch_stats` / `Session.current_batch_stats` compute stats directly
from the last published snapshot (so any layer works, watched or not), and
the `"none"` stats scope pauses/resumes the running watch aggregates without
hiding the watched cards (see `test_stats_scope` for the scope semantics).
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

import nansense
from nansense.watch import single_batch_stats
from tests.nansense.helpers import make_session, paused_worker, train_step


def test_single_batch_stats_matches_tensor_reductions() -> None:
    act = torch.tensor([[1.0, -2.0, 0.0], [3.0, 4.0, -5.0]])
    out = single_batch_stats(
        layer="L",
        phase="train",
        epoch=2,
        activation=act,
        gradient=None,
        patch_source=None,
        channel_limit=None,
        samples_per_channel=5,
        include_patches=False,
    )
    assert (out.layer, out.phase, out.epoch) == ("L", "train", 2)
    assert out.activations.n == 6
    assert out.activations.min == -5.0
    assert out.activations.max == 4.0
    # No gradient passed → an empty accumulator (n == 0), not a crash.
    assert out.gradients.n == 0
    assert out.patches is None


def test_single_batch_stats_gathers_patches_for_image_input() -> None:
    act = torch.rand(2, 4, 4, 4)  # (batch, channels, h, w)
    x = torch.rand(2, 3, 8, 8)
    out = single_batch_stats(
        layer="conv",
        phase="train",
        epoch=0,
        activation=act,
        gradient=None,
        patch_source=x,
        channel_limit=None,
        samples_per_channel=5,
        include_patches=True,
    )
    assert out.patches is not None
    tp = out.patches.by_type["max_pixel"]
    assert tp.values.shape[0] == 4  # one row per channel
    # include_patches=False skips the patch buffers entirely.
    no_patches = single_batch_stats(
        layer="conv",
        phase="train",
        epoch=0,
        activation=act,
        gradient=None,
        patch_source=x,
        channel_limit=None,
        samples_per_channel=5,
        include_patches=False,
    )
    assert no_patches.patches is None


def test_current_batch_stats_works_for_unwatched_layer() -> None:
    """The snapshot covers every layer, so current-batch stats need no watch."""
    session, model = make_session(epochs=1, phases={"train": 1})

    def loop() -> None:
        with session.batch(phase="train", epoch=0):
            train_step(model)

    with paused_worker(session, loop):
        # fc1 was never watched; current_batch_stats still produces its stats.
        assert session.watched_layers == frozenset()
        snap = session.current_batch_stats(layers=["fc1"])
        stats = snap.stats[("fc1", "train", 0)]
        assert stats.activations.n == 16  # 2 samples × 8 features
        assert stats.gradients.n == 16
        # The running watch aggregates remain empty — only this path sees it.
        assert session.watch_snapshot().stats == {}


def test_current_batch_stats_empty_before_any_batch() -> None:
    session, _model = make_session(epochs=1, phases={"train": 1})
    assert session.snapshot is None
    assert session.current_batch_stats(layers=["fc1"]).stats == {}


def test_stats_collecting_toggle_pauses_and_resumes() -> None:
    session, model = make_session(epochs=1, phases={"train": 3})
    session.watch("fc1")
    session.set_stats_scope("none")
    assert session.stats_collecting is False
    session.detach()

    # While paused, stepped batches accumulate nothing.
    for _ in range(2):
        with session.batch(phase="train", epoch=0):
            train_step(model)
    assert session.watch_snapshot().stats == {}

    # Resume — subsequent batches feed the running aggregate again.
    assert session.toggle_stats_collecting() is True
    with session.batch(phase="train", epoch=0):
        train_step(model)
    snap = session.watch_snapshot()
    # Only the one batch after re-enabling landed.
    assert snap.stats[("fc1", "train", 0)].activations.n == 16


def test_stats_toggle_off_still_publishes_snapshot() -> None:
    """Watched cards stay live (snapshot publishes) even with collection off."""
    session, model = make_session(epochs=1, phases={"train": 1})
    session.watch("fc1")
    session.set_stats_scope("none")

    def loop() -> None:
        with session.batch(phase="train", epoch=0):
            train_step(model)

    with paused_worker(session, loop):
        # The snapshot the main view renders from is still published ...
        assert session.snapshot is not None
        assert "fc1" in session.snapshot.activations
        # ... while no running stats were collected ...
        assert session.watch_snapshot().stats == {}
        # ... and the Current batch view still works off that snapshot.
        cur = session.current_batch_stats(layers=["fc1"])
        assert cur.stats[("fc1", "train", 0)].activations.n == 16


class _ConvNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(3, 2, kernel_size=3, padding=1)
        self.fc = nn.Linear(2 * 8 * 8, 3)

    def forward(self, x: Tensor) -> Tensor:
        return self.fc(torch.relu(self.conv(x)).flatten(1))


def test_current_batch_stats_gathers_patches_for_image_model() -> None:
    model = _ConvNet()
    session = nansense.start(model, epochs=1, phases={"train": 1})

    def loop() -> None:
        with session.batch(phase="train", epoch=0):
            x = torch.randn(2, 3, 8, 8)
            y = torch.randint(0, 3, (2,))
            model.zero_grad(set_to_none=True)
            nn.functional.cross_entropy(model(x), y).backward()

    with paused_worker(session, loop):
        snap = session.current_batch_stats(layers=["conv"], include_patches=True)
        patches = snap.stats[("conv", "train", 0)].patches
        assert patches is not None
        assert patches.by_type["max_pixel"].values.shape[0] == 2  # conv channels
