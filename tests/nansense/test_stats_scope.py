"""Tests for the three-way stats scope (`Session.set_stats_scope`).

`"watched"` (default) collects for the watched layers only, `"all"` for every
layer regardless of the watched set, and `"none"` pauses collection while
keeping every already-collected bucket frozen. Narrowing to `"watched"` drops
the buckets of layers outside the watched set.
"""

from __future__ import annotations

import pytest

from nansense.session import StatsScope
from tests.nansense.helpers import make_session, train_step

# TinyNet's fx-traced layer names, in graph order.
_TINYNET_LAYERS = ["x", "fc1", "relu", "fc2"]


def test_scope_all_collects_every_layer_without_watching() -> None:
    session, model = make_session(epochs=1, phases={"train": 2})
    session.set_stats_scope("all")
    session.detach()
    with session.batch(phase="train", epoch=0):
        train_step(model)
    stats = session.watch_snapshot(include_patches=False).stats
    assert {key[0] for key in stats} == set(_TINYNET_LAYERS)
    assert session.watched_layers == frozenset()
    assert session.stats_layers == frozenset(_TINYNET_LAYERS)


def test_scope_none_freezes_buckets_and_keeps_them_browsable() -> None:
    session, model = make_session(epochs=1, phases={"train": 4})
    session.watch("fc1")
    session.detach()
    with session.batch(phase="train", epoch=0):
        train_step(model)
    key = ("fc1", "train", 0)
    assert session.watch_snapshot().stats[key].activations.n == 16

    session.set_stats_scope("none")
    for _ in range(2):
        with session.batch(phase="train", epoch=0):
            train_step(model)
    # Frozen: nothing accumulated, nothing dropped — and the layer stays
    # browsable on the stats page while paused.
    assert session.watch_snapshot().stats[key].activations.n == 16
    assert "fc1" in session.stats_layers

    session.set_stats_scope("watched")
    with session.batch(phase="train", epoch=0):
        train_step(model)
    assert session.watch_snapshot().stats[key].activations.n == 32


def test_narrowing_scope_to_watched_prunes_other_buckets() -> None:
    session, model = make_session(epochs=1, phases={"train": 2})
    session.set_stats_scope("all")
    session.detach()
    with session.batch(phase="train", epoch=0):
        train_step(model)
    assert len(session.watch_snapshot().stats) == len(_TINYNET_LAYERS)

    session.watch("fc1")
    session.set_stats_scope("watched")
    snap = session.watch_snapshot()
    assert {key[0] for key in snap.stats} == {"fc1"}
    assert {key[0] for key in snap.weights} == {"fc1"}
    assert session.stats_layers == frozenset({"fc1"})


def test_toggle_restores_the_previous_scope() -> None:
    session, _model = make_session()
    session.set_stats_scope(StatsScope.ALL)
    assert session.toggle_stats_collecting() is False
    assert session.stats_scope is StatsScope.NONE
    assert session.toggle_stats_collecting() is True
    assert session.stats_scope is StatsScope.ALL


@pytest.mark.parametrize(
    ("scope", "watched", "expected"),
    [
        (StatsScope.NONE, ["fc1"], frozenset()),
        (StatsScope.WATCHED, ["fc1", "relu"], frozenset({"fc1", "relu"})),
        (StatsScope.ALL, [], frozenset(_TINYNET_LAYERS)),
    ],
)
def test_stats_layers_reflects_the_scope(
    scope: StatsScope, watched: list[str], expected: frozenset[str]
) -> None:
    session, _model = make_session()
    for name in watched:
        assert session.watch(name)
    session.set_stats_scope(scope)
    assert session.stats_layers == expected


def test_unknown_scope_raises() -> None:
    session, _model = make_session()
    with pytest.raises(ValueError):
        session.set_stats_scope("everything")
