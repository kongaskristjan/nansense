"""Tests for the value-vs-epoch series and figures in nansense.ui.epoch_stats."""

from __future__ import annotations

import pytest
import torch

from nansense.ui.epoch_stats import (
    DEAD_SERIES,
    EPOCH_STAT_SPECS,
    epoch_axis_dtick,
    epoch_stat_series,
    make_epoch_stats_figure,
    weight_stat_series,
)
from nansense.watch import (
    Kind,
    LayerStatsSnapshot,
    TensorAccumulator,
    WatchAccumulator,
)


def _history(
    *epoch_tensors: torch.Tensor,
    kinds: tuple[Kind, ...] = ("activation", "gradient"),
) -> list[LayerStatsSnapshot]:
    """One layer's train history with one update per epoch per kind."""
    acc = WatchAccumulator()
    for epoch, x in enumerate(epoch_tensors):
        for kind in kinds:
            acc.update(layer="a", phase="train", epoch=epoch, kind=kind, x=x)
    snap = acc.snapshot(include_patches=False)
    return snap.phase_history("a", "train")


def test_epoch_stat_series_values_track_epochs() -> None:
    history = _history(
        torch.tensor([[1.0, 3.0]]), torch.tensor([[2.0, 6.0]])
    )
    epochs, series = epoch_stat_series(history, "activation")
    assert epochs == [0, 1]
    assert list(series) == [s for s, _ in EPOCH_STAT_SPECS] + [DEAD_SERIES]
    assert series["mean"] == [2.0, 4.0]
    assert series["min"] == [1.0, 2.0]
    assert series["max"] == [3.0, 6.0]


def test_epoch_stat_series_dead_channels_for_activations_only() -> None:
    # Channel 0 all-zero in epoch 0, alive in epoch 1; epoch 1's arrival
    # collapses epoch 0's per-channel buffer but keeps its dead count.
    history = _history(
        torch.tensor([[0.0, 1.0], [0.0, 2.0]]), torch.ones(2, 2)
    )
    _, act = epoch_stat_series(history, "activation")
    assert act[DEAD_SERIES] == [1.0, 0.0]
    _, grad = epoch_stat_series(history, "gradient")
    assert DEAD_SERIES not in grad


def test_epoch_stat_series_empty_stream_yields_gaps() -> None:
    # No gradient updates at all: the nan/±inf stats become `None` gaps
    # (also what keeps the JSON pushed to the client valid).
    history = _history(torch.ones(2, 2), kinds=("activation",))
    _, series = epoch_stat_series(history, "gradient")
    assert series["mean"] == [None]
    assert series["min"] == [None] and series["max"] == [None]
    # `std` in particular: the snapshot property falls back to 0.0 on an
    # empty stream, which must gap here rather than plot a flat zero.
    assert series["std"] == [None]


@pytest.mark.parametrize(
    ("epochs", "expected"),
    [
        ([], 1),
        ([0], 1),
        ([0, 1, 2], 1),
        (list(range(9)), 1),
        (list(range(81)), 10),
    ],
)
def test_epoch_axis_dtick_stays_integral(
    epochs: list[int], expected: int
) -> None:
    assert epoch_axis_dtick(epochs) == expected


def test_make_epoch_stats_figure_trace_sets() -> None:
    stat_names = [s for s, _ in EPOCH_STAT_SPECS]
    act = make_epoch_stats_figure("activation", "t")
    assert [t.name for t in act.data] == stat_names + [DEAD_SERIES]
    # Dead channels ride a secondary count axis so they can't dwarf small
    # activation values; every value stat stays on the shared value axis.
    assert act.data[-1].yaxis == "y2"
    assert all(t.yaxis is None for t in act.data[:-1])
    assert act.layout.yaxis2.overlaying == "y"
    grad = make_epoch_stats_figure("gradient", "t")
    assert [t.name for t in grad.data] == stat_names
    assert all(t.yaxis is None for t in grad.data)


@pytest.mark.parametrize("kind", ["activation", "gradient"])
def test_make_epoch_stats_figure_enables_only_mean(kind: str) -> None:
    # Everything but the mean starts legend-deselected; a legend click
    # enables a stat, and refreshes never write `visible`, so it sticks.
    fig = make_epoch_stats_figure(kind, "t")
    by_name = {t.name: t.visible for t in fig.data}
    assert by_name.pop("mean") in (True, None)
    assert all(v == "legendonly" for v in by_name.values())


def _weight_stats(*values: float):
    acc = TensorAccumulator()
    acc.update(torch.tensor(list(values)))
    return acc.snapshot()


def test_weight_stat_series_matches_figure_trace_order() -> None:
    history = [(0, _weight_stats(1.0, 3.0)), (2, _weight_stats(3.0, 5.0))]
    epochs, series = weight_stat_series(history)
    assert epochs == [0, 2]
    # Same value stats as the activation/gradient series, no dead channels
    # — the shape the "weight"-kind figure's traces expect.
    assert list(series) == [s for s, _ in EPOCH_STAT_SPECS]
    assert series["mean"] == [2.0, 4.0]
    assert series["min"] == [1.0, 3.0]
    assert series["max"] == [3.0, 5.0]
