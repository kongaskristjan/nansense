"""Tests for the watch accumulator and its bin math."""

from __future__ import annotations

import math

import pytest
import torch

from playgrad.watch import (
    BINS_PER_DECADE,
    LOG10_MAX,
    LOG10_MIN,
    N_BINS,
    N_POS,
    ZERO_BIN,
    TensorAccumulator,
    WatchAccumulator,
    bin_index,
    bin_midpoint,
    histogram_edges,
)


def test_n_bins_matches_211() -> None:
    assert N_BINS == 211
    assert N_POS == 105
    assert ZERO_BIN == 105


def test_histogram_edges_have_correct_count_and_bounds() -> None:
    edges = histogram_edges()
    assert len(edges) == N_BINS + 1
    assert edges[0] == -(10**LOG10_MAX)
    assert edges[-1] == 10**LOG10_MAX
    # The two edges bracketing the zero bin are -1e-9 and +1e-9.
    assert edges[ZERO_BIN] == pytest.approx(-1e-9)
    assert edges[ZERO_BIN + 1] == pytest.approx(1e-9)
    # Strictly increasing.
    assert all(edges[i] < edges[i + 1] for i in range(N_BINS))


def test_powers_of_ten_are_bin_edges() -> None:
    """Every power-of-10 boundary in the range lines up with a bin edge."""
    edges = histogram_edges()
    edges_set = {round(math.log10(abs(e)), 9) for e in edges if e != 0 and abs(e) > 0}
    for k in range(LOG10_MIN, LOG10_MAX + 1):
        assert round(k, 9) in edges_set, f"10^{k} not on a bin edge"


@pytest.mark.parametrize(
    "value, expected_bin",
    [
        (0.0, ZERO_BIN),
        (1e-15, ZERO_BIN),  # below 1e-9 → zero band
        (-1e-15, ZERO_BIN),
        (1e-9, ZERO_BIN + 1),  # smallest positive bin
        (-1e-9, ZERO_BIN - 1),  # smallest negative bin
        (1.0, ZERO_BIN + 1 + BINS_PER_DECADE * 9),  # log10(1) = 0, 9 decades up from -9
        (-1.0, ZERO_BIN - 1 - BINS_PER_DECADE * 9),
        (1e6, N_BINS - 1),  # largest positive bin
        (-1e6, 0),  # largest negative bin
        (1e10, N_BINS - 1),  # overflow saturates into the top
        (-1e10, 0),  # overflow saturates into the bottom
        (float("inf"), N_BINS - 1),
        (float("-inf"), 0),
        (float("nan"), ZERO_BIN),
    ],
)
def test_bin_index(value: float, expected_bin: int) -> None:
    assert bin_index(value) == expected_bin


def test_bin_midpoint_zero_and_overflow() -> None:
    assert bin_midpoint(ZERO_BIN) == 0.0
    assert bin_midpoint(0) == -(10**LOG10_MAX)
    assert bin_midpoint(N_BINS - 1) == 10**LOG10_MAX


def test_bin_midpoint_is_geometric_mean_of_edges() -> None:
    """For a non-extreme positive bin, the midpoint is sqrt(lower * upper)."""
    edges = histogram_edges()
    # Pick a bin well inside the range.
    idx = ZERO_BIN + 50
    expected = math.sqrt(edges[idx] * edges[idx + 1])
    assert bin_midpoint(idx) == pytest.approx(expected, rel=1e-9)


def test_accumulator_starts_empty() -> None:
    snap = TensorAccumulator().snapshot()
    assert snap.n == 0
    assert snap.sum == 0.0
    assert math.isinf(snap.min) and snap.min > 0
    assert math.isinf(snap.max) and snap.max < 0
    assert math.isnan(snap.mean)
    assert math.isnan(snap.median)
    assert all(c == 0 for c in snap.hist)


def test_accumulator_aggregates_across_updates() -> None:
    acc = TensorAccumulator()
    acc.update(torch.tensor([1.0, 2.0, 3.0]))
    acc.update(torch.tensor([4.0, 5.0]))
    snap = acc.snapshot()
    assert snap.n == 5
    assert snap.sum == pytest.approx(15.0)
    assert snap.sum_sq == pytest.approx(55.0)  # 1+4+9+16+25
    assert snap.min == pytest.approx(1.0)
    assert snap.max == pytest.approx(5.0)
    assert snap.mean == pytest.approx(3.0)
    assert snap.std == pytest.approx(math.sqrt(2.0))


def test_accumulator_histogram_counts_match_input() -> None:
    """Each input value contributes exactly one count to the right bin."""
    acc = TensorAccumulator()
    values = [-1e3, -1.0, 0.0, 1.0, 1e3]
    acc.update(torch.tensor(values))
    snap = acc.snapshot()
    assert sum(snap.hist) == len(values)
    for v in values:
        assert snap.hist[bin_index(v)] >= 1


def test_accumulator_promotes_bf16_to_fp32() -> None:
    """bf16 inputs don't blow up sum_of_squares — fp32 reduces precisely."""
    acc = TensorAccumulator()
    # Many bf16 elements of 1.0 — sum_sq in bf16 would saturate early.
    x = torch.ones(10_000, dtype=torch.bfloat16)
    acc.update(x)
    snap = acc.snapshot()
    assert snap.n == 10_000
    # In bf16 this saturates around ~256; we want the true value.
    assert snap.sum_sq == pytest.approx(10_000.0, rel=1e-3)


def test_accumulator_handles_empty_input() -> None:
    acc = TensorAccumulator()
    acc.update(torch.tensor([]))
    snap = acc.snapshot()
    assert snap.n == 0


def test_overflow_values_land_in_extreme_bins() -> None:
    acc = TensorAccumulator()
    acc.update(torch.tensor([1e8, -1e8]))
    snap = acc.snapshot()
    assert snap.hist[N_BINS - 1] == 1  # extreme positive
    assert snap.hist[0] == 1  # extreme negative


def test_watch_accumulator_separates_layers_and_phases_and_epochs() -> None:
    acc = WatchAccumulator()
    acc.update(layer="a", phase="train", epoch=0, kind="activation", x=torch.tensor([1.0]))
    acc.update(layer="a", phase="train", epoch=0, kind="gradient", x=torch.tensor([0.1]))
    acc.update(layer="a", phase="val", epoch=0, kind="activation", x=torch.tensor([2.0]))
    acc.update(layer="b", phase="train", epoch=0, kind="activation", x=torch.tensor([3.0]))
    acc.update(layer="a", phase="train", epoch=1, kind="activation", x=torch.tensor([4.0]))

    snap = acc.snapshot()
    assert set(snap.stats) == {
        ("a", "train", 0),
        ("a", "val", 0),
        ("b", "train", 0),
        ("a", "train", 1),
    }
    assert snap.stats[("a", "train", 0)].activations.sum == pytest.approx(1.0)
    assert snap.stats[("a", "train", 0)].gradients.sum == pytest.approx(0.1)


def test_watch_accumulator_latest_per_phase_picks_max_epoch() -> None:
    acc = WatchAccumulator()
    acc.update(layer="a", phase="train", epoch=0, kind="activation", x=torch.tensor([1.0]))
    acc.update(layer="a", phase="train", epoch=2, kind="activation", x=torch.tensor([2.0]))
    acc.update(layer="a", phase="train", epoch=1, kind="activation", x=torch.tensor([3.0]))
    acc.update(layer="a", phase="val", epoch=0, kind="activation", x=torch.tensor([4.0]))

    latest = acc.snapshot().latest_per_phase("a")
    assert latest["train"].epoch == 2
    assert latest["train"].activations.sum == pytest.approx(2.0)
    assert latest["val"].epoch == 0


def test_watch_accumulator_forget_layer() -> None:
    acc = WatchAccumulator()
    acc.update(layer="a", phase="train", epoch=0, kind="activation", x=torch.tensor([1.0]))
    acc.update(layer="b", phase="train", epoch=0, kind="activation", x=torch.tensor([2.0]))
    acc.forget_layer("a")
    snap = acc.snapshot()
    assert ("a", "train", 0) not in snap.stats
    assert ("b", "train", 0) in snap.stats


def test_watch_accumulator_snapshot_filters_to_requested_layers() -> None:
    acc = WatchAccumulator()
    acc.update(layer="a", phase="train", epoch=0, kind="activation", x=torch.tensor([1.0]))
    acc.update(layer="b", phase="train", epoch=0, kind="activation", x=torch.tensor([2.0]))
    snap = acc.snapshot(layers=["a"])
    assert set(snap.stats) == {("a", "train", 0)}


def test_snapshot_median_is_histogram_midpoint() -> None:
    acc = TensorAccumulator()
    acc.update(torch.tensor([0.5, 0.5, 0.5, 0.5, 0.5]))
    snap = acc.snapshot()
    # All five samples land in the same bin; the median is its midpoint.
    median_bin = bin_index(0.5)
    assert snap.median == pytest.approx(bin_midpoint(median_bin))


def test_watch_accumulator_forget_epochs_from() -> None:
    acc = WatchAccumulator()
    for epoch in range(3):
        acc.update(layer="a", phase="train", epoch=epoch, kind="activation", x=torch.tensor([1.0]))
    acc.update(layer="b", phase="val", epoch=2, kind="activation", x=torch.tensor([2.0]))
    acc.forget_epochs_from(1)
    snap = acc.snapshot()
    assert set(snap.stats) == {("a", "train", 0)}
