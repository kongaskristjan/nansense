"""Tests for the watch accumulator and its bin math."""

from __future__ import annotations

import math

import pytest
import torch

from nansense.watch import (
    BINS_PER_DECADE,
    LOG10_MAX,
    LOG10_MIN,
    N_BINS,
    N_POS,
    ZERO_BIN,
    TensorAccumulator,
    WatchAccumulator,
    _bin_indices,
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


_BIN_INDEX_VALUES: list[float] = [
    0.0,
    1e-15,
    -1e-15,
    1e-9,
    -1e-9,
    1.0,
    -1.0,
    1e6,
    -1e6,
    1e10,
    -1e10,
    float("nan"),
    float("inf"),
    float("-inf"),
]


@pytest.mark.parametrize("value", _BIN_INDEX_VALUES)
def test_bin_indices_matches_scalar_bin_index(value: float) -> None:
    """The vectorised binning agrees with the scalar reference everywhere,
    including the non-finite cases (nan → zero bin, ±inf → overflow ends)."""
    idx = _bin_indices(torch.tensor([value], dtype=torch.float32))
    assert int(idx[0]) == bin_index(value)


def test_bin_indices_mixed_tensor_matches_scalar() -> None:
    """A tensor mixing finite and non-finite values bins element-wise like
    the scalar reference — in particular nan/+inf/-inf are not misbinned into
    the near-zero bins."""
    values = _BIN_INDEX_VALUES
    idx = _bin_indices(torch.tensor(values, dtype=torch.float32)).tolist()
    assert idx == [bin_index(v) for v in values]
    # The specific regression: nan and ±inf must land in 105/210/0, not the
    # 104/106 bins flanking zero.
    assert idx[-3:] == [ZERO_BIN, N_BINS - 1, 0]


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


def test_diverged_activation_lands_in_overflow_and_zero_bins() -> None:
    """A diverged activation (inf and nan) must spike the overflow/zero bins,
    not the near-zero bins — the whole point of a tool named 'nansense'."""
    acc = TensorAccumulator()
    acc.update(torch.tensor([float("inf"), float("-inf"), float("nan")]))
    snap = acc.snapshot()
    assert snap.hist[N_BINS - 1] == 1  # +inf → top overflow
    assert snap.hist[0] == 1  # -inf → bottom overflow
    assert snap.hist[ZERO_BIN] == 1  # nan → zero band
    # Nothing leaked into the bins flanking zero (the pre-fix misbinning).
    assert snap.hist[ZERO_BIN - 1] == 0
    assert snap.hist[ZERO_BIN + 1] == 0


def test_non_finite_values_do_not_poison_scalar_stats() -> None:
    """A NaN or inf must not poison min/max/sum for good — scalar reductions
    run over the finite values only, while the histogram still counts the
    non-finite ones in the overflow/zero bins."""
    acc = TensorAccumulator()
    acc.update(torch.tensor([1.0, 3.0]))
    acc.update(torch.tensor([float("nan"), float("inf"), float("-inf"), 5.0]))
    snap = acc.snapshot()
    # Finite population is {1, 3, 5} — three values, not the six fed in.
    assert snap.n == 3
    assert snap.min == 1.0
    assert snap.max == 5.0
    assert math.isfinite(snap.mean) and snap.mean == pytest.approx(3.0)
    assert math.isfinite(snap.std)
    # The non-finite values stay visible in the histogram.
    assert snap.hist[N_BINS - 1] == 1  # +inf
    assert snap.hist[0] == 1  # -inf
    assert snap.hist[ZERO_BIN] == 1  # nan in the zero band


def test_all_non_finite_update_leaves_scalar_stats_empty() -> None:
    acc = TensorAccumulator()
    acc.update(torch.tensor([float("nan"), float("inf")]))
    snap = acc.snapshot()
    assert snap.n == 0  # nothing finite contributed
    assert math.isnan(snap.mean)  # n == 0
    # but the divergence is still counted in the histogram.
    assert snap.hist[ZERO_BIN] == 1 and snap.hist[N_BINS - 1] == 1


def test_retain_layers_drops_unwatched_buckets() -> None:
    acc = WatchAccumulator()
    acc.update(layer="a", phase="train", epoch=0, kind="activation", x=torch.tensor([1.0]))
    acc.update(layer="b", phase="train", epoch=0, kind="activation", x=torch.tensor([2.0]))
    acc.update(layer="b", phase="val", epoch=1, kind="activation", x=torch.tensor([3.0]))
    assert {k[0] for k in acc.snapshot().stats} == {"a", "b"}
    acc.retain_layers(["a"])
    # Every "b" bucket (both phases and epochs) is gone; "a" survives.
    assert {k[0] for k in acc.snapshot().stats} == {"a"}


def test_retain_layers_keeps_all_epochs_of_a_kept_layer() -> None:
    acc = WatchAccumulator()
    acc.update(layer="a", phase="train", epoch=0, kind="activation", x=torch.tensor([1.0]))
    acc.update(layer="a", phase="train", epoch=1, kind="activation", x=torch.tensor([2.0]))
    acc.retain_layers(["a"])
    assert set(acc.snapshot().stats) == {("a", "train", 0), ("a", "train", 1)}


def test_watch_accumulator_diverged_activation_lands_in_overflow_bins() -> None:
    """The same divergence routed through WatchAccumulator's per-channel path
    bins each non-finite element into the overflow/zero bins."""
    acc = WatchAccumulator()
    # (B=2, C=1, H=2, W=1): inf, -inf, nan, and a finite value.
    x = torch.tensor([float("inf"), float("-inf"), float("nan"), 1.0]).reshape(2, 1, 2, 1)
    acc.update(layer="a", phase="train", epoch=0, kind="activation", x=x)
    act = acc.snapshot().stats[("a", "train", 0)].activations
    assert act.hist[N_BINS - 1] == 1
    assert act.hist[0] == 1
    assert act.hist[ZERO_BIN] == 1
    assert act.hist[ZERO_BIN - 1] == 0
    assert act.hist[ZERO_BIN + 1] == 0
    # The per-channel row mirrors the universal histogram for the lone channel.
    assert act.channel_hists is not None
    assert tuple(act.channel_hists[0]) == act.hist


def test_channel_limit_caps_per_channel_rows_but_not_universal() -> None:
    """The cap keeps per-channel rows for the first N channels only, while the
    universal histogram and scalars still cover every channel."""
    acc = TensorAccumulator()
    # (B=1, C=4, H=1, W=1): one distinct value per channel.
    x = torch.tensor([1.0, 2.0, 3.0, 4.0]).reshape(1, 4, 1, 1)
    acc.update(x, channel_limit=2)
    snap = acc.snapshot()
    # Per-channel rows: only the first 2 channels.
    assert snap.channel_hists is not None
    assert len(snap.channel_hists) == 2
    # Universal histogram + scalars cover all 4 channels.
    assert snap.n == 4
    assert sum(snap.hist) == 4
    assert snap.max == pytest.approx(4.0)


def test_channel_limit_none_keeps_all_channels() -> None:
    acc = TensorAccumulator()
    x = torch.arange(4.0).reshape(1, 4, 1, 1)
    acc.update(x, channel_limit=None)
    snap = acc.snapshot()
    assert snap.channel_hists is not None
    assert len(snap.channel_hists) == 4


def test_configure_flushes_only_on_change() -> None:
    acc = WatchAccumulator()
    acc.update(layer="a", phase="train", epoch=0, kind="activation", x=torch.tensor([1.0]))
    # Same values as the defaults -> no change, no flush.
    from nansense.patches import DEFAULT_SAMPLES_PER_CHANNEL
    from nansense.watch import DEFAULT_CHANNEL_LIMIT

    assert not acc.configure(
        channel_limit=DEFAULT_CHANNEL_LIMIT,
        samples_per_channel=DEFAULT_SAMPLES_PER_CHANNEL,
    )
    assert ("a", "train", 0) in acc.snapshot().stats
    # A real change flushes every bucket.
    assert acc.configure(channel_limit=8, samples_per_channel=3)
    assert acc.snapshot().stats == {}


def test_configure_channel_limit_applies_to_new_buckets() -> None:
    acc = WatchAccumulator()
    acc.configure(channel_limit=2, samples_per_channel=5)
    x = torch.arange(4.0).reshape(1, 4, 1, 1)
    acc.update(layer="a", phase="train", epoch=0, kind="activation", x=x)
    rows = acc.snapshot().stats[("a", "train", 0)].activations.channel_hists
    assert rows is not None and len(rows) == 2


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


def _patch_batch() -> tuple[torch.Tensor, torch.Tensor]:
    """A (act, x) pair small enough for fast patch updates."""
    act = torch.randn(2, 3, 4, 4)
    x = torch.randn(2, 3, 8, 8)
    return act, x


def test_watch_accumulator_update_patches_lands_in_snapshot() -> None:
    acc = WatchAccumulator()
    act, x = _patch_batch()
    acc.update_patches(layer="a", phase="train", epoch=0, act=act, x=x)
    snap = acc.snapshot()
    patches = snap.stats[("a", "train", 0)].patches
    assert patches is not None
    assert set(patches.by_type) == {
        "max_pixel",
        "min_pixel",
        "max_average",
        "min_average",
    }
    # Buckets without patch updates report None, not an empty snapshot.
    acc.update(layer="b", phase="train", epoch=0, kind="activation", x=torch.tensor([1.0]))
    assert acc.snapshot().stats[("b", "train", 0)].patches is None


def test_watch_accumulator_snapshot_can_skip_patches() -> None:
    acc = WatchAccumulator()
    act, x = _patch_batch()
    acc.update_patches(layer="a", phase="train", epoch=0, act=act, x=x)
    snap = acc.snapshot(include_patches=False)
    assert snap.stats[("a", "train", 0)].patches is None
    # The buffers themselves are untouched — a full snapshot still has them.
    assert acc.snapshot().stats[("a", "train", 0)].patches is not None


def test_watch_accumulator_evicts_older_epoch_patch_buffers() -> None:
    acc = WatchAccumulator()
    act, x = _patch_batch()
    acc.update_patches(layer="a", phase="train", epoch=0, act=act, x=x)
    acc.update_patches(layer="a", phase="val", epoch=0, act=act, x=x)
    acc.update_patches(layer="a", phase="train", epoch=1, act=act, x=x)
    snap = acc.snapshot()
    # The newer train epoch released epoch 0's train buffers; val (still on
    # epoch 0) and the new train bucket keep theirs.
    assert snap.stats[("a", "train", 0)].patches is None
    assert snap.stats[("a", "train", 1)].patches is not None
    assert snap.stats[("a", "val", 0)].patches is not None


def test_watch_accumulator_evicts_patches_when_bucket_precreated() -> None:
    """Histogram updates usually create the new epoch's bucket before any
    patch update arrives — the patch eviction must still fire then."""
    acc = WatchAccumulator()
    act, x = _patch_batch()
    acc.update_patches(layer="a", phase="train", epoch=0, act=act, x=x)
    acc.update(layer="a", phase="train", epoch=1, kind="activation", x=x)
    acc.update_patches(layer="a", phase="train", epoch=1, act=act, x=x)
    snap = acc.snapshot()
    assert snap.stats[("a", "train", 0)].patches is None
    assert snap.stats[("a", "train", 1)].patches is not None


@pytest.mark.parametrize(
    "shape",
    [(2, 3), (2, 3, 4), (2, 3, 2, 2)],
    ids=["2d", "3d", "4d"],
)
def test_accumulator_channel_hists_match_per_channel_counts(
    shape: tuple[int, ...],
) -> None:
    """Each dim-1 slice's values land in its own row, rows sum to `hist`."""
    torch.manual_seed(0)
    x = torch.randn(shape)
    acc = TensorAccumulator()
    acc.update(x)
    snap = acc.snapshot()
    assert snap.channel_hists is not None
    assert len(snap.channel_hists) == shape[1]
    for c in range(shape[1]):
        expected = [0] * N_BINS
        for v in x[:, c].reshape(-1).tolist():
            expected[bin_index(v)] += 1
        assert list(snap.channel_hists[c]) == expected
    summed = [sum(col) for col in zip(*snap.channel_hists)]
    assert tuple(summed) == snap.hist


def test_accumulator_1d_input_has_no_channel_hists() -> None:
    acc = TensorAccumulator()
    acc.update(torch.tensor([1.0, 2.0]))
    assert acc.snapshot().channel_hists is None


def test_accumulator_channel_count_change_collapses_to_universal() -> None:
    """A dim-1 size change (variable tokens) turns per-channel off for good."""
    acc = TensorAccumulator()
    acc.update(torch.ones(2, 3))
    assert acc.snapshot().channel_hists is not None
    acc.update(torch.ones(2, 4))
    snap = acc.snapshot()
    assert snap.channel_hists is None
    assert sum(snap.hist) == 14  # the universal histogram kept counting
    # Re-appearing with the original channel count does not re-enable it.
    acc.update(torch.ones(2, 3))
    assert acc.snapshot().channel_hists is None


def test_watch_accumulator_new_epoch_collapses_same_phase_channels_only() -> None:
    """A phase's new epoch releases only that phase's older channel buffers."""
    acc = WatchAccumulator()
    x = torch.ones(2, 3)
    for phase, epoch in [("train", 0), ("val", 0), ("train", 1)]:
        acc.update(layer="a", phase=phase, epoch=epoch, kind="activation", x=x)
        acc.update(layer="a", phase=phase, epoch=epoch, kind="gradient", x=x)
    snap = acc.snapshot(include_patches=False)
    old_train = snap.stats[("a", "train", 0)]
    assert old_train.activations.channel_hists is None
    assert old_train.gradients.channel_hists is None
    # The universal histogram of the collapsed bucket is untouched.
    assert sum(old_train.activations.hist) == 6
    assert snap.stats[("a", "val", 0)].activations.channel_hists is not None
    assert snap.stats[("a", "train", 1)].activations.channel_hists is not None
