"""Tests for experiments: deep dream and Captum attributions."""

from __future__ import annotations

import threading
from collections.abc import Iterator
from contextlib import contextmanager

import pytest
import torch
from torch import Tensor, nn

import nansense
from nansense.experiments import (
    EXPERIMENT_KINDS,
    _value_bounds,
    _zoom_in,
    available_experiment_kinds,
)
from nansense.session import Session
from tests.nansense.helpers import paused_session, train_step

CIFARISH_MEAN = (0.5, 0.4, 0.3)
CIFARISH_STD = (0.25, 0.2, 0.3)


class TinyClassifier(nn.Module):
    """Conv + BN + classifier head: fx-traceable, target-class compatible."""

    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(3, 4, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(4)
        self.fc = nn.Linear(4 * 4 * 4, 3)

    def forward(self, x: Tensor) -> Tensor:
        h = torch.relu(self.bn(self.conv(x)))
        return self.fc(h.flatten(1))


@contextmanager
def _paused_session(
    batch_size: int = 2,
) -> Iterator[tuple[Session, TinyClassifier]]:
    """A paused TinyClassifier session fed `batch_size`-sample image batches."""
    model = TinyClassifier()

    def step(m: TinyClassifier) -> None:
        train_step(m, input_shape=(3, 4, 4), batch_size=batch_size)

    with paused_session(model, step) as session:
        yield session, model


def _dream_params(**overrides: object) -> dict[str, object]:
    params: dict[str, object] = {
        "channels": 4,  # TinyClassifier.conv emits 4 channels
        "sample": 0,
        "steps": 5,
        "lr": 0.1,
        "diffusion": 0.1,
        "jitter": 1,
        "zoom": 1.0,
        "start": "sample",
        "clamp": True,
        "mean": CIFARISH_MEAN,
        "std": CIFARISH_STD,
    }
    params.update(overrides)
    return params


def test_pause_loop_marks_experiment_running_before_running_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The pause loop sets `_experiment_running` atomically with the dequeue
    (under the lock), so a `cancel_experiment(seq)` arriving before
    `run_experiment_guarded` starts is honoured rather than a silent no-op."""
    import nansense.experiments as experiments_mod

    called = threading.Event()
    running_at_entry: list[int | None] = []

    def spy(session: Session, request: object) -> None:
        # Record what the pause loop already set, then return without running
        # (and without clearing it): we only care about the dequeue invariant.
        running_at_entry.append(session._experiment_running)  # type: ignore[attr-defined]
        called.set()

    with _paused_session() as (session, _):
        monkeypatch.setattr(experiments_mod, "run_experiment_guarded", spy)
        seq = session.request_experiment(
            kind="deep_dream", layer="conv", params=_dream_params()
        )
        assert called.wait(timeout=5)
        # Already this seq at entry — the dequeue set it, not the (replaced)
        # run_experiment_guarded, closing the cancel-race window.
        assert running_at_entry == [seq]


def test_deep_dream_publishes_done_result_with_image() -> None:
    with _paused_session() as (session, _):
        assert session.input_batch_size == 2
        session.request_experiment(
            kind="deep_dream", layer="conv", params=_dream_params()
        )
        assert session.wait_for_experiment(timeout=10)
        result = session.experiment_result
        assert result is not None
        assert result.error is None
        assert result.done and result.step == result.total_steps == 5
        assert result.kind == "deep_dream" and result.layer == "conv"
        # One sample per channel over conv's 4 channels.
        assert result.image is not None and result.image.shape == (4, 3, 4, 4)
        assert result.image.device.type == "cpu"
        # Current-batch start shows the single chosen input.
        assert result.reference is not None and result.reference.shape == (1, 3, 4, 4)
        assert isinstance(result.objective, float)


def test_deep_dream_clips_channels_to_layer_count() -> None:
    with _paused_session(batch_size=10) as (session, _):
        session.request_experiment(
            kind="deep_dream",
            layer="conv",
            # More channels requested than conv has (4): clip to the layer.
            params=_dream_params(channels=99, start="noise", steps=1),
        )
        assert session.wait_for_experiment(timeout=10)
        result = session.experiment_result
        assert result is not None and result.error is None
        assert result.image is not None and result.image.shape == (4, 3, 4, 4)
        # The worker survived the clip and is still paused/responsive.
        assert session.pause_count == 1


@pytest.mark.parametrize(
    "start, channels, expected",
    [
        ("noise", 3, 3),  # under conv's 4-channel cap
        ("noise", 8, 4),  # clipped to conv's channel count
        ("sample", 2, 2),
        ("sample", 4, 4),
    ],
)
def test_deep_dream_channels_param(start: str, channels: int, expected: int) -> None:
    with _paused_session() as (session, _):
        session.request_experiment(
            kind="deep_dream",
            layer="conv",
            params=_dream_params(start=start, channels=channels, steps=2),
        )
        assert session.wait_for_experiment(timeout=10)
        result = session.experiment_result
        assert result is not None and result.error is None
        # One dreamed sample per channel.
        assert result.image is not None and result.image.shape == (expected, 3, 4, 4)
        if start == "noise":
            assert result.reference is None  # noise shows no input
        else:
            # Current batch shows the single shared starting sample.
            assert result.reference is not None
            assert result.reference.shape == (1, 3, 4, 4)


def test_deep_dream_noise_differs_across_runs() -> None:
    with _paused_session() as (session, _):
        images: list[Tensor] = []
        for _ in range(2):
            session.request_experiment(
                kind="deep_dream",
                layer="conv",
                params=_dream_params(start="noise", channels=2, steps=1),
            )
            assert session.wait_for_experiment(timeout=10)
            result = session.experiment_result
            assert result is not None and result.error is None
            assert result.reference is None  # noise start carries no input
            assert result.image is not None
            images.append(result.image)
        # Per-request-seeded noise → different dreams across runs.
        assert not torch.equal(images[0], images[1])


def test_deep_dream_clamps_to_displayable_range() -> None:
    with _paused_session() as (session, _):
        session.request_experiment(
            kind="deep_dream",
            layer="conv",
            params=_dream_params(steps=8, lr=10.0),  # huge lr would escape bounds
        )
        assert session.wait_for_experiment(timeout=10)
        result = session.experiment_result
        assert result is not None and result.error is None and result.image is not None
        lo, hi = _value_bounds(3, CIFARISH_MEAN, CIFARISH_STD)
        assert bool((result.image >= lo - 1e-5).all())
        assert bool((result.image <= hi + 1e-5).all())


def test_deep_dream_leaves_training_state_untouched() -> None:
    with _paused_session() as (session, model):
        running_mean = model.bn.running_mean
        assert running_mean is not None
        mean_before = running_mean.clone()
        grads_before = {
            n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None
        }
        assert grads_before  # the training step populated them
        rng_before = torch.get_rng_state()
        flags_before = [m.training for m in model.modules()]

        session.request_experiment(
            kind="deep_dream", layer="conv", params=_dream_params()
        )
        assert session.wait_for_experiment(timeout=10)
        assert session.experiment_result is not None
        assert session.experiment_result.error is None

        torch.testing.assert_close(running_mean, mean_before)
        for name, param in model.named_parameters():
            assert param.grad is not None
            torch.testing.assert_close(param.grad, grads_before[name])
        assert torch.equal(torch.get_rng_state(), rng_before)
        assert [m.training for m in model.modules()] == flags_before


def test_deep_dream_works_on_fx_intermediate_layer() -> None:
    with _paused_session() as (session, _):
        assert "relu" in session.layer_names
        session.request_experiment(
            kind="deep_dream", layer="relu", params=_dream_params(steps=3)
        )
        assert session.wait_for_experiment(timeout=10)
        result = session.experiment_result
        assert result is not None and result.error is None and result.image is not None


class VectorNet(nn.Module):
    """Vector input [B, 8]: deep dream must accept non-image network inputs."""

    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(8, 6)
        self.fc2 = nn.Linear(6, 3)

    def forward(self, x: Tensor) -> Tensor:
        return self.fc2(torch.relu(self.fc1(x)))


def test_deep_dream_works_on_vector_input() -> None:
    model = VectorNet()
    with paused_session(model, lambda m: train_step(m, input_shape=(8,))) as session:
        session.request_experiment(
            kind="deep_dream",
            layer="fc1",
            # fc1 emits 6 features; 3 channels → 3 vector samples of width 8.
            params={"channels": 3, "steps": 3, "lr": 0.1, "start": "noise"},
        )
        assert session.wait_for_experiment(timeout=10)
        result = session.experiment_result
        assert result is not None and result.error is None
        assert result.image is not None and result.image.shape == (3, 8)
        assert result.reference is None  # noise start carries no input


@pytest.mark.parametrize(
    "kind, params, channels",
    [
        ("gradcam", {"target": -1}, 1),  # Grad-CAM emits one heatmap channel
        ("neuron_gradient", {"channel": 0}, 3),
        ("neuron_ig", {"channel": 0, "ig_steps": 4}, 3),
        ("occlusion", {"channel": 0, "window": 2, "stride": 2}, 3),
    ],
)
def test_captum_methods_publish_attributions(
    kind: str, params: dict[str, object], channels: int
) -> None:
    # Captum runs on a batch now (like deep dream): the whole current batch
    # (two samples here), not a single example.
    with _paused_session() as (session, _):
        session.request_experiment(kind=kind, layer="conv", params=params)
        assert session.wait_for_experiment(timeout=15)
        result = session.experiment_result
        assert result is not None
        assert result.error is None
        assert result.done
        assert result.attribution is not None
        assert tuple(result.attribution.shape) == (2, channels, 4, 4)
        assert result.attribution.device.type == "cpu"
        assert result.reference is not None and result.reference.shape[0] == 2


def test_captum_batch_param_caps_samples() -> None:
    with _paused_session(batch_size=4) as (session, _):
        session.request_experiment(
            kind="neuron_gradient", layer="conv", params={"channel": 0, "batch": 2}
        )
        assert session.wait_for_experiment(timeout=15)
        result = session.experiment_result
        assert result is not None and result.error is None
        assert result.attribution is not None and result.attribution.shape[0] == 2


def test_occlusion_runs_on_fx_intermediate_layer() -> None:
    # Occlusion is retargeted to a layer-channel's activation, so it reads the
    # value through the fx interpreter and no longer needs an nn.Module.
    with _paused_session() as (session, _):
        assert "relu" in session.layer_names
        session.request_experiment(
            kind="occlusion",
            layer="relu",
            params={"channel": 0, "window": 2, "stride": 2},
        )
        assert session.wait_for_experiment(timeout=15)
        result = session.experiment_result
        assert result is not None and result.error is None
        assert result.attribution is not None
        assert tuple(result.attribution.shape) == (2, 3, 4, 4)


@pytest.mark.parametrize(
    "kind", ["gradcam", "neuron_gradient", "neuron_ig"]
)
def test_captum_module_kinds_on_fx_intermediate_publish_module_hint(
    kind: str,
) -> None:
    with _paused_session() as (session, _):
        session.request_experiment(kind=kind, layer="relu", params={})
        assert session.wait_for_experiment(timeout=10)
        result = session.experiment_result
        assert result is not None and result.error is not None
        assert "nn.Module" in result.error


@pytest.mark.parametrize(
    "layer, kind, available",
    [
        ("conv", "gradcam", True),
        ("relu", "gradcam", False),
        ("relu", "neuron_gradient", False),
        ("relu", "neuron_ig", False),
        ("relu", "deep_dream", True),
        ("relu", "occlusion", True),
    ],
)
def test_layer_available_by_kind(layer: str, kind: str, available: bool) -> None:
    from nansense.experiments import layer_available

    with _paused_session() as (session, _):
        assert layer in session.layer_names
        assert layer_available(session, layer, kind) is available


def test_register_auto_experiment_supersedes_queued_request() -> None:
    # No training loop runs, so requests stay queued; re-registering the same
    # key drops the superseded request so the pause loop never runs the stale
    # one — only the latest queued request for the key executes.
    session = nansense.start(TinyClassifier(), epochs=1, phases={"train": 1})
    first = session.register_auto_experiment(
        "page", kind="deep_dream", layer="conv", params={}
    )
    second = session.register_auto_experiment(
        "page", kind="deep_dream", layer="conv", params={}
    )
    queued = [r.seq for r in session._experiment_queue]
    assert queued == [second] and first not in queued


def test_auto_run_experiments_setting_defaults_on() -> None:
    session = nansense.start(TinyClassifier(), epochs=1, phases={"train": 1})
    assert session.auto_run_experiments is True
    session.set_auto_run_experiments(False)
    assert session.auto_run_experiments is False


def test_queued_experiments_publish_per_seq_results() -> None:
    # Two clients (browser tabs) request back to back: both run to
    # completion in order, and each result stays retrievable by its seq.
    with _paused_session() as (session, _):
        seq_a = session.request_experiment(
            kind="deep_dream", layer="conv", params=_dream_params(steps=2)
        )
        seq_b = session.request_experiment(
            kind="neuron_gradient", layer="conv", params={"channel": 0, "sample": 0}
        )
        assert session.wait_for_experiment(timeout=10)
        first = session.experiment_result_for(seq_a)
        assert first is not None and first.done and first.error is None
        assert first.kind == "deep_dream" and first.seq == seq_a
        assert first.image is not None
        second = session.experiment_result_for(seq_b)
        assert second is not None and second.done and second.error is None
        assert second.kind == "neuron_gradient" and second.seq == seq_b
        latest = session.experiment_result
        assert latest is not None and latest.seq == seq_b


def test_cancel_experiment_drops_one_seq_or_all() -> None:
    # No training loop is running, so requests stay queued.
    session = nansense.start(TinyClassifier(), epochs=1, phases={"train": 1})
    seq_a = session.request_experiment(
        kind="deep_dream", layer="conv", params={}
    )
    session.request_experiment(kind="deep_dream", layer="conv", params={})
    assert session.experiment_pending
    session.cancel_experiment(seq_a)
    assert session.experiment_pending  # the other request is untouched
    session.cancel_experiment()  # no seq: drop everything
    assert not session.experiment_pending


def test_experiment_results_evict_oldest_seq() -> None:
    with _paused_session() as (session, _):
        seqs = [
            session.request_experiment(
                kind="deep_dream", layer="conv", params=_dream_params(steps=1)
            )
            for _ in range(9)  # one more than _EXPERIMENT_RESULTS_KEPT
        ]
        assert session.wait_for_experiment(timeout=30)
        assert session.experiment_result_for(seqs[0]) is None  # evicted
        for seq in seqs[1:]:
            result = session.experiment_result_for(seq)
            assert result is not None and result.seq == seq


def test_request_experiment_rejects_unknown_kind() -> None:
    session = nansense.start(TinyClassifier(), epochs=1, phases={"train": 1})
    with pytest.raises(ValueError, match="unknown experiment kind"):
        session.request_experiment(kind="bogus", layer="conv", params={})


def test_value_bounds_inverts_display_normalization() -> None:
    lo, hi = _value_bounds(3, CIFARISH_MEAN, CIFARISH_STD)
    for c in range(3):
        assert float(lo[0, c, 0, 0]) == pytest.approx(
            -CIFARISH_MEAN[c] / CIFARISH_STD[c]
        )
        assert float(hi[0, c, 0, 0]) == pytest.approx(
            (1 - CIFARISH_MEAN[c]) / CIFARISH_STD[c]
        )
    # Without stats the input is assumed to be in [0, 1] already.
    lo, hi = _value_bounds(3, None, None)
    assert float(lo.min()) == 0.0 and float(hi.max()) == 1.0


@pytest.mark.parametrize(
    "size, zoom, changes",
    [
        (32, 1.1, True),
        (32, 1.0, False),
        (32, 1.001, False),  # rounds below one pixel — no-op
    ],
)
def test_zoom_in_keeps_shape(size: int, zoom: float, changes: bool) -> None:
    x = torch.randn(1, 3, size, size)
    zoomed = _zoom_in(x, zoom)
    assert zoomed.shape == x.shape
    assert torch.equal(zoomed, x) != changes


def test_available_kinds_offers_everything() -> None:
    assert available_experiment_kinds() == EXPERIMENT_KINDS
