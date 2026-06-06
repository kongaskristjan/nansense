"""Tests for experiments: deep dream and Captum attributions."""

from __future__ import annotations

import threading

import pytest
import torch
from torch import Tensor, nn

import playgrad
from playgrad.experiments import _value_bounds, _zoom_in
from playgrad.session import Session

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


def _train_step(model: TinyClassifier, batch_size: int) -> None:
    x = torch.randn(batch_size, 3, 4, 4)
    y = torch.randint(0, 3, (batch_size,))
    model.zero_grad(set_to_none=True)
    loss = nn.functional.cross_entropy(model(x), y)
    loss.backward()


def _paused_session(
    batch_size: int = 2,
) -> tuple[Session, TinyClassifier, threading.Thread]:
    model = TinyClassifier()
    session = playgrad.start(model, epochs=1, phases={"train": 2})

    def loop() -> None:
        for _ in range(2):
            with session.batch(phase="train", epoch=0):
                _train_step(model, batch_size)

    thread = threading.Thread(target=loop, daemon=True)
    thread.start()
    assert session.wait_until_paused(timeout=5)
    return session, model, thread


def _finish(session: Session, thread: threading.Thread) -> None:
    session.detach()
    thread.join(timeout=5)
    assert not thread.is_alive()


def _dream_params(**overrides: object) -> dict[str, object]:
    params: dict[str, object] = {
        "channel": 0,
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


def test_deep_dream_publishes_done_result_with_image() -> None:
    session, _, thread = _paused_session()
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
    # Without a "batch" param the dream covers the whole current batch.
    assert result.image is not None and result.image.shape == (2, 3, 4, 4)
    assert result.image.device.type == "cpu"
    assert result.reference is not None and result.reference.shape == (2, 3, 4, 4)
    assert isinstance(result.objective, float)
    _finish(session, thread)


def test_deep_dream_default_batch_caps_at_eight() -> None:
    session, _, thread = _paused_session(batch_size=10)
    session.request_experiment(
        kind="deep_dream",
        layer="conv",
        params=_dream_params(start="noise", steps=1),  # no "batch" param
    )
    assert session.wait_for_experiment(timeout=10)
    result = session.experiment_result
    assert result is not None and result.error is None
    assert result.image is not None and result.image.shape == (8, 3, 4, 4)
    _finish(session, thread)


@pytest.mark.parametrize(
    "start, batch, expected",
    [
        ("noise", 3, 3),  # noise draws exactly the requested count
        ("sample", 5, 2),  # the real input batch caps sample starts
        ("sample", 1, 1),
    ],
)
def test_deep_dream_batch_param(start: str, batch: int, expected: int) -> None:
    session, _, thread = _paused_session()
    session.request_experiment(
        kind="deep_dream",
        layer="conv",
        params=_dream_params(start=start, batch=batch, steps=2),
    )
    assert session.wait_for_experiment(timeout=10)
    result = session.experiment_result
    assert result is not None and result.error is None
    assert result.image is not None and result.image.shape == (expected, 3, 4, 4)
    assert result.reference is not None
    assert result.reference.shape == (expected, 3, 4, 4)
    _finish(session, thread)


def test_deep_dream_noise_differs_across_runs() -> None:
    session, _, thread = _paused_session()
    references: list[Tensor] = []
    for _ in range(2):
        session.request_experiment(
            kind="deep_dream",
            layer="conv",
            params=_dream_params(start="noise", batch=2, steps=1),
        )
        assert session.wait_for_experiment(timeout=10)
        result = session.experiment_result
        assert result is not None and result.error is None
        assert result.reference is not None
        references.append(result.reference)
    assert not torch.equal(references[0], references[1])
    _finish(session, thread)


def test_deep_dream_clamps_to_displayable_range() -> None:
    session, _, thread = _paused_session()
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
    _finish(session, thread)


def test_deep_dream_leaves_training_state_untouched() -> None:
    session, model, thread = _paused_session()
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
    _finish(session, thread)


def test_deep_dream_bad_channel_publishes_error() -> None:
    session, _, thread = _paused_session()
    session.request_experiment(
        kind="deep_dream", layer="conv", params=_dream_params(channel=99)
    )
    assert session.wait_for_experiment(timeout=10)
    result = session.experiment_result
    assert result is not None and result.done
    assert result.error is not None and "channel" in result.error
    # The worker survived and is still paused/responsive.
    assert session.pause_count == 1
    _finish(session, thread)


def test_deep_dream_works_on_fx_intermediate_layer() -> None:
    session, _, thread = _paused_session()
    assert "relu" in session.layer_names
    session.request_experiment(
        kind="deep_dream", layer="relu", params=_dream_params(steps=3)
    )
    assert session.wait_for_experiment(timeout=10)
    result = session.experiment_result
    assert result is not None and result.error is None and result.image is not None
    _finish(session, thread)


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
    session = playgrad.start(model, epochs=1, phases={"train": 2})

    def loop() -> None:
        for _ in range(2):
            with session.batch(phase="train", epoch=0):
                x = torch.randn(2, 8)
                y = torch.randint(0, 3, (2,))
                model.zero_grad(set_to_none=True)
                nn.functional.cross_entropy(model(x), y).backward()

    thread = threading.Thread(target=loop, daemon=True)
    thread.start()
    assert session.wait_until_paused(timeout=5)
    session.request_experiment(
        kind="deep_dream",
        layer="fc1",
        params={"channel": 0, "steps": 3, "lr": 0.1, "start": "noise", "batch": 3},
    )
    assert session.wait_for_experiment(timeout=10)
    result = session.experiment_result
    assert result is not None and result.error is None
    assert result.image is not None and result.image.shape == (3, 8)
    _finish(session, thread)


@pytest.mark.parametrize(
    "kind, params, expected_shape",
    [
        ("gradcam", {"target": -1, "sample": 0}, (1, 1, 4, 4)),
        ("neuron_gradient", {"channel": 0, "sample": 0}, (1, 3, 4, 4)),
        ("neuron_ig", {"channel": 0, "steps": 4, "sample": 0}, (1, 3, 4, 4)),
        (
            "occlusion",
            {"target": -1, "window": 2, "stride": 2, "sample": 0},
            (1, 3, 4, 4),
        ),
    ],
)
def test_captum_methods_publish_attributions(
    kind: str, params: dict[str, object], expected_shape: tuple[int, ...]
) -> None:
    session, _, thread = _paused_session()
    session.request_experiment(kind=kind, layer="conv", params=params)
    assert session.wait_for_experiment(timeout=15)
    result = session.experiment_result
    assert result is not None
    assert result.error is None
    assert result.done
    assert result.attribution is not None
    assert tuple(result.attribution.shape) == expected_shape
    assert result.attribution.device.type == "cpu"
    assert result.reference is not None
    _finish(session, thread)


def test_captum_on_fx_intermediate_publishes_module_hint() -> None:
    session, _, thread = _paused_session()
    session.request_experiment(
        kind="gradcam", layer="relu", params={"target": -1, "sample": 0}
    )
    assert session.wait_for_experiment(timeout=10)
    result = session.experiment_result
    assert result is not None and result.error is not None
    assert "nn.Module" in result.error
    _finish(session, thread)


def test_new_request_supersedes_previous_result() -> None:
    session, _, thread = _paused_session()
    session.request_experiment(
        kind="deep_dream", layer="conv", params=_dream_params(steps=2)
    )
    assert session.wait_for_experiment(timeout=10)
    first = session.experiment_result
    assert first is not None

    seq = session.request_experiment(
        kind="neuron_gradient", layer="conv", params={"channel": 0}
    )
    assert session.wait_for_experiment(timeout=10)
    second = session.experiment_result
    assert second is not None and second.seq == seq
    assert second.kind == "neuron_gradient"
    _finish(session, thread)


def test_request_experiment_rejects_unknown_kind() -> None:
    session = playgrad.start(TinyClassifier(), epochs=1, phases={"train": 1})
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
