"""Tests for experiments: deep dream and Captum attributions."""

from __future__ import annotations

import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from unittest import mock

import av
import pytest
import torch
from torch import Tensor, nn

import nansense
import nansense.recording
from nansense import experiments
from nansense.recording import RecordingManager
from nansense.experiments import (
    EXPERIMENT_KINDS,
    ExperimentQueueState,
    ExperimentResult,
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


def test_expired_time_limit_cuts_the_run_off_between_steps(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Past the wall-clock ceiling `should_abort` fires between steps: the
    run stops early and the progress so far still publishes as done."""
    import nansense.experiments as experiments_mod

    monkeypatch.setattr(experiments_mod, "_EXPERIMENT_TIME_LIMIT", 0.0)
    with _paused_session() as (session, _):
        session.request_experiment(
            kind="deep_dream", layer="conv", params=_dream_params()
        )
        assert session.wait_for_experiment(timeout=10)
        result = session.experiment_result
        assert result is not None and result.error is None
        # The deadline expired before the first step, so the run was cut off
        # immediately — but the result still lands, marked done.
        assert result.done
        assert result.step == 0 < result.total_steps


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


def test_deep_dream_minimize_lands_below_maximize() -> None:
    """The minimize toggle flips the step direction: from an identical
    (jitter-free, no-clamp) start, minimizing the channel objective lands below
    maximizing it."""
    common: dict[str, object] = {
        "start": "sample",
        "jitter": 0,
        "diffusion": 0.0,
        "clamp": False,
        "steps": 8,
        "lr": 0.2,
    }
    with _paused_session() as (session, _):
        session.request_experiment(
            kind="deep_dream",
            layer="conv",
            params=_dream_params(minimize=False, **common),
        )
        assert session.wait_for_experiment(timeout=10)
        maximized = session.experiment_result
        assert maximized is not None and maximized.objective is not None

        session.request_experiment(
            kind="deep_dream",
            layer="conv",
            params=_dream_params(minimize=True, **common),
        )
        assert session.wait_for_experiment(timeout=10)
        minimized = session.experiment_result
        assert minimized is not None and minimized.objective is not None

    assert minimized.objective < maximized.objective


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


@pytest.mark.parametrize(
    "kind, params",
    [
        ("deep_dream", _dream_params(steps=2)),
        ("gradcam", {"target": -1}),
        ("neuron_gradient", {"channel": 0}),
        ("neuron_ig", {"channel": 0, "ig_steps": 2}),
        ("occlusion", {"channel": 0, "window": 2, "stride": 2}),
    ],
)
def test_final_result_publishes_after_isolation_unwinds(
    kind: str, params: dict[str, object]
) -> None:
    """The final result is what every waiter wakes on, so the isolation must
    already be restored when it publishes. A kind that yields it from inside
    `isolated_model` only restores when its generator is next resumed, which
    leaves waiters racing the training thread for a model still in eval mode.
    """
    with _paused_session() as (session, model):
        expected = [m.training for m in model.modules()]
        flags_at_publish: list[list[bool]] = []
        publish = experiments._publish_experiment

        def spy(sess: Session, result: ExperimentResult) -> None:
            if result.done:
                flags_at_publish.append([m.training for m in model.modules()])
            publish(sess, result)

        with mock.patch.object(experiments, "_publish_experiment", spy):
            session.request_experiment(kind=kind, layer="conv", params=params)
            assert session.wait_for_experiment(timeout=15)
        assert session.experiment_result is not None
        assert session.experiment_result.error is None
        assert flags_at_publish == [expected]


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


class SequenceNet(nn.Module):
    """Rank-3 input [B, 4, 5]: renderable as neither an image nor a strip."""

    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(5, 3)
        self.fc2 = nn.Linear(4 * 3, 3)

    def forward(self, x: Tensor) -> Tensor:
        return self.fc2(torch.relu(self.fc1(x)).flatten(1))


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


# --- progress cadence and recorded runs -------------------------------


def _published_results(
    session: Session, **params: object
) -> tuple[int, list[experiments.ExperimentResult]]:
    """One deep-dream run's `seq` and every result it published, in order."""
    results: list[experiments.ExperimentResult] = []
    original = experiments._publish_experiment

    def spy(target: Session, result: experiments.ExperimentResult) -> None:
        results.append(result)
        original(target, result)

    with mock.patch.object(experiments, "_publish_experiment", spy):
        seq = session.request_experiment(
            kind="deep_dream", layer="conv", params=_dream_params(**params)
        )
        assert session.wait_for_experiment(timeout=20)
    return seq, results


def _published_steps(session: Session, **params: object) -> list[int]:
    """The `step` of every result one deep-dream run publishes."""
    return [result.step for result in _published_results(session, **params)[1]]


def test_deep_dream_publishes_a_sample_of_its_steps_by_default() -> None:
    """The starting image, then ~`_PUBLISH_COUNT` snapshots of the ascent —
    not one per step: the page only ever draws the freshest, so the rest are
    copies nobody sees."""
    with _paused_session() as (session, _):
        steps = _published_steps(session, steps=40)
        assert steps == [0, *(2 * n for n in range(1, 21))]


def test_all_steps_publishes_every_step() -> None:
    """The knob that makes a recorded run replay the whole ascent."""
    with _paused_session() as (session, _):
        steps = _published_steps(session, steps=40, all_steps=True)
        assert steps == list(range(0, 41))


def test_deep_dream_publishes_its_starting_noise_untouched() -> None:
    """Step 0 is the picture before the ascent, not one publish interval into
    it — which is what makes "this was built out of noise" something a viewer
    is shown rather than something they are asked to take on trust."""
    with _paused_session() as (session, _):
        seq, results = _published_results(session, steps=8, start="noise")
        first = results[0]
        assert first.step == 0 and not first.done and first.image is not None
        # The same noise `_dream_start` draws for this seq, with nothing done
        # to it: no step, no jitter, no diffusion, no clamp.
        request = experiments.ExperimentRequest(
            "deep_dream", "conv", _dream_params(start="noise"), seq
        )
        expected = experiments._dream_start(
            session, request, torch.Generator().manual_seed(seq), 4
        )
        assert isinstance(expected, Tensor)
        assert torch.equal(first.image, expected)
        # And the ascent moved off it.
        last = results[-1]
        assert last.image is not None and not torch.equal(last.image, first.image)


def test_deep_dream_step_zero_reports_the_objective_it_starts_from() -> None:
    """The published objective series begins at the starting image's own
    value, so "did this climb?" is a comparison against the start rather than
    against a frame already part of the way up."""
    with _paused_session() as (session, _):
        _, results = _published_results(session, steps=8, start="noise", lr=1.0)
        objectives: list[float] = []
        for result in results:
            assert isinstance(result.objective, float)
            objectives.append(result.objective)
        assert objectives[-1] > objectives[0]


def test_all_steps_is_a_declared_deep_dream_knob() -> None:
    specs = {s.key: s for s in experiments.EXPERIMENT_PARAMS["deep_dream"]}
    assert specs["all_steps"].kind == "bool"
    assert specs["all_steps"].default is False


def _recorded(session: Session, tmp_path: Path) -> RecordingManager:
    """Point the session's recordings at `tmp_path` (as the UI tests do)."""
    manager = RecordingManager(directory=tmp_path / "rec")
    session._recording_manager = manager
    return manager


def _frame_count(path: Path) -> int:
    with av.open(str(path)) as container:
        return sum(1 for _ in container.decode(video=0))


def test_a_recorded_deep_dream_writes_one_frame_per_published_step(
    tmp_path: Path,
) -> None:
    """`video=True` turns the run's progress into a playable MP4, and the path
    arrives on the same result that reports the run done."""
    with _paused_session() as (session, _):
        manager = _recorded(session, tmp_path)
        seq = session.request_experiment(
            kind="deep_dream",
            layer="conv",
            params=_dream_params(steps=40, all_steps=True),
            video=True,
        )
        assert session.wait_for_experiment(timeout=30)
        result = session.experiment_result_for(seq)
        assert result is not None and result.error is None and result.done
        assert result.video is not None
        path = Path(result.video)
        assert path.parent == manager.directory
        assert path.exists() and path.stat().st_size > 0
        # Forty steps plus the image they started from.
        assert _frame_count(path) == 41


def test_an_unrecorded_run_leaves_no_video(tmp_path: Path) -> None:
    with _paused_session() as (session, _):
        manager = _recorded(session, tmp_path)
        seq = session.request_experiment(
            kind="deep_dream", layer="conv", params=_dream_params()
        )
        assert session.wait_for_experiment(timeout=20)
        result = session.experiment_result_for(seq)
        assert result is not None and result.video is None
        assert not manager.directory.exists()


def test_a_cancelled_recording_keeps_the_frames_it_got(tmp_path: Path) -> None:
    """Aborting mid-ascent still finalizes a playable file — the point of
    encoding frame by frame instead of at the end."""
    with _paused_session() as (session, _):
        _recorded(session, tmp_path)
        seq = session.request_experiment(
            kind="deep_dream",
            layer="conv",
            params=_dream_params(steps=200, all_steps=True),
            video=True,
        )
        # Cancel once a few frames are in, then wait for the run to notice.
        deadline = time.monotonic() + 20
        while time.monotonic() < deadline:
            progress = session.experiment_result_for(seq)
            if progress is not None and progress.step >= 3:
                break
            time.sleep(0.01)
        session.cancel_experiment(seq)
        assert session.wait_for_experiment(timeout=20)
        result = session.experiment_result_for(seq)
        assert result is not None and result.done and result.step < 200
        assert result.video is not None
        # Every step it got to, plus the step-0 frame it opened on.
        assert _frame_count(Path(result.video)) == result.step + 1


def test_a_dream_on_a_vector_input_records_its_strips(tmp_path: Path) -> None:
    """Not an images-only feature: a flat input draws as a colormapped strip,
    so its ascent animates like any other."""
    model = VectorNet()
    with paused_session(model, lambda m: train_step(m, input_shape=(8,))) as session:
        _recorded(session, tmp_path)
        seq = session.request_experiment(
            kind="deep_dream",
            layer="fc1",
            params={
                "channels": 3,
                "steps": 4,
                "lr": 0.1,
                "start": "noise",
                "all_steps": True,
            },
            video=True,
        )
        assert session.wait_for_experiment(timeout=20)
        result = session.experiment_result_for(seq)
        assert result is not None and result.error is None
        assert result.video is not None
        assert _frame_count(Path(result.video)) == 5  # four steps and the start


def test_a_dream_with_nothing_to_draw_records_no_file(tmp_path: Path) -> None:
    """A rank the renderers have no picture for records nothing at all, rather
    than a clip of status lines with a blank where the image goes."""
    model = SequenceNet()
    step = lambda m: train_step(m, input_shape=(4, 5))  # noqa: E731 — [B, 4, 5]
    with paused_session(model, step) as session:
        manager = _recorded(session, tmp_path)
        seq = session.request_experiment(
            kind="deep_dream",
            layer="fc1",
            params={"channels": 2, "steps": 3, "lr": 0.1, "start": "noise"},
            video=True,
        )
        assert session.wait_for_experiment(timeout=20)
        result = session.experiment_result_for(seq)
        assert result is not None and result.error is None
        assert result.image is not None and result.image.ndim == 3
        assert result.video is None
        assert list(manager.directory.glob("*.mp4")) == []


def test_a_failing_recorder_costs_the_video_not_the_run(tmp_path: Path) -> None:
    """The experiment is the valuable thing: a render that blows up is
    reported beside its result, not in place of it."""
    with _paused_session() as (session, _):
        _recorded(session, tmp_path)
        with mock.patch.object(
            nansense.recording,
            "_experiment_result_frame",
            side_effect=RuntimeError("no pixels for you"),
        ):
            seq = session.request_experiment(
                kind="deep_dream",
                layer="conv",
                params=_dream_params(steps=5),
                video=True,
            )
            assert session.wait_for_experiment(timeout=20)
        result = session.experiment_result_for(seq)
        assert result is not None and result.done
        assert result.image is not None  # the run itself finished
        assert result.video is None
        assert result.error is not None and "no pixels for you" in result.error


def test_a_locked_session_records_no_experiment_videos() -> None:
    """Writing a file per request is unbounded work on the one thread a shared
    demo's visitors all queue for, so the flag is dropped with the other
    ceilings — at the request, before anything runs."""
    session = nansense.start(TinyClassifier(), epochs=1, phases={"train": 1})
    session.lock()
    seq = session.request_experiment(
        kind="deep_dream", layer="conv", params={}, video=True
    )
    queued = [r for r in session._experiment_queue if r.seq == seq]
    assert queued and queued[0].video is False


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


def test_experiment_queue_state_places_each_request() -> None:
    # A request publishes nothing until it has progress, so the UI asks the
    # queue whether "no result" means running, waiting, or gone. No training
    # loop runs here, so both requests stay queued behind nothing.
    session = nansense.start(TinyClassifier(), epochs=1, phases={"train": 1})
    first = session.request_experiment(kind="deep_dream", layer="conv", params={})
    second = session.request_experiment(kind="deep_dream", layer="conv", params={})
    assert session.experiment_queue_state(first) == ExperimentQueueState("queued", 0)
    assert session.experiment_queue_state(second) == ExperimentQueueState("queued", 1)
    # Never-requested and cancelled seqs are equally absent.
    assert session.experiment_queue_state(9999).stage == "absent"
    session.cancel_experiment(first)
    assert session.experiment_queue_state(first).stage == "absent"
    # The survivor moves up the line.
    assert session.experiment_queue_state(second) == ExperimentQueueState("queued", 0)


def test_experiment_queue_state_counts_the_running_request() -> None:
    # The running request is a wait of its own: a queued one behind it is
    # one deeper than its queue position suggests.
    session = nansense.start(TinyClassifier(), epochs=1, phases={"train": 1})
    running = session.request_experiment(kind="deep_dream", layer="conv", params={})
    queued = session.request_experiment(kind="deep_dream", layer="conv", params={})
    # Mimic the pause loop picking the first request up.
    session._experiment_queue.popleft()
    session._experiment_running = running
    assert session.experiment_queue_state(running) == ExperimentQueueState("running", 0)
    assert session.experiment_queue_state(queued) == ExperimentQueueState("queued", 1)


def test_auto_experiments_awaiting_their_turn_stay_queued_and_cancellable() -> None:
    """One publish runs every registration in turn, and the page of each one
    still waiting polls its state meanwhile — "gone" would read as a request
    that died, and a Cancel on it must not be a silent no-op."""
    session = nansense.start(TinyClassifier(), epochs=1, phases={"train": 1})
    seqs = [
        session.register_auto_experiment(
            f"page-{i}", kind="deep_dream", layer="conv", params=_dream_params(steps=1)
        )
        for i in range(3)
    ]
    seen: list[tuple[str, ...]] = []

    def observe(_session: Session, request: object) -> None:
        # Stands in for the run itself: records how every request reads at
        # the moment this one starts.
        seen.append(tuple(session.experiment_queue_state(s).stage for s in seqs))
        if len(seen) == 1:
            session.cancel_experiment(seqs[2])  # while it waits its turn

    with mock.patch.object(experiments, "run_experiment_guarded", observe):
        experiments.run_auto_experiments(session)

    # The two that hadn't started yet read as queued, not as gone.
    assert seen[0] == ("running", "queued", "queued")
    assert seen[1][1] == "running"  # the second one's turn
    # The cancel bit: the third was skipped, so only two ever started.
    assert len(seen) == 2


def test_auto_run_experiments_setting_defaults_on() -> None:
    session = nansense.start(TinyClassifier(), epochs=1, phases={"train": 1})
    assert session.auto_run_experiments is True
    session.set_auto_run_experiments(False)
    assert session.auto_run_experiments is False


def test_experiment_defaults_start_empty_and_accumulate() -> None:
    session = nansense.start(TinyClassifier(), epochs=1, phases={"train": 1})
    assert session.experiment_defaults == {}
    session.set_experiment_defaults(steps=150)
    session.set_experiment_defaults(channels=4)
    assert session.experiment_defaults == {"steps": 150, "channels": 4}
    # The getter hands out a copy — mutating it never leaks into the session.
    session.experiment_defaults["steps"] = 1
    assert session.experiment_defaults["steps"] == 150


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


def test_experiment_params_cover_every_kind() -> None:
    from nansense.experiments import EXPERIMENT_KINDS
    from nansense.experiments import EXPERIMENT_PARAMS

    assert set(EXPERIMENT_PARAMS) == set(EXPERIMENT_KINDS)
    for kind, specs in EXPERIMENT_PARAMS.items():
        assert specs, kind  # every experiment exposes at least one knob
        for spec in specs:
            assert spec.kind in ("int", "float", "bool", "select"), spec.key
            if spec.kind == "select":
                assert spec.options and spec.default in spec.options, spec.key
            if spec.kind in ("int", "float"):
                assert isinstance(spec.default, (int, float)), spec.key


def test_experiment_param_order_targeting_then_inputs() -> None:
    from nansense.experiments import EXPERIMENT_PARAMS

    for kind, specs in EXPERIMENT_PARAMS.items():
        keys = [s.key for s in specs]
        if kind == "deep_dream":
            # Channels first; Start from with Sample directly below it (Sample's
            # visibility follows Start). Minimize sits directly above Clamp.
            assert keys[0] == "channels", kind
            start_idx = keys.index("start")
            assert keys[start_idx + 1] == "sample", kind
            assert keys[keys.index("clamp") - 1] == "minimize", kind
        else:
            # Captum: the targeting knob first, Inputs directly below it.
            assert keys[0] in ("channel", "target"), kind
            assert keys[1] == "batch", kind


def test_deep_dream_exposes_minimize_toggle() -> None:
    from nansense.experiments import EXPERIMENT_PARAMS

    specs = {s.key: s for s in EXPERIMENT_PARAMS["deep_dream"]}
    assert "minimize" in specs
    minimize = specs["minimize"]
    assert minimize.kind == "bool" and minimize.default is False


def testdefault_param_values_apply_session_overrides() -> None:
    from nansense.experiments import EXPERIMENT_PARAMS, default_param_values

    plain = default_param_values({})
    assert plain["steps"] == 300 and plain["channels"] == 8
    seeded = default_param_values({"steps": 150, "channels": 4})
    assert seeded["steps"] == 150 and seeded["channels"] == 4
    # Everything not overridden keeps its built-in default, and every knob
    # of every kind gets a value.
    assert seeded["lr"] == plain["lr"]
    every_key = {s.key for specs in EXPERIMENT_PARAMS.values() for s in specs}
    assert set(seeded) == every_key


def test_experiment_descriptions_cover_every_kind() -> None:
    from nansense.experiments import EXPERIMENT_DESCRIPTIONS, EXPERIMENT_KINDS

    assert set(EXPERIMENT_DESCRIPTIONS) == set(EXPERIMENT_KINDS)
    for short, long in EXPERIMENT_DESCRIPTIONS.values():
        assert short and long  # both a tooltip and a pane description
    # Neuron Gradient calls out its grainy maps (point 4).
    assert "grain" in EXPERIMENT_DESCRIPTIONS["neuron_gradient"][1].lower()
