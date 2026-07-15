"""Tests for `Session.lock` — the shared-demo (playground) mode.

A locked session refuses run control, time travel, watch mutations, and
every global setting, pins the stats scope to `all`, and clamps/caps
experiment requests — while probes and experiments keep working.
"""

from __future__ import annotations

from collections.abc import Callable

import pytest

from nansense.experiments import _LOCKED_MAX_QUEUE
from nansense.restore import TimeTravelError
from nansense.session import Mode, Session, StatsScope, UpdateFrequency
from tests.nansense.helpers import make_session, paused_worker, train_step


def _locked_session() -> tuple[Session, object]:
    session, model = make_session(epochs=1, phases={"train": 2})
    session.lock()
    return session, model


def test_lock_forces_and_pins_the_all_scope() -> None:
    session, _model = _locked_session()
    assert session.locked
    assert session.stats_scope is StatsScope.ALL
    session.set_stats_scope("watched")
    assert session.stats_scope is StatsScope.ALL
    assert session.toggle_stats_collecting() is True
    assert session.stats_scope is StatsScope.ALL


@pytest.mark.parametrize(
    "control",
    [
        lambda s: s.step_batch(),
        lambda s: s.step_phase(),
        lambda s: s.step_epoch(),
        lambda s: s.step_run(),
        lambda s: s.step_until_position(phase_index=0, epoch=0, batch_idx=1),
        lambda s: s.detach(),
        lambda s: s.stop(),
    ],
)
def test_locked_run_controls_are_refused(
    control: Callable[[Session], None],
) -> None:
    session, _model = _locked_session()
    control(session)
    assert session.mode is Mode.STEP  # the construction-time mode, unchanged


def test_locked_settings_are_refused() -> None:
    session, _model = _locked_session()
    session.set_update_frequency(unit="batch", n=7)
    assert session.update_frequency == UpdateFrequency()
    assert session.set_watch_performance(channel_limit=1) is False
    assert session.watch_performance.channel_limit != 1
    before = session.debug_settings
    session.set_debug_settings(enabled=not before.enabled, interval=3)
    assert session.debug_settings == before
    session.disable_debug_check("nan_inf")
    assert session.debug_settings.check_nan_inf == before.check_nan_inf
    session.set_auto_run_experiments(False)
    assert session.auto_run_experiments is True
    session.set_experiment_defaults(steps=1)
    assert session.experiment_defaults == {}


def test_locked_watch_and_unwatch_are_refused() -> None:
    session, model = make_session(epochs=1, phases={"train": 2})
    assert session.watch("fc1")
    session.lock()
    assert session.watch("fc2") is False
    session.unwatch("fc1")
    assert session.watched_layers == frozenset({"fc1"})


def test_locked_time_travel_is_refused_with_a_reason() -> None:
    session, _model = _locked_session()
    with pytest.raises(TimeTravelError):
        session.request_time_travel(0)
    status = session.time_travel_status()
    assert status.available is False
    assert status.reason is not None and "demo" in status.reason


def test_locked_experiment_params_are_clamped() -> None:
    session, _model = _locked_session()
    seq = session.request_experiment(
        kind="deep_dream",
        layer="fc1",
        params={"steps": 100_000, "channels": 500, "lr": 0.1},
    )
    request = session._experiment_queue[-1]
    assert request.seq == seq
    assert request.params["steps"] == 300
    assert request.params["channels"] == 8
    assert request.params["lr"] == 0.1  # uncapped knobs pass through


def test_locked_experiment_queue_is_capped() -> None:
    session, _model = _locked_session()
    for _ in range(_LOCKED_MAX_QUEUE):
        session.request_experiment(kind="deep_dream", layer="fc1", params={})
    seq = session.request_experiment(kind="deep_dream", layer="fc1", params={})
    assert len(session._experiment_queue) == _LOCKED_MAX_QUEUE
    result = session.experiment_result_for(seq)
    assert result is not None and result.error is not None
    assert "queue is full" in result.error


def test_locked_probe_surface_is_refused() -> None:
    session, model = make_session(epochs=1, phases={"train": 2})
    session.lock()

    def loop() -> None:
        with session.batch(phase="train", epoch=0):
            train_step(model)

    with paused_worker(session, loop):
        # Pin, perturbations, and the forward mode are shared probe state,
        # so every mutation is refused.
        assert session.pin_current_batch() is False
        session.add_perturbation(
            input_name="x", sample=0, index=(0,), values=(1.0,)
        )
        assert session.perturbations == {}
        session.set_probe_mode("eval")
        assert session.probe_mode == "unchanged"
        session.close()


def test_a_pin_made_before_locking_sticks() -> None:
    session, model = make_session(epochs=1, phases={"train": 2})

    def loop() -> None:
        with session.batch(phase="train", epoch=0):
            train_step(model)

    with paused_worker(session, loop):
        assert session.pin_current_batch() is True
        session.lock()
        session.unpin_batch()
        session.clear_perturbations()
        assert session.is_pinned is True
        session.close()


def test_locked_session_still_runs_experiments() -> None:
    session, model = make_session(epochs=1, phases={"train": 2})
    session.lock()

    def loop() -> None:
        with session.batch(phase="train", epoch=0):
            train_step(model)

    with paused_worker(session, loop):
        # paused_worker's teardown detach() is refused on a locked session,
        # so resume via close() at the end instead.
        seq = session.request_experiment(
            kind="deep_dream",
            layer="fc1",
            params={"steps": 2, "channels": 2, "start": "sample"},
        )
        assert session.wait_for_experiment(timeout=10.0)
        result = session.experiment_result_for(seq)
        assert result is not None and result.done and result.error is None
        session.close()
