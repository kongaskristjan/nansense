"""Page decisions stay testable without constructing NiceGUI pages."""

import asyncio

import pytest

from nansense.session import StatsScope
from nansense.ui.controllers.experiment import ExperimentController
from nansense.ui.controllers.main import MainController
from nansense.ui.controllers.stats import (
    _PHASE_CURRENT_BATCH,
    _VIEW_GRAPHS,
    StatsController,
    _WatchPageState,
)
from tests.nansense.helpers import make_session


def test_layer_visibility_follows_scope_without_leaking_between_tabs() -> None:
    session, _ = make_session()
    session.watch("fc1")
    first = MainController(session, session.layer_names)
    second = MainController(session, session.layer_names)
    session.set_stats_scope(StatsScope.ALL)
    assert first.reconcile_scope() and second.reconcile_scope()
    first.toggle("fc2")
    assert first.shown_layers == {"fc1", "fc2"}
    assert second.shown_layers == session.watched_layers == {"fc1"}
    session.set_stats_scope(StatsScope.NONE)
    first.reconcile_scope()
    assert first.shown_layers == {"fc1", "fc2"}
    session.set_stats_scope(StatsScope.WATCHED)
    first.reconcile_scope()
    assert first.shown_layers == {"fc1"}
    first.toggle("fc2")
    assert second.shown_layers == {"fc1", "fc2"}


def test_experiment_replacement_and_cancel_are_isolated_to_the_page() -> None:
    session, _ = make_session()
    first = ExperimentController(session, session.layer_names, "fc1")
    second = ExperimentController(session, session.layer_names, "fc2")
    first.run({"steps": 1})
    second.run({"steps": 1})
    before = first.state.my_seq
    assert before is not None and second.state.my_seq is not None
    with pytest.raises(ValueError):
        first.run({"steps": "invalid"})
    assert first.state.my_seq == before
    assert session.experiment_queue_state(before).stage == "queued"
    first.run({"steps": 2})
    assert session.experiment_queue_state(before).stage == "absent"
    first.cancel()
    assert first.state.my_seq is not None
    assert session.experiment_queue_state(first.state.my_seq).stage == "absent"
    assert session.experiment_queue_state(second.state.my_seq).stage == "queued"


def test_experiment_edits_coalesce_and_recording_defers_the_pending_run() -> None:
    session, _ = make_session()
    session.set_auto_run_experiments(True)
    controller = ExperimentController(session, session.layer_names, "fc1")
    controller.schedule()
    controller.schedule()
    assert not controller.consume_auto_run(frozen=True)
    assert controller.consume_auto_run(frozen=False)
    assert not controller.consume_auto_run(frozen=False)


def test_graph_view_reconciles_current_batch_layer_to_aggregate_layers() -> None:
    session, _ = make_session()
    session.watch("fc1")
    state = _WatchPageState(selected_phase=_PHASE_CURRENT_BATCH, selected_layer="fc2")
    controller = StatsController(session, session.layer_names, state)
    assert "fc2" in controller.layer_options()
    controller.select_view(_VIEW_GRAPHS)
    assert state.selected_phase != _PHASE_CURRENT_BATCH
    assert state.selected_layer == "fc1"
    assert "fc2" not in controller.layer_options()


def test_refresh_coalesces_edits_and_recovers_after_render_failure() -> None:
    session, _ = make_session()
    controller = StatsController(session, session.layer_names, _WatchPageState())
    passes: list[int] = []

    async def render() -> None:
        passes.append(len(passes))
        if len(passes) == 1:
            await controller.refresh(render)
            await controller.refresh(render)

    async def fail() -> None:
        raise RuntimeError("renderer failed")

    async def exercise() -> None:
        await controller.refresh(render)
        assert passes == [0, 1]
        with pytest.raises(RuntimeError, match="renderer failed"):
            await controller.refresh(fail)
        await controller.refresh(render)
        assert passes == [0, 1, 2]

    asyncio.run(exercise())
