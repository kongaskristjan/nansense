"""MCP tools for the rest of the debugger: weights, probes, experiments,
time travel, recordings and settings.

These are the tools that *change* the session rather than reading it, so what
matters is that each one reaches the same `Session` method the matching UI
control does, and that the ones which cannot take effect say so instead of
returning a cheerful no-op — an agent has no button greying out to warn it.

`test_mcp` covers the orientation and run-control tools; `test_mcp_images`
covers the rendered views.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Coroutine
from pathlib import Path
from typing import Any, TypeVar

import torch
from mcp.client import Client
from mcp.types import TextContent
from torch import Tensor, nn

import nansense
from nansense.mcp_server import build_server
from nansense.mcp_views import (
    experiment_catalog_view,
    metrics_view,
    probe_view,
    recordings_view,
    settings_view,
    time_travel_view,
    weight_stats_view,
)
from nansense.session import Mode, Session

from .helpers import TinyNet, optimizer_train_step, paused_session, train_step

_T = TypeVar("_T")


def _run(coro: Coroutine[Any, Any, _T]) -> _T:
    return asyncio.run(coro)


def _call(session: Session, name: str, arguments: dict[str, Any] | None = None) -> Any:
    """Call one tool through a real MCP client and parse its JSON payload."""

    async def go() -> Any:
        async with Client(build_server(session)) as client:
            result = await client.call_tool(name, arguments or {})
            assert result.content, f"{name} returned no content"
            block = result.content[0]
            assert isinstance(block, TextContent)
            return json.loads(block.text)

    return _run(go())


class TinyClassifier(nn.Module):
    """Conv + classifier head over a small image; fx-traceable."""

    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(3, 4, kernel_size=3, padding=1)
        self.fc = nn.Linear(4 * 4 * 4, 3)

    def forward(self, x: Tensor) -> Tensor:
        return self.fc(torch.relu(self.conv(x)).flatten(1))


def _image_step(model: TinyClassifier) -> None:
    train_step(model, input_shape=(3, 4, 4), batch_size=2)


# --- weights ----------------------------------------------------------


def test_weight_stats_cover_values_gradients_and_optimizer_state() -> None:
    """The numeric half of the `/weights` page: the parameters themselves,
    which `get_layer_stats` (activations and their gradients) never shows."""
    model = TinyNet()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    with paused_session(
        model,
        lambda m: optimizer_train_step(m, optimizer),
        optimizer=optimizer,
    ) as session:
        view = weight_stats_view(session, layer="fc1")
        by_name = {entry["parameter"]: entry for entry in view["parameters"]}
        assert set(by_name) == {"fc1.weight", "fc1.bias"}
        weight = by_name["fc1.weight"]
        assert weight["weight"]["shape"] == [8, 4]
        assert isinstance(weight["weight"]["mean"], float)
        assert weight["gradient"] is not None
        # The live learning rate, not the one the optimizer was built with.
        assert weight["optimizer_hyperparameters"]["lr"] == 0.1


def test_weight_stats_explain_a_missing_optimizer() -> None:
    """No optimizer state has two very different causes, and an agent that
    reads "empty" as "the optimizer is broken" goes looking for nothing."""
    with paused_session(TinyNet()) as session:
        view = weight_stats_view(session, layer="fc1")
        assert "optimizer_note" in view
        assert "lazily" in view["optimizer_note"]


def test_weight_stats_separate_an_unknown_layer_from_a_weightless_one() -> None:
    with paused_session(TinyNet()) as session:
        assert "Unknown layer" in weight_stats_view(session, layer="nope")["error"]
        weightless = weight_stats_view(session, layer="relu")
        assert "no parameters" in weightless["error"]
        assert "Intermediates" in weightless["hint"]


def test_weight_stats_report_a_diverged_parameter_rather_than_nothing() -> None:
    """The same trap `tensor_stats_view` avoids for activations: scalars over
    an all-NaN tensor must not read as an empty one."""
    with paused_session(TinyNet()) as session:
        snapshot = session.snapshot
        assert snapshot is not None
        snapshot.weights["fc1.weight"] = torch.full((8, 4), float("nan"))
        summary = weight_stats_view(session, layer="fc1")["parameters"][0]
        blown = next(
            entry
            for entry in weight_stats_view(session, layer="fc1")["parameters"]
            if entry["parameter"] == "fc1.weight"
        )
        assert blown["weight"]["non_finite_count"] == 32
        assert "non-finite" in blown["weight"]["note"]
        assert summary is not None


# --- probes and perturbations ----------------------------------------


def test_pinning_a_batch_runs_a_probe_on_it() -> None:
    """A pin holds one input so stepping shows the model's changing response to
    a constant stimulus; the probe runs immediately while paused."""
    with paused_session(TinyClassifier(), _image_step) as session:
        view = _call(session, "pin_batch", {"timeout_seconds": 5.0})
        assert view["pinned"] is True
        assert view["active"] is True
        assert view["runs_completed"] >= 1
        assert view["pinned_position"]["phase"] == "train"


def test_pinning_with_nothing_captured_says_why() -> None:
    session = nansense.start(TinyNet(), epochs=1, phases={"train": 1})
    view = _call(session, "pin_batch", {"timeout_seconds": 0.5})
    assert "Nothing to pin" in view["error"]


def test_probe_mode_activates_probing_without_a_pin() -> None:
    """Eval mode is the interesting one — BatchNorm on running statistics —
    and selecting it is enough to start probing on its own."""
    with paused_session(TinyClassifier(), _image_step) as session:
        view = _call(
            session, "set_probe_mode", {"mode": "eval", "timeout_seconds": 5.0}
        )
        assert view["mode"] == "eval"
        assert view["active"] is True
        assert view["pinned"] is False


def test_a_perturbation_is_applied_and_reported() -> None:
    with paused_session(TinyClassifier(), _image_step) as session:
        view = _call(
            session,
            "add_perturbation",
            {"index": [1, 1], "values": [1.0, 0.0, -1.0], "timeout_seconds": 5.0},
        )
        assert "error" not in view
        assert view["perturbations"] == [
            {"input": "x", "sample": 0, "index": [1, 1], "values": [1.0, 0.0, -1.0]}
        ]
        # With edits active the strips switch to perturbed - original; an agent
        # reading a flat-zero strip needs to know that is the diff, not the data.
        assert "perturbed" in view["hint"]


def test_a_perturbation_that_does_not_fit_is_reported_not_swallowed() -> None:
    """Entries that don't match the base input are dropped at apply time, which
    on its own looks exactly like success."""
    with paused_session(TinyClassifier(), _image_step) as session:
        view = _call(
            session,
            "add_perturbation",
            # A 3-channel image needs three values, not one.
            {"index": [99, 99], "values": [1.0], "timeout_seconds": 5.0},
        )
        assert view["perturbations_applied"] is False
        assert "No perturbation reached the input" in view["error"]


def test_clearing_perturbations_returns_to_the_plain_input() -> None:
    with paused_session(TinyClassifier(), _image_step) as session:
        _call(
            session,
            "add_perturbation",
            {"index": [0, 0], "values": [1.0, 1.0, 1.0], "timeout_seconds": 5.0},
        )
        view = _call(session, "clear_perturbations")
        assert view["perturbations"] == []


def test_probe_status_of_an_idle_session_points_at_the_way_in() -> None:
    with paused_session(TinyNet()) as session:
        view = probe_view(session)
        assert view["active"] is False
        assert "pin_batch()" in view["hint"]


# --- experiments ------------------------------------------------------


def test_the_experiment_catalog_describes_kinds_params_and_layers() -> None:
    """Everything `run_experiment` needs, in one call: an agent cannot guess
    parameter keys, and the module-only kinds cannot run on fx intermediates."""
    with paused_session(TinyClassifier(), _image_step) as session:
        view = experiment_catalog_view(session)
        by_kind = {entry["kind"]: entry for entry in view["kinds"]}
        assert set(by_kind) >= {"deep_dream", "gradcam", "occlusion"}
        dream = by_kind["deep_dream"]
        assert dream["summary"] and dream["description"]
        assert {spec["key"] for spec in dream["params"]} >= {"steps", "channels"}
        # Grad-CAM needs an nn.Module, so the fx intermediate is excluded.
        assert "conv" in by_kind["gradcam"]["layers"]
        assert "relu" not in by_kind["gradcam"]["layers"]
        assert "relu" in by_kind["deep_dream"]["layers"]


def test_running_a_deep_dream_returns_its_result() -> None:
    with paused_session(TinyClassifier(), _image_step) as session:
        view = _call(
            session,
            "run_experiment",
            {
                "kind": "deep_dream",
                "layer": "conv",
                "params": {"channels": 1, "steps": 2},
                "timeout_seconds": 60.0,
            },
        )
        assert view["done"] is True, view
        assert view.get("error") is None
        assert view["produced"] == "image"
        assert view["params"]["steps"] == 2
        # The image travels as statistics here; `render_experiment` draws it.
        assert view["image"]["shape"][0] == 1


def test_unknown_experiment_params_are_reported_not_ignored() -> None:
    with paused_session(TinyClassifier(), _image_step) as session:
        view = _call(
            session,
            "run_experiment",
            {
                "kind": "deep_dream",
                "layer": "conv",
                "params": {"channels": 1, "steps": 2, "nonsense": 3},
                "timeout_seconds": 60.0,
            },
        )
        assert view["ignored_params"] == ["nonsense"]


def test_an_experiment_on_the_wrong_kind_of_layer_is_refused_with_the_reason() -> None:
    with paused_session(TinyClassifier(), _image_step) as session:
        view = _call(
            session, "run_experiment", {"kind": "gradcam", "layer": "relu"}
        )
        assert "nn.Module" in view["error"]
        assert "list_experiments" in view["hint"]


def test_an_unknown_layer_is_refused_before_the_request_is_armed() -> None:
    with paused_session(TinyClassifier(), _image_step) as session:
        view = _call(session, "run_experiment", {"kind": "deep_dream", "layer": "nope"})
        assert "Unknown layer" in view["error"]


def test_an_unknown_experiment_kind_never_reaches_the_session() -> None:
    """`kind` is an enum in the tool schema, so the SDK rejects a bad one before
    the handler runs — the agent gets the valid values back with the error."""

    async def go() -> Any:
        async with Client(build_server(session)) as client:
            return await client.call_tool(
                "run_experiment", {"kind": "nope", "layer": "conv"}
            )

    with paused_session(TinyClassifier(), _image_step) as session:
        result = _run(go())
        assert result.is_error, result.content


def test_an_experiment_result_that_was_never_published_says_so() -> None:
    with paused_session(TinyNet()) as session:
        view = _call(session, "get_experiment_result", {"seq": 999})
        assert "No result published" in view["error"]


# --- time travel ------------------------------------------------------


def _restorable(
    tmp_path: Path, *, epochs: int = 3
) -> tuple[Session, Any, TinyNet, torch.optim.SGD]:
    model = TinyNet()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    session = nansense.start(
        model, epochs=epochs, phases={"train": 2}, optimizer=optimizer
    )
    restorer = session.training_restorer(cache_dir=tmp_path / "cache")
    return session, restorer, model, optimizer


def test_time_travel_status_explains_an_unwired_loop() -> None:
    """Time travel needs the loop driven by `session.epochs()` /
    `restore_point()`; without it the answer is a reason, not `available: false`."""
    session = nansense.start(TinyNet(), epochs=2, phases={"train": 1})
    view = time_travel_view(session)
    assert view["available"] is False
    assert "restore_point" in view["reason"]


def test_time_travel_lists_the_epochs_it_can_reach(tmp_path: Path) -> None:
    session, restorer, _, _ = _restorable(tmp_path)
    restorer.save_epoch_start(0)
    restorer.save_epoch_start(1)
    view = time_travel_view(session)
    assert view["available"] is True
    assert view["cached_epochs"] == [0, 1]
    assert view["total_epochs"] == 3
    assert "discarded" in view["hint"]


def test_time_travel_arms_the_jump(tmp_path: Path) -> None:
    """With no worker thread nothing can reach the boundary, so the tool
    reports that it is still waiting — but the jump itself is armed."""
    session, restorer, _, _ = _restorable(tmp_path)
    restorer.save_epoch_start(0)
    view = _call(session, "time_travel", {"epoch": 0, "timeout_seconds": 0.0})
    assert view["travelled_to_epoch"] == 0
    assert session.mode is Mode.STEP
    assert "armed" in view["waiting"]


def test_time_travel_to_an_uncached_epoch_reports_the_reason(tmp_path: Path) -> None:
    session, restorer, _, _ = _restorable(tmp_path)
    restorer.save_epoch_start(0)
    view = _call(session, "time_travel", {"epoch": 2, "timeout_seconds": 0.0})
    assert "error" in view
    assert view["cached_epochs"] == [0]


# --- settings ---------------------------------------------------------


def test_settings_describe_what_each_knob_does() -> None:
    """An agent has no tooltips, so the meaning travels with the value."""
    with paused_session(TinyNet()) as session:
        view = settings_view(session)
        assert view["update_frequency"]["unit"] == "epoch"
        assert "without pausing" in view["update_frequency"]["description"]
        assert "flushes" in view["watch_performance"]["description"]


def test_update_frequency_rejects_a_phase_on_the_epoch_unit() -> None:
    """`set_update_frequency` raises for this; the tool must answer rather than
    let the exception surface as a tool error."""
    with paused_session(TinyNet()) as session:
        view = _call(
            session,
            "set_update_frequency",
            {"unit": "epoch", "n": 1, "phase": "train"},
        )
        assert "phase filter" in view["error"]
        assert view["known_phases"] == ["train"]


def test_watch_performance_change_warns_that_it_flushed_the_statistics() -> None:
    """Changing a cap reshapes the buffers and drops everything collected — an
    agent that then reads an empty history would conclude the run had reset."""
    with paused_session(TinyNet()) as session:
        session.watch("fc1")
        view = _call(session, "set_watch_performance", {"samples_per_channel": 3})
        assert view["statistics_flushed"] is True
        assert "dropped" in view["note"]
        assert view["watch_performance"]["samples_per_channel"] == 3


def test_watch_performance_reports_a_no_op_as_such() -> None:
    with paused_session(TinyNet()) as session:
        current = session.watch_performance.samples_per_channel
        view = _call(
            session, "set_watch_performance", {"samples_per_channel": current}
        )
        assert view["statistics_flushed"] is False
        assert "note" not in view


# --- custom metrics ---------------------------------------------------


def test_custom_metrics_come_back_as_series() -> None:
    """These exist only because the training script author wrote them, which
    makes their presence a signal in itself."""
    model = TinyNet()
    session = nansense.start(model, epochs=1, phases={"train": 1})
    # Nothing here drives the pause loop, so run detached rather than waiting
    # out the unserved-session timeout.
    session.detach()
    session.watch("fc1")

    @session.watch_metric("sparsity")
    def sparsity(ctx: Any) -> float:
        return float((ctx.activation > 0).float().mean())

    with session.batch(phase="train", epoch=0):
        train_step(model)

    view = metrics_view(session)
    assert view["series"], view
    entry = view["series"][0]
    assert entry["metric"] == "sparsity"
    assert entry["layer"] == "fc1"
    assert 0.0 <= entry["points"][0]["value"] <= 1.0


def test_no_custom_metrics_points_at_how_to_add_them() -> None:
    with paused_session(TinyNet()) as session:
        view = metrics_view(session)
        assert view["series"] == []
        assert "watch_metric" in view["hint"]


def test_auto_run_experiments_can_be_turned_off() -> None:
    """A browser page left on auto-run competes for the same paused training
    thread the agent's own experiments need."""
    with paused_session(TinyNet()) as session:
        assert settings_view(session)["auto_run_experiments"] is True
        view = _call(session, "set_auto_run_experiments", {"enabled": False})
        assert view["auto_run_experiments"] is False
        assert session.auto_run_experiments is False


# --- recordings -------------------------------------------------------


def test_starting_and_stopping_a_layer_recording(tmp_path: Path) -> None:
    from nansense.recording import RecordingManager

    with paused_session(TinyClassifier(), _image_step) as session:
        session._recording_manager = RecordingManager(directory=tmp_path)
        started = _call(session, "start_recording", {"view": "layers", "layers": ["conv"]})
        assert started["started"] == "main"
        assert [r["key"] for r in started["recordings"]] == ["main"]
        # One frame, driven the way the training thread drives them.
        session.recording.capture_frames(session)
        assert recordings_view(session)["recordings"][0]["frames"] == 1
        stopped = _call(session, "stop_recording", {"key": "main"})
        assert stopped["stopped"] == ["main"]
        assert stopped["files"] and Path(stopped["files"][0]).exists()


def test_a_second_recording_of_one_view_is_refused(tmp_path: Path) -> None:
    from nansense.recording import RecordingManager

    with paused_session(TinyClassifier(), _image_step) as session:
        session._recording_manager = RecordingManager(directory=tmp_path)
        _call(session, "start_recording", {"view": "layers", "layers": ["conv"]})
        again = _call(session, "start_recording", {"view": "layers", "layers": ["conv"]})
        assert "already recording" in again["error"]
        session.recording.delete_all()


def test_recording_a_view_with_no_data_is_refused_with_the_reason(
    tmp_path: Path,
) -> None:
    """Histogram and patch recordings read the watch accumulators; without a
    watched layer there is nothing to draw, every frame would be blank."""
    from nansense.recording import RecordingManager

    with paused_session(TinyNet()) as session:
        session._recording_manager = RecordingManager(directory=tmp_path)
        view = _call(session, "start_recording", {"view": "histograms"})
        assert "watch some layers first" in view["error"]


def test_weights_recording_needs_a_layer_with_parameters(tmp_path: Path) -> None:
    from nansense.recording import RecordingManager

    with paused_session(TinyNet()) as session:
        session._recording_manager = RecordingManager(directory=tmp_path)
        assert "Give `layer`" in _call(session, "start_recording", {"view": "weights"})["error"]
        view = _call(session, "start_recording", {"view": "weights", "layer": "relu"})
        assert "no parameters" in view["error"]


def test_stopping_a_recording_that_never_started_says_so(tmp_path: Path) -> None:
    from nansense.recording import RecordingManager

    with paused_session(TinyNet()) as session:
        session._recording_manager = RecordingManager(directory=tmp_path)
        view = _call(session, "stop_recording", {"key": "main"})
        assert "Nothing was recording" in view["error"]


def test_an_experiment_recording_keeps_its_request_rerunning(tmp_path: Path) -> None:
    """The page pins an auto experiment so every frame is a fresh rerun of the
    *same* seq; stopping must release it, or it reruns for the rest of the run."""
    from nansense.recording import RecordingManager

    with paused_session(TinyClassifier(), _image_step) as session:
        session._recording_manager = RecordingManager(directory=tmp_path)
        started = _call(
            session,
            "start_recording",
            {
                "view": "experiment",
                "layer": "conv",
                "kind": "deep_dream",
                "params": {"channels": 1, "steps": 2},
            },
        )
        assert started["started"] == "experiment:conv"
        # `expires_at is None` is what pins it: no heartbeat can expire it
        # while the recording holds the view.
        assert session._auto_experiments["experiment:conv"].expires_at is None
        _call(session, "stop_recording", {"key": "experiment:conv"})
        assert "experiment:conv" not in session._auto_experiments


def test_recordings_view_of_an_idle_session_explains_the_frame_source() -> None:
    with paused_session(TinyNet()) as session:
        view = recordings_view(session)
        assert view["recordings"] == []
        assert "visualization update" in view["hint"]


# --- histogram bin samples --------------------------------------------


def test_bin_samples_name_the_inputs_behind_a_bar() -> None:
    """A histogram says how many values fell in a bin; this says which — the
    step from "there is a spike in the overflow bin" to "these inputs cause it"."""
    from nansense.input_config import InputDisplay
    from nansense.mcp_images import bin_samples_image
    from nansense.watch import _bin_indices

    with paused_session(TinyClassifier(), _image_step) as session:
        snapshot = session.snapshot
        assert snapshot is not None
        activation = snapshot.activations["conv"]
        # Pick a bin that channel 0 actually populates, so the sampling has
        # something to find.
        bin_index = int(_bin_indices(activation[:, 0].reshape(-1))[0])
        rendered, values = bin_samples_image(
            session,
            layer="conv",
            channel=0,
            bin_index=bin_index,
            display=InputDisplay(),
            input_name="x",
        )
        assert values, rendered.note
        assert rendered.png is not None
        # The population is narrower than the bar, and saying so is the point.
        assert "last captured batch only" in rendered.note


def test_bin_samples_of_an_empty_bin_explain_the_narrower_population() -> None:
    """The bar may aggregate a whole epoch while only the last batch is still
    around to sample, so an empty answer is expected and must not read as a bug."""
    from nansense.input_config import InputDisplay
    from nansense.mcp_images import bin_samples_image

    with paused_session(TinyClassifier(), _image_step) as session:
        rendered, values = bin_samples_image(
            session,
            layer="conv",
            channel=0,
            bin_index=0,  # the extreme-negative end bin
            display=InputDisplay(),
            input_name="x",
        )
        assert values == []
        assert rendered.png is None
        assert "not retained" in rendered.note


def test_bin_samples_reach_the_agent_as_values_beside_the_picture() -> None:
    from nansense.watch import _bin_indices

    async def go() -> Any:
        async with Client(build_server(session)) as client:
            return await client.call_tool(
                "render_bin_samples",
                {"layer": "conv", "channel": 0, "bin_index": bin_index},
            )

    with paused_session(TinyClassifier(), _image_step) as session:
        snapshot = session.snapshot
        assert snapshot is not None
        bin_index = int(_bin_indices(snapshot.activations["conv"][:, 0].reshape(-1))[0])
        result = _run(go())
        texts = [b.text for b in result.content if getattr(b, "type", "") == "text"]
        images = [b for b in result.content if getattr(b, "type", "") == "image"]
        assert len(images) == 1
        payload = json.loads(texts[1])
        assert payload["samples"] and "value" in payload["samples"][0]


# --- per-epoch weight trend -------------------------------------------


def test_stats_history_includes_the_weight_trend() -> None:
    """The GRAPHS view plots weight statistics per epoch beside the activation
    ones; a run whose weights are drifting shows it here and nowhere else."""
    from nansense.mcp_views import stats_history_view

    model = TinyNet()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    session = nansense.start(
        model, epochs=1, phases={"train": 2}, optimizer=optimizer
    )
    session.detach()
    session.watch("fc1")
    for _ in range(2):
        with session.batch(phase="train", epoch=0):
            optimizer_train_step(model, optimizer)

    view = stats_history_view(session, layer="fc1")
    assert "fc1.weight" in view["weight_history"]
    point = view["weight_history"]["fc1.weight"][0]
    assert point["epoch"] == 0
    assert isinstance(point["mean"], float)
    assert "first watched batch" in view["weight_note"]
