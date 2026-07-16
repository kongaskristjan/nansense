"""Tests for custom instruments: scalar metrics and layer/weight tensors."""

from __future__ import annotations

from collections.abc import Callable, Sequence

import pytest
import torch
from torch import Tensor

import nansense
from nansense.instruments import (
    InstrumentManager,
    LayerContext,
    MetricsSnapshot,
    WeightContext,
)
from tests.nansense.helpers import (
    TinyNet,
    make_session,
    optimizer_train_step,
    paused_worker,
    train_step,
)


def _ctx(
    *,
    layer: str = "fc1",
    phase: str = "train",
    epoch: int = 0,
    batch_idx: int = 0,
    activation: Tensor | None = None,
) -> LayerContext:
    return LayerContext(
        layer=layer,
        phase=phase,
        epoch=epoch,
        batch_idx=batch_idx,
        module=None,
        activation=activation if activation is not None else torch.ones(2, 3),
        gradient=None,
        weights={},
        weight_gradients={},
        optimizer_state={},
    )


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"name": ""}, "non-empty"),
        ({"kind": "bogus"}, "unknown instrument kind"),
        ({"on": "step"}, "'batch' or 'epoch'"),
        ({"on": "batch", "reduce": "mean"}, "only applies"),
        ({"on": "epoch", "reduce": "median"}, "unknown reduce"),
        ({"fn": 42}, "callable"),
    ],
)
def test_register_rejects_invalid_arguments(
    kwargs: dict[str, object], match: str
) -> None:
    manager = InstrumentManager()
    full: dict = {"name": "m", "kind": "metric", "fn": lambda ctx: 0.0}
    full.update(kwargs)
    with pytest.raises(ValueError, match=match):
        manager.register(**full)


def test_register_rejects_duplicate_names_across_kinds() -> None:
    manager = InstrumentManager()
    manager.register("m", kind="metric", fn=lambda ctx: 0.0)
    with pytest.raises(ValueError, match="already registered"):
        manager.register("m", kind="layer_tensor", fn=lambda ctx: None)


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (1, 1.0),
        (0.5, 0.5),
        (True, 1.0),
        (torch.tensor(2.5), 2.5),
    ],
)
def test_metric_returns_coerce_to_float(value: object, expected: float) -> None:
    manager = InstrumentManager()
    manager.register("m", kind="metric", fn=lambda ctx: value)
    manager.run_metrics(_ctx())
    snap = manager.metrics_snapshot()
    assert snap.plots("fc1", "train")["m"][""].values == (expected,)


def test_metric_mapping_returns_one_series_per_key() -> None:
    manager = InstrumentManager()
    manager.register("m", kind="metric", fn=lambda ctx: {"lo": 1.0, "hi": 2.0})
    manager.run_metrics(_ctx())
    series = manager.metrics_snapshot().plots("fc1", "train")["m"]
    assert set(series) == {"lo", "hi"}
    assert series["lo"].values == (1.0,) and series["hi"].values == (2.0,)


def test_metric_none_skips_the_layer() -> None:
    manager = InstrumentManager()
    manager.register("m", kind="metric", fn=lambda ctx: None)
    manager.run_metrics(_ctx())
    assert manager.metrics_snapshot().series == {}
    assert manager.errors() == {}


@pytest.mark.parametrize(
    "fn",
    [
        lambda ctx: "nope",  # uncoercible type
        lambda ctx: torch.ones(3),  # non-scalar tensor
        lambda ctx: (_ for _ in ()).throw(RuntimeError("boom")),
    ],
)
def test_bad_metric_is_disabled_and_others_keep_running(fn) -> None:
    manager = InstrumentManager()
    manager.register("bad", kind="metric", fn=fn)
    manager.register("good", kind="metric", fn=lambda ctx: 1.0)
    manager.run_metrics(_ctx(batch_idx=0))
    manager.run_metrics(_ctx(batch_idx=1))
    assert "bad" in manager.errors()
    assert "fc1" in manager.errors()["bad"]
    good = manager.metrics_snapshot().plots("fc1", "train")["good"][""]
    assert good.values == (1.0, 1.0)
    assert "bad" not in manager.metrics_snapshot().plots("fc1", "train")


@pytest.mark.parametrize(
    ("reduce", "expected"),
    [
        ("mean", 2.0),
        ("sum", 6.0),
        ("min", 1.0),
        ("max", 3.0),
        ("last", 3.0),
        (lambda values: float(values[0]), 1.0),
    ],
)
def test_epoch_metrics_reduce_to_one_point(
    reduce: str | Callable[[Sequence[float]], float], expected: float
) -> None:
    manager = InstrumentManager()
    values = iter([1.0, 2.0, 3.0])
    manager.register(
        "m", kind="metric", fn=lambda ctx: next(values), on="epoch", reduce=reduce
    )
    for batch_idx in range(3):
        manager.run_metrics(_ctx(batch_idx=batch_idx))
    series = manager.metrics_snapshot().plots("fc1", "train")["m"][""]
    assert series.on == "epoch"
    assert series.xs == (0.0,)
    assert series.batches == (None,)
    assert series.values == (expected,)


def test_raising_reduce_disables_the_metric() -> None:
    def bad_reduce(values: Sequence[float]) -> float:
        raise RuntimeError("fold failed")

    manager = InstrumentManager()
    manager.register(
        "m", kind="metric", fn=lambda ctx: 1.0, on="epoch", reduce=bad_reduce
    )
    manager.run_metrics(_ctx())
    snap = manager.metrics_snapshot()
    assert snap.plots("fc1", "train")["m"][""].values == ()
    assert "fold failed" in manager.errors()["m"]


def test_batch_metric_points_sit_at_epoch_fractions() -> None:
    manager = InstrumentManager()
    manager.register("m", kind="metric", fn=lambda ctx: 1.0)
    for epoch in range(2):
        for batch_idx in range(4):
            manager.run_metrics(_ctx(epoch=epoch, batch_idx=batch_idx))
    series = manager.metrics_snapshot().plots("fc1", "train")["m"][""]
    assert series.xs == (0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75)
    assert series.epochs == (0, 0, 0, 0, 1, 1, 1, 1)


def test_eviction_forget_retain_and_epochs() -> None:
    manager = InstrumentManager()
    manager.register("m", kind="metric", fn=lambda ctx: 1.0)
    for layer in ("fc1", "fc2"):
        for epoch in range(3):
            manager.run_metrics(_ctx(layer=layer, epoch=epoch))

    manager.forget_epochs_from(2)
    fc1 = manager.metrics_snapshot().plots("fc1", "train")["m"][""]
    assert fc1.epochs == (0, 1)

    manager.forget_layer("fc2")
    assert not manager.metrics_snapshot().plots("fc2", "train")

    manager.retain_layers([])
    assert manager.metrics_snapshot().series == {}


def test_state_dict_roundtrips_through_torch_save(tmp_path) -> None:
    manager = InstrumentManager()
    manager.register("m", kind="metric", fn=lambda ctx: 1.5, on="epoch")
    manager.run_metrics(_ctx())
    path = tmp_path / "instruments.pt"
    torch.save(manager.state_dict(), path)
    state = torch.load(path, weights_only=True)
    restored = InstrumentManager()
    restored.load_state_dict(state)
    series = restored.metrics_snapshot().plots("fc1", "train")["m"][""]
    assert series.on == "epoch"
    assert series.values == (1.5,)
    # Restored data survives layer filtering and merges under live data.
    assert restored.metrics_snapshot(layers=["fc2"]).series == {}


def test_layer_tensor_shape_mismatch_disables() -> None:
    manager = InstrumentManager()
    manager.register("bad", kind="layer_tensor", fn=lambda ctx: torch.ones(5))
    manager.register(
        "good", kind="layer_tensor", fn=lambda ctx: ctx.activation * 2
    )
    out = manager.run_layer_tensors(_ctx(activation=torch.ones(2, 3)))
    assert set(out) == {"good"}
    assert out["good"].shape == (2, 3)
    assert "shape" in manager.errors()["bad"]


def test_weight_tensor_results_are_cpu_copies() -> None:
    manager = InstrumentManager()
    manager.register("w", kind="weight_tensor", fn=lambda ctx: ctx.weight * 0)
    weight = torch.ones(3, 4)
    ctx = WeightContext(
        layer="fc1",
        param="fc1.weight",
        phase="train",
        epoch=0,
        batch_idx=0,
        module=None,
        weight=weight,
        gradient=None,
        optimizer_state={},
        hyperparams={},
    )
    out = manager.run_weight_tensors(ctx)
    assert out["w"].shape == weight.shape
    assert out["w"].device.type == "cpu"
    assert out["w"].data_ptr() != weight.data_ptr()


def test_notify_rewind_calls_optional_hook_and_isolates_errors() -> None:
    class Stateful:
        def __init__(self) -> None:
            self.rewinds: list[int] = []

        def __call__(self, ctx: LayerContext) -> float:
            return 0.0

        def on_rewind(self, epoch: int) -> None:
            self.rewinds.append(epoch)

    class Broken:
        def __call__(self, ctx: LayerContext) -> float:
            return 0.0

        def on_rewind(self, epoch: int) -> None:
            raise RuntimeError("hook boom")

    manager = InstrumentManager()
    stateful = Stateful()
    manager.register("s", kind="metric", fn=stateful)
    manager.register("b", kind="metric", fn=Broken())
    manager.notify_rewind(3)
    assert stateful.rewinds == [3]
    assert "hook boom" in manager.errors()["b"]


# --- session integration -------------------------------------------------


def test_session_records_metrics_for_watched_layers_only() -> None:
    session, model = make_session(epochs=1, phases={"train": 3})
    session.watch("fc1")

    @session.watch_metric("act_mean")
    def act_mean(ctx: nansense.LayerContext) -> float:
        return float(ctx.activation.mean())

    session.detach()
    for _ in range(3):
        with session.batch(phase="train", epoch=0):
            train_step(model)

    snap = session.watch_metrics_snapshot()
    fc1 = snap.plots("fc1", "train")["act_mean"][""]
    assert len(fc1.values) == 3
    assert not snap.plots("fc2", "train")  # not watched, never evaluated
    session.close()


def test_session_layer_context_carries_weights_and_gradient() -> None:
    session, model = make_session(epochs=1, phases={"train": 1})
    session.watch("fc1")
    seen: list[nansense.LayerContext] = []

    session.watch_metric("probe")(lambda ctx: seen.append(ctx) or 0.0)

    session.detach()
    with session.batch(phase="train", epoch=0):
        train_step(model)

    (ctx,) = seen
    assert ctx.layer == "fc1" and ctx.phase == "train"
    assert ctx.module is model.fc1
    assert ctx.gradient is not None
    assert set(ctx.weights) == {"fc1.weight", "fc1.bias"}
    assert set(ctx.weight_gradients) == {"fc1.weight", "fc1.bias"}
    session.close()


def test_session_snapshot_carries_custom_tensors() -> None:
    model = TinyNet()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    session = nansense.start(
        model, epochs=1, phases={"train": 1}, optimizer=optimizer
    )
    session.watch("fc1")

    @session.watch_layer_tensor("zscore")
    def zscore(ctx: nansense.LayerContext) -> Tensor:
        a = ctx.activation
        return (a - a.mean()) / (a.std() + 1e-6)

    @session.watch_weight_tensor("grad_ratio")
    def grad_ratio(ctx: nansense.WeightContext) -> Tensor | None:
        if ctx.gradient is None:
            return None
        return ctx.gradient.abs() / (ctx.weight.abs() + 1e-8)

    def loop() -> None:
        with session.batch(phase="train", epoch=0):
            optimizer_train_step(model, optimizer)

    with paused_worker(session, loop):
        snap = session.snapshot
        assert snap is not None
        assert snap.custom_activations["fc1"]["zscore"].shape == (
            snap.activations["fc1"].shape
        )
        # Weight tensors cover every parameter of the watched layer.
        assert (
            snap.custom_weight_tensors["fc1.weight"]["grad_ratio"].shape
            == snap.weights["fc1.weight"].shape
        )
        # Unwatched layers contribute nothing.
        assert "fc2" not in snap.custom_activations
        assert "fc2.weight" not in snap.custom_weight_tensors
    session.close()


def test_session_raising_metric_does_not_kill_training() -> None:
    session, model = make_session(epochs=1, phases={"train": 2})
    session.watch("fc1")

    @session.watch_metric("broken")
    def broken(ctx: nansense.LayerContext) -> float:
        raise RuntimeError("boom")

    session.detach()
    for _ in range(2):
        with session.batch(phase="train", epoch=0):
            train_step(model)

    assert "boom" in session.instrument_errors["broken"]
    session.close()


def test_session_unwatch_drops_metric_series() -> None:
    session, model = make_session(epochs=1, phases={"train": 1})
    session.watch("fc1")
    session.watch_metric("m")(lambda ctx: 1.0)
    session.detach()
    with session.batch(phase="train", epoch=0):
        train_step(model)
    assert session.watch_metrics_snapshot().plots("fc1", "train")
    session.unwatch("fc1")
    assert not session.watch_metrics_snapshot().plots("fc1", "train")
    session.close()


def test_session_scope_none_freezes_metric_collection() -> None:
    session, model = make_session(epochs=1, phases={"train": 2})
    session.watch("fc1")
    session.watch_metric("m")(lambda ctx: 1.0)
    session.detach()
    with session.batch(phase="train", epoch=0):
        train_step(model)
    session.set_stats_scope("none")
    with session.batch(phase="train", epoch=0):
        train_step(model)
    series = session.watch_metrics_snapshot().plots("fc1", "train")["m"][""]
    assert len(series.values) == 1  # the scope-"none" batch added nothing
    session.close()


def test_locked_session_refuses_registration() -> None:
    session, _ = make_session()
    session.lock()
    with pytest.raises(RuntimeError, match="locked"):
        session.watch_metric("m")
    session.close()


def test_metrics_snapshot_type_is_exported() -> None:
    assert isinstance(InstrumentManager().metrics_snapshot(), MetricsSnapshot)
    assert nansense.MetricsSnapshot is MetricsSnapshot
