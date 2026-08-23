"""User-registered instruments: custom per-layer scalar metrics and tensors.

An *instrument* is a user callback the session evaluates for every layer the
stats scope collects (the watched set by default), on the training thread,
against the live device tensors. Three kinds exist, registered through the
`Session.watch_*` decorators:

- **metric** (`Session.watch_metric`) — returns a scalar (or a mapping of
  named scalars) per layer. Evaluated on every stats batch; `on="batch"`
  keeps each batch's value as its own point, `on="epoch"` folds the epoch's
  values through a reduction into one point. Both plot in the `/stats`
  GRAPHS view, one figure per metric below the built-in stats.
- **layer tensor** (`Session.watch_layer_tensor`) — returns a tensor shaped
  like the layer's activation. Evaluated on publish batches only and carried
  on the `BatchSnapshot` (`custom_activations`), rendered as an extra strip
  under the activation/gradient strips on the main page's layer card.
- **weight tensor** (`Session.watch_weight_tensor`) — returns a tensor shaped
  like a parameter, evaluated once per (layer, parameter) on publish batches
  and carried on `BatchSnapshot.custom_weight_tensors`, rendered on the
  `/weights` page next to the weight/gradient/optimizer strips.

The callback may be any callable — a plain function or a stateful object
implementing `__call__`. A stateful callable may also expose an optional
`on_rewind(epoch)` method, called when time travel rewinds to `epoch`, so
running state never leaks across abandoned timelines. Callbacks run under
`torch.no_grad()` and must not mutate the tensors they receive.

**Error isolation.** A raising (or invalid-result) instrument never kills
the training thread: the first failure disables the instrument, records the
error (`Session.instrument_errors`, also surfaced on the `/stats` page), and
prints one console line. Already-collected data stays browsable.

**Threading.** Registration and the store share one lock. Per-batch
evaluation reads a lock-free tuple cache of the enabled instruments (rebuilt
under the lock on register/disable), so an idle registry costs one attribute
read per batch. `metrics_snapshot()` copies the raw series under the lock and
assembles the frozen snapshot outside it, mirroring `WatchAccumulator`.

In distributed runs instruments are leader-only (rank-local, like the patch
buffers): followers never evaluate them.
"""

from __future__ import annotations

import math
import threading
from array import array
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal, cast

import torch
from torch import Tensor, nn

from nansense.console import console_print

InstrumentKind = Literal["metric", "layer_tensor", "weight_tensor"]
MetricCadence = Literal["batch", "epoch"]

# A metric callback may return a plain number, a 1-element tensor, a mapping
# of named scalars (one plot trace per key), or `None` to skip the layer.
MetricValue = float | int | Tensor
MetricResult = MetricValue | Mapping[str, MetricValue] | None

# `on="epoch"` reduction: a named fold or any `values -> float` callable.
MetricReduce = str | Callable[[Sequence[float]], float]

_NAMED_REDUCES: dict[str, Callable[[Sequence[float]], float]] = {
    "mean": lambda values: math.fsum(values) / len(values),
    "sum": lambda values: math.fsum(values),
    "min": min,
    "max": max,
    "last": lambda values: values[-1],
}


@dataclass(frozen=True)
class LayerContext:
    """One watched layer's live tensors for a batch, passed to instruments.

    `activation` / `gradient` are the layer's captured output and its
    retained grad (`None` in forward-only phases); `weights`,
    `weight_gradients`, and `optimizer_state` cover the layer's own
    parameters, keyed by qualified parameter name. All tensors are the live
    training-device tensors — treat them as read-only. `module` is the
    owning `nn.Module`, or `None` for fx intermediates (`relu`, `add`) and
    graph inputs.
    """

    layer: str
    phase: str
    epoch: int
    batch_idx: int
    module: nn.Module | None
    activation: Tensor
    gradient: Tensor | None
    weights: dict[str, Tensor]
    weight_gradients: dict[str, Tensor]
    optimizer_state: dict[str, dict[str, Tensor | float]]


@dataclass(frozen=True)
class WeightContext:
    """One (layer, parameter) pair's live tensors, passed to weight-tensor
    instruments once per parameter on publish batches.

    `weight` / `gradient` are the live parameter and its `.grad` (`None`
    before the first backward); `optimizer_state` is this parameter's state
    entries (empty without an optimizer, or before its lazy init) and
    `hyperparams` its param group's numeric knobs (`lr`, `momentum`, ...).
    All tensors are live training-device tensors — treat them as read-only.
    """

    layer: str
    param: str
    phase: str
    epoch: int
    batch_idx: int
    module: nn.Module | None
    weight: Tensor
    gradient: Tensor | None
    optimizer_state: dict[str, Tensor | float]
    hyperparams: dict[str, float]


@dataclass(frozen=True)
class MetricSeries:
    """One plot trace of a custom metric: points ordered by (epoch, batch).

    `xs` is the shared epoch-fraction axis: an `on="epoch"` point sits at
    its integer epoch, an `on="batch"` point at `epoch + batch_idx / n`
    where `n` spans the epoch's observed batches — so batch series line up
    under the per-epoch GRAPHS figures. `batches[i]` is `None` for reduced
    epoch points. Values are stored as-is; render paths gap non-finite ones.
    """

    on: MetricCadence
    xs: tuple[float, ...]
    epochs: tuple[int, ...]
    batches: tuple[int | None, ...]
    values: tuple[float, ...]


@dataclass(frozen=True)
class MetricsSnapshot:
    """Immutable view of every custom-metric series at a point in time.

    Keyed by `(layer, phase, metric, series)` — `series` is the mapping key
    for dict-returning metrics and `""` for plain scalar returns.
    """

    series: dict[tuple[str, str, str, str], MetricSeries] = field(
        default_factory=dict
    )

    def plots(
        self, layer: str, phase: str
    ) -> dict[str, dict[str, MetricSeries]]:
        """`metric -> series name -> series` for one layer and phase.

        The per-figure view behind the GRAPHS custom-metric plots, in sorted
        (metric, series) order so trace order is stable across refreshes.
        """
        out: dict[str, dict[str, MetricSeries]] = {}
        for (l, ph, metric, series), s in sorted(
            self.series.items(), key=lambda kv: kv[0]
        ):
            if l == layer and ph == phase:
                out.setdefault(metric, {})[series] = s
        return out


@dataclass
class _Instrument:
    """One registered instrument: identity, callback, and failure state."""

    name: str
    kind: InstrumentKind
    fn: Callable[..., Any]
    on: MetricCadence = "batch"
    reduce: MetricReduce = "mean"
    # Set by the first failure; a non-None error disables the instrument
    # (it drops out of the evaluation caches) but keeps its collected data.
    error: str | None = None


@dataclass
class _SeriesData:
    """Raw per-batch samples of one (layer, phase, epoch, metric, series)."""

    batches: array  # array("q") of batch indices
    values: array  # array("d") of samples

    @staticmethod
    def empty() -> _SeriesData:
        return _SeriesData(batches=array("q"), values=array("d"))


def _cpu_copy(t: Tensor) -> Tensor:
    """An independent CPU copy (never an alias, unlike `Tensor.cpu()`)."""
    return t.detach().to("cpu", copy=True)


def _scalar_value(value: object) -> float:
    """Coerce a metric return into a float; raises `TypeError` otherwise."""
    if isinstance(value, Tensor):
        if value.numel() != 1:
            raise TypeError(
                "expected a scalar, got a tensor of shape "
                f"{tuple(value.shape)}"
            )
        return float(value.detach().item())
    if isinstance(value, (int, float)):
        return float(value)
    raise TypeError(f"expected a number, got {type(value).__name__}")


def _scalar_series(result: object) -> dict[str, float]:
    """Split a metric return into `series name -> value` (`""` = unnamed)."""
    if isinstance(result, Mapping):
        return {str(key): _scalar_value(value) for key, value in result.items()}
    return {"": _scalar_value(result)}


class InstrumentManager:
    """Registry + scalar-metric store shared across threads (see module doc)."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._instruments: dict[str, _Instrument] = {}
        self._series: dict[tuple[str, str, int, str, str], _SeriesData] = {}
        # Lock-free per-batch caches of the enabled instruments per kind;
        # rebuilt under the lock whenever the enabled set changes.
        self._metrics: tuple[_Instrument, ...] = ()
        self._layer_tensors: tuple[_Instrument, ...] = ()
        self._weight_tensors: tuple[_Instrument, ...] = ()
        # Series restored from a frozen moment (browse-only; never grows).
        self._restored: MetricsSnapshot | None = None

    def register(
        self,
        name: str,
        *,
        kind: str,
        fn: Callable[..., Any],
        on: str = "batch",
        reduce: MetricReduce | None = None,
    ) -> None:
        """Add an instrument; raises `ValueError` on invalid arguments.

        Names are unique across all kinds — they label plots and strips, so
        a collision would merge unrelated instruments in the UI.
        """
        if not name:
            raise ValueError("instrument name must be a non-empty string")
        if not callable(fn):
            raise ValueError(f"instrument '{name}' must be callable")
        if kind not in ("metric", "layer_tensor", "weight_tensor"):
            raise ValueError(f"unknown instrument kind {kind!r}")
        if on not in ("batch", "epoch"):
            raise ValueError(f"on must be 'batch' or 'epoch', got {on!r}")
        if reduce is not None and on == "batch":
            raise ValueError("reduce only applies to on='epoch' metrics")
        if reduce is None:
            reduce = "mean"
        if isinstance(reduce, str) and reduce not in _NAMED_REDUCES:
            raise ValueError(
                f"unknown reduce {reduce!r}; expected one of "
                f"{sorted(_NAMED_REDUCES)} or a callable"
            )
        with self._lock:
            if name in self._instruments:
                raise ValueError(f"instrument {name!r} is already registered")
            self._instruments[name] = _Instrument(
                name=name,
                kind=cast(InstrumentKind, kind),
                fn=fn,
                on=cast(MetricCadence, on),
                reduce=reduce,
            )
            self._rebuild_caches_locked()

    def _rebuild_caches_locked(self) -> None:
        enabled = [i for i in self._instruments.values() if i.error is None]
        self._metrics = tuple(i for i in enabled if i.kind == "metric")
        self._layer_tensors = tuple(
            i for i in enabled if i.kind == "layer_tensor"
        )
        self._weight_tensors = tuple(
            i for i in enabled if i.kind == "weight_tensor"
        )

    def _disable(self, instrument: _Instrument, error: str) -> None:
        """Record `error` and drop the instrument from the enabled caches."""
        with self._lock:
            if instrument.error is None:
                instrument.error = error
                self._rebuild_caches_locked()
        console_print(
            f"NaNsense: {instrument.kind} instrument {instrument.name!r} "
            f"disabled after an error: {error}"
        )

    def has_metrics(self) -> bool:
        """Whether any enabled scalar metric exists (lock-free, per batch)."""
        return bool(self._metrics)

    def has_layer_tensors(self) -> bool:
        """Whether any enabled layer-tensor instrument exists (lock-free)."""
        return bool(self._layer_tensors)

    def has_weight_tensors(self) -> bool:
        """Whether any enabled weight-tensor instrument exists (lock-free)."""
        return bool(self._weight_tensors)

    def errors(self) -> dict[str, str]:
        """`name -> error` for every disabled instrument."""
        with self._lock:
            return {
                name: inst.error
                for name, inst in self._instruments.items()
                if inst.error is not None
            }

    def run_metrics(self, ctx: LayerContext) -> None:
        """Evaluate every enabled scalar metric for `ctx` and store results.

        Training thread. A raising callback or an uncoercible return
        disables that metric; the others still run.
        """
        for instrument in self._metrics:
            try:
                with torch.no_grad():
                    result = instrument.fn(ctx)
                if result is None:
                    continue
                series = _scalar_series(result)
            except Exception as e:  # noqa: BLE001 — isolation is the point
                self._disable(instrument, self._error_text(e, at=ctx.layer))
                continue
            with self._lock:
                for series_name, value in series.items():
                    key = (
                        ctx.layer,
                        ctx.phase,
                        ctx.epoch,
                        instrument.name,
                        series_name,
                    )
                    data = self._series.get(key)
                    if data is None:
                        data = _SeriesData.empty()
                        self._series[key] = data
                    data.batches.append(ctx.batch_idx)
                    data.values.append(value)

    def run_layer_tensors(self, ctx: LayerContext) -> dict[str, Tensor]:
        """Evaluate the layer-tensor instruments for `ctx`.

        Training thread, publish batches only. Returns `metric name -> CPU
        clone`; results must share the activation's shape, anything else
        disables the instrument.
        """
        out: dict[str, Tensor] = {}
        for instrument in self._layer_tensors:
            tensor = self._run_tensor(
                instrument, ctx, like=ctx.activation, at=ctx.layer
            )
            if tensor is not None:
                out[instrument.name] = tensor
        return out

    def run_weight_tensors(self, ctx: WeightContext) -> dict[str, Tensor]:
        """Evaluate the weight-tensor instruments for one parameter.

        Training thread, publish batches only. Returns `metric name -> CPU
        clone`; results must share the parameter's shape.
        """
        out: dict[str, Tensor] = {}
        for instrument in self._weight_tensors:
            tensor = self._run_tensor(
                instrument, ctx, like=ctx.weight, at=ctx.param
            )
            if tensor is not None:
                out[instrument.name] = tensor
        return out

    def _run_tensor(
        self,
        instrument: _Instrument,
        ctx: LayerContext | WeightContext,
        *,
        like: Tensor,
        at: str,
    ) -> Tensor | None:
        try:
            with torch.no_grad():
                result = instrument.fn(ctx)
            if result is None:
                return None
            if not isinstance(result, Tensor):
                raise TypeError(
                    f"expected a tensor, got {type(result).__name__}"
                )
            if result.shape != like.shape:
                raise ValueError(
                    f"shape {tuple(result.shape)} does not match the "
                    f"{'activation' if isinstance(ctx, LayerContext) else 'weight'}"
                    f" shape {tuple(like.shape)}"
                )
            return _cpu_copy(result)
        except Exception as e:  # noqa: BLE001 — isolation is the point
            self._disable(instrument, self._error_text(e, at=at))
            return None

    @staticmethod
    def _error_text(e: Exception, *, at: str) -> str:
        return f"{type(e).__name__} at {at!r}: {e}"

    def notify_rewind(self, epoch: int) -> None:
        """Tell stateful callables about a time-travel rewind to `epoch`.

        Calls the optional `on_rewind(epoch)` hook on every enabled
        instrument's callable, so accumulated state from the abandoned
        timeline can be dropped. A raising hook disables its instrument.
        """
        with self._lock:
            enabled = [
                i for i in self._instruments.values() if i.error is None
            ]
        for instrument in enabled:
            hook = getattr(instrument.fn, "on_rewind", None)
            if not callable(hook):
                continue
            try:
                hook(epoch)
            except Exception as e:  # noqa: BLE001 — isolation is the point
                self._disable(instrument, self._error_text(e, at="on_rewind"))

    def forget_layer(self, layer: str) -> None:
        """Drop all stored metric series for `layer` (e.g. on unwatch)."""
        with self._lock:
            for key in list(self._series):
                if key[0] == layer:
                    del self._series[key]

    def retain_layers(self, layers: Iterable[str]) -> None:
        """Drop series for any layer not in `layers`.

        Same training-thread reaper contract as
        `WatchAccumulator.retain_layers`: an unwatch on the UI thread can
        race a batch that recreates the just-forgotten series, so the sole
        writer re-reaps before every update pass.
        """
        keep = set(layers)
        with self._lock:
            for key in list(self._series):
                if key[0] not in keep:
                    del self._series[key]

    def forget_epochs_from(self, epoch: int) -> None:
        """Drop series for `epoch` and later (a time-travel rewind)."""
        with self._lock:
            for key in list(self._series):
                if key[2] >= epoch:
                    del self._series[key]

    def _reduce(
        self, instrument_name: str, reduce: MetricReduce, values: list[float]
    ) -> float | None:
        """Fold an epoch's samples into its point; disables on a bad fold."""
        fold = _NAMED_REDUCES[reduce] if isinstance(reduce, str) else reduce
        try:
            return float(fold(values))
        except Exception as e:  # noqa: BLE001 — isolation is the point
            with self._lock:
                instrument = self._instruments.get(instrument_name)
            if instrument is not None:
                self._disable(instrument, self._error_text(e, at="reduce"))
            return None

    def metrics_snapshot(
        self, *, layers: Iterable[str] | None = None
    ) -> MetricsSnapshot:
        """Assemble the frozen per-series view (all layers by default).

        UI thread. Copies the raw arrays under the lock, then computes the
        epoch-fraction x positions and the `on="epoch"` reductions outside
        it. Series restored from a frozen moment are merged in (live data
        wins on a key collision).
        """
        wanted = set(layers) if layers is not None else None
        with self._lock:
            meta = {
                name: (inst.on, inst.reduce)
                for name, inst in self._instruments.items()
                if inst.kind == "metric"
            }
            data = {
                key: (list(d.batches), list(d.values))
                for key, d in self._series.items()
                if wanted is None or key[0] in wanted
            }
            restored = self._restored
        # The epoch-fraction denominator spans every metric's observed
        # batches for that (layer, phase, epoch), so all of an epoch's batch
        # points share one x scale.
        denom: dict[tuple[str, str, int], int] = {}
        for (layer, phase, epoch, _, _), (batches, _) in data.items():
            if batches:
                bucket = (layer, phase, epoch)
                denom[bucket] = max(denom.get(bucket, 1), max(batches) + 1)
        grouped: dict[
            tuple[str, str, str, str], list[tuple[int, list[int], list[float]]]
        ] = {}
        for (layer, phase, epoch, metric, series), (batches, values) in sorted(
            data.items()
        ):
            grouped.setdefault((layer, phase, metric, series), []).append(
                (epoch, batches, values)
            )
        out: dict[tuple[str, str, str, str], MetricSeries] = {}
        for key, per_epoch in grouped.items():
            layer, phase, metric, _ = key
            on, reduce = meta.get(metric, ("batch", "mean"))
            xs: list[float] = []
            epochs: list[int] = []
            batches_out: list[int | None] = []
            values_out: list[float] = []
            for epoch, batches, values in per_epoch:
                if not values:
                    continue
                if on == "epoch":
                    value = self._reduce(metric, reduce, values)
                    if value is None:
                        continue
                    xs.append(float(epoch))
                    epochs.append(epoch)
                    batches_out.append(None)
                    values_out.append(value)
                else:
                    scale = denom[(layer, phase, epoch)]
                    for batch, value in zip(batches, values):
                        xs.append(epoch + batch / scale)
                        epochs.append(epoch)
                        batches_out.append(batch)
                        values_out.append(value)
            out[key] = MetricSeries(
                on=on,
                xs=tuple(xs),
                epochs=tuple(epochs),
                batches=tuple(batches_out),
                values=tuple(values_out),
            )
        if restored is not None:
            merged = {
                key: series
                for key, series in restored.series.items()
                if wanted is None or key[0] in wanted
            }
            merged.update(out)
            out = merged
        return MetricsSnapshot(series=out)

    def state_dict(self) -> dict[str, Any]:
        """The full metric view as plain data, for a frozen moment.

        Stores the already-reduced snapshot rather than the raw arrays: a
        restored moment is browse-only, and a callable `reduce` cannot be
        serialized — its reduced points can.
        """
        snapshot = self.metrics_snapshot()
        return {
            "series": [
                {
                    "layer": layer,
                    "phase": phase,
                    "metric": metric,
                    "series": series_name,
                    "on": series.on,
                    "xs": list(series.xs),
                    "epochs": list(series.epochs),
                    "batches": list(series.batches),
                    "values": list(series.values),
                }
                for (layer, phase, metric, series_name), series in (
                    snapshot.series.items()
                )
            ]
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Restore a `state_dict()` as the browse-only series overlay."""
        series: dict[tuple[str, str, str, str], MetricSeries] = {}
        for record in state["series"]:
            on = str(record["on"])
            key = (
                str(record["layer"]),
                str(record["phase"]),
                str(record["metric"]),
                str(record["series"]),
            )
            series[key] = MetricSeries(
                on="epoch" if on == "epoch" else "batch",
                xs=tuple(float(x) for x in record["xs"]),
                epochs=tuple(int(e) for e in record["epochs"]),
                batches=tuple(
                    None if b is None else int(b) for b in record["batches"]
                ),
                values=tuple(float(v) for v in record["values"]),
            )
        with self._lock:
            self._restored = MetricsSnapshot(series=series)
