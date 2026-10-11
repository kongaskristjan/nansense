"""NaNsense — visualization library for deep learning experiments.

The library provides a `Session` that hooks into a PyTorch model and
publishes per-batch activation/gradient snapshots for inspection in a
web UI. It deliberately contains no training logic; training lives in
`examples/`.
"""

from __future__ import annotations

from nansense.debugger import DebugError, DebugSettings, LayerReport
from nansense.experiments import ExperimentParam, Table
from nansense.extensions import ExperimentContext, ExperimentOutput, PauseContext
from nansense.instruments import (
    LayerContext,
    MetricSeries,
    MetricsSnapshot,
    WeightContext,
)
from nansense.moments import MomentError, load_moment
from nansense.restore import (
    TimeTravelError,
    TimeTravelJump,
    TimeTravelStatus,
    TrainingRestorer,
)
from nansense.schedule import BatchPosition, Schedule
from nansense.session import BatchSnapshot, Mode, Session, StatsScope, start
from nansense.ui import serve
from nansense.watch import (
    LayerStatsSnapshot,
    TensorStatsSnapshot,
    WatchSnapshot,
)

__all__ = [
    "BatchPosition",
    "BatchSnapshot",
    "DebugError",
    "DebugSettings",
    "ExperimentContext",
    "ExperimentOutput",
    "ExperimentParam",
    "LayerContext",
    "LayerReport",
    "LayerStatsSnapshot",
    "MetricSeries",
    "MetricsSnapshot",
    "Mode",
    "MomentError",
    "PauseContext",
    "Schedule",
    "Session",
    "StatsScope",
    "Table",
    "TensorStatsSnapshot",
    "TimeTravelError",
    "TimeTravelJump",
    "TimeTravelStatus",
    "TrainingRestorer",
    "WatchSnapshot",
    "WeightContext",
    "load_moment",
    "serve",
    "start",
]
