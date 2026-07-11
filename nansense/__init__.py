"""Nansense — visualization library for deep learning experiments.

The library provides a `Session` that hooks into a PyTorch model and
publishes per-batch activation/gradient snapshots for inspection in a
web UI. It deliberately contains no training logic; training lives in
`examples/`.
"""

from __future__ import annotations

from nansense.debugger import DebugError, DebugSettings, LayerReport
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
    "LayerReport",
    "LayerStatsSnapshot",
    "Mode",
    "Schedule",
    "Session",
    "StatsScope",
    "TensorStatsSnapshot",
    "TimeTravelError",
    "TimeTravelJump",
    "TimeTravelStatus",
    "TrainingRestorer",
    "WatchSnapshot",
    "serve",
    "start",
]
