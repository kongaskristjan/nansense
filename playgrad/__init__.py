"""Playgrad — visualization library for deep learning experiments.

The library provides a `Session` that hooks into a PyTorch model and
publishes per-batch activation/gradient snapshots for inspection in a
web UI. It deliberately contains no training logic; training lives in
`examples/`.
"""

from __future__ import annotations

from playgrad.restore import (
    TimeTravelError,
    TimeTravelJump,
    TimeTravelStatus,
    TrainingRestorer,
)
from playgrad.schedule import BatchPosition, Schedule
from playgrad.session import BatchSnapshot, Mode, Session, start
from playgrad.ui import serve
from playgrad.watch import (
    LayerStatsSnapshot,
    TensorStatsSnapshot,
    WatchSnapshot,
)

__all__ = [
    "BatchPosition",
    "BatchSnapshot",
    "LayerStatsSnapshot",
    "Mode",
    "Schedule",
    "Session",
    "TensorStatsSnapshot",
    "TimeTravelError",
    "TimeTravelJump",
    "TimeTravelStatus",
    "TrainingRestorer",
    "WatchSnapshot",
    "serve",
    "start",
]
