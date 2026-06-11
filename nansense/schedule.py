"""Schedule tracking: where in (phase, epoch) the current batch sits.

The schedule is fully declared at session construction time so that, when a
batch starts, we already know whether it is the last of its phase/epoch/run.
That lets the session decide *before* the forward pass whether to install
activation hooks for this batch, eliminating any reactive "phase just changed"
logic at `__exit__` time.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BatchPosition:
    phase: str
    epoch: int
    batch_idx: int
    is_last_in_phase: bool
    is_last_in_epoch: bool
    is_last_overall: bool


def format_position(position: BatchPosition) -> str:
    """The "epoch 0 | train batch 0" rendering shared by the top bar's
    live-position label and recorded frames' position banner."""
    return f"epoch {position.epoch} | {position.phase} batch {position.batch_idx}"


class Schedule:
    def __init__(self, epochs: int, phases: dict[str, int]) -> None:
        self._validate(epochs, phases)
        self._epochs = epochs
        self._phases: dict[str, int] = dict(phases)
        self._counters: dict[tuple[str, int], int] = {}

    @staticmethod
    def _validate(epochs: int, phases: dict[str, int]) -> None:
        if epochs <= 0:
            raise ValueError(f"epochs must be positive, got {epochs}")
        if not phases:
            raise ValueError("phases must be non-empty")
        for name, n in phases.items():
            if n <= 0:
                raise ValueError(f"phase {name!r} must declare a positive batch count, got {n}")

    @property
    def epochs(self) -> int:
        return self._epochs

    @property
    def phases(self) -> dict[str, int]:
        return dict(self._phases)

    @property
    def first_phase_name(self) -> str:
        return next(iter(self._phases))

    @property
    def last_phase_name(self) -> str:
        return next(reversed(self._phases))

    def rewind_to_epoch(self, epoch: int) -> None:
        """Forget batch counters for `epoch` and everything after it.

        Called when time travel jumps back to the start of `epoch`, so the
        re-run epochs advance from batch 0 again instead of tripping the
        more-batches-than-declared check.
        """
        for key in [k for k in self._counters if k[1] >= epoch]:
            del self._counters[key]

    def update(self, *, epochs: int | None = None, phases: dict[str, int] | None = None) -> None:
        new_epochs = self._epochs if epochs is None else epochs
        new_phases = self._phases if phases is None else dict(phases)
        self._validate(new_epochs, new_phases)
        self._epochs = new_epochs
        self._phases = new_phases

    def advance(self, phase: str, epoch: int) -> BatchPosition:
        if phase not in self._phases:
            raise ValueError(
                f"unknown phase {phase!r}; declared: {list(self._phases)}"
            )
        if not 0 <= epoch < self._epochs:
            raise ValueError(f"epoch {epoch} out of range [0, {self._epochs})")

        key = (phase, epoch)
        batch_idx = self._counters.get(key, 0)
        declared = self._phases[phase]
        if batch_idx >= declared:
            raise ValueError(
                f"more batches than declared for phase {phase!r} "
                f"(declared {declared}, got {batch_idx + 1})"
            )
        self._counters[key] = batch_idx + 1

        is_last_in_phase = batch_idx == declared - 1
        is_last_in_epoch = is_last_in_phase and phase == self.last_phase_name
        is_last_overall = is_last_in_epoch and epoch == self._epochs - 1
        return BatchPosition(
            phase=phase,
            epoch=epoch,
            batch_idx=batch_idx,
            is_last_in_phase=is_last_in_phase,
            is_last_in_epoch=is_last_in_epoch,
            is_last_overall=is_last_overall,
        )
