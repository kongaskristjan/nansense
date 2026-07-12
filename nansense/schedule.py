"""Schedule tracking: where in (phase, epoch) the current batch sits.

The schedule can be **declared** up front (`Schedule(epochs=N, phases={...})`)
or **discovered lazily** (`Schedule()` with no phases). When declared, a batch
knows on arrival whether it is the last of its phase/epoch/run, so the session
can decide *before* the forward pass whether to install hooks. When lazy, phase
names appear as `advance` first sees them and per-phase batch counts are learned
when a phase completes (`record_phase_length`), so those `is_last_*` flags only
become reliable from the second epoch on — the first epoch is the blind window.
The step modes that must work on the first epoch compare positions instead (see
`Session._should_capture`).
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class BatchPosition:
    """Where a batch sits in the run: the `phase` name plus 0-indexed `epoch`
    and `batch_idx`. This is the position record carried by
    `BatchSnapshot.position`, `DebugError.position` and `Session.live_position`.

    The `is_last_*` flags mark boundaries — the last batch of the phase, of
    the epoch, and of the whole run. They are best-effort: with a lazy
    schedule they stay `False` until the phase's batch count has been learned
    at the end of the first epoch.
    """

    phase: str
    epoch: int
    batch_idx: int
    is_last_in_phase: bool
    is_last_in_epoch: bool
    is_last_overall: bool


def format_position(
    position: BatchPosition,
    *,
    total_epochs: int | None = None,
    total_batches: int | None = None,
) -> str:
    """The "epoch 0 | train batch 0" rendering shared by the top bar's
    live-position label and recorded frames' position banner.

    `total_epochs` / `total_batches` are the run's known totals: when given,
    the count gets an "epoch 0/50" / "batch 0/196" suffix (both numbers stay
    0-indexed, matching the rest of the UI). Either is `None` while still
    unknown — a fully-lazy schedule before `session.epochs(n)`, or a phase's
    batch count during the first, unlearned epoch — and that part renders
    bare, so the suffix appears only once the total is real.
    """
    epoch = f"{position.epoch}/{total_epochs}" if total_epochs else f"{position.epoch}"
    batch = (
        f"{position.batch_idx}/{total_batches}"
        if total_batches
        else f"{position.batch_idx}"
    )
    return f"epoch {epoch} | {position.phase} batch {batch}"


class Schedule:
    """Batch-position bookkeeping, safe to touch from both session threads.

    `advance` and `record_phase_length` run on the training thread (every
    batch / each phase end) while the UI thread reads `epochs`/`phases`/
    `phase_order` and may `update`/`set_epochs`/`rewind_to_epoch`; a lock guards
    the shared counters and the phase/epoch fields so a mid-`advance` schedule
    swap can't yield an inconsistent `BatchPosition` or corrupt the counters.
    The lock is never held while touching the session's condition variable, so
    it only ever nests *inside* it — no lock-ordering cycle.

    `phases` may be `None` (lazy mode): phase names and counts are then learned
    by observation. A declared `phases` dict pins both up front and keeps the
    stricter validation (unknown phase / more batches than declared raise).
    """

    def __init__(
        self,
        epochs: int | None = None,
        phases: dict[str, int] | None = None,
    ) -> None:
        if epochs is not None and epochs <= 0:
            raise ValueError(f"epochs must be positive, got {epochs}")
        if phases is not None:
            self._validate_phases(phases)
        self._lock = threading.Lock()
        self._epochs = epochs
        # Declared mode pins the phase set and counts and keeps the strict
        # validation; lazy mode learns both and never raises on an unseen phase
        # or an over-count (the dataset may simply have grown).
        self._declared = phases is not None
        self._phase_order: list[str] = list(phases) if phases else []
        self._phase_counts: dict[str, int] = dict(phases) if phases else {}
        self._counters: dict[tuple[str, int], int] = {}

    @staticmethod
    def _validate_phases(phases: dict[str, int]) -> None:
        if not phases:
            raise ValueError("phases must be non-empty")
        for name, n in phases.items():
            if n <= 0:
                raise ValueError(
                    f"phase {name!r} must declare a positive batch count, got {n}"
                )

    @property
    def epochs(self) -> int | None:
        """Total epochs, or `None` until `set_epochs`/`session.epochs(n)` runs."""
        with self._lock:
            return self._epochs

    @property
    def phases(self) -> dict[str, int]:
        """Known phases mapped to their batch counts, in first-seen order.

        Declared mode returns the full dict up front; lazy mode grows it as each
        phase's count is learned (so it can be empty during the first epoch).
        """
        with self._lock:
            return {
                name: self._phase_counts[name]
                for name in self._phase_order
                if name in self._phase_counts
            }

    @property
    def phase_order(self) -> list[str]:
        """All phase names seen so far, in order — including ones whose count is
        not yet known (unlike `phases`, which only lists counted phases)."""
        with self._lock:
            return list(self._phase_order)

    def phase_count(self, phase: str) -> int | None:
        """The known batch count for `phase`, or `None` if not yet learned."""
        with self._lock:
            return self._phase_counts.get(phase)

    @property
    def first_phase_name(self) -> str | None:
        with self._lock:
            return self._phase_order[0] if self._phase_order else None

    @property
    def last_phase_name(self) -> str | None:
        with self._lock:
            return self._phase_order[-1] if self._phase_order else None

    def set_epochs(self, n: int) -> None:
        """Set the total epoch count (from `session.epochs(n)`)."""
        if n <= 0:
            raise ValueError(f"epochs must be positive, got {n}")
        with self._lock:
            self._epochs = n

    def rewind_to_epoch(self, epoch: int) -> None:
        """Forget batch counters for `epoch` and everything after it.

        Called when time travel jumps back to the start of `epoch`, so the
        re-run epochs advance from batch 0 again. The learned phase set and
        counts are kept (the shape is stable across the rewind).
        """
        with self._lock:
            for key in [k for k in self._counters if k[1] >= epoch]:
                del self._counters[key]

    def record_phase_length(self, phase: str, epoch: int, n: int) -> None:
        """Record the observed batch count of a just-completed phase (lazy mode).

        Called by `Session.batches` when its loop exhausts; the first
        observation teaches the count, later epochs update it if the dataset
        size changed. No-op in declared mode (counts are pinned up front).
        """
        if n <= 0:
            return
        with self._lock:
            if self._declared:
                return
            if phase not in self._phase_order:
                self._phase_order.append(phase)
            self._phase_counts[phase] = n

    def state_dict(self) -> dict[str, Any]:
        """The serializable schedule shape, for a frozen moment: the epoch
        count, phase order, and known per-phase batch counts (the totals
        behind the position label and the phase dropdowns)."""
        with self._lock:
            return {
                "epochs": self._epochs,
                "phase_order": list(self._phase_order),
                "phase_counts": dict(self._phase_counts),
            }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Restore a `state_dict()` shape (a frozen-moment reload).

        Batch counters reset — a restored moment drives no further batches —
        and the schedule stays undeclared, so nothing new is enforced."""
        epochs = state["epochs"]
        with self._lock:
            self._epochs = None if epochs is None else int(epochs)
            self._phase_order = [str(p) for p in state["phase_order"]]
            self._phase_counts = {
                str(name): int(n) for name, n in state["phase_counts"].items()
            }
            self._counters = {}

    def update(
        self, *, epochs: int | None = None, phases: dict[str, int] | None = None
    ) -> None:
        """Re-declare the schedule (Lightning re-declares per epoch).

        Passing `phases` switches to declared mode and replaces the phase set /
        counts; `_counters` are kept so the run does not restart.
        """
        with self._lock:
            if epochs is not None:
                if epochs <= 0:
                    raise ValueError(f"epochs must be positive, got {epochs}")
                self._epochs = epochs
            if phases is not None:
                self._validate_phases(phases)
                self._declared = True
                self._phase_order = list(phases)
                self._phase_counts = dict(phases)

    def advance(self, phase: str, epoch: int) -> BatchPosition:
        with self._lock:
            if self._epochs is not None and not 0 <= epoch < self._epochs:
                raise ValueError(f"epoch {epoch} out of range [0, {self._epochs})")
            if phase not in self._phase_order:
                if self._declared:
                    raise ValueError(
                        f"unknown phase {phase!r}; declared: {self._phase_order}"
                    )
                self._phase_order.append(phase)

            key = (phase, epoch)
            batch_idx = self._counters.get(key, 0)
            count = self._phase_counts.get(phase)  # None until learned/declared
            # Declared mode still rejects overrunning a pinned count; lazy mode
            # tolerates it (the count will be re-learned at phase end).
            if self._declared and count is not None and batch_idx >= count:
                raise ValueError(
                    f"more batches than declared for phase {phase!r} "
                    f"(declared {count}, got {batch_idx + 1})"
                )
            self._counters[key] = batch_idx + 1

            # All three flags are best-effort: they require the count to be
            # known (declared, or learned from a prior epoch). During the first
            # lazy epoch they stay False — position-based stepping covers it.
            is_last_in_phase = count is not None and batch_idx == count - 1
            is_last_in_epoch = (
                is_last_in_phase and phase == self._phase_order[-1]
            )
            is_last_overall = (
                is_last_in_epoch
                and self._epochs is not None
                and epoch == self._epochs - 1
            )
        return BatchPosition(
            phase=phase,
            epoch=epoch,
            batch_idx=batch_idx,
            is_last_in_phase=is_last_in_phase,
            is_last_in_epoch=is_last_in_epoch,
            is_last_overall=is_last_overall,
        )
