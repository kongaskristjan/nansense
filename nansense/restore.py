"""Time travel: epoch-start checkpoints and training-loop restoration.

Time travel lets the UI jump training back to the start of any epoch whose
state was cached to disk. It is opt-in: the user wraps their epoch loop in a
`TrainingRestorer` (`session.training_restorer(...)`), which

1. saves a checkpoint of the training state (model, optimizer, scheduler,
   RNG) at the start of every epoch, and
2. catches the `TimeTravelJump` exception that the session raises out of the
   paused batch when the UI requests a jump, restores the cached state, and
   lets the user's `while restorer.pending():` loop re-enter the epoch loop
   at `restorer.start_epoch`.

Without a restorer the session never raises `TimeTravelJump`, nothing is
written to disk, and the UI's time-travel button is disabled.

The intended loop shape::

    restorer = session.training_restorer(cache_dir=Path("models/latest"))
    while restorer.pending():
        with restorer:
            for epoch in restorer.epochs():
                ...  # one epoch of training

`TimeTravelJump` subclasses `BaseException` (like `KeyboardInterrupt`) so a
user's ``except Exception`` around the batch body cannot swallow a jump.
"""

from __future__ import annotations

import contextlib
import ctypes
import ctypes.util
import gc
import re
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import torch

if TYPE_CHECKING:
    from nansense.session import Session

DEFAULT_CACHE_DIR = Path(".nansense_cache")


def _mps_available() -> bool:
    return hasattr(torch, "mps") and torch.backends.mps.is_available()


def release_cpu_memory() -> None:
    """Return freed CPU allocations to the OS after a checkpoint load.

    `torch.load` materializes the whole checkpoint in CPU memory — the model
    parameters plus, for Adam-family optimizers, two moment tensors per
    parameter (so roughly 3x the model size). Once it has been copied into the
    live model/optimizer the buffers are dropped, but glibc keeps the freed
    arenas mapped, so RSS sits at the load peak ("never given back") until
    later allocations happen to reuse it. `gc.collect()` breaks any reference
    cycles holding the tensors, and `malloc_trim` (glibc only) hands the arenas
    back, so a time-travel jump's peak is reclaimed at the jump rather than
    lingering. A no-op on platforms without `malloc_trim` (musl, macOS).
    """
    gc.collect()
    try:
        libc = ctypes.CDLL(ctypes.util.find_library("c") or "libc.so.6")
        libc.malloc_trim(0)
    except (OSError, AttributeError):
        pass


def capture_rng() -> dict[str, Any]:
    """Snapshot the global torch (and per-device CUDA / MPS) RNG states.

    The counterpart of `restore_rng`; stashed alongside checkpoints so a
    time-travel replay reproduces DataLoader shuffling and dropout. The MPS
    state is captured on Apple Silicon so replays there are deterministic
    too, not just on CPU/CUDA.

    This is the calling process's RNG. Under DDP each rank captures its own
    (into its own `epoch_<n>.rank<r>.pt`), so a jump replays every rank's
    stochastic layers/augmentation deterministically — independent of the
    `DistributedSampler` shard order, which `set_epoch` reproduces on its own.
    """
    return {
        "torch": torch.get_rng_state(),
        "cuda": (
            torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
        ),
        "mps": torch.mps.get_rng_state() if _mps_available() else None,
    }


def restore_rng(rng: dict[str, Any] | None) -> None:
    """Best-effort RNG restore so the replayed epochs are deterministic.

    Restoring the global torch RNG also reproduces DataLoader shuffling:
    the loader draws its base seed from the global generator when its
    iterator is created. CUDA states are restored only when the device
    count still matches what was saved; the MPS state is restored when the
    backend is available (checkpoints predating MPS capture simply omit it).
    """
    if rng is None:
        return
    torch_state = rng.get("torch")
    if isinstance(torch_state, torch.Tensor):
        torch.set_rng_state(torch_state.cpu().to(torch.uint8))
    cuda_states = rng.get("cuda")
    if (
        isinstance(cuda_states, list)
        and torch.cuda.is_available()
        and len(cuda_states) == torch.cuda.device_count()
    ):
        torch.cuda.set_rng_state_all(
            [s.cpu().to(torch.uint8) for s in cuda_states]
        )
    mps_state = rng.get("mps")
    if isinstance(mps_state, torch.Tensor) and _mps_available():
        torch.mps.set_rng_state(mps_state.cpu().to(torch.uint8))


class TimeTravelJump(BaseException):
    """Raised by the session inside a batch to unwind to the restorer.

    A `BaseException` on purpose: the jump must travel through the user's
    training code (which may contain broad ``except Exception`` handlers)
    all the way to the ``with restorer:`` block that suppresses it.
    """

    def __init__(self, epoch: int) -> None:
        super().__init__(f"time travel to start of epoch {epoch}")
        self.epoch = epoch


class TimeTravelError(RuntimeError):
    """A time-travel request that cannot be honored (shown in the UI)."""


@dataclass(frozen=True)
class TimeTravelStatus:
    """UI-facing view of whether/where time travel can jump.

    `available` is False when no restorer wraps the training loop (or the
    run already completed); `reason` then carries the human-readable
    explanation for the disabled button's tooltip. `cached_epochs` are the
    epochs with a loadable checkpoint on disk, restricted to the current
    schedule's range. `total_epochs` is the run's known epoch count (`0`
    while a lazy schedule hasn't learned it), bounding the jump target picker.
    """

    available: bool
    reason: str | None
    cached_epochs: list[int]
    total_epochs: int


class EpochCache:
    """Disk cache of epoch-start training state, one file per epoch.

    Files are named `epoch_<n>.pt` inside `directory` and contain the
    model/optimizer/scheduler state dicts plus RNG states. Writes go
    through a temp file + atomic rename so a killed process can't leave a
    torn checkpoint behind. An existing file for the same epoch is simply
    overwritten — retraining past an epoch replaces the older timeline's
    entry.

    Under DDP each rank owns a separate file: rank 0 keeps the canonical
    `epoch_<n>.pt` (so the UI's `cached_epochs` and single-process layout are
    unchanged), while follower rank `r` writes `epoch_<n>.rank<r>.pt`. The
    model/optimizer/scheduler state dicts are replicated across ranks (DDP
    keeps them identical), so each rank's own file fully restores its state;
    the RNG snapshot is per-rank, so each rank's stochastic layers replay
    deterministically. `cached_epochs` only enumerates rank 0's files — the
    leader is the only one that drives the UI.
    """

    # Checkpoint file extension; subclasses with a different on-disk format
    # (e.g. the Lightning cache's `.ckpt`) override just this.
    FILE_SUFFIX: str = ".pt"

    def __init__(self, directory: Path, *, rank: int = 0) -> None:
        # The directory is created lazily on first save, so merely creating
        # a restorer (e.g. on a disabled session) leaves the disk untouched.
        self.directory = directory
        # 0 (the leader / single-process) writes the canonical filename;
        # followers tag theirs with `.rank<r>` so the ranks never collide.
        self.rank = rank

    def path_for(self, epoch: int) -> Path:
        suffix = "" if self.rank == 0 else f".rank{self.rank}"
        return self.directory / f"epoch_{epoch}{suffix}{self.FILE_SUFFIX}"

    def cached_epochs(self) -> list[int]:
        # Only rank 0's files are enumerated — the leader drives the UI, and
        # every rank caches the same set of epochs in lockstep anyway.
        if not self.directory.is_dir():
            return []
        file_re = re.compile(rf"^epoch_(\d+){re.escape(self.FILE_SUFFIX)}$")
        epochs: list[int] = []
        for entry in self.directory.iterdir():
            m = file_re.match(entry.name)
            if m is not None:
                epochs.append(int(m.group(1)))
        return sorted(epochs)

    def save(
        self,
        epoch: int,
        *,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer | None,
        scheduler: torch.optim.lr_scheduler.LRScheduler | None,
    ) -> None:
        payload: dict[str, Any] = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict() if optimizer is not None else None,
            "scheduler": scheduler.state_dict() if scheduler is not None else None,
            "rng": capture_rng(),
        }
        self.directory.mkdir(parents=True, exist_ok=True)
        path = self.path_for(epoch)
        tmp = path.with_name(path.name + ".tmp")
        torch.save(payload, tmp)
        tmp.replace(path)

    def load(self, epoch: int, *, mmap: bool = False) -> dict[str, Any]:
        """Load epoch `epoch`'s checkpoint; raises `TimeTravelError` if absent.

        `mmap=True` memory-maps the tensor storages from the file instead of
        reading them into RAM. The validation path
        (`Session.request_time_travel`) uses it: it only inspects keys and
        shapes, so mmap avoids materializing a full copy of the model and
        optimizer state in CPU memory on every jump request. The training
        thread's `_restore` loads for real (the default) since it copies the
        values back into the live state.
        """
        path = self.path_for(epoch)
        if not path.exists():
            raise TimeTravelError(f"no cached model for epoch {epoch} ({path})")
        try:
            payload = torch.load(
                path, map_location="cpu", weights_only=True, mmap=mmap
            )
        except Exception as e:  # corrupt file, unpicklable content, ...
            raise TimeTravelError(f"failed to load {path}: {e}") from e
        if not isinstance(payload, dict) or "model" not in payload:
            raise TimeTravelError(f"{path} is not a NaNsense epoch checkpoint")
        return payload


def _state_dict(payload: dict[str, Any], key: str) -> dict[str, Any] | None:
    """The state dict stored under `key`, or None when absent / not a dict."""
    value = payload.get(key)
    return cast("dict[str, Any]", value) if isinstance(value, dict) else None


def _group_params(group: Any) -> list[Any]:
    """The `params` entry of an optimizer param group, or `[]` when malformed."""
    if isinstance(group, dict):
        params = group.get("params")
        if isinstance(params, list):
            return params
    return []


def validate_model_state(
    payload: dict[str, Any], model: torch.nn.Module
) -> str | None:
    """Check a cached model state dict against the live model's shape.

    Returns a human-readable mismatch description, or `None` when the cached
    state can be loaded. Catches the "cache written by a previous run with a
    different model" case before the jump is committed, so a failed load
    never unwinds the training loop.
    """
    cached = _state_dict(payload, "model")
    if cached is None:
        return "checkpoint has no model state"
    current = model.state_dict()
    missing = sorted(set(current) - set(cached))
    unexpected = sorted(set(cached) - set(current))
    mismatched = sorted(
        name
        for name, tensor in current.items()
        if name in cached and tuple(cached[name].shape) != tuple(tensor.shape)
    )
    problems: list[str] = []
    if missing:
        problems.append(f"missing keys: {', '.join(missing[:5])}")
    if unexpected:
        problems.append(f"unexpected keys: {', '.join(unexpected[:5])}")
    if mismatched:
        shapes = [
            f"{n} (cached {tuple(cached[n].shape)} vs model "
            f"{tuple(current[n].shape)})"
            for n in mismatched[:5]
        ]
        problems.append(f"shape mismatches: {', '.join(shapes)}")
    if problems:
        return (
            "cached model does not match the current model — " + "; ".join(problems)
        )
    return None


def validate_optimizer_state(
    payload: dict[str, Any], optimizer: torch.optim.Optimizer | None
) -> str | None:
    """Check a cached optimizer state dict against the live optimizer.

    Returns a human-readable mismatch description, or `None` when the cached
    state can be loaded safely. Like `validate_model_state`, this runs at
    request time (UI thread) so an incompatible cache — written by a previous
    run whose optimizer had a different param-group layout, or a different
    optimizer class entirely — is rejected before any state is mutated,
    rather than detonating inside `optimizer.load_state_dict` (param-group
    count/size mismatch) or later at `optimizer.step()` (a class whose
    per-parameter state keys differ).

    Skipped (returns `None`) when the session has no optimizer or the
    checkpoint carries no optimizer state — preserving the no-optimizer flow.
    """
    if optimizer is None:
        return None
    cached = _state_dict(payload, "optimizer")
    if cached is None:
        return None
    cached_groups = cached.get("param_groups")
    current_groups = optimizer.state_dict().get("param_groups")
    if not isinstance(cached_groups, list) or not isinstance(current_groups, list):
        return "cached optimizer state is missing its param groups"
    if len(cached_groups) != len(current_groups):
        return (
            "cached optimizer does not match the current optimizer — "
            f"param group count differs (cached {len(cached_groups)} vs "
            f"optimizer {len(current_groups)})"
        )
    for i, (cached_group, current_group) in enumerate(
        zip(cached_groups, current_groups)
    ):
        cached_params = _group_params(cached_group)
        current_params = _group_params(current_group)
        if len(cached_params) != len(current_params):
            return (
                "cached optimizer does not match the current optimizer — "
                f"param group {i} has {len(cached_params)} params "
                f"(optimizer expects {len(current_params)})"
            )
    # A different optimizer class often keeps the same param-group layout but
    # writes different per-parameter state keys (SGD's `momentum_buffer` vs
    # Adam's `exp_avg`/`exp_avg_sq`), which only fails later at `step()`.
    # Catch it up-front by comparing the state keys the live optimizer already
    # populated against the cached ones for the same parameters.
    cached_state = cached.get("state")
    current_state = optimizer.state_dict().get("state")
    if isinstance(cached_state, dict) and isinstance(current_state, dict):
        for param_id, current_entry in current_state.items():
            cached_entry = cached_state.get(param_id)
            if not isinstance(current_entry, dict) or not isinstance(
                cached_entry, dict
            ):
                continue
            if set(current_entry) != set(cached_entry):
                return (
                    "cached optimizer does not match the current optimizer — "
                    "per-parameter state keys differ (likely a different "
                    "optimizer class)"
                )
    return None


def validate_scheduler_state(
    payload: dict[str, Any],
    scheduler: torch.optim.lr_scheduler.LRScheduler | None,
) -> str | None:
    """Check a cached scheduler state dict against the live scheduler.

    Returns a human-readable mismatch description, or `None` when the cached
    state can be loaded. Like the optimizer check, this runs at request time
    so a scheduler whose stored keys no longer match the live one is rejected
    before `scheduler.load_state_dict` runs on the training thread.

    Skipped (returns `None`) when the session has no scheduler or the
    checkpoint carries no scheduler state.
    """
    if scheduler is None:
        return None
    cached = _state_dict(payload, "scheduler")
    if cached is None:
        return None
    current = scheduler.state_dict()
    missing = sorted(set(current) - set(cached))
    unexpected = sorted(set(cached) - set(current))
    problems: list[str] = []
    if missing:
        problems.append(f"missing keys: {', '.join(missing[:5])}")
    if unexpected:
        problems.append(f"unexpected keys: {', '.join(unexpected[:5])}")
    if problems:
        return (
            "cached scheduler does not match the current scheduler — "
            + "; ".join(problems)
        )
    return None


class TrainingRestorer:
    """Restores the training loop at a cached epoch after a time-travel jump.

    The session drives this through two equivalent loop shapes. The flat one
    (`iter_epochs` + `epoch_guard`, surfaced as `session.epochs()` and
    `session.restore_point()`) is what hand-written loops use::

        for epoch in session.epochs(cache_dir=...):
            with session.restore_point():
                ...

    The nested one (`pending` + the restorer itself as a context manager) is
    what the Lightning integration transplants around `trainer.fit`::

        while restorer.pending():
            with restorer:
                for epoch in restorer.epochs():
                    ...

    Both re-enter at the start of an epoch: entering after a jump loads the
    cached epoch state back into the model / optimizer / scheduler and RNG,
    rewinds the session's schedule and watch statistics, and sets
    `start_epoch` to the jump target (`_apply_pending_jump`). The exit
    suppresses `TimeTravelJump` (and only that), so any other exception still
    propagates normally.
    """

    def __init__(
        self, session: Session | None = None, *, cache_dir: Path = DEFAULT_CACHE_DIR
    ) -> None:
        # `session` may be None for restorers created before the session
        # exists (e.g. the Lightning integration, where the session is built
        # inside the first `trainer.fit`); `Session.attach_restorer` binds it.
        self._session = session
        # Each rank persists its own file (followers tag theirs `.rank<r>`),
        # so a jump restores per-rank RNG while sharing the replicated
        # model/optimizer/scheduler. Outside DDP the rank is 0 (canonical
        # filename, single-process behaviour byte-identical).
        rank = 0 if session is None or session._dist is None else session._dist.rank
        self.cache = EpochCache(Path(cache_dir), rank=rank)
        self._start_epoch = 0
        self._finished = False
        self._jump_target: int | None = None
        # Detects a `while restorer.pending():` loop whose body forgot the
        # `with restorer:` — without it the loop would re-run training
        # forever. Any __enter__ between two pending() calls resets this.
        self._entered_since_pending = True

    @property
    def start_epoch(self) -> int:
        """First epoch the current attempt should train (0, or a jump target)."""
        return self._start_epoch

    @property
    def finished(self) -> bool:
        """Whether a `with` block completed without a time-travel jump."""
        return self._finished

    def _require_session(self) -> Session:
        if self._session is None:
            raise RuntimeError("restorer is not attached to a session yet")
        return self._session

    def epochs(self) -> range:
        """Epochs the current attempt should run: `start_epoch` to the end."""
        total = self._require_session().schedule.epochs
        if total is None:
            raise RuntimeError(
                "epoch count is unknown — set it via session.epochs(n) or "
                "nansense.start(epochs=n) before iterating restorer.epochs()"
            )
        return range(self._start_epoch, total)

    def resume_from(self, epoch: int) -> None:
        """Start the run at `epoch`, restoring its cached checkpoint.

        Behind `session.epochs(start_epoch=...)`, for resuming from a cache
        directory written by an earlier process (`cached_epochs` enumerates
        whatever is on disk, so a pre-baked cache is adopted as-is — this is
        what lets a hosted playground boot straight into a late epoch). The
        checkpoint is validated up-front like `Session.request_time_travel`
        (it must exist, load, and match the live model / optimizer /
        scheduler; raises `TimeTravelError` otherwise) and then armed like a
        pending jump: the first `restore_point()` entry — on the training
        thread, before any forward pass — loads the state back through the
        exact machinery a UI-requested jump uses.
        """
        session = self._require_session()
        total = session.schedule.epochs
        if total is not None and not 0 <= epoch < total:
            raise TimeTravelError(f"epoch {epoch} out of range [0, {total})")
        # Memory-mapped: only keys and shapes are inspected here; the real
        # load happens on the training thread in `_restore`.
        payload = self.cache.load(epoch, mmap=True)
        for error in (
            validate_model_state(payload, session.model),
            validate_optimizer_state(payload, session.optimizer),
            validate_scheduler_state(payload, session.scheduler),
        ):
            if error is not None:
                raise TimeTravelError(error)
        self._start_epoch = epoch
        self._jump_target = epoch

    def pending(self) -> bool:
        """Whether another `with restorer:` attempt should run."""
        if not self._finished and not self._entered_since_pending:
            raise RuntimeError(
                "restorer.pending() called twice without entering the "
                "`with restorer:` block — the loop body must wrap the "
                "epoch loop in `with restorer:`"
            )
        self._entered_since_pending = False
        return not self._finished

    def _apply_pending_jump(self) -> None:
        """Restore cached state when a jump was armed, else just note the entry.

        Shared by `__enter__` (the nested `while pending(): with restorer:`
        loop) and `epoch_guard` (the flat `for epoch in session.epochs():`
        loop): both re-enter at the start of an epoch and must roll the live
        state back to the jump target before that epoch trains. Marking
        `_entered_since_pending` here is also what satisfies both loops'
        "forgot the `with`" guard.
        """
        self._entered_since_pending = True
        if self._jump_target is not None:
            self._restore(self._jump_target)
            self._start_epoch = self._jump_target
            self._jump_target = None

    def iter_epochs(self) -> Iterator[int]:
        """Yield epoch indices for the flat `session.epochs()` loop.

        Drives the whole run as a generator: it yields `start_epoch …
        schedule.epochs - 1`, but after a time-travel jump (recorded by the
        `with session.restore_point():` block's exit) it re-yields the jump
        target and continues from there instead of advancing. The run is
        finished once the last epoch's body completes without a jump.

        Each body must be wrapped in `with session.restore_point():` — that
        context manager is what catches the `TimeTravelJump` (a `for` loop
        never throws its body's exception back into the iterator, so the
        generator cannot catch it itself), and a missing wrapper is reported
        rather than left to crash training or loop forever.
        """
        session = self._require_session()
        # `session.epochs(n)` set the count before reaching here (it errors
        # otherwise), so the schedule's epoch count is known.
        total = session.schedule.epochs
        assert total is not None
        epoch = self._start_epoch
        while True:
            session._current_epoch = epoch
            self._entered_since_pending = False
            yield epoch
            if not self._entered_since_pending:
                raise RuntimeError(
                    "the body of `for epoch in session.epochs()` must be "
                    "wrapped in `with session.restore_point():` — without it a "
                    "time-travel jump cannot be caught"
                )
            if self._jump_target is not None:
                # The next `restore_point()` entry consumes the target and
                # rolls state back; here we only pick the epoch to re-run.
                epoch = self._jump_target
            else:
                epoch += 1
                if epoch >= total:
                    self._finished = True
                    return

    @contextlib.contextmanager
    def epoch_guard(self) -> Iterator[TrainingRestorer]:
        """Per-epoch context for the flat loop (`session.restore_point()`).

        On entry it restores the cached state when the previous epoch's exit
        armed a jump; on exit it catches `TimeTravelJump` and re-arms it,
        suppressing the exception so `iter_epochs` re-yields the target.
        Unlike the restorer's own `__exit__`, completing one epoch does not
        mark the run finished — `iter_epochs` owns completion.
        """
        self._apply_pending_jump()
        try:
            yield self
        except TimeTravelJump as jump:
            # Suppressed (not re-raised): control returns to the
            # `for epoch in session.epochs():` loop, which resumes
            # `iter_epochs` to re-yield this target.
            self._jump_target = jump.epoch

    def __enter__(self) -> TrainingRestorer:
        self._apply_pending_jump()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: object,
    ) -> bool:
        if isinstance(exc, TimeTravelJump):
            self._jump_target = exc.epoch
            return True  # suppress: the while loop re-enters at start_epoch
        if exc is None:
            self._finished = True
        return False

    def save_epoch_start(self, epoch: int) -> None:
        """Checkpoint the training state at the start of `epoch`.

        Called by the session on the first batch of each epoch, before any
        forward pass — so the file holds exactly the state a jump to this
        epoch should restore (including any between-epoch user code such as
        `scheduler.step()` that already ran).
        """
        session = self._require_session()
        self.cache.save(
            epoch,
            model=session.model,
            optimizer=session.optimizer,
            scheduler=session.scheduler,
        )

    def _restore(self, epoch: int) -> None:
        """Load epoch `epoch`'s checkpoint back into the live training state.

        Runs on the training thread between attempts, so nothing races with
        a forward pass. The payload was already validated against the model,
        optimizer, and scheduler at request time
        (`Session.request_time_travel`), making a failing `load_state_dict`
        here practically impossible.
        """
        session = self._require_session()
        payload = self.cache.load(epoch)
        model_state = _state_dict(payload, "model")
        assert model_state is not None  # request-time validation guarantees this
        session.model.load_state_dict(model_state)
        optimizer_state = _state_dict(payload, "optimizer")
        if session.optimizer is not None and optimizer_state is not None:
            session.optimizer.load_state_dict(optimizer_state)
        scheduler_state = _state_dict(payload, "scheduler")
        if session.scheduler is not None and scheduler_state is not None:
            session.scheduler.load_state_dict(scheduler_state)
        restore_rng(_state_dict(payload, "rng"))
        # The checkpoint (model params plus the optimizer's moment tensors) was
        # just copied into the live state; drop it and hand the freed pages
        # back to the OS so the jump doesn't leave the load peak resident.
        del payload, model_state, optimizer_state, scheduler_state
        release_cpu_memory()
        session._rewind_to_epoch(epoch)
