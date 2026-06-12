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

import re
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import torch

if TYPE_CHECKING:
    from nansense.session import Session

DEFAULT_CACHE_DIR = Path("models/latest")


def capture_rng() -> dict[str, Any]:
    """Snapshot the global torch (and per-device CUDA) RNG states.

    The counterpart of `restore_rng`; stashed alongside checkpoints so a
    time-travel replay reproduces DataLoader shuffling and dropout.
    """
    return {
        "torch": torch.get_rng_state(),
        "cuda": (
            torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
        ),
    }


def restore_rng(rng: dict[str, Any] | None) -> None:
    """Best-effort RNG restore so the replayed epochs are deterministic.

    Restoring the global torch RNG also reproduces DataLoader shuffling:
    the loader draws its base seed from the global generator when its
    iterator is created. CUDA states are restored only when the device
    count still matches what was saved.
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
    schedule's range.
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
    """

    # Checkpoint file extension; subclasses with a different on-disk format
    # (e.g. the Lightning cache's `.ckpt`) override just this.
    FILE_SUFFIX: str = ".pt"

    def __init__(self, directory: Path) -> None:
        # The directory is created lazily on first save, so merely creating
        # a restorer (e.g. on a disabled session) leaves the disk untouched.
        self.directory = directory

    def path_for(self, epoch: int) -> Path:
        return self.directory / f"epoch_{epoch}{self.FILE_SUFFIX}"

    def cached_epochs(self) -> list[int]:
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

    def load(self, epoch: int) -> dict[str, Any]:
        """Load epoch `epoch`'s checkpoint; raises `TimeTravelError` if absent."""
        path = self.path_for(epoch)
        if not path.exists():
            raise TimeTravelError(f"no cached model for epoch {epoch} ({path})")
        try:
            payload = torch.load(path, map_location="cpu", weights_only=True)
        except Exception as e:  # corrupt file, unpicklable content, ...
            raise TimeTravelError(f"failed to load {path}: {e}") from e
        if not isinstance(payload, dict) or "model" not in payload:
            raise TimeTravelError(f"{path} is not a nansense epoch checkpoint")
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

    Created via `session.training_restorer(...)`; creating it is what
    enables time travel (and epoch caching) for the session. Use as::

        while restorer.pending():
            with restorer:
                for epoch in restorer.epochs():
                    ...

    `pending()` is True until a `with` block runs to completion without a
    jump. Entering the block after a jump loads the cached epoch state back
    into the model / optimizer / scheduler and RNG, rewinds the session's
    schedule and watch statistics, and sets `start_epoch` to the jump
    target. The `with` block's exit suppresses `TimeTravelJump` (and only
    that), so any other exception still propagates normally.
    """

    def __init__(
        self, session: Session | None = None, *, cache_dir: Path = DEFAULT_CACHE_DIR
    ) -> None:
        # `session` may be None for restorers created before the session
        # exists (e.g. the Lightning integration, where the session is built
        # inside the first `trainer.fit`); `Session.attach_restorer` binds it.
        self._session = session
        self.cache = EpochCache(Path(cache_dir))
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
        return range(self._start_epoch, self._require_session().schedule.epochs)

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

    def __enter__(self) -> TrainingRestorer:
        self._entered_since_pending = True
        if self._jump_target is not None:
            self._restore(self._jump_target)
            self._start_epoch = self._jump_target
            self._jump_target = None
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
        session._rewind_to_epoch(epoch)
