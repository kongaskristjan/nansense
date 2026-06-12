"""PyTorch Lightning integration: `NansenseCallback` and `fit_with_time_travel`.

The callback maps nansense's per-batch context onto Lightning's hook pairs
(`on_*_batch_start` / `on_*_batch_end`), so a stock `Trainer` gets the full
pause/step/inspect experience with no changes to the training code:

    callback = NansenseCallback(port=8080, model="net")
    trainer = L.Trainer(max_epochs=50, callbacks=[callback])
    trainer.fit(module, datamodule)

Time travel needs to own the retry loop around `trainer.fit`, which a
callback cannot do, so it ships as a wrapper:

    fit_with_time_travel(
        make_trainer, module, callback=callback, datamodule=datamodule
    )

The wrapper leans on Lightning's own checkpoint/resume: each epoch boundary
is checkpointed via `trainer.save_checkpoint`, and a jump re-invokes
`trainer.fit(ckpt_path=...)` — Lightning restores the model, optimizers,
schedulers, and loop counters, while the callback restores the RNG states it
stashed into the checkpoint. Nothing of Lightning's training loop is
reimplemented.
"""

from __future__ import annotations

import inspect
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.optim.lr_scheduler import LRScheduler

try:
    from lightning.pytorch import (
        Callback,
        LightningDataModule,
        LightningModule,
        Trainer,
    )
except ImportError as e:  # pragma: no cover - exercised only without lightning
    raise ImportError(
        "nansense.lightning requires the 'lightning' package; install it "
        "with `uv add lightning` (or `pip install lightning`)"
    ) from e

import nansense
from nansense.restore import (
    DEFAULT_CACHE_DIR,
    EpochCache,
    TimeTravelError,
    TrainingRestorer,
    capture_rng,
    restore_rng,
)
from nansense.session import Session, _BatchContext

_RNG_CHECKPOINT_KEY = "nansense_rng"

# Lightning's TRAIN_DATALOADERS / EVAL_DATALOADERS unions are wide and
# version-dependent; the wrapper only forwards these to `trainer.fit`.
type _Dataloaders = Any


class NansenseCallback(Callback):
    """Drive a nansense session from a Lightning `Trainer`.

    The session is created when `fit` starts (optimizers exist by then) and
    closed when it ends — the UI stays up for post-mortem browsing, exactly
    like a hand-written loop. Pass `model="net"` (an attribute path inside
    the LightningModule) to point nansense at the actual network; this is
    recommended whenever the module wraps its layers in a submodule, both
    for fx tracing and so input capture sees the real forward signature.

    Supported out of the box: automatic optimization, epoch-boundary
    validation (including `check_val_every_n_epoch > 1`), sanity-check
    skipping, and `enabled=False` as the zero-overhead off switch. Mid-epoch
    validation (`val_check_interval < 1.0` or step-based) is rejected with a
    clear error, and iterable-style dataloaders without a length are not
    supported — nansense declares the schedule up-front.
    """

    def __init__(
        self,
        *,
        port: int | None = None,
        host: str = "127.0.0.1",
        model: str | None = None,
        input_mean: tuple[float, ...] | None = None,
        input_std: tuple[float, ...] | None = None,
        enabled: bool = True,
    ) -> None:
        self._port = port
        self._host = host
        self._model_attr = model
        self._input_mean = input_mean
        self._input_std = input_std
        self._enabled = enabled
        self._session: Session | None = None
        self._restorer: LightningRestorer | None = None
        self._ctx: _BatchContext | None = None
        self._epoch = 0

    @property
    def session(self) -> Session | None:
        """The live session, or None before `fit` has started."""
        return self._session

    @property
    def state_prefix(self) -> str:
        """Prefix of the watched network's keys inside a Lightning ckpt."""
        return "" if self._model_attr is None else f"{self._model_attr}."

    # -- session lifecycle ------------------------------------------------

    def on_fit_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        if self._session is not None:  # a time-travel re-fit reuses the session
            return
        max_epochs = trainer.max_epochs
        if max_epochs is None or max_epochs <= 0:
            raise RuntimeError(
                "nansense requires Trainer(max_epochs=N) with a positive N — "
                "the schedule (and the UI's progress display) is declared "
                "up-front from it"
            )
        optimizers = trainer.optimizers
        scheduler_configs = trainer.lr_scheduler_configs
        scheduler = scheduler_configs[0].scheduler if scheduler_configs else None
        self._session = nansense.start(
            self._resolve_model(pl_module),
            epochs=max_epochs,
            # Real per-phase batch counts are not known until the dataloaders
            # are set up; this placeholder is replaced at every
            # `on_train_epoch_start`, before any batch advances the schedule.
            phases={"train": 1},
            enabled=self._enabled,
            optimizer=optimizers[0] if optimizers else None,
            scheduler=scheduler if isinstance(scheduler, LRScheduler) else None,
            port=self._port,
            host=self._host,
            input_mean=self._input_mean,
            input_std=self._input_std,
        )
        if self._restorer is not None:
            self._session.attach_restorer(self._restorer)
        self._save_epoch_zero(trainer)

    def _save_epoch_zero(self, trainer: Trainer) -> None:
        """Checkpoint the untrained state, enabling jumps back to epoch 0.

        Runs at `on_fit_start` — before Lightning sets up the dataloaders.
        The hook position matters for RNG determinism: creating a dataloader
        iterator draws a seed from the global RNG, and a resumed fit creates
        its first iterator eagerly (before `on_train_start`) while a running
        fit creates each next epoch's lazily. Anchoring the save *before*
        any dataloader setup, and the restore in `on_load_checkpoint`
        (likewise before setup), keeps the stream draws aligned between the
        original epoch and its replay. The same reasoning puts the other
        epochs' saves at `on_train_epoch_end`. (Lightning's sanity check
        runs under `isolate_rng()`, so it never shifts the stream.)
        """
        restorer = self._restorer
        if (
            self._active is None
            or restorer is None
            or restorer.finished
            or restorer.resume_ckpt_path is not None
        ):
            return
        restorer.save_for_epoch(trainer, 0)

    def on_fit_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        # Only successful completion reaches this hook — a TimeTravelJump
        # unwinds past it, keeping the session alive for the next attempt.
        if self._session is not None:
            self._session.close()

    def on_exception(
        self, trainer: Trainer, pl_module: LightningModule, exception: BaseException
    ) -> None:
        # Close an open batch context without publishing a snapshot. Covers
        # both a TimeTravelJump unwinding out of a paused batch and genuine
        # training errors.
        ctx, self._ctx = self._ctx, None
        if ctx is not None:
            ctx.__exit__(type(exception), exception, exception.__traceback__)

    def _resolve_model(self, pl_module: LightningModule) -> nn.Module:
        target: nn.Module = pl_module
        if self._model_attr is not None:
            for part in self._model_attr.split("."):
                attr = getattr(target, part)
                if not isinstance(attr, nn.Module):
                    raise TypeError(
                        f"model={self._model_attr!r}: {part!r} is not an "
                        f"nn.Module (got {type(attr).__name__})"
                    )
                target = attr
        return target

    # -- schedule ---------------------------------------------------------

    def on_train_epoch_start(
        self, trainer: Trainer, pl_module: LightningModule
    ) -> None:
        self._epoch = trainer.current_epoch
        self._update_schedule(trainer)

    def on_validation_epoch_start(
        self, trainer: Trainer, pl_module: LightningModule
    ) -> None:
        # Re-declare with the now-known val batch count. Usually a no-op;
        # it matters when val dataloaders were not yet set up at train epoch
        # start (e.g. num_sanity_val_steps=0 on the first epoch).
        if not trainer.sanity_checking:
            self._update_schedule(trainer)

    @property
    def _active(self) -> Session | None:
        session = self._session
        return session if session is not None and session.enabled else None

    def _update_schedule(self, trainer: Trainer) -> None:
        session = self._active
        if session is None:
            return
        n_train = trainer.num_training_batches
        if not isinstance(n_train, int) or n_train <= 0:
            raise RuntimeError(
                "nansense requires a sized train dataloader — the number of "
                f"batches per epoch must be known up-front (got {n_train!r})"
            )
        phases = {"train": n_train}
        n_val = self._val_batches(trainer)
        if n_val > 0:
            self._reject_mid_epoch_val(trainer, n_train)
            if self._val_runs_this_epoch(trainer):
                phases["val"] = n_val
        # Phases are re-declared per epoch (batch counters are kept), which
        # is what models val-every-N-epochs runs: epochs without validation
        # declare a train-only schedule.
        session.set_schedule(phases=phases)

    @staticmethod
    def _val_batches(trainer: Trainer) -> int:
        total = 0
        for n in trainer.num_val_batches:
            if not isinstance(n, int):
                raise RuntimeError(
                    "nansense requires sized val dataloaders — the number of "
                    f"batches per epoch must be known up-front (got {n!r})"
                )
            total += n
        return total

    @staticmethod
    def _val_runs_this_epoch(trainer: Trainer) -> bool:
        every_n = trainer.check_val_every_n_epoch
        if every_n is None:
            raise RuntimeError(
                "nansense does not support step-driven validation "
                "(check_val_every_n_epoch=None); validate at epoch boundaries"
            )
        return (trainer.current_epoch + 1) % every_n == 0

    @staticmethod
    def _reject_mid_epoch_val(trainer: Trainer, n_train: int) -> None:
        val_check_batch = getattr(trainer, "val_check_batch", None)
        if (
            isinstance(val_check_batch, (int, float))
            and val_check_batch < n_train
        ):
            raise RuntimeError(
                "nansense does not support mid-epoch validation "
                "(val_check_interval < 1.0); validate at epoch boundaries"
            )

    # -- per-batch context ------------------------------------------------

    def on_train_batch_start(
        self, trainer: Trainer, pl_module: LightningModule, batch: Any, batch_idx: int
    ) -> None:
        self._enter_batch("train")

    def on_train_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
    ) -> None:
        self._exit_batch()

    def on_validation_batch_start(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        if not trainer.sanity_checking:
            self._enter_batch("val")

    def on_validation_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        if not trainer.sanity_checking:
            self._exit_batch()

    def _enter_batch(self, phase: str) -> None:
        session = self._active
        if session is None:
            return
        ctx = session.batch(phase=phase, epoch=self._epoch)
        # `_ctx` is assigned only after a successful enter: a TimeTravelJump
        # raised from `__enter__` (jump armed while training was running)
        # must not leave a never-entered context for `on_exception` to exit.
        ctx.__enter__()
        self._ctx = ctx

    def _exit_batch(self) -> None:
        ctx, self._ctx = self._ctx, None
        if ctx is not None:
            # The pause (and a possible TimeTravelJump) happens in here, on
            # the training thread, inside Lightning's batch-end hook.
            ctx.__exit__(None, None, None)

    # -- time travel cooperation -------------------------------------------

    def on_train_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        # Fires after this epoch's validation: the canonical Lightning
        # checkpoint position (same as ModelCheckpoint), so a resume
        # continues cleanly with the next epoch.
        session = self._active
        restorer = self._restorer
        if session is None or restorer is None or restorer.finished:
            return
        next_epoch = self._epoch + 1
        if next_epoch < session.schedule.epochs:
            restorer.save_for_epoch(trainer, next_epoch)

    def on_save_checkpoint(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        checkpoint: dict[str, Any],
    ) -> None:
        # Lightning checkpoints do not include global RNG state; stash it so
        # a time-travel replay reproduces DataLoader shuffling and dropout.
        checkpoint[_RNG_CHECKPOINT_KEY] = capture_rng()

    def on_load_checkpoint(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        checkpoint: dict[str, Any],
    ) -> None:
        # Applied immediately: this hook fires before any dataloader setup,
        # mirroring the save anchors (see `_save_epoch_zero` for why the
        # position relative to iterator creation matters).
        rng = checkpoint.get(_RNG_CHECKPOINT_KEY)
        if isinstance(rng, dict):
            restore_rng(rng)


class _LightningCkptCache(EpochCache):
    """Epoch cache backed by Lightning checkpoints (`epoch_<n>.ckpt`).

    Saves go through `trainer.save_checkpoint` (see
    `LightningRestorer.save_for_epoch`), so `load` only needs to present the
    checkpoint's model weights to `Session.request_time_travel`'s validation
    — restoring is delegated to `trainer.fit(ckpt_path=...)`.
    """

    FILE_SUFFIX = ".ckpt"

    def __init__(self, directory: Path, *, state_prefix: str = "") -> None:
        super().__init__(directory)
        self._state_prefix = state_prefix

    def save(
        self,
        epoch: int,
        *,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer | None,
        scheduler: torch.optim.lr_scheduler.LRScheduler | None,
    ) -> None:
        raise NotImplementedError(
            "Lightning epoch checkpoints are written via trainer.save_checkpoint"
        )

    def load(self, epoch: int) -> dict[str, Any]:
        path = self.path_for(epoch)
        if not path.exists():
            raise TimeTravelError(f"no cached checkpoint for epoch {epoch} ({path})")
        try:
            # Lightning checkpoints carry loop/callback state beyond plain
            # tensors, so weights_only loading is not an option; the file is
            # one this process (or a previous run of it) wrote itself.
            payload = torch.load(path, map_location="cpu", weights_only=False)
        except Exception as e:  # corrupt file, unpicklable content, ...
            raise TimeTravelError(f"failed to load {path}: {e}") from e
        state = payload.get("state_dict") if isinstance(payload, dict) else None
        if not isinstance(state, dict):
            raise TimeTravelError(f"{path} is not a Lightning checkpoint")
        prefix = self._state_prefix
        if prefix:
            state = {
                name.removeprefix(prefix): tensor
                for name, tensor in state.items()
                if name.startswith(prefix)
            }
        # Present the weights under "model" so `validate_model_state` checks
        # them against the live (sub)module exactly like a nansense cache.
        return {"epoch": epoch, "model": state}


class LightningRestorer(TrainingRestorer):
    """Time-travel restorer that delegates state restore to Lightning.

    Instead of loading state dicts back into live objects, a jump records
    the target's checkpoint path; `fit_with_time_travel` then re-invokes
    `trainer.fit(ckpt_path=...)` and Lightning restores the model,
    optimizers, schedulers, and loop counters itself. Only nansense's own
    bookkeeping (schedule counters, watch statistics) is rewound here.
    """

    def __init__(
        self, *, cache_dir: Path = DEFAULT_CACHE_DIR, state_prefix: str = ""
    ) -> None:
        super().__init__(None, cache_dir=cache_dir)
        self.cache: EpochCache = _LightningCkptCache(
            Path(cache_dir), state_prefix=state_prefix
        )
        self._resume_ckpt: Path | None = None

    @property
    def resume_ckpt_path(self) -> str | None:
        """Checkpoint the next `trainer.fit` should resume from, if any."""
        return None if self._resume_ckpt is None else str(self._resume_ckpt)

    def save_for_epoch(self, trainer: Trainer, epoch: int) -> None:
        """Checkpoint the state a jump to the start of `epoch` restores."""
        self.cache.directory.mkdir(parents=True, exist_ok=True)
        path = self.cache.path_for(epoch)
        tmp = path.with_suffix(".ckpt.tmp")
        trainer.save_checkpoint(tmp)
        tmp.replace(path)

    def save_epoch_start(self, epoch: int) -> None:
        # The session calls this on each epoch's first batch; Lightning
        # checkpoints must instead be written at Lightning's own boundaries
        # (on_train_start / on_train_epoch_end), where resume semantics are
        # exact — the NansenseCallback drives `save_for_epoch` from there.
        return

    def _restore(self, epoch: int) -> None:
        self._resume_ckpt = self.cache.path_for(epoch)
        self._require_session()._rewind_to_epoch(epoch)


def fit_with_time_travel(
    make_trainer: Callable[[], Trainer],
    model: LightningModule,
    *,
    callback: NansenseCallback,
    train_dataloaders: _Dataloaders | None = None,
    val_dataloaders: _Dataloaders | None = None,
    datamodule: LightningDataModule | None = None,
    cache_dir: Path = DEFAULT_CACHE_DIR,
) -> None:
    """Run `trainer.fit` under nansense time travel.

    Equivalent to `make_trainer().fit(model, ...)` with `callback` attached,
    plus the UI's Time Travel button: every epoch boundary is checkpointed
    to `cache_dir`, and a jump re-enters training at the chosen epoch with
    model / optimizer / scheduler / RNG state restored.

    `make_trainer` is a factory because each jump needs a fresh `Trainer`:
    Lightning treats a trainer as single-use for `fit`, and after the
    `TimeTravelJump` unwinds, the old one has already torn itself down. The
    callback (and its session, UI included) lives across attempts.

    Note that metric loggers cannot time-travel — after a jump, an attached
    logger sees the replayed epochs again (overlapping curves or a fresh
    run per attempt, depending on the logger).
    """
    if callback.session is not None:
        raise RuntimeError(
            "fit_with_time_travel needs a fresh NansenseCallback — this one "
            "already drove a fit"
        )
    restorer = LightningRestorer(
        cache_dir=Path(cache_dir), state_prefix=callback.state_prefix
    )
    callback._restorer = restorer  # attached to the session at on_fit_start
    while restorer.pending():
        with restorer:
            trainer = make_trainer()
            # `Trainer.callbacks` is assigned dynamically by Lightning's
            # callback connector, hence the getattr indirection.
            callbacks: list[Callback] = getattr(trainer, "callbacks")
            if callback not in callbacks:
                callbacks.append(callback)
            fit_kwargs: dict[str, Any] = {}
            if restorer.resume_ckpt_path is not None:
                fit_kwargs["ckpt_path"] = restorer.resume_ckpt_path
                # The checkpoint carries loop/RNG state, not just weights;
                # older Lightning versions lack the parameter.
                if "weights_only" in inspect.signature(trainer.fit).parameters:
                    fit_kwargs["weights_only"] = False
            trainer.fit(
                model,
                train_dataloaders=train_dataloaders,
                val_dataloaders=val_dataloaders,
                datamodule=datamodule,
                **fit_kwargs,
            )
