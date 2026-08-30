# Wiring guide

Adding NaNsense to a training loop is a few lines of code. This page covers raw PyTorch and PyTorch Lightning; the full argument reference lives in the [API reference](https://kongaskristjan.github.io/nansense/dev/api/index.md). Prefer not to do it by hand? [Integrate with one prompt](https://kongaskristjan.github.io/nansense/dev/integrate/index.md) has a copy-paste prompt for your coding agent.

## Wire it into your loop: raw PyTorch

```
import torch
import nansense

# Init model, optimizer, criterion, dataloaders
model = ...
optimizer = ...
criterion = ...
train_dl, val_dl = ...

# Setup UI. The schedule is discovered as you train (phase names and batch
# counts are learned from the loop below); no need to declare them up front.
session = nansense.start(model, optimizer=optimizer, port=8080, enabled=True)

# Time travel needs an epoch cache. `session.epochs(50)` iterates like
# `range(50)` but checkpoints each epoch start; wrap each iteration's body in
# `with session.restore_point():` so a UI-requested jump can unwind it and
# re-enter at a different epoch. Without this loop, training runs once through
# and the Time Travel button is disabled.
for epoch in session.epochs(50, cache_dir=".nansense_cache"):
    with session.restore_point():
        # Training batch iteration
        for inputs, targets in session.batches(train_dl, phase="train"):
            optimizer.zero_grad()  # keep zero_grad at the beginning of the batch
            loss = criterion(model(inputs), targets)  # as NaNsense reads .grad when
            loss.backward()  # the batch exits, so zeroing after step() would
            optimizer.step()  # leave the weight-gradient views empty.
        # Validation batch iteration ...

# Close the UI (the served page stays up for post-mortem browsing)
session.close()
```

The pieces:

- [`nansense.start(model, ...)`](https://kongaskristjan.github.io/nansense/dev/api/#nansense.start) creates the [`Session`](https://kongaskristjan.github.io/nansense/dev/api/#nansense.Session) and, when `port=` is given, serves the UI. Pass `optimizer=` to get per-parameter optimizer state and live hyperparameters on the weights page, and `scheduler=` so time travel restores the LR schedule.
- [`session.batches(loader, phase=...)`](https://kongaskristjan.github.io/nansense/dev/api/#nansense.Session.batches) wraps each phase's dataloader — this is where NaNsense pauses, steps and captures.
- [`session.close()`](https://kongaskristjan.github.io/nansense/dev/api/#nansense.Session.close) marks training finished; the served page stays up for post-mortem browsing.
- `enabled=False` makes the whole session a near-zero-overhead no-op, so you can leave the wiring in place and switch the UI off with one flag.

To serve the UI separately from session creation (or pick a custom uvicorn log level), omit `port` and call [`nansense.serve(session, port=...)`](https://kongaskristjan.github.io/nansense/dev/api/#nansense.serve) yourself.

### Time travel

Time travel needs an epoch cache: drive the epoch loop with [`for epoch in session.epochs(N, cache_dir=...)`](https://kongaskristjan.github.io/nansense/dev/api/#nansense.Session.epochs) (default `.nansense_cache`) and wrap each iteration's body in [`with session.restore_point():`](https://kongaskristjan.github.io/nansense/dev/api/#nansense.Session.restore_point) as shown above. Each epoch start is checkpointed to disk — model, optimizer, scheduler and RNG state — and a UI-requested jump unwinds the loop body and re-enters it at the chosen epoch.

### The schedule

NaNsense learns phase names and batch counts as training runs. To show exact progress from the first epoch, pass the counts up front: `phases={"train": len(train_dl), "val": len(val_dl)}`. The bundled examples and Lightning integration do this for you.

A declared schedule is validated. If a phase does not run every epoch, update it with [`session.set_schedule(phases=...)`](https://kongaskristjan.github.io/nansense/dev/api/#nansense.Session.set_schedule).

### Data loading stalls between phases

If macOS or Windows pauses for several seconds between training and validation, set `num_workers=0` on your data loaders. Avoid `persistent_workers=True` when using time travel because replayed augmentations may differ.

## Wire it into your loop: PyTorch Lightning

```
import lightning as L
from nansense.lightning import NansenseCallback, fit_with_time_travel

# PyTorch Lightning modules
module = ...
datamodule = ...

# `model="net"` is the attribute path to the network inside your LightningModule, e.g. module.net
callback = NansenseCallback(port=8080, model="net", enabled=True)

# Time travel consumes the running fit, so the trainer comes from a factory:
# fit_with_time_travel builds a fresh Trainer for each jump-and-replay attempt.
trainer_factory = lambda: L.Trainer(max_epochs=50)
fit_with_time_travel(trainer_factory, module, datamodule=datamodule, callback=callback)
```

Attach a [`NansenseCallback(model="<attr path to the network>", ...)`](https://kongaskristjan.github.io/nansense/dev/api/#nansense.lightning.NansenseCallback) to your trainer and run the fit through [`fit_with_time_travel`](https://kongaskristjan.github.io/nansense/dev/api/#nansense.lightning.fit_with_time_travel), which owns the jump-and-replay loop. The callback accepts the same `port` / `host` / `open_browser` / `enabled` / `input_mean` / `input_std` / `input_transform` arguments as `start`.

## Displaying inputs correctly

- `input_mean` / `input_std`: the input normalization, so images display in their original colors.
- `input_transform`: a callable mapping a non-RGB image input `(N, C, H, W)` to a displayable `(N, 1|3, H, W)` image in `[0, 1]` (keeping `H × W`); without it, an input whose channel count isn't 1 or 3 shows a hint to add one. A flat `(N, C)` input needs none; it renders as a colormapped strip.
- For a multi-input model, `input_mean` / `input_std` / `input_transform` each take either one value for all inputs or a `dict` keyed by input name, and the input pane gains a dropdown to pick which input to view and perturb.

See `examples/multimodal/main.py` for all three in action.

## Distributed training (DDP)

For distributed training, call `nansense.start()` on every rank. Rank 0 serves the UI and controls the run; statistics include data from every rank. Use `session.epochs()` on every rank to enable time travel.

See `examples/standard/main.py --distributed`. Keep in mind that DDP support is currently **experimental**.
