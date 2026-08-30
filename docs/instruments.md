# Custom metrics & tensors

Instruments add your own per-layer metrics and tensors to the UI. Register a callback and NaNsense evaluates it for each watched layer using that batch's activations, gradients, weights, and optimizer state.

There are three kinds:

| Decorator | Returns | Shown |
|---|---|---|
| `session.watch_metric(name)` | scalar(s) per layer | `/stats` → GRAPHS, one plot per metric |
| `session.watch_layer_tensor(name)` | tensor shaped like the activation | main page, extra strip under activations/gradients |
| `session.watch_weight_tensor(name)` | tensor shaped like the parameter | `/weights`, next to weight/gradient/optimizer strips |

A runnable demo of all three lives in `examples/custom_metrics/`:

```bash
uv run examples/custom_metrics/main.py --nansense-port 8080
```

## Scalar metrics

```python
session = nansense.start(model, optimizer=optimizer, port=8080)

@session.watch_metric("sparsity")                      # one point per batch
def sparsity(ctx: nansense.LayerContext) -> float:
    return float((ctx.activation > 0).float().mean())

@session.watch_metric("grad_rms", on="epoch", reduce="mean")  # one point per epoch
def grad_rms(ctx: nansense.LayerContext) -> float | None:
    if ctx.gradient is None:                           # e.g. no-grad val forwards
        return None                                    # skip this layer/batch
    return float(ctx.gradient.square().mean().sqrt())
```

The callback receives a [`LayerContext`](api.md#nansense.LayerContext) and may return:

- a number (or a 1-element tensor) — one plot trace,
- a mapping of named scalars (`{"lo": ..., "hi": ...}`) — one trace per key,
- `None` — skip this layer for this batch (also a natural per-layer filter).

Use `on="batch"` to plot every batch. Use `on="epoch"` to reduce each epoch to one point with `"mean"` (the default), `"sum"`, `"min"`, `"max"`, `"last"`, or a custom `values -> float` function. The plots appear under **Stats → Graphs**.

## Layer tensors

```python
@session.watch_layer_tensor("zscore")
def zscore(ctx: nansense.LayerContext) -> torch.Tensor:
    a = ctx.activation
    return (a - a.mean()) / (a.std() + 1e-6)
```

The result must have the same shape as the activation. It appears as another strip in the layer card and is evaluated whenever the UI receives a new snapshot.

## Weight tensors

```python
@session.watch_weight_tensor("adam_dir")
def adam_dir(ctx: nansense.WeightContext) -> torch.Tensor | None:
    state = ctx.optimizer_state                        # this parameter's entries
    if "exp_avg" not in state:
        return None                                    # Adam state is lazy
    return state["exp_avg"] / (state["exp_avg_sq"].sqrt() + 1e-8)
```

Weight-tensor callbacks receive a [`WeightContext`](api.md#nansense.WeightContext). The result must match the parameter's shape and appears beside it on the weights page.

## The rules

- **Watched layers only.** Instruments run for the layers the stats scope collects — the watched set by default, every layer under scope `"all"`, nothing under `"none"`.
- **Training thread, live tensors, `no_grad`.** Callbacks run inside the batch context against the live device tensors: fast (no copies), but treat every tensor as read-only.
- **Errors never kill training.** A raising callback (or a wrong-shaped/typed return) disables that instrument, prints one console line, and reports on the `/stats` GRAPHS view and `session.instrument_errors`. Everything else keeps running.
- **Stateful instruments are just callables.** Pass any object with `__call__` — `session.watch_metric("drift")(DriftTracker(model))`. If it also defines `on_rewind(epoch)`, the session calls it when time travel rewinds, so cross-batch state doesn't leak across timelines. Stored series from rewound epochs are dropped automatically.
- **Names are unique** across all instrument kinds — they label the plots and strips.
- Under DDP, instruments run on the leader rank only (rank-local, like the extreme-input patches). On a locked (shared demo) session, register instruments before `session.lock()`.
