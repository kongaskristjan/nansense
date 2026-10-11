# Custom experiments & tools

Two decorators plug your own code into NaNsense:

- [`session.experiment`](#register-an-experiment) adds an experiment to the Experiment page, next to the built-in ones.
- [`session.tool`](#tools) adds a tool for coding agents (`nansense-cli`), which the UI's Tools page also offers.

Register your own experiment alongside the built-in ones (deep dream, Grad-CAM, …). It appears on the Experiment page, with a form for its knobs. Coding agents can list it with `list_experiments` and run it with `run_experiment`. It runs on the training thread between batches, using the model and captured inputs, and can return a table, headline numbers, an image or a note. Pause training to run it manually, or turn on **Live** to rerun it at each capture.

A runnable demo of both lives in `examples/custom_extensions/`:

```bash
uv run examples/custom_extensions/main.py --nansense-port 8080
```

## Register an experiment

```python
session = nansense.start(model, optimizer=optimizer, port=8080)

@session.experiment(
    "ablate_channels",
    label="Channel ablation",
    params=[nansense.ExperimentParam("top", "Channels to list", "int", default=8, minimum=1)],
)
def ablate_channels(ctx: nansense.ExperimentContext) -> nansense.ExperimentOutput:
    """Accuracy drop when each channel of the layer is zeroed."""
    _, targets = ctx.batch_item                    # the paused batch, as the loader yielded it
    targets = targets.to(ctx.device)

    def accuracy(logits):
        return float((logits.argmax(1) == targets).float().mean())

    with torch.no_grad():
        baseline = accuracy(ctx.forward())         # forward() defaults to the paused inputs
        drops = []
        for c in range(ctx.activation(ctx.layer).shape[1]):
            def zero(name, act, c=c):              # called with tensor outputs
                if name != ctx.layer:
                    return None                    # None keeps the output as it is
                act = act.clone()
                act[:, c] = 0
                return act                         # a tensor replaces it
            drops.append((c, baseline - accuracy(ctx.forward(edit=zero))))
    drops.sort(key=lambda row: -row[1])
    return nansense.ExperimentOutput(
        table=nansense.Table(("channel", "accuracy drop"), drops[: ctx.params["top"]]),
        scalars={"baseline accuracy": baseline},
    )
```

The name must be lowercase `snake_case` and must not clash with another experiment. `label` is the text in the dropdown. The summary and description default to the function's docstring. The summary is the first line, shown as the dropdown's tooltip and in `list_experiments`; the description is the whole docstring, shown at the foot of the page's left pane.

## Knobs

Each [`ExperimentParam`](api.md#nansense.ExperimentParam) becomes a form widget on the page and a key in `run_experiment`'s `params`:

| `kind` | Widget | Default must be |
|---|---|---|
| `"int"`, `"float"` | number field (`minimum` applies; integers step by 1, floats use `step`) | a number |
| `"bool"` | switch | `True` or `False` |
| `"select"` | dropdown of `options` (`{value: label}`) | one of the option values |

Values reach the function validated, in `ctx.params[key]`. Numbers must be finite; values below `minimum` are raised to it, and `"int"` values are converted to integers.

## What the function gets

The [`ExperimentContext`](api.md#nansense.ExperimentContext) holds:

- `model` and `device`: the model at the batch boundary, and the device it lives on.
- `layer`: the layer chosen on the page, or passed as `run_experiment`'s `layer`.
- `params`: the knob values.
- `inputs`: the last snapshot's model inputs, on `device`.
- `batch_item`: the current batch as your loader yielded it (targets included), available while its batch context publishes or pauses. It is `None` outside that context or if the loop doesn't use `session.batches` or pass `item=` to `session.batch`.
- `forward(*inputs, edit=None)`: runs the model, defaulting to `ctx.inputs`. `edit(layer, output)` receives tensor outputs and can replace them by returning a tensor; returning `None` keeps the output. With a traced model it sees graph nodes; otherwise it sees named modules.
- `activation(layer, *inputs)`: reruns the model and returns one layer's tensor output, preserving its autograd graph when gradients are enabled.
- `should_abort()`: check it inside long loops. It turns true on cancellation, after the 90-second limit, or when a resume, rewind or close is requested. Stopping is cooperative: it cannot interrupt a long call.

## What it returns

An [`ExperimentOutput`](api.md#nansense.ExperimentOutput) with at least one of:

- `table`: a [`Table`](api.md#nansense.Table) of numbers or text. The page shows it as a sortable table; `get_experiment_result` returns its rows as JSON. Results keep the first 2,000 rows and add a note if rows were cut.
- `scalars`: headline numbers, shown as a row of tiles.
- `note`: free text.
- `image`: an input-space batch, drawn like the model's input.
- `attribution` and `reference`: one signed map per sample, drawn with the diverging colormap beside the inputs it explains. This also enables the page's "Overlay on input" switch.

To stream progress, write a generator that yields several outputs. Each output publishes when the next arrives; the last publishes as the final result after model state is restored. `step` and `total_steps` label the progress; without them, each yield counts as one step.

## Guarantees

- **Model state is restored.** The function runs in eval mode by default (`mode="train"` or `"unchanged"` changes that). Afterwards NaNsense restores training flags, existing buffers, the PyTorch RNG state for CPU and the model's device, and existing parameter values and `.grad`. You can edit parameter values or call `loss.backward()` for a what-if. Keep the model structure intact; optimizer state and other side effects are not restored.
- **Run errors become results.** An exception in the function publishes an error result with its type, message and traceback file and line. It does not stop training or unregister the experiment. Registration and invalid requests still raise errors.
- **Registration adds no per-batch callbacks.** Your code only runs when an experiment is requested, manually or through **Live**.
- `layers="modules"` limits the experiment to layers backed by an `nn.Module`, such as `conv1`. Other layers, such as a functional `relu`, cannot run it.
- **PyTorch Lightning.** Register in `NansenseCallback(on_session=...)`, which runs once the session exists.
- **Locked sessions.** A locked session (a shared demo) hides custom experiments unless they are registered with `locked_limits`, e.g. `{"top": 32}`. Those caps clamp numeric knobs there; `{}` offers the experiment uncapped. Register before `session.lock()`.
- **Distributed runs.** Under DDP the experiment runs on the leader rank only, against its own shard of the batch.

## Tools

A tool is a function an agent calls by name, with typed arguments:

```python
from typing import Annotated
import pydantic

@session.tool()
def confusion(
    ctx: nansense.ToolContext,
    batches: Annotated[int, pydantic.Field(ge=1, description="Validation batches to evaluate")] = 8,
) -> dict[str, object]:
    """Confusion matrix and accuracy for up to `batches` validation batches."""
    matrix = torch.zeros(4, 4, dtype=torch.long)
    with torch.no_grad():
        for i, (images, labels) in enumerate(val_loader):
            if i >= batches or ctx.should_abort():
                break
            predicted = ctx.model(images.to(ctx.device)).argmax(1).cpu()
            matrix.index_put_((labels, predicted), torch.ones_like(labels), accumulate=True)
    return {"accuracy": float(matrix.diag().sum() / matrix.sum()), "matrix": matrix}
```

List the tool with `nansense-cli --port 8080 tools` and call it with `nansense-cli --port 8080 confusion --batches 4`. Add `--help` to see its docstring and argument flags. Completed calls return JSON. The UI's top-bar **Tools** button opens a page with a form for each tool and its results. The button appears only when tools are registered.

- **Arguments** follow the context parameter and should use types with a JSON schema, such as `int`, `float`, `bool`, `str`, `Literal[...]`, lists of these, or `X | None`. `Annotated[..., pydantic.Field(description=..., ge=...)]` adds help text and a constraint. Arguments without defaults are required, including nullable arguments. `*args` and `**kwargs` are not supported.
- **Results** can contain numbers, strings, booleans, `None`, lists, dicts, tensors with at most 10,000 elements, or a [`Table`](api.md#nansense.Table). Tensors become JSON arrays or scalars; tables become objects with `columns` and `rows`. A returned dict becomes the top-level JSON object; other values arrive as `{"result": ...}`.
- **Errors.** Exceptions and returned dicts with a top-level `error` key are reported as tool errors; `nansense-cli` exits with a nonzero status.
- **When it runs.** By default, calls run on the training thread while paused, in evaluation mode. Pause before calling to avoid waiting in the queue. If the agent's wait expires, it receives a `seq` and `stage`; the call remains queued or running. Pause training if needed, then use `get_tool_result --seq N` to poll it. Up to 16 recent call results are retained.
- **Restoration.** As with custom experiments, training flags, existing buffers, parameter values, `.grad` and PyTorch RNG state for CPU and the model's device are restored after the call. Keep the model structure intact; optimizer state and other side effects are not restored. Set `mode="train"` or `mode="unchanged"` to change how the model runs.
- **Timeouts are cooperative.** `timeout` defaults to 60 seconds, measured from when execution starts. After that, `ctx.should_abort()` returns true; check it in long loops. It does not forcibly stop the function. The agent waits up to `timeout + 1` seconds, capped at 300 seconds, before receiving the call's status. This wait includes time in the queue and does not cancel the call.
- **Snapshot-only tools.** `on_training_thread=False` starts the call on a worker thread without waiting for a pause. Read only [`ctx.snapshot`](api.md#nansense.BatchSnapshot) and `ctx.inputs`, which are on CPU. Do not access the live model, including through `ctx.forward()` or `ctx.activation()`. Before the first capture, `snapshot` is `None` and `inputs` is empty.
- **Locked sessions** allow only tools registered with `public=True`. Register before `session.lock()`.

The [`ToolContext`](api.md#nansense.ToolContext) has everything an experiment's context has except `layer` and `params`, plus `snapshot`.
