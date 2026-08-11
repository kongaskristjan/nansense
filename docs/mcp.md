# Debug with a coding agent (MCP)

NaNsense speaks [MCP](https://modelcontextprotocol.io), so a coding agent can drive the debugger itself: pause your training run, read any layer's activations and gradients, and find out which layer started producing NaNs — without you clicking through the UI and pasting screenshots back into the chat.

The agent connects to the same session the browser shows. Both are front-ends onto one paused training loop, so you can watch in the UI while the agent works.

## Connect

The endpoint is served alongside the UI, on the same port. Start your run as usual:

```bash
uv run --group cpu examples/standard/main.py --nansense-port 8080
```

The startup banner prints the command to register it:

```text
┌──────────────────────────────────────────────────────────────┐
│ NaNsense UI is running at:                                   │
│ http://127.0.0.1:8080                                        │
│                                                              │
│ Debug this run from a coding agent:                          │
│ claude mcp add --transport http nansense http://127.0.0.1:8080/mcp │
└──────────────────────────────────────────────────────────────┘
```

Run that command in the directory you're working in, and the agent has the tools. Other MCP clients take the same URL — the transport is streamable HTTP.

To turn the endpoint off, pass `mcp=False` to [`serve`][nansense.serve].

!!! note "It listens on localhost"

    The endpoint is bound wherever you bound the UI, and it can control your training run. On the default loopback bind it also rejects requests carrying a foreign `Host` header, so a web page you visit cannot reach it. Don't expose the port to an untrusted network.

## What the agent can do

Ask in plain language — "why has my loss stopped falling?" — and the agent works through the tools below.

| Tool | What it does |
| --- | --- |
| `get_status` | Where the run sits: paused or running, position, watched layers, any numerical warning |
| `get_architecture` | Layer names, hyperparameters and the compute graph |
| `get_layer_stats` | Activation and gradient statistics for the last captured batch, for any layer |
| `get_stats_history` | One layer's statistics per epoch — the trend across the run |
| `get_debug_report` | Which layers produced NaN/Inf or collapsing gradients, with per-layer percentages |
| `step`, `run`, `run_until`, `pause`, `detach` | The top bar's run controls |
| `refresh` | Publish a fresh snapshot from a free-running session without pausing it |
| `watch_layers`, `unwatch_layers`, `set_stats_scope` | Choose which layers collect running statistics |
| `configure_debug_checks`, `silence_debug_check` | Tune the numerical-error checks |

| `get_weight_stats` | A layer's parameters, their gradients, and the optimizer state moving them |
| `get_metrics` | Custom scalar metrics the training script registered with `watch_metric` |
| `get_settings`, `set_update_frequency`, `set_watch_performance` | The settings dialog |

Statistics come from two places, and the distinction matters when you read the agent's reasoning. `get_layer_stats` reads the last captured batch and covers **any** layer. `get_stats_history` reads the running accumulators, which only cover **watched** layers — so the agent watches a layer first, lets a few epochs run, and then asks for the trend.

## What the agent can see

The `render_*` tools return the views as pictures — the same ones the browser draws, rendered server-side and sent as images the agent looks at directly.

| Tool | The view |
| --- | --- |
| `render_layer` | Per-channel activation and gradient strips: red positive, blue negative, NaN/Inf as transparent holes |
| `render_input` | One sample of the model's input, denormalized |
| `render_weights` | A layer's kernels, gradients and optimizer state, plus the live learning rate |
| `render_histogram` | Value distributions of watched layers, over the signed-log bins |
| `render_extreme_patches` | The inputs that most (and least) excite each channel |
| `render_bin_samples` | The inputs behind one histogram bar — which samples landed in it |

This is worth its tokens when the numbers say *that* something is wrong but not *where*. A layer's mean and standard deviation look healthy while one channel of sixty-four is dead; the strip shows that at a glance. A conv filter that has collapsed to noise, an activation saturating along one edge of the image, a gradient histogram with a spike in the overflow bin — all are shapes, not scalars.

Pictures are capped at 1568 pixels on the longest side and downscaled past it, with a note saying so, since a wide layer would otherwise arrive as an unreadably large image.

## What the agent can try

Reading a paused run is one thing; the rest of the UI is about *interrogating* it.

| Tool | What it does |
| --- | --- |
| `time_travel`, `get_time_travel_status` | Restart at an earlier epoch, with the model, optimizer and scheduler restored |
| `pin_batch`, `unpin_batch`, `set_probe_mode`, `get_probe_status` | Re-run the model on one fixed input at every capture |
| `add_perturbation`, `clear_perturbations` | Edit that input and see which layers move |
| `list_experiments`, `run_experiment`, `get_experiment_result`, `render_experiment`, `cancel_experiment` | Deep dream and the Captum attributions — `run_experiment(video=True)` records a dream's whole ascent to MP4 |
| `set_auto_run_experiments` | Stop open experiment pages re-running on the thread you need |
| `start_recording`, `stop_recording`, `discard_recording`, `list_recordings` | Record a view to MP4, one frame per visualization update |
| `save_snapshot` | Save one still of a view as a PNG file to hand a human |

**Time travel is the one that changes how debugging goes.** An agent that has run past a divergence normally has to restart the script. Instead it can jump back to the epoch before it, this time with `configure_debug_checks(interval_batches=1)` and the suspect layers watched — so the second pass sees what the first one missed. It needs the training loop driven by `session.epochs()` with `session.restore_point()`; `get_time_travel_status` says whether yours is.

**Probes hold the input still.** Stepping normally changes the weights *and* the batch at once, which makes it hard to say which caused a change. Pinning a batch re-runs the model on that one input at every capture, so what you see between steps is the weights alone. `set_probe_mode("eval")` runs the probe with BatchNorm on its running statistics — a model that looks fine in train mode and broken in eval usually has statistics that have drifted.

**Experiments run on the paused training thread**, so pause first, or the request queues until the next pause. There is a wall-clock ceiling on each run, and `run_experiment` returns statistics while `render_experiment` draws the result. A result you poll before it exists is not simply missing: `get_experiment_result` reports the request's `stage` — `running`, `queued` (with `queued_ahead`), or `absent` — so waiting longer and re-requesting are told apart.

**A deep dream can hand back the whole ascent.** `render_experiment` shows what a channel converged *to*; `run_experiment(..., video=True)` writes the run itself to an MP4 and returns the path, which is how you tell an image that formed steadily from one that went somewhere and came back. The page streams every step to a watching human, but a tool call only sees the snapshot it polled — so the video is the agent-side equivalent, and it records the ~20 evenly spaced snapshots a run publishes unless `params: {"all_steps": true}` asks for one frame per step. Both cost the run time: frames are drawn on the training thread, inside the same ceiling as the ascent.

**Recordings capture change over time.** One frame per visualization update — so `set_update_frequency` is the frame rate, and a run that stays paused records nothing. Start one, let training run, then stop it for the file path to show a human — or discard it, if the take went wrong, so no half-finished file is left looking like a result. `save_snapshot` is the same thing at length one: a single PNG of a view as it stands, written immediately, with nothing to start or stop. Use it for the file you want a human to open — the `render_*` tools return the picture to *you*.

## A worked example

A typical exchange, once the run is paused:

> **You:** Training loss goes to NaN around epoch 3. Find out where it starts.

The agent calls `get_status` to see the run is paused at epoch 0, then `configure_debug_checks(interval_batches=1)` so the checks run on every batch instead of every hundredth, then `run` and polls until the debugger trips. `get_debug_report` names the layer and the fraction of its gradient that went non-finite, and `get_layer_stats` on that layer and the one feeding it shows which of the two was already unhealthy a batch earlier.

Because the agent is driving the same session, you can open the UI at any point and see exactly the batch it stopped on.

## Things worth knowing

**Two positions.** `live_position` is where training is now; `snapshot_position` is the batch the statistics describe. They're identical while paused and diverge once the run advances freely — the tools report both so the agent doesn't read stale numbers as current.

**Non-finite values survive.** JSON has no NaN literal, so the tools report them as the strings `"nan"`, `"inf"` and `"-inf"` rather than as nulls. A layer whose values are *all* non-finite reports a `non_finite_count` instead of pretending it captured nothing — the statistics themselves only ever describe the finite values.

**Long steps don't hang.** A `step` over a slow epoch returns after a timeout with a note that the run is still going; the command stays in effect and the agent polls `get_status`.

**Locked sessions refuse.** On a [locked](playground.md) demo session the control tools return an explicit refusal rather than silently doing nothing, so an agent can't loop forever stepping a run that never moves. Inspection keeps working.
