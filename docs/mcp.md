# Debug with a coding agent (MCP)

NaNsense speaks [MCP](https://modelcontextprotocol.io), so a coding agent can drive the debugger itself: pause your training run, read any layer's activations and gradients, and find out which layer started producing NaNs — without you clicking through the UI and pasting screenshots back into the chat.

The agent connects to the same session the browser shows. Both are front-ends onto one paused training loop, so you can watch in the UI while the agent works.

## Connect

The endpoint is served alongside the UI, on the same port. Start your run as usual:

```bash
uv run --group cuda examples/standard/main.py --nansense-port 8080
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

Statistics come from two places, and the distinction matters when you read the agent's reasoning. `get_layer_stats` reads the last captured batch and covers **any** layer. `get_stats_history` reads the running accumulators, which only cover **watched** layers — so the agent watches a layer first, lets a few epochs run, and then asks for the trend.

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
