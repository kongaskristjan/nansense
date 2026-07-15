# Integrate with one prompt

Run this prompt from your project with a coding agent (Claude Code, Cursor, …): it reads the NaNsense docs and wires the debugger into your training loop.

<div class="nansense-prompt" markdown="1">

```text
Add the NaNsense PyTorch debugger to this project's training loop.

Read the docs first: https://kongaskristjan.github.io/nansense/llms-full.txt
(page-by-page index is here https://kongaskristjan.github.io/nansense/llms.txt)

Requirements: PyTorch >= 2.3, Python 3.10-3.14, and an eager training loop you can wrap (PyTorch Lightning is also supported). If this project can't meet these, ask me how to proceed.

Then:
1. Add NaNsense to the project's dependencies (`pip install nansense`).
2. Wire nansense.start() / session.epochs() / session.batches() / session.close() into the training loop as the wiring guide shows; enable time travel if there's a clean epoch loop.
3. Add a `--debugger-port PORT` CLI flag and pass `port=that_port, enabled=that_port is not None`, so NaNsense stays a zero-overhead no-op unless a port is given.
4. Show me the exact command to launch training with the debugger.
```

### Prompt settings:

<label class="nansense-prompt-option">
<input type="checkbox" id="nansense-port-only" checked>
<span>Gate NaNsense behind a <code>--debugger-port PORT</code> flag, so it stays a zero-overhead no-op unless a port is passed <em>(recommended)</em></span>
</label>


</div>

## Prefer to look first?

- The [Wiring guide](wiring.md) shows the exact code the prompt produces, for raw PyTorch and PyTorch Lightning — plus time travel, input display and distributed training.
- [Getting started](getting-started.md) runs a bundled example on your machine, without touching your project.
- The [Playground](playground.md) is a trained network you can inspect right in the browser, no install at all.
