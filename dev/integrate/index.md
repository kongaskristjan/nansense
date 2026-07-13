# Integrate with one prompt

Run this prompt from your project with a coding agent (Claude Code, Cursor, …): it reads the nansense docs and wires the debugger into your training loop.

Gate nansense behind a `--debugger-port PORT` flag, so it stays a zero-overhead no-op unless a port is passed *(recommended)*

```
Add the nansense PyTorch debugger to this project's training loop.

Read the docs first — everything in one file: https://kongaskristjan.github.io/nansense/llms-full.txt
(page-by-page index: https://kongaskristjan.github.io/nansense/llms.txt)

Requirements: PyTorch >= 2.3, Python 3.10-3.14, and an eager training loop you can wrap (PyTorch Lightning is also supported). If this project can't meet these, stop and ask me how to proceed.

Then:
1. Add nansense to the project's dependencies (`pip install nansense`).
2. Wire nansense.start() / session.epochs() / session.batches() / session.close() into the training loop as the wiring guide shows; enable time travel if there's a clean epoch loop.
3. Add a `--debugger-port PORT` CLI flag and pass `port=that_port, enabled=that_port is not None`, so nansense stays a zero-overhead no-op unless a port is given.
4. Show me the diff and the exact command to launch training with the debugger.
```

## Prefer to look first?

Plenty of people want to see what the change looks like before letting an agent touch their training loop:

- The [Wiring guide](https://kongaskristjan.github.io/nansense/dev/wiring/index.md) shows the exact code the prompt produces, for raw PyTorch and PyTorch Lightning — plus time travel, input display and distributed training.
- [Getting started](https://kongaskristjan.github.io/nansense/dev/getting-started/index.md) runs a bundled example on your machine, without touching your project.
- The [Playground](https://kongaskristjan.github.io/nansense/dev/playground/index.md) is a trained network you can inspect right in the browser, no install at all.
