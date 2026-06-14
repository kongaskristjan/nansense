---
name: run-example
description: Launch a nansense example (web UI on --nansense-port) and verify it in a browser with the Playwright MCP. Use whenever a change should be checked in the running app — UI changes, new views or layers, example changes — or when asked to run, screenshot, or visually verify an example.
---

# Run and verify a nansense example

## Launch

1. Pick a free port (check with `netstat -tnlp`); if it turns out taken, just
   try another. Never kill processes on ports you didn't open — they may
   belong to the user or other concurrent agents.
2. Start an example in the background and capture its PID:

   ```bash
   uv run examples/standard/main.py --nansense-port <PORT> --device cpu
   ```

   - `examples/standard/main.py` — full wiring (scheduler, time travel):
     ResNet/ViT/LeNet on MNIST/CIFAR10/Imagenette
     (`--model resnet|resnet_deep|vit|lenet`,
     `--dataset mnist|cifar10|imagenette`); `--blocks-per-stage 1` makes
     the ResNet small and fast, `--dataset mnist --model lenet` is the
     lightest full-wiring combination. Add `--distributed` (under
     `torchrun`) to exercise the DDP path.
   - `examples/lightning/main.py` — the PyTorch Lightning wiring
     (`NansenseCallback` + `fit_with_time_travel`); use when testing
     `nansense/lightning.py` changes.
   - `examples/game_of_life/main.py` — synthetic data, no download, fast
     startup; the lightest choice for a quick UI check.
3. If running in a worktree, `data/` is gitignored — symlink it from the main
   tree first to skip the dataset download.
4. Poll until the UI is up — training pauses on the first batch, so the page
   serves right after startup (first run may download the dataset first):

   ```bash
   curl -s -o /dev/null -w "%{http_code}" http://127.0.0.1:<PORT>/  # until 200
   ```

## Verify (Playwright MCP)

- `browser_navigate` to `http://127.0.0.1:<PORT>/`, then `browser_snapshot`
  for structure or `browser_take_screenshot` for rendering.
- Pages: `/` (top bar, architecture diagram, layer cards, input pane),
  `/watch` (toggle "Watch" on a layer card first), `/weights?layer=...`,
  `/experiment?layer=...`.
- Drive training from the top bar (step batch / epoch) when activations need
  to change between checks.
- After UI changes, also check `browser_console_messages` for JS errors.

## Tear down

Kill only the PID captured at launch (`kill <PID>`), then `browser_close`.
Avoid `pkill -f` by pattern — it can hit the user's or other agents' sessions.
