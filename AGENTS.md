# Instructions for AI agents

## Project

- Python 3.13
- Package manager: `uv`
  - Installing/running: `uv sync --group cuda`, `uv run --group cuda examples/...`.
  - If the group is not in memory, investigate with `nvidia-smi` and other commands.
  - After investigating, write this group into memory so you don't waste time researching the same thing again.
- Directory structure:
  - 'nansense/' - NaNsense visualization library (no training)
  - `examples/` - Runnable Python examples (fully contains training logic, each example in a separate subdirectory)
  - `tests/` - Tests for both examples and the NaNsense library
  - `assets/` - Assets
  - `docs/` + `mkdocs.yml` - User documentation site (MkDocs Material), published to GitHub Pages
  - `README.md` - Landing page: pitch, hero video, prominent links (playground, one-prompt integration, docs), a teaser example command; keep it short — examples and wiring details live in `docs/`
  - `INTERNALS.md` - High level overview of NaNsense library internals

## Prompts

- Ask questions before instead of guessing, especially with
  - UX decisions
  - library architecture, that may have future impact on refactorability

## Sub-agents

- Use sub-agents generously, according to your own best judgement
  - A tiny task can be done on main agent, but bigger ones generally are handed to sub-agents
  - A rule of thumb: the main agent gets many different implementation requests and should not fill it's context
- If a large model is used for main agent, then easier tasks can be delegated to smaller models, eg. Fable -> Opus

## Worktrees

- Branch every new change request from the current branch using a git worktree.
  - Make, test and commit your changes in a worktree and then rebase the changes to the original branch.
  - Use a meaningful name for the worktree.
  - For automatic merges, additional verification after the merge is usually not necessary, but more complex ones may warrant additional testing.
- Merge back to the original branch (including main branch if this is where we started)
  - Never report a task done with its commits reachable only from the worktree branch. If something genuinely blocks, report to user.
  - Never push unless instructed so

## MCP server

- `nansense/mcp_server.py` (tools) + `nansense/mcp_views.py` (JSON views) expose the
  debugger to coding agents; `serve()` mounts it alongside the UI.
- **Keep the MCP surface at feature parity with the UI** — both are front-ends onto
  the same `Session`. Add or change the matching tool in the same commit as a UI
  capability. Parity is about capability, not presentation: agents may get JSON/text input where a page might draw pixels.

## Code quality

- Consider moving files to subdirectories if a large number of files appear in `lib/` or `tests/`
- Proactively refactor clearly redundant or suboptimal code. Refactor big functions into smaller ones if reasonable.
- Avoid verbosity in
  - comments
  - tests. Avoid overlapping tests just because of "more testing" of same feature

## Type hints

- All function signatures must have type hints (parameters and return types).
- Variables whose type cannot be inferred from the during initialization must have type hints (e.g. `items: list[str] = []`).
- Do not annotate variables where the type can be inferred from the right-hand side.

## Testing

- Framework: pytest
- Keep tests reasonably fast: no sleeps, no many-batch neural network training, use small tensors etc.
- Use `pytest.mark.parametrize` for testing multiple inputs instead of duplicating test functions.
- The suite runs in parallel by default (`pytest-xdist`, configured in `pyproject.toml`). Pass `-n0` for a serial run when debugging a failure or reading print output.
  - Every worker re-imports torch and re-collects the suite, so that fixed cost dominates the run — a test that is slow *once* costs far more than the same milliseconds spread over many tests.
  - A module-scoped fixture runs once *per worker*. When one is expensive, mark the module `pytest.mark.xdist_group("<name>")` so its tests stay on a single worker (see `tests/examples/playground/test_main.py`).

## Commit discipline

- Every user requested change should be accompanied by a commit. Don't ask for permission, just do it as the last step.
  - If multiple unrelated changes are requested within one prompt, the separate commits should be created.
- Most commits should include corresponding test additions or changes. High level changes should incorporate documentation changes.
- Before committing (assuming `cuda` group: `uv run --group cuda pytest && uv run --group cuda ty check`
- Before committing: run the code. For any UI changes, you can run some of the examples and use the playwright MCP. If you're changing interactive or non-trivial parts of the documentation, verify the functionality and *visual aesthetics* with Playwright.
  - Use any `--nansense-port [NUMBER]`. If a port has been taken, just try another one. Don't kill sessions on other ports, as they may have been started by the user or other concurrent agents.
- Keep `README.md` short and lean: only update it when previously documented behaviour becomes invalid, not to advertise every new feature. Keep `INTERNALS.md` up to date.
- Don't push unless explicitly commanded so.

## Docs site

- `docs/` is the user-facing documentation (usage guides + public API); `INTERNALS.md` stays a repo file and is intentionally not part of the site.
- The API reference (`docs/api.md`) renders public docstrings via mkdocstrings - keep docstrings of public symbols current when changing them.
- Verify docs changes with `uv run --only-group docs mkdocs build --strict` (also useful: `mkdocs serve`).
