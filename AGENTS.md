# Instructions for AI agents

## Project

- Python 3.13
- Package manager: `uv`
  - Always use `cuda` group: `uv sync --group cuda`, `uv run --group cuda examples/...`.
- Directory structure:
  - 'nansense/' - NaNsense visualization library (no training)
  - `examples/` - Runnable Python examples (fully contains training logic, each example in a separate subdirectory)
  - `tests/` - Tests for both examples and the NaNsense library
  - `assets/` - Assets
  - `docs/` + `mkdocs.yml` - User documentation site (MkDocs Material), published to GitHub Pages
  - `README.md` - Landing page: pitch, hero video, prominent links (playground, one-prompt integration, docs), a teaser example command; keep it short — examples and wiring details live in `docs/`
  - `INTERNALS.md` - High level overview of NaNsense library internals

## Prompts

- If the user request is unclear, ask questions instead of proceeding to implement

## Worktrees

- When running in agents mode, branch every new change request `main` using a git worktree.
  - Make, test and commit your changes in a worktree and then rebase the changes to `main`.
  - Use a meaningful name for the worktree.
  - For automatic merges, additional verification after the merge is usually not necessary.
  - For complex/manual merges, run the tests/checks again. For UI conflicts, verify with Playwright.
  - Fast forward merges if there's no conflicts.

## Code quality

- Consider moving files to subdirectories if a large number of files appear in `lib/` or `tests/`
- Proactively refactor clearly redundant or suboptimal code. Refactor big functions into smaller ones if reasonable.

## Type hints

- All function signatures must have type hints (parameters and return types).
- Variables whose type cannot be inferred from the during initialization must have type hints (e.g. `items: list[str] = []`).
- Do not annotate variables where the type can be inferred from the right-hand side.

## Testing

- Framework: pytest
- Keep tests reasonably fast: no sleeps, no many-batch neural network training, use small tensors etc.
- Use `pytest.mark.parametrize` for testing multiple inputs instead of duplicating test functions.

## Commit discipline

- Every user requested change should be accompanied by a commit. Don't ask for permission, just do it as the last step.
  - If multiple unrelated changes are requested within one prompt, the separate commits should be created.
  - You can amend the last commit if it clearly introduced a bug.
- Most commits should include corresponding test additions or changes. High level changes should incorporate documentation changes.
- Before committing: `uv run --group cuda pytest && uv run --group cuda ty check`
- Before committing: run the code. For UI testing, you can run some of the examples and use the playwright MCP.
  - Use any `--nansense-port [NUMBER]`. If a port has been taken, just try another one. Don't kill sessions on other ports, as they may have been started by the user or other concurrent agents.
- Keep `README.md` short and lean: only update it when previously documented behaviour becomes invalid, not to advertise every new feature. Keep `INTERNALS.md` up to date.
- Don't push unless explicitly commanded so.

## Docs site

- `docs/` is the user-facing documentation (usage guides + public API); `INTERNALS.md` stays a repo file and is intentionally not part of the site.
- `.github/workflows/docs.yml` deploys with `mike`: pushes to `main` update the `dev` version, published releases update `X.Y` + `latest` (the default).
- The API reference (`docs/api.md`) renders public docstrings via mkdocstrings — keep docstrings of public symbols current when changing them.
- Verify docs changes with `uv run --only-group docs mkdocs build --strict` (also useful: `mkdocs serve`).

