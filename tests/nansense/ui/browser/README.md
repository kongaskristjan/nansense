# Browser smoke tests

This optional suite checks browser-visible behavior against a tiny, deterministic
CPU training run: pinning and stepping, settings persistence, recording controls,
stats navigation, and a Captum experiment. It fails on browser console errors and
saves screenshots and the server log in a temporary directory. It uses its own
available loopback port and terminates only the server process it starts.

Set up the project's CPU environment, then install the optional browser tools
outside the repository:

```sh
uv sync --group cpu
npm install --prefix /tmp/nansense-browser-tools playwright@1.62.1
/tmp/nansense-browser-tools/node_modules/.bin/playwright install chromium
NODE_PATH=/tmp/nansense-browser-tools/node_modules node tests/nansense/ui/browser/smoke.cjs
```

Run from the repository root. `PYTHON` can override the default `.venv/bin/python`.
The browser suite is separate from pytest so ordinary library tests need neither
Node nor browser downloads. Reuse an existing matching Playwright installation by
setting `NODE_PATH` to its `node_modules` directory.
