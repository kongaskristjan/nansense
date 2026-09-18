// Optional integration suite; see README.md in this directory.
const assert = require('node:assert/strict');
const { spawn } = require('node:child_process');
const fs = require('node:fs/promises');
const os = require('node:os');
const path = require('node:path');
const net = require('node:net');
const { chromium, expect } = require('playwright/test');

async function unusedPort() {
  const socket = net.createServer();
  await new Promise(resolve => socket.listen(0, '127.0.0.1', resolve));
  const port = socket.address().port;
  await new Promise(resolve => socket.close(resolve));
  return port;
}

async function main() {
  const root = path.resolve(__dirname, '../../../..');
  const port = await unusedPort();
  const artifacts = await fs.mkdtemp(path.join(os.tmpdir(), 'nansense-browser-'));
  const server = spawn(process.env.PYTHON || path.join(root, '.venv/bin/python'),
    ['-m', 'tests.nansense.ui.browser.server', '--port', String(port)],
    { cwd: artifacts, env: { ...process.env, PYTHONPATH: root } });
  let log = '';
  server.stdout.on('data', chunk => { log += chunk; });
  server.stderr.on('data', chunk => { log += chunk; });
  let browser;
  try {
    browser = await chromium.launch({ headless: true });
    const page = await browser.newPage({ viewport: { width: 1440, height: 1000 } });
    const errors = [];
    page.on('pageerror', error => errors.push(String(error)));
    page.on('console', message => {
      if (message.type() === 'error') errors.push(message.text());
    });
    const url = `http://127.0.0.1:${port}`;
    await expect(async () => {
      assert.equal(server.exitCode, null, log);
      const response = await page.request.get(url);
      assert.equal(response.status(), 200);
    }).toPass({ timeout: 30000 });
    await page.goto(url);
    await expect(page.getByRole('button', { name: 'Step Batch', exact: true })).toBeEnabled();
    await expect(page.getByText('conv1', { exact: true }).first()).toBeVisible();
    const pane = page.locator('[data-resize-pane="main-input"]');
    const initial = await pane.boundingBox();
    await page.setViewportSize({ width: 900, height: 1000 });
    await expect(async () => assert.ok((await pane.boundingBox()).width < initial.width)).toPass();
    await page.setViewportSize({ width: 1440, height: 1000 });
    const handle = await page.locator('[data-resize-key="main-input"]').boundingBox();
    await page.mouse.move(handle.x + handle.width / 2, handle.y + 100);
    await page.mouse.down();
    await page.mouse.move(handle.x - 180, handle.y + 100, { steps: 8 });
    await page.mouse.up();
    await expect(async () => assert.ok((await pane.boundingBox()).width > initial.width + 150)).toPass();
    await page.getByRole('switch', { name: 'Pin batch', exact: true }).click();
    await expect(page.getByRole('switch', { name: 'Pin batch', exact: true })).toBeChecked();
    await page.getByRole('button', { name: 'Step Batch', exact: true }).click();
    await expect(page.getByText('epoch 0/1 | train batch 1/20', { exact: true })).toBeVisible();
    await page.screenshot({ path: path.join(artifacts, 'main.png'), fullPage: true, animations: 'disabled' });
    await page.getByRole('button', { name: 'Settings', exact: true }).click();
    const dialog = page.getByRole('dialog');
    await expect(dialog.getByText('Recording', { exact: true })).toBeVisible();
    await dialog.getByLabel('n', { exact: true }).fill('2');
    await dialog.getByLabel('n', { exact: true }).press('Tab');
    await dialog.getByRole('button', { name: 'Record', exact: true }).click();
    await expect(dialog.getByText('Currently recording', { exact: true })).toBeVisible();
    await expect(dialog.getByLabel('n', { exact: true })).toBeDisabled();
    await dialog.getByRole('button', { name: 'Close', exact: true }).click();
    await page.getByRole('button', { name: 'Step Batch', exact: true }).click();
    await page.getByRole('button', { name: 'Settings', exact: true }).click();
    await expect(dialog.getByText('Currently recording', { exact: true })).toBeVisible();
    await dialog.getByText('Currently recording', { exact: true }).scrollIntoViewIfNeeded();
    await page.screenshot({ path: path.join(artifacts, 'settings.png'), fullPage: true, animations: 'disabled' });
    await dialog.getByRole('button', { name: 'Delete', exact: true }).click();
    await expect(dialog.getByText('Currently recording', { exact: true })).toHaveCount(0);
    await expect(dialog.getByLabel('n', { exact: true })).toBeEnabled();
    await expect(dialog.getByLabel('n', { exact: true })).toHaveValue('2');
    await dialog.getByRole('button', { name: 'Close', exact: true }).click();
    await page.goto(`${url}/stats?layer=conv1`);
    await expect(page.getByRole('button', { name: 'Settings', exact: true })).toBeVisible();
    await expect(page.getByText("Loading this layer's statistics…", { exact: true })).toHaveCount(0);
    await expect(page.locator('.js-plotly-plot').first()).toBeVisible();
    await page.getByRole('checkbox', { name: 'Log y', exact: true }).click();
    await expect(page.getByRole('checkbox', { name: 'Log y', exact: true })).toBeChecked();
    await expect(async () => {
      const axis = await page.locator('.js-plotly-plot').first().evaluate(plot => plot.layout.yaxis.type);
      assert.equal(axis, 'log');
    }).toPass();
    await page.screenshot({ path: path.join(artifacts, 'stats.png'), fullPage: true, animations: 'disabled' });
    await page.getByLabel('View', { exact: true }).click();
    await page.getByRole('option', { name: 'MIN/MAX', exact: true }).click();
    await expect(page.getByRole('checkbox', { name: 'Log y', exact: true })).toBeHidden();
    await expect(page.locator('img[src^="data:image"]').first()).toBeVisible();
    await page.screenshot({ path: path.join(artifacts, 'stats-minmax.png'), fullPage: true, animations: 'disabled' });
    await page.goto(`${url}/experiment?layer=conv1`);
    await expect(page.getByText('Parameters', { exact: true })).toBeVisible();
    await page.getByLabel('Experiment', { exact: true }).click();
    await page.getByRole('option', { name: 'Neuron Gradient (Captum)', exact: true }).click();
    await expect(page.getByText(/^Neuron Gradient \(Captum\) — done$/)).toBeVisible({ timeout: 15000 });
    await expect(page.getByText('Sample 0', { exact: true })).toBeVisible();
    await page.screenshot({ path: path.join(artifacts, 'experiment.png'), fullPage: true, animations: 'disabled' });
    assert.deepEqual(errors, [], 'Browser errors');
    console.log(`Browser smoke passed. Screenshots: ${artifacts}`);
  } finally {
    if (browser) await browser.close();
    server.kill('SIGTERM');
    await new Promise(resolve => {
      if (server.exitCode !== null || server.signalCode !== null) resolve();
      else server.once('exit', resolve);
    });
    await fs.writeFile(path.join(artifacts, 'server.log'), log);
    console.log(`Server log: ${path.join(artifacts, 'server.log')}`);
  }
}

main().catch(error => { console.error(error); process.exitCode = 1; });
