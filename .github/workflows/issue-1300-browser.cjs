const { chromium } = require('playwright');
const fs = require('node:fs');

(async () => {
  const [id, log] = process.argv.slice(2);
  const browser = await chromium.launch();
  try {
    const page = await browser.newPage();
    const record = (event) => fs.appendFileSync(log, JSON.stringify(event) + '\n');
    page.on('request', (req) => record({ at: Date.now(), method: req.method(),
      url: req.url(), body: req.postData() }));
    page.on('response', (res) => record({ at: Date.now(), status: res.status(), url: res.url() }));
    page.on('requestfailed', (req) => record({ url: req.url(), error: req.failure() }));
    page.on('dialog', (dialog) => dialog.accept());
    await page.goto('http://127.0.0.1:18000/');
    await page.getByRole('button', { name: 'Models', exact: true }).click();
    await page.getByRole('searchbox', { name: 'Search models' }).fill('HTDemucs');
    const row = page.locator('.model-row').filter({ has: page.locator('option[value="' + id + '"]') });
    await row.locator('select').selectOption(id, { timeout: 45000 });
    const request = page.waitForResponse(res => res.url().endsWith('/v1/ui/models/install') &&
      res.request().method() === 'POST', { timeout: 60000 });
    await row.getByRole('button', { name: 'Download', exact: true }).click({ timeout: 55000 });
    const response = await request;
    if (!response.ok()) throw new Error(await response.text());
    await Promise.race([
      row.getByRole('button', { name: 'Open in Studio', exact: true }).waitFor({ timeout: 180000 }),
      row.getByText('Download failed', { exact: true }).first().waitFor({ timeout: 180000 }),
      page.waitForEvent('requestfailed', { predicate: req => req.url().includes('/install-status'), timeout: 180000 }),
    ]);
  } finally {
    await browser.close();
  }
})().catch(err => { console.error(err); process.exitCode = 1; });
