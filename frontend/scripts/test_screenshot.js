const puppeteer = require('puppeteer-core');
const path = require('path');

(async () => {
  const browser = await puppeteer.launch({
    executablePath: 'C:\\Program Files (x86)\\Microsoft\\Edge\\Application\\msedge.exe',
    headless: 'new',
    args: ['--no-sandbox', '--disable-setuid-sandbox']
  });

  const page = await browser.newPage();
  await page.setViewport({ width: 1440, height: 900 });
  await page.goto('http://localhost:3000', { waitUntil: 'networkidle0', timeout: 15000 });

  const outPath = path.resolve(__dirname, '../../reports/screenshots/01_overview_page.png');
  await page.screenshot({ path: outPath, fullPage: true });
  console.log('Saved screenshot to:', outPath);

  await browser.close();
})();
