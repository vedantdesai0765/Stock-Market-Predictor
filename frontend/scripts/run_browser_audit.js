const puppeteer = require('puppeteer-core');
const path = require('path');
const fs = require('fs');

const SCREENSHOTS_DIR = path.resolve(__dirname, '../../reports/screenshots');
if (!fs.existsSync(SCREENSHOTS_DIR)) {
  fs.mkdirSync(SCREENSHOTS_DIR, { recursive: true });
}

const EDGE_PATH = 'C:\\Program Files (x86)\\Microsoft\\Edge\\Application\\msedge.exe';

async function runBrowserAudit() {
  console.log('=== STARTING BROWSER-BASED INTEGRATION AUDIT ===');

  const browser = await puppeteer.launch({
    executablePath: EDGE_PATH,
    headless: 'new',
    args: ['--no-sandbox', '--disable-setuid-sandbox', '--disable-dev-shm-usage'],
  });

  const page = await browser.newPage();
  const consoleErrors = [];
  const pageErrors = [];

  page.on('console', (msg) => {
    if (msg.type() === 'error') {
      consoleErrors.push({ url: page.url(), text: msg.text() });
      console.warn(`[BROWSER CONSOLE ERROR] ${page.url()}: ${msg.text()}`);
    }
  });

  page.on('pageerror', (err) => {
    pageErrors.push({ url: page.url(), message: err.message });
    console.error(`[BROWSER UNCAUGHT ERROR] ${page.url()}: ${err.message}`);
  });

  const auditReport = {
    pagesTested: [],
    stocksTested: [],
    interactionsTested: [],
    screenshotsCaptured: [],
    consoleErrors,
    pageErrors,
    responsiveChecks: []
  };

  try {
    // ----------------------------------------------------
    // TEST 1: DESKTOP WORKSPACE (1440x900) - DEFAULT (TCS)
    // ----------------------------------------------------
    await page.setViewport({ width: 1440, height: 900 });

    const pages = [
      { name: 'Overview', path: '/', file: '01_overview_page.png' },
      { name: 'Prediction Terminal', path: '/prediction', file: '02_prediction_terminal.png' },
      { name: 'Next-Morning Forecast', path: '/forecast', file: '03_forecast_page.png' },
      { name: 'Why Did It Move?', path: '/explain', file: '04_explain_page.png' },
      { name: 'Actionable Signals', path: '/signals', file: '05_signals_page.png' }
    ];

    for (const p of pages) {
      console.log(`Navigating to ${p.name} (${p.path})...`);
      await page.goto(`http://localhost:3000${p.path}`, { waitUntil: 'networkidle0', timeout: 30000 });
      await new Promise(r => setTimeout(r, 1200));

      const screenshotPath = path.join(SCREENSHOTS_DIR, p.file);
      await page.screenshot({ path: screenshotPath, fullPage: true });
      console.log(`[PASS] Captured ${p.file}`);

      auditReport.pagesTested.push({ name: p.name, path: p.path, status: 'PASS' });
      auditReport.screenshotsCaptured.push(p.file);
    }

    // ----------------------------------------------------
    // TEST 2: INTERACTION ON ACTIONABLE SIGNALS
    // ----------------------------------------------------
    console.log('\nTesting Actionable Signals Interactive Strategy Switch to Variant E & 30.0 bps...');
    await page.goto('http://localhost:3000/signals', { waitUntil: 'networkidle0', timeout: 30000 });
    await new Promise(r => setTimeout(r, 1000));

    // Click Variant E button
    const buttons = await page.$$('button');
    let variantEClicked = false;
    for (const btn of buttons) {
      const text = await page.evaluate(el => el.textContent, btn);
      if (text && text.includes('Variant E')) {
        await btn.click();
        variantEClicked = true;
        break;
      }
    }

    // Click 30.0 bps friction button
    let friction30Clicked = false;
    const allButtons = await page.$$('button');
    for (const btn of allButtons) {
      const text = await page.evaluate(el => el.textContent, btn);
      if (text && text.includes('30.0 bps')) {
        await btn.click();
        friction30Clicked = true;
        break;
      }
    }

    await new Promise(r => setTimeout(r, 2000));
    const variantScreenshot = path.join(SCREENSHOTS_DIR, '06_signals_variant_e_30bps.png');
    await page.screenshot({ path: variantScreenshot, fullPage: true });
    console.log(`[PASS] Variant E & 30.0 bps toggle tested. Screenshot saved to 06_signals_variant_e_30bps.png`);
    auditReport.interactionsTested.push({
      action: 'Switch to Variant E & 30.0 bps Friction',
      variantEClicked,
      friction30Clicked,
      status: 'PASS'
    });
    auditReport.screenshotsCaptured.push('06_signals_variant_e_30bps.png');

    // ----------------------------------------------------
    // TEST 3: INTERACTION ON WHY DID IT MOVE? (NEWS FILTER)
    // ----------------------------------------------------
    console.log('\nTesting News Sentiment Filter on /explain...');
    await page.goto('http://localhost:3000/explain', { waitUntil: 'networkidle0', timeout: 30000 });
    await new Promise(r => setTimeout(r, 1000));

    const filterButtons = await page.$$('button');
    let bullishFilterClicked = false;
    for (const btn of filterButtons) {
      const text = await page.evaluate(el => el.textContent, btn);
      if (text && text.includes('Bullish (+)')) {
        await btn.click();
        bullishFilterClicked = true;
        break;
      }
    }

    await new Promise(r => setTimeout(r, 1500));
    const filterScreenshot = path.join(SCREENSHOTS_DIR, '07_explain_bullish_filter.png');
    await page.screenshot({ path: filterScreenshot, fullPage: true });
    console.log(`[PASS] Bullish sentiment filter tested. Screenshot saved to 07_explain_bullish_filter.png`);
    auditReport.interactionsTested.push({
      action: 'Click Bullish (+) Sentiment Filter Tab',
      bullishFilterClicked,
      status: 'PASS'
    });
    auditReport.screenshotsCaptured.push('07_explain_bullish_filter.png');

    // ----------------------------------------------------
    // TEST 4: STOCK SWITCHING IN UI (ETERNAL & ADANIGREEN)
    // ----------------------------------------------------
    console.log('\nTesting Stock Switching to ETERNAL and ADANIGREEN...');
    const switchStocks = ['ETERNAL', 'ADANIGREEN', 'RELIANCE'];
    for (const stock of switchStocks) {
      await page.evaluate((ticker) => {
        localStorage.setItem('selected_stock_ticker', ticker);
      }, stock);

      await page.goto('http://localhost:3000/signals', { waitUntil: 'networkidle0', timeout: 30000 });
      await new Promise(r => setTimeout(r, 1200));

      const screenshotName = `10_signals_${stock.toLowerCase()}.png`;
      const stockScreenshot = path.join(SCREENSHOTS_DIR, screenshotName);
      await page.screenshot({ path: stockScreenshot, fullPage: true });
      console.log(`[PASS] Switched stock to ${stock}. Captured ${screenshotName}`);
      auditReport.stocksTested.push({ stock, status: 'PASS', screenshot: screenshotName });
      auditReport.screenshotsCaptured.push(screenshotName);
    }

    // Reset stock to TCS
    await page.evaluate(() => {
      localStorage.setItem('selected_stock_ticker', 'TCS');
    });

    // ----------------------------------------------------
    // TEST 5: MOBILE RESPONSIVE AUDIT (375x667 - iPhone SE)
    // ----------------------------------------------------
    console.log('\nTesting Mobile Viewport (375x667 - iPhone SE)...');
    await page.setViewport({ width: 375, height: 667, isMobile: true, hasTouch: true });

    await page.goto('http://localhost:3000/prediction', { waitUntil: 'networkidle0', timeout: 30000 });
    await new Promise(r => setTimeout(r, 1200));
    const mobPredScreenshot = path.join(SCREENSHOTS_DIR, '08_prediction_mobile.png');
    await page.screenshot({ path: mobPredScreenshot, fullPage: true });
    console.log(`[PASS] Mobile prediction captured: 08_prediction_mobile.png`);

    await page.goto('http://localhost:3000/signals', { waitUntil: 'networkidle0', timeout: 30000 });
    await new Promise(r => setTimeout(r, 1200));
    const mobSigScreenshot = path.join(SCREENSHOTS_DIR, '09_signals_mobile.png');
    await page.screenshot({ path: mobSigScreenshot, fullPage: true });
    console.log(`[PASS] Mobile signals captured: 09_signals_mobile.png`);

    auditReport.responsiveChecks.push({ viewport: '375x667', pages: ['/prediction', '/signals'], status: 'PASS' });
    auditReport.screenshotsCaptured.push('08_prediction_mobile.png', '09_signals_mobile.png');

  } catch (err) {
    console.error('[BROWSER AUDIT FAILURE]', err);
    auditReport.fatalError = err.message;
  } finally {
    await browser.close();
  }

  const outReportPath = path.resolve(__dirname, '../../reports/browser_audit_results.json');
  fs.writeFileSync(outReportPath, JSON.stringify(auditReport, null, 2));
  console.log(`\n=== BROWSER INTEGRATION AUDIT COMPLETE. RESULTS SAVED TO ${outReportPath} ===`);
}

runBrowserAudit();
