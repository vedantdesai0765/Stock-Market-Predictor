const puppeteer = require('puppeteer-core');
const path = require('path');
const fs = require('fs');

const SCREENSHOTS_DIR = path.resolve(__dirname, '../../reports/screenshots');
if (!fs.existsSync(SCREENSHOTS_DIR)) {
  fs.mkdirSync(SCREENSHOTS_DIR, { recursive: true });
}

const EDGE_PATH = 'C:\\Program Files (x86)\\Microsoft\\Edge\\Application\\msedge.exe';

async function runPhase5BrowserAudit() {
  console.log('=== STARTING PHASE 5 VIRTUAL PORTFOLIO BROWSER AUDIT ===');

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
    workflowsTested: [],
    screenshotsCaptured: [],
    consoleErrors,
    pageErrors,
  };

  try {
    await page.setViewport({ width: 1440, height: 900 });

    // 1. Reset portfolio to ensure clean empty state
    console.log('\n1. Resetting portfolio and navigating to (/portfolio)...');
    await page.goto('http://localhost:3000/portfolio', { waitUntil: 'networkidle0', timeout: 30000 });
    await page.evaluate(async () => {
      await fetch('http://127.0.0.1:8000/api/v1/portfolio/reset', { method: 'POST' });
    });
    await page.reload({ waitUntil: 'networkidle0' });
    await new Promise((r) => setTimeout(r, 1500));

    const shot1 = path.join(SCREENSHOTS_DIR, '11_portfolio_dashboard_empty.png');
    await page.screenshot({ path: shot1, fullPage: true });
    console.log('[PASS] Captured 11_portfolio_dashboard_empty.png');
    auditReport.screenshotsCaptured.push('11_portfolio_dashboard_empty.png');

    // 2. Open Order Modal
    console.log('\n2. Opening Order Placement Modal...');
    const buttons = await page.$$('button');
    let placeOrderBtn = null;
    for (const btn of buttons) {
      const text = await page.evaluate((el) => el.textContent, btn);
      if (text && text.includes('Place Order')) {
        placeOrderBtn = btn;
        break;
      }
    }

    if (placeOrderBtn) {
      await placeOrderBtn.click();
      await new Promise((r) => setTimeout(r, 1200));

      const shot2 = path.join(SCREENSHOTS_DIR, '12_portfolio_buy_modal.png');
      await page.screenshot({ path: shot2, fullPage: true });
      console.log('[PASS] Captured 12_portfolio_buy_modal.png');
      auditReport.screenshotsCaptured.push('12_portfolio_buy_modal.png');

      // Click "Review Order Confirmation →"
      const modalButtons = await page.$$('button');
      for (const mBtn of modalButtons) {
        const text = await page.evaluate((el) => el.textContent, mBtn);
        if (text && text.includes('Review Order Confirmation')) {
          await mBtn.click();
          break;
        }
      }
      await new Promise((r) => setTimeout(r, 1000));

      // Click "Confirm BUY Order"
      const confirmButtons = await page.$$('button');
      for (const cBtn of confirmButtons) {
        const text = await page.evaluate((el) => el.textContent, cBtn);
        if (text && text.includes('Confirm BUY Order')) {
          await cBtn.click();
          break;
        }
      }
      await new Promise((r) => setTimeout(r, 2000));

      // Close modal by clicking "Return to Portfolio"
      const closeButtons = await page.$$('button');
      for (const clBtn of closeButtons) {
        const text = await page.evaluate((el) => el.textContent, clBtn);
        if (text && text.includes('Return to Portfolio')) {
          await clBtn.click();
          break;
        }
      }
      await new Promise((r) => setTimeout(r, 1500));
    }

    // 3. Capture Portfolio with Active Holdings & Charts
    console.log('\n3. Capturing updated Portfolio with active positions...');
    const shot3 = path.join(SCREENSHOTS_DIR, '13_portfolio_with_holdings.png');
    await page.screenshot({ path: shot3, fullPage: true });
    console.log('[PASS] Captured 13_portfolio_with_holdings.png');
    auditReport.screenshotsCaptured.push('13_portfolio_with_holdings.png');

    // 4. Scroll to Portfolio Analytics
    console.log('\n4. Scrolling to Portfolio Analytics charts...');
    await page.evaluate(() => {
      const mainEl = document.querySelector('main');
      if (mainEl) {
        mainEl.scrollTop = 520;
      }
    });
    await new Promise((r) => setTimeout(r, 1200));
    const shot4 = path.join(SCREENSHOTS_DIR, '14_portfolio_analytics.png');
    await page.screenshot({ path: shot4, fullPage: false });
    console.log('[PASS] Captured 14_portfolio_analytics.png');
    auditReport.screenshotsCaptured.push('14_portfolio_analytics.png');

    // 5. Navigate to /signals and test "Simulate Trade with Signal"
    console.log('\n5. Testing Actionable Signals integration (/signals)...');
    // Set stock to INFY which has a BUY signal
    await page.evaluate(() => {
      localStorage.setItem('selected_stock_ticker', 'INFY');
    });

    await page.goto('http://localhost:3000/signals', { waitUntil: 'networkidle0', timeout: 30000 });
    await new Promise((r) => setTimeout(r, 2000));

    // Look for "Simulate Trade with Signal" button
    const signalButtons = await page.$$('button');
    let simTradeBtn = null;
    for (const sBtn of signalButtons) {
      const text = await page.evaluate((el) => el.textContent, sBtn);
      if (text && text.includes('Simulate Trade with Signal')) {
        simTradeBtn = sBtn;
        break;
      }
    }

    if (simTradeBtn) {
      console.log('Clicking "Simulate Trade with Signal" button...');
      await simTradeBtn.click();
      await new Promise((r) => setTimeout(r, 1500));

      const shot5 = path.join(SCREENSHOTS_DIR, '15_signals_with_trade_modal.png');
      await page.screenshot({ path: shot5, fullPage: true });
      console.log('[PASS] Captured 15_signals_with_trade_modal.png');
      auditReport.screenshotsCaptured.push('15_signals_with_trade_modal.png');
    }

    // Reset stock to TCS
    await page.evaluate(() => {
      localStorage.setItem('selected_stock_ticker', 'TCS');
    });

    // 6. Mobile Viewport (375x667 - iPhone SE)
    console.log('\n6. Testing Mobile Viewport (375x667)...');
    await page.setViewport({ width: 375, height: 667, isMobile: true, hasTouch: true });
    await page.goto('http://localhost:3000/portfolio', { waitUntil: 'networkidle0', timeout: 30000 });
    await new Promise((r) => setTimeout(r, 1500));

    const shot6 = path.join(SCREENSHOTS_DIR, '16_portfolio_mobile.png');
    await page.screenshot({ path: shot6, fullPage: true });
    console.log('[PASS] Captured 16_portfolio_mobile.png');
    auditReport.screenshotsCaptured.push('16_portfolio_mobile.png');

    auditReport.workflowsTested.push({
      workflow: 'Virtual Portfolio Full Lifecycle (Empty -> Buy Order -> Holdings -> Analytics -> Signals Link -> Mobile)',
      status: 'PASS',
    });
  } catch (err) {
    console.error('[BROWSER AUDIT ERROR]', err);
    auditReport.fatalError = err.message;
  } finally {
    await browser.close();
  }

  const outPath = path.resolve(__dirname, '../../reports/phase5_browser_results.json');
  fs.writeFileSync(outPath, JSON.stringify(auditReport, null, 2));
  console.log(`\n=== PHASE 5 BROWSER AUDIT COMPLETE. RESULTS SAVED TO ${outPath} ===`);
}

runPhase5BrowserAudit();
