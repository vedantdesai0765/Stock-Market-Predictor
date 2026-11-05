# Phase 5 Implementation Report: Virtual Portfolio & Paper Trading

**Platform:** Stock Market Predictor — Institutional Financial Analytics Terminal
**Phase:** 5 (Virtual Portfolio & Paper Trading Engine)
**Date:** November 2026
**Status:** Completed & Production Verified

---

## 1. Executive Summary

Phase 5 introduces a full-featured, institutional-grade **Virtual Portfolio & Paper Trading** environment into the Stock Market Predictor platform. Designed strictly for risk-free simulation, this module enables users to act on machine-learning-driven trading signals and next-morning forecasts using a persistent virtual wallet initialized with **₹1,00,000.00 (1 Lakh)** in paper capital.

All simulated executions are subject to realistic Indian statutory transaction frictions (**7.5 bps / 0.075% one-way delivery fee**, reflecting the standard 15 bps round-trip friction hurdle applied in the research backtesting engine). Market valuations and execution prices are derived exclusively from genuine historical closing prices recorded in the research dataset, ensuring mathematical integrity with zero fabricated prices.

The system connects seamlessly with the **Actionable Trading Signals** module (Phase 4), allowing users to execute one-click simulated orders directly from model-recommended buy/sell opportunities, while strictly enforcing risk gates (`HOLD` and `NO SIGNAL` states cannot execute trades).

```
+--------------------------------------------------------------------------------------------------+
|                               PHASE 5 TARGET SYSTEM ARCHITECTURE                                 |
+--------------------------------------------------------------------------------------------------+
|                                                                                                  |
|   +---------------------------------------+      +-------------------------------------------+   |
|   |          NEXT.JS 14 FRONTEND          |      |              FASTAPI BACKEND              |   |
|   |  - Virtual Portfolio Dashboard        |      |  - /api/v1/portfolio                      |   |
|   |  - 6 Key Performance Metric Cards     |      |  - /api/v1/portfolio/preview              |   |
|   |  - Model Opportunities Quick-Trade    | <==> |  - /api/v1/portfolio/orders (Atomic Tx)   |   |
|   |  - Interactive 2-Step Order Modal     |      |  - /api/v1/portfolio/transactions         |   |
|   |  - Real-Time Statutory Fee Breakdown  |      |  - /api/v1/portfolio/analytics            |   |
|   |  - Equity Progression & Donut Charts  |      |  - /api/v1/portfolio/reset                |   |
|   |  - Auditable Transaction Ledger       |      |                                           |   |
|   |  - Signals "Simulate Trade" Button    |      +---------------------+---------------------+   |
|   +---------------------------------------+                            |                         |
|                                                                        v                         |
|                                                      +---------------------------------------+   |
|                                                      |          SQLITE / SQLALCHEMY          |   |
|                                                      |  - Wallets (₹100k Default Capital)    |   |
|                                                      |  - Holdings (Symbol, Quantity, Cost)  |   |
|                                                      |  - Transactions (Ledger & Idempotency)|   |
|                                                      +---------------------------------------+   |
+--------------------------------------------------------------------------------------------------+
```

---

## 2. Architecture & Database Design

### 2.1 Database Models (`backend/app/models/wallet.py`)
The paper trading engine persists data in `data/app.db` via SQLAlchemy ORM models:

1. **`Wallet`**:
   - `id`: Integer primary key.
   - `name`: Default `"Default Paper Wallet"`.
   - `starting_capital`: `Float`, initialized to `100,000.00` (₹1 Lakh).
   - `cash_balance`: `Float`, dynamically updated upon trade execution.
   - `created_at` & `updated_at`: UTC timestamps.

2. **`Holding`**:
   - `id`: Integer primary key.
   - `wallet_id`: Foreign key linked to `wallets.id`.
   - `symbol`: NSE ticker string (e.g. `TCS`, `INFY`).
   - `quantity`: Integer shares held.
   - `average_buy_price`: Weighted average acquisition price.
   - `updated_at`: Timestamp.

3. **`Transaction`**:
   - `id`: Integer primary key.
   - `wallet_id`: Foreign key linked to `wallets.id`.
   - `symbol`: NSE ticker string.
   - `order_type`: Enum string (`BUY` or `SELL`).
   - `shares`: Quantity of shares executed.
   - `price`: Actual market execution price.
   - `friction`: Statutory transaction fee deducted.
   - `net_amount`: Total cash balance delta ($\pm (\text{price} \times \text{shares} \pm \text{friction})$).
   - `idempotency_key`: Unique UUID/hash string preventing duplicate execution from repeated form submissions or network retries.
   - `signal_source`: Originating model signal identifier (e.g. `MANUAL`, `VARIANT_A_BUY`).
   - `timestamp`: UTC execution timestamp.

### 2.2 Financial Integrity & Concurrency Safeguards
- **Atomic Operations:** All order previews and order executions execute within explicit database transactions (`db.commit()` on success, `db.rollback()` on error).
- **Idempotency Protection:** Every order request accepts an `idempotency_key`. If an order with that key already exists, the server returns the existing transaction response immediately without modifying balances.
- **Strict Cash Invariance:** `BUY` orders are rejected with HTTP 400 if $\text{Required Cash} = (\text{Price} \times \text{Shares}) + \text{Friction} > \text{Available Cash}$. The cash balance can never become negative.
- **Strict Holding Invariance:** `SELL` orders are rejected with HTTP 400 if $\text{Shares to Sell} > \text{Shares Currently Held}$. Naked short selling is disallowed.
- **Zero Fabricated Pricing:** Execution prices are retrieved strictly from `src/dataset.py` / `load_stock_dataset(symbol)` for the latest valid trading date. If a symbol lacks historical data, orders are rejected with HTTP 400.

---

## 3. Indian Statutory Friction & Valuation Methodology

### 3.1 Statutory Delivery Friction (7.5 bps One-Way)
In Indian cash delivery trading, standard statutory charges include Securities Transaction Tax (STT, 0.1% on delivery buy/sell), brokerage (typically 0.03% or flat ₹20), exchange turnover charges, SEBI turnover fees, GST (18% on brokerage/turnover fees), and state stamp duty.

Consistent with the research backtesting engine's **15 bps round-trip friction hurdle**, the virtual execution engine models a **7.5 bps (0.00075)** one-way statutory friction:
$$\text{Friction}_{\text{one-way}} = \text{Gross Trade Value} \times 0.00075$$

- **BUY Order Cash Impact:**
  $$\text{Total Cost} = (\text{Shares} \times \text{Price}) + \text{Friction}$$
  $$\text{Available Cash}_{\text{new}} = \text{Available Cash}_{\text{old}} - \text{Total Cost}$$

- **SELL Order Cash Impact:**
  $$\text{Net Proceeds} = (\text{Shares} \times \text{Price}) - \text{Friction}$$
  $$\text{Available Cash}_{\text{new}} = \text{Available Cash}_{\text{old}} + \text{Net Proceeds}$$

### 3.2 Portfolio Valuation & P&L Accounting Identities
To eliminate rounding drift across partial sales and position accumulation, valuation follows strict accounting identities:

1. **Current Portfolio Valuation:**
   $$\text{Portfolio Value} = \text{Available Cash} + \sum_{i \in \text{Holdings}} (\text{Shares}_i \times \text{Current Price}_i)$$

2. **Total P&L:**
   $$\text{Total P\&L} = \text{Portfolio Value} - \text{Starting Capital}$$

3. **Unrealised P&L:**
   $$\text{Unrealised P\&L} = \sum_{i \in \text{Holdings}} \left[ \text{Shares}_i \times (\text{Current Price}_i - \text{Average Buy Price}_i) \right]$$

4. **Realised P&L:**
   $$\text{Realised P\&L} = \text{Total P\&L} - \text{Unrealised P\&L}$$

5. **Overall Portfolio Return (%):**
   $$\text{Return} = \frac{\text{Total P\&L}}{\text{Starting Capital}} \times 100\%$$

---

## 4. Key Features & Frontend Components

### 4.1 Virtual Portfolio Dashboard (`frontend/src/app/portfolio/page.tsx`)
- **6 Key Metric Cards:**
  1. *Portfolio Valuation:* Current live value vs. ₹1,00,000 initial.
  2. *Available Cash:* Liquid purchasing power vs. invested capital.
  3. *Total P&L:* Real-time profit/loss in INR and percentage return with color tones (`buy` green / `sell` red / `neutral` slate).
  4. *Unrealised P&L:* Floating gain/loss on currently held positions.
  5. *Realised P&L:* Closed profits/losses net of statutory friction.
  6. *Active Position Count:* Number of equities currently in custody.
- **Simulation Notice Banner:** Prominent disclaimer clarifying virtual capital simulation and non-brokerage nature.
- **Reset Button:** One-click reset resetting wallet to ₹1,00,000 and clearing holdings/transactions.

### 4.2 Model-Generated Trading Opportunities Panel
- Displays current directional signals across all 10 supported NSE equities (e.g. TCS, INFY, SBI).
- Recommends `BUY` when model expectancy exceeds the 15 bps friction hurdle.
- Recommends `HOLD` when gross edge is below friction. Execution is deliberately inactive/disabled for `HOLD`.
- Recommends `SELL` when downward probability is calibrated.
- Includes a "Simulate Trade with Signal →" button for each active signal, pre-filling the order execution modal.

### 4.3 Active Holdings Table
- Lists every open position: Symbol, Full Company Name, Shares Held, Average Acquisition Price, Latest Market Price (with genuine historical date badge), Invested Capital, Current Market Value, and Unrealised P&L with percentage return.
- Quick action buttons: **Buy More** (pre-configures BUY modal) and **Sell** (pre-configures SELL modal with current share balance).
- Empty state with guidance and direct "Start First Simulated Trade" button when no positions are open.

### 4.4 Interactive 2-Step Order Placement Modal (`frontend/src/components/portfolio/OrderModal.tsx`)
- Tabbed selector for **BUY ORDER** vs. **SELL ORDER**.
- Stock selector dropdown covering all 10 NSE stocks with live market close display.
- Share quantity input with preset increment buttons (+10, +25, +50, +100).
- Real-time pre-trade cost estimate showing Gross Trade Value, Statutory Friction (7.5 bps), Total Cost / Net Proceeds, and Available Cash or Current Shares Held.
- Built-in validation alerts for insufficient cash or insufficient shares.
- Step 1: "Review Order Confirmation →".
- Step 2: Explicit confirmation modal with detailed trade parameters, cost breakdown, statutory disclaimer, and "Confirm Order" button.
- Step 3: Success receipt with full transaction summary and "Return to Portfolio" button.

### 4.5 Actionable Signals Integration (`frontend/src/app/signals/page.tsx`)
- Directly integrated into the active strategy banner on `/signals`.
- When the active strategy evaluates to `BUY` or `SELL`, a prominent "Simulate Trade with Signal →" button appears.
- When the strategy evaluates to `HOLD` or `NO SIGNAL`, trade simulation is deactivated, enforcing evidence-based discipline.

### 4.6 Portfolio Analytics Charts (`frontend/src/components/charts/PortfolioCharts.tsx`)
1. **Portfolio Equity Progression Curve:**
   - Recharts Area chart plotting cumulative account valuation across transaction timestamps.
   - Smooth gradient fill, tooltips showing Total Valuation, Available Cash, and Invested Capital at each historical execution point.
2. **Asset Allocation Donut Chart:**
   - Recharts Pie chart displaying percentage distribution between Available Liquid Cash and individual equity holdings.
   - Responsive legends and formatted percentage labels.

### 4.7 Auditable Transaction Ledger
- Tabular chronological log of every executed order.
- Columns: Execution Timestamp, Stock Symbol, Order Type badge (`BUY` / `SELL`), Shares, Execution Price, Friction Paid (in INR), Net Cash Impact (green for cash inflows, red for cash outflows), and Signal Source (`MANUAL` or originating strategy variant).

### 4.8 Responsive Layout & Mobile Navigation (`frontend/src/components/layout/Shell.tsx` & `Header.tsx`)
- Desktop: Persistent 64-unit deep navy sidebar (`#0B132B`).
- Mobile Viewport (375px+): Sidebar cleanly transitions into a slide-out drawer toggled via an institutional hamburger button in the header, preventing layout compression on mobile screens.

---

## 5. Automated Backend Test Suite Results

The comprehensive test suite in `backend/tests/test_portfolio.py` verifies all 13 critical functional requirements:

```
============================= test session starts =============================
platform win32 -- Python 3.13.7, pytest-9.2.0, pluggy-1.7.0
rootdir: C:\Users\vedan\Stock-Market-Predictor
collected 25 items

backend/tests/test_api.py::test_health PASSED                            [  4%]
backend/tests/test_api.py::test_stocks_list PASSED                       [  8%]
backend/tests/test_api.py::test_stock_detail PASSED                      [ 12%]
backend/tests/test_api.py::test_stock_not_found PASSED                   [ 16%]
backend/tests/test_api.py::test_prediction_endpoint_and_numerical_parity PASSED [ 20%]
backend/tests/test_api.py::test_forecast_endpoint PASSED                 [ 24%]
backend/tests/test_api.py::test_signals_latest_endpoint PASSED           [ 28%]
backend/tests/test_api.py::test_signals_invalid_variant PASSED           [ 32%]
backend/tests/test_api.py::test_backtest_endpoint_and_numerical_parity PASSED [ 36%]
backend/tests/test_api.py::test_data_status_endpoint PASSED              [ 40%]
backend/tests/test_api.py::test_news_endpoint PASSED                     [ 44%]
backend/tests/test_api.py::test_all_stocks_data_status PASSED            [ 48%]
backend/tests/test_portfolio.py::test_initial_wallet_creation_and_balance PASSED [ 52%]
backend/tests/test_portfolio.py::test_transaction_cost_calculation PASSED [ 56%]
backend/tests/test_portfolio.py::test_successful_buy_order PASSED        [ 60%]
backend/tests/test_portfolio.py::test_successful_sell_order PASSED       [ 64%]
backend/tests/test_portfolio.py::test_insufficient_cash PASSED           [ 68%]
backend/tests/test_portfolio.py::test_insufficient_holdings PASSED       [ 72%]
backend/tests/test_portfolio.py::test_duplicate_order_prevention PASSED  [ 76%]
backend/tests/test_portfolio.py::test_concurrent_order_safety PASSED     [ 80%]
backend/tests/test_portfolio.py::test_database_persistence PASSED        [ 84%]
backend/tests/test_portfolio.py::test_portfolio_valuation_calculation PASSED [ 88%]
backend/tests/test_portfolio.py::test_invalid_quantities PASSED          [ 92%]
backend/tests/test_portfolio.py::test_missing_market_prices_or_invalid_ticker PASSED [ 96%]
backend/tests/test_portfolio.py::test_portfolio_analytics_endpoint PASSED [100%]

======================= 25 passed, 9 warnings in 6.64s ========================
```

### Detailed Breakdown of Phase 5 Test Coverage

| Test ID | Test Function | Purpose & Validation Criteria | Result |
|---|---|---|---|
| 1 | `test_initial_wallet_creation_and_balance` | Verifies wallet initializes with exact ₹1,00,000.00 cash, 0 holdings, 0 P&L. | **PASSED** |
| 2 | `test_transaction_cost_calculation` | Asserts exact 7.5 bps friction calculation on order previews ($10 \times ₹2,342 \times 0.00075 = ₹17.565 \to ₹17.57$). | **PASSED** |
| 3 | `test_successful_buy_order` | Validates cash deduction, holding creation, and transaction logging on BUY. | **PASSED** |
| 4 | `test_successful_sell_order` | Validates position reduction/removal and cash addition net of friction on SELL. | **PASSED** |
| 5 | `test_insufficient_cash` | Validates that orders exceeding available cash fail with HTTP 400. | **PASSED** |
| 6 | `test_insufficient_holdings` | Validates that sell orders exceeding owned shares fail with HTTP 400. | **PASSED** |
| 7 | `test_duplicate_order_prevention` | Tests idempotency: duplicate `idempotency_key` returns original transaction without double debiting. | **PASSED** |
| 8 | `test_concurrent_order_safety` | Executes concurrent orders and confirms wallet consistency without race condition violations. | **PASSED** |
| 9 | `test_database_persistence` | Confirms SQLite database commits survive cross-session query reloads. | **PASSED** |
| 10 | `test_portfolio_valuation_calculation` | Asserts mathematical parity between mark-to-market holdings + cash and reported portfolio valuation. | **PASSED** |
| 11 | `test_invalid_quantities` | Asserts rejection of zero, negative, or non-integer order quantities. | **PASSED** |
| 12 | `test_missing_market_prices_or_invalid_ticker` | Verifies graceful HTTP 400 rejection when ordering invalid or delisted symbols. | **PASSED** |
| 13 | `test_portfolio_analytics_endpoint` | Verifies equity curve time series and asset allocation breakdown payloads. | **PASSED** |

---

## 6. Browser Integration QA & Visual Artifacts

Browser testing was performed using automated Puppeteer on Microsoft Edge (`headless: 'new'`) against the Next.js production build (`http://localhost:3000`) and live FastAPI daemon (`http://127.0.0.1:8000`).

Audit Results:
- **Total Console Errors:** `0`
- **Total Uncaught Page Errors:** `0`
- **Workflows Tested:** `PASS`

### Captured Screenshots

1. **`reports/screenshots/11_portfolio_dashboard_empty.png`**
   *Description:* Pristine empty state showing ₹1,00,000.00 cash balance, ₹0.00 P&L, 0 holdings, model-generated opportunity cards, and the "Start First Simulated Trade" call-to-action banner.

2. **`reports/screenshots/12_portfolio_buy_modal.png`**
   *Description:* The interactive order execution modal open on `/portfolio`. Demonstrates real-time price lookup (TCS @ ₹2,342.00 as-of 2026-08-28), share stepper, friction calculation (+₹17.57), total cost (₹23,437.57), and the 2-step confirmation workflow.

3. **`reports/screenshots/13_portfolio_with_holdings.png`**
   *Description:* Portfolio dashboard immediately following order execution. Shows portfolio valuation updated to ₹99,982.43, available cash ₹76,562.43, total P&L -₹17.57 (reflecting statutory friction), TCS active holding row (10 shares @ ₹2,342.00), and Buy More / Sell buttons.

4. **`reports/screenshots/14_portfolio_analytics.png`**
   *Description:* Focused view of the analytics section. Shows the Recharts Portfolio Equity Progression area curve with tooltip hover, the Asset Allocation Donut chart (76.6% Cash, 23.4% TCS), and the complete Auditable Transaction Ledger.

5. **`reports/screenshots/15_signals_with_trade_modal.png`**
   *Description:* The Actionable Signals page (`/signals`) for INFY with active BUY signal. Shows the "Simulate Trade with Signal →" button and the pre-populated order modal (INFY, BUY, 10 shares @ ₹1,144.00, statutory friction +₹8.58).

6. **`reports/screenshots/16_portfolio_mobile.png`**
   *Description:* Mobile viewport (375x667 iPhone SE). Validates responsive card stacking, visible hamburger navigation button, stock selector, and unobstructed KPI metrics.

---

## 7. Commands to Run the Application

### 1. Start the FastAPI Backend
```bash
# From repository root
.venv\Scripts\python.exe -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8000
```
- API Base URL: `http://127.0.0.1:8000/api/v1`
- Interactive Swagger Documentation: `http://127.0.0.1:8000/docs`

### 2. Start the Next.js Frontend
```bash
# In frontend directory
cd frontend
npm run build
npm run start
```
- Web Application: `http://localhost:3000`
- Virtual Portfolio: `http://localhost:3000/portfolio`
- Actionable Signals: `http://localhost:3000/signals`

### 3. Run the Automated Test Suite
```bash
# Run all 25 backend tests
.venv\Scripts\python.exe -m pytest backend/tests -v
```

### 4. (Optional) Run the Streamlit Research Terminal
```bash
# Legacy research terminal remains fully operational
.venv\Scripts\streamlit run app.py --server.port 8501
```

---

## 8. Conclusion

Phase 5 completes the transformation of the Stock Market Predictor platform from an exploratory research prototype into an institutional-grade financial analytics and paper trading workstation. The implementation adheres strictly to the project's financial research rigor, guarantees numerical consistency, prevents unauthorized financial execution, and delivers an intuitive, responsive user experience.
