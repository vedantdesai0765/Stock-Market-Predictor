# Phase 5 Implementation Plan: Virtual Portfolio & Paper Trading

**Project:** Stock Market Predictor
**Module:** Phase 5 — Virtual Portfolio, Simulated Execution & Analytics
**Date:** November 4, 2026
**Status:** Planned

---

## 1. Architectural Overview & Existing Infrastructure Reuse

The goal of Phase 5 is to build a complete institutional paper trading platform connected to our existing Python prediction and signal engines without duplicate models or logic.

### Components to Reuse
1. **SQLAlchemy 2.0 Infrastructure (`backend/app/database/`):**
   - [`backend/app/database/base.py`](file:///C:/Users/vedan/Stock-Market-Predictor/backend/app/database/base.py): Shared `DeclarativeBase`.
   - [`backend/app/database/session.py`](file:///C:/Users/vedan/Stock-Market-Predictor/backend/app/database/session.py): `engine`, `SessionLocal`, `get_db`, SQLite/PostgreSQL configuration.
2. **Existing ORM Models (`backend/app/models/wallet.py`):**
   - `User`, `Wallet`, `Position`, `Transaction`, `Watchlist`.
   - Update default initial virtual capital from ₹10,00,000 to **₹1,00,000** (1 Lakh) to strictly align with user specifications.
3. **Existing Market Pricing:**
   - Retrieve latest prices from `data/raw/<TICKER>_raw.csv` and `StockService` so simulated trades execute strictly on genuine historical closes with verified timestamps.
4. **Existing Actionable Signals Engine (`src/signals.py`):**
   - Connect live recommendations (`BUY`/`SELL`) from the 10 supported stocks directly to the portfolio order interface with explicit user confirmation.

---

## 2. Backend Design

### A. Schemas (`backend/app/schemas/wallet.py`)
- `WalletSummaryResponse`:
  - `initial_capital`: float (₹1,00,000.00)
  - `available_cash`: float
  - `invested_capital`: float
  - `portfolio_value`: float
  - `total_pnl`: float
  - `realised_pnl`: float
  - `unrealised_pnl`: float
  - `total_return_pct`: float
- `HoldingItem`:
  - `ticker`: str
  - `display_name`: str
  - `quantity`: int
  - `avg_buy_price`: float
  - `current_price`: float
  - `price_date`: str
  - `invested_amount`: float
  - `current_value`: float
  - `unrealised_pnl`: float
  - `return_pct`: float
- `OrderPreviewRequest` & `OrderPreviewResponse`:
  - Computes gross amount, statutory friction (7.5 bps one-way), net amount, cash validity, and max executable shares.
- `ExecuteOrderRequest` & `ExecuteOrderResponse`:
  - Ticker, side (`BUY`/`SELL`), quantity, signal source, idempotency key.
- `PortfolioAnalyticsResponse`:
  - Equity history points, asset allocation breakdown (Cash vs. Stocks), transaction ledger.

### B. Service Layer (`backend/app/services/wallet_service.py`)
- `get_or_create_default_wallet(db: Session)`: Guarantees a persistent primary wallet with ₹1,00,000 starting cash.
- `get_portfolio_summary(db: Session)`: Re-values open positions against the latest market prices, calculates realised/unrealised P&L and return %.
- `preview_order(request: OrderPreviewRequest, db: Session)`: Pre-execution validation and fee calculation.
- `execute_order(request: ExecuteOrderRequest, db: Session)`: Atomic execution, balance/holding validation, transaction ledger recording, and position rebalancing.
- `reset_portfolio(db: Session)`: Clears positions/transactions and resets cash to ₹1,00,000 for reproducible test sessions.
- `get_analytics(db: Session)`: Equity progression, asset allocation, and transaction logs.

### C. API Endpoints (`backend/app/api/v1/endpoints/portfolio.py`)
- `GET /api/v1/portfolio`: Full portfolio summary, metrics, and holdings list.
- `POST /api/v1/portfolio/preview`: Pre-trade cost and balance preview.
- `POST /api/v1/portfolio/orders`: Place BUY or SELL paper trading order.
- `GET /api/v1/portfolio/transactions`: Paginated historical trade ledger.
- `GET /api/v1/portfolio/analytics`: Chart data (equity curve, allocation, P&L breakdown).
- `POST /api/v1/portfolio/reset`: Reset portfolio to default state.

---

## 3. Frontend Design

### A. Virtual Portfolio Dashboard (`frontend/src/app/portfolio/page.tsx`)
1. **Header & Simulation Notice:**
   - Prominent amber banner: `SIMULATED TRADING · VIRTUAL CAPITAL ₹1,00,000 (NO REAL MONEY)`.
   - Reset wallet button with confirmation dialog.
2. **KPI Metric Cards:**
   - Total Portfolio Value, Available Cash, Invested Capital, Total P&L, Realised P&L, Unrealised P&L, Overall Return %.
   - Plain-English tooltips on every term (`InfoTooltip`).
3. **Actionable Signal Quick-Trade Panel:**
   - Highlights current `BUY` or `SELL` opportunities across the 10 stocks from the signals engine.
   - One-click trigger to open pre-filled order modal.
4. **Holdings Dashboard Table:**
   - Real-time valuations, average buy price, current price, invested amount, current value, unrealised P&L, return %.
   - Action buttons: "Buy More", "Sell".
5. **Interactive Analytics Charts (`frontend/src/components/charts/PortfolioCharts.tsx`):**
   - Portfolio Equity Curve (cumulative portfolio value).
   - Asset Allocation Donut Chart (Cash vs. Stock weights).
   - Realised vs. Unrealised P&L comparison.
6. **Auditable Transaction History Table:**
   - Date, Ticker, Side badge (green BUY, red SELL), Quantity, Price, Friction paid, Net cash flow, Signal source.
7. **Order Placement Modal:**
   - Interactive BUY/SELL form.
   - Dynamic price and statutory friction calculator.
   - Cash/holding validation feedback.
   - Two-step confirmation screen.

### B. Actionable Signals Page Integration (`frontend/src/app/signals/page.tsx`)
- When signal is `BUY` or `SELL`, add an **"Execute Simulated Trade"** button that opens the order modal pre-filled with the ticker, direction, and price.
- Neutral states (`HOLD`, `NO SIGNAL`) disable order creation with a clear explanation.

---

## 4. Verification & Testing Strategy
1. **Backend Tests (`backend/tests/test_portfolio.py`):**
   - 13 comprehensive unit & integration tests covering wallet creation, buy, sell, insufficient cash/shares, transaction costs, duplicate prevention, and valuations.
2. **Automated Browser QA:**
   - Test order placement, modal interaction, holding updates, and responsive rendering with Puppeteer.
   - Capture screenshots for `reports/screenshots/`.
3. **Build & Regression Suite:**
   - `pytest backend/tests -v`
   - `npm run build`
