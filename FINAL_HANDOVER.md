# FINAL HANDOVER: STOCK MARKET PREDICTOR

**Platform:** Stock Market Predictor — Institutional Financial Analytics Terminal
**Document:** Project Handover, Architecture Specification & Operational Runbook
**Version:** 1.0.0 (Production-Ready Release)
**Date:** November 2026

---

## 1. Project Overview

The **Stock Market Predictor** is an institutional-grade financial analytics and decision-support workstation designed for Indian equities traded on the National Stock Exchange (NSE). The platform evolved from an exploratory machine learning and Streamlit research codebase into a decoupled, high-performance web platform powered by a **FastAPI (Python)** backend and a **Next.js 14 (React 19 / TypeScript / Tailwind CSS)** frontend.

The system is engineered around econometric rigor:
- **Walk-forward cross-validation** with a **52-day training embargo** to eliminate lookahead bias.
- **Isotonic probability calibration** to convert raw model scores into true directional convictions.
- **Company-targeted FinBERT sentiment analysis** on corporate announcements and financial news.
- **Indian transaction-cost awareness**, enforcing a **15 bps round-trip friction hurdle** before generating actionable trading signals.
- **Risk-free paper trading environment** with ₹1,00,000 initial virtual capital and **7.5 bps statutory one-way delivery friction**.

---

## 2. Core Features Breakdown

### 2.1 Completed & Fully Tested Features
1. **Prediction Terminal (`/prediction`):** Directional 3-day return prediction ($Close_{t+3} > Close_t$), calibrated conviction probability gauge, out-of-sample accuracy vs. majority-class baseline, confidence gating flags, and model feature importances.
2. **Next-Morning Forecast (`/forecast`):** Next trading session opening price point estimate $\hat{Open}_{t+1}$, 90% confidence bands $[\hat{Open}_{low}, \hat{Open}_{high}]$, historical directional accuracy, and residual variance tracking. Strictly isolated from the 3-day directional model.
3. **Why Did It Move? (`/explain`):** Entity-level FinBERT sentiment polarity extraction ($S \in [-1.0, +1.0]$), interactive stock price and news event correlation timeline, corporate announcement drawer, and sentiment distribution breakdown.
4. **Actionable Trading Signals (`/signals`):** Evidence-based decision engine operationalizing five systematic strategy variants (A: Baseline Long-Only, B: Risk-Managed Long-Only, C: Dynamic Long/Short, D: Multi-Factor Consensus, E: Regime-Adaptive). Backtesting equity curves, Sharpe/Sortino ratios, drawdown analysis, and historical trade ledger.
5. **Virtual Portfolio & Paper Trading (`/portfolio`):** Persistent virtual wallet initialized with ₹1,00,000 virtual capital. Available cash tracking, real-time unrealised/realised P&L accounting, 10-stock active holdings table, interactive 2-step Buy/Sell order execution modal, Recharts equity progression curve, asset allocation donut chart, and complete transaction ledger.
6. **Actionable Signals to Paper Trading Link:** "Simulate Trade with Signal →" button on `/signals` and `/portfolio` allowing one-click execution from active BUY or SELL signals. Trade execution is strictly disabled for `HOLD` and `NO SIGNAL`.
7. **References & Methodology Compendium (`/references` & `/methodology`):** 10 organized academic citation categories with titles, authors, DOIs, and takeaways. Mathematical formulations for all models, research vs. backtest comparison table, and data coverage disclosures.
8. **Settings & System Diagnostics (`/settings`):** FastAPI connection health, response latency, registered stock models, database engine, 10-stock coverage matrix, and explicit atomic wallet reset modal requiring typing "RESET".
9. **Responsive Shell Navigation:** Desktop persistent deep navy sidebar (`#0B132B`) with responsive slide-out drawer on mobile viewports (tested at 375x667 iPhone SE).
10. **Automated Test Suite:** 25 passing pytest unit/integration tests and automated Puppeteer browser integration suite with zero console errors.

### 2.2 Partially Implemented / Fallback Features
- **Legacy Streamlit Terminal (`app.py`):** Fully operational on port 8501. Serves as a standalone fallback tool for researchers who prefer the original monolithic Streamlit interface.

### 2.3 Known Limitations
1. **Historical News Density:** Curated company-targeted news archives are continuous from 2020 through 2026. Prior to 2020, financial headline density is discrete rather than continuous.
2. **Live News Tier:** The optional NewsAPI developer tier restricts live searches to a rolling 30-day window and 100 requests per day. The platform gracefully falls back to local curated news archives when no API key is supplied.
3. **Market Hours Live Feed:** The primary research dataset represents an audited historical snapshot through August 28, 2026. The terminal discloses this status prominently in the top header (`Research Snapshot: 2026-08-28`).

### 2.4 Future Scope & Roadmap
1. **WebSockets for Real-Time NSE Feeds:** Integrate streaming tick-by-tick WebSocket data from registered Indian broker APIs (e.g. Zerodha Kite Connect, Angel One SmartAPI, Upstox).
2. **Multi-User Authentication:** Upgrade from single-tenant local SQLite wallets to multi-tenant OAuth2 / JWT authentication with user accounts and cloud PostgreSQL storage.
3. **Options & Derivatives Module:** Expand from cash equity delivery into NIFTY/BANKNIFTY index options and weekly expiry payoff simulations.

---

## 3. Technology Stack & Architecture

```
+----------------------------------------------------------------------------------------------------+
|                                      SYSTEM ARCHITECTURE                                           |
+----------------------------------------------------------------------------------------------------+
|                                                                                                    |
|    FRONTEND (Next.js 14 App Router, React 19, TypeScript, Tailwind CSS, shadcn/ui, Recharts)        |
|    Port: 3000 | Client Base URL: http://localhost:3000                                             |
|                                                                                                    |
|          |  HTTP / REST (JSON API)                                                                 |
|          v                                                                                         |
|                                                                                                    |
|    BACKEND (FastAPI, Uvicorn, Pydantic v2, SQLAlchemy ORM)                                          |
|    Port: 8000 | API Prefix: /api/v1 | Docs: /docs                                                  |
|                                                                                                    |
|          +----------------------------+----------------------------+                               |
|          |                            |                            |                               |
|          v                            v                            v                               |
|   [ Research Engine ]         [ Database Layer ]           [ Sentiment Study ]                     |
|   - src/pipeline.py           - data/app.db (SQLite)       - sentiment_study/src/                  |
|   - src/signals.py            - Wallets (₹100k Capital)    - Entity FinBERT Tagging                |
|   - src/backtest.py           - Holdings (Marked to Mkt)   - News Archives                         |
|   - src/dataset.py            - Transactions (Audit Log)                                           |
|   - 10 Trained Models                                                                              |
+----------------------------------------------------------------------------------------------------+
```

### Stack Details
- **Frontend:** Next.js 14.2.35, React 19, TypeScript 5, Tailwind CSS 3.4, Lucide React icons, TanStack React Query v5, Recharts 2.15.
- **Backend:** Python 3.11/3.13, FastAPI, Uvicorn, Pydantic v2, SQLAlchemy 2.0, AnyIO.
- **Machine Learning & Analytics:** Scikit-learn, LightGBM, Pandas, NumPy, SciPy, Ta (Technical Analysis library).
- **NLP & Sentiment:** PyTorch, HuggingFace Transformers, FinBERT, SEntFiN 1.0 entity disambiguation.
- **Database:** SQLite 3 (default `data/app.db`) with full PostgreSQL connection string support via `DATABASE_URL`.
- **Testing:** Pytest, HTTPX, Puppeteer-core with Microsoft Edge headless browser automation.

---

## 4. Repository Structure

```
Stock-Market-Predictor/
├── backend/                        # FastAPI Service Layer
│   ├── app/
│   │   ├── api/v1/                 # Versioned REST endpoints
│   │   │   ├── endpoints/          # stocks, portfolio, etc.
│   │   │   └── api.py              # Router aggregation
│   │   ├── core/                   # config.py, errors.py
│   │   ├── database/               # base.py, session.py
│   │   ├── models/                 # wallet.py (SQLAlchemy ORM)
│   │   ├── schemas/                # Pydantic v2 schemas
│   │   ├── services/               # Adapters wrapping src/ modules
│   │   └── main.py                 # FastAPI application entrypoint
│   ├── tests/                      # Pytest backend test suite
│   │   ├── test_api.py             # Phase 1-4 endpoint tests
│   │   └── test_portfolio.py       # Phase 5 portfolio tests (13 tests)
│   └── Dockerfile                  # Production container definition
├── frontend/                       # Next.js 14 Frontend Application
│   ├── public/                     # Static icons & brand assets
│   ├── scripts/                    # Puppeteer browser QA scripts
│   ├── src/
│   │   ├── app/                    # Next.js App Router pages
│   │   │   ├── explain/page.tsx    # Why Did It Move?
│   │   │   ├── forecast/page.tsx   # Next-Morning Forecast
│   │   │   ├── methodology/page.tsx# Methodology alias
│   │   │   ├── portfolio/page.tsx  # Virtual Portfolio & Execution
│   │   │   ├── prediction/page.tsx # Prediction Terminal
│   │   │   ├── references/page.tsx # 10-category References
│   │   │   ├── settings/page.tsx   # Diagnostics & Demonstration
│   │   │   ├── signals/page.tsx    # Actionable Trading Signals
│   │   │   ├── layout.tsx          # Root layout with Shell
│   │   │   └── page.tsx            # Overview Dashboard
│   │   ├── components/             # UI, layout, brand, and chart components
│   │   ├── context/                # StockContext (global selected stock)
│   │   ├── hooks/                  # TanStack Query data fetching hooks
│   │   ├── lib/                    # api.ts (Axios/fetch client), utils.ts
│   │   └── types/                  # TypeScript interface definitions
│   ├── Dockerfile                  # Next.js multi-stage container build
│   └── package.json                # Dependencies and build scripts
├── data/                           # Data storage
│   ├── raw/                        # 10 raw OHLCV CSV files (TCS, INFY, RELIANCE, etc.)
│   ├── processed/                  # Processed feature matrices & sentiment CSVs
│   └── app.db                      # Local SQLite database for virtual wallets
├── models/                         # Pre-trained scikit-learn models per equity
│   ├── TCS/                        # model.joblib, metadata.json
│   ├── RELIANCE/
│   ├── INFY/
│   └── ... (10 equities)
├── reports/                        # Deliverables, QA audits, and screenshots
│   ├── screenshots/                # 13 verified browser screenshots (11_ to 23_)
│   ├── final_qa_report.md          # Comprehensive QA report
│   ├── security_audit_final.md     # Security audit report
│   └── deployment_guide.md         # Deployment & operations guide
├── sentiment_study/                # FinBERT and SEntFiN research codebase
├── src/                            # Core research prediction engine
│   ├── backtest.py                 # Out-of-sample strategy backtester
│   ├── config.py                   # Central stock registry & path resolver
│   ├── dataset.py                  # OHLCV loading and feature engineering
│   ├── pipeline.py                 # Training & prediction pipeline
│   └── signals.py                  # 5-variant trading decision engine
├── app.py                          # Streamlit legacy fallback application
├── docker-compose.yml              # Multi-container orchestration
├── requirements.txt                # Python backend dependencies
└── README.md                       # Main repository readme
```

---

## 5. Step-by-Step Installation & Runbook (Windows PowerShell)

### Step 1: Environment Setup
Ensure you have cloned the repository and are in the root directory:
```powershell
cd C:\Users\vedan\Stock-Market-Predictor
```

Copy the environment template:
```powershell
Copy-Item .env.example .env
```

### Step 2: Backend Setup & Execution
Activate the Python virtual environment and launch FastAPI:
```powershell
# Activate virtual environment
.venv\Scripts\Activate.ps1

# Install requirements (if not already installed)
pip install -r requirements.txt

# Start FastAPI on port 8000
uvicorn backend.app.main:app --host 127.0.0.1 --port 8000
```
- API Base URL: `http://127.0.0.1:8000/api/v1`
- Swagger Interactive Documentation: `http://127.0.0.1:8000/docs`

### Step 3: Frontend Setup & Execution
In a second PowerShell terminal:
```powershell
cd C:\Users\vedan\Stock-Market-Predictor\frontend

# Install dependencies (if needed)
npm ci

# Build optimized production bundle
npm run build

# Start production server on port 3000
npm run start
```
- Web Application: `http://localhost:3000`

### Step 4: Run Automated Tests
In a third PowerShell terminal:
```powershell
cd C:\Users\vedan\Stock-Market-Predictor

# Run all 25 backend unit and integration tests
.venv\Scripts\python.exe -m pytest backend/tests -v

# Run browser automated audit
node frontend/scripts/run_phase6_final_audit.js
```

### Step 5: (Optional) Run Streamlit Fallback
```powershell
cd C:\Users\vedan\Stock-Market-Predictor
.venv\Scripts\streamlit run app.py --server.port 8501
```

---

## 6. Ten Supported Equities

| Symbol | Yahoo Ticker | Display Name | Sector |
|---|---|---|---|
| `TCS` | `TCS.NS` | Tata Consultancy Services | Information Technology |
| `RELIANCE` | `RELIANCE.NS` | Reliance Industries | Energy & Conglomerate |
| `INFY` | `INFY.NS` | Infosys Limited | Information Technology |
| `HDFCBANK` | `HDFCBANK.NS` | HDFC Bank | Banking & Financials |
| `SBI` | `SBIN.NS` | State Bank of India | Public Sector Banking |
| `ITC` | `ITC.NS` | ITC Limited | Fast-Moving Consumer Goods |
| `ADANIENT` | `ADANIENT.NS` | Adani Enterprises | Infrastructure & Trading |
| `VEDL` | `VEDL.NS` | Vedanta Limited | Natural Resources & Mining |
| `ETERNAL` | `ETERNAL.NS` | Eternal Limited (Zomato) | Consumer Internet Technology |
| `ADANIGREEN` | `ADANIGREEN.NS` | Adani Green Energy | Renewable Power Generation |

---

## 7. Virtual Wallet & Demonstration Procedure

To conduct a live presentation, academic demonstration, or testing review:

1. **Access Settings:** Navigate to [`http://localhost:3000/settings`](http://localhost:3000/settings).
2. **Review Diagnostics:** Confirm that the FastAPI connection health is green (`ONLINE`) and all 10 stock models are registered.
3. **Trigger Demonstration Reset:** Click **Reset Demonstration Wallet**.
4. **Explicit Confirmation:** In the modal, type `RESET` in capital letters and click **Confirm Atomic Reset**.
5. **Verify Pristine State:** Navigate to [`http://localhost:3000/portfolio`](http://localhost:3000/portfolio). Observe that:
   - Portfolio Valuation is exactly ₹1,00,000.00.
   - Available Cash is ₹1,00,000.00.
   - Active Holdings table displays the empty state guidance banner.
   - Transaction ledger contains 0 rows.
6. **Execute Simulated Orders:**
   - Click **Place Order** or click **Simulate Trade with Signal →** on one of the active `BUY` recommendations (e.g. INFY).
   - Review the 7.5 bps statutory fee calculation.
   - Click **Review Order Confirmation →**, then **Confirm Order**.
   - Notice immediate atomic updating of cash, holdings, equity progression curve, and donut chart.
7. **Test Disciplined Risk Gating:**
   - Navigate to [`http://localhost:3000/signals`](http://localhost:3000/signals).
   - Switch to TCS where the model recommends `HOLD` (edge below 15 bps).
   - Verify that the trade button is deactivated, explaining that the model abstains to protect capital from churn.

---

## 8. Data Refresh & Model Maintenance

To update market data or retrain models:
1. **Fetch Latest OHLCV Data:** Execute `src/collect.py` to pull updated daily price bars from Yahoo Finance.
2. **Feature Generation:** Run `src/dataset.py` to calculate technical features (RSI, MACD, Bollinger Bands, ATR, volume trends).
3. **Model Retraining:** Execute `src/pipeline.py` to retrain scikit-learn models using purged walk-forward cross-validation with a 52-day embargo.
4. **Sentiment Refresh:** Execute `sentiment_study/src/export_company_sentiment.py` to process new headlines through FinBERT and export updated entity sentiment matrices.

---

## 9. Conclusion & Certification

The Stock Market Predictor terminal stands certified for production deployment. It combines statistical and econometric integrity with an institutional design system, verified security postures, and comprehensive operational documentation.
