# Phase 4 Deliverable Report: Financial News Explainability & Actionable Trading Signals

**Project:** Stock Market Predictor
**Phase:** Phase 4 — "Why Did It Move?" (Sentiment & News Explainability) and "Actionable Signals" (Quantitative Trading Strategy & Walk-Forward Backtesting)
**Date:** November 4, 2026
**Status:** Complete & Verified (`npm run build` passed with 0 errors across 12 static routes; `pytest backend/tests` passed with 12/12 tests)

---

## 1. Executive Summary

Phase 4 of the Stock Market Predictor migration has been completed. This phase delivers the two remaining major research and analytical engines of the platform:

1. **Why Did It Move? (`/explain`):**
   - Financial news and entity-targeted FinBERT explainability interface.
   - Merged price-and-news interactive timeline displaying historical closing prices overlaid with positive, negative, and neutral sentiment event markers.
   - Comprehensive news explorer with sentiment pill filters (All, Positive, Negative, Neutral), live text search, and expandable article cards showing entity relevance and confidence scores.
   - Interactive lead-lag analysis chart evaluating information diffusion across $\pm 5$ trading days using Spearman rank correlation.
   - Plain-English explanations of entity-targeted transformer architectures, headline noise reduction, and academic research takeaways.

2. **Actionable Trading Signals & Backtesting (`/signals`):**
   - Live quantitative trading recommendation interface featuring strict, accessible signal badges (**BUY** in emerald green, **SELL** in rose red, **HOLD** in amber with a pause icon, and **NO SIGNAL** in neutral slate).
   - Interactive selector across all **5 Strategy Variants (A through E)** from the research codebase:
     - **Variant A:** Base Model (Walk-Forward Directional LightGBM).
     - **Variant B:** Calibrated Probability + Friction Gate (Isotonic Calibration + 15 bps Indian statutory hurdle).
     - **Variant C:** Trend & Volatility Regime Filter (SMA20/SMA50 + Expanding Volatility Quantile).
     - **Variant D:** Multi-Factor Confirmation (Model + RSI/MACD + FinBERT Sentiment Alignment).
     - **Variant E:** Full Production Ensemble (Combined Calibrated + Regime + Multi-Factor).
   - Multi-factor confirmation matrix inspecting five independent decision drivers.
   - Dynamic transaction cost friction selector ($0\text{ bps}$, realistic Indian delivery $15\text{ bps}$, and conservative $30\text{ bps}$).
   - Institutional performance metrics (CAGR, Sharpe, Sortino, Max Drawdown, Profit Factor, Win Rate, and Total Trades).
   - Interactive Recharts Equity Curve (Strategy vs. Buy & Hold vs. Cash benchmark indexed to ₹1,00,000) and peak-to-trough Underwater Drawdown area chart.
   - Searchable historical trades log table detailing entry/exit dates, prices, position direction, returns, and net P&L.
   - Quantitative risk and methodology accordion documenting the 17-day embargo, out-of-sample walk-forward protocol, and multiple testing caveats.

---

## 2. Architecture & File Inventory

### Backend Components

| File | Purpose |
|---|---|
| [`backend/app/schemas/news.py`](file:///C:/Users/vedan/Stock-Market-Predictor/backend/app/schemas/news.py) | Extended with `SentimentTimelinePoint` model and added `timeline: List[SentimentTimelinePoint]` to `NewsResponse`. |
| [`backend/app/services/news_service.py`](file:///C:/Users/vedan/Stock-Market-Predictor/backend/app/services/news_service.py) | Merges daily historical OHLCV closing prices with entity-targeted FinBERT sentiment items within the overlapping news date window to build a continuous event timeline. |
| [`backend/app/schemas/backtest.py`](file:///C:/Users/vedan/Stock-Market-Predictor/backend/app/schemas/backtest.py) | Backtest response schemas including equity curves, drawdown series, strategy metrics, and trade records. |
| [`backend/app/services/backtest_service.py`](file:///C:/Users/vedan/Stock-Market-Predictor/backend/app/services/backtest_service.py) | Service adapter linking `src/backtest.py` and `src/signals.py` for all 10 NSE stocks and all 5 strategy variants. |

### Frontend Components

| File | Purpose |
|---|---|
| [`frontend/src/types/api.ts`](file:///C:/Users/vedan/Stock-Market-Predictor/frontend/src/types/api.ts) | Extended TypeScript definitions for `SentimentTimelinePoint`, `NewsResponse`, `BacktestResponse`, and `StrategyVariant`. |
| [`frontend/src/components/charts/ExplainCharts.tsx`](file:///C:/Users/vedan/Stock-Market-Predictor/frontend/src/components/charts/ExplainCharts.tsx) | Recharts `SentimentPriceTimelineChart` (closing price area chart overlaid with colored sentiment scatter dots and date click selection) and `LeadLagChart` (Spearman rank correlation bar chart). |
| [`frontend/src/components/charts/BacktestCharts.tsx`](file:///C:/Users/vedan/Stock-Market-Predictor/frontend/src/components/charts/BacktestCharts.tsx) | Recharts `BacktestEquityChart` (indexed growth of Strategy, Buy & Hold, and Cash from ₹1,00,000) and `UnderwaterDrawdownChart` (peak-to-trough percentage drawdown). |
| [`frontend/src/app/explain/page.tsx`](file:///C:/Users/vedan/Stock-Market-Predictor/frontend/src/app/explain/page.tsx) | Complete "Why Did It Move?" page incorporating all 6 analytical sections. |
| [`frontend/src/app/signals/page.tsx`](file:///C:/Users/vedan/Stock-Market-Predictor/frontend/src/app/signals/page.tsx) | Complete "Actionable Signals & Backtesting" page incorporating all 8 quantitative sections. |

---

## 3. Section-by-Section Breakdown

### Part A: "Why Did It Move?" (`/explain`)

1. **Stock Overview Header:**
   - Selected ticker and full corporate name (e.g., Reliance Industries, Tata Consultancy Services, HDFC Bank).
   - Current price, daily price change percentage, and latest market date.
   - Data hygiene metadata: news coverage date span, sentiment model freshness, and total analyzed headline count.

2. **Price & News Interactive Timeline:**
   - Visualizes closing price movements over time with sentiment overlay.
   - Positive sentiment events marked in emerald green dots, negative sentiment in rose red, neutral in slate grey.
   - Clicking any date or point automatically filters the headline explorer below to inspect the news published on that trading session.

3. **Filterable News Explorer:**
   - Quick-toggle sentiment pills: **All**, **Positive**, **Negative**, **Neutral**, with counts.
   - Real-time text search across headline titles and publisher sources.
   - Expandable headline cards with FinBERT entity target attribution, calibrated sentiment confidence scores (e.g., 94.2% Positive), publisher name, and time of release.

4. **FinBERT Sentiment Analysis Explanations:**
   - Dedicated explanatory panels detailing entity-targeted financial NLP.
   - Explains how generic sentiment models fail on market text (e.g., *"HDFC Bank beats estimates while rivals tumble"* targets HDFC Bank as positive, rather than averaging rival negativity).
   - High-confidence threshold filtering ($p \ge 0.70$) to discard ambiguous financial market commentary.

5. **Lead-Lag Information Diffusion Analysis:**
   - Interactive bar chart displaying Spearman rank correlation between sentiment and stock price returns across $\tau \in [-5, +5]$ trading days.
   - Visualizes contemporaneous impact ($\tau = 0$) versus predictive lead ($\tau < 0$) and post-announcement drift ($\tau > 0$).

6. **Empirical Research Findings:**
   - Documented key takeaways: sentiment edge decay occurs within 24–48 hours; high news volume amplifies price volatility; macro news requires industry-level entity disambiguation.

---

### Part B: "Actionable Signals & Backtesting" (`/signals`)

1. **Current Actionable Signal Banner:**
   - Prominently displays the current recommendation: **BUY** (Emerald), **SELL** (Rose), **HOLD** (Amber with Pause icon), or **NO SIGNAL** (Slate Grey with neutral icon).
   - Direct plain-English rationale explaining why the signal was emitted.
   - Explicit distinction between **HOLD** (frictions exceed expected edge) and **NO SIGNAL** (abstention due to low model conviction).
   - Statutory friction hurdle reminder ($15\text{ bps}$ delivery costs).

2. **Strategy Variant Selector (Variants A–E):**
   - Fast, interactive switching across all five validated research strategies:
     - **Variant A (Base Model):** Pure walk-forward directional LightGBM model.
     - **Variant B (Calibrated + Friction):** Isotonically calibrated probabilities with an explicit 15 bps return hurdle.
     - **Variant C (Regime Filter):** Restricts trades to favorable trend (SMA20 > SMA50) and volatility regimes.
     - **Variant D (Multi-Factor):** Requires model conviction to align with technical indicators (RSI/MACD) and FinBERT sentiment.
     - **Variant E (Full Ensemble):** Production combination of calibration, regime gating, and multi-factor confirmation.

3. **Multi-Factor Confirmation Matrix:**
   - Visual breakdown of the 5 independent decision components:
     1. *ML Directional Probability* (Calibrated $P_{\text{up}}$ vs. threshold).
     2. *Trend Filter* (Price vs. 20-day / 50-day Simple Moving Average).
     3. *Momentum Filter* (14-day RSI and MACD signal crossover).
     4. *FinBERT Sentiment Score* (3-day rolling sentiment polarity).
     5. *Transaction Cost Hurdle* (Estimated gross return vs. 15 bps delivery friction).

4. **Execution Settings & Friction Selector:**
   - Interactive toggle to stress-test performance under varying friction assumptions:
     - **0 bps:** Theoretical zero-cost benchmark (idealized).
     - **15 bps:** Realistic Indian delivery friction (STT 0.1%, Brokerage, Exchange Turnover, Stamp Duty, GST, SEBI fee, slippage).
     - **30 bps:** Conservative stress test for high slippage or smaller capital accounts.

5. **Performance Metrics Dashboard:**
   - Six core institutional quantitative metrics:
     - **Strategy CAGR vs. Buy & Hold CAGR:** Compound annual growth rate comparison.
     - **Sharpe Ratio:** Annualized risk-adjusted excess return over risk-free rate ($6.5\%$).
     - **Sortino Ratio:** Downside-risk-adjusted return penalizing only negative volatility.
     - **Max Drawdown:** Worst peak-to-trough portfolio decline.
     - **Profit Factor:** Gross profits divided by gross losses.
     - **Win Rate & Total Trades:** Percentage of closed positions with positive net return and trade sample size.

6. **Interactive Equity & Drawdown Charts:**
   - **Backtest Equity Chart:** Tracks portfolio value starting at ₹1,00,000 across the backtest horizon for the Strategy, Buy & Hold, and Cash benchmark.
   - **Underwater Drawdown Chart:** Visualizes portfolio drawdowns below previous high-water marks over time.

7. **Historical Trades Log Table:**
   - Detailed audit log of every trade executed by the backtest engine:
     - Entry Date and Exit Date (3-day non-overlapping holding period).
     - Position Type (`LONG` / `SHORT`).
     - Entry Price (₹) and Exit Price (₹).
     - Net Return (%) and Net P&L (₹) after all statutory transaction costs.
     - Trigger rationale and exit condition.
   - Searchable by date or position type, with pagination.

8. **Quantitative Risk & Methodology Accordion:**
   - Progressive disclosure documenting:
     - **Walk-Forward Validation:** 5-fold expanding window with zero future data leakage.
     - **17-Day Embargo:** Mandatory buffer quarantine between training and testing windows to eliminate autocorrelation artifacts from 3-day holding targets.
     - **Non-Overlapping Holding Periods:** Prevents statistical pseudo-replication in trade log evaluations.
     - **Indian Transaction Costs:** Complete statutory breakdown of the 15 bps friction model.
     - **Multiple Testing Caveat:** Transparent disclosure of selection bias across 10 stocks and 5 variants.

---

## 4. Verification & Testing

### Backend Test Suite
Executed `pytest backend/tests -v` using the project Python virtual environment:
```text
backend/tests/test_api.py::test_health_endpoint PASSED                           [  8%]
backend/tests/test_api.py::test_stocks_list_endpoint PASSED                      [ 16%]
backend/tests/test_api.py::test_stock_prediction_endpoint PASSED                 [ 25%]
backend/tests/test_api.py::test_stock_prediction_invalid_ticker PASSED           [ 33%]
backend/tests/test_api.py::test_stock_forecast_endpoint PASSED                   [ 41%]
backend/tests/test_api.py::test_stock_news_endpoint PASSED                       [ 50%]
backend/tests/test_api.py::test_stock_backtest_endpoint PASSED                   [ 58%]
backend/tests/test_api.py::test_portfolio_wallet_endpoints PASSED                [ 66%]
backend/tests/test_api.py::test_portfolio_trade_lifecycle PASSED                 [ 75%]
backend/tests/test_api.py::test_portfolio_insufficient_balance PASSED            [ 83%]
backend/tests/test_api.py::test_portfolio_sell_insufficient_shares PASSED        [ 91%]
backend/tests/test_api.py::test_portfolio_performance_endpoint PASSED           [100%]

============================== 12 passed in 1.48s ==============================
```

### Frontend Production Build
Executed `npm run build` in `frontend/`:
```text
   ▲ Next.js 15.2.4
   - Environments: .env

   ✓ Compiled successfully
   Linting and checking validity of types ...
   Collecting page data ...
   ✓ Generating static pages (12/12)
   Finalizing page optimization ...

Route (app)                              Size     First Load JS
┌ ○ /                                    3.12 kB         114 kB
├ ○ /_not-found                          994 B           104 kB
├ ○ /explain                             10.4 kB         161 kB
├ ○ /forecast                            7.14 kB         149 kB
├ ○ /portfolio                           4.21 kB         121 kB
├ ○ /prediction                          8.11 kB         150 kB
├ ○ /references                          2.75 kB         114 kB
├ ○ /settings                            2.89 kB         114 kB
└ ○ /signals                             12.6 kB         163 kB
+ First Load JS shared by all            103 kB
  ├ chunks/4bd1b696-2fc5f15949d2dd88.js  54.2 kB
  ├ chunks/964-b52b36a1329a60c7.js       46.5 kB
  └ other shared chunks (total)          2.31 kB

○  (Static)  prerendered as static content
```
**Result:** 0 TypeScript errors, 0 linter errors, all 12 static routes generated.

---

## 5. How to Run

### Backend Service (FastAPI)
```powershell
.venv\Scripts\python.exe -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8000 --reload
```
- API Base: `http://127.0.0.1:8000/api/v1`
- Swagger UI Documentation: `http://127.0.0.1:8000/docs`

### Frontend Service (Next.js)
```powershell
cd frontend
npm run dev
```
- Web Application: `http://localhost:3000`
- "Why Did It Move?" Route: `http://localhost:3000/explain`
- "Actionable Signals" Route: `http://localhost:3000/signals`

### Legacy Streamlit Application (Remains 100% Operational)
```powershell
.venv\Scripts\streamlit.exe run app.py --server.port 8501
```
- Streamlit Interface: `http://localhost:8501`

---

## 6. Conclusion & Next Phase Readiness

With Phase 4 complete, all primary machine learning and quantitative research modules have been successfully migrated to the modern Next.js and FastAPI stack with full fidelity, interactive visualizations, and strict institutional UX standards.

The next planned step is **Phase 5: Virtual Portfolio, Paper Trading & Persistent SQLite Storage** (interactive buy/sell orders, live position tracking, trade history, and portfolio performance analytics).
