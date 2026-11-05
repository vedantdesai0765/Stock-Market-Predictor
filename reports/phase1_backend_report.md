# Phase 1 Deliverable Report: FastAPI Backend Foundation

**Project:** Stock Market Predictor
**Target Architecture:** Next.js + Tailwind CSS Frontend with FastAPI Backend
**Author:** Pair Programming Assistant
**Date:** November 4, 2026
**Status:** Complete & Verified (12/12 Automated Tests Passing)

---

## 1. Executive Summary

Phase 1 of the frontend migration plan (`reports/frontend_migration_plan.md`) has been completed. A modular, production-style **FastAPI backend** foundation has been built inside `backend/` that exposes all existing Python machine learning models, directional predictions, next-morning forecasting, actionable trading signals, and walk-forward backtesting without modifying or duplicating any core mathematical logic.

### Key Accomplishments
1. **Zero Mathematical Deviation:** The backend wraps `src/pipeline.py`, `src/signals.py`, `src/backtest.py`, `src/config.py`, and forecast/news datasets via clean adapter services. Direct numerical parity tests confirm exact matches for prediction probabilities, confidence, backtest CAGR, Sharpe ratio, and drawdown.
2. **Standardized REST API (`/api/v1`):** Complete OpenAPI 3.1 / Swagger specification generated at `/docs` with strict Pydantic v2 schemas and validation for requests and responses.
3. **Transparent Data Freshness Service (`/data-status`):** Distinguishes raw OHLCV market data timestamps from model training dates and out-of-sample evaluation cutoffs, ensuring historical test data is never misrepresented as a live broker feed.
4. **Relational Database Schema (SQLAlchemy 2.0 + SQLite):** Schema initialized at `data/app.db` with models for `User`, `Wallet`, `Position`, `Transaction`, and `Watchlist` to support paper trading in subsequent phases.
5. **Comprehensive Automated Test Suite:** 12 integration and numerical parity tests in `backend/tests/` passed in 2.91s (`pytest`).
6. **Streamlit Integrity:** The existing Streamlit dashboard (`app.py`) on port 8501 remains untouched, fully functional, and runnable side-by-side with FastAPI.

---

## 2. Directory Structure

```text
backend/
├── app/
│   ├── __init__.py
│   ├── main.py                     # FastAPI app factory, CORS, DB lifespan, router mounting
│   ├── api/
│   │   ├── __init__.py
│   │   └── v1/
│   │       ├── __init__.py
│   │       ├── api.py              # Consolidated v1 router
│   │       └── endpoints/
│   │           ├── __init__.py
│   │           ├── health.py       # Health check & DB connection diagnostics
│   │           └── stocks.py       # REST endpoints for stocks, predictions, signals, etc.
│   ├── core/
│   │   ├── __init__.py
│   │   ├── config.py               # Path resolution, environment variables, CORS origins
│   │   └── errors.py               # Custom exceptions & consistent JSON error handler
│   ├── database/
│   │   ├── __init__.py
│   │   ├── base.py                 # DeclarativeBase for SQLAlchemy 2.0 ORM
│   │   └── session.py              # Engine, SessionLocal, get_db dependency, init_db()
│   ├── models/
│   │   ├── __init__.py
│   │   └── wallet.py               # ORM models for User, Wallet, Position, Transaction, Watchlist
│   ├── schemas/
│   │   ├── __init__.py
│   │   ├── stock.py                # StockItem, StockListResponse
│   │   ├── prediction.py           # PredictionResponse, FeatureImportanceItem, CoverageCurvePoint
│   │   ├── forecast.py             # ForecastResponse, ForecastSummaryItem, AnnualForecastItem
│   │   ├── signals.py              # LatestSignalResponse
│   │   ├── backtest.py             # BacktestResponse, BacktestMetricsResponse, BacktestEquityPoint
│   │   ├── news.py                 # NewsResponse, NewsArticleItem, LeadLagPoint
│   │   ├── status.py               # DataStatusResponse
│   │   └── wallet.py               # Pydantic schemas for User, Wallet, Position, Transaction
│   └── services/
│       ├── __init__.py
│       ├── stock_service.py        # Adapter for src/config.py
│       ├── prediction_service.py   # Adapter for src/pipeline.py
│       ├── forecast_service.py     # Adapter for reports/forecast/*.csv
│       ├── signals_service.py      # Adapter for src/signals.py
│       ├── backtest_service.py     # Adapter for src/backtest.py
│       ├── news_service.py         # Adapter for data/processed/news/ & explain reports
│       └── status_service.py       # Lineage, dataset boundaries & timestamp separator
└── tests/
    ├── conftest.py                 # Pytest TestClient fixtures & DB setup
    └── test_api.py                 # 12 automated integration and numerical parity tests
```

---

## 3. Implemented API Endpoints

All endpoints are versioned under `/api/v1` and documentable via Swagger UI at `http://localhost:8000/docs`.

| Method | Endpoint | Description | Service Adapter |
|---|---|---|---|
| `GET` | `/api/v1/health` | System health, environment, DB connectivity, stock count | Direct DB / `config.list_stocks()` |
| `GET` | `/api/v1/stocks` | Lists all 10 registered stocks and artifact availability | `StockService.get_all_stocks()` |
| `GET` | `/api/v1/stocks/{ticker}` | Returns single stock metadata and file flags | `StockService.get_stock()` |
| `GET` | `/api/v1/stocks/{ticker}/prediction` | Latest 3-day directional prediction, probability, confidence gate | `PredictionService.get_prediction()` |
| `GET` | `/api/v1/stocks/{ticker}/forecast` | Next-morning open forecast, overnight drift, model comparisons | `ForecastService.get_forecast()` |
| `GET` | `/api/v1/stocks/{ticker}/signals/latest` | Actionable BUY/SELL/HOLD signal, calibrated prob, cost hurdle | `SignalsService.get_latest_signal()` |
| `GET` | `/api/v1/stocks/{ticker}/backtest` | Walk-forward backtest equity curves, trade log, CAGR/Sharpe | `BacktestService.run_backtest()` |
| `GET` | `/api/v1/stocks/{ticker}/data-status` | Data lineage, raw date boundaries, training timestamp | `StatusService.get_data_status()` |
| `GET` | `/api/v1/stocks/{ticker}/news` | FinBERT targeted news sentiment, label counts, lead-lag curve | `NewsService.get_news_and_sentiment()` |

---

## 4. Verification & Numerical Parity

To ensure that the FastAPI layer does not alter research numbers or introduce regressions, automated tests were executed via `pytest`:

```text
============================= test session starts =============================
platform win32 -- Python 3.13.7, pytest-9.2.0, pluggy-1.7.0
plugins: anyio-4.10.0
collected 12 items

backend/tests/test_api.py::test_health PASSED                            [  8%]
backend/tests/test_api.py::test_stocks_list PASSED                       [ 16%]
backend/tests/test_api.py::test_stock_detail PASSED                      [ 25%]
backend/tests/test_api.py::test_stock_not_found PASSED                   [ 33%]
backend/tests/test_api.py::test_prediction_endpoint_and_numerical_parity PASSED [ 41%]
backend/tests/test_api.py::test_forecast_endpoint PASSED                 [ 50%]
backend/tests/test_api.py::test_signals_latest_endpoint PASSED           [ 58%]
backend/tests/test_api.py::test_signals_invalid_variant PASSED           [ 66%]
backend/tests/test_api.py::test_backtest_endpoint_and_numerical_parity PASSED [ 75%]
backend/tests/test_api.py::test_data_status_endpoint PASSED              [ 83%]
backend/tests/test_api.py::test_news_endpoint PASSED                     [ 91%]
backend/tests/test_api.py::test_all_stocks_data_status PASSED            [100%]

======================= 12 passed, 9 warnings in 2.91s ========================
```

### Numerical Parity Test Highlights (TCS):
- **Prediction Probability Parity:** `pipeline.predict_latest("TCS")` probability matches API `probability_up` exactly to 4 decimal places (`0.4957`).
- **Confidence Parity:** `confidence` matches API value `0.0043` with abstention gate triggering `NO SIGNAL` because `0.0043 < 0.1319`.
- **Backtest Parity:** Strategy CAGR, Sharpe ratio, maximum drawdown, and total trade counts from `GET /api/v1/stocks/TCS/backtest?variant=A` match `backtest.run_walk_forward_backtest("TCS")` to within floating-point tolerance (`abs=1e-4`).

---

## 5. Temporal Lineage & Data Freshness Integrity

One of the central requirements of Phase 1 was solving the ambiguity between historical evaluation dates and real-time market data. The `/api/v1/stocks/{ticker}/data-status` endpoint provides strict lineage separation:

```json
{
  "ticker": "TCS",
  "display_name": "Tata Consultancy Services",
  "raw_data_file": "TCS_raw.csv",
  "raw_data_exists": true,
  "raw_data_last_date": "2026-08-28",
  "raw_data_rows": 5485,
  "model_exists": true,
  "model_trained_at": "2026-11-01T05:21:04",
  "oof_eval_last_date": "2026-08-18",
  "news_file_exists": true,
  "news_last_date": "2025-12-31",
  "news_count": 4149,
  "status_summary": "Stock: Tata Consultancy Services (TCS.NS). Raw OHLCV data: Available up to 2026-08-28 (5485 records). Model artifacts: Trained on 2026-11-01T05:21:04. Out-of-sample evaluation boundary: 2026-08-18. News sentiment feed: Available up to 2025-12-31 (4149 articles). Mode: Historical research snapshot (live broker feed disabled).",
  "is_live_market": false
}
```

This guarantees that:
- Historical out-of-sample test dates (e.g. August 2026) are never fabricated or presented as current live market sessions.
- Research snapshots are clearly labeled as static evaluation records.

---

## 6. How to Run the Backend

To run the FastAPI backend locally:

```powershell
# From C:\Users\vedan\Stock-Market-Predictor
.venv\Scripts\python.exe -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8000 --reload
```

- **Interactive Swagger UI:** [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- **ReDoc Documentation:** [http://127.0.0.1:8000/redoc](http://127.0.0.1:8000/redoc)
- **Health Check:** [http://127.0.0.1:8000/api/v1/health](http://127.0.0.1:8000/api/v1/health)

---

## 7. Next Steps: Ready for Phase 2

Phase 1 is now completed and fully tested. We are ready to proceed to:
- **Phase 2: Next.js + Tailwind CSS Foundation & Shell**
  - Initialize Next.js project with App Router, TypeScript, and Tailwind CSS.
  - Configure shadcn/ui components, dark theme, navigation sidebar, and header.
  - Setup API client / React Query hooks fetching from `http://127.0.0.1:8000/api/v1`.
