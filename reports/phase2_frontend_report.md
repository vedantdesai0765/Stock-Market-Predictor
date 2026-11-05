# Phase 2 Deliverable Report: Next.js Frontend Foundation

**Project:** Stock Market Predictor
**Target Architecture:** Next.js (App Router) + React + TypeScript + Tailwind CSS + TanStack Query
**Author:** Pair Programming Assistant
**Date:** November 4, 2026
**Status:** Complete & Verified

---

## 1. Executive Summary

Phase 2 of the approved frontend migration has been completed. A responsive, institutional financial analytics frontend has been built in the `frontend/` directory. It interfaces with the FastAPI backend foundation established in Phase 1 and preserves the existing Python research pipeline and Streamlit application (`app.py` on port 8501).

### Key Accomplishments
1. **Application Shell & Design System:**
   - Deep navy sidebar (`#0B132B`) with institutional branding and legacy Streamlit port indicator.
   - Light neutral workspace (`#F8FAFC`) with white cards, subtle borders (`#E2E8F0`), and dark navy typography (`#0F172A`).
   - Clean color-coded financial metrics: emerald green for positive/BUY signals, rose red for negative/SELL signals, amber for delayed/historical research snapshots, sky blue/teal for indicators and accents.
2. **8-Point Institutional Navigation:**
   - **Overview** (`/`): High-level market summary, latest close, actionable signal, 3-day direction, morning forecast, and module links.
   - **Prediction Terminal** (`/prediction`): Directional edge, confidence abstention, walk-forward accuracy, and feature weights.
   - **Next-Morning Forecast** (`/forecast`): Overnight gap predictability, multi-model evaluation (LightGBM, LogReg, XGBoost).
   - **Why Did It Move?** (`/explain`): Entity-targeted FinBERT headline sentiment and lead-lag analysis.
   - **Actionable Signals** (`/signals`): BUY/SELL/HOLD decision engine with 15 bps Indian statutory friction and backtest metrics.
   - **Virtual Portfolio** (`/portfolio`): Prepared for Phase 5 paper trading execution.
   - **References** (`/references`): Foundational academic literature and methodology citations.
   - **Settings** (`/settings`): Backend endpoint diagnostic status and connection parameters.
3. **Reusable Component Architecture:**
   - `StockSelector`: Dropdown allowing selection across all ten supported Indian stocks (`TCS`, `RELIANCE`, `INFY`, `HDFCBANK`, `SBI`, `ITC`, `ADANIENT`, `VEDL`, `ETERNAL`, `ADANIGREEN`).
   - `MetricCard`: Institutional KPI display with primary value, delta, secondary badge, and tooltip.
   - `SignalBadge`, `FreshnessBadge`, `RegimeBadge`: Clear visual states distinguishing live data from historical snapshots.
   - `LoadingSkeleton` & `ErrorState`: Graceful degradation for loading and backend connection errors.
   - `ModuleCard`: Direct entry points for the four specialized research engines.
4. **Data Integrity & Strict Lineage:**
   - Consumes actual FastAPI endpoints only; zero fabricated mock stock prices or signals.
   - Clearly labels data as a static research snapshot (`is_live_market = false`) with explicit date boundaries.

---

## 2. Directory Structure

```text
frontend/
├── .env.local                          # NEXT_PUBLIC_API_URL=http://127.0.0.1:8000/api/v1
├── next.config.mjs                     # Next.js 14 configuration
├── package.json                        # Dependencies (Next.js, React 18, TanStack Query, Recharts, Tailwind)
├── postcss.config.mjs                  # PostCSS plugins
├── tailwind.config.ts                  # Institutional financial palette & typography
├── tsconfig.json                       # Strict TypeScript configuration
└── src/
    ├── app/
    │   ├── globals.css                 # Base styles & custom scrollbars
    │   ├── layout.tsx                  # Root layout with QueryProvider and Shell
    │   ├── page.tsx                    # Overview Page
    │   ├── prediction/page.tsx         # Directional Prediction Terminal
    │   ├── forecast/page.tsx           # Next-Morning Forecast Page
    │   ├── explain/page.tsx            # Why Did It Move? Page
    │   ├── signals/page.tsx            # Actionable Signals & Backtest Page
    │   ├── portfolio/page.tsx          # Virtual Portfolio Placeholder
    │   ├── references/page.tsx         # Research References Page
    │   └── settings/page.tsx           # Terminal Diagnostics Page
    ├── components/
    │   ├── layout/
    │   │   ├── Sidebar.tsx             # Deep navy navigation sidebar
    │   │   ├── Header.tsx              # Stock selector, live IST clock, freshness badge
    │   │   └── Shell.tsx               # Responsive layout shell
    │   ├── providers/
    │   │   └── QueryProvider.tsx       # TanStack Query & Stock context provider
    │   └── ui/
    │       ├── Badge.tsx               # Signal, freshness, and regime badges
    │       ├── MetricCard.tsx          # Institutional KPI metric card
    │       ├── StockSelector.tsx       # 10-stock dropdown selector
    │       ├── ModuleCard.tsx          # Research module preview card
    │       ├── LoadingSkeleton.tsx     # Skeleton loaders
    │       └── ErrorState.tsx          # API connection error card
    ├── context/
    │   └── StockContext.tsx            # Active stock selection state
    ├── hooks/
    │   └── useStockData.ts             # TanStack Query hooks for backend endpoints
    ├── lib/
    │   ├── api.ts                      # Typed API client
    │   └── utils.ts                    # Formatting helpers (INR currency, percent, bps)
    └── types/
        └── api.ts                      # Strict TypeScript interfaces matching backend schemas
```

---

## 3. Independent Execution Instructions

The backend and frontend are independently runnable and decoupled.

### Step 1: Start the FastAPI Backend
Open a terminal in `C:\Users\vedan\Stock-Market-Predictor`:
```powershell
.venv\Scripts\python.exe -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8000 --reload
```
- API Base URL: `http://127.0.0.1:8000/api/v1`
- Swagger Docs: `http://127.0.0.1:8000/docs`

### Step 2: Start the Next.js Frontend
Open a separate terminal in `C:\Users\vedan\Stock-Market-Predictor\frontend`:
```powershell
npm run dev
```
- Frontend UI: `http://localhost:3000`

### Step 3: Run the Legacy Streamlit Dashboard (Optional)
The Streamlit dashboard remains untouched and operational on port 8501:
```powershell
.venv\Scripts\streamlit.exe run app.py --server.port 8501
```

---

## 4. Next Steps: Phase 3

With the Next.js application shell, state management, and API layer complete, the project is ready for:
- **Phase 3: Prediction Terminal & Next-Morning Forecast Pages Migration**
  - Implement interactive candlestick / price charts with TradingView Lightweight Charts or Recharts.
  - Implement probability gauge, walk-forward fold performance breakdown, and feature importance bar charts.
  - Implement overnight forecast drift tables and morning open comparison charts.
