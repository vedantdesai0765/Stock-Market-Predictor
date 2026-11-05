# Phase 3 Deliverable Report: Visual Identity, Intuitive UX & Module Expansion

**Project:** Stock Market Predictor
**Phase:** Phase 3 — Brand Identity, Accessible UX, Directional Prediction Terminal & Next-Morning Forecast
**Date:** November 4, 2026
**Status:** Complete & Verified (`npm run build` completed with 0 errors across 12 static routes)

---

## 1. Executive Summary

Phase 3 of the Stock Market Predictor frontend development has been completed. This phase focused on three key areas:
1. **Brand Identity & Vector Logo:** A custom SVG vector logo representing the project's core purpose (three ascending bars, rising trendline, and an upward forecasting arrow) with scalable full and compact variants, and an integrated SVG favicon.
2. **Intuitive UX & Plain-English Explanations:** Complete elimination of unexplained jargon. Every machine learning, statistical, and quantitative finance term (RSI, MACD, CAGR, Sharpe ratio, Walk-forward validation, Calibrated probability, Embargo, Isotonic calibration) now includes plain-English labels, interactive explanation tooltips (`InfoTooltip`), and expandable technical deep-dives using progressive disclosure.
3. **Strict Color-Coded Signal Design System:**
   - **BUY / UP:** Green badge (`#15803D`), upward arrow icon, green-tinted card background.
   - **SELL / DOWN:** Red badge (`#DC2626`), downward arrow icon, red-tinted card background.
   - **HOLD:** Amber badge (`#D97706`), Pause icon, amber-tinted background, explicitly explained as a deliberate capital preservation choice where gross returns do not overcome 15 bps statutory costs.
   - **NO SIGNAL:** Neutral grey badge (`#475569`), MinusCircle icon, neutral-tinted background, explicitly defined as model abstention due to low statistical conviction.
   - **Strict Rule Enforced:** HOLD and NO SIGNAL are never interchangeable, and neutral model states are never converted into synthetic BUY/SELL signals.
4. **Phase 3 Core Module Implementations:**
   - **Directional Prediction Terminal (`/prediction`):** Live walk-forward predictions, visual conviction spectrum gauge, Recharts horizontal bar chart for top stationary feature weights, Recharts coverage vs. accuracy operating tradeoff curve, and expandable 5-fold walk-forward validation parameters.
   - **Next-Morning Forecast (`/forecast`):** Overnight gap directional predictability, 12-year aggregate evaluation (3,000+ sessions), Recharts comparison bar chart (Baseline Guess vs. Technicals vs. News-Enhanced), multi-model evaluation table, annual historical breakdown (2013–2025), and an intuitive guide explaining US markets close, GIFT Nifty, and overnight corporate announcements.

---

## 2. Brand Identity & Logo Specifications

Implemented in `frontend/src/components/brand/Logo.tsx` and `frontend/src/app/icon.svg`:

- **Design Elements:**
  - Three ascending vertical bars with teal-to-blue-to-green gradient fills (`#0D9488`, `#0284C7`, `#2563EB`).
  - Slicing upward trendline with gradient stroke (`#0D9488` to `#22C55E`).
  - Upward forecasting arrow head (`#22C55E`) with a data pulse coordinate dot (`#86EFAC`).
  - Rounded navy container (`#0B132B` to `#1E293B`) with subtle slate border.
- **Brand Text:**
  - Primary: **Stock Market Predictor** (Inter SemiBold, white on dark sidebar, slate-900 on light).
  - Subtitle: **Institutional Analytics Terminal** (JetBrains Mono uppercase tracking, sky-400).
- **Scalability:**
  - `LogoIcon`: Compact vector mark (sizes: sm 24px, md 32px, lg 40px, xl 48px).
  - `Logo`: Full horizontal brand mark for the navigation sidebar.
  - `icon.svg`: Directly loaded by modern browsers as the site favicon.

---

## 3. Accessible UX & Signal Color System

### Plain-English Translations & Tooltips (`InfoTooltip`)

| Technical Jargon | Plain-English Label | User-Facing Explanation | Technical Context in Tooltip |
|---|---|---|---|
| **Calibrated Probability** | Model Confidence | "Estimated likelihood of upward price movement. 50% is completely neutral." | Isotonic regression fit on out-of-sample walk-forward predictions |
| **Abstention Threshold** | Conviction Gate | "Required distance from 50/50 neutral before the model emits an actionable forecast." | Minimum operating point chosen to improve edge without falling below 20% coverage |
| **Walk-Forward Validation** | Chronological Testing | "Testing models only on future dates, exactly mimicking real-world live trading." | 5-Fold expanding window with 17-day embargo to eliminate label leakage |
| **Embargo** | Information Buffer | "17-day quarantine between training and testing sets to prevent overlapping return leakage." | Eliminates autocorrelation artifacts from 3-day non-overlapping targets |
| **Market Regime** | Market Environment | "Prevailing trend and volatility context: Trending Bull, Trending Bear, High Volatility, or Choppy." | Moving averages (SMA20/SMA50) and expanding volatility quantiles |
| **15 bps Friction** | Transaction Cost Hurdle | "Statutory round-trip costs (STT, brokerage, stamp duty, exchange fees, slippage)." | Realistic Indian delivery friction (0.15% round-trip) |

### Strict Signal Matrix

```text
┌──────────────┬──────────────┬──────────────┬────────────────────────────────────────────────────────┐
│ Signal       │ Badge Color  │ Icon         │ Meaning & Accessibility Rationale                     │
├──────────────┼──────────────┼──────────────┼────────────────────────────────────────────────────────┤
│ BUY / UP     │ Emerald      │ TrendingUp   │ Conviction and expected returns exceed costs           │
│ SELL / DOWN  │ Rose         │ TrendingDown │ Downward probability justifies capital preservation   │
│ HOLD         │ Amber        │ Pause        │ Expected returns do not exceed 15 bps friction         │
│ NO SIGNAL    │ Slate Grey   │ MinusCircle  │ Model conviction is below operating threshold          │
└──────────────┴──────────────┴──────────────┴────────────────────────────────────────────────────────┘
```

---

## 4. Implemented Module Architecture

### 1. Directional Prediction Terminal (`/prediction`)
- **Conviction Spectrum Bar:** Displays the balance between Bearish (`p_down`) and Bullish (`p_up`) probabilities, with a black center marker at 50% neutral and operating gate boundaries.
- **Feature Weights Chart:** Recharts horizontal bar chart highlighting top stationary financial drivers (e.g., Price vs. 20-Day SMA, 5-Day Price Velocity, Normalized Volatility). Positive bars are colored emerald green; negative bars rose red.
- **Coverage vs. Accuracy Tradeoff Curve:** Recharts line chart proving that as the model's confidence threshold increases, accuracy on remaining trading days rises.
- **Expandable Statistical Accordion:** Progressive disclosure revealing the 17-day embargo, 5-fold walk-forward validation setup, and academic integrity notes.

### 2. Next-Morning Opening Forecast (`/forecast`)
- **Overnight Gap Analytics:** Predicts the direction of tomorrow morning's opening price gap (09:15 AM IST) relative to today's closing price.
- **Model Comparison Chart:** Recharts bar chart comparing zero-intelligence baseline guessing against LightGBM, Logistic Regression, and XGBoost with technicals and news features.
- **Multi-Model Summary Table:** Details sample sizes ($n \approx 3,100$), edge over prior guessing ($+6.9\%$ to $+7.1\%$), and p-values ($p < 0.001$).
- **Annual Historical Breakdown Table:** Collapsible table displaying year-by-year performance across all 12 years (2013–2025).
- **Overnight Dynamics Guide:** Explains how US market closing sentiment (Dow Jones, S&P 500), GIFT Nifty morning trading, and overnight corporate filings shape opening prices.

---

## 5. Build & Compilation Verification

Ran `npm run build` in `frontend/`:

```text
 ✓ Compiled successfully
   Linting and checking validity of types ...
   Collecting page data ...
 ✓ Generating static pages (12/12)
   Finalizing page optimization ...
   Collecting build traces ...

Route (app)                              Size     First Load JS
┌ ○ /                                    5.98 kB         122 kB
├ ○ /_not-found                          876 B          88.4 kB
├ ○ /explain                             4.71 kB         108 kB
├ ○ /forecast                            4.37 kB         212 kB
├ ○ /icon.svg                            0 B                0 B
├ ○ /portfolio                           1.55 kB        89.1 kB
├ ○ /prediction                          13 kB           218 kB
├ ○ /references                          1.77 kB        89.3 kB
├ ○ /settings                            1.94 kB        98.1 kB
└ ○ /signals                             6.13 kB         109 kB
+ First Load JS shared by all            87.5 kB

○  (Static)  prerendered as static content
```

**Status:** 0 compilation errors, 0 type errors, 0 lint warnings.

---

## 6. How to Run the Updated Platform

### Terminal 1 — Start the FastAPI Backend
```powershell
# From C:\Users\vedan\Stock-Market-Predictor
.venv\Scripts\python.exe -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8000 --reload
```
- API Base: `http://127.0.0.1:8000/api/v1`
- Swagger UI: `http://127.0.0.1:8000/docs`

### Terminal 2 — Start the Next.js Frontend
```powershell
# From C:\Users\vedan\Stock-Market-Predictor\frontend
npm run dev
```
- Frontend UI: `http://localhost:3000`

---

## 7. Next Steps: Phase 4 & Phase 5

With the visual identity, accessible UX, Prediction Terminal, and Next-Morning Forecast fully implemented, the project is ready for:
- **Phase 4: Why Did It Move? & Actionable Signals Deep-Dive**
  - Interactive FinBERT entity sentiment timeline and price overlay.
  - Lead-lag cross-correlation visualization across $\pm 5$ trading day horizons.
  - Interactive walk-forward backtesting equity curve simulator with adjustable transaction friction and strategy variant selectors (A to E).
- **Phase 5: Virtual Portfolio Execution & End-to-End Hardening**
