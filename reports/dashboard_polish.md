# Dashboard Polish and Presentation Readiness Report

**Project:** Stock Market Predictor  
**Date:** November 2, 2026  
**Auditor / Presenter Engineer:** Quantitative Analytics Assistant  
**Artifact Directory:** `reports/dashboard_polish.md`  
**Updated Screenshots:** `screenshots/dashboard_terminal_polished.png`, `screenshots/why_did_it_move_polished.png`, `screenshots/dashboard_terminal_reliance.png`

---

## 1. Executive Summary

In preparation for tomorrow's presentation, a comprehensive UI/UX polish and scientific presentation-readiness pass was executed on the Stock Market Predictor terminal (`app.py` and `src/explain_page.py`).

The underlying machine-learning models, 5-fold walk-forward validation methodology, datasets, and predictions were **100% preserved**. The changes focused entirely on visual hierarchy, presentation clarity, elimination of layout collisions, transparent explanations of quantitative metrics, and immediate communication of the core empirical findings:

1. **Weak-Form Market Efficiency on Technical Indicators:** Demonstrates that after eliminating target overlap and enforcing a 17-trading-day embargo, technical models struggle to generate persistent out-of-sample edge (-0.9% average edge across 10 large-cap NSE equities), correctly triggering the system's honest abstention mechanism (`NO ACTIONABLE SIGNAL`).
2. **Semi-Strong Efficiency on News Sentiment:** Proves through contemporaneous event studies and entity-targeted FinBERT that news explains same-day returns ($\rho = +0.056, p < 10^{-4}$ for Sensex; $\rho = +0.030, p = 0.005$ pooled), but exhibits zero next-day predictive drift ($\rho = +0.007, p = 0.618$).

---

## 2. Summary of Dashboard Improvements

### 2.1 Top Navigation & Institutional Research Badges (`app.py`)
- **Institutional Research Header Badge Strip:** Added prominent metadata chips at the very top:
  - `NSE EQUITIES` • `5-FOLD WALK-FORWARD CV` • `NON-OVERLAPPING TARGETS (H=3)` • `17-DAY EMBARGO GAP` • `RESEARCH PROTOCOL · NOT FINANCIAL ADVICE`.
  - Sets an immediate tone of scientific rigor and regulatory compliance before evaluators even ask questions.
- **Executive Briefing & Evaluation Framework (Presenter Guide):** Added a clean collapsible card below the market snapshot bar. Presenters can open this accordion to explain:
  - The problem of naïve financial ML papers reporting 90%+ accuracy due to target leakage and serial autocorrelation.
  - The four-pillar zero-leakage validation design.
  - Why the terminal abstains rather than making unprofitable trades.

### 2.2 Plotly Chart Layouts & Collision Fixes (`app.py`)
- **Candlestick Chart Range Selector Fix:** Moved the `1M`, `3M`, `6M`, `1Y`, `ALL` rangeselector buttons up (`y = 1.14`) and increased the top margin (`t = 75`), completely resolving the visual collision where buttons overlapped the chart title and legends.
- **Axis Labels and Hovers:** Added explicit axis labels (`Price (INR ₹)` and `Trading Date`), and formatted OHLC tooltips with Indian Rupee formatting (`₹%{close:,.2f}`).
- **Secondary Indicator Polish:**
  - **RSI Chart:** Added overbought (`70`) and oversold (`30`) dashed reference lines with clean, color-coded annotations and explicit axis limits.
  - **Volume Chart:** Formatted daily volume in shares with clean thousands-separated hover tooltips.
- **Validation Diagnostics Layout:**
  - **Accuracy vs. Confidence Coverage Chart:** Repositioned the legend to the bottom (`y = -0.28`), preventing collision with the chart title. Converted axes from decimals (`0.52`) to percentages (`52%`).
  - **Out-of-Fold Accuracy Chart:** Added percentage labels directly on top of each fold bar (e.g., `55.8%`, `59.1%`) and added a 50% chance dashed reference line.
  - **Feature Coefficients Chart:** Replaced raw variable names (e.g., `hl_range_norm`, `bb_bandwidth_zscore`) with human-readable labels (`High-Low Range (Norm)`, `BB Bandwidth (Z-Score)`, `Ichimoku Cloud Thickness`, etc.) and set left margin `l = 150` to prevent text truncation.

### 2.3 Clarity on Prediction and Confidence Metrics (`app.py`)
- **Model Signal Card:** Clarified that the target represents the sign of the 3-day forward log return ($t$ to $t+3$) with an honest `NO ACTIONABLE SIGNAL` amber badge when edge is non-positive.
- **Probability vs. Confidence Card:** Explicitly separated the directional probability ($P(\text{UP})$) from confidence, displaying:
  $$\text{Confidence: } |P(\text{UP}) - 0.50| = 0.084 \quad (\text{Gate Threshold: } 0.140)$$
  This eliminates evaluator confusion regarding raw probabilities versus abstention thresholds.
- **Hit Rate Card:** Delineated un-gated full walk-forward accuracy from gated operating points to avoid displaying unachievable accuracy figures during abstention.

### 2.4 Deep-Dive Event Study Polish ("Why did it move?" / `src/explain_page.py`)
- **Executive Finding Callout Banner:** Added a prominent indigo banner at the top:
  > **Core Empirical Finding:** News explains today's move, but does not forecast tomorrow ($\rho = +0.056, p < 10^{-4}$ same-day vs. $\rho = +0.007, p = 0.618$ next-day).
- **Multiselect Chip Styling:** Replaced the default harsh red Streamlit filter tags with sleek slate-indigo chips matching the dark financial terminal theme.
- **Day Explorer Table:** Formatted sentiment scores (`Towards Company` and `Plain FinBERT`) to signed 3-decimal format (e.g. `+0.742`, `-0.510`), preventing raw floating-point clutter.
- **The Largest Moves Table:** Formatted percentage returns (e.g. `+15.52%`, `-30.85%`) and sentiment scores consistently.
- **Evidence Quintiles & Event Study:** Added an empirical guide callout explaining how green same-day bars scale across quintiles while grey next-day bars stay flat near zero.
- **Dynamic Module Reloading:** Added `importlib.reload(explain_page)` in `app.py` to ensure that any edits in `src/explain_page.py` are immediately hot-reloaded by Streamlit without requiring server restarts.

### 2.5 Query Parameter Navigation Support (`app.py`)
- Enabled direct URL navigation via Streamlit query parameters:
  - `?stock=INFY`
  - `?stock=INFY&view=explain`
  - `?stock=RELIANCE`
- Allows the presenter to bookmark specific demo pages or switch smoothly between demonstrations.

---

## 3. Visual Verification & Screenshot Evidence

All screenshots have been generated using automated headless Chrome CDP and saved with new filenames (preserving the original screenshots):

| Screenshot File | Resolution | Content Description |
|---|---|---|
| `screenshots/dashboard_terminal_polished.png` | 1600 × 2600 | Infosys (INFY) full terminal: badge strip, executive briefing, snapshot bar, verdict banner, candlestick chart, indicators, diagnostics, and compliance footer. |
| `screenshots/why_did_it_move_polished.png` | 1600 × 2600 | Infosys (INFY) "Why did it move?" view: research callout banner, 3,176 headlines, 2,109 filings, Day Explorer table, largest moves, and quintiles chart. |
| `screenshots/dashboard_terminal_reliance.png` | 1600 × 2600 | Reliance Industries (RELIANCE) full terminal: clean indicators, out-of-fold bar chart, and feature coefficients. |
| `screenshots/why_did_it_move_infy.png` *(Preserved)* | Original | Original baseline screenshot preserved for historical comparison. |
| `screenshots/why_did_it_move_sensex.png` *(Preserved)* | Original | Original market baseline screenshot preserved for historical comparison. |

---

## 4. Presentation Tips for Tomorrow

When presenting the project to evaluators or faculty, follow this narrative arc:

### Step 1: The Problem with Traditional Stock Prediction Projects (1–2 minutes)
- Point to the **VALIDATION METHODOLOGY** checklist in the sidebar and the **Executive Briefing** expander.
- Explain: *"Most stock market prediction projects naively train an LSTM or XGBoost on raw prices with overlapping rolling windows. They report 90% accuracy, but it is 100% lookahead bias and target autocorrelation. When deployed, they lose capital immediately."*

### Step 2: Scientific Rigour & Honest Abstention (2–3 minutes)
- Show **Infosys (INFY)** in the main terminal (`?stock=INFY`):
  - Walk-Forward Accuracy: **53.9%**
  - Honest Baseline: **55.2%**
  - Edge Over Baseline: **-1.3%**
  - Overall Verdict: **NO ACTIONABLE SIGNAL**
- Key talking point: *"Our system does not force trades. Under strict expanding-window cross-validation with a 17-day embargo, technical indicators provide no persistent alpha over the majority/persistence baseline. The model honestly abstains."*
- Show the **OUT-OF-FOLD ACCURACY STABILITY** chart (consistent ~50%–55% across all 5 folds) and **TOP 10 FEATURE COEFFICIENTS** (standardized L2 logistic weights).

### Step 3: "Why Did It Move?" — News Sentiment Event Study (3–4 minutes)
- Switch to **"Why did it move?"** (`?stock=INFY&view=explain`):
  - Point to the 3,176 headlines and 2,109 company filings.
  - Explain: *"We built an entity-targeted FinBERT to score sentiment specifically towards the company, not just overall market tone."*
  - Look at the **Largest Moves** table (e.g. 10 April 2003: -30.85% drop on *'Infosys says the bad times have just begun'*).
  - Point to Section 04 **HOW CLOSELY NEWS TONE AND PRICE MOVES LINE UP**:
    - Same-day correlation is positive and statistically significant ($\rho = +0.051, p = 0.038$).
    - Next-day correlation drops to zero ($\rho = +0.007$).
    - In the **Quintiles Chart**, green bars scale upward while grey bars remain flat.
  - Conclusion: *"Prices incorporate news instantly on the day it breaks (Semi-Strong Market Efficiency). Financial news is an explanatory tool, not a crystal ball for tomorrow's market."*

### Recommended Stocks for the Demo:
- **INFY (Infosys):** Best for demonstrating both technical validation and extensive 20-year news coverage.
- **RELIANCE:** High-volume blue-chip stock demonstrating stable multi-fold diagnostics.
- **SENSEX (in Evidence scope):** Demonstrates pooled macroeconomic significance ($n = 4,000+$ days).

---

## 5. Exact Launch Command for Tomorrow

To launch the dashboard for the presentation:

```powershell
cd C:\Users\vedan\Stock-Market-Predictor
.venv\Scripts\streamlit run app.py
```

Or with Python directly:

```powershell
cd C:\Users\vedan\Stock-Market-Predictor
.venv\Scripts\python.exe -m streamlit run app.py
```

Direct URLs:
- **Main Terminal:** `http://localhost:8501?stock=INFY`
- **Why Did It Move?:** `http://localhost:8501?stock=INFY&view=explain`
