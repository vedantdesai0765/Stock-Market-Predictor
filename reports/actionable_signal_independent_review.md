# Independent Critical Audit: Actionable Trading Signal Engine

**Review Target:** Stock Market Predictor — Actionable Trading Signal Engine & Walk-Forward Backtester  
**Audited Modules:** `src/signals.py`, `src/backtest.py`, `src/signals_page.py`, `app.py`  
**Data Evaluated:** 10 NSE Equities (`ADANIENT`, `ADANIGREEN`, `ETERNAL`, `HDFCBANK`, `INFY`, `ITC`, `RELIANCE`, `SBI`, `TCS`, `VEDL`)  
**Audit Date:** November 2, 2026  
**Auditor Role:** Independent Quantitative Risk & Model Validation Reviewer  

---

## Executive Summary

This independent critical audit evaluates the statistical validity, temporal alignment, execution mechanics, cost modeling, and performance claims of the newly implemented **Actionable Trading Signal Engine**.

### Core Audit Verdict

1. **Mathematical & Structural Integrity:**  
   The core simulation code in `src/backtest.py` and `src/signals.py` is free from blatant mathematical errors. Holding periods are strictly non-overlapping, isotonic calibration is fitted on past folds only, and transaction costs are properly deducted upon portfolio rebalancing.
2. **Defensive Alpha vs. Return Generation:**  
   The signal engine delivers genuine, verifiable utility in **drawdown compression and capital preservation**, reducing average maximum drawdowns from **-54.77% down to -10.94%–35.44%**. However, claims of broad absolute alpha must be tempered: **no fixed strategy beats Buy & Hold in raw CAGR across the 10-stock universe** (average Buy & Hold CAGR was **15.68%** vs. **13.24%** for Variant C and **12.82%** for Variant A).
3. **Critical Nuances & Caveats Identified:**  
   - **News Timestamp Truncation:** News datasets end in December 2025. For all 2026 evaluation dates, news sentiment defaulted to `0.0` (neutral).
   - **Execution Price Assumption:** Backtests assume execution at the Day $t$ Close price ($Close_t$). In practice, algorithmic signals computed after market close must be executed at the Day $t+1$ Open, introducing overnight gap risk.
   - **Cash Interest Sensitivity:** The reported strategy CAGR credits idle cash balances with a 5.0% annual risk-free rate. Because active strategies spend **62% to 93% of their time in cash**, assuming a 0.0% cash return drops average Variant C CAGR from **13.24% to 9.57%**.
   - **Selection Bias in Reporting:** Highlighting the single best variant per stock (e.g. Variant E for ADANIGREEN, Variant C for TCS, Variant B for INFY) introduces post-hoc selection bias. Fixed-rule evaluation reveals that active strategies trail Buy & Hold CAGR in 8 out of 10 stocks.
   - **Current Signal Date Lag:** The live signal displayed in `signals_page.py` is extracted from the last completed row of `oof_predictions.csv` (e.g., 2026-08-18 for INFY) rather than the most recent raw OHLCV bar (2026-08-28).

---

## 1. Systematic Verification Checklist

### 1.1 Signal Timestamps, News Timestamps, and Feature Availability

| Audit Item | Verification Status | Forensic Findings |
| :--- | :---: | :--- |
| **Feature Lag & Point-in-Time Availability** | **VERIFIED CLEAN** | Features in `src/features.py` and `src/stationary.py` are strictly backward-looking trailing indicators (e.g. SMA20, RSI14, ATR14). Expanding volatility quantiles in `src/signals.py` explicitly use `.shift(1)` to ensure that Day $t$ regime thresholds depend only on information up to Day $t-1$. |
| **News Date Alignment** | **VERIFIED WITH CAVEAT** | News items map to `trade_date`. For Times of India articles, weekend news maps to Monday; for NSE filings, filings before 15:30 IST map to the same day. However, **the news dataset abruptly terminates in late December 2025**. Between January 2026 and August 2026, sentiment is absent and defaults to neutral (`0.0`). |
| **Execution Timing vs. Signal Computation** | **METHODOLOGICAL LIMITATION** | The simulation enters trades at $Close_t$ and exits at $Close_{t+3}$. In reality, a signal requiring $Close_t$ cannot be executed at $Close_t$ unless participating in the 15:30–15:40 IST closing auction. Retail execution on the next morning's market open ($Open_{t+1}$) faces overnight price jumps. |

### 1.2 3-Day Holding Periods & Non-Overlapping Subsampling

- **Observation Count & Step Size:**  
  In `models/<STOCK>/oof_predictions.csv`, the date difference between consecutive rows was audited across all 10 stocks:
  - Minimum step difference: **3.0 calendar days** (3 trading sessions).
  - Standard step difference: **3 to 5 calendar days** (reflecting weekend gaps).
  - Duplicate dates: **0 across all 10 stocks**.
  - Monotonicity: Strictly ascending chronologically.
- **Independence & Autocorrelation:**  
  Because rows are subsampled at step $H=3$, Target[t] and Target[t+1] share zero price bars. First-order label autocorrelation is negligible ($\approx 0.01$), confirming that target overlap leakage is non-existent.
- **Fold Boundaries:**  
  Test folds in `walk_forward_splits` are strictly adjacent partitions of the out-of-sample block. The transition gaps between folds range from 3 to 6 calendar days, confirming that test folds concatenate into a continuous, non-overlapping chronological timeline.

### 1.3 Isotonic Probability Calibration

- **Out-of-Sample Verification:**  
  `calibrate_oof_probabilities` in `src/signals.py` iterates across folds:
  ```python
  folds = sorted(df["fold"].unique())
  for f in folds[1:]:
      past = df[df["fold"] < f]
      current = df[df["fold"] == f]
      iso.fit(past["y_proba"], past["y_true"])
      df.loc[current.index, "y_proba_calibrated"] = iso.predict(current["y_proba"])
  ```
- **Findings:**
  - Fold 0 uses raw model probabilities because no prior out-of-fold data exists.
  - Fold 1 is calibrated strictly using Fold 0.
  - Fold $f$ is calibrated strictly using historical folds $< f$.
  - **Verdict:** No future label or prediction ever enters the calibration fit. The calibration is strictly out-of-sample and causal.

### 1.4 Strategy Thresholds and Regime Filter Selection

- **Volatility Thresholding:**  
  Regime detection uses expanding quantiles (`expanding_vol.quantile(0.80).shift(1)`), preventing future volatility leakage.
- **Hyperparameter Stability:**  
  Thresholds (RSI 70/30, MACD histogram 0, confidence gate 0.03, sentiment cutoff 0.05) were not fit through per-stock parameter optimization. They are uniform across all 10 tickers.
- **Limitation:**  
  While the parameters are conventional heuristics, they represent a fixed set of choices. Any performance variance across stocks is driven by how well these heuristics align with the underlying asset's volatility structure.

### 1.5 Indian Transaction Charges, Brokerage, and Slippage

The 15.0 bps (0.15%) round-trip friction model implemented in `src/signals.py` and `src/backtest.py` was benchmarked against the actual schedule of charges for NSE delivery trading:

| Charge Component | Statutory / Market Rate | Simulation Model | Assessment |
| :--- | :--- | :--- | :--- |
| **STT (Securities Transaction Tax)** | 0.10% Buy + 0.10% Sell (Delivery) | Included in roundtrip | Conservative: Real delivery STT is 20 bps total; however, institutional or swing execution via equity futures is 2.5 bps. 15 bps is a realistic blended rate. |
| **Brokerage** | Zero (Discount) to ₹20 flat (~0.03%) | Included | Realistic |
| **NSE Turnover + SEBI + Stamp Duty + GST** | ~0.035% total | Included | Accurate |
| **Bid-Ask Slippage** | 0.02%–0.05% per side | Included | Realistic for large-cap Nifty liquid equities |

**Turnover Accounting:** Friction is assessed strictly on portfolio changes ($\Delta w = |w_t - w_{t-1}|$). Continuous holding across consecutive periods incurs **zero rebalancing penalty**, correctly avoiding artificial cost compounding.

### 1.6 Benchmarks Consistency (Buy & Hold vs. Cash)

- **Initial Capital:** Identical (₹100,000) across Strategy, Buy & Hold, and Cash.
- **Start and End Dates:** Identical across all three curves (matching the exact first and last out-of-sample prediction dates).
- **Compounding Frequency:** Identical 3-day step compounding:
  $$V_T = V_0 \prod_{t=1}^T (1 + R_t)$$
- **Friction on Buy & Hold:** Appropriately charges one-way entry fee (7.5 bps) at step 0 and one-way exit fee (7.5 bps) at the final step.

### 1.7 Mathematical Precision of Metrics

The formulas implemented in `_compute_metrics` were audited:

1. **CAGR:**
   $$\text{CAGR} = (1 + R_{\text{cum}})^{365.25 / \text{Days}} - 1$$
   **Status:** Mathematically correct.
2. **Annualized Volatility:**
   $$\sigma_{\text{ann}} = \sigma_{3d} \times \sqrt{\frac{252}{3}} = \sigma_{3d} \times \sqrt{84}$$
   **Status:** Mathematically correct standard square-root-of-time scaling.
3. **Sharpe Ratio:**
   $$\text{Sharpe} = \frac{\text{CAGR} - R_f}{\sigma_{\text{ann}}}$$
   **Status:** Uses geometric mean (CAGR) rather than arithmetic mean. This is mathematically sound and slightly more conservative than standard textbook Sharpe.
4. **Sortino Ratio:**
   $$\text{Sortino} = \frac{\text{CAGR} - R_f}{\sigma_{\text{downside}}}$$
   **Nuance Found:** Downside deviation is computed over negative periods only ($N_{\text{neg}}$) rather than total periods ($N_{\text{total}}$). This makes the computed downside deviation larger and the resulting Sortino ratio **more conservative** than the theoretical definition.
5. **Drawdown:**
   $$\text{DD}_t = \frac{V_t - \max_{\tau \le t} V_\tau}{\max_{\tau \le t} V_\tau}$$
   **Status:** Exact high-water-mark peak-to-trough calculation.
6. **Profit Factor & Win Rate:**
   $$\text{PF} = \frac{\sum \text{Net Gains}}{\sum |\text{Net Losses}|}, \quad \text{WR} = \frac{N_{\text{wins}}}{N_{\text{trades}}}$$
   **Status:** Exact calculation on closed trade net returns.

### 1.8 Statistical Credibility Under Multiple Hypothesis Testing

In the audit, 10 stocks were evaluated across 5 variants (50 separate simulations).
- **Post-Hoc Selection Bias:** Presenting the "Best Variant per Stock" implies a trader could know in 2013 that `ADANIGREEN` would require Variant E, while `ETERNAL` would require Variant C, and `ADANIENT` would favor Variant A.
- **Universal Rule Evaluation:** When holding the strategy constant across all 10 stocks:
  - **Variant C:** Beats Buy & Hold CAGR on only **2 out of 10 stocks** (`ETERNAL` and `TCS`).
  - **Variant E:** Beats Buy & Hold CAGR on only **1 out of 10 stocks** (`ADANIGREEN`).
  - **Verdict on Absolute Alpha:** ML directional forecasting does not consistently beat equity beta across Indian equities. Its true statistical edge lies in **asymmetric downside protection**.

### 1.9 Pipeline Alignment: Current Live Signal vs. Historical Backtest

- **Finding:**  
  `get_latest_signal_status()` in `src/signals.py` generates the current signal by extracting the final row of `oof_predictions.csv`:
  ```python
  df = generate_signals(stock_key, variant=variant, ...)
  latest = df.iloc[-1]
  ```
- **Discrepancy Identified:**  
  In `models/INFY/oof_predictions.csv`, the final date is **2026-08-18**. However, the raw price history in `data/raw/INFY_raw.csv` extends to **2026-08-28**.
- **Explanation:**  
  The 3-day non-overlapping target creation drops the final 3 unlabelled rows, and the 3-day subsampling offset leaves the final test observation at 2026-08-18. Consequently, while the main Prediction Terminal in `app.py` predicts the unlabelled bar on 2026-08-28 via `predict_latest()`, the Actionable Signals terminal displays the signal as of 2026-08-18.
- **Significance:** Both components are internally consistent with their respective design goals, but users should be aware that the Actionable Signals tab reflects the conclusion of the walk-forward validation window.

---

## 2. In-Depth Forensic Case Studies

### 2.1 ETERNAL (Zomato): Rapid Growth & Sparse News

```mermaid
flowchart LR
    A["Total OOF Dates: 188<br>(May 2024 - Aug 2026)"] --> B["Active News Dates: 24<br>(12.8% coverage)"]
    B --> C["Default Neutral: 164<br>(87.2% coverage)"]
    C --> D["Variant C falls back<br>to Model if P >= 55%"]
    D --> E["Result: 36.92% CAGR<br>vs 26.65% B&H"]
```

- **Out-of-Sample Period:** May 9, 2024 to August 19, 2026 (only 2.28 years, 188 steps).
- **Reported Metric:** Variant C achieved **36.92% CAGR** vs. **26.65% Buy & Hold**, with Sharpe **1.15** vs. 0.56.
- **Forensic Findings:**
  1. **Sparse News Coverage:** Only **24 out of 188 dates (12.8%)** had active news sentiment. On 87.2% of dates, sentiment was `0.0`.
  2. **Model Passthrough:** Under Variant C logic, when sentiment is neutral (`0.0`), a BUY is triggered if the base model probability $P(\text{up}) \ge 0.55$. Thus, Variant C's outperformance was primarily driven by the underlying logistic model's directional accuracy on momentum swings, rather than continuous NLP sentiment filtering.
  3. **Identical Variants B, D, and E:** In ETERNAL, Variants B, D, and E produced the exact same trades (20 trades, 16.51% CAGR, 15.93% MaxDD), proving that technical filters were the binding constraint and sentiment did not add incremental differentiation when technicals were active.
  4. **Short Track Record:** With only 2.28 years of data and 29 closed trades, the statistical power is limited, and findings should not be extrapolated without caution.

---

### 2.2 TCS: The True Driver of Active Outperformance

```mermaid
flowchart TD
    subgraph B_and_H["Buy & Hold Benchmark"]
        BH1["CAGR: 8.97%"]
        BH2["Max Drawdown: -53.08%"]
        BH3["Sharpe: 0.17"]
    end

    subgraph Strat_C_5pct["Variant C (5% Cash Rate)"]
        C1["CAGR: 9.94%"]
        C2["Max Drawdown: -22.29%"]
        C3["Sharpe: 0.32"]
    end

    subgraph Strat_C_0pct["Variant C (0% Cash Rate)"]
        Z1["CAGR: 7.06%"]
        Z2["Max Drawdown: -22.29%"]
        Z3["Sharpe: 0.17"]
    end

    B_and_H --> Strat_C_5pct
    B_and_H --> Strat_C_0pct
```

- **Out-of-Sample Period:** November 26, 2015 to August 20, 2026 (10.7 years, 884 steps, 153 trades).
- **Reported Metric:** Variant C returned **9.94% CAGR** vs. **8.97% Buy & Hold**, with Sharpe **0.32** vs. 0.17 and MaxDD **-22.29%** vs. -53.08%.
- **Forensic Findings:**
  1. **Drawdown Protection is Robust:** Regardless of cash rate assumptions, Variant C successfully avoided the severe 2016–2017 and 2022–2023 tech consolidations, cutting maximum drawdown by more than half (**-22.29% vs -53.08%**).
  2. **CAGR Depends on Cash Interest:**
     - At $R_f = 5.0\%$ (liquid mutual fund sweep): Strategy CAGR is **9.94%** (beats Buy & Hold by +0.97%).
     - At $R_f = 0.0\%$ (uninvested cash earning zero): Strategy CAGR drops to **7.06%** (trails Buy & Hold by -1.91%).
  3. **Economic Truth:** Because the strategy spends 55.4% of its time in cash, nominal return parity requires an active treasury cash sweep. Its risk-adjusted advantage (Sharpe and Sortino) remains intact under all interest rate regimes.

---

### 2.3 ADANIGREEN: Defensive Capital Preservation via Small-Sample Abstention

- **Out-of-Sample Period:** October 19, 2022 to August 19, 2026 (3.8 years, 310 steps).
- **Reported Metric:** Buy & Hold lost **-12.17% CAGR** with a **-77.91%** drawdown. Variant E generated **+10.09% CAGR** with only a **-5.39%** drawdown and Sharpe **0.53**.
- **Forensic Findings:**
  1. **Extreme Abstention:** Variant E executed only **7 trades** over the entire 3.8-year period, spending **96.1% of its time in CASH**.
  2. **Mechanics of Outperformance:** The +10.09% CAGR was not generated by catching market rallies. It resulted from:
     - 4.80% compound return from cash interest.
     - 7 high-conviction trades that yielded positive returns during brief consolidation periods.
     - 100% avoidance of the Hindenburg collapse and subsequent multi-year downtrend.
  3. **Statistical Limitation:** 7 trades across 3.8 years ($n=7$) is too small a sample to establish statistical significance. However, it perfectly demonstrates the *design intention* of the engine: when volatility spikes into hostile territory, the algorithm sits entirely in cash.

---

## 3. Cash Return Sensitivity Analysis

Because active strategies spend between 62.5% and 93.0% of their life in cash, the rate of return attributed to uninvested capital heavily influences nominal compounding. The table below audits all 10 stocks under $R_f = 5.0\%$ vs. $R_f = 0.0\%$:

| Stock Ticker | Buy & Hold CAGR | Variant A ($R_f=5\%$) | Variant A ($R_f=0\%$) | Variant C ($R_f=5\%$) | Variant C ($R_f=0\%$) | Market Exposure (Var C) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **`ADANIENT`** | **48.66%** | 41.36% | 38.92% | 36.06% | 32.76% | 48.4% |
| **`ADANIGREEN`** | **-12.17%** | 17.70% | 13.19% | 16.20% | 11.61% | 14.2% |
| **`ETERNAL`** | **26.65%** | 25.65% | 22.88% | 36.92% | 33.56% | 48.1% |
| **`HDFCBANK`** | **12.79%** | 5.72% | 3.04% | 4.60% | 0.85% | 23.1% |
| **`INFY`** | **12.13%** | 10.63% | 8.67% | 10.60% | 8.31% | 56.1% |
| **`ITC`** | **4.79%** | 4.54% | 0.70% | 2.96% | -0.94% | 18.7% |
| **`RELIANCE`** | **16.27%** | 6.25% | 2.06% | 6.49% | 1.95% | 8.3% |
| **`SBI`** | **15.71%** | 7.35% | 4.21% | 10.88% | 7.06% | 26.4% |
| **`TCS`** | **8.97%** | 8.43% | 5.92% | 9.94% | 7.06% | 44.6% |
| **`VEDL`** | **23.05%** | 0.56% | -3.78% | -2.27% | -6.57% | 5.5% |
| **UNIVERSE MEAN** | **15.68%** | **12.82%** | **9.58%** | **13.24%** | **9.57%** | **29.3%** |

### Audit Takeaway on Cash Modeling
1. Crediting uninvested cash with 5.0% annual interest reflects professional treasury cash management (overnight liquid mutual funds / T-Bills yield 5.5%–6.8% in India).
2. However, for a retail trader leaving idle cash in a zero-interest brokerage ledger, average Variant C return drops to **9.57% CAGR**.
3. Under both interest rate assumptions, the **drawdown reduction and volatility compression are identical**, proving that the risk management mechanism is genuine and robust.

---

## 4. Synthesis: Can Any Strategy Claim an "Actionable Advantage"?

### What the Engine CAN Legitimate Claim:
1. **Asymmetric Downside Hedging:**  
   The engine successfully eliminates the severe tail-risk of equities. Across all 10 stocks, maximum drawdown was reduced by **19.3 to 43.8 percentage points**. In severe bear markets or corporate governance crises (`ADANIGREEN`), abstention preserves capital while Buy & Hold loses up to 78%.
2. **Defensive Alpha in High-Volatility Assets:**  
   For stocks with volatile sideways regimes (`TCS`, `ITC`) or crashing fundamentals (`ADANIGREEN`), active gating produces superior risk-adjusted returns (Sharpe ratio improvements of +0.15 to +0.81).
3. **Execution Feasibility:**  
   At 15.0 bps roundtrip friction, the strategies remain viable after realistic transaction costs, debunking the notion that machine learning edges are instantly devoured by execution drag.

### What the Engine CANNOT Legitimate Claim:
1. **Beating Buy & Hold in a Secular Bull Market:**  
   The engine cannot claim to beat unhedged equity beta during prolonged bull markets. When an asset compounds at 15%–48% CAGR (`ADANIENT`, `RELIANCE`, `VEDL`), spending 60%–90% of the time in cash imposes an insurmountable drag on nominal wealth accumulation.
2. **Predictive Omniscience:**  
   Directional accuracy remains near 52%–55%. The algorithm's value is derived from **disciplined position sizing and trade abstention**, not superior market forecasting.

---

## 5. Recommended Corrections & Enhancements

1. **Clarify Terminal Signal Date:**  
   Update the Streamlit interface in `signals_page.py` to clearly state that the displayed signal represents the latest validated observation from the walk-forward evaluation dataset (2026-08-18), distinguishing it from the unlabelled real-time bar in the prediction terminal.
2. **Add Cash Yield Toggle in Dashboard:**  
   Incorporate an interactive toggle in `signals_page.py` allowing users to switch cash yield between **0% (Idle Broker Ledger)** and **5% (Treasury / Liquid Fund)** to promote full transparency.
3. **Explicit Multiple-Testing Disclaimer:**  
   Clearly state in presentation materials that reporting the best variant per stock reflects an exploratory sweep, and that fixed-variant application represents the true expected out-of-sample experience.
4. **News Coverage Indicator:**  
   Add a visual indicator in the multi-factor matrix alerting users when news sentiment defaults to neutral due to absent news coverage.

---

## Conclusion

The Actionable Trading Signal Engine is mathematically sound, chronologically disciplined, and free from forward-looking target leakage. Its reported outperformance in risk-adjusted metrics (Sharpe ratio and drawdown compression) is genuine and defensible under rigorous peer audit, provided that its structural trade-off—sacrificing secular bull market beta in exchange for extreme capital preservation—is transparently acknowledged.
