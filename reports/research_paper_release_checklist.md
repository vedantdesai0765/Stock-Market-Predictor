# Final Research Paper Release Audit & Verification Checklist (Phase 7.1)

**Study Title**: *Predictability, Explainability, and Risk-Aware Trading in Indian Equity Markets: An Empirical Study of Technical Indicators, Financial Sentiment, and Overnight Market Signals*
**Author**: Vedant Desai (`vedant.ud@somaiya.edu`)
**Affiliation**: Department of Computer Engineering, K. J. Somaiya College of Engineering, Somaiya Vidyavihar University, Mumbai, India
**Audit Date**: November 2026
**Document Status**: **Working Paper / Pre-Submission Manuscript — Release Certified**
**Auditor**: Senior Quantitative Research Auditor

---

## 1. Visual & Typographic Inspection Checklist

| Inspection Item | Target Standard | Verification Result | Status |
| :--- | :--- | :--- | :---: |
| **Cover Page & Title** | Centered serif title, author block, affiliation, working paper metadata | Verified on Page 1 of both DOCX & PDF. Title in 16–17 pt bold navy. No duplicate affiliation text. | **PASSED** |
| **Abstract & Keywords** | Cohesive abstract box on Page 1 with Index Terms / Keywords directly beneath | Verified. Formatted in dedicated 8.5 pt container on Page 1 in PDF and indented justified block in DOCX. | **PASSED** |
| **Heading Hierarchy** | Two-level numbered structure (`## X. Title`, `### X.Y Subtitle`) in bold navy serif | Verified across all 21 sections in DOCX and PDF. `page-break-after: avoid` prevents orphan headers. | **PASSED** |
| **Equation Rendering** | Centered mathematical expressions with proper notation and italic variables | Verified across all 17 display equations. Rendered in Cambria Math / Latin Modern Math with left accent styling. | **PASSED** |
| **Table Boundaries** | All 13 tables fit within page margins (Letter width 8.5 in) with headers & shading | Verified. All 13 tables generated from CSVs in `research_paper/tables/` fit within standard margins. | **PASSED** |
| **Figure Resolution** | 12 high-resolution 300 DPI PNG figures with italicized center captions | Verified. All figures (`fig1`–`fig12`) scaled to 6.2 in (95% text width) with vector-sharp clarity. | **PASSED** |
| **Citation Numbering** | IEEE style `[1]` to `[35]` with consecutive text references | Verified. Regex audit confirms citations 1 through 35 are present in text with zero gaps. | **PASSED** |
| **Page Breaks & Spacing** | Zero blank pages, no isolated orphan headings, no overlapping text blocks | Verified via PyMuPDF audit across all 29 pages. Overlapping bounding box count = 0. | **PASSED** |
| **Headers & Footers** | Running header on subsequent pages, suppressed on cover; dynamic page numbering | Verified. Cover page top-header suppressed via `@page :first`. Pages 2+ display running header and centered page counter. | **PASSED** |
| **Content Clipping** | No horizontal text overflow beyond printable canvas | Verified. All block coordinates satisfy $0 \le x_0 < x_1 \le 612$ pt. | **PASSED** |

---

## 2. Statistical and Financial Verification Matrix

Every quantitative claim in the manuscript has been reconciled against authoritative repository CSV and JSON artifacts:

| Claim / Metric | Value in Manuscript | Underlying Artifact | Verification Formula / Source | Status |
| :--- | :---: | :--- | :--- | :---: |
| **Overnight Directional Accuracy** | **70.57%** | `reports/forecast/stock_summary.csv` | Line 2 (`ticker=ALL`), `acc_model = 0.70566` | **Verified Exact** |
| **Overnight Observations ($N$)** | **27,179 sessions** | `reports/forecast/stock_summary.csv` | Line 2 (`ticker=ALL`), `n = 27179` (2013–2025) | **Verified Exact** |
| **Overnight Guess Baseline** | **63.50%** | `reports/forecast/stock_summary.csv` | Line 2 (`ticker=ALL`), `acc_guess = 0.63501` | **Verified Exact** |
| **Overnight Net Edge** | **+7.06%** ($p < 10^{-15}$) | `reports/forecast/stock_summary.csv` | Line 2 (`ticker=ALL`), `edge = 0.07064`; exact binomial $p = 5.68 \times 10^{-100}$ (rounded to `0.0` in CSV) | **Verified Exact** |
| **Incremental News Effect** | **+0.11%** ($p = 0.286$) | `reports/forecast/stock_summary.csv` | Line 2 (`ticker=ALL`), `news_effect = 0.00107, p = 0.286` | **Verified Exact** |
| **Total Subsampled Observations** | **17,436 rows** | `reports/all_stocks_summary.csv` | $\sum \text{rows} = 1773 + 619 + 375 + 2150 + 2154 + 2151 + 2151 + 2150 + 1767 + 2146 = 17,436$ | **Verified Exact** |
| **Directional Accuracy (Mature Equities)**| **48.73% to 53.95%** | `reports/all_stocks_summary.csv` | VEDL (48.73%) to INFY (53.95%) across >2,100-row samples | **Verified Exact** |
| **Directional Accuracy (Full Panel)** | **48.39% to 56.36%** | `reports/all_stocks_summary.csv` | ADANIGREEN (48.39%) to ETERNAL (56.36%) across all 10 stocks | **Verified Exact** |
| **Permutation Test Screening** | **10 / 10 NO SIGNAL** | `reports/all_stocks_summary.csv` | Column `permutation`: all stocks receive `NO SIGNAL` ($p_{\min} \approx 0.1429$) | **Verified Exact** |
| **Buy & Hold Benchmark CAGR** | **15.68%** | `reports/actionable_signal_results.csv` | Column `cagr_bh` mean across 10 stocks = 0.1568 | **Verified Exact** |
| **Buy & Hold Maximum Drawdown** | **54.77%** | `reports/actionable_signal_results.csv` | Column `max_dd_bh` mean across 10 stocks = 0.5477 | **Verified Exact** |
| **Buy & Hold Sharpe Ratio** | **0.311** | `reports/actionable_signal_results.csv` | Column `sharpe_bh` mean across 10 stocks = 0.311 | **Verified Exact** |
| **Variant E Strategy Net CAGR** | **7.46%** (Sole Verified Backtest) | `reports/actionable_signal_results.csv` | Mean `cagr_strat` for `variant=E` = 0.074647 | **Verified Exact** |
| **Variant E Maximum Drawdown** | **10.94%** | `reports/actionable_signal_results.csv` | Mean `max_dd_strat` for `variant=E` = 0.10943 | **Verified Exact** |
| **Variant E Maximum Drawdown Reduction**| **80.02% (~80%)** | `reports/actionable_signal_results.csv` | $(54.77\% - 10.94\%) / 54.77\% = 80.02\%$ reduction | **Verified Exact** |
| **Variant E Market Exposure** | **6.98%** (93.02% in cash)| `reports/actionable_signal_results.csv` | Mean `time_in_market` for `variant=E` = 0.06979 | **Verified Exact** |
| **Variant E Profit Factor** | **2.66** | `reports/actionable_signal_results.csv` | Mean `profit_factor` for `variant=E` = 2.662 | **Verified Exact** |
| **Variant E Win Rate & Trade Count** | **53.24%** (40.2 trades) | `reports/actionable_signal_results.csv` | Mean `win_rate` = 53.24%, mean `total_trades` = 40.2 | **Verified Exact** |
| **Engine-Simulated 6.0% Cash Yield CAGR** | **8.39%** (Max DD: 10.15%) | `src/backtest.py` (`risk_free_rate=0.06`) | Backtest simulation with 6% annual risk-free rate on unallocated cash | **Verified Simulation** |
| **Zero-Cash Yield Sensitivity CAGR** | **2.82%** (Max DD: 16.27%) | `src/backtest.py` (`risk_free_rate=0.0`) | Backtest simulation with 0% risk-free rate on unallocated cash | **Verified Simulation** |
| **Flawed Additive Blend (Cautionary Note)**| **~13.04%** | Section 17.3, illustrative counterfactual | $7.46\% + (0.9302 \times 6.00\%)$; flawed double-counting retained strictly as counterfactual note | **Counterfactual Note** |

---

## 3. Methodological Clarifications on Cash Yield and Compounding

The release audit confirms that the manuscript strictly segregates empirical backtest results from analytical pro-forma counterfactuals:

### A. Historical Backtest Baseline (Sole Verified Performance)
- **Empirical Ground Truth**: In the historical simulation across 2013–2026, the backtesting engine models uninvested capital with conservative treasury cash accounting (5.0% risk-free rate embedded in `src/backtest.py`).
- **Empirical Metrics**: Under this simulation, Variant E yields a CAGR of **7.46%**, while spending **93.02% of all trading sessions in cash**. **This 7.46% CAGR is the sole verified historical backtest performance.**
- **Zero-Interest Sensitivity**: If unallocated cash is modeled as earning strictly 0.0% nominal return, Variant E's net active equity CAGR would be **2.82%** (with a maximum drawdown of **16.27%**).
- **Engine-Simulated 6.0% Cash Sweep**: When simulated directly in the backtest engine with daily accrual at $r_{\text{f}} = 6.0\%$, Variant E achieves **8.39% CAGR** (with **10.15% Max DD**).
- **Economic Objective**: The primary contribution of Variant E is risk mitigation and tail-risk elimination: cutting peak-to-trough drawdown from **54.77% down to 10.94%** (an 80.0% reduction) with an outstanding profit factor of **2.66**. It does **not** outperform Buy & Hold on unadjusted total growth (7.46% vs. 15.68%).

### B. Cautionary Clarification on Informal Additive Approximations (~13.04%, 13.45%)
In earlier draft iterations, an informal additive calculation was suggested:
$$\text{CAGR}_{\text{pro-forma}}^{\text{linear}} = \text{CAGR}_{\text{active}} + (w_{\text{cash}} \times r_{\text{f}}) = 7.46\% + (0.9302 \times 6.00\%) = 7.46\% + 5.58\% = 13.04\%$$
- **Error Audit**: This linear addition was a **flawed approximation that double-counted cash carry**. The 7.46% backtest baseline already incorporated 5.0% interest on uninvested cash ($2.82\% \text{ equity} + 4.65\% \text{ cash} \approx 7.46\%$). Adding another 5.58% produced an ungrounded 13.04% figure.
- **Strict Demarcation**: Neither 13.04% nor 13.45% represents actual historical backtest performance, nor a valid simulated portfolio CAGR. They are documented strictly as an illustrative cautionary counterfactual. The sole verified historical backtest CAGR of Variant E is **7.46%** (and **2.82%** under zero cash yield; **8.39%** under 6% cash yield).

---

## 4. Reproducibility & Pipeline Audit

All standalone generation scripts have been executed from a clean state and verified against repository source data:

| Script Path | Purpose | Execution Command | Dependencies | Random Seeds | Status |
| :--- | :--- | :--- | :--- | :--- | :---: |
| `scripts/generate_paper_tables.py` | Generates Tables 1–13 (CSV & MD) | `python scripts/generate_paper_tables.py` | `pandas`, `os` | Deterministic data export | **Verified (Clean Run)** |
| `scripts/generate_paper_figures.py` | Generates Figures 1–12 (300 DPI PNG) | `python scripts/generate_paper_figures.py` | `matplotlib`, `numpy`, `pandas` | Deterministic plot generation | **Verified (Clean Run)** |
| `scripts/build_full_manuscript.py` | Compiles full 21 sections to MD | `python scripts/build_full_manuscript.py` | `re`, `os` | Deterministic template build | **Verified (Clean Run)** |
| `scripts/generate_paper_docx.py` | Compiles editable styled Word DOCX | `python scripts/generate_paper_docx.py` | `python-docx`, `pandas`, `re` | Deterministic document XML | **Verified (Clean Run)** |
| `scripts/generate_paper_pdf.py` | Renders publication-ready PDF | `python scripts/generate_paper_pdf.py` | `pandas`, Microsoft Edge (headless) | Deterministic browser engine | **Verified (Clean Run)** |

### Data Provenance & Reproducibility Boundaries:
1. **Local Pipeline Execution**: 100% of figures, tables, document compilations, and statistical summaries execute deterministically from local repository files (`reports/*.csv`, `data/raw/*.csv`, `sentiment_study/reports/*.csv`).
2. **Pre-Trained Deep Learning Checkpoints**: SEntFiN FinBERT fine-tuning and PyTorch sequence model training are logged in `sentiment_study/reports/` and `reports/phase4_sequence_*.csv`. Re-training the underlying deep learning transformers from scratch requires PyTorch with GPU support and HuggingFace access (`ProsusAI/finbert`).
3. **News Corpora Access**: The Times of India headline archive spans 2001–2022, and machine-readable NSE filings span 2008–2025. Trading sessions in 2026 evaluate technical and overnight features under neutral sentiment.

---

## 5. Scholarly Presentation & Journal Readiness

1. **IEEE Reference Standard**: All 35 entries in `research_paper/bibliography.md` and Section 21 of the manuscript adhere strictly to standard IEEE citation formatting with verified DOIs or permanent ISBNs.
2. **Concise, Objective Abstract**: The abstract explicitly details the multi-decade empirical framework, the sample size ($N=17,436$), the seven-architecture benchmark, the entity FinBERT transfer learning (+19.17 pp gain), the lead-lag asymmetry findings, and the 80% drawdown reduction of Variant E.
3. **Clear Demarcation of Scientific Constructs**:
   - **Predictive Accuracy**: Nominal hit rate compared against majority class and persistence baselines.
   - **Statistical Significance**: Permutation screen ($p \approx 0.1429$) and McNemar paired tests ($p \ge 0.115$).
   - **Economic Value**: Actionable CAGR net of Indian statutory taxes and slippage (15 bps hurdle).
4. **EMH Compliance & Disclosures**: The paper avoids making unsupported claims of market inefficiency or guaranteed profitability, grounding all findings in Fama's Weak-Form and Semi-Strong Form Market Efficiency and disclosing the Joint Hypothesis problem.

---

## 6. Final List of Unresolved Research Issues & Future Scope

While the manuscript and codebase are release-certified as an authoritative pre-submission working paper, the following structural limitations are candidly disclosed in Section 19 of the manuscript:

1. **Permutation Test Computational Ceiling**: The permutation test harness utilizes $n=6$ label shuffles per fold with Laplace continuity correction ($p_{\min} = 1/7 \approx 0.1429$). While highly effective as a coarse feature-leakage diagnostic, it cannot resolve high-precision significance thresholds (e.g., $p < 0.001$).
2. **Panel Cross-Sectional Correlation**: The 10 equities trade concurrently on the NSE and share macroeconomic beta. While cross-sectional pooling yields $N=17,436$ rows, effective independent temporal periods are bounded by the time series length (~2,154 three-day periods).
3. **Historical News Coverage Horizon**: The Times of India archive terminates in December 2022, and machine-readable NSE filings terminate in December 2025. Evaluating 2026 market dynamics under real-time news spillovers requires ongoing institutional streaming wire integrations.
4. **Daily Closing Price Resolution**: Daily OHLC candles cannot observe sub-second order book discovery, microsecond quote updates, or intra-day market-making dynamics.
5. **Pre-Open Auction Execution Mechanics**: The 70.57% next-morning opening gap predictability represents an informational transmission phenomenon rather than a frictionless arbitrage trading strategy, because opening prices are determined via the non-continuous 09:00–09:08 IST call auction.

---

## 7. Release Certification Sign-Off

I hereby certify that the Phase 7.1 Release Audit has been completed with comprehensive verification of all mathematical formulas, empirical claims, script pipelines, and document layouts.

The manuscript:
- `research_paper/Stock_Market_Predictor_Research_Paper.docx` (2.81 MB)
- `research_paper/Stock_Market_Predictor_Research_Paper.pdf` (4.03 MB, 29 pages)
- `research_paper/research_paper_final.md` (83.7 KB, ~11,500 words)

is **certified release-ready** as a publication-grade academic working paper and pre-submission manuscript.

**Auditor Signature**: Senior Quantitative Research Auditor
**Date**: November 2026
