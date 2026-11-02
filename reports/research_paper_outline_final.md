# Complete Research Paper Outline and Planning Specification (Final Audited Version)

**Project**: Stock Market Predictor — Multi-Decade Technical Predictability, Entity-Level Financial NLP, and Market Efficiency  
**Target Output**: Comprehensive Academic Research Paper  
**Authoritative Results Source**: [`reports/research_paper_results_final.md`](file:///C:/Users/vedan/Stock-Market-Predictor/reports/research_paper_results_final.md)  
**Date**: November 2026  
**Status**: Final Audited Paper Blueprint  

---

## Executive Summary & Paper Architecture

This document establishes the definitive, audited architectural blueprint for the research paper. It incorporates the complete results from [`reports/research_paper_results_final.md`](file:///C:/Users/vedan/Stock-Market-Predictor/reports/research_paper_results_final.md), applies strict econometric qualifications regarding sample independence, stationarity, and market efficiency, and provides an audited chain of custody linking every claim to concrete project artifacts.

```
┌──────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                   PAPER ARCHITECTURE & FLOW                                     │
└──────────────────────────────────────────────────────────────────────────────────────────────────┘
                                                 │
          ┌──────────────────────────────────────┴──────────────────────────────────────┐
          ▼                                                                            ▼
  PART I: THE TECHNICAL FOUNDATION                             PART II: THE SENTIMENT EXPANSION
  - 10 NSE Equities (2000–2026, 17,636 subsampled rows)        - News Corpora: ToI (2001–22), NSE (2008–25)
  - Scale-Free Dimensionless Indicators (52 features)          - Entity FinBERT ([CLS] Co [SEP] Head [SEP])
  - Expanding Walk-Forward CV (5 Folds, 17-bar / 52d Embargo)  - Gold Standard: SEntFiN 1.0 (14,404 labels)
  - Seven-Model Benchmark (Linear Regularized to Seq DL)       - Paired Ablation: 20 Equity-Corpus Setups
          │                                                                            │
          ▼                                                                            ▼
  EMPIRICAL FINDING 1:                                         EMPIRICAL FINDING 2:
  - Multi-decade technical edge collapses                      - Entity NLP improves accuracy to 87.7%
  - 10/10 stocks: NO SIGNAL vs. Honest Majority Base           - Incremental next-day edge: Δ = +0.09% (p > 0.11)
  - Architecture invariance (< 1.32 pp spread under T+G)       - 0/20 stocks achieve statistical significance
  - Sequence DL models suffer variance shrinkage (< 0.19)      - News explains today (ρ = +0.056); dead tomorrow
          │                                                                            │
          └──────────────────────────────────────┬─────────────────────────────────────┘
                                                 ▼
                              PART III: SYNTHESIS & DEMYSTIFICATION
                              - Controlled Reproduction of the "Three Accuracy Illusions" (77%, 86%, 99%)
                              - Strict Alignment with Weak-Form & Semi-Strong Form EMH (Fama, 1970)
                              - Fama Joint Hypothesis Caveats & Absence of "EMH Proof" Claims
                              - Practical Implications for Algorithmic Trading & Decision-Support Systems
```

---

## 1. Proposed Title and Abstract Direction

### 1.1. Recommended Final Title
**Predictability, Attribution, and Market Efficiency: An Empirical Investigation of Technical Indicators and Entity-Targeted Financial Sentiment in Indian Equities**

*Rationale*: This title precisely captures the tripartite structure of the study: (1) multi-decade empirical testing of technical indicators, (2) fine-tuning of sentence-pair language models for entity-level polarity attribution, and (3) econometric evaluation of market efficiency on the National Stock Exchange of India (NSE).

### 1.2. Structured Abstract (285 words)

> Contemporary financial machine learning literature frequently reports directional classification accuracies exceeding 70% to 90%. However, such findings often reflect methodological vulnerabilities, including lookahead leakage, target autocorrelation, and unanchored baselines. This study presents a rigorous empirical investigation into the out-of-sample predictability of Indian equities across full market cycles. Evaluating 26 years of daily trading data (January 2000 to August 2026) across ten major National Stock Exchange (NSE) listings ($N=17,636$ subsampled, non-overlapping 3-day holding periods), we implement an expanding 5-fold walk-forward validation framework with strict train-test embargo buffers and scale-free, dimensionless indicator transformations. Benchmarking seven architectures—spanning $L_2$-regularized Logistic Regression, decision-tree ensembles (Random Forest, GBDT, XGBoost, LightGBM), and recurrent neural networks (PyTorch LSTM and GRU)—we find that all models converge within a narrow 1.32 percentage point spread under inter-market features, with zero equities demonstrating an edge distinguishable from random label permutations (`NO SIGNAL`). Recurrent sequence models exhibit severe variance shrinkage ($\sigma(\hat{r})/\sigma(r) \le 0.19$) and fail to outperform a trivial zero-return benchmark. To evaluate unstructured text spillovers, we fine-tune an entity-targeted FinBERT classifier on the SEntFiN 1.0 benchmark, achieving 87.71% accuracy (+19.17 pp over generic FinBERT) and resolving multi-firm attribution ambiguity. However, in paired walk-forward ablations across 12,426 Times of India headlines and 11,740 regulatory filings, incorporating company sentiment yields an unweighted mean accuracy delta of $+0.09\%$ ($+0.28\%$ weighted), with zero of twenty equity-corpus configurations achieving statistical significance ($p \ge 0.115$). Lead-lag cross-correlations demonstrate a significant contemporaneous association ($\rho = +0.0562, p < 10^{-4}$) that collapses to noise on forward trading sessions ($\rho = +0.0070, p = 0.618$). Our findings are consistent with Weak-Form and Semi-Strong Form Market Efficiency at daily horizons.

---

## 2. Keywords

`Financial Machine Learning`, `Efficient Market Hypothesis`, `Entity-Targeted Sentiment Analysis`, `FinBERT`, `Walk-Forward Validation`, `Lead-Lag Asymmetry`, `Technical Indicators`, `Emerging Markets`, `National Stock Exchange of India`, `Data Leakage`.

---

## 3. Complete 13-Section Paper Outline

### Section 1: Title, Authors, and Affiliations
- **Title**: *Predictability, Attribution, and Market Efficiency: An Empirical Investigation of Technical Indicators and Entity-Targeted Financial Sentiment in Indian Equities*
- **Authors**: [Author Names to be confirmed by user, e.g., Vedant et al.]
- **Affiliations**: [Department / Institution / Laboratory]
- **Target Venues**: *Journal of Financial Data Science*, *Quantitative Finance*, *Expert Systems with Applications*, or *ACM Transactions on Management Information Systems*.

### Section 2: Abstract
- Structured 285-word synthesis (as detailed in Section 1.2).

### Section 3: Keywords
- Standardized academic indexing terms (as listed in Section 2).

### Section 4: Introduction
- **4.1 Context and Motivation**: Algorithmic price discovery and retail participation in emerging markets, focusing on the National Stock Exchange of India (NSE).
- **4.2 The "Replication Crisis" in Financial ML**: High nominal accuracies reported in literature; the critical gap between theoretical papers and real-world execution.
- **4.3 The Entity Attribution Bottleneck**: The structural inadequacy of document-level sentiment models in multi-company news environments.
- **4.4 Disciplinary Guardrails and Methodology**: Eliminating selection bias via 26-year multi-decade data expansion, non-overlapping labels, scale-free indicators, and walk-forward CV.
- **4.5 Summary of Seven Key Contributions**:
  1. *Multi-Decade Walk-Forward Evaluation*: 17,636 subsampled observations across 10 NSE equities (2000–2026), demonstrating edge collapse over full market cycles.
  2. *Architecture Invariance in Low-SNR Regimes*: Empirical parity across linear models, tree ensembles, and deep neural nets (spread $< 1.32$ pp).
  3. *Recurrent Sequence Failure Diagnostics*: Quantifying variance shrinkage and failure to beat zero-return benchmarks in LSTM/GRU regressors.
  4. *Entity-Targeted Financial NLP*: Sentence-pair FinBERT yielding 87.7% accuracy on SEntFiN 1.0 (+19.2 pp over generic FinBERT).
  5. *Paired Econometric Sentiment Ablation*: 20 equity-corpus ablations proving company news provides no statistically significant out-of-sample forward edge.
  6. *Characterization of Lead-Lag Asymmetry*: Resolving the news paradox—news contemporaneously explains today's price moves ($\rho = +0.0562, p < 10^{-4}$), but fails to forecast tomorrow ($\rho = +0.0070, p = 0.618$).
  7. *Controlled Deconstruction of Literature High-Accuracy Traps*: Controlled replication isolating the exact mechanisms generating 77%, 86%, and 99% accuracy illusions.

### Section 5: Literature Review
- **5.1 The Efficient Market Hypothesis and Price Predictability**:
  - Foundations: Fama (1970) Weak, Semi-Strong, and Strong form definitions.
  - The Joint Hypothesis Problem (Fama, 1970, 1991): Why no empirical test can definitively "prove" market efficiency.
  - Data-snooping and selection luck across sample horizons: White (2000), Hansen (2005), Timmermann & Granger (2004).
- **5.2 Financial NLP and Sentiment Extraction**:
  - Domain lexicons: Loughran & McDonald (2011).
  - Pre-trained transformer language models: FinBERT (Araci, 2019; Yang, Uy, & Huang, 2021).
  - Limitations of whole-document polarity aggregation in financial journalism.
- **5.3 Entity-Level and Aspect-Based Sentiment in Financial Text**:
  - The SEntFiN 1.0 gold-standard corpus: Sinha, Kedas, Kumar, & Malo (2022).
  - Sentence-pair classification for directed sentiment extraction.
- **5.4 Methodological Traps and Evaluation Protocols in Financial Machine Learning**:
  - Overlapping labels, lookahead leakage, and temporal cross-validation: Marcos López de Prado (2018).
  - Panel data cross-sectional dependence and standard error clustering: Petersen (2009), Cameron & Miller (2015).
  - Transaction costs and microstructure frictions in Indian equity trading.

### Section 6: Problem Statement and Research Objectives
- **6.1 Mathematical Formulation of the Prediction Task**:
  - Let $P_{i,t}$ denote the daily adjusted closing price of equity $i$ on trading day $t$.
  - Discrete forward return: $R_{i, t, t+h} = \frac{P_{i, t+h} - P_{i,t}}{P_{i,t}}$.
  - Directional binary target: $y_{i,t}^{(h)} = \mathbb{I}(R_{i, t, t+h} > 0) \in \{0, 1\}$.
  - Non-overlapping subsampling constraint: Evaluating on sub-grid $\{t_k\}_{k=1}^N$ where $t_{k+1} - t_k \ge h$ (specifically $h=3, \text{step}=3$).
- **6.2 Seven Formal Research Objectives (RO1 to RO7)**:
  - **RO1 (Multi-Decade Technical Predictability)**: Test whether stationary technical price indicators provide an out-of-sample directional edge over honest baselines across full market cycles (2000–2026).
  - **RO2 (Architecture Complexity vs. Parity)**: Benchmark seven model families under identical walk-forward splits to test whether non-linear or deep models outperform convex regularized linear models in low-SNR equity data.
  - **RO3 (Sequential Deep Learning & Autoregressive Memory)**: Determine whether recurrent neural networks (LSTM, GRU) learn non-trivial conditional return distributions or succumb to variance shrinkage and fail zero-return benchmarks.
  - **RO4 (Entity-Level Sentiment Disambiguation)**: Evaluate whether formulating FinBERT as a sentence-pair classifier resolves multi-company polarity attribution ambiguity.
  - **RO5 (Incremental Predictive Value of News Sentiment)**: Conduct paired econometric ablation to evaluate whether incorporating entity-targeted news sentiment produces a statistically significant next-day predictive edge.
  - **RO6 (Lead-Lag Asymmetry: Contemporaneous Explanation vs. Forward Prediction)**: Quantify the temporal lead-lag structure between news sentiment and equity returns to determine whether financial news functions as an explanation or a forecast.
  - **RO7 (Deconstruction of Literature High-Accuracy Claims)**: Empirically reproduce and isolate the methodological flaws (lookahead, autocorrelation, unit-root regression) responsible for published claims of 70%–90% accuracy.

### Section 7: Dataset Description
- **7.1 Historical Equity Price Panel**:
  - 10 diverse NSE equities: ADANIENT, ADANIGREEN, ETERNAL (Zomato), HDFCBANK, INFY, ITC, RELIANCE, SBI, TCS, VEDL.
  - Exact historical date ranges from raw files and metadata:
    - *Mature Listings* (HDFCBANK, INFY, ITC, RELIANCE, SBI, VEDL): Raw data from 2000-01-03 to 2026-08-28/31 ($>6,588$ daily sessions). Feature modeling frames span 2000-06-12 to 2026-08-25 (accounting for 60-day indicator burn-in and forward targets).
    - *TCS*: Raw data from 2004-08-25 to 2026-08-28 (5,426 rows); modeling frame spans 2005-02-02 to 2026-08-25.
    - *ADANIENT*: Raw data from 2004-07-01 to 2026-08-28 (5,465 rows); modeling frame spans 2004-12-07 to 2026-08-25.
    - *ADANIGREEN*: Raw data from IPO 2018-06-18 to 2026-08-28 (2,023 rows); modeling frame spans 2018-11-28 to 2026-08-25.
    - *ETERNAL*: Raw data from IPO 2021-07-23 to 2026-08-28 (1,257 rows); modeling frame spans 2021-12-30 to 2026-08-25.
  - Data hygiene: Split/dividend adjustment, corporate restructuring audits (2015 Adani demergers, 2026 Vedanta restructuring), elimination of non-trading holiday rows.
  - Subsampled panel total: 17,636 non-overlapping 3-day holding periods.
- **7.2 Macroeconomic and Inter-Market Features**:
  - Domestic index return (Nifty 50), volatility index (India VIX), overnight US closing returns (S&P 500), and morning Asian market opens (Nikkei 225, KOSPI).
- **7.3 Unstructured Financial News and Regulatory Disclosure Corpora**:
  - *Times of India (ToI) Historical Archive (2001–2022)*: 12,426 company-headline pairs matched to project equities.
  - *NSE Corporate Regulatory Filings (2008–2025)*: 11,740 machine-parsed regulatory announcements and disclosures stamped with exact dissemination times.
  - *New York Times (NYT) Global News Archive (1990–2025)*: International macro and market sentiment context.
- **7.4 Gold-Standard Benchmark (SEntFiN 1.0)**:
  - 14,404 entity-headline annotations across 10,753 Economic Times headlines, partitioned 80/10/10 by headline hash to prevent data leakage.

### Section 8: Methodology
- **8.1 Data Preprocessing, Cleaning, and Subsampling**:
  - Corporate action normalization.
  - Step=3 subsampling protocol for $h=3$ to eliminate moving-average target autocorrelation ($\rho = 0.806 \to -0.044$).
  - Subsampling offset analysis: Evaluating phase sensitivity across offsets $k \in \{0, 1, 2\}$.
- **8.2 Feature Engineering (52 Scale-Free Dimensionless Indicators)**:
  - Removal of nominal price levels ($P_t, \text{SMA}_k$) that destroy cross-asset transferability.
  - Percentage deviations: $\Delta_{\text{SMA20}} = \frac{P_t - \text{SMA}_{20}(t)}{\text{SMA}_{20}(t)}$, $\Delta_{\text{SMA50}} = \frac{P_t - \text{SMA}_{50}(t)}{\text{SMA}_{50}(t)}$.
  - Moving average interactions: $\frac{\text{SMA}_{20}(t) - \text{SMA}_{50}(t)}{\text{SMA}_{50}(t)}$.
  - Leakage-free trailing rolling z-scores (window = 60 days): $z_t = \frac{x_t - \mu_{t-1}}{\sigma_{t-1}}$.
  - Edge-safe Relative Strength Index (RSI): Smooth boundary handling preventing division by zero on consecutive gain streaks (0, 50, 100 limit conditions).
  - Normalized MACD: $\text{MACD}_{\text{norm}} = \frac{\text{MACD}_t}{P_t}$.
  - Normalized Ichimoku cloud indicators: Normalized Tenkan/Kijun spreads, Senkou Span ratios, Cloud thickness.
  - Normalized Bollinger Bands: Percent $B = \frac{P_t - \text{Lower}_t}{\text{Upper}_t - \text{Lower}_t}$ and bandwidth $\frac{\text{Upper}_t - \text{Lower}_t}{\text{SMA}_{20}(t)}$.
  - Log-return volatility ratios across trailing windows (5, 10, 20, 60 days).
  - *Methodological Note on Stationarity*: Clarify that features are constructed as dimensionless, scale-free transformations to eliminate price-level drift, but covariance stationarity is not claimed to be formally proven via unit-root hypothesis tests.
- **8.3 Walk-Forward Validation Harness, Embargo, and Baselines**:
  - Expanding-window walk-forward cross-validation across 5 folds.
  - *Audited Embargo Mechanism*: In daily space, `embargo = 52`; in subsampled space ($h=3, \text{step}=3$), `embargo = 17` rows ($17 \times 3 = 51 \approx 52$ trading days) dropped from the *end of each training fold* before the test fold starts.
    - *What it protects against*: (1) **Purging**: Completely purges the forward target horizon $h=3$, preventing forward label overlap into the test period. (2) **Autocorrelation buffer**: Functions as an econometric decorrelation gap preventing the model from exploiting short-term serial autocorrelation or transient error persistence across the train/test seam.
    - *Clarification on Lookback*: Trailing indicator lookbacks (e.g., 52 bars) into past history do not leak future test labels in walk-forward evaluation because past prices are legitimately available in live execution.
  - Baselines: Training Majority Class baseline, Naive Persistence baseline ($\hat{y}_{t+1} = y_t$), and Zero-Return benchmark ($\hat{r}_{t+1} = 0$).
- **8.4 Seven-Model Benchmarking Suite**:
  - $L_2$-regularized Logistic Regression ($C=0.1$).
  - Random Forest (500 estimators, $\text{min\_samples\_leaf}=20$, $\sqrt{p}$ subsampling).
  - Gradient Boosted Decision Trees: HistGradientBoosting ($\text{max\_leaf\_nodes}=15, \eta=0.05$), XGBoost (depth=4, $\lambda=1.0$), LightGBM (GOSS).
  - Recurrent Neural Networks: PyTorch 2-layer stacked LSTM and GRU (hidden dim = 64, recurrent dropout $p=0.2$, 30-day sequential lookbacks).
- **8.5 Deep Learning Sequence Architecture and Diagnostics**:
  - Continuous return regression ($h=3$) on TCS and RELIANCE.
  - Evaluation against Zero-Return benchmark RMSE ($\hat{r}=0$) and Persistence RMSE.
  - Variance shrinkage metric: $\text{Shrinkage} = \frac{\sigma(\hat{r})}{\sigma(r)}$.
- **8.6 Entity-Targeted FinBERT Architecture**:
  - Sentence-pair classification:
    $$\text{Input} = \text{[CLS]} \circ \text{Target Company} \circ \text{[SEP]} \circ \text{News Headline} \circ \text{[SEP]}$$
  - Fine-tuning: Pre-trained ProsusAI/finbert; AdamW ($\text{lr} = 3 \times 10^{-5}$, weight decay 0.01); 3 epochs on SEntFiN 1.0.
  - Benchmark evaluation against standard generic FinBERT.
- **8.7 Econometric Sentiment Ablation Framework**:
  - Paired comparison: Baseline $B$ vs. Augmented $B + C_{\text{ent}}$ across 20 equity-corpus conditions.
  - Paired McNemar test with continuity correction on discordant predictions $(b, c)$:
    $$\chi^2 = \frac{(|b - c| - 1)^2}{b + c}$$
  - Placebo control: Temporally shifting sentiment by $+1$ calendar year.
- **8.8 News-Price Relationship Analysis ("Why Did It Move?" Framework)**:
  - Regex filtering of circular price-reporting headlines (`"down X%"`, `"drops"`, `"surges"`).
  - Spearman rank correlation across lags $-1, 0, +1$.
  - Sentiment quintile sorting ($Q_1$ to $Q_5$) and daily return spread estimation.
  - Explanatory coverage analysis of extreme market moves ($|\Delta| > 1.5\%$) across 4 categories: Matching, Opposite, Neutral, No News.
- **8.9 Statistical Diagnostic Limitations and Power Constraints**:
  - Permutation test limitation: $n=6$ label shuffles per fold with Laplace correction:
    $$p_{\min} = \frac{0 + 1}{6 + 1} = \frac{1}{7} \approx 0.1429$$
  - Coarse diagnostic function: screen against feature leakage rather than high-resolution hypothesis test.
  - Panel dependence qualification: Concurrent trading on NSE induces cross-sectional correlation; effective degrees of freedom governed by time series length, not pooled rows.
  - Statutory transaction cost modeling: STT (0.10%), exchange fees, GST, SEBI turnover fees, and bid-ask slippage.

### Section 9: Results and Discussion
*(Adopts 100% of the authoritative content and tables from `reports/research_paper_results_final.md`)*
- **9.1 Multi-Decade Walk-Forward Directional Prediction Performance Across 10 Equities** (Table 1: 17,636 rows, all 10 stocks `NO SIGNAL`, dissipation of truncated-window apparent edges).
- **9.2 Baseline Benchmarking and the Three Accuracy Illusions** (Table 2 & ASCII Taxonomy: Majority Class Illusion, Autocorrelation Trap, Contemporaneous Look-Ahead Mirage).
- **9.3 Model Architecture Comparison on Next-Day Forecasting** (Table 3: 7 models, 1.32 pp spread under $T+G$, 2.06 pp spread under $T$, offset robustness of Logistic Regression).
- **9.4 Deep Learning Sequence Models: LSTM and GRU Diagnostics** (Table 4: continuous regression failure on TCS and RELIANCE, `beats_zero = False`, severe variance shrinkage to 9.6%–19.3%).
- **9.5 Entity-Targeted FinBERT Evaluation on SEntFiN Benchmark** (Table 5: +19.17 pp overall, +23.78 pp on multi-entity headlines).
- **9.6 Company-Targeted Sentiment Ablation Across 10 Equities** (Table 6: 20 equity-corpus conditions, unweighted mean delta $+0.09\%$, sample-weighted mean $+0.28\%$, 0/20 stocks significant under McNemar test, $p \ge 0.115$).
- **9.7 News-Price Relationship and the Explanatory Power of Media** (Tables 7, 8, 9: lead-lag asymmetry, $\rho = +0.0562 \to +0.0070$, quintile spread $+0.20\% \to -0.035\%$, big move coverage showing 52%–97% unexplained single-stock volatility).
- **9.8 Controlled Demonstration of Literature Accuracy Pitfalls** (Table 10: empirical reproduction of 77% headline illusion, 86% moving average illusion, 99% price-level $R^2$ illusion).
- **9.9 Synthesis of Key Findings** (Seven empirical takeaways).

### Section 10: Practical Implications
- **10.1 For Quantitative Traders and Portfolio Managers**:
  - Hurdle rates: Transaction costs ($0.25\%–0.35\%$ round-trip) render gross technical edges unviable.
  - Perils of model complexity: Regularized convex linear models provide greater parameter stability than tree ensembles or neural nets in low-SNR environments.
- **10.2 For Financial NLP Researchers and System Architects**:
  - The absolute necessity of entity-level disambiguation in financial NLP pipelines.
  - The critical peril of circular price-reporting headlines in financial sentiment research.
- **10.3 For Market Regulators and Financial Surveillance**:
  - Empirical confirmation of rapid information absorption on the National Stock Exchange of India.
  - Public filings dissemination efficiency.
- **10.4 For Dashboard and Decision-Support Systems (Illustrative Software Architecture)**:
  - User interface design that strictly segregates **retrospective explanation** from **forward prediction** (e.g., the Streamlit "Why did it move?" interface).
  - Implementation of automated confidence-based abstention thresholds to filter uncertain market regimes.
  - *Clear Visual Demarcation*: Explicitly present UI screenshots as software implementation outputs, not statistical evidence.

### Section 11: Limitations and Future Work
- **11.1 Daily Frequency and Latency Truncation**: Public daily closes vs. millisecond algorithmic execution.
- **11.2 Public News Archives vs. Machine-Readable Wires**: Journalistic print cycles vs. direct Bloomberg/Reuters ultra-low-latency feeds.
- **11.3 Hypothesis Testing Power**: Computational bounding of the $n=6$ permutation screen ($p_{\min} = 1/7$); outlining requirements for distributed HPC permutation testing ($n=1,000$).
- **11.4 Promising Research Trajectories**: Multimodal earnings call audio; inter-firm supply chain dependency graphs; high-frequency auction order books.

### Section 12: Conclusion
- **12.1 Concluding Synthesis**: Final appraisal of market efficiency in Indian equities; affirmation that stationary technical patterns and public news headlines do not confer a reliable directional edge at daily horizons once rigorous evaluation standards are enforced.
- **12.2 Methodological Mandate**: A call for scientific integrity, leakage auditing, baseline anchoring, and sample window expansion in financial AI.

### Section 13: References
- Complete, verified bibliographic citations with DOIs (as cataloged in Section 13).

---

## 4. Section-by-Section Source Mapping Table

| Paper Section & Subsection | Source Code / Notebooks | Input Data / Raw Corpora | Result Reports / Artifacts | Figures & Visual Assets |
| :--- | :--- | :--- | :--- | :--- |
| **Section 4: Introduction** | `src/config.py`, `src/pipeline.py` | `data/raw/*.csv` | `reports/all_stocks_summary.csv` | `reports/figures/walk_forward_folds.png` |
| **Section 5: Literature Review** | N/A (Academic literature) | SEntFiN 1.0 dataset | N/A | N/A |
| **Section 6: Problem Statement & ROs** | `src/targets.py`, `src/config.py` | `data/raw/*.csv` | `reports/phase3_results.csv` | `reports/figures/target_autocorrelation.png` |
| **Section 7.1: Equity Price Panel** | `scripts/extend_history_2000.py`, `scripts/fix_raw_data.py`, `src/collect.py` | `data/raw/*_raw.csv` (10 stocks) | `models/*/metadata.json` | `reports/figures/return_distribution.png` |
| **Section 7.2: Inter-Market Features** | `sentiment_study/src/news_prep.py` | Yahoo Finance (S&P, Nikkei, VIX) | `data/processed/*_dataset.csv` | N/A |
| **Section 7.3: News & Filings Corpora** | `sentiment_study/src/news_prep.py`, `src/config.py` | `data/raw/india_news/`, `data/raw/sentfin/` | `data/processed/*_company_sentiment.csv` | N/A |
| **Section 7.4: SEntFiN Benchmark** | `sentiment_study/src/train_entity_model.py` | `data/raw/sentfin/SEntFiN.csv` | `sentiment_study/reports/entity_model_eval.json` | N/A |
| **Section 8.1: Subsampling & Offsets** | `src/targets.py`, `notebooks/01_data_collection.ipynb` | `data/raw/*.csv` | `reports/phase4_offset_sensitivity.csv` | `reports/figures/phase4_offset_sensitivity.png` |
| **Section 8.2: Scale-Free Indicators** | `src/features.py`, `src/stationary.py`, `src/ichimoku.py`, `src/bollinger.py` | `data/raw/*.csv` | `reports/phase2_results.csv` | `reports/figures/TCS_feature_set_comparison.png` |
| **Section 8.3: Walk-Forward & Embargo** | `src/evaluation.py`, `src/pipeline.py` | `data/processed/*_dataset.csv` | `reports/phase2_window_stability.csv` | `reports/figures/walk_forward_folds.png` |
| **Section 8.4: Seven-Model Suite** | `src/modeling.py`, `sentiment_study/src/run_all.py` | `data/processed/*_dataset.csv` | `sentiment_study/reports/comparison_all.csv` | N/A |
| **Section 8.5: LSTM/GRU Diagnostics** | `src/sequence_models.py` | `data/processed/*_dataset.csv` | `reports/phase4_sequence_*.csv`, `comparison_dl.csv` | `reports/figures/phase4_sequence_models.png` |
| **Section 8.6: Entity FinBERT Setup** | `sentiment_study/src/train_entity_model.py`, `score_finbert.py` | `data/raw/finbert/`, SEntFiN | `sentiment_study/reports/entity_model_eval.json` | N/A |
| **Section 8.7: Sentiment Ablation** | `sentiment_study/src/export_company_sentiment.py` | `data/processed/*_company_sentiment.csv` | `reports/company_sentiment_ablation.csv` | N/A |
| **Section 8.8: News-Price Analysis** | `sentiment_study/src/explain_moves.py`, `src/explain_page.py` | Processed news + returns | `sentiment_study/reports/explain/*.csv` | Tables 7–9 |
| **Section 8.9: Permutation & Limitations** | `src/pipeline.py`, `src/evaluation.py` | Shuffled targets | `reports/all_stocks_summary.csv` | `reports/figures/phase1_signal_was_window_luck.png` |
| **Section 9.1: 10-Stock Results** | `src/pipeline.py` | 10 NSE stocks (2000–2026) | `reports/all_stocks_summary.csv` | Table 1 |
| **Section 9.2: Baselines & Illusions** | `src/evaluation.py`, `sentiment_study/src/run_all.py` | Multi-corpus datasets | `leakage_demo.csv`, `phase3_overlap.csv` | Table 2, ASCII Taxonomy |
| **Section 9.3: Seven-Model Benchmark** | `src/modeling.py` | SENSEX next-day panel | `sentiment_study/reports/comparison_all.csv` | Table 3, `phase4_offset_sensitivity.png` |
| **Section 9.4: Deep Learning Failure** | `src/sequence_models.py` | TCS, RELIANCE, SENSEX, S&P | `phase4_sequence_*.csv`, `comparison_dl.csv` | Table 4, `phase4_sequence_models.png` |
| **Section 9.5: Entity FinBERT Eval** | `sentiment_study/src/train_entity_model.py` | SEntFiN test split | `sentiment_study/reports/entity_model_eval.json` | Table 5 |
| **Section 9.6: Sentiment Ablation** | `sentiment_study/src/run_all.py` | 10 stocks $\times$ 2 corpora | `reports/company_sentiment_ablation.csv` | Table 6 |
| **Section 9.7: Lead-Lag & Explain** | `sentiment_study/src/explain_moves.py` | SENSEX + 10 NSE stocks | `sentiment_study/reports/explain/*.csv` | Tables 7, 8, 9 |
| **Section 9.8: Demystifying High Acc** | `sentiment_study/src/run_all.py` | Leakage demonstration suite | `sentiment_study/reports/leakage_demo.csv` | Table 10 |
| **Section 10: Practical Implications** | `app.py`, `src/explain_page.py` | Production pipelines | `reports/phase3_operating_points.csv` | `coverage_accuracy.png`, `why_did_it_move_*.png` (UI Outputs) |
| **Section 11: Limitations** | `src/pipeline.py`, `src/validate.py` | Project configurations | Runtime and metadata logs | `reports/figures/phase1_signal_was_window_luck.png` |

---

## 5. Research Objectives to Experiment Mapping Matrix

| Research Objective | Core Theoretical Hypothesis | Exact Experimental Protocol | Empirical Metrics Evaluated | Primary Source Artifact | Paper Location & Table |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **RO1: Multi-Decade Technical Predictability** | Technical price patterns provide out-of-sample directional edge across full market cycles. | 26-year panel (2000–2026), 10 stocks, $h=3, \text{step}=3$, 5-fold expanding walk-forward, 17-bar embargo. | Walk-forward accuracy, fold standard deviation, net edge over training majority, permutation $p$-value. | `reports/all_stocks_summary.csv` | Section 9.1, Table 1 |
| **RO2: Architecture Complexity vs. Parity** | Non-linear tree ensembles and sequence models outperform regularized linear classifiers in low-SNR equity data. | Benchmarking 7 architectures on 3,248 SENSEX walk-forward days under technical ($T$) and macro ($T+G$) features. | Test accuracy, ROC-AUC, balanced accuracy, McNemar test across feature additions, offset stability. | `sentiment_study/reports/comparison_all.csv` | Section 9.3, Table 3 |
| **RO3: Sequence Modeling & Variance Shrinkage** | Recurrent neural networks (LSTM, GRU) learn conditional mean returns better than naive baselines. | 30-day sequence lookbacks for continuous 3-day return regression on TCS and RELIANCE; next-day classification. | Test RMSE vs. Persistence RMSE and Zero-Return benchmark RMSE ($\hat{r}=0$), shrinkage factor $\frac{\sigma(\hat{r})}{\sigma(r)}$. | `reports/phase4_sequence_TCS.csv`, `phase4_sequence_RELIANCE.csv`, `comparison_dl.csv` | Section 9.4, Table 4 |
| **RO4: Entity-Level Sentiment Attribution** | Sentence-pair FinBERT resolves polarity attribution in multi-company financial headlines. | Fine-tune ProsusAI/finbert on SEntFiN 1.0 ([CLS] entity [SEP] headline [SEP]); test on 1,440 held-out pairs. | Accuracy and Macro-$F_1$ on full test set and multi-entity sub-cohort ($\ge 2$ companies). | `sentiment_study/reports/entity_model_eval.json` | Section 9.5, Table 5 |
| **RO5: Incremental Value of Company Sentiment** | Adding entity-targeted news sentiment improves next-day directional forecasting accuracy. | Walk-forward ablation ($B$ vs. $B+C_{\text{ent}}$) across 10 stocks and 2 news corpora (ToI headlines, NSE filings). | Out-of-sample accuracy delta ($\Delta$), ROC-AUC delta, paired McNemar test $p$-value, 1-year placebo check. | `reports/company_sentiment_ablation.csv`, `company_sentiment_comparison.csv` | Section 9.6, Table 6 |
| **RO6: Lead-Lag Asymmetry (Explain vs. Predict)** | Financial news correlates contemporaneously with price moves but does not forecast forward returns. | Filter circular price reports; compute Spearman rank correlation across lags $-1, 0, +1$; evaluate quintile spreads and big moves. | Spearman $\rho$ and $p$-values at lags $-1, 0, +1$; Quintile $Q_5 - Q_1$ daily return spreads; big move coverage shares. | `sentiment_study/reports/explain/lead_lag.csv`, `quintiles.csv`, `big_moves_coverage.csv` | Section 9.7, Tables 7, 8, 9 |
| **RO7: Demystification of Literature Claims** | Published 70%–90% accuracies stem from specific methodological flaws (lookahead, overlap, non-stationarity). | Controlled ablation intentionally introducing: same-day news, rolling MA target, unit-root level regression. | Accuracy, $R^2$, and $1-\text{MAPE}$ under invalid vs. valid evaluation setups on identical market data. | `sentiment_study/reports/leakage_demo.csv` | Section 9.8, Table 10 |

---

## 6. Recommended Figure and Table Placement Plan

The paper incorporates 10 empirical tables, 8 analytical figures, and 2 illustrative system interface screenshots.

| Item Identifier | Asset Type & Status | Source Path in Project | Target Section | Caption and Description | Scientific Function in Paper |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Table 1** | Empirical Table | `reports/all_stocks_summary.csv` | Section 9.1 | Multi-Decade Walk-Forward Directional Prediction Performance Across 10 NSE Equities ($h=3$ Days). | Demonstrates complete absence of technical edge across 26 years; all 10 stocks `NO SIGNAL`. |
| **Figure 1** | Analytical Figure | `reports/figures/walk_forward_folds.png` | Section 8.3 | Expanding 5-Fold Walk-Forward Cross-Validation Scheme with 17-Bar Embargo Buffer. | Illustrates temporal partitioning and embargo preventing indicator boundary leakage. |
| **Figure 2** | Analytical Figure | `reports/figures/return_distribution.png` | Section 7.1 | Empirical Distribution and Heavy Tails of Daily Log-Returns Across Evaluated Equities. | Confirms finite variance and empirical distributions of differenced logarithmic price series. |
| **Figure 3** | Analytical Figure | `reports/figures/TCS_feature_set_comparison.png` | Section 8.2 | Feature Space Comparison: Nominal Raw Price Levels vs. Dimensionless Scale-Free Indicators. | Visualizes how price levels drift across regimes while dimensionless ratios remain bounded and stable. |
| **Figure 4** | Analytical Figure | `reports/figures/target_autocorrelation.png` | Section 8.1 | Decay of Forward Target Return Autocorrelation: Overlapping Rolling Returns vs. Subsampled Grid. | Proves that subsampling ($\text{step}=3$) collapses artificial target autocorrelation from 0.81 to noise ($-0.04$). |
| **Table 2** | Empirical Table | `sentiment_study/reports/comparison_all.csv` | Section 9.2 | Benchmark Comparison of Baseline Strategies vs. Trained Models Across Experimental Setups. | Anchors model accuracies against honest majority and persistence baselines; introduces the Three Illusions. |
| **Figure 5** | Conceptual Diagram | Derived ASCII / TikZ Box | Section 9.2 | Structural Taxonomy of the Three Accuracy Illusions in Financial Machine Learning. | Conceptual diagram categorizing Majority Trap, Autocorrelation Trap, and Contemporaneous Mirage. |
| **Table 3** | Empirical Table | `sentiment_study/reports/comparison_all.csv` | Section 9.3 | Performance Comparison Across Seven Architectures on SENSEX Next-Day Directional Forecasting. | Proves architecture invariance (< 1.32 pp spread) and establishes the efficacy of $L_2$ regularization. |
| **Figure 6** | Analytical Figure | `reports/figures/phase4_offset_sensitivity.png` | Section 8.1 / 9.3 | Out-of-Sample Performance Sensitivity Across Subsampling Offsets ($k \in \{0, 1, 2\}$). | Reveals high instability in tree ensembles across sampling phases vs. strict stability of Logistic Regression. |
| **Table 4** | Empirical Table | `phase4_sequence_*.csv`, `comparison_dl.csv` | Section 9.4 | Deep Learning Sequence Model Performance and Diagnostic Failure Analysis (TCS, RELIANCE, SENSEX, S&P). | Quantifies failure to beat Zero-Return benchmark (`beats_zero = False`) and extreme variance shrinkage. |
| **Figure 7** | Analytical Figure | `reports/figures/phase4_sequence_models.png` | Section 9.4 | Predicted Returns vs. Realized Returns: Sequence Model Variance Shrinkage Under Empirical Optimization. | Graphically depicts how LSTM and GRU predictions compress into a narrow horizontal band around zero drift. |
| **Table 5** | Empirical Table | `sentiment_study/reports/entity_model_eval.json` | Section 9.5 | Entity-Targeted FinBERT vs. Standard Generic FinBERT on the SEntFiN Benchmark. | Documents decisive NLP fidelity improvement (+19.2 pp overall, +23.8 pp on multi-entity headlines). |
| **Table 6** | Empirical Table | `reports/company_sentiment_ablation.csv` | Section 9.6 | Company-Targeted Sentiment Ablation Across 10 Equities (Times of India Headlines & NSE Filings). | Proves that news sentiment provides zero statistically significant edge (0/20 stocks significant, all $p \ge 0.115$). |
| **Table 7** | Empirical Table | `sentiment_study/reports/explain/lead_lag.csv` | Section 9.7 | Lead-Lag Rank Correlation Profile (Spearman $\rho$) Between News Sentiment and Equity Returns. | Demonstrates fundamental asymmetry: significant same-day correlation collapses on forward trading days. |
| **Table 8** | Empirical Table | `sentiment_study/reports/explain/quintiles.csv` | Section 9.7 | Daily Mean Returns Across Clean News Sentiment Quintiles on the BSE SENSEX. | Shows $+0.20\%$/day contemporaneous spread vanishes and reverses to $-0.035\%$/day on the next trading session. |
| **Table 9** | Empirical Table | `sentiment_study/reports/explain/big_moves_coverage.csv` | Section 9.7 | Explanatory Coverage of Extreme Price Movements ($|\Delta| > 1.5\%$) Across Indian Equities. | Reveals that 52.8% to 97.0% of single-stock big moves occur with zero explanatory press headlines. |
| **Screenshot 1** | Illustrative UI Output | `screenshots/why_did_it_move_infy.png` | Section 10.4 | Decision-Support System Interface: "Why Did It Move?" Explanatory Interface for Infosys (INFY). | *Illustrative system artifact only*: demonstrates user-interface separation of news explanation from prediction. |
| **Screenshot 2** | Illustrative UI Output | `screenshots/why_did_it_move_sensex.png` | Section 10.4 | Decision-Support System Interface: Market-Wide Lead-Lag, Quintiles, and Event Study Diagnostics. | *Illustrative system artifact only*: showcases market-wide empirical diagnostics rendered dynamically in Streamlit. |
| **Table 10** | Empirical Table | `sentiment_study/reports/leakage_demo.csv` | Section 9.8 | Controlled Demonstration of Literature Accuracy Pitfalls on Identical Market Data. | Deconstructs 77% headline, 86% moving average, and 99% level regression illusions. |
| **Figure 8** | Analytical Figure | `reports/figures/phase1_signal_was_window_luck.png` | Section 9.1 / 11.3 | Window Selection Luck: Dissipation of Apparent Technical Predictive Edges as Sample Horizon Expands. | Proves that apparent edges found in short sample windows (2018–2024) evaporate over multi-decade cycles. |
| **Figure 9** | Analytical Figure | `reports/figures/coverage_accuracy.png` | Section 10.1 / 10.4 | Confidence Threshold vs. Accuracy and Coverage Curve for Out-of-Sample Predictions. | Details production abstention logic: trading only when predicted probability exceeds confidence thresholds. |

---

## 7. Author Decisions Required Prior to Full Paper Drafting

The following 5 decisions cannot be determined from repository artifacts and must be provided by the author:

1. **Authorship and Institutional Affiliations**: Full names, ordering, academic/institutional affiliations, and corresponding author email address.
2. **Target Publication Venue & Formatting Style**: Selected journal/conference template (e.g., IEEE 2-column, ACM double-column, Elsevier single-column, or standard working paper markdown).
3. **Hardware Environment Details**: Exact GPU model (e.g., NVIDIA A100 80GB, RTX 4090, V100, T4) and approximate training wall-clock runtimes for FinBERT fine-tuning and walk-forward cross-validation.
4. **NSE Regulatory Filings Cutoff Specification**: Confirmation of the exact calendar cutoff date in 2025 for regulatory disclosures (e.g., August 2025 vs. December 2025).
5. **Backtesting / Trading Simulation Scope**: Confirmation of whether an optional backtesting appendix with transaction cost modeling (STT, slippage) is desired, or if the paper should remain strictly focused on econometric and statistical evaluation.

---

## 8. Verified Bibliography and References

1. **Araci, D. (2019)**. *FinBERT: Financial sentiment analysis with pre-trained language models*. arXiv preprint arXiv:1908.10063.
2. **Cameron, A. C., & Miller, D. L. (2015)**. A practitioner’s guide to cluster-robust inference. *Journal of Human Resources*, 50(2), 317–372. https://doi.org/10.3368/jhr.50.2.317
3. **De Prado, M. L. (2018)**. *Advances in Financial Machine Learning*. John Wiley & Sons, Inc., Hoboken, New Jersey.
4. **Fama, E. F. (1970)**. Efficient capital markets: A review of theory and empirical work. *The Journal of Finance*, 25(2), 383–417. https://doi.org/10.2307/2325486
5. **Hansen, P. R. (2005)**. A test for superior predictive ability. *Journal of Business & Economic Statistics*, 23(4), 365–380. https://doi.org/10.1198/073500105000000063
6. **Loughran, T., & McDonald, B. (2011)**. When is a word liability? Texts, words, and financial performance. *The Journal of Finance*, 66(1), 35–65. https://doi.org/10.1111/j.1540-6261.2010.01625.x
7. **Petersen, M. A. (2009)**. Estimating standard errors in finance panel data sets: Comparing approaches. *The Review of Financial Studies*, 22(1), 435–480. https://doi.org/10.1093/rfs/hhn053
8. **Sinha, A., Kedas, S., Kumar, R., & Malo, P. (2022)**. SEntFiN 1.0: Entity-aware sentiment analysis for financial news. *Journal of the Operational Research Society*, 73(11), 2534–2546. https://doi.org/10.1080/01605682.2021.1994646
9. **Timmermann, A., & Granger, C. W. (2004)**. Efficient market hypothesis and forecasting. *International Journal of Forecasting*, 20(1), 15–27. https://doi.org/10.1016/S0169-2070(03)00012-8
10. **White, H. (2000)**. A reality check for data snooping. *Econometrica*, 68(5), 1097–1126. https://doi.org/10.1111/1468-0262.00152
11. **Yang, Y., Uy, M. C. S., & Huang, A. (2021)**. FinBERT: A large language model for extracting information from financial text. *Management Science*, 67(5), 2568–2592. https://doi.org/10.1287/mnsc.2020.3604
