# Complete Research Paper Outline and Planning Specification

**Project**: Stock Market Predictor — Multi-Decade Technical Predictability, Entity-Level Financial NLP, and Market Efficiency  
**Target Output**: Comprehensive Academic Research Paper  
**Authoritative Results Source**: [`reports/research_paper_results_final.md`](file:///C:/Users/vedan/Stock-Market-Predictor/reports/research_paper_results_final.md)  
**Date**: November 2026  
**Status**: Authoritative Paper Outline & Blueprint  

---

## Executive Summary & Paper Architecture

This document provides the definitive architectural blueprint for authoring the complete research paper based on the Stock Market Predictor project. The empirical results in [`reports/research_paper_results_final.md`](file:///C:/Users/vedan/Stock-Market-Predictor/reports/research_paper_results_final.md) serve as the strict, immutable foundation for all quantitative claims. 

The paper investigates directional predictability across 26 years of National Stock Exchange (NSE) equity data (2000–2026), benchmarks seven machine learning and deep learning architectures, evaluates entity-targeted FinBERT for financial sentiment attribution, conducts rigorous paired sentiment ablation studies, and analyzes the lead-lag relationship between financial news and stock returns.

```
┌──────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                   PAPER ARCHITECTURE & FLOW                                     │
└──────────────────────────────────────────────────────────────────────────────────────────────────┘
                                                 │
          ┌──────────────────────────────────────┴──────────────────────────────────────┐
          ▼                                                                            ▼
  PART I: THE TECHNICAL FOUNDATION                             PART II: THE SENTIMENT EXPANSION
  - 10 NSE Equities (2000–2026, 17,636 rows)                   - News Corpora: ToI (2001–22), NSE (2008–25)
  - Stationary Indicators (52 scale-free features)             - Entity FinBERT ([CLS] Co [SEP] Head [SEP])
  - Expanding Walk-Forward CV (5 Folds, 52d Embargo)           - Gold Standard: SEntFiN 1.0 (14,404 labels)
  - Seven-Model Benchmark (Linear to Sequence DL)              - Paired Ablation: 20 Equity-Corpus Setups
          │                                                                            │
          ▼                                                                            ▼
  EMPIRICAL FINDING 1:                                         EMPIRICAL FINDING 2:
  - Multi-decade technical edge collapses                      - Entity NLP improves accuracy to 87.7%
  - 10/10 stocks: NO SIGNAL vs. Majority Base                  - Incremental next-day edge: Δ = +0.09% (p > 0.11)
  - Architecture invariance (< 1.32 pp spread)                 - 0/20 stocks achieve statistical significance
  - DL models suffer variance shrinkage (< 0.19)               - News explains today (ρ = +0.056); dead tomorrow
          │                                                                            │
          └──────────────────────────────────────┬─────────────────────────────────────┘
                                                 ▼
                              PART III: SYNTHESIS & DEMYSTIFICATION
                              - Resolution of the Literature "Accuracy Illusions"
                              - Controlled Demonstration of 77%, 86%, and 99% Pitfalls
                              - Consistency with Weak & Semi-Strong EMH (Fama, 1970)
                              - Practical Implications for Algorithmic Trading & Dashboards
```

---

## 1. Proposed Title and Abstract Direction

### 1.1. Candidate Titles

1. **Primary Recommendation (Comprehensive & Impactful)**:  
   *Predictability, Attribution, and Market Efficiency: An Empirical Investigation of Technical Indicators and Entity-Targeted Financial Sentiment in Indian Equities*
2. **Alternative A (Methodological Focus)**:  
   *Beyond the High-Accuracy Illusion: Multi-Decade Walk-Forward Directional Prediction, Entity FinBERT, and the Asymmetry of Financial News in NSE Equities*
3. **Alternative B (Asset Pricing / Market Efficiency Focus)**:  
   *Weak and Semi-Strong Market Efficiency in Emerging Markets: Evaluating 26 Years of Price Dynamics, Deep Sequence Models, and Regulatory News Sentiment*

### 1.2. Abstract Direction (Draft Text: 280 words)

> Contemporary financial literature frequently reports machine learning classification accuracies between 70% and 90% in equity directional forecasting. However, such claims often stem from methodological artifacts, including lookahead leakage, target autocorrelation, and unanchored baselines. This study presents a rigorous empirical investigation into the out-of-sample predictability of Indian equities across full market cycles. Evaluating 26 years of daily trading data (2000–2026) across ten major National Stock Exchange (NSE) listings ($N=17,636$ subsampled 3-day holding periods), we implement an expanding 5-fold walk-forward validation framework with strict embargo periods and scale-free stationary indicators. Benchmarking seven architectures—ranging from $L_2$-regularized Logistic Regression to gradient-boosted decision trees (Random Forest, GBDT, XGBoost, LightGBM) and deep recurrent networks (PyTorch LSTM and GRU)—we find that all models converge tightly within a 1.32–2.06 percentage point spread, with zero stocks demonstrating an edge distinguishable from random label permutations (`NO SIGNAL`). Recurrent sequence models exhibit severe prediction variance shrinkage ($\sigma(\hat{r})/\sigma(r) \le 0.19$) and fail to beat a trivial zero-return benchmark. To evaluate informational spillovers from unstructured text, we fine-tune an entity-targeted FinBERT classifier on the SEntFiN 1.0 benchmark, achieving 87.71% accuracy (+19.17 pp over generic FinBERT) and resolving multi-firm attribution ambiguity. However, in paired walk-forward ablations across 12,426 Times of India headlines and 11,740 regulatory filings, incorporating company sentiment yields an unweighted mean accuracy delta of only $+0.09\%$ ($+0.28\%$ weighted), with zero of twenty equity-corpus configurations achieving statistical significance ($p \ge 0.115$). Lead-lag analysis reveals a significant contemporaneous correlation ($\rho = +0.0562, p < 10^{-4}$) that collapses on the forward trading day ($\rho = +0.0070, p = 0.618$). Our findings are fully consistent with Weak-Form and Semi-Strong Form Market Efficiency: news explains today's price movements but provides no tradable edge for tomorrow.

---

## 2. Keywords

`Financial Machine Learning`, `Efficient Market Hypothesis`, `Entity-Targeted Sentiment Analysis`, `FinBERT`, `Walk-Forward Validation`, `Lead-Lag Asymmetry`, `Technical Indicators`, `Emerging Markets`, `National Stock Exchange of India`, `Data Leakage`.

---

## 3. Complete 13-Section Paper Outline

### Section 1: Title, Authors, and Affiliations
- **Working Title**: *Predictability, Attribution, and Market Efficiency: An Empirical Investigation of Technical Indicators and Entity-Targeted Financial Sentiment in Indian Equities*
- **Author Placeholder**: [Author Names, e.g., Vedant et al.]
- **Affiliations**: [Department / Institution / Laboratory]
- **Target Venues**: *Journal of Financial Data Science*, *Quantitative Finance*, *Expert Systems with Applications*, or *IEEE Transactions on Computational Social Systems*.

### Section 2: Abstract
- Structured 250–300 word synthesis (as formulated in Section 1.2).

### Section 3: Keywords
- Standardized academic indexing terms (as formulated in Section 2).

### Section 4: Introduction
- **4.1 The Promise and Peril of Financial Machine Learning**: The rise of algorithmic execution in emerging economies; high expectations placed on modern non-linear learners and Transformer-based NLP.
- **4.2 The "Replication Crisis" in Financial ML**: Proliferation of published papers reporting 70%–90% directional accuracy; discordance between academic claims and practical quantitative trading reality.
- **4.3 The Entity Attribution Bottleneck**: The historical failure of financial sentiment analysis to properly assign polarity to specific corporate entities in multi-firm headlines.
- **4.4 Methodological Scope and Disciplinary Guardrails**: Scope of the 26-year study on the National Stock Exchange of India (NSE); commitment to leakage-free evaluation, stationarity, and open-science replicability.
- **4.5 Summary of Novel Contributions**:
  1. Multi-decade walk-forward benchmarking of 10 mature NSE equities eliminating window-selection bias.
  2. Rigorous empirical demonstration of architecture invariance between linear shrinkage models and complex neural networks.
  3. Formal diagnosis of recurrent neural network failure modes (variance shrinkage and zero-return benchmark failure).
  4. Formulation and evaluation of Entity-Targeted FinBERT on multi-company financial headlines.
  5. Decisive 10-stock paired ablation demonstrating the econometric non-significance of public company news for next-day returns.
  6. Empirical characterization of the fundamental lead-lag asymmetry: news as contemporaneous explanation vs. forward prediction.
  7. Controlled taxonomy and demonstration of the "Three Accuracy Illusions" that explain high-accuracy literature claims.

### Section 5: Literature Review
- **5.1 The Efficient Market Hypothesis and Price Predictability**:
  - Fama (1970): Weak-form, semi-strong form, and strong-form definitions.
  - Data-snooping and window-selection luck: White (2000), Hansen (2005), Timmermann & Granger (2004).
  - Challenges in emerging market efficiency (BSE/NSE structural transitions).
- **5.2 Financial Natural Language Processing: From Lexicons to Transformers**:
  - Domain-specific lexicons: Loughran & McDonald (2011).
  - Financial pre-trained language models: FinBERT (Araci, 2019; Yang, Uy, & Huang, 2021).
  - Limitations of whole-document sentiment aggregation.
- **5.3 Entity-Level and Aspect-Based Sentiment Analysis in Finance**:
  - The SEntFiN benchmark: Sinha, Kedas, Kumar, & Malo (2022).
  - Sentence-pair classification formulations for entity-directed polarity extraction.
- **5.4 Methodological Pitfalls and Evaluation Standards in Financial ML**:
  - Lookahead leakage, non-stationarity, and cross-validation hazards: Marcos López de Prado (2018).
  - Clustered errors and cross-sectional panel dependence: Petersen (2009), Cameron & Miller (2015).
  - Transaction costs and microstructure frictions in Indian equities.

### Section 6: Problem Statement and Research Objectives
- **6.1 Mathematical Formulation of the Prediction Task**:
  - Let $P_{i,t}$ denote the daily adjusted closing price of asset $i$ on trading day $t$.
  - Forward $h$-day return: $R_{i, t, t+h} = \frac{P_{i, t+h} - P_{i,t}}{P_{i,t}}$.
  - Binary directional target: $y_{i,t}^{(h)} = \mathbb{I}(R_{i, t, t+h} > 0) \in \{0, 1\}$.
  - Non-overlapping subsampling condition: Evaluation strictly over sub-grid $\{t_k\}_{k=1}^N$ where $t_{k+1} - t_k \ge h$ (specifically $h=3, \text{step}=3$).
- **6.2 Formal Research Objectives (RO1 to RO7)**:
  - **RO1 (Multi-Decade Technical Signal)**: Determine whether stationary technical indicators provide a statistically significant out-of-sample directional edge across multi-decade NSE market cycles.
  - **RO2 (Architecture Sensitivity and Invariance)**: Evaluate whether non-linear tree ensembles or recurrent architectures outperform convex regularized linear models in low signal-to-noise equity regimes.
  - **RO3 (Sequence Modeling and Variance Shrinkage)**: Investigate whether recurrent neural networks (LSTM, GRU) capture conditional autoregressive dynamics or collapse toward unconditional zero-mean predictions.
  - **RO4 (Entity-Level Sentiment Disambiguation)**: Assess whether modifying FinBERT into a sentence-pair architecture resolves polarity attribution in multi-entity financial news.
  - **RO5 (Incremental Predictive Value of Sentiment)**: Test whether entity-targeted news sentiment provides statistically significant incremental accuracy over technical and macro baselines in next-day forecasting.
  - **RO6 (Lead-Lag Asymmetry: Explanation vs. Prediction)**: Quantify the contemporaneous vs. forward correlation between clean news sentiment and equity returns to determine the precise informational role of financial media.
  - **RO7 (Deconstruction of Literature High-Accuracy Claims)**: Empirically reproduce and isolate the methodological flaws responsible for published claims of 70%–90% prediction accuracy.

### Section 7: Dataset Description
- **7.1 Historical Equity Price Panel**:
  - Ten NSE equities spanning diverse sectors: IT (TCS, INFY), Banking (HDFCBANK, SBI), Energy/Conglomerate (RELIANCE), FMCG (ITC), Commodities (VEDL), Conglomerate/Infrastructure (ADANIENT), Renewables (ADANIGREEN), and Consumer Tech (ETERNAL / Zomato).
  - Temporal expansion: January 3, 2000 to September 1, 2026 for mature listings (over 6,500 trading days); August 2004 for TCS; July 2004 for ADANIENT; June 2018 for ADANIGREEN; July 2021 for ETERNAL.
  - Data hygiene: Audited for corporate actions (2015 Adani demergers, 2026 Vedanta restructuring); removal of non-trading holiday artifacts.
  - Total pooled sample: 17,636 non-overlapping 3-day holding periods.
- **7.2 Macroeconomic and Inter-Market Features**:
  - Global market spillover features: Overnight S&P 500 closing returns, early Asian market opens (Nikkei 225, KOSPI).
  - Domestic macro and volatility features: Nifty 50 benchmark returns, India VIX implied volatility index.
- **7.3 Unstructured Financial News Corpora**:
  - *Times of India (ToI) Historical Archive (2001–2022)*: 12,426 company-headline pairs matched to the 10 target tickers.
  - *National Stock Exchange (NSE) Corporate Regulatory Filings (2008–2025)*: 11,740 machine-parsed regulatory announcements and disclosures stamped with exact dissemination times.
  - *New York Times (NYT) Global Corpus (1990–2025)*: US market context and global macroeconomic news.
- **7.4 Gold-Standard Benchmark (SEntFiN 1.0)**:
  - 14,404 human-annotated entity-headline pairs (10,753 Economic Times headlines) with explicit entity-specific polarity (negative, neutral, positive).

### Section 8: Methodology
- **8.1 Data Preprocessing, Cleaning, and Stationarity**:
  - Corporate action normalization and split adjustments.
  - Subsampling methodology: Setting $\text{step}=3$ for $h=3$ to eliminate moving-average target autocorrelation ($\rho = 0.806 \to -0.044$).
  - Subsampling offset analysis: Evaluating phase sensitivity across offsets $k \in \{0, 1, 2\}$.
- **8.2 Feature Engineering (52 Dimensionless Stationary Indicators)**:
  - Elimination of raw price levels ($P_t, \text{SMA}_k$) that destroy out-of-sample and cross-ticker transferability.
  - Dimensionless price-position ratios: $\Delta_{\text{SMA20}} = \frac{P_t - \text{SMA}_{20}(t)}{\text{SMA}_{20}(t)}$, $\Delta_{\text{SMA50}} = \frac{P_t - \text{SMA}_{50}(t)}{\text{SMA}_{50}(t)}$.
  - Moving-average interaction spreads: $\frac{\text{SMA}_{20}(t) - \text{SMA}_{50}(t)}{\text{SMA}_{50}(t)}$.
  - Leakage-free trailing rolling z-scores (window = 60 days): $z_t = \frac{x_t - \mu_{t-1}}{\sigma_{t-1}}$.
  - Edge-safe Relative Strength Index (RSI): Smooth boundary handling preventing division by zero on consecutive gain streaks (0, 50, 100 limit conditions).
  - Normalized MACD: $\text{MACD}_{\text{norm}} = \frac{\text{MACD}_t}{P_t}$.
  - Stationary Ichimoku indicators: Normalized Tenkan-sen/Kijun-sen spreads, Senkou Span ratios, Cloud thickness $\frac{\text{SpanA} - \text{SpanB}}{P_t}$.
  - Stationary Bollinger Bands: Percent $B = \frac{P_t - \text{Lower}_t}{\text{Upper}_t - \text{Lower}_t}$ and normalized bandwidth $\frac{\text{Upper}_t - \text{Lower}_t}{\text{SMA}_{20}(t)}$.
  - Log-return volatility ratios across multiple trailing horizons (5, 10, 20, 60 days).
- **8.3 Walk-Forward Validation Harness and Baselines**:
  - Expanding-window walk-forward cross-validation across 5 folds.
  - Embargo protocol: Setting an embargo period $E=52$ trading days (equal to the maximum indicator lookback, Senkou Span B) between training and test sets to eliminate feature boundary leakage (de Prado, 2018).
  - Explicit baseline implementations: Training Majority Class baseline, Naive Persistence baseline ($\hat{y}_{t+1} = y_t$), and Zero-Return benchmark ($\hat{r}_{t+1} = 0$).
- **8.4 Seven-Model Benchmarking Suite**:
  - Linear/Convex: $L_2$-regularized Logistic Regression ($C=0.1$).
  - Bagging Ensemble: Random Forest (500 estimators, $\text{min\_samples\_leaf}=20$, $\sqrt{p}$ feature subsampling).
  - Boosting Ensembles: Scikit-Learn HistGradientBoosting ($\text{max\_leaf\_nodes}=15, \eta=0.05$), XGBoost (second-order gradients, depth=4, $\lambda=1.0$), LightGBM (GOSS, histogram binning).
  - Deep Recurrent Neural Networks: PyTorch 2-layer stacked LSTM and GRU (hidden dimension = 64, recurrent dropout $p=0.2$, 30-day sequential lookback windows).
- **8.5 Deep Learning Sequence Architecture and Diagnostics**:
  - Continuous regression formulation vs. directional classification.
  - Evaluation of Root Mean Squared Error (RMSE) against Naive Persistence and Zero-Return ($\hat{r}=0$) benchmarks.
  - Variance shrinkage quantification: $\text{Shrinkage} = \frac{\sigma(\hat{r})}{\sigma(r)}$.
- **8.6 Natural Language Processing & Entity-Targeted FinBERT**:
  - The sentence-pair classification framework:
    $$\text{Sequence} = \text{[CLS]} \circ \text{Target Company} \circ \text{[SEP]} \circ \text{News Headline} \circ \text{[SEP]}$$
  - Model fine-tuning: Pre-trained ProsusAI/finbert weights; AdamW optimizer ($\text{lr} = 3 \times 10^{-5}$, weight decay 0.01); 3 epochs on SEntFiN 1.0 (80/10/10 split partitioned by headline hash to prevent leakage).
  - Multi-entity performance evaluation against standard generic FinBERT.
- **8.7 Econometric Sentiment Ablation Framework**:
  - Paired feature comparison: Baseline Information Set $B$ vs. Augmented Information Set $B + C_{\text{ent}}$.
  - Evaluation across 20 distinct equity-corpus experimental conditions.
  - Paired statistical testing: McNemar test with continuity correction on discordant pairs $(b, c)$:
    $$\chi^2 = \frac{(|b - c| - 1)^2}{b + c}$$
  - Placebo control verification: Shifting sentiment series temporally by $+1$ calendar year.
- **8.8 News-Price Relationship Analysis ("Why Did It Move?" Framework)**:
  - Removal of circular price-reporting headlines via regex templates (`"down X%"`, `"plunges"`, `"surges"`).
  - Spearman rank correlation across temporal lags: Preceding day ($t-1$), Same day ($t$), and Forward day ($t+1$).
  - Sentiment quintile sorting ($Q_1$ to $Q_5$) and daily return spread estimation.
  - Explanatory coverage analysis of extreme market moves ($|\Delta| > 1.5\%$) across 4 categories: Matching, Opposite, Neutral, and No News.
- **8.9 Statistical Diagnostic Limitations and Power Constraints**:
  - Permutation testing procedure: $n=6$ shuffles per fold with Laplace continuity correction:
    $$p_{\min} = \frac{0 + 1}{6 + 1} = \frac{1}{7} \approx 0.1429$$
  - Diagnostic function as a coarse feature-leakage check rather than a high-resolution hypothesis test.
  - Discussion of cross-sectional panel dependence and clustered standard errors (Petersen, 2009).
  - Transaction cost friction modeling: STT (0.10%), exchange turnover charges, GST, SEBI fees, and bid-ask slippage.

### Section 9: Results and Discussion
*(Incorporates the full authoritative content from `reports/research_paper_results_final.md`)*
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
- **10.1 For Quantitative Traders and Asset Managers**:
  - The fatal impact of transaction costs on marginal nominal edges (minimum hurdle $+0.25\%$ to $+0.35\%$ per round trip).
  - The danger of overfitting complex architectures in low-SNR regimes; the superiority of regularized convex models.
- **10.2 For Financial NLP Researchers and System Architects**:
  - The mandatory necessity of entity-level disambiguation in financial text processing.
  - The critical peril of circular price-reporting headlines in NLP evaluation pipelines.
- **10.3 For Market Regulators and Financial Surveillance**:
  - Assessment of informational efficiency on the National Stock Exchange of India.
  - Dissemination latency of corporate regulatory filings (NSE announcements).
- **10.4 For Dashboard and Decision-Support System Design**:
  - Designing UI/UX that transparently separates **explanation** from **prediction** (e.g., the Streamlit "Why did it move?" interface vs. forward forecasting modules).
  - Implementing automated abstention thresholds based on prediction confidence to prevent bad trade execution.

### Section 11: Limitations and Future Work
- **11.1 Daily Frequency and Execution Latency Truncation**:
  - Public daily closing prices vs. intraday millisecond microstructure.
  - Price discovery occurring within seconds of electronic regulatory filing dissemination.
- **11.2 Public News Archives vs. Low-Latency Machine Feeds**:
  - Times of India print daily cycle vs. direct Bloomberg/Reuters/Dow Jones machine-readable feeds.
- **11.3 Permutation Harness Computational Bounding**:
  - Re-acknowledging the $n=6$ permutation limitation ($p_{\min} = 1/7$) and proposing distributed HPC implementations with $n=1,000$ permutations.
- **11.4 Promising Avenues for Future Research**:
  - Multimodal financial forecasting: Incorporating audio recordings and tone from quarterly earnings conference calls.
  - Graph neural networks modeling supply chain, cross-holding, and customer-supplier dependency graphs among NSE firms.
  - High-frequency tick-level order book dynamics during opening and closing auctions.

### Section 12: Conclusion
- **12.1 Concluding Synthesis**: Final appraisal of market efficiency in the world's most populous democracy; affirmation that past technical patterns and public news headlines do not confer a systematic directional edge at daily horizons once rigorous evaluation standards are enforced.
- **12.2 Methodological Mandate**: A call for scientific integrity, leakage auditing, baseline anchoring, and sample window expansion in financial machine learning research.

### Section 13: References
- Complete, verified academic citations (as cataloged in Section 13).

---

## 4. Section-by-Section Source Mapping Table

The following table establishes an unbroken chain of custody between every section of the paper and the project's source code, notebooks, data files, experiment reports, and visual artifacts.

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
| **Section 8.2: Stationary Indicators** | `src/features.py`, `src/stationary.py`, `src/ichimoku.py`, `src/bollinger.py` | `data/raw/*.csv` | `reports/phase2_results.csv` | `reports/figures/TCS_feature_set_comparison.png` |
| **Section 8.3: Walk-Forward & Embargo** | `src/evaluation.py`, `src/pipeline.py` | `data/processed/*_dataset.csv` | `reports/phase2_window_stability.csv` | `reports/figures/walk_forward_folds.png` |
| **Section 8.4: Seven-Model Suite** | `src/modeling.py`, `sentiment_study/src/run_all.py` | `data/processed/*_dataset.csv` | `sentiment_study/reports/comparison_all.csv` | N/A |
| **Section 8.5: LSTM/GRU Diagnostics** | `src/sequence_models.py` | `data/processed/*_dataset.csv` | `reports/phase4_sequence_*.csv`, `comparison_dl.csv` | `reports/figures/phase4_sequence_models.png` |
| **Section 8.6: Entity FinBERT Setup** | `sentiment_study/src/train_entity_model.py`, `score_finbert.py` | `data/raw/finbert/`, SEntFiN | `sentiment_study/reports/entity_model_eval.json` | N/A |
| **Section 8.7: Sentiment Ablation** | `sentiment_study/src/export_company_sentiment.py` | `data/processed/*_company_sentiment.csv` | `reports/company_sentiment_ablation.csv` | N/A |
| **Section 8.8: News-Price Analysis** | `sentiment_study/src/explain_moves.py`, `src/explain_page.py` | Processed news + returns | `sentiment_study/reports/explain/*.csv` | `screenshots/why_did_it_move_*.png` |
| **Section 8.9: Permutation & Limitations** | `src/pipeline.py`, `src/evaluation.py` | Shuffled targets | `reports/all_stocks_summary.csv` | `reports/figures/phase1_signal_was_window_luck.png` |
| **Section 9.1: 10-Stock Results** | `src/pipeline.py` | 10 NSE stocks (2000–2026) | `reports/all_stocks_summary.csv` | Table 1 |
| **Section 9.2: Baselines & Illusions** | `src/evaluation.py`, `sentiment_study/src/run_all.py` | Multi-corpus datasets | `leakage_demo.csv`, `phase3_overlap.csv` | Table 2, ASCII Taxonomy |
| **Section 9.3: Seven-Model Benchmark** | `src/modeling.py` | SENSEX next-day panel | `sentiment_study/reports/comparison_all.csv` | Table 3, `phase4_offset_sensitivity.png` |
| **Section 9.4: Deep Learning Failure** | `src/sequence_models.py` | TCS, RELIANCE, SENSEX, S&P | `phase4_sequence_*.csv`, `comparison_dl.csv` | Table 4, `phase4_sequence_models.png` |
| **Section 9.5: Entity FinBERT Eval** | `sentiment_study/src/train_entity_model.py` | SEntFiN test split | `sentiment_study/reports/entity_model_eval.json` | Table 5 |
| **Section 9.6: Sentiment Ablation** | `sentiment_study/src/run_all.py` | 10 stocks $\times$ 2 corpora | `reports/company_sentiment_ablation.csv` | Table 6 |
| **Section 9.7: Lead-Lag & Explain** | `sentiment_study/src/explain_moves.py` | SENSEX + 10 NSE stocks | `sentiment_study/reports/explain/*.csv` | Tables 7, 8, 9, `why_did_it_move_*.png` |
| **Section 9.8: Demystifying High Acc** | `sentiment_study/src/run_all.py` | Leakage demonstration suite | `sentiment_study/reports/leakage_demo.csv` | Table 10 |
| **Section 10: Practical Implications** | `app.py`, `src/explain_page.py` | Production pipelines | `reports/phase3_operating_points.csv` | `reports/figures/coverage_accuracy.png` |
| **Section 11: Limitations** | `src/pipeline.py`, `src/validate.py` | Project configurations | Runtime and metadata logs | `reports/figures/phase1_signal_was_window_luck.png` |

---

## 5. Research Objectives to Experiment Mapping Table

This matrix maps each formal research objective (RO1 through RO7) to its precise experimental setup, empirical metric, project artifact, and paper location.

| Research Objective | Core Theoretical Hypothesis | Exact Experimental Protocol | Empirical Metrics Evaluated | Primary Source Artifact | Paper Location & Table |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **RO1: Multi-Decade Technical Signal** | Technical price patterns provide out-of-sample directional edge across full market cycles. | 26-year panel (2000–2026), 10 stocks, $h=3, \text{step}=3$, 5-fold expanding walk-forward, 52d embargo. | Walk-forward accuracy, fold standard deviation, net edge over training majority, permutation $p$-value. | `reports/all_stocks_summary.csv` | Section 9.1, Table 1 |
| **RO2: Architecture Complexity vs. Parity** | Non-linear tree ensembles and sequence models outperform regularized linear classifiers in low-SNR equity data. | Benchmarking 7 architectures on 3,248 SENSEX walk-forward days under technical ($T$) and macro ($T+G$) features. | Test accuracy, ROC-AUC, balanced accuracy, McNemar test across feature additions, offset stability. | `sentiment_study/reports/comparison_all.csv` | Section 9.3, Table 3 |
| **RO3: Sequence Modeling & Variance Shrinkage** | Recurrent neural networks (LSTM, GRU) learn conditional mean returns better than naive baselines. | 30-day sequence lookbacks for continuous 3-day return regression on TCS and RELIANCE; next-day classification. | Test RMSE vs. Persistence RMSE and Zero-Return benchmark RMSE ($\hat{r}=0$), shrinkage factor $\frac{\sigma(\hat{r})}{\sigma(r)}$. | `reports/phase4_sequence_TCS.csv`, `phase4_sequence_RELIANCE.csv`, `comparison_dl.csv` | Section 9.4, Table 4 |
| **RO4: Entity-Level Sentiment Attribution** | Sentence-pair FinBERT resolves polarity attribution in multi-company financial headlines. | Fine-tune ProsusAI/finbert on SEntFiN 1.0 ([CLS] entity [SEP] headline [SEP]); test on 1,440 held-out pairs. | Accuracy and Macro-$F_1$ on full test set and multi-entity sub-cohort ($\ge 2$ companies). | `sentiment_study/reports/entity_model_eval.json` | Section 9.5, Table 5 |
| **RO5: Incremental Value of Company Sentiment** | Adding entity-targeted news sentiment improves next-day directional forecasting accuracy. | Walk-forward ablation ($B$ vs. $B+C_{\text{ent}}$) across 10 stocks and 2 news corpora (ToI headlines, NSE filings). | Out-of-sample accuracy delta ($\Delta$), ROC-AUC delta, paired McNemar test $p$-value, 1-year placebo check. | `reports/company_sentiment_ablation.csv`, `company_sentiment_comparison.csv` | Section 9.6, Table 6 |
| **RO6: Lead-Lag Asymmetry (Explain vs. Predict)** | Financial news correlates contemporaneously with price moves but does not forecast forward returns. | Filter circular price reports; compute Spearman rank correlation across lags $-1, 0, +1$; evaluate quintile spreads and big moves. | Spearman $\rho$ and $p$-values at lags $-1, 0, +1$; Quintile $Q_5 - Q_1$ daily return spreads; big move coverage shares. | `sentiment_study/reports/explain/lead_lag.csv`, `quintiles.csv`, `big_moves_coverage.csv` | Section 9.7, Tables 7, 8, 9 |
| **RO7: Demystification of Literature Claims** | Published 70%–90% accuracies stem from specific methodological flaws (lookahead, overlap, non-stationarity). | Controlled ablation intentionally introducing: same-day news, rolling MA target, unit-root level regression. | Accuracy, $R^2$, and $1-\text{MAPE}$ under invalid vs. valid evaluation setups on identical market data. | `sentiment_study/reports/leakage_demo.csv` | Section 9.8, Table 10 |

---

## 6. Recommended Figure and Table Placement Plan

The paper integrates 10 empirical tables (from the authoritative results draft), 8 analytical figures (from `reports/figures/`), and 2 production dashboard screenshots (from `screenshots/`).

| Item Identifier | Asset Type | Source Path in Project | Target Paper Section | Caption and Explanatory Description | Narrative Function & Key Takeaway |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Table 1** | Empirical Table | `reports/all_stocks_summary.csv` | Section 9.1 | Multi-Decade Walk-Forward Directional Prediction Performance Across 10 NSE Equities ($h=3$ Days). | Demonstrates complete absence of technical edge across 26 years; all 10 stocks `NO SIGNAL`. |
| **Figure 1** | Analytical Plot | `reports/figures/walk_forward_folds.png` | Section 8.3 | Expanding 5-Fold Walk-Forward Cross-Validation Scheme with 52-Day Embargo Period. | Illustrates rigorous temporal partitioning and embargo preventing indicator boundary leakage. |
| **Figure 2** | Analytical Plot | `reports/figures/return_distribution.png` | Section 7.1 | Empirical Distribution and Stationarity of Daily Log-Returns Across Evaluated Equities. | Confirms heavy tails, finite variance, and stationarity of differenced logarithmic price series. |
| **Figure 3** | Analytical Plot | `reports/figures/TCS_feature_set_comparison.png` | Section 8.2 | Feature Space Comparison: Nominal Raw Price Indicators vs. Dimensionless Stationary Indicators. | Visualizes how price levels drift across regimes while stationary ratios remain bounded and stable. |
| **Figure 4** | Analytical Plot | `reports/figures/target_autocorrelation.png` | Section 8.1 | Decay of Forward Target Return Autocorrelation: Overlapping Rolling Returns vs. Non-Overlapping Grid. | Proves that subsampling ($\text{step}=3$) collapses artificial target autocorrelation from 0.81 to noise ($-0.04$). |
| **Table 2** | Empirical Table | `sentiment_study/reports/comparison_all.csv` | Section 9.2 | Benchmark Comparison of Baseline Strategies vs. Trained Models Across Experimental Setups. | Anchors model accuracies against honest majority and persistence baselines; introduces the Three Illusions. |
| **Figure 5** | Taxonomy Box | Derived ASCII / TikZ in Section 9.2 | Section 9.2 | Structural Taxonomy of the Three Accuracy Illusions in Financial Machine Learning. | Conceptual diagram categorizing Majority Trap, Autocorrelation Trap, and Contemporaneous Mirage. |
| **Table 3** | Empirical Table | `sentiment_study/reports/comparison_all.csv` | Section 9.3 | Performance Comparison Across Seven Architectures on SENSEX Next-Day Directional Forecasting. | Proves architecture invariance (< 1.32 pp spread) and establishes the efficacy of $L_2$ regularization. |
| **Figure 6** | Analytical Plot | `reports/figures/phase4_offset_sensitivity.png` | Section 8.1 / 9.3 | Out-of-Sample Performance Sensitivity Across Subsampling Offsets ($k \in \{0, 1, 2\}$). | Reveals high instability in tree ensembles across sampling phases vs. strict stability of Logistic Regression. |
| **Table 4** | Empirical Table | `phase4_sequence_*.csv`, `comparison_dl.csv` | Section 9.4 | Deep Learning Sequence Model Performance and Diagnostic Failure Analysis (TCS, RELIANCE, SENSEX, S&P). | Quantifies failure to beat Zero-Return benchmark (`beats_zero = False`) and extreme variance shrinkage. |
| **Figure 7** | Analytical Plot | `reports/figures/phase4_sequence_models.png` | Section 9.4 | Predicted Returns vs. Realized Returns: Sequence Model Variance Shrinkage Under Empirical Optimization. | Graphically depicts how LSTM and GRU predictions compress into a narrow horizontal band around zero drift. |
| **Table 5** | Empirical Table | `sentiment_study/reports/entity_model_eval.json` | Section 9.5 | Entity-Targeted FinBERT vs. Standard Generic FinBERT on the SEntFiN Benchmark. | Documents decisive NLP fidelity improvement (+19.2 pp overall, +23.8 pp on multi-entity headlines). |
| **Table 6** | Empirical Table | `reports/company_sentiment_ablation.csv` | Section 9.6 | Company-Targeted Sentiment Ablation Across 10 Equities (Times of India Headlines & NSE Filings). | Proves that news sentiment provides zero statistically significant edge (0/20 stocks significant, all $p \ge 0.115$). |
| **Table 7** | Empirical Table | `sentiment_study/reports/explain/lead_lag.csv` | Section 9.7 | Lead-Lag Rank Correlation Profile (Spearman $\rho$) Between News Sentiment and Equity Returns. | Demonstrates fundamental asymmetry: significant same-day correlation collapses on forward trading days. |
| **Table 8** | Empirical Table | `sentiment_study/reports/explain/quintiles.csv` | Section 9.7 | Daily Mean Returns Across Clean News Sentiment Quintiles on the BSE SENSEX. | Shows $+0.20\%$/day contemporaneous spread vanishes and reverses to $-0.035\%$/day on the next trading session. |
| **Table 9** | Empirical Table | `sentiment_study/reports/explain/big_moves_coverage.csv` | Section 9.7 | Explanatory Coverage of Extreme Price Movements ($|\Delta| > 1.5\%$) Across Indian Equities. | Reveals that 52.8% to 97.0% of single-stock big moves occur with zero explanatory press headlines. |
| **Screenshot 1** | System UI | `screenshots/why_did_it_move_infy.png` | Section 10.4 | Streamlit Decision-Support Dashboard: "Why Did It Move?" Explanatory Interface for Infosys (INFY). | Demonstrates practical application: aligning news timeline, sentiment scores, and historical move explanations. |
| **Screenshot 2** | System UI | `screenshots/why_did_it_move_sensex.png` | Section 10.4 | Streamlit Decision-Support Dashboard: Market-Wide Lead-Lag, Quintiles, and Event Study Diagnostics. | Showcases market-wide empirical diagnostics rendered dynamically for end-user quantitative exploration. |
| **Table 10** | Empirical Table | `sentiment_study/reports/leakage_demo.csv` | Section 9.8 | Controlled Demonstration of Literature Accuracy Pitfalls on Identical Market Data. | Deconstructs 77% headline, 86% moving average, and 99% level regression illusions. |
| **Figure 8** | Analytical Plot | `reports/figures/phase1_signal_was_window_luck.png` | Section 9.1 / 11.3 | Window Selection Luck: Dissipation of Apparent Technical Predictive Edges as Sample Horizon Expands. | Proves that apparent edges found in short sample windows (2018–2024) evaporate over multi-decade cycles. |
| **Figure 9** | Analytical Plot | `reports/figures/coverage_accuracy.png` | Section 10.1 / 10.4 | Confidence Threshold vs. Accuracy and Coverage Curve for Out-of-Sample Predictions. | Details production abstention logic: trading only when predicted probability exceeds confidence thresholds. |

---

## 7. Missing Information and Questions Requiring User Input

Before drafting the full research paper text, the following specific details, metadata, and author preferences must be confirmed by the user:

### 7.1. Authorship, Affiliations, and Target Venue
- [ ] **Author Names & Order**: Please provide the exact author name(s) and desired author ordering for the manuscript title page.
- [ ] **Affiliations & Addresses**: Institutional departments, universities/laboratories, city, state, country, and postal codes.
- [ ] **Corresponding Author & Email**: Name and contact email to be designated for correspondence.
- [ ] **Target Journal or Conference Format**:
  - *ACM Format* (`acmart` double-column or single-column)
  - *IEEE Format* (`IEEEtran` double-column)
  - *Elsevier / ScienceDirect Format* (e.g., *Expert Systems with Applications*)
  - *Springer / Taylor & Francis Format* (e.g., *Quantitative Finance*)
  - *Standard Academic Markdown / Working Paper Format*

### 7.2. Hardware, Compute Environment, and Training Runtimes
- [ ] **Hardware Architecture**: Specify the GPU hardware used during the fine-tuning of Entity-Targeted FinBERT (e.g., NVIDIA A100 80GB, RTX 4090, T4, or V100).
- [ ] **Compute Runtime Details**: Approximate wall-clock execution time for:
  - SEntFiN 1.0 fine-tuning (3 epochs).
  - Multi-decade walk-forward 10-stock model training.
  - Recurrent neural network (LSTM/GRU) sequence model training.

### 7.3. Dataset Provenance and Regulatory Filing Specifics
- [ ] **NSE Regulatory Filings Cutoff Date**: Confirm the exact cutoff date in 2025 for corporate filings (e.g., December 31, 2025, or an earlier month).
- [ ] **Corporate Filing Scraping Protocol**: Confirm whether filing text was ingested via the official NSE API / NEAPS portal, direct XML/JSON disclosure feeds, or an archival repository.
- [ ] **Corporate Restructuring Treatments**: Confirm whether any additional corporate adjustments (beyond the 2015 Adani demergers and 2026 Vedanta restructuring noted in `metadata.json`) should be explicitly highlighted in the methodology.

### 7.4. Economic / Trading Simulation Assumptions (If an Appendix is Desired)
- [ ] **Transaction Cost Assumptions**: Confirm preferred standard parameters for simulated equity trading:
  - Delivery trades: 0.10% STT + exchange fees $\approx 0.15\%$ one-way ($0.30\%$ round-trip).
  - Intraday trades: 0.025% STT + brokerage $\approx 0.05\%$ round-trip.
  - Bid-ask spread slippage: Fixed 5 bps or 10 bps.
- [ ] **Portfolio Sizing / Capital Allocation**: Confirm whether a backtesting appendix is requested (e.g., long-short quintile portfolio, dollar-neutral market factor) or if the paper should remain focused on econometric and statistical evaluation.

---

## 8. Verified Bibliography and References

Every citation below has been verified against authoritative bibliographic databases, including full author lists, verified publication years, exact article titles, and standard journal/venue references:

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
