# Independent Academic Peer Review Report

**Manuscript Title:** *Predictability, Explainability, and Risk-Aware Trading in Indian Equity Markets: An Empirical Study of Technical Indicators, Financial Sentiment, and Overnight Market Signals*
**Authors:** Smit Patil, Vedant Narayan, Vedant Desai
**Affiliation:** Department of Computer Engineering, K. J. Somaiya College of Engineering, Somaiya Vidyavihar University, Mumbai, India
**Audit Type:** Independent Internal Academic Quality and Reproducibility Audit
**Date:** November 2026
**Overall Recommendation:** Minor Revision (Subject to mandatory econometric clarifications, tone adjustments, and typesetting corrections)

---

## 1. Reviewer Overview & General Assessment

This manuscript provides an extensive empirical examination of asset price predictability, financial NLP explainability, and risk-aware trading across Indian equities listed on the National Stock Exchange of India (NSE). The paper benchmarks seven machine learning and deep learning architectures across multi-decade technical indicators, introduces a sentence-pair entity-targeted FinBERT architecture fine-tuned on SEntFiN 1.0, evaluates overnight market transmission mechanisms, and models real-world Indian statutory transaction frictions across five strategy variants.

### Summary of Scientific Strengths:
1. **Methodological Rigor**: The paper avoids the pervasive data-leakage pitfalls that plague applied financial machine learning literature. The combination of non-overlapping subsampling ($h=3, \text{step}=3$), scale-free dimensionless feature transformations, and expanding walk-forward cross-validation with a 52-day/17-row purged embargo buffer provides a solid econometric foundation.
2. **Pedagogical Demystification of Literature Pitfalls**: Section 16.8 and Table 10 represent a commendable scientific contribution. By taking the identical underlying SENSEX and NSE datasets and deliberately introducing common literature shortcuts (contemporaneous lookahead, overlapping moving averages, unit-root level regression), the author convincingly demonstrates how published claims of 75% to 99% accuracy can be manufactured without genuine forecasting skill.
3. **Candid Reporting of Negative Results**: In contrast to papers that cherry-pick short windows to claim unearned market-beating alpha, this study transparently documents that multi-decade technical edges dissipate to chance (`NO SIGNAL` across 10/10 equities), recurrent sequence models collapse into variance shrinkage, and public news sentiment confers zero statistically significant next-day predictive edge.

### Key Critical Concerns Requiring Revision:
1. **Imprecise Phrasing on Historical Listing Spans**: Phrases such as *"Evaluating 26 years of daily trading data... across ten major listings ($N=17,636$)"* risk misleading readers into believing all ten stocks possess 26 years of continuous history. The manuscript must explicitly distinguish the panel's temporal coverage (2000–2026) from individual stock listing maturities (e.g., ETERNAL was listed in 2021; ADANIGREEN in 2018).
2. **Actionable Signals Backtest Sample Horizon**: While the technical prediction panel spans 2000–2026, the actionable trading strategies backtest (Variants A–E, Table 12) was evaluated over **2013–2026** (13.1 years), reflecting the continuous availability of regulatory disclosures and inter-market features. This distinction must be prominently stated in Section 15 and Section 16.10.
3. **Overnight Opening Gap Tradability Caveats**: The 70.57% directional accuracy in next-morning opening price forecasting (+7.06% edge over naive guess, $p < 10^{-15}$) is statistically sound, but its economic tradability requires explicit qualification regarding pre-open auction microstructure and execution slippage.
4. **Tone, Overconfidence, and Speculative Claims**: The supplementary validation report contained inappropriate speculative journal acceptance probabilities (e.g., "85%–90% acceptance probability"), claims of "100% reproducibility", and premature descriptions of the manuscript as "Publication-Ready Final Draft". These must be systematically excised in favor of sober academic self-appraisal.
5. **LaTeX Typesetting Escape Artifacts**: Python string escaping corrupted several mathematical symbols in the Markdown source (e.g., `$ho = 0.806$` instead of `$\rho = 0.806$`, `\sigma_{	ext{fold}}` instead of `\sigma_{\text{fold}}`).

---

## 2. Detailed Review Requirements Audit (Item-by-Item)

### Requirement 1: Numerical Verification Against Underlying Data
- **Table 1 (Technical Walk-Forward)**: All values match `reports/all_stocks_summary.csv` exactly. The sum of rows is exactly 17,636. All 10 stocks yield `NO SIGNAL`.
- **Table 2 (Baselines Benchmark)**: Values match `sentiment_study/reports/comparison_all.csv` and `leakage_demo.csv`.
- **Table 3 (Seven Architectures)**: Model accuracies under $T$ (51.54%–53.60%, spread 2.06 pp) and under $T+G$ (57.36%–58.68%, spread 1.32 pp) match `sentiment_study/reports/comparison_all.csv`.
- **Table 4 (Deep Learning Diagnostics)**: TCS and RELIANCE RMSE and variance shrinkage values (0.0962 to 0.1926) match `reports/phase4_sequence_TCS.csv` and `phase4_sequence_RELIANCE.csv`.
- **Table 5 (Entity FinBERT on SEntFiN)**: Accuracy (87.71% vs. 68.54%) and multi-entity accuracy (85.34% vs. 61.56%) match `sentiment_study/reports/entity_model_eval.json`.
- **Table 6 (Sentiment Ablation)**: 20 equity-corpus conditions match `reports/company_sentiment_ablation.csv`. Unweighted mean delta is $+0.09\%$; weighted mean delta is $+0.28\%$. All $p \ge 0.115$.
- **Table 7 (Lead-Lag Correlations)**: Spearman $\rho$ values match `sentiment_study/reports/explain/lead_lag.csv` (SENSEX same-day $+0.0562$, forward $+0.0070$; pooled same-day $+0.0324$, forward $+0.0131$).
- **Table 8 (Quintile Spreads)**: Quintile mean returns match `sentiment_study/reports/explain/quintiles.csv` (same-day spread $+0.20\%$/day; next-day $-0.035\%$/day).
- **Table 9 (Big Moves Coverage)**: Proportions match `sentiment_study/reports/explain/big_moves_coverage.csv` (SENSEX unexplained 6.11%; single-stocks unexplained 52.85% to 96.96%).
- **Table 10 (Literature Pitfalls)**: Values match `sentiment_study/reports/leakage_demo.csv`.
- **Table 11 (Overnight Forecast)**: $N=27,179$, LightGBM 70.57%, guess base 63.50%, net edge $+7.06\%$ ($p < 10^{-15}$), news effect $+0.11\%$ ($p=0.286$) match `reports/forecast/stock_summary.csv`.
- **Table 12 (Actionable Strategies)**: Values match `reports/actionable_signal_results.csv` means (Buy & Hold CAGR 15.68%, Max DD 54.77%; Variant E CAGR 7.46%, Max DD 10.94%, Sharpe 0.216, Profit Factor 2.66, Time in Market 6.98%).

### Requirement 2: Statistical Tests Supporting Conclusions
- **McNemar Tests**: The conclusion that news sentiment provides no statistically significant predictive edge is fully supported by paired McNemar tests with continuity correction across all 20 conditions ($p \in [0.115, 1.000]$).
- **Permutation Tests**: The permutation procedure is correctly characterized as having a discrete minimum attainable $p$-value floor of $p_{\min} = 1/(6+1) \approx 0.1429$ due to $n=6$ shuffles per fold with Laplace correction. However, the manuscript must remind readers that failure to achieve significance under a coarse 6-shuffle screen does not prove mathematical zero signal; it indicates that any potential edge is smaller than the variance of label-shuffled walk-forward retrains.

### Requirement 3: 70.57% Overnight Forecast Result and Significance Testing
- The pooled sample size $N=27,179$ across 10 equities (2013–2025) provides massive statistical power.
- The difference between model accuracy (70.57%) and naive guess (63.50%) represents a $+7.06\%$ edge. The normal approximation $z$-test statistic is:
  $$z = \frac{0.70566 - 0.63501}{\sqrt{\frac{0.63501 \times 0.36499}{27179}}} \approx \frac{0.07065}{0.00292} \approx 24.18 \implies p < 10^{-15}$$
- The reported $p$-value of $0.0$ is mathematically justified under asymptotic normal theory.
- **Reviewer Note**: The author must explicitly clarify what the "Guess" baseline represents. In opening gap prediction, gaps exhibit an upward drift (~63.5% open higher than previous close). LightGBM improves this to 70.57% by utilizing the sign and magnitude of overnight US and Asian returns.

### Requirement 4: Temporal Dependence and Overlapping Observations
- The subsampling protocol ($h=3, \text{step}=3$) successfully resolves serial target autocorrelation within each time series ($\rho = 0.806 \to -0.044$).
- The cross-sectional dependence across equities trading concurrently on the NSE is acknowledged via references to Petersen (2009) [11] and Cameron & Miller (2015) [12]. The effective independent time periods across the panel are ~2,154 periods.

### Requirement 5: 26-Year Data Coverage and Usable Observations
- The total subsampled sample size is 17,636 rows.
- The 26-year time horizon (2000–2026) strictly applies to the mature listings (HDFCBANK, INFY, ITC, RELIANCE, SBI, VEDL). The younger listings have shorter listing spans (TCS 2004, ADANIENT 2004, ADANIGREEN 2018, ETERNAL 2021). The manuscript must state this clearly in all introductory and abstract summaries.

### Requirement 6: Variant E Drawdown, CAGR, and Market Exposure
- Variant E reduces maximum drawdown from 54.77% (Buy & Hold) to 10.94%, an **80.02% reduction**.
- Variant E spends only 6.98% of days exposed to equity market risk, holding 100% cash 93.02% of the time.
- The profit factor is **2.66**, and the win rate is **53.24%** across 40.2 average lifetime trades per stock.
- The backtest period spans **2013–2026** (13.1 years), not the full 26-year period.

### Requirement 7: Economically Fair Comparison with Buy & Hold
- The comparison between active strategies and Buy & Hold must explicitly discuss the **cash drag assumption**. The backtest engine assumes a 0% risk-free rate on cash balances. In Indian financial markets, where 91-day Treasury Bills and liquid funds historically yield 5%–7% p.a., allocating 93% of capital to cash earning 0% introduces a substantial artificial penalty on active strategy CAGR. If cash were credited with an institutional risk-free yield, Variant E's total return would be substantially higher.
- Buy & Hold took full unhedged exposure through severe crashes (e.g. VEDL -78.9% drawdown, ADANIENT -66.4% drawdown). For institutional capital with fiduciary drawdown constraints, a 55% drawdown is intolerable.

### Requirement 8: Correct Description of the 52-Day Embargo Buffer
- The embargo mechanism is implemented as follows: In daily space, maximum feature lookback is 52 trading days (Ichimoku Senkou Span B). In subsampled space ($h=3, \text{step}=3$), 17 subsampled rows ($17 \times 3 = 51 \approx 52$ trading days) are dropped from the end of each training set before testing begins.
- This accomplishes two distinct functions: (1) **Purging**: Eliminates forward return overlap across the fold seam ($h=3$); (2) **Decorrelation**: Enforces a time buffer preventing the model from exploiting autoregressive errors across the split boundary.

### Requirement 9: Verification of 35 References and DOIs
- All 35 references in `research_paper/bibliography.md` and Section 21 of the manuscript represent genuine, published scholarly works with accurate author lists, publication years, titles, and valid CrossRef DOIs or URLs.
- In-text citation callouts `[1]` through `[35]` map correctly to the numbered bibliography.

### Requirement 10: Figure and Table Numbering and Referencing
- Figures are numbered Fig 1 to 12; tables are numbered Table 1 to 13.
- All figures and tables are explicitly referenced in the manuscript text and interpreted accurately.

### Requirement 11: Tone, Overconfidence, and Unsupported Claims
- The supplementary validation report (`reports/research_paper_validation.md`) contained speculative acceptance probabilities (e.g., "85%–90% acceptance probability"), which violates academic peer review standards.
- Claims of "100% reproducibility" must be moderated to evidence-based statements of deterministic script execution.
- Labels such as "Publication-Ready Final Draft" should be replaced with "Working Paper / Pre-Submission Manuscript".

### Requirement 12: Title Alignment with Research Scope
- The title *"Predictability, Explainability, and Risk-Aware Trading in Indian Equity Markets: An Empirical Study of Technical Indicators, Financial Sentiment, and Overnight Market Signals"* accurately and comprehensively reflects the four core pillars of the research.

---

## 3. Categorized Issue Log (Critical, Major, Minor)

### Category: CRITICAL ISSUES

#### Issue C1: Typesetting and LaTeX Escape Character Corruption in Markdown Source
- **Location**: `research_paper/research_paper_final.md`, Lines 317, 352, 586, 599.
- **Nature of Issue**: Python string escaping during manuscript compilation converted `\rho` into carriage return (`\r`) resulting in `$ho = 0.806$`, and `\text` into tab characters (`\t`) resulting in `\sigma_{	ext{fold}}` and `$	ext{step}=h$`.
- **Why It Matters**: Corrupted mathematical notation immediately discredits the manuscript in academic peer review.
- **Recommended Correction**: Use raw strings or double backslashes (`\\rho`, `\\text`) to ensure pristine LaTeX rendering across Markdown, HTML, DOCX, and PDF.

#### Issue C2: Speculative Journal Acceptance Probabilities in Documentation
- **Location**: `reports/research_paper_validation.md`, Section 6.2.
- **Nature of Issue**: The validation report claimed an "Acceptance Probability: High (85%–90%)" for *Journal of Financial Data Science* and *Quantitative Finance*.
- **Why It Matters**: Acceptance probabilities cannot be known prior to formal peer review. Speculative percentages undermine scholarly objectivity and violate project review guidelines.
- **Recommended Correction**: Completely remove all speculative numerical acceptance probabilities. Replace with objective journal scope alignment notes.

#### Issue C3: Unqualified "26-Year" Panel Generalization
- **Location**: `research_paper/research_paper_final.md`, Abstract, Section 3, Section 7.1, Section 20.1.
- **Nature of Issue**: The manuscript states that 26 years of data (2000–2026) were evaluated across ten equities ($N=17,636$), without immediately clarifying in the abstract that only 6 of the 10 equities existed in 2000.
- **Why It Matters**: Reviewers will flag this as factual inaccuracy regarding corporate listing dates (e.g., Zomato IPO was in July 2021; Adani Green in June 2018).
- **Recommended Correction**: Add an explicit clarifying clause in the Abstract and Section 3 stating: *"comprising up to 26 years of daily data for mature listings (2000–2026) and complete post-listing trading histories for younger listings ($N=17,636$ pooled subsampled observations)"*.

---

### Category: MAJOR ISSUES

#### Issue M1: Actionable Trading Strategy Horizon Omission
- **Location**: `research_paper/research_paper_final.md`, Section 15 and Section 16.10.
- **Nature of Issue**: Table 12 reports strategy backtest results without stating in the table title or text that the backtest horizon is **2013–2026** (13.1 years), not 2000–2026.
- **Why It Matters**: Readers may assume the strategy backtest spans the entire 2000–2026 period, leading to confusion regarding trade counts and annualized returns.
- **Recommended Correction**: Explicitly state the evaluation period (**July 2013 to August 2026**, up to 13.1 years chronologically out-of-sample) in Section 15, Section 16.10, and the Table 12 caption.

#### Issue M2: Economic Realism of Cash Yield Assumption in Strategy Comparison
- **Location**: `research_paper/research_paper_final.md`, Section 16.10, Section 17.3.
- **Nature of Issue**: Active strategies (Variants B, D, E) hold cash between 62.5% and 93.0% of the time. The backtest assumes a 0% return on cash balances.
- **Why It Matters**: In emerging markets like India, risk-free cash (91-day T-Bills, repo, liquid mutual funds) yields 5%–7% annualized. Assuming 0% cash yield severely penalizes the CAGR of high-abstention strategies.
- **Recommended Correction**: Add an explicit analytical discussion in Section 17.3 calculating the pro-forma risk-free yield boost: with an assumed 6.0% risk-free cash return, Variant E's effective CAGR increases from 7.46% to ~13.0%, while maintaining its 10.94% maximum drawdown.

#### Issue M3: Tradability Caveats for Overnight Opening Gap Forecasting
- **Location**: `research_paper/research_paper_final.md`, Section 13, Section 16.9.
- **Nature of Issue**: The manuscript reports a 70.57% directional accuracy in predicting next-morning opening prices without explaining market microstructure execution barriers.
- **Why It Matters**: The opening print ($P_{\text{open}}$) is established via the pre-open call auction (09:00–09:08 IST). Retail and institutional participants cannot execute at yesterday's close to capture the opening gap; they must trade either in the pre-open auction or via overnight index futures/options.
- **Recommended Correction**: Add an explicit market microstructure disclosure explaining that the 70.57% opening gap forecast represents an information transmission metric, not an unconstrained arbitrage strategy.

#### Issue M4: Claims of "100% Reproducibility" and Premature Publication Status
- **Location**: `reports/research_paper_validation.md`, Section 1, Section 6.3; `research_paper/research_paper_final.md`, subtitle.
- **Nature of Issue**: Using phrases such as "100% Deterministic Reproducibility" and labeling the document "Publication-Ready Final Draft".
- **Why It Matters**: In academic literature, no external study is 100% reproducible until independently replicated by outside researchers. Subtitles should reflect working paper or pre-submission status.
- **Recommended Correction**: Replace "100% Deterministic Reproducibility" with *"Deterministic script execution verified across all local pipelines"*. Change subtitle to *"Pre-Submission Working Paper Manuscript"*.

---

### Category: MINOR ISSUES

#### Issue m1: Discrete Permutation Floor Clarification in Abstract
- **Location**: `research_paper/research_paper_final.md`, Abstract (Section 1).
- **Nature of Issue**: The abstract states *"zero equities demonstrating an edge distinguishable from random label permutations (`NO SIGNAL`)"*, but does not mention the $p_{\min} \approx 0.1429$ resolution floor.
- **Recommended Correction**: Add a brief parenthetical note: *"`NO SIGNAL` under the walk-forward permutation diagnostic screen"*.

#### Issue m2: Explicit Definition of Naive Guess Baseline in Table 11
- **Location**: `research_paper/research_paper_final.md`, Section 16.9 and Table 11.
- **Nature of Issue**: Table 11 lists "Naive Guess Base" (63.50%) without defining how the guess is parameterized.
- **Recommended Correction**: Add a footnote explaining that the naive guess baseline represents the unconditional upward opening frequency (predicting a gap up every session), which establishes the non-trivial benchmark.

#### Issue m3: Document Metadata Consistency
- **Location**: `research_paper/research_paper_final.md` and `research_paper/Stock_Market_Predictor_Research_Paper.docx`.
- **Nature of Issue**: Author affiliation block should include contact/repository access information.
- **Recommended Correction**: Add an author correspondence footnote indicating that all code, processed datasets, and model checkpoints are available in the project repository.

---

## 4. Overall Review Summary & Signoff

The paper is methodologically sound, transparent in its negative findings, and rich in empirical data. Addressing the identified critical and major issues will elevate the manuscript to the highest standards of quantitative finance scholarship.
