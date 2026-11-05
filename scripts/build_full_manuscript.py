"""
build_full_manuscript.py
Constructs the complete 21-section academic research paper manuscript in Markdown format
at research_paper/research_paper_final.md.
Incorporates all internal audit corrections and rigorous econometric disclosures,
using raw strings to prevent LaTeX typesetting escape character corruption.
"""

import os

OUTPUT_FILE = "research_paper/research_paper_final.md"

def build_manuscript():
    sections = []

    # Header & Title
    sections.append(r"""# Predictability, Explainability, and Risk-Aware Trading in Indian Equity Markets: An Empirical Study of Technical Indicators, Financial Sentiment, and Overnight Market Signals

**Smit Patil**, **Vedant Narayan**, and **Vedant Desai**
Department of Computer Engineering
K. J. Somaiya College of Engineering, Somaiya Vidyavihar University, Mumbai, India
*Working Paper — Pre-Submission Manuscript*
*Date: November 2026*
*Correspondence: Department of Computer Engineering, K. J. Somaiya College of Engineering, Somaiya Vidyavihar University, Mumbai, India. All data processing scripts, feature engineering pipelines, model checkpoints, and evaluation harnesses are available for replication in the project repository.*

---
""")

    # 1. Abstract
    sections.append(r"""## 1. Abstract

Contemporary financial machine learning literature frequently reports directional classification accuracies exceeding 70% to 90%. However, such findings often reflect methodological vulnerabilities, including lookahead data leakage, target autocorrelation, and unanchored baselines. This study presents a rigorous empirical investigation into the out-of-sample predictability of Indian equities across multi-decade market cycles. Evaluating up to 26 years of daily trading data for mature listings (January 2000 to August 2026) alongside complete post-listing histories for younger listings ($N=17,436$ pooled subsampled, non-overlapping 3-day holding periods across ten National Stock Exchange listings), we implement an expanding 5-fold walk-forward validation framework with strict 52-trading-day (17-subsampled-bar) embargo buffers and scale-free, dimensionless indicator transformations. Benchmarking seven architectures—spanning $L_2$-regularized Logistic Regression, decision-tree ensembles (Random Forest, GBDT, XGBoost, LightGBM), and recurrent neural networks (PyTorch LSTM and GRU)—we find that all models converge within a narrow 1.32 percentage point spread under inter-market features, with zero equities demonstrating an edge distinguishable from random label permutations under the walk-forward diagnostic screen (`NO SIGNAL`). Recurrent sequence models exhibit severe variance shrinkage ($\sigma(\hat{r})/\sigma(r) \le 0.19$) and fail to outperform a trivial zero-return benchmark. To evaluate unstructured text spillovers, we fine-tune an entity-targeted FinBERT classifier on the SEntFiN 1.0 benchmark, achieving 87.71% accuracy (+19.17 pp over generic FinBERT) and resolving multi-firm attribution ambiguity. However, in paired walk-forward ablations across 12,426 Times of India headlines and 11,740 regulatory filings, incorporating company sentiment yields an unweighted mean accuracy delta of $+0.09\%$ ($+0.28\%$ weighted), with zero of twenty equity-corpus configurations achieving statistical significance ($p \ge 0.115$). Lead-lag cross-correlations demonstrate a significant contemporaneous association ($\rho = +0.0562, p < 10^{-4}$) that collapses to noise on forward trading sessions ($\rho = +0.0070, p = 0.618$). In contrast, next-morning opening price forecasting ($N=27,179$, 2013–2025) achieves 70.57% directional accuracy vs. a 63.50% training-majority guess baseline (+7.06% edge, $p < 10^{-15}$), driven almost entirely by overnight US and Asian market returns rather than news sentiment; however, we emphasize that this reflects international information transmission rather than an executable trading arbitrage due to pre-open call auction pricing and opening bid-ask spreads. Finally, backtesting five actionable trading strategy variants across all ten stocks over a 13.1-year out-of-sample period (2013–2026) under Indian statutory transaction costs (15 bps friction) demonstrates that while unhedged Buy & Hold yields a higher nominal CAGR of 15.68%, it incurs a devastating 54.77% average maximum drawdown. A regime-adaptive strategy (Variant E) achieves a verified historical backtest CAGR of 7.46% (and 2.82% under a zero-cash sensitivity), drastically reducing maximum drawdown by 80% (down to 10.94%) with a profit factor of 2.66 while requiring active market risk exposure only 6.98% of the time. Our findings are consistent with Weak-Form and Semi-Strong Form Market Efficiency at daily horizons and provide actionable guardrails for quantitative trading systems.
""")

    # 2. Keywords
    sections.append(r"""## 2. Keywords

`Financial Machine Learning`, `Efficient Market Hypothesis`, `Entity-Targeted Sentiment Analysis`, `FinBERT`, `Walk-Forward Validation`, `Lead-Lag Asymmetry`, `Technical Indicators`, `Emerging Markets`, `National Stock Exchange of India`, `Data Leakage`, `Risk-Aware Trading Strategies`, `Overnight Gap Forecasting`.
""")

    # 3. Introduction
    sections.append(r"""## 3. Introduction

Financial machine learning has experienced an extraordinary surge in academic literature and commercial deployment over the past decade. Driven by the democratization of high-performance computing, the open-source availability of deep-learning libraries, and the proliferation of transformer-based natural language processing (NLP) models, researchers and practitioners increasingly deploy complex algorithmic pipelines to predict equity price movements.

Nowhere is this phenomenon more pronounced than in major emerging markets such as India. The National Stock Exchange of India (NSE) is currently the world's largest derivatives exchange by contract volume and ranks among the top global equity venues by cash market transactions [35]. With millions of new retail participants entering the market since 2020 alongside sophisticated domestic and foreign institutional investors, the question of whether electronic equity prices can be systematically forecasted using technical indicators, news sentiment, and global macro spillovers carries profound economic and theoretical importance.

However, modern financial machine learning faces what several leading scholars have described as an institutional "replication crisis" [6]. Published papers in reputable computer science and applied artificial intelligence venues regularly claim out-of-sample directional prediction accuracies of 70%, 80%, or even 90% [14]. Yet, when quantitative investment firms attempt to deploy these strategies in live electronic execution, the claimed profitability invariably evaporates.

This paper bridges the divide between algorithmic theory and empirical reality. By conducting an exhaustive investigation across ten major Indian equities spanning January 2000 through August 2026 ($N=17,436$ pooled subsampled observations, covering up to 26 years for mature listings and full post-IPO histories for newer equities), this study systematically investigates the boundaries of predictability across three distinct information sets: historical technical indicators, entity-targeted financial news sentiment, and overnight global market transmissions.

![Figure 1: Research Architecture](figures/fig1_research_architecture.png)
*Figure 1: End-to-end econometric and machine learning research architecture spanning data hygiene, feature engineering, walk-forward cross-validation with purged embargoes, seven model architectures, entity NLP, lead-lag explainability, and transaction-cost-aware trading strategy evaluation.*
""")

    # 4. Research Motivation and Objectives
    sections.append(r"""## 4. Research Motivation and Objectives

### 4.1. Motivation
The primary motivation of this study stems from three unresolved dilemmas in empirical quantitative finance:
1. **The Horizon and Selection Vulnerability**: Many published trading algorithms evaluate models on short, convenient sample windows—typically 3 to 5 years (e.g., 2018–2022)—which often coincide with persistent secular bull regimes. In such windows, models may appear highly profitable simply by passively capturing systematic market beta, rather than discovering true alpha.
2. **The Document-Level Sentiment Fallacy**: Standard financial NLP pipelines score news articles using document-level sentiment models (such as default FinBERT [7], [8] or dictionary lexicons [10]). In multi-company financial journalism, this approach catastrophically misattributes polarity. For example, in a headline such as *"Reliance Retail gains market share as Tata Motors slumps on supply chain constraints"*, a generic sentiment model outputs a single diluted score, failing to recognize that the news is simultaneously bullish for Reliance and bearish for Tata.
3. **The Disconnect Between Prediction and Implementation**: Machine learning research often focuses exclusively on classification accuracy or mean squared error (MSE), ignoring real-world market microstructure frictions, statutory transaction taxes, bid-ask slippage, and maximum drawdown risk. An algorithm with 54% accuracy can easily become bankrupt under real-world transaction costs if it trades too frequently or suffers extreme tail-risk drawdowns.

### 4.2. Formal Research Questions (RQ1 to RQ5)
To resolve these dilemmas, this research addresses five formal research questions:
- **RQ1**: Do stationary, scale-free technical price indicators provide out-of-sample directional predictability over honest baselines across multi-decade market cycles on the National Stock Exchange of India?
- **RQ2**: Does resolving multi-company attribution ambiguity through entity-targeted language models yield an incremental, statistically significant edge in short-horizon equity return forecasting?
- **RQ3**: Does overnight global market information (US equity closing returns and Asian morning market opens) provide exploitable directional information for next-morning opening price forecasting?
- **RQ4**: Can risk-aware trading strategy variants preserve capital and improve risk-adjusted performance (Sharpe, Sortino, Maximum Drawdown) after accounting for statutory Indian transaction costs?
- **RQ5**: What specific methodological traps (data leakage, autocorrelation, non-stationarity) are responsible for published high-accuracy claims in financial machine learning, and how can they be deconstructed?
""")

    # 5. Literature Review
    sections.append(r"""## 5. Literature Review

### 5.1. The Efficient Market Hypothesis and Technical Predictability
The theoretical foundation of asset price predictability traces to Eugene Fama's Efficient Market Hypothesis (EMH) [1], [2]. Fama categorized market efficiency into three distinct forms based on the information set $\Phi_t$:
- **Weak-Form Efficiency**: The information set $\Phi_t^{\text{weak}}$ includes only past price and volume history. Under weak-form efficiency, technical analysis and historical chart patterns cannot yield risk-adjusted excess returns:
  $$\mathbb{E}[R_{i, t+h} \mid \Phi_t^{\text{weak}}] = \mathbb{E}[R_{i, t+h}]$$
- **Semi-Strong Form Efficiency**: The information set $\Phi_t^{\text{semi}}$ includes all publicly available information, including corporate financial statements, regulatory disclosures, and press headlines. Under semi-strong efficiency, prices adjust instantaneously to public announcements, precluding profitable trading on published news.
- **Strong-Form Efficiency**: The information set $\Phi_t^{\text{strong}}$ encompasses all information, both public and private/insider information.

Crucially, Fama [1], [2] emphasized the **Joint Hypothesis Problem**: any empirical test of market efficiency is simultaneously a test of market efficiency and the assumed asset pricing model of equilibrium expected returns. Consequently, empirical research cannot definitively "prove" that a market is efficient; it can only demonstrate that, within the bounds of a specified model and information set, no exploitable excess return remains after risk and transaction costs.

Subsequent econometric research by White [3] and Hansen [4] established rigorous frameworks for testing "data-snooping" and selection luck. When dozens of technical trading rules or hyperparameter combinations are evaluated on a single historical series, standard statistical tests dramatically overstate significance due to multiple hypothesis testing. Timmermann and Granger [5] noted that financial forecasting relationships are inherently non-stationary; patterns that appear profitable in one economic regime invariably decay as market participants adapt and arbitrage away the edge.

### 5.2. Financial Natural Language Processing and Sentiment Extraction
The application of textual analysis to financial markets began with manually curated domain dictionaries. Loughran and McDonald [10] demonstrated that standard psychological lexicons (such as Harvard General Inquirer) severely misclassify financial terms (e.g., words like *"cost"*, *"liability"*, and *"foreign"* are treated as negative despite being ordinary accounting terms). Their domain-specific lexicon established a benchmark for financial text analysis. Tetlock [13] utilized media pessimism to predict downward price pressure followed by mean-reversion, while Bollen et al. [14] popularized social media sentiment forecasting.

The advent of transformer neural architectures [25] and bidirectional pre-training (BERT) [26] revolutionized NLP. Araci [7] introduced **FinBERT**, adapting BERT to financial corpora, while Yang et al. [8] pre-trained FinBERT on massive financial filings and corporate communications, demonstrating superior domain-specific classification over general-purpose models.

### 5.3. Entity-Level and Aspect-Based Financial Sentiment
Despite the success of transformer models, standard document-level classification fails in complex multi-entity news environments. In financial journalism, a single headline frequently references multiple corporations, competitors, regulatory bodies, and macroeconomic indicators. Sinha, Kedas, Kumar, and Malo [9] addressed this bottleneck by constructing **SEntFiN 1.0**, a gold-standard dataset of 14,404 entity-headline annotations across 10,753 Economic Times headlines. SEntFiN demonstrated that financial sentiment is directed toward specific entities, laying the groundwork for entity-targeted sentence-pair language modeling.

### 5.4. Methodological Traps and Evaluation Protocols in Financial Machine Learning
Marcos López de Prado [6] synthesized the econometric pitfalls unique to financial machine learning:
1. **Target Overlap and Serial Autocorrelation**: When rolling multi-day returns are used as targets (e.g., 5-day or 10-day holding horizons sampled daily), consecutive labels share identical underlying days. This induces extreme serial correlation that invalidates standard independent and identically distributed (i.i.d.) cross-validation assumptions.
2. **Train-Test Boundary Leakage**: Lagged technical indicators with extended lookbacks (e.g., 52-day moving averages or Ichimoku cloud spans) can bleed future information across cross-validation seams unless purged and embargoed.
3. **Panel Cross-Sectional Dependence**: In panel datasets of multiple equities trading concurrently, equities share systematic market risk. Petersen [11] and Cameron & Miller [12] demonstrated that pooling panel observations without clustering standard errors severely overstates degrees of freedom and downward-biases standard errors.
""")

    # 6. Research Gap and Contributions
    sections.append(r"""## 6. Research Gap and Contributions

### 6.1. The Identified Research Gaps
Despite thousands of published papers on stock price prediction, four major gaps persist:
1. **Absence of Multi-Decade Walk-Forward Evaluation**: Most studies test models on 2 to 5 years of recent data using standard shuffled $k$-fold cross-validation, which violates temporal causality and overfits to transient market regimes.
2. **Lack of Entity-Targeted Sentiment Ablation**: Extant literature rarely quantifies the incremental predictive value of news sentiment via paired econometric ablations (comparing identical models with and without sentiment).
3. **Failure to Distinguish Explanation from Prediction**: Studies frequently report high correlations between same-day news sentiment and price moves without testing whether news published on day $t$ retains any predictive power on day $t+1$.
4. **Neglect of Real-World Statutory Frictions**: Very few academic papers test trading strategies under real-world statutory taxes and slippage (such as Indian Securities Transaction Tax, GST, and SEBI turnover fees).

### 6.2. Seven Key Research Contributions
This study resolves these gaps through seven primary contributions:
1. **Multi-Decade Walk-Forward Empirical Panel**: We evaluate up to 26 years of daily data (2000–2026) across 10 NSE equities ($N=17,436$ pooled subsampled observations) using an expanding 5-fold walk-forward cross-validation harness with a 52-day (17-bar) embargo.
2. **Empirical Architecture Parity**: We benchmark seven distinct machine-learning architectures (from $L_2$-regularized Logistic Regression to XGBoost, LightGBM, and stacked PyTorch LSTMs/GRUs), proving that model complexity provides no edge in low signal-to-noise regimes.
3. **Deep Learning Sequence Failure Diagnostics**: We quantify the failure of recurrent sequence models (LSTM and GRU) on financial return series, documenting extreme variance shrinkage ($\sigma(\hat{r})/\sigma(r) \le 0.19$) and the failure to outperform a trivial zero-return benchmark.
4. **Entity-Targeted FinBERT Architecture**: We formulate entity sentiment extraction as a sentence-pair classification task, boosting SEntFiN accuracy from 68.54% to 87.71% (+19.17 pp) and resolving multi-entity ambiguity.
5. **Rigorous Sentiment Ablation**: Across 20 equity-corpus conditions (12,426 ToI headlines and 11,740 regulatory filings), paired McNemar tests demonstrate that adding company sentiment yields zero statistically significant improvement ($p \ge 0.115$).
6. **Characterization of Lead-Lag Asymmetry**: We resolve the financial news paradox, demonstrating that news exhibits a significant contemporaneous relationship with price moves ($\rho = +0.0562, p < 10^{-4}$) that collapses to noise on forward trading sessions ($\rho = +0.0070, p = 0.618$).
7. **Actionable Risk-Aware Trading System**: We develop and backtest five actionable strategy variants over a 13.1-year out-of-sample period (2013–2026) under a 15 bps statutory Indian friction hurdle, proving that regime-adaptive rules (Variant E) cut maximum drawdown by 80% while achieving a profit factor of 2.66.
""")

    # 7. Dataset and Data Sources
    sections.append(r"""## 7. Dataset and Data Sources

The empirical analysis is conducted across four rich datasets spanning equity price history, inter-market macroeconomic indicators, unstructured financial journalism, and official regulatory disclosures.

### 7.1. Historical Equity Price Panel (10 NSE Equities)
The equity universe comprises ten prominent National Stock Exchange (NSE) listings selected to span diverse market capitalizations, cyclical sectors, and listing maturities:
- **Information Technology**: Tata Consultancy Services (`TCS`), Infosys (`INFY`).
- **Energy, Conglomerate, and Materials**: Reliance Industries (`RELIANCE`), Vedanta (`VEDL`).
- **Banking and Financial Services**: HDFC Bank (`HDFCBANK`), State Bank of India (`SBI`).
- **Fast-Moving Consumer Goods (FMCG)**: ITC Limited (`ITC`).
- **Infrastructure, Green Energy, and Internet IPOs**: Adani Enterprises (`ADANIENT`), Adani Green Energy (`ADANIGREEN`), Eternal Portfolio / Zomato (`ETERNAL`).

While mature equities (HDFCBANK, INFY, ITC, RELIANCE, SBI, VEDL) possess over 26 years of continuous listing history from January 3, 2000 to August 28, 2026 (>6,588 daily sessions), newer listings are evaluated over their full post-listing lifecycles: TCS from August 2004 (1,767 subsampled rows), ADANIENT from July 2004 (1,773 rows), ADANIGREEN from June 2018 (619 rows), and ETERNAL from July 2021 (375 rows). All series were adjusted for corporate splits, bonus issues, dividends, and corporate restructurings (including the 2015 Adani demergers and the 2026 Vedanta restructuring). Non-trading holiday rows were audited and purged.

### 7.2. Inter-Market Macroeconomic Indicators
To capture global transmission channels, the panel includes:
- **Domestic Benchmarks**: S&P BSE SENSEX, Nifty 50, and India VIX (National Stock Exchange Volatility Index).
- **Overnight US Equity Markets**: S&P 500 daily closing return and Dow Jones Industrial Average.
- **Early Morning Asian Markets**: Nikkei 225 (Japan) and KOSPI (South Korea) morning opening returns.

### 7.3. Unstructured Financial Journalism and Regulatory Disclosures
Textual information was extracted from two extensive Indian corpora:
1. **Times of India (ToI) Historical News Archive (2001–2022)**: 12,426 company-headline pairs matched to the 10 project equities, filtered to eliminate non-business reporting.
2. **NSE Corporate Regulatory Disclosures (2008–2025)**: 11,740 machine-parsed regulatory announcements submitted under SEBI (Listing Obligations and Disclosure Requirements) Regulations, stamped with exact exchange dissemination timestamps.

### 7.4. SEntFiN 1.0 Gold-Standard Benchmark
For supervised fine-tuning and evaluation of entity sentiment models, we utilize the **SEntFiN 1.0** dataset [9], comprising 14,404 entity-headline annotations across 10,753 Economic Times headlines. The dataset is partitioned into an 80% train split, a 10% validation split, and a 10% test split (1,440 pairs) using MD5 headline hashing to prevent train-test contamination.
""")

    # 8. Data Preprocessing
    sections.append(r"""## 8. Data Preprocessing

### 8.1. Mathematical Formulation of the Prediction Task
Let $P_{i,t}$ denote the adjusted closing price of equity $i$ on trading day $t$. The discrete multi-day forward return over holding horizon $h$ is defined as:
$$R_{i, t, t+h} = \frac{P_{i, t+h} - P_{i,t}}{P_{i,t}}$$

The binary directional target is formulated as:
$$y_{i,t}^{(h)} = \mathbb{I}\left(R_{i, t, t+h} > 0\right) = \begin{cases} 1 & \text{if } R_{i, t, t+h} > 0 \\ 0 & \text{if } R_{i, t, t+h} \le 0 \end{cases}$$

### 8.2. Subsampling Protocol to Eliminate Target Autocorrelation
When multi-day forward returns ($h=3$) are calculated on consecutive daily sessions ($t, t+1, t+2$), adjacent target labels share $h-1=2$ days of identical price movements. This induces severe moving-average serial autocorrelation in the target series:
$$\text{Corr}\left(R_{i, t, t+3}, R_{i, t+1, t+4}\right) \approx \frac{2}{3} \approx 0.67$$

Empirical measurement on raw rolling targets confirms an artificial autocorrelation of $\rho = 0.806$ (`reports/phase3_overlap.csv`). Under such conditions, a model that simply memorizes the preceding day's direction achieves spurious accuracy.

To eliminate this artifact, we enforce a strict subsampling protocol ($\text{step}=h=3$). The dataset is evaluated exclusively on the sub-grid:
$$\mathcal{T}_k = \{t_0 + k, t_0 + k + 3, t_0 + k + 6, \dots\}, \quad k \in \{0, 1, 2\}$$

Subsampling completely collapses the target serial autocorrelation from $\rho = 0.806$ to $\rho = -0.044$ (indistinguishable from white noise), ensuring that each evaluated holding period represents an independent, non-overlapping price event.

### 8.3. Subsampling Phase Offset Sensitivity Analysis
A key vulnerability identified in multi-day subsampling is sensitivity to the phase offset $k$. In a 3-day holding period, there are three equally valid starting days ($k=0$ Monday, $k=1$ Tuesday, $k=2$ Wednesday). A robust model must exhibit consistent out-of-sample performance across all three offsets.
""")

    # 9. Feature Engineering
    sections.append(r"""## 9. Feature Engineering

### 9.1. Elimination of Nominal Price Drift
A primary driver of false discovery in financial ML is the inclusion of raw price levels ($P_t, \text{SMA}_{20}(t)$) as input features. Because equity prices exhibit non-stationary unit-root behavior with secular upward drift over multi-decade cycles, models trained on raw prices suffer catastrophic domain shift when predicting future market regimes.

To ensure stationarity and cross-asset transferability, we engineer **52 scale-free, dimensionless indicator transformations**:

1. **Percentage Moving Average Deviations**:
   $$\Delta_{\text{SMA}k}(t) = \frac{P_t - \text{SMA}_k(P, t)}{\text{SMA}_k(P, t)}, \quad k \in \{5, 10, 20, 50, 200\}$$

2. **Moving Average Spread Oscillators**:
   $$\text{Spread}_{20, 50}(t) = \frac{\text{SMA}_{20}(P, t) - \text{SMA}_{50}(P, t)}{\text{SMA}_{50}(P, t)}$$

3. **Leakage-Free Trailing Rolling Z-Scores**:
   $$z_t = \frac{x_t - \mu_{t-1}(W)}{\sigma_{t-1}(W)}, \quad W = 60 \text{ trading days}$$
   *Crucially, the mean $\mu_{t-1}$ and standard deviation $\sigma_{t-1}$ are estimated strictly over trailing historical windows ending on day $t-1$, preventing contemporaneous boundary leakage.*

4. **Edge-Safe Relative Strength Index (RSI)**:
   $$\text{RSI}_t = 100 - \frac{100}{1 + \text{RS}_t}, \quad \text{RS}_t = \frac{\text{EMA}_{14}(\text{Gain})}{\text{EMA}_{14}(\text{Loss})}$$
   *Boundary conditions are explicitly handled to prevent division-by-zero during consecutive winning streaks.*

5. **Normalized Moving Average Convergence Divergence (MACD)**:
   $$\text{MACD}_{\text{norm}}(t) = \frac{\text{EMA}_{12}(P, t) - \text{EMA}_{26}(P, t)}{P_t}$$

6. **Normalized Bollinger Bands**:
   $$\%B_t = \frac{P_t - \text{Lower}_t}{\text{Upper}_t - \text{Lower}_t}, \quad \text{Bandwidth}_t = \frac{\text{Upper}_t - \text{Lower}_t}{\text{SMA}_{20}(P, t)}$$

7. **Normalized Ichimoku Kinko Hyo Indicators**:
   $$\text{Tenkan-Kijun Spread} = \frac{\text{Tenkan}_t - \text{Kijun}_t}{P_t}, \quad \text{Cloud Thickness} = \frac{\text{Span A}_t - \text{Span B}_t}{P_t}$$

8. **Realized Volatility Ratios**: Ratios of short-term (5-day, 10-day) to long-term (20-day, 60-day) log-return variance.
""")

    # 10. Methodology and Model Architecture
    sections.append(r"""## 10. Methodology and Model Architecture

### 10.1. Seven-Model Benchmarking Suite
To investigate whether complex non-linear models discover technical interactions missed by linear models, seven distinct architectures are benchmarked:
1. **$L_2$-Regularized Logistic Regression**: A convex linear classifier with penalty parameter $C=0.1$:
   $$\min_{\mathbf{w}, b} \frac{1}{2} \|\mathbf{w}\|^2 + C \sum_{i=1}^N \log\left(1 + \exp\left(-y_i (\mathbf{w}^T \mathbf{x}_i + b)\right)\right)$$
2. **Random Forest (RF)**: A bagging ensemble of 500 de-correlated decision trees with $\text{min\_samples\_leaf}=20$ and feature subsampling $\sqrt{p}$ [19].
3. **Histogram Gradient Boosting (Scikit-Learn GBDT)**: Tree boosting with shallow depth ($\text{max\_leaf\_nodes}=15$) and learning rate $\eta=0.05$ [20].
4. **XGBoost**: Extreme gradient boosting with second-order Taylor loss approximations and $L_2$ leaf regularization [21].
5. **LightGBM**: Gradient-based one-side sampling (GOSS) and exclusive feature bundling for optimal histogram discretization [22].
6. **PyTorch LSTM**: A 2-layer stacked Long Short-Term Memory network (hidden dimension = 64, recurrent dropout $p=0.2$) processing 30-day sequential rolling windows [23].
7. **PyTorch GRU**: A 2-layer stacked Gated Recurrent Unit network optimized for parameter parsimony on noisy sequential inputs [24].

### 10.2. Honest Baseline Hierarchy
Every evaluated model is strictly anchored against four non-trivial baselines:
1. **Training Majority Baseline**: The prevailing frequency of the dominant directional class within the training folds.
2. **Naive Persistence Baseline**: Predicting that tomorrow's direction equals today's observed direction ($\hat{y}_{t+1} = y_t$).
3. **Zero-Return Benchmark ($\hat{r}=0$)**: For continuous return regression, predicting zero change, which minimizes MSE for a martingale difference sequence.
4. **Stratified Random Baseline**: Drawing predictions from a Bernoulli distribution matching historical upward frequencies.
""")

    # 11. Three-Day Directional Prediction
    sections.append(r"""## 11. Three-Day Directional Prediction

### 11.1. Expanding Walk-Forward Validation Scheme
To replicate real-world algorithmic trading without lookahead bias, models are evaluated using an **expanding 5-fold walk-forward cross-validation** harness:
- Fold 1: Train on initial 35% of history, evaluate on subsequent test block.
- Folds 2–5: Expand training window to incorporate preceding test data, retraining models from scratch.

![Figure 2: Walk-Forward Validation & Embargo](figures/fig2_walk_forward_embargo.png)
*Figure 2: Expanding 5-fold walk-forward cross-validation scheme with 52-trading-day (17-subsampled-bar) purged embargo buffer.*

### 11.2. The 52-Trading-Day Purged Embargo Buffer
A critical requirement identified by de Prado [6] is the implementation of an **embargo buffer**. Because technical indicators utilize trailing lookbacks of up to 52 days (the lookback for Ichimoku Senkou Span B), training samples immediately preceding the test period share historical data points with the earliest test samples.

Furthermore, in forward-horizon prediction ($h=3$), training rows at the very end of a fold project into future days. To prevent boundary contamination, we drop **52 trading days (17 rows in subsampled space)** from the end of each training set before out-of-sample testing begins. This accomplishes two objectives:
1. **Purging**: Eliminates forward target overlap across the train/test seam.
2. **Decorrelation Gap**: Acts as an econometric buffer preventing the model from exploiting transient autoregressive errors across the split boundary.
""")

    # 12. Entity-Targeted Financial Sentiment Analysis
    sections.append(r"""## 12. Entity-Targeted Financial Sentiment Analysis

### 12.1. Sentence-Pair FinBERT Architecture
To resolve the multi-entity attribution failure of generic financial NLP, we formulate sentiment classification as a sentence-pair inference task:
$$\text{Input Sequence} = \text{[CLS]} \circ \text{Target Entity} \circ \text{[SEP]} \circ \text{Headline Text} \circ \text{[SEP]}$$

By placing the target company name in Segment A and the headline in Segment B, the bidirectional cross-attention mechanism forces BERT's self-attention heads to attend specifically to the syntactic modifiers and predicates associated with the queried firm.

```
┌──────────────────────────────────────────────────────────────────────────────────┐
│                   SENTENCE-PAIR ENTITY FINBERT CLASSIFIER                        │
└──────────────────────────────────────────────────────────────────────────────────┘
  [CLS]   Target Company   [SEP]         Headline Context Text            [SEP]
    │           │            │                     │                        │
  ┌──────────────────────────────────────────────────────────────────────────────┐
  │                 12-Layer Bidirectional Transformer (BERT-Base)               │
  │                  Cross-Attention Heads Attend to Entity Modifiers            │
  └──────────────────────────────────────────────────────────────────────────────┘
                                        │
                                      [CLS]
                                        │
                         ┌──────────────┴──────────────┐
                         ▼                             ▼
                 Dense Linear Layer            Softmax Classifier
                         │                             │
                         ▼                             ▼
                 [Positive, Negative, Neutral] Directed Polarity
```

### 12.2. Model Fine-Tuning and Optimization
The model was initialized from `ProsusAI/finbert` and fine-tuned on the SEntFiN 1.0 training split (11,524 pairs) over 3 epochs using:
- **Optimizer**: AdamW ($\beta_1 = 0.9, \beta_2 = 0.999, \epsilon = 10^{-8}$) with weight decay 0.01.
- **Learning Rate**: $2 \times 10^{-5}$ with linear warmup over the first 10% of steps followed by linear decay.
- **Batch Size**: 32; Maximum sequence length: 128 tokens; Mixed-precision FP16 training.
""")

    # 13. Overnight Forecasting
    sections.append(r"""## 13. Overnight Forecasting

While technical indicators operate on closing prices ($P_{i,t}$), modern electronic markets experience substantial overnight price discovery. Between the 15:30 IST close of Indian equity markets and the 09:15 IST opening bell the following morning, global capital markets assimilate substantial information:
1. **US Markets**: S&P 500 and Dow Jones trade until 01:30 IST (summer) / 02:30 IST (winter).
2. **Asian Markets**: Nikkei 225 and KOSPI begin trading at 05:30 IST, several hours before Indian markets open.

To test whether overnight inter-market transmissions provide exploitable opening predictability, we construct a next-morning opening forecast engine ($N=27,179$ sessions across 10 equities and SENSEX, 2013–2025). The target is the direction of the opening price relative to the previous close:
$$y_{i,t}^{\text{open}} = \mathbb{I}\left(P_{i,t}^{\text{open}} > P_{i,t-1}^{\text{close}}\right)$$

Models are trained on overnight US index returns, early morning Asian returns, India VIX changes, and overnight regulatory disclosures.

### 13.1. Market Microstructure and Execution Considerations
A critical econometric caveat must be emphasized regarding opening price forecasting:
The official NSE opening price is determined via the pre-open call auction held between 09:00 and 09:08 IST. Market participants cannot purchase shares at yesterday's close $P_{i,t-1}^{\text{close}}$ after observing overnight US and Asian market movements; the pre-open auction incorporates global information before standard continuous trading commences at 09:15 IST. Furthermore, executing market orders at the open pays $P_{i,t}^{\text{open}}$, meaning that the opening price gap has already been realized and absorbed into the asset's price. Attempting to trade following the open exposes capital to opening bid-ask spreads (typically 10–30 bps in Indian cash equities), execution slippage, and daytime intraday mean reversion. Consequently, the 70.57% directional forecast accuracy reported in Section 16.9 represents an econometric measure of international information transmission, rather than an exploitable, risk-free trading arbitrage.
""")

    # 14. Walk-Forward Validation and Statistical Testing
    sections.append(r"""## 14. Walk-Forward Validation and Statistical Testing

### 14.1. Paired McNemar Test for Sentiment Ablation
To determine whether adding entity sentiment ($B + C_{\text{ent}}$) provides a statistically significant improvement over baseline features ($B$), we utilize the paired **McNemar test with Edward's continuity correction** [27]. Given the contingency table of out-of-sample predictions:
- $b$: Number of sessions where Baseline is correct and Augmented is incorrect.
- $c$: Number of sessions where Baseline is incorrect and Augmented is correct.

The test statistic is formulated as:
$$\chi^2 = \frac{(|b - c| - 1)^2}{b + c} \sim \chi_1^2$$

### 14.2. Monte Carlo Permutation Testing and its Power Floor
To test whether observed out-of-sample performance exceeds random chance, the walk-forward evaluation incorporates Monte Carlo label permutations. Targets $y_{i,t}$ are randomly shuffled within each fold while preserving feature correlation structures.

Crucially, in the walk-forward production pipeline (`src/pipeline.py`), permutation testing was constrained to $n=6$ shuffles per fold to bound computational runtime across multi-decade retrains. Incorporating the standard Laplace continuity correction ($+1$), the minimum attainable empirical $p$-value is:
$$p_{\min} = \frac{0 + 1}{6 + 1} = \frac{1}{7} \approx 0.1429$$

This discrete floor means that the pipeline's permutation screen **cannot mathematically resolve conventional $\alpha = 0.05$ significance thresholds**. It functions strictly as a **coarse diagnostic check against catastrophic feature leakage** rather than a high-resolution hypothesis test.

### 14.3. Paired Hypothesis Testing for Overnight Gap Forecasting
To evaluate the statistical significance of next-morning opening gap predictability ($N=27,179$), the machine learning model (LightGBM) is evaluated against a strictly out-of-fold **Naive Training Majority Baseline**:
1. **Out-of-Fold Baseline Construction**: For each test year $y \in [2013, 2025]$, the naive majority guess prediction $\hat{y}_{\text{guess}, t}$ is determined solely by the majority class of historical training observations strictly prior to year $y$ ($t < y\text{-01-01} - 7\text{ days}$). No future test data from year $y$ influences the baseline.
2. **Paired Observation Structure**: Model predictions $\hat{y}_{\text{model}, t}$ and baseline predictions $\hat{y}_{\text{guess}, t}$ are evaluated synchronously on the exact same dates and stock sessions ($N=27,179$).
3. **Contingency Formulation**:
   - $n_{11} = 14,094$: Both model and naive baseline are correct.
   - $n_{00} = 4,835$: Both model and naive baseline are incorrect.
   - $n_{01} = 5,085$: Naive baseline is incorrect, but LightGBM model is correct.
   - $n_{10} = 3,165$: Naive baseline is correct, but LightGBM model is incorrect.
   - Total discordant pairs: $n_{\text{disc}} = n_{01} + n_{10} = 8,250$.
4. **Hypothesis Framework**:
   - **Null Hypothesis ($H_0$)**: The model and naive baseline have identical marginal classification accuracies; the discordant prediction outcomes are equally likely, $P(n_{01}) = P(n_{10}) = 0.5$.
   - **Alternative Hypothesis ($H_1$)**: The model and naive baseline have differing marginal classification accuracies; discordant prediction outcomes are not equally likely, $P(n_{01}) \ne P(n_{10})$ (two-sided). Because $n_{01} = 5,085 > n_{10} = 3,165$, the observed deviation strictly favors the model under the two-sided test.
5. **Statistical Test Statistics**:
   - McNemar's continuity-corrected chi-square: $\chi^2 = \frac{(|5085 - 3165| - 1)^2}{8250} = \mathbf{446.37}$ ($df = 1, p = 4.45 \times 10^{-99}$).
   - Exact two-sided binomial test via `scipy.stats.binomtest(5085, 8250, 0.5, alternative='two-sided')`: $p = \mathbf{5.68 \times 10^{-100}}$.
   - **Reporting Convention**: In compliance with mathematical standards prohibiting non-degenerate $p = 0.0$ claims, this astronomical significance is reported as $p < 10^{-15}$.
""")

    # 15. Transaction-Cost-Aware Trading Strategies
    sections.append(r"""## 15. Transaction-Cost-Aware Trading Strategies

To bridge machine learning classification and practical investment execution, five distinct trading strategy variants were implemented and backtested across all ten equities over an out-of-sample period spanning **July 2013 to August 2026 (13.1 years)**, corresponding to the period of continuous availability of machine-parsed regulatory disclosures and inter-market features:

- **Variant A (Baseline Long-Only)**: Enters a 100% long position when the predicted probability of an upward move exceeds 50% ($P(\text{Up}) > 0.50$) and the model's gross expected return exceeds the statutory transaction cost hurdle.
- **Variant B (Risk-Managed Long-Only)**: Extends Variant A by enforcing a strict 2% trailing stop-loss and a mandatory 5-day time-based exit, cutting tail-risk exposure.
- **Variant C (Dynamic Long/Short)**: Allows both long and short participation based on directional conviction, shorting when $P(\text{Up}) < 0.45$.
- **Variant D (Multi-Factor Consensus)**: Demands simultaneous alignment between three independent signals: ML model probability ($P(\text{Up}) > 0.55$), technical oscillators (RSI $< 65$ and positive MACD histogram), and positive FinBERT news sentiment.
- **Variant E (Regime-Adaptive Allocation)**: Utilizes trailing 20-day realized volatility and moving-average regime filters, actively taking equity positions only during favorable market regimes and allocating 100% to risk-free cash during elevated volatility regimes.

### 15.1. Indian Market Statutory Transaction Frictions
All backtests incorporate an explicit **15 basis points (0.15%) round-trip friction hurdle**, reflecting actual Indian institutional execution costs:
- Securities Transaction Tax (STT): 0.10% on delivery turnover.
- Exchange Turnover Fees, GST, and SEBI Turnover Charges: ~0.015%.
- Bid-Ask Spread and Execution Slippage: ~0.035%.
""")

    # 16. Experimental Results
    sections.append(r"""## 16. Experimental Results

### 16.1. Directional Prediction Performance Across 10 Equities
Table 1 summarizes the out-of-sample walk-forward evaluation across all ten equities ($N=17,436$ pooled subsampled observations).

### Table 1: Multi-Decade Walk-Forward Directional Prediction Performance Across 10 NSE Equities (h=3 Days)

| Equity Ticker | Historical Data Start | Subsampled Samples ($N$) | Walk-Forward Accuracy ($\pm \sigma_{\text{fold}}$) | Training Majority Baseline | Net Predictive Edge | Monte Carlo Permutation Verdict |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **ADANIENT** | 2004-07 | 1,773 | 0.5310 $\pm$ 0.0199 | 0.5320 | $-0.0010$ | NO SIGNAL |
| **ADANIGREEN** | 2018-06 | 619 | 0.4839 $\pm$ 0.0684 | 0.4871 | $-0.0032$ | NO SIGNAL |
| **ETERNAL** | 2021-07 | 375 | 0.5636 $\pm$ 0.0357 | 0.5420 | $+0.0216$ | NO SIGNAL |
| **HDFCBANK** | 2000-01 | 2,150 | 0.5098 $\pm$ 0.0325 | 0.5274 | $-0.0177$ | NO SIGNAL |
| **INFY** | 2000-01 | 2,154 | 0.5395 $\pm$ 0.0363 | 0.5525 | $-0.0131$ | NO SIGNAL |
| **ITC** | 2000-01 | 2,151 | 0.5149 $\pm$ 0.0194 | 0.4935 | $+0.0214$ | NO SIGNAL |
| **RELIANCE** | 2000-01 | 2,151 | 0.4972 $\pm$ 0.0121 | 0.5186 | $-0.0214$ | NO SIGNAL |
| **SBI** | 2000-01 | 2,150 | 0.5153 $\pm$ 0.0344 | 0.5395 | $-0.0242$ | NO SIGNAL |
| **TCS** | 2004-08 | 1,767 | 0.5070 $\pm$ 0.0292 | 0.5149 | $-0.0079$ | NO SIGNAL |
| **VEDL** | 2000-01 | 2,146 | 0.4873 $\pm$ 0.0612 | 0.5358 | $-0.0485$ | NO SIGNAL |
| **Total / Pooled** | 2000–2026 | **17,436** | 0.5129 $\pm$ 0.0234 | 0.5243 | $-0.0114$ | **10 / 10 NO SIGNAL** |

*Note: Models utilize $L_2$-regularized Logistic Regression ($C=0.1$) on stationary technical indicators. Net edge represents Accuracy minus Majority Baseline. Permutation testing applies $n=6$ shuffles per fold with Laplace continuity correction.*

Across the full sample comprising **17,436 non-overlapping observations**, **not a single stock demonstrates a directional edge that clearly separates from shuffled label permutations**. Every ticker receives a verdict of `NO SIGNAL` under the walk-forward diagnostic screen. For mature multi-decade listings with over 2,100 subsampled periods (HDFCBANK, INFY, ITC, RELIANCE, SBI, VEDL), walk-forward accuracies converge tightly between **48.73% and 53.95%**, falling short of their training majority base rates (51.8% to 55.3%). Across the complete ten-equity panel (including younger listings ADANIGREEN and ETERNAL), directional accuracies span 48.39% to 56.36%, with zero equities achieving statistical significance over baseline.

### 16.2. Baseline Benchmarking and the Three Accuracy Illusions
Table 2 details the comparative performance of baseline rules against machine learning models.

### Table 2: Benchmark Comparison of Baseline Strategies vs. Trained Models Across Experimental Setups

| Experimental Setup | Evaluated Model | Model Nominal Accuracy | Prevailing Majority Baseline | Naive Persistence Baseline | Net Edge over Best Baseline | Effective Predictive Signal |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **INFY 3-Day Direction ($h=3$)** | Logistic Regression | 53.95% | **55.25%** | 50.12% | $-1.31\%$ | Negative Edge (Sub-baseline) |
| **RELIANCE 3-Day Direction ($h=3$)** | Logistic Regression | 49.72% | **51.86%** | 49.65% | $-2.14\%$ | Negative Edge (Sub-baseline) |
| **SENSEX Next-Day ($h=1$, Technical $T$)** | Logistic Regression | 51.54% | **53.11%** | 50.84% | $-1.57\%$ | Negative Edge (Sub-baseline) |
| **SENSEX Next-Day ($h=1$, $T+G$)** | Random Forest | **58.68%** | 53.11% | 51.20% | $+5.57\%$ | Positive Edge (Macro Spillover) |
| **SENSEX Big Moves ($|\Delta| > 1.5\%$)** | XGBoost ($T+G+S$) | 70.94% | **71.37%** | 58.20% | $-0.43\%$ | Negative Edge (Majority Artifact) |
| **10-Day Moving Average Direction** | Walk-Forward GBDT | **86.43%** | 54.20% | 86.43% | $\pm 0.00\%$ | Spurious Autocorrelation Trap |
| **Same-Day Lookahead (Today's News)** | XGBoost ($T+G+S_0$) | **66.20%** | 53.12% | 51.04% | $+13.08\%$ | Contemporaneous Lookahead Trap |

These comparisons expose the **Three Accuracy Illusions**:
1. **The Majority Class Trap**: 70.94% accuracy on big moves appears impressive, but the baseline of non-occurrence is 71.37%. The model has a negative net edge ($-0.43\%$).
2. **The Target Autocorrelation Trap**: 86.43% accuracy on rolling moving average direction simply repeats yesterday's direction because 9 of 10 days overlap ($\rho = 0.806$).
3. **The Lookahead Mirage**: 66.20% accuracy pairing same-day news with closing returns reflects retrospective description, not tradable forward forecasting.

### 16.3. Seven-Architecture Benchmarking and Architecture Parity
Table 3 compares all seven architectures on the BSE SENSEX across 3,248 walk-forward sessions (2009–2022).

### Table 3: Performance Comparison Across Seven Architectures on SENSEX Next-Day Directional Forecasting

| Architecture | Model Family | Feature Set | Test Accuracy | ROC-AUC | Balanced Accuracy | Net Edge vs. Majority Base | Sentiment Ablation Effect ($\Delta_{\text{sent}}$) | Sentiment McNemar Test ($p$-value) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | Linear / Convex | $T$ | 51.54% | 0.5233 | 50.65% | $-1.57\%$ | $+0.55\%$ | $p = 0.548$ (n.s.) |
| **Random Forest** | Bagging Ensemble | $T$ | 53.60% | 0.5422 | 50.19% | $+0.49\%$ | $-1.29\%$ | $p = 0.069$ (n.s.) |
| **Gradient Boosting** | Boosting Ensemble | $T$ | 52.71% | 0.5344 | 51.55% | $-0.40\%$ | $+0.37\%$ | $p = 0.690$ (n.s.) |
| **XGBoost** | Boosting Ensemble | $T$ | 53.02% | 0.5367 | 51.79% | $-0.09\%$ | $-0.34\%$ | $p = 0.708$ (n.s.) |
| **LightGBM** | Boosting Ensemble | $T$ | 52.43% | 0.5343 | 51.57% | $-0.68\%$ | $+0.12\%$ | $p = 0.914$ (n.s.) |
| **PyTorch LSTM** | Recurrent Sequence | $T$ | 51.91% | 0.5215 | 50.82% | $-1.20\%$ | $+0.25\%$ | $p = 0.837$ (n.s.) |
| **PyTorch GRU** | Recurrent Sequence | $T$ | 52.89% | 0.5309 | 51.14% | $-0.22\%$ | $-1.08\%$ | $p = 0.340$ (n.s.) |
| **Logistic Regression** | Linear / Convex | $T+G$ | 58.59% | 0.6110 | 58.45% | $+5.48\%$ | $-0.12\%$ | $p = 0.889$ (n.s.) |
| **Random Forest** | Bagging Ensemble | $T+G$ | **58.68%** | **0.6172** | 58.36% | **$+5.57\%$** | $+0.25\%$ | $p = 0.533$ (n.s.) |
| **Gradient Boosting** | Boosting Ensemble | $T+G$ | 57.42% | 0.6068 | 57.20% | $+4.31\%$ | $+1.05\%$ | $p = 0.130$ (n.s.) |
| **XGBoost** | Boosting Ensemble | $T+G$ | 57.36% | 0.6053 | 57.18% | $+4.25\%$ | $+0.99\%$ | $p = 0.160$ (n.s.) |
| **LightGBM** | Boosting Ensemble | $T+G$ | 57.64% | 0.6048 | 57.44% | $+4.53\%$ | $-0.40\%$ | $p = 0.605$ (n.s.) |
| **PyTorch LSTM** | Recurrent Sequence | $T+G$ | 58.13% | 0.6140 | 57.88% | $+5.02\%$ | $-0.37\%$ | $p = 0.695$ (n.s.) |
| **PyTorch GRU** | Recurrent Sequence | $T+G$ | 58.34% | 0.6202 | 58.10% | $+5.23\%$ | $+0.49\%$ | $p = 0.567$ (n.s.) |

![Figure 3: Model Architecture Comparison](figures/fig3_model_architecture_comparison.png)
*Figure 3: Out-of-sample performance parity across seven architectures on SENSEX next-day directional forecasting under Technical ($T$) and Technical + Global Inter-market ($T+G$) feature sets.*

Under technical indicators alone ($T$), all seven models perform within a tight 2.06 pp band (51.54% to 53.60%). Under inter-market features ($T+G$), all models improve to 57.36%–58.68% (a spread of only 1.32 pp). A simple $L_2$-regularized Logistic Regression model ($C=0.1$) achieves 58.59% accuracy and 0.6110 AUC, matching Random Forest (58.68%) while exhibiting superior offset phase stability across $k \in \{0, 1, 2\}$ (`reports/phase4_offset_sensitivity.csv`).

### 16.4. Deep Learning Sequence Diagnostics (LSTM & GRU)
Table 4 presents the diagnostic evaluation of recurrent sequence architectures.

### Table 4: Deep Learning Sequence Model Performance and Diagnostic Failure Analysis

| Asset / Market Scope | Architecture | Model Formulation | Directional Accuracy | Net Edge vs. Base | Continuous RMSE | Naive Persistence RMSE | Zero-Return Benchmark RMSE | Prediction Shrinkage Factor |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **TCS (2004–2026)** | PyTorch LSTM | 3-Day Return Regressor | 49.33% | $-2.46\%$ | 0.013117 | 0.018418 | **0.013069** | **0.0962** |
| **TCS (2004–2026)** | PyTorch GRU | 3-Day Return Regressor | 51.24% | $-0.55\%$ | 0.013248 | 0.018418 | **0.013069** | **0.1196** |
| **RELIANCE (2000–2026)** | PyTorch LSTM | 3-Day Return Regressor | 50.22% | $-1.64\%$ | 0.009109 | 0.012354 | **0.008930** | **0.1849** |
| **RELIANCE (2000–2026)** | PyTorch GRU | 3-Day Return Regressor | 50.65% | $-1.21\%$ | 0.009133 | 0.012354 | **0.008930** | **0.1926** |
| **India SENSEX (Next-Day)** | PyTorch LSTM | 1-Day Classifier ($T$) | 51.91% | $-1.20\%$ | — | — | — | — |
| **India SENSEX (Next-Day)** | PyTorch LSTM | 1-Day Classifier ($T+S$) | 52.16% | $-0.95\%$ | — | — | — | — |
| **India SENSEX (Next-Day)** | PyTorch GRU | 1-Day Classifier ($T$) | 52.89% | $-0.22\%$ | — | — | — | — |
| **India SENSEX (Next-Day)** | PyTorch GRU | 1-Day Classifier ($T+S$) | 51.82% | $-1.29\%$ | — | — | — | — |
| **US S&P 500 (Next-Day)** | PyTorch LSTM | 1-Day Classifier ($T$) | 53.34% | $-0.36\%$ | — | — | — | — |
| **US S&P 500 (Next-Day)** | PyTorch LSTM | 1-Day Classifier ($T+S$) | 53.36% | $-0.34\%$ | — | — | — | — |
| **US S&P 500 (Next-Day)** | PyTorch GRU | 1-Day Classifier ($T$) | 53.43% | $-0.27\%$ | — | — | — | — |
| **US S&P 500 (Next-Day)** | PyTorch GRU | 1-Day Classifier ($T+S$) | 52.51% | $-1.19\%$ | — | — | — | — |

![Figure 4: Deep Learning Variance Shrinkage](figures/fig4_deep_learning_shrinkage.png)
*Figure 4: Recurrent sequence model diagnostics: (A) Prediction variance shrinkage ratio $\sigma(\hat{r})/\sigma(r)$, and (B) TCS 3-day return RMSE comparison showing failure to beat zero-return benchmark.*

The sequence diagnostics reveal two fatal failure modes:
1. **Failure to Beat Zero-Return Benchmark**: Neither LSTM nor GRU outperforms a trivial forecast of zero return ($\hat{r}=0$, `beats_zero = False`). On TCS, $\hat{r}=0$ achieves an RMSE of 0.013069, superior to LSTM (0.013117) and GRU (0.013248). On RELIANCE, zero forecast achieves 0.008930, superior to LSTM (0.009109) and GRU (0.009133).
2. **Extreme Variance Shrinkage**: Predictions compress to a narrow band around zero (shrinkage factor 9.6% to 19.3% of realized volatility). Backpropagation minimizes $L_2$ loss by avoiding directional commitments in low-SNR data.

### 16.5. Entity-Targeted FinBERT on SEntFiN Benchmark
Table 5 evaluates the sentence-pair Entity FinBERT on the SEntFiN 1.0 test split (1,440 pairs).

### Table 5: Entity-Targeted FinBERT vs. Standard Generic FinBERT on SEntFiN 1.0 Benchmark

| Test Benchmark Evaluation Split | Sample Size ($N$) | Evaluation Metric | Standard Generic FinBERT | Fine-Tuned Entity FinBERT | Net Performance Improvement |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Full SEntFiN Test Set** | 1,440 pairs | Classification Accuracy | 68.54% | **87.71%** | **$+19.17\%$ pts** |
| **Full SEntFiN Test Set** | 1,440 pairs | Macro-Averaged $F_1$ | 0.6870 | **0.8783** | **$+0.1913$** |
| **Multi-Entity Headlines ($\ge 2$ Companies)** | 614 pairs | Classification Accuracy | 61.56% | **85.34%** | **$+23.78\%$ pts** |
| **Multi-Entity Headlines ($\ge 2$ Companies)** | 614 pairs | Macro-Averaged $F_1$ | 0.6122 | **0.8527** | **$+0.2405$** |

![Figure 5: Entity FinBERT Performance](figures/fig5_entity_finbert_sentfin.png)
*Figure 5: Entity-Targeted FinBERT vs. Generic FinBERT performance on SEntFiN 1.0 benchmark: (A) Classification accuracy, and (B) Macro-averaged F1 score.*

Entity-Targeted FinBERT achieves **87.71% accuracy** (+19.17 pp gain) and **0.8783 macro-$F_1$** on the full test set. On headlines mentioning two or more competing firms (614 pairs), where standard FinBERT degrades to near-chance (61.56%), Entity FinBERT maintains **85.34% accuracy** (+23.78 pp gain), resolving polarity attribution ambiguity.

### 16.6. Company-Targeted Sentiment Ablation Across 10 Equities
Table 6 summarizes the paired ablation across 20 equity-corpus conditions.

### Table 6: Company-Targeted Sentiment Ablation Across 10 Equities (Times of India Headlines & NSE Filings)

| Stock Ticker | Text Corpus Window | Subsampled Rows | News Days ($N_{\text{news}}$) | Accuracy Without Sent ($B$) | Accuracy With Sent ($B+C_{\text{ent}}$) | Accuracy Delta ($\Delta$) | ROC-AUC Without Sent | ROC-AUC With Sent | Net Edge over Majority | Paired McNemar Test ($p$-value) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **TCS** | ToI Headlines | 1,405 | 959 | 53.08% | 53.06% | $-0.02\%$ | 0.5160 | 0.5059 | $-0.70\%$ | $p = 1.000$ |
| **TCS** | NSE Filings | 1,492 | 1,124 | 50.95% | 51.48% | $+0.53\%$ | 0.5216 | 0.5262 | $+0.27\%$ | $p = 0.694$ |
| **RELIANCE** | ToI Headlines | 1,790 | 2,127 | 50.17% | 50.39% | $+0.22\%$ | 0.4982 | 0.4969 | $-1.79\%$ | $p = 0.864$ |
| **RELIANCE** | NSE Filings | 1,490 | 784 | 49.13% | 49.53% | $+0.40\%$ | 0.4893 | 0.4952 | $-0.81\%$ | $p = 0.766$ |
| **INFY** | ToI Headlines | 1,793 | 1,682 | 55.86% | 55.86% | $\pm 0.00\%$ | 0.5596 | 0.5599 | $-0.20\%$ | $p = 1.000$ |
| **INFY** | NSE Filings | 1,491 | 918 | 54.96% | 55.23% | $+0.27\%$ | 0.5494 | 0.5425 | $+1.07\%$ | $p = 0.856$ |
| **HDFCBANK** | ToI Headlines | 1,791 | 316 | 51.56% | 51.68% | $+0.12\%$ | 0.5182 | 0.5134 | $-1.45\%$ | $p = 1.000$ |
| **HDFCBANK** | NSE Filings | 1,490 | 336 | 53.02% | 52.89% | $-0.13\%$ | 0.5159 | 0.5186 | $+0.54\%$ | $p = 1.000$ |
| **SBI** | ToI Headlines | 1,789 | 1,479 | 49.50% | 51.96% | $+2.46\%$ | 0.5075 | 0.5222 | $-1.12\%$ | $p = 0.115$ |
| **SBI** | NSE Filings | 1,489 | 615 | 52.35% | 53.42% | $+1.07\%$ | 0.5468 | 0.5443 | $+3.36\%$ | $p = 0.341$ |
| **ITC** | ToI Headlines | 1,790 | 377 | 51.62% | 51.73% | $+0.11\%$ | 0.5218 | 0.5294 | $+1.01\%$ | $p = 1.000$ |
| **ITC** | NSE Filings | 1,486 | 303 | 48.06% | 49.54% | $+1.48\%$ | 0.4991 | 0.5005 | $-0.27\%$ | $p = 0.135$ |
| **ADANIENT** | ToI Headlines | 1,412 | 470 | 52.27% | 51.28% | $-0.99\%$ | 0.5307 | 0.5252 | $+7.64\%$ | $p = 0.265$ |
| **ADANIENT** | NSE Filings | 1,492 | 510 | 49.20% | 48.26% | $-0.94\%$ | 0.4916 | 0.4910 | $-4.82\%$ | $p = 0.337$ |
| **VEDL** | ToI Headlines | 1,785 | 407 | 49.71% | 50.61% | $+0.90\%$ | 0.4928 | 0.4862 | $-0.76\%$ | $p = 0.374$ |
| **VEDL** | NSE Filings | 1,488 | 611 | 50.79% | 50.12% | $-0.67\%$ | 0.4951 | 0.4887 | $+0.82\%$ | $p = 0.653$ |
| **ETERNAL** | ToI Headlines | 16 | 4 | — | — | — | — | — | — | Insufficient Data |
| **ETERNAL** | NSE Filings | 375 | 130 | 56.36% | 55.86% | $-0.50\%$ | 0.5575 | 0.5585 | $+4.49\%$ | $p = 1.000$ |
| **ADANIGREEN**| ToI Headlines | 263 | 28 | 56.37% | 54.07% | $-2.30\%$ | 0.4911 | 0.4580 | $-1.32\%$ | $p = 0.375$ |
| **ADANIGREEN**| NSE Filings | 619 | 192 | 48.39% | 48.06% | $-0.33\%$ | 0.4934 | 0.4897 | $+1.29\%$ | $p = 1.000$ |

![Figure 6: Sentiment Ablation Forest Plot](figures/fig6_sentiment_ablation_forest.png)
*Figure 6: Company-targeted sentiment ablation forest plot across 20 equity-corpus conditions (all p >= 0.115 under paired McNemar tests; 0 of 20 conditions statistically significant).*

Adding company sentiment yields an unweighted mean delta of **$+0.09\%$** and a weighted mean delta of **$+0.28\%$**. Across all 20 conditions, **zero stocks achieve statistical significance** under the paired McNemar test ($p \ge 0.115$). In low-coverage stocks (ADANIGREEN, ADANIENT), sparse sentiment degrades performance by introducing noise. A 1-year temporal placebo test produced identical random noise ($\Delta \in [-0.011, +0.002]$, all $p > 0.40$).

### 16.7. Lead-Lag Asymmetry, Quintiles, and Explanatory Coverage
Tables 7, 8, and 9 characterize the temporal relationship between news and equity prices.

### Table 7: Lead-Lag Rank Correlation Profile (Spearman rho) Between News Sentiment and Equity Returns

| Market Scope / Asset | Evaluated News Subset | Preceding Return ($\text{Lag } -1$) | Same-Day Return ($\text{Lag } 0$) | Next-Day Return ($\text{Lag } +1$) | Same-Day Significance | Next-Day Significance |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **SENSEX Index** | All Headlines (Raw) | $+0.1152$ | $+0.1720$ | $+0.0205$ | $p < 10^{-14}$ | $p = 0.143$ (n.s.) |
| **SENSEX Index** | Price Reports Removed | $+0.0573$ | **$+0.0562$** | **$+0.0070$** | **$p = 0.00006$** | **$p = 0.618$ (n.s.)** |
| **TCS** | Price Reports Removed | $+0.0514$ | $+0.0340$ | $+0.0169$ | $p = 0.295$ | $p = 0.602$ (n.s.) |
| **RELIANCE** | Price Reports Removed | $+0.0428$ | $+0.0161$ | $-0.0225$ | $p = 0.463$ | $p = 0.304$ (n.s.) |
| **INFY** | Price Reports Removed | $+0.0621$ | $+0.0513$ | $+0.0330$ | $p = 0.038$ | $p = 0.181$ (n.s.) |
| **HDFCBANK** | Price Reports Removed | $+0.0210$ | $+0.0368$ | $+0.0123$ | $p = 0.515$ | $p = 0.828$ (n.s.) |
| **SBI** | Price Reports Removed | $+0.0380$ | $+0.0180$ | $+0.0230$ | $p = 0.492$ | $p = 0.378$ (n.s.) |
| **ITC** | Price Reports Removed | $+0.0194$ | $-0.0106$ | $-0.0565$ | $p = 0.838$ | $p = 0.274$ (n.s.) |
| **ADANIENT** | Price Reports Removed | $+0.0490$ | $+0.0643$ | $+0.1257$ | $p = 0.160$ | $p = 0.006$ |
| **VEDL** | Price Reports Removed | $+0.0312$ | $+0.0462$ | $+0.0217$ | $p = 0.353$ | $p = 0.662$ (n.s.) |
| **Pooled 10 Stocks** | Price Reports Removed | $+0.0415$ | **$+0.0324$** | **$+0.0131$** | **$p = 0.0044$** | **$p = 0.248$ (n.s.)** |

![Figure 7: Lead-Lag Cross-Correlation](figures/fig7_lead_lag_cross_correlation.png)
*Figure 7: Lead-lag rank cross-correlation profile (Spearman rho) illustrating contemporaneous peak at Lag 0 and forward collapse at Lag +1.*

### Table 8: Daily Mean Returns Across Clean News Sentiment Quintiles on the BSE SENSEX

| Sentiment Quintile | Number of Sessions ($N$) | Same-Day Mean Return | Next-Day Mean Return |
| :--- | :---: | :---: | :---: |
| **Quintile 1 (Most Negative)** | 1,024 | $-0.0564\%$ | $+0.0689\%$ |
| **Quintile 2** | 1,023 | $+0.0179\%$ | $-0.0262\%$ |
| **Quintile 3 (Neutral)** | 1,024 | $+0.0475\%$ | $+0.0875\%$ |
| **Quintile 4** | 1,023 | $+0.1292\%$ | $+0.1089\%$ |
| **Quintile 5 (Most Positive)** | 1,024 | $+0.1434\%$ | $+0.0343\%$ |
| **Spread: Quintile 5 minus Quintile 1** | — | **$+0.1998\%$ / day** | **$-0.0345\%$ / day** |

![Figure 8: Sentiment Quintile Returns](figures/fig8_sentiment_quintile_returns.png)
*Figure 8: Daily mean returns across sentiment quintiles on BSE SENSEX, showing same-day +0.20% spread vs. next-day collapse to -0.035%.*

### Table 9: Explanatory Coverage of Extreme Price Movements ($|\Delta| > 1.5\%$)

| Scope / Asset | Total Big Move Sessions | Matching News Share | Opposite News Share | Neutral News Share | No News Share (Unexplained) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **SENSEX Index** | 262 | 45.80% (120) | 39.31% (103) | 8.78% (23) | **6.11% (16)** |
| **TCS** | 217 | 11.06% (24) | 7.83% (17) | 15.67% (34) | **65.44% (142)** |
| **RELIANCE** | 263 | 12.93% (34) | 9.89% (26) | 24.34% (64) | **52.85% (139)** |
| **INFY** | 263 | 14.83% (39) | 6.08% (16) | 16.35% (43) | **62.74% (165)** |
| **HDFCBANK** | 263 | 1.52% (4) | 0.76% (2) | 0.76% (2) | **96.96% (255)** |
| **SBI** | 263 | 4.56% (12) | 4.56% (12) | 18.25% (48) | **72.62% (191)** |
| **ITC** | 263 | 2.28% (6) | 3.80% (10) | 2.66% (7) | **91.25% (240)** |
| **ADANIENT** | 219 | 3.65% (8) | 2.74% (6) | 5.94% (13) | **87.67% (192)** |
| **VEDL** | 263 | 2.66% (7) | 1.14% (3) | 2.66% (7) | **93.54% (246)** |

The lead-lag profile resolves the financial news paradox:
1. **Contemporaneous Absorption**: News sentiment correlates significantly with same-day returns ($\rho = +0.0562, p = 0.00006$ on SENSEX; $\rho = +0.0324, p = 0.0044$ pooled), yielding a $+0.20\%$ daily spread between top and bottom quintiles.
2. **Forward Collapse**: On the next day, correlation collapses to zero ($\rho = +0.0070, p = 0.618$), and the quintile spread vanishes ($-0.035\%$). News explains today; it does not forecast tomorrow.
3. **Unexplained Single-Stock Volatility**: While only 6.11% of SENSEX big moves lack news, **52.8% to 97.0% of single-stock extreme moves occur with zero company headlines in the major press**.

### 16.8. Controlled Demonstration of Literature Accuracy Pitfalls
Table 10 deconstructs how published papers generate artificial 70%–90% accuracies.

### Table 10: Controlled Demonstration of Literature Accuracy Pitfalls on Identical Market Data

| Experimental Evaluation Protocol | Valid Setup? | Reported Metric | Nominal Value | Mechanism of Artificial Performance Inflation |
| :--- | :---: | :---: | :---: | :--- |
| **Honest Out-of-Sample Walk-Forward ($T+G+S$)** | **Yes** | Direction Accuracy | **57.24%** | Rigorous expanding walk-forward CV, non-overlapping labels, strict embargo. |
| **Same-Day Headline Polarity vs. Return** | **No** | Direction Accuracy | **76.96%** | **Contemporaneous Lookahead**: FinBERT scores headlines describing today's close ($N=2,270$). |
| **Next-Day Implied by Same Headline Rule** | **Yes** | Direction Accuracy | **55.02%** | Proper alignment resets accuracy to unconditional base rate (~53%). |
| **5-Day Return Horizon with Shuffled 5-Fold CV** | **No** | Direction Accuracy | **60.95%** | **Data Leakage**: Random $k$-fold CV causes adjacent days to share 4 of 5 return days. |
| **5-Day Return with Walk-Forward & Embargo** | **Yes** | Direction Accuracy | **55.29%** | Embargo prevents boundary leakage; accuracy drops to majority base (~56%). |
| **10-Day Moving Average Direction (Walk-Forward)** | **No** | Direction Accuracy | **86.43%** | **Trivial Target**: Predicting $\text{SMA}_{10}(t+1) > \text{SMA}_{10}(t)$ when 9 of 10 prices are already known. |
| **Next-Day Price Level Linear Regression** | **No** | Coefficient ($R^2$) | **0.9980** | **Non-Stationarity**: Regressing $P_{t+1}$ on $P_t$ yields near-unity fit due to integrated unit-root drift. |
| **Next-Day Price Level Linear Regression** | **No** | Metric ($1 - \text{MAPE}$) | **0.9925** | Reported as "99.2% accuracy"; simply reflects that tomorrow's price is close to today's price. |
| **Direction Implied by Level Regression** | **Yes** | Direction Accuracy | **52.54%** | The level model that achieved 99.2% $1-\text{MAPE}$ has zero directional predictive power. |

![Figure 12: Accuracy Traps Deconstruction](figures/fig12_reproduced_accuracy_traps.png)
*Figure 12: Controlled deconstruction of the three literature accuracy traps comparing nominal reported metrics against genuine out-of-sample edge.*

### 16.9. Overnight Opening Price Forecasting Performance
Table 11 presents next-morning opening gap predictability ($N=27,179$).

### Table 11: Next-Morning Opening Price Forecast Performance Across Indian Equities (LightGBM, N=27,179)

| Ticker | Samples ($N$) | Period | Naive Guess Base | LightGBM Model | With News Model | Net Edge vs. Guess | $p$-value (vs. Guess) | News Effect ($\Delta$) | $p$-value (News) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **ADANIENT** | 3,069 | 2013–2025 | 67.81% | 72.27% | 72.37% | $+4.46\%$ | $p < 10^{-15}$ | $+0.10\%$ | $p = 0.804$ |
| **ADANIGREEN** | 1,580 | 2019–2025 | 71.96% | 71.71% | 72.15% | $-0.25\%$ | $p = 0.883$ | $+0.44\%$ | $p = 0.265$ |
| **ETERNAL** | 841 | 2022–2025 | 66.71% | 70.27% | 68.97% | $+3.57\%$ | $p = 0.049$ | $-1.31\%$ | $p = 0.027$ |
| **HDFCBANK** | 3,140 | 2013–2025 | 56.50% | 69.01% | 69.20% | $+12.52\%$ | $p < 10^{-15}$ | $+0.19\%$ | $p = 0.556$ |
| **INFY** | 3,122 | 2013–2025 | 58.49% | 68.00% | 68.26% | $+9.51\%$ | $p < 10^{-15}$ | $+0.26\%$ | $p = 0.456$ |
| **ITC** | 2,984 | 2013–2025 | 65.85% | 69.77% | 69.71% | $+3.92\%$ | $p = 0.0001$ | $-0.07\%$ | $p = 0.908$ |
| **RELIANCE** | 3,133 | 2013–2025 | 62.94% | 71.15% | 71.31% | $+8.20\%$ | $p < 10^{-15}$ | $+0.16\%$ | $p = 0.635$ |
| **SBI** | 3,102 | 2013–2025 | 66.34% | 74.08% | 74.31% | $+7.74\%$ | $p < 10^{-15}$ | $+0.23\%$ | $p = 0.483$ |
| **TCS** | 3,122 | 2013–2025 | 60.28% | 67.17% | 67.49% | $+6.89\%$ | $p < 10^{-15}$ | $+0.32\%$ | $p = 0.440$ |
| **VEDL** | 3,086 | 2013–2025 | 64.91% | 72.62% | 72.49% | $+7.71\%$ | $p < 10^{-15}$ | $-0.13\%$ | $p = 0.699$ |
| **POOLED (ALL)** | **27,179** | 2013–2025 | **63.50%** | **70.57%** | **70.67%** | **$+7.06\%$** | **$p < 10^{-15}$** | **$+0.11\%$** | **$p = 0.286$** |

*Note: The Naive Guess Baseline represents the unconditional proportion of trading sessions where the opening price exceeded the previous day's close. LightGBM improves upon this baseline to 70.57% ($+7.06\%$ net edge, $p < 10^{-15}$) by conditioning on overnight US index returns and early Asian market opens.*

![Figure 9: Overnight Forecast Performance](figures/fig9_overnight_forecast_performance.png)
*Figure 9: Next-morning opening price forecast accuracy across 10 Indian equities (LightGBM vs. naive guess baseline, N=27,179).*

In next-morning opening forecasting, LightGBM achieves **70.57% directional accuracy** vs. a 63.50% training-majority guess baseline (+7.06% net edge, $p < 10^{-15}$). HDFCBANK (+12.52%), INFY (+9.51%), and RELIANCE (+8.20%) display substantial edges. Crucially, adding company announcements produces an incremental gain of only $+0.11\%$ ($p = 0.286$), proving that opening gap direction is governed almost exclusively by overnight global macro spillovers.

*Methodological Note on Significance Testing and Numerical Precision*: In the underlying experiment logs (e.g., `reports/forecast/stock_summary.csv`), the paired statistical comparison between model accuracy and naive majority guess is evaluated using McNemar's paired test / exact binomial test via `scipy.stats.binomtest` on discordant prediction pairs across the synchronous 2x2 contingency table ($N=27,179$). As formally defined in Section 14.3:
- Concordant pairs: $n_{11} = 14,094$ (both correct), $n_{00} = 4,835$ (both wrong).
- Discordant pairs: $n_{01} = 5,085$ (guess wrong, model correct) vs. $n_{10} = 3,165$ (guess correct, model wrong); total discordant $n_{\text{disc}} = 8,250$.
- Continuity-corrected McNemar statistic: $\chi^2 = 446.37$ ($df=1$, $p = 4.45 \times 10^{-99}$).
- Exact two-sided binomial $p$-value: $p = 5.68 \times 10^{-100}$.

Under standard 5-decimal floating-point export formatting (`pd.DataFrame.round(5)`), this astronomical significance rounds to literal `0.0`. In accordance with rigorous mathematical and statistical reporting standards, we report this result as $p < 10^{-15}$ rather than a literal zero, as non-degenerate statistical hypothesis test $p$-values are strictly bounded above zero.

*Dependence-Aware Robustness and Date-Block Bootstrap Audit*: Conventional McNemar testing treats each of the $N = 27,179$ stock-date observations as independent. However, because observations originate from 10 equities trading concurrently across 3,205 unique dates (~8.48 stocks per day), cross-sectional co-movement is substantial: pairwise correlation of correctness differences ($d_{i,t} = \mathbb{I}(\text{model correct})_{i,t} - \mathbb{I}(\text{baseline correct})_{i,t}$) averages $r \approx +0.298$, and actual gap outcomes co-move with mean $r \approx +0.342$. Furthermore, daily aggregated accuracy differences exhibit mild multi-day persistence (Ljung-Box $Q = 30.68, p < 0.001$ across lags 1–5).

To test whether the $+7.06\%$ net edge withstands cross-sectional clustering and temporal dependence, a Date-Block Bootstrap ($R = 5,000$ resamples) was conducted across the 3,205 unique trading sessions, resampling blocks of dates while preserving all synchronous cross-sectional stock observations and model-vs-baseline pairings:
- **Daily Date-Clustering ($B = 1$ day)**: $\text{SE}_{\text{boot}} = 0.590\%$ (standard-error inflation factor $\text{SEIF} = 1.78\times$ over naive i.i.d. $\text{SE} = 0.331\%$; variance inflation factor $\text{VIF} = 3.17\times$), $95\%$ CI: $[+5.89\%, +8.18\%]$, $z = 11.97$, asymptotic $p = 5.27 \times 10^{-33}$, empirical $p_{\text{boot}} < 0.0002$.
- **Weekly Block Bootstrap ($B = 5$ days)**: $\text{SE}_{\text{boot}} = 0.650\%$ ($\text{SEIF} = 1.96\times, \text{VIF} = 3.85\times$), $95\%$ CI: $[+5.84\%, +8.37\%]$, $z = 10.87$, asymptotic $p = 1.63 \times 10^{-27}$, empirical $p_{\text{boot}} < 0.0002$.
- **Monthly Block Bootstrap ($B = 21$ days)**: $\text{SE}_{\text{boot}} = 0.757\%$ ($\text{SEIF} = 2.28\times, \text{VIF} = 5.21\times$), $95\%$ CI: $[+5.59\%, +8.56\%]$, $z = 9.34$, asymptotic $p = 9.85 \times 10^{-21}$, empirical $p_{\text{boot}} < 0.0002$.
- **Quarterly Block Bootstrap ($B = 63$ days)**: $\text{SE}_{\text{boot}} = 0.882\%$ ($\text{SEIF} = 2.66\times, \text{VIF} = 7.09\times$), $95\%$ CI: $[+5.45\%, +8.86\%]$, $z = 8.01$, asymptotic $p = 1.17 \times 10^{-15}$, empirical $p_{\text{boot}} < 0.0002$.

*Sample Size Hierarchy and Block-Length Sensitivity*: Econometric precision must clearly distinguish between four hierarchical aggregation dimensions: (1) 27,179 nominal stock-date observations, (2) 3,205 unique trading-date clusters, (3) approximately 152 non-overlapping monthly blocks ($3,205 / 21 \approx 152.6$), and (4) an effective independent sample size of $N_{\text{eff}} = 27,179 / \text{VIF} \approx 27,179 / 5.21 \approx 5,217$ independent stock-equivalent observations (or $T_{\text{eff}} \approx 3,205 / 3.17 \approx 1,011$ independent date clusters). These dimensions are not interchangeable. Selecting $B = 21$ trading days represents a methodological sensitivity decision corresponding to the empirical decay of serial autocorrelation ($|\rho_k| \le 0.026$ for $k \ge 15$), rather than a uniquely optimal window. The statistical significance of the $+7.06\%$ out-of-fold advantage is invariant to the choice of block length: asymptotic $p$-values range from $10^{-33}$ to $10^{-15}$, and the empirical bootstrap distribution bounded at $p_{\text{boot}} < 0.0002$ ($0$ of $5,000$ centered replicates reached the null boundary).

*Four-Way Conceptual Separation*: In financial econometrics, rigorous interpretation demands maintaining strict conceptual separation between four distinct dimensions:
1. **Predictive Accuracy (70.57%)**: Measures statistical classification alignment with directional opening gaps, demonstrating that overnight inter-market signals transmit substantial directional information into Indian market opens.
2. **Statistical Significance (Asymptotic $p \approx 9.85 \times 10^{-21}$; Empirical $p_{\text{boot}} < 0.0002$)**: Verifies that the accuracy advantage over historical drift is not an artifact of random sampling noise, cross-sectional clustering, or multi-day persistence.
3. **Economic Profitability**: Fails to materialize because Indian equities establish opening prices via the Pre-Open Call Auction (09:00–09:08 IST). Orders cannot be placed at $P_{t-1}^{\text{close}}$ after observing overnight moves; orders executed at $P_t^{\text{open}}$ capture zero opening gap return.
4. **Tradability After Execution Costs**: Any daytime continuation trade executed from open ($P_t^{\text{open}}$) to close ($P_t^{\text{close}}$) incurs statutory Indian frictions (15 bps STT, turnover charges, GST) and wide opening bid-ask spreads (10–30 bps), resulting in net negative returns. Statistical significance must never be conflated with commercial trading viability.

### 16.10. Actionable Trading Strategy Variants Backtest Results (2013–2026)
Table 12 presents the backtest results across all ten equities evaluated over a 13.1-year out-of-sample period (July 2013 to August 2026) under a 15 bps friction hurdle.

### Table 12: Actionable Trading Strategy Variants Comparison Across 10 Equities (2013–2026, Net of 15 bps Friction)

| Strategy Option | Operational Description | Mean Net CAGR | Sharpe Ratio | Maximum Drawdown | Win Rate | Profit Factor | Average Trades | Market Exposure |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Buy & Hold Benchmark** | Passive 100% equity holding | 15.68% | 0.311 | 54.77% | — | — | 1.0 | 100.0% |
| **Variant A (Baseline Long-Only)** | Enters when $P(\text{Up}) > 0.50$ & edge $> 15$ bps | 12.82% | 0.239 | 35.44% | 53.49% | 1.61 | 107.0 | 37.51% |
| **Variant B (Risk-Managed)** | 2% trailing stop-loss, 5-day exit | 6.75% | 0.098 | 20.25% | 52.15% | 1.25 | 59.5 | 10.92% |
| **Variant C (Dynamic Long/Short)** | Active long & short participation | 13.24% | 0.268 | 31.70% | 55.16% | 1.56 | 100.8 | 29.35% |
| **Variant D (Multi-Factor Consensus)** | Requires ML + RSI/MACD + FinBERT | 6.79% | 0.111 | 17.93% | 51.94% | 1.25 | 57.0 | 10.52% |
| **Variant E (Regime-Adaptive)** | Dynamic allocation; cash in high-vol regimes | 7.46% | 0.216 | **10.94%** | 53.24% | **2.66** | 40.2 | **6.98%** |

![Figure 10: Strategy Drawdown Comparison](figures/fig10_strategy_drawdown_comparison.png)
*Figure 10: Strategy risk-adjusted comparison: (A) Maximum drawdown reduction, and (B) CAGR vs. market exposure.*

![Figure 11: Transaction Cost Sensitivity](figures/fig11_transaction_cost_sensitivity.png)
*Figure 11: Net CAGR decay curves as transaction friction increases from 0 to 50 bps.*

We explicitly emphasize that Variant E does **not** outperform Buy & Hold in nominal compound growth: Buy & Hold generated a significantly higher nominal CAGR (15.68% vs. 7.46%). However, Buy & Hold suffered a catastrophic average maximum drawdown of 54.77% (exceeding 66% in ADANIENT and 78% in VEDL). In contrast, **Variant E reduced maximum drawdown to only 10.94% (an 80.0% drawdown reduction)** with an outstanding profit factor of 2.66. Crucially, this defensive profile is achieved by remaining in cash 93.02% of the time and exposing capital to equity market risk only 6.98% of sessions during favorable low-volatility regimes.
""")

    # 17. Discussion
    sections.append(r"""## 17. Discussion

### 17.1. Theoretical Alignment with Market Efficiency
Our multi-decade empirical findings are strongly consistent with Fama's Efficient Market Hypothesis [1], [2]:
1. **Weak-Form EMH**: The complete dissipation of directional edges across multi-decade cycles ($N=17,436$, Table 1) demonstrates that stationary technical indicators alone do not contain tradable directional alpha at daily frequencies once lookahead and autocorrelation are purged.
2. **Semi-Strong Form EMH**: The instantaneous same-day correlation ($\rho = +0.0562$) and next-day collapse ($\rho = +0.0070$) demonstrate rapid informational absorption. By the time news is published in morning newspapers, Indian equity markets have already incorporated the information into closing prices.

### 17.2. Model Complexity in Low Signal-to-Noise Regimes
The empirical parity observed across architectures (spread $< 1.32$ pp, Table 3) provides a vital lesson for quantitative researchers. In low-SNR environments, high-capacity non-linear models (gradient-boosted trees and deep neural nets) do not discover hidden structural relationships; instead, they risk fitting idiosyncratic sample noise. A convex, heavily regularized linear model ($L_2$ Logistic Regression, $C=0.1$) achieves identical accuracy while offering robust phase-shift stability across sampling offsets.

### 17.3. The Economics of Risk-Aware Trading and Cash Drag: Empirical vs. Pro-Forma Scenarios
The backtest results in Table 12 highlight the crucial distinction between nominal return, risk-adjusted survival, and cash-drag mechanics. We explicitly distinguish between the **empirical historical backtest** and **hypothetical pro-forma cash-yield scenarios**:

1. **Empirical Historical Backtest Baseline (Sole Verified Performance)**:
   - In the historical simulation across 2013–2026, the backtesting engine (`src/backtest.py`) models uninvested capital with conservative institutional cash accounting (5.0% risk-free rate on cash balances).
   - Under this verified implementation, Variant E achieves a net CAGR of **7.46%**, a maximum drawdown of **10.94%**, and an average active market exposure of only **6.98%** (spending 93.02% of trading days in cash). **This 7.46% CAGR is the sole verified empirical backtest performance** of the strategy.
   - **Zero-Cash Sensitivity Scenario**: If unallocated cash is modeled as earning strictly 0.0% nominal return, Variant E's net active equity CAGR across all ten equities is **2.82%** (with a maximum drawdown of **16.27%**).
   - **Engine-Simulated 6.0% Sovereign T-Bill Scenario**: When the backtest simulation is re-run with unallocated cash earning 6.0% p.a. (compounded at the daily bar frequency), the simulated portfolio CAGR across all ten equities is **8.39%** (with a maximum drawdown of **10.15%**).
   - In contrast, unhedged Buy & Hold achieves 15.68% nominal CAGR but suffers an average maximum drawdown of **54.77%** (exceeding 66% in ADANIENT and 78% in VEDL). Variant E does not compete with Buy & Hold on unadjusted total return; its structural advantage is delivering an **80.0% reduction in maximum drawdown** and a profit factor of 2.66.

2. **Cautionary Note on Informal Additive Approximations (~13.04%, 13.45%)**:
   - In earlier draft iterations and informal discussions, an analytical additive calculation was proposed: $(1 - \text{Exposure}) \times r_{\text{f}} = 0.9302 \times 6.00\% = 5.58\%$, added to active return to yield ~$13.04\%$ ($7.46\% + 5.58\%$), or a compounded projection of $13.45\%$.
   - **Methodological Error Audit**: We explicitly clarify that this additive calculation was a **flawed approximation that inadvertently double-counted cash carry**. Because the 7.46% backtest baseline already incorporated the 5.0% treasury cash rate ($2.82\% \text{ equity} + 4.65\% \text{ cash} \approx 7.46\%$), adding an extra 5.58% created an ungrounded return figure.
   - Neither 13.04% nor 13.45% represents actual historical backtest performance, nor a valid simulated portfolio CAGR. They are documented here strictly as an illustrative cautionary counterfactual. The sole verified historical backtest CAGR of Variant E is **7.46%** (and **2.82%** under zero cash yield; **8.39%** under 6% cash yield).
""")

    # 18. Explainability and Practical Implications
    sections.append(r"""## 18. Explainability and Practical Implications

### 18.1. For Quantitative Traders and System Architects
1. **Strictly Purge and Embargo Walk-Forward Partitions**: Always implement an embargo gap equal to or greater than the maximum feature lookback and target holding horizon.
2. **Subsample Multi-Day Targets**: Never evaluate overlapping rolling targets on daily grids without subsampling ($\text{step}=h$).
3. **Anchor Against Honest Baselines**: Never report nominal accuracy without comparing against training majority class and naive persistence baselines.
4. **Account for Indian Microstructure Costs**: Evaluate strategies net of statutory STT, GST, SEBI fees, and realistic slippage (minimum 15 bps hurdle).

### 18.2. For Financial NLP Researchers
1. **Mandatory Entity Disambiguation**: Generic sentiment models are unviable in multi-company news environments. Entity-targeted sentence-pair architectures are essential for accurate polarity attribution.
2. **Strip Circular Price Reports**: Filter price-reporting headlines (`"surges"`, `"slumps"`) to avoid confusing price effects with sentiment causes.

### 18.3. Decision-Support System Architecture
To operationalize these principles in user-facing software (such as the Streamlit and Next.js interfaces developed in this project), user interfaces must **strictly segregate retrospective explanation from forward prediction**. The "Why Did It Move?" module provides contemporaneous post-market attribution, while the "Prediction Terminal" presents forward forecasts with explicit confidence abstention thresholds.
""")

    # 19. Limitations
    sections.append(r"""## 19. Limitations

Table 13 synthesizes the critical methodological disclosures and limitations of this study.

### Table 13: Summary of Methodological Disclosures and Research Limitations

| Dimension | Methodological Implementation | Econometric & Practical Purpose |
| :--- | :--- | :--- |
| **Target Horizon & Subsampling** | $h=3$ non-overlapping subsampling ($\text{step}=3$) | Eliminates moving-average serial target autocorrelation ($\rho$ drops from 0.81 to $-0.04$). |
| **Walk-Forward Embargo Buffer** | 52 trading days (17 subsampled rows) dropped at train seams | Purges forward label overlap and decorrelates transient errors across train/test seams. |
| **Permutation Testing Resolution** | $n=6$ shuffles per fold with Laplace correction ($p_{\min} = 1/7 = 0.1429$) | Bounded computational diagnostic screening feature leakage; not a high-resolution hypothesis test. |
| **Cross-Sectional Panel Dependence** | Concurrent NSE trading across all 10 equities (2000–2026) | Effective degrees of freedom governed by time series length (~2,154 periods), not pooled 17,436 rows. |
| **News Corpora Coverage Window** | ToI archive (2001–2022), NSE filings (2008–2025), Cutoff Dec 31, 2025 | 2026 trading sessions evaluated under neutral sentiment; news does not explain 2026 volatility. |
| **Indian Transaction Friction Hurdle** | 15 basis points round-trip (STT 0.10%, GST, SEBI fee, slippage) | Enforces realistic break-even hurdle; gross edge must exceed 15 bps to yield net positive CAGR. |
| **Cash Allocation & Backtest Mechanics** | Uninvested capital held in cash (5.0% risk-free baseline; 2.82% under 0% cash sensitivity) | Low time-in-market strategies (Variants B, D, E) eliminate tail drawdowns at the expense of nominal compounding (7.46% vs. Buy & Hold's 15.68%). |

### 19.1. Additional Latency and Data Limitations
- **Daily Frequency vs. Millisecond Microstructure**: Price discovery on the NSE occurs within milliseconds. Daily closes cannot capture sub-second electronic order book discovery.
- **Public News vs. Institutional Feeds**: The news corpora represent public post-event reporting. Institutional quantitative funds utilize direct low-latency machine-readable feeds (Bloomberg, Reuters) with microsecond timestamps.
- **Fama Joint Hypothesis Caveat**: These empirical findings do not "prove" that markets are perfectly efficient in an absolute philosophical sense; they establish that within the evaluated information sets and linear/ML architectures, no exploitable excess return remains.

### 19.2. Disclosures on Review Methodology, Data Cutoffs, and News Sparsity
- **Internal Audit vs. External Peer Review**: We explicitly disclose that the validation process conducted on this working paper was an **internal quality, code replication, and econometric audit**, not an external double-blind journal peer review.
- **Historical Data Cutoffs**: News corpora have explicit historical termination boundaries: the Times of India headline archive terminates in December 2022, while official NSE regulatory announcements extend through December 2025. Evaluations conducted during 2026 operate under neutral sentiment defaults.
- **Single-Stock News Sparsity**: As quantified in Section 16.7 (Table 10), between 52.8% and 97.0% of single-stock large price moves occur without company-specific press coverage, reflecting the fundamental reality that public financial journalism is index-centric and cannot explain idiosyncratic single-stock volatility.
""")

    # 20. Conclusion and Future Work
    sections.append(r"""## 20. Conclusion and Future Work

### 20.1. Concluding Synthesis
This study conducted an empirical investigation into equity predictability, financial NLP explainability, and risk-aware trading across multi-decade National Stock Exchange of India data.

Evaluating 17,436 subsampled observations across ten equities under an expanding 5-fold walk-forward cross-validation harness with 52-day embargoes, we find that:
1. Stationary technical price indicators do not provide a statistically valid out-of-sample directional edge over honest baselines (`NO SIGNAL` across 10/10 equities under the permutation diagnostic screen).
2. Model architectures converge to within 1.32 percentage points of each other, with regularized linear models matching complex tree ensembles while offering superior stability.
3. Recurrent sequence models (LSTM, GRU) succumb to severe variance shrinkage and fail to beat a zero-return benchmark.
4. Entity-targeted FinBERT achieves 87.71% accuracy on SEntFiN 1.0, successfully resolving multi-firm attribution ambiguity.
5. In paired walk-forward ablations across 12,426 headlines and 11,740 regulatory filings, company news provides zero statistically significant forward edge ($p \ge 0.115$). News contemporaneously explains today's price movements ($\rho = +0.0562, p < 10^{-4}$), but collapses to noise on forward sessions ($\rho = +0.0070, p = 0.618$).
6. Overnight opening gaps ($N=27,179$, 2013–2025) exhibit 70.57% directional predictability driven by US and Asian market spillovers; however, this reflects international information transmission rather than an executable trading arbitrage.
7. Regime-adaptive trading strategies cut maximum drawdown by 80% (from 54.8% to 10.9%) under realistic statutory Indian transaction costs, achieving 7.46% verified historical backtest CAGR (and 2.82% under zero cash yield; 8.39% with 6% cash yield) with an exceptional profit factor of 2.66 while requiring active equity market exposure only 6.98% of the time.

### 20.2. Future Work
Promising research trajectories include:
- High-frequency limit order book (LOB) dynamics and trade-level microstructure flow.
- Inter-firm supply chain dependency networks and regulatory graph convolutional networks.
- Multimodal audio processing of corporate earnings conference calls.
""")

    # 21. References
    sections.append(r"""## 21. References

[1] E. F. Fama, "Efficient Capital Markets: A Review of Theory and Empirical Work," *The Journal of Finance*, vol. 25, no. 2, pp. 383–417, May 1970. DOI: [10.2307/2325486](https://doi.org/10.2307/2325486).

[2] E. F. Fama, "Efficient Capital Markets: II," *The Journal of Finance*, vol. 46, no. 5, pp. 1575–1617, Dec. 1991. DOI: [10.1111/j.1540-6261.1991.tb04636.x](https://doi.org/10.1111/j.1540-6261.1991.tb04636.x).

[3] H. White, "A Reality Check for Data Snooping," *Econometrica*, vol. 68, no. 5, pp. 1097–1126, Sep. 2000. DOI: [10.1111/1468-0262.00152](https://doi.org/10.1111/1468-0262.00152).

[4] P. R. Hansen, "A Test for Superior Predictive Ability," *Journal of Business & Economic Statistics*, vol. 23, no. 4, pp. 365–380, Oct. 2005. DOI: [10.1198/073500105000000063](https://doi.org/10.1198/073500105000000063).

[5] A. Timmermann and C. W. J. Granger, "Efficient Market Hypothesis and Forecasting," *International Journal of Forecasting*, vol. 20, no. 1, pp. 15–27, Jan. 2004. DOI: [10.1016/S0169-2070(03)00012-8](https://doi.org/10.1016/S0169-2070(03)00012-8).

[6] M. López de Prado, *Advances in Financial Machine Learning*. Hoboken, NJ, USA: John Wiley & Sons, Inc., 2018. ISBN: 978-1-119-48208-6.

[7] D. Araci, "FinBERT: Financial Sentiment Analysis with Pre-trained Language Models," *arXiv preprint arXiv:1908.10063*, Aug. 2019. Available: [https://arxiv.org/abs/1908.10063](https://arxiv.org/abs/1908.10063).

[8] Y. Yang, M. C. S. Uy, and A. Huang, "FinBERT: A Large Language Model for Extracting Information from Financial Text," *Management Science*, vol. 67, no. 5, pp. 2568–2592, May 2021. DOI: [10.1287/mnsc.2020.3604](https://doi.org/10.1287/mnsc.2020.3604).

[9] A. Sinha, S. Kedas, R. Kumar, and P. Malo, "SEntFiN 1.0: Entity-Aware Sentiment Analysis for Financial News," *Journal of the Operational Research Society*, vol. 73, no. 11, pp. 2534–2546, Nov. 2022. DOI: [10.1080/01605682.2021.1994646](https://doi.org/10.1080/01605682.2021.1994646).

[10] T. Loughran and B. McDonald, "When Is a Word Liability? Texts, Words, and Financial Performance," *The Journal of Finance*, vol. 66, no. 1, pp. 35–65, Feb. 2011. DOI: [10.1111/j.1540-6261.2010.01625.x](https://doi.org/10.1111/j.1540-6261.2010.01625.x).

[11] M. A. Petersen, "Estimating Standard Errors in Finance Panel Data Sets: Comparing Approaches," *The Review of Financial Studies*, vol. 22, no. 1, pp. 435–480, Jan. 2009. DOI: [10.1093/rfs/hhn053](https://doi.org/10.1093/rfs/hhn053).

[12] A. C. Cameron and D. L. Miller, "A Practitioner’s Guide to Cluster-Robust Inference," *Journal of Human Resources*, vol. 50, no. 2, pp. 317–372, Spring 2015. DOI: [10.3368/jhr.50.2.317](https://doi.org/10.3368/jhr.50.2.317).

[13] P. C. Tetlock, "Giving Content to Investor Sentiment: The Role of Media in the Stock Market," *The Journal of Finance*, vol. 62, no. 3, pp. 1139–1168, Jun. 2007. DOI: [10.1111/j.1540-6261.2007.01232.x](https://doi.org/10.1111/j.1540-6261.2007.01232.x).

[14] J. Bollen, H. Mao, and X. Zeng, "Twitter Mood Predicts the Stock Market," *Journal of Computational Science*, vol. 2, no. 1, pp. 1–8, Mar. 2011. DOI: [10.1016/j.jocs.2010.12.007](https://doi.org/10.1016/j.jocs.2010.12.007).

[15] J. J. Murphy, *Technical Analysis of the Financial Markets: A Comprehensive Guide to Trading Methods and Applications*. New York, NY, USA: New York Institute of Finance, 1999. ISBN: 978-0-7352-0066-1.

[16] J. W. Wilder, *New Concepts in Technical Trading Systems*. Greensboro, NC, USA: Trend Research, 1978. ISBN: 978-0-89459-008-5.

[17] G. Appel, *Technical Analysis: Power Tools for Active Investors*. Upper Saddle River, NJ, USA: Financial Times Prentice Hall, 2005. ISBN: 978-0-13-147929-6.

[18] J. Bollinger, *Bollinger on Bollinger Bands*. New York, NY, USA: McGraw-Hill Education, 2002. ISBN: 978-0-07-137368-5.

[19] L. Breiman, "Random Forests," *Machine Learning*, vol. 45, no. 1, pp. 5–32, Oct. 2001. DOI: [10.1023/A:1010933404324](https://doi.org/10.1023/A:1010933404324).

[20] J. H. Friedman, "Greedy Function Approximation: A Gradient Boosting Machine," *The Annals of Statistics*, vol. 29, no. 5, pp. 1189–1232, Oct. 2001. DOI: [10.1214/aos/1013203451](https://doi.org/10.1214/aos/1013203451).

[21] T. Chen and C. Guestrin, "XGBoost: A Scalable Tree Boosting System," in *Proc. 22nd ACM SIGKDD Int. Conf. Knowledge Discovery and Data Mining (KDD '16)*, San Francisco, CA, USA, Aug. 2016, pp. 785–794. DOI: [10.1145/2939672.2939785](https://doi.org/10.1145/2939672.2939785).

[22] G. Ke, Q. Meng, T. Finley, T. Wang, W. Chen, W. Ma, Q. Ye, and T.-Y. Liu, "LightGBM: A Highly Efficient Gradient Boosting Decision Tree," in *Advances in Neural Information Processing Systems (NeurIPS 2017)*, Long Beach, CA, USA, Dec. 2017, vol. 30, pp. 3146–3154.

[23] S. Hochreiter and J. Schmidhuber, "Long Short-Term Memory," *Neural Computation*, vol. 9, no. 8, pp. 1735–1780, Nov. 1997. DOI: [10.1162/neco.1997.9.8.1735](https://doi.org/10.1162/neco.1997.9.8.1735).

[24] K. Cho, B. van Merriënboer, Ç. Gülçehre, D. Bahdanau, F. Bougares, H. Schwenk, and Y. Bengio, "Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation," in *Proc. Conf. Empirical Methods in Natural Language Processing (EMNLP 2014)*, Doha, Qatar, Oct. 2014, pp. 1724–1734. DOI: [10.3115/v1/D14-1179](https://doi.org/10.3115/v1/D14-1179).

[25] A. Vaswani, N. Shazeer, N. Parmar, J. Uszkoreit, L. Jones, A. N. Gomez, Ł. Kaiser, and I. Polosukhin, "Attention Is All You Need," in *Advances in Neural Information Processing Systems (NeurIPS 2017)*, Long Beach, CA, USA, Dec. 2017, vol. 30, pp. 5998–6008.

[26] J. Devlin, M.-W. Chang, K. Lee, and K. Toutanova, "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding," in *Proc. Conf. North American Chapter Association for Computational Linguistics (NAACL-HLT 2019)*, Minneapolis, MN, USA, Jun. 2019, pp. 4171–4186. DOI: [10.18653/v1/N19-1423](https://doi.org/10.18653/v1/N19-1423).

[27] Q. McNemar, "Note on the Sampling Error of the Difference Between Correlated Proportions or Percentages," *Psychometrika*, vol. 12, no. 2, pp. 153–157, Jun. 1947. DOI: [10.1007/BF02295996](https://doi.org/10.1007/BF02295996).

[28] F. X. Diebold and R. S. Mariano, "Comparing Predictive Accuracy," *Journal of Business & Economic Statistics*, vol. 13, no. 3, pp. 253–263, Jul. 1995. DOI: [10.1080/07350015.1995.10509923](https://doi.org/10.1080/07350015.1995.10509923).

[29] D. I. Harvey, S. J. Leybourne, and P. Newbold, "Testing the Equality of Prediction Mean Squared Errors," *International Journal of Forecasting*, vol. 13, no. 2, pp. 281–291, Jun. 1997. DOI: [10.1016/S0169-2070(96)00719-4](https://doi.org/10.1016/S0169-2070(96)00719-4).

[30] W. F. Sharpe, "Mutual Fund Performance," *The Journal of Business*, vol. 39, no. 1, pp. 119–138, Jan. 1966. DOI: [10.1086/294846](https://doi.org/10.1086/294846).

[31] F. A. Sortino and L. N. Price, "Performance Measurement in a Downside Risk Framework," *The Journal of Investing*, vol. 3, no. 3, pp. 59–64, Fall 1994. DOI: [10.3905/joi.3.3.59](https://doi.org/10.3905/joi.3.3.59).

[32] R. F. Engle, "Autoregressive Conditional Heteroscedasticity with Estimates of the Variance of United Kingdom Inflation," *Econometrica*, vol. 50, no. 4, pp. 987–1007, Jul. 1982. DOI: [10.2307/1912773](https://doi.org/10.2307/1912773).

[33] T. Bollerslev, "Generalized Autoregressive Conditional Heteroskedasticity," *Journal of Econometrics*, vol. 31, no. 3, pp. 307–327, Apr. 1986. DOI: [10.1016/0304-4076(86)90063-1](https://doi.org/10.1016/0304-4076(86)90063-1).

[34] R. J. Hyndman and G. Athanasopoulos, *Forecasting: Principles and Practice*, 3rd ed. Melbourne, Australia: OTexts, 2021. Available: [https://otexts.com/fpp3/](https://otexts.com/fpp3/).

[35] National Stock Exchange of India Ltd., *Indian Securities Market: A Review (Fact Book 2024)*. Mumbai, India: NSE India Knowledge Center, 2024. Available: [https://www.nseindia.com](https://www.nseindia.com).
""")

    content = "\n\n".join(sections)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write(content)

    word_count = len(content.split())
    print(f"Corrected manuscript generated at {OUTPUT_FILE} with {len(sections)} sections and ~{word_count} words.")

if __name__ == '__main__':
    build_manuscript()
