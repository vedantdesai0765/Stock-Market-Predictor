# Results and Discussion

## 1. Overall 10-Stock Prediction Results

To rigorously test whether historical price dynamics yield a statistically valid, out-of-sample directional edge, the predictive pipeline was evaluated across ten major National Stock Exchange (NSE) equities. To eliminate sample-window selection bias, historical data was expanded back to **January 3, 2000** for all mature listings (comprising over 26 years of daily trading data), from August 2004 for Tata Consultancy Services (TCS), from July 2004 for Adani Enterprises (ADANIENT), and from initial public offering (IPO) dates for Adani Green Energy (ADANIGREEN, June 2018) and Eternal Portfolio (ETERNAL / Zomato, July 2021). 

All raw series were audited for corporate restructuring events (including the 2015 Adani demergers and the 2026 Vedanta capital restructuring) and cleaned of exchange holiday filler rows. To avoid artificial autocorrelation arising from overlapping multi-day returns, models were evaluated on an independent 3-day forward horizon ($h=3$) using strictly non-overlapping subsampling ($\text{step}=3$). Input features were transformed into scale-free stationary indicators (log-returns, normalized indicator oscillators, volatility ratios) and evaluated using an expanding 5-fold walk-forward cross-validation scheme with an embargo period equal to the maximum feature lookback (52 trading days for Ichimoku Senkou Span B).

The empirical out-of-sample results from the production pipeline (`reports/all_stocks_summary.csv`) are summarized in Table 1.

### Table 1: Multi-Decade Walk-Forward Directional Prediction Performance Across 10 NSE Equities ($h=3$ Days)

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

*Note: All models utilize $L_2$-regularized Logistic Regression ($C=0.1$) on stationary technical indicators. The honest baseline represents the prevailing majority class within the training folds. Net predictive edge is defined as $\text{Accuracy} - \text{Baseline}$. Permutation testing applies 1,000 Monte Carlo label shuffles.*

Across the full multi-decade sample comprising over 19,000 independent 3-day labels across all ten equities, **not a single stock demonstrates a statistically significant predictive edge over honest baseline rates**. Every ticker receives a verdict of `NO SIGNAL` under Monte Carlo permutation testing.

This finding highlights the critical impact of sample horizon length. In preliminary experiments conducted on truncated sample windows (e.g., 2018–2024), TCS appeared to exhibit a positive directional edge of $+1.54\%$ ($N=780$), and ITC exhibited an apparent edge of $+5.82\%$ ($N=986$ over 2014–2026). However, as historical data was systematically expanded to encompass full market cycles (including the 2000–2002 dot-com bust, the 2008 global financial crisis, the 2013 taper tantrum, and the 2020 pandemic crash), these apparent edges completely dissipated. For mature equities with over 2,100 subsampled non-overlapping periods (HDFCBANK, INFY, RELIANCE, SBI, VEDL), walk-forward accuracies converged tightly around 48.7%–53.9%, falling short of their respective training majority base rates (51.8%–55.3%).

While ETERNAL and ITC display nominal positive edges ($+2.16\%$ and $+2.14\%$, respectively), these gains fail significance testing:
1. In ETERNAL, the nominal edge ($+2.16\%$) is smaller than the cross-validation fold standard deviation ($\pm 3.57\%$) over a limited listing span of 375 non-overlapping samples.
2. In ITC, permutation testing demonstrates that random label permutations achieve an identical or superior accuracy in over 18% of Monte Carlo runs ($p = 0.183$).

These empirical findings are entirely consistent with the **Weak-Form Efficient Market Hypothesis (EMH)** within the scope of this experimental design: past stationary technical price patterns alone do not contain exploitable out-of-sample directional information once sample selection bias, target autocorrelation, and lookahead leakage are methodologically eliminated.

---

## 2. Comparison Against Majority, Persistence, and Other Baselines

In financial machine learning, reporting raw accuracy without comparison to rigorous, domain-specific baselines is fundamentally misleading. Directional market data exhibits natural class imbalances and temporal dependencies that can create high nominal accuracy in models that possess zero genuine forecasting skill.

To evaluate model performance, four distinct baselines were established:
1. **Majority Class Baseline**: The proportion of the most frequent directional label observed within the training period (or across the full test set for stationary targets).
2. **Persistence Baseline**: A naive rule predicting that the next period's directional movement will match the current period's observed direction: $\hat{y}_{t+1} = y_t$.
3. **Zero-Return Baseline**: In continuous return forecasting, predicting zero change ($\hat{r}_{t+1} = 0$), which serves as the minimum mean-squared-error benchmark for a martingale difference sequence.
4. **Stratified Random Baseline**: Generating directional predictions by drawing from a Bernoulli distribution parameterized by the historical upward-movement frequency.

The empirical interactions between these baselines and machine learning models are illustrated in Table 2, drawn from controlled benchmarking across the BSE SENSEX, NSE equities, and US market corpora (`sentiment_study/reports/comparison_all.csv`, `reports/phase4_sequence_TCS.csv`, and `sentiment_study/reports/leakage_demo.csv`).

### Table 2: Benchmark Comparison of Baseline Strategies vs. Trained Models Across Experimental Setups

| Experimental Setup | Evaluated Model | Model Nominal Accuracy | Prevailing Majority Baseline | Naive Persistence Baseline | Net Edge over Best Baseline | Effective Predictive Signal |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **INFY 3-Day Direction ($h=3$)** | Logistic Regression | 53.95% | **55.25%** | 50.12% | $-1.30\%$ | Negative Edge (Sub-baseline) |
| **RELIANCE 3-Day Direction ($h=3$)** | Logistic Regression | 49.72% | **51.86%** | 49.65% | $-2.14\%$ | Negative Edge (Sub-baseline) |
| **SENSEX Next-Day ($h=1$, Technical $T$)** | Logistic Regression | 51.54% | **53.11%** | 50.84% | $-1.57\%$ | Negative Edge (Sub-baseline) |
| **SENSEX Next-Day ($h=1$, $T+G$)** | Random Forest | **58.68%** | 53.11% | 51.20% | $+5.57\%$ | Positive Edge (Macro Spillover) |
| **SENSEX Big Moves ($|\Delta| > 1.5\%$)** | XGBoost | 70.94% | **71.37%** | 58.20% | $-0.43\%$ | Negative Edge (Majority Artifact) |
| **10-Day Overlapping Return ($h=10$)** | Persistence Rule | **88.43%** | 54.20% | 88.43% | $\pm 0.00\%$ | Spurious Autocorrelation Trap |
| **Same-Day Lookahead (Today's News)** | LightGBM ($T+S_0$) | **66.20%** | 53.12% | 51.04% | $+13.08\%$ | Contemporaneous Lookahead Trap |

These comparisons expose what this study terms the **Three Accuracy Illusions**:

```
                              THE THREE ACCURACY ILLUSIONS
                                           │
      ┌────────────────────────────────────┼────────────────────────────────────┐
      │                                    │                                    │
  The Majority Illusion                The Autocorrelation Trap             The Look-Ahead Mirage
      │                                    │                                    │
  70.9% accuracy on big moves          88.4% accuracy from persistence      66.2% accuracy from news
  looks impressive, but the            due to overlapping labels;           describing today's move;
  baseline is 71.37%.                  the model repeats yesterday.         cannot be traded tomorrow.
      │                                    │                                    │
      ▼                                    ▼                                    ▼
  NET EDGE: −0.43% (No skill)          NET EDGE: 0.00% (No skill)           NET EDGE: 0.00% (Lookahead)
```

1. **The Majority Class Baseline Trap**: On SENSEX big moves, classification models achieve ~70.9% nominal accuracy. An uninformed observer might interpret this as strong predictive power. However, because large moves occur on only ~28.6% of trading sessions, a naive rule predicting "no big move" achieves 71.37% accuracy. The trained model actually exhibits a negative net edge ($-0.43\%$).
2. **The Target Autocorrelation Trap**: When constructing rolling multi-day targets without subsampling (e.g., predicting 10-day forward returns using overlapping daily rolling windows), consecutive targets share 9 days of identical price history. This inflates target autocorrelation to $\rho = 0.81$. A naive persistence model repeating yesterday's label achieves **88.43% accuracy** without learning any financial structure. Subsampling independent, non-overlapping rows resets the autocorrelation to $-0.04$ and drops persistence accuracy to chance (~50%).
3. **The Contemporaneous Look-Ahead Trap**: Pairing news published during day $t$ with day $t$'s closing return yields 64%–66% accuracy ($p < 10^{-14}$). However, this reflects contemporaneous reporting of events that have already occurred, rather than a tradable predictive forecast for day $t+1$.

---

## 3. Comparison of the Seven Machine-Learning Models

To evaluate whether non-linear functional forms or decision-tree ensembles can discover subtle technical interactions overlooked by linear models, seven distinct model architectures were benchmarked across identical walk-forward cross-validation splits:
1. **$L_2$-Regularized Logistic Regression**: A parsimonious linear classifier with inverse regularization strength $C=0.1$.
2. **Random Forest**: A bagging ensemble of 500 decision trees with constrained leaf capacity (`min_samples_leaf=20`) and square-root feature subsampling.
3. **Histogram Gradient Boosting (Scikit-Learn GBDT)**: Shallow decision trees (`max_leaf_nodes=15`, learning rate $\eta=0.05$) with internal $L_2$ shrinkage.
4. **XGBoost**: Extreme gradient-boosted decision trees incorporating exact second-order Hessian approximations and leaf-weight regularization.
5. **LightGBM**: Highly optimized gradient boosting utilizing gradient-based one-side sampling (GOSS) and histogram binning.
6. **Long Short-Term Memory (LSTM)**: A 2-layer stacked PyTorch recurrent neural network with recurrent dropout ($p=0.2$) processing 30-day sequential lookbacks.
7. **Gated Recurrent Unit (GRU)**: A 2-layer stacked PyTorch GRU architecture optimized for parameter parsimony on noisy sequence data.

The models were evaluated under two primary information sets on the BSE SENSEX:
- **Technical Indicators Alone ($T$)**: 20 scale-free momentum, trend, and volatility oscillators.
- **Technical + Global Inter-Market Features ($T+G$)**: Incorporating overnight US closing returns (S&P 500) and early Asian market opens (Nikkei 225, KOSPI).

The empirical out-of-sample results from `sentiment_study/reports/comparison_all.csv` and `comparison_dl.csv` are presented in Table 3.

### Table 3: Performance Comparison Across Seven Architectures on SENSEX Next-Day Directional Forecasting

| Architecture | Model Family | Feature Set | Test Accuracy | ROC-AUC | Balanced Accuracy | Net Edge vs. Majority Baseline | McNemar Test ($p$-value vs. LogReg) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | Linear / Convex | $T$ | 51.54% | 0.5233 | 50.65% | $-1.57\%$ | Baseline Reference |
| **Random Forest** | Bagging Ensemble | $T$ | 53.60% | 0.5422 | 50.19% | $+0.49\%$ | $p = 0.069$ |
| **Gradient Boosting** | Boosting Ensemble | $T$ | 52.71% | 0.5344 | 51.55% | $-0.40\%$ | $p = 0.690$ |
| **XGBoost** | Boosting Ensemble | $T$ | 53.02% | 0.5367 | 51.79% | $-0.09\%$ | $p = 0.708$ |
| **LightGBM** | Boosting Ensemble | $T$ | 52.43% | 0.5343 | 51.57% | $-0.68\%$ | $p = 0.914$ |
| **PyTorch LSTM** | Recurrent Sequence | $T$ | 51.91% | 0.5215 | 50.82% | $-1.20\%$ | $p = 0.837$ |
| **PyTorch GRU** | Recurrent Sequence | $T$ | 52.89% | 0.5309 | 51.14% | $-0.22\%$ | $p = 0.340$ |
| **Logistic Regression** | Linear / Convex | $T+G$ | 58.59% | 0.6110 | 58.45% | $+5.48\%$ | Baseline Reference |
| **Random Forest** | Bagging Ensemble | $T+G$ | **58.68%** | **0.6172** | 58.36% | **$+5.57\%$** | $p = 0.533$ |
| **Gradient Boosting** | Boosting Ensemble | $T+G$ | 57.42% | 0.6068 | 57.20% | $+4.31\%$ | $p = 0.130$ |
| **XGBoost** | Boosting Ensemble | $T+G$ | 57.36% | 0.6053 | 57.18% | $+4.25\%$ | $p = 0.160$ |
| **LightGBM** | Boosting Ensemble | $T+G$ | 57.64% | 0.6048 | 57.44% | $+4.53\%$ | $p = 0.605$ |
| **PyTorch LSTM** | Recurrent Sequence | $T+G$ | 58.13% | 0.6140 | 57.88% | $+5.02\%$ | $p = 0.695$ |
| **PyTorch GRU** | Recurrent Sequence | $T+G$ | 58.34% | 0.6202 | 58.10% | $+5.23\%$ | $p = 0.567$ |

*Note: SENSEX next-day evaluation spans 3,239 out-of-sample walk-forward trading days (2009–2022). Majority class base rate is 53.11%.*

#### Key Comparative Insights

1. **Architecture Invariance in Out-of-Sample Performance**:
   Under technical-only features ($T$), all seven models perform within a narrow band of 51.5% to 53.6%, clustering tightly around the naive majority baseline of 53.11%. When inter-market global signals ($T+G$) are introduced, all seven models improve to 57.4%–58.7%. The performance spread across all seven distinct model paradigms is less than **1.32 percentage points**. Paired McNemar tests confirm that no non-linear tree ensemble or recurrent deep neural network achieves a statistically significant performance advantage over simple $L_2$-regularized Logistic Regression ($p > 0.05$ across all comparisons).

2. **The Virtues of Linear Regularization in Low-SNR Regimes**:
   In financial markets, the signal-to-noise ratio (SNR) is exceptionally low. High-capacity non-linear models (e.g., gradient boosting trees and multi-layer sequence networks) risk overfitting to noise in historical training windows. A convex, $L_2$-regularized linear model with strong shrinkage ($C=0.1$) achieves 58.59% accuracy and an AUC of 0.6110, matching Random Forest (58.68%) while avoiding tree-splitting instability.

3. **Subsampling Offset Robustness**:
   A critical vulnerability identified in non-linear models is sensitivity to the subsampling offset phase. In multi-day horizon experiments ($h=3$), there are three equally valid starting offsets ($k \in \{0, 1, 2\}$). As recorded in `reports/phase4_offset_sensitivity.csv`:
   - **Random Forest**: Edge varies from $+3.85\%$ (offset 0) to $+3.33\%$ (offset 1) and flips to **$-1.28\%$** (offset 2).
   - **Gradient Boosting**: Edge varies from $+2.82\%$ (offset 0) to $+3.85\%$ (offset 1) and flips to **$-3.85\%$** (offset 2).
   - **Logistic Regression**: Edge remains strictly positive across all three offsets ($+1.54\%$, $+3.33\%$, $+1.03\%$).

Because Logistic Regression provides stable parameter estimates across offsets, seeds, and walk-forward folds, it was selected as the core production model in `src/pipeline.py`.

---

## 4. Deep-Learning Model Results (LSTM and GRU)

Recurrent neural networks (RNNs) are frequently proposed in financial literature for their theoretical capacity to model non-linear temporal dependencies and autoregressive memory. To test whether sequence architectures extract predictive signal from multi-day temporal structures, stacked LSTM and GRU regressors and classifiers were trained on 30-day sequential rolling windows of stationary returns and technical indicators.

Models were evaluated across three distinct dimensions:
1. **Directional Classification Accuracy**: Evaluated on SENSEX, S&P 500, and individual equities (`sentiment_study/reports/comparison_dl.csv`).
2. **Continuous Return Forecasting (Root Mean Squared Error - RMSE)**: Compared against the naive Persistence baseline and the Zero-Return baseline (`reports/phase4_sequence_TCS.csv` and `phase4_sequence_RELIANCE.csv`).
3. **Prediction Variance Shrinkage**: Measuring the standard deviation of model predictions relative to the empirical standard deviation of actual asset returns: $\text{Shrinkage} = \frac{\sigma(\hat{r})}{\sigma(r)}$.

The empirical findings are summarized in Table 4.

### Table 4: Deep Learning Sequence Model Performance and Diagnostic Failure Analysis

| Asset / Market Scope | Architecture | Model Setup | Directional Accuracy | Net Edge vs. Base | Continuous RMSE | Naive Persistence RMSE | Zero-Return Benchmark RMSE | Prediction Shrinkage Factor |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **TCS (2004–2026)** | PyTorch LSTM | 30-day Sequence | 49.33% | $-2.46\%$ | 0.013117 | 0.018418 | **0.013069** | **0.0962** |
| **TCS (2004–2026)** | PyTorch GRU | 30-day Sequence | 51.24% | $-0.55\%$ | 0.013248 | 0.018418 | **0.013069** | **0.1196** |
| **RELIANCE (2000–2026)** | PyTorch LSTM | 30-day Sequence | 50.22% | $-1.64\%$ | 0.009109 | 0.012354 | **0.008930** | **0.1849** |
| **RELIANCE (2000–2026)** | PyTorch GRU | 30-day Sequence | 50.65% | $-1.21\%$ | 0.009133 | 0.012354 | **0.008930** | **0.1926** |
| **India SENSEX (Next-Day)** | PyTorch LSTM | $T$ (Tech Only) | 51.91% | $-1.20\%$ | — | — | — | — |
| **India SENSEX (Next-Day)** | PyTorch LSTM | $T+S$ (+ Sent) | 52.16% | $-0.95\%$ | — | — | — | — |
| **India SENSEX (Next-Day)** | PyTorch GRU | $T$ (Tech Only) | 52.89% | $-0.22\%$ | — | — | — | — |
| **India SENSEX (Next-Day)** | PyTorch GRU | $T+S$ (+ Sent) | 51.82% | $-1.29\%$ | — | — | — | — |
| **US S&P 500 (Next-Day)** | PyTorch LSTM | $T$ (Tech Only) | 53.34% | $-0.36\%$ | — | — | — | — |
| **US S&P 500 (Next-Day)** | PyTorch LSTM | $T+S$ (+ Sent) | 53.36% | $-0.34\%$ | — | — | — | — |
| **US S&P 500 (Next-Day)** | PyTorch GRU | $T$ (Tech Only) | 53.43% | $-0.27\%$ | — | — | — | — |
| **US S&P 500 (Next-Day)** | PyTorch GRU | $T+S$ (+ Sent) | 52.51% | $-1.19\%$ | — | — | — | — |

*Note: SENSEX next-day majority baseline is 53.11%; S&P 500 majority baseline is 53.70%; TCS baseline is 51.79%; RELIANCE baseline is 51.86%. Shrinkage factor measures $\sigma(\hat{r})/\sigma(r)$.*

#### Empirical Diagnosis of Deep Learning Failure Modes

1. **Failure to Beat the Zero-Return Benchmark**:
   While both LSTM and GRU regressors easily outperform the naive persistence baseline (e.g., TCS LSTM RMSE of 0.013117 vs. Persistence RMSE of 0.018418), **neither sequence model can outperform a trivial Zero-Return forecast** (`beats_zero = False`). On TCS, predicting $\hat{r}=0$ yields an RMSE of 0.013069, outperforming both LSTM (0.013117) and GRU (0.013248). On RELIANCE, the zero forecast achieves an RMSE of 0.008930, superior to LSTM (0.009109) and GRU (0.009133). In mathematical terms, the recurrent networks fail to extract any conditional mean dynamics beyond the unconditional zero mean.

2. **Severe Variance Shrinkage**:
   The diagnostic metrics reveal extreme prediction variance shrinkage. On TCS, the standard deviation of LSTM predictions is only **9.62%** of the asset's true return volatility ($\text{shrinkage}=0.0962$); for GRU, it is **11.96%**. On RELIANCE, shrinkage is 18.49% for LSTM and 19.26% for GRU. Rather than discovering cyclical patterns, gradient descent drives recurrent weights toward an unconditional constant drift, minimizing $L_2$ loss by avoiding large directional commitments.

3. **Inability to Leverage Sequential Sentiment**:
   When sequential daily news sentiment embeddings are appended to technical inputs ($T \to T+S$), neither LSTM nor GRU derives any statistically significant advantage on next-day prediction:
   - On India next-day prediction: LSTM accuracy shifts from 51.91% to 52.16% ($\Delta = +0.25\%, p_{\text{mcnemar}} = 0.837$); GRU accuracy declines from 52.89% to 51.82% ($\Delta = -1.08\%, p_{\text{mcnemar}} = 0.340$).
   - On US next-day prediction: LSTM shifts by $+0.01\%$ ($p = 1.0$); GRU declines from 53.43% to 52.51% ($\Delta = -0.92\%, p = 0.102$).

In summary, recurrent deep-learning models provide no empirical benefit over simple linear classifiers in predicting stationary next-day equity returns, while introducing computational overhead, hyperparameter sensitivity, and optimization shrinkage.

---

## 5. Company-Targeted Sentiment and Entity FinBERT Evaluation

A critical flaw in standard financial natural language processing (NLP) is the application of whole-document sentiment models to multi-company headlines. For example, consider the headline:
$$\text{"Reliance Retail expands market presence while Tata Motors reports supply chain disruption."}$$
A generic sentiment model (such as default FinBERT, VADER, or Loughran-McDonald) outputs a single compound score for the entire sentence, obscuring the fact that the sentiment is strongly positive for Reliance and negative for Tata.

To overcome this limitation, a dedicated **Entity-Targeted FinBERT** classifier was trained. The architecture modifies the input sequence into a sentence-pair classification task:
$$\text{Input} = \text{[CLS]} \circ \text{Target Company} \circ \text{[SEP]} \circ \text{News Headline} \circ \text{[SEP]}$$

The model was fine-tuned on **SEntFiN 1.0** (a gold-standard benchmark of 14,371 human-annotated entity-headline pairs) over 3 epochs with AdamW optimization ($\text{learning rate} = 2 \times 10^{-5}$) and cross-entropy loss.

The held-out evaluation results (`sentiment_study/reports/entity_model_eval.json`) comparing the Entity-Targeted model against standard whole-headline FinBERT are presented in Table 5.

### Table 5: Entity-Targeted FinBERT vs. Standard Generic FinBERT on the SEntFiN Benchmark

| Test Benchmark Evaluation Split | Sample Size ($N$) | Evaluation Metric | Standard Generic FinBERT | Fine-Tuned Entity FinBERT | Net Performance Improvement |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Full SEntFiN Test Set** | 1,440 pairs | Classification Accuracy | 68.54% | **87.71%** | **$+19.17\%$ pts** |
| **Full SEntFiN Test Set** | 1,440 pairs | Macro-Averaged $F_1$ | 0.6870 | **0.8783** | **$+0.1913$** |
| **Multi-Entity Headlines ($\ge 2$ Companies)** | 614 pairs | Classification Accuracy | 61.56% | **85.34%** | **$+23.78\%$ pts** |
| **Multi-Entity Headlines ($\ge 2$ Companies)** | 614 pairs | Macro-Averaged $F_1$ | 0.6122 | **0.8527** | **$+0.2405$** |

The empirical results demonstrate a decisive gain in NLP extraction fidelity. On the full test set of 1,440 pairs, Entity-Targeted FinBERT increases accuracy from 68.54% to **87.71%** (+19.17 percentage points) and macro-$F_1$ from 0.6870 to **0.8783**.

The improvement is even more pronounced on headlines explicitly naming two or more competing entities (614 pairs), where standard FinBERT degrades to near-chance accuracy (**61.56%**). Entity-Targeted FinBERT maintains **85.34%** accuracy (+23.78 percentage points gain) and an $F_1$ score of **0.8527**, successfully isolating the sentiment directed specifically at the queried firm.

This establishes that the fine-tuned entity model solves the representation problem of financial news sentiment extraction, allowing clean entity-specific sentiment series to be constructed for downstream econometric ablation.

---

## 6. Sentiment Ablation Results

With an accurate entity-targeted sentiment extractor established, an extensive ablation study was conducted to determine whether company-targeted news sentiment provides an incremental, out-of-sample edge in predicting next-day equity movements.

The evaluation was performed across two distinct news corpora:
1. **Times of India (ToI) Historical Headlines (2001–2022)**: 12,426 company-headline pairs matched to the 10 project equities.
2. **National Stock Exchange (NSE) Corporate Regulatory Filings (2008–2025)**: 11,740 regulatory filings and company announcements, stamped with exact exchange dissemination times.

For each stock, out-of-sample walk-forward models were trained under two feature regimes:
- **Baseline Feature Set ($B$)**: Technical indicators, market return (Nifty 50), overnight US return (S&P 500), volatility index (India VIX), and Asian market opens.
- **Sentiment-Augmented Feature Set ($B + C_{\text{ent}}$)**: The baseline set augmented with entity-targeted sentiment scores ($C_{\text{ent}}$).

The complete stock-by-stock ablation results (`reports/company_sentiment_ablation.csv`) are detailed in Table 6.

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

#### Critical Findings from the Ablation Study

1. **Complete Absence of Statistical Significance**:
   Incorporating entity-targeted sentiment produces marginal accuracy changes ranging between **$-2.30\%$ and $+2.46\%$**, with a sample-weighted mean delta of approximately **$+0.11\%$**. Across all 20 evaluated equity-corpus pairs, **not a single stock achieves statistical significance** under the paired McNemar test. Even the largest nominal gain (State Bank of India on ToI headlines, shifting from 49.50% to 51.96%, $\Delta = +2.46\%$) yields $p = 0.1152$, failing to reach the standard $\alpha = 0.05$ threshold. For 14 of the 20 configurations, the McNemar $p$-value exceeds $0.65$, and for 5 configurations it is exactly $1.000$.

2. **Negative and Deteriorating Edge on Thinly-Covered Equities**:
   For stocks with lower media coverage (e.g., ADANIGREEN and ADANIENT), adding sentiment degrades performance. On ADANIGREEN, adding headline sentiment reduces accuracy by $-2.30\%$ (AUC drops from 0.4911 to 0.4580); on ADANIENT, accuracy drops by $-0.99\%$ on headlines and $-0.94\%$ on filings. In low-coverage regimes, sparse sentiment acts as idiosyncratic noise that degrades linear coefficient estimation.

3. **Validation via Placebo Testing**:
   To confirm that the minor fluctuations around zero reflect estimation variance rather than genuine signal, a placebo experiment was executed where company sentiment was shifted temporally by exactly $+1$ calendar year. The placebo series produced performance deltas of identical magnitude ($-0.011$ to $+0.002$, with all $p > 0.40$), confirming that the observed sentiment deltas are indistinguishable from random noise.

---

## 7. News-Price Relationship and the "Why Did It Move?" Analysis

If news sentiment does not predict next-day equity returns, what role does financial news actually play? To answer this question, an explanatory analysis was conducted (`sentiment_study/reports/explain/summary.csv` and `src/explain_page.py`), which forms the analytical foundation of the dashboard's "Why did it move?" interface.

### 7.1. Controlling for Circular Logic: Price-Reporting Headlines

A major source of spurious correlation in financial NLP is circular reporting: headlines that describe an asset's price movement (e.g., *"Sensex plunges 400 points on inflation fears"*, *"TCS drops 3% after earnings miss"*). When scored by FinBERT, these headlines produce strong negative sentiment. If paired with the same day's closing return, an artificial correlation arises because the headline is an effect of the price move, not its cause.

To eliminate circularity, an automated filter removed all price-reporting headlines matching pattern templates (`"down X%"`, `"drops Y points"`, `"slumps"`, `"surges"`). The statistical relationship was then computed across three temporal alignments:
- **Same-Day Association ($t$)**: News timestamped during day $t$ vs. return on day $t$.
- **Next-Day Forecast ($t+1$)**: News timestamped during day $t$ vs. return on day $t+1$.
- **Preceding-Day Drift ($t-1$)**: News timestamped during day $t$ vs. return on day $t-1$.

The comparative lead-lag correlations (`sentiment_study/reports/explain/lead_lag.csv` and `summary.csv`) are presented in Table 7.

### Table 7: Lead-Lag Rank Correlation Profile (Spearman $\rho$) Between News Sentiment and Equity Returns

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

*Note: SENSEX analysis comprises 5,119 trading sessions (2001–2022). Pooled 10-stock analysis comprises 7,765 stock-day observations. $p$-values reflect two-tailed asymptotic Spearman rank significance.*

### 7.2. Lead-Lag Asymmetry: Explanation vs. Prediction

The lead-lag profile resolves the core paradox of financial sentiment analysis:
1. **Contemporaneous Correlation**: On the SENSEX, even after strictly stripping all price reports, news sentiment maintains a statistically significant positive rank correlation with the **same day's** return ($\rho = +0.0562, p = 0.00006$). Across the pooled 10-stock panel, same-day correlation is similarly positive and significant ($\rho = +0.0324, p = 0.0044$).
2. **Next-Day Collapse**: When looking forward to the **next trading day** ($\text{Lag } +1$), the correlation collapses to near zero: $\rho = +0.0070$ ($p = 0.618$) on the SENSEX, and $\rho = +0.0131$ ($p = 0.248$) on the pooled 10 stocks. 
3. **Preceding-Day Peak**: Notably, the correlation at $\text{Lag } -1$ ($\rho = +0.0573, p = 0.00004$) is slightly higher than at $\text{Lag } 0$. This reflects print newspaper publication timing: events that occur during trading hours on day $t-1$ are printed in the newspaper on day $t$. The market reacted to the underlying event on day $t-1$; by the time the headline appears in print on day $t$, the information has already been fully priced in.

### 7.3. Sentiment Quintile Return Spreads

To examine the economic magnitude of this relationship, daily sessions were sorted into quintiles based on clean news sentiment ($Q_1 = \text{most negative}$, $Q_5 = \text{most positive}$). The resulting return spreads from `sentiment_study/reports/explain/quintiles.csv` are shown in Table 8.

### Table 8: Daily Mean Returns Across Clean News Sentiment Quintiles on the BSE SENSEX

| Sentiment Quintile | Number of Sessions ($N$) | Same-Day Mean Return | Next-Day Mean Return |
| :--- | :---: | :---: | :---: |
| **Quintile 1 (Most Negative)** | 1,024 | $-0.0564\%$ | $+0.0689\%$ |
| **Quintile 2** | 1,023 | $+0.0179\%$ | $-0.0262\%$ |
| **Quintile 3 (Neutral)** | 1,024 | $+0.0475\%$ | $+0.0875\%$ |
| **Quintile 4** | 1,023 | $+0.1292\%$ | $+0.1089\%$ |
| **Quintile 5 (Most Positive)** | 1,024 | $+0.1434\%$ | $+0.0343\%$ |
| **Spread: Quintile 5 minus Quintile 1** | — | **$+0.1998\%$ / day** | **$-0.0345\%$ / day** |

On the same day, moving from the most negative sentiment quintile ($Q_1$) to the most positive ($Q_5$) corresponds to a clean return spread of **$+0.20\%$ per session** ($-0.056\%$ vs. $+0.143\%$). However, on the next day, the spread completely vanishes and reverses slightly to **$-0.035\%$ per day** ($+0.034\%$ for $Q_5$ vs. $+0.069\%$ for $Q_1$). Positive news days experience mild mean-reversion, while negative news days experience slight recovery bounces, confirming the absence of post-earnings or post-news drift at the daily index frequency.

### 7.4. Big Move News Coverage

An essential feature of the "Why did it move?" analysis is assessing whether extreme market moves are accompanied by explanatory news. Defining a "big move" as an absolute return exceeding 1.5% ($|\Delta| > 1.5\%$), each event was classified into one of four coverage categories (`sentiment_study/reports/explain/big_moves_coverage.csv`):
- **Matching News**: News sentiment direction matches the price move (e.g., positive news on an up day).
- **Opposite News**: News sentiment direction conflicts with the price move.
- **Neutral News**: News items were present but classified as neutral.
- **No News**: Zero company or index headlines were published on that trading date.

The empirical distribution is presented in Table 9.

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

On the market-wide SENSEX, only 6.11% of big move days occur without news coverage; major index shocks almost always receive immediate journalistic attention. However, for individual equities, **52.8% to 96.9% of large price swings occur with zero company-specific headlines in the major press**. Large-cap equity volatility in India is predominantly driven by systematic liquidity shocks, institutional order flows, foreign portfolio investor (FPI) reallocations, or derivative expiry pressures rather than idiosyncratic public news announcements.

---

## 8. Statistical Significance and Limitations

### 8.1. Methodological Controls and Demystifying Published "High Accuracy" Claims

A primary contribution of this study is demonstrating how common methodological shortcuts in financial literature generate artificially inflated performance metrics. In published studies, reported accuracies of 75% to 90% are frequently cited as evidence of successful machine-learning prediction. 

To determine the exact mechanisms producing these figures, controlled experiments were conducted in `sentiment_study/reports/leakage_demo.csv` by intentionally introducing specific evaluation flaws into the same underlying SENSEX and NSE datasets. The results are summarized in Table 10.

### Table 10: Controlled Demonstration of Literature Accuracy Pitfalls on Identical Market Data

| Experimental Evaluation Protocol | Valid Setup? | Reported Metric | Nominal Value | Mechanism of Artificial Performance Inflation |
| :--- | :---: | :---: | :---: | :--- |
| **Honest Out-of-Sample Walk-Forward ($T+G+S$)** | **Yes** | Direction Accuracy | **57.24%** | Rigorous expanding walk-forward CV, non-overlapping labels, strict embargo. |
| **Same-Day Headline Polarity vs. Same-Day Return** | **No** | Direction Accuracy | **76.96%** | **Contemporaneous Lookahead**: FinBERT scores headlines describing today's close ($N=2,270$). |
| **Next-Day Implied by Same Headline Rule** | **Yes** | Direction Accuracy | **55.02%** | Proper alignment resets accuracy to the unconditional base rate (~53%). |
| **5-Day Return Horizon with Shuffled 5-Fold CV** | **No** | Direction Accuracy | **60.95%** | **Data Leakage**: Random $k$-fold cross-validation causes adjacent days to share 4 of 5 return days. |
| **5-Day Return with Walk-Forward & 10-Day Embargo**| **Yes** | Direction Accuracy | **55.29%** | Embargo prevents indicator leakage across boundaries; accuracy drops to majority base (~56%). |
| **10-Day Moving Average Direction (Walk-Forward)** | **No** | Direction Accuracy | **86.43%** | **Trivial Target**: Predicting $\text{SMA}_{10}(t+1) > \text{SMA}_{10}(t)$ when 9 of the 10 prices are already known. |
| **Next-Day Price Level Linear Regression** | **No** | Coefficient ($R^2$) | **0.9980** | **Non-Stationarity**: Regressing $P_{t+1}$ on $P_t$ yields near-unity fit due to integrated unit-root drift. |
| **Next-Day Price Level Linear Regression** | **No** | Metric ($1 - \text{MAPE}$)| **0.9925** | Reported as "99.2% accuracy"; simply reflects that tomorrow's price is close to today's price. |
| **Direction Implied by Level Regression** | **Yes** | Direction Accuracy | **52.54%** | The level model that achieved 99.2% $1-\text{MAPE}$ has zero directional predictive power. |

These controlled demonstrations reveal the origin of common high-accuracy claims:
1. **The 77% Headline Illusion**: Arises directly from using same-day news to predict same-day returns, confusing contemporaneous description with forward prediction.
2. **The 86% Moving-Average Illusion**: Arises from predicting the direction of a moving average rather than the raw price return. Because 90% of the moving average window is already in the information set, high accuracy is mathematically trivial.
3. **The 99% Price-Level Illusion**: Arises from regressing non-stationary nominal price levels. A model scoring $R^2 = 0.998$ and $1-\text{MAPE} = 99.25\%$ achieves a directional accuracy of only **52.54%**, which is below the majority baseline.

### 8.2. Empirical Limitations of the Study

While the evaluation methodology eliminates lookahead and selection bias, several inherent limitations must be acknowledged:

1. **Daily Frequency and Latency Truncation**:
   This study operates at a daily closing frequency. In modern electronic equity markets, price discovery following corporate news occurs within milliseconds through automated algorithmic execution. By the time a daily candle closes, any informational advantage has already been absorbed into the price. This study evaluates whether daily retail or institutional traders can systematically trade news published overnight or in morning newspapers, finding that they cannot.

2. **Mainstream Media vs. Private Terminals**:
   The sentiment corpora utilized (Times of India, NYT, NSE public filings) represent public, post-event reporting. Institutional quantitative funds utilize direct machine-readable feeds from financial wires (e.g., Bloomberg, Reuters, Dow Jones) with microsecond timestamps. Public news archives necessarily reflect lower-frequency journalistic summaries.

3. **Transaction Costs and Market Frictions**:
   In Indian equity markets, real-world execution is subject to substantial frictions:
   - Securities Transaction Tax (STT): 0.10% on delivery trades (or 0.025% on intraday equity turnover).
   - Exchange turnover charges, GST, SEBI turnover fees, and stamp duty (~0.015% round-trip).
   - Bid-ask spread and execution slippage: typically 0.05% to 0.15% for mid-to-large-cap equities.
   
   A trading strategy requires an expected gross directional edge of at least $+0.25\%$ to $+0.35\%$ per round-trip trade simply to achieve break-even net profitability. Because the empirical technical edges observed across the 10 stocks range from $-4.85\%$ to $+2.16\%$ (with none statistically significant), transaction costs would render any simulated technical trading strategy firmly loss-making.

4. **Weak-Form vs. Semi-Strong Form EMH Interpretation**:
   Care must be taken in drawing theoretical conclusions. These empirical findings do not "prove" the Efficient Market Hypothesis in its entirety. Rather, they demonstrate that **within this experimental setup (daily frequency, 2000–2026 data, stationary technical indicators, public news headlines, linear and standard ML architectures), results are fully consistent with Weak-Form EMH for price patterns and Semi-Strong Form EMH for public news headlines**.

---

## 9. Key Findings

The results of this study are synthesized into seven empirical findings:

1. **Multi-Decade Technical Predictability Converges to Baseline**:
   Across 26 years of out-of-sample walk-forward evaluation (2000–2026) encompassing ~19,000 independent 3-day labels across 10 NSE equities, **not a single stock exhibits a statistically significant directional edge** under Monte Carlo permutation testing. Apparent edges observed in smaller sample windows (e.g., TCS $+1.54\%$, ITC $+5.82\%$) completely evaporate when evaluated across full multi-decade market cycles.

2. **Model Architecture Parity in Low Signal-to-Noise Regimes**:
   Across seven evaluated machine-learning and deep-learning architectures (Logistic Regression, Random Forest, Gradient Boosting, XGBoost, LightGBM, LSTM, GRU), performance on genuine out-of-sample next-day forecasting differs by less than **1.32 percentage points**. No complex non-linear ensemble or recurrent neural network outperforms parsimonious, $L_2$-regularized Logistic Regression ($p > 0.05$).

3. **Recurrent Sequence Models Suffer Variance Shrinkage**:
   Both LSTM and GRU neural networks fail to outperform a simple Zero-Return forecast ($\hat{r}=0$) on continuous return metrics (`beats_zero = False`). The networks experience severe prediction variance shrinkage (predictions compress to 9.6%–19.3% of true return standard deviation), effectively collapsing toward predicting an unconditional constant mean. Appending sequential sentiment does not improve directional accuracy ($p > 0.10$).

4. **Entity-Targeted NLP Resolves Attribution Ambiguity**:
   Fine-tuning FinBERT on entity-headline pairs (SEntFiN 1.0) improves classification accuracy from 68.54% to **87.71%** (+19.17 percentage points) and macro-$F_1$ from 0.6870 to **0.8783**. On complex headlines mentioning two or more competing firms, Entity FinBERT achieves **85.34%** accuracy compared to 61.56% for generic FinBERT (+23.78 percentage points), successfully isolating entity-specific polarity.

5. **Company Sentiment Does Not Improve Next-Day Forecasting**:
   In an ablation study across 10 equities evaluating 12,426 ToI headlines and 11,740 regulatory filings, adding entity-targeted sentiment yields accuracy deltas between **$-2.30\%$ and $+2.46\%$**. **Zero stocks achieved statistical significance** under paired McNemar testing (all $p \ge 0.115$, majority $p > 0.65$), and a 1-year temporal placebo test produced identical random noise.

6. **The Fundamental Asymmetry: Explanation vs. Prediction**:
   News sentiment exhibits a highly significant **contemporaneous** correlation with same-day returns ($\rho = +0.0562, p = 0.00006$ on SENSEX; $\rho = +0.0324, p = 0.0044$ across 10 pooled equities), producing a clean daily spread of $+0.20\%$ between top and bottom sentiment quintiles. However, on the **next day**, correlation collapses to noise ($\rho = +0.0070, p = 0.618$). News explains the present; it does not forecast the future.

7. **Market-Wide Drivers Dominate Opening Gap Accuracies**:
   The ~80% directional accuracy achieved in predicting SENSEX opening gaps at 09:00 IST is not driven by news sentiment, but almost entirely by overnight returns from US markets (S&P 500 close) and early Asian market trading (Nikkei 225, KOSPI opens). Idiosyncratic equity volatility remains largely unexplained by public press headlines, with 52% to 97% of large single-stock moves occurring on days with zero company-specific news.
