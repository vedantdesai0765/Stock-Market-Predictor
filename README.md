# Stock Market Predictor

Next-move direction prediction for Indian equities, built on technical
indicators with a rigorously validated machine-learning pipeline.

A B.Tech final-year project. The emphasis is on **methodological
correctness over headline accuracy**: every claim below is backed by
walk-forward validation, permutation testing, and seed/offset robustness
checks, and results that failed those checks were discarded rather than
reported.

---

## Headline results

### 1. 10-Stock Predictor Pipeline (2000–2026 Expanded History)

To eliminate small-sample window bias, historical stock data was expanded back to **2000-01-03** for all mature equities (TCS from its August 2004 listing, ADANIENT from July 2004 when clean data begins; ADANIGREEN and ETERNAL from their respective IPO listings).

Data collection was executed via `scripts/extend_history_2000.py` and audited via `scripts/fix_raw_data.py`, removing holiday filler rows and adjusting for corporate demergers (ADANIENT 2015 spin-offs and VEDL 2026 capital restructuring).

Evaluated across an independent 3-day forward horizon using scale-free stationary features and non-overlapping subsampling, the production artifacts in `reports/all_stocks_summary.csv` show:

| Stock | Data From | Subsampled Rows | Walk-Forward Accuracy | Honest Baseline | Predictive Edge | Permutation Verdict |
|---|---|---|---|---|---|---|
| **ADANIENT** | 2004-07 | 1,773 | 0.5310 ± 0.0199 | 0.5320 | −0.0010 | NO SIGNAL |
| **ADANIGREEN**| 2018-06 | 619 | 0.4839 ± 0.0684 | 0.4871 | −0.0032 | NO SIGNAL |
| **ETERNAL** | 2021-07 | 375 | 0.5636 ± 0.0357 | 0.5420 | +0.0216 | NO SIGNAL |
| **HDFCBANK** | 2000-01 | 2,150 | 0.5098 ± 0.0325 | 0.5274 | −0.0177 | NO SIGNAL |
| **INFY** | 2000-01 | 2,154 | 0.5395 ± 0.0363 | 0.5525 | −0.0131 | NO SIGNAL |
| **ITC** | 2000-01 | 2,151 | 0.5149 ± 0.0194 | 0.4935 | +0.0214 | NO SIGNAL |
| **RELIANCE** | 2000-01 | 2,151 | 0.4972 ± 0.0121 | 0.5186 | −0.0214 | NO SIGNAL |
| **SBI** | 2000-01 | 2,150 | 0.5153 ± 0.0344 | 0.5395 | −0.0242 | NO SIGNAL |
| **TCS** | 2004-08 | 1,767 | 0.5070 ± 0.0292 | 0.5149 | −0.0079 | NO SIGNAL |
| **VEDL** | 2000-01 | 2,146 | 0.4873 ± 0.0612 | 0.5358 | −0.0485 | NO SIGNAL |

#### Key Finding on Historical Expansion
Across a full 26-year horizon (~2,150 independent 3-day labels per stock), **not a single stock exhibits an edge that survives Monte Carlo permutation testing (`NO SIGNAL`)**.
- Earlier positive signals observed on shorter slices—such as TCS exhibiting +1.54% edge on 780 rows or ITC exhibiting +5.82% on 2014–2026 data—**disappear entirely** when subjected to multi-decade history.
- ETERNAL shows a nominal edge of +2.16%, but with only 375 rows (under five years of listing), this edge is smaller than its fold spread (±3.57%) and fails permutation validation.
- These empirical findings are consistent with the **Weak-Form Efficient Market Hypothesis (EMH)** for Indian equities: within this experimental setup, once sample selection bias and target autocorrelation are removed, past technical indicators provide no statistically significant predictive advantage over honest market baselines.

---

### 2. Does News Sentiment Improve Next-Day Forecasting? (`sentiment_study/`)

A 20+ year multi-market study evaluated whether adding news sentiment ($S$) to technical ($T$) and global-market ($G$) indicators improves out-of-sample directional prediction. Testing spanned four multi-year corpora (Times of India 2001–2022, NYT 1990–2025, FNSPID 2009–2023, and 78,467 NSE corporate announcements 2008–2025):

| Task & Market Scope | Without Sentiment | With Sentiment | Net Sentiment Effect | Statistical Significance |
|---|---|---|---|---|
| **Sensex next-day direction (2009–2022)** | 57.4% – 58.7% | 57.2% – 58.9% | −1.3 to +1.0 pts | $p > 0.10$ (n.s., McNemar) |
| **S&P 500 next-day direction (1998–2025)** | 52.1% – 53.4% | 52.9% – 53.5% | 0.0 to +1.2 pts | AUC +1.5 to +2.3 pts ($p \approx 0.001$) |
| **Sensex opening gap at 09:00 IST** | 79.3% – 80.1% | 78.8% – 80.4% | −1.3 to +0.8 pts | $p > 0.20$ (n.s.) |
| **Nifty opening gap at 09:00 IST (2012–2025)** | 76.1% – 77.9% | 75.6% – 77.8% | −0.5 to −0.1 pts | $p > 0.30$ (n.s.) |
| **NSE corporate filing events (40 stocks, opening gap)** | 69.5% – 69.9% | 69.2% – 69.7% | −0.3 to −0.2 pts | $p > 0.35$ (n.s.) |
| **Same-day direction (Look-ahead diagnostic)** | 51.6% – 58.8% | 57.6% – 66.2% | **+4.4 to +12.1 pts** | **$p < 10^{-14}$ (Contemporaneous)** |

#### The Fundamental Role & Limits of Sentiment
1. **Contemporaneous Explanation vs. Predictive Forecast**: News sentiment is strongly correlated with the **same day's** market movement ($p < 10^{-14}$), because daily news explains what is currently unfolding. However, for **next-day** forecasting, news sentiment yields no statistically significant advantage ($p > 0.05$). By the time headlines are printed, prices have already adjusted.
2. **The Source of High Gap Accuracy**: The ~80% opening-gap accuracy on Sensex does **not** stem from news sentiment; it is driven almost entirely by overnight returns from US markets (S&P 500 close) and early Asian trading (Nikkei/KOSPI opens).
3. **Investigating High Literature Accuracy Claims via Controlled Leakage Demonstrations**: When exploring how certain studies report 75%–90% sentiment accuracy, controlled experiments in `sentiment_study/reports/leakage_demo.csv` demonstrated how specific methodological pitfalls can artificially inflate reported performance: same-day lookahead leakage produced 77% accuracy, moving-average target overlap created multi-day autocorrelation that elevated persistence accuracy to 86%, and non-stationary raw price level features allowed models to achieve spurious 99% directional fit. When these evaluation artifacts are controlled, out-of-sample directional predictive gains from sentiment become statistically indistinguishable from baseline noise.

---

### 3. Company-Targeted Sentiment & Entity-Level FinBERT

Generic headline sentiment fails because an article saying *"Reliance expands market share while Tata steel mills face headwinds"* cannot be assigned a single compound score.

To solve this, a dedicated **Entity-Level FinBERT** (`[CLS] company [SEP] headline`) was trained on **SEntFiN 1.0** (14,371 human-annotated financial entity–headline pairs):

| Sentiment Model Architecture | SEntFiN Test Accuracy (1,440 pairs) | Macro-F1 | Multi-Company Headlines (614 pairs) |
|---|---|---|---|
| **Standard FinBERT** (whole-headline scoring) | 68.54% | 0.6870 | 61.56% |
| **Fine-Tuned Entity FinBERT** (company-targeted) | **87.71%** | **0.8783** | **85.34%** |

#### Company Sentiment Ablation Across All 10 Stocks
Each of the 10 project equities was paired with its specific Times of India headlines (12,426 pairs) and company NSE filings (11,740 filings). In `reports/company_sentiment_ablation.csv`:
- Incorporating company-targeted sentiment into walk-forward models resulted in accuracy deltas between **−2.30%** and **+2.46%**.
- **Zero stocks achieved statistical significance** (all paired McNemar test $p$-values $\ge 0.1152$, with most between $0.69$ and $1.0$).
- A placebo test (shifting entity sentiment by exactly one year) generated identical random performance noise ($\pm 0.5\%$).
- **Sparsity Constraint**: Times of India names an individual stock on only 1% to 40% of trading days (Reliance 40%, Infosys 32%, Adani Green <4%), while the archive ceases in March 2022.

---

### 4. "Why Did It Move?" — Explanatory Move Decomposition

Because sentiment explains rather than predicts, the Streamlit dashboard features a dedicated view: **"Why did it move?"** (implemented in `src/explain_page.py`).

This interface maps daily returns against company-targeted news headlines and NSE corporate announcements, allowing users to dissect why significant moves occurred. To prevent circular logic, price-reporting headlines (e.g., *"Sensex plunges 300 points"*, *"TCS drops 2%"*) were detected and filtered out (`sentiment_study/src/explain_moves.py`):

| Diagnostic Metric (2001–2022, `reports/explain/summary.csv`) | Raw Headlines | Price Reports Removed |
|---|---|---|
| **Sensex: Same-day return correlation (Spearman $\rho$)** | 0.172 | **0.056** ($p = 0.0001$) |
| **Sensex: Next-day return correlation (Spearman $\rho$)** | 0.021 | **0.007** (n.s.) |
| **Sensex: Return spread (Quintile 5 positive vs Quintile 1 negative)** | +0.60% / day | **+0.20% / day** (Next day: −0.03%) |
| **10 Stocks Pooled: Same-day news correlation** | 0.059 | **0.032** ($p = 0.0044$) |
| **10 Stocks Pooled: Next-day news correlation** | 0.019 | **0.013** (n.s.) |

### 5. "Next-Morning Forecast" — Where a Genuine Signal Exists

The third dashboard view (`src/forecast_page.py`, tables from `scripts/build_forecast_results.py`) shows, **stock by stock**,
how well tomorrow's direction is predicted with information available before 09:15 IST: the stock's own indicators, the Nifty,
the S&P 500 overnight return, the VIX and the Asian opening moves. All numbers are walk-forward (never-seen years), logistic regression:

| Prediction (2013–2025, 10 stocks) | Model | Simple guess | Gain | Stocks with a significant signal | Adding company news |
|---|---|---|---|---|---|
| Next morning's opening price | 70.6% | 63.5% | +7.1 pts (p < 0.001) | 9 of 10 | −0.03 pts (n.s.) |
| Next day's closing price | 56.2% | 51.6% | +4.7 pts (p < 0.001) | 7 of 10 | −0.07 pts (n.s.) |

The signal comes from overnight world markets, not from news. These are tested historical results, not a live
forecast, and a forecast made before the open cannot be traded at the previous close.

---

## Seven Models Evaluated

The project implemented and benchmarked seven distinct machine learning and deep learning architectures across multi-decade walk-forward splits (`sentiment_study/reports/comparison_all.csv` and `comparison_dl.csv`):

```
                                  MODEL BENCHMARKING SUITE
                                             │
      ┌──────────────────────────────────────┼──────────────────────────────────────┐
      │                                      │                                      │
Linear & Regularised                   Tree Ensembles                         Sequence Models
      │                                      │                                      │
┌─────┴──────────────┐       ┌───────────────┼───────────────┐       ┌──────────────┴──────────────┐
│ Logistic Regression│       │ Random Forest │  Gradient     │       │     PyTorch LSTM            │
│ (L2-Regularised)   │       │               │  Boosting     │       │                             │
└────────────────────┘       └───────────────┼───────────────┘       └─────────────────────────────┘
                                             │                                      │
                             ┌───────────────┼───────────────┐       ┌──────────────┴──────────────┐
                             │    XGBoost    │   LightGBM    │       │     PyTorch GRU             │
                             └───────────────┴───────────────┘       └─────────────────────────────┘
```

1. **Logistic Regression (L2-Regularized)**:
   - *Design*: Standardized linear classifier with $C=0.1$ regularization.
   - *Role & Findings*: Serves as the parsimonious linear reference. Consistently matched or outperformed non-linear ensembles on genuine out-of-sample forward testing (Sensex next-day accuracy: 58.59%, AUC: 0.6110). Because it is convex and robust to small sample sizes, it was selected as the default production model in `src/pipeline.py`.
2. **Random Forest**:
   - *Design*: Bagging ensemble of 500 trees with heavily constrained leaf sizes (`min_samples_leaf=20`) and square-root feature subsampling.
   - *Role & Findings*: Effectively mitigated single-tree variance, scoring 58.68% next-day accuracy on Sensex. However, it exhibited sensitivity to non-overlapping sampling offsets in smaller datasets.
3. **Histogram Gradient Boosting (Scikit-Learn GBDT)**:
   - *Design*: Shallow gradient-boosted decision trees (`max_leaf_nodes=15`, $lr=0.05$, L2 regularization).
   - *Role & Findings*: Captured non-linear indicator interactions (57.42% accuracy, AUC 0.6068), but proved fragile under offset perturbation tests, flipping sign across equally valid subsampling phases.
4. **XGBoost**:
   - *Design*: Extreme gradient boosted trees with second-order gradient regularization and depth control.
   - *Role & Findings*: Scored 57.36% on next-day Sensex prediction (AUC 0.6053). On large-scale US panel testing (Nasdaq-100), XGBoost achieved 51.25% direction accuracy and 70.01% on big-move filtering.
5. **LightGBM**:
   - *Design*: Highly efficient leaf-wise gradient boosting optimized for histogram-binned stationary indicators.
   - *Role & Findings*: Yielded 57.64% accuracy (AUC 0.6048) on Sensex next-day prediction. In opening-gap prediction ($T+G$), LightGBM scored 58.34%.
6. **LSTM (Long Short-Term Memory Network)**:
   - *Design*: 2-layer stacked PyTorch LSTM with dropout (0.2) predicting stationary log-returns from 30-day trailing sequences.
   - *Role & Findings*: Evaluated in `src/sequence_models.py` and `sentiment_study/src/seq_models.py`. While LSTM beat the naive persistence baseline (repeating yesterday's return), it failed to beat a **Zero-Return baseline** (`beats_zero = False`, RMSE 0.0131 vs 0.0130). Output predictions shrunk to just 9.6% of real return volatility—effectively collapsing into predicting a historical constant.
7. **GRU (Gated Recurrent Unit Network)**:
   - *Design*: 2-layer stacked PyTorch GRU regressor with fewer parameters than LSTM for improved regularization on noisy financial time series.
   - *Role & Findings*: Achieved comparable directional accuracy (51.24% on TCS returns, 58.34% on Sensex) but exhibited the identical variance-shrinkage failure mode (shrinkage factor 0.1196), indicating that within this setup, recurrent architectures struggled to extract actionable temporal autoregressive signal from stationary return sequences beyond estimating a constant mean.

---

## Methodological Rigour & What It Prevents

| Scientific Technique | What It Prevents |
|---|---|
| **Expanding Walk-Forward CV** | A single train/test split's accuracy swinging up to 5.9 points based purely on arbitrary date cutoffs. |
| **52-Day & 17-Day Embargo** | Rolling technical indicators (e.g., 52-day Senkou Span B) straddling the train/test boundary and causing leakage. |
| **Scale-Free Stationary Features** | Models memorizing nominal rupee price levels (e.g., "TCS > 3000") rather than scale-invariant pattern dynamics. |
| **Non-Overlapping Target Sampling** | Forward returns over $h$ days sharing $h-1$ days of identical data, which inflates label autocorrelation to 0.81 and persistence baselines to 0.88. |
| **Monte Carlo Permutation Testing** | False discovery. Shuffled target labels must collapse to chance ($\approx 0.50$); models maintaining high accuracy on shuffled data are flagged for leakage. |
| **Multi-Seed Robustness (7 Seeds)** | Reporting the luckiest random seed initialization out of a distribution. |
| **Sampling Offset Sensitivity** | Directional edges that depend on which of $h$ equally valid subsampling days was picked. |

---

## Model Accuracy vs. Actionable Trading Advantage

A central contribution of this project is proving that **nominal accuracy is completely meaningless in isolation**. In financial machine learning, high accuracy frequently disguises zero predictive skill:

```
                            THE THREE ACCURACY ILLUSIONS
                                         │
    ┌────────────────────────────────────┼────────────────────────────────────┐
    │                                    │                                    │
The Majority Illusion                The Autocorrelation Trap             The Look-Ahead Mirage
    │                                    │                                    │
71% accuracy on big moves            88% accuracy from persistence        66% accuracy from news
looks impressive, but the            due to overlapping labels;           describing today's move;
baseline is 71.37%.                  the model repeats yesterday.         cannot be traded tomorrow.
    │                                    │                                    │
    ▼                                    ▼                                    ▼
NET EDGE: −0.37% (No skill)          NET EDGE: 0.00% (No skill)           NET EDGE: 0.00% (Lookahead)
```

1. **The Majority Class Baseline Trap**:
   - In `reports/all_stocks_summary.csv`, INFY displays **53.95%** walk-forward accuracy. However, its training majority baseline is **55.25%**. The model possesses a negative edge (**−1.31%**).
   - In big-move classification, models score **~71% accuracy**. But because the majority class of big moves is 71.37%, the model has zero actionable edge.
2. **The Target Autocorrelation Trap**:
   - At a 10-day forward horizon, consecutive target labels share 9 days of overlapping returns (autocorrelation = 0.81). A naive persistence model repeating yesterday's label scores **88% accuracy**. Subsampling non-overlapping rows drops autocorrelation to −0.04 and resets the persistence baseline to ~50%.
3. **The Contemporaneous Look-Ahead Trap**:
   - Pairing same-day news headlines with daily returns yields **64%–66% accuracy** ($p < 10^{-14}$). But this reflects contemporaneous description, not prediction. Shifting news by 1 trading day causes accuracy to collapse to the noise floor (~51%–53%).

#### Criteria for a Statistically Valid, Actionable Edge
To claim an authentic predictive advantage, an experiment must simultaneously satisfy five criteria:
1. **Positive Net Edge**: $\text{Accuracy} > \text{Best Honest Baseline}$ (Majority, Persistence, and Stratified Random).
2. **Signal Exceeds Variance**: $\text{Edge} > \sigma_{\text{fold}}$ (edge must exceed fold-to-fold cross-validation standard deviation).
3. **Permutation Significance**: Empirical $p < 0.05$ and $\text{Real Accuracy} > \text{Mean}_{\text{shuffled}} + 2\sigma_{\text{shuffled}}$.
4. **Seed & Offset Robustness**: Edge remains strictly positive across all 7 random seeds and all $h$ sampling offsets.
5. **Economic Viability**: Expected return must survive round-trip transaction costs, Securities Transaction Tax (STT), exchange turnover charges, and bid-ask slippage (~0.10% per trade).

---

## Quick start

```bash
pip install -r requirements.txt
```

```bash
python src/validate.py          # health check
```

Then, in order:

| Notebook | Purpose |
|---|---|
| `01_data_collection.ipynb` | Download price history |
| `00_setup_and_build.ipynb` | Build feature datasets |
| `04_model_training_and_artifacts.ipynb` | Train and save models |

```bash
streamlit run app.py            # dashboard: 'Prediction terminal', 'Why did it move?' and 'Next-morning forecast' views
```

Sentiment study (uses the shipped scored news, no re-download needed):

```bash
python sentiment_study/src/export_to_main.py      # refresh data/processed/IN_market_news_sentiment.csv
python sentiment_study/src/final_tables.py        # with vs without sentiment tables + significance
```

See `sentiment_study/README.md` for the full pipeline.

To add a stock, run `05_add_new_stock.ipynb` and follow
[docs/ADDING_A_STOCK.md](docs/ADDING_A_STOCK.md).

---

## Project structure

```
app.py                Streamlit dashboard (reads models/ and data/processed/)
requirements.txt      Python dependencies
.env.example          Template for the optional NewsAPI key (copy to .env, never commit .env)
src/                  Main pipeline
  config.py           Stock registry: the ONLY place stocks are defined
  collect.py          Registry-driven data download
  dataio.py           Robust CSV loading and cleaning
  features.py         Technical indicators (single source of truth)
  ichimoku.py         Ichimoku Cloud features
  bollinger.py        Bollinger Band features
  stationary.py       Scale-free feature transforms
  targets.py          Label engineering, non-overlapping sampling
  evaluation.py       Walk-forward CV, baselines, permutation test
  abstention.py       Confidence gating, coverage/accuracy curves
  modeling.py         Model zoo, seed/offset robustness harness
  sequence_models.py  LSTM / GRU in PyTorch
  pipeline.py         Train and save artifacts
  dataset.py          Build one stock's modelling dataset (+ optional sentiment merge)
  sentiment_analysis.py  Legacy NewsAPI/VADER/FinBERT scorer (kept for reference)
  validate.py         Project health check
  explain_page.py     Dashboard view "Why did it move?"
  forecast_page.py    Dashboard view "Next-morning forecast" (stock by stock)
scripts/
  extend_history_2000.py      Extend stocks back to 2000, rebuild datasets, retrain
  fix_raw_data.py             Remove holiday filler rows, adjust demergers, retrain
  build_forecast_results.py   Tables for the "Next-morning forecast" view (reports/forecast/)
  company_sentiment_ablation.py  With vs without each stock's own sentiment
  phase1_audit.py             Data / artifact / config consistency audit
data/
  raw/                <STOCK>_raw.csv price history (2000 or listing date onwards)
  processed/          <STOCK>_dataset.csv model inputs,
                      <STOCK>_company_sentiment.csv company-targeted sentiment,
                      IN_market_news_sentiment.csv market-wide sentiment,
                      TCS_news_sentiment.csv old Reddit proxy (reference only),
                      news/<STOCK>_news_items.csv every headline/filing about the stock, scored
models/<STOCK>/       model.joblib, metadata.json, out-of-fold predictions, coverage curve
reports/              Result tables (all_stocks_summary.csv, phase*.csv, company_sentiment_ablation.csv)
  figures/            Charts used in the write-up
notebooks/            Step-by-step notebooks (setup, EDA, features, targets, training, Ichimoku, Bollinger)
docs/                 ADDING_A_STOCK.md
sentiment_study/      Leakage-controlled sentiment study (self-contained, see its README)
  src/                Data download, news scoring (FinBERT, VADER, LM, entity-level FinBERT),
                      walk-forward experiments, LSTM/GRU, Methods 1 and 4, company-targeted study
  data/processed/     Scored headlines, filings and daily sentiment tables
  data/raw/sentfin/   SEntFiN 1.0 training data for the entity-level model (MIT)
  reports/            Result tables and out-of-fold predictions
  reports/explain/    "Why did it move?" statistics (explain_moves.py)
```

Large or licence-restricted files (Yahoo price downloads, NYT and FNSPID headlines, the raw NSE
filings JSON, the 440 MB fine-tuned model) are listed in `.gitignore`: they stay on your computer
and are not uploaded. `sentiment_study/src/download_data.py` and `train_entity_model.py` rebuild them.

Analysis notebooks 02, 03, 06, 07 document the experiments behind each
design decision and are worth reading before changing anything.

---

## Sentiment analysis

Sentiment now comes from a 20-year news archive instead of the Reddit r/worldnews
proxy (ROC-AUC 0.47 for TCS, kept as `data/processed/TCS_news_sentiment.csv` for reference):

| Corpus | Coverage | Used for |
|---|---|---|
| Times of India headlines (Harvard Dataverse, CC0) | 2001–2022, 276k market headlines | Indian market sentiment |
| New York Times headlines + abstracts (NYT Archive API) | 1990–2025, minute timestamps | US market, overnight news |
| FNSPID company news | 2009–2023, 38 Nasdaq-100 stocks | Company-level panel |
| NSE corporate announcements | 2008–2025, 40 Nifty stocks, exact times | Company event days |

**Per-stock, company-targeted files (default since 31 Oct 2026).** `config.py` registers
`data/processed/<KEY>_company_sentiment.csv` for every stock, written by
`python sentiment_study/src/export_company_sentiment.py`:

| Column | Meaning |
|---|---|
| `finbert_sentiment` | entity-level FinBERT score towards this company (ToI headlines naming it), EMA half-life 3 days |
| `vader_sentiment` | VADER on the same headlines, same smoothing |
| `company_news_count` | headlines naming the company that day |
| `filings_sentiment` | entity-level score of the company's own NSE filings (15:30–09:00 IST window), same smoothing |
| `filings_count` | non-routine filings that day |

Headline columns are 0 after March 2022 (archive end). The market-wide file
`IN_market_news_sentiment.csv` (`export_to_main.py`) is kept for the comparison. The main pipeline
still trains with `with_sentiment=False`; `python scripts/company_sentiment_ablation.py` reruns
with vs without for every stock.

Rebuilding the targeted scores: `sentiment_study/src/train_entity_model.py` (≈30 min on 2 CPU cores;
needs `data/raw/finbert` and `data/raw/sentfin/SEntFiN.csv`) → `company_news.py` → `score_entity.py`
→ `export_company_sentiment.py`. Score in full precision: int8 quantisation flattens the fine-tuned model.

---

## Known limitations

- **Company news is sparse.** Times of India names a given company on 1–40% of
  trading days (Reliance and Infosys most; Adani Green and Zomato under 4%); filings
  cover 7–27% of days. No free, dense, 20-year company-level Indian news archive exists.
- **The Times of India archive ends in March 2022**, so sentiment-enabled
  datasets stop there.
- **Probabilities are not calibrated.** The Brier score sits near 0.25.
  Treat confidence as a relative ranking, not a literal probability.
- **No transaction costs or slippage** are modelled. Over the full 2000–2026 history, no stock exhibits an edge that survives Monte Carlo permutation testing (`NO SIGNAL`), and any small nominal edge observed on short-window samples would be completely erased by Securities Transaction Tax (STT), brokerage fees, and bid-ask slippage (~0.10% per trade).
- **Effect sizes are within the noise floor.** Out-of-sample directional prediction on stationary financial series hovers at the random/majority baseline (~50–53%), consistent with the Weak-Form Efficient Market Hypothesis.

---

## Future work

- Live news integration via the `collect_news` interface
- Company-level Indian news with timestamps (results-PDF tone, LiveMint/Moneycontrol)
- LLM-based headline scoring in place of FinBERT
- Triple-barrier labelling (López de Prado)
- Continuous order-flow and intraday microstructure signals

---

## Not investment advice

This is an academic research project. Over multi-decade walk-forward validation (2000–2026), no model demonstrates a statistically significant predictive advantage over honest market baselines. Do not trade on this system.
