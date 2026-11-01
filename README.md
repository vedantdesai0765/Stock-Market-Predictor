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

### 1. Does news sentiment improve prediction? (sentiment study, `sentiment_study/`)

Leakage-controlled comparison over 20+ years: technical (T), global-market (G) and
news-sentiment (S) features; Logistic Regression, Random Forest, Gradient Boosting,
XGBoost, LightGBM, LSTM and GRU; walk-forward by year with paired McNemar and
block-bootstrap tests.

| Task (out-of-sample, market-wide sentiment) | Without sentiment | With sentiment | Sentiment effect |
|---|---|---|---|
| Sensex next-day direction, 2009–2022 (Times of India news) | 57.4–58.7% | 57.2–58.9% | −1.3 to +1.0 pts, n.s. |
| S&P 500 next-day direction, 1998–2025 (NYT news to 09:00 ET) | 52.1–53.4% | 52.9–53.5% | 0 to +1.2 pts; AUC +1.5 to +2.3 pts |
| Sensex opening gap at 09:00 IST (US close + Asian opens) | 79.3–80.1% | 78.8–80.4% | −1.3 to +0.8 pts, n.s. |
| Nifty opening gap at 09:00 IST, 2012–2025 | 76.1–77.9% | 75.6–77.8% | n.s. |
| NSE company filing events, 40 stocks, 2013–2025 (opening gap) | 69.5–69.9% | 69.2–69.7% | n.s. |
| Same-day direction (look-ahead: news describes the move) | 52–58% | 58–66% | +4 to +12 pts (explanatory only) |

Sentiment explains the day it describes but adds at most 1–2.5 points to genuine
forecasts. The ~80% opening-gap accuracy comes from overnight US and Asian prices.
Setups that report 77–99% "accuracy" on the same data (same-day headlines, moving-average
targets, price-level R²) are reproduced and explained in `sentiment_study/reports/leakage_demo.csv`.
Full write-up: `sentiment_study/README.md`.

### 2. Per-stock pipeline (`src/`, 3-day horizon, non-overlapping labels)

Price history now starts in 2000 for six stocks (TCS from its August 2004 listing, ADANIENT
from mid-2004, when Yahoo's series becomes clean; ADANIGREEN and ETERNAL from their listings).
Extended with `scripts/extend_history_2000.py` and cleaned with `scripts/fix_raw_data.py` (holiday filler rows
removed; the ADANIENT 2015 and VEDL 2026 demergers adjusted). Current artifacts in `reports/all_stocks_summary.csv`:

| Stock | Data from | Rows | Accuracy | Baseline | Edge | Permutation |
|---|---|---|---|---|---|---|
| ADANIENT | 2004-07 | 1773 | 0.5310 ± 0.0199 | 0.5320 | -0.0010 | NO SIGNAL |
| ADANIGREEN | 2018-06 | 619 | 0.4839 ± 0.0684 | 0.4871 | -0.0032 | NO SIGNAL |
| ETERNAL | 2021-07 | 375 | 0.5636 ± 0.0357 | 0.5420 | +0.0216 | NO SIGNAL |
| HDFCBANK | 2000-01 | 2150 | 0.5098 ± 0.0325 | 0.5274 | -0.0177 | NO SIGNAL |
| INFY | 2000-01 | 2154 | 0.5395 ± 0.0363 | 0.5525 | -0.0131 | NO SIGNAL |
| ITC | 2000-01 | 2151 | 0.5149 ± 0.0194 | 0.4935 | +0.0214 | NO SIGNAL |
| RELIANCE | 2000-01 | 2151 | 0.4972 ± 0.0121 | 0.5186 | -0.0214 | NO SIGNAL |
| SBI | 2000-01 | 2150 | 0.5153 ± 0.0344 | 0.5395 | -0.0242 | NO SIGNAL |
| TCS | 2004-08 | 1767 | 0.5070 ± 0.0292 | 0.5149 | -0.0079 | NO SIGNAL |
| VEDL | 2000-01 | 2146 | 0.4873 ± 0.0612 | 0.5358 | -0.0485 | NO SIGNAL |

No stock shows an edge that survives the permutation test. ETERNAL's +2.2-point edge (375 rows, under
five years) is smaller than its fold spread (±3.6 points) and fails the permutation test.
The earlier TCS result (+1.5 points on 780 rows) and the ITC result on 2014–2026 data
(+5.8 points) both disappear on the longer sample. **Price direction is not predictable in
aggregate**, which is the expected outcome under the efficient-market hypothesis. The
contribution is the measurement apparatus that establishes it credibly.

### 3. Company-targeted sentiment (each stock scored on news about itself)

Each of the 10 stocks gets its own daily score. Inputs are the Times of India headlines that name the
company (12,426 headline–company pairs, 2001–2022) and the company's own NSE filings (11,740, 2008–2025).
The scorer is an **entity-level FinBERT**: `[CLS] company [SEP] headline`, fine-tuned on SEntFiN 1.0
(14,371 labelled company–headline pairs). It is asked how the headline reads *for that company*,
so a headline naming two companies can get two different scores. Plain FinBERT gives the whole headline one score.

| Sentiment scorer (SEntFiN held-out test, 1,440 pairs) | Accuracy | Macro-F1 | Multi-company headlines |
|---|---|---|---|
| Plain FinBERT (one score per headline) | 68.5% | 0.687 | 61.6% |
| Entity-level FinBERT (score towards the named company) | **87.7%** | **0.878** | **85.3%** |

| Task (walk-forward, 10-stock panel) | Without | With company-targeted sentiment | Effect |
|---|---|---|---|
| Opening gap, Times of India period 2007–2022 | 71.4–72.1% | 71.3–72.0% | −0.2 to −0.1 pts |
| Next-day close, Times of India period 2007–2022 | 57.3–58.2% | 57.4–58.0% | −0.2 to +0.1 pts, n.s. |
| Opening gap, NSE filings 2013–2025 | 70.6–70.7% | 70.6–70.8% | 0 to +0.1 pts, n.s. |
| Next-day close, NSE filings 2013–2025 | 56.2–56.7% | 56.2–56.9% | −0.1 to +0.2 pts, n.s. |
| Main pipeline, per stock (logistic, 3-day horizon) | 47–56% | 47–56% | −2.3 to +2.5 pts, none p < 0.05 |

The targeted scorer measures sentiment far better, but the better measurement does not turn into better
forecasts. Targeted scoring beats plain FinBERT by at most +0.3 points, and a placebo (scores shifted by one
year) moves accuracy by the same ±0.5 points. On news days, the targeted score correlates with the **same
day's** return (Spearman 0.076, p < 10⁻¹⁰) more strongly than the plain score (0.066). Its link to the next
day is small (0.02–0.05). By the time a headline is dated, the price has mostly moved. Details:
`sentiment_study/reports/company_sentiment_comparison.csv`, `company_sentiment_per_stock.csv`,
`reports/company_sentiment_ablation.csv`.

### 4. Why did it move? Sentiment as an explanation of the day's move

The dashboard's **Why did it move?** view puts every headline and filing about a company next to that day's
price move, scored for sentiment towards the company. `sentiment_study/src/explain_moves.py` measures how
closely news tone and same-day moves line up. Headlines that only report the move ("Sensex down 200 points",
"Infosys shares fall 3%") are removed, because they describe the move rather than explain it:

| Measure (2001–2022) | All headlines | Price reports removed |
|---|---|---|
| Sensex: same-day correlation with return (Spearman) | 0.172 | 0.056 (p < 0.001) |
| Sensex: correlation with the next day's return | 0.021 | 0.007 |
| Sensex: return gap, most positive vs most negative fifth of days | 0.60 pts/day | 0.20 pts/day (next day −0.03) |
| Sensex: largest 5% of moves where news tone matched / opposed the move | | 46% / 39% |
| 10 stocks pooled: same-day correlation on company-news days | 0.059 | 0.032 (p = 0.004) |
| Sensex same-day up/down accuracy, logistic (57.7% without sentiment) | 60.9% (+3.2 pts, p < 0.001) | 58.2% (+0.5 pts, n.s.) |

News tone lines up with the same day's move weakly but measurably, and not at all the next day. Most of
the apparent explanatory power comes from the 2–4% of headlines that report the move itself; a newspaper
dated d also often reports news from day d−1, so part of the move shows up the day before the headline.
Tables: `sentiment_study/reports/explain/`.

---

## What makes this rigorous

| Technique | What it prevents |
|---|---|
| Walk-forward validation | A single train/test split's accuracy swings 5.9 points depending only on where the line is drawn |
| 52-day embargo | Rolling-window features straddling the train/test boundary |
| Permutation testing | Leakage. Shuffled labels must collapse to ~0.50 |
| Stationary features | Models memorising price levels instead of patterns |
| Non-overlapping labels | Target autocorrelation of 0.81 inflating the persistence baseline to 0.88 |
| Seed averaging | Reporting the luckiest of seven random seeds |
| Offset testing | Edges that depend on an arbitrary sampling choice |

### Three false positives this apparatus caught

**Phase 1 reported TCS at 59.31%, +8.9 points over baseline, permutation-
clean.** Sliding the same 565-row window across the decade showed that
window was the only one of ten with a positive edge; the other nine
averaged −2.2 points. Permutation testing detects leakage but cannot
detect an unrepresentative slice of history.

**The LSTM appeared to beat its baseline.** It beat *persistence*, but
loses to simply predicting zero — and its outputs carry only 9.6% of the
volatility of real returns. It had collapsed to predicting a constant.

**Gradient boosting looked like the best model** at +2.8 points, until the
offset test showed it flipping to −3.9 points on one of three equally
valid sampling offsets. Logistic regression, at a smaller +1.5 points, was
positive on all three and became the default.

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
streamlit run app.py            # dashboard: 'Prediction terminal' and 'Why did it move?' views
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
scripts/
  extend_history_2000.py      Extend stocks back to 2000, rebuild datasets, retrain
  fix_raw_data.py             Remove holiday filler rows, adjust demergers, retrain
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
- **No transaction costs or slippage** are modelled. A 1.5-point edge would
  likely not survive them.
- **Effect sizes are near the noise floor.** Report them as such.

---

## Future work

- Live news integration via the `collect_news` interface
- Company-level Indian news with timestamps (results-PDF tone, LiveMint/Moneycontrol)
- LLM-based headline scoring in place of FinBERT
- Triple-barrier labelling (López de Prado)
- Longer histories for all stocks

---

## Not investment advice

This is an academic project. The measured edge is small, unvalidated on
live data, and ignores transaction costs. Do not trade on it.
