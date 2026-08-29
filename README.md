# Stock Market Predictor

Next-move direction prediction for Indian equities, built on technical
indicators with a rigorously validated machine-learning pipeline.

A B.Tech final-year project. The emphasis is on **methodological
correctness over headline accuracy**: every claim below is backed by
walk-forward validation, permutation testing, and seed/offset robustness
checks, and results that failed those checks were discarded rather than
reported.

---

## Headline result

| Stock | Rows | Accuracy | Baseline | Edge | Permutation |
|---|---|---|---|---|---|
| TCS | 780 | 0.5333 ± 0.0525 | 0.5179 | **+0.0154** | SIGNAL |
| RELIANCE | 372 | 0.4679 ± 0.0685 | 0.5111 | −0.0432 | NO SIGNAL |

The honest summary: **one-day price direction is not predictable in
aggregate.** Restricting to independent labels at a three-day horizon
yields a small edge on TCS (~1.5 percentage points) that survives
permutation testing and holds across all random seeds and sampling
offsets. Reliance shows no signal at any configuration — it has only two
years of history.

This is the expected outcome under the efficient-market hypothesis. The
contribution is the measurement apparatus that establishes it credibly.

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
streamlit run app.py            # dashboard
```

To add a stock, run `05_add_new_stock.ipynb` and follow
[docs/ADDING_A_STOCK.md](docs/ADDING_A_STOCK.md).

---

## Project structure

```
src/
  config.py       Stock registry — the ONLY place stocks are defined
  collect.py      Registry-driven data download
  dataio.py       Robust CSV loading
  features.py     Technical indicators (single source of truth)
  stationary.py   Scale-free feature transforms
  targets.py      Label engineering, non-overlapping sampling
  evaluation.py   Walk-forward CV, baselines, permutation test
  abstention.py   Confidence gating, coverage/accuracy curves
  modeling.py     Model zoo, seed/offset robustness harness
  sequence_models.py  LSTM / GRU in PyTorch
  pipeline.py     Train and save artifacts
  validate.py     Project health check
  ichimoku.py     Ichimoku Cloud features
  bollinger.py    Bollinger Band features
```

Analysis notebooks 02, 03, 06, 07 document the experiments behind each
design decision and are worth reading before changing anything.

---

## Sentiment analysis

The project implements VADER and FinBERT scoring, but **sentiment is
excluded from the model**.

The available corpus is Reddit r/worldnews (2008–2016), not company news.
Measured as a predictor of TCS direction its ROC-AUC was **0.47** — below
chance. The machinery is correct; the input data was wrong for the task.
The dashboard displays these scores with that caveat attached.

Using them properly requires a historical company-news archive. The free
NewsAPI tier serves only ~30 days, far too little to train on.

---

## Known limitations

- **Reliance has insufficient history** (493 raw rows, ~2 years). Its
  registry entry starts in 2022; extending it to 2014 would likely change
  its verdict.
- **Probabilities are not calibrated.** The Brier score sits near 0.25.
  Treat confidence as a relative ranking, not a literal probability.
- **No transaction costs or slippage** are modelled. A 1.5-point edge would
  likely not survive them.
- **Effect sizes are near the noise floor.** Report them as such.

---

## Future work

- Live news integration via the `collect_news` interface
- Historical company-news corpus to test sentiment properly
- Triple-barrier labelling (López de Prado)
- Longer histories for all stocks

---

## Not investment advice

This is an academic project. The measured edge is small, unvalidated on
live data, and ignores transaction costs. Do not trade on it.
