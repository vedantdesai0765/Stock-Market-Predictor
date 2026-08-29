# Adding a New Stock

This project is designed so that adding a stock requires **no code changes
except one entry in a registry**. This document is the complete procedure.

Estimated time: about 10 minutes, most of it waiting for training.

---

## Before you start

Run the health check. It confirms the pipeline is intact before you change
anything, so if something breaks later you know it was your change.

```python
# In a notebook, or: python src/validate.py
import sys; sys.path.insert(0, "src")
from validate import run_all_checks
run_all_checks()
```

Proceed only if there are **0 failures**. Warnings are fine.

---

## The five steps

### 1. Find the ticker symbol

Look it up on finance.yahoo.com. Indian equities need a suffix:

| Exchange | Suffix | Example |
|---|---|---|
| NSE | `.NS` | `INFY.NS` |
| BSE | `.BO` | `500209.BO` |

Getting this wrong is the most common failure. `collect_prices()` raises a
clear error if the ticker returns nothing.

### 2. Generate the registry entry

```python
from collect import registry_template
print(registry_template("INFY", "INFY.NS", "Infosys", start="2014-01-01"))
```

This prints a ready-to-paste block.

### 3. Paste it into `src/config.py`

Find the `STOCKS` dictionary and add the block inside it. Save the file.
**Restart the kernel** — Python caches imported modules, so the new entry
will not appear until you do.

```python
STOCKS: dict[str, StockConfig] = {
    "TCS": StockConfig(...),
    "RELIANCE": StockConfig(...),
    "INFY": StockConfig(          # <-- your new block
        key="INFY",
        ticker="INFY.NS",
        display_name="Infosys",
        raw_filename="INFY_raw.csv",
        start="2014-01-01",
        end="2024-01-01",
        sentiment_filename=None,
        notes="Added 2026-08-29.",
    ),
}
```

### 4. Download and verify

```python
from collect import collect_prices, verify_raw_file
collect_prices("INFY")
verify_raw_file("INFY")
```

**Check the row count.** You want 900+ rows, roughly four years of daily
data. Below that, the pipeline automatically falls back to a shorter
prediction horizon, which is a weaker question. Reliance sits at 493 rows
and shows no signal at any horizon — that is a data problem, not a model
problem.

### 5. Train

```python
from pipeline import TrainConfig, train_stock
train_stock("INFY", cfg=TrainConfig())
```

This writes `models/INFY/` and the stock appears in the dashboard
automatically. Nothing else needs editing.

Or just run **`notebooks/05_add_new_stock.ipynb`**, which walks through all
five steps with checks at each stage.

---

## Reading the output

Training prints a block like this:

```
=== Training INFY (Infosys) ===
  full frame     : 2290 rows, 52 features
  modelling frame: 759 rows (horizon=3, k=0.0)
  walk-forward   : 0.5333 +/- 0.0525 (baseline 0.5179, edge +0.0154)
  seed-averaged  : 0.5333 (edge +0.0154) -> POSITIVE
  permutation    : SIGNAL (shuffled 0.4979)
  operating point: coverage 100%, accuracy 0.5333, edge +0.0308
  saved -> models/INFY/
```

What to look at, in order of importance:

| Line | Meaning | Action |
|---|---|---|
| `permutation: LEAKAGE` | Model reads the answer somewhere | **Stop.** Do not use it. Investigate. |
| `permutation: NO SIGNAL` | Pipeline clean, no skill | Valid result. Report it as such. |
| `edge` negative | Loses to the baseline | Valid result. Dashboard shows a warning. |
| `edge` positive + `SIGNAL` | Real but small skill | Usable. Still check offsets. |
| `horizon` fell below 3 | Not enough history | Re-download with an earlier start date. |

**A negative edge is not a bug.** Most stocks will show no predictable
next-move direction — that is what the efficient-market literature
predicts, and the project measured it rigorously across four phases. The
dashboard displays a red banner for such stocks rather than a
confident-looking signal.

---

## Always check offset robustness

At horizon 3 there are three equally valid ways to subsample
non-overlapping labels. Which one you happen to use has no financial
meaning, so any edge that depends on it is an artifact.

Section 10 of notebook 05 runs this automatically. In Phase 4 both tree
ensembles flipped negative on one of the three offsets while logistic
regression stayed positive on all three — which is why logistic is the
default model.

**Report the mean edge across offsets, never the best one.**

---

## Troubleshooting

**`KeyError: Unknown stock 'INFY'`**
The registry entry was not saved, or the kernel was not restarted.

**`ValueError: No data returned for ticker`**
Wrong symbol. Check the `.NS` / `.BO` suffix on finance.yahoo.com.

**`ValueError: only N modelling rows after trying horizons [...]`**
Too little history even at horizon 1. Re-download with an earlier start
date, or pick a stock with a longer listing history.

**`FileNotFoundError: No trained artifacts`**
Run `train_stock(key)` before opening the dashboard.

**Dashboard does not show the new stock**
The dropdown only lists stocks with saved artifacts. Confirm
`models/<KEY>/model.joblib` exists, then restart Streamlit.

**`InconsistentVersionWarning` when loading a model**
The model was pickled with a different scikit-learn version. Retrain with
`train_all(overwrite=True)` to regenerate artifacts under your version.

---

## What NOT to change

These carry the correctness guarantees earned over Phases 0–4. Changing
them silently invalidates every reported result.

| File | Why it is load-bearing |
|---|---|
| `features.py` | Single source of truth for indicators; fixes the fabricated-target bug |
| `stationary.py` | Makes features scale-free, which is what allows cross-stock transfer |
| `targets.py` | Non-overlapping sampling; labels computed before subsampling |
| `evaluation.py` | Walk-forward folds, embargo, baselines, permutation test |

If you must change one, re-run notebooks 01 through 04 and regenerate every
number in `reports/`. Do not mix old and new results in one table.
