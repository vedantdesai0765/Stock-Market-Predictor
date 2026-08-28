# src/evaluation.py
"""
Walk-forward evaluation harness.

Why this module exists
----------------------
The original project reported a single 80/20 split. On TCS that means a
113-row test set, where flipping four predictions moves accuracy by 3.5
percentage points. Any claim of the form "sentiment improved accuracy by
2 points" was inside the noise floor and could not be defended.

This module replaces that with:

  1. Expanding-window walk-forward splits -> several independent test folds,
     so results carry a mean AND a standard deviation.
  2. An embargo gap between train and test, so rolling-window features
     computed near the boundary do not straddle the split.
  3. Three reference baselines reported alongside every result, because
     accuracy alone is uninterpretable on a near-50/50 target.

Nothing here is stock-specific.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Iterator, Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, roc_auc_score)

try:
    import config
except ImportError:  # pragma: no cover
    from src import config


# ---------------------------------------------------------------------------
# Splitting
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Fold:
    """One train/test split, with the embargo already applied."""

    index: int
    train_idx: np.ndarray
    test_idx: np.ndarray
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    embargo_dropped: int

    def describe(self) -> dict:
        return {
            "fold": self.index,
            "train_rows": len(self.train_idx),
            "test_rows": len(self.test_idx),
            "train_period": f"{self.train_start.date()} .. {self.train_end.date()}",
            "test_period": f"{self.test_start.date()} .. {self.test_end.date()}",
            "embargoed": self.embargo_dropped,
        }


def walk_forward_splits(dates: pd.Series,
                        n_splits: int = 5,
                        embargo: int = 52,
                        min_train_size: Optional[int] = None,
                        expanding: bool = True) -> list[Fold]:
    """
    Build expanding (or rolling) walk-forward folds over a sorted date series.

    Parameters
    ----------
    dates : pd.Series of datetime64, sorted ascending, same length as the data
    n_splits : number of test folds
    embargo : trading days removed from the END of each training block.
        Set this to at least the longest rolling window in your feature set.
        Ichimoku Senkou B uses 52 bars, so 52 is the safe default here.
    min_train_size : rows required in the first training block. Defaults to
        roughly half the dataset, which keeps folds reasonably sized on the
        small (n=443) Reliance set.
    expanding : True  -> training block grows each fold (recommended)
                False -> fixed-width rolling window

    Returns
    -------
    list[Fold]
    """
    n = len(dates)
    dates = pd.to_datetime(pd.Series(dates).reset_index(drop=True))

    if not dates.is_monotonic_increasing:
        raise ValueError("dates must be sorted ascending before splitting")
    if n_splits < 2:
        raise ValueError(f"n_splits must be >= 2, got {n_splits}")

    if min_train_size is None:
        min_train_size = max(int(n * 0.5), 100)

    usable = n - min_train_size
    if usable < n_splits * 20:
        raise ValueError(
            f"Not enough rows for {n_splits} folds: {n} total, "
            f"{min_train_size} reserved for initial training, "
            f"leaving {usable} for testing. "
            f"Reduce n_splits or min_train_size."
        )

    test_size = usable // n_splits
    folds: list[Fold] = []

    for i in range(n_splits):
        test_start = min_train_size + i * test_size
        test_end = test_start + test_size if i < n_splits - 1 else n

        train_end_raw = test_start
        train_end = max(train_end_raw - embargo, 0)
        train_start = 0 if expanding else max(train_end - min_train_size, 0)

        if train_end - train_start < 50:
            raise ValueError(
                f"Fold {i} has only {train_end - train_start} training rows "
                f"after a {embargo}-day embargo. Lower the embargo or n_splits."
            )

        train_idx = np.arange(train_start, train_end)
        test_idx = np.arange(test_start, test_end)

        folds.append(Fold(
            index=i,
            train_idx=train_idx,
            test_idx=test_idx,
            train_start=dates.iloc[train_start],
            train_end=dates.iloc[train_end - 1],
            test_start=dates.iloc[test_start],
            test_end=dates.iloc[test_end - 1],
            embargo_dropped=train_end_raw - train_end,
        ))

    return folds


def describe_folds(folds: Sequence[Fold]) -> pd.DataFrame:
    """Human-readable fold table, for the report."""
    return pd.DataFrame([f.describe() for f in folds]).set_index("fold")


# ---------------------------------------------------------------------------
# Baselines
# ---------------------------------------------------------------------------

def baseline_majority(y_train: pd.Series, y_test: pd.Series) -> np.ndarray:
    """Always predict whichever class dominated the TRAINING data."""
    majority = int(y_train.mean() >= 0.5)
    return np.full(len(y_test), majority)


def baseline_persistence(y_train: pd.Series, y_test: pd.Series) -> np.ndarray:
    """
    Predict that tomorrow repeats today's direction.

    Uses the last training label to seed the first test prediction, so no
    test information leaks backwards.
    """
    seed = int(y_train.iloc[-1])
    shifted = y_test.shift(1)
    shifted.iloc[0] = seed
    return shifted.astype(int).to_numpy()


def baseline_random(y_train: pd.Series, y_test: pd.Series,
                    random_state: int = config.RANDOM_STATE) -> np.ndarray:
    """Stratified random guess drawn from the training class distribution."""
    dummy = DummyClassifier(strategy="stratified", random_state=random_state)
    dummy.fit(np.zeros((len(y_train), 1)), y_train)
    return dummy.predict(np.zeros((len(y_test), 1)))


BASELINES: dict[str, Callable] = {
    "majority": baseline_majority,
    "persistence": baseline_persistence,
    "random": baseline_random,
}


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(y_true, y_pred, y_proba=None) -> dict:
    """Accuracy, precision, recall, F1 and (optionally) ROC-AUC."""
    out = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }
    if y_proba is not None and len(np.unique(y_true)) > 1:
        try:
            out["roc_auc"] = roc_auc_score(y_true, y_proba)
        except ValueError:
            out["roc_auc"] = np.nan
    else:
        out["roc_auc"] = np.nan
    return out


# ---------------------------------------------------------------------------
# The harness
# ---------------------------------------------------------------------------

@dataclass
class WalkForwardResult:
    """Everything produced by one walk-forward evaluation run."""

    label: str
    per_fold: pd.DataFrame
    baselines: pd.DataFrame
    folds: list[Fold]
    feature_names: list[str]
    oof_predictions: pd.DataFrame
    fitted_models: list = field(default_factory=list)

    # -- summaries ---------------------------------------------------------

    def summary(self) -> pd.Series:
        """Mean and std of every metric across folds."""
        means = self.per_fold.mean(numeric_only=True)
        stds = self.per_fold.std(numeric_only=True)
        out = {}
        for k in means.index:
            out[f"{k}_mean"] = means[k]
            out[f"{k}_std"] = stds[k]
        out["n_folds"] = len(self.folds)
        return pd.Series(out, name=self.label)

    def headline(self) -> str:
        s = self.summary()
        return (f"{self.label}: accuracy "
                f"{s['accuracy_mean']:.4f} +/- {s['accuracy_std']:.4f} "
                f"over {int(s['n_folds'])} folds")

    def edge_over_baselines(self) -> pd.Series:
        """Mean accuracy minus each baseline's mean accuracy."""
        model_acc = self.per_fold["accuracy"].mean()
        return pd.Series(
            {f"vs_{b}": model_acc - self.baselines[b].mean()
             for b in self.baselines.columns},
            name=self.label,
        )

    def report(self) -> None:
        print(f"\n{'=' * 62}")
        print(f"  {self.label}")
        print(f"{'=' * 62}")
        print(f"  Features : {len(self.feature_names)}")
        print(f"  Folds    : {len(self.folds)}")
        print()
        print("  Per-fold metrics")
        print(self.per_fold.round(4).to_string())
        print()
        print("  Baseline accuracy per fold")
        print(self.baselines.round(4).to_string())
        print()
        s = self.summary()
        print(f"  MEAN accuracy : {s['accuracy_mean']:.4f} "
              f"+/- {s['accuracy_std']:.4f}")
        print(f"  MEAN f1       : {s['f1_mean']:.4f} +/- {s['f1_std']:.4f}")
        if not np.isnan(s.get("roc_auc_mean", np.nan)):
            print(f"  MEAN roc_auc  : {s['roc_auc_mean']:.4f} "
                  f"+/- {s['roc_auc_std']:.4f}")
        print()
        for name, val in self.edge_over_baselines().items():
            verdict = "BEATS" if val > 0 else "LOSES TO"
            print(f"  {name:18s} {val:+.4f}   ({verdict} baseline)")
        print()
        print(f"  VERDICT: {self.verdict()}")

    def verdict(self) -> str:
        """
        A blunt readout. An edge smaller than one fold-to-fold standard
        deviation is not evidence of anything.
        """
        acc_mean = self.per_fold["accuracy"].mean()
        acc_std = self.per_fold["accuracy"].std()
        best_baseline = self.baselines.mean().max()
        edge = acc_mean - best_baseline

        if edge <= 0:
            return ("No signal. Model does not beat the best baseline "
                    f"({best_baseline:.4f}).")
        if edge < acc_std:
            return (f"Inconclusive. Edge of {edge:+.4f} is smaller than the "
                    f"fold-to-fold std ({acc_std:.4f}); likely noise.")
        return (f"Signal. Edge of {edge:+.4f} exceeds the fold-to-fold std "
                f"({acc_std:.4f}).")


def walk_forward_evaluate(df: pd.DataFrame,
                          feature_cols: Sequence[str],
                          model,
                          label: str = "model",
                          target_col: str = "Target",
                          date_col: str = "Date",
                          n_splits: int = 5,
                          embargo: int = 52,
                          min_train_size: Optional[int] = None,
                          expanding: bool = True,
                          keep_models: bool = False,
                          verbose: bool = False) -> WalkForwardResult:
    """
    Evaluate one model across walk-forward folds.

    The model is cloned and refit from scratch on every fold, so no state
    leaks between folds. Baselines are computed on the identical splits.
    """
    feature_cols = [c for c in feature_cols if c in df.columns]
    if not feature_cols:
        raise ValueError("No usable feature columns found in the DataFrame.")

    leaky = sorted(set(feature_cols) & config.FORBIDDEN_FEATURES)
    if leaky:
        raise ValueError(
            f"Refusing to evaluate: forbidden columns in feature set: {leaky}"
        )

    df = df.sort_values(date_col).reset_index(drop=True)
    X = df[list(feature_cols)]
    y = df[target_col].astype(int)

    folds = walk_forward_splits(
        df[date_col], n_splits=n_splits, embargo=embargo,
        min_train_size=min_train_size, expanding=expanding,
    )

    rows, base_rows, oof_parts, models = [], [], [], []

    for fold in folds:
        X_tr, y_tr = X.iloc[fold.train_idx], y.iloc[fold.train_idx]
        X_te, y_te = X.iloc[fold.test_idx], y.iloc[fold.test_idx]

        est = clone(model)
        est.fit(X_tr, y_tr)
        y_pred = est.predict(X_te)

        y_proba = None
        if hasattr(est, "predict_proba"):
            try:
                y_proba = est.predict_proba(X_te)[:, 1]
            except Exception:
                y_proba = None

        metrics = compute_metrics(y_te, y_pred, y_proba)
        metrics["fold"] = fold.index
        rows.append(metrics)

        base_row = {"fold": fold.index}
        for bname, bfunc in BASELINES.items():
            base_row[bname] = accuracy_score(y_te, bfunc(y_tr, y_te))
        base_rows.append(base_row)

        oof_parts.append(pd.DataFrame({
            "fold": fold.index,
            date_col: df[date_col].iloc[fold.test_idx].to_numpy(),
            "y_true": y_te.to_numpy(),
            "y_pred": y_pred,
            "y_proba": y_proba if y_proba is not None else np.nan,
        }))

        if keep_models:
            models.append(est)

        if verbose:
            print(f"    fold {fold.index}: acc={metrics['accuracy']:.4f} "
                  f"(train {len(X_tr)}, test {len(X_te)})")

    per_fold = pd.DataFrame(rows).set_index("fold")
    baselines = pd.DataFrame(base_rows).set_index("fold")
    oof = pd.concat(oof_parts, ignore_index=True)

    return WalkForwardResult(
        label=label,
        per_fold=per_fold,
        baselines=baselines,
        folds=folds,
        feature_names=list(feature_cols),
        oof_predictions=oof,
        fitted_models=models,
    )


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------

def compare_feature_sets(df: pd.DataFrame,
                         feature_sets: dict[str, Sequence[str]],
                         model,
                         verbose: bool = True,
                         **kwargs) -> tuple[pd.DataFrame, dict]:
    """
    Run several feature configurations through IDENTICAL folds.

    This is the tool Phase 2 needs: "did stationary features help?" becomes
    a table with mean +/- std rather than one number versus another.

    Returns
    -------
    (comparison_table, {label: WalkForwardResult})
    """
    results: dict[str, WalkForwardResult] = {}
    rows = []

    for label, cols in feature_sets.items():
        cols = [c for c in cols if c in df.columns]
        if not cols:
            if verbose:
                print(f"  skipping '{label}': no columns present")
            continue

        res = walk_forward_evaluate(df, cols, model, label=label, **kwargs)
        results[label] = res

        s = res.summary()
        rows.append({
            "feature_set": label,
            "n_features": len(cols),
            "accuracy": round(s["accuracy_mean"], 4),
            "std": round(s["accuracy_std"], 4),
            "f1": round(s["f1_mean"], 4),
            "roc_auc": round(s["roc_auc_mean"], 4),
            "vs_majority": round(res.edge_over_baselines()["vs_majority"], 4),
            "vs_persistence": round(
                res.edge_over_baselines()["vs_persistence"], 4),
        })

        if verbose:
            print(f"  {res.headline()}")

    table = pd.DataFrame(rows)
    if not table.empty:
        table = table.sort_values("accuracy", ascending=False)
        table = table.set_index("feature_set")

    return table, results


def permutation_test(df: pd.DataFrame,
                     feature_cols: Sequence[str],
                     model,
                     n_permutations: int = 8,
                     target_col: str = "Target",
                     random_state: int = config.RANDOM_STATE,
                     verbose: bool = True,
                     **kwargs) -> dict:
    """
    The single most important sanity check in the project.

    Shuffle the target and re-run the identical evaluation. A correct
    pipeline collapses to chance (~0.50) on shuffled labels. If accuracy
    stays high, the model is exploiting leakage rather than learning, and
    every other number in the report is worthless.

    Returns a dict with the real score, the shuffled distribution, and a
    pass/fail verdict.
    """
    real = walk_forward_evaluate(
        df, feature_cols, model, label="real", **kwargs
    ).per_fold["accuracy"].mean()

    rng_seeds = range(n_permutations)
    shuffled = []
    for seed in rng_seeds:
        d = df.copy()
        rng = np.random.default_rng(random_state + seed)
        d[target_col] = rng.permutation(d[target_col].to_numpy())
        shuffled.append(
            walk_forward_evaluate(
                d, feature_cols, model, label=f"perm{seed}", **kwargs
            ).per_fold["accuracy"].mean()
        )

    shuffled = np.asarray(shuffled)
    gap = real - shuffled.mean()
    # p-value: how often did a shuffled run match the real one?
    p_value = float((shuffled >= real).sum() + 1) / (n_permutations + 1)

    # Two INDEPENDENT questions, which must not be conflated:
    #
    #   (a) Is the pipeline leaking?  -> do shuffled targets still score well?
    #       Shuffled labels carry no information, so anything meaningfully
    #       above chance here means the model is reading the answer somewhere.
    #
    #   (b) Does the model have signal? -> is real meaningfully above shuffled?
    #
    # A model can be leak-free and still have no signal (gap ~ 0). Reporting
    # that as "leakage" sends you hunting a bug that does not exist.
    leak_free = bool(shuffled.mean() < 0.55)
    has_signal = bool(gap > 2 * shuffled.std())

    if not leak_free:
        verdict = "LEAKAGE"
        message = ("LEAKAGE SUSPECTED - shuffled targets still score "
                   f"{shuffled.mean():.4f}, well above chance. The model is "
                   "reading the answer from somewhere. Do not trust any "
                   "other metric until this is found.")
    elif has_signal:
        verdict = "SIGNAL"
        message = ("PASS - shuffled targets collapse to chance "
                   f"({shuffled.mean():.4f}), and the real target scores "
                   f"{gap:+.4f} above that. The edge reflects learnable "
                   "structure, not leakage.")
    else:
        verdict = "NO SIGNAL"
        message = ("NO SIGNAL - the pipeline is clean (shuffled sits at "
                   f"chance, {shuffled.mean():.4f}), but the real target only "
                   f"scores {gap:+.4f} above it. Nothing is being learned. "
                   "This is a modelling problem, NOT a leakage bug.")

    out = {
        "real_accuracy": round(float(real), 4),
        "shuffled_mean": round(float(shuffled.mean()), 4),
        "shuffled_std": round(float(shuffled.std()), 4),
        "shuffled_min": round(float(shuffled.min()), 4),
        "shuffled_max": round(float(shuffled.max()), 4),
        "gap": round(float(gap), 4),
        "p_value": round(p_value, 4),
        "leak_free": leak_free,
        "has_signal": has_signal,
        "verdict": verdict,
    }

    if verbose:
        print("\n  Permutation test")
        print("  " + "-" * 40)
        for k, v in out.items():
            print(f"    {k:16s} : {v}")
        print(f"\n    {message}")

    return out


def baseline_table(df: pd.DataFrame,
                   target_col: str = "Target",
                   date_col: str = "Date",
                   **kwargs) -> pd.DataFrame:
    """
    Baseline accuracies alone, with no model involved.

    Useful as the first thing printed for any new stock: it tells you the
    number a model must exceed before it has demonstrated anything.
    """
    df = df.sort_values(date_col).reset_index(drop=True)
    y = df[target_col].astype(int)
    folds = walk_forward_splits(df[date_col], **kwargs)

    rows = []
    for fold in folds:
        y_tr, y_te = y.iloc[fold.train_idx], y.iloc[fold.test_idx]
        row = {"fold": fold.index}
        for bname, bfunc in BASELINES.items():
            row[bname] = accuracy_score(y_te, bfunc(y_tr, y_te))
        rows.append(row)

    table = pd.DataFrame(rows).set_index("fold")
    table.loc["MEAN"] = table.mean()
    return table.round(4)
