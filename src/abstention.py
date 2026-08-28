# src/abstention.py
"""
Confidence-gated prediction: letting the model decline to answer.

Why this exists
---------------
Accuracy over all days is the wrong headline for a near-50/50 task. A model
forced to commit on every single day will spend most of its predictions on
days that are unpredictable, and its accuracy will sit near the base rate no
matter how good it is on the days that matter.

The alternative is to report two numbers together:

    coverage - the fraction of days the model chose to act on
    accuracy - how often it was right, on those days only

"Right 57% of the time on the 25% of days it committed" is both more useful
and more honest than "52% overall". It is also directly actionable: the
dashboard shows a signal when there is one, and stays quiet otherwise.

This module is model-agnostic. It consumes the out-of-fold predictions that
evaluation.walk_forward_evaluate() already produces, so nothing needs to be
refitted.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import accuracy_score, brier_score_loss


# ---------------------------------------------------------------------------
# Confidence
# ---------------------------------------------------------------------------

def confidence_from_proba(proba: pd.Series | np.ndarray) -> np.ndarray:
    """
    Distance from the decision boundary, in [0, 0.5].

    A probability of 0.50 means no view at all (confidence 0.0); 0.90 or
    0.10 means a strong view either way (confidence 0.40).
    """
    return np.abs(np.asarray(proba, dtype=float) - 0.5)


# ---------------------------------------------------------------------------
# Coverage / accuracy
# ---------------------------------------------------------------------------

def coverage_accuracy_curve(oof: pd.DataFrame,
                            quantiles: tuple[float, ...] = (
                                0.0, 0.1, 0.2, 0.3, 0.4,
                                0.5, 0.6, 0.7, 0.8, 0.9),
                            min_samples: int = 20,
                            proba_col: str = "y_proba",
                            pred_col: str = "y_pred",
                            true_col: str = "y_true") -> pd.DataFrame:
    """
    Accuracy as a function of how selective the model is allowed to be.

    For each confidence quantile q, keep only predictions in the top
    (1 - q) fraction by confidence and measure accuracy there. The majority
    baseline is recomputed ON THE SAME SUBSET, because a confident subset
    may also be a class-imbalanced one — comparing against the full-sample
    baseline would flatter the model.

    A healthy result is monotonic: accuracy and edge both rise as coverage
    falls. A flat or falling curve means the model's probabilities carry no
    information about its own reliability.
    """
    df = oof.dropna(subset=[proba_col]).copy()
    if df.empty:
        raise ValueError("No rows with probabilities; does the model expose "
                         "predict_proba?")

    df["confidence"] = confidence_from_proba(df[proba_col])
    n_total = len(df)

    rows = []
    for q in quantiles:
        cut = df["confidence"].quantile(q)
        sel = df[df["confidence"] >= cut]
        if len(sel) < min_samples:
            continue

        acc = accuracy_score(sel[true_col], sel[pred_col])
        maj = max(sel[true_col].mean(), 1 - sel[true_col].mean())

        rows.append({
            "confidence_quantile": q,
            "coverage": round(len(sel) / n_total, 4),
            "n_predictions": len(sel),
            "accuracy": round(float(acc), 4),
            "subset_majority": round(float(maj), 4),
            "edge": round(float(acc - maj), 4),
            "min_confidence": round(float(cut), 4),
        })

    return pd.DataFrame(rows)


def evaluate_with_abstention(oof: pd.DataFrame,
                             confidence_threshold: float = 0.05,
                             proba_col: str = "y_proba",
                             pred_col: str = "y_pred",
                             true_col: str = "y_true") -> dict:
    """
    Apply a single confidence gate and report the resulting trade-off.

    Returns coverage, accuracy on committed predictions, the majority
    baseline on that same subset, and the edge between them.
    """
    df = oof.dropna(subset=[proba_col]).copy()
    df["confidence"] = confidence_from_proba(df[proba_col])

    committed = df[df["confidence"] >= confidence_threshold]
    if committed.empty:
        return {
            "confidence_threshold": confidence_threshold,
            "coverage": 0.0,
            "n_committed": 0,
            "accuracy": np.nan,
            "subset_majority": np.nan,
            "edge": np.nan,
        }

    acc = accuracy_score(committed[true_col], committed[pred_col])
    maj = max(committed[true_col].mean(), 1 - committed[true_col].mean())

    return {
        "confidence_threshold": confidence_threshold,
        "coverage": round(len(committed) / len(df), 4),
        "n_committed": len(committed),
        "accuracy": round(float(acc), 4),
        "subset_majority": round(float(maj), 4),
        "edge": round(float(acc - maj), 4),
    }


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------

def calibration_table(oof: pd.DataFrame,
                      n_bins: int = 5,
                      proba_col: str = "y_proba",
                      true_col: str = "y_true") -> pd.DataFrame:
    """
    Do the predicted probabilities mean what they say?

    Bin predictions by probability and compare the mean predicted value in
    each bin against the observed frequency. A well-calibrated model has
    these roughly equal: among days it called 60% likely to rise, about 60%
    should have risen.

    This matters for the dashboard. If a confidence figure is displayed to a
    user, it should be truthful.
    """
    df = oof.dropna(subset=[proba_col]).copy()
    df["bin"] = pd.qcut(df[proba_col], q=n_bins, duplicates="drop")

    grouped = df.groupby("bin", observed=True).agg(
        n=(true_col, "size"),
        mean_predicted=(proba_col, "mean"),
        observed_frequency=(true_col, "mean"),
    ).reset_index()

    grouped["gap"] = grouped["mean_predicted"] - grouped["observed_frequency"]
    grouped["bin"] = grouped["bin"].astype(str)
    return grouped.round(4)


def brier_score(oof: pd.DataFrame,
                proba_col: str = "y_proba",
                true_col: str = "y_true") -> float:
    """
    Mean squared error of the probabilities. Lower is better; 0.25 is what
    you get by predicting 0.5 for everything.
    """
    df = oof.dropna(subset=[proba_col])
    return round(float(brier_score_loss(df[true_col], df[proba_col])), 4)


def calibrate_probabilities(oof: pd.DataFrame,
                            proba_col: str = "y_proba",
                            true_col: str = "y_true",
                            fold_col: str = "fold") -> pd.DataFrame:
    """
    Isotonic recalibration, fitted fold-by-fold.

    Fold f's calibrator is fitted on folds < f only, so no future outcome
    ever informs a calibrated probability. Fold 0 is left uncalibrated
    because it has no history to learn from.
    """
    df = oof.dropna(subset=[proba_col]).copy().sort_values(fold_col)
    df["y_proba_calibrated"] = df[proba_col]

    folds = sorted(df[fold_col].unique())
    for f in folds[1:]:
        past = df[df[fold_col] < f]
        current = df[df[fold_col] == f]
        if past[true_col].nunique() < 2 or len(past) < 30:
            continue

        iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        iso.fit(past[proba_col], past[true_col])
        df.loc[current.index, "y_proba_calibrated"] = iso.predict(
            current[proba_col])

    return df


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def abstention_report(oof: pd.DataFrame, label: str = "model") -> None:
    """Print the coverage/accuracy curve and calibration summary."""
    print(f"\n{'=' * 66}")
    print(f"  Confidence-gated evaluation: {label}")
    print(f"{'=' * 66}")

    curve = coverage_accuracy_curve(oof)
    print("\n  Coverage / accuracy trade-off")
    print(curve.to_string(index=False))

    if len(curve) >= 2:
        first, last = curve.iloc[0], curve.iloc[-1]
        direction = ("rises" if last["accuracy"] > first["accuracy"]
                     else "does not rise")
        print(f"\n  Accuracy {direction} as coverage falls "
              f"({first['coverage']:.0%} -> {last['coverage']:.0%}: "
              f"{first['accuracy']:.4f} -> {last['accuracy']:.4f})")

    print(f"\n  Brier score: {brier_score(oof)}  (0.25 = uninformative)")
    print("\n  Calibration")
    print(calibration_table(oof).to_string(index=False))
