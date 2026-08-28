# src/pipeline.py
"""
Training pipeline: build, fit, and save every artifact one stock needs.

    from pipeline import train_stock
    train_stock("TCS")

This writes models/TCS/ containing the fitted classifier, the feature list,
the confidence operating point, and a metrics file. The dashboard then
LOADS those artifacts instead of retraining on every button press, which is
what app.py did before Phase 4.

Adding a new stock stays a one-line change in config.STOCKS: this module
reads the registry and never hardcodes a ticker.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd

try:
    import config
    from abstention import (calibration_table, coverage_accuracy_curve,
                            evaluate_with_abstention)
    from dataset import build_dataset
    from evaluation import permutation_test, walk_forward_evaluate
    from modeling import build_model, evaluate_across_seeds
    from stationary import add_stationary_features, get_stationary_features
    from targets import build_directional_dataset
except ImportError:  # pragma: no cover
    from src import config
    from src.abstention import (calibration_table, coverage_accuracy_curve,
                                evaluate_with_abstention)
    from src.dataset import build_dataset
    from src.evaluation import permutation_test, walk_forward_evaluate
    from src.modeling import build_model, evaluate_across_seeds
    from src.stationary import (add_stationary_features,
                                get_stationary_features)
    from src.targets import build_directional_dataset


ARTIFACT_VERSION = "phase4.1"


# ---------------------------------------------------------------------------
# Training configuration
# ---------------------------------------------------------------------------

@dataclass
class TrainConfig:
    """
    Every knob in one place, saved alongside the model.

    Defaults come from the Phase 3 sweep and the Phase 4 offset test.

    horizon=3 with non-overlapping labels was the only Phase 3 configuration
    whose edge exceeded its own fold standard deviation.

    model_name defaults to LOGISTIC REGRESSION, not a tree ensemble. On the
    offset test the trees flip negative on one of the three equally valid
    sampling offsets (gradient boosting -0.0385, random forest -0.0128)
    while regularised logistic stays positive on all three (+0.0154,
    +0.0333, +0.0103). Gradient boosting has the higher single-offset mean,
    but an edge that depends on an arbitrary sampling choice is not an edge.
    With a few hundred rows and a weak signal, the linear model is the
    honest default; the ensembles remain available via model_name.
    """

    model_name: str = "logistic"
    horizon: int = 3
    k: float = 0.0
    non_overlapping: bool = True
    n_splits: int = 5
    embargo: int = 17
    confidence_threshold: float = 0.05
    min_rows: int = 200


# ---------------------------------------------------------------------------
# Dataset assembly
# ---------------------------------------------------------------------------

def prepare_modelling_frame(stock_key: str,
                            cfg: TrainConfig,
                            verbose: bool = False
                            ) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """
    Build the full feature frame and the label-filtered modelling frame.

    Returns (full_frame, modelling_frame, feature_columns). The full frame
    is kept because the dashboard needs prices and indicators for charts,
    not just the rows that survived label filtering.
    """
    full = build_dataset(stock_key, with_sentiment=False,
                         save=False, verbose=False)
    full = add_stationary_features(full).dropna().reset_index(drop=True)

    features = get_stationary_features(full)

    modelling = build_directional_dataset(
        full,
        horizon=cfg.horizon,
        k=cfg.k,
        non_overlapping=cfg.non_overlapping,
    )

    if verbose:
        print(f"  full frame     : {len(full)} rows, {len(features)} features")
        print(f"  modelling frame: {len(modelling)} rows "
              f"(horizon={cfg.horizon}, k={cfg.k})")

    return full, modelling, features


# ---------------------------------------------------------------------------
# Operating point selection
# ---------------------------------------------------------------------------

def select_operating_point(curve: pd.DataFrame,
                           min_coverage: float = 0.20,
                           min_gain: float = 0.01) -> tuple[float, bool, str]:
    """
    Choose the confidence threshold to ship.

    Returns (threshold, abstention_useful, reason).

    Rules, in order:

      1. Only consider points covering at least `min_coverage` of days.
         Without this floor the search always drifts to the most selective
         point, where the sample is tiny and the "edge" is mostly luck —
         the trap Phase 3 flagged when the best TCS edge sat on 66 rows.

      2. Abstaining must EARN its place. It is only adopted if it improves
         the edge by at least `min_gain` over predicting on every day.
         Otherwise the model predicts always, and metadata records that
         abstention did not help.

    Point 2 matters because a threshold can look attractive purely by
    shrinking the sample. If full coverage is already the best operating
    point, that is a legitimate finding and should be reported as one, not
    dressed up with a token confidence gate.
    """
    if curve is None or curve.empty:
        return 0.0, False, "no curve available"

    full = curve.iloc[0]
    full_edge = float(full["edge"])

    eligible = curve[curve["coverage"] >= min_coverage]
    if eligible.empty:
        return 0.0, False, "no point met the coverage floor"

    best = eligible.loc[eligible["edge"].idxmax()]
    best_edge = float(best["edge"])

    if best_edge <= 0:
        return 0.0, False, f"no positive edge at any coverage >= {min_coverage:.0%}"

    if best_edge - full_edge < min_gain:
        return 0.0, False, (f"abstention gained only "
                            f"{best_edge - full_edge:+.4f}, below {min_gain}")

    return float(best["min_confidence"]), True, (
        f"edge improves {full_edge:+.4f} -> {best_edge:+.4f} "
        f"at {best['coverage']:.0%} coverage")


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_stock(stock_key: str,
                cfg: Optional[TrainConfig] = None,
                run_permutation: bool = True,
                run_seeds: bool = True,
                save: bool = True,
                verbose: bool = True) -> dict:
    """
    Train, validate and save one stock's classifier.

    Artifacts written to models/<KEY>/:
        model.joblib      - classifier fitted on ALL modelling rows
        metadata.json     - config, feature list, metrics, provenance
        oof_predictions.csv - out-of-fold predictions for the coverage curve
        coverage_curve.csv  - accuracy vs coverage

    The saved model is fitted on the full modelling frame, while every
    reported metric comes from walk-forward folds. Those are different
    objects on purpose: you deploy the model that has seen the most data,
    but you never quote its in-sample score.
    """
    cfg = cfg or TrainConfig()
    stock = config.get_stock(stock_key)

    if verbose:
        print(f"\n=== Training {stock.key} ({stock.display_name}) ===")

    full, modelling, features = prepare_modelling_frame(
        stock_key, cfg, verbose=verbose)

    # Non-overlapping sampling divides row count by the horizon, so a stock
    # with little history can fall below the usable minimum. Rather than
    # failing outright, step the horizon down until it fits. A shorter
    # horizon is a weaker question, but a trained model with an honest
    # label beats no model at all — and this keeps "just add a stock"
    # working for tickers with short histories.
    tried = [cfg.horizon]
    while len(modelling) < cfg.min_rows and cfg.horizon > 1:
        new_h = cfg.horizon - 1 if cfg.horizon <= 3 else cfg.horizon // 2
        cfg = TrainConfig(**{**asdict(cfg), "horizon": new_h,
                             "embargo": max(cfg.embargo // 2, 5)})
        tried.append(new_h)
        full, modelling, features = prepare_modelling_frame(
            stock_key, cfg, verbose=False)
        if verbose:
            print(f"  horizon {tried[-2]} gave too few rows; "
                  f"retrying at horizon {new_h} -> {len(modelling)} rows")

    if len(modelling) < cfg.min_rows:
        raise ValueError(
            f"{stock.key}: only {len(modelling)} modelling rows after trying "
            f"horizons {tried}, need {cfg.min_rows}. This stock has too "
            f"little history; download a longer window in notebook 01."
        )

    # --- walk-forward metrics -------------------------------------------
    model = build_model(cfg.model_name)
    result = walk_forward_evaluate(
        modelling, features, model,
        label=f"{stock.key}:{cfg.model_name}",
        n_splits=cfg.n_splits, embargo=cfg.embargo,
    )
    walk_acc = float(result.per_fold["accuracy"].mean())
    walk_std = float(result.per_fold["accuracy"].std())
    baseline = float(result.baselines.mean().max())

    if verbose:
        print(f"  walk-forward   : {walk_acc:.4f} +/- {walk_std:.4f} "
              f"(baseline {baseline:.4f}, edge {walk_acc - baseline:+.4f})")

    # --- seed robustness -------------------------------------------------
    seed_summary = None
    if run_seeds:
        rob = evaluate_across_seeds(
            modelling, features, cfg.model_name,
            n_splits=cfg.n_splits, embargo=cfg.embargo,
        )
        seed_summary = rob.summary()
        if verbose:
            print(f"  seed-averaged  : {rob.mean_accuracy:.4f} "
                  f"(edge {rob.mean_edge:+.4f}) -> {rob.verdict()}")

    # --- leakage check ---------------------------------------------------
    perm_summary = None
    if run_permutation:
        perm_summary = permutation_test(
            modelling, features, build_model(cfg.model_name),
            n_permutations=6, n_splits=cfg.n_splits,
            embargo=cfg.embargo, verbose=False,
        )
        if verbose:
            print(f"  permutation    : {perm_summary['verdict']} "
                  f"(shuffled {perm_summary['shuffled_mean']:.4f})")

    # --- confidence operating point --------------------------------------
    curve = coverage_accuracy_curve(result.oof_predictions, min_samples=15)
    chosen_threshold, abstention_useful, reason = select_operating_point(curve)
    cfg = TrainConfig(**{**asdict(cfg),
                         "confidence_threshold": chosen_threshold})
    operating = evaluate_with_abstention(
        result.oof_predictions,
        confidence_threshold=chosen_threshold,
    )
    operating["abstention_useful"] = abstention_useful
    operating["selection_reason"] = reason
    if verbose:
        print(f"  operating point: coverage {operating['coverage']:.0%}, "
              f"accuracy {operating['accuracy']:.4f}, "
              f"edge {operating['edge']:+.4f}")
        print(f"  abstention     : "
              f"{'ADOPTED' if abstention_useful else 'not used'} ({reason})")

    # --- final fit on all modelling rows ---------------------------------
    final_model = build_model(cfg.model_name)
    final_model.fit(modelling[features], modelling["Target"])

    metadata = {
        "artifact_version": ARTIFACT_VERSION,
        "stock_key": stock.key,
        "ticker": stock.ticker,
        "display_name": stock.display_name,
        "trained_at": datetime.now().isoformat(timespec="seconds"),
        "config": asdict(cfg),
        "n_full_rows": len(full),
        "n_modelling_rows": len(modelling),
        "n_features": len(features),
        "features": features,
        "date_range": [str(full["Date"].min().date()),
                       str(full["Date"].max().date())],
        "walk_forward_accuracy": round(walk_acc, 4),
        "walk_forward_std": round(walk_std, 4),
        "baseline": round(baseline, 4),
        "edge": round(walk_acc - baseline, 4),
        "seed_robustness": seed_summary,
        "permutation": perm_summary,
        "operating_point": operating,
    }

    if save:
        out_dir = stock.model_dir
        out_dir.mkdir(parents=True, exist_ok=True)

        joblib.dump(final_model, out_dir / "model.joblib")
        with open(out_dir / "metadata.json", "w") as fh:
            json.dump(metadata, fh, indent=2, default=str)
        result.oof_predictions.to_csv(
            out_dir / "oof_predictions.csv", index=False)
        curve.to_csv(out_dir / "coverage_curve.csv", index=False)

        if verbose:
            print(f"  saved -> models/{stock.key}/")

    return {
        "metadata": metadata,
        "model": final_model,
        "result": result,
        "curve": curve,
        "full": full,
        "modelling": modelling,
        "features": features,
    }


def train_all(cfg: Optional[TrainConfig] = None,
              verbose: bool = True,
              **kwargs) -> dict:
    """Train every stock in the registry, skipping any that fail."""
    out = {}
    for key in config.list_stocks():
        try:
            out[key] = train_stock(key, cfg=cfg, verbose=verbose, **kwargs)
        except Exception as exc:
            print(f"\n  !! {key} skipped: {type(exc).__name__}: {exc}")
    return out


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def artifacts_exist(stock_key: str) -> bool:
    stock = config.get_stock(stock_key)
    return ((stock.model_dir / "model.joblib").exists()
            and (stock.model_dir / "metadata.json").exists())


def load_artifacts(stock_key: str) -> dict:
    """
    Load a saved model and its metadata.

    Raises a clear, actionable error if training has not been run, rather
    than silently retraining — which is what hid the fabricated LSTM
    numbers in the old dashboard.
    """
    stock = config.get_stock(stock_key)
    model_path = stock.model_dir / "model.joblib"
    meta_path = stock.model_dir / "metadata.json"

    if not model_path.exists() or not meta_path.exists():
        raise FileNotFoundError(
            f"No trained artifacts for {stock.key}.\n"
            f"Expected: {model_path}\n"
            f"Run notebooks/04_model_training_and_artifacts.ipynb, or:\n"
            f"    from pipeline import train_stock; train_stock('{stock.key}')"
        )

    model = joblib.load(model_path)
    with open(meta_path) as fh:
        metadata = json.load(fh)

    curve_path = stock.model_dir / "coverage_curve.csv"
    curve = pd.read_csv(curve_path) if curve_path.exists() else None

    return {"model": model, "metadata": metadata, "curve": curve}


def predict_latest(stock_key: str,
                   full_frame: Optional[pd.DataFrame] = None) -> dict:
    """
    Predict the most recent row, with a confidence gate.

    Returns a dict including `signal`, which is "UP", "DOWN" or "NO SIGNAL".
    The abstention case is a real output, not an error: on most days the
    model has nothing worth saying, and the dashboard should show that
    rather than inventing a direction.
    """
    art = load_artifacts(stock_key)
    model, meta = art["model"], art["metadata"]
    cfg = meta["config"]

    if full_frame is None:
        full_frame = build_dataset(stock_key, with_sentiment=False,
                                   save=False, verbose=False)
        full_frame = (add_stationary_features(full_frame)
                      .dropna().reset_index(drop=True))

    features = meta["features"]
    missing = [c for c in features if c not in full_frame.columns]
    if missing:
        raise ValueError(f"Frame is missing trained features: {missing[:5]}")

    latest = full_frame[features].iloc[[-1]]
    proba = float(model.predict_proba(latest)[0, 1])
    confidence = abs(proba - 0.5)
    threshold = cfg["confidence_threshold"]

    if confidence < threshold:
        signal = "NO SIGNAL"
    else:
        signal = "UP" if proba > 0.5 else "DOWN"

    return {
        "stock": stock_key,
        "date": str(full_frame["Date"].iloc[-1].date()),
        "close": float(full_frame["Close"].iloc[-1]),
        "probability_up": round(proba, 4),
        "confidence": round(confidence, 4),
        "confidence_threshold": threshold,
        "signal": signal,
        "horizon_days": cfg["horizon"],
        "expected_accuracy": meta["operating_point"]["accuracy"],
        "expected_coverage": meta["operating_point"]["coverage"],
    }
