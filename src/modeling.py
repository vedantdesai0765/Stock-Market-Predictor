# src/modeling.py
"""
Model zoo and robustness harness.

Two jobs:

1. A single place where every classifier is defined, so notebooks and the
   dashboard cannot drift apart on hyperparameters.

2. A robustness harness. Phase 3 produced a result that looked good at
   seed 42 (+0.0385) but averaged +0.0189 across seven seeds, and flipped
   negative at one of three valid sampling offsets. A single configuration
   is not evidence. Nothing from Phase 4 onward gets reported without
   seed and offset variation attached.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Sequence

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import (HistGradientBoostingClassifier,
                              RandomForestClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    import config
    from evaluation import walk_forward_evaluate
except ImportError:  # pragma: no cover
    from src import config
    from src.evaluation import walk_forward_evaluate


# ---------------------------------------------------------------------------
# Model definitions
# ---------------------------------------------------------------------------

def make_random_forest(random_state: int = config.RANDOM_STATE,
                       n_estimators: int = 500,
                       min_samples_leaf: int = 20) -> RandomForestClassifier:
    """
    Random Forest. min_samples_leaf is deliberately high: with a few hundred
    rows and 50+ features, unconstrained trees memorise the training window.
    """
    return RandomForestClassifier(
        n_estimators=n_estimators,
        min_samples_leaf=min_samples_leaf,
        max_features="sqrt",
        random_state=random_state,
        n_jobs=-1,
    )


def make_gradient_boosting(random_state: int = config.RANDOM_STATE,
                           max_iter: int = 300,
                           learning_rate: float = 0.05,
                           max_leaf_nodes: int = 15,
                           min_samples_leaf: int = 20,
                           l2_regularization: float = 1.0
                           ) -> HistGradientBoostingClassifier:
    """
    Histogram gradient boosting — scikit-learn's built-in equivalent of
    LightGBM. Chosen over XGBoost/LightGBM deliberately: it needs no extra
    dependency, and the project's requirements.txt is already fragile.

    Regularisation is heavy (shallow trees, low learning rate, L2) because
    the datasets are small and the signal is weak.
    """
    return HistGradientBoostingClassifier(
        max_iter=max_iter,
        learning_rate=learning_rate,
        max_leaf_nodes=max_leaf_nodes,
        min_samples_leaf=min_samples_leaf,
        l2_regularization=l2_regularization,
        early_stopping=False,
        random_state=random_state,
    )


def make_logistic(random_state: int = config.RANDOM_STATE,
                  C: float = 0.1) -> Pipeline:
    """
    Regularised logistic regression, scaled.

    Included as a deliberately simple reference. If a linear model matches
    the forests, the extra complexity is not earning its place — a useful
    thing to be able to state in a report.
    """
    return Pipeline([
        ("scale", StandardScaler()),
        ("clf", LogisticRegression(C=C, max_iter=2000,
                                   random_state=random_state)),
    ])


def make_stratified_dummy(random_state: int = config.RANDOM_STATE
                          ) -> DummyClassifier:
    """A model that learns nothing. Any real model must beat it."""
    return DummyClassifier(strategy="stratified", random_state=random_state)


MODEL_FACTORIES: dict[str, Callable[..., object]] = {
    "random_forest": make_random_forest,
    "gradient_boosting": make_gradient_boosting,
    "logistic": make_logistic,
    "dummy": make_stratified_dummy,
}


def build_model(name: str, random_state: int = config.RANDOM_STATE, **kwargs):
    """Look up and construct a model by name."""
    key = name.strip().lower()
    if key not in MODEL_FACTORIES:
        raise KeyError(
            f"Unknown model '{name}'. Available: {sorted(MODEL_FACTORIES)}")
    return MODEL_FACTORIES[key](random_state=random_state, **kwargs)


def list_models() -> list[str]:
    return sorted(MODEL_FACTORIES)


# ---------------------------------------------------------------------------
# Robustness
# ---------------------------------------------------------------------------

DEFAULT_SEEDS = (0, 1, 7, 42, 99, 123, 2024)


@dataclass
class RobustnessResult:
    """Accuracy of one model across many seeds (and optionally offsets)."""

    model_name: str
    accuracies: list[float]
    edges: list[float]
    seeds: list[int]
    baseline: float
    offsets: list[int] = field(default_factory=list)
    offset_edges: list[float] = field(default_factory=list)

    @property
    def mean_accuracy(self) -> float:
        return float(np.mean(self.accuracies))

    @property
    def seed_std(self) -> float:
        return float(np.std(self.accuracies))

    @property
    def mean_edge(self) -> float:
        return float(np.mean(self.edges))

    @property
    def all_seeds_positive(self) -> bool:
        return bool(all(e > 0 for e in self.edges))

    @property
    def all_offsets_positive(self) -> bool:
        if not self.offset_edges:
            return True
        return bool(all(e > 0 for e in self.offset_edges))

    @property
    def offsets_tested(self) -> bool:
        return len(self.offset_edges) > 0

    @property
    def is_deterministic(self) -> bool:
        """
        Some models (gradient boosting, logistic regression) ignore the seed
        entirely, so seed_std is 0 by construction. That is NOT evidence of
        robustness, and must not be reported as though it were.
        """
        return self.seed_std == 0.0

    def verdict(self) -> str:
        """
        Deliberately strict. An effect that depends on the seed or on an
        arbitrary sampling offset is not a finding.
        """
        if self.mean_edge <= 0:
            return "NO EDGE"
        if not self.all_seeds_positive:
            return "SEED-DEPENDENT"
        if self.offsets_tested and not self.all_offsets_positive:
            return "OFFSET-FRAGILE"
        if not self.is_deterministic and self.mean_edge < self.seed_std:
            return "WITHIN NOISE"
        if not self.offsets_tested:
            return "POSITIVE (offsets untested)"
        return "ROBUST"

    def summary(self) -> dict:
        return {
            "model": self.model_name,
            "mean_accuracy": round(self.mean_accuracy, 4),
            "seed_std": round(self.seed_std, 4),
            "mean_edge": round(self.mean_edge, 4),
            "min_edge": round(float(np.min(self.edges)), 4),
            "max_edge": round(float(np.max(self.edges)), 4),
            "baseline": round(self.baseline, 4),
            "n_seeds": len(self.seeds),
            "deterministic": self.is_deterministic,
            "all_seeds_positive": self.all_seeds_positive,
            "offsets_tested": self.offsets_tested,
            "verdict": self.verdict(),
        }


def evaluate_across_seeds(df: pd.DataFrame,
                          feature_cols: Sequence[str],
                          model_name: str,
                          seeds: Sequence[int] = DEFAULT_SEEDS,
                          verbose: bool = False,
                          **eval_kwargs) -> RobustnessResult:
    """
    Run walk-forward evaluation once per seed.

    Reporting the best seed is a form of overfitting to randomness. This
    returns the distribution so the mean can be reported instead.
    """
    accs, edges, base = [], [], None

    for seed in seeds:
        model = build_model(model_name, random_state=seed)
        res = walk_forward_evaluate(df, feature_cols, model,
                                    label=f"{model_name}_s{seed}",
                                    **eval_kwargs)
        acc = float(res.per_fold["accuracy"].mean())
        base = float(res.baselines.mean().max())
        accs.append(acc)
        edges.append(acc - base)
        if verbose:
            print(f"    seed {seed:5d}: acc={acc:.4f} edge={acc - base:+.4f}")

    return RobustnessResult(
        model_name=model_name,
        accuracies=accs,
        edges=edges,
        seeds=list(seeds),
        baseline=base if base is not None else float("nan"),
    )


def evaluate_across_offsets(base_df: pd.DataFrame,
                            feature_cols: Sequence[str],
                            model_name: str,
                            dataset_builder: Callable[[pd.DataFrame, int],
                                                      pd.DataFrame],
                            n_offsets: int,
                            seed: int = config.RANDOM_STATE,
                            verbose: bool = False,
                            **eval_kwargs) -> tuple[list[int], list[float]]:
    """
    Vary the sampling offset used when building non-overlapping datasets.

    At horizon h there are h equally valid ways to subsample. Which one you
    happen to pick has no financial meaning, so a result that depends on it
    is an artifact. Phase 3's horizon-3 result flipped negative on one of
    three offsets — this function makes that visible by default.
    """
    offsets, edges = [], []

    for off in range(n_offsets):
        ds = dataset_builder(base_df, off)
        if len(ds) < 150:
            continue
        model = build_model(model_name, random_state=seed)
        res = walk_forward_evaluate(ds, feature_cols, model, **eval_kwargs)
        acc = float(res.per_fold["accuracy"].mean())
        base = float(res.baselines.mean().max())
        offsets.append(off)
        edges.append(acc - base)
        if verbose:
            print(f"    offset {off}: n={len(ds)} acc={acc:.4f} "
                  f"edge={acc - base:+.4f}")

    return offsets, edges


def compare_models(df: pd.DataFrame,
                   feature_cols: Sequence[str],
                   model_names: Sequence[str] = ("random_forest",
                                                 "gradient_boosting",
                                                 "logistic",
                                                 "dummy"),
                   seeds: Sequence[int] = DEFAULT_SEEDS,
                   verbose: bool = True,
                   **eval_kwargs) -> tuple[pd.DataFrame, dict]:
    """
    Seed-averaged comparison of several models on identical folds.

    Returns (table, {name: RobustnessResult}).
    """
    results, rows = {}, []

    for name in model_names:
        if verbose:
            print(f"  {name} ...")
        r = evaluate_across_seeds(df, feature_cols, name, seeds=seeds,
                                  verbose=False, **eval_kwargs)
        results[name] = r
        rows.append(r.summary())

    table = pd.DataFrame(rows).set_index("model")
    table = table.sort_values("mean_edge", ascending=False)
    return table, results
