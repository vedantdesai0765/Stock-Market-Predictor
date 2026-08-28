# src/sequence_models.py
"""
Real LSTM and GRU models, trained and saved to disk.

What this replaces
------------------
app.py contained a function called simulate_lstm_gru() which did this:

    pl = yte + rng.normal(0, 0.015, len(yte))
    pg = yte + rng.normal(0, 0.013, len(yte))

It took the ground-truth test values and added Gaussian noise. The
"LSTM vs GRU Predictions" chart and both RMSE cards in the dashboard were
noise drawn around the correct answer. GRU always "won" because 0.013 is
smaller than 0.015. That function is deleted in Phase 4 and replaced by
what is in this module.

Two design decisions worth defending
------------------------------------
1. PyTorch, not TensorFlow. The project's requirements.txt lists torch and
   does not list tensorflow, so notebook 05 (which imports Keras) cannot
   actually run on a clean install. Building on torch fixes that.

2. The models predict RETURNS, not price levels. Notebook 05 regressed
   scaled Close, which makes any sequence model look excellent while it is
   really just learning that tomorrow's price is close to today's. Its
   reported RMSE of 0.02 was largely persistence. Returns are stationary
   and have no such shortcut, so the numbers mean something.

Every result is reported against a persistence baseline (predict that the
next return equals the last one). A sequence model that cannot beat
persistence has not learned anything, however good its RMSE looks.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Sequence

import numpy as np
import pandas as pd

try:
    import config
except ImportError:  # pragma: no cover
    from src import config

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover
    TORCH_AVAILABLE = False


def _require_torch() -> None:
    if not TORCH_AVAILABLE:
        raise ImportError(
            "PyTorch is not installed. Run:  %pip install torch\n"
            "(torch is listed in requirements.txt; tensorflow is not.)"
        )


# ---------------------------------------------------------------------------
# Sequence construction
# ---------------------------------------------------------------------------

def make_sequences(values: np.ndarray,
                   seq_len: int = 30) -> tuple[np.ndarray, np.ndarray]:
    """
    Turn a 1-D series into (X, y) supervised pairs.

        X[i] = values[i : i + seq_len]
        y[i] = values[i + seq_len]

    No shuffling: order is preserved so a chronological split stays valid.
    """
    if len(values) <= seq_len:
        raise ValueError(
            f"Need more than {seq_len} observations, got {len(values)}")

    X = np.lib.stride_tricks.sliding_window_view(
        values[:-1], seq_len)[: len(values) - seq_len]
    y = values[seq_len:]
    return X.astype(np.float32), y.astype(np.float32)


# ---------------------------------------------------------------------------
# Architectures
# ---------------------------------------------------------------------------

if TORCH_AVAILABLE:

    class SequenceRegressor(nn.Module):
        """
        Two stacked recurrent layers with dropout, then a linear head.

        Mirrors notebook 05's architecture so the comparison is fair, but
        the cell type is switchable and the target is returns.
        """

        def __init__(self, cell: Literal["lstm", "gru"] = "lstm",
                     hidden_size: int = 32, num_layers: int = 2,
                     dropout: float = 0.2, input_size: int = 1):
            super().__init__()
            rnn_cls = nn.LSTM if cell == "lstm" else nn.GRU
            self.rnn = rnn_cls(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                dropout=dropout if num_layers > 1 else 0.0,
                batch_first=True,
            )
            self.head = nn.Linear(hidden_size, 1)

        def forward(self, x):
            out, _ = self.rnn(x)
            return self.head(out[:, -1, :]).squeeze(-1)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

@dataclass
class SequenceResult:
    """Everything one trained sequence model produced."""

    cell: str
    rmse: float
    mae: float
    directional_accuracy: float
    persistence_rmse: float
    persistence_directional: float
    zero_rmse: float
    prediction_shrinkage: float
    y_true: np.ndarray
    y_pred: np.ndarray
    train_losses: list[float]
    n_train: int
    n_test: int

    @property
    def beats_persistence(self) -> bool:
        return self.rmse < self.persistence_rmse

    @property
    def beats_zero(self) -> bool:
        """
        The baseline that actually matters for return regression.

        Daily returns have mean near zero, so a model that outputs zero for
        everything scores RMSE equal to the standard deviation of returns.
        Beating PERSISTENCE is easy — persistence chases noise and roughly
        doubles the error. Beating ZERO requires genuine predictive content.
        A model can look impressive against persistence while being strictly
        worse than a constant.
        """
        return self.rmse < self.zero_rmse

    def summary(self) -> dict:
        return {
            "model": self.cell.upper(),
            "rmse": round(self.rmse, 6),
            "mae": round(self.mae, 6),
            "directional_accuracy": round(self.directional_accuracy, 4),
            "persistence_rmse": round(self.persistence_rmse, 6),
            "zero_rmse": round(self.zero_rmse, 6),
            "beats_persistence": self.beats_persistence,
            "beats_zero": self.beats_zero,
            "prediction_shrinkage": round(self.prediction_shrinkage, 4),
            "n_train": self.n_train,
            "n_test": self.n_test,
        }

    def verdict(self) -> str:
        if not self.beats_zero:
            return ("NO SIGNAL - worse than predicting zero; the model has "
                    "collapsed toward a constant.")
        if self.prediction_shrinkage < 0.25:
            return ("NEAR-CONSTANT - beats zero, but predictions carry only "
                    f"{self.prediction_shrinkage:.0%} of real volatility.")
        return "Predictions carry meaningful variation."


def train_sequence_model(returns: pd.Series | np.ndarray,
                         cell: Literal["lstm", "gru"] = "lstm",
                         seq_len: int = 30,
                         hidden_size: int = 32,
                         num_layers: int = 2,
                         dropout: float = 0.2,
                         epochs: int = 40,
                         batch_size: int = 32,
                         lr: float = 1e-3,
                         test_fraction: float = 0.2,
                         random_state: int = config.RANDOM_STATE,
                         verbose: bool = False) -> SequenceResult:
    """
    Train one recurrent model to predict the next daily return.

    Scaling note
    ------------
    Returns are standardised using the mean and std of the TRAINING slice
    only. Notebook 05 and app.py both called scaler.fit_transform() on the
    whole series before splitting, which leaks the test period's scale into
    training. That is fixed here.
    """
    _require_torch()

    torch.manual_seed(random_state)
    np.random.seed(random_state)

    r = pd.Series(returns).dropna().to_numpy(dtype=np.float64)
    if len(r) < seq_len + 60:
        raise ValueError(
            f"Not enough observations: {len(r)} (need at least {seq_len + 60})")

    split = int(len(r) * (1 - test_fraction))

    # Fit the scaler on training data ONLY
    mu, sigma = r[:split].mean(), r[:split].std()
    sigma = sigma if sigma > 0 else 1.0
    scaled = (r - mu) / sigma

    X, y = make_sequences(scaled, seq_len=seq_len)
    seq_split = split - seq_len
    if seq_split < 30:
        raise ValueError("Training split too small after sequencing.")

    X_tr = torch.from_numpy(X[:seq_split]).unsqueeze(-1)
    y_tr = torch.from_numpy(y[:seq_split])
    X_te = torch.from_numpy(X[seq_split:]).unsqueeze(-1)
    y_te = y[seq_split:]

    model = SequenceRegressor(cell=cell, hidden_size=hidden_size,
                              num_layers=num_layers, dropout=dropout)
    optimiser = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    n = len(X_tr)
    losses = []
    model.train()
    for epoch in range(epochs):
        perm = torch.randperm(n)
        epoch_loss = 0.0
        for start in range(0, n, batch_size):
            idx = perm[start:start + batch_size]
            optimiser.zero_grad()
            pred = model(X_tr[idx])
            loss = loss_fn(pred, y_tr[idx])
            loss.backward()
            optimiser.step()
            epoch_loss += float(loss) * len(idx)
        losses.append(epoch_loss / n)
        if verbose and (epoch + 1) % 10 == 0:
            print(f"    epoch {epoch + 1:3d}  loss={losses[-1]:.5f}")

    model.eval()
    with torch.no_grad():
        pred_scaled = model(X_te).numpy()

    # Back to return units
    y_true = y_te * sigma + mu
    y_pred = pred_scaled * sigma + mu

    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    mae = float(np.mean(np.abs(y_true - y_pred)))
    directional = float(np.mean(np.sign(y_pred) == np.sign(y_true)))

    # Persistence: next return equals the last observed return
    last_obs = X[seq_split:, -1] * sigma + mu
    persistence_rmse = float(np.sqrt(np.mean((y_true - last_obs) ** 2)))
    persistence_dir = float(np.mean(np.sign(last_obs) == np.sign(y_true)))

    # The baseline that matters: predict zero for every day
    zero_rmse = float(np.sqrt(np.mean(y_true ** 2)))

    # How much of the real variation do the predictions actually reproduce?
    true_std = float(np.std(y_true))
    shrinkage = float(np.std(y_pred) / true_std) if true_std > 0 else 0.0

    return SequenceResult(
        cell=cell,
        rmse=rmse,
        mae=mae,
        directional_accuracy=directional,
        persistence_rmse=persistence_rmse,
        persistence_directional=persistence_dir,
        zero_rmse=zero_rmse,
        prediction_shrinkage=shrinkage,
        y_true=y_true,
        y_pred=y_pred,
        train_losses=losses,
        n_train=len(X_tr),
        n_test=len(X_te),
    )


def compare_sequence_models(returns: pd.Series | np.ndarray,
                            seeds: Sequence[int] = (0, 42, 123),
                            verbose: bool = True,
                            **kwargs) -> tuple[pd.DataFrame, dict]:
    """
    Train LSTM and GRU across several seeds.

    Neural network training is stochastic, so a single run tells you very
    little. Reporting "GRU beat LSTM" from one seed is exactly the mistake
    the simulated version made by construction.
    """
    rows, store = [], {}

    for cell in ("lstm", "gru"):
        per_seed = []
        for seed in seeds:
            res = train_sequence_model(returns, cell=cell,
                                       random_state=seed, **kwargs)
            per_seed.append(res)
            if verbose:
                print(f"  {cell.upper()} seed {seed}: "
                      f"rmse={res.rmse:.6f} dir={res.directional_accuracy:.4f}")
        store[cell] = per_seed

        rmses = [r.rmse for r in per_seed]
        dirs = [r.directional_accuracy for r in per_seed]
        shrink = [r.prediction_shrinkage for r in per_seed]
        rows.append({
            "model": cell.upper(),
            "rmse_mean": round(float(np.mean(rmses)), 6),
            "rmse_std": round(float(np.std(rmses)), 6),
            "directional_mean": round(float(np.mean(dirs)), 4),
            "persistence_rmse": round(per_seed[0].persistence_rmse, 6),
            "zero_rmse": round(per_seed[0].zero_rmse, 6),
            "beats_persistence": bool(
                np.mean(rmses) < per_seed[0].persistence_rmse),
            "beats_zero": bool(np.mean(rmses) < per_seed[0].zero_rmse),
            "shrinkage": round(float(np.mean(shrink)), 4),
        })

    return pd.DataFrame(rows).set_index("model"), store


# ---------------------------------------------------------------------------
# Persistence to disk
# ---------------------------------------------------------------------------

def save_sequence_model(model, path: Path | str) -> Path:
    """Save weights so the dashboard can load them instead of retraining."""
    _require_torch()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)
    return path


def load_sequence_model(path: Path | str,
                        cell: Literal["lstm", "gru"] = "lstm",
                        hidden_size: int = 32,
                        num_layers: int = 2):
    """Reconstruct a saved model."""
    _require_torch()
    model = SequenceRegressor(cell=cell, hidden_size=hidden_size,
                              num_layers=num_layers)
    model.load_state_dict(torch.load(Path(path), map_location="cpu"))
    model.eval()
    return model
