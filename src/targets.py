# src/targets.py
"""
Target (label) engineering.

The problem this module solves
------------------------------
Every phase up to now asked the same question: "will tomorrow's close be
higher than today's?" That question is close to unanswerable, and the
reason is in the labels rather than the model.

If a stock closes at 1000.00 and then 1000.20, the label is UP. If it then
closes at 999.80, the label is DOWN. Both are 0.02% moves — indistinguishable
from the bid-ask spread or a single large order. Roughly two thirds of a
daily-direction dataset consists of labels like these: pure noise, dressed
up as signal. They cannot be learned, and their presence drowns out the
minority of days that carry real information.

This module provides targets that ask better questions:

  1. threshold_target        - ignore moves too small to matter
  2. volatility_scaled_target- define "too small" per stock, per period
  3. non_overlapping_subset  - stop consecutive labels from sharing data

Leakage policy
--------------
Forward returns are, by construction, future information — that is what a
label IS. The rules are:
  - Labels are never used as features.
  - Rows whose label cannot be computed (the final `horizon` rows) are NaN,
    never silently filled.
  - The volatility used to SCALE a threshold is trailing-only, so the
    decision of "what counts as a big move" uses no future data.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Class encoding used throughout
DOWN, FLAT, UP = -1, 0, 1
CLASS_NAMES = {DOWN: "DOWN", FLAT: "FLAT", UP: "UP"}


# ---------------------------------------------------------------------------
# Forward returns
# ---------------------------------------------------------------------------

def forward_return(df: pd.DataFrame,
                   horizon: int = 1,
                   price_col: str = "Close") -> pd.Series:
    """
    Return realised over the next `horizon` bars.

        fwd[t] = price[t + horizon] / price[t] - 1

    The final `horizon` rows are NaN because their outcome is unknown.
    """
    if horizon < 1:
        raise ValueError(f"horizon must be >= 1, got {horizon}")
    price = df[price_col]
    return price.shift(-horizon) / price - 1.0


def forward_return_open_to_close(df: pd.DataFrame,
                                 horizon: int = 1) -> pd.Series:
    """
    A tradeable variant: buy at tomorrow's OPEN, sell at the close `horizon`
    bars later.

    Close-to-close returns are not executable — you cannot trade at a close
    you have already observed. This version can actually be acted on.
    """
    if horizon < 1:
        raise ValueError(f"horizon must be >= 1, got {horizon}")
    entry = df["Open"].shift(-1)
    exit_ = df["Close"].shift(-horizon)
    return exit_ / entry - 1.0


# ---------------------------------------------------------------------------
# Idea 1 — fixed threshold
# ---------------------------------------------------------------------------

def threshold_target(df: pd.DataFrame,
                     horizon: int = 1,
                     threshold: float = 0.005,
                     price_col: str = "Close") -> pd.Series:
    """
    Three-class label with a dead zone around zero.

        return >  +threshold  ->  UP   (+1)
        return <  -threshold  ->  DOWN (-1)
        otherwise             ->  FLAT ( 0)

    `threshold` is a fraction: 0.005 = 0.5%.

    The FLAT class is not a failure mode — it is the model being allowed to
    say "nothing worth acting on here".
    """
    if threshold < 0:
        raise ValueError(f"threshold must be >= 0, got {threshold}")

    fwd = forward_return(df, horizon=horizon, price_col=price_col)

    target = pd.Series(FLAT, index=df.index, dtype="float64")
    target[fwd > threshold] = UP
    target[fwd < -threshold] = DOWN
    target[fwd.isna()] = np.nan
    return target


# ---------------------------------------------------------------------------
# Idea 2 — volatility-scaled threshold
# ---------------------------------------------------------------------------

def trailing_volatility(df: pd.DataFrame,
                        window: int = 20,
                        price_col: str = "Close") -> pd.Series:
    """
    Rolling standard deviation of daily returns, trailing only.

    Used to decide what counts as a "big" move. Because it is trailing, the
    threshold at row t is knowable at row t.
    """
    ret = df[price_col].pct_change()
    return ret.rolling(window, min_periods=max(window // 2, 5)).std()


def volatility_scaled_target(df: pd.DataFrame,
                             horizon: int = 1,
                             k: float = 0.5,
                             vol_window: int = 20,
                             price_col: str = "Close") -> pd.Series:
    """
    Three-class label whose dead zone adapts to each stock's own volatility.

        threshold[t] = k * trailing_vol[t] * sqrt(horizon)

    Why this matters: a fixed 0.5% threshold is wrong for the same reason
    absolute price levels were wrong in Phase 2 — it does not transfer. A
    0.5% move is routine for a volatile smallcap and a major event for a
    stable largecap. Scaling by the stock's own recent volatility makes
    "big move" mean the same thing everywhere, which is what allows a new
    stock to be dropped in without retuning anything.

    The sqrt(horizon) term reflects that volatility grows with the square
    root of time under a random walk, so a 5-day threshold should be about
    2.2x a 1-day threshold, not 5x.
    """
    if k < 0:
        raise ValueError(f"k must be >= 0, got {k}")

    fwd = forward_return(df, horizon=horizon, price_col=price_col)
    vol = trailing_volatility(df, window=vol_window, price_col=price_col)
    thresh = k * vol * np.sqrt(horizon)

    target = pd.Series(FLAT, index=df.index, dtype="float64")
    target[fwd > thresh] = UP
    target[fwd < -thresh] = DOWN
    target[fwd.isna() | thresh.isna()] = np.nan
    return target


def big_move_mask(df: pd.DataFrame,
                  horizon: int = 1,
                  k: float = 0.5,
                  vol_window: int = 20,
                  price_col: str = "Close") -> pd.Series:
    """
    Boolean mask of days whose forward move exceeds the volatility-scaled
    threshold in either direction.

    This supports the cleanest formulation of the question:

        "GIVEN that a significant move happens, can we predict its direction?"

    Train and test only on these rows, and the task becomes a well-posed
    binary problem with an honest majority baseline — no abstention metrics
    needed.
    """
    fwd = forward_return(df, horizon=horizon, price_col=price_col)
    vol = trailing_volatility(df, window=vol_window, price_col=price_col)
    thresh = k * vol * np.sqrt(horizon)
    return (fwd.abs() > thresh) & fwd.notna() & thresh.notna()


# ---------------------------------------------------------------------------
# Idea 3 — non-overlapping sampling
# ---------------------------------------------------------------------------

def target_autocorrelation(target: pd.Series, lag: int = 1) -> float:
    """
    Autocorrelation of the label series.

    Near zero is healthy. High values mean consecutive labels are near-copies
    of each other, which inflates the persistence baseline and makes a model
    look good for doing nothing.
    """
    return float(target.dropna().autocorr(lag))


def non_overlapping_subset(df: pd.DataFrame,
                           horizon: int = 1,
                           offset: int = 0) -> pd.DataFrame:
    """
    Keep every `horizon`-th row so that no two labels share any data.

    Why this is necessary
    ---------------------
    At horizon h, Target[t] and Target[t+1] are computed from windows that
    share h-1 days. At h=10 the label autocorrelation is about 0.73, and a
    model that simply repeats yesterday's answer scores roughly 0.88. Any
    "accuracy improvement" from a longer horizon is then mostly an illusion:
    the model AND the baseline both inflate, and the baseline inflates more.

    Subsampling costs rows (2350 -> 235 at h=10) but every remaining label
    is independent, and the persistence baseline falls back to ~0.50 where
    it belongs. Fewer honest rows beat many correlated ones.
    """
    if horizon < 1:
        raise ValueError(f"horizon must be >= 1, got {horizon}")
    if horizon == 1:
        return df.reset_index(drop=True)
    return df.iloc[offset::horizon].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def target_report(target: pd.Series, name: str = "target") -> dict:
    """Class balance, coverage and autocorrelation for one label series."""
    clean = target.dropna()
    counts = clean.value_counts()
    total = len(clean)

    non_flat = clean[clean != FLAT]
    coverage = len(non_flat) / total if total else np.nan

    if len(non_flat):
        up_share = float((non_flat == UP).mean())
        directional_majority = max(up_share, 1 - up_share)
    else:
        up_share = directional_majority = np.nan

    return {
        "target": name,
        "n_labelled": total,
        "pct_UP": round(float(counts.get(UP, 0)) / total, 4) if total else np.nan,
        "pct_FLAT": round(float(counts.get(FLAT, 0)) / total, 4) if total else np.nan,
        "pct_DOWN": round(float(counts.get(DOWN, 0)) / total, 4) if total else np.nan,
        "coverage": round(coverage, 4),
        "up_share_of_moves": round(up_share, 4) if total else np.nan,
        "directional_majority": round(directional_majority, 4) if total else np.nan,
        "autocorr_lag1": round(target_autocorrelation(target), 4),
    }


def compare_targets(df: pd.DataFrame, specs: dict) -> pd.DataFrame:
    """
    Build several target definitions and tabulate their properties.

    `specs` maps a label to a callable taking df and returning a Series:

        {'binary 1d': lambda d: threshold_target(d, 1, 0.0),
         'vol k=0.5': lambda d: volatility_scaled_target(d, 1, 0.5)}
    """
    rows = [target_report(fn(df), name) for name, fn in specs.items()]
    return pd.DataFrame(rows).set_index("target")


# ---------------------------------------------------------------------------
# Dataset assembly
# ---------------------------------------------------------------------------

def build_directional_dataset(df: pd.DataFrame,
                              horizon: int = 1,
                              k: float = 0.5,
                              vol_window: int = 20,
                              non_overlapping: bool = True,
                              price_col: str = "Close") -> pd.DataFrame:
    """
    Produce the "big moves only" dataset.

    Steps, IN THIS ORDER:
      1. Compute the forward return and the big-move mask on the FULL frame.
      2. Optionally subsample every `horizon`-th row of the full frame, so
         kept rows sit `horizon` real trading days apart.
      3. Keep only rows flagged as big moves.
      4. Attach a binary Target: 1 = UP, 0 = DOWN.

    The resulting question is well-posed: given that something significant
    happened, which way did it go? The majority class of this subset is the
    honest baseline.

    Ordering matters, and getting it wrong is easy
    ----------------------------------------------
    An earlier version of this function subsampled FIRST and then computed
    the forward return on the already-subsampled frame. Because shift(-h)
    then moved h *subsampled* rows rather than h trading days, a 5-day
    horizon silently became a 25-day horizon. The symptom was a persistence
    baseline stuck at 0.79 even after subsampling, when it should have
    fallen to ~0.50. Labels must always be computed on the full frame, at
    full time resolution, before any row is dropped.
    """
    work = df.reset_index(drop=True).copy()

    # 1. Labels on the FULL frame, at full time resolution
    work["fwd_return"] = forward_return(work, horizon=horizon,
                                        price_col=price_col)
    work["_is_big_move"] = big_move_mask(work, horizon=horizon, k=k,
                                         vol_window=vol_window,
                                         price_col=price_col)

    # 2. Subsample the full frame so kept rows are `horizon` days apart
    if non_overlapping and horizon > 1:
        work = work.iloc[::horizon]

    # 3. Keep the significant moves
    out = work[work["_is_big_move"]].copy()

    # 4. Binary direction
    out["Target"] = (out["fwd_return"] > 0).astype(int)

    return out.drop(columns=["_is_big_move"]).reset_index(drop=True)
