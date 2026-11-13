"""
Build one morning's forecast, and grade past forecasts. Pure functions: prices in, dictionaries out.
The scripts in scripts/ handle downloading and files.
"""
from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd

from . import features as F, model as M
from .sessions import IST, MARKET_OPEN
from .panel import NAMES, STOCKS

ASIA_LABELS = {"^N225": ("n225", "Nikkei 225", "Japan"), "^KS11": ("ks11", "KOSPI", "South Korea"),
               "^TWII": ("twii", "Taiex", "Taiwan"), "^HSI": ("hsi", "Hang Seng", "Hong Kong")}
OPEN_KNOWN = dt.time(9, 20)   # the opening price is not trusted before this time (IST)
MIN_HISTORY = 260        # trading days needed for the 200-day average plus a margin
HISTORY_COLUMNS = [
    "target_date", "ticker", "based_on", "prev_close", "prob_up", "call", "baseline_call",
    "generated_at", "on_time",
    "open_price", "actual", "correct", "baseline_correct", "graded_at",
    "close_prob_up", "close_call", "close_baseline_call", "close_price", "close_actual",
    "close_correct", "close_baseline_correct",
]


def _pct(log_return: float) -> float:
    return float((np.exp(log_return) - 1) * 100)


def _call(p: float) -> str:
    return "HIGHER" if p > 0.5 else "LOWER"


def _confidence(evidence: dict, p: float) -> dict:
    conf = max(p, 1 - p)
    for band in evidence.get("confidence", []):
        if band["from"] <= conf < band["to"] or (band["to"] >= 1.0 and conf >= band["from"]):
            level = {0.5: "Low", 0.6: "Moderate", 0.7: "High", 0.8: "Very high"}.get(band["from"], "")
            return {"level": level, "tested_accuracy": band["accuracy"], "tested_n": band["n"]}
    return {"level": "", "tested_accuracy": None, "tested_n": 0}


def cut_prices(px: dict, stock_px: dict, target: pd.Timestamp):
    """Keep only what is public on the morning of `target`, before the NSE opens."""
    out_px = {}
    for tk, d in px.items():
        if tk in F.ASIA:
            out_px[tk] = d.loc[:target]                             # today's Asian open is already known
        else:
            out_px[tk] = d.loc[:target - pd.Timedelta(days=1)]      # Nifty, S&P 500, VIX: up to yesterday
    out_stock = {k: d.loc[:target - pd.Timedelta(days=1)] for k, d in stock_px.items()}
    return out_px, out_stock


def market_snapshot(px: dict, target: pd.Timestamp, based_on: pd.Timestamp) -> list:
    """The overnight inputs, in plain numbers, for the strip at the top of the page."""
    items = []
    sp = px[F.SP500]
    spr = np.log(sp.close).diff()
    recent = spr.loc[based_on:]                 # US sessions since the last Indian close
    if len(recent):
        items.append({"key": "sp500", "label": "S&P 500", "region": "United States",
                      "change_pct": round(_pct(recent.sum()), 2), "level": round(float(sp.close.iloc[-1]), 2),
                      "session": str(recent.index[-1].date()), "available": True})
    else:
        items.append({"key": "sp500", "label": "S&P 500", "region": "United States", "change_pct": None,
                      "level": round(float(sp.close.iloc[-1]), 2), "session": str(sp.index[-1].date()),
                      "available": False, "note": "No US session since the last Indian close"})
    vix = px[F.VIX].close
    vchg = _pct(np.log(vix.iloc[-1] / vix.iloc[-2])) if len(vix) > 1 else None
    items.append({"key": "vix", "label": "VIX", "region": "US volatility index",
                  "change_pct": round(vchg, 2) if vchg is not None else None,
                  "level": round(float(vix.iloc[-1]), 2), "session": str(vix.index[-1].date()), "available": True})
    for tk in F.ASIA:
        key, label, region = ASIA_LABELS[tk]
        d = px[tk]
        if len(d) and d.index[-1] == target and len(d) > 1:
            gap = _pct(np.log(d.open.iloc[-1] / d.close.iloc[-2]))
            items.append({"key": key, "label": label, "region": region, "change_pct": round(gap, 2),
                          "level": round(float(d.open.iloc[-1]), 2), "session": str(target.date()),
                          "available": True})
        else:
            items.append({"key": key, "label": label, "region": region, "change_pct": None, "level": None,
                          "session": str(target.date()), "available": False,
                          "note": "Closed today, or not open yet"})
    return items


def _drivers(spec: dict, x: pd.Series, snap: dict) -> list:
    """The three groups of inputs that push this forecast hardest, in plain words."""
    c = M.contributions(spec, x)
    total = sum(abs(v) for v in c.values()) or 1.0
    asia_ok = [m for m in snap.values() if m["key"] in ("n225", "ks11", "twii", "hsi") and m["available"]]
    asia_avg = float(np.mean([m["change_pct"] for m in asia_ok])) if asia_ok else None
    sp = snap["sp500"]
    text = {
        "us": ("US market overnight",
               f"S&P 500 {sp['change_pct']:+.2f}% overnight" if sp["available"] else "No US session overnight"),
        "vix": ("Market fear (VIX)",
                f"VIX at {snap['vix']['level']:.1f}" + (f" ({snap['vix']['change_pct']:+.1f}%)"
                                                         if snap["vix"]["change_pct"] is not None else "")),
        "asia": ("Asian markets this morning",
                 f"Asian markets opened {asia_avg:+.2f}% on average" if asia_avg is not None else "No Asian market open yet"),
        "nifty": ("Nifty trend", f"Nifty {_pct(x['nifty_r1']):+.2f}% yesterday"),
        "stock": ("The stock's own trend", f"Stock {_pct(x['ret_l1']):+.2f}% yesterday, RSI {x['rsi_14']:.0f}"),
    }
    rows = [{"key": g, "label": text[g][0], "detail": text[g][1],
             "direction": "up" if v > 0 else "down", "weight": round(abs(v) / total, 3)}
            for g, v in c.items()]
    rows.sort(key=lambda r: -r["weight"])
    return rows[:3]


def make_forecast(px: dict, stock_px: dict, target_date: dt.date, model: dict,
                  generated_at: dt.datetime, expected_based_on: dt.date | None = None) -> dict:
    """Forecast the session of `target_date` for every stock. Prices may contain later rows; they are cut."""
    target = pd.Timestamp(target_date)
    px, stock_px = cut_prices(px, stock_px, target)
    nifty_last = px[F.NIFTY].index[-1]
    based_on = pd.Timestamp(expected_based_on) if expected_based_on else nifty_last
    notes = []
    if nifty_last != based_on:
        notes.append(f"Nifty data ends {nifty_last.date()}, expected {based_on.date()}.")
    snap_list = market_snapshot(px, target, based_on)
    snap = {m["key"]: m for m in snap_list}
    if not any(snap[k]["available"] for k in ("n225", "ks11", "twii", "hsi")):
        notes.append("No Asian market had opened when this forecast was made.")
    if not snap["sp500"]["available"]:
        notes.append("There was no US session between the last Indian close and this forecast.")

    open_spec, close_spec = model["open"], model["close"]
    base_open = "HIGHER" if open_spec["base_rate_up"] > 0.5 else "LOWER"
    base_close = "HIGHER" if close_spec["base_rate_up"] > 0.5 else "LOWER"
    on_time = generated_at.astimezone(IST) < dt.datetime.combine(target_date, MARKET_OPEN, IST)

    calls = []
    for key in model.get("stocks", list(STOCKS)):
        entry = {"ticker": key, "name": NAMES.get(key, key)}
        d = stock_px.get(key)
        problem = None
        if d is None or len(d) < MIN_HISTORY:
            problem = "Not enough price history was downloaded."
        elif d.index[-1] != based_on:
            problem = f"Latest price is from {d.index[-1].date()}, expected {based_on.date()}."
        if problem is None:
            x = F.stock_frame(d, px, next_day=target).iloc[-1]
            if x[F.TECH].isna().any():
                problem = "Some indicators could not be computed."
        if problem:
            calls.append({**entry, "available": False, "reason": problem})
            continue
        feats = x[F.FEATURES].astype(float)
        p_open = float(M.predict(open_spec, pd.DataFrame([feats]))[0])
        p_close = float(M.predict(close_spec, pd.DataFrame([feats]))[0])
        calls.append({
            **entry, "available": True,
            "prev_close": round(float(d.close.iloc[-1]), 2),
            "prob_up": round(p_open, 4), "call": _call(p_open), "baseline_call": base_open,
            "confidence": _confidence(model["evidence"]["open"], p_open),
            "drivers": _drivers(open_spec, feats, snap),
            "close": {"prob_up": round(p_close, 4), "call": _call(p_close), "baseline_call": base_close,
                      "confidence": _confidence(model["evidence"]["close"], p_close)},
        })

    ok = [c for c in calls if c["available"]]
    ev = model["evidence"]["open"]
    return {
        "schema": 1,
        "status": "ok" if ok else "unavailable",
        "target_date": str(target_date),
        "based_on": str(based_on.date()),
        "generated_at": generated_at.astimezone(IST).isoformat(timespec="seconds"),
        "on_time": bool(on_time),
        "summary": {"higher": sum(c["call"] == "HIGHER" for c in ok),
                    "lower": sum(c["call"] == "LOWER" for c in ok),
                    "unavailable": len(calls) - len(ok)},
        "market": snap_list,
        "calls": calls,
        "model": {"name": "Logistic regression, pooled across 10 stocks",
                  "trained_through": open_spec["train_through"],
                  "tested": ev["backtest"], "after_study": ev.get("after_study"),
                  "simple_guess": base_open, "base_rate_up": round(open_spec["base_rate_up"], 4)},
        "notes": notes,
    }


def closed_notice(today: dt.date, next_session: dt.date, reason: str, generated_at: dt.datetime) -> dict:
    return {"schema": 1, "status": "market_closed", "target_date": str(today), "reason": reason,
            "next_session": str(next_session),
            "generated_at": generated_at.astimezone(IST).isoformat(timespec="seconds"),
            "market": [], "calls": [], "notes": []}


def failure_notice(today: dt.date, message: str, generated_at: dt.datetime) -> dict:
    return {"schema": 1, "status": "unavailable", "target_date": str(today), "reason": message,
            "generated_at": generated_at.astimezone(IST).isoformat(timespec="seconds"),
            "market": [], "calls": [], "notes": [message]}


# ----------------------------------------------------------------------------- history and grading

def history_rows(forecast: dict) -> pd.DataFrame:
    rows = []
    for c in forecast["calls"]:
        if not c.get("available"):
            continue
        rows.append({
            "target_date": forecast["target_date"], "ticker": c["ticker"], "based_on": forecast["based_on"],
            "prev_close": c["prev_close"], "prob_up": c["prob_up"], "call": c["call"],
            "baseline_call": c["baseline_call"], "generated_at": forecast["generated_at"],
            "on_time": bool(forecast["on_time"]),
            "close_prob_up": c["close"]["prob_up"], "close_call": c["close"]["call"],
            "close_baseline_call": c["close"]["baseline_call"],
        })
    return pd.DataFrame(rows, columns=HISTORY_COLUMNS)


def merge_history(history: pd.DataFrame, new: pd.DataFrame) -> pd.DataFrame:
    """Add a morning's forecasts. An on-time forecast that already exists is never replaced."""
    if history is None or history.empty:
        return new.copy()
    if new.empty:
        return history
    keep = history.copy()
    key = lambda df: df.target_date.astype(str) + "|" + df.ticker.astype(str)
    existing = dict(zip(key(keep), zip(keep.on_time.astype(str).str.lower() == "true", keep.actual.notna())))
    add, drop = [], []
    for k, row in zip(key(new), new.to_dict("records")):
        if k not in existing:
            add.append(row)
        else:
            was_on_time, graded = existing[k]
            if not was_on_time and not graded and row["on_time"]:
                drop.append(k)
                add.append(row)
    if drop:
        keep = keep[~key(keep).isin(drop)]
    if add:
        keep = pd.concat([keep, pd.DataFrame(add, columns=HISTORY_COLUMNS)], ignore_index=True)
    return keep.sort_values(["target_date", "ticker"]).reset_index(drop=True)


def grade(history: pd.DataFrame, stock_px: dict, now: dt.datetime, void_after_days: int = 4) -> pd.DataFrame:
    """Fill in what really happened for every forecast whose session has data.

    Opening call: open(target) against close(previous session), both from the same download.
    Closing call: close(target) against close(previous session), once the session has finished.
    A session with no price bar after `void_after_days` (unexpected market closure) is marked VOID.
    """
    if history is None or history.empty:
        return history
    h = history.copy()
    for col in ("actual", "close_actual", "graded_at", "correct", "baseline_correct",
                "close_correct", "close_baseline_correct"):
        h[col] = h[col].astype(object)
    now_ist = now.astimezone(IST)
    stamp = now_ist.isoformat(timespec="seconds")
    for i, row in h.iterrows():
        d = stock_px.get(row.ticker)
        if d is None or not len(d):
            continue
        target = pd.Timestamp(row.target_date)
        day_over = now_ist.date() > target.date() or (now_ist.date() == target.date() and now_ist.hour >= 16)
        opened = now_ist >= dt.datetime.combine(target.date(), OPEN_KNOWN, IST)
        if not opened:
            continue                                    # the session has not started: nothing to check yet
        has_bar = target in d.index
        if not has_bar:
            if pd.isna(row.actual) and (now_ist.date() - target.date()).days >= void_after_days:
                h.at[i, "actual"] = "VOID"
                h.at[i, "close_actual"] = "VOID"
                h.at[i, "graded_at"] = stamp
            continue
        before = d.loc[:target - pd.Timedelta(days=1)]
        prev_close = float(before.close.iloc[-1]) if len(before) else float(row.prev_close)
        bar = d.loc[target]
        if pd.isna(row.actual) and pd.notna(bar.open) and bar.open > 0:
            gap = np.log(float(bar.open) / prev_close)
            actual = "FLAT" if abs(gap) <= 1e-6 else ("HIGHER" if gap > 0 else "LOWER")
            h.at[i, "open_price"] = round(float(bar.open), 2)
            h.at[i, "actual"] = actual
            h.at[i, "graded_at"] = stamp
            if actual != "FLAT":
                h.at[i, "correct"] = bool(row.call == actual)
                h.at[i, "baseline_correct"] = bool(row.baseline_call == actual)
        if pd.isna(row.close_actual) and day_over and pd.notna(bar.close):
            move = np.log(float(bar.close) / prev_close)
            actual = "FLAT" if move == 0 else ("HIGHER" if move > 0 else "LOWER")
            h.at[i, "close_price"] = round(float(bar.close), 2)
            h.at[i, "close_actual"] = actual
            if actual != "FLAT":
                h.at[i, "close_correct"] = bool(row.close_call == actual)
                h.at[i, "close_baseline_correct"] = bool(row.close_baseline_call == actual)
    return h


def _truth(s: pd.Series) -> pd.Series:
    return s.astype(str).str.lower().map({"true": True, "false": False})


def _score(correct: pd.Series, baseline: pd.Series) -> dict:
    c, b = _truth(correct).dropna(), _truth(baseline).dropna()
    n = int(len(c))
    return {"n": n, "correct": int(c.sum()), "accuracy": round(float(c.mean()), 4) if n else None,
            "baseline_correct": int(b.sum()), "baseline_accuracy": round(float(b.mean()), 4) if len(b) else None}


def scorecard(history: pd.DataFrame, now: dt.datetime, recent_days: int = 20) -> dict:
    """Live record of the forecasts that were made before the market opened."""
    base = {"schema": 1, "updated_at": now.astimezone(IST).isoformat(timespec="seconds")}
    empty = {**base, "since": None, "mornings": 0, "open": _score(pd.Series(dtype=object), pd.Series(dtype=object)),
             "close": _score(pd.Series(dtype=object), pd.Series(dtype=object)), "days": [], "per_stock": [],
             "excluded_late": 0}
    if history is None or history.empty:
        return empty
    h = history.copy()
    on_time = h.on_time.astype(str).str.lower() == "true"
    late = int((~on_time & h.actual.isin(["HIGHER", "LOWER"])).sum())
    g = h[on_time & h.actual.isin(["HIGHER", "LOWER"])]
    if g.empty:
        return {**empty, "excluded_late": late}
    ok = _truth(g.correct)
    days = (g.assign(ok=ok).groupby("target_date").agg(n=("ok", "size"), correct=("ok", "sum")).reset_index())
    per = (g.assign(ok=ok).groupby("ticker").agg(n=("ok", "size"), correct=("ok", "sum")).reset_index())
    gc = h[on_time & h.close_actual.isin(["HIGHER", "LOWER"])]
    return {
        **base,
        "since": str(g.target_date.min()),
        "mornings": int(g.target_date.nunique()),
        "open": _score(g.correct, g.baseline_correct),
        "close": _score(gc.close_correct, gc.close_baseline_correct),
        "days": [{"date": str(r.target_date), "n": int(r.n), "correct": int(r.correct)}
                 for r in days.tail(recent_days).itertuples()],
        "per_stock": [{"ticker": r.ticker, "name": NAMES.get(r.ticker, r.ticker), "n": int(r.n),
                       "correct": int(r.correct), "accuracy": round(r.correct / r.n, 4)}
                      for r in per.itertuples()],
        "excluded_late": late,
    }
