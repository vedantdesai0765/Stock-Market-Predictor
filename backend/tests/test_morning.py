"""
backend/tests/test_morning.py
Tests for the Morning Call: the model code in src/morning/ and the API endpoints that serve it.
All prices here are synthetic, so the tests need no downloads and no archived data.
"""

import datetime as dt
import json

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from backend.app.core.config import settings
from backend.app.services.morning_service import MorningService
from morning import sessions as cal, features as F, live, model as M, panel as P, store

IST = cal.IST
TARGET = dt.date(2026, 10, 27)            # a Tuesday; the previous session is Monday 26 October


def _walk(index, seed, start=100.0, vol=0.012):
    rng = np.random.default_rng(seed)
    r = rng.normal(0.0003, vol, len(index))
    close = start * np.exp(np.cumsum(r))
    open_ = np.r_[start, close[:-1]] * np.exp(rng.normal(0, vol / 2, len(index)))
    high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, vol / 3, len(index))))
    low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, vol / 3, len(index))))
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close, "adjclose": close,
                         "volume": rng.integers(1e5, 1e6, len(index)).astype(float)}, index=index)


@pytest.fixture(scope="module")
def prices():
    days = pd.bdate_range("2024-01-01", "2026-10-30")
    px = {tk: _walk(days, i, start=1000 + 50 * i) for i, tk in enumerate(F.MARKET_TICKERS)}
    px[F.VIX] = _walk(days, 99, start=16, vol=0.05)
    stock_px = {k: _walk(days, 10 + i, start=500 + 100 * i) for i, k in enumerate(P.STOCKS)}
    return px, stock_px


@pytest.fixture(scope="module")
def model():
    return M.load_json(settings.MORNING_MODEL_PATH)


def _forecast(prices, model, target=TARGET, hour=8):
    px, stock_px = prices
    now = dt.datetime.combine(target, dt.time(hour, 0), IST)
    return live.make_forecast(px, stock_px, target, model, now, expected_based_on=dt.date(2026, 10, 26))


# ----------------------------------------------------------------------------- model file

def test_model_file_is_complete(model):
    for part in ("open", "close"):
        spec = model[part]
        assert spec["features"] == F.FEATURES
        assert len(spec["coef"]) == len(spec["mean"]) == len(spec["scale"]) == len(spec["impute"]) == len(F.FEATURES)
    tested = model["evidence"]["open"]["backtest"]
    assert tested["accuracy"] > tested["baseline"]
    assert 0.70 < tested["accuracy"] < 0.72          # the published walk-forward result (70.6%)
    assert tested["years_beating_guess"] == tested["years_tested"] == 13


def test_json_model_is_a_probability(model):
    X = pd.DataFrame(np.zeros((3, len(F.FEATURES))), columns=F.FEATURES)
    X.iloc[1] = np.nan                               # missing inputs are filled with training means
    p = M.predict(model["open"], X)
    assert ((p > 0) & (p < 1)).all()


# ----------------------------------------------------------------------------- forecast

def test_forecast_covers_every_stock(prices, model):
    fc = _forecast(prices, model)
    assert fc["status"] == "ok" and fc["on_time"] is True
    assert fc["target_date"] == "2026-10-27" and fc["based_on"] == "2026-10-26"
    assert [c["ticker"] for c in fc["calls"]] == list(P.STOCKS)
    for c in fc["calls"]:
        assert c["available"] and c["call"] in ("HIGHER", "LOWER")
        assert (c["call"] == "HIGHER") == (c["prob_up"] > 0.5)
        assert len(c["drivers"]) == 3 and abs(sum(d["weight"] for d in c["drivers"])) <= 1.0001
    assert fc["summary"]["higher"] + fc["summary"]["lower"] == 10
    assert {m["key"] for m in fc["market"]} == {"sp500", "vix", "n225", "ks11", "twii", "hsi"}


def test_forecast_never_uses_the_future(prices, model):
    """Changing anything that happens after the forecast is made must not change the forecast."""
    px, stock_px = prices
    base = _forecast(prices, model)
    t = pd.Timestamp(TARGET)
    px2 = {k: d.copy() for k, d in px.items()}
    st2 = {k: d.copy() for k, d in stock_px.items()}
    for d in list(px2.values()) + list(st2.values()):
        d.loc[d.index > t, ["open", "high", "low", "close", "adjclose"]] *= 1.5       # every later day
    for k, d in st2.items():
        d.loc[t, ["open", "high", "low", "close", "adjclose"]] *= 1.2                  # today's Indian session
    for tk in (F.NIFTY, F.SP500, F.VIX):
        px2[tk].loc[t, ["open", "high", "low", "close", "adjclose"]] *= 1.2            # today's Nifty / US session
    for tk in F.ASIA:
        px2[tk].loc[t, ["high", "low", "close", "adjclose"]] *= 1.2                    # only the Asian OPEN is known
    now = dt.datetime.combine(TARGET, dt.time(8, 0), IST)
    again = live.make_forecast(px2, st2, TARGET, model, now, expected_based_on=dt.date(2026, 10, 26))
    assert [c["prob_up"] for c in again["calls"]] == [c["prob_up"] for c in base["calls"]]
    assert again["market"] == base["market"]


def test_asian_open_does_change_the_forecast(prices, model):
    px, stock_px = prices
    base = _forecast(prices, model)
    px2 = {k: d.copy() for k, d in px.items()}
    for tk in F.ASIA:
        px2[tk].loc[pd.Timestamp(TARGET), "open"] *= 1.03
    now = dt.datetime.combine(TARGET, dt.time(8, 0), IST)
    up = live.make_forecast(px2, stock_px, TARGET, model, now, expected_based_on=dt.date(2026, 10, 26))
    assert all(a["prob_up"] > b["prob_up"] for a, b in zip(up["calls"], base["calls"]))


def test_stale_stock_is_marked_unavailable(prices, model):
    px, stock_px = prices
    st2 = dict(stock_px)
    st2["INFY"] = stock_px["INFY"].loc[:"2026-10-22"]
    now = dt.datetime.combine(TARGET, dt.time(8, 0), IST)
    fc = live.make_forecast(px, st2, TARGET, model, now, expected_based_on=dt.date(2026, 10, 26))
    infy = next(c for c in fc["calls"] if c["ticker"] == "INFY")
    assert infy["available"] is False and "expected 2026-10-26" in infy["reason"]
    assert fc["summary"]["unavailable"] == 1 and fc["status"] == "ok"


def test_forecast_made_after_the_open_is_late(prices, model):
    assert _forecast(prices, model, hour=11)["on_time"] is False


# ----------------------------------------------------------------------------- history, grading, scorecard

def test_grading_and_scorecard(prices, model):
    px, stock_px = prices
    fc = _forecast(prices, model)
    hist = live.merge_history(pd.DataFrame(columns=live.HISTORY_COLUMNS), live.history_rows(fc))
    assert len(hist) == 10

    visible = {k: d.loc[:str(TARGET)] for k, d in stock_px.items()}
    early = live.grade(hist, visible, dt.datetime.combine(TARGET, dt.time(8, 30), IST))
    assert early.actual.isna().all()                                   # nothing is graded before the open

    mid = live.grade(hist, visible, dt.datetime.combine(TARGET, dt.time(9, 45), IST))
    assert mid.actual.isin(["HIGHER", "LOWER", "FLAT"]).all() and mid.close_actual.isna().all()
    for row in mid.itertuples():
        d = stock_px[row.ticker]
        truth = "HIGHER" if d.open[pd.Timestamp(TARGET)] > d.close[pd.Timestamp("2026-10-26")] else "LOWER"
        assert row.actual == truth and row.correct == (row.call == truth)

    done = live.grade(mid, visible, dt.datetime.combine(TARGET, dt.time(16, 30), IST))
    assert done.close_actual.isin(["HIGHER", "LOWER", "FLAT"]).all()

    card = live.scorecard(done, dt.datetime.combine(TARGET, dt.time(16, 30), IST))
    assert card["mornings"] == 1 and card["open"]["n"] == 10
    assert card["open"]["correct"] == int(done.correct.astype(str).str.lower().eq("true").sum())
    assert card["days"] == [{"date": "2026-10-27", "n": 10, "correct": card["open"]["correct"]}]


def test_on_time_forecast_is_never_replaced(prices, model):
    first = live.history_rows(_forecast(prices, model))
    later = first.copy()
    later["prob_up"] = 0.99
    merged = live.merge_history(first, later)
    assert len(merged) == 10 and (merged.prob_up != 0.99).all()


def test_late_forecasts_do_not_count(prices, model):
    px, stock_px = prices
    hist = live.history_rows(_forecast(prices, model, hour=11))
    visible = {k: d.loc[:str(TARGET)] for k, d in stock_px.items()}
    now = dt.datetime.combine(TARGET, dt.time(16, 30), IST)
    card = live.scorecard(live.grade(hist, visible, now), now)
    assert card["open"]["n"] == 0 and card["excluded_late"] > 0


def test_missing_session_is_voided(prices, model):
    px, stock_px = prices
    hist = live.history_rows(_forecast(prices, model))
    no_bar = {k: d.drop(pd.Timestamp(TARGET)) for k, d in stock_px.items()}
    graded = live.grade(hist, no_bar, dt.datetime.combine(TARGET + dt.timedelta(days=5), dt.time(10, 0), IST))
    assert (graded.actual == "VOID").all()


def test_calendar():
    holidays = cal.load_holidays(settings.ROOT_DIR / "live" / "nse_holidays.json")
    assert not cal.is_trading_day(dt.date(2026, 11, 24), holidays)       # Guru Nanak Jayanti
    assert not cal.is_trading_day(dt.date(2026, 11, 14), holidays)       # Saturday
    assert cal.next_trading_day(dt.date(2026, 11, 23), holidays) == dt.date(2026, 11, 25)
    assert cal.prev_trading_day(dt.date(2026, 11, 16), holidays) == dt.date(2026, 11, 13)


# ----------------------------------------------------------------------------- API

@pytest.fixture()
def live_dir(tmp_path, monkeypatch, prices, model):
    """A live/ folder holding one graded morning."""
    px, stock_px = prices
    fc = _forecast(prices, model)
    now = dt.datetime.combine(TARGET, dt.time(16, 30), IST)
    hist = live.grade(live.history_rows(fc), {k: d.loc[:str(TARGET)] for k, d in stock_px.items()}, now)
    store.write_json(fc, tmp_path / store.FORECAST_FILE)
    store.write_history(hist, tmp_path / store.HISTORY_FILE)
    store.write_json(live.scorecard(hist, now), tmp_path / store.SCORECARD_FILE)
    (tmp_path / store.HOLIDAY_FILE).write_text(
        (settings.ROOT_DIR / "live" / store.HOLIDAY_FILE).read_text(encoding="utf-8"), encoding="utf-8")
    monkeypatch.setattr(settings, "LIVE_DIR", tmp_path)
    monkeypatch.setattr(settings, "MORNING_DATA_URL", "")
    MorningService.clear_cache()
    yield tmp_path
    MorningService.clear_cache()


def test_morning_call_endpoint(client: TestClient, live_dir):
    r = client.get("/api/v1/morning-call")
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "ok" and data["target_date"] == "2026-10-27"
    assert len(data["calls"]) == 10 and len(data["market"]) == 6
    first = data["calls"][0]
    assert first["result"]["actual"] in ("HIGHER", "LOWER", "FLAT")
    assert first["confidence"]["tested_accuracy"] > 0.5
    assert data["model"]["tested"]["accuracy"] > data["model"]["tested"]["baseline"]
    assert data["next_call_date"] >= data["today"]


def test_morning_call_phases(live_dir):
    at = lambda d, h, m=0: MorningService.get_morning_call(dt.datetime.combine(d, dt.time(h, m), IST))
    same_day = at(TARGET, 10)
    assert same_day["is_today"] and same_day["market_phase"] == "open"
    assert same_day["next_call_date"] == "2026-10-28"
    next_morning = at(dt.date(2026, 10, 28), 7, 30)                     # today's call has not arrived yet
    assert not next_morning["is_today"] and next_morning["market_phase"] == "pre_open"
    assert next_morning["next_call_date"] == "2026-10-28"
    weekend = at(dt.date(2026, 10, 31), 12)
    assert weekend["market_phase"] == "holiday" and weekend["next_call_date"] == "2026-11-02"
    assert weekend["reason"] == "Weekend"


def test_scorecard_endpoint(client: TestClient, live_dir):
    r = client.get("/api/v1/morning-call/scorecard?recent_days=3")
    assert r.status_code == 200
    data = r.json()
    assert data["mornings"] == 1 and data["open"]["n"] + 0 <= 10
    assert len(data["recent"]) == 1 and data["recent"][0]["date"] == "2026-10-27"
    assert data["tested"]["years"] == "2013-2025"
    assert [b["from"] for b in data["confidence_bands"]] == [0.5, 0.6, 0.7, 0.8]
    accs = [b["accuracy"] for b in data["confidence_bands"]]
    assert accs == sorted(accs)                                         # more confident = more often right


def test_endpoints_before_the_first_forecast(client: TestClient, tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "LIVE_DIR", tmp_path)
    monkeypatch.setattr(settings, "MORNING_DATA_URL", "")
    MorningService.clear_cache()
    r = client.get("/api/v1/morning-call")
    assert r.status_code == 200 and r.json()["status"] == "not_started" and r.json()["calls"] == []
    s = client.get("/api/v1/morning-call/scorecard")
    assert s.status_code == 200 and s.json()["mornings"] == 0 and s.json()["tested"]["n"] > 20000
    MorningService.clear_cache()


def test_falls_back_to_local_files_when_github_is_unreachable(client: TestClient, live_dir, monkeypatch):
    monkeypatch.setattr(settings, "MORNING_DATA_URL", "http://127.0.0.1:9/nowhere")
    MorningService.clear_cache()
    r = client.get("/api/v1/morning-call")
    assert r.status_code == 200 and r.json()["status"] == "ok" and r.json()["source"] == "local"
