"""
backend/app/services/morning_service.py
Serves the Morning Call: the daily forecast, its results and the live scorecard.

The numbers are produced outside the API, by scripts/morning_call.py and
scripts/grade_morning_call.py, which a scheduled GitHub Actions job runs every weekday and
commits to the live/ folder. This service only reads those files:

  1. from MORNING_DATA_URL (the public live/ folder on GitHub) when that is configured, so a new
     forecast shows up within a few minutes without redeploying the API;
  2. otherwise, or if GitHub cannot be reached, from the live/ folder shipped with the code.
"""

from __future__ import annotations

import csv
import datetime as dt
import io
import json
import logging
import threading
import time
import urllib.request
from typing import Any, Optional, Tuple

from backend.app.core.config import settings

log = logging.getLogger("uvicorn.error")

IST = dt.timezone(dt.timedelta(hours=5, minutes=30), "IST")
MARKET_OPEN, MARKET_CLOSE = dt.time(9, 15), dt.time(15, 30)
FORECAST_FILE, HISTORY_FILE, SCORECARD_FILE, HOLIDAY_FILE = (
    "latest_forecast.json", "forecast_history.csv", "scorecard.json", "nse_holidays.json")


def _bool(v: Any) -> Optional[bool]:
    s = str(v).strip().lower()
    return True if s == "true" else False if s == "false" else None


def _num(v: Any) -> Optional[float]:
    try:
        f = float(v)
        return f if f == f else None
    except (TypeError, ValueError):
        return None


class MorningService:
    _cache: dict = {}
    _lock = threading.Lock()

    # ------------------------------------------------------------------ file access

    @classmethod
    def clear_cache(cls) -> None:
        with cls._lock:
            cls._cache.clear()

    @classmethod
    def _read_text(cls, name: str) -> Tuple[Optional[str], str]:
        """Returns (text, source). Cached for MORNING_CACHE_SECONDS."""
        now = time.time()
        with cls._lock:
            hit = cls._cache.get(name)
            if hit and hit[0] > now:
                return hit[1], hit[2]
        text, source, ttl = None, "local", settings.MORNING_CACHE_SECONDS
        if settings.MORNING_DATA_URL:
            try:
                req = urllib.request.Request(f"{settings.MORNING_DATA_URL}/{name}",
                                             headers={"User-Agent": "stock-market-predictor-api"})
                with urllib.request.urlopen(req, timeout=6) as resp:       # noqa: S310 (fixed https URL)
                    text, source = resp.read().decode("utf-8"), "github"
            except Exception as exc:                                       # noqa: BLE001
                log.warning("Morning Call: could not read %s from GitHub (%s); using the local copy.", name, exc)
                ttl = 60          # try GitHub again soon
        if text is None:
            path = settings.LIVE_DIR / name
            if path.exists():
                text = path.read_text(encoding="utf-8")
        with cls._lock:
            cls._cache[name] = (now + ttl, text, source)
        return text, source

    @classmethod
    def _read_json(cls, name: str) -> Tuple[Optional[dict], str]:
        text, source = cls._read_text(name)
        if not text:
            return None, source
        try:
            return json.loads(text), source
        except ValueError:
            log.warning("Morning Call: %s is not valid JSON.", name)
            return None, source

    @classmethod
    def _history(cls) -> list:
        text, _ = cls._read_text(HISTORY_FILE)
        if not text:
            return []
        return list(csv.DictReader(io.StringIO(text)))

    @classmethod
    def _holidays(cls) -> dict:
        raw, _ = cls._read_json(HOLIDAY_FILE)
        out: dict = {}
        for days in (raw or {}).values():
            if isinstance(days, dict):
                out.update(days)
        return out

    @staticmethod
    def _model_evidence() -> dict:
        try:
            return json.loads(settings.MORNING_MODEL_PATH.read_text(encoding="utf-8")).get("evidence", {})
        except (OSError, ValueError):
            return {}

    # ------------------------------------------------------------------ calendar

    @staticmethod
    def _is_trading_day(day: dt.date, holidays: dict) -> bool:
        return day.weekday() < 5 and day.isoformat() not in holidays

    @classmethod
    def _next_trading_day(cls, day: dt.date, holidays: dict) -> dt.date:
        d = day + dt.timedelta(days=1)
        while not cls._is_trading_day(d, holidays):
            d += dt.timedelta(days=1)
        return d

    # ------------------------------------------------------------------ public API

    @classmethod
    def get_morning_call(cls, now: Optional[dt.datetime] = None) -> dict:
        now = (now or dt.datetime.now(IST)).astimezone(IST)
        today = now.date()
        holidays = cls._holidays()
        trading_today = cls._is_trading_day(today, holidays)
        if not trading_today:
            phase = "holiday"
        elif now.time() < MARKET_OPEN:
            phase = "pre_open"
        elif now.time() < MARKET_CLOSE:
            phase = "open"
        else:
            phase = "closed"

        forecast, source = cls._read_json(FORECAST_FILE)
        forecast = forecast or {}
        target = forecast.get("target_date")
        is_today = target == today.isoformat()
        waiting_for_today = trading_today and not is_today and now.time() < MARKET_OPEN
        next_call = today if waiting_for_today else cls._next_trading_day(today, holidays)

        out = {
            "status": forecast.get("status", "not_started"),
            "today": today.isoformat(),
            "server_time": now.isoformat(timespec="seconds"),
            "is_today": is_today,
            "market_phase": phase,
            "next_call_date": next_call.isoformat(),
            "target_date": target,
            "based_on": forecast.get("based_on"),
            "generated_at": forecast.get("generated_at"),
            "on_time": forecast.get("on_time"),
            "reason": forecast.get("reason"),
            "next_session": forecast.get("next_session"),
            "summary": forecast.get("summary"),
            "market": forecast.get("market", []),
            "calls": forecast.get("calls", []),
            "model": forecast.get("model"),
            "notes": forecast.get("notes", []),
            "source": source,
        }
        if not is_today and not trading_today and out["status"] != "market_closed":
            out["reason"] = holidays.get(today.isoformat(), "Weekend")

        # attach what actually happened to each call
        results = {r["ticker"]: r for r in cls._history() if r.get("target_date") == target}
        for call in out["calls"]:
            r = results.get(call.get("ticker"))
            if r and (r.get("actual") or r.get("close_actual")):
                call["result"] = {
                    "open_price": _num(r.get("open_price")), "actual": r.get("actual") or None,
                    "correct": _bool(r.get("correct")),
                    "close_price": _num(r.get("close_price")), "close_actual": r.get("close_actual") or None,
                    "close_correct": _bool(r.get("close_correct")),
                }
        return out

    @classmethod
    def get_scorecard(cls, recent_days: int = 5) -> dict:
        card, source = cls._read_json(SCORECARD_FILE)
        card = dict(card or {})
        graded = [r for r in cls._history()
                  if r.get("actual") in ("HIGHER", "LOWER", "FLAT") and _bool(r.get("on_time"))]
        dates = sorted({r["target_date"] for r in graded}, reverse=True)[:max(0, recent_days)]
        recent = []
        for d in dates:
            rows = sorted((r for r in graded if r["target_date"] == d), key=lambda r: r["ticker"])
            recent.append({"date": d, "results": [
                {"ticker": r["ticker"], "call": r["call"], "prob_up": _num(r["prob_up"]) or 0.0,
                 "actual": r["actual"], "correct": _bool(r.get("correct"))} for r in rows]})
        ev = cls._model_evidence().get("open", {})
        return {
            "updated_at": card.get("updated_at"), "since": card.get("since"),
            "mornings": card.get("mornings", 0),
            "open": card.get("open") or {}, "close": card.get("close") or {},
            "days": card.get("days", []), "per_stock": card.get("per_stock", []),
            "excluded_late": card.get("excluded_late", 0),
            "recent": recent,
            "tested": ev.get("backtest"), "after_study": ev.get("after_study"),
            "confidence_bands": ev.get("confidence", []),
            "source": source,
        }
