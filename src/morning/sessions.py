"""NSE trading-day helpers (Indian Standard Time)."""
from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

IST = dt.timezone(dt.timedelta(hours=5, minutes=30), "IST")
MARKET_OPEN = dt.time(9, 15)
MARKET_CLOSE = dt.time(15, 30)


def now_ist() -> dt.datetime:
    return dt.datetime.now(IST)


def load_holidays(path) -> dict:
    """{date string: occasion} for every year in live/nse_holidays.json."""
    path = Path(path)
    if not path.exists():
        return {}
    raw = json.loads(path.read_text(encoding="utf-8"))
    out = {}
    for year, days in raw.items():
        if isinstance(days, dict):
            out.update(days)
    return out


def is_trading_day(day: dt.date, holidays: dict) -> bool:
    return day.weekday() < 5 and day.isoformat() not in holidays


def next_trading_day(day: dt.date, holidays: dict) -> dt.date:
    d = day + dt.timedelta(days=1)
    while not is_trading_day(d, holidays):
        d += dt.timedelta(days=1)
    return d


def prev_trading_day(day: dt.date, holidays: dict) -> dt.date:
    d = day - dt.timedelta(days=1)
    while not is_trading_day(d, holidays):
        d -= dt.timedelta(days=1)
    return d
