"""
The 8:00 am job: make this morning's forecast for all 10 stocks.

Runs automatically on GitHub Actions every weekday (.github/workflows/morning-call.yml).
It downloads last night's US market, this morning's Asian opens and the latest Indian prices from
Yahoo Finance, applies the saved model (models/morning/model.json) and writes:

    live/latest_forecast.json     today's 10 calls, the inputs and the reasons
    live/forecast_history.csv     one row per stock per morning (never rewritten once on time)
    live/scorecard.json           the live record so far (re-graded with the fresh prices)

Run by hand:      python scripts/morning_call.py
Replay a past morning from an archive (testing only, writes to another folder):
    python scripts/morning_call.py --prices FILE --now 2026-10-27T08:00:00+05:30 --out /tmp/replay
"""
import argparse
import datetime as dt
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from morning import sessions as cal, data, features as F, live, model as M, panel as P, store


def load_prices(args, period):
    if args.prices:
        allpx = data.load_long_file(args.prices)
        px = {t: allpx[t] for t in F.MARKET_TICKERS}
        stock_px = {}
        for k, t in P.STOCKS.items():
            stock_px[k] = allpx[t] if t in allpx else data.load_raw_csv(ROOT / "data" / "raw" / f"{k}_raw.csv")
        return px, stock_px
    got = data.download(F.MARKET_TICKERS + list(P.STOCKS.values()), period=period)
    return {t: got[t] for t in F.MARKET_TICKERS}, {k: got[t] for k, t in P.STOCKS.items()}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--prices", help="archived long-format price file instead of downloading (replay)")
    ap.add_argument("--now", help="pretend it is this time, ISO format with offset (replay)")
    ap.add_argument("--out", default=str(ROOT / "live"), help="folder for the output files")
    ap.add_argument("--force", action="store_true", help="replace today's forecast even if one exists")
    args = ap.parse_args()

    now = dt.datetime.fromisoformat(args.now) if args.now else cal.now_ist()
    if now.tzinfo is None:
        now = now.replace(tzinfo=cal.IST)
    today = now.astimezone(cal.IST).date()
    out = Path(args.out)
    holidays = cal.load_holidays(ROOT / "live" / store.HOLIDAY_FILE)
    forecast_path = out / store.FORECAST_FILE

    if not cal.is_trading_day(today, holidays):
        reason = holidays.get(today.isoformat(), "Weekend")
        nxt = cal.next_trading_day(today, holidays)
        store.write_json(live.closed_notice(today, nxt, reason, now), forecast_path)
        print(f"{today}: market closed ({reason}). Next session {nxt}.")
        return 0

    existing = store.read_json(forecast_path, {})
    if (not args.force and existing.get("status") == "ok" and existing.get("target_date") == str(today)
            and existing.get("on_time")):
        print(f"{today}: an on-time forecast already exists (made {existing.get('generated_at')}). Nothing to do.")
        return 0

    try:
        px, stock_px = load_prices(args, period="3y")
    except Exception as exc:                                     # noqa: BLE001
        msg = f"Price download failed: {exc}"
        print(msg, file=sys.stderr)
        if existing.get("target_date") != str(today) or existing.get("status") != "ok":
            store.write_json(live.failure_notice(today, "Price data could not be downloaded this morning.", now),
                             forecast_path)
        return 1

    model = M.load_json(ROOT / "models" / "morning" / "model.json")
    forecast = live.make_forecast(px, stock_px, today, model, now,
                                  expected_based_on=cal.prev_trading_day(today, holidays))
    store.write_json(forecast, forecast_path)

    history = live.merge_history(store.read_history(out / store.HISTORY_FILE), live.history_rows(forecast))
    if args.prices:   # replay: hide everything after "now"
        stock_px = {k: d.loc[:str(today)] for k, d in stock_px.items()}
    history = live.grade(history, stock_px, now)
    store.write_history(history, out / store.HISTORY_FILE)
    store.write_json(live.scorecard(history, now), out / store.SCORECARD_FILE)

    s = forecast.get("summary", {})
    print(f"{today}: forecast {forecast['status']} (based on close of {forecast['based_on']}, "
          f"on time: {forecast['on_time']}). Higher {s.get('higher')}, lower {s.get('lower')}, "
          f"unavailable {s.get('unavailable')}.")
    for c in forecast["calls"]:
        if c["available"]:
            print(f"  {c['ticker']:<11} {c['call']:<6} p(up)={c['prob_up']:.3f}  {c['drivers'][0]['label']}: {c['drivers'][0]['detail']}")
        else:
            print(f"  {c['ticker']:<11} unavailable: {c['reason']}")
    for n in forecast["notes"]:
        print("  note:", n)
    return 0 if forecast["status"] == "ok" else 1


if __name__ == "__main__":
    sys.exit(main())
