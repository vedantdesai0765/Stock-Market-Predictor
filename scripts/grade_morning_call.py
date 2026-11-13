"""
The grading job: check past forecasts against what really happened.

Runs automatically on GitHub Actions at about 9:45 am (opening prices) and 4:30 pm (closing prices)
on weekdays. It downloads the latest prices, fills in the result of every forecast that can now be
checked, and rebuilds live/scorecard.json. Only forecasts made before 09:15 count in the scorecard.

Run by hand:   python scripts/grade_morning_call.py
"""
import argparse
import datetime as dt
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from morning import sessions as cal, data, live, panel as P, store


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--prices", help="archived long-format price file instead of downloading (replay)")
    ap.add_argument("--now", help="pretend it is this time, ISO format with offset (replay)")
    ap.add_argument("--out", default=str(ROOT / "live"), help="folder holding the live files")
    args = ap.parse_args()

    now = dt.datetime.fromisoformat(args.now) if args.now else cal.now_ist()
    if now.tzinfo is None:
        now = now.replace(tzinfo=cal.IST)
    out = Path(args.out)
    history = store.read_history(out / store.HISTORY_FILE)
    if history.empty:
        store.write_json(live.scorecard(history, now), out / store.SCORECARD_FILE)
        print("No forecasts to grade yet.")
        return 0

    try:
        if args.prices:
            allpx = data.load_long_file(args.prices)
            today = str(now.astimezone(cal.IST).date())
            stock_px = {k: (allpx[t] if t in allpx else data.load_raw_csv(ROOT / "data" / "raw" / f"{k}_raw.csv")).loc[:today]
                        for k, t in P.STOCKS.items()}
        else:
            got = data.download(list(P.STOCKS.values()), period="3mo")
            stock_px = {k: got[t] for k, t in P.STOCKS.items()}
    except Exception as exc:                                     # noqa: BLE001
        print(f"Price download failed: {exc}", file=sys.stderr)
        return 1

    before = history.actual.notna().sum(), history.close_actual.notna().sum()
    history = live.grade(history, stock_px, now)
    store.write_history(history, out / store.HISTORY_FILE)
    card = live.scorecard(history, now)
    store.write_json(card, out / store.SCORECARD_FILE)
    print(f"Graded {history.actual.notna().sum() - before[0]} opening calls and "
          f"{history.close_actual.notna().sum() - before[1]} closing calls.")
    o = card["open"]
    if o["n"]:
        print(f"Live record since {card['since']}: {o['correct']}/{o['n']} opening calls correct "
              f"({o['accuracy']:.1%}); simple guess {o['baseline_correct']}/{o['n']} ({o['baseline_accuracy']:.1%}).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
