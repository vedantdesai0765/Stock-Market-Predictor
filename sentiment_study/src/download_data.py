"""
Reproduce every raw input of the study on your own machine.

    pip install yfinance requests huggingface_hub
    python src/download_data.py            # ~1.1 GB total

Writes into data/raw/:
  yahoo_daily_prices_long.csv      Yahoo Finance daily OHLCV (indices + large caps)
  india-news-headlines.csv         Times of India headlines 2001-2022 (Harvard Dataverse, CC0)
  nyt/nyt_YYYY.parquet             NYT headlines+abstracts with timestamps (HF mirror of NYT Archive API)
  finbert/                         ProsusAI/finbert weights
"""
import os, pathlib, requests
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"; RAW.mkdir(parents=True, exist_ok=True)

TICKERS = ['^BSESN', '^NSEI', '^NSEBANK', '^INDIAVIX', '^GSPC', '^DJI', '^IXIC', '^VIX',
           'RELIANCE.NS', 'TCS.NS', 'INFY.NS', 'HDFCBANK.NS', 'SBIN.NS', 'ITC.NS', 'ICICIBANK.NS',
           'HINDUNILVR.NS', 'LT.NS', 'BHARTIARTL.NS', 'AXISBANK.NS', 'WIPRO.NS', 'BAJFINANCE.NS',
           'MARUTI.NS', 'SUNPHARMA.NS', 'VEDL.NS', 'ADANIENT.NS', 'TATASTEEL.NS', 'ONGC.NS', 'KOTAKBANK.NS',
           'AAPL', 'MSFT', 'JPM', 'XOM', 'GE', 'KO', 'WMT', 'JNJ', 'PG', 'IBM', 'INTC', 'BAC']


def prices():
    import yfinance as yf
    out = []
    for t in TICKERS:
        d = yf.download(t, period="max", interval="1d", auto_adjust=False, progress=False)
        if isinstance(d.columns, pd.MultiIndex): d.columns = d.columns.get_level_values(0)
        d = d.rename(columns=str.lower).rename(columns={"adj close": "adjclose"}).reset_index()
        d = d.rename(columns={"Date": "date"}); d["ticker"] = t
        out.append(d[["ticker", "date", "open", "high", "low", "close", "adjclose", "volume"]])
        print(t, len(d), d.date.min().date())
    pd.concat(out).to_csv(RAW / "yahoo_daily_prices_long.csv", index=False)


def toi():
    url = "https://dataverse.harvard.edu/api/access/datafile/6175512"   # doi:10.7910/DVN/DPQMQH
    with requests.get(url, stream=True, timeout=600) as r, open(RAW / "india-news-headlines.csv", "wb") as f:
        r.raise_for_status()
        for chunk in r.iter_content(1 << 20): f.write(chunk)


def nyt(y0=1990, y1=2025):
    from huggingface_hub import hf_hub_download
    (RAW / "nyt").mkdir(exist_ok=True)
    for y in range(y0, y1 + 1):
        hf_hub_download("bguzzo2k/nyt_100y_news_headlines", f"yearly_parquet/nyt_{y}.parquet",
                        repo_type="dataset", local_dir=RAW / "nyt")


def finbert():
    from huggingface_hub import snapshot_download
    snapshot_download("ProsusAI/finbert", local_dir=RAW / "finbert",
                      allow_patterns=["*.json", "vocab.txt", "model.safetensors"])


V2_EXTRA = ['^N225', '^KS11', '^TWII', '^HSI', 'HCLTECH.NS', 'TECHM.NS', 'ASIANPAINT.NS', 'ULTRACEMCO.NS', 'NTPC.NS',
            'POWERGRID.NS', 'M&M.NS', 'TITAN.NS', 'NESTLEIND.NS', 'DRREDDY.NS', 'CIPLA.NS', 'BPCL.NS', 'COALINDIA.NS',
            'HINDALCO.NS', 'JSWSTEEL.NS', 'GRASIM.NS', 'EICHERMOT.NS', 'HEROMOTOCO.NS', 'BAJAJ-AUTO.NS', 'INDUSINDBK.NS']


def prices_v2():
    """prices for Method 1 (Asian opens) and Method 4 (40 NSE stocks) -> data/raw/yahoo_prices_v2.csv"""
    global TICKERS
    keep = TICKERS
    TICKERS = [t for t in keep if t.endswith(".NS") or t in ("^BSESN", "^NSEI", "^GSPC", "^VIX")] + V2_EXTRA
    prices()
    os.replace(RAW / "yahoo_daily_prices_long.csv", RAW / "yahoo_prices_v2.csv")
    TICKERS = keep


def nse_announcements(symbols, y0=2008, y1=2025):
    """NSE corporate announcements (exact dissemination times). NSE blocks plain scripts at times;
    if this returns 403, open nseindia.com in a browser first or use the shipped JSON."""
    import json, time
    s = requests.Session()
    s.headers.update({"User-Agent": "Mozilla/5.0", "Accept": "application/json", "Referer": "https://www.nseindia.com/"})
    s.get("https://www.nseindia.com", timeout=30)
    rows = []
    for sym in symbols:
        for y in range(y0, y1 + 1):
            r = s.get("https://www.nseindia.com/api/corporate-announcements",
                      params={"index": "equities", "symbol": sym, "from_date": f"01-01-{y}", "to_date": f"31-12-{y}"}, timeout=60)
            r.raise_for_status()
            for x in r.json():
                rows.append({"symbol": sym, "an_dt": x.get("an_dt"), "dis": x.get("exchdisstime"), "desc": x.get("desc"),
                             "text": (x.get("attchmntText") or "")[:700], "sm_name": x.get("sm_name")})
            time.sleep(0.5)
    json.dump(rows, open(RAW / "nse_announcements_2008_2025.json", "w"))


if __name__ == "__main__":
    prices(); prices_v2(); toi(); nyt(); finbert()
