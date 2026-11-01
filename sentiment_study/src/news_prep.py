"""
Build the two headline corpora used in the study.

ToI  : Times of India headlines (Harvard Dataverse DVN/DPQMQH, CC0),
       2001-01-02 .. 2022-03-31, DATE-ONLY timestamps (IST calendar day).
NYT  : New York Times headlines + abstracts (NYT Archive API via HF
       bguzzo2k/nyt_100y_news_headlines), 1990..2025, minute timestamps (UTC).

Both are reduced to a finance/market-relevant subset with a transparent
category + keyword rule so the paper can state exactly what was used.
"""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.dirname(__file__))
from paths import RAW, INTERIM, PROCESSED, REPORTS, OOF, FINBERT, first, price_file
import glob, re, sys
import pandas as pd

UP = str(RAW)
OUT = str(INTERIM)

TOI_KW = re.compile(
    r"\b(sensex|nifty|bse|nse|stock|stocks|shares?|equit(y|ies)|market|markets|"
    r"investor|investors|fii|fiis|fpi|ipo|rbi|repo|interest rate|rate cut|rate hike|"
    r"inflation|wpi|cpi|gdp|growth|economy|economic|recession|slowdown|rupee|forex|"
    r"crude|oil price|gold|profit|profits|loss|losses|earnings|revenue|quarter|q[1-4]|"
    r"bank|banks|banking|npa|budget|fiscal|deficit|tax|gst|exports?|imports?|"
    r"industrial output|iip|manufacturing|sebi|mutual fund|bonds?|yields?|"
    r"rally|rallies|crash|slump|plunge|surge|tumble|soar)\b", re.I)

NYT_KW = re.compile(
    r"\b(stocks?|shares|wall street|dow|s\.?&\s?p|nasdaq|market|markets|investors?|"
    r"fed|federal reserve|interest rates?|rate cut|rate hike|treasur(y|ies)|bonds?|yields?|"
    r"inflation|deflation|economy|economic|economists?|recession|gdp|growth|jobs|"
    r"unemployment|payrolls?|jobless|layoffs?|earnings|profits?|revenue|quarter|"
    r"banks?|banking|credit|debt|deficit|oil|crude|opec|dollar|currency|"
    r"ipo|merger|acquisition|bankrupt\w*|default|bailout|stimulus|tariffs?|trade war|"
    r"rally|rallies|crash|slump|plunge|surge|tumble|soar|sell-?off|bull|bear)\b", re.I)


def toi():
    n = pd.read_csv(f"{UP}/india-news-headlines.csv", dtype={"publish_date": str})
    n["date"] = pd.to_datetime(n.publish_date, format="%Y%m%d")
    n = n.rename(columns={"headline_text": "text", "headline_category": "category"})
    n = n.dropna(subset=["text"])
    cat = n.category.fillna("")
    biz = cat.str.contains(r"^business|markets|econom|budget|india-business", regex=True)
    kw = n.text.str.contains(TOI_KW)
    n["is_business_cat"] = biz
    n["is_market"] = biz | kw
    n = n.drop_duplicates(subset=["date", "text"])
    n[["date", "category", "text", "is_business_cat", "is_market"]].to_parquet(f"{OUT}/toi_all.parquet")
    m = n[n.is_market]
    print("ToI all", len(n), "market subset", len(m))
    print(m.groupby(m.date.dt.year).size().to_string())


def nyt():
    fs = sorted(glob.glob(f"{UP}/**/nyt_*.parquet", recursive=True))
    parts = []
    for f in fs:
        d = pd.read_parquet(f).reset_index(drop=True)
        d["headline"] = d.headline.fillna("").astype(str)
        d["abstract"] = d.abstract.fillna("").astype(str)
        txt = d.headline + " " + d.abstract
        keep = txt.str.contains(NYT_KW)
        d = d[keep & (d.headline.str.len() > 3)]
        parts.append(d)
    d = pd.concat(parts, ignore_index=True)
    d = d.drop_duplicates(subset=["headline", "abstract"])
    d["ts_utc"] = d.date
    d["ts_et"] = d.date.dt.tz_convert("America/New_York")
    # print-edition items are stamped 00:00 local -> keep them, they are
    # available before the 09:30 open of that day.
    d = d[["ts_utc", "ts_et", "headline", "abstract"]].reset_index(drop=True)
    d.to_parquet(f"{OUT}/nyt_fin.parquet")
    print("NYT finance subset", len(d))
    print(d.groupby(d.ts_et.dt.year).size().to_string())


if __name__ == "__main__":
    which = sys.argv[1:] or ["toi", "nyt"]
    if "toi" in which: toi()
    if "nyt" in which: nyt()
