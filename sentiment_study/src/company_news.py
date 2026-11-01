"""
Company-targeted news: find Times of India headlines that mention each of the 10 project stocks.
Case-sensitive alias regexes; each match keeps the matched surface form as the target entity,
so the entity-level model scores sentiment *towards that company* (not the headline overall).
"""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.dirname(__file__))
from paths import RAW, INTERIM, PROCESSED, REPORTS, OOF, FINBERT, ROOT, first, price_file

import re, sys, pandas as pd

ALIASES = {
    "RELIANCE": r"\bReliance(?! (?:on|upon|of|in|Anil|Communications?|Comm|Power|Capital|Infra(?:structure)?|Energy|Natural|Big|Entertainment|Home|General|Life|Mutual|MF|Money|Securities|Health|ADA|ADAG|Anil|Defence|Naval|Telecom|Globalcom|Broadcast|Media|Nippon|Mediaworks|Group's Anil))\b|\bRIL\b|\bMukesh Ambani\b|\bJio\b",
    "TCS": r"\bTCS\b|\bTata Consultancy\b",
    "INFY": r"\bInfosys\b|\bInfy\b",
    "HDFCBANK": r"\bHDFC Bank\b",
    "SBI": r"\bSBI\b(?! (?:Life|Cards?|MF|Mutual|Caps?|Capital|General|Funds?|Mutual Fund))|\bState Bank of India\b",
    "ITC": r"\bITC\b(?! (?:Grand|Maurya|Kakatiya|Gardenia|Windsor|Sonar|Mughal|Rajputana|Maratha|Kohenur|Royal))",
    "ADANIENT": r"\bAdani Enterprises\b|\bAdani Group\b|\bGautam Adani\b|\bAdani\b(?! (?:Ports?|Power|Green|Transmission|Gas|Total|Wilmar|Energy|Electricity|Pharma))",
    "ADANIGREEN": r"\bAdani Green\b|\bAdani Group\b|\bGautam Adani\b",
    "VEDL": r"(?<!per the )\bVedanta(?! (?:Society|philosophy|Kesari|Hospital|Cancer|Institute))\b|\bSesa Goa\b|\bSesa Sterlite\b|\bSterlite Industries\b|\bAnil Agarwal\b",
    "ETERNAL": r"\bZomato\b|\bBlinkit\b|\bEternal Ltd\b",
}
START = {"ADANIGREEN": "2018-01-01", "ETERNAL": "2021-01-01"}   # pre-listing mentions are irrelevant


def match_toi(toi):
    rows = []
    for stk, pat in ALIASES.items():
        rx = re.compile(pat)
        m = toi.text.str.extract(f"({pat})", expand=False)
        hit = toi[m.notna()].copy()
        hit["entity"] = m[m.notna()].values
        hit["stock"] = stk
        if stk in START:
            hit = hit[hit.date >= START[stk]]
        rows.append(hit[["stock", "date", "category", "text", "entity"]])
    out = pd.concat(rows, ignore_index=True).drop_duplicates(["stock", "date", "text"])
    # how many of the 10 stocks each headline names (multi-company headlines are where targeting matters)
    out["n_companies"] = out.groupby(["date", "text"]).stock.transform("nunique")
    return out


if __name__ == "__main__":
    toi = pd.read_parquet(first(INTERIM / "toi_all.parquet"), columns=["date", "category", "text"])
    out = match_toi(toi)
    out.to_parquet(INTERIM / "toi_company_matches.parquet")
    print(out.groupby("stock").agg(n=("text", "size"), first=("date", "min"), last=("date", "max"),
                                    days=("date", "nunique")))
    for s in ALIASES:
        print("\n==", s); print(out[out.stock == s].sample(6, random_state=1)[["entity", "text"]].to_string()[:900])
