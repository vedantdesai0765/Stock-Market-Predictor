# Sentiment study: does news sentiment improve stock-movement prediction?

A leakage-controlled, 20+ year comparison of technical-only vs technical + news-sentiment
models (Logistic Regression, Random Forest, Gradient Boosting, XGBoost, LightGBM, LSTM, GRU)
on Indian (Sensex + Times of India) and US (S&P 500 + NYT; Nasdaq-100 + FNSPID) markets.

## Data (all free)

| File (data/raw/) | Source | Coverage |
|---|---|---|
| yahoo_daily_prices_long.csv | Yahoo Finance (yfinance) | Sensex 1997-, S&P 500 1970-, NSE large caps 1996-/2002- |
| india-news-headlines.csv | Harvard Dataverse doi:10.7910/DVN/DPQMQH (CC0) | 2001-01-02 .. 2022-03-31, date only |
| nyt/nyt_YYYY.parquet | HF bguzzo2k/nyt_100y_news_headlines (NYT Archive API) | 1990-2025, minute timestamps |
| fnspid_ndx100_slim.parquet | HF benstaf/FNSPID-filtered-nasdaq-100 (columns Date, Article_title, Stock_symbol) | 2009-2023 |
| finbert/ | HF ProsusAI/finbert | model weights |

`python src/download_data.py` fetches prices, ToI, NYT and FinBERT. FNSPID: download the two parquet
files from the HF page and keep only the Date / Article_title / Stock_symbol / Publisher columns.
Paths are set at the top of each script (`UP`, `PRICE_FILE`, `MD`, ...); point them at data/raw.

## Pipeline

```
python src/news_prep.py          # market/finance subsets of ToI and NYT
python src/score_lexicon.py      # VADER + Loughran-McDonald (minutes)
python src/score_finbert.py      # FinBERT on ToI (+ score_finbert_nyt.py for NYT market subset)
python src/fnspid_prep.py        # company news scoring
python src/run_all.py main       # India grid      (T, T+S, T+G, T+G+S, S  x 5 models x 3 targets)
python src/run_all.py us         # US grid
python src/run_all.py robust     # strict 16:00 cut-off, placebo, lexicon-only
python src/run_all.py vol        # realised-volatility regression
python src/run_all.py stocks     # individual stocks
python src/gap_decomp.py         # overnight gap vs open-to-close
python src/panel.py              # Nasdaq-100 company-level panel
python src/run_dl.py IN|US       # LSTM / GRU
python src/leakage_demo.py       # how 77-99% "accuracy" arises
python src/final_tables.py       # paired McNemar + block-bootstrap -> reports/comparison_all.csv
python src/seventy.py            # routes to ~70%: Sensex gap, volatility direction
python src/method1.py '{"tag":"M1S"}'                                         # Method 1, Sensex 2001-2022
python src/method1.py '{"tag":"M1N","index":"^NSEI","start":"2007-09-17","end":"2025-12-30","use_toi":false,"fty":2012}'
python src/method4.py '{"prep":"data/raw/nse_announcements_2008_2025.json"}'  # score NSE filings (FinBERT)
python src/method4.py            # Method 4 event-day panel
```

### Methods 1 and 4 (added 31 Oct 2026)

* Method 1 - opening gap at 09:00 IST (US close, VIX, Nikkei/KOSPI/Taiex/Hang Seng opens, overnight ToI+NYT sentiment):
  Sensex 2009-2022 accuracy 79.3-80.1% (AUC 0.81) without sentiment, 78.8-80.4% with it; Nifty 2012-2025 76.1-77.9% vs 75.6-77.8%.
  Results: reports/method1_summary.csv.
* Method 4 - 78,467 NSE corporate announcements (40 Nifty stocks, 2008-2025, exact dissemination times),
  overnight filing events: opening-gap accuracy 69.5-69.9% without sentiment, 69.2-69.7% with it (majority 64.9%);
  results days AUC +1.0 to +1.2 points with sentiment for two models. Results: reports/method4_summary.csv.
* Data: data/raw/yahoo_prices_v2.parquet (adds Asian indices and 20 more NSE stocks),
  data/raw/nse_announcements_2008_2025.json, data/processed/nse_announcements_scored.parquet,
  predictions in reports/oof_all_4.parquet.

### Company-targeted sentiment (added 31 Oct 2026)

Each of the 10 project stocks is scored on news about itself, by a model that reads sentiment
*towards the named company* rather than the tone of the whole headline.

```
python src/train_entity_model.py 2   # FinBERT -> entity-level classifier on SEntFiN 1.0 (data/raw/sentfin, MIT); ~30 min, 2 CPU cores
python src/company_news.py           # ToI headlines naming each stock (needs data/interim/toi_all.parquet from news_prep.py)
python src/score_entity.py           # score (company, headline) and (company, filing) pairs; full precision, not int8
python src/company_study.py '{"study":"TOI"}'   # 2001-2022 headlines, walk-forward 2007-2022
python src/company_study.py '{"study":"NSE"}'   # 2008-2025 filings, walk-forward 2013-2025
python src/company_tables.py         # paired McNemar + block bootstrap, all days and company-news days
python src/export_company_sentiment.py          # writes ../data/processed/<KEY>_company_sentiment.csv
```

* Scorer quality on the SEntFiN test split (1,440 pairs): entity-level 87.7% accuracy / 0.878 macro-F1,
  plain FinBERT 68.5% / 0.687; on headlines naming 2+ companies 85.3% vs 61.6% (reports/entity_model_eval.json).
* Feature sets: B = stock technicals + market (Nifty, S&P 500 overnight, VIX, Asian opens); Cent = targeted
  company sentiment; Cfb = the same headlines scored by plain FinBERT; Cplc = Cent shifted one year (placebo);
  M = market-wide ToI sentiment.
* Results (reports/company_sentiment_comparison.csv): adding Cent moves accuracy -0.2 to +0.2 points
  (opening gap ~70-72%, next-day close ~56-58%); targeted vs plain scoring -0.4 to +0.1 (all days); the placebo moves it by
  the same amount. Per stock: reports/company_sentiment_per_stock.csv.
* Shipped: data/processed/toi_company_scored.parquet, nse_company_scored.parquet (scored pairs),
  reports/oof_company_all.parquet (all out-of-fold predictions; company_tables.py unpacks it).
  The fine-tuned model (440 MB) is not shipped; train_entity_model.py rebuilds it.

### Why did it move? (explanatory analysis, added 1 Nov 2026)

```
python src/explain_moves.py   # writes ../data/processed/news/<STOCK>_news_items.csv and reports/explain/*.csv
```

News is matched to the trading day it can explain (headline dated d -> first trading day >= d; filing before
15:30 IST -> that day, otherwise the next). Every statistic is also computed with price-report headlines
("Sensex down 200 points") removed. Outputs: lead_lag.csv (correlation k days before/after), quintiles.csv,
event_study.csv (cumulative abnormal return around strong news days), big_moves_coverage.csv,
big_moves_top.csv, same_day_accuracy.csv, summary.csv. With price reports removed: Sensex same-day
Spearman 0.056 (p < 0.001), next day 0.007; 10 stocks pooled 0.032 (p = 0.004). The dashboard's
"Why did it move?" view reads these files.

## Information sets (row t -> target r(t+1) = log close(t+1)/close(t))

* ToI: headlines dated [date(t), date(t+1)) -> known by midnight IST, before the 09:15 open.
* NYT 'preopen': stamped (09:00 ET t, 09:00 ET t+1]; 'close': (16:00 ET t-1, 16:00 ET t].
* FNSPID: articles dated [date(t), date(t+1)).
* `up_today` pairs the same news with r(t) and lagged price features: LEAKAGE DEMONSTRATION ONLY.

## Feature sets
T = technical (20-21), G = global/market (overnight S&P 500 return for India, VIX), S = sentiment (23).

## Results
See reports/comparison_all.csv (every with/without pair, with McNemar p, bootstrap CIs for
delta-accuracy and delta-AUC), reports/leakage_demo.csv, reports/vol_regression.csv, and
reports/oof_all_1..3.parquet (every out-of-fold prediction; `analyze.load()` reads them when
reports/oof/ is empty, so every table can be recomputed without retraining).

## Shipped intermediate data (skip the 2-3 hours of FinBERT scoring)

data/processed/ holds every scored headline (FinBERT, VADER, Loughran-McDonald) and the daily
sentiment tables; data/raw/ holds the price files as parquet. With only these, `experiment.py`,
`panel.py`, `seq_models.py`, `leakage_demo.py` and `final_tables.py` run as-is.

## Headline numbers (see the plan document for the full write-up)

* India next-day accuracy: T 51.5-53.6%, T+G 57.4-58.7%, T+G+S 57.2-58.9% (always-up 53.1%).
* US next-day: 51.7-53.8% for every set (always-up 53.7%); sentiment AUC +1.5 to +2.3 pts with news to 09:00 ET, ~0 with news to 16:00 ET.
* Same-day (look-ahead) accuracy with sentiment: India 64-66%, US 58-59% (+4 to +12 pts, p < 0.001).
* Leaky setups on the same data: 77% (same-day headline sign), 86% (moving-average direction), 99% (price level).
