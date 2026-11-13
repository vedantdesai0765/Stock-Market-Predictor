"""
Morning Call: the live version of the overnight (next-morning opening price) model.

    features.py   feature construction, identical to the research study
                  (sentiment_study/src/company_study.py, feature set "B")
    data.py       price loading (Yahoo Finance download, or archived files)
    model.py      training, walk-forward testing, JSON model, explanations
    sessions.py   NSE trading-day helpers

Everything here uses only information that is public before the NSE opens at 09:15 IST.
"""
