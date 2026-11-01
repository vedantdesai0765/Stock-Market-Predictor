"""Full experiment grid. Results -> reports/oof/*.parquet, log -> reports/run_all.log"""
import sys, os, time
sys.path.insert(0, os.path.dirname(__file__))
import experiment as E

ALL = ["LogReg", "RandomForest", "GradBoost", "XGBoost", "LightGBM"]
SETS = ["T", "T+S", "T+G", "T+G+S", "S"]
stage = sys.argv[1] if len(sys.argv) > 1 else "main"
t0 = time.time()
if stage == "main":
    E.run("IN", ["up_next", "big_next", "up_today"], SETS, ALL)
if stage == "us":
    E.run("US", ["up_next", "big_next", "up_today"], ["T", "T+S", "T+G", "T+G+S"], ALL, nyt_mode="preopen")
    E.run("US", ["up_next", "big_next", "up_today"], ["S"], ["LogReg", "LightGBM"], nyt_mode="preopen")
if stage == "robust":
    E.run("US", ["up_next", "big_next"], ["T", "T+S", "T+G", "T+G+S"], ["LogReg", "LightGBM"], nyt_mode="close", tag="_close")
    E.run("IN", ["up_next", "big_next"], ["T+G", "T+G+S"], ["LogReg", "LightGBM"], tag="_placebo", placebo=True)
    E.run("US", ["up_next", "big_next"], ["T+G", "T+G+S"], ["LogReg", "LightGBM"], tag="_placebo", placebo=True)
    E.run("IN", ["up_next", "big_next"], ["T", "T+S", "T+G", "T+G+S"], ["LogReg", "LightGBM"], has_fb=False, tag="_lex")
    E.run("US", ["up_next", "big_next"], ["T", "T+S", "T+G", "T+G+S"], ["LogReg", "LightGBM"], has_fb=False, tag="_lex")
if stage == "vol":
    E.run("IN", ["rv5_next"], ["T", "T+S", "T+G", "T+G+S"], ["Ridge", "LightGBM"], kind="reg")
    E.run("US", ["rv5_next"], ["T", "T+S", "T+G", "T+G+S"], ["Ridge", "LightGBM"], kind="reg")
if stage == "stocks":
    for tk in ["RELIANCE.NS", "INFY.NS", "HDFCBANK.NS", "SBIN.NS", "ITC.NS", "TCS.NS"]:
        if os.path.exists(f"{E.OUT}/IN_{tk.split('.')[0]}__big_next__LightGBM__T+G+S.parquet"): continue
        E.run("IN", ["up_next", "big_next"], ["T+G", "T+G+S"], ["LogReg", "LightGBM"], ticker=tk, tag="_" + tk.split(".")[0])
    for tk in ["AAPL", "MSFT", "JPM", "XOM", "KO", "GE"]:
        E.run("US", ["up_next", "big_next"], ["T+G", "T+G+S"], ["LogReg", "LightGBM"], ticker=tk, tag="_" + tk)
print("STAGE DONE", stage, round(time.time() - t0), "s", flush=True)
