# src/validate.py
"""
Project health check.

Run this after any change — adding a stock, upgrading a library, cloning
fresh — to confirm the pipeline is intact. It answers one question:

    "If I add a stock right now, will everything work?"

Each check is independent and reports PASS / WARN / FAIL with an
actionable message. Nothing here modifies data or models.
"""

from __future__ import annotations

import importlib
import sys
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


PASS, WARN, FAIL = "PASS", "WARN", "FAIL"


@dataclass
class Check:
    name: str
    status: str
    detail: str

    @property
    def symbol(self) -> str:
        return {PASS: "[ok]", WARN: "[!!]", FAIL: "[XX]"}[self.status]


# ---------------------------------------------------------------------------
# Individual checks
# ---------------------------------------------------------------------------

def check_dependencies() -> list[Check]:
    """Every import the pipeline needs, with the optional ones marked."""
    required = {
        "numpy": "core", "pandas": "core", "sklearn": "core",
        "joblib": "core", "ta": "indicators", "matplotlib": "plots",
    }
    optional = {
        "yfinance": "downloading new price data",
        "torch": "LSTM / GRU sequence models",
        "streamlit": "the dashboard",
        "plotly": "the dashboard",
        "transformers": "FinBERT sentiment",
        "vaderSentiment": "VADER sentiment",
        "requests": "news collection",
    }

    checks = []
    for mod, why in required.items():
        try:
            importlib.import_module(mod)
            checks.append(Check(f"import {mod}", PASS, why))
        except ImportError:
            checks.append(Check(f"import {mod}", FAIL,
                                f"required for {why}. pip install {mod}"))

    for mod, why in optional.items():
        try:
            importlib.import_module(mod)
            checks.append(Check(f"import {mod}", PASS, why))
        except ImportError:
            checks.append(Check(f"import {mod}", WARN,
                                f"missing — {why} unavailable"))

    return checks


def check_project_modules() -> list[Check]:
    """All first-party modules import cleanly."""
    modules = [
        "config", "dataio", "features", "dataset", "stationary",
        "targets", "evaluation", "abstention", "modeling",
        "pipeline", "collect", "ichimoku", "bollinger",
    ]
    checks = []
    for mod in modules:
        try:
            importlib.import_module(mod)
            checks.append(Check(f"src/{mod}.py", PASS, "imports cleanly"))
        except Exception as exc:
            checks.append(Check(f"src/{mod}.py", FAIL,
                                f"{type(exc).__name__}: {exc}"))
    return checks


def check_directories() -> list[Check]:
    import config
    checks = []
    for d in config.ALL_DIRS:
        if d.exists():
            checks.append(Check(f"{d.name}/", PASS, "exists"))
        else:
            checks.append(Check(f"{d.name}/", FAIL,
                                "missing — run config.ensure_dirs()"))
    return checks


def check_registry() -> list[Check]:
    """Every registered stock has a readable raw file with enough history."""
    import config
    from collect import verify_raw_file

    checks = []
    keys = config.list_stocks()

    if not keys:
        return [Check("registry", FAIL, "config.STOCKS is empty")]

    checks.append(Check("registry", PASS,
                        f"{len(keys)} stock(s): {', '.join(keys)}"))

    for key in keys:
        res = verify_raw_file(key, verbose=False)
        if not res.get("ok"):
            checks.append(Check(f"raw data: {key}", FAIL,
                                str(res.get("problem"))))
        elif res.get("warning"):
            checks.append(Check(f"raw data: {key}", WARN,
                                f"{res['rows']} rows ({res['years']}y) — "
                                f"short history, horizon will fall back"))
        else:
            checks.append(Check(f"raw data: {key}", PASS,
                                f"{res['rows']} rows ({res['years']}y)"))
    return checks


def check_datasets(quick: bool = True) -> list[Check]:
    """Each stock's dataset builds without error and passes the audit."""
    import config
    from dataset import build_dataset
    from features import audit_dataset
    from stationary import add_stationary_features, get_stationary_features

    checks = []
    for key in config.list_stocks():
        try:
            df = build_dataset(key, with_sentiment=False,
                               save=False, verbose=False)
            if not quick:
                df = add_stationary_features(df).dropna().reset_index(drop=True)
                cols = get_stationary_features(df)
                audit = audit_dataset(df, cols)
                if audit.get("leaky_features"):
                    checks.append(Check(f"dataset: {key}", FAIL,
                                        f"leaky: {audit['leaky_features']}"))
                    continue
                if audit.get("has_inf"):
                    checks.append(Check(f"dataset: {key}", FAIL,
                                        "contains infinities"))
                    continue
            checks.append(Check(f"dataset: {key}", PASS,
                                f"{len(df)} rows built"))
        except Exception as exc:
            checks.append(Check(f"dataset: {key}", FAIL,
                                f"{type(exc).__name__}: {exc}"))
    return checks


def check_artifacts() -> list[Check]:
    """Trained models exist, load, and carry honest metadata."""
    import config
    from pipeline import artifacts_exist, load_artifacts

    checks = []
    for key in config.list_stocks():
        if not artifacts_exist(key):
            checks.append(Check(f"model: {key}", WARN,
                                "not trained — run notebook 04"))
            continue
        try:
            art = load_artifacts(key)
            meta = art["metadata"]
            edge = meta.get("edge", 0)
            perm = (meta.get("permutation") or {}).get("verdict", "n/a")

            if perm == "LEAKAGE":
                checks.append(Check(f"model: {key}", FAIL,
                                    "permutation test indicates LEAKAGE"))
            elif edge <= 0:
                checks.append(Check(f"model: {key}", WARN,
                                    f"loads, but edge {edge:+.4f} "
                                    f"({perm}) — no demonstrated skill"))
            else:
                checks.append(Check(f"model: {key}", PASS,
                                    f"edge {edge:+.4f}, {perm}"))
        except Exception as exc:
            checks.append(Check(f"model: {key}", FAIL,
                                f"{type(exc).__name__}: {exc}"))
    return checks


def check_secrets() -> list[Check]:
    """No API keys committed anywhere in the tree."""
    import config

    checks = []
    suspicious = []

    for path in config.PROJECT_ROOT.rglob("*"):
        if not path.is_file():
            continue
        if any(p in path.parts for p in
               (".git", "__pycache__", ".ipynb_checkpoints", "data")):
            continue
        if path.suffix not in (".py", ".ipynb", ".md", ".txt", ".json"):
            continue
        if path.name == ".env":
            continue
        try:
            text = path.read_text(errors="ignore")
        except Exception:
            continue
        for marker in ("apiKey", "API_KEY", "api_key"):
            if marker in text:
                # A 32-char hex string next to an api key marker is a real key
                import re
                if re.search(r"[\"'][0-9a-f]{32}[\"']", text):
                    suspicious.append(str(path.relative_to(
                        config.PROJECT_ROOT)))
                break

    if suspicious:
        checks.append(Check("committed secrets", FAIL,
                            f"possible API key in: {', '.join(sorted(set(suspicious))[:3])}"))
    else:
        checks.append(Check("committed secrets", PASS,
                            "no hardcoded keys detected"))

    gitignore = config.PROJECT_ROOT / ".gitignore"
    if gitignore.exists() and ".env" in gitignore.read_text():
        checks.append(Check(".env ignored", PASS, ".env is in .gitignore"))
    else:
        checks.append(Check(".env ignored", WARN,
                            "add '.env' to .gitignore"))

    return checks


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run_all_checks(deep: bool = False) -> pd.DataFrame:
    """
    Run every check and print a report.

    deep=True also builds stationary features and runs the leakage audit,
    which is slower but more thorough.
    """
    groups = [
        ("Dependencies", check_dependencies),
        ("Project modules", check_project_modules),
        ("Directories", check_directories),
        ("Stock registry", check_registry),
        ("Datasets", lambda: check_datasets(quick=not deep)),
        ("Trained artifacts", check_artifacts),
        ("Secrets", check_secrets),
    ]

    rows = []
    print("=" * 68)
    print("  PROJECT HEALTH CHECK")
    print("=" * 68)

    for title, fn in groups:
        print(f"\n{title}")
        print("-" * len(title))
        try:
            results = fn()
        except Exception as exc:
            results = [Check(title, FAIL, f"{type(exc).__name__}: {exc}")]
        for c in results:
            print(f"  {c.symbol} {c.name:24s} {c.detail}")
            rows.append({"group": title, "check": c.name,
                         "status": c.status, "detail": c.detail})

    table = pd.DataFrame(rows)
    n_fail = int((table["status"] == FAIL).sum())
    n_warn = int((table["status"] == WARN).sum())
    n_pass = int((table["status"] == PASS).sum())

    print("\n" + "=" * 68)
    print(f"  {n_pass} passed, {n_warn} warnings, {n_fail} failures")
    if n_fail:
        print("\n  FAILURES must be fixed before adding a stock:")
        for _, r in table[table["status"] == FAIL].iterrows():
            print(f"    - {r['check']}: {r['detail']}")
    elif n_warn:
        print("\n  No failures. Warnings are safe to proceed with.")
    else:
        print("\n  Everything green. Ready to add stocks.")
    print("=" * 68)

    return table


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / "src"))
    run_all_checks(deep="--deep" in sys.argv)
