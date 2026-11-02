# Repository Security Audit and Credential Hygiene Report

**Repository**: Stock Market Predictor (`C:\Users\vedan\Stock-Market-Predictor`)  
**Audit Date**: November 2026  
**Auditor**: Antigravity Automated Security Audit  
**Status**: Current Working Tree Clean | Historical Commit Exposure Identified | Hardening Implemented  

---

## 1. Executive Summary

A comprehensive defensive security audit of the Stock Market Predictor repository was conducted across all tracked files, configuration files, Jupyter notebooks, Python source modules, and the complete 63-commit Git history graph.

### Summary of Audit Status
* **Current Working Tree**: **CLEAN**. Zero unencrypted API keys, tokens, credentials, or private keys exist in any active notebook (code or output cells), Python module, configuration file, or documentation file.
* **Tracked Secrets**: **CLEAN**. `.env` is not tracked by Git. No certificates, credential JSONs, or private keys are tracked.
* **Environment Template**: **CLEAN**. `.env.example` contains solely non-secret placeholders (`NEWSAPI_KEY=your_key_here`).
* **Git History Exposure**: **LEGACY EXPOSURE IDENTIFIED**. A 32-character NewsAPI key was historically committed in commit `d489095c` (February 22, 2026) within `notebooks/01_data_collection.ipynb`. While the key was removed from the active working tree in commit `53b67624` (August 29, 2026), it remains permanently recorded in the immutable Git object history.
* **Configuration Hardening**: **COMPLETED**. `.gitignore` has been updated with comprehensive exclusions for virtual environments (`.venv/`), general `.env` variations, certificate/key files, credential JSONs, and operating system artifacts.

---

## 2. Findings & Severity Matrix

| Finding ID | Component / Area | Description | Severity | Current Status |
| :--- | :--- | :--- | :---: | :---: |
| **SEC-01** | Git History (`notebooks/01_data_collection.ipynb`) | Legacy 32-character NewsAPI credential committed in historical commit `d489095c`. | **Medium / High** | Remediated in working tree (commit `53b67624`); historical commit requires credential rotation / history purge. |
| **SEC-02** | Repository Hygiene (`.gitignore`) | Incomplete ignore rules: `.venv/`, `venv/`, `.env.*`, and certificate/credential patterns were not explicitly excluded. | **Low** | **RESOLVED**. Comprehensive exclusions added to `.gitignore`. |
| **SEC-03** | Model Serialization (`models/**/*.joblib`) | Scikit-learn pipelines serialized via Python pickle (`joblib`). Loading untrusted `.joblib` files poses arbitrary code execution risks. | **Informational** | **ACCEPTABLE**. Models are internal project artifacts loaded by `app.py`. |
| **SEC-04** | API Key Loader (`src/collect.py`) | Fallback search for `.env` at project root reads line-by-line without exposing secrets. | **Informational** | **VERIFIED CLEAN**. Safely returns `None` when `.env` is absent. |

---

## 3. Current Working Tree Audit

An exhaustive automated scan of all files in the current repository working tree (excluding `.git` and `.venv`) evaluated 211 tracked files and uncommitted working files against credential signatures (AWS keys, GitHub tokens, generic API keys, private keys, and 32-character hex tokens).

### 3.1. Jupyter Notebooks (`notebooks/01` through `notebooks/07`)
* **Code Cells**: Audited for hardcoded credentials. All API integrations in [`notebooks/01_data_collection.ipynb`](file:///C:/Users/vedan/Stock-Market-Predictor/notebooks/01_data_collection.ipynb) utilize `src.collect._load_api_key()` to read from the environment or a local gitignored `.env` file.
* **Markdown Instructions**: Only generic documentation references and public registration links (`https://newsapi.org/register`) are present.
* **Output Cells**: Audited for leaked token outputs or API error dumps containing authorization headers. Zero credentials or tokens are present in any notebook output metadata.

### 3.2. Configuration and Environment Files
* [**`.env.example`**](file:///C:/Users/vedan/Stock-Market-Predictor/.env.example):
  ```bash
  # Copy this file to .env and fill in your key.
  # .env is gitignored and must never be committed.
  #
  # Free key: https://newsapi.org/register
  NEWSAPI_KEY=your_key_here
  ```
  Verified to contain only the dummy placeholder string `your_key_here`.
* **`.env`**:
  Verified as **absent or untracked**. It does not exist in `git ls-files` and is strictly ignored by `.gitignore`.

### 3.3. Python Source Modules (`src/`, `scripts/`, `sentiment_study/src/`)
* **News Collection Module** ([`src/collect.py`](file:///C:/Users/vedan/Stock-Market-Predictor/src/collect.py)):
  Function `_load_api_key(env_var="NEWSAPI_KEY")` safely queries `os.environ` and parses local `.env` if present, defaulting cleanly to `None`. No fallback strings contain credentials.
* **Validation Suite** ([`src/validate.py`](file:///C:/Users/vedan/Stock-Market-Predictor/src/validate.py)):
  Function `check_secrets()` scans the repository for committed 32-character hex strings near API key markers, correctly ignoring `.venv`, `venv`, and `env` virtual environment directories.

### 3.4. Documentation and UI Screenshots
* [**`README.md`**](file:///C:/Users/vedan/Stock-Market-Predictor/README.md): No tokens or authorization headers present.
* [**`screenshots/why_did_it_move_infy.png`**](file:///C:/Users/vedan/Stock-Market-Predictor/screenshots/why_did_it_move_infy.png) & [**`screenshots/why_did_it_move_sensex.png`**](file:///C:/Users/vedan/Stock-Market-Predictor/screenshots/why_did_it_move_sensex.png): User interface screenshots render news timelines and econometric metrics; no credential inputs or API key values are visible.

---

## 4. Git History Exposure Analysis (SEC-01)

A historical commit scan across all 63 commits in the repository identified a single historical credential exposure:

### 4.1. Incident Timeline
1. **Commit `d489095c4fa94a4ac135ebcb4232ee720c70b091`**
   * **Date**: Sunday, February 22, 2026, 18:49:58 +0530
   * **Author**: Vedant Narayan <vedant.narayan@somaiya.edu>
   * **Commit Message**: `feature engineering started and dataset collected`
   * **Exposed File**: `notebooks/01_data_collection.ipynb`
   * **Nature of Exposure**: A live 32-character hexadecimal NewsAPI key was assigned directly to variable `API_KEY` in Cell 0:
     ```python
     # [REDACTED HISTORICAL DIFF - VALUE MASKED]
     + "API_KEY = \"[REDACTED_32_CHAR_HEX_KEY]\"\n"
     + "BASE_URL = \"https://newsapi.org/v2/everything\"\n"
     ```
2. **Commit `53b676246103254086e08e7cd58cee1a8ed35e4d`**
   * **Date**: Saturday, August 29, 2026, 09:41:58 +0530
   * **Author**: SmitPatil2004 <smitpatil2004@gmail.com>
   * **Commit Message**: `upd`
   * **Action Taken**: Removed the hardcoded string from `notebooks/01_data_collection.ipynb`, added `.env.example`, and transitioned news collection to `_load_api_key()`.

### 4.2. Security Implication
Although commit `53b67624` cleanly excised the secret from the working directory, Git stores historical snapshots immutably. Anyone with clone access to the repository can retrieve the historical commit via `git show d489095c`.

---

## 5. Recommended Remediation & Safe Procedures

### 5.1. Immediate Action: Credential Rotation (User Responsibility)
* **Status**: Required.
* **Procedure**:
  1. Log into your [NewsAPI Account](https://newsapi.org/account).
  2. Invalidate / delete the compromised API key.
  3. Generate a new API key.
  4. Place the new key inside a local `.env` file at the project root (`NEWSAPI_KEY=your_new_key`).
  5. Never paste the new key into code or Git-tracked files.

### 5.2. Permanent Git History Sanitization (Optional / Recommended for Public Repositories)
To completely expunge the compromised commit from the Git object database prior to making the repository public:

> [!WARNING]
> Rewriting Git history modifies commit hashes and requires all collaborators to re-clone the repository. Do not perform this on a live production branch without coordinating with your team. Per strict execution guidelines, this operation was **not** run automatically.

#### Option A: Using `git-filter-repo` (Recommended by Git)
```bash
# 1. Install git-filter-repo
pip install git-filter-repo

# 2. Create a clean backup clone
git clone --no-local . ../Stock-Market-Predictor-backup

# 3. Replace the secret in history with a placeholder string
# Create a replacement expressions file: replace.txt
# Format: [EXPOSED_SECRET_VALUE] ==> your_key_here
git filter-repo --replace-text replace.txt

# 4. Force push the sanitized history to remote (only after verification)
# git push origin --force --all
```

#### Option B: Using BFG Repo-Cleaner
```bash
# 1. Download BFG Repo-Cleaner
# 2. Create passwords.txt containing the exact secret string
java -jar bfg.jar --replace-text passwords.txt .
git reflog expire --expire=now --all && git gc --prune=now --aggressive
```

---

## 6. Safe Cleanup Performed (Phase 3)

### 6.1. `.gitignore` Hardening
The repository [`.gitignore`](file:///C:/Users/vedan/Stock-Market-Predictor/.gitignore) was enhanced to ensure that private keys, credentials, local environment overrides, and virtual environments can never be accidentally staged:

```diff
--- a/.gitignore
+++ b/.gitignore
@@ -3,9 +3,39 @@
 __pycache__/
 *.pyc
 
-# Ignore environment and editor files
-.env
-.vscode/
+# Virtual environments
+.venv/
+venv/
+env/
+ENV/
+
+# Ignore environment and secret files
+.env
+.env.*
+!.env.example
+*.pem
+*.key
+*.p12
+*.pfx
+credentials.json
+client_secret*.json
+token.json
+
+# Editor and OS files
+.vscode/
+.idea/
+.DS_Store
+Thumbs.db
+*.swp
+*.swo
+
+# Build and test artifacts
+build/
+dist/
+*.egg-info/
+.pytest_cache/
+.coverage
+htmlcov/
```

### 6.2. Integrity and Functionality Check
Following the `.gitignore` update:
1. `git status` confirmed all existing user modifications and untracked report drafts were preserved without disruption.
2. `python -m src.validate` was executed cleanly:
   - **57 checks passed**.
   - **Committed secrets**: `[ok] no hardcoded keys detected`.
   - **Environment configuration**: `[ok] .env is in .gitignore`.
   - Zero application failures.

---

## 7. Before & After Security Status Summary

| Check Item | Initial State (Before Audit) | Final State (After Audit & Cleanup) | Verification |
| :--- | :---: | :---: | :---: |
| **Active Working Tree Secrets** | 0 detected | 0 detected | Scanned all .py, .ipynb, .md, .json |
| **Notebook Output Secrets** | 0 detected | 0 detected | Parsed JSON output arrays in notebooks 01–07 |
| **`.env.example` Hygiene** | Safe placeholder (`your_key_here`) | Safe placeholder (`your_key_here`) | Verified file contents |
| **Virtual Environment Staging Risk** | High (`.venv/` not in `.gitignore`) | **Resolved** (`.venv/`, `venv/`, `env/` ignored) | Tested in `.gitignore` |
| **Alternate `.env.*` Risk** | High (Only exact `.env` ignored) | **Resolved** (`.env.*` ignored; `!.env.example` kept) | Tested pattern match |
| **Certificate / Key File Risk** | Unprotected (`*.pem`, `*.key` unlisted) | **Resolved** (`*.pem`, `*.key`, `*.p12` ignored) | Added to `.gitignore` |
| **OS / Editor Leakage Risk** | Partial (`.vscode/` only) | **Resolved** (`.idea/`, `.DS_Store`, `Thumbs.db` added) | Added to `.gitignore` |
| **Git History Object Graph** | 1 key in commit `d489095c` | Documented; remediation instructions provided | Traced to commit `d489095c` |
| **Validation Test Suite** | 57 passed, 8 warnings, 0 failures | 57 passed, 8 warnings, 0 failures | Verified via `src.validate` |
