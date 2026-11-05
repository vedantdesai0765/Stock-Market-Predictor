# Systematic Corrections Specification and Implementation Plan

**Study:** *Predictability, Explainability, and Risk-Aware Trading in Indian Equity Markets: An Empirical Study of Technical Indicators, Financial Sentiment, and Overnight Market Signals*
**Scope:** Actionable corrections addressing all issues identified in `reports/research_paper_independent_review.md`
**Date:** November 2026
**Status:** Approved for Implementation

---

## 1. Summary of Required Modifications

| Issue ID | Severity | File(s) to Modify | Affected Sections | Description of Required Action | Status |
| :---: | :---: | :--- | :--- | :--- | :---: |
| **C1** | **Critical** | `research_paper_final.md`, `build_full_manuscript.py` | Lines 317, 352, 586, 599 | Fix Python string escaping artifacts (`$ho \to \rho$`, `	ext \to \text`). | Planned |
| **C2** | **Critical** | `reports/research_paper_validation.md` | Section 6.2 | Remove all speculative journal acceptance percentages (85%–90%). | Planned |
| **C3** | **Critical** | `research_paper_final.md`, `claim_to_source_mapping.md` | Abstract, Sec 3, Sec 7.1, Sec 20 | Explicitly qualify the 26-year panel vs. shorter individual listing histories. | Planned |
| **M1** | **Major** | `research_paper_final.md`, `tables/table12_*` | Sec 15, Sec 16.10, Table 12 | State that actionable signal backtest covers 2013–2026 (13.1 years), not 26 years. | Planned |
| **M2** | **Major** | `research_paper_final.md` | Sec 16.10, Sec 17.3 | Add economic analysis of cash drag (0% vs. 6% T-Bill yield) on Variant E CAGR. | Planned |
| **M3** | **Major** | `research_paper_final.md` | Sec 13, Sec 16.9 | Add market microstructure execution caveats regarding the 70.57% opening gap forecast. | Planned |
| **M4** | **Major** | `research_paper_validation.md`, `research_paper_final.md` | Subtitle, Validation Sec 1 & 6 | Replace "100% Reproducibility" and "Publication-Ready Final Draft" with modest phrasing. | Planned |
| **m1** | **Minor** | `research_paper_final.md` | Abstract (Sec 1) | Clarify that permutation screen is a diagnostic leakage check with $p_{\min} \approx 0.1429$. | Planned |
| **m2** | **Minor** | `research_paper_final.md`, `tables/table11_*` | Sec 16.9, Table 11 | Explicitly define the "Naive Guess Base" as the unconditional gap-up frequency. | Planned |
| **m3** | **Minor** | `research_paper_final.md` | Author Block & Footnote | Add repository access footnote for replication data and model weights. | Planned |

---

## 2. Detailed Section-by-Section Corrections

### Correction C1: Fix LaTeX Math Typesetting and String Escapes
- **Action**: In `scripts/build_full_manuscript.py`, use raw string literals or double backslashes so that:
  - `\rho` is never converted to a carriage return.
  - `\text` is never converted to a tab character.
  - `\sigma` and `\Delta` render cleanly.
- **Verification**: Inspect generated Markdown to ensure no `$ho` or `	ext` remains.

### Correction C2: Remove Speculative Acceptance Probabilities
- **Action**: In `reports/research_paper_validation.md`, Section 6.2, replace the numerical probabilities ("Acceptance Probability: High (85%–90%)") with objective editorial scope alignment:
  - *"Target Venue Alignment: The manuscript aligns well with the editorial focus of The Journal of Financial Data Science, Quantitative Finance, and Expert Systems with Applications due to its multi-decade empirical scope, focus on algorithmic demystification, and adherence to econometric best practices."*

### Correction C3: Clarify Multi-Decade Panel vs. Individual Listing Maturities
- **Action**: In Abstract, Introduction (Section 3), and Section 7.1 of `research_paper/research_paper_final.md`:
  - Modify Abstract: *"Evaluating up to 26 years of daily trading data for mature listings (January 2000 to August 2026) alongside complete post-listing histories for younger equities ($N=17,636$ pooled subsampled, non-overlapping 3-day holding periods across ten National Stock Exchange listings)..."*
  - In Section 7.1, reiterate: *"While mature equities (HDFCBANK, INFY, ITC, RELIANCE, SBI, VEDL) possess over 26 years of continuous listing history (>6,588 daily sessions), newer listings are evaluated over their full post-listing lifecycles (TCS from 2004, ADANIENT from 2004, ADANIGREEN from 2018, and ETERNAL from 2021)."*

### Correction M1: Specify Actionable Signals Backtest Sample Horizon (2013–2026)
- **Action**: In Section 15 and Section 16.10, explicitly declare:
  - Title of Table 12: *"Actionable Trading Strategy Variants Comparison Across 10 Equities (2013–2026, Net of 15 bps Friction)"*.
  - Text: *"The actionable signal strategies were evaluated over an out-of-sample period spanning July 2013 to August 2026 (up to 13.1 years), corresponding to the continuous availability of machine-parsed regulatory disclosures and inter-market features."*

### Correction M2: Address Economic Realism of Cash Yield Assumption
- **Action**: In Section 17.3 ("The Economics of Risk-Aware Trading"), add the following analytical paragraph:
  - *"It is crucial to note that the backtesting framework models uninvested capital as yielding exactly 0.0% nominal return. In reality, institutional quantitative funds allocate idle cash to overnight repo, 91-day Government of India Treasury Bills, or liquid mutual funds, which historically generated 5.0% to 7.0% annualized risk-free yield over the 2013–2026 period. Because Variant E spends 93.02% of trading sessions in cash, crediting idle balances with an institutional risk-free yield of 6.0% p.a. would increase Variant E's net annualized return from 7.46% to approximately 13.04% ($7.46\% + 0.9302 \times 6.0\%$), closely approaching Buy & Hold's 15.68% CAGR while preserving its dramatic 80% maximum drawdown reduction (10.94% vs. 54.77%)."*

### Correction M3: Microstructure Caveats for Opening Gap Predictability
- **Action**: In Section 13 and Section 16.9, add explicit execution disclosures:
  - *"Market Microstructure Disclosure: Although LightGBM achieves 70.57% directional accuracy in predicting opening price movement ($P_{\text{open}} > P_{\text{close}, t-1}$), this metric measures information transmission rather than a frictionless arbitrage strategy. The official NSE opening price is determined via the pre-open call auction between 09:00 and 09:08 IST. Market participants cannot purchase shares at yesterday's close to capture the opening gap; executing at the open requires submitting orders into the pre-open auction, which entails execution uncertainty and auction price impact. Consequently, the 70.57% opening gap forecast is best utilized as a regime filter or execution benchmark for pre-existing positions rather than a standalone intraday retail strategy."*

### Correction M4: Replace Exaggerated Claims of Reproducibility
- **Action**:
  - In `reports/research_paper_validation.md`, change "100% Deterministic Reproducibility" to: *"Deterministic script execution verified across all local pipelines"*.
  - In `research_paper/research_paper_final.md`, change subtitle to: *"Working Paper — Pre-Submission Manuscript"*.

### Correction m1: Permutation Power Floor Qualification in Abstract
- **Action**: In the Abstract, revise the permutation sentence to:
  - *"...with zero equities demonstrating an edge distinguishable from random label permutations under the walk-forward diagnostic screen (`NO SIGNAL`)."*

### Correction m2: Footnote on Naive Guess Baseline in Table 11
- **Action**: Add a footnote to Table 11:
  - *"Note: The Naive Guess Baseline (63.50% pooled) represents the historical proportion of sessions where the equity opened higher than the prior day's close. LightGBM improves upon this unconditional upward drift to 70.57% ($+7.06\%$ edge, $p < 10^{-15}$) by conditioning on overnight US market closes and early Asian market opens."*

### Correction m3: Author Footnote with Replication Information
- **Action**: In the Author Block of `research_paper_final.md`:
  - Add: *"Correspondence: Vedant Desai (`vedant.ud@somaiya.edu`), Department of Computer Engineering, K. J. Somaiya College of Engineering, Somaiya Vidyavihar University, Mumbai, India. All data processing scripts, feature engineering pipelines, model checkpoints, and evaluation harnesses are available for replication in the project repository."*

---

## 3. Execution Sequence

1. Update `scripts/build_full_manuscript.py` with all text corrections and raw string literals (`r"""..."""`).
2. Run `scripts/build_full_manuscript.py` to regenerate `research_paper/research_paper_final.md`.
3. Update `research_paper/claim_to_source_mapping.md` to reflect the 2013–2026 actionable signal window and pro-forma cash yield calculations.
4. Update `reports/research_paper_validation.md` to remove speculative acceptance probabilities and tone down reproducibility claims.
5. Regenerate `research_paper/Stock_Market_Predictor_Research_Paper.docx` via `scripts/generate_paper_docx.py`.
6. Regenerate `research_paper/Stock_Market_Predictor_Research_Paper.pdf` via `scripts/generate_paper_pdf.py`.
7. Verify that all regenerated files are complete, properly sized, and free of formatting errors.
