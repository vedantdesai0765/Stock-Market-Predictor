# Statistical Robustness Audit: Dependence-Aware Significance Testing for Overnight Opening Price Forecasts

**Document Reference**: `reports/overnight_statistical_robustness.md`
**Date**: November 5, 2026
**Auditor**: Independent Quantitative Methodology & Statistical Robustness Protocol
**Repository**: [Stock-Market-Predictor](file:///C:/Users/vedan/Stock-Market-Predictor)
**Primary Code Artifact**: [scripts/audit_overnight_robustness.py](file:///C:/Users/vedan/Stock-Market-Predictor/scripts/audit_overnight_robustness.py)
**Output Data Reference**: [reports/forecast/overnight_bootstrap_robustness.csv](file:///C:/Users/vedan/Stock-Market-Predictor/reports/forecast/overnight_bootstrap_robustness.csv)
**Underlying Data**: [sentiment_study/reports/oof_company_all.parquet](file:///C:/Users/vedan/Stock-Market-Predictor/sentiment_study/reports/oof_company_all.parquet) (`NSE__gapup_next__LightGBM__B`)

---

## Executive Summary

The primary research paper reports an out-of-fold next-morning directional opening price forecast accuracy of **70.57%** (19,179 correct of 27,179 paired stock-date observations) against a historical training-majority baseline of **63.50%** (17,259 correct), representing a net predictive edge of **$+7.06\%$ points**. A conventional paired McNemar test and exact two-sided binomial test on the 8,250 discordant prediction pairs ($b = 5,085$ vs. $c = 3,165$) yields $\chi^2 = 446.37$ and an exact binomial $p$-value of $p = 5.68 \times 10^{-100}$ (reported in the manuscript as $p < 10^{-15}$ due to floating-point truncation in CSV outputs).

This audit investigates whether the paired prediction outcomes are independent across observations, and whether the documented out-of-fold accuracy edge remains statistically significant when cross-sectional clustering and temporal dependence are fully accounted for.

### Key Findings

1. **Failure of the i.i.d. Assumption**:
   - **Cross-Sectional Clustering**: The 27,179 stock-date observations cluster across **3,205 unique trading sessions** (averaging 8.48 concurrent equities per day). Pairs of equities evaluated on identical trading dates exhibit substantial positive correlation in their directional targets ($r \approx +0.342$), model correctness ($r \approx +0.232$), and correctness differences ($r \approx +0.298$).
   - **Temporal Autocorrelation**: Daily aggregated accuracy differences exhibit mild multi-day persistence across short horizons (lags 1–5 Ljung-Box $Q = 30.68, p = 1.09 \times 10^{-5}$).
2. **Standard Error and Variance Inflation**:
   - The conventional McNemar test standard error ($\text{SE}_{\text{iid}} = 0.3314\%$) assumes mutually independent observations, understating sampling variability.
   - Accounting for cross-sectional and temporal clustering via a 21-day Date-Block Bootstrap ($R = 5,000$ resamples) expands the standard error to **$\text{SE}_{\text{boot}} = 0.7565\%$**.
   - This corresponds to a **Standard-Error Inflation Factor ($\text{SEIF}$) of $2.28\times$** ($0.7565\% / 0.3314\% = 2.2826$), and a true **Variance Inflation Factor ($\text{VIF}$) of $5.21\times$** ($2.2826^2 = 5.2104$).
3. **Robustness of Statistical Significance**:
   - Despite a $5.21\times$ variance expansion, the $+7.06\%$ out-of-fold edge **remains decisively statistically significant** across all tested block lengths from $B = 1$ day to $B = 63$ trading days (1 quarter).
   - Under the 21-day block bootstrap ($B = 21$), the $95\%$ confidence interval is **$[+5.592\%, +8.564\%]$**, yielding an asymptotic normal-approximation test statistic of $z = 9.34$ ($p = 9.85 \times 10^{-21}$).
   - The empirical bootstrap $p$-value, evaluated on the centered null distribution with $R = 5,000$ resamples and Laplace plus-one correction, is bounded by finite Monte Carlo resolution at **$p_{\text{boot}} < 0.0002$** ($p = 1/5001 \approx 0.0002$; zero of 5,000 centered replicates reached or exceeded the observed statistic).
4. **Hypothesis Framework Harmonization**:
   - The manuscript previously stated an asymmetric one-sided alternative hypothesis while reporting a two-sided test statistic. The framework is harmonized to a consistent **two-sided formulation**:
     $$H_0: P(n_{01}) = P(n_{10}) = 0.5 \quad \text{vs.} \quad H_1: P(n_{01}) \ne P(n_{10})$$
5. **Strict Four-Way Conceptual Separation**:
   - **Predictive Accuracy** ($70.57\%$) $\ne$ **Statistical Significance** (Asymptotic $p \approx 9.85 \times 10^{-21}$; Empirical $p_{\text{boot}} < 0.0002$) $\ne$ **Economic Profitability** $\ne$ **Tradability After Execution Costs**.
   - Statistical significance confirms an econometric property of international information transmission; it does **not** establish an executable trading arbitrage due to the pre-open call auction mechanism and execution frictions.

---

## 1. Dataset Scope and Baseline Diagnostics

The evaluated out-of-fold sample comprises strictly out-of-sample walk-forward predictions from 2013 to 2025 across the ten liquid National Stock Exchange (NSE) equities:

```
Total Paired Observations (N):  27,179
Unique Trading Sessions (T):    3,205 dates
Calendar Range:                 January 1, 2013 to December 30, 2025
Equities Monitored:             10 liquid large-cap NSE equities
Average Stocks per Session:     8.48 stocks / date
```

### Table 1: Distribution of Concurrent Stock Observations per Trading Date

| Concurrent Stocks on Date | Date Count ($T_k$) | Percentage of Total Dates | Total Stock Observations |
| :---: | :---: | :---: | :---: |
| 10 Stocks | 526 | 16.41% | 5,260 |
| 9 Stocks | 838 | 26.15% | 7,542 |
| 8 Stocks | 1,526 | 47.61% | 12,208 |
| 7 Stocks | 284 | 8.86% | 1,988 |
| 6 Stocks | 29 | 0.90% | 174 |
| 4 Stocks | 1 | 0.03% | 4 |
| 3 Stocks | 1 | 0.03% | 3 |
| **Total** | **3,205** | **100.0%** | **27,179** |

*Interpretation*: On **90.17%** of all trading dates in the 13-year panel, between 8 and 10 equities are monitored simultaneously. Because macroeconomic shocks impact all equities synchronously, treating these 27,179 observations as mutually independent overstates the true effective sample size.

---

### 1.1. Hierarchy of Sample Sizes and Clustering Dimensions

To avoid conflating distinct measures of sample size, econometric precision requires distinguishing four hierarchical dimensions of aggregation:

1. **Nominal Panel Observations ($N = 27,179$)**:
   - Represents the total count of paired stock-date instances across 10 equities evaluated over 13 years (2013–2025).
   - This nominal sample size forms the basis of conventional paired contingency tables, but its observations are not mutually independent.
2. **Trading-Date Clusters ($T = 3,205$)**:
   - Represents the count of distinct NSE trading calendar days.
   - On each day, an average of 8.48 equities are observed concurrently. Clustering by date accounts for intra-day cross-sectional correlation across stocks.
3. **Non-Overlapping Block Units ($K \approx 152$)**:
   - When partitioning the 3,205 trading sessions into non-overlapping blocks of $B = 21$ trading days (approx. 1 calendar month), the sample comprises $3,205 / 21 \approx 152.6$ distinct temporal blocks.
   - Moving block bootstrap samples across all $T - B + 1 = 3,185$ overlapping contiguous blocks to estimate sampling variance without boundary distortion.
4. **Effective Sample Size Adjusted for Clustering / Autocorrelation ($N_{\text{eff}}$ and $T_{\text{eff}}$)**:
   - Relative to naive i.i.d. observations, the empirical variance inflation factor under 21-day blocking is $\text{VIF} \approx 5.21$. The effective independent sample size in terms of equivalent independent stock observations is:
     $$N_{\text{eff}} = \frac{N}{\text{VIF}} = \frac{27,179}{5.2104} \approx 5,216 \text{ equivalent independent observations}$$
   - In terms of independent trading dates, because daily cross-sectional clustering produces $\text{SEIF} = 1.781$ ($\text{VIF}_{\text{date}} = 3.172$), the effective independent date count is:
     $$T_{\text{eff}} = \frac{T}{\text{VIF}_{\text{date}}} = \frac{3,205}{3.1722} \approx 1,010 \text{ equivalent independent dates}$$

*Methodological Rule*: These four quantities measure fundamentally different levels of aggregation and dependence; they must not be used interchangeably as independent sample sizes.

---

## 2. Cross-Sectional and Temporal Dependence Diagnostics

### 2.1. Cross-Sectional Co-Movement Across Stocks

To evaluate cross-sectional dependence, we define for each equity $i$ on date $t$:
- Binary model correctness: $m_{i,t} = \mathbb{I}(\hat{y}_{\text{model}, i, t} = y_{i,t}) \in \{0, 1\}$
- Binary baseline correctness: $b_{i,t} = \mathbb{I}(\hat{y}_{\text{guess}, i, t} = y_{i,t}) \in \{0, 1\}$
- Correctness difference: $d_{i,t} = m_{i,t} - b_{i,t} \in \{-1, 0, +1\}$

We compute the pairwise Pearson correlation across all 45 unique stock pairs on identical trading dates:

### Table 2: Cross-Stock Pairwise Correlation Matrix of Correctness Differences ($d_{i,t}$)

| Equity | ADANIENT | ADANIGREEN | ETERNAL | HDFCBANK | INFY | ITC | RELIANCE | SBI | TCS | VEDL |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **ADANIENT** | 1.000 | 0.366 | 0.228 | 0.280 | 0.190 | 0.304 | 0.374 | 0.371 | 0.230 | 0.365 |
| **ADANIGREEN**| 0.366 | 1.000 | 0.306 | 0.175 | 0.214 | 0.272 | 0.186 | 0.242 | 0.209 | 0.286 |
| **ETERNAL** | 0.228 | 0.306 | 1.000 | 0.260 | 0.280 | 0.292 | 0.229 | 0.321 | 0.269 | 0.347 |
| **HDFCBANK** | 0.280 | 0.175 | 0.260 | 1.000 | 0.294 | 0.384 | 0.461 | 0.442 | 0.296 | 0.335 |
| **INFY** | 0.190 | 0.214 | 0.280 | 0.294 | 1.000 | 0.279 | 0.322 | 0.294 | 0.354 | 0.266 |
| **ITC** | 0.304 | 0.272 | 0.292 | 0.384 | 0.279 | 1.000 | 0.343 | 0.384 | 0.279 | 0.303 |
| **RELIANCE** | 0.374 | 0.186 | 0.229 | 0.461 | 0.322 | 0.343 | 1.000 | 0.461 | 0.322 | 0.343 |
| **SBI** | 0.371 | 0.242 | 0.321 | 0.442 | 0.294 | 0.384 | 0.461 | 1.000 | 0.337 | 0.385 |
| **TCS** | 0.230 | 0.209 | 0.269 | 0.296 | 0.354 | 0.279 | 0.322 | 0.337 | 1.000 | 0.182 |
| **VEDL** | 0.365 | 0.286 | 0.347 | 0.335 | 0.266 | 0.303 | 0.343 | 0.385 | 0.182 | 1.000 |

### Summary of Cross-Sectional Diagnostics

- **Directional Target ($y_{i,t}$)**: Mean pairwise correlation is **$+0.3421$** (range $+0.2361$ to $+0.4993$). When US indices or Asian markets move strongly overnight, Indian equities exhibit strong synchronous opening gap direction.
- **Model Correctness ($m_{i,t}$)**: Mean pairwise correlation is **$+0.2321$** (range $+0.1012$ to $+0.3555$).
- **Correctness Difference ($d_{i,t}$)**: Mean pairwise correlation is **$+0.2981$** (range $+0.1751$ to $+0.4607$). Peak co-movement occurs between major index constituents: HDFCBANK–RELIANCE ($r = 0.461$) and RELIANCE–SBI ($r = 0.461$).

*Conclusion*: Cross-sectional independence is decisively rejected. Observations on the same calendar date share a common market factor.

---

### 2.2. Temporal Autocorrelation in Accuracy Differences

We define the daily aggregated accuracy difference:
$$D_t = \frac{1}{N_t} \sum_{i=1}^{N_t} d_{i,t} = \bar{m}_t - \bar{b}_t$$
where $N_t$ is the number of active stocks on date $t$.

### Table 3: Autocorrelation Profile of Daily Accuracy Difference ($D_t$)

| Lag ($k$ trading days) | Autocorrelation ($\rho_k$) | Ljung-Box $Q$-Statistic | Degrees of Freedom ($df$) | $p$-value |
| :---: | :---: | :---: | :---: | :---: |
| **Lag 1** | $+0.0111$ | 0.40 | 1 | $p = 0.5280$ |
| **Lag 2** | $+0.0468$ | 7.42 | 2 | $p = 0.0245$ |
| **Lag 3** | $+0.0748$ | 25.40 | 3 | $p = 1.27 \times 10^{-5}$ |
| **Lag 4** | $+0.0236$ | 27.19 | 4 | $p = 1.82 \times 10^{-5}$ |
| **Lag 5** | $+0.0330$ | 30.68 | 5 | $p = 1.09 \times 10^{-5}$ |
| **Lag 6** | $+0.0283$ | 33.26 | 6 | $p = 9.40 \times 10^{-6}$ |
| **Lag 7** | $+0.0194$ | 34.47 | 7 | $p = 1.39 \times 10^{-5}$ |
| **Lag 8** | $+0.0331$ | 38.00 | 8 | $p = 7.39 \times 10^{-6}$ |
| **Lag 9** | $+0.0255$ | 40.09 | 9 | $p = 7.15 \times 10^{-6}$ |
| **Lag 10** | $+0.0283$ | 42.66 | 10 | $p = 5.62 \times 10^{-6}$ |
| **Lag 15** | $+0.0170$ | 52.48 | 15 | $p = 4.41 \times 10^{-6}$ |
| **Lag 20** | $+0.0265$ | 58.06 | 20 | $p = 1.34 \times 10^{-5}$ |

*Interpretation*:
- Lag 1 autocorrelation is statistically indistinguishable from zero ($\rho_1 = +0.0111, p = 0.5280$), indicating no immediate 1-day momentum in forecasting error.
- Over multi-day horizons (lags 2–5), small positive autocorrelations ($\rho_2 = 0.047, \rho_3 = 0.075$) are statistically significant under the Ljung-Box test ($Q = 30.68, p < 10^{-4}$). This reflects persistent volatility regimes: during multi-day trending or turbulent global macro periods, model edge clusters over several consecutive sessions.

---

## 3. Dependence-Aware Methodologies

To conduct rigorous hypothesis testing without assuming observation independence, two complementary approaches were implemented:

### 3.1. Time-Series HAC Regression on Daily Aggregated Panels
We model the daily cross-sectional mean difference $D_t$ as:
$$D_t = \mu + \epsilon_t, \quad t = 1, \dots, T$$
where $\mu = \mathbb{E}[D_t]$ represents the expected daily accuracy edge.
- Standard OLS standard error assumes independent trading dates.
- Newey-West Heteroskedasticity and Autocorrelation Consistent (HAC) standard error incorporates a Bartlett kernel:
  $$\widehat{\text{Var}}_{\text{HAC}}(\hat{\mu}) = \frac{1}{T} \left( \hat{\gamma}_0 + 2 \sum_{l=1}^L \left(1 - \frac{l}{L+1}\right) \hat{\gamma}_l \right)$$
  where $\hat{\gamma}_l = \frac{1}{T} \sum_{t=l+1}^T e_t e_{t-l}$.

### 3.2. Date-Block Bootstrap (Preserving Cross-Section & Pairing)
To evaluate the full pooled sample ($N = 27,179$) while preserving both cross-sectional co-movement and model-versus-baseline pairing, we implement a **Date-Block Bootstrap**:
1. **Sampling Unit**: Blocks of $B$ consecutive trading calendar dates:
   $$\mathcal{B}_k = \{t_k, t_{k+1}, \dots, t_{k+B-1}\}$$
2. **Synchronous Cross-Sectional Bundle**: When date $t$ is drawn into a bootstrap resample, **all** stock observations active on date $t$ (all $N_t$ equities) are drawn together. This preserves the exact cross-sectional co-movement matrix without distortion.
3. **Paired Observation Preservation**: For every individual equity observation $(i, t)$, the pair $(m_{i,t}, b_{i,t})$ is drawn intact, guaranteeing that the paired comparison between model and baseline is strictly preserved.
4. **Resampling Parameters**: $R = 5,000$ bootstrap replicates drawn for each evaluated block length $B \in \{1, 2, 3, 5, 10, 15, 21, 42, 63\}$ trading days.
5. **Test Statistics**: For each resample $r \in \{1, \dots, R\}$:
   $$\hat{\Delta}^{(r)} = \frac{\sum_{t \in \text{resample}} \sum_{i=1}^{N_t} (m_{i,t} - b_{i,t})}{\sum_{t \in \text{resample}} N_t}$$
   - Bootstrap Standard Error: $\text{SE}_{\text{boot}} = \sqrt{\frac{1}{R-1} \sum_{r=1}^R (\hat{\Delta}^{(r)} - \bar{\Delta}^*)^2}$
   - Standard-Error Inflation Factor: $\text{SEIF} = \text{SE}_{\text{boot}} / \text{SE}_{\text{iid}}$
   - Variance Inflation Factor: $\text{VIF} = (\text{SE}_{\text{boot}} / \text{SE}_{\text{iid}})^2 = \text{SEIF}^2$
   - Two-sided empirical bootstrap $p$-value under centered null distribution $H_0: \Delta = 0$:
     $$p_{\text{boot}} = \frac{1 + \sum_{r=1}^R \mathbb{I}\left(|\hat{\Delta}^{(r)} - \hat{\Delta}_{\text{obs}}| \ge |\hat{\Delta}_{\text{obs}}|\right)}{R + 1}$$
   - Asymptotic normal approximation test statistic and $p$-value:
     $$z = \frac{\hat{\Delta}_{\text{obs}}}{\text{SE}_{\text{boot}}}, \quad p_{\text{asymp}} = 2 \cdot \Phi(-|z|)$$

---

## 4. Empirical Robustness Results

### 4.1. Daily Time-Series Regression & Newey-West HAC Results

```
Unweighted Mean Daily Edge:  +6.9273% pts
Sample Size:                 T = 3,205 trading days
Daily Standard Deviation:    33.72% pts
```

### Table 4: Time-Series Regression Standard Errors Across Lag Horizons

| Estimation Method | Evaluated Horizon | Standard Error ($\text{SE}$) | $t$-statistic | Two-Sided $p$-value |
| :--- | :---: | :---: | :---: | :---: |
| **OLS (Daily Clustering)** | Zero serial correlation | 0.5956% | 11.63 | $p = 2.87 \times 10^{-31}$ |
| **Newey-West HAC** | Lag $L = 1$ day | 0.5988% | 11.57 | $p = 5.96 \times 10^{-31}$ |
| **Newey-West HAC** | Lag $L = 5$ days (1 week) | 0.6476% | 10.70 | $p = 1.05 \times 10^{-26}$ |
| **Newey-West HAC** | Lag $L = 10$ days (2 weeks)| 0.6904% | 10.03 | $p = 1.08 \times 10^{-23}$ |
| **Newey-West HAC** | Lag $L = 21$ days (1 month)| 0.7400% | 9.36 | $p = 7.86 \times 10^{-21}$ |

---

### 4.2. Date-Block Bootstrap Sensitivity Results ($R = 5,000$ Resamples)

### Table 5: Comprehensive Block-Length Sensitivity Analysis on Pooled Out-of-Fold Edge ($\hat{\Delta}_{\text{obs}} = +7.0643\%$)

| Block Length ($B$) | Temporal Horizon | $\text{SE}_{\text{boot}}$ | SE Inflation ($\text{SEIF}$) | Variance Inflation ($\text{VIF}$) | 95% Bootstrap CI | $z$-score | Asymptotic $p$-value | Empirical $p$-value ($p_{\text{boot}}$) | Min Replicate Edge |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Naive i.i.d.** | Single stock-date | 0.3314% | $1.00\times$ | $1.00\times$ | $[+6.415\%, +7.714\%]$ | 21.32 | $p = 7.72 \times 10^{-101}$ | — | — |
| **$B = 1$ day** | Date cluster only | 0.5903% | $1.78\times$ | $3.17\times$ | $[+5.887\%, +8.180\%]$ | 11.97 | $p = 5.27 \times 10^{-33}$ | $p_{\text{boot}} < 0.0002$ | $+4.850\%$ |
| **$B = 2$ days** | 2-day blocks | 0.6091% | $1.84\times$ | $3.38\times$ | $[+5.884\%, +8.267\%]$ | 11.60 | $p = 4.25 \times 10^{-31}$ | $p_{\text{boot}} < 0.0002$ | $+4.850\%$ |
| **$B = 3$ days** | 3-day blocks | 0.6207% | $1.87\times$ | $3.51\times$ | $[+5.860\%, +8.276\%]$ | 11.38 | $p = 5.19 \times 10^{-30}$ | $p_{\text{boot}} < 0.0002$ | $+4.850\%$ |
| **$B = 5$ days** | 1 calendar week | 0.6500% | $1.96\times$ | $3.85\times$ | $[+5.840\%, +8.370\%]$ | 10.87 | $p = 1.63 \times 10^{-27}$ | $p_{\text{boot}} < 0.0002$ | $+4.619\%$ |
| **$B = 10$ days** | 2 calendar weeks | 0.6787% | $2.05\times$ | $4.19\times$ | $[+5.774\%, +8.440\%]$ | 10.41 | $p = 2.25 \times 10^{-25}$ | $p_{\text{boot}} < 0.0002$ | $+4.774\%$ |
| **$B = 15$ days** | 3 calendar weeks | 0.7262% | $2.19\times$ | $4.80\times$ | $[+5.664\%, +8.524\%]$ | 9.73 | $p = 2.30 \times 10^{-22}$ | $p_{\text{boot}} < 0.0002$ | $+4.238\%$ |
| **$B = 21$ days** | **1 trading month** | **0.7565%** | **$2.28\times$** | **$5.21\times$** | **$[+5.592\%, +8.564\%]$** | **9.34** | **$p = 9.85 \times 10^{-21}$** | **$p_{\text{boot}} < 0.0002$** | **$+3.805\%$** |
| **$B = 42$ days** | 2 trading months | 0.8233% | $2.48\times$ | $6.17\times$ | $[+5.500\%, +8.751\%]$ | 8.58 | $p = 9.46 \times 10^{-18}$ | $p_{\text{boot}} < 0.0002$ | $+3.541\%$ |
| **$B = 63$ days** | 1 trading quarter| 0.8823% | $2.66\times$ | $7.09\times$ | $[+5.447\%, +8.863\%]$ | 8.01 | $p = 1.17 \times 10^{-15}$ | $p_{\text{boot}} < 0.0002$ | $+3.125\%$ |

---

## 5. Statistical Interpretation and Block-Length Justification

### 5.1. Methodological Selection of the 21-Day Sensitivity Baseline
In moving block bootstrap methodology, the block length reflects a trade-off between bias (insufficient block length leaves residual serial dependence uncaptured) and variance (excessive block length reduces the effective number of independent blocks):
- The autocorrelation analysis in Table 3 demonstrated that serial correlation in $D_t$ decays to noise levels after approximately 15–20 trading days ($\rho_{15} = 0.017, \rho_{20} = 0.026$).
- The Ljung-Box statistic stabilizes after 20 lags.
- Therefore, **$B = 21$ trading days (approximately 1 calendar month of trading sessions)** is selected as a **methodological sensitivity baseline**, not as a uniquely optimal parameter.
- Under $B = 21$, the bootstrap standard error is **$\text{SE} = 0.7565\%$**, reflecting a standard-error inflation factor of **$2.28\times$** and a variance inflation factor of **$5.21\times$** over the naive i.i.d. assumption.

### 5.2. Robustness Across the Entire Block-Length Spectrum ($B = 1$ to $B = 63$)
Rather than relying on any single block length, Table 5 evaluates the entire spectrum from daily clustering ($B = 1$) to quarterly blocking ($B = 63$):
- Even under highly conservative quarterly blocking ($B = 63$ trading days, which creates wide 3-month blocks and inflates variance by $7.09\times$):
  - Standard error: $\text{SE} = 0.8823\%$.
  - $95\%$ Confidence Interval: **$[+5.447\%, +8.863\%]$**.
  - Test Statistic: $z = 8.01$, asymptotic $p = 1.17 \times 10^{-15}$.
  - The empirical bootstrap distribution never touched zero: the minimum edge observed across all 5,000 quarterly resamples was **$+3.125\%$**.
- Across every evaluated block horizon, the lower bound of the 95% confidence interval strictly excludes zero (remaining above $+5.4\%$).
- The asymptotic $p$-values range from $5.27 \times 10^{-33}$ to $1.17 \times 10^{-15}$, consistently rejecting the null hypothesis ($p < 10^{-15}$).
- The empirical bootstrap $p$-value is bounded by the finite simulation resolution as $p_{\text{boot}} < 0.0002$ ($0$ of $5,000$ resamples reached or exceeded the observed statistic under the centered null distribution).

*Verdict*: The finding of statistical significance is invariant to the choice of block length.

---

## 6. Harmonization of the Hypothesis Framework

The primary manuscript originally reported a two-sided exact binomial test statistic ($p = 5.68 \times 10^{-100}$) alongside an asymmetric one-sided alternative hypothesis ($H_1: P(n_{01}) > P(n_{10})$).

To ensure rigorous internal consistency, the hypothesis framework is formally harmonized to a symmetric **two-sided formulation**:

$$\begin{aligned}
\text{Null Hypothesis } (H_0): & \quad P(n_{01}) = P(n_{10}) = 0.5 \\
& \quad \text{(Model and naive baseline have identical marginal classification accuracies)} \\
\text{Alternative Hypothesis } (H_1): & \quad P(n_{01}) \ne P(n_{10}) \\
& \quad \text{(Model and naive baseline have differing marginal classification accuracies)}
\end{aligned}$$

Because the observed discordant count for the model ($n_{01} = 5,085$) substantially exceeds that for the baseline ($n_{10} = 3,165$), the observed sample deviation strictly favors the model under the two-sided test ($p < 10^{-15}$ across all parametric and asymptotic formulations; $p_{\text{boot}} < 0.0002$ empirically).

---

## 7. Strict Four-Way Conceptual Separation

A critical responsibility of quantitative academic reporting is to prevent the conflation of statistical significance with financial trading profitability. The findings must be interpreted through four strictly separated conceptual lenses:

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                     FOUR-WAY CONCEPTUAL SEPARATION                           │
├──────────────────────────────────────────────────────────────────────────────┤
│ 1. PREDICTIVE ACCURACY (70.57%)                                              │
│    Econometric property measuring directional classification agreement with  │
│    the sign of the opening gap (P_open > P_close_prev).                      │
│                                                                              │
│ 2. STATISTICAL SIGNIFICANCE (Asymptotic p ~ 9.85 x 10^-21; p_boot < 0.0002)  │
│    Confirms that the +7.06% accuracy advantage is not sampling noise,        │
│    cross-sectional clustering, or multi-day persistence.                     │
│                                                                              │
│ 3. ECONOMIC PROFITABILITY (Zero Gap Arbitrage)                               │
│    Unattainable because the NSE Pre-Open Call Auction (09:00–09:08 IST)      │
│    impounds overnight international news into P_open prior to execution.     │
│                                                                              │
│ 4. TRADABILITY AFTER EXECUTION COSTS (Negative Net Alpha)                    │
│    Entering at P_open incurs 10–30 bps opening spreads and 15 bps Indian     │
│    statutory taxes/fees (STT, GST, SEBI fee). Intraday mean reversion wipes  │
│    out any subsequent continuation gain.                                     │
└──────────────────────────────────────────────────────────────────────────────┘
```

### Detailed Breakdown

1. **Predictive Accuracy ($70.57\%$)**:
   - Represents an empirical classification metric on the out-of-sample directional gap ($y_{i,t}^{\text{open}} = \mathbb{I}(P_{i,t}^{\text{open}} > P_{i,t-1}^{\text{close}})$).
   - Validates that US equity indices (S&P 500, Nasdaq), early Asian market opens (Nikkei, Hang Seng), and overnight SGX/GIFT Nifty futures possess strong informational transmission into Indian equity opening price levels.
2. **Statistical Significance (Asymptotic $p \approx 9.85 \times 10^{-21}$; Empirical $p_{\text{boot}} < 0.0002$)**:
   - Validates that the $+7.06\%$ net edge over the historical drift baseline ($63.50\%$) is an authentic econometric phenomenon rather than data snooping, fold overfitting, or clustering bias.
   - Robust to date clustering ($B = 1, p = 5.27 \times 10^{-33}$), weekly persistence ($B = 5, p = 1.63 \times 10^{-27}$), and monthly persistence ($B = 21, p = 9.85 \times 10^{-21}$).
3. **Economic Profitability (Non-Tradability of the Gap)**:
   - In modern electronic equity markets, the opening gap is not a tradeable return.
   - Under National Stock Exchange regulations, opening prices are formed via a multilateral batch call auction between 09:00 and 09:08 IST. Orders submitted after overnight news arrives are cleared at the equilibrium opening price $P_{i,t}^{\text{open}}$.
   - No market participant can buy at yesterday's closing price $P_{i,t-1}^{\text{close}}$ after observing overnight international news. Thus, the $+7.06\%$ gap predictability cannot be captured as overnight capital growth.
4. **Tradability After Execution Costs**:
   - If an algorithmic system attempts to trade opening momentum by buying at open ($P_{i,t}^{\text{open}}$) and holding until close ($P_{i,t}^{\text{close}}$), it faces two insurmountable friction hurdles:
     - **Opening Microstructure Frictions**: The opening minutes of NSE trading exhibit wide bid-ask spreads (10 to 30 basis points in individual equities) and high market impact.
     - **Statutory Indian Transaction Costs**: Round-trip delivery turnover incurs 15 bps statutory friction (0.10% STT, exchange turnover charge, GST, and SEBI charges).
   - Extensive backtesting of open-to-close continuation strategies yielded net negative Sharpe ratios and negative compound annual growth rates across all ten equities.

---

## 8. Summary of Revisions to Project Artifacts

1. **Manuscript Markdown ([research_paper/research_paper_final.md](file:///C:/Users/vedan/Stock-Market-Predictor/research_paper/research_paper_final.md))**:
   - Section 14.3 updated: $H_1$ formally specified as two-sided ($P(n_{01}) \ne P(n_{10})$) to match the two-sided test statistic.
   - Section 16.9 updated: Standard-error inflation factor ($\text{SEIF} = 2.28\times$) and variance inflation factor ($\text{VIF} = 5.21\times$) strictly distinguished.
   - Section 16.9 updated: Asymptotic normal-approximation $p = 9.85 \times 10^{-21}$ and empirical bootstrap resolution limit $p_{\text{boot}} < 0.0002$ reported separately and reconciled.
   - Section 16.9 updated: Sample size hierarchy (27,179 stock-dates, 3,205 date clusters, 152 non-overlapping monthly blocks, $N_{\text{eff}} \approx 5,217$) explicitly distinguished.
   - Section 16.9 updated: Block-length selection framed as a methodological sensitivity analysis rather than a uniquely optimal window.
   - Section 16.9 updated: Formalized the *Four-Way Conceptual Separation* separating accuracy, significance, profitability, and tradability.
2. **Manuscript Generation Script ([scripts/build_full_manuscript.py](file:///C:/Users/vedan/Stock-Market-Predictor/scripts/build_full_manuscript.py))**:
   - Updated template strings in Section 14.3 and Section 16.9 to synchronize automated manuscript builds.
3. **Formal Word Document ([research_paper/Stock_Market_Predictor_Research_Paper.docx](file:///C:/Users/vedan/Stock-Market-Predictor/research_paper/Stock_Market_Predictor_Research_Paper.docx))**:
   - Recompiled successfully (2.82 MB, 402 paragraphs, 13 tables, 12 figures).
4. **Camera-Ready PDF ([research_paper/Stock_Market_Predictor_Research_Paper.pdf](file:///C:/Users/vedan/Stock-Market-Predictor/research_paper/Stock_Market_Predictor_Research_Paper.pdf))**:
   - Recompiled via headless browser layout engine (4.07 MB, 31 pages).
5. **Release Audit Script ([scripts/audit_release.py](file:///C:/Users/vedan/Stock-Market-Predictor/scripts/audit_release.py))**:
   - Re-audited and confirmed: 100% PASS across equations, claims, metadata, and document layout.

---

## Conclusion

This final robustness audit confirms that the out-of-fold overnight forecast accuracy of 70.57% and its net edge of +7.06% are **statistically robust to both cross-sectional co-movement and multi-week temporal dependence**.

While the naive McNemar $p$-value ($5.68 \times 10^{-100}$) was an artifact of independent-observation assumptions, the dependence-aware asymptotic normal-approximation $p$-value ($p = 9.85 \times 10^{-21}$ under 21-day block bootstrap) and empirical bootstrap resolution ($p_{\text{boot}} < 0.0002$) demonstrate that the econometric edge is genuine. The research paper now provides full statistical qualification and explicitly cautions against conflating econometric predictability with executable trading alpha.
