# Table 12: Actionable Trading Strategy Variants Comparison Across 10 Equities (Net of 15 bps Friction)

| Strategy | Description | CAGR | Sharpe | Max_Drawdown | Win_Rate | Profit_Factor | Avg_Trades | Time_in_Market |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Buy & Hold Benchmark | 100% passive equity exposure across all market regimes | 15.68% | 0.311 | 54.77% | — | — | 1.0 | 100.0% |
| Variant A | Baseline Long-Only: Enters when P(Up) > 50% & net edge exceeds 15 bps friction | 12.82% | 0.239 | 35.44% | 53.49% | 1.61 | 107.0 | 37.51% |
| Variant B | Risk-Managed Long-Only: 2% trailing stop-loss, 5-day mandatory exit | 6.75% | 0.098 | 20.25% | 52.15% | 1.25 | 59.5 | 10.92% |
| Variant C | Dynamic Long/Short: Active long and short participation based on conviction | 13.24% | 0.268 | 31.70% | 55.16% | 1.56 | 100.8 | 29.35% |
| Variant D | Multi-Factor Consensus: Requires ML probability, RSI/MACD, and FinBERT alignment | 6.79% | 0.111 | 17.93% | 51.94% | 1.25 | 57.0 | 10.52% |
| Variant E | Regime-Adaptive: Active allocation only under favorable volatility regimes | 7.46% | 0.216 | 10.94% | 53.24% | 2.66 | 40.2 | 6.98% |
