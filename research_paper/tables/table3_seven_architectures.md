# Table 3: Performance Comparison Across Seven Architectures on SENSEX Next-Day Directional Forecasting

| Architecture | Family | Feature_Set | Accuracy | AUC | Balanced_Acc | Net_Edge | Sent_Delta | McNemar_p |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Logistic Regression | Linear / Convex | T | 51.54% | 0.5233 | 50.65% | -1.57% | +0.55% | 0.548 (n.s.) |
| Random Forest | Bagging Ensemble | T | 53.60% | 0.5422 | 50.19% | +0.49% | -1.29% | 0.069 (n.s.) |
| Gradient Boosting | Boosting Ensemble | T | 52.71% | 0.5344 | 51.55% | -0.40% | +0.37% | 0.690 (n.s.) |
| XGBoost | Boosting Ensemble | T | 53.02% | 0.5367 | 51.79% | -0.09% | -0.34% | 0.708 (n.s.) |
| LightGBM | Boosting Ensemble | T | 52.43% | 0.5343 | 51.57% | -0.68% | +0.12% | 0.914 (n.s.) |
| PyTorch LSTM | Recurrent Sequence | T | 51.91% | 0.5215 | 50.82% | -1.20% | +0.25% | 0.837 (n.s.) |
| PyTorch GRU | Recurrent Sequence | T | 52.89% | 0.5309 | 51.14% | -0.22% | -1.08% | 0.340 (n.s.) |
| Logistic Regression | Linear / Convex | T+G | 58.59% | 0.6110 | 58.45% | +5.48% | -0.12% | 0.889 (n.s.) |
| Random Forest | Bagging Ensemble | T+G | 58.68% | 0.6172 | 58.36% | +5.57% | +0.25% | 0.533 (n.s.) |
| Gradient Boosting | Boosting Ensemble | T+G | 57.42% | 0.6068 | 57.20% | +4.31% | +1.05% | 0.130 (n.s.) |
| XGBoost | Boosting Ensemble | T+G | 57.36% | 0.6053 | 57.18% | +4.25% | +0.99% | 0.160 (n.s.) |
| LightGBM | Boosting Ensemble | T+G | 57.64% | 0.6048 | 57.44% | +4.53% | -0.40% | 0.605 (n.s.) |
| PyTorch LSTM | Recurrent Sequence | T+G | 58.13% | 0.6140 | 57.88% | +5.02% | -0.37% | 0.695 (n.s.) |
| PyTorch GRU | Recurrent Sequence | T+G | 58.34% | 0.6202 | 58.10% | +5.23% | +0.49% | 0.567 (n.s.) |
