"""
generate_paper_tables.py
Extracts and generates all 13 standalone academic tables in both CSV and Markdown formats
inside research_paper/tables/.
"""

import os
import pandas as pd

TABLES_DIR = "research_paper/tables"
os.makedirs(TABLES_DIR, exist_ok=True)

# -------------------------------------------------------------
# TABLE 1: Multi-Decade Walk-Forward Directional Prediction
# -------------------------------------------------------------
table1_data = [
    {"Ticker": "ADANIENT", "Data_Start": "2004-07", "Samples_N": 1773, "WalkForward_Accuracy": "0.5310 ± 0.0199", "Majority_Baseline": 0.5320, "Net_Edge": -0.0010, "Verdict": "NO SIGNAL"},
    {"Ticker": "ADANIGREEN", "Data_Start": "2018-06", "Samples_N": 619, "WalkForward_Accuracy": "0.4839 ± 0.0684", "Majority_Baseline": 0.4871, "Net_Edge": -0.0032, "Verdict": "NO SIGNAL"},
    {"Ticker": "ETERNAL", "Data_Start": "2021-07", "Samples_N": 375, "WalkForward_Accuracy": "0.5636 ± 0.0357", "Majority_Baseline": 0.5420, "Net_Edge": 0.0216, "Verdict": "NO SIGNAL"},
    {"Ticker": "HDFCBANK", "Data_Start": "2000-01", "Samples_N": 2150, "WalkForward_Accuracy": "0.5098 ± 0.0325", "Majority_Baseline": 0.5274, "Net_Edge": -0.0177, "Verdict": "NO SIGNAL"},
    {"Ticker": "INFY", "Data_Start": "2000-01", "Samples_N": 2154, "WalkForward_Accuracy": "0.5395 ± 0.0363", "Majority_Baseline": 0.5525, "Net_Edge": -0.0131, "Verdict": "NO SIGNAL"},
    {"Ticker": "ITC", "Data_Start": "2000-01", "Samples_N": 2151, "WalkForward_Accuracy": "0.5149 ± 0.0194", "Majority_Baseline": 0.4935, "Net_Edge": 0.0214, "Verdict": "NO SIGNAL"},
    {"Ticker": "RELIANCE", "Data_Start": "2000-01", "Samples_N": 2151, "WalkForward_Accuracy": "0.4972 ± 0.0121", "Majority_Baseline": 0.5186, "Net_Edge": -0.0214, "Verdict": "NO SIGNAL"},
    {"Ticker": "SBI", "Data_Start": "2000-01", "Samples_N": 2150, "WalkForward_Accuracy": "0.5153 ± 0.0344", "Majority_Baseline": 0.5395, "Net_Edge": -0.0242, "Verdict": "NO SIGNAL"},
    {"Ticker": "TCS", "Data_Start": "2004-08", "Samples_N": 1767, "WalkForward_Accuracy": "0.5070 ± 0.0292", "Majority_Baseline": 0.5149, "Net_Edge": -0.0079, "Verdict": "NO SIGNAL"},
    {"Ticker": "VEDL", "Data_Start": "2000-01", "Samples_N": 2146, "WalkForward_Accuracy": "0.4873 ± 0.0612", "Majority_Baseline": 0.5358, "Net_Edge": -0.0485, "Verdict": "NO SIGNAL"},
    {"Ticker": "POOLED / TOTAL", "Data_Start": "2000-2026", "Samples_N": 17436, "WalkForward_Accuracy": "0.5129 ± 0.0234", "Majority_Baseline": 0.5243, "Net_Edge": -0.0114, "Verdict": "10 / 10 NO SIGNAL"}
]
def df_to_markdown(df):
    headers = [str(c) for c in df.columns]
    header_line = "| " + " | ".join(headers) + " |"
    separator_line = "| " + " | ".join(["---"] * len(headers)) + " |"
    rows = []
    for _, row in df.iterrows():
        rows.append("| " + " | ".join(str(val).replace("\n", " ") for val in row) + " |")
    return "\n".join([header_line, separator_line] + rows)

df1 = pd.DataFrame(table1_data)
df1.to_csv(f"{TABLES_DIR}/table1_walk_forward_10stocks.csv", index=False)
with open(f"{TABLES_DIR}/table1_walk_forward_10stocks.md", "w", encoding="utf-8") as f:
    f.write("# Table 1: Multi-Decade Walk-Forward Directional Prediction Performance Across 10 NSE Equities (h=3 Days)\n\n")
    f.write(df_to_markdown(df1))

# -------------------------------------------------------------
# TABLE 2: Benchmark Comparison of Baseline Strategies vs Models
# -------------------------------------------------------------
table2_data = [
    {"Experimental_Setup": "INFY 3-Day Direction (h=3)", "Model": "Logistic Regression", "Nominal_Accuracy": "53.95%", "Majority_Baseline": "55.25%", "Persistence_Baseline": "50.12%", "Net_Edge": "-1.30%", "Verdict": "Negative Edge (Sub-baseline)"},
    {"Experimental_Setup": "RELIANCE 3-Day Direction (h=3)", "Model": "Logistic Regression", "Nominal_Accuracy": "49.72%", "Majority_Baseline": "51.86%", "Persistence_Baseline": "49.65%", "Net_Edge": "-2.14%", "Verdict": "Negative Edge (Sub-baseline)"},
    {"Experimental_Setup": "SENSEX Next-Day (h=1, Technical T)", "Model": "Logistic Regression", "Nominal_Accuracy": "51.54%", "Majority_Baseline": "53.11%", "Persistence_Baseline": "50.84%", "Net_Edge": "-1.57%", "Verdict": "Negative Edge (Sub-baseline)"},
    {"Experimental_Setup": "SENSEX Next-Day (h=1, T+G)", "Model": "Random Forest", "Nominal_Accuracy": "58.68%", "Majority_Baseline": "53.11%", "Persistence_Baseline": "51.20%", "Net_Edge": "+5.57%", "Verdict": "Positive Edge (Macro Spillover)"},
    {"Experimental_Setup": "SENSEX Big Moves (|Δ| > 1.5%)", "Model": "XGBoost (T+G+S)", "Nominal_Accuracy": "70.94%", "Majority_Baseline": "71.37%", "Persistence_Baseline": "58.20%", "Net_Edge": "-0.43%", "Verdict": "Negative Edge (Majority Trap)"},
    {"Experimental_Setup": "10-Day Moving Average Direction", "Model": "Walk-Forward GBDT", "Nominal_Accuracy": "86.43%", "Majority_Baseline": "54.20%", "Persistence_Baseline": "86.43%", "Net_Edge": "±0.00%", "Verdict": "Spurious Autocorrelation Trap"},
    {"Experimental_Setup": "Same-Day Headline Lookahead", "Model": "XGBoost (T+G+S0)", "Nominal_Accuracy": "66.20%", "Majority_Baseline": "53.12%", "Persistence_Baseline": "51.04%", "Net_Edge": "+13.08%", "Verdict": "Contemporaneous Lookahead Mirage"}
]
df2 = pd.DataFrame(table2_data)
df2.to_csv(f"{TABLES_DIR}/table2_baseline_benchmarks.csv", index=False)
with open(f"{TABLES_DIR}/table2_baseline_benchmarks.md", "w", encoding="utf-8") as f:
    f.write("# Table 2: Benchmark Comparison of Baseline Strategies vs. Trained Models Across Experimental Setups\n\n")
    f.write(df_to_markdown(df2))

# -------------------------------------------------------------
# TABLE 3: Performance Comparison Across Seven Architectures
# -------------------------------------------------------------
table3_data = [
    {"Architecture": "Logistic Regression", "Family": "Linear / Convex", "Feature_Set": "T", "Accuracy": "51.54%", "AUC": "0.5233", "Balanced_Acc": "50.65%", "Net_Edge": "-1.57%", "Sent_Delta": "+0.55%", "McNemar_p": "0.548 (n.s.)"},
    {"Architecture": "Random Forest", "Family": "Bagging Ensemble", "Feature_Set": "T", "Accuracy": "53.60%", "AUC": "0.5422", "Balanced_Acc": "50.19%", "Net_Edge": "+0.49%", "Sent_Delta": "-1.29%", "McNemar_p": "0.069 (n.s.)"},
    {"Architecture": "Gradient Boosting", "Family": "Boosting Ensemble", "Feature_Set": "T", "Accuracy": "52.71%", "AUC": "0.5344", "Balanced_Acc": "51.55%", "Net_Edge": "-0.40%", "Sent_Delta": "+0.37%", "McNemar_p": "0.690 (n.s.)"},
    {"Architecture": "XGBoost", "Family": "Boosting Ensemble", "Feature_Set": "T", "Accuracy": "53.02%", "AUC": "0.5367", "Balanced_Acc": "51.79%", "Net_Edge": "-0.09%", "Sent_Delta": "-0.34%", "McNemar_p": "0.708 (n.s.)"},
    {"Architecture": "LightGBM", "Family": "Boosting Ensemble", "Feature_Set": "T", "Accuracy": "52.43%", "AUC": "0.5343", "Balanced_Acc": "51.57%", "Net_Edge": "-0.68%", "Sent_Delta": "+0.12%", "McNemar_p": "0.914 (n.s.)"},
    {"Architecture": "PyTorch LSTM", "Family": "Recurrent Sequence", "Feature_Set": "T", "Accuracy": "51.91%", "AUC": "0.5215", "Balanced_Acc": "50.82%", "Net_Edge": "-1.20%", "Sent_Delta": "+0.25%", "McNemar_p": "0.837 (n.s.)"},
    {"Architecture": "PyTorch GRU", "Family": "Recurrent Sequence", "Feature_Set": "T", "Accuracy": "52.89%", "AUC": "0.5309", "Balanced_Acc": "51.14%", "Net_Edge": "-0.22%", "Sent_Delta": "-1.08%", "McNemar_p": "0.340 (n.s.)"},
    {"Architecture": "Logistic Regression", "Family": "Linear / Convex", "Feature_Set": "T+G", "Accuracy": "58.59%", "AUC": "0.6110", "Balanced_Acc": "58.45%", "Net_Edge": "+5.48%", "Sent_Delta": "-0.12%", "McNemar_p": "0.889 (n.s.)"},
    {"Architecture": "Random Forest", "Family": "Bagging Ensemble", "Feature_Set": "T+G", "Accuracy": "58.68%", "AUC": "0.6172", "Balanced_Acc": "58.36%", "Net_Edge": "+5.57%", "Sent_Delta": "+0.25%", "McNemar_p": "0.533 (n.s.)"},
    {"Architecture": "Gradient Boosting", "Family": "Boosting Ensemble", "Feature_Set": "T+G", "Accuracy": "57.42%", "AUC": "0.6068", "Balanced_Acc": "57.20%", "Net_Edge": "+4.31%", "Sent_Delta": "+1.05%", "McNemar_p": "0.130 (n.s.)"},
    {"Architecture": "XGBoost", "Family": "Boosting Ensemble", "Feature_Set": "T+G", "Accuracy": "57.36%", "AUC": "0.6053", "Balanced_Acc": "57.18%", "Net_Edge": "+4.25%", "Sent_Delta": "+0.99%", "McNemar_p": "0.160 (n.s.)"},
    {"Architecture": "LightGBM", "Family": "Boosting Ensemble", "Feature_Set": "T+G", "Accuracy": "57.64%", "AUC": "0.6048", "Balanced_Acc": "57.44%", "Net_Edge": "+4.53%", "Sent_Delta": "-0.40%", "McNemar_p": "0.605 (n.s.)"},
    {"Architecture": "PyTorch LSTM", "Family": "Recurrent Sequence", "Feature_Set": "T+G", "Accuracy": "58.13%", "AUC": "0.6140", "Balanced_Acc": "57.88%", "Net_Edge": "+5.02%", "Sent_Delta": "-0.37%", "McNemar_p": "0.695 (n.s.)"},
    {"Architecture": "PyTorch GRU", "Family": "Recurrent Sequence", "Feature_Set": "T+G", "Accuracy": "58.34%", "AUC": "0.6202", "Balanced_Acc": "58.10%", "Net_Edge": "+5.23%", "Sent_Delta": "+0.49%", "McNemar_p": "0.567 (n.s.)"}
]
df3 = pd.DataFrame(table3_data)
df3.to_csv(f"{TABLES_DIR}/table3_seven_architectures.csv", index=False)
with open(f"{TABLES_DIR}/table3_seven_architectures.md", "w", encoding="utf-8") as f:
    f.write("# Table 3: Performance Comparison Across Seven Architectures on SENSEX Next-Day Directional Forecasting\n\n")
    f.write(df_to_markdown(df3))

# -------------------------------------------------------------
# TABLE 4: Deep Learning Sequence Diagnostics (LSTM & GRU)
# -------------------------------------------------------------
table4_data = [
    {"Scope": "TCS (2004-2026)", "Architecture": "PyTorch LSTM", "Formulation": "3-Day Regressor", "Direction_Acc": "49.33%", "Net_Edge": "-2.46%", "RMSE_Model": "0.013117", "RMSE_Persistence": "0.018418", "RMSE_ZeroReturn": "0.013069", "Shrinkage_Factor": "0.0962", "Beats_Zero": False},
    {"Scope": "TCS (2004-2026)", "Architecture": "PyTorch GRU", "Formulation": "3-Day Regressor", "Direction_Acc": "51.24%", "Net_Edge": "-0.55%", "RMSE_Model": "0.013248", "RMSE_Persistence": "0.018418", "RMSE_ZeroReturn": "0.013069", "Shrinkage_Factor": "0.1196", "Beats_Zero": False},
    {"Scope": "RELIANCE (2000-2026)", "Architecture": "PyTorch LSTM", "Formulation": "3-Day Regressor", "Direction_Acc": "50.22%", "Net_Edge": "-1.64%", "RMSE_Model": "0.009109", "RMSE_Persistence": "0.012354", "RMSE_ZeroReturn": "0.008930", "Shrinkage_Factor": "0.1849", "Beats_Zero": False},
    {"Scope": "RELIANCE (2000-2026)", "Architecture": "PyTorch GRU", "Formulation": "3-Day Regressor", "Direction_Acc": "50.65%", "Net_Edge": "-1.21%", "RMSE_Model": "0.009133", "RMSE_Persistence": "0.012354", "RMSE_ZeroReturn": "0.008930", "Shrinkage_Factor": "0.1926", "Beats_Zero": False},
    {"Scope": "India SENSEX (Next-Day)", "Architecture": "PyTorch LSTM", "Formulation": "1-Day Classifier (T)", "Direction_Acc": "51.91%", "Net_Edge": "-1.20%", "RMSE_Model": "—", "RMSE_Persistence": "—", "RMSE_ZeroReturn": "—", "Shrinkage_Factor": "—", "Beats_Zero": "—"},
    {"Scope": "India SENSEX (Next-Day)", "Architecture": "PyTorch LSTM", "Formulation": "1-Day Classifier (T+S)", "Direction_Acc": "52.16%", "Net_Edge": "-0.95%", "RMSE_Model": "—", "RMSE_Persistence": "—", "RMSE_ZeroReturn": "—", "Shrinkage_Factor": "—", "Beats_Zero": "—"},
    {"Scope": "India SENSEX (Next-Day)", "Architecture": "PyTorch GRU", "Formulation": "1-Day Classifier (T)", "Direction_Acc": "52.89%", "Net_Edge": "-0.22%", "RMSE_Model": "—", "RMSE_Persistence": "—", "RMSE_ZeroReturn": "—", "Shrinkage_Factor": "—", "Beats_Zero": "—"},
    {"Scope": "India SENSEX (Next-Day)", "Architecture": "PyTorch GRU", "Formulation": "1-Day Classifier (T+S)", "Direction_Acc": "51.82%", "Net_Edge": "-1.29%", "RMSE_Model": "—", "RMSE_Persistence": "—", "RMSE_ZeroReturn": "—", "Shrinkage_Factor": "—", "Beats_Zero": "—"},
    {"Scope": "US S&P 500 (Next-Day)", "Architecture": "PyTorch LSTM", "Formulation": "1-Day Classifier (T)", "Direction_Acc": "53.34%", "Net_Edge": "-0.36%", "RMSE_Model": "—", "RMSE_Persistence": "—", "RMSE_ZeroReturn": "—", "Shrinkage_Factor": "—", "Beats_Zero": "—"},
    {"Scope": "US S&P 500 (Next-Day)", "Architecture": "PyTorch LSTM", "Formulation": "1-Day Classifier (T+S)", "Direction_Acc": "53.36%", "Net_Edge": "-0.34%", "RMSE_Model": "—", "RMSE_Persistence": "—", "RMSE_ZeroReturn": "—", "Shrinkage_Factor": "—", "Beats_Zero": "—"},
    {"Scope": "US S&P 500 (Next-Day)", "Architecture": "PyTorch GRU", "Formulation": "1-Day Classifier (T)", "Direction_Acc": "53.43%", "Net_Edge": "-0.27%", "RMSE_Model": "—", "RMSE_Persistence": "—", "RMSE_ZeroReturn": "—", "Shrinkage_Factor": "—", "Beats_Zero": "—"},
    {"Scope": "US S&P 500 (Next-Day)", "Architecture": "PyTorch GRU", "Formulation": "1-Day Classifier (T+S)", "Direction_Acc": "52.51%", "Net_Edge": "-1.19%", "RMSE_Model": "—", "RMSE_Persistence": "—", "RMSE_ZeroReturn": "—", "Shrinkage_Factor": "—", "Beats_Zero": "—"}
]
df4 = pd.DataFrame(table4_data)
df4.to_csv(f"{TABLES_DIR}/table4_deep_learning_diagnostics.csv", index=False)
with open(f"{TABLES_DIR}/table4_deep_learning_diagnostics.md", "w", encoding="utf-8") as f:
    f.write("# Table 4: Deep Learning Sequence Model Performance and Diagnostic Failure Analysis\n\n")
    f.write(df_to_markdown(df4))

# -------------------------------------------------------------
# TABLE 5: Entity-Targeted FinBERT vs Generic FinBERT on SEntFiN
# -------------------------------------------------------------
table5_data = [
    {"Split": "Full SEntFiN Test Set", "Samples_N": 1440, "Metric": "Classification Accuracy", "Generic_FinBERT": "68.54%", "Entity_FinBERT": "87.71%", "Gain": "+19.17 pp"},
    {"Split": "Full SEntFiN Test Set", "Samples_N": 1440, "Metric": "Macro-Averaged F1", "Generic_FinBERT": "0.6870", "Entity_FinBERT": "0.8783", "Gain": "+0.1913"},
    {"Split": "Multi-Entity Headlines (>= 2 firms)", "Samples_N": 614, "Metric": "Classification Accuracy", "Generic_FinBERT": "61.56%", "Entity_FinBERT": "85.34%", "Gain": "+23.78 pp"},
    {"Split": "Multi-Entity Headlines (>= 2 firms)", "Samples_N": 614, "Metric": "Macro-Averaged F1", "Generic_FinBERT": "0.6122", "Entity_FinBERT": "0.8527", "Gain": "+0.2405"}
]
df5 = pd.DataFrame(table5_data)
df5.to_csv(f"{TABLES_DIR}/table5_entity_finbert_sentfin.csv", index=False)
with open(f"{TABLES_DIR}/table5_entity_finbert_sentfin.md", "w", encoding="utf-8") as f:
    f.write("# Table 5: Entity-Targeted FinBERT vs. Standard Generic FinBERT on SEntFiN 1.0 Benchmark\n\n")
    f.write(df_to_markdown(df5))

# -------------------------------------------------------------
# TABLE 6: Company-Targeted Sentiment Ablation Across 10 Equities
# -------------------------------------------------------------
df6 = pd.read_csv('reports/company_sentiment_ablation.csv')
df6.to_csv(f"{TABLES_DIR}/table6_sentiment_ablation.csv", index=False)
with open(f"{TABLES_DIR}/table6_sentiment_ablation.md", "w", encoding="utf-8") as f:
    f.write("# Table 6: Company-Targeted Sentiment Ablation Across 10 Equities (Times of India & NSE Filings)\n\n")
    f.write(df_to_markdown(df6))

# -------------------------------------------------------------
# TABLE 7: Lead-Lag Rank Correlation Profile (Spearman rho)
# -------------------------------------------------------------
table7_data = [
    {"Scope": "SENSEX Index", "Corpus_Filter": "All Headlines (Raw)", "Lag_minus1": "+0.1152", "Lag_0_SameDay": "+0.1720", "Lag_plus1_NextDay": "+0.0205", "SameDay_Significance": "p < 10^-14", "NextDay_Significance": "p = 0.143 (n.s.)"},
    {"Scope": "SENSEX Index", "Corpus_Filter": "Price Reports Removed", "Lag_minus1": "+0.0573", "Lag_0_SameDay": "+0.0562", "Lag_plus1_NextDay": "+0.0070", "SameDay_Significance": "p = 0.00006", "NextDay_Significance": "p = 0.618 (n.s.)"},
    {"Scope": "TCS", "Corpus_Filter": "Price Reports Removed", "Lag_minus1": "+0.0514", "Lag_0_SameDay": "+0.0340", "Lag_plus1_NextDay": "+0.0169", "SameDay_Significance": "p = 0.295", "NextDay_Significance": "p = 0.602 (n.s.)"},
    {"Scope": "RELIANCE", "Corpus_Filter": "Price Reports Removed", "Lag_minus1": "+0.0428", "Lag_0_SameDay": "+0.0161", "Lag_plus1_NextDay": "-0.0225", "SameDay_Significance": "p = 0.463", "NextDay_Significance": "p = 0.304 (n.s.)"},
    {"Scope": "INFY", "Corpus_Filter": "Price Reports Removed", "Lag_minus1": "+0.0621", "Lag_0_SameDay": "+0.0513", "Lag_plus1_NextDay": "+0.0330", "SameDay_Significance": "p = 0.038", "NextDay_Significance": "p = 0.181 (n.s.)"},
    {"Scope": "HDFCBANK", "Corpus_Filter": "Price Reports Removed", "Lag_minus1": "+0.0210", "Lag_0_SameDay": "+0.0368", "Lag_plus1_NextDay": "+0.0123", "SameDay_Significance": "p = 0.515", "NextDay_Significance": "p = 0.828 (n.s.)"},
    {"Scope": "SBI", "Corpus_Filter": "Price Reports Removed", "Lag_minus1": "+0.0380", "Lag_0_SameDay": "+0.0180", "Lag_plus1_NextDay": "+0.0230", "SameDay_Significance": "p = 0.492", "NextDay_Significance": "p = 0.378 (n.s.)"},
    {"Scope": "ITC", "Corpus_Filter": "Price Reports Removed", "Lag_minus1": "+0.0194", "Lag_0_SameDay": "-0.0106", "Lag_plus1_NextDay": "-0.0565", "SameDay_Significance": "p = 0.838", "NextDay_Significance": "p = 0.274 (n.s.)"},
    {"Scope": "ADANIENT", "Corpus_Filter": "Price Reports Removed", "Lag_minus1": "+0.0490", "Lag_0_SameDay": "+0.0643", "Lag_plus1_NextDay": "+0.1257", "SameDay_Significance": "p = 0.160", "NextDay_Significance": "p = 0.006"},
    {"Scope": "VEDL", "Corpus_Filter": "Price Reports Removed", "Lag_minus1": "+0.0312", "Lag_0_SameDay": "+0.0462", "Lag_plus1_NextDay": "+0.0217", "SameDay_Significance": "p = 0.353", "NextDay_Significance": "p = 0.662 (n.s.)"},
    {"Scope": "POOLED 10 STOCKS", "Corpus_Filter": "Price Reports Removed", "Lag_minus1": "+0.0415", "Lag_0_SameDay": "+0.0324", "Lag_plus1_NextDay": "+0.0131", "SameDay_Significance": "p = 0.0044", "NextDay_Significance": "p = 0.248 (n.s.)"}
]
df7 = pd.DataFrame(table7_data)
df7.to_csv(f"{TABLES_DIR}/table7_lead_lag_correlations.csv", index=False)
with open(f"{TABLES_DIR}/table7_lead_lag_correlations.md", "w", encoding="utf-8") as f:
    f.write("# Table 7: Lead-Lag Rank Correlation Profile (Spearman rho) Between News Sentiment and Equity Returns\n\n")
    f.write(df_to_markdown(df7))

# -------------------------------------------------------------
# TABLE 8: Quintile Spreads on BSE SENSEX
# -------------------------------------------------------------
table8_data = [
    {"Quintile": "Quintile 1 (Most Negative)", "Sessions_N": 1024, "SameDay_Mean_Return": "-0.0564%", "NextDay_Mean_Return": "+0.0689%"},
    {"Quintile": "Quintile 2", "Sessions_N": 1023, "SameDay_Mean_Return": "+0.0179%", "NextDay_Mean_Return": "-0.0262%"},
    {"Quintile": "Quintile 3 (Neutral)", "Sessions_N": 1024, "SameDay_Mean_Return": "+0.0475%", "NextDay_Mean_Return": "+0.0875%"},
    {"Quintile": "Quintile 4", "Sessions_N": 1023, "SameDay_Mean_Return": "+0.1292%", "NextDay_Mean_Return": "+0.1089%"},
    {"Quintile": "Quintile 5 (Most Positive)", "Sessions_N": 1024, "SameDay_Mean_Return": "+0.1434%", "NextDay_Mean_Return": "+0.0343%"},
    {"Quintile": "Spread: Quintile 5 - Quintile 1", "Sessions_N": "—", "SameDay_Mean_Return": "+0.1998% / day", "NextDay_Mean_Return": "-0.0345% / day"}
]
df8 = pd.DataFrame(table8_data)
df8.to_csv(f"{TABLES_DIR}/table8_sentiment_quintiles.csv", index=False)
with open(f"{TABLES_DIR}/table8_sentiment_quintiles.md", "w", encoding="utf-8") as f:
    f.write("# Table 8: Daily Mean Returns Across Clean News Sentiment Quintiles on the BSE SENSEX\n\n")
    f.write(df_to_markdown(df8))

# -------------------------------------------------------------
# TABLE 9: Explanatory Coverage of Extreme Price Movements
# -------------------------------------------------------------
table9_data = [
    {"Scope": "SENSEX Index", "Big_Moves_N": 262, "Matching_News": "45.80% (120)", "Opposite_News": "39.31% (103)", "Neutral_News": "8.78% (23)", "No_News_Unexplained": "6.11% (16)"},
    {"Scope": "TCS", "Big_Moves_N": 217, "Matching_News": "11.06% (24)", "Opposite_News": "7.83% (17)", "Neutral_News": "15.67% (34)", "No_News_Unexplained": "65.44% (142)"},
    {"Scope": "RELIANCE", "Big_Moves_N": 263, "Matching_News": "12.93% (34)", "Opposite_News": "9.89% (26)", "Neutral_News": "24.34% (64)", "No_News_Unexplained": "52.85% (139)"},
    {"Scope": "INFY", "Big_Moves_N": 263, "Matching_News": "14.83% (39)", "Opposite_News": "6.08% (16)", "Neutral_News": "16.35% (43)", "No_News_Unexplained": "62.74% (165)"},
    {"Scope": "HDFCBANK", "Big_Moves_N": 263, "Matching_News": "1.52% (4)", "Opposite_News": "0.76% (2)", "Neutral_News": "0.76% (2)", "No_News_Unexplained": "96.96% (255)"},
    {"Scope": "SBI", "Big_Moves_N": 263, "Matching_News": "4.56% (12)", "Opposite_News": "4.56% (12)", "Neutral_News": "18.25% (48)", "No_News_Unexplained": "72.62% (191)"},
    {"Scope": "ITC", "Big_Moves_N": 263, "Matching_News": "2.28% (6)", "Opposite_News": "3.80% (10)", "Neutral_News": "2.66% (7)", "No_News_Unexplained": "91.25% (240)"},
    {"Scope": "ADANIENT", "Big_Moves_N": 219, "Matching_News": "3.65% (8)", "Opposite_News": "2.74% (6)", "Neutral_News": "5.94% (13)", "No_News_Unexplained": "87.67% (192)"},
    {"Scope": "VEDL", "Big_Moves_N": 263, "Matching_News": "2.66% (7)", "Opposite_News": "1.14% (3)", "Neutral_News": "2.66% (7)", "No_News_Unexplained": "93.54% (246)"}
]
df9 = pd.DataFrame(table9_data)
df9.to_csv(f"{TABLES_DIR}/table9_big_moves_coverage.csv", index=False)
with open(f"{TABLES_DIR}/table9_big_moves_coverage.md", "w", encoding="utf-8") as f:
    f.write("# Table 9: Explanatory Coverage of Extreme Price Movements (|Δ| > 1.5%)\n\n")
    f.write(df_to_markdown(df9))

# -------------------------------------------------------------
# TABLE 10: Controlled Deconstruction of Accuracy Pitfalls
# -------------------------------------------------------------
table10_data = [
    {"Protocol": "Honest Out-of-Sample Walk-Forward (T+G+S)", "Valid_Setup": "Yes", "Reported_Metric": "Direction Accuracy", "Nominal_Value": "57.24%", "Mechanism": "Rigorous expanding walk-forward CV, non-overlapping labels, strict embargo."},
    {"Protocol": "Same-Day Headline Polarity vs. Return", "Valid_Setup": "No", "Reported_Metric": "Direction Accuracy", "Nominal_Value": "76.96%", "Mechanism": "Contemporaneous Lookahead: FinBERT scores headlines describing today's close (N=2,270)."},
    {"Protocol": "Next-Day Implied by Same Headline Rule", "Valid_Setup": "Yes", "Reported_Metric": "Direction Accuracy", "Nominal_Value": "55.02%", "Mechanism": "Proper alignment resets accuracy to unconditional base rate (~53%)."},
    {"Protocol": "5-Day Return Horizon with Shuffled 5-Fold CV", "Valid_Setup": "No", "Reported_Metric": "Direction Accuracy", "Nominal_Value": "60.95%", "Mechanism": "Data Leakage: Random k-fold CV causes adjacent days to share 4 of 5 return days."},
    {"Protocol": "5-Day Return with Walk-Forward & Embargo", "Valid_Setup": "Yes", "Reported_Metric": "Direction Accuracy", "Nominal_Value": "55.29%", "Mechanism": "Embargo prevents boundary leakage; accuracy drops to majority base (~56%)."},
    {"Protocol": "10-Day Moving Average Direction (Walk-Forward)", "Valid_Setup": "No", "Reported_Metric": "Direction Accuracy", "Nominal_Value": "86.43%", "Mechanism": "Trivial Target: Predicting SMA10(t+1) > SMA10(t) when 9 of 10 prices are already known."},
    {"Protocol": "Next-Day Price Level Linear Regression", "Valid_Setup": "No", "Reported_Metric": "Coefficient (R^2)", "Nominal_Value": "0.9980", "Mechanism": "Non-Stationarity: Regressing Pt+1 on Pt yields near-unity fit due to integrated unit-root drift."},
    {"Protocol": "Next-Day Price Level Linear Regression", "Valid_Setup": "No", "Reported_Metric": "Metric (1 - MAPE)", "Nominal_Value": "0.9925", "Mechanism": "Reported as '99.2% accuracy'; simply reflects that tomorrow's price is close to today's price."},
    {"Protocol": "Direction Implied by Level Regression", "Valid_Setup": "Yes", "Reported_Metric": "Direction Accuracy", "Nominal_Value": "52.54%", "Mechanism": "The level model that achieved 99.2% 1-MAPE has zero directional predictive power."}
]
df10 = pd.DataFrame(table10_data)
df10.to_csv(f"{TABLES_DIR}/table10_accuracy_pitfalls_deconstruction.csv", index=False)
with open(f"{TABLES_DIR}/table10_accuracy_pitfalls_deconstruction.md", "w", encoding="utf-8") as f:
    f.write("# Table 10: Controlled Demonstration of Literature Accuracy Pitfalls on Identical Market Data\n\n")
    f.write(df_to_markdown(df10))

# -------------------------------------------------------------
# TABLE 11: Overnight Forecast Performance by Equity (N=27,179)
# -------------------------------------------------------------
df_f = pd.read_csv('reports/forecast/stock_summary.csv')
# Filter LightGBM next morning
df_f_lgb = df_f[(df_f['model'] == 'LightGBM') & (df_f['target'] == "Next morning's opening price")].copy()
table11_data = []
for _, row in df_f_lgb.iterrows():
    table11_data.append({
        "Ticker": row['ticker'],
        "Samples_N": int(row['n']),
        "Period": f"{int(row['first_year'])}-{int(row['last_year'])}",
        "Guess_Baseline": f"{row['acc_guess']*100:.2f}%",
        "Model_Accuracy": f"{row['acc_model']*100:.2f}%",
        "With_News_Accuracy": f"{row['acc_news']*100:.2f}%",
        "Net_Edge": f"{row['edge']*100:+.2f}%",
        "p_vs_Guess": f"{row['p_vs_guess']:.4f}",
        "News_Effect": f"{row['news_effect']*100:+.2f}%",
        "p_News": f"{row['p_news']:.4f}"
    })
df11 = pd.DataFrame(table11_data)
df11.to_csv(f"{TABLES_DIR}/table11_overnight_opening_forecast.csv", index=False)
with open(f"{TABLES_DIR}/table11_overnight_opening_forecast.md", "w", encoding="utf-8") as f:
    f.write("# Table 11: Next-Morning Opening Price Forecast Performance Across Indian Equities (LightGBM, N=27,179)\n\n")
    f.write(df_to_markdown(df11))

# -------------------------------------------------------------
# TABLE 12: Strategy Variants Comparison Across 10 Equities
# -------------------------------------------------------------
df_strat = pd.read_csv('reports/actionable_signal_results.csv')
v_means = df_strat.groupby('variant')[['cagr_strat', 'cagr_bh', 'sharpe_strat', 'sharpe_bh', 'max_dd_strat', 'max_dd_bh', 'win_rate', 'profit_factor', 'total_trades', 'time_in_market']].mean().reset_index()

table12_data = []
# Add Buy & Hold baseline row
bh_cagr = v_means['cagr_bh'].iloc[0] * 100
bh_sharpe = v_means['sharpe_bh'].iloc[0]
bh_dd = v_means['max_dd_bh'].iloc[0] * 100
table12_data.append({
    "Strategy": "Buy & Hold Benchmark",
    "Description": "100% passive equity exposure across all market regimes",
    "CAGR": f"{bh_cagr:.2f}%",
    "Sharpe": f"{bh_sharpe:.3f}",
    "Max_Drawdown": f"{bh_dd:.2f}%",
    "Win_Rate": "—",
    "Profit_Factor": "—",
    "Avg_Trades": "1.0",
    "Time_in_Market": "100.0%"
})

desc_map = {
    'A': "Baseline Long-Only: Enters when P(Up) > 50% & net edge exceeds 15 bps friction",
    'B': "Risk-Managed Long-Only: 2% trailing stop-loss, 5-day mandatory exit",
    'C': "Dynamic Long/Short: Active long and short participation based on conviction",
    'D': "Multi-Factor Consensus: Requires ML probability, RSI/MACD, and FinBERT alignment",
    'E': "Regime-Adaptive: Active allocation only under favorable volatility regimes"
}

for _, row in v_means.iterrows():
    v = row['variant']
    table12_data.append({
        "Strategy": f"Variant {v}",
        "Description": desc_map[v],
        "CAGR": f"{row['cagr_strat']*100:.2f}%",
        "Sharpe": f"{row['sharpe_strat']:.3f}",
        "Max_Drawdown": f"{row['max_dd_strat']*100:.2f}%",
        "Win_Rate": f"{row['win_rate']*100:.2f}%",
        "Profit_Factor": f"{row['profit_factor']:.2f}",
        "Avg_Trades": f"{row['total_trades']:.1f}",
        "Time_in_Market": f"{row['time_in_market']*100:.2f}%"
    })
df12 = pd.DataFrame(table12_data)
df12.to_csv(f"{TABLES_DIR}/table12_actionable_strategies.csv", index=False)
with open(f"{TABLES_DIR}/table12_actionable_strategies.md", "w", encoding="utf-8") as f:
    f.write("# Table 12: Actionable Trading Strategy Variants Comparison Across 10 Equities (Net of 15 bps Friction)\n\n")
    f.write(df_to_markdown(df12))

# -------------------------------------------------------------
# TABLE 13: Summary of Methodological Disclosures & Limitations
# -------------------------------------------------------------
table13_data = [
    {"Dimension": "Target Horizon & Subsampling", "Methodological_Implementation": "h=3 non-overlapping subsampling (step=3)", "Econometric_Purpose": "Eliminates moving-average serial target autocorrelation (rho drops from 0.81 to -0.04)."},
    {"Dimension": "Walk-Forward Embargo Buffer", "Methodological_Implementation": "52 trading days (17 subsampled rows) dropped at train seams", "Econometric_Purpose": "Purges forward label overlap and decorrelates transient errors across train/test seams."},
    {"Dimension": "Permutation Testing Resolution", "Methodological_Implementation": "n=6 shuffles per fold with Laplace correction (p_min = 1/7 = 0.1429)", "Econometric_Purpose": "Bounded computational diagnostic screening feature leakage; not a high-resolution hypothesis test."},
    {"Dimension": "Cross-Sectional Panel Dependence", "Methodological_Implementation": "Concurrent NSE trading across all 10 equities (2000-2026)", "Econometric_Purpose": "Effective degrees of freedom governed by time series length (~2,154 periods), not pooled 17,636 rows."},
    {"Dimension": "News Corpora Coverage Window", "Methodological_Implementation": "ToI archive (2001-2022), NSE filings (2008-2025), Cutoff Dec 31, 2025", "Econometric_Purpose": "2026 trading sessions evaluated under neutral sentiment; news does not explain 2026 volatility."},
    {"Dimension": "Indian Transaction Friction Hurdle", "Methodological_Implementation": "15 basis points round-trip (STT 0.10%, GST, SEBI fee, slippage)", "Econometric_Purpose": "Enforces realistic break-even hurdle; gross edge must exceed 15 bps to yield net positive CAGR."},
    {"Dimension": "Cash Allocation & CAGR", "Methodological_Implementation": "Uninvested capital held in cash (assumed 0% risk-free rate)", "Econometric_Purpose": "Low time-in-market strategies (Variants B, D, E) exhibit lower nominal CAGR due to cash drag."}
]
df13 = pd.DataFrame(table13_data)
df13.to_csv(f"{TABLES_DIR}/table13_methodological_disclosures.csv", index=False)
with open(f"{TABLES_DIR}/table13_methodological_disclosures.md", "w", encoding="utf-8") as f:
    f.write("# Table 13: Summary of Methodological Disclosures and Research Limitations\n\n")
    f.write(df_to_markdown(df13))

print("All 13 standalone tables generated in CSV and Markdown formats.")
