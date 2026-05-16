# 📈 Stock Trend Prediction using NLP and Deep Learning

## 📌 Overview
This project focuses on predicting stock market trends using a hybrid approach combining:

- Machine Learning
- Deep Learning
- NLP-based Sentiment Analysis
- Technical Indicators

The system analyzes historical stock prices and financial news to predict stock movement trends.

Developed as a **B.Tech Final Year Project**, this project demonstrates the application of AI techniques in financial market prediction.

---

# 🚀 Features Implemented

## ✅ Data Collection
- Historical stock data using Yahoo Finance API
- Financial news collection using News API

### Stocks Used
- TCS
- Reliance Industries

---

## ✅ Data Preprocessing
- Missing value handling
- Datetime conversion
- Dataset cleaning
- Time-series formatting

---

## ✅ Exploratory Data Analysis (EDA)
Performed analysis and visualization of:
- Stock price trends
- Volume trends
- Moving averages
- Historical market behavior

---

## ✅ Feature Engineering
Implemented technical indicators:
- SMA 20 & SMA 50
- EMA 20
- RSI
- MACD
- Daily Returns

---

# 🤖 Machine Learning

## Random Forest Classifier
Implemented a baseline ML model for stock movement prediction.

### Results
| Stock | Accuracy |
| TCS | ~53% |
| Reliance | ~50% |

---

# 🧠 Deep Learning Models

## LSTM (Long Short-Term Memory)
Implemented sequence-based stock prediction using TensorFlow/Keras.

### Result
- RMSE ≈ 0.0297

---

## GRU (Gated Recurrent Unit)
Implemented GRU-based sequential stock prediction model.

### Result
- RMSE ≈ 0.0219

### Observation
GRU achieved better performance than LSTM for the current dataset.

---

# 🧠 NLP Sentiment Analysis
Implemented VADER sentiment analysis on financial news headlines.

### Completed Work
- News preprocessing
- Sentiment score generation
- Sentiment-enhanced features
- Integration with LSTM and GRU models

---

# 🔗 Hybrid NLP + Deep Learning Pipeline

Financial News
      ↓
VADER Sentiment Analysis
      ↓
Sentiment Scores
      ↓
Feature Engineering
      ↓
LSTM / GRU Models
      ↓
Stock Trend Prediction

# 🧪 Technologies Used

## Programming
- Python

## Libraries
- Pandas
- NumPy
- Scikit-learn
- TensorFlow
- Keras
- Matplotlib
- Seaborn
- vaderSentiment

## Tools
- VS Code
- Jupyter Notebook
- GitHub

---

# 📂 Project Structure

```text
Stock-Market-Predictor/
│
├── data/
├── notebooks/
├── src/
├── models/
├── app/
├── requirements.txt
└── README.md
```

---

# ▶️ How to Run

## Install Dependencies

```bash
pip install -r requirements.txt
```

## Run Preprocessing

```bash
python src/data_preprocessing.py
```

## Train Baseline Model

```bash
python src/train_model.py
```

## Run Deep Learning Notebook

Open:

```text
notebooks/05_deep_learning_models.ipynb
```

---

# ✅ Current Progress

- Data collection completed
- Feature engineering completed
- Random Forest implemented
- LSTM implemented
- GRU implemented
- VADER sentiment analysis integrated
- Hybrid NLP + Deep Learning pipeline completed
- Comparative evaluation completed

---

# 🚀 Future Enhancements

- Streamlit dashboard
- FinBERT integration
- Multi-stock deep learning support
- Hyperparameter optimization
- Real-time prediction system

---

# 📌 Conclusion

This project successfully combines Machine Learning, Deep Learning, and NLP-based sentiment analysis for stock trend prediction. Comparative evaluation showed that GRU performed better than LSTM for the current dataset.
