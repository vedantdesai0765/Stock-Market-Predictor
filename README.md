# 📈 Stock Price Movement Prediction using Machine Learning

## 📌 Project Overview
This project aims to predict **stock price movement (Up/Down)** using Machine Learning by analyzing historical stock market data and technical indicators.  
The system follows a **modular and scalable pipeline** including data collection, preprocessing, feature engineering, and model training.  

It is developed as a **B.Tech Final Year (Mega) Project** and demonstrates the application of **Machine Learning techniques in financial market analysis**.

---

## 🎯 Objectives
- Collect and analyze historical stock market data  
- Perform exploratory data analysis (EDA) to identify trends  
- Clean and preprocess raw time-series data  
- Apply feature engineering using technical indicators  
- Build and evaluate Machine Learning models  
- Predict next-day stock price movement (Up/Down)  
- Validate performance across multiple stocks  
- Prepare the system for deployment via an interactive dashboard  

---

## 🧠 Problem Statement
Stock market prices are volatile and influenced by multiple factors, making prediction challenging.  

This project focuses on predicting whether a stock’s price will **increase or decrease on the next trading day** using historical price data and derived technical indicators.

---

## ⚙️ Methodology

### 1️⃣ Data Collection
- Historical stock price data collected from **Yahoo Finance**
- Stocks implemented:
  - TCS (Tata Consultancy Services)
  - Reliance Industries
- Data stored in CSV format for reproducibility

---

### 2️⃣ Exploratory Data Analysis (EDA)
- Performed using Jupyter notebooks  
- Analysis includes:
  - Price trends  
  - Volume behavior  
  - Moving averages  
  - Volatility patterns  

---

### 3️⃣ Data Cleaning & Preprocessing
- Implemented as a reusable pipeline in `src/data_preprocessing.py`
- Steps include:
  - Date parsing and chronological sorting  
  - Removal of corrupted rows  
  - Handling missing values  
  - Structuring time-series data  
  - Saving cleaned output for downstream tasks  

---

### 4️⃣ Feature Engineering
- Implemented and validated in notebooks  
- Technical indicators used:
  - Simple Moving Average (SMA 20, SMA 50)  
  - Exponential Moving Average (EMA 20)  
  - Relative Strength Index (RSI 14)  
  - MACD & Signal Line  
  - Daily Returns  

- Binary Target Variable:
  - **1 → Price goes UP next day**
  - **0 → Price goes DOWN next day**

---

### 5️⃣ Model Training & Evaluation
- Implemented in `src/train_model.py`
- Key steps:
  - Time-series–aware train-test split (no shuffling)  
  - 80% training, 20% testing (chronological split)  
  - Training a baseline **Random Forest classifier**  
  - Evaluation using accuracy metric and classification report  

---

## 📊 Current Results

| Stock      | Accuracy |
|------------|----------|
| TCS        | ~50%     |
| Reliance   | ~49–50%  |

### 🔎 Observation
- Accuracy is close to random baseline (~50%)  
- Stock direction prediction is highly noisy  
- Technical indicators alone are weak predictors  
- This motivates integrating sentiment analysis in the next phase  

---

## 🧠 Technologies Used
- **Programming Language:** Python  
- **Data Analysis:** Pandas, NumPy  
- **Visualization (EDA):** Matplotlib, Seaborn  
- **Machine Learning:** Scikit-learn  
- **Deep Learning (Planned):** TensorFlow / Keras (LSTM)  
- **Data Source:** Yahoo Finance  
- **Dashboard (Planned):** Streamlit  

---

## 📂 Project Structure

## 📂 Project Structure

```text
Stock-Market-Predictor/
│
├── data/
│   ├── raw/                     # Raw stock market data
│   └── processed/               # Processed ML-ready datasets
│
├── notebooks/
│   ├── 01_data_collection.ipynb
│   ├── 02_eda.ipynb
│   ├── 03_feature_engineering.ipynb
│   └── 04_model_training.ipynb
│
├── src/
│   ├── data_preprocessing.py    # Data cleaning & preprocessing
│   └── train_model.py           # Model training & evaluation
│
├── models/                      # Saved trained models (ignored in Git)
│
├── app/                         # Streamlit dashboard (planned)
│
├── requirements.txt
├── .gitignore
└── README.md

---

### ▶️ How to Run the Project
1️⃣ Install Dependencies
- pip install -r requirements.txt
2️⃣ Run Data Preprocessing
- python src/data_preprocessing.py
3️⃣ Train the Machine Learning Model
- python src/train_model.py

## Current Project Status
✅ Completed
- Data collection (TCS & Reliance)
- Exploratory Data Analysis (EDA)
- Feature engineering using technical indicators
- Modular data preprocessing pipeline
- Baseline Machine Learning model training
- Multi-stock validation
- Clean Git workflow

## ⏳ In Progress / Planned
- Sentiment analysis integration
- Deep Learning model (LSTM)
- Hyperparameter tuning
- Streamlit-based interactive dashboard
- Final report and presentation

## 🚀 Future Work
- Merge financial news sentiment with stock data
- Compare ML vs LSTM performance
- Improve model generalization
- Deploy an interactive dashboard












