# src/data_preprocessing.py
import pandas as pd
def merge_sentiment_features(features_path, sentiment_path, output_path, trim_to_news=True):
    """
    Merge stock features with daily sentiment scores.
    Forward-fill sentiment to handle weekends or days without news.
    """
    # Load features and sentiment
    features_df = pd.read_csv(features_path)
    sentiment_df = pd.read_csv(sentiment_path)
    
    # Ensure date columns are datetime
    features_df['Date'] = pd.to_datetime(features_df['Date'])
    sentiment_df['date'] = pd.to_datetime(sentiment_df['date'])
    
    if trim_to_news and not sentiment_df.empty:
        min_date = sentiment_df['date'].min()
        max_date = sentiment_df['date'].max()
        features_df = features_df[(features_df['Date'] >= min_date) & (features_df['Date'] <= max_date)]
        print(f"Trimming stock features to match news date range: {min_date.date()} to {max_date.date()}")
        
        if features_df.empty:
            print("WARNING: After trimming, the stock features dataset is empty!")
            print("This means there is no date overlap between your stock data and news data.")
            print("Please ensure your news dataset covers the trading period.")
            print("Returning without overwriting the dataset.")
            return features_df
    
    # Merge left on features dates
    merged_df = pd.merge(features_df, sentiment_df, left_on='Date', right_on='date', how='left')
    
    # Drop the redundant 'date' column
    if 'date' in merged_df.columns:
        merged_df = merged_df.drop('date', axis=1)
        
    # Forward fill sentiment scores for days without news, then fill remaining with 0
    merged_df['vader_sentiment'] = merged_df['vader_sentiment'].ffill().fillna(0)
    merged_df['finbert_sentiment'] = merged_df['finbert_sentiment'].ffill().fillna(0)
    
    # Save the merged dataset
    merged_df.to_csv(output_path, index=False)
    print(f"Saved merged dataset to {output_path}")
    return merged_df
    
def load_raw_data(filepath):
    """
    Load raw stock market data from a CSV file.
    """
    df = pd.read_csv(filepath)
    return df


def preprocess_data(df):
    """
    Clean and preprocess stock market data.
    - Convert Date to datetime
    - Sort by Date
    - Remove missing values
    - Set Date as index
    """
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date')
    df = df.dropna()
    df.set_index('Date', inplace=True)
    return df


def run_preprocessing(input_path, output_path):
    """
    End-to-end preprocessing pipeline.
    Reads raw data, preprocesses it, and saves processed data.
    """
    df = load_raw_data(input_path)
    df = preprocess_data(df)
    df.to_csv(output_path)


if __name__ == "__main__":
    # Example usage for merging sentiment
    features_file = "data/processed/TCS_features.csv"
    sentiment_file = "data/processed/TCS_news_sentiment.csv"
    merged_output = "data/processed/TCS_features_sentiment.csv"
    
    merge_sentiment_features(features_file, sentiment_file, merged_output)
