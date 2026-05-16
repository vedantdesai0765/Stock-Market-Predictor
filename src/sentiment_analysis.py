import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from transformers import pipeline
import warnings
from tqdm import tqdm

warnings.filterwarnings("ignore")
tqdm.pandas()

def load_news_data(filepath):
    """Load raw news data."""
    return pd.read_csv(filepath)

def setup_analyzers():
    """Initialize VADER and FinBERT analyzers."""
    vader = SentimentIntensityAnalyzer()
    
    # Initialize FinBERT pipeline
    finbert = pipeline("sentiment-analysis", model="ProsusAI/finbert")
    return vader, finbert

def get_finbert_score(finbert_pipeline, text):
    """Map FinBERT labels to numerical scores."""
    try:
        # Truncate text to 512 tokens to avoid exceeding max length
        result = finbert_pipeline(str(text)[:512])[0]
        label = result['label']
        if label == 'positive':
            return 1
        elif label == 'negative':
            return -1
        else:
            return 0
    except Exception as e:
        print(f"Error processing text with FinBERT: {e}")
        return 0

def analyze_sentiment(df):
    """Apply VADER and FinBERT to the dataset."""
    print("Initializing models...")
    vader, finbert = setup_analyzers()
    
    # Combine title and description for richer context
    df['text'] = df['title'].fillna("") + " " + df['description'].fillna("")
    
    print("Calculating VADER sentiment...")
    df['vader_sentiment'] = df['text'].progress_apply(
        lambda x: vader.polarity_scores(str(x))['compound']
    )
    
    print("Calculating FinBERT sentiment...")
    df['finbert_sentiment'] = df['text'].progress_apply(
        lambda x: get_finbert_score(finbert, x)
    )
    
    return df

def aggregate_daily_sentiment(df):
    """Aggregate sentiment scores by date."""
    # Ensure date is datetime
    df['date'] = pd.to_datetime(df['date'])
    
    # Group by date and calculate the mean for the day
    daily_sentiment = df.groupby('date')[['vader_sentiment', 'finbert_sentiment']].mean().reset_index()
    
    return daily_sentiment

def run_sentiment_analysis(input_path, output_path):
    print(f"Loading data from {input_path}")
    df = load_news_data(input_path)
    
    df = analyze_sentiment(df)
    
    daily_df = aggregate_daily_sentiment(df)
    
    print(f"Saving aggregated sentiment to {output_path}")
    daily_df.to_csv(output_path, index=False)
    print("Done!")

if __name__ == "__main__":
    input_file = "data/raw/TCS_news_raw.csv"
    output_file = "data/processed/TCS_news_sentiment.csv"
    run_sentiment_analysis(input_file, output_file)
