import pandas as pd

def load_raw_data(filepath):
    """
    Load raw stock market data from CSV
    """
    return pd.read_csv(filepath)


def preprocess_data(df):
    """
    Clean and preprocess stock market data
    """
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date')
    df = df.dropna()
    df.set_index('Date', inplace=True)
    return df


def run_preprocessing(input_path, output_path):
    """
    End-to-end preprocessing pipeline
    """
    df = load_raw_data(input_path)
    df = preprocess_data(df)
    df.to_csv(output_path)
