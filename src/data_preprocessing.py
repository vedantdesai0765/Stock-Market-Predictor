import pandas as pd


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
    # Example usage
    input_file = "data/raw/TCS_raw.csv"
    output_file = "data/processed/TCS_processed.csv"
    run_preprocessing(input_file, output_file)
