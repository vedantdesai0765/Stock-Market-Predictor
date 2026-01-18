import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report, confusion_matrix


def load_data(filepath):
    """
    Load processed stock data from CSV
    """
    df = pd.read_csv(filepath)
    return df


def create_target(df):
    """
    Create target variable:
    1 -> Next day's close price is higher (UP)
    0 -> Next day's close price is lower (DOWN)
    """
    df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
    df = df.dropna()
    return df


def train_model(df):
    """
    Train Random Forest classifier on numeric features only
    """

    # Drop target and non-numeric columns like Date
    X = df.drop(['Target', 'Date'], axis=1, errors='ignore')
    X = X.select_dtypes(include=['number'])

    y = df['Target']

    # Time-series aware split (no shuffling)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )

    # Model
    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )

    # Train
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    # Week-1 evaluation metrics (baseline verification)
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    accuracy = accuracy_score(y_test, y_pred)

    print(f"Model Accuracy: {accuracy:.2f}")

    return model


if __name__ == "__main__":
    data_path = "data/processed/TCS_processed.csv"
    model_path = "models/random_forest_model.pkl"

    # Load and prepare data
    df = load_data(data_path)
    df = create_target(df)

    # Train model
    model = train_model(df)

    # Save model
    joblib.dump(model, model_path)
    print("Model saved successfully.")
