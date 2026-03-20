import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib
import os

# ── CONSTANTS ────────────────────────────────────────────────────────────────
NUMERIC_COLS = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
CATEGORICAL_COLS = ['cp', 'restecg', 'slope', 'ca', 'thal']
IMPUTE_MODA_COLS = ['restecg', 'thalach', 'exang', 'fbs', 'oldpeak', 'slope', 'thal', 'ca']


# ── FUNCTIONS ─────────────────────────────────────────────────────────────────
def fix_impossible_zeros(df):
    """Replaces medically impossible zeros with NaN in chol and trestbps."""
    df = df.copy()
    df['chol'] = df['chol'].replace(0, np.nan)
    df['trestbps'] = df['trestbps'].replace(0, np.nan)
    return df


def impute_nulls(df):
    """Imputes null values with mean for continuous variables and mode for categorical variables."""
    df = df.copy()
    df['chol'] = df['chol'].fillna(df['chol'].mean())
    df['trestbps'] = df['trestbps'].fillna(df['trestbps'].mean())
    for col in IMPUTE_MODA_COLS:
        df[col] = df[col].fillna(df[col].mode()[0])
    return df


def fix_outliers(df):
    """Fixes clinically impossible outliers."""
    df = df.copy()
    df['oldpeak'] = df['oldpeak'].clip(lower=0)
    return df


def encode_categoricals(df):
    """Applies One-Hot Encoding to nominal categorical variables."""
    df = df.copy()
    df = pd.get_dummies(df, columns=CATEGORICAL_COLS, drop_first=True, dtype=int)
    return df


def scale_numerics(df, scaler=None, save_path=None):
    """
    Scales numeric variables with StandardScaler.
    - If scaler=None → trains a new scaler (fit_transform) and saves it in save_path
    - If scaler!=None → uses the existing scaler (transform)
    """
    df = df.copy()
    if scaler is None:
        scaler = StandardScaler()
        df[NUMERIC_COLS] = scaler.fit_transform(df[NUMERIC_COLS])
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            joblib.dump(scaler, save_path)
    else:
        df[NUMERIC_COLS] = scaler.transform(df[NUMERIC_COLS])
    return df, scaler


def preprocess_data(df, scaler=None, scaler_path="./models/scaler.pkl"):
    """
    Main function that executes the entire preprocessing pipeline.
    - scaler=None → trains and saves the scaler (used in train_model.py)
    - scaler!=None → uses the existing scaler (used in streamlit_app.py)
    """
    df = fix_impossible_zeros(df)
    df = impute_nulls(df)
    df = fix_outliers(df)
    df = encode_categoricals(df)
    df, scaler = scale_numerics(df, scaler=scaler, save_path=scaler_path)
    return df, scaler


# ── MAIN EXECUTION ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    df = pd.read_csv("./dataset/heart.csv")
    df_processed, scaler = preprocess_data(df)
    df_processed.to_csv("./dataset/heart_processed.csv", index=False)
    print(f"Dataset processed: {df_processed.shape[0]} rows × {df_processed.shape[1]} columns")
    print(f"Scaler saved to ./models/scaler.pkl")