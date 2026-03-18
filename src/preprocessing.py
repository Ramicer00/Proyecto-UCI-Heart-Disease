import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib
import os

# ── CONSTANTES ────────────────────────────────────────────────────────────────
NUMERIC_COLS = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
CATEGORICAL_COLS = ['cp', 'restecg', 'slope', 'ca', 'thal']
IMPUTE_MODA_COLS = ['restecg', 'thalach', 'exang', 'fbs', 'oldpeak', 'slope', 'thal', 'ca']


# ── FUNCIONES ─────────────────────────────────────────────────────────────────
def fix_impossible_zeros(df):
    """Reemplaza ceros médicamente imposibles por NaN en chol y trestbps."""
    df = df.copy()
    df['chol'] = df['chol'].replace(0, np.nan)
    df['trestbps'] = df['trestbps'].replace(0, np.nan)
    return df


def impute_nulls(df):
    """Imputa nulos con media en variables continuas y moda en categóricas."""
    df = df.copy()
    df['chol'] = df['chol'].fillna(df['chol'].mean())
    df['trestbps'] = df['trestbps'].fillna(df['trestbps'].mean())
    for col in IMPUTE_MODA_COLS:
        df[col] = df[col].fillna(df[col].mode()[0])
    return df


def fix_outliers(df):
    """Corrige outliers clínicamente imposibles."""
    df = df.copy()
    df['oldpeak'] = df['oldpeak'].clip(lower=0)
    return df


def encode_categoricals(df):
    """Aplica One-Hot Encoding a variables categóricas nominales."""
    df = df.copy()
    df = pd.get_dummies(df, columns=CATEGORICAL_COLS, drop_first=True, dtype=int)
    return df


def scale_numerics(df, scaler=None, save_path=None):
    """
    Escala variables numéricas con StandardScaler.
    - Si scaler=None → entrena un scaler nuevo (fit_transform) y lo guarda en save_path
    - Si scaler!=None → usa el scaler existente (transform)
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


def preprocess_data(df, scaler=None, scaler_path="../models/scaler.pkl"):
    """
    Función principal que ejecuta todo el pipeline de preprocessing.
    - scaler=None → entrena y guarda el scaler (usado en train_model.py)
    - scaler!=None → usa el scaler existente (usado en streamlit_app.py)
    """
    df = fix_impossible_zeros(df)
    df = impute_nulls(df)
    df = fix_outliers(df)
    df = encode_categoricals(df)
    df, scaler = scale_numerics(df, scaler=scaler, save_path=scaler_path)
    return df, scaler


# ── EJECUCIÓN DIRECTA ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    df = pd.read_csv("../data/heart.csv")
    df_processed, scaler = preprocess_data(df)
    df_processed.to_csv("../data/heart_processed.csv", index=False)
    print(f"Dataset procesado: {df_processed.shape[0]} filas × {df_processed.shape[1]} columnas")
    print(f"Scaler guardado en ../models/scaler.pkl")