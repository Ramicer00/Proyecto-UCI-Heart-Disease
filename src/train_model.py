import pandas as pd
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
import xgboost as xgb

# ── CONSTANTES ────────────────────────────────────────────────────────────────
DATA_PATH = "./dataset/heart_processed.csv"
MODELS_DIR = "./models"
TEST_SIZE = 0.2
RANDOM_STATE = 42


# ── FUNCIONES ─────────────────────────────────────────────────────────────────
def load_data(path=DATA_PATH):
    """Carga el dataset procesado y separa features y target."""
    df = pd.read_csv(path)
    X = df.drop("target", axis=1)
    y = df["target"]
    return train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y)


def train_logistic_regression(X_train, y_train):
    """Entrena un modelo de Logistic Regression."""
    model = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
    model.fit(X_train, y_train)
    return model


def train_random_forest(X_train, y_train):
    """Entrena un modelo de Random Forest."""
    model = RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE)
    model.fit(X_train, y_train)
    return model


def train_xgboost(X_train, y_train):
    """
    Entrena XGBoost con hiperparámetros optimizados.
    Parámetros obtenidos via RandomizedSearchCV en notebooks/tuning.ipynb:
    subsample=0.8, n_estimators=150, max_depth=3,
    learning_rate=0.05, colsample_bytree=0.6
    """
    model = xgb.XGBClassifier(
        subsample=0.8,
        n_estimators=150,
        max_depth=3,
        learning_rate=0.05,
        colsample_bytree=0.6,
        eval_metric='logloss',
        random_state=RANDOM_STATE
    )
    model.fit(X_train, y_train)
    return model


def save_model(model, filename):
    """Guarda el modelo en la carpeta models/."""
    os.makedirs(MODELS_DIR, exist_ok=True)
    path = os.path.join(MODELS_DIR, filename)
    joblib.dump(model, path)


# ── EJECUCIÓN PRINCIPAL ───────────────────────────────────────────────────────
if __name__ == "__main__":
    # Cargar datos
    X_train, X_test, y_train, y_test = load_data()

    # Entrenar modelos
    lr_model = train_logistic_regression(X_train, y_train)
    rf_model = train_random_forest(X_train, y_train)
    xgb_model = train_xgboost(X_train, y_train)

    # Guardar modelos
    save_model(lr_model, "lr_model.pkl")
    save_model(rf_model, "rf_model.pkl")
    save_model(xgb_model, "xgb_model.pkl")

    # Guardar test set para evaluación reproducible
    X_test.to_csv("./dataset/X_test.csv", index=False)
    y_test.to_csv("./dataset/y_test.csv", index=False)

    print("Modelos guardados correctamente en models/")
    print("Test set guardado en dataset/")
