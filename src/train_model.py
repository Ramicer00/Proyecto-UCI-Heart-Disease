import pandas as pd
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb

# ── CONSTANTS ────────────────────────────────────────────────────────────────
DATA_PATH = "./dataset/heart_processed.csv"
MODELS_DIR = "./models"
TEST_SIZE = 0.2
RANDOM_STATE = 42


# ── FUNCTIONS ─────────────────────────────────────────────────────────────────
def load_data(path=DATA_PATH):
    """Loads the processed dataset and separates features and target."""
    df = pd.read_csv(path)
    X = df.drop("target", axis=1)
    y = df["target"]
    return train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y)


def train_logistic_regression(X_train, y_train):
    """Trains a Logistic Regression model."""
    model = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
    model.fit(X_train, y_train)
    return model


def train_random_forest(X_train, y_train):
    """Trains a Random Forest model."""
    model = RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE)
    model.fit(X_train, y_train)
    return model


def train_xgboost(X_train, y_train):
    """
    Trains XGBoost with optimized hyperparameters.
    Parameters obtained via RandomizedSearchCV in notebooks/tuning.ipynb:
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
    """Saves the model in the models/ folder."""
    os.makedirs(MODELS_DIR, exist_ok=True)
    path = os.path.join(MODELS_DIR, filename)
    joblib.dump(model, path)


# ── MAIN EXECUTION ───────────────────────────────────────────────────────
if __name__ == "__main__":
    # Load data
    X_train, X_test, y_train, y_test = load_data()

    # Train models
    lr_model = train_logistic_regression(X_train, y_train)
    rf_model = train_random_forest(X_train, y_train)
    xgb_model = train_xgboost(X_train, y_train)

    # Save models
    save_model(lr_model, "lr_model.pkl")
    save_model(rf_model, "rf_model.pkl")
    save_model(xgb_model, "xgb_model.pkl")

    # Save test set for reproducible evaluation
    X_test.to_csv("./dataset/X_test.csv", index=False)
    y_test.to_csv("./dataset/y_test.csv", index=False)

    print("Models saved correctly in models/")
    print("Test set saved in dataset/")
