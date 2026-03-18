import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.metrics import (
    roc_auc_score, roc_curve,
    confusion_matrix, classification_report
)

# ── CONSTANTES ────────────────────────────────────────────────────────────────
MODELS_DIR = "../models"
DATA_DIR = "../data"

# ── CARGA DE RECURSOS ─────────────────────────────────────────────────────────
@st.cache_resource
def load_models():
    return {
        "Logistic Regression": joblib.load(f"{MODELS_DIR}/lr_model.pkl"),
        "Random Forest":       joblib.load(f"{MODELS_DIR}/rf_model.pkl"),
        "XGBoost":             joblib.load(f"{MODELS_DIR}/xgb_model.pkl")
    }

@st.cache_data
def load_test_data():
    X_test = pd.read_csv(f"{DATA_DIR}/X_test.csv")
    y_test = pd.read_csv(f"{DATA_DIR}/y_test.csv").squeeze()
    return X_test, y_test

# ── PREPROCESSING DEL INPUT ───────────────────────────────────────────────────
def preprocess_input(input_dict, X_test_columns):
    """
    Convierte el input del formulario al mismo formato que el modelo espera.
    Aplica One-Hot Encoding y reordena columnas según X_test.
    """
    scaler = joblib.load(f"{MODELS_DIR}/scaler.pkl")
    df = pd.DataFrame([input_dict])

    # One-Hot Encoding
    cat_cols = ['cp', 'restecg', 'slope', 'ca', 'thal']
    df = pd.get_dummies(df, columns=cat_cols, drop_first=True, dtype=int)

    # Agregar columnas faltantes con 0
    for col in X_test_columns:
        if col not in df.columns:
            df[col] = 0

    df = df[X_test_columns]

    # Escalar variables numéricas
    numeric_cols = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
    df[numeric_cols] = scaler.transform(df[numeric_cols])

    return df


# ── PÁGINAS ───────────────────────────────────────────────────────────────────
def page_prediction(models, X_test):
    st.title("🫀 Heart Disease Prediction")
    st.markdown("Fill in the patient data to predict the presence of heart disease.")

    col1, col2, col3 = st.columns(3)

    with col1:
        age      = st.slider("Age", 20, 80, 50)
        sex      = st.selectbox("Sex", [0, 1], format_func=lambda x: "Female" if x == 0 else "Male")
        cp       = st.selectbox("Chest Pain Type (cp)", [0, 1, 2, 3])
        trestbps = st.slider("Resting Blood Pressure (trestbps)", 80, 200, 130)
        chol     = st.slider("Cholesterol (chol)", 85, 610, 246)

    with col2:
        fbs     = st.selectbox("Fasting Blood Sugar > 120 mg/dl (fbs)", [0, 1])
        restecg = st.selectbox("Resting ECG (restecg)", [0, 1, 2])
        thalach = st.slider("Max Heart Rate (thalach)", 60, 210, 140)
        exang   = st.selectbox("Exercise Induced Angina (exang)", [0, 1])
        oldpeak = st.slider("ST Depression (oldpeak)", 0.0, 6.2, 0.0, step=0.1)

    with col3:
        slope = st.selectbox("Slope of ST Segment (slope)", [0, 1, 2])
        ca    = st.selectbox("Major Vessels (ca)", [0, 1, 2, 3])
        thal  = st.selectbox("Thalassemia (thal)", [0, 1, 2, 3])

    model_name = st.selectbox("Select Model", list(models.keys()))

    if st.button("Predict"):
        input_dict = {
            "age": age, "sex": sex, "cp": cp, "trestbps": trestbps,
            "chol": chol, "fbs": fbs, "restecg": restecg, "thalach": thalach,
            "exang": exang, "oldpeak": oldpeak, "slope": slope,
            "ca": ca, "thal": thal
        }

        input_processed = preprocess_input(input_dict, X_test.columns)
        model = models[model_name]
        prediction = model.predict(input_processed)[0]
        probability = model.predict_proba(input_processed)[0][1]

        st.divider()
        col_res1, col_res2 = st.columns(2)

        with col_res1:
            if prediction == 1:
                st.error("🔴 Heart Disease Detected")
            else:
                st.success("🟢 No Heart Disease Detected")

        with col_res2:
            st.metric(
                label="Probability of Heart Disease",
                value=f"{probability * 100:.1f}%"
            )


def page_evaluation(models, X_test, y_test):
    st.title("📊 Model Evaluation")

    tab1, tab2, tab3 = st.tabs(["Confusion Matrix", "ROC Curve", "Feature Importance"])

    with tab1:
        st.subheader("Confusion Matrix")
        fig, axs = plt.subplots(1, 3, figsize=(18, 5))
        for ax, (name, model) in zip(axs, models.items()):
            y_pred = model.predict(X_test)
            cm = confusion_matrix(y_test, y_pred)
            sns.heatmap(
                cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['No Disease', 'Disease'],
                yticklabels=['No Disease', 'Disease']
            )
            ax.set_title(name)
            ax.set_xlabel("Predicted")
            ax.set_ylabel("Actual")
        plt.tight_layout()
        st.pyplot(fig)

    with tab2:
        st.subheader("ROC Curve")
        fig, ax = plt.subplots(figsize=(8, 6))
        for name, model in models.items():
            y_proba = model.predict_proba(X_test)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, y_proba)
            auc = roc_auc_score(y_test, y_proba)
            ax.plot(fpr, tpr, label=f"{name} (AUC = {auc:.3f})")
        ax.plot([0, 1], [0, 1], 'k--', label="Random classifier")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("ROC Curve Comparison")
        ax.legend()
        plt.tight_layout()
        st.pyplot(fig)

    with tab3:
        st.subheader("Feature Importance")
        fig, axs = plt.subplots(1, 2, figsize=(18, 6))
        for ax, name in zip(axs, ["Random Forest", "XGBoost"]):
            model = models[name]
            importance = pd.Series(
                model.feature_importances_,
                index=X_test.columns
            ).sort_values(ascending=False).head(15)
            sns.barplot(x=importance.values, y=importance.index, ax=ax, palette="Blues_r")
            ax.set_title(name)
            ax.set_xlabel("Importance")
        plt.tight_layout()
        st.pyplot(fig)


# ── NAVEGACIÓN ────────────────────────────────────────────────────────────────
def main():
    st.set_page_config(
        page_title="Heart Disease Predictor",
        page_icon="🫀",
        layout="wide"
    )

    models = load_models()
    X_test, y_test = load_test_data()

    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Prediction", "Model Evaluation"])

    if page == "Prediction":
        page_prediction(models, X_test)
    elif page == "Model Evaluation":
        page_evaluation(models, X_test, y_test)


if __name__ == "__main__":
    main()