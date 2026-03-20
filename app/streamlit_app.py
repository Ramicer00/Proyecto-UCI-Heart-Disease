import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.metrics import (
    roc_auc_score, roc_curve,
    confusion_matrix
)

# ── CONSTANTS ────────────────────────────────────────────────────────────────
MODELS_DIR = "./models"
DATA_DIR = "./dataset"

# ── LOAD RESOURCES ─────────────────────────────────────────────────────────
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

# ── INPUT PREPROCESSING ───────────────────────────────────────────────────
def preprocess_input(input_dict, X_test_columns):
    """
    Converts form input to the format expected by the model.
    Applies One-Hot Encoding and reorders columns to match training data.
    """
    scaler = joblib.load(f"{MODELS_DIR}/scaler.pkl")
    df = pd.DataFrame([input_dict])

    # One-Hot Encoding
    cat_cols = ['cp', 'restecg', 'slope', 'ca', 'thal']
    df = pd.get_dummies(df, columns=cat_cols, drop_first=True, dtype=int)

    # Add missing columns with zeros
    for col in X_test_columns:
        if col not in df.columns:
            df[col] = 0

    df = df[X_test_columns]

    # Scale numeric variables
    numeric_cols = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
    df[numeric_cols] = scaler.transform(df[numeric_cols])

    return df


# ── PAGES ───────────────────────────────────────────────────────────────
def page_prediction(models, X_test):
    st.markdown("<h1 style='text-align: center;'>Heart Disease Prediction</h1>", unsafe_allow_html=True)
    st.write("")
    st.markdown("Fill in the patient data to predict the presence of heart disease")

    col1, col2, col3 = st.columns(3)

    with col1:
        age      = st.number_input("Age", min_value=20, max_value=80, value=50)
        sex      = st.selectbox("Sex", [0, 1], format_func=lambda x: "Female" if x == 0 else "Male")
        cp       = st.selectbox("Chest Pain Type (cp)", [0, 1, 2, 3])
        trestbps = st.number_input("Resting Blood Pressure (trestbps)", min_value=80, max_value=200, value=130)
        chol     = st.number_input("Cholesterol (chol)", min_value=85, max_value=610, value=246)

    with col2:
        fbs     = st.selectbox("Fasting Blood Sugar > 120 mg/dl (fbs)", [0, 1])
        restecg = st.selectbox("Resting ECG (restecg)", [0, 1, 2])
        thalach = st.number_input("Max Heart Rate (thalach)", min_value=60, max_value=210, value=140)
        exang   = st.selectbox("Exercise Induced Angina (exang)", [0, 1])

    with col3:
        slope = st.selectbox("Slope of ST Segment (slope)", [0, 1, 2])
        ca    = st.selectbox("Major Vessels (ca)", [0, 1, 2, 3])
        thal  = st.selectbox("Thalassemia (thal)", [0, 1, 2, 3])
        oldpeak = st.number_input("ST Depression (oldpeak)", min_value=0.0, max_value=6.2, value=0.0, step=0.1)

    st.divider()
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
    st.markdown("<h1 style='text-align: center;'>Model Evaluation</h1>", unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["Confusion Matrix", "ROC Curve", "Feature Importance"])

    with tab1:
        st.markdown("<h2 style='text-align: center;'>Confusion Matrix</h2>", unsafe_allow_html=True)
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

        # ── RESULTS SUMMARY ────────────────────────────────────────────────────────
        st.write("")
        st.markdown("### Results Summary")

        col1, col2, col3 = st.columns(3)

        for col, (name, model) in zip([col1, col2, col3], models.items()):
            y_pred = model.predict(X_test)
            cm = confusion_matrix(y_test, y_pred)
            tn, fp, fn, tp = cm.ravel()

            with col:
                st.markdown(f"**{name}**")
                st.markdown(f"✅ **{tn}** healthy patients correctly identified")
                st.markdown(f"✅ **{tp}** sick patients correctly identified")
                st.markdown(f"⚠️ **{fp}** healthy patients incorrectly flagged as sick")
                st.markdown(f"🔴 **{fn}** sick patients missed (false negatives)")
        # ── MODEL METRICS ────────────────────────────────────────────────────────
        st.write("")
        st.markdown("### Model Metrics")

        metrics_rows = []
        for name, model in models.items():
            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test)[:, 1]
            tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
            metrics_rows.append({
                "Model":             name,
                "Accuracy":          f"{(tn + tp) / (tn + fp + fn + tp) * 100:.1f}%",
                "Recall":  f"{tp / (tp + fn) * 100:.1f}%",
                "Precision": f"{tp / (tp + fp) * 100:.1f}%",
                "F1":      f"{2 * tp / (2 * tp + fp + fn) * 100:.1f}%",
            })

        st.dataframe(
            pd.DataFrame(metrics_rows).set_index("Model"),
            use_container_width=True
        )

    with tab2:
        st.markdown("<h2 style='text-align: center;'>ROC Curve</h2>", unsafe_allow_html=True)
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
        st.markdown("<h2 style='text-align: center;'>Feature Importance</h2>", unsafe_allow_html=True)
        fig, axs = plt.subplots(2, 1, figsize=(14, 16))
        for ax, name in zip(axs, ["Random Forest", "XGBoost"]):
            model = models[name]
            importance = pd.Series(
                model.feature_importances_,
                index=X_test.columns
            ).sort_values(ascending=False).head(15)
            sns.barplot(x=importance.values, y=importance.index, ax=ax, palette="Blues_r")
            ax.set_title(name)
            ax.set_xlabel("Importance")
        plt.tight_layout(h_pad=4.0)
        st.pyplot(fig, use_container_width=True)


# ── NAVIGATION ────────────────────────────────────────────────────────────────
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