import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.metrics import (
    classification_report, roc_auc_score,
    roc_curve, confusion_matrix
)

# ── CONSTANTES ────────────────────────────────────────────────────────────────
DATA_DIR = "../dataset"
MODELS_DIR = "../models"


# ── FUNCIONES ─────────────────────────────────────────────────────────────────
def load_data():
    """Carga el test set generado por train_model.py."""
    X_test = pd.read_csv(f"{DATA_DIR}/X_test.csv")
    y_test = pd.read_csv(f"{DATA_DIR}/y_test.csv").squeeze()
    return X_test, y_test


def load_models():
    """Carga los modelos guardados."""
    models = {
        "Logistic Regression": joblib.load(f"{MODELS_DIR}/lr_model.pkl"),
        "Random Forest":       joblib.load(f"{MODELS_DIR}/rf_model.pkl"),
        "XGBoost":             joblib.load(f"{MODELS_DIR}/xgb_model.pkl")
    }
    return models


def plot_confusion_matrix(models, X_test, y_test):
    """Grafica la matriz de confusión para cada modelo."""
    fig, axs = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("Confusion Matrix", fontsize=16)

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
    plt.show()


def plot_roc_curve(models, X_test, y_test):
    """Grafica la curva ROC comparando los 3 modelos."""
    plt.figure(figsize=(8, 6))

    for name, model in models.items():
        y_proba = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        auc = roc_auc_score(y_test, y_proba)
        plt.plot(fpr, tpr, label=f"{name} (AUC = {auc:.3f})")

    plt.plot([0, 1], [0, 1], 'k--', label="Random classifier")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve Comparison")
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_feature_importance(models, X_test):
    """Grafica la importancia de variables para Random Forest y XGBoost."""
    fig, axs = plt.subplots(1, 2, figsize=(18, 6))
    fig.suptitle("Feature Importance", fontsize=16)

    for ax, name in zip(axs, ["Random Forest", "XGBoost"]):
        model = models[name]
        importance = pd.Series(
            model.feature_importances_,
            index=X_test.columns
        ).sort_values(ascending=False).head(15)

        sns.barplot(x=importance.values, y=importance.index, ax=ax, palette="Blues_r")
        ax.set_title(name)
        ax.set_xlabel("Importance")
        ax.set_ylabel("Feature")

    plt.tight_layout()
    plt.show()


def print_metrics(models, X_test, y_test):
    """Imprime classification report y ROC AUC por modelo."""
    for name, model in models.items():
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]
        print(f"\n── {name} ──────────────────────────────")
        print(classification_report(y_test, y_pred,
              target_names=["No Disease", "Disease"]))
        print(f"ROC AUC: {roc_auc_score(y_test, y_proba):.4f}")


def print_model_comparison(models, X_test, y_test):
    """Genera una tabla comparativa de métricas entre modelos."""
    from sklearn.metrics import accuracy_score, f1_score, recall_score

    rows = []
    for name, model in models.items():
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]
        rows.append({
            "Model":     name,
            "Accuracy":  round(accuracy_score(y_test, y_pred), 4),
            "Recall (Disease)": round(recall_score(y_test, y_pred), 4),
            "F1 (Disease)":     round(f1_score(y_test, y_pred), 4),
            "ROC AUC":   round(roc_auc_score(y_test, y_proba), 4)
        })

    comparison = pd.DataFrame(rows).set_index("Model")
    print("\n── Model Comparison ──────────────────────────────")
    print(comparison.to_string())
    return comparison


# ── EJECUCIÓN PRINCIPAL ───────────────────────────────────────────────────────
if __name__ == "__main__":
    X_test, y_test = load_data()
    models = load_models()

    print_metrics(models, X_test, y_test)
    print_model_comparison(models, X_test, y_test)
    plot_confusion_matrix(models, X_test, y_test)
    plot_roc_curve(models, X_test, y_test)
    plot_feature_importance(models, X_test)