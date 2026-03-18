import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.base import BaseEstimator
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
    roc_curve,
    auc,
)
from sklearn.preprocessing import LabelBinarizer


def compute_summary(y_test: pd.Series, y_pred: np.ndarray) -> dict:
    
    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "f1_weighted": f1_score(y_test, y_pred, average="weighted", zero_division=0),
        "classification_report": classification_report(y_test, y_pred, zero_division=0),
    }


def plot_confusion_matrix(y_test: pd.Series, y_pred: np.ndarray, class_names: list[str]) -> plt.Figure:
    
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix")
    fig.tight_layout()
    return fig


def plot_feature_importance(importance_df: pd.DataFrame, top_n: int = 10) -> plt.Figure:
    
    df = importance_df.head(top_n)
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.barplot(data=df, x="importance", y="feature", palette="viridis", ax=ax)
    ax.set_title(f"Top {top_n} Feature Importances")
    ax.set_xlabel("Importance")
    ax.set_ylabel("")
    fig.tight_layout()
    return fig


def plot_roc_curve(
    model: BaseEstimator,
    X_test: np.ndarray,
    y_test: pd.Series,
    class_names: list[str],
) -> plt.Figure | None:
    
    if not hasattr(model, "predict_proba"):
        return None

    y_prob = model.predict_proba(X_test)
    fig, ax = plt.subplots(figsize=(6, 5))

    if len(class_names) == 2:
        fpr, tpr, _ = roc_curve(y_test, y_prob[:, 1])
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
    else:
        lb = LabelBinarizer()
        y_bin = lb.fit_transform(y_test)
        for i, cls in enumerate(class_names):
            fpr, tpr, _ = roc_curve(y_bin[:, i], y_prob[:, i])
            roc_auc = auc(fpr, tpr)
            ax.plot(fpr, tpr, label=f"{cls} (AUC={roc_auc:.2f})")

    ax.plot([0, 1], [0, 1], "k--", label="Random")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend(loc="lower right")
    fig.tight_layout()
    return fig