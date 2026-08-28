import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    precision_recall_curve
)

from credit_fraud_pack.config import COST_FALSE_NEGATIVE, COST_FALSE_POSITIVE


# Metrics columns in the order they should appear wherever they're tabulated
METRIC_COLUMNS = ["precision", "recall", "f1", "auprc", "roc_auc",
                  "expected_cost"]


def classification_metrics(y_true, y_pred, y_score=None):
    """Score one set of predictions against the fraud label.

    :param y_true: True 0 or 1 labels
    :param y_pred: Predicted 0 or 1 labels, already thresholded
    :param y_score: Predicted probability that fraud(1) occurred.
    :return: dictionary mapping metric name to value
    """
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    metrics = {
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "auprc": np.nan,
        "roc_auc": np.nan,
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        "expected_cost": fn * COST_FALSE_NEGATIVE + fp * COST_FALSE_POSITIVE
    }

    if y_score is not None:
        metrics["auprc"] = average_precision_score(y_true, y_score)
        metrics["roc_auc"] = roc_auc_score(y_true, y_score)

    return metrics


def evaluate_classifier(estimator, X, y, threshold=0.5):
    """Score a fitted sklearn estimator on (X, y).

    :param estimator: Fitted classifier exposing predict_proba,
        with classes ordered [0, 1] (the default for this dataset).
    :param X: Features to predict on, usually the held-out test set.
    :param y: True labels for X.
    :param threshold: Probability cut-off for labelling a row as fraud.
    :return: the dictionary from classification_metrics
    """
    y_score = estimator.predict_proba(X)[:, 1]
    y_pred = (y_score >= threshold).astype(int)
    return classification_metrics(y, y_pred, y_score)


def plot_confusion_matrix(y_true, y_pred, title=None, ax=None):
    """Draw a 2x2 confusion matrix of raw counts.

    :return: The matplotlib Axes, so the caller can save or restyle it.
    """
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    if ax is None:
        _, ax = plt.subplots(figsize=(4.5, 4))

    sns.heatmap(
        cm, annot=True, fmt=",d", cmap="Blues", cbar=False,
        xticklabels=["Legit", "Fraud"], yticklabels=["Legit", "Fraud"], ax=ax
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title or "Confusion Matrix")
    return ax


def plot_precision_recall_curve(y_true, y_score, label=None, ax=None):
    """Plot a precision-recall curve annotated with its AUPRC.

    :param y_score: Predicted probability of the positive class.
    :return: The matplotlib Axes.
    """
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    auprc = average_precision_score(y_true, y_score)

    if ax is None:
        _, ax = plt.subplots(figsize=(6, 5))

    ax.plot(recall, precision, label=f"{label or 'model'}   (AUPRC = {auprc:.3f})")

    prevalence = np.mean(y_true)
    ax.axhline(prevalence, ls="--", color="grey",
               label=f"No-skill ({prevalence:.4f})")

    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-recall curve")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.legend(loc="upper right")
    return ax


def build_comparison_table(results, columns=METRIC_COLUMNS):
    """Stack per-model metric dictionaries into one comparison DataFrame

    :param results: Mapping of model name -> metrics dict.
    :param columns: Which metrics to keep, in display order.
    :return: DataFrame indexed by model name, one column per metric.
    """
    table = pd.DataFrame(results).T
    table = table[columns]
    table.index.name = "model"
    return table
