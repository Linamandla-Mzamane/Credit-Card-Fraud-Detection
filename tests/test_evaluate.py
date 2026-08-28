import numpy as np

from sklearn.linear_model import LogisticRegression

from credit_fraud_pack.config import COST_FALSE_POSITIVE, COST_FALSE_NEGATIVE
from credit_fraud_pack.evaluate import classification_metrics, evaluate_classifier


Y_TRUE = np.array([0, 0, 0, 1, 1, 1])
Y_PRED = np.array([0, 1, 0, 1, 1, 0])


def test_confusion_matrix():
    m = classification_metrics(Y_TRUE, Y_PRED)
    assert (m["tn"], m["fp"], m["fn"], m["tp"]) == (2, 1, 1, 2)


def test_precision_recall_f1():
    m = classification_metrics(Y_TRUE, Y_PRED)
    assert m["precision"] == 2 / 3
    assert m["recall"] == 2 / 3
    assert m["f1"] == 2 / 3


def test_expected_cost_uses_config_constants():
    m = classification_metrics(Y_TRUE, Y_PRED)
    expected = 1 * COST_FALSE_NEGATIVE + 1 * COST_FALSE_POSITIVE
    assert m["expected_cost"] == expected


def test_ranking_metrics_nan_without_scores():
    m = classification_metrics(Y_TRUE, Y_PRED)
    assert np.isnan(m["auprc"])
    assert np.isnan(m["roc_auc"])


def test_ranking_metrics_present_with_scores():
    y_score = np.array([0.1, 0.2, 0.3, 0.8, 0.9, 0.7])
    m = classification_metrics(Y_TRUE, Y_PRED, y_score)
    assert m["auprc"] == 1.0
    assert m["roc_auc"] == 1.0


def test_all_negative_predictions_do_not_raise():
    m = classification_metrics(Y_TRUE, np.zeros_like(Y_TRUE))
    assert m["precision"] == 0.0
    assert m["recall"] == 0.0
    assert (m["tn"], m["fp"], m["fn"], m["tp"]) == (3, 0, 3, 0)


def _perfectly_separable():
    """Tiny 2-feature dataset a linear model can classify with no errors."""

    X = np.array([[0.0, 0.0], [0.1, 0.1], [0.2, 0.0],
                  [5.0, 5.0], [5.1, 4.9], [4.9, 5.1],])
    y = np.array([0, 0, 0, 1, 1, 1])
    return X, y


def test_evaluate_classifier_scores_a_fitted_model():
    X, y = _perfectly_separable()
    model = LogisticRegression().fit(X, y)

    m = evaluate_classifier(model, X, y)

    assert m["precision"] == 1.0
    assert m["recall"] == 1.0
    assert m["auprc"] == 1.0
    assert (m["fp"], m["fn"]) == (0, 0)


def test_evaluate_classifier_threshold_shifts_predictions():
    X, y = _perfectly_separable()
    model = LogisticRegression().fit(X, y)

    m = evaluate_classifier(model, X, y, threshold=1.01)

    assert m["tp"] == 0
    assert m["fn"] == 3
    assert m["recall"] == 0.0


def test_evaluate_classifier_matches_manual_call():
    X, y = _perfectly_separable()
    model = LogisticRegression().fit(X, y)

    y_score = model.predict_proba(X)[:, 1]
    y_pred = (y_score >= 0.5).astype(int)
    expected = classification_metrics(y, y_pred, y_score)

    assert evaluate_classifier(model, X, y) == expected