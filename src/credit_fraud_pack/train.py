"""Fit the final fraud model and save it for deployment.

The model, its hyperparameters and its decision threshold were all chosen in
notebooks/credit_card_fraud_detection.ipynb. This script rebuilds that single
model from scratch, so it can be reproduced without re-running the notebook.

Run from the project root:
    python -m credit_fraud_pack.train
"""
import joblib
from sklearn.model_selection import FixedThresholdClassifier, train_test_split
from xgboost import XGBClassifier

from credit_fraud_pack.config import (FEATURE_COLS, MODELS_DIR, RANDOM_SEED,
                                      TARGET_COL, TEST_SIZE)
from credit_fraud_pack.data import load_raw_data
from credit_fraud_pack.evaluate import classification_metrics
from credit_fraud_pack.pipeline import build_model_pipeline

# Chosen by the 5-fold GridSearchCV in the notebook's XGBoost section.
# scale_pos_weight stays at its default of 1: the threshold-tuning section
# showed class weighting adds nothing once the threshold is tuned.
XGB_PARAMS = {
    "n_estimators": 600,
    "learning_rate": 0.05,
    "max_depth": 3,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
}

# Chosen in the notebook's threshold-tuning section: the cut-off with the
# lowest expected cost on out-of-fold training scores.
DECISION_THRESHOLD = 0.095

MODEL_FILE = MODELS_DIR / "xgboost_fraud_model.joblib"


def build_final_model():
    """Build the unfitted final model: the XGBoost pipeline with its threshold.

    :return: A FixedThresholdClassifier whose predict() flags fraud when the
        predicted probability is at least DECISION_THRESHOLD.
    """
    xgb = XGBClassifier(
        objective="binary:logistic",
        eval_metric="aucpr",
        tree_method="hist",
        random_state=RANDOM_SEED,
        n_jobs=-1,
        **XGB_PARAMS,
    )

    return FixedThresholdClassifier(
        build_model_pipeline(xgb),
        threshold=DECISION_THRESHOLD,
        # The threshold applies to probabilities, not raw XGBoost margins.
        response_method="predict_proba",
    )


def main():
    # Reproduce the notebook's data preparation exactly: same duplicate
    # removal, same stratified split, same seed. If any of this differs,
    # the test set differs and the check below will not match the notebook.
    df = load_raw_data().drop_duplicates().reset_index(drop=True)
    X, y = df[FEATURE_COLS], df[TARGET_COL]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        stratify=y,
        random_state=RANDOM_SEED,
    )

    model = build_final_model()
    model.fit(X_train, y_train)

    # Use predict(), not a manual threshold, because predict() is what a
    # deployed service will call. This checks the saved object itself.
    y_pred = model.predict(X_test)
    y_score = model.predict_proba(X_test)[:, 1]
    metrics = classification_metrics(y_test, y_pred, y_score)

    print(f"Threshold:      {DECISION_THRESHOLD}")
    print(f"Precision:      {metrics['precision']:.3f}")
    print(f"Recall:         {metrics['recall']:.3f}")
    print(f"Missed frauds:  {metrics['fn']}")
    print(f"False alarms:   {metrics['fp']}")
    print(f"Expected cost:  {metrics['expected_cost']:,.0f}")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_FILE)
    print(f"Saved model to {MODEL_FILE}")


if __name__ == "__main__":
    main()