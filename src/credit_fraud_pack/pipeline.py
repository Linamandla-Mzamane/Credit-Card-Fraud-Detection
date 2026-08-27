from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from credit_fraud_pack.config import SCALE_COLS, PCA_COLS

def build_preprocessor() -> ColumnTransformer:
    """Build the feature preprocessor.

    Scales `Time` and `Amount` with a `StandardScaler` and passes the
    PCA components (`V1-V28`) through unchanged. Every other column is
    dropped, so the estimator only ever sees the agreed feature set.

    :return: A ColumnTransformer that emits a pandas DataFrame.
    """
    ct = ColumnTransformer(
        transformers=[
            ("scale", StandardScaler(), SCALE_COLS),
            ("pca", "passthrough", PCA_COLS),
        ],
        remainder="drop",
        verbose_feature_names_out=False
    )
    return ct.set_output(transform="pandas")

def build_model_pipeline(estimator) -> Pipeline:
    """Wrap estimator behind the standard preprocessor

    :param estimator: A scikit-learn compatible classifier to fit
        the processed features.
    :return: A pipeline that applies build_preprocessor() before fitting
        with the given estimator.
    """
    return Pipeline([
        ("preprocessor", build_preprocessor()),
        ("classifier", estimator)
    ])