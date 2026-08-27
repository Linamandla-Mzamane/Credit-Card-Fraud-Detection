from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

def build_preprocessor() -> ColumnTransformer:
    """
    :return: A ColumnTransformer that scales Time and Amount
        and passes the PCA components (V1-V28) through unchanged
    """
    ct = ColumnTransformer(
        transformers=[("scale_time_amount", StandardScaler(), ["Time", "Amount"])],
        remainder="passthrough",
        verbose_feature_names_out=False
    )
    return ct.set_output(transform="pandas")

def build_model_pipeline(estimator) -> Pipeline:
    """
    :param estimator: A scikit-learn compatible classifier to fit
        the processed features.
    :return: A pipline that applies build_preprocessor() before fitting
        with the given estimator.
    """
    return Pipeline([
        ("preprocessor", build_preprocessor()),
        ("classifier", estimator)
    ])