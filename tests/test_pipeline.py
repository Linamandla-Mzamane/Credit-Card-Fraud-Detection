import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression

from credit_fraud_pack.pipeline import (build_preprocessor,
                                        build_model_pipeline)

from credit_fraud_pack.config import FEATURE_COLS

def _tiny_df(n=6):
    rng = np.random.default_rng(0)
    data ={"Time": np.arange(n) * 100.0,
           "Amount": rng.uniform(1, 500, n)}

    for col in (f"V{i}" for i in range(1, 29)):
        data[col] = rng.standard_normal(n)

    return pd.DataFrame(data)[FEATURE_COLS]


def test_build_preprocessor_scales_time_amount():
    result = build_preprocessor().fit_transform(_tiny_df())
    assert np.isclose(result["Time"].mean(), 0.0, atol=1e-8)
    assert np.isclose(result["Time"].std(ddof=0), 1.0, atol=1e-8)

def test_build_preprocessor_leaves_v_columns_unchanged():
    df = _tiny_df()
    result = build_preprocessor().fit_transform(df)

    assert list(result["V1"]) == list(df["V1"])
    assert list(result["V28"]) == list(df["V28"])

def test_build_preprocessor_returns_dataframe_with_expected_columns():
    df = _tiny_df()
    result = build_preprocessor().fit_transform(df)
    assert isinstance(result, pd.DataFrame)
    assert set(result.columns) == set(df.columns)

def test_build_preprocessor_drops_unexpected_columns():
    df = _tiny_df()
    df["hour_of_day"] = 3
    result = build_preprocessor().fit_transform(df)
    assert "hour_of_day" not in result.columns

def test_build_model_pipeline_fits_and_predicts():
    df =_tiny_df()
    y = pd.Series([0, 1, 0, 1, 0, 1])
    pipeline = build_model_pipeline(LogisticRegression())
    pipeline.fit(df, y)
    predictions = pipeline.predict(df)
    assert len(predictions) == len(df)
    assert set(predictions).issubset({0, 1})

