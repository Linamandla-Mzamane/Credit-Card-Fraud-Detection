import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression

from credit_fraud_pack.pipeline import (build_preprocessor,
                                        build_model_pipeline)

def _tiny_df():
    return pd.DataFrame({
        "Time":[0.0, 100.0, 200.0, 300.0],
        "Amount": [10.0, 20.0, 30.0, 40.0],
        "V1": [1.1, -2.2, 3.3, -4.4],
        "V2": [0.5, 0.5, 0.5, 0.5]
    })

def test_build_preprocessor_scales_time_amount():
    result = build_preprocessor().fit_transform(_tiny_df())

    assert np.isclose(result["Time"].mean(), 0.0, atol=1e-8)
    assert np.isclose(result["Time"].std(ddof=0), 1.0, atol=1e-8)

def test_build_preprocessor_leaves_v_columns_unchanged():
    df = _tiny_df()
    result = build_preprocessor().fit_transform(df)

    assert list(result["V1"]) == list(df["V1"])

def test_build_model_pipeline_fits_and_predicts():
    df =_tiny_df()
    y = pd.Series([0, 1, 0, 1])

    pipeline = build_model_pipeline(LogisticRegression())
    pipeline.fit(df, y)
    predictions = pipeline.predict(df)

    assert len(predictions) == len(df)
    assert set(predictions).issubset({0, 1})

