import numpy as np
import pandas as pd
import pytest

from src.models.evaluate import evaluate_binary
from src.models.train_ctr_model import CATEGORICAL_COLS, NUMERIC_COLS, build_pipeline
from sklearn.tree import DecisionTreeClassifier


def test_evaluate_binary_perfect_predictions():
    y_true = np.array([0, 0, 1, 1])
    y_proba = np.array([0.01, 0.02, 0.98, 0.99])
    metrics = evaluate_binary(y_true, y_proba)
    assert metrics.accuracy == 1.0
    assert metrics.roc_auc == 1.0


def test_evaluate_binary_metrics_bounded():
    rng = np.random.default_rng(0)
    y_true = rng.integers(0, 2, size=200)
    y_proba = rng.random(200)
    metrics = evaluate_binary(y_true, y_proba)
    for value in metrics.as_dict().values():
        assert 0.0 <= value or value == pytest.approx(0.0, abs=1e-6)


def _synthetic_frame(n: int = 300) -> pd.DataFrame:
    rng = np.random.default_rng(1)
    df = pd.DataFrame(
        {
            "device_type": rng.choice(["ios", "mid_android"], size=n),
            "os_version": rng.choice(["v12", "v13"], size=n),
            "region": rng.choice(["north", "south"], size=n),
            "ad_category": rng.choice(["gaming", "news"], size=n),
            "content_type": rng.choice(["lockscreen", "video"], size=n),
            "hour_of_day": rng.integers(0, 24, size=n),
            "day_of_week": rng.integers(0, 7, size=n),
            "historical_ctr": rng.random(n) * 0.3,
            "session_length_log": rng.random(n) * 5,
            "is_peak_hour": rng.integers(0, 2, size=n),
            "is_weekend": rng.integers(0, 2, size=n),
            "device_tier": rng.integers(1, 4, size=n),
        }
    )
    y = (df["historical_ctr"] > 0.15).astype(int)
    return df, y


def test_build_pipeline_trains_and_predicts():
    X, y = _synthetic_frame()
    pipeline = build_pipeline(DecisionTreeClassifier(max_depth=3, random_state=0))
    pipeline.fit(X[CATEGORICAL_COLS + NUMERIC_COLS], y)
    proba = pipeline.predict_proba(X[CATEGORICAL_COLS + NUMERIC_COLS])[:, 1]
    assert proba.shape[0] == len(X)
    assert (proba >= 0).all() and (proba <= 1).all()
