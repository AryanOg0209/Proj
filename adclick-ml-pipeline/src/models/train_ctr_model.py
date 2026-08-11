"""
CTR classifier: Decision Tree vs. Random Forest, selected by held-out ROC-AUC.

Reads the row-level Parquet features written by the Spark ETL job, trains
both a single decision tree (interpretable baseline) and a random forest
(stronger ensemble), evaluates both, and persists the winner plus the
fitted preprocessing pipeline as a single joblib artifact so it can be
loaded directly by the FastAPI serving layer.

Usage:
    python -m src.models.train_ctr_model --features data/features/impressions \
        --out models/ctr_model.joblib
"""

from __future__ import annotations

import argparse
import json
import os

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier

from src.models.evaluate import evaluate_binary

CATEGORICAL_COLS = ["device_type", "os_version", "region", "ad_category", "content_type"]
NUMERIC_COLS = [
    "hour_of_day",
    "day_of_week",
    "historical_ctr",
    "session_length_log",
    "is_peak_hour",
    "is_weekend",
    "device_tier",
]
TARGET = "clicked"


def load_features(path: str) -> pd.DataFrame:
    return pd.read_parquet(path)


def build_pipeline(estimator) -> Pipeline:
    preprocess = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_COLS),
        ],
        remainder="passthrough",
    )
    return Pipeline(steps=[("preprocess", preprocess), ("model", estimator)])


def train(features_path: str, out_path: str, test_size: float = 0.2, seed: int = 42) -> dict:
    df = load_features(features_path)
    X = df[CATEGORICAL_COLS + NUMERIC_COLS]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )

    candidates = {
        "decision_tree": DecisionTreeClassifier(max_depth=8, min_samples_leaf=50, random_state=seed),
        "random_forest": RandomForestClassifier(
            n_estimators=150, max_depth=10, min_samples_leaf=20, n_jobs=-1, random_state=seed
        ),
    }

    results = {}
    best_name, best_pipeline, best_auc = None, None, -1.0
    for name, estimator in candidates.items():
        pipeline = build_pipeline(estimator)
        pipeline.fit(X_train, y_train)
        proba = pipeline.predict_proba(X_test)[:, 1]
        metrics = evaluate_binary(y_test.to_numpy(), proba)
        results[name] = metrics.as_dict()
        print(f"[{name}] {metrics.as_dict()}")
        if metrics.roc_auc > best_auc:
            best_name, best_pipeline, best_auc = name, pipeline, metrics.roc_auc

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    joblib.dump(
        {
            "pipeline": best_pipeline,
            "categorical_cols": CATEGORICAL_COLS,
            "numeric_cols": NUMERIC_COLS,
            "model_name": best_name,
        },
        out_path,
    )
    with open(out_path.replace(".joblib", "_metrics.json"), "w") as f:
        json.dump({"best_model": best_name, "results": results}, f, indent=2)

    print(f"Selected '{best_name}' (ROC-AUC={best_auc:.4f}), saved to {out_path}")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", default="data/features/impressions")
    parser.add_argument("--out", default="models/ctr_model.joblib")
    args = parser.parse_args()
    train(args.features, args.out)


if __name__ == "__main__":
    main()
