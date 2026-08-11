"""
FastAPI microservice that serves the trained CTR model — the
"creation of microservices and APIs for serving ML models" piece of the
pipeline. Loads whichever model artifact is available at startup (sklearn
decision tree / random forest by default) and exposes:

    GET  /health          -- liveness + which models are loaded
    POST /predict/ctr      -- click probability for one impression

Run locally:
    uvicorn src.serving.app:app --reload

Run in Docker: see Dockerfile.
"""

from __future__ import annotations

import math
import os

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException

from src.models.train_ctr_model import CATEGORICAL_COLS, NUMERIC_COLS
from src.serving.schemas import CTRPrediction, HealthResponse, ImpressionRequest

MODEL_PATH = os.environ.get("CTR_MODEL_PATH", "models/ctr_model.joblib")

app = FastAPI(
    title="AdClick ML Serving API",
    description="Serves the CTR-prediction model trained by the adclick-ml-pipeline.",
    version="1.0.0",
)

_state: dict = {"pipeline": None, "model_name": None}


@app.on_event("startup")
def load_model() -> None:
    if os.path.exists(MODEL_PATH):
        bundle = joblib.load(MODEL_PATH)
        _state["pipeline"] = bundle["pipeline"]
        _state["model_name"] = bundle["model_name"]
    else:
        # Service can still boot (e.g. for the health check / CI) without a
        # trained artifact present; predictions will 503 until one is trained.
        _state["pipeline"] = None
        _state["model_name"] = None


def _to_feature_row(req: ImpressionRequest) -> pd.DataFrame:
    is_peak_hour = int(req.hour_of_day in {8, 9, 18, 19, 20, 21})
    is_weekend = int(req.day_of_week in {5, 6})
    device_tier = {"flagship_android": 3, "ios": 3, "mid_android": 2}.get(
        req.device_type, 1
    )
    row = {
        "device_type": req.device_type,
        "os_version": req.os_version,
        "region": req.region,
        "ad_category": req.ad_category,
        "content_type": req.content_type,
        "hour_of_day": req.hour_of_day,
        "day_of_week": req.day_of_week,
        "historical_ctr": req.historical_ctr,
        "session_length_log": math.log1p(req.session_length_sec),
        "is_peak_hour": is_peak_hour,
        "is_weekend": is_weekend,
        "device_tier": device_tier,
    }
    return pd.DataFrame([row])[CATEGORICAL_COLS + NUMERIC_COLS]


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    loaded = [_state["model_name"]] if _state["pipeline"] is not None else []
    return HealthResponse(status="ok", models_loaded=loaded)


@app.post("/predict/ctr", response_model=CTRPrediction)
def predict_ctr(req: ImpressionRequest) -> CTRPrediction:
    if _state["pipeline"] is None:
        raise HTTPException(
            status_code=503,
            detail=f"No trained model found at {MODEL_PATH}. Run the training pipeline first.",
        )
    row = _to_feature_row(req)
    proba = float(_state["pipeline"].predict_proba(row)[:, 1][0])
    return CTRPrediction(click_probability=round(proba, 6), model_used=_state["model_name"])
