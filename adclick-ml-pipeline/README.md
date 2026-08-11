# AdClick ML Pipeline

An end-to-end machine learning pipeline for a mobile ad/content platform: **Spark ETL → user
segmentation (clustering) → CTR prediction (decision trees / random forest / neural net) →
model serving via a FastAPI microservice**, with tests and CI.

Built to mirror the day-to-day of an ML Engineer on a mobile ads/content platform (lock-screen
ads, in-feed content, etc.) — architecting experimentation pipelines, deploying models, and
exposing them as services other teams can call.

## Architecture

```
                  ┌─────────────────────┐
 generate_data.py │  raw impression CSV  │
                  └──────────┬───────────┘
                             ▼
                  ┌─────────────────────┐
     PySpark ETL  │  clean + feature-    │   src/etl/spark_pipeline.py
                  │  engineer @ scale    │
                  └──────────┬───────────┘
                             ▼
              ┌──────────────┴──────────────┐
              ▼                              ▼
   data/features/impressions       data/features/user_profile
   (row-level, for CTR model)      (per-user, for clustering)
              │                              │
              ▼                              ▼
  ┌───────────────────────┐      ┌────────────────────────┐
  │ Decision Tree /        │      │ Spark MLlib KMeans      │
  │ Random Forest (sklearn)│      │ user segmentation       │
  │ + PyTorch MLP          │      └────────────────────────┘
  └───────────┬────────────┘
              ▼
   models/ctr_model.joblib
              │
              ▼
   ┌────────────────────────────┐
   │ FastAPI microservice        │  src/serving/app.py
   │ POST /predict/ctr           │  (Dockerized)
   │ GET  /health                │
   └────────────────────────────┘
```

## Why this project

Built specifically to demonstrate the core requirements of an ML Engineer / Data Scientist
role on a mobile content/ads platform:

| Requirement | Where it shows up |
|---|---|
| Python + Spark for data science | `src/etl/spark_pipeline.py` (PySpark DataFrame API + MLlib) |
| Managing & deploying ML models | `models/train_ctr_model.py`, `src/serving/app.py` (FastAPI serving) |
| ML techniques: clustering, decision trees, neural networks | KMeans (Spark MLlib), Decision Tree / Random Forest (scikit-learn), MLP (PyTorch) — all three, trained and compared on the same problem |
| Data structures & transformation methods | Spark DataFrame transformations, window functions, aggregations in `spark_pipeline.py` |
| Deploying data pipelines for efficient ML workflows | `src/pipeline.py` — single reproducible CLI entry point chaining every stage |
| Microservices/APIs for serving ML models | Dockerized FastAPI service (`Dockerfile`, `docker-compose.yml`) |
| Linux command-line proficiency | `scripts/run_pipeline.sh`, Docker, GitHub Actions CI all shell-driven |
| Math/stats | Feature engineering (log-transform, percentile rank), evaluation via ROC-AUC / log-loss / silhouette score |

## Quickstart

```bash
cd adclick-ml-pipeline
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Run everything: generate data -> Spark ETL -> clustering -> CTR models
python -m src.pipeline run --rows 200000 --k 4

# Serve the trained CTR model
uvicorn src.serving.app:app --reload
# then: curl -X POST localhost:8000/predict/ctr -H "Content-Type: application/json" -d '{
#   "device_type": "flagship_android", "os_version": "v13", "region": "south",
#   "ad_category": "gaming", "content_type": "lockscreen",
#   "hour_of_day": 19, "day_of_week": 5, "historical_ctr": 0.12, "session_length_sec": 240
# }'
```

Or via the shell script (bootstraps its own venv):

```bash
./scripts/run_pipeline.sh 200000 4
```

Or with Docker (after training locally so `models/` is populated):

```bash
docker compose up --build
```

## Project layout

```
adclick-ml-pipeline/
├── data/generate_data.py        # synthetic mobile-ad-impression data generator
├── src/
│   ├── etl/spark_pipeline.py    # PySpark cleaning + feature engineering
│   ├── models/
│   │   ├── train_clustering.py  # Spark MLlib KMeans user segmentation
│   │   ├── train_ctr_model.py   # Decision Tree vs Random Forest (sklearn)
│   │   ├── train_nn_model.py    # PyTorch MLP baseline
│   │   └── evaluate.py          # shared metrics (accuracy/precision/recall/F1/ROC-AUC)
│   ├── serving/app.py           # FastAPI model-serving microservice
│   └── pipeline.py              # CLI orchestrator for the whole pipeline
├── tests/                       # pytest coverage for ETL, models, and API
├── scripts/run_pipeline.sh
├── Dockerfile / docker-compose.yml
└── .github/workflows/ci.yml     # installs deps, runs tests, smoke-tests ETL
```

## Testing

```bash
pytest -v
```

CI (`.github/workflows/ci.yml`) runs the full test suite plus a smoke test of data generation
and the Spark ETL job on every push/PR.

## Notes on the data

`data/generate_data.py` produces synthetic impression-level events (device, region, ad
category, time-of-day, user's historical CTR, session length) with a hand-crafted latent
click-probability function, so the labels have real, learnable structure rather than being
pure noise — useful for demonstrating the pipeline without needing access to proprietary
ad-serving data.
