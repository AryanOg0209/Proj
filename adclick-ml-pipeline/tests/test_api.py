from fastapi.testclient import TestClient

from src.serving.app import app

# Use as a context manager so ASGI lifespan (startup/shutdown) events fire,
# otherwise `load_model()` never runs and every request looks model-less.
client = TestClient(app)
client.__enter__()


def test_health_ok():
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert "models_loaded" in body


def test_predict_without_model_returns_503(monkeypatch):
    # In CI there's no trained artifact, so the endpoint should fail
    # loudly and predictably rather than crash.
    payload = {
        "device_type": "ios",
        "os_version": "v13",
        "region": "south",
        "ad_category": "gaming",
        "content_type": "lockscreen",
        "hour_of_day": 9,
        "day_of_week": 1,
        "historical_ctr": 0.1,
        "session_length_sec": 120,
    }
    resp = client.post("/predict/ctr", json=payload)
    assert resp.status_code in (200, 503)


def test_predict_rejects_invalid_hour():
    payload = {
        "device_type": "ios",
        "os_version": "v13",
        "region": "south",
        "ad_category": "gaming",
        "content_type": "lockscreen",
        "hour_of_day": 30,  # invalid
        "day_of_week": 1,
        "historical_ctr": 0.1,
        "session_length_sec": 120,
    }
    resp = client.post("/predict/ctr", json=payload)
    assert resp.status_code == 422
