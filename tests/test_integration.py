import os
import httpx

BASE_URL = os.environ.get("INTEGRATION_BASE_URL", "http://localhost:8000")
API_KEY = os.environ.get("API_KEY", "")

HEADERS = {"X-API-Key": API_KEY, "Content-Type": "application/json"}

VALID_INPUT = {"sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2}


def test_health_reachable_without_key():
    response = httpx.get(f"{BASE_URL}/api/v1/health")
    assert response.status_code == 200
    assert response.json()["model_loaded"] is True


def test_predict_end_to_end():
    response = httpx.post(f"{BASE_URL}/api/v1/predict", json=VALID_INPUT, headers=HEADERS)
    assert response.status_code == 200
    body = response.json()
    assert body["prediction"] == "setosa"
    assert "request_id" in body


def test_predict_batch_end_to_end():
    batch = {"inputs": [VALID_INPUT, VALID_INPUT]}
    response = httpx.post(f"{BASE_URL}/api/v1/predict-batch", json=batch, headers=HEADERS)
    assert response.status_code == 200
    assert response.json()["count"] == 2


def test_predict_without_key_rejected():
    response = httpx.post(f"{BASE_URL}/api/v1/predict", json=VALID_INPUT)
    assert response.status_code == 401


def test_metrics_reachable_and_contains_entropy_metric():
    response = httpx.get(f"{BASE_URL}/metrics", headers=HEADERS)
    assert response.status_code == 200
    assert "iris_prediction_entropy_bits" in response.text