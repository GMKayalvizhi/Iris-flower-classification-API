"""
Security and validation edge-case tests (Task 17).

Covers: missing/invalid API key rejection, unexpected extra fields on
both single-predict and batch-predict schemas, and confirms a valid
key still succeeds. The unauthenticated /health endpoint is covered
separately in test_health.py, not here.
"""

from app.config import settings


def test_predict_without_api_key_returns_401(client, valid_input):
    """No X-API-Key header at all -> 401."""
    client.headers.pop("X-API-Key", None)     # truly remove the header
    response = client.post("/api/v1/predict", json=valid_input)
    assert response.status_code == 401


def test_predict_with_invalid_api_key_returns_401(client, valid_input):
    """Wrong key value -> 401, same as a missing key."""
    response = client.post(
        "/api/v1/predict",
        json=valid_input,
        headers={"X-API-Key": "definitely-the-wrong-key"},
    )
    assert response.status_code == 401


def test_predict_with_valid_api_key_succeeds(client, valid_input):
    """Sanity check: the correct key still works end-to-end."""
    response = client.post(
        "/api/v1/predict",
        json=valid_input,
        headers={"X-API-Key": settings.API_KEY},
    )
    assert response.status_code == 200


def test_predict_rejects_unexpected_extra_field(client, valid_input):
    """extra='forbid' on IrisInput -> unknown field is a 422, not silently dropped."""
    payload = {**valid_input, "unexpected_field": "should be rejected"}
    response = client.post(
        "/api/v1/predict",
        json=payload,
        headers={"X-API-Key": settings.API_KEY},
    )
    assert response.status_code == 422


def test_predict_batch_rejects_unexpected_extra_field(client, valid_input):
    """
    Same check as above, but on PredictionBatchInput -- this schema was
    missing extra='forbid' until this task; this test is what would have
    caught that gap.
    """
    payload = {
        "inputs": [valid_input],
        "unexpected_field": "should be rejected",
    }
    response = client.post(
        "/api/v1/predict-batch",
        json=payload,
        headers={"X-API-Key": settings.API_KEY},
    )
    assert response.status_code == 422


def test_predict_without_api_key_returns_401(client, valid_input):
    """No X-API-Key header at all -> 401."""
    client.headers.pop("X-API-Key", None)     # truly remove the header
    response = client.post("/api/v2/predict", json=valid_input)
    assert response.status_code == 401


def test_health_does_not_require_api_key():
    from fastapi.testclient import TestClient
    from app.main import app
    with TestClient(app) as bare_client:      # no default key header
        response = bare_client.get("/api/v1/health")
    assert response.status_code == 200