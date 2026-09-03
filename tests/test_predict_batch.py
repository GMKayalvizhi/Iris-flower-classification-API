from app.config import settings


def test_batch_valid_input_returns_200(client, batch_input):
    response = client.post("/api/v1/predict-batch", json=batch_input)
    assert response.status_code == 200
    body = response.json()
    assert len(body["predictions"]) == 2
    assert body["count"] == 2


def test_batch_response_matches_expected_shape(client, batch_input):
    response = client.post("/api/v1/predict-batch", json=batch_input)
    assert response.status_code == 200
    body = response.json()
    assert set(body.keys()) == {"predictions", "count", "request_id"}
    for item in body["predictions"]:
        assert set(item.keys()) == {"prediction", "confidence", "model_version"}
        assert isinstance(item["prediction"], str)
        assert isinstance(item["confidence"], float)
        assert isinstance(item["model_version"], str)


def test_batch_predictions_preserve_input_order(client):
    ordered_input = {
        "inputs": [
            {"sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2},  # setosa-like
            {"sepal_length": 6.3, "sepal_width": 2.9, "petal_length": 5.6, "petal_width": 1.8},  # virginica-like
        ]
    }
    response = client.post("/api/v1/predict-batch", json=ordered_input)
    assert response.status_code == 200
    predictions = response.json()["predictions"]
    assert predictions[0]["prediction"] == "setosa"
    assert predictions[1]["prediction"] == "virginica"


def test_batch_single_item_is_accepted(client, valid_input):
    single_item_batch = {"inputs": [valid_input]}
    response = client.post("/api/v1/predict-batch", json=single_item_batch)
    assert response.status_code == 200
    assert response.json()["count"] == 1


def test_batch_empty_list_returns_422(client):
    empty_batch = {"inputs": []}
    response = client.post("/api/v1/predict-batch", json=empty_batch)
    assert response.status_code == 422


def test_batch_invalid_item_returns_422(client, valid_input):
    bad_batch = {"inputs": [valid_input, {**valid_input, "sepal_length": 4.0}]}
    response = client.post("/api/v1/predict-batch", json=bad_batch)
    assert response.status_code == 422


def test_batch_request_id_present_and_consistent(client, batch_input):
    response = client.post("/api/v1/predict-batch", json=batch_input)
    assert response.status_code == 200
    body = response.json()
    assert "X-Request-ID" in response.headers
    assert response.headers["X-Request-ID"] == body["request_id"]


def test_batch_internal_failure_returns_500_with_safe_message(client, batch_input, monkeypatch):
    def broken_predict(*args, **kwargs):
        raise RuntimeError("simulated internal failure")

    from app.state import ml_models
    original_model = ml_models["iris_classifier"]
    monkeypatch.setattr(original_model, "predict", broken_predict)

    response = client.post("/api/v1/predict-batch", json=batch_input)
    assert response.status_code == 500
    body = response.json()
    assert body["detail"]["message"] == "Batch prediction failed"
    assert "request_id" in body["detail"]


def test_batch_logs_batch_size_and_duration(client, batch_input, caplog):
    with caplog.at_level("INFO"):
        response = client.post("/api/v1/predict-batch", json=batch_input)

    assert response.status_code == 200
    assert any(
        "batch_size=2" in record.message and "batch_prediction_duration_ms=" in record.message
        for record in caplog.records
    )


# ---------- MAX_BATCH_SIZE enforcement (Task 12 config-driven behavior) ----------

def test_max_batch_size_is_enforced_from_current_settings(client, valid_input):
    # Uses settings.MAX_BATCH_SIZE dynamically rather than hardcoding
    # "100", so this stays correct even if the configured default changes.
    at_limit = {"inputs": [valid_input] * settings.MAX_BATCH_SIZE}
    response = client.post("/api/v1/predict-batch", json=at_limit)
    assert response.status_code == 200

    over_limit = {"inputs": [valid_input] * (settings.MAX_BATCH_SIZE + 1)}
    response = client.post("/api/v1/predict-batch", json=over_limit)
    assert response.status_code == 422


def test_max_batch_size_is_dynamically_configurable(client, valid_input, monkeypatch):
    # Proves the limit is genuinely config-driven, not hardcoded: change
    # settings.MAX_BATCH_SIZE mid-test (no restart) and confirm the API
    # obeys the new value immediately. Only works because
    # PredictionBatchInput checks settings.MAX_BATCH_SIZE inside a
    # field_validator (evaluated per-request), not a static
    # Field(max_length=...) (evaluated once, at import time).
    monkeypatch.setattr(settings, "MAX_BATCH_SIZE", 3)

    within_new_limit = {"inputs": [valid_input] * 3}
    response = client.post("/api/v1/predict-batch", json=within_new_limit)
    assert response.status_code == 200

    over_new_limit = {"inputs": [valid_input] * 4}
    response = client.post("/api/v1/predict-batch", json=over_new_limit)
    assert response.status_code == 422