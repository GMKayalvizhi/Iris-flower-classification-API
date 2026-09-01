import pytest
from fastapi.testclient import TestClient
from app.main import app


@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c


VALID_INPUT = {
    "sepal_length": 5.1,
    "sepal_width": 3.5,
    "petal_length": 1.4,
    "petal_width": 0.2,
}


def test_valid_input_returns_200(client):
    response = client.post("api/v1/predict", json=VALID_INPUT)
    assert response.status_code == 200
    assert "prediction" in response.json()


def test_response_matches_prediction_output_shape(client):
    response = client.post("api/v1/predict", json=VALID_INPUT)
    assert response.status_code == 200
    body = response.json()
    assert set(body.keys()) == {"prediction", "confidence", "model_version", "request_id"}
    assert isinstance(body["prediction"], str)
    assert isinstance(body["confidence"], float)
    assert isinstance(body["model_version"], str)
    assert isinstance(body["request_id"], str)


def test_boundary_values_are_accepted(client):
    boundary_input = {
        "sepal_length": 4.3,   # min
        "sepal_width": 4.4,    # max
        "petal_length": 6.9,   # max
        "petal_width": 0.1,    # min
    }
    response = client.post("api/v1/predict", json=boundary_input)
    assert response.status_code == 200


def test_value_below_minimum_returns_422(client):
    bad_input = {**VALID_INPUT, "sepal_length": 4.0}
    response = client.post("api/v1/predict", json=bad_input)
    assert response.status_code == 422


def test_value_above_maximum_returns_422(client):
    bad_input = {**VALID_INPUT, "petal_width": 3.0}
    response = client.post("api/v1/predict", json=bad_input)
    assert response.status_code == 422


def test_missing_field_returns_422(client):
    incomplete_input = {
        "sepal_length": 5.1,
        "sepal_width": 3.5,
        "petal_length": 1.4,
    }
    response = client.post("api/v1/predict", json=incomplete_input)
    assert response.status_code == 422


def test_non_numeric_input_returns_422(client):
    bad_type_input = {**VALID_INPUT, "sepal_length": "not-a-number"}
    response = client.post("api/v1/predict", json=bad_type_input)
    assert response.status_code == 422


def test_extra_field_returns_422(client):
    bad_input = {**VALID_INPUT, "extra_field": "hello"}
    response = client.post("api/v1/predict", json=bad_input)
    assert response.status_code == 422


def test_internal_failure_returns_500_with_safe_message(client, monkeypatch):
    def broken_predict(*args, **kwargs):
        raise RuntimeError("simulated internal failure")

    from app.main import ml_models
    original_model = ml_models["iris_classifier"]
    monkeypatch.setattr(original_model, "predict", broken_predict)

    response = client.post("api/v1/predict", json=VALID_INPUT)
    assert response.status_code == 500
    body = response.json()
    assert body["detail"]["message"] == "Prediction failed"
    assert "request_id" in body["detail"]


def test_value_error_returns_400_with_safe_message(client, monkeypatch):
    def broken_predict(*args, **kwargs):
        raise ValueError("simulated shape mismatch")

    from app.main import ml_models
    original_model = ml_models["iris_classifier"]
    monkeypatch.setattr(original_model, "predict", broken_predict)

    response = client.post("api/v1/predict", json=VALID_INPUT)
    assert response.status_code == 400
    body = response.json()
    assert body["detail"] == "Invalid input shape or value for prediction"
    assert "request_id" in body


# ---------- Task 9: logging & request ID tests ----------

def test_request_id_present_in_response_header(client):
    response = client.post("api/v1/predict", json=VALID_INPUT)
    assert response.status_code == 200
    assert "X-Request-ID" in response.headers


def test_request_id_matches_between_header_and_body(client):
    response = client.post("api/v1/predict", json=VALID_INPUT)
    assert response.status_code == 200
    header_id = response.headers["X-Request-ID"]
    body_id = response.json()["request_id"]
    assert header_id == body_id


def test_successful_prediction_logs_info_level(client, caplog):
    with caplog.at_level("INFO"):
        response = client.post("api/v1/predict", json=VALID_INPUT)

    assert response.status_code == 200
    assert any("prediction=" in record.message for record in caplog.records)


def test_value_error_logs_at_error_level(client, monkeypatch, caplog):
    def broken_predict(*args, **kwargs):
        raise ValueError("simulated shape mismatch")

    from app.main import ml_models
    original_model = ml_models["iris_classifier"]
    monkeypatch.setattr(original_model, "predict", broken_predict)

    with caplog.at_level("ERROR"):
        response = client.post("api/v1/predict", json=VALID_INPUT)

    assert response.status_code == 400
    error_logs = [r for r in caplog.records if r.levelname == "ERROR"]
    assert len(error_logs) > 0
    assert any("ValueError" in record.message for record in error_logs)


def test_internal_failure_logs_at_error_level(client, monkeypatch, caplog):
    def broken_predict(*args, **kwargs):
        raise RuntimeError("simulated internal failure")

    from app.main import ml_models
    original_model = ml_models["iris_classifier"]
    monkeypatch.setattr(original_model, "predict", broken_predict)

    with caplog.at_level("ERROR"):
        response = client.post("api/v1/predict", json=VALID_INPUT)

    assert response.status_code == 500
    error_logs = [r for r in caplog.records if r.levelname == "ERROR"]
    assert len(error_logs) > 0
    assert any("Unexpected error" in record.message for record in error_logs)


def test_every_request_gets_logged_with_duration(client, caplog):
    with caplog.at_level("INFO"):
        response = client.get("api/v1/health")

    assert response.status_code == 200
    assert any(
        "method=GET" in record.message and "duration_ms=" in record.message
        for record in caplog.records
    )

 #---------- Task 11: /predict-batch tests ----------
 
BATCH_INPUT = {
    "inputs": [
        {"sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2},
        {"sepal_length": 6.7, "sepal_width": 3.1, "petal_length": 4.7, "petal_width": 1.5},
    ]
}
 
 
def test_batch_valid_input_returns_200(client):
    response = client.post("/api/v1/predict-batch", json=BATCH_INPUT)
    assert response.status_code == 200
    body = response.json()
    assert len(body["predictions"]) == 2
    assert body["count"] == 2
 
 
def test_batch_response_matches_expected_shape(client):
    response = client.post("/api/v1/predict-batch", json=BATCH_INPUT)
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
 
 
def test_batch_single_item_is_accepted(client):
    single_item_batch = {"inputs": [VALID_INPUT]}
    response = client.post("/api/v1/predict-batch", json=single_item_batch)
    assert response.status_code == 200
    assert response.json()["count"] == 1
 
 
def test_batch_max_size_100_is_accepted(client):
    max_batch = {"inputs": [VALID_INPUT] * 100}
    response = client.post("/api/v1/predict-batch", json=max_batch)
    assert response.status_code == 200
    assert response.json()["count"] == 100
 
 
def test_batch_over_max_size_returns_422(client):
    oversized_batch = {"inputs": [VALID_INPUT] * 101}
    response = client.post("/api/v1/predict-batch", json=oversized_batch)
    assert response.status_code == 422
 
 
def test_batch_empty_list_returns_422(client):
    empty_batch = {"inputs": []}
    response = client.post("/api/v1/predict-batch", json=empty_batch)
    assert response.status_code == 422
 
 
def test_batch_invalid_item_returns_422(client):
    bad_batch = {"inputs": [VALID_INPUT, {**VALID_INPUT, "sepal_length": 4.0}]}
    response = client.post("/api/v1/predict-batch", json=bad_batch)
    assert response.status_code == 422
 
 
def test_batch_request_id_present_and_consistent(client):
    response = client.post("/api/v1/predict-batch", json=BATCH_INPUT)
    assert response.status_code == 200
    body = response.json()
    assert "X-Request-ID" in response.headers
    assert response.headers["X-Request-ID"] == body["request_id"]
 
 
def test_batch_internal_failure_returns_500_with_safe_message(client, monkeypatch):
    def broken_predict(*args, **kwargs):
        raise RuntimeError("simulated internal failure")
 
    from app.state import ml_models
    original_model = ml_models["iris_classifier"]
    monkeypatch.setattr(original_model, "predict", broken_predict)
 
    response = client.post("/api/v1/predict-batch", json=BATCH_INPUT)
    assert response.status_code == 500
    body = response.json()
    assert body["detail"]["message"] == "Batch prediction failed"
    assert "request_id" in body["detail"]
 
 
def test_batch_logs_batch_size_and_duration(client, caplog):
    with caplog.at_level("INFO"):
        response = client.post("/api/v1/predict-batch", json=BATCH_INPUT)
 
    assert response.status_code == 200
    assert any(
        "batch_size=2" in record.message and "batch_prediction_duration_ms=" in record.message
        for record in caplog.records
    )
 
 
# ---------- Task 11: /model-info tests ----------
 
def test_model_info_returns_200(client):
    response = client.get("/api/v1/model-info")
    assert response.status_code == 200
 
 
def test_model_info_response_matches_expected_shape(client):
    response = client.get("/api/v1/model-info")
    assert response.status_code == 200
    body = response.json()
    assert set(body.keys()) == {
        "model_type", "model_version", "trained_on",
        "feature_names", "target_names", "n_estimators", "test_accuracy",
    }
    assert isinstance(body["model_type"], str)
    assert isinstance(body["feature_names"], list)
    assert isinstance(body["n_estimators"], int)
    assert isinstance(body["test_accuracy"], float)
 
 
def test_model_info_returns_real_values_not_placeholders(client):
    response = client.get("/api/v1/model-info")
    assert response.status_code == 200
    body = response.json()
    assert body["model_type"] == "RandomForestClassifier"
    assert len(body["feature_names"]) == 4
    assert len(body["target_names"]) == 3
    assert 0.0 < body["test_accuracy"] <= 1.0
 
 
def test_model_info_matches_predict_response_model_version(client):
    info_response = client.get("/api/v1/model-info")
    predict_response = client.post("/api/v1/predict", json=VALID_INPUT)
    assert info_response.json()["model_version"] == predict_response.json()["model_version"]
 
 
def test_model_info_failure_returns_500_with_safe_message(client, monkeypatch):
    from app.state import ml_models
    original_info = ml_models["model_info"]
    # Simulate a malformed model_info.json missing a required field
    monkeypatch.setitem(ml_models, "model_info", {"model_type": "RandomForestClassifier"})
 
    response = client.get("/api/v1/model-info")
    assert response.status_code == 500
    body = response.json()
    assert body["detail"]["message"] == "Failed to retrieve model info"
    assert "request_id" in body["detail"]
 
    ml_models["model_info"] = original_info
 
 
def test_model_info_logs_at_info_level(client, caplog):
    with caplog.at_level("INFO"):
        response = client.get("/api/v1/model-info")
 
    assert response.status_code == 200
    assert any("model-info requested" in record.message for record in caplog.records)
 








