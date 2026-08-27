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
    response = client.post("/predict", json=VALID_INPUT)
    assert response.status_code == 200
    assert "prediction" in response.json()

def test_response_matches_prediction_output_shape(client):
    response = client.post("/predict", json=VALID_INPUT)
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
    response = client.post("/predict", json=boundary_input)
    assert response.status_code == 200


def test_value_below_minimum_returns_422(client):
    bad_input = {**VALID_INPUT, "sepal_length": 4.0}
    response = client.post("/predict", json=bad_input)
    assert response.status_code == 422


def test_value_above_maximum_returns_422(client):
    bad_input = {**VALID_INPUT, "petal_width": 3.0}
    response = client.post("/predict", json=bad_input)
    assert response.status_code == 422


def test_missing_field_returns_422(client):
    incomplete_input = {
        "sepal_length": 5.1,
        "sepal_width": 3.5,
        "petal_length": 1.4,
    }
    response = client.post("/predict", json=incomplete_input)
    assert response.status_code == 422


def test_non_numeric_input_returns_422(client):
    bad_type_input = {**VALID_INPUT, "sepal_length": "not-a-number"}
    response = client.post("/predict", json=bad_type_input)
    assert response.status_code == 422

def test_extra_field_returns_422(client):
    bad_input = {**VALID_INPUT, "extra_field": "hello"}
    response = client.post("/predict", json=bad_input)
    assert response.status_code == 422

def test_internal_failure_returns_500_with_safe_message(client, monkeypatch):
    def broken_predict(*args, **kwargs):
        raise RuntimeError("simulated internal failure")

    from app.main import ml_models
    original_model = ml_models["iris_classifier"]
    monkeypatch.setattr(original_model, "predict", broken_predict)

    response = client.post("/predict", json=VALID_INPUT)
    assert response.status_code == 500
    assert response.json() == {"detail": "Prediction failed"}

def test_value_error_returns_400_with_safe_message(client, monkeypatch):
    def broken_predict(*args, **kwargs):
        raise ValueError("simulated shape mismatch")

    from app.main import ml_models
    original_model = ml_models["iris_classifier"]
    monkeypatch.setattr(original_model, "predict", broken_predict)

    response = client.post("/predict", json=VALID_INPUT)
    assert response.status_code == 400
    assert response.json() == {"detail": "Invalid input shape or value for prediction"}