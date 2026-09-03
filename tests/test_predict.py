def test_valid_input_returns_200(client, valid_input):
    response = client.post("/api/v1/predict", json=valid_input)
    assert response.status_code == 200
    assert "prediction" in response.json()


def test_response_matches_prediction_output_shape(client, valid_input):
    response = client.post("/api/v1/predict", json=valid_input)
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
    response = client.post("/api/v1/predict", json=boundary_input)
    assert response.status_code == 200


def test_value_below_minimum_returns_422(client, valid_input):
    bad_input = {**valid_input, "sepal_length": 4.0}
    response = client.post("/api/v1/predict", json=bad_input)
    assert response.status_code == 422


def test_value_above_maximum_returns_422(client, valid_input):
    bad_input = {**valid_input, "petal_width": 3.0}
    response = client.post("/api/v1/predict", json=bad_input)
    assert response.status_code == 422


def test_missing_field_returns_422(client):
    incomplete_input = {
        "sepal_length": 5.1,
        "sepal_width": 3.5,
        "petal_length": 1.4,
    }
    response = client.post("/api/v1/predict", json=incomplete_input)
    assert response.status_code == 422


def test_non_numeric_input_returns_422(client, valid_input):
    bad_type_input = {**valid_input, "sepal_length": "not-a-number"}
    response = client.post("/api/v1/predict", json=bad_type_input)
    assert response.status_code == 422


def test_extra_field_returns_422(client, valid_input):
    bad_input = {**valid_input, "extra_field": "hello"}
    response = client.post("/api/v1/predict", json=bad_input)
    assert response.status_code == 422


def test_internal_failure_returns_500_with_safe_message(client, valid_input, monkeypatch):
    def broken_predict(*args, **kwargs):
        raise RuntimeError("simulated internal failure")

    from app.state import ml_models
    original_model = ml_models["iris_classifier"]
    monkeypatch.setattr(original_model, "predict", broken_predict)

    response = client.post("/api/v1/predict", json=valid_input)
    assert response.status_code == 500
    body = response.json()
    assert body["detail"]["message"] == "Prediction failed"
    assert "request_id" in body["detail"]


def test_value_error_returns_400_with_safe_message(client, valid_input, monkeypatch):
    def broken_predict(*args, **kwargs):
        raise ValueError("simulated shape mismatch")

    from app.state import ml_models
    original_model = ml_models["iris_classifier"]
    monkeypatch.setattr(original_model, "predict", broken_predict)

    response = client.post("/api/v1/predict", json=valid_input)
    assert response.status_code == 400
    body = response.json()
    assert body["detail"] == "Invalid input shape or value for prediction"
    assert "request_id" in body


def test_request_id_present_in_response_header(client, valid_input):
    response = client.post("/api/v1/predict", json=valid_input)
    assert response.status_code == 200
    assert "X-Request-ID" in response.headers


def test_request_id_matches_between_header_and_body(client, valid_input):
    response = client.post("/api/v1/predict", json=valid_input)
    assert response.status_code == 200
    header_id = response.headers["X-Request-ID"]
    body_id = response.json()["request_id"]
    assert header_id == body_id


def test_successful_prediction_logs_info_level(client, valid_input, caplog):
    with caplog.at_level("INFO"):
        response = client.post("/api/v1/predict", json=valid_input)

    assert response.status_code == 200
    assert any("prediction=" in record.message for record in caplog.records)


def test_value_error_logs_at_error_level(client, valid_input, monkeypatch, caplog):
    def broken_predict(*args, **kwargs):
        raise ValueError("simulated shape mismatch")

    from app.state import ml_models
    original_model = ml_models["iris_classifier"]
    monkeypatch.setattr(original_model, "predict", broken_predict)

    with caplog.at_level("ERROR"):
        response = client.post("/api/v1/predict", json=valid_input)

    assert response.status_code == 400
    error_logs = [r for r in caplog.records if r.levelname == "ERROR"]
    assert len(error_logs) > 0
    assert any("ValueError" in record.message for record in error_logs)


def test_internal_failure_logs_at_error_level(client, valid_input, monkeypatch, caplog):
    def broken_predict(*args, **kwargs):
        raise RuntimeError("simulated internal failure")

    from app.state import ml_models
    original_model = ml_models["iris_classifier"]
    monkeypatch.setattr(original_model, "predict", broken_predict)

    with caplog.at_level("ERROR"):
        response = client.post("/api/v1/predict", json=valid_input)

    assert response.status_code == 500
    error_logs = [r for r in caplog.records if r.levelname == "ERROR"]
    assert len(error_logs) > 0
    assert any("Unexpected error" in record.message for record in error_logs)