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


def test_model_info_matches_predict_response_model_version(client, valid_input):
    info_response = client.get("/api/v1/model-info")
    predict_response = client.post("/api/v1/predict", json=valid_input)
    assert info_response.json()["model_version"] == predict_response.json()["model_version"]


def test_model_info_failure_returns_500_with_safe_message(client, monkeypatch):
    from app.state import ml_models
    original_info = ml_models["model_info"]
    # Simulate a malformed model_info.json missing required fields
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