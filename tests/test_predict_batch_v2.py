from app.config import settings


def test_v2_batch_valid_input_returns_200(client, batch_input):
    response = client.post("/api/v2/predict-batch", json=batch_input)
    assert response.status_code == 200
    body = response.json()
    assert len(body["predictions"]) == 2
    assert body["count"] == 2


def test_v2_batch_response_matches_expected_shape(client, batch_input):
    response = client.post("/api/v2/predict-batch", json=batch_input)
    assert response.status_code == 200
    body = response.json()
    assert set(body.keys()) == {"predictions", "count", "model_version", "request_id"}
    assert isinstance(body["model_version"], str)
    for item in body["predictions"]:
        assert set(item.keys()) == {"prediction", "confidence", "probabilities"}
        assert isinstance(item["probabilities"], dict)
        assert set(item["probabilities"].keys()) == {"setosa", "versicolor", "virginica"}


def test_v2_batch_probabilities_sum_to_one_per_item(client, batch_input):
    response = client.post("/api/v2/predict-batch", json=batch_input)
    assert response.status_code == 200
    for item in response.json()["predictions"]:
        assert abs(sum(item["probabilities"].values()) - 1.0) < 1e-6


def test_v2_batch_empty_list_returns_422(client):
    response = client.post("/api/v2/predict-batch", json={"inputs": []})
    assert response.status_code == 422


def test_v2_batch_respects_same_max_batch_size_as_v1(client, valid_input):
    # Confirms v2 shares v1's request schema (PredictionBatchInput),
    # including the Task 12 config-driven size limit -- not a
    # reimplemented, potentially inconsistent copy of it.
    over_limit = {"inputs": [valid_input] * (settings.MAX_BATCH_SIZE + 1)}
    response = client.post("/api/v2/predict-batch", json=over_limit)
    assert response.status_code == 422


def test_v2_batch_internal_failure_returns_500_with_safe_message(client, batch_input, monkeypatch):
    def broken_predict(*args, **kwargs):
        raise RuntimeError("simulated internal failure")

    from app.state import ml_models
    original_model = ml_models["iris_classifier"]
    monkeypatch.setattr(original_model, "predict", broken_predict)

    response = client.post("/api/v2/predict-batch", json=batch_input)
    assert response.status_code == 500
    body = response.json()
    assert body["detail"]["message"] == "Batch prediction failed"
    assert "request_id" in body["detail"]


def test_v2_batch_request_id_present_and_consistent(client, batch_input):
    response = client.post("/api/v2/predict-batch", json=batch_input)
    assert response.status_code == 200
    body = response.json()
    assert response.headers["X-Request-ID"] == body["request_id"]


def test_v1_and_v2_batch_endpoints_return_different_item_shapes(client, batch_input):
    # The batch-endpoint version of the mini challenge from
    # test_versioning.py -- same proof, applied to /predict-batch.
    v1_response = client.post("/api/v1/predict-batch", json=batch_input)
    v2_response = client.post("/api/v2/predict-batch", json=batch_input)

    assert v1_response.status_code == 200
    assert v2_response.status_code == 200

    v1_item = v1_response.json()["predictions"][0]
    v2_item = v2_response.json()["predictions"][0]

    assert "probabilities" not in v1_item
    assert "probabilities" in v2_item
    assert v1_item["prediction"] == v2_item["prediction"]