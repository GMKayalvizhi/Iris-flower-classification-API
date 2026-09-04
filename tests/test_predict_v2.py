def test_v2_valid_input_returns_200(client, valid_input):
    response = client.post("/api/v2/predict", json=valid_input)
    assert response.status_code == 200


def test_v2_response_matches_expected_shape(client, valid_input):
    response = client.post("/api/v2/predict", json=valid_input)
    assert response.status_code == 200
    body = response.json()
    assert set(body.keys()) == {"prediction", "confidence", "probabilities", "model_version", "request_id"}
    assert isinstance(body["prediction"], str)
    assert isinstance(body["confidence"], float)
    assert isinstance(body["probabilities"], dict)
    assert isinstance(body["model_version"], str)
    assert isinstance(body["request_id"], str)


def test_v2_probabilities_cover_all_three_species(client, valid_input):
    response = client.post("/api/v2/predict", json=valid_input)
    assert response.status_code == 200
    probabilities = response.json()["probabilities"]
    assert set(probabilities.keys()) == {"setosa", "versicolor", "virginica"}


def test_v2_probabilities_sum_to_approximately_one(client, valid_input):
    response = client.post("/api/v2/predict", json=valid_input)
    assert response.status_code == 200
    probabilities = response.json()["probabilities"]
    assert abs(sum(probabilities.values()) - 1.0) < 1e-6


def test_v2_probabilities_are_each_between_zero_and_one(client, valid_input):
    response = client.post("/api/v2/predict", json=valid_input)
    assert response.status_code == 200
    probabilities = response.json()["probabilities"]
    assert all(0.0 <= p <= 1.0 for p in probabilities.values())


def test_v2_winning_probability_matches_confidence(client, valid_input):
    # The "confidence" field should always equal the probability of the
    # predicted class within the "probabilities" breakdown -- they're
    # two views of the same number, not independently computed.
    response = client.post("/api/v2/predict", json=valid_input)
    assert response.status_code == 200
    body = response.json()
    predicted_species = body["prediction"]
    assert body["probabilities"][predicted_species] == body["confidence"]


def test_v2_same_validation_rules_as_v1(client, valid_input):
    # v2 reuses IrisInput (unchanged), so the same bounds apply --
    # this is REQUEST validation shared between versions, not part of
    # the breaking change, which is only about the RESPONSE shape.
    bad_input = {**valid_input, "sepal_length": 4.0}
    response = client.post("/api/v2/predict", json=bad_input)
    assert response.status_code == 422


def test_v2_internal_failure_returns_500_with_safe_message(client, valid_input, monkeypatch):
    def broken_predict(*args, **kwargs):
        raise RuntimeError("simulated internal failure")

    from app.state import ml_models
    original_model = ml_models["iris_classifier"]
    monkeypatch.setattr(original_model, "predict", broken_predict)

    response = client.post("/api/v2/predict", json=valid_input)
    assert response.status_code == 500
    body = response.json()
    assert body["detail"]["message"] == "Prediction failed"
    assert "request_id" in body["detail"]


def test_v2_request_id_present_in_header_and_body(client, valid_input):
    response = client.post("/api/v2/predict", json=valid_input)
    assert response.status_code == 200
    assert response.headers["X-Request-ID"] == response.json()["request_id"]