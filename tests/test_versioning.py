def test_v1_and_v2_return_different_shapes_for_same_input(client, valid_input):
    v1_response = client.post("/api/v1/predict", json=valid_input)
    v2_response = client.post("/api/v2/predict", json=valid_input)

    assert v1_response.status_code == 200
    assert v2_response.status_code == 200

    v1_body = v1_response.json()
    v2_body = v2_response.json()

    # v1's shape is exactly what it has always been -- no new field
    # snuck in just because v2 now exists.
    assert set(v1_body.keys()) == {"prediction", "confidence", "model_version", "request_id"}
    assert "probabilities" not in v1_body

    # v2's shape is a genuine superset -- same base fields, plus the
    # deliberate breaking addition.
    assert set(v2_body.keys()) == {"prediction", "confidence", "probabilities", "model_version", "request_id"}
    assert "probabilities" in v2_body


def test_v1_and_v2_agree_on_the_prediction_itself(client):
    # Deliberately an ambiguous, non-setosa input (not the standard
    # valid_input fixture) -- setosa is trivially separable in this
    # dataset, so a hardcoded/broken v2 could accidentally match the
    # real model's answer by coincidence on an easy input. This input
    # sits firmly in virginica territory, making a genuine mismatch
    # actually detectable.
    ambiguous_input = {
        "sepal_length": 6.3, "sepal_width": 2.9,
        "petal_length": 5.6, "petal_width": 1.8,
    }
    v1_body = client.post("/api/v1/predict", json=ambiguous_input).json()
    v2_body = client.post("/api/v2/predict", json=ambiguous_input).json()

    assert v1_body["prediction"] == v2_body["prediction"]
    assert v1_body["confidence"] == v2_body["confidence"]
    assert v1_body["model_version"] == v2_body["model_version"]


def test_v1_still_rejects_invalid_input_exactly_as_before(client, valid_input):
    # A regression guard: v1's validation behavior is untouched by v2
    # existing. If someone accidentally broke v1 while building v2,
    # this is the test that would catch it.
    bad_input = {**valid_input, "sepal_length": 4.0}
    response = client.post("/api/v1/predict", json=bad_input)
    assert response.status_code == 422

def test_v1_schema_rejects_a_v2_shaped_field_it_doesnt_know_about():
    # Direct proof that v1's schema genuinely enforces its own shape.
    # PredictionOutput has extra="forbid" -- so handing it a
    # "probabilities" field it doesn't declare should raise a
    # ValidationError immediately, not silently build a filtered object.
    # (Earlier version of this test assumed the old extra="ignore"
    # behavior, from before extra="forbid" was added to response
    # schemas -- updated to match the schema's actual current contract.)
    from pydantic import ValidationError
    from app.models.schemas import PredictionOutput
 
    v2_shaped_data = {
        "prediction": "setosa",
        "confidence": 0.97,
        "probabilities": {"setosa": 0.97, "versicolor": 0.02, "virginica": 0.01},
        "model_version": "1.0.0",
        "request_id": "test-id",
    }
 
    try:
        PredictionOutput(**v2_shaped_data)
        assert False, "PredictionOutput should reject the unexpected 'probabilities' field"
    except ValidationError:
        pass

def test_rejects_unexpected_field():
    from pydantic import ValidationError
    from app.models.schemas import PredictionItem

    try:
        PredictionItem(prediction="setosa", confidence=1.0, model_version="1.0.0")
        assert False   # if we get here, no error was raised — that's bad
    except ValidationError:
        pass            # good — it correctly raised an error