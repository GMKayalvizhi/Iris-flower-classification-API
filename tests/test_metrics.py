from app.metrics import PREDICTION_ENTROPY_BITS


def _entropy_count_and_sum(api_version):
    for metric in PREDICTION_ENTROPY_BITS.collect():
        count = sum_ = None
        for sample in metric.samples:
            if sample.labels.get("api_version") != api_version:
                continue
            if sample.name.endswith("_count"):
                count = sample.value
            elif sample.name.endswith("_sum"):
                sum_ = sample.value
        if count is not None and sum_ is not None:
            return count, sum_
    return 0.0, 0.0


def test_metrics_endpoint_returns_prometheus_text(client):
    response = client.get("/metrics")
    assert response.status_code == 200
    assert "text/plain" in response.headers["content-type"]


def test_metrics_endpoint_does_not_require_api_key(client):
    response = client.get("/metrics", headers={})
    assert response.status_code == 200


def test_successful_prediction_records_entropy_observation(client, valid_input):
    count_before, _ = _entropy_count_and_sum("v1")
    response = client.post("/api/v1/predict", json=valid_input)
    assert response.status_code == 200
    count_after, _ = _entropy_count_and_sum("v1")
    assert count_after == count_before + 1


def test_confident_prediction_records_near_zero_entropy(client, valid_input):
    # valid_input is a textbook setosa case -- the model should be fully
    # decisive, so this one observation's contribution to entropy is ~0 bits.
    count_before, sum_before = _entropy_count_and_sum("v1")
    response = client.post("/api/v1/predict", json=valid_input)
    assert response.status_code == 200
    count_after, sum_after = _entropy_count_and_sum("v1")
    this_observation = sum_after - sum_before
    assert count_after == count_before + 1
    assert this_observation < 0.1


def test_batch_prediction_records_one_observation_per_item(client, batch_input):
    count_before, _ = _entropy_count_and_sum("v1")
    response = client.post("/api/v1/predict-batch", json=batch_input)
    assert response.status_code == 200
    count_after, _ = _entropy_count_and_sum("v1")
    assert count_after == count_before + len(batch_input["inputs"])


def test_failed_validation_does_not_record_entropy(client, valid_input):
    bad_input = {**valid_input, "sepal_length": 4.0}  # below minimum -> 422
    count_before, _ = _entropy_count_and_sum("v1")
    response = client.post("/api/v1/predict", json=bad_input)
    assert response.status_code == 422
    count_after, _ = _entropy_count_and_sum("v1")
    assert count_after == count_before


def test_v2_prediction_labeled_separately_from_v1(client, valid_input):
    v1_count_before, _ = _entropy_count_and_sum("v1")
    v2_count_before, _ = _entropy_count_and_sum("v2")
    response = client.post("/api/v2/predict", json=valid_input)
    assert response.status_code == 200
    v1_count_after, _ = _entropy_count_and_sum("v1")
    v2_count_after, _ = _entropy_count_and_sum("v2")
    assert v2_count_after == v2_count_before + 1
    assert v1_count_after == v1_count_before