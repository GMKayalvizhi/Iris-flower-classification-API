def test_root_returns_200(client):
    response = client.get("/")
    assert response.status_code == 200


def test_health_returns_200_and_expected_shape(client):
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    body = response.json()
    assert set(body.keys()) == {"status", "model_loaded"}
    assert body["status"] == "ok"
    assert body["model_loaded"] is True


def test_health_is_logged_with_duration(client, caplog):
    with caplog.at_level("INFO"):
        response = client.get("/api/v1/health")

    assert response.status_code == 200
    assert any(
        "method=GET" in record.message and "duration_ms=" in record.message
        for record in caplog.records
    )