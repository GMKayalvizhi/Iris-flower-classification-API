from app.config import settings


def test_settings_has_expected_fields():
    # A basic sanity check that Settings loaded and every field has a
    # usable value -- catches a broken/empty .env before it causes
    # confusing failures elsewhere.
    assert settings.MODEL_PATH
    assert settings.MODEL_INFO_PATH
    assert settings.LOG_LEVEL
    assert isinstance(settings.MAX_BATCH_SIZE, int)
    assert settings.MAX_BATCH_SIZE > 0
    assert settings.API_TITLE


def test_api_title_from_settings_appears_in_root_response(client):
    response = client.get("/")
    assert response.status_code == 200
    assert settings.API_TITLE in response.json()["message"]