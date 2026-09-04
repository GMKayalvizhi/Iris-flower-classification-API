# Central configuration for the Iris API.
# Settings are loaded from environment variables and a local .env file.
# This keeps configuration separate from code and allows the same
# application to run across different environments without code changes.
#
# This module has no dependencies on other app modules to avoid
# circular import issues.

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Every field here has a sane default, so the app still runs with no
    .env file present at all (e.g. a fresh clone before anyone's set
    one up). Environment variables / .env values override these
    defaults when present -- pydantic-settings handles that matching
    automatically by field name.
    """

    MODEL_PATH: str = "ml/saved_model/model.joblib"
    MODEL_INFO_PATH: str = "ml/saved_model/model_info.json"
    LOG_LEVEL: str = "INFO"
    MAX_BATCH_SIZE: int = 100
    API_TITLE: str = "Iris Flower Classification API"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
    )


# Instantiated once, at import time. Every other module imports this
# same `settings` object rather than instantiating Settings() again --
# one object, read many times, same pattern as ml_models in state.py.
settings = Settings()