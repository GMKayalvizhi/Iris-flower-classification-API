# Central configuration for the Iris API.
# Settings are loaded from environment variables and a local .env file.
# This keeps configuration separate from code and allows the same
# application to run across different environments without code changes.
#
# This module has no dependencies on other app modules to avoid
# circular import issues.
import secrets
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Every field here has a sane default, except API_KEY: a secret must never
    have a public default, so if it is not set a random key is generated at
    startup (and every client gets 401 until a real key is configured).
    Environment variables / .env values override these defaults when present.
    """

    MODEL_PATH: str = "ml/saved_model/model.joblib"
    MODEL_INFO_PATH: str = "ml/saved_model/model_info.json"
    LOG_LEVEL: str = "INFO"
    MAX_BATCH_SIZE: int = 100
    API_TITLE: str = "Iris Flower Classification API"
    API_KEY: str = Field(default_factory=lambda: secrets.token_hex(16))  # random if unset; real value comes from .env or the host's environment
    ALLOWED_ORIGINS: str = "http://localhost:3000"  # comma-separated list


    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
    )


# Instantiated once, at import time. Every other module imports this
# same `settings` object rather than instantiating Settings() again --
# one object, read many times, same pattern as ml_models in state.py.
settings = Settings()