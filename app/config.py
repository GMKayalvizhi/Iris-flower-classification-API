# app/config.py
#
# Central configuration for the Iris API, sourced from environment
# variables (and a local .env file for development convenience). This
# follows the twelve-factor app principle: config lives in the
# environment, not hardcoded in code, so the exact same code can run in
# local / staging / production with different values -- no code edits,
# no per-environment branches in the app itself.
#
# IMPORTANT: this module must never import anything from the rest of
# app/. Every other module (logging_config.py, schemas.py, main.py)
# imports `settings` FROM here at import time or startup time. If this
# file imported something from, say, app.state, and that module (or one
# of its dependents) imported config.py back, you'd get the same kind
# of circular import problem Task 10 solved with app/state.py -- so
# config.py has to sit below everything else, with zero dependencies on
# the rest of the app.

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