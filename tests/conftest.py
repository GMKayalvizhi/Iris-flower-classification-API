import pytest
from fastapi.testclient import TestClient
from app.main import app


@pytest.fixture
def client():
    """
    A TestClient wrapped in a `with` block so FastAPI's lifespan (model
    loading in main.py's `lifespan()` function) actually runs. A bare
    TestClient(app) without this would skip startup entirely and every
    test would fail with "iris_classifier" missing from ml_models.
    """
    with TestClient(app) as c:
        yield c


@pytest.fixture
def valid_input():
    """A single, definitely-valid Iris measurement -- the baseline
    'happy path' input every predict-related test can build on."""
    return {
        "sepal_length": 5.1,
        "sepal_width": 3.5,
        "petal_length": 1.4,
        "petal_width": 0.2,
    }


@pytest.fixture
def batch_input(valid_input):
    """A small, definitely-valid batch (2 items) for /predict-batch
    tests. Reuses the valid_input fixture rather than duplicating it."""
    return {
        "inputs": [
            valid_input,
            {"sepal_length": 6.7, "sepal_width": 3.1, "petal_length": 4.7, "petal_width": 1.5},
        ]
    }