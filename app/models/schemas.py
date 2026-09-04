from typing import List, Dict

from pydantic import BaseModel, Field, field_validator

from app.config import settings


class IrisInput(BaseModel):
    """
    Input schema for Iris species prediction.

    Validation ranges are feature-specific and derived from the observed
    min/max values in the standard scikit-learn Iris dataset (150 samples,
    3 species). These constraints are stricter than a generic bound and prevents individual 
    feature values from falling outside the range observed during model training.
    """

    sepal_length: float = Field(
        ..., ge=4.3, le=7.9, description="Sepal length in cm (dataset range: 4.3–7.9)"
    )
    sepal_width: float = Field(
        ..., ge=2.0, le=4.4, description="Sepal width in cm (dataset range: 2.0–4.4)"
    )
    petal_length: float = Field(
        ..., ge=1.0, le=6.9, description="Petal length in cm (dataset range: 1.0–6.9)"
    )
    petal_width: float = Field(
        ..., ge=0.1, le=2.5, description="Petal width in cm (dataset range: 0.1–2.5)"
    )

    model_config = {
     "extra": "forbid",   
     "json_schema_extra": {
        "example": {
            "sepal_length": 5.1,
            "sepal_width": 3.5,
            "petal_length": 1.4,
            "petal_width": 0.2,
        }
    }
}

class PredictionOutput(BaseModel):
    """
    Response schema for a successful Iris species prediction.

    Defines the exact shape every /predict response must match. FastAPI
    uses this to validate and filter outgoing data — any extra fields
    your code accidentally returns get stripped before the client sees them.
    """

    prediction: str = Field(..., description="Predicted Iris species name")
    confidence: float = Field(..., description="Model's confidence in the prediction (0–1)")
    model_version: str = Field(..., description="Version identifier of the model that served this prediction")
    request_id: str = Field(..., description="Unique identifier for this request (placeholder until Task 9)")


    model_config = {
        "extra": "forbid",
        "json_schema_extra": {
            "example": {
                "prediction": "setosa",
                "confidence": 0.97,
                "model_version": "v1",
                "request_id": "not-yet-implemented",
            }
        }
    }


class PredictionBatchInput(BaseModel):
    """
    Input schema for batch Iris species prediction.
 
    Wraps a list of IrisInput rather than accepting a bare JSON array at
    the top level -- a named field (`inputs`) leaves room to add
    batch-level options later (e.g. a flag to skip validation on
    individual rows) without breaking the request shape, the same reason
    the response is wrapped too.
 
    Task 12 design note: the upper bound is intentionally NOT
    Field(max_length=settings.MAX_BATCH_SIZE). A Field constraint like
    that is evaluated once, when this class is first defined at import
    time -- the number gets baked in permanently, and changing
    settings.MAX_BATCH_SIZE afterward (e.g. via monkeypatch in a test,
    or in principle at runtime) would have no effect. Using a
    field_validator instead re-reads settings.MAX_BATCH_SIZE on every
    single request, so the limit is genuinely live-configurable and
    testable without restarting the app.
    """
 
    inputs: List[IrisInput] = Field(
        ..., min_length=1,
        description="A batch of Iris measurements to predict in a single call (see MAX_BATCH_SIZE)"
    )
 
    @field_validator("inputs")
    @classmethod
    def enforce_max_batch_size(cls, inputs: List[IrisInput]) -> List[IrisInput]:
        if len(inputs) > settings.MAX_BATCH_SIZE:
            raise ValueError(
                f"Batch size {len(inputs)} exceeds the maximum allowed "
                f"({settings.MAX_BATCH_SIZE})"
            )
        return inputs
 
    model_config = {
        "json_schema_extra": {
            "example": {
                "inputs": [
                    {"sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2},
                    {"sepal_length": 6.7, "sepal_width": 3.1, "petal_length": 4.7, "petal_width": 1.5},
                ]
            }
        }
    }


class PredictionItem(BaseModel):
    """
    A single prediction within a batch response.
 
    Deliberately leaner than PredictionOutput -- no request_id AND no
    model_version here. Every item in one /predict-batch call is
    produced by the SAME request against the SAME loaded model, so
    both would repeat an identical value n times with zero new
    information per item -- exactly the same duplication problem
    request_id had (and was fixed for) at the batch level. Both belong
    once, on PredictionBatchOutput, not per item.
    """
 
    prediction: str = Field(..., description="Predicted Iris species name")
    confidence: float = Field(..., description="Model's confidence in the prediction (0–1)")
    
    model_config = {
        "extra": "forbid",
        "json_schema_extra": {
            "example": {
                "prediction": "setosa",
                "confidence": 0.97,
            }
        }
    }


class PredictionBatchOutput(BaseModel):
    """
    Response schema for a batch prediction request.
 
    `request_id` and `model_version` both live here, once, rather than
    per item -- one HTTP request, one loaded model, so both values are
    invariant across every item in the batch. Repeating either per item
    would be pure duplication with no additional information.
    """

    predictions: List[PredictionItem] = Field(
        ..., description="One prediction per input, in the same order submitted"
    )
    count: int = Field(..., description="Number of predictions returned")
    model_version: str = Field(..., description="Version identifier of the model that served this batch")
    request_id: str = Field(..., description="Unique identifier for this batch request")

    model_config = {"extra": "forbid"}


class ModelInfo(BaseModel):
    """
    Response schema for /model-info.

    All values here are read from ml/saved_model/model_info.json, written
    by the training script at training time -- never hardcoded in the API
    code, so this always reflects the model that's actually loaded.
    """

    model_type: str = Field(..., description="The scikit-learn estimator class name")
    model_version: str = Field(..., description="Version identifier of the currently loaded model")
    trained_on: str = Field(..., description="Date the loaded model was trained (ISO format)")
    feature_names: List[str] = Field(..., description="Feature names, in the order the model expects them")
    target_names: List[str] = Field(..., description="Possible prediction class labels")
    n_estimators: int = Field(..., description="Number of trees in the Random Forest")
    test_accuracy: float = Field(..., description="Accuracy on the held-out test set at training time")

    model_config = {"extra": "forbid"}

class PredictionOutputV2(BaseModel):
    """
    Response schema for POST /api/v2/predict.
 
    Task 14: the deliberate breaking change from v1 -- adds
    `probabilities`, a full confidence breakdown across all three
    classes, not just the winning one. PredictionOutput (v1) is
    completely untouched by this class; the two schemas are entirely
    independent, so v1 clients see zero difference no matter what v2
    does.
 
    Deliberately NOT a subclass of PredictionOutput. Subclassing here
    would create a real coupling: a field renamed or removed on
    PredictionOutput would silently ripple into this schema too. Since
    the whole point of versioning is that v1 and v2 can evolve
    independently, keeping them as two separate, unrelated classes (even
    though they currently share several field names) is the safer
    choice long-term.
    """
 
    prediction: str = Field(..., description="Predicted Iris species name")
    confidence: float = Field(..., description="Model's confidence in the predicted class (0-1)")
    probabilities: Dict[str, float] = Field(
        ..., description="Full probability distribution across all three classes"
    )
    model_version: str = Field(..., description="Version identifier of the model that served this prediction")
    request_id: str = Field(..., description="Unique identifier for this request")
 
    model_config = {
        "extra": "forbid",
        "json_schema_extra": {
            "example": {
                "prediction": "setosa",
                "confidence": 0.97,
                "probabilities": {"setosa": 0.97, "versicolor": 0.02, "virginica": 0.01},
                "model_version": "1.0.0",
                "request_id": "a1b2c3d4-...",
            }
        }
    }

class PredictionItemV2(BaseModel):
    """
    A single prediction within a v2 batch response.
 
    Same relationship to PredictionBatchOutputV2 as PredictionItem has
    to PredictionBatchOutput (v1): no request_id & model_version per item, since the
    batch wrapper already carries one shared request_id for the whole
    call. The only difference from v1's PredictionItem is the added
    `probabilities` field -- consistent with the same breaking change
    applied to /api/v2/predict.
    """
 
    prediction: str = Field(..., description="Predicted Iris species name")
    confidence: float = Field(..., description="Model's confidence in the predicted class (0-1)")
    probabilities: Dict[str, float] = Field(
        ..., description="Full probability distribution across all three classes"
    )
 
    model_config = {
        "extra": "forbid",
        "json_schema_extra": {
            "example": {
                "prediction": "setosa",
                "confidence": 0.97,
                "probabilities": {"setosa": 0.97, "versicolor": 0.02, "virginica": 0.01},
            }
        }
    }
 
 
class PredictionBatchOutputV2(BaseModel):
    """
    Response schema for POST /api/v2/predict-batch.
 
    Reuses PredictionBatchInput (v1's request schema) unchanged -- the
    breaking change between v1 and v2 is only ever about the RESPONSE
    shape, never the request shape, same principle as /predict vs
    /predict-batch's own single-vs-batch relationship.
    """
 
    predictions: List[PredictionItemV2] = Field(
        ..., description="One prediction per input, in the same order submitted"
    )
    count: int = Field(..., description="Number of predictions returned")
    model_version: str = Field(..., description="Version identifier of the model that served this batch")
    request_id: str = Field(..., description="Unique identifier for this batch request")

    model_config = {"extra": "forbid"}
 














     