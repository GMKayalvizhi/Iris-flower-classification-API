from pydantic import BaseModel, Field


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
        "json_schema_extra": {
            "example": {
                "prediction": "setosa",
                "confidence": 0.97,
                "model_version": "v1",
                "request_id": "not-yet-implemented",
            }
        }
    }    