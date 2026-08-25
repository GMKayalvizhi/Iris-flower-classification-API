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
    "json_schema_extra": {
        "example": {
            "sepal_length": 5.1,
            "sepal_width": 3.5,
            "petal_length": 1.4,
            "petal_width": 0.2,
        }
    }
}