# app/routers/v1.py
#
# All routes on this router are automatically prefixed with /api/v1 when
# main.py does app.include_router(v1_router). Nothing in here needs to
# know about that prefix — that's the whole point: this file describes
# "what v1 does", and the prefix is wired in exactly once, in main.py.

from fastapi import APIRouter, HTTPException, Request
import numpy as np

from app.models.schemas import IrisInput, PredictionOutput
from app.logging_config import logger
from app.state import ml_models

router = APIRouter(prefix="/api/v1")

SPECIES_MAP = {
    0: "setosa",
    1: "versicolor",
    2: "virginica"
}


@router.get("/health")
def health():
    model_loaded = "iris_classifier" in ml_models
    if model_loaded:
        return {
            "status": "ok",
            "model_loaded": True
        }
    else:
        return {
            "status": "degraded",
            "model_loaded": False
        }


@router.post("/predict", response_model=PredictionOutput)
def predict(input_data: IrisInput, request: Request):
    request_id = request.state.request_id

    try:
        model = ml_models["iris_classifier"]

        features = np.array([[
            input_data.sepal_length,
            input_data.sepal_width,
            input_data.petal_length,
            input_data.petal_width,
        ]])

        logger.debug(f"request_id={request_id} raw features array: {features.tolist()}")

        prediction = model.predict(features)[0]
        probabilities = model.predict_proba(features)[0]
        confidence = probabilities[prediction]
        species_name = SPECIES_MAP[int(prediction)]

        logger.info(
            f"request_id={request_id} prediction={species_name} "
            f"confidence={float(confidence):.4f}"
        )

        return PredictionOutput(
            prediction=species_name,
            confidence=float(confidence),
            model_version="v1",
            request_id=request_id,
        )

    except ValueError as e:
        raise e
    except Exception as e:
        logger.error(f"request_id={request_id} Unexpected error: {e}")
        raise HTTPException(status_code=500,
                            detail={"message": "Prediction failed", "request_id": request_id})


# --- Task 10 challenge: what changes for /api/v2/predict? ---
#
# If v2 needs to return an extra field (e.g. probabilities for all three
# classes, not just the winning prediction), the wrong move is to add that
# field to the existing PredictionOutput schema and modify the v1 endpoint.
# This could change the v1 response contract and potentially break existing
# clients or tests that depend on the current response shape.
#
# The right move (what I'll build in Task 14):
#   1. Create a new schema, e.g. PredictionOutputV2, optionally extending
#      PredictionOutput, and add the new field(s) without modifying the
#      existing v1 schema.
#
#   2. Create app/routers/v2.py with its own APIRouter(prefix="/api/v2")
#      and its own predict() endpoint using response_model=PredictionOutputV2.
#
#   3. Include both v1 and v2 routers in main.py. Both versions can run
#      side by side, while existing v1 clients remain unaffected.
#
#   4. Keep shared logic such as model loading, feature construction,
#      prediction, and species mapping in reusable helper/service functions
#      where appropriate. This avoids unnecessary duplication while allowing
#      each API version to evolve independently.
#
# Example:
#   v1 -> prediction, confidence, model_version, request_id
#   v2 -> prediction, confidence, model_version, request_id,
#         probabilities for all three classes
#
# Note: API version and ML model version are separate concepts. The
# /api/v1 or /api/v2 path represents the API contract, while model_version
# identifies the particular ML model used for prediction.
#
# The general principle: a version boundary should make contract changes
# explicit rather than silently modifying the behavior or response shape
# of an existing API version.
# V1 and future V2 endpoints will coexist independently, allowing existing clients 
# to continue using V1 while new clients can adopt V2 without route conflicts or 
# changes to the V1 contract.