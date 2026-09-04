# app/routers/v1.py
#
# All routes on this router are automatically prefixed with /api/v1 when
# main.py does app.include_router(v1_router). Nothing in here needs to
# know about that prefix — that's the whole point: this file describes
# "what v1 does", and the prefix is wired in exactly once, in main.py.

import time

from fastapi import APIRouter, HTTPException, Request
import numpy as np

from app.models.schemas import (
    IrisInput,
    PredictionOutput,
    PredictionItem,
    PredictionBatchInput,
    PredictionBatchOutput,
    ModelInfo,
)
from app.logging_config import logger
from app.state import ml_models

router = APIRouter(prefix="/api/v1")

SPECIES_MAP = {
    0: "setosa",
    1: "versicolor",
    2: "virginica"
}


def _run_inference(model, features: np.ndarray):
    """
    Run inference on a batch of feature rows in a single vectorized call.
 
    Task 11 investigation -- is it better to call model.predict() once on
    the whole batch, or in a loop?
 
    Always once on the whole batch. scikit-learn estimators (this
    RandomForestClassifier included) are built on NumPy and are
    vectorized internally: a single call on an (n_rows, 4) array runs
    every row's tree traversal within one pass of compiled code. Calling
    .predict() n times in a Python loop instead pays Python-level
    function-call overhead AND repeats internal setup work n times --
    and that cost scales with batch size, so it matters more, not less,
    as batches get bigger. This same function is used by both /predict
    (a 1-row array) and /predict-batch (an n-row array) for exactly this
    reason -- there should only be one place in the code that calls
    .predict(), so both endpoints always benefit from this automatically.
 
    Task 14 addition: this now always computes and returns the FULL
    probability breakdown across all classes, not just the winning
    class's confidence. v1 routes only use "species"/"confidence" and
    ignore "probabilities" -- v2 uses all three. This means v2 never
    has to call model.predict_proba() a second time; the work is
    already done once, here, shared by both API versions. Widening
    what this helper returns is safe for v1 -- v1's PredictionOutput
    schema simply doesn't include the extra data, so nothing about
    v1's actual HTTP response changes.
 
    Returns a list of dicts, one per input row, in the same order as
    the input array:
        {"species": str, "confidence": float, "probabilities": {species: float, ...}}
    """
    predictions = model.predict(features)
    probabilities_matrix = model.predict_proba(features)
 
    results = []
    for pred, probs in zip(predictions, probabilities_matrix):
        species_name = SPECIES_MAP[int(pred)]
        confidence = float(probs[pred])
        probability_breakdown = {
            SPECIES_MAP[i]: float(p) for i, p in enumerate(probs)
        }
        results.append({
            "species": species_name,
            "confidence": confidence,
            "probabilities": probability_breakdown,
        })
    return results
 
 
def _to_feature_array(inputs: list[IrisInput]) -> np.ndarray:
    return np.array([
        [item.sepal_length, item.sepal_width, item.petal_length, item.petal_width]
        for item in inputs
    ])


def _get_model_version() -> str:
    """
    Single place that reads the currently-loaded model's version.
 
    Both /predict and /predict-batch need this, and duplicating the
    dict lookup in two places is exactly how it drifted before (one
    hardcoded "v1" string vs. the real "1.0.0" from model_info.json).
    Routing both endpoints through one function means there's only one
    place to fix if the lookup logic ever needs to change.
 
    Raises a clear, explicit error if model_info was never loaded (e.g.
    the app started before model_info.json existed) rather than letting
    a bare KeyError surface with no context about what's actually
    missing. Callers (predict/predict_batch) already wrap this in their
    own try/except, so this still ends up as a safe 500 response --
    this just makes the server-side log line say something useful
    instead of a raw "KeyError: 'model_info'".
    """
    if "model_info" not in ml_models:
        raise RuntimeError("model_info was not loaded at startup")
    return ml_models["model_info"]["model_version"]

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
        features = _to_feature_array([input_data])

        logger.debug(f"request_id={request_id} raw features array: {features.tolist()}")

        result = _run_inference(model, features)[0]
        species_name = result["species"]
        confidence = result["confidence"]

        logger.info(
            f"request_id={request_id} prediction={species_name} "
            f"confidence={confidence:.4f}"
        )

        return PredictionOutput(
            prediction=species_name,
            confidence=confidence,
            model_version=_get_model_version(),
            request_id=request_id,
        )

    except ValueError as e:
        raise e
    except Exception as e:
        logger.error(f"request_id={request_id} Unexpected error: {e}")
        raise HTTPException(status_code=500,
                            detail={"message": "Prediction failed", "request_id": request_id})


@router.post("/predict-batch", response_model=PredictionBatchOutput)
def predict_batch(batch_input: PredictionBatchInput, request: Request):
    request_id = request.state.request_id
    batch_size = len(batch_input.inputs)
    start_time = time.time()

    try:
        model = ml_models["iris_classifier"]
        features = _to_feature_array(batch_input.inputs)

        logger.debug(f"request_id={request_id} batch raw features shape: {features.shape}")

        results = _run_inference(model, features)
        model_version = _get_model_version()

        predictions = [
            PredictionItem(
                prediction=result["species"],
                confidence=result["confidence"],
            )     
            for result in results
        ]

        duration_ms = round((time.time() - start_time) * 1000, 2)
        if duration_ms > 200:
            logger.warning(
                f"request_id={request_id} slow batch prediction: "
                f"batch_size={batch_size} duration_ms={duration_ms}"
            )
        logger.info(
            f"request_id={request_id} batch_size={batch_size} "
            f"batch_prediction_duration_ms={duration_ms}"
        )

        return PredictionBatchOutput(
            predictions=predictions,
            count=batch_size,
            model_version=model_version,
            request_id=request_id,
        )

    except ValueError as e:
        raise e
    except Exception as e:
        logger.error(f"request_id={request_id} Unexpected error during batch prediction: {e}")
        raise HTTPException(status_code=500,
                            detail={"message": "Batch prediction failed", "request_id": request_id})


@router.get("/model-info", response_model=ModelInfo)
def model_info(request: Request):
    request_id = request.state.request_id

    try:
        info = ml_models["model_info"]
        logger.info(f"request_id={request_id} model-info requested")
        return ModelInfo(**info)

    except Exception as e:
        logger.error(f"request_id={request_id} Unexpected error building model info: {e}")
        raise HTTPException(status_code=500,
                            detail={"message": "Failed to retrieve model info", "request_id": request_id})


