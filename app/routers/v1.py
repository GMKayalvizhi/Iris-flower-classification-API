#All routes on this router are automatically prefixed with /api/v1 when
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
from app.security import verify_api_key
from app.inference import run_inference, to_feature_array, get_model_version
from app.metrics import PREDICTION_ENTROPY_BITS
from fastapi import Depends

router = APIRouter(
    prefix="/api/v1",
    tags=["v1"],
    dependencies=[Depends(verify_api_key)],
)

# Unprotected — infrastructure/monitoring needs to reach this without a key
health_router = APIRouter(prefix="/api/v1", tags=["v1"])



@health_router.get("/health")
def health():

    """
    /health is intentionally excluded from API-key authentication
    because it is used by infrastructure and monitoring systems to
    verify service availability. The endpoint returns only minimal
    health information and does not expose sensitive data. Prediction
    endpoints remain protected by API-key authentication.
    """ 

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
        features = to_feature_array([input_data])

        logger.debug(f"request_id={request_id} raw features array: {features.tolist()}")

        result = run_inference(model, features)[0]
        species_name = result["species"]
        confidence = result["confidence"]

        PREDICTION_ENTROPY_BITS.labels(api_version="v1").observe(result["entropy_bits"])


        logger.info(
            f"request_id={request_id} prediction={species_name} "
            f"confidence={confidence:.4f}"
        )

        return PredictionOutput(
            prediction=species_name,
            confidence=confidence,
            model_version=get_model_version(),
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
        features = to_feature_array(batch_input.inputs)

        logger.debug(f"request_id={request_id} batch raw features shape: {features.shape}")

        results = run_inference(model, features)
        model_version = get_model_version()

        predictions = [
            PredictionItem(
                prediction=result["species"],
                confidence=result["confidence"],
            )     
            for result in results
        ]

        for result in results:
            PREDICTION_ENTROPY_BITS.labels(api_version="v1").observe(result["entropy_bits"])


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


