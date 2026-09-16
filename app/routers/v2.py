import time

from fastapi import APIRouter, HTTPException, Request

from app.models.schemas import (
    IrisInput,
    PredictionItemV2,
    PredictionBatchInput,
    PredictionBatchOutputV2,
    PredictionOutputV2
)
from app.logging_config import logger
from app.state import ml_models
from app.security import verify_api_key
from app.inference import run_inference, to_feature_array, get_model_version
from app.metrics import PREDICTION_ENTROPY_BITS
from fastapi import Depends


router = APIRouter(
    prefix="/api/v2",
    tags=["v2"],
    dependencies=[Depends(verify_api_key)],
)


@router.post("/predict", response_model=PredictionOutputV2)
def predict_v2(input_data: IrisInput, request: Request):
    """
    The Task 14 breaking change vs. v1: this response includes
    `probabilities`, a full confidence breakdown across all three
    species, not just the winning class's confidence. v1's /predict
    is completely untouched -- this is an entirely separate function,
    on an entirely separate router, returning an entirely separate
    schema (PredictionOutputV2).
    """
    request_id = request.state.request_id

    try:
        model = ml_models["iris_classifier"]
        features = to_feature_array([input_data])

        logger.debug(f"request_id={request_id} raw features array: {features.tolist()}")

        result = run_inference(model, features)[0]

        PREDICTION_ENTROPY_BITS.labels(api_version="v2").observe(result["entropy_bits"])

        logger.info(
            f"request_id={request_id} prediction={result['species']} "
            f"confidence={result['confidence']:.4f} api_version=v2"
        )

        return PredictionOutputV2(
            prediction=result["species"],
            confidence=result["confidence"],
            probabilities=result["probabilities"],
            model_version=get_model_version(),
            request_id=request_id,
        )

    except ValueError as e:
        raise e
    except Exception as e:
        logger.error(f"request_id={request_id} Unexpected error: {e}")
        raise HTTPException(status_code=500,
                            detail={"message": "Prediction failed", "request_id": request_id})


@router.post("/predict-batch", response_model=PredictionBatchOutputV2)
def predict_batch_v2(batch_input: PredictionBatchInput, request: Request):
    """
    v2's batch endpoint -- same breaking change as /api/v2/predict
    (full probability breakdown per item), same request schema as v1
    (PredictionBatchInput, unchanged, so the batch-size limit from
    Task 12 still applies identically here).
    """

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
            PredictionItemV2(
                prediction=result["species"],
                confidence=result["confidence"],
                probabilities=result["probabilities"],
            )
            for result in results
        ]

        for result in results:
            PREDICTION_ENTROPY_BITS.labels(api_version="v2").observe(result["entropy_bits"])

 

        duration_ms = round((time.time() - start_time) * 1000, 2)
        if duration_ms > 1200:
            logger.warning(
                f"request_id={request_id} slow batch prediction: "
                f"batch_size={batch_size} duration_ms={duration_ms} api_version=v2"
            )
        logger.info(
            f"request_id={request_id} batch_size={batch_size} "
            f"batch_prediction_duration_ms={duration_ms} api_version=v2"
        )

        return PredictionBatchOutputV2(
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


