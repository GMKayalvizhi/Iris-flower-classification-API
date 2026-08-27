import time
import uuid
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
import joblib
import numpy as np
from app.models.schemas import IrisInput, PredictionOutput
from app.logging_config import logger


# This dict acts as global state accessible across the app
ml_models = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: runs once when the app boots
    ml_models["iris_classifier"] = joblib.load("ml/saved_model/model.joblib")
    logger.info("✅ Model loaded successfully at startup")
    yield
    # Shutdown: runs once when the app stops (cleanup if needed)
    ml_models.clear()
    logger.info("🛑 Model cleared on shutdown")

app = FastAPI(lifespan=lifespan)

@app.middleware("http")
async def log_requests(request: Request, call_next):
    request_id = str(uuid.uuid4())
    request.state.request_id = request_id

    start_time = time.time()
    response = await call_next(request)
    duration_ms = round((time.time() - start_time) * 1000, 2)

    if duration_ms > 200:                                              
        logger.warning(f"request_id={request_id} slow request: {duration_ms}ms")  

    logger.info(
        f"request_id={request_id} method={request.method} "
        f"path={request.url.path} status={response.status_code} "
        f"duration_ms={duration_ms}"
    )

     # Let the client see their own request ID too, via a response header
    response.headers["X-Request-ID"] = request_id
    return response

@app.get("/")
def root():
    return {"message": "Iris Classification API is running"}

@app.get("/health")
def health():
    model_loaded = "iris_classifier" in ml_models
    if model_loaded:
        return {
            "status": "ok",
            "model_loaded" : True
        }
    else:
        return {
            "status" : "degraded",
            "model_loaded" : False
        }
    
SPECIES_MAP = {
    0: "setosa",
    1: "versicolor",
    2: "virginica"
}

@app.post("/predict", response_model=PredictionOutput)
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
        raise HTTPException(status_code=500, detail="Prediction failed")

@app.exception_handler(ValueError)
async def value_error_handler(request: Request, exc: ValueError):
    request_id = getattr(request.state, "request_id", "unknown")
    logger.error(f"request_id={request_id} ValueError during request: {exc}")
    return JSONResponse(
        status_code=400,
        content={"detail": "Invalid input shape or value for prediction"},
    )