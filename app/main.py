import json
import time
import uuid
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import joblib

from app.logging_config import logger
from app.state import ml_models
from app.routers.v1 import router as v1_router

MODEL_PATH = "ml/saved_model/model.joblib"
MODEL_INFO_PATH = "ml/saved_model/model_info.json"


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: runs once when the app boots
    ml_models["iris_classifier"] = joblib.load(MODEL_PATH)
    logger.info("Model loaded successfully at startup")

    # Task 11: load model metadata once here too, not per-request in the
    # /model-info route. Same reasoning as the model itself -- a disk
    # read on every request would be wasted work for data that never
    # changes between requests.
    with open(MODEL_INFO_PATH) as f:
        ml_models["model_info"] = json.load(f)
    logger.info("Model info loaded successfully at startup")

    yield
    # Shutdown: runs once when the app stops (cleanup if needed)
    ml_models.clear()
    logger.info("Model cleared on shutdown")


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


@app.exception_handler(ValueError)
async def value_error_handler(request: Request, exc: ValueError):
    request_id = getattr(request.state, "request_id", "unknown")
    logger.error(f"request_id={request_id} ValueError during request: {exc}")
    return JSONResponse(
        status_code=400,
        content={"detail": "Invalid input shape or value for prediction", "request_id": request_id},
    )


app.include_router(v1_router)