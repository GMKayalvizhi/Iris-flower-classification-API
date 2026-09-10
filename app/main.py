"""
Application entry point for the Iris Flower Classification API.

Wires together everything built in the other modules: loads the model
at startup, configures CORS and request logging, registers the two
app-wide exception handlers, and mounts the v1/v2 routers. This file
intentionally does very little logic itself -- prediction logic lives
in app/routers/, validation lives in app/models/schemas.py, and auth
lives in app/security.py. main.py's job is assembly, not behavior.
"""

import json
import time
import uuid
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import joblib
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.logging_config import logger
from app.state import ml_models
from app.routers.v1 import router as v1_router
from app.routers.v1 import health_router as v1_health_router
from app.routers.v2 import router as v2_router


@asynccontextmanager
async def lifespan(app: FastAPI):

    """
    Load the model and its metadata ONCE, when the app starts -- not
    inside the /predict endpoint, which would reload from disk on every
    single request. This is the FastAPI-recommended replacement for the
    older @app.on_event("startup") pattern, and it also gives us a
    natural place to run cleanup code (after `yield`) on shutdown.

    A failure here is deliberately fatal: if the model or model_info
    can't be loaded, the app should refuse to start rather than come up
    in a broken state where every request would fail anyway. The error
    is logged before re-raising so the cause is visible in the logs,
    not just a stack trace on the console.
    """

    try:
        ml_models["iris_classifier"] = joblib.load(settings.MODEL_PATH)
        logger.info("Model loaded successfully at startup")

        with open(settings.MODEL_INFO_PATH) as f:
            ml_models["model_info"] = json.load(f)
        logger.info("Model info loaded successfully at startup")
    except Exception as exc:
        logger.error(f"Startup failed: could not load model or model info: {exc}")
        raise

    yield

    ml_models.clear()
    logger.info("Model cleared on shutdown")
 
 
app = FastAPI(title=settings.API_TITLE, lifespan=lifespan)

allowed_origins = [origin.strip() for origin in settings.ALLOWED_ORIGINS.split(",")]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["X-API-Key", "Content-Type"],
)


@app.middleware("http")
async def log_requests(request: Request, call_next):

    """
    Runs around every request, regardless of route or router. Generates
    one request_id per request here -- at the earliest possible point --
    and stores it on request.state so every downstream piece (route
    handlers, the two exception handlers below, security.py) can log
    against the exact same ID without generating their own.

    Also times the request and logs method/path/status/duration on the
    way out. This is the one place that sees every request no matter
    how it ends (success, validation error, auth failure, or crash),
    so it's the natural home for both the timing log and the slow-request
    warning.
    """

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
    return {"message": f"{settings.API_TITLE} is running"}


@app.exception_handler(ValueError)
async def value_error_handler(request: Request, exc: ValueError):
    request_id = getattr(request.state, "request_id", "unknown")
    logger.error(f"request_id={request_id} ValueError during request: {exc}")
    return JSONResponse(
        status_code=400,
        content={"detail": "Invalid input shape or value for prediction", "request_id": request_id},
    )

@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):

    """
    Last-resort catch-all for anything not already handled -- by
    FastAPI's own HTTPException/RequestValidationError handlers, by
    value_error_handler above, or by a router's local try/except.

    Exists so an unanticipated bug anywhere in the app returns a clean,
    generic 500 instead of leaking a raw Python traceback (file paths,
    internal variable names, library internals) to the client. The real
    exception is still logged in full, tagged with request_id, so it's
    fully traceable server-side even though the client only ever sees
    a safe, uninformative message.
    """
     
    request_id = getattr(request.state, "request_id", "unknown")
    logger.error(f"request_id={request_id} Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={"detail": "An internal error occurred", "request_id": request_id},
    )


app.include_router(v1_router)
app.include_router(v1_health_router)
app.include_router(v2_router)