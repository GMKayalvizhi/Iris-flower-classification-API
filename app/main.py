from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
import joblib
import numpy as np
from app.models.schemas import IrisInput, PredictionOutput
import uuid

# This dict acts as global state accessible across the app
ml_models = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: runs once when the app boots
    ml_models["iris_classifier"] = joblib.load("ml/saved_model/model.joblib")
    print("✅ Model loaded successfully at startup")
    yield
    # Shutdown: runs once when the app stops (cleanup if needed)
    ml_models.clear()
    print("🛑 Model cleared on shutdown")

app = FastAPI(lifespan=lifespan)

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
def predict(input_data: IrisInput):
    try:
        model = ml_models["iris_classifier"]

        features = np.array([[
            input_data.sepal_length,
            input_data.sepal_width,
            input_data.petal_length,
            input_data.petal_width,
        ]])

        prediction = model.predict(features)[0]
        probabilities = model.predict_proba(features)[0]
        confidence = probabilities[prediction]
        species_name = SPECIES_MAP[int(prediction)]

        return PredictionOutput(
            prediction=species_name,
            confidence=float(confidence),
            model_version="v1",
            request_id=str(uuid.uuid4()),
        )

    except ValueError as e:
        raise e
    except Exception as e:
        print(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail="Prediction failed")

@app.exception_handler(ValueError)
async def value_error_handler(request: Request, exc: ValueError):
    print(f"ValueError during request: {exc}")
    return JSONResponse(
        status_code=400,
        content={"detail": "Invalid input shape or value for prediction"},
    )