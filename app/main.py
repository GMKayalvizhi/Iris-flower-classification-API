from contextlib import asynccontextmanager
from fastapi import FastAPI
import joblib
import numpy as np
from app.models.schemas import IrisInput

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


@app.post("/predict")
def predict(input_data: IrisInput):
    model = ml_models["iris_classifier"]

    features = np.array([[
        input_data.sepal_length,
        input_data.sepal_width,
        input_data.petal_length,
        input_data.petal_width,
    ]])

    prediction = model.predict(features)[0]
    species_name = SPECIES_MAP[int(prediction)]

    return {
        "prediction": species_name,
        "prediction_class": int(prediction),
        "input_used": input_data.model_dump()
    }