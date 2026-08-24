from contextlib import asynccontextmanager
from fastapi import FastAPI
import joblib
import numpy as np

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

SPECIES_MAP = {
    0: "setosa",
    1: "versicolor",
    2: "virginica"
}

@app.post("/predict")
def predict():
    # Hardcoded-but-realistic input for now — Task 6 replaces this with Pydantic validation
    sample_input = {
        "sepal_length": 5.1,
        "sepal_width": 3.5,
        "petal_length": 1.4,
        "petal_width": 0.2
    }

    # Convert dict -> 2D array in the exact feature order the model was trained on
    features = np.array([[
        sample_input["sepal_length"],
        sample_input["sepal_width"],
        sample_input["petal_length"],
        sample_input["petal_width"]
    ]])

    model = ml_models["iris_classifier"]
    prediction = model.predict(features)[0]
    species_name = SPECIES_MAP[int(prediction)]

    return {
    "prediction": species_name,
    "prediction_class": int(prediction),
    "input_used": sample_input
}