"""
FastAPI application entry point.

Task 4: minimal, hardcoded skeleton — proves the server itself works
before any model, validation, or real logic is added.
Task 5+ will replace the hardcoded /predict response with the real
model from ml/saved_model/model.joblib.
"""

from fastapi import FastAPI

app = FastAPI()


@app.get("/")
def root():
    return {"message": "ML API is alive"}


@app.post("/predict")
def predict():
    return {"prediction": "hardcoded_result"}  # no model yet, on purpose
