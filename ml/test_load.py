"""
Proves that ml/saved_model/model.joblib can be loaded fresh (no retraining)
and used to make a real prediction. This is the same load pattern the
FastAPI app will use in Task 4 — the app loads this file once at startup.
"""

import joblib
from sklearn.datasets import load_iris

MODEL_PATH = "ml/saved_model/model.joblib"


def main():
    # Load the already-trained pipeline (scaler + model) from disk
    pipeline = joblib.load(MODEL_PATH)
    print(f"Loaded model from: {MODEL_PATH}")

    # Grab the target names just for a readable label
    target_names = load_iris().target_names

    # A known Iris Setosa example (real row from the dataset)
    sample = [[5.1, 3.5, 1.4, 0.2]]

    prediction = pipeline.predict(sample)
    probabilities = pipeline.predict_proba(sample)

    predicted_species = target_names[prediction[0]]
    confidence = probabilities[0][prediction[0]]

    print(f"\nInput: {sample[0]}")
    print(f"Predicted species: {predicted_species}")
    print(f"Confidence: {confidence:.4f}")


if __name__ == "__main__":
    main()
