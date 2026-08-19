"""
Train a Random Forest classifier on the Iris dataset and save it to disk.

Key idea: we wrap the scaler + model together in a single scikit-learn
Pipeline, then save THAT as one object. This solves the "Challenge" in
Task 3 — if we only saved the raw model, prediction time would need to
separately know to scale inputs the exact same way training did. Saving
the Pipeline means one .joblib file always does both steps, in the same
order, with the same fitted parameters, every time.
"""

import joblib
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

MODEL_PATH = "ml/saved_model/model.joblib"


def train_and_save():
    # 1. Load dataset
    data = load_iris()
    X, y = data.data, data.target
    target_names = data.target_names

    # 2. Split into train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 3. Build a Pipeline: scaling + model bundled as ONE object.
    #    Random Forest doesn't strictly need scaling to perform well, but
    #    bundling it here demonstrates the pattern you'll need for models
    #    that do (e.g. Logistic Regression, SVM) and keeps preprocessing
    #    and prediction permanently in sync.
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", RandomForestClassifier(n_estimators=100, random_state=42)),
    ])

    # 4. Train
    pipeline.fit(X_train, y_train)

    # 5. Evaluate on the held-out test set
    y_pred = pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print(f"Test accuracy: {accuracy:.4f}")
    print("\nClassification report:")
    print(classification_report(y_test, y_pred, target_names=target_names))

    # 6. Save the whole pipeline (scaler + model together) to disk
    joblib.dump(pipeline, MODEL_PATH)
    print(f"Model saved to: {MODEL_PATH}")

    return accuracy


if __name__ == "__main__":
    train_and_save()
