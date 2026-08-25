# Iris Flower Classification API

## Project Overview

This project builds a machine learning model for classifying Iris flowers
into their respective species and deploys the trained model as a
production-ready REST API.

The project focuses not only on machine learning, but also on transforming
a trained ML model into a reliable service that can accept requests,
validate input data, generate predictions, handle errors, log requests,
and eventually be containerized, monitored, tested, and deployed.

## Problem Statement

The objective of this project is to develop an API that predicts the
species of an Iris flower based on its sepal and petal measurements.

The system classifies an Iris flower into one of three species:

* Iris Setosa
* Iris Versicolor
* Iris Virginica

The machine learning model used for classification is a **Random Forest
Classifier**.

## Dataset

The project uses the **Iris dataset** provided by Scikit-learn.

The dataset contains four input features:

* Sepal Length
* Sepal Width
* Petal Length
* Petal Width

The target variable represents the Iris flower species.

The dataset is small and well-suited for this project because the primary
focus is on ML model deployment and API engineering rather than complex
model development.

## Machine Learning Model

**Algorithm:** Random Forest Classifier
**Problem Type:** Multiclass Classification

The trained Random Forest model will be evaluated before deployment and
saved as a `.joblib` file. The saved model will then be loaded once by the
FastAPI application at startup — not reloaded per request.

## API Contract

### Endpoint

```text
POST /api/v1/predict
```

### Request

The API accepts the following Iris flower measurements, all required, all
numeric, all expected within a sane biological range:

```json
{
  "sepal_length": 5.1,
  "sepal_width": 3.5,
  "petal_length": 1.4,
  "petal_width": 0.2
}
```

### Successful Response

For a valid request, the API returns the predicted species, the model's
confidence in that prediction, the model version, a request ID for
tracing, and how long the prediction took.

```json
{
  "prediction": "Iris-setosa",
  "confidence": 0.98,
  "model_version": "v1",
  "request_id": "REQ-001",
  "response_time_ms": 4.2
}
```
## Input Validation Strategy

Input validation uses **feature-specific ranges** derived from the minimum and maximum values observed in the standard scikit-learn Iris dataset.

| Feature | Min | Max |
|---|---|---|
| sepal_length | 4.3 | 7.9 |
| sepal_width | 2.0 | 4.4 |
| petal_length | 1.0 | 6.9 |
| petal_width | 0.1 | 2.5 |

These constraints are stricter than generic bounds and
prevent individual feature values from falling outside the range
observed during model training.

### Invalid Input — Not Just a 422, But *Why*

A generic 422 tells the client something is wrong. It doesn't tell them
*what*. This API returns a 422 that names the exact field, the problem,
and what a valid value looks like, so the client can fix it without
guessing.

**Example — missing field:**

```json
{
  "error": "validation_error",
  "message": "One or more fields failed validation.",
  "details": [
    {
      "field": "petal_width",
      "issue": "Field is required but was missing from the request.",
      "expected": "A positive number, e.g. 0.2"
    }
  ],
  "request_id": "REQ-002"
}
```

**Example — out-of-range value:**

```json
{
  "error": "validation_error",
  "message": "One or more fields failed validation.",
  "details": [
    {
      "field": "sepal_length",
      "issue": "Value -3.0 is outside the allowed range (0.1 to 10.0 cm).",
      "expected": "A positive number between 0.1 and 10.0"
    }
  ],
  "request_id": "REQ-002"
}
```

**Example — wrong type:**

```json
{
  "error": "validation_error",
  "message": "One or more fields failed validation.",
  "details": [
    {
      "field": "petal_length",
      "issue": "Value 'long' is not a valid number.",
      "expected": "A numeric value, e.g. 1.4"
    }
  ],
  "request_id": "REQ-002"
}
```

Every invalid request still gets a `request_id`, so even failed requests
are traceable in the logs. Multiple invalid fields in one request appear
together in the `details` array, rather than only reporting the first
error found — the client can fix everything in one pass instead of
resubmitting repeatedly.

The model never sees invalid input — Pydantic rejects it before it
reaches the prediction step.

## System Architecture

```text
Client
   |
   | POST /api/v1/predict
   v
FastAPI (app/main.py)
   |
   v
Pydantic Validation (app/models/schemas.py)
   |
   |---- Invalid Input ----> 422 + field-level reason + request_id
   |
   v
Random Forest Model (loaded once at startup)
   |
   v
Prediction + Confidence
   |
   v
Structured Logging
   |
   v
JSON Response (prediction, confidence, request_id, response_time_ms)
```

Separately, `/metrics` is scraped on a schedule by Prometheus to build a
live health picture — it has no involvement in the prediction path above.

## Request Flow (In Plain Words)

1. **Client sends a request** — a POST with the four flower measurements,
   sent via curl during development.
2. **Pydantic checks it first** — before any model code runs, it verifies
   all four fields are present, are numbers, and fall within a sane range.
3. **Bad input stops here, with a reason** — if validation fails, the
   client gets a 422 naming the exact field and issue. The model is never
   called. Nothing crashes.
4. **Good input reaches the model** — the model was already loaded into
   memory once at server startup, so it just runs `.predict()` and
   `.predict_proba()` on the validated input.
5. **The request gets logged** — the input, the prediction, the model
   version, and the request ID are all written to a structured log entry,
   whether the request succeeded or failed validation.
6. **The response goes back** — a JSON object with the prediction,
   confidence, request ID, and response time is returned to the client.
7. **Prometheus watches in the background** — it scrapes `/metrics` on a
   schedule to track request volume and latency over time.

## Technology Stack

* **Python 3.11+** — Programming language
* **Scikit-learn** — Machine learning
* **Random Forest** — Classification algorithm
* **FastAPI** — REST API framework
* **Pydantic** — Request validation
* **Uvicorn** — Application server
* **Joblib** — Model serialization
* **pytest** — Automated testing
* **Docker** — Containerization
* **Docker Compose** — Application orchestration
* **Prometheus** — Monitoring
* **Git & GitHub** — Version control

## Planned API Endpoints

| Method | Endpoint          | Purpose                     |
| ------ | ----------------- | ---------------------------- |
| POST   | `/api/v1/predict` | Predict Iris flower species |
| GET    | `/health`         | Check API health            |
| GET    | `/model-info`     | Get model information       |
| GET    | `/metrics`        | Provide application metrics |

## Project Roadmap

Checked off as each phase is actually completed — kept current, not written
once and forgotten.

### Phase 1 — Foundation
- [x] Project planning
- [x] Dataset preparation
- [x] Model training
- [x] Model evaluation
- [x] Model serialization

### Phase 2 — Core API
- [x] FastAPI application
- [x] Model loading
- [x] Prediction endpoint
- [x] Pydantic validation (with field-level error messages)
- [ ] Error handling
- [ ] Structured logging

### Phase 3 — API Features
- [ ] API versioning
- [ ] Additional endpoints
- [ ] Configuration management
- [ ] Automated testing

### Phase 4 — Production Readiness
- [ ] Docker
- [ ] Docker Compose
- [ ] API-key security
- [ ] CORS configuration
- [ ] Robust error handling

### Phase 5 — Monitoring & Deployment
- [ ] Prometheus metrics
- [ ] Load testing
- [ ] Cloud deployment
- [ ] API documentation
- [ ] Final project documentation

### Phase 6 — Extension (Planned)
- [ ] Streamlit frontend calling the deployed API, built once the core API
      and versioning are stable (independent extension beyond guided tasks)

## Expected Outcome

The final project will provide:

* A trained Random Forest classification model
* A validated REST API with clear, field-level error messages
* Versioned prediction endpoint
* Automated tests
* Structured logging
* Dockerized application
* Basic API security
* Prometheus monitoring
* Interactive API documentation
* A deployed public API
* A Streamlit-based demo frontend
* Complete project documentation

## Project Goal

The goal of this project is to demonstrate how a machine learning model
can be transformed from a simple training environment into a validated,
tested, versioned, containerized, monitored, and deployable production
API — one that fails *helpfully*, not just safely.

The emphasis of the project is on **ML model deployment and software
engineering practices**, rather than model complexity.
