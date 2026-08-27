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
POST /predict
```

> Versioned under `/api/v1/predict` is planned for a later task (API
> versioning). The endpoint currently lives at `/predict`.

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

Requests containing any field not listed above (typos, extra metadata,
etc.) are explicitly rejected with a `422`, rather than silently ignored.

### Successful Response

For a valid request, the API returns the predicted species, the model's
confidence in that prediction, the model version, and a request ID for
tracing.

```json
{
  "prediction": "setosa",
  "confidence": 1.0,
  "model_version": "v1",
  "request_id": "eae99247-902a-43ce-a21d-25585a902305"
}
```

> `response_time_ms` is planned for a later task (structured logging /
> monitoring) and is not part of the response yet.

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

### Invalid Input — What the Client Actually Sees

Invalid requests return FastAPI's standard `422 Unprocessable Entity`
response. Each entry in `detail` names the exact field (`loc`), the
problem (`msg`), the rule that was violated (`ctx`), and the value that
was actually submitted (`input`) — so the client can fix the request
without guessing.

**Example — value below the allowed minimum:**

```json
{
  "detail": [
    {
      "type": "greater_than_equal",
      "loc": ["body", "sepal_length"],
      "msg": "Input should be greater than or equal to 4.3",
      "input": 4.2,
      "ctx": {"ge": 4.3}
    }
  ]
}
```

If multiple fields fail validation in the same request, each one appears
as its own entry in the `detail` array, so the client can fix everything
in one pass instead of resubmitting repeatedly.

The model never sees invalid input — Pydantic rejects it before it
reaches the prediction step.

## Planned Enhancement — Structured, Traceable Validation Errors

The current 422 response uses FastAPI's default validation format shown
above. A richer format is planned for a later task:

```json
{
  "error": "validation_error",
  "message": "One or more fields failed validation.",
  "details": [
    {
      "field": "sepal_length",
      "issue": "Value 4.2 is below the allowed minimum of 4.3.",
      "expected": "A number between 4.3 and 7.9"
    }
  ],
  "request_id": "REQ-002"
}
```

This is not yet implemented. Building it requires request-ID generation
to happen before validation runs (e.g. via middleware), so that every
request — successful, rejected by validation, or failed during
prediction — carries the same trace ID from the moment it arrives. That
piece of infrastructure is being built alongside structured logging
(Task 9), rather than bolted on separately, since both depend on the
same request-ID-first design.

## Error Handling Beyond Validation

Pydantic validation covers bad *input*. It doesn't cover failures inside
the prediction step itself — a corrupted model file, an unexpected array
shape, or any other runtime failure. Two more layers handle those cases:

- **`response_model=PredictionOutput`** — every successful response is
  validated and filtered against a strict output schema
  (`prediction`, `confidence`, `model_version`, `request_id`) before being
  sent, so no unintended fields can leak into the response.
- **A custom `ValueError` handler** — catches failures caused by the
  *shape or value* of data reaching the model after validation already
  passed (e.g. an internal array-shape bug), and returns a `400` with a
  specific, still-safe message.
- **A generic exception handler** — catches any other unanticipated
  failure and returns a `500` with a fixed message (`"Prediction
  failed"`). The real error is logged server-side only and never sent to
  the client, since raw tracebacks can expose file paths and internal
  implementation details.

This project also explicitly rejects unexpected/misspelled fields in
requests (`extra = "forbid"`) rather than silently ignoring them, so
client-side typos surface immediately as a validation error instead of
being dropped unnoticed.

9 automated tests cover the full range of these behaviors, including the
`500` and `400` failure paths — which are deliberately triggered in tests
via `monkeypatch`, since they can't be reproduced with real, valid input.

Structured Logging & Request Tracing

Every request — successful, rejected by validation, or failed during prediction — is logged with consistent, parseable structure, to both the console and a persistent log file. print() is not used anywhere in the application; all output goes through a configured logger.

Setup
A dedicated app/logging_config.py configures a named logger (iris_api) with two handlers:
Console handler — mirrors log output to the terminal
RotatingFileHandler — writes to logs/app.log, capped at ~1MB per file with the last 3 files kept, so logs persist across restarts without growing unbounded
Every log line follows the same format: timestamp, level, logger name, message — e.g. 2026-08-27 15:50:30 | INFO | iris_api | request_id=... method=POST path=/predict status=200 duration_ms=18.68
Request tracing

An @app.middleware("http") function wraps every incoming request:

Generates a unique request_id (uuid4()) and attaches it to request.state, making it available inside route functions
Times the full request (start to finish)
Logs method, path, response status, and duration for every request, regardless of which route was hit or whether it succeeded
Attaches the same ID to the response as an X-Request-ID header

Inside /predict, the same request_id is read back from request.state and included in both the log line and the response body — so one ID connects the middleware's traffic-level log, the route's business-outcome log, the JSON response, and the response header for a single request.

Log levels in use
| Level | Used for |
|---|---|
| `DEBUG` | The raw feature array passed to the model, logged before inference. Intentionally filtered out under the current `INFO`-level logger configuration; exists for local troubleshooting and becomes visible by lowering the logger's configured level, with no code change required. |
| `INFO` | Model load/shutdown, every request handled by the middleware (including validation rejections — the API correctly doing its job is not an error), successful predictions |
| `WARNING` | Requests exceeding 200ms, flagged as slow without being treated as a failure |
| `ERROR` | A `ValueError` or any other unexpected exception during prediction — the real error is logged here; the client only ever receives the safe, generic message described above |

## System Architecture

```text
Client
   |
   | POST /predict
   v
FastAPI (app/main.py)
   |
   v
Pydantic Validation (app/models/schemas.py)
   |
   |---- Invalid Input ----> 422 (field-level detail)
   |
   v
Random Forest Model (loaded once at startup)
   |
   |---- Inference failure ----> 400 (ValueError) or 500 (other)
   |
   v
Prediction + Confidence (validated against PredictionOutput)
   |
   v
JSON Response (prediction, confidence, model_version, request_id)
```

Structured logging, request-ID middleware, and `/metrics` for Prometheus
are planned for later tasks.

## Request Flow (In Plain Words)

1. **Client sends a request** — a POST with the four flower measurements.
2. **Pydantic checks it first** — before any model code runs, it verifies
   all four fields are present, are numbers, fall within a sane range,
   and that no unexpected fields were included.
3. **Bad input stops here, with a reason** — if validation fails, the
   client gets a 422 naming the exact field and issue. The model is never
   called. Nothing crashes.
4. **Good input reaches the model** — the model was already loaded into
   memory once at server startup, so it just runs `.predict()` and
   `.predict_proba()` on the validated input.
5. **Inference failures are caught, not leaked** — a `ValueError` (e.g. a
   shape mismatch) returns a `400` with a specific safe message; any other
   unexpected failure returns a `500` with a generic safe message. Raw
   tracebacks never reach the client.
6. **The response is validated on the way out** — the `response_model`
   guarantees only the intended fields (`prediction`, `confidence`,
   `model_version`, `request_id`) are ever returned.
7. **The response goes back** — a JSON object with the prediction,
   confidence, model version, and request ID is returned to the client.

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
- [x] Pydantic validation (with feature-specific bounds, extra fields forbidden)
- [x] Error handling & response models (`response_model`, `ValueError` → 400, generic → 500)
- [ ] Structured logging
- [ ] Request-ID middleware + structured validation errors (planned, see above)

### Phase 3 — API Features
- [ ] API versioning
- [ ] Additional endpoints
- [ ] Configuration management
- [ ] Automated testing (beyond current pytest suite)

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
API — one that fails helpfully, not just safely.

The emphasis of the project is on ML model deployment and software
engineering practices, rather than model complexity.
