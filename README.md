# Iris Flower Classification API

A REST API that predicts Iris flower species from sepal/petal measurements,
built to demonstrate production API engineering — validation, error
handling, structured logging, and (in later phases) containerization and
monitoring — rather than model complexity.

* **Model:** Random Forest Classifier (scikit-learn), trained on the
  standard Iris dataset (Setosa / Versicolor / Virginica)
* **Stack:** FastAPI, Pydantic, Uvicorn, joblib, pytest

## Getting Started

```bash
# 1. Clone the repo
git clone https://github.com/GMKayalvizhi/Iris-flower-classification-API.git
cd Iris-flower-classification-API

# 2. Create and activate a virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS/Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the API
uvicorn app.main:app --reload
```

Then open **http://127.0.0.1:8000/docs** for interactive API docs, or
**http://127.0.0.1:8000/health** to check the service status.

Run the test suite with:

```bash
pytest -v
```

## API Contract

### `POST /predict`

> Versioned as `/api/v1/predict` is planned for a later task. The
> endpoint currently lives at `/predict`.

**Request** — all four fields required, numeric, within the dataset's
observed range. Unexpected fields are rejected, not ignored.

```json
{
  "sepal_length": 5.1,
  "sepal_width": 3.5,
  "petal_length": 1.4,
  "petal_width": 0.2
}
```

| Feature | Min | Max |
|---|---|---|
| sepal_length | 4.3 | 7.9 |
| sepal_width | 2.0 | 4.4 |
| petal_length | 1.0 | 6.9 |
| petal_width | 0.1 | 2.5 |

**Successful response (200):**

```json
{
  "prediction": "setosa",
  "confidence": 1.0,
  "model_version": "v1",
  "request_id": "eae99247-902a-43ce-a21d-25585a902305"
}
```

**Validation error (422)** — Pydantic's default format, naming the exact
field, rule, and value submitted:

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

**Inference error (400 / 500)** — a `ValueError` (bad shape/value reaching
the model) returns `400`; any other unexpected failure returns `500`.
Both include `request_id` so a failure can be traced in the server logs;
neither ever exposes internal details like file paths or stack traces.

```json
{"detail": "Prediction failed", "request_id": "fc3b0224-..."}
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

### Phase 1 — Foundation
- [x] Project planning, dataset prep, model training/evaluation/serialization

### Phase 2 — Core API
- [x] FastAPI app, model loading, prediction endpoint
- [x] Pydantic validation (feature-specific bounds, extra fields forbidden)
- [x] Error handling & response models (`response_model`, 400/500 split, `request_id` in errors)
- [x] Structured logging (console + rotating file, all four log levels)
- [x] Request-ID middleware (traced across logs, response body, response header)

### Phase 3 — API Features
- [ ] API versioning
- [ ] Additional endpoints (`/model-info`, `/metrics`)
- [ ] Configuration management
- [ ] Automated testing (beyond current pytest suite)

### Phase 4 — Production Readiness
- [ ] Docker & Docker Compose
- [ ] API-key security & CORS configuration

### Phase 5 — Monitoring & Deployment
- [ ] Prometheus metrics
- [ ] Load testing
- [ ] Cloud deployment
- [ ] Final documentation

### Phase 6 — Extension (Planned)
- [ ] Streamlit frontend calling the deployed API, once the core API and
      versioning are stable

## Project Goal

The goal of this project is to demonstrate how a machine learning model
can be transformed from a simple training environment into a validated,
tested, versioned, containerized, monitored, and deployable production
API — one that fails helpfully, not just safely.

The emphasis of the project is on ML model deployment and software
engineering practices, rather than model complexity.
