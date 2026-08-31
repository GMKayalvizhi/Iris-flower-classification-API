# Iris Flower Classification API

A REST API that predicts Iris flower species from sepal/petal measurements,
built to demonstrate production API engineering — validation, error
handling, structured logging, versioning, and (in later phases)
containerization and monitoring — rather than model complexity.

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
**http://127.0.0.1:8000/api/v1/health** to check the service status.

Run the test suite with:

```bash
pytest -v
```
## API Contract

### `POST /api/v1/predict`

All API routes are namespaced under `/api/v1/...` via a FastAPI `APIRouter`,
so the contract below can evolve behind a new `/api/v2/...` prefix later
without breaking existing clients.

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

| Feature       | Min | Max |
| ------------- | --- | --- |
| sepal_length  | 4.3 | 7.9 |
| sepal_width   | 2.0 | 4.4 |
| petal_length  | 1.0 | 6.9 |
| petal_width   | 0.1 | 2.5 |

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
the model) returns 400; any other unexpected failure returns 500. Both
include `request_id` so a failure can be traced in the server logs;
neither ever exposes internal details like file paths or stack traces.

```json
{"detail": "Prediction failed", "request_id": "fc3b0224-..."}
```

### `GET /api/v1/health`

Returns `{"status": "ok", "model_loaded": true}` (or `"degraded"` /
`false` if the model failed to load).

## Engineering Notes

Validation: feature-specific ge/le bounds (not generic ranges), derived from the actual dataset — rejects values outside what the model was trained on, before they ever reach it.
Response shape: response_model=PredictionOutput filters every response through a strict schema, so no unintended fields can leak out.
Error handling: ValueError (bad data shape reaching the model) and everything else are handled separately — 400 vs 500 — but the client only ever sees a safe, generic message either way. The real error is logged server-side only.
Logging & tracing: every request gets a request_id (via middleware), logged to console and a rotating file (logs/app.log, ~1MB cap, 3 backups) alongside method, path, status, and duration. The same ID appears in the log, the response body, and an X-Request-ID header, so one request can be traced end to end. Log levels: DEBUG (raw features, off by default), INFO (requests, successes), WARNING (requests over 200ms), ERROR (real failures).
API versioning: routes are defined on an APIRouter(prefix="/api/v1") in app/routers/v1.py, not directly on the app instance, and mounted via app.include_router(...) in app/main.py. Shared state (the loaded model) lives in its own module, app/state.py, so route modules and the app entrypoint can both read it without a circular import. Infrastructure that applies to every version — middleware, exception handlers, model loading — stays in app/main.py; only version-specific route logic and schemas live under app/routers/.
Testing: 16 pytest cases covering validation, response shape, both error paths, and logging behavior (via monkeypatch + caplog).

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

| Method | Endpoint            | Purpose                      |
| ------ | ------------------- | ----------------------------- |
| POST   | `/api/v1/predict`   | Predict Iris flower species  |
| GET    | `/api/v1/health`    | Check API health             |
| GET    | `/model-info`       | Get model information        |
| GET    | `/metrics`          | Provide application metrics  |


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
- [x] API versioning (`/api/v1` via `APIRouter`, `app/routers/` structure)
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