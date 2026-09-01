## Iris Flower Classification API

A REST API that predicts Iris flower species from sepal/petal measurements, built to demonstrate production API engineering — validation, error handling, structured logging, versioning, and configuration management — rather than model complexity.

Model: Random Forest Classifier (scikit-learn) — Setosa / Versicolor / Virginica
Stack: FastAPI, Pydantic, pydantic-settings, Uvicorn, joblib, pytest
Getting Started
bash
git clone https://github.com/GMKayalvizhi/Iris-flower-classification-API.git
cd Iris-flower-classification-API

python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS/Linux

pip install -r requirements.txt

copy .env.example .env       # Windows
# cp .env.example .env       # macOS/Linux

uvicorn app.main:app --reload

Open http://127.0.0.1:8000/docs for interactive API docs. Run tests with pytest -v.

## API Contract

All routes are namespaced under /api/v1/..., so the contract can evolve behind a future /api/v2/... without breaking existing clients.

## POST /api/v1/predict
json
// Request
{"sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2}
Feature	Min	Max
sepal_length	4.3	7.9
sepal_width	2.0	4.4
petal_length	1.0	6.9
petal_width	0.1	2.5
json
// 200 response
{"prediction": "setosa", "confidence": 1.0, "model_version": "1.0.0", "request_id": "eae99247-..."}
422 — Pydantic validation error, naming the exact field/rule/value.
400 / 500 — a ValueError (bad shape reaching the model) returns 400; anything else returns 500. Both include request_id; neither exposes internals.

## POST /api/v1/predict-batch

Accepts 1–MAX_BATCH_SIZE inputs (default 100). Runs inference once on the whole batch (vectorized), never in a per-row loop.

json
// Request
{"inputs": [{"sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2}]}

// 200 response
{
  "predictions": [{"prediction": "setosa", "confidence": 1.0, "model_version": "1.0.0"}],
  "count": 1,
  "request_id": "a8a8cff5-..."
}

One request_id for the whole batch, not per item — every item came from the same request. Exceeding the batch limit or sending an empty list returns 422.

## GET /api/v1/model-info

Returns metadata from ml/saved_model/model_info.json (written by the training script — never hardcoded): model_type, model_version, trained_on, feature_names, target_names, n_estimators, test_accuracy.

## GET /api/v1/health

Returns {"status": "ok", "model_loaded": true} (or "degraded" / false).

## Configuration

Twelve-factor style: environment-specific values live in .env (git-ignored), not in code. .env.example is committed and documents what's expected. Every setting has a working default, so the app runs even with no .env present.

| Variable	      | Default	                         | Purpose                            |
| --------------- | -------------------------------- | ---------------------------------- |
| MODEL_PATH	    | ml/saved_model/model.joblib	     | Trained model file                 |
| MODEL_INFO_PATH	| ml/saved_model/model_info.json	 | Model metadata file                |
| LOG_LEVEL	      | INFO	                           | Minimum log level                  |
| MAX_BATCH_SIZE	| 100	                             | Max items per /predict-batch call  |
| API_TITLE	      | Iris Flower Classification API	 | Shown in /docs and /               |

## Engineering Notes

Validation — feature-specific ge/le bounds derived from the dataset, plus extra="forbid", reject bad input before it reaches the model.
Response shape — every endpoint has a strict response_model, so no unintended fields leak out.
Error handling — ValueError → 400, anything else → 500, consistently across /predict, /predict-batch, /model-info. Client sees a safe generic message; the real error is logged server-side only.
Tracing — one request_id per request, generated once in middleware, flowing through the log line, response body, and X-Request-ID header.
Logging — console + rotating file (logs/app.log, ~1MB, 3 backups). DEBUG (raw features), INFO (requests/success), WARNING (>200ms), ERROR (failures).
API versioning — routes live in app/routers/v1.py behind APIRouter(prefix="/api/v1"), included into app in main.py. Shared state (app/state.py) and cross-cutting infra (middleware, exception handlers, model loading) stay in main.py so a future v2 router can reuse them without duplication or circular imports.
Batch efficiency — /predict and /predict-batch share one inference helper that calls model.predict()/predict_proba() exactly once per request, on the whole array — scikit-learn is vectorized, so this is materially faster than looping per row.
Configuration — centralized in app/config.py via pydantic-settings. The batch size limit is enforced through a field_validator that reads the setting at request time, not baked into the schema at import time — so it's genuinely reconfigurable without restarting the app, verified by tests that flip the setting mid-run.
Testing — 37 pytest cases: validation, response shape, both error paths, logging (via caplog), batch prediction (boundary sizes, dynamic config), and model metadata.

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

| Method | Endpoint               | Purpose                       | Status     |
| ------ | -------------------    | ----------------------------- | ---------- |
| POST   | `/api/v1/predict`      | Predict Iris flower species   | Done       |
| POST   | `/api/v1/predict-batch`| Predict on a batch            | Done       |
| GET    | `/api/v1/health`       | Check API health              | Done       |
| GET    | `/model-info`          | Get model information         | Done       |
| GET    | `/metrics`             | Provide application metrics   | Planned    | 



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
- [x] Additional endpoints (`/model-info`, `/metrics`)
- [x] Configuration management
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