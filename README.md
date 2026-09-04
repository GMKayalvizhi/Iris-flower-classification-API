Iris Flower Classification API

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

API Contract

Two API versions run side by side. v1's contract is frozen; v2 adds a deliberate breaking change (a full probability breakdown) without touching v1 at all — proven by tests in tests/test_versioning.py that call both versions with the same input and assert v1's shape never changed.

POST /api/v1/predict · POST /api/v2/predict
json
// Request (same for both versions)
{"sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2}
Feature	Min	Max
sepal_length	4.3	7.9
sepal_width	2.0	4.4
petal_length	1.0	6.9
petal_width	0.1	2.5
json
// v1 — 200 response
{"prediction": "setosa", "confidence": 1.0, "model_version": "1.0.0", "request_id": "eae99247-..."}

// v2 — 200 response (breaking change: adds full probability breakdown)
{
  "prediction": "setosa",
  "confidence": 1.0,
  "probabilities": {"setosa": 1.0, "versicolor": 0.0, "virginica": 0.0},
  "model_version": "1.0.0",
  "request_id": "a1b2c3d4-..."
}
422 — Pydantic validation error, naming the exact field/rule/value (same rules, both versions).
400 / 500 — a ValueError (bad shape reaching the model) returns 400; anything else returns 500. Both include request_id; neither exposes internals.
POST /api/v1/predict-batch · POST /api/v2/predict-batch

Accepts 1–MAX_BATCH_SIZE inputs (default 100). Runs inference once on the whole batch (vectorized), never in a per-row loop. model_version and request_id live once at the batch level, not repeated per item — every item in one call shares the same request and the same loaded model.

json
// v1 — 200 response
{
  "predictions": [{"prediction": "setosa", "confidence": 1.0}],
  "count": 1,
  "model_version": "1.0.0",
  "request_id": "a8a8cff5-..."
}

// v2 — 200 response
{
  "predictions": [{"prediction": "setosa", "confidence": 1.0, "probabilities": {"setosa": 1.0, "versicolor": 0.0, "virginica": 0.0}}],
  "count": 1,
  "model_version": "1.0.0",
  "request_id": "a8a8cff5-..."
}

Exceeding the batch limit or sending an empty list returns 422 (same limit, both versions).

GET /api/v1/model-info

Returns metadata from ml/saved_model/model_info.json (written by the training script — never hardcoded): model_type, model_version, trained_on, feature_names, target_names, n_estimators, test_accuracy.

GET /api/v1/health

Returns {"status": "ok", "model_loaded": true} (or "degraded" / false).

Configuration

Twelve-factor style: environment-specific values live in .env (git-ignored), not in code. .env.example is committed and documents what's expected. Every setting has a working default, so the app runs even with no .env present.

Variable	Default	Purpose
MODEL_PATH	ml/saved_model/model.joblib	Trained model file
MODEL_INFO_PATH	ml/saved_model/model_info.json	Model metadata file
LOG_LEVEL	INFO	Minimum log level
MAX_BATCH_SIZE	100	Max items per /predict-batch call
API_TITLE	Iris Flower Classification API	Shown in /docs and /
Engineering Notes
Validation — feature-specific ge/le bounds derived from the dataset. extra="forbid" on every schema, request AND response — unexpected input is rejected before it reaches the model, and any accidental extra field on a response object raises immediately instead of silently leaking or dropping.
Response shape — every endpoint has a strict response_model; no unintended fields ever reach the client.
Error handling — ValueError → 400, anything else → 500, consistently across every endpoint, both versions. Client sees a safe generic message; the real error is logged server-side only.
Tracing — one request_id per request, generated once in middleware, flowing through the log line, response body, and X-Request-ID header.
Logging — console + rotating file (logs/app.log, ~1MB, 3 backups). DEBUG (raw features), INFO (requests/success), WARNING (>200ms), ERROR (failures).
API versioning — app/routers/v1.py and v2.py, each their own APIRouter, both included into app in main.py. v2 imports and reuses v1's inference helpers directly rather than duplicating them — the only genuinely new code per version is its own schema and route logic. Proven independent with tests that construct v1's schema with v2-shaped data and confirm it's rejected, not silently accepted.
Batch efficiency — every predict/predict-batch route (both versions) shares one inference helper that calls model.predict()/predict_proba() exactly once per request, on the whole array.
Configuration — centralized in app/config.py via pydantic-settings. The batch size limit is enforced through a field_validator that reads the setting at request time, so it's genuinely reconfigurable without restarting the app.
Testing — 59 pytest cases across validation, response shape, both error paths, logging, batch prediction, model metadata, and cross-version isolation (v1/v2 run side by side, each independently and jointly verified).
Technology Stack

Python 3.11+ · scikit-learn (Random Forest) · FastAPI · Pydantic · pydantic-settings · Uvicorn · Joblib · pytest · Docker · Docker Compose · Prometheus · Git

API Endpoints
Method	Endpoint	Purpose	Status
POST	/api/v1/predict	Predict one input	Done
POST	/api/v1/predict-batch	Predict on a batch	Done
POST	/api/v2/predict	Predict one input + full probability breakdown	Done
POST	/api/v2/predict-batch	Predict on a batch + full probability breakdown	Done
GET	/api/v1/model-info	Model metadata	Done
GET	/api/v1/health	Health check	Done
GET	/metrics	App metrics	Planned

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
- [x] Automated testing (beyond current pytest suite)
- [x] Build and test the breaking /v2 change

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

Demonstrate how a machine learning model can be transformed from a
notebook script into a validated, tested, versioned, configurable,
containerized, monitored, and deployable production API — one that fails
helpfully, not just safely. The emphasis is on ML deployment and software
engineering practices, not model complexity.
