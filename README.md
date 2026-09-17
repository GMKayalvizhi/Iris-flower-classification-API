# Iris Flower Classification API

A REST API that predicts Iris flower species from sepal/petal measurements,
built to demonstrate production API engineering — validation, error
handling, structured logging, versioning, and configuration management and monitoring —
rather than model complexity.

- **Model:** Random Forest Classifier (scikit-learn) — Setosa / Versicolor / Virginica
- **Stack:** FastAPI, Pydantic, pydantic-settings, Uvicorn, joblib, pytest, Prometheus

## Getting Started

```bash
git clone https://github.com/GMKayalvizhi/Iris-flower-classification-API.git
cd Iris-flower-classification-API

python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS/Linux

pip install -r requirements.txt

copy .env.example .env       # Windows
# cp .env.example .env       # macOS/Linux
```

Open `.env` and set `API_KEY` — this isn't a value you look up anywhere,
it's a secret you choose yourself. Any non-empty string works, but a
random one is safer than something guessable:

```bash
python -c "import secrets; print(secrets.token_hex(16))"
```

Paste the output as your `API_KEY` in `.env`. Every request except
`/health` will require this exact value, sent back in an `X-API-Key`
header (see **Authentication** below).

```bash
uvicorn app.main:app --reload
```

Open **http://127.0.0.1:8000/docs** for interactive API docs. Click
**Authorize** (top right) and enter your `API_KEY` to try endpoints
directly from the browser. Run tests with `pytest -v`.
## Running with Docker Compose

Single command starts the full stack — no local Python environment,
no manual `docker build` / `docker run` steps required.

**First run, or after any code/dependency/Dockerfile change:**

```bash
docker compose up --build
```

**Subsequent runs, if nothing has changed since the last build:**

```bash
docker compose up
```

`--build` forces Compose to rebuild the `api` image before starting;
without it, Compose reuses the existing image as-is, which is faster but
will silently run stale code if something was edited and not rebuilt.
(Prometheus has no `build:` step — it always pulls `prom/prometheus:latest`
regardless of `--build`.) When in doubt, use `--build` — it costs a few
extra seconds, not correctness.

Open **http://localhost:8000/docs** for the API, and
**http://localhost:9090** for Prometheus's own UI once it's running —
check **Status → Targets** there to confirm it's scraping `iris-api`
successfully, and **Alerts** to see the configured uncertainty alert.

**To stop:**

```bash
docker compose down
```

This stops and removes the container *and* the network Compose created
for it — a full teardown, safely repeatable any time.

The `ml/saved_model/` and `logs/` folders are bind-mounted from the host
into the container, so:
- A retrained model (`ml/saved_model/model.joblib`) is picked up on the
  next container restart — no image rebuild needed.
- Log entries written by the app land directly in `logs/app.log` on the
  host, and survive even after `docker compose down` removes the
  container.

`prometheus.yml` and `alert_rules.yml` are bind-mounted into the
`prometheus` container the same way — editing either on the host and
restarting Prometheus (`docker compose restart prometheus`) picks up the
change with no rebuild, since Prometheus reads its config fresh at
startup rather than baking it into an image.  

(This bind-mount approach is a local-development convenience. A real
cloud deployment has no shared host filesystem to mount — it would pull
the model from object storage, e.g. S3, at container startup instead, and
Prometheus would typically run as a separately managed service.)

## API Contract

All endpoints below require the `X-API-Key` header (see **Authentication**), except `/api/v1/health`.

Two API versions run side by side. v1's contract is frozen; v2 adds a
deliberate breaking change (a full probability breakdown) without
touching v1 at all — proven by tests in `tests/test_versioning.py` that
call both versions with the same input and assert v1's shape never changed.

### `POST /api/v1/predict` · `POST /api/v2/predict`

```json
// Request (same for both versions)
{"sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2}
```

| Feature | Min | Max |
|---|---|---|
| sepal_length | 4.3 | 7.9 |
| sepal_width | 2.0 | 4.4 |
| petal_length | 1.0 | 6.9 |
| petal_width | 0.1 | 2.5 |

```json
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
```

- **422** — Pydantic validation error, naming the exact field/rule/value (same rules, both versions).
- **400 / 500** — a `ValueError` (bad shape reaching the model) returns 400; anything else returns 500. Both include `request_id`; neither exposes internals.

### `POST /api/v1/predict-batch` · `POST /api/v2/predict-batch`

Accepts 1–`MAX_BATCH_SIZE` inputs (default 100). Runs inference once on
the whole batch (vectorized), never in a per-row loop. `model_version`
and `request_id` live once at the batch level, not repeated per item —
every item in one call shares the same request and the same loaded model.

```json
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
```

Exceeding the batch limit or sending an empty list returns 422 (same limit, both versions).

### `GET /api/v1/model-info`

Returns metadata from `ml/saved_model/model_info.json` (written by the
training script — never hardcoded): `model_type`, `model_version`,
`trained_on`, `feature_names`, `target_names`, `n_estimators`, `test_accuracy`.

### `GET /api/v1/health`

Returns `{"status": "ok", "model_loaded": true}` (or `"degraded"` / `false`).

### `GET /metrics`

Exposes live operational data in Prometheus text format — generic HTTP
metrics (request counts, latency, payload sizes) from
`prometheus-fastapi-instrumentator`, plus one custom, ML-specific metric:

- **`iris_prediction_entropy_bits`** (histogram, labeled by `api_version`)
  — the Shannon entropy of each successful prediction's probability
  distribution. `0` bits means the model was fully decisive; up to
  `log₂(3) ≈ 1.585` bits means the input landed right on a decision
  boundary between two species. Recorded only on successful predictions
  (a `422`/`400`/`500` never contributes an observation), so the metric
  reflects genuine model uncertainty, not request failures.

  Chosen over a simpler per-class request counter because it catches a
  different failure mode: a class counter shows *what* the model is
  predicting and can catch output-distribution drift, but says nothing
  about individual predictions becoming less decisive while the overall
  class mix looks normal. Entropy catches that directly, and it's
  computed from probability data (`model.predict_proba()`) the app was
  already calculating for `/api/v2/predict`'s response — no extra model
  calls, no extra endpoint-specific logic.

Unauthenticated by design, for the same reason as `/health`: Prometheus
itself calls this endpoint on a schedule (every 5s per `prometheus.yml`)
with no credentials, so requiring `X-API-Key` here would make the entire
monitoring stack silently fail every scrape.

An alert rule (`alert_rules.yml`, loaded by the bundled Prometheus
container) watches the 15-minute rolling average of this metric and
fires a `warning`-severity alert if it stays above `1.0` bits for a
sustained 10 minutes — a single ambiguous prediction is expected model
behavior (versicolor/virginica genuinely overlap), but a sustained rise
is a signal worth investigating (data quality, distribution shift).

## Authentication

Every endpoint except `/api/v1/health` requires an `X-API-Key` header
matching the configured `API_KEY`. Health checks are left open
deliberately — infrastructure tooling (load balancers, uptime monitors)
needs to reach them without holding a secret.

```bash
curl -X POST http://localhost:8000/api/v1/predict \
  -H "X-API-Key: your-key-here" \
  -H "Content-Type: application/json" \
  -d '{"sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2}'
```

- **401** — missing or incorrect key. Comparison is constant-time
  (`secrets.compare_digest`) to avoid leaking timing information about
  a partially-correct key.
- Failed attempts are logged (with `request_id`) but never log the
  submitted key itself.

## CORS

Only origins listed in `ALLOWED_ORIGINS` (comma-separated in `.env`) can
call this API from browser JavaScript. Not left wildcarded — an
explicit allowlist is required, since `allow_credentials=True` combined
with a wildcard origin is both a real security risk and something
browsers reject outright.

## Configuration

Twelve-factor style: environment-specific values live in `.env`
(git-ignored), not in code. `.env.example` is committed and documents
what's expected. Every setting has a working default, so the app runs
even with no `.env` present.

| Variable | Default | Purpose |
|---|---|---|
| `MODEL_PATH` | `ml/saved_model/model.joblib` | Trained model file |
| `MODEL_INFO_PATH` | `ml/saved_model/model_info.json` | Model metadata file |
| `LOG_LEVEL` | `INFO` | Minimum log level |
| `MAX_BATCH_SIZE` | `100` | Max items per `/predict-batch` call |
| `API_TITLE` | `Iris Flower Classification API` | Shown in `/docs` and `/` |
| `API_KEY` | *(placeholder)* | Required for all endpoints except `/health` |
| `ALLOWED_ORIGINS` | `http://localhost:3000` | Comma-separated list of origins allowed by CORS |

## Engineering Notes

- **Validation** — feature-specific `ge`/`le` bounds derived from the dataset. `extra="forbid"` on every schema, request AND response — unexpected input is rejected before it reaches the model, and any accidental extra field on a response object raises immediately instead of silently leaking or dropping.
- **Response shape** — every endpoint has a strict `response_model`; no unintended fields ever reach the client.
- **Error handling** — `ValueError` → 400, anything else → 500, consistently across every endpoint, both versions. Client sees a safe generic message; the real error is logged server-side only.
- **Tracing** — one `request_id` per request, generated once in middleware, flowing through the log line, response body, and `X-Request-ID` header.
- **Logging** — console + rotating file (`logs/app.log`, ~1MB, 3 backups). DEBUG (raw features), INFO (requests/success), WARNING (>200ms), ERROR (failures).
- - **Startup failure handling** — model/config loading in the `lifespan` function is wrapped in a try/except that logs the real error to `logs/app.log` before re-raising, so a bad `MODEL_PATH` or corrupted model file leaves a traceable record instead of only a console message that disappears when the terminal closes.
- **API versioning** — `app/routers/v1.py` and `v2.py`, each their own `APIRouter`, both included into `app` in `main.py`. v2 imports and reuses v1's inference helpers directly rather than duplicating them — the only genuinely new code per version is its own schema and route logic. Proven independent with tests that construct v1's schema with v2-shaped data and confirm it's rejected, not silently accepted.
- **Batch efficiency** — every predict/predict-batch route (both versions) shares one inference helper that calls `model.predict()`/`predict_proba()` exactly once per request, on the whole array.
- **Configuration** — centralized in `app/config.py` via `pydantic-settings`. The batch size limit is enforced through a `field_validator` that reads the setting at *request time*, so it's genuinely reconfigurable without restarting the app.
- **Containerization** — single-stage `python:3.11-slim` build, layered so `requirements.txt` installs in its own cached layer separate from app code, keeping rebuilds fast. `.dockerignore` excludes `venv/`, `.env`, `logs/`, and test artifacts from the image.
- **Authentication & CORS** — every route except `/health` requires `X-API-Key`, checked via a FastAPI `Security` dependency applied at the router level (not per-endpoint, so nothing new can accidentally ship unprotected). CORS origins are explicitly allowlisted via `ALLOWED_ORIGINS`, never wildcarded. 
- **Testing** — 65 pytest cases across validation, response shape, both error paths, logging, batch prediction, model metadata, cross-version isolation, and authentication/authorization edge cases.
- **Monitoring** — `prometheus-fastapi-instrumentator` wires up generic HTTP metrics automatically (`http_requests_total`, `http_request_duration_seconds`, payload sizes, in-progress requests). One custom metric was added on top — `iris_prediction_entropy_bits` (see **API Contract → `GET /metrics`** above) — chosen deliberately over the simpler per-class counter the task suggested, because it captures per-prediction model uncertainty rather than just output distribution. A bundled Prometheus container (`docker-compose.yml`) scrapes `/metrics` every 5s and evaluates an alert rule on sustained high uncertainty (`alert_rules.yml`). 
- **Testing** — pytest suite across validation, response shape, both error paths, logging, batch prediction, model metadata, cross-version isolation, authentication/authorization edge cases, and dedicated coverage for the `/metrics` endpoint and the entropy metric's correctness (`tests/test_metrics.py`) — including that failed/invalid requests never record an entropy observation, and that batch calls record one observation per item, not one per request.


## Technology Stack

Python 3.11+ · scikit-learn (Random Forest) · FastAPI · Pydantic · pydantic-settings · Uvicorn · Joblib · pytest · Docker · Docker Compose · Prometheus · prometheus-fastapi-instrumentator ·Git

## API Endpoints

| Method | Endpoint | Purpose | Status |
|---|---|---|---|
| POST | `/api/v1/predict` | Predict one input | Done |
| POST | `/api/v1/predict-batch` | Predict on a batch | Done |
| POST | `/api/v2/predict` | Predict one input + full probability breakdown | Done |
| POST | `/api/v2/predict-batch` | Predict on a batch + full probability breakdown | Done |
| GET | `/api/v1/model-info` | Model metadata | Done |
| GET | `/api/v1/health` | Health check | Done |
| GET | `/metrics` | Prometheus metrics (HTTP + custom entropy metric) | Done |

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
- [x] Additional endpoints (`/predict-batch`, `/model-info`)
- [x] Configuration management (`pydantic-settings`, `.env` / `.env.example`)
- [x] Automated testing (59 pytest cases, organized in `tests/`)
- [x] Build and test the breaking `/v2` change (full parity with v1, cross-version isolation proven by tests)

### Phase 4 — Production Readiness
- [x] Docker
- [x] Docker Compose
- [x] API-key security & CORS configuration

### Phase 5 — Monitoring & Deployment
- [x] Prometheus metrics (`/metrics`)
- [x] Load testing
- [ ] Cloud deployment
- [ ] Final documentation

### Phase 6 — Extension (Planned)
- [ ] Streamlit frontend calling the deployed API, once the core API and
      versioning are stable
- [ ] Alertmanager integration (route the existing `HighPredictionUncertainty`
      alert to a real notification channel — currently visible only in
      Prometheus's own UI)


## Project Goal

Demonstrate how a machine learning model can be transformed from a
notebook script into a validated, tested, versioned, configurable,
containerized, monitored, and deployable production API — one that fails
helpfully, not just safely. The emphasis is on ML deployment and software
engineering practices, not model complexity.
