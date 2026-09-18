# Testing

## What I tested

- **Integration tests** (`tests/test_integration.py`) against the live
  Docker container over real HTTP: /health, /predict, /predict-batch,
  missing API key → 401, /metrics. All 5 passed.
- **Load test** (Locust, `locustfile.py`) at 50, 100, and 200 concurrent
  users against /predict and related endpoints.

## What broke

Under load, response time grew far beyond what a simple ML API should
take, and got worse the more users were added -- at 50 users, average
response time was 2 seconds; nothing failed, but it was silently slow.

Root cause: only 4 worker processes were running (not matched to the
8 available CPU cores), and NumPy's math library was spawning its own
extra threads inside each one -- so threads were fighting each other
for the same cores instead of working in parallel.

## What I fixed

`Dockerfile`: forced NumPy's math libraries to run single-threaded
(`OMP_NUM_THREADS=1`, etc.) and raised `--workers` from 4 to 8 to match
the available cores.

## Proof

| Users | Avg (before) | Avg (after) | Improvement | Throughput (after) | Failures |
|---|---|---|---|---|---|
| 50  | 2021ms | 134ms  | 15x  | 110.2 req/s | 0% |
| 100 | 4380ms | 438ms  | 10x | 123.6 req/s | 0% |
| 200 | 8805ms | 1301ms | 6.8x | 113.4 req/s | 0% |

CPU usage during the 100-user test rose from ~187% to ~546% (of an
800% ceiling) after the fix -- confirming the extra CPU time was now
going toward real work instead of threads competing for the same cores.

At 200 users, throughput held steady rather than climbing further
(113.4 req/s vs. 123.6 at 100 users) -- the system is running at its
available 8-core capacity, and further improvement past this point
would need more cores or multiple container instances rather than a
configuration change.