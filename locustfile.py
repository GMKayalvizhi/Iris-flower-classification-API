"""
Locust load test for the Iris API.

Task 19: basic load test, 50-200 concurrent requests, to see how the
system holds up under concurrency and surface real issues (slow
responses, failures, container instability) that unit/integration
tests can't catch on their own. Exercises both v1 and v2 -- since
v2's extra probability-breakdown work runs through the same shared
run_inference() helper, this also checks that the added computation
doesn't behave differently under load.

Run with the web UI:
    locust -f locustfile.py --host http://localhost:8000

Run headless, e.g. 100 users, ramping up at 10/sec, for 60 seconds:
    locust -f locustfile.py --host http://localhost:8000 \
        --users 100 --spawn-rate 10 --run-time 60s --headless \
        --csv=load_test_results
"""

import os

from locust import HttpUser, task, between

API_KEY = os.environ.get("API_KEY", "")

VALID_INPUT = {
    "sepal_length": 5.1,
    "sepal_width": 3.5,
    "petal_length": 1.4,
    "petal_width": 0.2,
}

BATCH_INPUT = {"inputs": [VALID_INPUT, VALID_INPUT, VALID_INPUT]}


class IrisApiUser(HttpUser):
    # Small pause between a simulated user's requests -- real clients
    # don't fire back-to-back with zero delay. Keeps the test closer to
    # realistic traffic rather than an artificial instant burst.
    wait_time = between(0.1, 0.5)

    def on_start(self):
        self.headers = {"X-API-Key": API_KEY, "Content-Type": "application/json"}

    @task(5)
    def predict_v1(self):
        self.client.post("/api/v1/predict", json=VALID_INPUT, headers=self.headers)

    @task(5)
    def predict_v2(self):
        self.client.post("/api/v2/predict", json=VALID_INPUT, headers=self.headers)

    @task(1)
    def predict_batch_v1(self):
        self.client.post("/api/v1/predict-batch", json=BATCH_INPUT, headers=self.headers)

    @task(1)
    def predict_batch_v2(self):
        self.client.post("/api/v2/predict-batch", json=BATCH_INPUT, headers=self.headers)

    @task(1)
    def health(self):
        # No API key needed -- unauthenticated by design.
        self.client.get("/api/v1/health")