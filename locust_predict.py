from locust import HttpUser, task, between


class PredictUser(HttpUser):
    wait_time = between(1, 2)

    @task
    def predict(self):
        self.client.post(
            "/api/v1/predict",
            json={
                "sepal_length": 5.1,
                "sepal_width": 3.5,
                "petal_length": 1.4,
                "petal_width": 0.2,
            },
            headers={
                "X-API-Key": "your-actual-real-key"
            }
        )