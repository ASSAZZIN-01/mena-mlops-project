"""Locust scenario for the public Nginx sentiment gateway."""

from __future__ import annotations

import os
from itertools import cycle

from locust import HttpUser, between, task

from mena_mlops.load_testing import valid_prediction_response

REVIEWS = [
    "المنتج ممتاز والجودة عالية جدا",
    "الخدمة سيئة والتوصيل تأخر كثيرا",
    "المنتج عادي وليس سيئا",
    "السعر مناسب والتجربة جيدة",
]
review_cycle = cycle(REVIEWS)


class SentimentGatewayUser(HttpUser):
    """Simulate interactive review submissions through Nginx."""

    host = os.getenv("LOCUST_HOST", "http://localhost:8080")
    wait_time = between(0.1, 0.5)

    @task(10)
    def predict(self) -> None:
        with self.client.post(
            "/predict",
            json={"text": next(review_cycle)},
            name="POST /predict",
            catch_response=True,
        ) as response:
            if response.status_code != 200:
                response.failure(f"HTTP {response.status_code}: {response.text[:200]}")
                return
            try:
                if not valid_prediction_response(response.json()):
                    response.failure("invalid prediction contract")
            except (ValueError, TypeError) as error:
                response.failure(f"invalid JSON response: {error}")

    @task
    def health(self) -> None:
        with self.client.get(
            "/health",
            name="GET /health",
            catch_response=True,
        ) as response:
            if response.status_code != 200:
                response.failure(f"HTTP {response.status_code}")
