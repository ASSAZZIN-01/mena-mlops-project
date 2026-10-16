"""Locust scenario for the public Nginx sentiment gateway."""

from __future__ import annotations

import os
from itertools import cycle

from locust import HttpUser, between, task

from mena_mlops.load_testing import (
    valid_batch_prediction_response,
    valid_prediction_response,
)

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


class SentimentBatchUser(HttpUser):
    """Simulate vectorized offline evaluation through Nginx."""

    host = os.getenv("LOCUST_HOST", "http://localhost:8080")
    wait_time = between(0.1, 0.5)
    batch_size = int(os.getenv("LOCUST_BATCH_SIZE", "32"))
    _batch_number = 0

    @task
    def predict_batch(self) -> None:
        self.__class__._batch_number += 1
        batch_number = self.__class__._batch_number
        texts = [
            REVIEWS[index % len(REVIEWS)] for index in range(self.batch_size)
        ]
        with self.client.post(
            "/predict_batch",
            json={
                "texts": texts,
                "source": "batch",
                "input_ids": [
                    f"locust-{batch_number:08d}-{index:03d}"
                    for index in range(self.batch_size)
                ],
                "feedback_policy": "audit_only",
            },
            headers={"X-Source": "load-test"},
            name=f"POST /predict_batch ({self.batch_size})",
            catch_response=True,
        ) as response:
            if response.status_code != 200:
                response.failure(f"HTTP {response.status_code}: {response.text[:200]}")
                return
            try:
                if not valid_batch_prediction_response(response.json()):
                    response.failure("invalid batch prediction contract")
            except (ValueError, TypeError) as error:
                response.failure(f"invalid JSON response: {error}")
