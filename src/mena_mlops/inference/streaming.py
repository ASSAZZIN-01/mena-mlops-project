"""Redis Streams inference consumer."""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from redis import Redis
from redis.exceptions import ResponseError

from mena_mlops.feedback import FeedbackStore

LOGGER = logging.getLogger(__name__)


class StreamPredictor(Protocol):
    model_version: str

    def predict(self, text: str) -> tuple[str, dict[str, float]]: ...


@dataclass(frozen=True)
class ReviewEvent:
    """Validated review event received from the input stream."""

    text: str
    event_id: str | None = None


def parse_review_event(fields: dict[str, object]) -> ReviewEvent:
    """Validate a Redis stream review payload."""

    raw_text = fields.get("text")
    if not isinstance(raw_text, str) or not raw_text.strip():
        raise ValueError("stream event requires a non-empty text field")
    raw_event_id = fields.get("event_id")
    event_id = raw_event_id if isinstance(raw_event_id, str) else None
    return ReviewEvent(text=raw_text, event_id=event_id)


def build_prediction_event(
    message_id: str,
    review: ReviewEvent,
    predictor: StreamPredictor,
) -> dict[str, str]:
    """Create a serializable prediction event for the output stream."""

    label, probabilities = predictor.predict(review.text)
    return {
        "message_id": message_id,
        "event_id": review.event_id or "",
        "text": review.text,
        "prediction": label,
        "probabilities": json.dumps(
            probabilities,
            ensure_ascii=False,
            sort_keys=True,
        ),
        "model_version": predictor.model_version,
    }


class RedisInferenceConsumer:
    """Consume, predict, publish, and acknowledge Redis stream messages."""

    def __init__(
        self,
        client: Redis,
        predictor: StreamPredictor,
        *,
        input_stream: str = "reviews:input",
        output_stream: str = "reviews:predictions",
        group: str = "mena-inference",
        consumer: str = "worker-1",
        feedback_store: FeedbackStore | None = None,
        feedback_min_confidence: float = 0.25,
        feedback_max_confidence: float = 0.75,
    ) -> None:
        self.client = client
        self.predictor = predictor
        self.input_stream = input_stream
        self.output_stream = output_stream
        self.group = group
        self.consumer = consumer
        self.feedback_store = feedback_store
        self.feedback_min_confidence = feedback_min_confidence
        self.feedback_max_confidence = feedback_max_confidence

    def ensure_group(self) -> None:
        """Create the consumer group once, without hiding connection errors."""

        try:
            self.client.xgroup_create(
                self.input_stream,
                self.group,
                id="0",
                mkstream=True,
            )
        except ResponseError as error:
            if "BUSYGROUP" not in str(error):
                raise

    def run_once(self, *, block_ms: int = 1000, count: int = 10) -> int:
        """Process one Redis read batch and return the acknowledged count."""

        messages = self.client.xreadgroup(
            self.group,
            self.consumer,
            {self.input_stream: ">"},
            count=count,
            block=block_ms,
        )
        acknowledged = 0
        for _, entries in messages:
            for message_id, fields in entries:
                try:
                    review = parse_review_event(fields)
                    prediction = build_prediction_event(
                        message_id,
                        review,
                        self.predictor,
                    )
                    self.client.xadd(self.output_stream, prediction)
                    if self.feedback_store is not None:
                        probabilities = json.loads(prediction["probabilities"])
                        confidence = max(probabilities.values())
                        if (
                            self.feedback_min_confidence
                            <= confidence
                            <= self.feedback_max_confidence
                        ):
                            self.feedback_store.add_uncertain(
                                review.text,
                                prediction["prediction"],
                                confidence,
                            )
                    self.client.xack(self.input_stream, self.group, message_id)
                    acknowledged += 1
                except (TypeError, ValueError, RuntimeError):
                    LOGGER.exception(
                        "Failed to process Redis message %s; leaving it pending",
                        message_id,
                    )
        return acknowledged


def create_consumer() -> RedisInferenceConsumer:
    """Build a configured consumer for the local deployment."""

    from mena_mlops.serving.app import _load_predictor

    feedback_store = FeedbackStore(
        Path(os.getenv("FEEDBACK_DB", "data/feedback/reviewed.db"))
    )
    return RedisInferenceConsumer(
        Redis.from_url(os.getenv("REDIS_URL", "redis://localhost:6379/0")),
        _load_predictor(),
        input_stream=os.getenv("REDIS_INPUT_STREAM", "reviews:input"),
        output_stream=os.getenv("REDIS_OUTPUT_STREAM", "reviews:predictions"),
        group=os.getenv("REDIS_CONSUMER_GROUP", "mena-inference"),
        consumer=os.getenv("REDIS_CONSUMER_NAME", "worker-1"),
        feedback_store=feedback_store,
        feedback_min_confidence=float(
            os.getenv("FEEDBACK_MIN_CONFIDENCE", "0.25")
        ),
        feedback_max_confidence=float(
            os.getenv("FEEDBACK_MAX_CONFIDENCE", "0.75")
        ),
    )
