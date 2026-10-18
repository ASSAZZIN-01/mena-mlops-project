"""Progressively promote a candidate and roll back automatically on failure."""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Callable
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from mena_mlops.deployment.canary import apply_split, split_for_candidate
from mena_mlops.observability import emit_error, emit_event


def _json_request(url: str, payload: dict[str, object] | None = None) -> object:
    data = None
    headers = {}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
    request = Request(url, data=data, headers=headers)
    with urlopen(request, timeout=10) as response:
        return json.load(response)


def _prometheus_value(prometheus_url: str, query: str) -> float | None:
    query_string = urlencode({"query": query})
    query_url = f"{prometheus_url.rstrip('/')}/api/v1/query?{query_string}"
    payload = _json_request(query_url)
    if not isinstance(payload, dict) or payload.get("status") != "success":
        raise RuntimeError(f"Prometheus query failed: {query}")
    data = payload.get("data")
    if not isinstance(data, dict):
        raise RuntimeError("Prometheus response did not contain data")
    result = data.get("result")
    if not isinstance(result, list) or not result:
        return None
    value = result[0].get("value")
    if not isinstance(value, list) or len(value) < 2:
        raise RuntimeError("Prometheus response did not contain a value")
    return float(value[1])


def _check_prediction_contract(gateway_url: str, probes: int) -> str | None:
    expected_labels = {"negative", "neutral", "positive"}
    for index in range(probes):
        payload = _json_request(
            f"{gateway_url.rstrip('/')}/predict",
            {
                "text": "هذا المنتج ممتاز",
                "source": "live",
                "input_id": f"canary-{index}",
            },
        )
        if not isinstance(payload, dict):
            return "gateway response is not a JSON object"
        if payload.get("label") not in expected_labels:
            return "gateway response has an invalid label"
        probabilities = payload.get("probabilities")
        if not isinstance(probabilities, dict) or set(probabilities) != expected_labels:
            return "gateway response has an invalid probability map"
        total = sum(float(value) for value in probabilities.values())
        if abs(total - 1.0) > 0.01:
            return f"probabilities sum to {total:.4f}, not approximately 1"
        current_version = payload.get("model_version")
        if not isinstance(current_version, str) or not current_version:
            return "gateway response has no model version"
    return None


def check_stage(
    prometheus_url: str,
    gateway_url: str,
    *,
    max_error_rate: float,
    max_p95_seconds: float,
    probes: int,
    minimum_observations: int = 10,
    maximum_relative_regression: float = 0.20,
) -> list[str]:
    failures: list[str] = []
    candidate_observations = _prometheus_value(
        prometheus_url,
        'sum(increase(bentoml_service_request_duration_seconds_count{'
        'job="model-candidate"}[2m]))',
    )
    stable_observations = _prometheus_value(
        prometheus_url,
        'sum(increase(bentoml_service_request_duration_seconds_count{'
        'job="model-stable"}[2m]))',
    )
    if (
        candidate_observations is None
        or stable_observations is None
        or candidate_observations < minimum_observations
        or stable_observations < minimum_observations
    ):
        failures.append(
            "insufficient stable/candidate observations for canary decision"
        )
    candidate_up = _prometheus_value(
        prometheus_url,
        'sum(up{job="model-candidate"})',
    )
    if candidate_up != 1:
        failures.append("candidate service is not healthy in Prometheus")

    error_rate = _prometheus_value(
        prometheus_url,
        'sum(rate(bentoml_service_request_duration_seconds_count{'
        'job="model-candidate",http_response_code=~"5.."}[2m])) / '
        'sum(rate(bentoml_service_request_duration_seconds_count{'
        'job="model-candidate"}[2m]))',
    )
    if error_rate is not None and error_rate > max_error_rate:
        failures.append(
            f"candidate HTTP error rate {error_rate:.3f} exceeds "
            f"{max_error_rate:.3f}"
        )

    p95 = _prometheus_value(
        prometheus_url,
        'histogram_quantile(0.95, sum by (le) (rate('
        'bentoml_service_request_duration_seconds_bucket{'
        'job="model-candidate"}[2m])))',
    )
    if p95 is not None and p95 > max_p95_seconds:
        failures.append(
            f"candidate p95 latency {p95:.3f}s exceeds "
            f"{max_p95_seconds:.3f}s"
        )
    stable_p95 = _prometheus_value(
        prometheus_url,
        'histogram_quantile(0.95, sum by (le) (rate('
        'bentoml_service_request_duration_seconds_bucket{'
        'job="model-stable"}[2m])))',
    )
    if (
        p95 is not None
        and stable_p95 is not None
        and p95 > stable_p95 * (1 + maximum_relative_regression)
    ):
        failures.append(
            f"candidate p95 latency {p95:.3f}s regresses against stable "
            f"{stable_p95:.3f}s"
        )

    contract_failure = _check_prediction_contract(gateway_url, probes)
    if contract_failure is not None:
        failures.append(contract_failure)
    return failures


def run_canary(
    *,
    prometheus_url: str,
    gateway_url: str,
    stages: tuple[int, ...] = (5, 20, 50, 100),
    duration_seconds: int = 600,
    max_error_rate: float = 0.05,
    max_p95_seconds: float = 1.0,
    probes: int = 10,
    minimum_observations: int = 10,
    maximum_relative_regression: float = 0.20,
    apply: Callable[[object], None] = apply_split,
    sleep: Callable[[float], None] = time.sleep,
) -> None:
    for candidate_percent in stages:
        stage_started = time.monotonic()
        emit_event(
            "canary_stage_started",
            "canary",
            status="started",
            candidate_percent=candidate_percent,
            duration_seconds=duration_seconds,
        )
        apply(split_for_candidate(candidate_percent))
        contract_failure = _check_prediction_contract(gateway_url, probes)
        if contract_failure is not None:
            apply(split_for_candidate(0))
            emit_event(
                "canary_rollback",
                "canary",
                severity="ERROR",
                status="failed",
                candidate_percent=candidate_percent,
                reason=contract_failure,
                duration_ms=round((time.monotonic() - stage_started) * 1000, 2),
            )
            raise RuntimeError(
                f"Canary stage {candidate_percent}% failed: {contract_failure}"
            )
        if duration_seconds:
            emit_event(
                "canary_observation_started",
                "canary",
                status="running",
                candidate_percent=candidate_percent,
                duration_seconds=duration_seconds,
            )
            sleep(duration_seconds)
        failures = check_stage(
            prometheus_url,
            gateway_url,
            max_error_rate=max_error_rate,
            max_p95_seconds=max_p95_seconds,
            probes=probes,
            minimum_observations=minimum_observations,
            maximum_relative_regression=maximum_relative_regression,
        )
        if failures:
            apply(split_for_candidate(0))
            emit_event(
                "canary_rollback",
                "canary",
                severity="ERROR",
                status="failed",
                candidate_percent=candidate_percent,
                failures=failures,
                duration_ms=round((time.monotonic() - stage_started) * 1000, 2),
            )
            raise RuntimeError(
                f"Canary stage {candidate_percent}% failed: "
                + "; ".join(failures)
            )
        emit_event(
            "canary_stage_completed",
            "canary",
            status="passed",
            candidate_percent=candidate_percent,
            duration_ms=round((time.monotonic() - stage_started) * 1000, 2),
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prometheus-url", default="http://localhost:9090")
    parser.add_argument("--gateway-url", default="http://localhost:8080")
    parser.add_argument("--duration-seconds", type=int, default=600)
    parser.add_argument("--max-error-rate", type=float, default=0.05)
    parser.add_argument("--max-p95-seconds", type=float, default=1.0)
    parser.add_argument("--probes", type=int, default=10)
    parser.add_argument("--minimum-observations", type=int, default=10)
    parser.add_argument(
        "--maximum-relative-regression",
        type=float,
        default=0.20,
        help="Maximum candidate p95 regression versus stable, as a ratio.",
    )
    args = parser.parse_args()
    if (
        args.duration_seconds < 0
        or args.probes < 1
        or args.minimum_observations < 1
        or args.maximum_relative_regression < 0
    ):
        parser.error("canary thresholds must be non-negative and probes positive")
    try:
        run_canary(
            prometheus_url=args.prometheus_url,
            gateway_url=args.gateway_url,
            duration_seconds=args.duration_seconds,
            max_error_rate=args.max_error_rate,
            max_p95_seconds=args.max_p95_seconds,
            probes=args.probes,
            minimum_observations=args.minimum_observations,
            maximum_relative_regression=args.maximum_relative_regression,
        )
    except Exception as error:
        apply_split(split_for_candidate(0))
        emit_error("canary_run_failed", "canary", error, rollback_percent=0)
        raise


if __name__ == "__main__":
    main()
