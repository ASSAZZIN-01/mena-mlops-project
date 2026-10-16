"""Import one immutable Vertex promotion decision into the local dashboard registry."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from urllib.parse import urlparse

from google.cloud import storage

from mena_mlops.release_registry import ReleaseRegistry


def read_decision(uri: str) -> dict[str, object]:
    parsed = urlparse(uri)
    if parsed.scheme != "gs" or not parsed.netloc or not parsed.path:
        raise ValueError(f"expected a GCS decision URI, got {uri!r}")
    payload = (
        storage.Client()
        .bucket(parsed.netloc)
        .blob(parsed.path.lstrip("/"))
        .download_as_text()
    )
    decision = json.loads(payload)
    if not isinstance(decision, dict):
        raise ValueError("promotion decision must be a JSON object")
    return decision


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--decision", required=True)
    parser.add_argument(
        "--registry",
        type=Path,
        default=Path("reports/control-plane/releases.db"),
    )
    args = parser.parse_args()
    decision = read_decision(args.decision)
    parts = urlparse(args.decision).path.rstrip("/").split("/")
    if len(parts) < 3:
        raise ValueError("decision URI does not contain a release ID")
    release_id = parts[-3]
    registry = ReleaseRegistry(args.registry)
    registry.start_release(
        release_id,
        trigger="imported",
        vertex_job_id=release_id,
    )
    registry.record_decision(
        release_id,
        decision_uri=args.decision,
        decision=decision,
        payload={"source": "sync_release_registry"},
    )
    if decision.get("status") != "approved":
        registry.record_deployment(
            release_id,
            status="not_attempted",
            payload={"reason": "release decision was rejected"},
        )
    print(release_id)


if __name__ == "__main__":
    main()
