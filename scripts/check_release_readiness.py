"""Show the human-readable production decision for a candidate release."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from urllib.parse import urlparse

from google.cloud import storage


def _read_decision(uri: str) -> dict[str, object]:
    parsed = urlparse(uri)
    if parsed.scheme == "gs":
        if not parsed.netloc or not parsed.path:
            raise ValueError(f"invalid GCS decision URI: {uri}")
        payload = (
            storage.Client()
            .bucket(parsed.netloc)
            .blob(parsed.path.lstrip("/"))
            .download_as_bytes()
        )
        decision = json.loads(payload)
    else:
        decision = json.loads(Path(uri).read_text(encoding="utf-8"))
    if not isinstance(decision, dict):
        raise ValueError("release decision must be a JSON object")
    return decision


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--decision",
        required=True,
        help="Local promotion_decision.json or a gs:// URI to it.",
    )
    parser.add_argument("--json", action="store_true", dest="as_json")
    args = parser.parse_args()

    decision = _read_decision(args.decision)
    if args.as_json:
        print(json.dumps(decision, indent=2, ensure_ascii=False))
        return

    status = str(decision.get("status", "unknown")).upper()
    selected = decision.get("selected_variant") or "none"
    quality = decision.get("quality_gate", {})
    optimization = decision.get("optimization_gate", {})
    print(f"Release decision: {status}")
    print(f"Selected deployment variant: {selected}")
    print(
        "Quality gate: "
        f"{'PASS' if quality.get('accepted') else 'FAIL'}"
    )
    for failure in quality.get("failures", []):
        print(f"  - {failure}")
    print(
        "Optimization gate: "
        f"{'PASS' if optimization.get('accepted') else 'FAIL'}"
    )
    for failure in optimization.get("failures", []):
        print(f"  - {failure}")
    if status == "APPROVED":
        print("Action: the candidate may be deployed with the approved deploy command.")
    else:
        print("Action: do not deploy this candidate.")
    raise SystemExit(0 if status == "APPROVED" else 1)


if __name__ == "__main__":
    main()
