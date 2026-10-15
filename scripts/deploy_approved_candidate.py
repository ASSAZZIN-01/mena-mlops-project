"""Deploy an approved bundle to the local candidate slot and start a canary."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
from pathlib import Path
from urllib.parse import urlparse

from google.cloud import storage

from mena_mlops.observability import emit_event


def _read_decision(uri: str) -> dict[str, object]:
    parsed = urlparse(uri)
    if parsed.scheme == "gs":
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


def _download_bundle(source: str, destination: Path) -> None:
    parsed = urlparse(source)
    destination.mkdir(parents=True, exist_ok=True)
    if parsed.scheme == "gs":
        if not parsed.netloc or not parsed.path:
            raise ValueError(f"invalid GCS bundle URI: {source}")
        client = storage.Client()
        prefix = parsed.path.lstrip("/").rstrip("/") + "/"
        blobs = list(client.bucket(parsed.netloc).list_blobs(prefix=prefix))
        if not blobs:
            raise FileNotFoundError(f"no release files found below {source}")
        for blob in blobs:
            relative = blob.name[len(prefix) :]
            if relative and "/" not in relative:
                blob.download_to_filename(destination / relative)
        return

    source_path = Path(source)
    if not source_path.is_dir():
        raise FileNotFoundError(f"local bundle directory does not exist: {source}")
    for item in source_path.iterdir():
        if item.is_file():
            shutil.copy2(item, destination / item.name)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--decision",
        required=True,
        help="Local promotion_decision.json or a gs:// URI to it.",
    )
    parser.add_argument(
        "--bundle",
        help="Local bundle directory or gs:// release prefix. "
        "For GCS decisions, the decision's parent prefix is used by default.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("models/candidate-release"),
    )
    parser.add_argument("--candidate-percent", type=int, default=5)
    parser.add_argument("--skip-compose", action="store_true")
    parser.add_argument("--manual-canary", action="store_true")
    parser.add_argument("--stage-duration-seconds", type=int, default=600)
    args = parser.parse_args()

    if not 1 <= args.candidate_percent <= 99:
        parser.error("--candidate-percent must be between 1 and 99")
    if args.stage_duration_seconds < 0:
        parser.error("--stage-duration-seconds must be non-negative")
    decision = _read_decision(args.decision)
    if decision.get("status") != "approved":
        raise SystemExit("Refusing deployment: release decision is not approved.")
    selected_variant = decision.get("selected_variant")
    model_file = {
        "onnx-int8": "model-int8.onnx",
        "onnx-fp32": "model.onnx",
    }.get(str(selected_variant))
    if model_file is None:
        raise SystemExit(
            "Refusing deployment: approved bundle has an unsupported variant."
        )

    bundle = args.bundle
    parsed_decision = urlparse(args.decision)
    if bundle is None and parsed_decision.scheme == "gs":
        bundle = args.decision.rsplit("/", 1)[0]
    if bundle is None:
        parser.error("--bundle is required for a local decision file")

    _download_bundle(bundle, args.output_dir)
    required = (model_file, "tokenizer.json")
    missing = [name for name in required if not (args.output_dir / name).is_file()]
    if missing:
        raise RuntimeError(f"release bundle is missing: {', '.join(missing)}")

    stable_percent = 100 - args.candidate_percent
    emit_event(
        "deployment_release_downloaded",
        "deployment",
        status="approved",
        decision_uri=args.decision,
        output_dir=str(args.output_dir),
        selected_variant=selected_variant,
        model_file=model_file,
    )
    emit_event(
        "deployment_canary_started",
        "deployment",
        status="started",
        stable_percent=stable_percent,
        candidate_percent=args.candidate_percent,
    )
    if not args.skip_compose:
        environment = os.environ.copy()
        environment.update(
            {
                "CANDIDATE_MODEL_BACKEND": "onnx",
                "CANDIDATE_MODEL_PATH": str(args.output_dir.resolve()),
                "CANDIDATE_MODEL_FILE": model_file,
                "CANDIDATE_MODEL_VERSION": args.output_dir.name,
                "STABLE_WEIGHT": str(stable_percent),
                "CANDIDATE_WEIGHT": str(args.candidate_percent),
            }
        )
        subprocess.run(
            [
                "docker",
                "compose",
                "up",
                "-d",
                "model-candidate",
                "prometheus",
                "nginx",
            ],
            check=True,
            env=environment,
        )
        emit_event("candidate_services_started", "deployment", status="running")
        if not args.manual_canary:
            subprocess.run(
                [
                    "uv",
                    "run",
                    "python",
                    "scripts/auto_canary.py",
                    "--duration-seconds",
                    str(args.stage_duration_seconds),
                ],
                check=True,
                env=environment,
            )
    else:
        emit_event(
            "candidate_services_start_skipped",
            "deployment",
            status="skipped",
        )


if __name__ == "__main__":
    main()
