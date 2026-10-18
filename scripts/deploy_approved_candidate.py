"""Deploy an approved bundle to the local candidate slot and start a canary."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from urllib.parse import urlparse

from google.cloud import storage

from mena_mlops.deployment.canary import docker_compose_command
from mena_mlops.deployment_identity import build_deployment_identity
from mena_mlops.feedback import FeedbackStore
from mena_mlops.observability import emit_event
from mena_mlops.release_registry import ReleaseRegistry


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
            if relative:
                target = destination / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                blob.download_to_filename(target)
        return

    source_path = Path(source)
    if not source_path.is_dir():
        raise FileNotFoundError(f"local bundle directory does not exist: {source}")
    for item in source_path.iterdir():
        target = destination / item.name
        if item.is_dir():
            shutil.copytree(item, target, dirs_exist_ok=True)
        elif item.is_file():
            shutil.copy2(item, target)


def _write_deployed_baseline(
    decision: dict[str, object],
    identity: dict[str, object],
    path: Path,
) -> None:
    selected_variant = str(decision["selected_variant"])
    results = decision.get("results")
    if not isinstance(results, dict) or not isinstance(
        results.get(selected_variant), dict
    ):
        raise RuntimeError("promotion decision has no selected variant benchmark")
    result = results[selected_variant]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "model_id": identity["model_id"],
                "selected_variant": selected_variant,
                "quality": result["quality"],
                "latency": result["latency"],
                "hardware": {
                    "backend": "onnxruntime",
                    "provider": "CPUExecutionProvider",
                },
                "model_card_uri": identity.get("model_card_uri"),
                "release_uri": identity.get("release_uri"),
                "promotion_decision_uri": identity.get(
                    "promotion_decision_uri"
                ),
                "deployed_at": identity.get("deployed_at"),
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )


def _release_id_from_decision_uri(uri: str) -> str:
    path = urlparse(uri).path.rstrip("/")
    parts = path.split("/")
    if len(parts) < 3:
        raise ValueError(f"cannot derive release ID from decision URI: {uri}")
    return parts[-3]


def _promote_stable(
    *,
    compose: list[str],
    environment: dict[str, str],
    candidate_environment: dict[str, str],
) -> None:
    """Switch stable only after Compose readiness, restoring the old slot on failure."""

    previous = {
        key: environment.get(key)
        for key in (
            "STABLE_MODEL_PATH",
            "STABLE_MODEL_FILE",
            "STABLE_MODEL_VERSION",
        )
    }
    promoted = environment | candidate_environment
    promoted.update({"STABLE_WEIGHT": "100", "CANDIDATE_WEIGHT": "0"})
    try:
        subprocess.run(
            [*compose, "up", "-d", "--wait", "model-stable", "nginx"],
            check=True,
            env=promoted,
        )
    except subprocess.CalledProcessError as error:
        rollback = environment.copy()
        rollback.update(
            {
                key: value
                for key, value in previous.items()
                if value is not None
            }
        )
        rollback.update({"STABLE_WEIGHT": "100", "CANDIDATE_WEIGHT": "0"})
        subprocess.run(
            [*compose, "up", "-d", "--wait", "model-stable", "nginx"],
            check=True,
            env=rollback,
        )
        raise RuntimeError(
            "stable promotion failed; previous stable service was restored"
        ) from error


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
    parser.add_argument(
        "--deployed-baseline-path",
        type=Path,
        default=Path("reports/retraining/deployed_baseline.json"),
    )
    args = parser.parse_args()

    if not 1 <= args.candidate_percent <= 99:
        parser.error("--candidate-percent must be between 1 and 99")
    if args.stage_duration_seconds < 0:
        parser.error("--stage-duration-seconds must be non-negative")
    decision = _read_decision(args.decision)
    release_id = _release_id_from_decision_uri(args.decision)
    registry = ReleaseRegistry(
        Path(
            os.getenv(
                "CONTROL_PLANE_DB",
                "reports/control-plane/releases.db",
            )
        )
    )
    if decision.get("status") != "approved":
        registry.record_deployment(
            release_id,
            status="not_attempted",
            payload={
                "reason": "release decision was not approved",
                "decision_uri": args.decision,
            },
        )
        raise SystemExit("Refusing deployment: release decision is not approved.")
    model_version = registry.allocate_model_version(release_id)
    registry_version = decision.get("registry_version")
    if isinstance(registry_version, str) and registry_version.strip():
        registry.set_registry_version(model_version, registry_version)
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

    release_output_dir = args.output_dir.parent / model_version
    _download_bundle(bundle, release_output_dir)
    required = (model_file, "tokenizer.json")
    missing = [
        name for name in required if not (release_output_dir / name).is_file()
    ]
    if missing:
        raise RuntimeError(f"release bundle is missing: {', '.join(missing)}")
    identity = build_deployment_identity(
        decision=decision,
        decision_uri=args.decision,
        bundle=bundle,
        output_dir=release_output_dir,
        selected_variant=str(selected_variant),
    )
    identity["deployment_version"] = model_version
    (release_output_dir / "deployment_identity.json").write_text(
        json.dumps(identity, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    stable_percent = 100 - args.candidate_percent
    emit_event(
        "deployment_release_downloaded",
        "deployment",
        status="approved",
        decision_uri=args.decision,
        output_dir=str(release_output_dir),
        selected_variant=selected_variant,
        model_file=model_file,
        model_version=model_version,
        registry_version=registry_version,
        model_id=identity["model_id"],
        model_card_sha256=identity["model_card_sha256"],
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
                "CANDIDATE_MODEL_PATH": str(release_output_dir.resolve()),
                "CANDIDATE_MODEL_FILE": model_file,
                "CANDIDATE_MODEL_VERSION": model_version,
                "STABLE_WEIGHT": str(stable_percent),
                "CANDIDATE_WEIGHT": str(args.candidate_percent),
            }
        )
        subprocess.run(
            [
                *docker_compose_command(),
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
                    sys.executable,
                    "scripts/auto_canary.py",
                    "--duration-seconds",
                    str(args.stage_duration_seconds),
                ],
                check=True,
                env=environment,
            )
            promotion_environment = {
                "STABLE_MODEL_PATH": str(release_output_dir.resolve()),
                "STABLE_MODEL_FILE": model_file,
                "STABLE_MODEL_VERSION": model_version,
            }
            _promote_stable(
                compose=docker_compose_command(),
                environment=environment,
                candidate_environment=promotion_environment,
            )
            registry.set_model_version_status(model_version, "stable")
            emit_event(
                "deployment_release_promoted",
                "deployment",
                status="stable",
                model_version=model_version,
                rollback_version=registry.get_model_release_pointers().get(
                    "rollback"
                ),
            )
        _write_deployed_baseline(
            decision,
            identity,
            args.deployed_baseline_path,
        )
        registry.record_deployment(
            release_id,
            status="deployed",
            payload={
            "candidate_percent": args.candidate_percent,
            "model_id": identity["model_id"],
            "model_version": model_version,
            },
        )
        feedback_store = FeedbackStore(
            Path(
                os.getenv(
                    "FEEDBACK_DB",
                    "data/feedback/reviewed.db",
                )
            )
        )
        feedback_store.mark_successful_deployment(
            feedback_store.latest_reviewed_id(),
            reviewed_at=feedback_store.latest_reviewed_at(),
        )
    else:
        emit_event(
            "candidate_services_start_skipped",
            "deployment",
            status="skipped",
        )


if __name__ == "__main__":
    main()
