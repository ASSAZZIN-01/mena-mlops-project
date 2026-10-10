"""Download and verify a public deployment model bundle."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from urllib.request import urlopen

DEFAULT_BUNDLE_URL = (
    "https://storage.googleapis.com/"
    "mena-mlops-public-models/models/onnx-int8-v1"
)


def download(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with urlopen(url, timeout=120) as response, destination.open("wb") as output:
        while chunk := response.read(1024 * 1024):
            output.write(chunk)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle-url", default=DEFAULT_BUNDLE_URL)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("models/arabert-debug-variants"),
    )
    args = parser.parse_args()

    manifest_path = args.output_dir / "manifest.json"
    download(f"{args.bundle_url.rstrip('/')}/manifest.json", manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for entry in manifest["files"].values():
        filename = Path(entry["object"]).name
        destination = args.output_dir / filename
        download(f"{args.bundle_url.rstrip('/')}/{filename}", destination)
        actual = sha256(destination)
        if actual != entry["sha256"]:
            raise RuntimeError(
                f"checksum mismatch for {filename}: "
                f"expected {entry['sha256']}, got {actual}"
            )
        print(f"verified {destination}")


if __name__ == "__main__":
    main()
