"""Build local private identity-only manifests from metadata-authorized roots."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
from typing import Any

from .identity import (build_realiad_identity_manifest,
                       build_visa_1cls_identity_manifest,
                       validate_identity_manifest, write_manifest_exclusive)


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_REALIAD_ROOT = Path(r"G:\dataset\realiad")
DEFAULT_REALIAD_JSON_ROOT = Path(r"G:\dataset\realiad\jsons\realiad_jsons")
DEFAULT_VISA_ROOT = Path(r"G:\dataset\VisA")


def _git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, check=True,
                          capture_output=True, text=True).stdout.strip()


def _json_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")


def build_all(*, realiad_root: Path, realiad_json_root: Path,
              visa_root: Path, output_dir: Path, workers: int = 4) -> dict[str, Any]:
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise FileExistsError("Gate 1 identity output path already exists; refusing to overwrite")
    output_dir.mkdir(parents=True, exist_ok=False)
    started = time.time()
    revision = _git("rev-parse", "HEAD")
    branch = _git("branch", "--show-current")
    # No official archive/release receipt has been supplied; leaving the
    # frozen release digest unset deliberately blocks role-hash certification.
    realiad = build_realiad_identity_manifest(
        realiad_root, realiad_json_root, verified_official_release_sha256=None,
        workers=workers)
    visa = build_visa_1cls_identity_manifest(
        visa_root, verified_official_release_sha256=None, workers=workers)
    realiad_schema = validate_identity_manifest(realiad)
    visa_schema = validate_identity_manifest(visa)
    realiad_sha = write_manifest_exclusive(output_dir / "realiad_identity_manifest.private.json", realiad)
    visa_sha = write_manifest_exclusive(output_dir / "visa_1cls_identity_manifest.private.json", visa)
    receipt = {
        "schema": "final-benchmark.identity-census-receipt.v1",
        "created_unix_seconds": int(started),
        "branch": branch, "source_commit": revision,
        "frozen_gate0_commit": "a34b870fd4700bfe970e459d2963a602f04b1341",
        "realiad": {
            "manifest_sha256": realiad_sha,
            "local_snapshot_sha256": realiad["local_snapshot_sha256"],
            "identity_census_status": realiad["identity_census_status"],
            "role_manifest_status": realiad["role_manifest_status"],
            "official_provenance_status": realiad["official_provenance_status"],
            "category_count": realiad["category_count"],
            "category_error_count": len(realiad["category_errors"]),
            "schema_validation": realiad_schema,
        },
        "visa_1cls": {
            "manifest_sha256": visa_sha,
            "local_snapshot_sha256": visa["local_snapshot_sha256"],
            "identity_census_status": visa["identity_census_status"],
            "role_manifest_status": visa["role_manifest_status"],
            "official_provenance_status": visa["official_provenance_status"],
            "category_count": visa["category_count"],
            "official_group_key_columns": visa["official_group_key_columns"],
            "schema_validation": visa_schema,
        },
        "test_outcome_access": "NOT_ACCESSED",
        "mask_access": "NOT_ACCESSED",
        "image_decoding_or_inference": "NOT_PERFORMED",
        "external_official_release_receipt": "NOT_PRESENT; role/event hashes withheld",
    }
    receipt_payload = _json_bytes(receipt) + b"\n"
    receipt_path = output_dir / "identity_census_receipt.json"
    with receipt_path.open("xb") as stream:
        stream.write(receipt_payload)
        stream.flush()
        os.fsync(stream.fileno())
    receipt_sha = hashlib.sha256(receipt_payload).hexdigest()
    with (output_dir / "identity_census_receipt.sha256").open("x", encoding="ascii") as stream:
        stream.write(receipt_sha + "  identity_census_receipt.json\n")
        stream.flush()
        os.fsync(stream.fileno())
    receipt["receipt_sha256"] = receipt_sha
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--realiad-root", type=Path, default=DEFAULT_REALIAD_ROOT)
    parser.add_argument("--realiad-json-root", type=Path, default=DEFAULT_REALIAD_JSON_ROOT)
    parser.add_argument("--visa-root", type=Path, default=DEFAULT_VISA_ROOT)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    receipt = build_all(realiad_root=args.realiad_root,
                        realiad_json_root=args.realiad_json_root,
                        visa_root=args.visa_root, output_dir=args.output_dir,
                        workers=args.workers)
    # Only sanitized counts/status and hashes are written to stdout.
    print(json.dumps(receipt, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
