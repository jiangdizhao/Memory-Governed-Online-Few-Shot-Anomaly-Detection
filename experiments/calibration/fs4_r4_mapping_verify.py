"""Post-seal helper: project labels from a private mapping after validation."""
from __future__ import annotations

import json
from pathlib import Path
import sys

from .fs4_durable_ledger import canonical, digest
from .fs4_r2_security import WindowsPrivateStore
from .fs4_r4_broker import verified_mapping


def project(private_root: Path, run_root: Path, expected_manifest_sha256: str) -> dict:
    # Independently require a valid global Stage A seal before opening mapping.
    from .fs4_research_stage_a_verify import verify_stage_a
    sealed = verify_stage_a(run_root)
    if sealed["manifest_sha256"] != expected_manifest_sha256:
        raise ValueError("sealed manifest differs")
    manifest = json.loads(WindowsPrivateStore(run_root, create=False).read("manifest.json"))
    if digest(manifest) != expected_manifest_sha256:
        raise ValueError("sealed manifest differs")
    execution = manifest["research_execution_manifest"]
    if execution.get("schema") != "FS4_R4_BLINDED_EXECUTION_V1":
        raise ValueError("blinded registration required")
    mapping = verified_mapping(private_root, execution)
    return {"schema": "FS4_R4_OUTCOME_PROJECTION_V1", "manifest_sha256": expected_manifest_sha256,
            "source_mapping_sha256": execution["source_mapping_sha256"],
            "labels": {event: row["label"] for event, row in mapping.items()}}


if __name__ == "__main__":
    if len(sys.argv) != 4:
        raise SystemExit("expected private root, run root and sealed manifest digest")
    sys.stdout.buffer.write(canonical(project(Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3])))
