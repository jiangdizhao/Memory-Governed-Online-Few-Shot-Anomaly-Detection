"""Independent CPU-only Stage B evaluator for trusted local research.

This module does not import or instantiate online controllers or memory. It
first launches a read-only Stage A verifier, and only then reads authorized
outcome labels. Same-account process separation is not adversarial isolation.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

from .fs4_r2_security import WindowsPrivateStore


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def _sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def evaluate(*, run_root: Path, grant_path: Path, labels_root: Path, labels_name: str, result_root: Path) -> dict:
    """Validate sealed Stage A before opening the authorized outcome projection."""
    run_root, grant_path, labels_root, result_root = map(Path, (run_root, grant_path, labels_root, result_root))
    if any(not p.is_absolute() for p in (run_root, grant_path, labels_root, result_root)):
        raise ValueError("evaluation paths must be absolute")
    if any(p == run_root or p.is_relative_to(run_root) or run_root.is_relative_to(p) for p in (labels_root, result_root)):
        raise ValueError("outcome and result locations must be outside Stage A storage")
    if labels_root == result_root or labels_root.is_relative_to(result_root) or result_root.is_relative_to(labels_root):
        raise ValueError("result and outcome stores must be separate")
    # No label file is opened before successful validation of the complete journal.
    verified = subprocess.run([sys.executable, "-m", "experiments.calibration.fs4_research_stage_a_verify", str(run_root)],
                              capture_output=True, check=True)
    stage_a = json.loads(verified.stdout)
    grant = json.loads(grant_path.read_bytes())
    keys = {"schema", "scope", "run_id", "manifest_sha256", "source_commit", "model_state_sha256",
            "dataset_identity_sha256", "fixed_support_sha256_by_category", "stream_sha256_by_category",
            "stage_a_sha256", "labels_sha256"}
    if set(grant) != keys or grant["schema"] != "FS4_LOCAL_RESEARCH_STAGE_B_GRANT_V1" or grant["scope"] != "AGGREGATE_METRICS":
        raise ValueError("invalid research evaluation authorization")
    for key in keys - {"schema", "scope", "labels_sha256"}:
        if grant[key] != stage_a[key]:
            raise ValueError(f"Stage B grant {key} does not match sealed Stage A")
    label_data = WindowsPrivateStore(labels_root, create=False).read(labels_name)
    if _sha(label_data) != grant["labels_sha256"]:
        raise ValueError("authorized outcome projection checksum differs")
    labels = json.loads(label_data)
    events = stage_a["events"]
    if (not isinstance(labels, dict) or set(labels) != {row["event"] for row in events} or
            any(isinstance(y, bool) or y not in (0, 1) for y in labels.values())):
        raise ValueError("outcome projection must contain exactly one binary label per registered event")
    if any(labels[event] != value for event, value in stage_a["queried_feedback"].items()):
        raise ValueError("outcome projection disagrees with verified queried feedback")
    metrics: dict[str, dict[str, dict[str, int]]] = {}
    for row in events:
        category = metrics.setdefault(row["category"], {})
        label = labels[row["event"]]
        for name, threshold in row["thresholds"].items():
            predicted = int(row["scores"][name.rsplit("_", 1)[1]] >= threshold)
            counts = category.setdefault(name, {"tp": 0, "tn": 0, "fp": 0, "fn": 0})
            key = ("tp" if predicted else "fn") if label else ("fp" if predicted else "tn")
            counts[key] += 1
    for category in metrics.values():
        for counts in category.values():
            normal, defect = counts["tn"] + counts["fp"], counts["tp"] + counts["fn"]
            counts["fpr"] = counts["fp"] / normal if normal else None
            counts["fnr"] = counts["fn"] / defect if defect else None
            counts["balanced_accuracy"] = (1 - (counts["fpr"] + counts["fnr"]) / 2
                                           if normal and defect else None)
    result = {"schema": "FS4_LOCAL_RESEARCH_STAGE_B_RESULT_V1", "run_id": grant["run_id"],
              "stage_a_sha256": stage_a["stage_a_sha256"], "grant_sha256": _sha(_canonical(grant)),
              "scope": grant["scope"], "metrics": metrics}
    result_data = _canonical(result) + b"\n"
    store = WindowsPrivateStore(result_root, create=True)
    store.write("INCOMPLETE.json", _canonical({"status": "INCOMPLETE"}) + b"\n")
    store.write("result.json", result_data)
    seal = {"schema": "FS4_LOCAL_RESEARCH_STAGE_B_SEAL_V1", "stage_a_sha256": stage_a["stage_a_sha256"],
            "result_sha256": _sha(result_data), "result_size": len(result_data)}
    seal_data = _canonical(seal) + b"\n"
    store.write("stage_b_seal.json", seal_data)
    store.write("COMPLETE.json", _canonical({"schema": "FS4_LOCAL_RESEARCH_COMPLETE_V1",
        "stage_b_seal_sha256": _sha(seal_data), "result_sha256": _sha(result_data)}) + b"\n")
    return verify_completion(result_root)


def verify_completion(result_root: Path) -> dict:
    store = WindowsPrivateStore(Path(result_root), create=False)
    result_bytes, seal_bytes, complete_bytes = (store.read(name) for name in ("result.json", "stage_b_seal.json", "COMPLETE.json"))
    try:
        result, seal, complete = map(json.loads, (result_bytes, seal_bytes, complete_bytes))
    except (ValueError, TypeError) as exc:
        raise ValueError("research Stage B completion integrity check failed") from exc
    if (seal.get("schema") != "FS4_LOCAL_RESEARCH_STAGE_B_SEAL_V1" or
            complete.get("schema") != "FS4_LOCAL_RESEARCH_COMPLETE_V1" or
            seal.get("result_sha256") != _sha(result_bytes) or seal.get("result_size") != len(result_bytes) or
            seal.get("stage_a_sha256") != result.get("stage_a_sha256") or
            complete.get("result_sha256") != _sha(result_bytes) or complete.get("stage_b_seal_sha256") != _sha(seal_bytes)):
        raise ValueError("research Stage B completion integrity check failed")
    return result


if __name__ == "__main__":
    if len(sys.argv) != 6:
        raise SystemExit("usage: evaluator RUN_ROOT GRANT LABELS_ROOT LABELS_NAME RESULT_ROOT")
    value = evaluate(run_root=Path(sys.argv[1]), grant_path=Path(sys.argv[2]), labels_root=Path(sys.argv[3]),
                     labels_name=sys.argv[4], result_root=Path(sys.argv[5]))
    sys.stdout.buffer.write(_canonical(value))
