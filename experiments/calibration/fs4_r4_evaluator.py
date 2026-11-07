"""CPU-only R4 evaluation; no online controller or memory import."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

from .fs4_r2_security import WindowsPrivateStore
from .fs4_research_evaluator import _canonical, verify_completion


def evaluate_blinded(*, run_root: Path, private_root: Path, grant_path: Path, result_root: Path) -> dict[str, Any]:
    run_root, private_root, grant_path, result_root = map(Path, (run_root, private_root, grant_path, result_root))
    if any(not p.is_absolute() for p in (run_root, private_root, grant_path, result_root)):
        raise ValueError("evaluation paths must be absolute")
    if any(a == b or a.is_relative_to(b) or b.is_relative_to(a) for a, b in
           ((run_root, private_root), (run_root, result_root), (private_root, result_root))):
        raise ValueError("Stage A, private mapping and results must be separate")
    # This must finish successfully before the private mapping process starts.
    verified = subprocess.run([sys.executable, "-m", "experiments.calibration.fs4_research_stage_a_verify", str(run_root)],
                              capture_output=True, check=True)
    stage_a = json.loads(verified.stdout)
    grant = json.loads(grant_path.read_bytes())
    keys = {"schema", "scope", "run_id", "manifest_sha256", "source_commit", "model_state_sha256",
            "dataset_identity_sha256", "fixed_support_sha256_by_category", "stream_sha256_by_category",
            "stage_a_sha256", "source_mapping_sha256"}
    if set(grant) != keys or grant["schema"] != "FS4_R4_STAGE_B_GRANT_V1" or grant["scope"] != "AGGREGATE_METRICS":
        raise ValueError("invalid R4 Stage B authorization")
    for key in keys - {"schema", "scope", "source_mapping_sha256"}:
        if grant[key] != stage_a[key]: raise ValueError("Stage B authorization differs from sealed Stage A")
    projected = subprocess.run([sys.executable, "-m", "experiments.calibration.fs4_r4_mapping_verify",
                                str(private_root), str(run_root), stage_a["manifest_sha256"]], capture_output=True, check=True)
    projection = json.loads(projected.stdout)
    if projection["source_mapping_sha256"] != grant["source_mapping_sha256"]:
        raise ValueError("Stage B mapping commitment differs")
    labels = projection["labels"]
    if set(labels) != {row["event"] for row in stage_a["events"]}:
        raise ValueError("Stage B projection differs from registered stream")
    if any(labels[event] != label for event, label in stage_a["queried_feedback"].items()):
        raise ValueError("Stage B outcomes disagree with queried feedback")
    metrics: dict[str, Any] = {}
    for row in stage_a["events"]:
        category = metrics.setdefault(row["category"], {})
        for name, threshold in row["thresholds"].items():
            predicted = int(row["scores"][name.rsplit("_", 1)[1]] >= threshold)
            label = labels[row["event"]]
            counts = category.setdefault(name, {"tp": 0, "tn": 0, "fp": 0, "fn": 0})
            counts[("tp" if predicted else "fn") if label else ("fp" if predicted else "tn")] += 1
    for category in metrics.values():
        for counts in category.values():
            normal, defect = counts["tn"] + counts["fp"], counts["tp"] + counts["fn"]
            counts["fpr"] = counts["fp"] / normal if normal else None
            counts["fnr"] = counts["fn"] / defect if defect else None
            counts["balanced_accuracy"] = 1 - (counts["fpr"] + counts["fnr"]) / 2 if normal and defect else None
    sha = lambda value: hashlib.sha256(value).hexdigest()
    projection_bytes = _canonical(projection) + b"\n"
    result = {"schema": "FS4_LOCAL_RESEARCH_STAGE_B_RESULT_V1", "run_id": grant["run_id"],
              "stage_a_sha256": stage_a["stage_a_sha256"], "grant_sha256": sha(_canonical(grant)),
              "scope": grant["scope"], "projection_sha256": sha(projection_bytes), "metrics": metrics}
    result_bytes = _canonical(result) + b"\n"
    store = WindowsPrivateStore(result_root, create=True)
    store.write("INCOMPLETE.json", _canonical({"status": "INCOMPLETE"}) + b"\n")
    store.write("outcome_projection.json", projection_bytes)
    store.write("result.json", result_bytes)
    seal = {"schema": "FS4_LOCAL_RESEARCH_STAGE_B_SEAL_V1", "stage_a_sha256": stage_a["stage_a_sha256"],
            "result_sha256": sha(result_bytes), "result_size": len(result_bytes)}
    seal_bytes = _canonical(seal) + b"\n"
    store.write("stage_b_seal.json", seal_bytes)
    store.write("COMPLETE.json", _canonical({"schema": "FS4_LOCAL_RESEARCH_COMPLETE_V1",
        "stage_b_seal_sha256": sha(seal_bytes), "result_sha256": sha(result_bytes)}) + b"\n")
    return verify_blinded_completion(result_root)


def verify_blinded_completion(result_root: Path) -> dict[str, Any]:
    """Recheck the separate completion seal and its private outcome projection."""
    result = verify_completion(result_root)
    data = WindowsPrivateStore(Path(result_root), create=False).read("outcome_projection.json")
    if hashlib.sha256(data).hexdigest() != result.get("projection_sha256"):
        raise ValueError("Stage B projection checksum differs")
    return result


if __name__ == "__main__":
    if len(sys.argv) != 5:
        raise SystemExit("expected Stage A root, private mapping root, grant and result root")
    value = evaluate_blinded(run_root=Path(sys.argv[1]), private_root=Path(sys.argv[2]),
                             grant_path=Path(sys.argv[3]), result_root=Path(sys.argv[4]))
    sys.stdout.buffer.write(_canonical(value))
