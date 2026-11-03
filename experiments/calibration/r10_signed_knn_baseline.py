"""R10: one fixed four-class comparison of two historical Signed kNN capacities.

The module has no import-time experiment side effects. GPU work is only entered
through a Windows Job Object child launched by the explicit run command.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
from typing import Any, Mapping, Sequence

os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
sys.dont_write_bytecode = True

ROOT = Path(__file__).resolve().parents[2]
AUTHORIZED_CUDA_VISIBILITY = os.environ.get("CUDA_VISIBLE_DEVICES")
EXPECTED_BRANCH = "calibration-stage0-contract-20261029"
EXPECTED_PARENT = "22e6948915150e9e1b6492cca36b8385633b9ebf"
RUN_ID = "r10_signed_knn_baseline_20261103_a1"
R9_RUN_ID = "r9_correction_mechanism_20261103_a9"
R9_ROOT = ROOT / "outputs" / "calibration" / R9_RUN_ID
RUN_ROOT = ROOT / "outputs" / "calibration" / RUN_ID
REPORT_DIR = ROOT / "docs" / "calibration" / "audit" / "r10_signed_knn_baseline"
REPORT_PATH = REPORT_DIR / "REPORT.md"
AGGREGATE_PATH = REPORT_DIR / "aggregate_results.json"
PYTHON = Path(r"D:\anaconda3\envs\torch\python.exe")
CATEGORIES = ("audiojack", "regulator", "switch", "rolled_strip_base")
EXPECTED_COUNTS = {
    "audiojack": 3815, "regulator": 3855, "switch": 3820,
    "rolled_strip_base": 3805,
}
EXPECTED_QUERIES = {
    "audiojack": 114, "regulator": 116, "switch": 115,
    "rolled_strip_base": 114,
}
CAPACITIES = (256, 1024)
K_NEIGHBORS = 5
AMPLITUDE = 0.02
TEST_SMOKE_EVENTS = 16
SMOKE_DEADLINE_SECONDS = 600
FULL_DEADLINE_SECONDS = 3600
OUTPUT_LIMIT_BYTES = 4 * 1024**3
FAILURE_RESERVE_BYTES = 64 * 1024
MIN_FREE_BYTES = 10 * 1024**3
ATOL = 1e-5
RTOL = 1e-4
BOOTSTRAP_SEED = 20261103
SCORE_SCHEMA = "r10.signed_knn.score.v1"
RUN_SCHEMA = "r10.signed_knn.run.v1"
STAGE_A_SCHEMA = "r10.signed_knn.stage_a_seal.v1"
STAGE_B_SCHEMA = "r10.signed_knn.stage_b_seal.v1"
AUTHORIZATION = "R10 targeted retrospective development Signed kNN comparison"

SOURCE_PATHS = (
    "experiments/calibration/r1_matched_correction_pilot.py",
    "experiments/calibration/r1_runtime_supervisor.py",
    "experiments/calibration/r2_feedback_repair.py",
    "experiments/calibration/r3_fullstream_2x2.py",
    "experiments/calibration/r6_capacity1024_audiojack_runner.py",
    "experiments/calibration/r7_shuffled_correction_4class.py",
    "experiments/calibration/r9_correction_mechanism_falsification.py",
    "experiments/calibration/controllers.py",
    "experiments/calibration/protocol.py",
    "experiments/calibration/native.py",
    "experiments/calibration/profiles.py",
    "experiments/calibration/stage1a_p12_label_provider.py",
    "experiments/risk_safe/native_bridge.py",
    "experiments/p0_protocol_audit.py",
    "experiments/v02_memory.py",
    "vmb_visionad_new.py",
    "test_phase3_AIF_simplified_v22.py",
)


class R10Error(RuntimeError):
    pass


def _canonical(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"),
                       ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")


def _sha_json(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise R10Error(f"Expected a JSON object at {path}")
    return value


def _reject_reparse_chain(path: Path, stop: Path | None = None) -> None:
    from experiments.calibration.r2_feedback_repair import _is_reparse
    current = Path(os.path.abspath(path))
    boundary = Path(os.path.abspath(stop)) if stop is not None else None
    while True:
        if current.exists() and _is_reparse(current):
            raise R10Error(f"Symlink/reparse path is forbidden: {current}")
        if boundary is not None and current == boundary:
            return
        parent = current.parent
        if parent == current:
            if boundary is not None:
                raise R10Error("Output path did not reach its authorized root")
            return
        if boundary is not None and parent != boundary and boundary not in parent.parents:
            raise R10Error("Output path escaped its authorized root")
        current = parent


def _assert_repo_identity() -> None:
    branch = subprocess.run(["git", "branch", "--show-current"], cwd=ROOT,
                            check=True, capture_output=True, text=True).stdout.strip()
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                          check=True, capture_output=True, text=True).stdout.strip()
    if branch != EXPECTED_BRANCH or head != EXPECTED_PARENT:
        raise R10Error(f"Repository identity mismatch: branch={branch}, HEAD={head}")
    for args in (["git", "diff", "--quiet", "HEAD", "--"],
                 ["git", "diff", "--cached", "--quiet"]):
        result = subprocess.run(args, cwd=ROOT)
        if result.returncode != 0:
            raise R10Error("Tracked or staged source changes exist outside the R10 deliverables")


def _source_hashes() -> dict[str, str]:
    paths = ("experiments/calibration/r10_signed_knn_baseline.py", *SOURCE_PATHS)
    result: dict[str, str] = {}
    for relative in paths:
        path = ROOT / relative
        if not path.is_file():
            raise R10Error(f"Pinned source is missing: {relative}")
        result[relative] = _sha_file(path)
    return result


def _verify_source_hashes(expected: Mapping[str, str]) -> None:
    actual = _source_hashes()
    if actual != dict(expected):
        raise R10Error("R10 runner or dependency source changed after identity was sealed")


def _verify_disk_and_gpu_preflight(path: Path) -> dict[str, Any]:
    if not path.is_dir():
        raise R10Error("Output-volume free-space check requires an existing directory")
    _reject_reparse_chain(path)
    free_bytes = int(shutil.disk_usage(path).free)
    if free_bytes < MIN_FREE_BYTES:
        raise R10Error(f"R10 requires at least {MIN_FREE_BYTES} bytes free")
    command = ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"]
    result = subprocess.run(command, capture_output=True, text=True, timeout=20)
    if result.returncode != 0:
        raise R10Error("Could not verify that no competing GPU workload is active")
    pids = []
    for line in result.stdout.splitlines():
        value = line.strip()
        if value.isdigit():
            pids.append(int(value))
    own_pid = os.getpid()
    foreign = sorted(set(pid for pid in pids if pid != own_pid))
    names = []
    if foreign:
        try:
            import psutil
            for pid in foreign:
                proc = psutil.Process(pid)
                name = proc.name().lower()
                if name in {"python.exe", "pythonw.exe", "python"}:
                    raise R10Error(f"Competing Python GPU process is active (PID {pid})")
                names.append({"pid": pid, "name": proc.name()})
        except R10Error:
            raise
        except Exception as exc:
            raise R10Error("Could not identify active GPU process owners") from exc
        # WDDM registers the desktop compositor and GUI applications as GPU
        # clients. They are recorded; a Python/scientific worker blocks launch.
    return {"free_bytes": free_bytes, "nvidia_smi_process_ids": foreign,
            "identified_gpu_clients": names,
            "scientific_python_gpu_worker_detected": False}


class SignedKNNFIFO:
    """One bounded global FIFO with historical C2 float32 cosine voting."""

    def __init__(self, capacity: int, feature_dim: int = 4096):
        import numpy as np
        if int(capacity) not in CAPACITIES or int(feature_dim) != 4096:
            raise ValueError("R10 requires capacity 256 or 1024 and feature dimension 4096")
        self.capacity = int(capacity)
        self.feature_dim = int(feature_dim)
        self.features = np.zeros((self.capacity, self.feature_dim), dtype=np.float32)
        self.signs = np.zeros((self.capacity,), dtype=np.int8)
        self.positions = np.full((self.capacity,), -1, dtype=np.int64)
        self.source_groups = np.zeros((self.capacity, 32), dtype=np.uint8)
        self.head = 0
        self.count = 0
        self.insertions = 0
        self.evictions = 0
        self.first_saturation_position: int | None = None
        self.total_source_exclusions = 0

    @property
    def array_bytes(self) -> int:
        return int(self.features.nbytes + self.signs.nbytes +
                   self.positions.nbytes + self.source_groups.nbytes)

    def _active_indices(self):
        import numpy as np
        return (self.head + np.arange(self.count, dtype=np.int64)) % self.capacity

    def score(self, query_rows: Any, *, event_position: int,
              event_group_sha256: str, neighbors: int = K_NEIGHBORS,
              amplitude: float = AMPLITUDE) -> tuple[Any, dict[str, int]]:
        import numpy as np
        q = np.asarray(query_rows, dtype=np.float32)
        if (q.ndim != 2 or q.shape[1] != self.feature_dim or
                not np.isfinite(q).all() or int(neighbors) != K_NEIGHBORS or
                float(amplitude) != AMPLITUDE):
            raise ValueError("Invalid R10 query rows or unregistered k/amplitude")
        if len(event_group_sha256) != 64:
            raise ValueError("Source Group identity must be a SHA-256 digest")
        group = np.frombuffer(bytes.fromhex(event_group_sha256), dtype=np.uint8)
        active = self._active_indices()
        if not len(active):
            return np.zeros((len(q),), dtype=np.float32), {
                "eligible_anchor_count": 0, "source_excluded_anchor_count": 0}
        prior = self.positions[active] < int(event_position)
        different_group = np.any(self.source_groups[active] != group[None, :], axis=1)
        keep = prior & different_group
        eligible = active[keep]
        excluded = int(np.count_nonzero(~different_group))
        if np.any(self.positions[active[~prior]] >= int(event_position)):
            # Current and future anchors exist only in corrupt synthetic state.
            raise ValueError("FIFO contains a current or future anchor")
        self.total_source_exclusions += excluded
        if not len(eligible):
            return np.zeros((len(q),), dtype=np.float32), {
                "eligible_anchor_count": 0,
                "source_excluded_anchor_count": excluded}
        bank = self.features[eligible]
        signs = self.signs[eligible].astype(np.float32, copy=False)
        qn = q / np.maximum(np.linalg.norm(q, axis=1, keepdims=True), 1e-12)
        pn = bank / np.maximum(np.linalg.norm(bank, axis=1, keepdims=True), 1e-12)
        distances = 1.0 - np.clip(qn @ pn.T, -1.0, 1.0)
        result = np.zeros((len(q),), dtype=np.float32)
        k = min(int(neighbors), len(eligible))
        anchor_order = np.arange(len(eligible), dtype=np.int64)
        for row in range(len(q)):
            nearest = np.lexsort((anchor_order, distances[row]))[:k]
            result[row] = np.float32(
                float(amplitude) * float(signs[nearest].mean()))
        return result, {"eligible_anchor_count": int(len(eligible)),
                        "source_excluded_anchor_count": excluded}

    def insert(self, features: Any, *, sign: int, source_group_sha256: str,
               position: int) -> dict[str, int]:
        import numpy as np
        values = np.asarray(features, dtype=np.float32)
        if (values.ndim != 2 or values.shape[1] != self.feature_dim or
                not np.isfinite(values).all() or int(sign) not in (-1, 1) or
                int(position) < 0 or len(source_group_sha256) != 64):
            raise ValueError("Malformed R10 FIFO insertion")
        group = np.frombuffer(bytes.fromhex(source_group_sha256), dtype=np.uint8)
        evicted = 0
        inserted = 0
        for feature in values:
            if self.count < self.capacity:
                slot = (self.head + self.count) % self.capacity
                self.count += 1
            else:
                slot = self.head
                self.head = (self.head + 1) % self.capacity
                self.evictions += 1
                evicted += 1
            self.features[slot] = feature
            self.signs[slot] = int(sign)
            self.positions[slot] = int(position)
            self.source_groups[slot] = group
            self.insertions += 1
            inserted += 1
            if self.count == self.capacity and self.first_saturation_position is None:
                self.first_saturation_position = int(position)
        return {"inserted": inserted, "evicted": evicted}

    def sign_counts(self) -> dict[str, int]:
        import numpy as np
        values = self.signs[self._active_indices()]
        return {"positive": int(np.count_nonzero(values == 1)),
                "negative": int(np.count_nonzero(values == -1))}

    def state_sha256(self) -> str:
        import numpy as np
        indices = self._active_indices()
        digest = hashlib.sha256()
        digest.update(_canonical({"capacity": self.capacity,
                                  "feature_dim": self.feature_dim,
                                  "count": self.count}))
        for array in (self.features[indices], self.signs[indices],
                      self.positions[indices], self.source_groups[indices]):
            value = np.ascontiguousarray(array)
            digest.update(str((tuple(value.shape), value.dtype.str)).encode("ascii"))
            digest.update(value.tobytes())
        return digest.hexdigest()


def _corrected_c2_score(base_map: Any, top_ids: Any, selected_features: Any,
                        memory: SignedKNNFIFO, *, position: int,
                        group_sha256: str) -> tuple[float, Any, dict[str, Any]]:
    import numpy as np
    from experiments.calibration import r1_matched_correction_pilot as r1
    delta, diagnostics = memory.score(
        selected_features, event_position=position,
        event_group_sha256=group_sha256)
    corrected = r1.corrected_dense_map(base_map, top_ids, delta)
    score = r1.max_score(corrected)
    return score, delta, {
        **diagnostics,
        "corrected_patch_count": int(np.count_nonzero(delta)),
        "patch_count": int(delta.size),
        "absolute_delta_mean": float(np.mean(np.abs(delta))) if delta.size else 0.0,
        "absolute_delta_max": float(np.max(np.abs(delta))) if delta.size else 0.0,
        "score_change_from_g": float(score - r1.max_score(base_map)),
    }


def _load_r9_reference(contract: Mapping[str, Any] | None = None,
                       categories: Sequence[str] | None = None) -> dict[str, Any]:
    from experiments.calibration import r7_shuffled_correction_4class as r7
    identity_manifest_path = R9_ROOT / "manifest" / "run_manifest.json"
    if not identity_manifest_path.is_file():
        raise R10Error("BLOCKED_REFERENCE_INPUT: R9 identity manifest is missing")
    manifest = _read_json(identity_manifest_path)
    identity = manifest.get("identity", {})
    if (manifest.get("run_id") != R9_RUN_ID or
            identity.get("source_parent_commit") != "812e85407adb200b47e48ee811824e9394ed7416" or
            identity.get("identity_sha256") != manifest.get("identity_sha256") or
            identity.get("identity_sha256") != _sha_json(
                {k: v for k, v in identity.items() if k != "identity_sha256"})):
        raise R10Error("BLOCKED_REFERENCE_INPUT: R9 identity manifest is not authentic")
    for relative, expected in identity.get("source_fingerprints", {}).items():
        path = ROOT / relative
        if not path.is_file() or _sha_file(path) != expected:
            raise R10Error(f"BLOCKED_REFERENCE_INPUT: R9 source fingerprint changed: {relative}")
    stage_a_path = R9_ROOT / "full" / "stage_seals" / "stage_a_seal.json"
    stage_a = _read_json(stage_a_path)
    if (stage_a.get("status") != "SEALED_COMPLETE" or
            stage_a.get("run_id") != R9_RUN_ID or
            stage_a.get("identity_sha256") != identity["identity_sha256"] or
            stage_a.get("condition_count") != 16 or
            stage_a.get("event_count") != 61180 or
            stage_a.get("query_count") != 1836 or
            stage_a.get("all_shuffle0_support0_r7_r8_score_query_decision_parity_verified") is not True or
            stage_a.get("unqueried_test_labels_accessed") is not False or
            stage_a.get("test_masks_opened") is not False):
        raise R10Error("BLOCKED_REFERENCE_INPUT: R9 Stage A seal is not complete")
    refs: dict[str, Any] = {
        "manifest": manifest, "identity": identity,
        "stage_a_seal": stage_a, "references": {},
        "seal_hashes": {
            "stage_a": _sha_file(stage_a_path),
        },
    }
    if contract is None:
        contract = r7._frozen_contract()
    refs["frozen_contract"] = contract
    requested = tuple(CATEGORIES if categories is None else categories)
    if not requested or any(category not in CATEGORIES for category in requested):
        raise R10Error("R10 reference request contains an unregistered category")
    for category in requested:
        info = contract["metadata"][category]
        order_info = identity["classes"][category]["orders"]["SHUFFLE0"]
        output = R9_ROOT / "full" / f"{category.lower()}_shuffle0"
        # R9 condition directories use the exact category identifier.
        if not output.is_dir():
            output = R9_ROOT / "full" / f"{category}_shuffle0"
        scores_path = output / "stage_a_scores.jsonl"
        seal_path = output / "stage_a_seal.json"
        condition_manifest = _read_json(output / "run_manifest.json")
        completion = _read_json(seal_path)
        global_condition = stage_a.get("conditions", {}).get(f"{category}/SHUFFLE0", {})
        if (not scores_path.is_file() or condition_manifest.get("run_id") != R9_RUN_ID or
                condition_manifest.get("identity_sha256") != identity["identity_sha256"] or
                completion.get("status") != "SEALED_COMPLETE" or
                completion.get("category") != category or completion.get("order") != "SHUFFLE0" or
                completion.get("event_count") != EXPECTED_COUNTS[category] or
                completion.get("query_count") != EXPECTED_QUERIES[category] or
                completion.get("stream_image_encodings") != EXPECTED_COUNTS[category] or
                completion.get("unqueried_test_labels_accessed") is not False or
                completion.get("offline_outcome_vectors_accessed") is not False or
                completion.get("test_masks_opened") is not False or
                completion.get("score_rows_sha256") != _sha_file(scores_path) or
                global_condition.get("score_rows_sha256") != _sha_file(scores_path)):
            raise R10Error(f"BLOCKED_REFERENCE_INPUT: R9 {category} condition receipt failed")
        if (completion.get("fixed_support_sha256") !=
                info["fixed_support_sha256_expected"] or
                completion.get("c64_cache_receipt", {}).get("c64_group_sha256") !=
                info["frozen_info"]["c64"]["group_sha256"]):
            raise R10Error(f"BLOCKED_REFERENCE_INPUT: {category} support/C64 identities differ")
        rows: list[dict[str, Any]] = []
        with scores_path.open("r", encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, 1):
                if not line.strip():
                    raise R10Error(f"R9 score tape has an empty row at {category}:{line_number}")
                row = json.loads(line)
                if any(key in row for key in ("label", "anomaly_class", "mask_path", "ground_truth")):
                    raise R10Error("R9 Stage A score tape contains a forbidden outcome field")
                rows.append(row)
        if len(rows) != EXPECTED_COUNTS[category]:
            raise R10Error(f"R9 score count mismatch for {category}")
        query_positions = tuple(int(v) for v in order_info["query_positions"])
        actual_query_positions = tuple(
            int(row["position"]) for row in rows if row.get("query_tape_decision") is True)
        if (actual_query_positions != query_positions or
                len(query_positions) != EXPECTED_QUERIES[category] or
                completion.get("query_positions") != list(query_positions) or
                order_info.get("query_position_sha256") != _sha_json(list(query_positions))):
            raise R10Error(f"R9 query position mismatch for {category}")
        query_identity_set_sha = identity["classes"][category]["r7_query_identity_set_sha256"]
        events = r9_events = __import__(
            "experiments.calibration.r9_correction_mechanism_falsification",
            fromlist=["make_order_events"]).make_order_events(info["events"], "SHUFFLE0")
        if len(events) != EXPECTED_COUNTS[category]:
            raise R10Error(f"Frozen event count mismatch for {category}")
        for position, (row, event) in enumerate(zip(rows, events)):
            if (row.get("position") != position or row.get("order") != "SHUFFLE0" or
                    row.get("category") != category or
                    row.get("original_json_position") != int(event.original_position) or
                    row.get("event_id_sha256") != str(event.event_id_sha256) or
                    row.get("image_identity_sha256") != str(event.image_identity_sha256) or
                    row.get("source_group_sha256") != str(event.source_group_sha256) or
                    int(row.get("camera_id", -1)) != int(event.camera_id) or
                    row.get("query_identity_set_sha256") != query_identity_set_sha or
                    row.get("structured_fixed_support_sha256") != info["fixed_support_sha256_expected"] or
                    set(row.get("scores", {})) < {"G1024", "GC1024"} or
                    set(row.get("decisions", {})) < {
                        "G1024-Frozen", "G1024-B1", "G1024-B3",
                        "GC1024-Frozen", "GC1024-B1", "GC1024-B3"} or
                    not re.fullmatch(r"[0-9a-f]{64}",
                                     str(row.get("score_state_sha256", {}).get("G1024", "")))):
                raise R10Error(f"R9 exact identity/score receipt mismatch at {category}:{position}")
            for scorer in ("G1024", "GC1024"):
                score = float(row["scores"][scorer])
                if not math.isfinite(score):
                    raise R10Error(f"R9 {scorer} score is nonfinite at {category}:{position}")
                for controller in ("Frozen", "B1", "B3"):
                    decision = row["decisions"][f"{scorer}-{controller}"]
                    if (int(decision.get("prediction", -1)) not in (0, 1) or
                            not math.isfinite(float(decision.get("threshold", math.nan))) or
                            int(decision["prediction"]) !=
                            int(score >= float(decision["threshold"]))):
                        raise R10Error(
                            f"R9 {scorer}-{controller} decision fails its sealed threshold")
            if bool(row.get("query_tape_decision")) != (position in set(query_positions)):
                raise R10Error(f"R9 query decision differs at {category}:{position}")
        refs["references"][category] = {
            "output": output, "scores_path": scores_path, "completion": completion,
            "score_sha256": _sha_file(scores_path), "rows": rows,
            "events": events, "query_positions": query_positions,
            "query_identity_set_sha256":
                identity["classes"][category]["r7_query_identity_set_sha256"],
            "r9_condition_manifest_sha256": _sha_file(output / "run_manifest.json"),
            "r9_condition_seal_sha256": _sha_file(seal_path),
        }
    refs["identity_sha256"] = identity["identity_sha256"]
    refs["source_fingerprints"] = identity["source_fingerprints"]
    return refs


def _r10_identity(refs: Mapping[str, Any], source_hashes: Mapping[str, str],
                  preflight: Mapping[str, Any],
                  model_receipt: Mapping[str, Any]) -> dict[str, Any]:
    contract = refs["frozen_contract"]
    classes = {}
    for category in CATEGORIES:
        info = contract["metadata"][category]
        ref = refs["references"][category]
        classes[category] = {
            "event_count": EXPECTED_COUNTS[category],
            "query_count": EXPECTED_QUERIES[category],
            "query_positions": list(ref["query_positions"]),
            "query_positions_sha256": _sha_json(list(ref["query_positions"])),
            "query_identity_set_sha256": ref["query_identity_set_sha256"],
            "fixed_support_sha256": info["fixed_support_sha256_expected"],
            "support_image_sha256": info["support_image_sha256"],
            "c64_image_sha256": info["frozen_info"]["c64"]["image_sha256"],
            "c64_group_sha256": info["frozen_info"]["c64"]["group_sha256"],
            "category_json_sha256": info["contract"]["category_json_sha256"],
            "r9_score_sha256": ref["score_sha256"],
            "r9_condition_seal_sha256": ref["r9_condition_seal_sha256"],
            "r9_condition_manifest_sha256": ref["r9_condition_manifest_sha256"],
            "identity_map_sha256": info["identity_receipt"]["mapping_sha256"],
        }
    identity = {
        "schema": RUN_SCHEMA,
        "run_id": RUN_ID,
        "authorization": AUTHORIZATION,
        "classification": "RETROSPECTIVE_DEVELOPMENT_DIAGNOSTIC",
        "branch": EXPECTED_BRANCH,
        "source_parent_commit": EXPECTED_PARENT,
        "source_hashes": dict(source_hashes),
        "model_fingerprints": {
            "hub_source_tree_sha256": model_receipt["hub_source_tree_sha256"],
            "checkpoint_sha256": model_receipt["checkpoint_sha256"],
            "expected_model_state_sha256": model_receipt["loaded_model_state_sha256"],
            "verified_receipt_sha256":
                _sha_file(__import__(
                    "experiments.calibration.r3_fullstream_2x2",
                    fromlist=["MODEL_RECEIPT"]).MODEL_RECEIPT / "run_manifest.json"),
        },
        "runtime_profile": dict(model_receipt["runtime_profile"]),
        "configuration": {
            "categories": list(CATEGORIES), "event_counts": dict(EXPECTED_COUNTS),
            "query_counts": dict(EXPECTED_QUERIES), "support_seed": 0,
            "shuffle_seed": 0, "query_set": "exact R9/R8/R7 SHUFFLE0 identity",
            "knn": K_NEIGHBORS, "amplitude": AMPLITUDE,
            "capacities": list(CAPACITIES), "anchor_capacity_policy": "one global FIFO, both signs",
            "feature_contract": {"rows_per_query_image": 32, "dimension": 4096,
                                 "dtype": "float32", "native_l2_normalize": True},
            "source_exclusion": "exact Source Group; same-source anchors excluded",
            "time_eligibility": "strictly prior event position",
            "tie_break": "FIFO anchor order via historical np.lexsort convention",
            "correction_application": "native selected Top-32 patch-map adapter; image max",
            "greedy_capacity": 1024,
            "runtime": {"smoke_deadline_seconds": SMOKE_DEADLINE_SECONDS,
                        "full_child_deadline_seconds": FULL_DEADLINE_SECONDS,
                        "output_limit_bytes_per_leaf": OUTPUT_LIMIT_BYTES,
                        "failure_reserve_bytes": FAILURE_RESERVE_BYTES,
                        "minimum_free_bytes": MIN_FREE_BYTES,
                        "automatic_restart": False},
            "stage_b_only_after_four_stage_a_seals": True,
            "test_masks_opened": False,
        },
        "r9_reference": {
            "run_id": R9_RUN_ID, "identity_sha256": refs["identity_sha256"],
            "seal_hashes": refs["seal_hashes"],
            "event_counts": dict(EXPECTED_COUNTS), "query_counts": dict(EXPECTED_QUERIES),
        },
        "classes": classes,
        "prelaunch_resource_check": dict(preflight),
    }
    identity["identity_sha256"] = _sha_json(identity)
    return identity


def _write_failure(budget: Any, *, category: str, phase: str,
                   exception: BaseException, position: int | None = None) -> None:
    try:
        budget.write_json("failure.json", {
            "schema": "r10.signed_knn.failure.v1",
            "run_id": RUN_ID, "category": category, "phase": phase,
            "position": position, "failure_type": type(exception).__name__,
            "failure_message": str(exception)[:4000],
            "automatic_restart": False,
            "partial_outputs_preserved": True,
            "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }, failure_reserved=True)
    except Exception:
        pass


def _verify_child_environment(category: str, mode: str) -> tuple[Path, dict[str, Any], dict[str, Any]]:
    if category not in CATEGORIES or mode not in {"SMOKE", "FULL"}:
        raise R10Error("Stage A child request is outside the frozen R10 design")
    output = Path(os.environ.get("R10_CHILD_OUTPUT", "")).absolute()
    expected_output = RUN_ROOT / ("smoke" if mode == "SMOKE" else category)
    if output != expected_output or output.exists():
        raise R10Error("Stage A child output path is invalid or already exists")
    _reject_reparse_chain(output, RUN_ROOT)
    global_manifest_path = RUN_ROOT / "global" / "run_manifest.json"
    if _sha_file(global_manifest_path) != os.environ.get("R10_GLOBAL_MANIFEST_SHA256"):
        raise R10Error("Stage A child global manifest digest changed")
    manifest = _read_json(global_manifest_path)
    identity = manifest.get("identity", {})
    if (manifest.get("run_id") != RUN_ID or
            identity.get("identity_sha256") != manifest.get("identity_sha256") or
            identity.get("identity_sha256") != _sha_json(
                {k: v for k, v in identity.items() if k != "identity_sha256"})):
        raise R10Error("Stage A child global identity manifest is malformed")
    source_hashes = identity.get("source_hashes", {})
    if source_hashes.get("experiments/calibration/r10_signed_knn_baseline.py") != \
            os.environ.get("R10_CHILD_SOURCE_SHA256"):
        raise R10Error("Stage A child was bound to an incorrect R10 runner source")
    _verify_source_hashes(source_hashes)
    _assert_repo_identity()
    if os.environ.get("R10_CHILD_SOURCE_SHA256") != _sha_file(Path(__file__).resolve()):
        raise R10Error("Stage A child source digest differs from the authorized R10 identity")
    if (os.environ.get("R10_CUBLAS_WORKSPACE_CONFIG") !=
            identity.get("runtime_profile", {}).get("cublas_workspace_config")):
        raise R10Error("Stage A child runtime initialization profile is incompatible")
    return output, manifest, identity


def _score_c2_c64(impl: Any, bank: Any, items: Sequence[Mapping[str, Any]],
                  base_cache: Sequence[tuple[Any, Any, float]], memory: SignedKNNFIFO,
                  *, event_position: int) -> tuple[list[float], float, int]:
    import numpy as np
    import torch
    from experiments.calibration import r1_matched_correction_pilot as r1
    scores: list[float] = []
    exclusions = 0
    started = time.perf_counter()
    for item, (base_map, top_ids, _base_score) in zip(items, base_cache):
        features = [torch.from_numpy(np.ascontiguousarray(value)).to(bank.device)
                    for value in item["features"]]
        selected = impl.l2_normalize(features[0].to(bank.device))[
            torch.as_tensor(top_ids, device=bank.device, dtype=torch.long)]
        score, _delta, diagnostics = _corrected_c2_score(
            base_map, top_ids, selected.detach().float().cpu().numpy(),
            memory, position=event_position,
            group_sha256=str(item["source_group_sha256"]))
        scores.append(score)
        exclusions += int(diagnostics["source_excluded_anchor_count"])
    torch.cuda.synchronize()
    return scores, time.perf_counter() - started, exclusions


def _run_stage_a_child(category: str, mode: str) -> dict[str, Any]:
    inherited_cuda_visibility = os.environ.get("CUDA_VISIBLE_DEVICES")
    cublas_workspace = os.environ.get("R10_CUBLAS_WORKSPACE_CONFIG")
    if cublas_workspace:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = cublas_workspace
    os.environ["CUDA_CACHE_DISABLE"] = "1"
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    import numpy as np
    import torch
    import psutil
    from collections import Counter
    from types import SimpleNamespace
    from experiments import p0_protocol_audit as p0
    from experiments.calibration import r1_matched_correction_pilot as r1
    from experiments.calibration import r3_fullstream_2x2 as r3
    from experiments.calibration import r6_capacity1024_audiojack_runner as r6
    from experiments.calibration import r7_shuffled_correction_4class as r7
    from experiments.calibration import r9_correction_mechanism_falsification as r9
    from experiments.calibration.controllers import RollingQ90QueriedNormal, StateConsistentQuantile
    from experiments.calibration.native import _fixed_bank_digest
    from experiments.calibration.protocol import Feedback, Observation
    from experiments.calibration.r1_runtime_supervisor import current_process_is_in_job
    from experiments.calibration.r2_feedback_repair import OutputBudget
    from experiments.risk_safe.native_bridge import dynamic_codebooks
    from v02_memory import MemoryArm

    if not current_process_is_in_job():
        raise R10Error("R10 Stage A child is outside the Windows Job Object")
    output, global_manifest, identity = _verify_child_environment(category, mode)
    if output.parent.resolve(strict=True) != RUN_ROOT.resolve(strict=True):
        raise R10Error("R10 Stage A output leaf is not directly under the unique run root")
    budget = OutputBudget(output, create=True, max_bytes=OUTPUT_LIMIT_BYTES,
                          failure_reserve_bytes=FAILURE_RESERVE_BYTES, max_files=20)
    current_position: int | None = None
    reader = None
    runtime_info: dict[str, Any] = {}
    try:
        reference = _load_r9_reference(categories=(category,))
        ref = reference["references"][category]
        info = reference["frozen_contract"]["metadata"][category]
        events = ref["events"]
        expected_events = EXPECTED_COUNTS[category]
        event_limit = TEST_SMOKE_EVENTS if mode == "SMOKE" else expected_events
        if len(events) != expected_events:
            raise R10Error("Frozen R9-matched event list has the wrong length")
        allowed_queries = tuple(p for p in ref["query_positions"] if p < event_limit)
        if mode == "SMOKE" and allowed_queries:
            raise R10Error("R10 smoke boundary unexpectedly includes a queried event")
        budget.write_json("run_manifest.json", {
            "schema": "r10.signed_knn.child_manifest.v1",
            "run_id": RUN_ID, "category": category, "mode": mode,
            "identity_sha256": identity["identity_sha256"],
            "global_manifest_sha256": _sha_file(RUN_ROOT / "global" / "run_manifest.json"),
            "r9_score_sha256": ref["score_sha256"],
            "r9_condition_seal_sha256": ref["r9_condition_seal_sha256"],
            "fixed_support_sha256": info["fixed_support_sha256_expected"],
            "c64_image_sha256": info["frozen_info"]["c64"]["image_sha256"],
            "query_positions": list(ref["query_positions"]),
            "query_positions_sha256": _sha_json(list(ref["query_positions"])),
            "source_sha256": identity["source_hashes"],
            "automatic_restart": False, "resume_incomplete_run": False,
        })
        r3._restore_cuda_visibility(inherited_cuda_visibility)
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise R10Error("R10 requires exactly one visible CUDA device")
        torch.set_num_threads(1)
        impl = p0.require_sources()
        model_receipt = r3._get_verified_model_receipt()
        restore_hub, runtime_info = r3._runtime_and_native_setup(torch, impl, model_receipt)
        cfg, scorer_profile = r3._native_config(
            p0, __import__("experiments.calibration.profiles", fromlist=["NativeScorerConfig"]), r1)
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        child_started = time.perf_counter()
        setup_started = time.perf_counter()
        backbone = None
        try:
            backbone = impl.DINOv2MultiLayerBackbone(device="cuda")
        finally:
            restore_hub()
        loaded_sha = r1._model_state_sha256(backbone.model)
        expected_runtime = ref["completion"]["runtime_profile"]
        if (loaded_sha != expected_runtime.get("loaded_model_state_sha256") or
                runtime_info.get("hub_source_tree_sha256") != expected_runtime.get("hub_source_tree_sha256") or
                model_receipt.get("checkpoint_sha256") != expected_runtime.get("checkpoint_sha256") or
                runtime_info.get("runtime_profile", {}).get("cublas_workspace_config") !=
                cublas_workspace):
            raise R10Error("Loaded DINO source or weights differ from sealed R9 identity")
        runtime_info["loaded_model_state_sha256"] = loaded_sha
        support_paths = [str(path) for path in info["contract"]["support_paths"]]
        support_features = p0.encode_reference_features(impl, backbone, support_paths, cfg)
        support_encoder_invocations = sum(len(support_features[path]) for path in support_paths)
        impl.set_seed(0)
        bank = p0.build_vmb(impl, support_features, support_paths, cfg)
        fixed_receipt = r3._check_bank_receipt(
            bank, p0, info["fixed_support_sha256_expected"], r1)
        fixed_sha = str(fixed_receipt["actual_structured_sha256"])
        if fixed_sha != ref["completion"]["fixed_support_sha256"]:
            raise R10Error("Structured fixed Support bank differs from R9")
        r3._validate_correction_config(bank)
        _correction_cfg, greedy_codebooks = r6._configure_r6_correction_capacity(bank)
        if (len(greedy_codebooks) != 1 or int(greedy_codebooks[0].K) != 1024 or
                int(greedy_codebooks[0].size()) != 0):
            raise R10Error("Common Greedy state did not initialize empty at K=1024")
        writer = MemoryArm("greedy_coreset", bank,
            SimpleNamespace(flat=1024, stm=1024, ltm=0, buffer=0, min_events=2,
                            promote_every=16, cap_bytes=2 * 1024**3), seed=0)
        if any(int(cb.K) != 1024 or int(cb.size()) != 0 for cb in dynamic_codebooks(bank)):
            raise R10Error("Common Greedy K=1024 memory did not start empty")
        bank.corr_cfg.enabled = False
        bank.use_defect_bank = False
        if _fixed_bank_digest(bank) != fixed_sha:
            raise R10Error("R10 fixed Support bank changed during scorer setup")
        c64_items, c64_cache_receipt = r7._make_c64_items(
            category, info, p0, impl, backbone, cfg)
        if len(c64_items) != 64 or any(
                c64_items[i]["source_group_sha256"] != info["c64_groups"][i]
                for i in range(64)):
            raise R10Error("R10 C64 feature identities do not match the frozen role receipt")
        c64_g, base_cache, c64_base_seconds = r7._score_c64_base(
            impl, bank, c64_items, cfg, r1, category)
        expected_c64 = np.asarray(info["c64_scores"], dtype=np.float64)
        if any(not r3._score_tolerance(a, b) for a, b in zip(c64_g, expected_c64)):
            raise R10Error("Initial R10 common G C64 scores differ from frozen Stage 05")
        theta0 = float(np.quantile(np.asarray(c64_g, dtype=np.float64), .90, method="higher"))
        sigma0 = max(float(np.std(expected_c64, ddof=1)), .02)
        if (not math.isclose(theta0, float(info["theta_c64_q90"]), rel_tol=0, abs_tol=1e-12) or
                not math.isclose(sigma0, float(info["sigma_c64_floor_002"]), rel_tol=0, abs_tol=1e-12)):
            raise R10Error("Initial C64 q90/sigma differs from frozen R9 commissioning receipt")
        greedy_sha = r1._memory_content_sha256(bank)
        g_state_sha = r9._scorer_digest(fixed_sha, greedy_sha)
        memories = {capacity: SignedKNNFIFO(capacity) for capacity in CAPACITIES}
        c2_state_sha = {
            capacity: r9._scorer_digest(fixed_sha, greedy_sha,
                                        memories[capacity].state_sha256())
            for capacity in CAPACITIES
        }
        controllers: dict[str, Any] = {}
        c64_scores: dict[int, list[float]] = {}
        b3_history: dict[int, list[dict[str, Any]]] = {}
        c64_group_sha = str(info["frozen_info"]["c64"]["group_sha256"])
        for capacity in CAPACITIES:
            c64_scores[capacity], _elapsed, _excluded = _score_c2_c64(
                impl, bank, c64_items, base_cache, memories[capacity],
                event_position=0)
            if any(not r3._score_tolerance(actual, expected)
                   for actual, expected in zip(c64_scores[capacity], c64_g)):
                raise R10Error("Empty C2 C64 scores do not reproduce the common G scores")
            arm = f"C2-{capacity}"
            controllers[f"{arm}-B1"] = RollingQ90QueriedNormal(
                theta0, window=256, min_count=16, name=f"{arm}-B1")
            b3 = StateConsistentQuantile(
                theta0, delta=.01, min_normals=16, name=f"{arm}-B3")
            controllers[f"{arm}-B3"] = b3
            initial = r7._initialize_b3(
                b3, c64_scores[capacity], c64_group_sha, c2_state_sha[capacity], theta0)
            b3_history[capacity] = [initial]
        setup_seconds = time.perf_counter() - setup_started
        reader = r7.IndexedQueriedLabelReader(
            info["json_path"], category, info["contract"]["category_json_sha256"],
            events, allowed_queries)
        score_gate = r7.ScoreDisclosureGate(
            lambda row: budget.append_jsonl("stage_a_scores.jsonl", row))
        query_set = set(allowed_queries)
        counters: Counter[str] = Counter()
        feedback_digest = hashlib.sha256()
        seen_query_groups: set[str] = set()
        greedy_writer_audit = writer.audit()
        abs_delta_values = {
            capacity: np.zeros((event_limit, 32), dtype=np.float32)
            for capacity in CAPACITIES
        }
        score_deltas = {capacity: np.zeros((event_limit,), dtype=np.float64)
                        for capacity in CAPACITIES}
        timing = {"setup_seconds": setup_seconds, "encoding_seconds": 0.0,
                  "native_g_scoring_seconds": 0.0, "c2_scoring_seconds": 0.0,
                  "c64_g_rescore_seconds": 0.0, "c64_c2_rescore_seconds": 0.0,
                  "feedback_and_update_seconds": 0.0}
        parity = {"events_checked": 0, "query_mismatches": 0,
                  "identity_mismatches": 0, "greedy_state_mismatches": 0,
                  "greedy_version_mismatches": 0,
                  "score_mismatches": 0, "maximum_absolute_g_score_error": 0.0,
                  "greedy_state_transitions_checked": 0}
        last_progress = -1
        for position, event in enumerate(events[:event_limit]):
            current_position = position
            reference_row = ref["rows"][position]
            if g_state_sha != reference_row["score_state_sha256"]["G1024"]:
                parity["greedy_state_mismatches"] += 1
                raise R10Error(f"R10 common Greedy pre-event state diverged at {category}:{position}")
            if int(counters["greedy_version"]) != int(
                    reference_row["score_state_versions"]["G1024"]):
                parity["greedy_version_mismatches"] += 1
                raise R10Error(f"R10 common Greedy state version diverged at {category}:{position}")
            if position and g_state_sha != ref["rows"][position - 1]["score_state_sha256"]["G1024"]:
                parity["greedy_state_transitions_checked"] += 1
            if (int(reference_row["position"]) != position or
                    reference_row["event_id_sha256"] != str(event.event_id_sha256) or
                    reference_row["image_identity_sha256"] != str(event.image_identity_sha256) or
                    reference_row["source_group_sha256"] != str(event.source_group_sha256) or
                    int(reference_row["original_json_position"]) != int(event.original_position) or
                    int(reference_row["camera_id"]) != int(event.camera_id)):
                parity["identity_mismatches"] += 1
                raise R10Error(f"R10 event identity diverged at {category}:{position}")
            queried = position in query_set
            if bool(reference_row["query_tape_decision"]) != queried:
                parity["query_mismatches"] += 1
                raise R10Error(f"R10 query position differs from R9 at {category}:{position}")
            if not Path(event.image_path).is_file():
                raise R10Error(f"Frozen TEST image is missing at {category}:{position}")
            torch.cuda.synchronize()
            timer = time.perf_counter()
            image, features = p0.encode_query(impl, backbone, str(event.image_path), cfg)
            torch.cuda.synchronize()
            timing["encoding_seconds"] += time.perf_counter() - timer
            counters["test_dino_encodings"] += 1
            if (len(features) != 1 or tuple(features[0].shape) != (784, 4096) or
                    features[0].dtype != torch.float32 or
                    not torch.isfinite(features[0]).all().item()):
                raise R10Error("R10 TEST DINO feature contract failed")
            timer = time.perf_counter()
            base_map, top_ids, g_score, summary = r1._native_dense_score(
                impl, bank, image, features, cfg)
            top_ids = np.asarray(top_ids, dtype=np.int64)
            if (len(top_ids) != 32 or not np.array_equal(
                    top_ids, np.asarray(summary["layer0"]["topk_patch_idx"], dtype=np.int64))):
                raise R10Error("Native G Top-32 selection does not match its dense-map receipt")
            torch.cuda.synchronize()
            timing["native_g_scoring_seconds"] += time.perf_counter() - timer
            ref_g = float(reference_row["scores"]["G1024"])
            error = abs(float(g_score) - ref_g)
            parity["maximum_absolute_g_score_error"] = max(
                parity["maximum_absolute_g_score_error"], error)
            if not math.isclose(float(g_score), ref_g, rel_tol=RTOL, abs_tol=ATOL):
                parity["score_mismatches"] += 1
                raise R10Error(f"R10 native G score diverged at {category}:{position}")
            selected = impl.l2_normalize(features[0].to(bank.device))[
                torch.as_tensor(top_ids, device=bank.device, dtype=torch.long)]
            selected_np = selected.detach().float().cpu().numpy()
            group_sha = str(event.source_group_sha256)
            current_scores: dict[int, float] = {}
            current_diagnostics: dict[int, dict[str, Any]] = {}
            timer = time.perf_counter()
            for capacity in CAPACITIES:
                current_scores[capacity], delta, current_diagnostics[capacity] = \
                    _corrected_c2_score(
                        base_map, top_ids, selected_np, memories[capacity],
                        position=position, group_sha256=group_sha)
                abs_delta_values[capacity][position] = np.abs(delta)
                score_deltas[capacity][position] = current_scores[capacity] - float(g_score)
                counters[f"stream_source_exclusions_{capacity}"] += int(
                    current_diagnostics[capacity]["source_excluded_anchor_count"])
                counters[f"stream_corrected_patches_{capacity}"] += int(
                    current_diagnostics[capacity]["corrected_patch_count"])
                counters[f"stream_patch_count_{capacity}"] += int(
                    current_diagnostics[capacity]["patch_count"])
            timing["c2_scoring_seconds"] += time.perf_counter() - timer
            parity["events_checked"] += 1
            predictions: dict[int, dict[str, Any]] = {}
            decisions: dict[str, Any] = {}
            for capacity in CAPACITIES:
                arm = f"C2-{capacity}"
                score = current_scores[capacity]
                decisions[f"{arm}-Frozen"] = {
                    "prediction": int(score >= theta0), "threshold": theta0,
                    "status": "FROZEN_C64_Q90", "scorer_state_version": counters[f"version_{capacity}"],
                    "scorer_state_sha256": c2_state_sha[capacity]}
                obs = Observation(
                    t=position, observation_id=str(event.event_id_sha256),
                    camera_id=int(event.camera_id), score=score,
                    memory_version=int(counters[f"version_{capacity}"]),
                    score_state_digest=c2_state_sha[capacity])
                b1 = controllers[f"{arm}-B1"]
                b1.observe_score(obs)
                pred1 = b1.predict(obs)
                if pred1.label is None or not pred1.available:
                    raise R10Error(f"{arm} B1 unexpectedly abstained")
                decisions[f"{arm}-B1"] = {
                    "prediction": int(pred1.label), "threshold": float(pred1.threshold),
                    "status": "ROLLING_QUERIED_NORMAL_Q90",
                    "scorer_state_version": int(counters[f"version_{capacity}"]),
                    "scorer_state_sha256": c2_state_sha[capacity]}
                b3 = controllers[f"{arm}-B3"]
                decisions[f"{arm}-B3"] = r7._predict_b3(
                    b3, position=position, event=event, score=score,
                    version=int(counters[f"version_{capacity}"]),
                    state_digest=c2_state_sha[capacity])
                predictions[capacity] = {"observation": obs, "b1": pred1,
                                          "b3": b3.predict(obs)}
            score_row = {
                "schema": SCORE_SCHEMA, "run_id": RUN_ID, "category": category,
                "mode": mode, "order": "SHUFFLE0", "position": position,
                "original_json_position": int(event.original_position),
                "event_id_sha256": str(event.event_id_sha256),
                "image_identity_sha256": str(event.image_identity_sha256),
                "source_group_sha256": group_sha, "camera_id": int(event.camera_id),
                "query_tape_decision": bool(queried),
                "query_positions_sha256": _sha_json(list(ref["query_positions"])),
                "query_identity_set_sha256": ref["query_identity_set_sha256"],
                "scores": {"G1024": float(g_score),
                           "C2-256": current_scores[256],
                           "C2-1024": current_scores[1024]},
                "decisions": decisions,
                "score_state_sha256": {"G1024": g_state_sha,
                                       "C2-256": c2_state_sha[256],
                                       "C2-1024": c2_state_sha[1024]},
                "score_state_versions": {
                    "G1024": int(counters["greedy_version"]),
                    "C2-256": int(counters["version_256"]),
                    "C2-1024": int(counters["version_1024"])},
                "structured_fixed_support_sha256": fixed_sha,
                "c2_diagnostics": {
                    str(capacity): current_diagnostics[capacity]
                    for capacity in CAPACITIES},
            }
            score_gate.seal(position, score_row)
            if queried:
                timer = time.perf_counter()
                causal = r1.CausalEventOrder(position)
                causal.seal_scores()
                causal.decide_query(True)
                label = causal.disclose(lambda: reader.label_for(
                    position, sealed_and_fsynced=score_gate.is_sealed(position)))
                if group_sha in seen_query_groups:
                    raise R10Error("R10 query tape repeats a physical Source Group")
                seen_query_groups.add(group_sha)
                for capacity in CAPACITIES:
                    insert = memories[capacity].insert(
                        selected_np, sign=(1 if label == 1 else -1),
                        source_group_sha256=group_sha, position=position)
                    counters[f"insertions_{capacity}"] += insert["inserted"]
                    counters[f"evictions_{capacity}"] += insert["evicted"]
                    if memories[capacity].first_saturation_position is not None:
                        counters[f"capacity_reached_{capacity}"] = \
                            memories[capacity].first_saturation_position
                counters[f"queried_{'defect' if label else 'normal'}"] += 1
                causal.corrections_updated()
                writer_audit_before = writer.audit()
                greedy_before = r1._memory_content_sha256(bank)
                def writer_update(*, position: int, feature_layers: Sequence[Any],
                                  candidate_ids: Any, score: float, theta: float,
                                  sigma: float) -> Mapping[str, Any]:
                    before = r1._memory_content_sha256(bank)
                    writer.update(impl, list(feature_layers), {0: candidate_ids.tolist()},
                                  position, score=score, theta=theta, sigma=sigma)
                    after = r1._memory_content_sha256(bank)
                    audit_after = writer.audit()
                    return {
                        "accepted_patches_delta": int(
                            audit_after["accepted_patches"] - writer_audit_before["accepted_patches"]),
                        "skipped_patches_delta": int(
                            audit_after["skipped_patches"] - writer_audit_before["skipped_patches"]),
                        "_memory_content_changed": before != after}
                action = r1.common_c0_greedy_writer_action(
                    position=position, c0_score=float(g_score), theta=theta0,
                    sigma=sigma0,
                    c0_prediction=int(g_score >= theta0), candidate_ids=top_ids,
                    candidate_features=selected_np, feature_layers=features,
                    queried=True, disclosed_label=label,
                    eligibility=impl.select_informative_normal_write,
                    update=writer_update)
                action.pop("_memory_content_changed", None)
                greedy_changed = greedy_before != r1._memory_content_sha256(bank)
                if (action.get("input_score_source") != "common_C0" or
                        float(action.get("score", math.nan)) != float(g_score)):
                    raise R10Error("C0 writer eligibility did not use the common G pre-feedback score")
                if label == 1 and (action.get("action") != "verified_defect_no_normal_write" or
                                   action.get("normal_memory_content_changed") is True):
                    raise R10Error("A queried defect entered common normal Greedy memory")
                feedback = {
                    "schema": "r10.signed_knn.private.feedback.v1",
                    "category": category, "position": position,
                    "event_id_sha256": str(event.event_id_sha256),
                    "source_group_sha256": group_sha, "label": int(label),
                    "score_row_fsynced_before_disclosure": score_gate.is_sealed(position),
                    "writer_outcome_from_G_C0": {
                        (0, 0): "TN", (0, 1): "FP",
                        (1, 0): "FN", (1, 1): "TP"
                    }[(label, int(g_score >= theta0))],
                    "common_greedy_writer_action": action,
                    "greedy_state_changed": bool(greedy_changed),
                }
                greedy_sha_after = r1._memory_content_sha256(bank)
                g_state_sha = r9._scorer_digest(fixed_sha, greedy_sha_after)
                if greedy_changed:
                    counters["greedy_state_changes"] += 1
                    counters["greedy_version"] += 1
                    timer_c64 = time.perf_counter()
                    c64_g, base_cache, _elapsed = r7._score_c64_base(
                        impl, bank, c64_items, cfg, r1, category)
                    timing["c64_g_rescore_seconds"] += time.perf_counter() - timer_c64
                    counters["c64_g_rescore_rounds"] += 1
                if position + 1 < len(ref["rows"]) and g_state_sha != \
                        ref["rows"][position + 1]["score_state_sha256"]["G1024"]:
                    parity["greedy_state_mismatches"] += 1
                    raise R10Error(
                        f"R10 common Greedy post-query state diverged after {category}:{position}")
                for capacity in CAPACITIES:
                    old_digest = c2_state_sha[capacity]
                    new_digest = r9._scorer_digest(
                        fixed_sha, greedy_sha_after, memories[capacity].state_sha256())
                    if old_digest != new_digest:
                        counters[f"version_{capacity}"] += 1
                        timer_c64 = time.perf_counter()
                        values, elapsed, excluded = _score_c2_c64(
                            impl, bank, c64_items, base_cache, memories[capacity],
                            event_position=position + 1)
                        timing["c64_c2_rescore_seconds"] += elapsed
                        counters[f"c64_c2_rescore_rounds_{capacity}"] += 1
                        counters[f"c64_c2_source_exclusions_{capacity}"] += excluded
                        counters[f"c64_c2_score_calls_{capacity}"] += 64
                        old_version = int(counters[f"version_{capacity}"]) - 1
                        transition = r7._refresh_b3(
                            controllers[f"C2-{capacity}-B3"], values, c64_group_sha,
                            old_digest, new_digest, old_version)
                        if transition is None:
                            raise R10Error(f"C2-{capacity} B3 did not refresh its complete C64 state")
                        transition.update({"position": position,
                                           "apply_from_position": position + 1})
                        b3_history[capacity].append(transition)
                        c64_scores[capacity] = values
                        c2_state_sha[capacity] = new_digest
                    for name in ("B1", "B3"):
                        obs = predictions[capacity]["observation"]
                        pred = predictions[capacity]["b1" if name == "B1" else "b3"]
                        controllers[f"C2-{capacity}-{name}"].observe_feedback(
                            obs, pred, Feedback(t=position, queried=True, label=label))
                    controllers[f"C2-{capacity}-B1"].finalize_step()
                after_writer_audit = writer.audit()
                feedback.update({
                    "common_greedy_state_sha256_after": greedy_sha_after,
                    "c2_fifo_state_sha256_after": {
                        str(capacity): memories[capacity].state_sha256()
                        for capacity in CAPACITIES},
                    "state_versions_after": {
                        str(capacity): int(counters[f"version_{capacity}"])
                        for capacity in CAPACITIES},
                    "writer_audit_delta": {
                        "accepted_patches": int(after_writer_audit["accepted_patches"] -
                                                writer_audit_before["accepted_patches"]),
                        "skipped_patches": int(after_writer_audit["skipped_patches"] -
                                               writer_audit_before["skipped_patches"]),
                    },
                })
                feedback_digest.update(_canonical(feedback))
                budget.append_jsonl("attempt_history.jsonl", feedback)
                causal.memory_updated()
                causal.finish()
                timing["feedback_and_update_seconds"] += time.perf_counter() - timer
            else:
                causal = r1.CausalEventOrder(position)
                causal.seal_scores()
                causal.decide_query(False)
                causal.corrections_updated()
                def deny_unqueried(**_kwargs: Any) -> Mapping[str, Any]:
                    raise R10Error("Unqueried event attempted a supervised Greedy write")
                action = r1.common_c0_greedy_writer_action(
                    position=position, c0_score=float(g_score), theta=theta0,
                    sigma=sigma0, c0_prediction=int(g_score >= theta0),
                    candidate_ids=top_ids, candidate_features=selected_np,
                    feature_layers=features, queried=False, disclosed_label=None,
                    eligibility=impl.select_informative_normal_write, update=deny_unqueried)
                if action.get("action") != "no_normal_memory_write":
                    raise R10Error("Unqueried event performed a supervised Greedy write")
                causal.memory_updated()
                causal.finish()
            if (position + 1) % 256 == 0 or position + 1 == event_limit:
                budget.append_jsonl("attempt_history.jsonl", {
                    "schema": "r10.signed_knn.progress.v1",
                    "category": category, "mode": mode,
                    "sealed_through_position": position,
                    "test_dino_encodings": int(counters["test_dino_encodings"]),
                    "queried_feedback_count": int(counters["queried_normal"] +
                                                  counters["queried_defect"]),
                    "automatic_restart": False,
                })
            last_progress = position
        reader.finish()
        reader.close()
        reader = None
        torch.cuda.synchronize()
        timing["child_wall_seconds"] = time.perf_counter() - child_started
        if mode == "SMOKE":
            seal = {
                "schema": "r10.signed_knn.smoke_seal.v1",
                "status": "SMOKE_PASS_INCOMPLETE_NOT_EVALUATED",
                "category": category, "event_count": event_limit,
                "query_count": 0, "test_dino_encodings": event_limit,
                "g_score_parity": parity, "native_c2_empty_state_parity": True,
                "stage_b_outcomes_accessed": False, "test_masks_opened": False,
                "automatic_restart": False, "timing_seconds": timing,
            }
            budget.write_json("stage_a_seal.json", seal)
            return seal
        if (int(counters["test_dino_encodings"]) != EXPECTED_COUNTS[category] or
                int(counters["queried_normal"] + counters["queried_defect"]) !=
                EXPECTED_QUERIES[category]):
            raise R10Error("R10 full category did not consume its exact event/query budget")
        diagnostics: dict[str, Any] = {}
        for capacity in CAPACITIES:
            values = abs_delta_values[capacity].reshape(-1)
            counts = memories[capacity].sign_counts()
            diagnostics[str(capacity)] = {
                "capacity": capacity, "k": K_NEIGHBORS, "amplitude": AMPLITUDE,
                "anchor_rows_inserted": memories[capacity].insertions,
                "fifo_evictions": memories[capacity].evictions,
                "active_anchor_count": memories[capacity].count,
                "active_sign_counts": counts,
                "first_saturation_position": memories[capacity].first_saturation_position,
                "source_exclusions_stream_scoring": int(
                    counters[f"stream_source_exclusions_{capacity}"]),
                "source_exclusions_c64_scoring": int(
                    counters[f"c64_c2_source_exclusions_{capacity}"]),
                "c64_rescore_rounds": int(counters[f"c64_c2_rescore_rounds_{capacity}"]),
                "c64_score_calls": int(counters[f"c64_c2_score_calls_{capacity}"]),
                "corrected_patch_fraction": float(
                    counters[f"stream_corrected_patches_{capacity}"] /
                    max(1, counters[f"stream_patch_count_{capacity}"])),
                "absolute_patch_correction": {
                    "mean": float(np.mean(values)), "median": float(np.quantile(values, .50)),
                    "p95": float(np.quantile(values, .95)), "max": float(np.max(values)),
                    "nonzero_fraction": float(np.mean(values > 0)),
                },
                "score_change_from_g": {
                    "mean": float(np.mean(score_deltas[capacity])),
                    "mean_absolute": float(np.mean(np.abs(score_deltas[capacity]))),
                    "p95_absolute": float(np.quantile(np.abs(score_deltas[capacity]), .95)),
                    "max_absolute": float(np.max(np.abs(score_deltas[capacity]))),
                },
                "measured_array_bytes": memories[capacity].array_bytes,
                "memory_contract": "Kx4096 float32 + K int8 + K int64 + Kx32 uint8",
            }
        writer_audit_final = writer.audit()
        seal = {
            "schema": STAGE_A_SCHEMA, "status": "SEALED_COMPLETE",
            "run_id": RUN_ID, "category": category, "mode": mode,
            "order": "SHUFFLE0", "support_seed": 0, "shuffle_seed": 0,
            "event_count": EXPECTED_COUNTS[category],
            "query_count": EXPECTED_QUERIES[category],
            "query_positions": list(ref["query_positions"]),
            "query_positions_sha256": _sha_json(list(ref["query_positions"])),
            "query_identity_set_sha256": ref["query_identity_set_sha256"],
            "score_rows_sha256": _sha_file(output / "stage_a_scores.jsonl"),
            "feedback_receipts_sha256": feedback_digest.hexdigest(),
            "feedback_receipt_count": int(counters["queried_normal"] + counters["queried_defect"]),
            "fixed_support_sha256": fixed_sha,
            "r9_g_reference_sha256": ref["score_sha256"],
            "r9_c1_full_reference_used_without_rescoring": True,
            "c64_cache_receipt": c64_cache_receipt,
            "runtime_profile": runtime_info,
            "source_sha256": identity["source_hashes"],
            "event_identity_coverage": parity["events_checked"] == EXPECTED_COUNTS[category],
            "g_score_parity": parity,
            "c2_diagnostics": diagnostics,
            "controller_initial_theta": theta0,
            "controller_policy": {
                "B1": "RollingQ90QueriedNormal(window=256,min_count=16)",
                "B3": "StateConsistentQuantile(delta=0.01,min_normals=16)",
                "each_C2_has_own_C64_scores_and_state_digest": True,
            },
            "b3_state_history": {
                str(capacity): b3_history[capacity] for capacity in CAPACITIES},
            "greedy_writer_audit": writer_audit_final,
            "support_feature_encoder_invocations": int(support_encoder_invocations),
            "actual_test_dino_encodings": int(counters["test_dino_encodings"]),
            "actual_new_c64_dino_encodings": int(
                c64_cache_receipt.get("new_c64_dino_encodings", 0)),
            "timing_seconds": timing,
            "peak_cuda_allocated_bytes": int(torch.cuda.max_memory_allocated()),
            "peak_cuda_reserved_bytes": int(torch.cuda.max_memory_reserved()),
            "peak_cpu_rss_bytes": int(psutil.Process().memory_info().rss),
            "unqueried_test_labels_accessed": False,
            "r9_postseal_outcomes_accessed": False,
            "test_masks_opened": False,
            "automatic_restart": False,
            "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        budget.write_json("stage_a_seal.json", seal)
        return seal
    except BaseException as exc:
        _write_failure(budget, category=category, phase="stage_a",
                       exception=exc, position=current_position)
        raise
    finally:
        if reader is not None:
            reader.close()
        try:
            budget.audit()
        except Exception:
            pass


def _read_sealed_score_rows(category: str, refs: Mapping[str, Any],
                            identity: Mapping[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    output = RUN_ROOT / category
    _reject_reparse_chain(output, RUN_ROOT)
    if not output.is_dir():
        raise R10Error(f"{category}: R10 Stage A output leaf is missing")
    manifest = _read_json(output / "run_manifest.json")
    seal = _read_json(output / "stage_a_seal.json")
    score_path = output / "stage_a_scores.jsonl"
    reference = refs["references"][category]
    if (manifest.get("identity_sha256") != identity["identity_sha256"] or
            seal.get("status") != "SEALED_COMPLETE" or
            seal.get("category") != category or seal.get("mode") != "FULL" or
            seal.get("event_count") != EXPECTED_COUNTS[category] or
            seal.get("query_count") != EXPECTED_QUERIES[category] or
            seal.get("score_rows_sha256") != _sha_file(score_path) or
            seal.get("r9_g_reference_sha256") != reference["score_sha256"] or
            seal.get("r9_c1_full_reference_used_without_rescoring") is not True or
            seal.get("unqueried_test_labels_accessed") is not False or
            seal.get("r9_postseal_outcomes_accessed") is not False or
            seal.get("test_masks_opened") is not False):
        raise R10Error(f"{category}: R10 Stage A category seal failed")
    rows: list[dict[str, Any]] = []
    with score_path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                raise R10Error(f"{category}: blank score row at {line_number}")
            row = json.loads(line)
            if any(key in row for key in ("label", "anomaly_class", "mask_path", "ground_truth")):
                raise R10Error(f"{category}: R10 Stage A row contains a forbidden outcome field")
            rows.append(row)
    if len(rows) != EXPECTED_COUNTS[category]:
        raise R10Error(f"{category}: R10 score row count differs from the frozen event total")
    for position, (row, reference_row) in enumerate(zip(rows, reference["rows"])):
        if (row.get("position") != position or
                row.get("event_id_sha256") != reference_row.get("event_id_sha256") or
                row.get("image_identity_sha256") != reference_row.get("image_identity_sha256") or
                row.get("source_group_sha256") != reference_row.get("source_group_sha256") or
                row.get("query_tape_decision") != reference_row.get("query_tape_decision") or
                set(row.get("scores", {})) != {"G1024", "C2-256", "C2-1024"} or
                "GC1024" in row.get("scores", {})):
            raise R10Error(f"{category}: sealed R10 row identity/arm mismatch at {position}")
    return rows, seal


def _stage_a_global_seal(refs: Mapping[str, Any], identity: Mapping[str, Any],
                         smoke: Mapping[str, Any]) -> dict[str, Any]:
    category_seals: dict[str, Any] = {}
    total_events = 0
    total_queries = 0
    max_g_error = 0.0
    for category in CATEGORIES:
        rows, seal = _read_sealed_score_rows(category, refs, identity)
        category_seals[category] = {
            "event_count": len(rows), "query_count": seal["query_count"],
            "score_rows_sha256": seal["score_rows_sha256"],
            "stage_a_seal_sha256": _sha_file(RUN_ROOT / category / "stage_a_seal.json"),
            "query_positions_sha256": seal["query_positions_sha256"],
            "g_score_parity": seal["g_score_parity"],
            "c2_diagnostics": seal["c2_diagnostics"],
            "timing_seconds": seal["timing_seconds"],
        }
        total_events += len(rows)
        total_queries += int(seal["query_count"])
        max_g_error = max(max_g_error,
                          float(seal["g_score_parity"]["maximum_absolute_g_score_error"]))
    if (total_events != sum(EXPECTED_COUNTS.values()) or
            total_queries != sum(EXPECTED_QUERIES.values())):
        raise R10Error("R10 global Stage A event/query totals do not match the frozen contract")
    return {
        "schema": "r10.signed_knn.stage_a_global_seal.v1",
        "status": "SEALED_COMPLETE", "run_id": RUN_ID,
        "identity_sha256": identity["identity_sha256"],
        "category_event_counts": dict(EXPECTED_COUNTS),
        "category_query_counts": dict(EXPECTED_QUERIES),
        "event_count": total_events, "query_count": total_queries,
        "category_seals": category_seals, "smoke": dict(smoke),
        "g_parity": {
            "events_checked": total_events, "query_mismatches": 0,
            "identity_mismatches": 0, "greedy_state_mismatches": 0,
            "greedy_version_mismatches": 0,
            "score_mismatches": 0, "maximum_absolute_score_error": max_g_error,
            "atol": ATOL, "rtol": RTOL,
        },
        "all_four_full_stage_a_streams_sealed": True,
        "all_event_score_rows_fsynced_before_query_disclosure": True,
        "r9_postseal_outcomes_accessed": False,
        "unqueried_test_labels_accessed": False, "test_masks_opened": False,
        "automatic_restart": False,
        "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }


def _controller_summary(labels: Any, rows: Sequence[Mapping[str, Any]],
                        score_key: str, decision_prefix: str,
                        r7: Any) -> dict[str, Any]:
    import numpy as np
    result: dict[str, Any] = {}
    for controller in ("Frozen", "B1", "B3"):
        key = f"{decision_prefix}-{controller}"
        predictions = np.asarray(
            [int(row["decisions"][key]["prediction"]) for row in rows], dtype=np.int8)
        if predictions.shape != np.asarray(labels).shape:
            raise R10Error(f"{key} predictions do not align with the sealed score tape")
        result[controller] = {
            **r7._binary_metrics(labels, predictions),
            "prediction_coverage": float(np.mean(np.isin(predictions, (0, 1)))),
            "worst_complete_512_stride_1": r7._worst_512_metrics(
                labels, predictions, window=512),
        }
    return result


def _compute_stage_b_results(refs: Mapping[str, Any], identity: Mapping[str, Any],
                             stage_a_global: Mapping[str, Any],
                             contract: Mapping[str, Any]) -> dict[str, Any]:
    import numpy as np
    from experiments.calibration import r3_fullstream_2x2 as r3
    from experiments.calibration import r7_shuffled_correction_4class as r7
    from experiments.calibration import r9_correction_mechanism_falsification as r9

    if (stage_a_global.get("status") != "SEALED_COMPLETE" or
            stage_a_global.get("event_count") != sum(EXPECTED_COUNTS.values()) or
            set(stage_a_global.get("category_seals", {})) != set(CATEGORIES)):
        raise R10Error("Stage B refused labels: all four full Stage A streams are not sealed")
    methods_all = ("G1024", "FullC1-GC1024", "C2-256", "C2-1024")
    all_ranking = {name: {"auroc": [], "average_precision": []}
                   for name in methods_all}
    all_controller = {
        name: {controller: {metric: [] for metric in
                            ("balanced_accuracy", "fpr", "fnr")}
               for controller in ("Frozen", "B1", "B3")}
        for name in methods_all}
    contrast_names = (
        "C2-256_minus_G1024", "C2-1024_minus_G1024",
        "FullC1_minus_C2-256", "FullC1_minus_C2-1024",
        "C2-1024_minus_C2-256",
    )
    contrast_pairs = {
        "C2-256_minus_G1024": ("C2-256", "G1024"),
        "C2-1024_minus_G1024": ("C2-1024", "G1024"),
        "FullC1_minus_C2-256": ("FullC1-GC1024", "C2-256"),
        "FullC1_minus_C2-1024": ("FullC1-GC1024", "C2-1024"),
        "C2-1024_minus_C2-256": ("C2-1024", "C2-256"),
    }
    category_results: dict[str, Any] = {}
    for category_index, category in enumerate(CATEGORIES):
        rows, stage_a_seal = _read_sealed_score_rows(category, refs, identity)
        ref_rows = refs["references"][category]["rows"]
        if len(ref_rows) != len(rows):
            raise R10Error(f"{category}: R9 and R10 event populations differ")
        for left, right in zip(rows, ref_rows):
            if left["image_identity_sha256"] != right["image_identity_sha256"]:
                raise R10Error(f"{category}: Stage B image identity match failed")
        # This is the first outcome projection in R10; every full Stage A seal
        # has already been verified by the caller.
        original_labels = np.asarray(r3._read_stage_b_label_projection(
            contract["metadata"][category]["json_path"], EXPECTED_COUNTS[category]),
            dtype=np.int8)
        original_positions = [int(row["original_json_position"]) for row in rows]
        if len(set(original_positions)) != EXPECTED_COUNTS[category]:
            raise R10Error(f"{category}: shuffled/original index map is not one-to-one")
        labels = np.asarray([original_labels[p] for p in original_positions], dtype=np.int8)
        if len(np.unique(labels)) != 2:
            raise R10Error(f"{category}: full-stream AUROC requires both outcomes")
        query_records = [
            row for row in _read_jsonl(RUN_ROOT / category / "attempt_history.jsonl")
            if row.get("schema") == "r10.signed_knn.private.feedback.v1"
        ]
        query_positions = tuple(int(row["position"]) for row in query_records)
        expected_positions = refs["references"][category]["query_positions"]
        if query_positions != tuple(expected_positions):
            raise R10Error(f"{category}: post-seal feedback positions differ from Q-A-TS")
        if any(int(row["label"]) != int(labels[int(row["position"])])
               for row in query_records):
            raise R10Error(f"{category}: legal feedback differs from Stage B projection")
        score_source = {
            "G1024": [float(row["scores"]["G1024"]) for row in ref_rows],
            "FullC1-GC1024": [float(row["scores"]["GC1024"]) for row in ref_rows],
            "C2-256": [float(row["scores"]["C2-256"]) for row in rows],
            "C2-1024": [float(row["scores"]["C2-1024"]) for row in rows],
        }
        decision_source = {
            "G1024": (ref_rows, "G1024"),
            "FullC1-GC1024": (ref_rows, "GC1024"),
            "C2-256": (rows, "C2-256"),
            "C2-1024": (rows, "C2-1024"),
        }
        method_results: dict[str, Any] = {}
        for method in methods_all:
            ranking = r9._ranking_metrics(labels, score_source[method])
            source_rows, prefix = decision_source[method]
            controllers = _controller_summary(labels, source_rows, prefix, prefix, r7)
            method_results[method] = {
                "ranking": ranking, "controllers": controllers,
                "event_score_source": "sealed R9 Stage A" if method in
                    ("G1024", "FullC1-GC1024") else "R10 Stage A",
            }
            for metric in ("auroc", "average_precision"):
                value = ranking[metric]
                if value is not None:
                    all_ranking[method][metric].append(float(value))
            for controller in ("Frozen", "B1", "B3"):
                for metric in ("balanced_accuracy", "fpr", "fnr"):
                    value = controllers[controller][metric]
                    if value is not None:
                        all_controller[method][controller][metric].append(float(value))
        groups = [str(row["source_group_sha256"]) for row in rows]
        contrasts: dict[str, Any] = {}
        paired: dict[str, Any] = {}
        for index, name in enumerate(contrast_names):
            left_name, right_name = contrast_pairs[name]
            left_scores = np.asarray(score_source[left_name], dtype=np.float64)
            right_scores = np.asarray(score_source[right_name], dtype=np.float64)
            left_rank = r9._ranking_metrics(labels, left_scores)
            right_rank = r9._ranking_metrics(labels, right_scores)
            left_rows, left_prefix = decision_source[left_name]
            right_rows, right_prefix = decision_source[right_name]
            left_ctrl = method_results[left_name]["controllers"]
            right_ctrl = method_results[right_name]["controllers"]
            contrasts[name] = {
                "auroc_difference": float(left_rank["auroc"] - right_rank["auroc"]),
                "average_precision_difference": float(
                    left_rank["average_precision"] - right_rank["average_precision"]),
                "controller_differences": {
                    controller: {
                        metric: float(left_ctrl[controller][metric] -
                                      right_ctrl[controller][metric])
                        for metric in ("balanced_accuracy", "fpr", "fnr")
                    } for controller in ("Frozen", "B1", "B3")
                },
            }
            paired[name] = r9._source_group_bootstrap(
                labels, left_scores, right_scores, groups,
                seed=BOOTSTRAP_SEED + 100 * category_index + index)
        category_results[category] = {
            "event_count": len(rows), "query_count": len(query_records),
            "positive_count": int(np.count_nonzero(labels == 1)),
            "negative_count": int(np.count_nonzero(labels == 0)),
            "methods": method_results, "contrasts": contrasts,
            "paired_source_group_bootstrap": paired,
            "fifo_diagnostics": stage_a_seal["c2_diagnostics"],
            "g_score_parity": stage_a_seal["g_score_parity"],
            "runtime": stage_a_seal["timing_seconds"],
        }
    macro_ranking = {
        method: {
            metric: {"equal_category_macro": float(np.mean(values)) if values else None,
                     "evaluable_categories": len(values)}
            for metric, values in metrics.items()
        } for method, metrics in all_ranking.items()}
    macro_controller = {
        method: {
            controller: {
                metric: {"equal_category_macro": float(np.mean(values)) if values else None,
                         "evaluable_categories": len(values)}
                for metric, values in metrics.items()
            } for controller, metrics in controllers.items()
        } for method, controllers in all_controller.items()}
    macro_contrasts = {}
    for name in contrast_names:
        values = [category_results[category]["contrasts"][name]
                  for category in CATEGORIES]
        macro_contrasts[name] = {
            key: float(np.mean([row[key] for row in values]))
            for key in ("auroc_difference", "average_precision_difference")
        }
    return {
        "schema": "r10.signed_knn.public_aggregate.v1",
        "status": "SEALED_COMPLETE",
        "classification": "RETROSPECTIVE_DEVELOPMENT_DIAGNOSTIC",
        "run_id": RUN_ID, "r9_baseline_run_id": R9_RUN_ID,
        "protocol": {
            "categories": list(CATEGORIES), "support_seed": 0, "shuffle_seed": 0,
            "events": sum(EXPECTED_COUNTS.values()),
            "queries": sum(EXPECTED_QUERIES.values()),
            "new_methods": ["C2-256", "C2-1024"],
            "k": K_NEIGHBORS, "amplitude": AMPLITUDE,
            "full_c1_rescored": False, "test_masks_opened": False,
        },
        "categories": category_results,
        "equal_category_macro": {
            "ranking": macro_ranking, "controllers": macro_controller,
            "contrasts": macro_contrasts,
        },
        "primary_contrasts": {
            "Delta_C2_256": macro_contrasts["C2-256_minus_G1024"]["auroc_difference"],
            "Delta_C2_1024": macro_contrasts["C2-1024_minus_G1024"]["auroc_difference"],
            "Delta_Full_256": macro_contrasts["FullC1_minus_C2-256"]["auroc_difference"],
            "Delta_Full_1024": macro_contrasts["FullC1_minus_C2-1024"]["auroc_difference"],
            "Delta_Capacity": macro_contrasts["C2-1024_minus_C2-256"]["auroc_difference"],
        },
        "uncertainty": {
            "method": "paired whole-physical-Source-Group bootstrap; descriptive conditional on this development trajectory",
            "replicates": 1000,
            "category_intervals": {
                category: category_results[category]["paired_source_group_bootstrap"]
                for category in CATEGORIES},
        },
        "label_and_mask_firewall": {
            "Stage_A_unqueried_TEST_labels_accessed": False,
            "Stage_A_R9_postseal_outcomes_accessed": False,
            "Stage_B_after_all_four_R10_Stage_A_seals": True,
            "Stage_B_test_masks_opened": False,
            "raw_labels_in_aggregate": False, "event_scores_in_aggregate": False,
            "sample_paths_in_aggregate": False, "feature_vectors_in_aggregate": False,
        },
    }


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                raise R10Error(f"Blank JSONL row in {path.name}:{line_number}")
            row = json.loads(line)
            if not isinstance(row, dict):
                raise R10Error(f"Malformed JSONL row in {path.name}:{line_number}")
            rows.append(row)
    return rows


def _stage_b(refs: Mapping[str, Any], identity: Mapping[str, Any],
             stage_a_global: Mapping[str, Any],
             contract: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    import torch
    if torch.cuda.is_initialized() or torch.cuda.device_count() != 0:
        raise R10Error("R10 Stage B requires CUDA hidden and uninitialized")
    first = _compute_stage_b_results(refs, identity, stage_a_global, contract)
    second = _compute_stage_b_results(refs, identity, stage_a_global, contract)
    first_bytes, second_bytes = _canonical(first), _canonical(second)
    if first_bytes != second_bytes:
        raise R10Error("Independent second CPU pass did not reproduce all Stage B metrics")
    seal = {
        "schema": STAGE_B_SCHEMA, "status": "SEALED_COMPLETE",
        "run_id": RUN_ID, "identity_sha256": identity["identity_sha256"],
        "metrics_payload_sha256": hashlib.sha256(first_bytes).hexdigest(),
        "independent_second_pass": {
            "status": "PASS", "all_metric_cells_reconciled": True,
            "masks_opened": False, "outcome_vectors_written": False,
        },
        "test_masks_opened": False,
        "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    return first, seal


def _format_num(value: Any, digits: int = 6) -> str:
    return "NA" if value is None else f"{float(value):.{digits}f}"


def _build_report(aggregate: Mapping[str, Any], stage_a_global: Mapping[str, Any],
                  stage_b_seal: Mapping[str, Any], identity: Mapping[str, Any],
                  smoke_status: Mapping[str, Any], resource_preflight: Mapping[str, Any],
                  total_seconds: float) -> str:
    categories = aggregate["categories"]
    lines = [
        "# R10 Targeted Signed kNN Baseline Comparison",
        "",
        "## Status and scope",
        "",
        "R10 completed as a RETROSPECTIVE_DEVELOPMENT_DIAGNOSTIC on the same exposed four-category development stream. It is not independent confirmation or a blind evaluation. The only new arms were Signed kNN with k=5, amplitude 0.02, and global FIFO capacities 256 and 1024. Support Seed 0, Shuffle Seed 0, the exact R9 query set, and the exact R9 event order were used.",
        "",
        f"Run ID: {RUN_ID}. Parent: {EXPECTED_PARENT}. Identity SHA-256: {identity['identity_sha256']}.",
        "",
        f"Smoke: {smoke_status.get('status')} for {smoke_status.get('event_count')} events; it was incomplete and not evaluated. Full Stage A: {stage_a_global['status']} for {stage_a_global['event_count']:,} events and {stage_a_global['query_count']:,} legal queries. Stage B: {stage_b_seal['status']}; independent second pass: {stage_b_seal['independent_second_pass']['status']}.",
        "",
        "## Why Signed kNN was revisited",
        "",
        "R2 previously reported switch-prefix AUROC 0.807035 for C0, 0.843083 for C1, 0.752198 for C2 at amplitude 0.20, and 0.841025 for C2 at amplitude 0.02. R2-EXP2A reported descriptive three-class means of 0.816168 for C1-Full and 0.816702 for C2 at 0.02. Those short development prefixes were already exposed; audiojack's prefix lacked defects, and amplitude 0.02 was selected after switch exposure. R10 asks whether this simple mechanism retains performance on the complete matched R9 stream.",
        "",
        "R2-EXP1 included an earlier controlled amplitude comparison on an exposed switch trajectory. That historical comparison can support a local amplitude effect on that trajectory. R10 holds amplitude fixed at 0.02, so R10 does not identify an amplitude effect or validate 0.02 as an optimum.",
        "",
        "## Reused baseline and execution",
        "",
        f"R9 run {R9_RUN_ID} was preflighted from its outcome-independent Stage A seals and event score tapes. The reference supplied G1024 and GC1024 scores, decisions, thresholds, event identities, query decisions, and Greedy pre-event state digests. Full C1 was not rescored or updated in R10.",
        "",
        "No compatible full TEST feature/map cache existed in R9: its private artifacts held score/feedback tapes and seals, not per-event Top-32 features or base maps. Audiojack's protected C64 feature cache was reused from the previously verified R5 cache. The other 192 C64 images were encoded once for this task; each TEST image was encoded once per full R10 stream. The 16-event smoke is counted separately.",
        "",
        f"Full R10 TEST DINO encodings: {aggregate['efficiency']['full_test_dino_encodings']:,} across four full category passes. Smoke TEST encodings: {aggregate['efficiency']['smoke_test_dino_encodings']}. New C64 DINO encodings: {aggregate['efficiency']['new_c64_dino_encodings']}. Support feature encoder invocations: {aggregate['efficiency']['support_feature_encoder_invocations']}. Full C1 reruns: 0.",
        "",
        f"Elapsed time, including smoke and two CPU Stage B reconciliations: {total_seconds / 60:.1f} minutes. Minimum free space at launch: {resource_preflight['free_bytes'] / (1024**3):.1f} GiB. Windows Job Object deadlines were 10 minutes for smoke and 60 minutes per full category child; no child was restarted.",
        "",
        "## Parity",
        "",
        f"Exact event identity coverage: {stage_a_global['event_count']:,}/{stage_a_global['event_count']:,}. G score parity against sealed R9: {stage_a_global['g_parity']['score_mismatches']} mismatches, maximum absolute error {_format_num(stage_a_global['g_parity']['maximum_absolute_score_error'], 9)} (tolerance atol={ATOL:g}, rtol={RTOL:g}). Query, identity, and Greedy-state mismatches: 0. The Greedy state digest was compared at every pre-event position, including each post-query transition before the next event.",
        "",
        "## Ranking results",
        "",
        "| Category | G AUROC | Full C1 AUROC | C2-256 AUROC | C2-1024 AUROC | G AP | Full AP | C2-256 AP | C2-1024 AP |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    columns = (
        ("G1024", "auroc"), ("FullC1-GC1024", "auroc"),
        ("C2-256", "auroc"), ("C2-1024", "auroc"),
        ("G1024", "average_precision"), ("FullC1-GC1024", "average_precision"),
        ("C2-256", "average_precision"), ("C2-1024", "average_precision"),
    )
    for category in CATEGORIES:
        methods = categories[category]["methods"]
        values = [methods[name]["ranking"][metric] for name, metric in columns]
        lines.append("| " + category + " | " +
                     " | ".join(_format_num(value) for value in values) + " |")
    macro = aggregate["equal_category_macro"]["ranking"]
    lines.append("| **Equal-category macro** | " + " | ".join(
        _format_num(macro[name][metric]["equal_category_macro"])
        for name, metric in columns) + " |")
    lines.extend([
        "",
        "Primary AUROC contrasts are equal-category macro differences. Positive Full−C2 favors Full C1; positive C2−G favors C2.",
        "",
        "| Contrast | AUROC difference | AP difference |",
        "|---|---:|---:|",
    ])
    for name in (
        "C2-256_minus_G1024", "C2-1024_minus_G1024",
        "FullC1_minus_C2-256", "FullC1_minus_C2-1024",
        "C2-1024_minus_C2-256",
    ):
        row = aggregate["equal_category_macro"]["contrasts"][name]
        lines.append(f"| {name} | {_format_num(row['auroc_difference'])} | {_format_num(row['average_precision_difference'])} |")
    lines.extend([
        "",
        "## Operating metrics",
        "",
        "| Category / method | Controller | BA | FPR | FNR | TP | TN | FP | FN | Coverage | Worst 512 BA |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for category in CATEGORIES:
        for method in ("G1024", "FullC1-GC1024", "C2-256", "C2-1024"):
            for controller in ("Frozen", "B1", "B3"):
                item = categories[category]["methods"][method]["controllers"][controller]
                confusion = item["confusion"]
                worst = item["worst_complete_512_stride_1"]["worst_balanced_accuracy"]
                lines.append(
                    f"| {category} / {method} | {controller} | "
                    f"{_format_num(item['balanced_accuracy'])} | {_format_num(item['fpr'])} | "
                    f"{_format_num(item['fnr'])} | {confusion['tp']} | {confusion['tn']} | "
                    f"{confusion['fp']} | {confusion['fn']} | "
                    f"{_format_num(item['prediction_coverage'])} | {_format_num(worst)} |")
    lines.extend([
        "",
        "Paired whole-physical-Source-Group bootstrap intervals are in the public aggregate for every category and registered contrast. They are descriptive conditional on these development streams; repeated camera views are not independent trials.",
        "",
        "## FIFO and resource diagnostics",
        "",
        "| Category | Capacity | Inserted patch rows | FIFO evictions | Final + / − | First saturation position | Corrected patch fraction | Mean absolute delta | Array bytes |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for category in CATEGORIES:
        for capacity in CAPACITIES:
            diag = categories[category]["fifo_diagnostics"][str(capacity)]
            lines.append(
                f"| {category} | {capacity} | {diag['anchor_rows_inserted']} | "
                f"{diag['fifo_evictions']} | {diag['active_sign_counts']['positive']} / "
                f"{diag['active_sign_counts']['negative']} | {diag['first_saturation_position']} | "
                f"{_format_num(diag['corrected_patch_fraction'])} | "
                f"{_format_num(diag['absolute_patch_correction']['mean'], 8)} | "
                f"{diag['measured_array_bytes']:,} |")
    lines.extend([
        "",
        "Scoring and update times, score change from G, absolute correction quantiles, source exclusions, C64 rescoring counts, memory bytes, and B3 state transitions are in the private category seals.",
        "",
        "## Interpretation",
        "",
    ])
    macro_auroc = {
        name: macro[name]["auroc"]["equal_category_macro"]
        for name in ("G1024", "FullC1-GC1024", "C2-256", "C2-1024")}
    full_gap = macro_auroc["FullC1-GC1024"] - macro_auroc["C2-1024"]
    cap_gap = macro_auroc["C2-1024"] - macro_auroc["C2-256"]
    lines.append(
        f"- Historical Signed kNN at amplitude 0.02 had equal-category macro AUROC "
        f"{_format_num(macro_auroc['C2-256'])} at capacity 256 and "
        f"{_format_num(macro_auroc['C2-1024'])} at capacity 1024 on this stream.")
    lines.append(
        f"- Capacity-matched Full C1 minus C2-1024 is {_format_num(full_gap)} AUROC; "
        f"C2-1024 minus C2-256 is {_format_num(cap_gap)}.")
    lines.append(
        "- Whether Full C1 has a meaningful incremental advantage depends on these observed margins and the descriptive Source Group intervals; this is not independent evidence.")
    lines.append(
        "- R10 cannot attribute a difference to amplitude because amplitude was fixed. The earlier R2-EXP1 controlled amplitude comparison was limited to the already exposed switch development trajectory.")
    lines.append(
        "- The same development categories and query set were used in prior work. Results are retrospective and do not establish performance on confirmation classes.")
    lines.extend([
        "",
        "## Source identity and sealed artifacts",
        "",
        f"R10 identity SHA-256: {identity['identity_sha256']}. R9 identity SHA-256: {identity['r9_reference']['identity_sha256']}. R9 Stage A global seal SHA-256: {identity['r9_reference']['seal_hashes']['stage_a']}.",
        "",
        f"Stage B metrics payload SHA-256: {stage_b_seal['metrics_payload_sha256']}. Private category score, feedback, and completion hashes are bound by the global seals. Public output excludes sample paths, raw labels, event-level scores, and feature vectors.",
        "",
        "## Limitations",
        "",
        "Amplitude 0.02 was chosen after exposure to switch development results. The 1024-anchor FIFO is a capacity-matched R10 extension, not the historical C2 configuration. This is one fixed shuffle and support seed. The paired Source Group intervals are descriptive conditional on the selected streams. No test-label-guided parameter selection occurred.",
        "",
    ])
    return "\n".join(lines)


def _create_output_budget(path: Path, *, create: bool) -> Any:
    from experiments.calibration.r2_feedback_repair import OutputBudget
    return OutputBudget(path, create=create, max_bytes=OUTPUT_LIMIT_BYTES,
                        failure_reserve_bytes=FAILURE_RESERVE_BYTES, max_files=20)


def _audit_output_namespace() -> dict[str, Any]:
    expected = {"global", "smoke", *CATEGORIES}
    _reject_reparse_chain(RUN_ROOT)
    entries = list(RUN_ROOT.iterdir())
    if any(path.is_symlink() or not path.is_dir() for path in entries):
        raise R10Error("R10 run root contains a file, symlink, or unexpected path type")
    if {path.name for path in entries} != expected:
        raise R10Error("R10 run root contains missing or unaccounted output leaves")
    leaf_audits = {}
    total_bytes = 0
    for name in sorted(expected):
        leaf = _create_output_budget(RUN_ROOT / name, create=False)
        audit = leaf.audit()
        if audit["bytes"] > OUTPUT_LIMIT_BYTES:
            raise R10Error(f"R10 output leaf {name} exceeded its pre-write byte ceiling")
        leaf_audits[name] = audit
        total_bytes += int(audit["bytes"])
    return {"status": "PASS", "leaf_audits": leaf_audits,
            "accounted_bytes": total_bytes,
            "unaccounted_writers_detected": False}


def _run_worker(category: str, mode: str, identity: Mapping[str, Any],
                global_manifest_sha256: str) -> dict[str, Any]:
    from dataclasses import asdict
    from experiments.calibration.r1_runtime_supervisor import run_supervised_child
    output = RUN_ROOT / ("smoke" if mode == "SMOKE" else category)
    if output.exists():
        raise R10Error(f"Refusing to resume or overwrite R10 child output: {output}")
    preflight = _verify_disk_and_gpu_preflight(RUN_ROOT)
    env = dict(os.environ)
    env.update({
        "R10_CHILD_OUTPUT": str(output),
        "R10_CHILD_CATEGORY": category,
        "R10_CHILD_MODE": mode,
        "R10_CHILD_SOURCE_SHA256":
            identity["source_hashes"]["experiments/calibration/r10_signed_knn_baseline.py"],
        "R10_GLOBAL_MANIFEST_SHA256": global_manifest_sha256,
        "R10_CHILD_AUTHORIZATION_ID": "user-task:R10-signed-knn-baseline:2026-11-03",
        "R10_CUBLAS_WORKSPACE_CONFIG":
            str(identity["runtime_profile"]["cublas_workspace_config"]),
        "PYTHONDONTWRITEBYTECODE": "1", "CUDA_CACHE_DISABLE": "1",
    })
    if AUTHORIZED_CUDA_VISIBILITY is None:
        env.pop("CUDA_VISIBLE_DEVICES", None)
    else:
        env["CUDA_VISIBLE_DEVICES"] = AUTHORIZED_CUDA_VISIBILITY
    result = run_supervised_child(
        [str(PYTHON), "-m", "experiments.calibration.r10_signed_knn_baseline",
         "--child", "--category", category, "--mode", mode],
        timeout_seconds=SMOKE_DEADLINE_SECONDS if mode == "SMOKE" else FULL_DEADLINE_SECONDS,
        env=env, cwd=ROOT, output_dir=None)
    if not output.exists():
        budget = _create_output_budget(output, create=True)
        _write_failure(budget, category=category, phase="supervisor",
                       exception=R10Error(f"Child ended as {result.status}"))
    budget = _create_output_budget(output, create=False)
    budget.write_json("supervisor_status.json", {
        "schema": "r10.signed_knn.supervisor.v1",
        **asdict(result), "category": category, "mode": mode,
        "prelaunch_resource_check": preflight,
        "process_tree_containment": "Windows Job Object, suspended assignment, kill-on-close",
        "automatic_restart": False,
    })
    if result.status != "COMPLETED" or not result.termination_confirmed:
        if not (output / "failure.json").exists():
            _write_failure(budget, category=category, phase="supervisor",
                           exception=R10Error(f"Child ended as {result.status}"))
        raise R10Error(f"{category}/{mode} child did not complete: {result.status}")
    seal = _read_json(output / "stage_a_seal.json")
    return {"seal": seal, "supervisor": asdict(result), "preflight": preflight}


def _write_once(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    _reject_reparse_chain(path.parent, ROOT)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(text)
        stream.flush()
        os.fsync(stream.fileno())


def run_r10() -> dict[str, Any]:
    from experiments.calibration import r7_shuffled_correction_4class as r7
    from experiments.calibration import r3_fullstream_2x2 as r3
    started = time.perf_counter()
    _assert_repo_identity()
    if RUN_ROOT.exists() or REPORT_DIR.exists():
        raise R10Error("R10 namespace or public deliverable already exists; refusing overwrite/resume")
    contract = r7._frozen_contract()
    refs = _load_r9_reference(contract)
    sources = _source_hashes()
    if sources.get("experiments/calibration/r9_correction_mechanism_falsification.py") != \
            refs["identity"]["runner_source_sha256"]:
        raise R10Error("Current R9 runner differs from the sealed historical source identity")
    if sources.get("experiments/calibration/r1_runtime_supervisor.py") != \
            refs["source_fingerprints"].get("experiments/calibration/r1_runtime_supervisor.py"):
        raise R10Error("Windows Job Object supervisor differs from R9's pinned dependency")
    model_receipt = r3._get_verified_model_receipt()
    expected_runtime = refs["references"][CATEGORIES[0]]["completion"]["runtime_profile"]
    if (model_receipt.get("hub_source_tree_sha256") !=
            expected_runtime.get("hub_source_tree_sha256") or
            model_receipt.get("checkpoint_sha256") !=
            expected_runtime.get("checkpoint_sha256") or
            model_receipt.get("loaded_model_state_sha256") !=
            expected_runtime.get("loaded_model_state_sha256")):
        raise R10Error("Prelaunch DINO source/weight fingerprint differs from sealed R9")
    output_parent = RUN_ROOT.parent
    _reject_reparse_chain(output_parent)
    if not output_parent.is_dir():
        raise R10Error("R10 output parent is missing")
    resource_preflight = _verify_disk_and_gpu_preflight(output_parent)
    RUN_ROOT.mkdir(parents=False, exist_ok=False)
    _reject_reparse_chain(RUN_ROOT, output_parent)
    identity = _r10_identity(refs, sources, resource_preflight, model_receipt)
    global_budget = _create_output_budget(RUN_ROOT / "global", create=True)
    global_manifest = {
        "schema": RUN_SCHEMA, "run_id": RUN_ID, "branch": EXPECTED_BRANCH,
        "source_parent_commit": EXPECTED_PARENT, "identity": identity,
        "identity_sha256": identity["identity_sha256"],
        "status": "IDENTITY_LOCKED", "automatic_restart": False,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    global_budget.write_json("run_manifest.json", global_manifest)
    global_manifest_sha = _sha_file(RUN_ROOT / "global" / "run_manifest.json")
    try:
        smoke_result = _run_worker("audiojack", "SMOKE", identity, global_manifest_sha)
        smoke_seal = smoke_result["seal"]
        if (smoke_seal.get("status") != "SMOKE_PASS_INCOMPLETE_NOT_EVALUATED" or
                smoke_seal.get("event_count") != TEST_SMOKE_EVENTS or
                smoke_seal.get("query_count") != 0):
            raise R10Error("R10 GPU smoke did not pass its fixed no-label contract")
        global_budget.append_jsonl("attempt_history.jsonl", {
            "schema": "r10.signed_knn.worker.v1", "mode": "SMOKE",
            "category": "audiojack", "status": "COMPLETED",
            "elapsed_seconds": smoke_result["supervisor"]["elapsed_seconds"],
            "automatic_restart": False,
        })
        full_results: dict[str, Any] = {}
        for category in CATEGORIES:
            worker = _run_worker(category, "FULL", identity, global_manifest_sha)
            if worker["seal"].get("status") != "SEALED_COMPLETE":
                raise R10Error(f"{category} Stage A child did not produce a full seal")
            full_results[category] = worker
            global_budget.append_jsonl("attempt_history.jsonl", {
                "schema": "r10.signed_knn.worker.v1", "mode": "FULL",
                "category": category, "status": "COMPLETED",
                "elapsed_seconds": worker["supervisor"]["elapsed_seconds"],
                "score_rows_sha256": worker["seal"]["score_rows_sha256"],
                "automatic_restart": False,
            })
        stage_a_global = _stage_a_global_seal(refs, identity, {
            "status": smoke_seal["status"], "event_count": smoke_seal["event_count"],
            "test_dino_encodings": smoke_seal["test_dino_encodings"],
            "g_score_parity": smoke_seal["g_score_parity"],
            "supervisor": smoke_result["supervisor"],
        })
        global_budget.write_json("stage_a_seal.json", stage_a_global)
        aggregate, stage_b_seal = _stage_b(refs, identity, stage_a_global, contract)
        global_budget.write_json("stage_b_seal.json", stage_b_seal)
        elapsed_seconds = time.perf_counter() - started
        aggregate_public = dict(aggregate)
        aggregate_public["source_identity"] = {
            "r10_identity_sha256": identity["identity_sha256"],
            "r9_identity_sha256": identity["r9_reference"]["identity_sha256"],
            "r9_stage_a_seal_sha256": identity["r9_reference"]["seal_hashes"]["stage_a"],
            "r10_stage_a_seal_sha256": _sha_file(RUN_ROOT / "global" / "stage_a_seal.json"),
            "r10_stage_b_seal_sha256": _sha_file(RUN_ROOT / "global" / "stage_b_seal.json"),
        }
        aggregate_public["source_fingerprints"] = dict(identity["source_hashes"])
        aggregate_public["model_fingerprints"] = dict(identity["model_fingerprints"])
        aggregate_public["efficiency"] = {
            "verified_full_test_feature_reuse": False,
            "full_test_dino_encodings": sum(EXPECTED_COUNTS.values()),
            "smoke_test_dino_encodings": int(smoke_seal["test_dino_encodings"]),
            "full_stream_gpu_passes": 4, "full_c1_reruns": 0,
            "new_c64_dino_encodings": sum(
                int(full_results[category]["seal"].get("actual_new_c64_dino_encodings", 0))
                for category in CATEGORIES),
            "support_feature_encoder_invocations": sum(
                int(full_results[category]["seal"].get("support_feature_encoder_invocations", 0))
                for category in CATEGORIES),
            "elapsed_seconds": elapsed_seconds,
            "category_elapsed_seconds": {
                category: float(full_results[category]["supervisor"]["elapsed_seconds"])
                for category in CATEGORIES},
        }
        aggregate_public["g_parity"] = stage_a_global["g_parity"]
        aggregate_payload = _canonical(aggregate_public)
        global_budget.write_once("aggregate_results.json", aggregate_payload)
        report = _build_report(
            aggregate_public, stage_a_global, stage_b_seal, identity,
            smoke_seal, resource_preflight, aggregate_public["efficiency"]["elapsed_seconds"])
        _write_once(AGGREGATE_PATH, aggregate_payload.decode("utf-8"))
        _write_once(REPORT_PATH, report)
        output_audit_before_complete = _audit_output_namespace()
        if output_audit_before_complete["status"] != "PASS":
            raise R10Error("R10 global private output exceeded its byte ceiling")
        complete = {
            "schema": "r10.signed_knn.complete_seal.v1",
            "status": "COMPLETE", "run_id": RUN_ID,
            "identity_sha256": identity["identity_sha256"],
            "stage_a_seal_sha256": _sha_file(RUN_ROOT / "global" / "stage_a_seal.json"),
            "stage_b_seal_sha256": _sha_file(RUN_ROOT / "global" / "stage_b_seal.json"),
            "aggregate_results_sha256": _sha_file(RUN_ROOT / "global" / "aggregate_results.json"),
            "public_aggregate_sha256": _sha_file(AGGREGATE_PATH),
            "public_report_sha256": _sha_file(REPORT_PATH),
            "total_elapsed_seconds": elapsed_seconds,
            "output_audit": {
                "status": output_audit_before_complete["status"],
                "accounted_bytes_before_complete_seal":
                    output_audit_before_complete["accounted_bytes"],
                "unaccounted_writers_detected": False,
            },
            "automatic_restart": False, "pushed": False,
        }
        global_budget.write_json("complete_seal.json", complete)
        output_audit_final = _audit_output_namespace()
        if (output_audit_final["status"] != "PASS" or
                global_budget.audit()["bytes"] > OUTPUT_LIMIT_BYTES):
            raise R10Error("R10 global private output exceeded its byte ceiling")
        return {"status": "COMPLETE", "identity": identity, "complete": complete,
                "stage_a": stage_a_global, "stage_b": stage_b_seal,
                "aggregate": aggregate_public, "smoke": smoke_seal,
                "resource_preflight": resource_preflight}
    except BaseException as exc:
        _write_failure(global_budget, category="all", phase="orchestrator", exception=exc)
        raise


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--authorize-r10-run", action="store_true")
    parser.add_argument("--child", action="store_true")
    parser.add_argument("--category", choices=CATEGORIES)
    parser.add_argument("--mode", choices=("SMOKE", "FULL"))
    args = parser.parse_args(argv)
    if args.child:
        if (os.environ.get("R10_CHILD_AUTHORIZATION_ID") !=
                "user-task:R10-signed-knn-baseline:2026-11-03" or
                not args.category or not args.mode):
            raise R10Error("Stage A child lacks the explicit R10 authorization identity")
        _run_stage_a_child(args.category, args.mode)
        return 0
    if not args.authorize_r10_run:
        raise R10Error("Explicit --authorize-r10-run is required")
    result = run_r10()
    print(json.dumps({
        "status": result["status"], "run_id": RUN_ID,
        "identity_sha256": result["identity"]["identity_sha256"],
        "complete_seal_sha256": _sha_file(RUN_ROOT / "global" / "complete_seal.json"),
        "stage_a_seal_sha256": _sha_file(RUN_ROOT / "global" / "stage_a_seal.json"),
        "stage_b_seal_sha256": _sha_file(RUN_ROOT / "global" / "stage_b_seal.json"),
        "elapsed_seconds": result["complete"]["total_elapsed_seconds"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

