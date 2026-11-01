"""R3-FULL-2X2: retrospective full-stream dual-memory factorial evaluation.

Stage A scores each event once per normal-memory state, seals all four arm
scores, then discloses only legal Q-A-TS feedback. Stage B is a separate CPU
child and cannot start until all four category trajectories are sealed.
"""
from __future__ import annotations

import argparse
import ast
import gzip
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[2]
OUTPUT_BASE = ROOT / "outputs" / "calibration"
STAGE05 = OUTPUT_BASE / "stage05_exploratory_20261029"
EXP2A = OUTPUT_BASE / "r1_digest_contract_v1_exp2a_r2exp2a_20261101_a4"
MODEL_RECEIPT = OUTPUT_BASE / "r1_digest_contract_v1_engineering_r1s4b-switch-s0-512-20261031-a1"
REPORT_DIR = ROOT / "docs" / "calibration" / "audit" / "r3_fullstream_2x2_20261102"
DATASET_ROOT = Path(r"G:\dataset\realiad")
JSON_ROOT = Path(r"G:\dataset\realiad\jsons\realiad_jsons")
EXPECTED_BASELINE = "b23f6956b6dd3e8359d27050f3fe27713ef3a69e"
CATEGORIES = ("audiojack", "regulator", "switch", "rolled_strip_base")
EXPECTED_COUNTS = {"audiojack": 3815, "regulator": 3855,
                   "switch": 3820, "rolled_strip_base": 3805}
ARMS = ("S", "SC", "G", "GC")
OUTPUT_LIMIT = 4 * 1024**3
FAILURE_RESERVE = 64 * 1024
MAX_FILES = 20
MIN_FREE_BYTES = 10 * 1024**3
TOTAL_DEADLINE_SECONDS = 8 * 60 * 60
CATEGORY_DEADLINE_SECONDS = 120 * 60
SMOKE_DEADLINE_SECONDS = 15 * 60
TEST_DEADLINE_SECONDS = 180
BOOTSTRAP_REPLICATES = 1000
BOOTSTRAP_SEED = 20261102
ATOL = 1e-5
RTOL = 1e-4
RUN_SCHEMA = "r3.fullstream.2x2.run.v1"
SCORE_SCHEMA = "r3.fullstream.2x2.score.v1"
HISTORY_SCHEMA = "r3.fullstream.2x2.history.v1"
STAGE_A_SEAL_SCHEMA = "r3.fullstream.2x2.stage_a_seal.v1"
STAGE_B_SEAL_SCHEMA = "r3.fullstream.2x2.stage_b_seal.v1"
AUTHORIZATION = "R3-FULL-2X2 user-authorized retrospective full-stream development evaluation"
CORRECTION_CONFIG = {
    "implementation": "frozen v22 SignedCorrectionCodebook / C1-Full",
    "capacity": 256, "corr_weight": 1.0,
    "beta_scale": 0.5, "beta_min": 0.02, "beta_max": 0.20,
    "outcome_weights": {"FP": 1.30, "FN": 1.20, "TN": 0.60, "TP": 0.80},
    "patch_beta_floor": 0.75, "patch_beta_cap": 1.25, "patch_beta_power": 1.0,
    "radius0": 0.10, "radius_min": 0.05, "radius_max": 0.25,
    "kappa0": 1.0, "kappa_inc": 0.10, "kappa_max": 5.0,
    "merge_eps": 0.05, "min_dN_for_store_fp": 0.10,
    "min_dN_for_store_fn": 0.12, "sign_aware_write": True,
    "patch_weighted_beta": True, "sign_aware_retention": True,
    "corr_replace_margin": 0.0, "keep_defect_bank": False,
}
KNOWN_R1_HUB_TREE_SHA256 = "b7c0037f90671c4932393bd56de8c8a205b29a6eb22dea563b6c8a88c471da15"
KNOWN_CHECKPOINT_SHA256 = "36e4deffbaef061a2576705b0c36f93621e2ae20bf6274694821b0b492551b51"
KNOWN_R1_MODEL_STATE_SHA256 = "1f26464ba4055403e550f1d5c1333fb012631d98661df6993cabea730d17e3f9"
# This pre-repair runner sealed every Stage A score but its Stage B restricted
# label reader rejected legal trailing whitespace after the final TEST object.
# Reuse is permitted only for that exact runner SHA after matching every frozen
# input fingerprint and the complete category receipts below.
STAGE_A_REUSE_COMPATIBLE_RUNNERS = {
    "06df65ef04d2c63468becd3fb11939f9859d4a327b5f381c22c68c6b3d26c991":
        "stage-b-projection-and-parent-reporting-repairs-only",
}
RELEVANT_SOURCES = (
    "experiments/calibration/stage05.py",
    "experiments/calibration/stage1a_p12_label_provider.py",
    "experiments/calibration/r1_matched_correction_pilot.py",
    "experiments/calibration/r2_exp2a_c1_strength.py",
    "experiments/calibration/r2_feedback_repair.py",
    "experiments/calibration/r1_runtime_supervisor.py",
    "experiments/calibration/native.py",
    "experiments/calibration/profiles.py",
    "experiments/calibration/query_tape.py",
    "experiments/calibration/controllers.py",
    "experiments/p0_protocol_audit.py",
    "experiments/risk_safe/native_bridge.py",
    "experiments/v02_memory.py",
    "test_phase3_AIF_simplified_v22.py",
    "vmb_visionad_new.py",
)
_OUTPUT_NAMES = {
    "run_manifest.json", "attempt_history.jsonl", "stage_a_scores.jsonl",
    "stage_a_seal.json", "stage_b_seal.json", "stage_a_reuse_receipt.json",
    "aggregate_results.json", "complete_seal.json", "supervisor_status.json",
    "failure.json",
}


class R3Error(RuntimeError):
    pass


def canonical_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=True, sort_keys=True,
                       separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while True:
            block = stream.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def sha256_json(value: Any) -> str:
    return sha256_bytes(canonical_bytes(value).rstrip(b"\n"))


def _git(*args: str) -> str:
    result = subprocess.run(["git", *args], cwd=ROOT, check=True,
                            capture_output=True, text=True)
    return result.stdout.strip()


def _check_repository_identity() -> dict[str, str]:
    branch = _git("branch", "--show-current")
    head = _git("rev-parse", "HEAD")
    if branch != "calibration-stage0-contract-20261029":
        raise R3Error(f"Unexpected branch: {branch}")
    if head != EXPECTED_BASELINE:
        raise R3Error(f"Expected source parent {EXPECTED_BASELINE}; found {head}")
    changed = _git("status", "--porcelain", "--", *RELEVANT_SOURCES)
    if changed:
        raise R3Error(f"Relevant tracked experiment sources are dirty: {changed}")
    return {"branch": branch, "source_parent_commit": head}


def _source_fingerprints() -> dict[str, str]:
    return {name: sha256_file(ROOT / name) for name in RELEVANT_SOURCES}


def _verify_source_binding(manifest: Mapping[str, Any]) -> None:
    _check_repository_identity()
    if (manifest.get("source_parent_commit") != EXPECTED_BASELINE or
            manifest.get("runner_source_sha256") != sha256_file(Path(__file__).resolve()) or
            manifest.get("source_fingerprints") != _source_fingerprints()):
        raise R3Error("Run manifest source fingerprint differs from the executing child")


def _verify_reusable_stage_a_binding(manifest: Mapping[str, Any]) -> str:
    """Verify frozen Stage A inputs for a narrowly versioned Stage B-only repair."""
    _check_repository_identity()
    runner_sha = str(manifest.get("runner_source_sha256", ""))
    if runner_sha == sha256_file(Path(__file__).resolve()):
        _verify_source_binding(manifest)
        return "same-runner-source"
    compatibility = STAGE_A_REUSE_COMPATIBLE_RUNNERS.get(runner_sha)
    if compatibility is None:
        raise R3Error("Completed Stage A source is not in the explicit reuse compatibility contract")
    if (manifest.get("source_parent_commit") != EXPECTED_BASELINE or
            manifest.get("source_fingerprints") != _source_fingerprints()):
        raise R3Error("Stage A reuse source dependencies differ from the frozen run")
    return compatibility


def _identity_sha_with_runner(identity: Mapping[str, Any], runner_sha: str) -> str:
    normalized = dict(identity)
    normalized.pop("identity_sha256", None)
    normalized["runner_source_sha256"] = str(runner_sha)
    return sha256_json(normalized)


def _overall_wall_seconds(started: float, reused_category_seconds: float) -> float:
    return float(reused_category_seconds) + max(0.0, time.monotonic() - float(started))


def _output_path(run_id: str) -> Path:
    if not re.fullmatch(r"r3_fullstream_2x2_[a-zA-Z0-9_-]{3,80}", run_id):
        raise R3Error("Invalid R3 run ID")
    path = OUTPUT_BASE / run_id
    if path.parent.resolve(strict=True) != OUTPUT_BASE.resolve(strict=True):
        raise R3Error("R3 output directory escaped outputs/calibration")
    if path.exists():
        raise FileExistsError(f"Refusing output reuse: {path}")
    return path


def _existing_output_path(run_id: str) -> Path:
    """Resolve an already-created R3 namespace for a supervised child only."""
    if not re.fullmatch(r"r3_fullstream_2x2_[a-zA-Z0-9_-]{3,80}", run_id):
        raise R3Error("Invalid R3 run ID")
    parent = OUTPUT_BASE.resolve(strict=True)
    path = OUTPUT_BASE / run_id
    if path.is_symlink() or not path.is_dir():
        raise R3Error("Existing R3 output path is missing or unsafe")
    if path.resolve(strict=True).parent != parent:
        raise R3Error("Existing R3 output directory escaped outputs/calibration")
    return path


def _budget(root: Path, *, create: bool = False):
    from experiments.calibration.r2_feedback_repair import (
        OutputBudget, MAX_OUTPUT_FILES, FAILURE_RESERVE_BYTES)
    return OutputBudget(root, create=create, max_bytes=OUTPUT_LIMIT,
                        failure_reserve_bytes=FAILURE_RESERVE,
                        max_files=MAX_FILES)


def _restore_cuda_visibility(value: str | None) -> None:
    """Undo the CPU-only R2 module's process-wide CUDA visibility side effect."""
    if value is None:
        os.environ.pop("CUDA_VISIBLE_DEVICES", None)
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = value


def _child_environment(base: Mapping[str, str], stage: str,
                       authorized_cuda_visibility: str | None) -> dict[str, str]:
    env = dict(base)
    if stage == "STAGE_A":
        _restore = authorized_cuda_visibility
        if _restore is None:
            env.pop("CUDA_VISIBLE_DEVICES", None)
        else:
            env["CUDA_VISIBLE_DEVICES"] = _restore
        env.pop("R3_STAGE_B_CPU_ONLY", None)
    elif stage == "STAGE_B":
        env["CUDA_VISIBLE_DEVICES"] = ""
        env["R3_STAGE_B_CPU_ONLY"] = "1"
    else:
        raise R3Error(f"Unsupported supervised child stage: {stage}")
    env["PYTHONHASHSEED"] = "0"
    env["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    env["HF_HUB_OFFLINE"] = "1"
    env["TRANSFORMERS_OFFLINE"] = "1"
    return env


class BufferedJsonl:
    """Append only through OutputBudget; flush at query/durable boundaries."""

    def __init__(self, budget: Any, name: str):
        if name not in _OUTPUT_NAMES:
            raise R3Error(f"Output name is not authorized: {name}")
        self.budget, self.name = budget, name
        self.pending: list[bytes] = []

    def add(self, row: Mapping[str, Any]) -> bytes:
        payload = canonical_bytes(dict(row))
        self.pending.append(payload)
        return payload

    def flush(self) -> None:
        if not self.pending:
            return
        payload = b"".join(self.pending)
        if self.name in self.budget.created:
            self.budget.append(self.name, payload)
        else:
            self.budget.write_once(self.name, payload)
        self.pending.clear()


def _disclose_query_label_after_seal(scores_out: BufferedJsonl, *,
                                    query_decision: bool, category: str,
                                    position: int, sample_key: str,
                                    label_reader: Any) -> int | None:
    """Never call the label reader for an unqueried event; fsync current scores first."""
    if not query_decision:
        return None
    if label_reader is None:
        raise R3Error("Frozen query position has no restricted label reader")
    scores_out.flush()
    return int(label_reader(category, int(position), str(sample_key)))


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise R3Error(f"Expected JSON object: {path.name}")
    return value


def _load_exp2a_score_references() -> tuple[dict[str, Any], dict[tuple[str, int], dict[str, Any]]]:
    """Read only sealed EXP2A score rows, never feedback or update diagnostics."""
    run_manifest = _read_json(EXP2A / "run_manifest.json")
    stage_a = _read_json(EXP2A / "stage_a_seal.json")
    experiment = _read_json(EXP2A / "experiment_seal.json")
    scores_path = EXP2A / "event_scores.jsonl"
    if (experiment.get("status") != "COMPLETE" or
            stage_a.get("status") != "SEALED_COMPLETE" or
            experiment.get("stage_a_seal_sha256") != sha256_file(EXP2A / "stage_a_seal.json") or
            stage_a.get("score_rows_sha256") != sha256_file(scores_path)):
        raise R3Error("EXP2A first-512 score trajectory lacks a valid complete seal")
    if run_manifest.get("run_id") != "r2exp2a_20261101_a4":
        raise R3Error("EXP2A input is not the exact authorized a4 run")
    counts = stage_a.get("category_event_counts", {})
    if any(int(counts.get(cat, -1)) != 512 for cat in CATEGORIES):
        raise R3Error("EXP2A score seal is not 512 rows for each category")
    refs: dict[tuple[str, int], dict[str, Any]] = {}
    with scores_path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            row = json.loads(line)
            category, position = str(row.get("category")), int(row.get("position", -1))
            if category not in CATEGORIES or not 0 <= position < 512:
                raise R3Error(f"Malformed EXP2A score row {line_number}")
            if row.get("schema") != "r2.exp2a.score_row.v1":
                raise R3Error("EXP2A score row schema mismatch")
            if "label" in row or "anomaly_class" in row:
                raise R3Error("EXP2A score reference unexpectedly contains labels")
            key = (category, position)
            if key in refs:
                raise R3Error(f"Duplicate EXP2A score row: {category}:{position}")
            if not {"c0", "c1_full"}.issubset(row.get("scores", {})):
                raise R3Error("EXP2A score row lacks C0/C1-Full references")
            refs[key] = row
    if len(refs) != 4 * 512:
        raise R3Error("EXP2A reference set is incomplete")
    contract = run_manifest.get("common_trajectory_contract", {})
    for claim in ("C0 map/query/prediction/common Greedy writer shared",
                  "C0 prediction defines all arm outcomes",
                  "C1 Codebook state independent by arm",
                  "one feature extraction per event shared by all arms",
                  "scores sealed before queried label disclosure"):
        if contract and contract.get(claim) is not True:
            raise R3Error(f"EXP2A causal receipt failed: {claim}")
    if contract and (contract.get("masks accessed") is not False or
                     contract.get("unqueried labels accessed") is not False):
        raise R3Error("EXP2A causal receipt reports unauthorized access")
    return {"stage_a_seal_sha256": sha256_file(EXP2A / "stage_a_seal.json"),
            "score_rows_sha256": sha256_file(scores_path),
            "experiment_seal_sha256": sha256_file(EXP2A / "experiment_seal.json")}, refs


def load_frozen_contract(*, verify_image_paths: bool = True) -> dict[str, Any]:
    """Load only frozen identities, score-only tapes, and C64 support scores."""
    import numpy as np
    from experiments.calibration.stage05 import _absolute_image_path
    from experiments.calibration.controllers import theta0_from_c64
    from experiments.calibration.profiles import QueryProtocolConfig, canonical_json as stage05_canonical_json

    if not (STAGE05 / "run_manifest.json").is_file():
        raise R3Error("Frozen Stage 0.5 inputs are missing")
    run = _read_json(STAGE05 / "run_manifest.json")
    frozen = _read_json(STAGE05 / "frozen_inputs.json")
    score_manifest = _read_json(STAGE05 / "score_tape_manifest.json")
    if (run.get("overall_status") != "EXPLORATORY_AUDIT_COMPLETE_NOT_STAGE1_VALIDATION" or
            run.get("confirmation_classes_opened") is not False or
            frozen.get("schema") != "stage05.frozen_inputs.v1" or
            frozen.get("freeze_verification_status") != "PASS" or
            frozen.get("stage1_or_stage2_run") is not False):
        raise R3Error("Stage 0.5 frozen-input receipt is incompatible")
    query_path = STAGE05 / "query_tapes.json.gz"
    if (sha256_file(query_path) != run.get("query_tape_archive_sha256") or
            sha256_file(query_path) != frozen.get("query_tapes_file_sha256")):
        raise R3Error("Frozen Stage 0.5 query tape archive hash mismatch")
    if score_manifest.get("status") != "PASS" or score_manifest.get("controller_or_test_label_access") is not False:
        raise R3Error("Stage 0.5 native score manifest is not sealed")
    with gzip.open(query_path, "rt", encoding="utf-8") as stream:
        tapes = json.load(stream)
    if set(tapes) != set(CATEGORIES):
        raise R3Error("Frozen Q-A-TS archive has an unexpected category inventory")

    out: dict[str, Any] = {
        "stage05_run_manifest_sha256": sha256_file(STAGE05 / "run_manifest.json"),
        "frozen_inputs_sha256": sha256_file(STAGE05 / "frozen_inputs.json"),
        "score_tape_manifest_sha256": sha256_file(STAGE05 / "score_tape_manifest.json"),
        "query_tape_archive_sha256": sha256_file(query_path),
        "query_tape_file_sha256_receipt": frozen["query_tapes_file_sha256"],
        "stage05_model_provenance": score_manifest.get("model_provenance", {}),
        "stage05_scoring_source_hashes": score_manifest.get("scoring_source_hashes", {}),
        "stage05_current_source_hashes": {},
        "classes": {},
    }
    for rel, expected_hash in score_manifest.get("scoring_source_hashes", {}).items():
        source_path = ROOT / rel
        if source_path.is_file():
            actual_hash = sha256_file(source_path)
            out["stage05_current_source_hashes"][rel] = actual_hash
    for category in CATEGORIES:
        info = frozen["classes"][category]
        expected_n = EXPECTED_COUNTS[category]
        stream_rows = info.get("stream", [])
        if len(stream_rows) != expected_n or int(info.get("stream_images", -1)) != expected_n:
            raise R3Error(f"{category}: frozen TEST event count differs from protocol")
        if (info.get("support_c64_disjoint") is not True or
                info.get("train_test_groups_disjoint") is not True or
                len(info["supports"]["0"]["images"]) != 4 or
                int(info["c64"]["count"]) != 64):
            raise R3Error(f"{category}: frozen Support/C64 identity contract failed")
        class_json = JSON_ROOT / f"{category}.json"
        class_sha = sha256_file(class_json)
        if class_sha != info.get("class_json_sha256"):
            raise R3Error(f"{category}: current JSON source differs from frozen public identity")

        tape = tapes[category]["0.03"]["37"]
        expected_tape = frozen["tape_digests_by_seed37"][category]["0.03"]
        tape_payload = {
            "schema": "calibration.query_tape.v1",
            "policy": "Q-A-TS",
            "config": asdict(QueryProtocolConfig()),
            "n_images": expected_n,
            "n_groups": len({str(row.get("opaque_group", "")) for row in stream_rows}),
            "requested_budget": int(tape["requested_budget"]),
            "early_budget": 0,
            "stream_digest": str(tape["stream_digest"]),
            "bin_edges": list(tape["bin_edges"]),
            "bin_quotas": list(tape["bin_quotas"]),
            "events": list(tape["events"]),
        }
        observed_digest = hashlib.sha256(
            stage05_canonical_json(tape_payload).encode("utf-8")).hexdigest()
        if tape.get("digest") != expected_tape or observed_digest != expected_tape:
            raise R3Error(f"{category}: existing seed-37 Q-A-TS digest differs from frozen receipt")
        positions: list[int] = []
        group_by_position: dict[int, str] = {}
        for event in tape.get("events", []):
            position = int(event.get("position", -1))
            if not 0 <= position < expected_n or position in group_by_position:
                raise R3Error(f"{category}: malformed or duplicate frozen query position")
            row = stream_rows[position]
            group = str(row.get("opaque_group", ""))
            identity_hash = sha256_bytes(
                ("image-identity\0" + str(row.get("relative_id", ""))).encode("utf-8"))
            if (group != event.get("official_sample_key") or
                    int(row.get("camera_id", -1)) != int(event.get("camera_id", -2)) or
                    identity_hash != event.get("relative_id")):
                raise R3Error(f"{category}:{position}: Q-A-TS identity differs from frozen stream")
            positions.append(position)
            group_by_position[position] = group
        budget = int(math.floor(0.03 * expected_n + 0.5))
        if (int(tape.get("requested_budget", -1)) != budget or len(positions) != budget or
                len(set(group_by_position.values())) != budget):
            raise R3Error(f"{category}: existing Q-A-TS query budget/Source Group rule failed")
        positions.sort()

        c64_row = score_manifest["classes"][category]["score_tapes"]["0"]
        c64_path = (STAGE05 / str(c64_row["file"])).resolve(strict=True)
        if STAGE05.resolve(strict=True) not in c64_path.parents:
            raise R3Error(f"{category}: C64 score tape escaped frozen Stage 0.5")
        if (int(c64_row.get("c64_score_count", -1)) != 64 or
                sha256_file(c64_path) != c64_row.get("sha256") or
                c64_row.get("c64_group_sha256") != info["c64"]["group_sha256"]):
            raise R3Error(f"{category}: C64 score receipt is missing or checksum-invalid")
        with np.load(c64_path, allow_pickle=False) as archive:
            c64_scores = np.asarray(archive["c64_scores"], dtype=np.float64)
            c64_identity = str(archive["c64_identity_sha256"].item())
        if (c64_scores.shape != (64,) or not np.isfinite(c64_scores).all() or
                c64_identity != info["c64"]["image_sha256"]):
            raise R3Error(f"{category}: C64 score identity/values are invalid")
        theta = float(theta0_from_c64(c64_scores.tolist()))
        sigma = max(float(c64_scores.std(ddof=1)), 0.02)

        supports = info["supports"]["0"]
        support_groups = {
            sha256_bytes(("official-sample-group\0" + str(group)).encode("utf-8"))
            for group in supports["groups"]}
        c64_groups = {
            sha256_bytes(("official-sample-group\0" + str(group)).encode("utf-8"))
            for group in info["c64"]["groups"]}
        stream_groups = {str(row.get("opaque_group", "")) for row in stream_rows}
        if (len(support_groups) != 4 or len(c64_groups) != 64 or
                support_groups & c64_groups or support_groups & stream_groups or
                c64_groups & stream_groups):
            raise R3Error(f"{category}: Support/C64/TEST Source Groups are not disjoint")
        support_paths = [Path(_absolute_image_path(DATASET_ROOT, info["prefix"], rel))
                         for rel in supports["images"]]
        if verify_image_paths and any(not path.is_file() for path in support_paths):
            raise R3Error(f"{category}: a frozen Support image is missing")
        event_paths = [Path(_absolute_image_path(DATASET_ROOT, info["prefix"],
                                                  str(row["relative_id"])))
                       for row in stream_rows]
        if verify_image_paths and any(not path.is_file() for path in event_paths):
            raise R3Error(f"{category}: a frozen TEST image is missing")

        out["classes"][category] = {
            "category_json_sha256": class_sha,
            "frozen_class_json_sha256": info["class_json_sha256"],
            "stream_count": expected_n,
            "stream_rows": stream_rows,
            "event_paths": event_paths,
            "prefix": str(info["prefix"]),
            "support_paths": support_paths,
            "support_image_sha256": supports.get("image_sha256"),
            "support_group_sha256": sha256_json(sorted(support_groups)),
            "c64_image_sha256": info["c64"]["image_sha256"],
            "c64_group_sha256": info["c64"]["group_sha256"],
            "c64_score_tape_sha256": c64_row["sha256"],
            "fixed_support_sha256_expected": str(c64_row["fixed_support_sha256"]),
            "theta_c64_q90": theta,
            "sigma_c64_floor_002": sigma,
            "query_positions": positions,
            "query_positions_sha256": sha256_json(positions),
            "query_group_sha256_by_position": group_by_position,
            "query_tape_sha256": expected_tape,
            "query_count": budget,
        }
    return out


def _source_group_hash(value: str) -> str:
    return sha256_bytes(("official-sample-group\0" + value).encode("utf-8"))


def _event_id(category: str, position: int, rel_id: str,
              image_sha: str, group_sha: str) -> str:
    return sha256_json({"category": category, "position": int(position),
                        "relative_id_sha256": sha256_bytes(rel_id.encode("utf-8")),
                        "image_sha256": image_sha, "source_group_sha256": group_sha})


def _write_failure(root: Path, stage: str, exc: BaseException,
                   details: Mapping[str, Any] | None = None) -> None:
    if not root.is_dir():
        return
    try:
        budget = _budget(root)
        budget.write_json("failure.json", {
            "schema": "r3.fullstream.2x2.failure.v1",
            "stage": stage, "failure_type": type(exc).__name__,
            "message": str(exc)[:2000], "finished_utc": time.strftime(
                "%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "automatic_restart": False, "details": dict(details or {}),
        }, failure_reserved=True)
    except Exception:
        pass


def _append_history(budget: Any, row: Mapping[str, Any]) -> bytes:
    payload = canonical_bytes(dict(row))
    if "attempt_history.jsonl" in budget.created:
        budget.append("attempt_history.jsonl", payload)
    else:
        budget.write_once("attempt_history.jsonl", payload)
    return payload


def _runtime_and_native_setup(torch: Any, impl: Any, expected_model: Mapping[str, Any]) -> tuple[Any, dict[str, Any]]:
    """Configure the historical math-only runtime and force cached local DINO sources."""
    runtime = __import__(
        "experiments.calibration.r1_matched_correction_pilot",
        fromlist=["_configure_historical_runtime"],
    )._configure_historical_runtime(impl, torch)
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise R3Error("Expected exactly one authorized CUDA device")
    runtime["cuda_device_index"] = 0
    expected_runtime = expected_model.get("runtime_profile", {})
    if runtime["cuda_device_name"] != expected_runtime.get("cuda_device_name"):
        raise R3Error("CUDA device differs from the verified local R1 execution receipt")
    if str(torch.__version__) != expected_runtime.get("torch_version"):
        raise R3Error("PyTorch version differs from the frozen local runtime")

    hub_root = Path(torch.hub.get_dir())
    candidates = sorted(p for p in hub_root.glob("facebookresearch_dinov2*")
                        if p.is_dir() and (p / "hubconf.py").is_file())
    if len(candidates) != 1:
        raise R3Error("Expected one already-cached local DINOv2 source tree")
    local_repo = candidates[0]
    hub_path, hub_tree_sha = __import__(
        "experiments.calibration.r1_matched_correction_pilot",
        fromlist=["_hub_source_sha256"],
    )._hub_source_sha256(hub_root)
    if (hub_tree_sha != KNOWN_R1_HUB_TREE_SHA256 or
            hub_tree_sha != expected_model.get("hub_source_tree_sha256")):
        raise R3Error("Local DINO source fingerprint differs from the verified R1 receipt")
    checkpoint = hub_root / "checkpoints" / "dinov2_vitl14_reg4_pretrain.pth"
    checkpoint_sha = sha256_file(checkpoint) if checkpoint.is_file() else None
    if (checkpoint_sha != KNOWN_CHECKPOINT_SHA256 or
            checkpoint_sha != expected_model.get("checkpoint_sha256")):
        raise R3Error("Local DINO checkpoint is missing or fingerprint-incompatible")

    original_load = torch.hub.load
    original_state_load = torch.hub.load_state_dict_from_url
    original_download = torch.hub.download_url_to_file

    def local_load(repo_or_dir: Any, model: str, *args: Any, **kwargs: Any):
        if str(repo_or_dir) not in {"facebookresearch/dinov2", str(local_repo)}:
            raise R3Error(f"Unexpected torch.hub source requested: {repo_or_dir}")
        kwargs["source"] = "local"
        return original_load(str(local_repo), model, *args, **kwargs)

    def cached_weights(url: str, *args: Any, **kwargs: Any):
        filename = Path(str(url).split("?", 1)[0]).name
        local_file = checkpoint.parent / filename
        if filename != checkpoint.name or not local_file.is_file():
            raise R3Error("DINO requested a weight file outside the verified local checkpoint")
        kwargs["model_dir"] = str(checkpoint.parent)
        return original_state_load(url, *args, **kwargs)

    def deny_download(*_args: Any, **_kwargs: Any) -> None:
        raise R3Error("Network weight/source download is disabled for this experiment")

    torch.hub.load = local_load
    torch.hub.load_state_dict_from_url = cached_weights
    torch.hub.download_url_to_file = deny_download
    return (lambda: _restore_hub(torch, original_load, original_state_load,
                                 original_download)), {
        "runtime_profile": runtime,
        "hub_source_name": Path(hub_path).name,
        "hub_source_tree_sha256": hub_tree_sha,
        "checkpoint_name": checkpoint.name,
        "checkpoint_sha256": checkpoint_sha,
        "expected_model_state_sha256": expected_model.get("loaded_model_state_sha256"),
    }


def _restore_hub(torch: Any, load: Any, state_load: Any, download: Any) -> None:
    torch.hub.load = load
    torch.hub.load_state_dict_from_url = state_load
    torch.hub.download_url_to_file = download


def _get_verified_model_receipt() -> dict[str, Any]:
    receipt = _read_json(MODEL_RECEIPT / "run_manifest.json")
    if (receipt.get("status") != "ENGINEERING_PASS" or
            receipt.get("task") != "R1" or
            receipt.get("profile_id") != "r1_switch_s0_512_digest_contract_v1" or
            receipt.get("hub_source_tree_sha256") != KNOWN_R1_HUB_TREE_SHA256 or
            receipt.get("checkpoint_sha256") != KNOWN_CHECKPOINT_SHA256 or
            receipt.get("loaded_model_state_sha256") != KNOWN_R1_MODEL_STATE_SHA256):
        raise R3Error("Known local R1 model/source verification receipt is absent or incompatible")
    return receipt


def _check_bank_receipt(bank: Any, p0: Any, expected: str,
                        r1: Any) -> dict[str, Any]:
    from experiments.calibration.native import _fixed_bank_digest
    evidence = r1._fixed_bank_digest_evidence(bank, p0, expected, _fixed_bank_digest)
    if not evidence["matches_expected_structured_digest"]:
        raise R3Error(
            "Stage 0.5 structured fixed-bank mismatch; "
            f"expected={evidence['expected_sha256']} actual={evidence['actual_structured_sha256']} "
            f"raw={evidence['actual_raw_prototype_sha256']} layers={evidence['layer_profile']}"
        )
    return evidence


def _native_config(p0: Any, profiles: Any, r1: Any) -> tuple[Any, Any]:
    scorer_profile = profiles.NativeScorerConfig()
    cfg = p0.Config(
        device="cuda", augmentation_seed=0, bank_seed=0,
        fixed_capacity=25000, fixed_per_image=5000, augmentations=4,
        include_identity=False, multiplier=2.5, feature_fuse="concat",
        topk_patches=32, image_aggregation="max", image_topk=64,
        image_quantile=0.95, knn=5,
    )
    r1._validate_p0_profile(cfg, scorer_profile)
    return cfg, scorer_profile


def _validate_correction_config(bank: Any) -> None:
    cfg = bank.corr_cfg
    expected = {
        "K_correction": CORRECTION_CONFIG["capacity"],
        "corr_weight": CORRECTION_CONFIG["corr_weight"],
        "beta_scale": CORRECTION_CONFIG["beta_scale"],
        "beta_min": CORRECTION_CONFIG["beta_min"],
        "beta_max": CORRECTION_CONFIG["beta_max"],
        "outcome_weight_fp": CORRECTION_CONFIG["outcome_weights"]["FP"],
        "outcome_weight_fn": CORRECTION_CONFIG["outcome_weights"]["FN"],
        "outcome_weight_tn": CORRECTION_CONFIG["outcome_weights"]["TN"],
        "outcome_weight_tp": CORRECTION_CONFIG["outcome_weights"]["TP"],
        "patch_beta_floor": CORRECTION_CONFIG["patch_beta_floor"],
        "patch_beta_cap": CORRECTION_CONFIG["patch_beta_cap"],
        "patch_beta_power": CORRECTION_CONFIG["patch_beta_power"],
        "radius0": CORRECTION_CONFIG["radius0"],
        "radius_min": CORRECTION_CONFIG["radius_min"],
        "radius_max": CORRECTION_CONFIG["radius_max"],
        "kappa0": CORRECTION_CONFIG["kappa0"],
        "kappa_inc": CORRECTION_CONFIG["kappa_inc"],
        "kappa_max": CORRECTION_CONFIG["kappa_max"],
        "merge_eps": CORRECTION_CONFIG["merge_eps"],
        "min_dN_for_store_fp": CORRECTION_CONFIG["min_dN_for_store_fp"],
        "min_dN_for_store_fn": CORRECTION_CONFIG["min_dN_for_store_fn"],
        "corr_replace_margin": CORRECTION_CONFIG["corr_replace_margin"],
        "enable_sign_aware_write": True,
        "enable_patch_weighted_beta": True,
        "enable_sign_aware_retention": True,
        "keep_defect_bank": False,
    }
    mismatches = {name: (getattr(cfg, name, None), value)
                  for name, value in expected.items()
                  if getattr(cfg, name, None) != value}
    if mismatches:
        raise R3Error(f"Native C1-Full configuration differs from the frozen profile: {mismatches}")


def _score_tolerance(actual: float, expected: float) -> bool:
    return abs(float(actual) - float(expected)) <= ATOL + RTOL * abs(float(expected))


def _run_category_child(run_id: str, category: str, *, smoke_events: int = 0) -> dict[str, Any]:
    cuda_visibility_before_imports = os.environ.get("CUDA_VISIBLE_DEVICES")
    import numpy as np
    import torch
    from types import SimpleNamespace
    from experiments import p0_protocol_audit as p0
    from experiments.calibration import stage05
    from experiments.calibration import r1_matched_correction_pilot as r1
    from experiments.calibration import r2_exp2a_c1_strength as exp2a
    from experiments.calibration.native import _fixed_bank_digest
    from experiments.risk_safe.native_bridge import dynamic_codebooks
    from v02_memory import MemoryArm

    if category not in CATEGORIES:
        raise R3Error("Stage A child received an unauthorized category")
    output = _existing_output_path(run_id)
    run_manifest = _read_json(output / "run_manifest.json")
    if (run_manifest.get("authorization") != AUTHORIZATION or
            run_manifest.get("source_parent_commit") != EXPECTED_BASELINE or
            run_manifest.get("mode") not in {"FULL_STAGE_A", "SMOKE_ONLY"}):
        raise R3Error("Stage A child identity/authorization does not match its parent manifest")
    _verify_source_binding(run_manifest)
    if run_manifest.get("mode") == "SMOKE_ONLY" and not 0 < smoke_events <= 256:
        raise R3Error("Smoke child event limit must be between 1 and 256")

    contract = load_frozen_contract(verify_image_paths=False)
    info = contract["classes"][category]
    output_budget = _budget(output)
    # r2_feedback_repair intentionally hides CUDA at import for CPU-only work.
    # Stage A is GPU-authorized, so restore the inherited selection after creating
    # its accountant and before the first CUDA visibility check.
    _restore_cuda_visibility(cuda_visibility_before_imports)
    scores_out = BufferedJsonl(output_budget, "stage_a_scores.jsonl")
    category_started = time.monotonic()
    query_receipt_digest = hashlib.sha256()
    category_score_digest = hashlib.sha256()
    query_positions = tuple(int(x) for x in info["query_positions"])
    event_limit = min(info["stream_count"], smoke_events or info["stream_count"])
    allowed_queries = tuple(position for position in query_positions if position < event_limit)
    group_map = {int(pos): str(value) for pos, value in
                 info["query_group_sha256_by_position"].items() if int(pos) in allowed_queries}
    label_reader = None
    backbone = None
    scorer_time = encoding_time = correction_time = writer_time = 0.0
    parity = {"checked_events": 0, "g_score_checks": 0, "gc_score_checks": 0,
              "g_max_abs_error": 0.0, "gc_max_abs_error": 0.0,
              "numeric_mismatches": 0, "initial_map_max_abs_error": 0.0,
              "initial_score_max_abs_error": 0.0}
    runtime_info: dict[str, Any] = {}
    counters = {
        "actual_image_encodings": 0, "query_events": 0,
        "query_normal": 0, "query_anomalous": 0,
        "greedy_write_events": 0, "greedy_accepted_patch_count": 0,
        "greedy_skipped_patch_count": 0, "correction": {
            "SC": Counter(), "GC": Counter()},
        "first_greedy_memory_divergence_position": None,
        "same_base_prewrite_checks": 0, "same_correction_predivergence_checks": 0,
        "source_exclusion_checks": 0,
    }
    score_rows_written = 0
    try:
        torch.set_num_threads(1)
        visible_device_count = int(torch.cuda.device_count())
        if visible_device_count != 1:
            raise R3Error(
                "Authorized GPU visibility changed inside supervised child: "
                f"device_count={visible_device_count}, "
                f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')!r}, "
                f"cuda_available={torch.cuda.is_available()}, "
                f"cuda_initialized={torch.cuda.is_initialized()}"
            )
        impl = p0.require_sources()
        expected_model = _get_verified_model_receipt()
        restore_hub, runtime_info = _runtime_and_native_setup(torch, impl, expected_model)
        cfg, scorer_profile = _native_config(p0, __import__(
            "experiments.calibration.profiles", fromlist=["NativeScorerConfig"]), r1)
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        model_started = time.perf_counter()
        try:
            backbone = impl.DINOv2MultiLayerBackbone(device="cuda")
        finally:
            restore_hub()
        loaded_model_sha = r1._model_state_sha256(backbone.model)
        if loaded_model_sha != expected_model.get("loaded_model_state_sha256"):
            raise R3Error("Loaded DINO weights differ from the verified R1 model state")
        runtime_info["loaded_model_state_sha256"] = loaded_model_sha

        support_paths = [str(path) for path in info["support_paths"]]
        support_features = p0.encode_reference_features(impl, backbone, support_paths, cfg)
        support_encoder_invocations = sum(len(support_features[path]) for path in support_paths)
        selected = list(support_paths)
        impl.set_seed(0)
        static_bank = p0.build_vmb(impl, support_features, selected, cfg)
        impl.set_seed(0)
        greedy_bank = p0.build_vmb(impl, support_features, selected, cfg)
        static_receipt = _check_bank_receipt(
            static_bank, p0, info["fixed_support_sha256_expected"], r1)
        greedy_receipt = _check_bank_receipt(
            greedy_bank, p0, info["fixed_support_sha256_expected"], r1)
        if static_receipt["actual_structured_sha256"] != greedy_receipt["actual_structured_sha256"]:
            raise R3Error("Static and Greedy initial fixed banks are not identical")
        _validate_correction_config(static_bank)
        _validate_correction_config(greedy_bank)
        if (static_bank.corr_cb["p0"]["id"][0] is greedy_bank.corr_cb["p0"]["id"][0] or
                int(static_bank.corr_cb["p0"]["id"][0].size()) != 0 or
                int(greedy_bank.corr_cb["p0"]["id"][0].size()) != 0):
            raise R3Error("SC and GC correction banks are not independent empty codebooks")
        if static_bank.corr_cb["p0"]["id"][0].K != 256 or greedy_bank.corr_cb["p0"]["id"][0].K != 256:
            raise R3Error("C1-Full capacity differs from the frozen native 256-anchor profile")

        settings = SimpleNamespace(flat=256, stm=256, ltm=1, buffer=1,
                                   min_events=2, promote_every=16,
                                   cap_bytes=2 * 1024**3)
        greedy_writer = MemoryArm("greedy_coreset", greedy_bank, settings, seed=0)
        if any(int(cb.size()) != 0 or int(cb.K) != 256
               for cb in dynamic_codebooks(greedy_bank)):
            raise R3Error("GREEDY_NORMAL did not start empty at native K=256")
        if any(int(cb.size()) != 0 for cb in dynamic_codebooks(static_bank)):
            raise R3Error("STATIC_NORMAL has unexpected mutable normal prototypes")
        static_dynamic = dynamic_codebooks(static_bank)
        greedy_dynamic = dynamic_codebooks(greedy_bank)
        if (len(static_dynamic) != len(greedy_dynamic) or
                any(left is right or left.protos is right.protos
                    for left, right in zip(static_dynamic, greedy_dynamic))):
            raise R3Error("STATIC_NORMAL and GREEDY_NORMAL share mutable dynamic tensors")
        static_memory_initial = r1._memory_content_sha256(static_bank)

        # Native predictor path must remain correction-disabled; correction is applied below.
        static_bank.corr_cfg.enabled = False
        greedy_bank.corr_cfg.enabled = False
        static_bank.use_defect_bank = False
        greedy_bank.use_defect_bank = False
        theta, sigma = float(info["theta_c64_q90"]), float(info["sigma_c64_floor_002"])
        c1_source_groups: dict[str, dict[int, set[str]]] = {"SC": {}, "GC": {}}
        c1_stats: dict[str, Counter[str]] = {"SC": Counter(), "GC": Counter()}
        c1_source_exclusions = Counter()
        c1_active_peak = {"SC": 0, "GC": 0}
        memory_version = 0
        first_divergence: int | None = None
        writer_audit_before = greedy_writer.audit()
        label_json = JSON_ROOT / f"{category}.json"
        from experiments.calibration.stage1a_p12_label_provider import QueriedPositionLabelReader
        label_reader = QueriedPositionLabelReader(
            category=category, path=label_json,
            expected_source_sha256=info["category_json_sha256"],
            positions=allowed_queries,
            group_sha256_by_position=group_map,
        ) if allowed_queries else None

        exp2a_refs: dict[int, dict[str, Any]] = {}
        if event_limit >= min(info["stream_count"], 512):
            _, all_refs = _load_exp2a_score_references()
            exp2a_refs = {pos: all_refs[(category, pos)] for pos in range(512)}

        from experiments.calibration.r1_matched_correction_pilot import (
            _native_dense_score, _native_reference_and_debug,
            _c1_score_delta, corrected_dense_map, max_score,
            common_c0_greedy_writer_action,
            _model_state_sha256, _process_rss_bytes,
        )

        progress = 0
        for position, meta in enumerate(info["stream_rows"][:event_limit]):
            image_path = info["event_paths"][position]
            if not image_path.is_file():
                raise R3Error(f"{category}:{position}: authorized image disappeared")
            image_sha = sha256_file(image_path)
            torch.cuda.synchronize()
            start = time.perf_counter()
            image, features = p0.encode_query(impl, backbone, str(image_path), cfg)
            torch.cuda.synchronize()
            encoding_time += time.perf_counter() - start
            counters["actual_image_encodings"] += 1
            if (len(features) != 1 or tuple(features[0].shape) != (784, 4096) or
                    features[0].dtype != torch.float32 or
                    not torch.isfinite(features[0]).all().item()):
                raise R3Error(f"{category}:{position}: native fused descriptor contract failed")
            group_sha = str(meta["opaque_group"])
            queried = position in set(allowed_queries)
            start = time.perf_counter()
            static_map, static_ids, static_base, static_summary = _native_dense_score(
                impl, static_bank, image, features, cfg)
            greedy_map, greedy_ids, greedy_base, greedy_summary = _native_dense_score(
                impl, greedy_bank, image, features, cfg)
            torch.cuda.synchronize()
            scorer_time += time.perf_counter() - start
            if (not np.array_equal(static_ids, np.asarray(
                    static_summary["layer0"]["topk_patch_idx"], dtype=np.int64)) or
                    not np.array_equal(greedy_ids, np.asarray(
                        greedy_summary["layer0"]["topk_patch_idx"], dtype=np.int64))):
                raise R3Error(f"{category}:{position}: a native branch Top-32 receipt differs")
            if first_divergence is None:
                base_map_error = float(np.max(np.abs(static_map - greedy_map)))
                base_score_error = abs(float(static_base) - float(greedy_base))
                parity["initial_map_max_abs_error"] = max(
                    parity["initial_map_max_abs_error"], base_map_error)
                parity["initial_score_max_abs_error"] = max(
                    parity["initial_score_max_abs_error"], base_score_error)
                counters["same_base_prewrite_checks"] += 1
                if (not np.allclose(static_map, greedy_map, atol=ATOL, rtol=RTOL) or
                        not _score_tolerance(static_base, greedy_base)):
                    raise R3Error(f"{category}:{position}: Static/Greedy base differed before first memory write")
            if position == 0:
                for branch_bank in (static_bank, greedy_bank):
                    ref_score, ref_ids, ref_map = _native_reference_and_debug(
                        impl, branch_bank, image, features, cfg)
                    if (not np.array_equal(ref_ids, static_ids if branch_bank is static_bank else greedy_ids) or
                            not np.array_equal(ref_map, static_map if branch_bank is static_bank else greedy_map) or
                            not _score_tolerance(ref_score, static_base if branch_bank is static_bank else greedy_base)):
                        raise R3Error("Native map adapter failed direct reference parity on the first smoke event")

            start = time.perf_counter()
            selected_static = impl.l2_normalize(features[0].to(static_bank.device))[
                torch.as_tensor(static_ids, device=static_bank.device, dtype=torch.long)]
            selected_greedy = impl.l2_normalize(features[0].to(greedy_bank.device))[
                torch.as_tensor(greedy_ids, device=greedy_bank.device, dtype=torch.long)]
            delta_static, detail_static = _c1_score_delta(
                impl, static_bank, selected_static, c1_source_groups["SC"], group_sha,
                event_position=position)
            delta_greedy, detail_greedy = _c1_score_delta(
                impl, greedy_bank, selected_greedy, c1_source_groups["GC"], group_sha,
                event_position=position)
            c1_source_exclusions["SC"] += int(detail_static["same_source_excluded_anchor_slots"])
            c1_source_exclusions["GC"] += int(detail_greedy["same_source_excluded_anchor_slots"])
            counters["source_exclusion_checks"] += 2
            sc_map = corrected_dense_map(static_map, static_ids, delta_static)
            gc_map = corrected_dense_map(greedy_map, greedy_ids, delta_greedy)
            scores = {"S": float(static_base), "SC": max_score(sc_map),
                      "G": float(greedy_base), "GC": max_score(gc_map)}
            correction_time += time.perf_counter() - start
            if not all(math.isfinite(value) for value in scores.values()):
                raise R3Error(f"{category}:{position}: nonfinite factorial score")
            if scores["S"] != max_score(static_map) or scores["G"] != max_score(greedy_map):
                raise R3Error(f"{category}:{position}: native max aggregation changed")
            if first_divergence is None:
                counters["same_correction_predivergence_checks"] += 1
                if (not np.allclose(delta_static, delta_greedy, atol=ATOL, rtol=RTOL) or
                        not _score_tolerance(scores["SC"], scores["GC"])):
                    raise R3Error(f"{category}:{position}: correction branches diverged before normal memory")

            event_id = _event_id(category, position, str(meta["relative_id"]),
                                 image_sha, group_sha)
            source_identity = sha256_bytes(str(meta["relative_id"]).encode("utf-8"))
            score_row = {
                "schema": SCORE_SCHEMA, "category": category, "position": position,
                "event_id_sha256": event_id,
                "source_identity_sha256": source_identity,
                "source_group_sha256": group_sha,
                "image_content_sha256": image_sha,
                "camera_id": int(meta["camera_id"]),
                "prequential_order_position": position,
                "timestamp_semantics": "metadata only; not asserted to be physical chronology",
                "query_tape_decision": queried,
                "theta_c64_q90": theta,
                "scores": scores,
                "predictions_at_frozen_threshold": {
                    arm: int(value >= theta) for arm, value in scores.items()},
                "greedy_memory_version_before": memory_version,
                "SC_correction_occupancy_before": int(static_bank.corr_cb["p0"]["id"][0].size()),
                "GC_correction_occupancy_before": int(greedy_bank.corr_cb["p0"]["id"][0].size()),
                "SC_same_group_excluded_slots": int(detail_static["same_source_excluded_anchor_slots"]),
                "GC_same_group_excluded_slots": int(detail_greedy["same_source_excluded_anchor_slots"]),
            }
            encoded = scores_out.add(score_row)
            category_score_digest.update(encoded)
            score_rows_written += 1
            if queried or (position + 1) % 256 == 0:
                # Includes this query's four sealed scores; fsync precedes any label access.
                scores_out.flush()
            if position < 512 and exp2a_refs:
                reference = exp2a_refs[position]
                if (reference.get("source_group_sha256") != group_sha or
                        bool(reference.get("query_tape_decision")) != queried or
                        reference.get("class_json_sha256") != info["category_json_sha256"]):
                    raise R3Error(f"{category}:{position}: EXP2A event/query identity mismatch")
                exp2a_scores = reference["scores"]
                for arm, actual, expected, metric_key in (
                        ("G", scores["G"], exp2a_scores["c0"], "g"),
                        ("GC", scores["GC"], exp2a_scores["c1_full"], "gc")):
                    error = abs(float(actual) - float(expected))
                    parity[f"{metric_key}_score_checks"] += 1
                    parity[f"{metric_key}_max_abs_error"] = max(
                        parity[f"{metric_key}_max_abs_error"], error)
                    if not _score_tolerance(actual, expected):
                        parity["numeric_mismatches"] += 1
                        from experiments.calibration.native import _fixed_bank_digest
                        raise R3Error(
                            f"{category}:{position}: first-512 EXP2A {arm} parity mismatch "
                            f"before feedback disclosure; actual={actual:.9g}, "
                            f"expected={expected:.9g}, abs_error={error:.3g}, "
                            f"memory_version={memory_version}, "
                            f"greedy_memory_state_sha256={r1._memory_state_sha256(greedy_bank, memory_version)}, "
                            f"static_fixed_sha256={_fixed_bank_digest(static_bank)}, "
                            f"greedy_fixed_sha256={_fixed_bank_digest(greedy_bank)}, "
                            f"greedy_top32_sha256={sha256_json(np.asarray(greedy_ids, dtype=np.int64).tolist())}, "
                            f"theta={theta:.9g}, C0={greedy_base:.9g}, SC={scores['SC']:.9g}, "
                            f"GC={scores['GC']:.9g}"
                        )
                parity["checked_events"] += 1
            if queried:
                disclosed = _disclose_query_label_after_seal(
                    scores_out, query_decision=queried, category=category,
                    position=position, sample_key=str(meta["sample_key"]),
                    label_reader=label_reader)
                if disclosed is None:
                    raise R3Error("Query feedback was unexpectedly withheld")
                label = disclosed
                if label not in (0, 1):
                    raise R3Error("Restricted query reader returned a nonbinary label")
                counters["query_events"] += 1
                if label == 0:
                    counters["query_normal"] += 1
                else:
                    counters["query_anomalous"] += 1
                correction_updates: dict[str, Any] = {}
                for branch, branch_bank, branch_features, branch_ids, branch_base in (
                        ("SC", static_bank, features, static_ids, static_base),
                        ("GC", greedy_bank, features, greedy_ids, greedy_base)):
                    prediction = int(float(branch_base) >= theta)
                    outcome = {(0, 0): "TN", (0, 1): "FP",
                               (1, 0): "FN", (1, 1): "TP"}[(label, prediction)]
                    sign = 1 if label == 1 else -1
                    cfg_c1 = branch_bank.corr_cfg
                    cb = branch_bank.corr_cb["p0"]["id"][0]
                    _result, layer, diag = exp2a._capture_update(
                        impl, branch_bank, r1, "c1_full", cfg_c1, cb,
                        branch_features, np.asarray(branch_ids, dtype=np.int64),
                        sign, float(branch_base), theta, outcome, position,
                        c1_source_groups[branch], group_sha)
                    updates = {key: int(value) for key, value in layer.items()
                               if isinstance(value, (int, np.integer))}
                    c1_stats[branch].update(updates)
                    c1_active_peak[branch] = max(c1_active_peak[branch], int(cb.size()))
                    correction_updates[branch] = {
                        "base_score": float(branch_base), "base_prediction": prediction,
                        "outcome": outcome, "sign": sign, "native_stats": updates,
                        "occupancy_after": int(cb.size()),
                        "native_keep_count": int(sum(bool(x) for x in diag["native_gate_keep_mask"])),
                    }

                writer_start = time.perf_counter()
                audit_before = greedy_writer.audit()
                writer_memory_change: dict[str, bool] = {"changed": False}

                def writer_update(*, position: int, feature_layers: Sequence[Any],
                                  candidate_ids: Any, score: float, theta: float,
                                  sigma: float) -> Mapping[str, Any]:
                    memory_before = r1._memory_content_sha256(greedy_bank)
                    greedy_writer.update(
                        impl, list(feature_layers), {0: candidate_ids.tolist()}, position,
                        score=score, theta=theta, sigma=sigma)
                    memory_after = r1._memory_content_sha256(greedy_bank)
                    writer_memory_change["changed"] = memory_after != memory_before
                    audit_after = greedy_writer.audit()
                    return {
                        "accepted_patches_delta": int(audit_after["accepted_patches"] -
                                                      audit_before["accepted_patches"]),
                        "skipped_patches_delta": int(audit_after["skipped_patches"] -
                                                     audit_before["skipped_patches"]),
                        "_memory_content_changed": writer_memory_change["changed"],
                    }

                normalized_greedy_candidates = impl.l2_normalize(
                    features[0].to(greedy_bank.device))[torch.as_tensor(
                        greedy_ids, device=greedy_bank.device, dtype=torch.long)]
                writer_action = common_c0_greedy_writer_action(
                    position=position, c0_score=float(greedy_base), theta=theta,
                    sigma=sigma, c0_prediction=int(greedy_base >= theta),
                    candidate_ids=np.asarray(greedy_ids, dtype=np.int64),
                    candidate_features=normalized_greedy_candidates.detach().float().cpu().numpy(),
                    feature_layers=features, queried=True, disclosed_label=label,
                    eligibility=impl.select_informative_normal_write,
                    update=writer_update)
                writer_time += time.perf_counter() - writer_start
                memory_changed = writer_memory_change["changed"]
                writer_action.pop("_memory_content_changed", None)
                if memory_changed:
                    memory_version += 1
                    if first_divergence is None:
                        first_divergence = position
                        counters["first_greedy_memory_divergence_position"] = position
                if writer_action.get("action") == "native_greedy_normal_update":
                    counters["greedy_write_events"] += 1
                    counters["greedy_accepted_patch_count"] += int(
                        writer_action.get("accepted_patches_delta", 0))
                    counters["greedy_skipped_patch_count"] += int(
                        writer_action.get("skipped_patches_delta", 0))
                history_row = {
                    "schema": HISTORY_SCHEMA, "record_type": "queried_feedback_and_updates",
                    "category": category, "position": position,
                    "event_id_sha256": event_id,
                    "source_group_sha256": group_sha,
                    "disclosure_after_four_score_seal": True,
                    "label": label,
                    "correction_updates": correction_updates,
                    "greedy_writer_action": writer_action,
                    "greedy_memory_version_after": memory_version,
                    "greedy_memory_content_changed": memory_changed,
                }
                history_bytes = _append_history(output_budget, history_row)
                query_receipt_digest.update(history_bytes)
                progress += 1
            if (position + 1) % 256 == 0:
                scores_out.flush()
                checkpoint = {
                    "schema": HISTORY_SCHEMA,
                    "record_type": "diagnostic_progress_checkpoint",
                    "category": category,
                    "sealed_through_position": position,
                    "score_rows_written": score_rows_written,
                    "queried_feedback_count": counters["query_events"],
                    "actual_image_encodings": counters["actual_image_encodings"],
                    "greedy_memory_version": memory_version,
                    "greedy_memory_state_sha256": r1._memory_state_sha256(
                        greedy_bank, memory_version),
                    "static_normal_fixed_bank_sha256": _fixed_bank_digest(static_bank),
                    "SC_correction_state_sha256": r1._correction_state_sha256(
                        static_bank, c1_source_groups["SC"]),
                    "GC_correction_state_sha256": r1._correction_state_sha256(
                        greedy_bank, c1_source_groups["GC"]),
                    "checkpoint_resumable": False,
                    "recovery_rule": "Restart this incomplete category from position zero; completed categories may be integrity-verified and reused.",
                }
                _append_history(output_budget, checkpoint)
                print(f"R3_PROGRESS category={category} events={position+1}/{event_limit} "
                      f"queries={counters['query_events']} encodings={counters['actual_image_encodings']}",
                      flush=True)
            del image, features, static_map, greedy_map, sc_map, gc_map

        scores_out.flush()
        if label_reader is not None:
            if tuple(label_reader.decoded_positions) != allowed_queries:
                raise R3Error("Queried label reader did not follow the complete fixed tape prefix")
            label_reader.close()
        if run_manifest["mode"] == "FULL_STAGE_A" and event_limit != EXPECTED_COUNTS[category]:
            raise R3Error("Full category did not process its frozen complete event count")
        if r1._memory_content_sha256(static_bank) != static_memory_initial:
            raise R3Error("STATIC_NORMAL changed during the full prequential stream")
        if any(int(cb.size()) for cb in dynamic_codebooks(static_bank)):
            raise R3Error("STATIC_NORMAL received dynamic normal prototypes")
        if (_fixed_bank_digest(static_bank) != info["fixed_support_sha256_expected"] or
                _fixed_bank_digest(greedy_bank) != info["fixed_support_sha256_expected"]):
            raise R3Error("Fixed Support changed by online memory/correction updates")
        if static_bank.corr_cb["p0"]["id"][0] is greedy_bank.corr_cb["p0"]["id"][0]:
            raise R3Error("Correction bank object identity became shared")
        final_audit = greedy_writer.audit()
        torch.cuda.synchronize()
        result = {
            "schema": HISTORY_SCHEMA, "record_type": "category_complete",
            "category": category, "events_processed": event_limit,
            "query_events_processed": counters["query_events"],
            "score_rows_sha256": category_score_digest.hexdigest(),
            "query_receipts_sha256": query_receipt_digest.hexdigest(),
            "fixed_support_sha256": _fixed_bank_digest(greedy_bank),
            "fixed_support_sha256_static": _fixed_bank_digest(static_bank),
            "fixed_support_digest_algorithm": "native._fixed_bank_digest.v1",
            "fixed_support_raw_diagnostic_sha256_greedy": p0.fixed_bank_profile(greedy_bank)["sha256"],
            "fixed_support_raw_diagnostic_sha256_static": p0.fixed_bank_profile(static_bank)["sha256"],
            "theta_c64_q90": theta, "sigma_c64_floor_002": sigma,
            "query_tape_sha256": info["query_tape_sha256"],
            "query_positions_prefix_sha256": sha256_json(list(allowed_queries)),
            "greedy_memory_version_final": memory_version,
            "greedy_memory_content_sha256_final": r1._memory_content_sha256(greedy_bank),
            "greedy_memory_state_sha256_final": r1._memory_state_sha256(greedy_bank, memory_version),
            "greedy_writer_audit_final": final_audit,
            "greedy_write_events": counters["greedy_write_events"],
            "greedy_accepted_patch_count": counters["greedy_accepted_patch_count"],
            "greedy_skipped_patch_count": counters["greedy_skipped_patch_count"],
            "correction_stats": {key: dict(value) for key, value in c1_stats.items()},
            "correction_final_occupancy": {
                "SC": int(static_bank.corr_cb["p0"]["id"][0].size()),
                "GC": int(greedy_bank.corr_cb["p0"]["id"][0].size())},
            "correction_max_occupancy": c1_active_peak,
            "source_exclusion_reads": dict(c1_source_exclusions),
            "source_exclusion_checks": counters["source_exclusion_checks"],
            "consistency_checks": {
                "same_base_before_first_write": counters["same_base_prewrite_checks"],
                "same_correction_before_first_write": counters["same_correction_predivergence_checks"],
                "static_normal_unchanged": True,
                "greedy_writer_input": "G pre-event native base score/top-32 only",
                "corrections_independent": True,
            },
            "first512_exp2a_parity": parity if exp2a_refs else {"status": "NOT_APPLICABLE_SMOKE_PREFIX"},
            "runtime_profile": runtime_info["runtime_profile"],
            "hub_source_tree_sha256": runtime_info["hub_source_tree_sha256"],
            "checkpoint_sha256": runtime_info["checkpoint_sha256"],
            "loaded_model_state_sha256": loaded_model_sha,
            "support_encoder_invocations": support_encoder_invocations,
            "support_and_model_setup_seconds": time.perf_counter() - model_started,
            "actual_stream_image_encodings": counters["actual_image_encodings"],
            "encoding_seconds": encoding_time,
            "native_base_scoring_seconds": scorer_time,
            "correction_scoring_seconds": correction_time,
            "greedy_writer_seconds": writer_time,
            "wall_seconds": time.monotonic() - category_started,
            "peak_cuda_allocated_bytes": int(torch.cuda.max_memory_allocated()),
            "peak_cuda_reserved_bytes": int(torch.cuda.max_memory_reserved()),
            "peak_cpu_rss_bytes": r1._process_rss_bytes(),
            "smoke_only": run_manifest["mode"] == "SMOKE_ONLY",
            "offline_evaluation_performed": False,
        }
        _append_history(output_budget, result)
        if backbone is not None:
            backbone.cleanup()
        return result
    except BaseException as exc:
        try:
            scores_out.flush()
        except Exception:
            pass
        try:
            if label_reader is not None:
                label_reader.close()
        except Exception:
            pass
        try:
            if backbone is not None:
                backbone.cleanup()
        except Exception:
            pass
        _write_failure(output, "STAGE_A_CHILD", exc, {"category": category})
        raise


def _weighted_rank_metrics(scores: Any, labels: Any, weights: Any | None = None) -> dict[str, float | None]:
    import numpy as np
    score = np.asarray(scores, dtype=np.float64)
    y = np.asarray(labels, dtype=np.int8)
    w = np.ones(len(y), dtype=np.float64) if weights is None else np.asarray(weights, dtype=np.float64)
    if score.ndim != 1 or y.shape != score.shape or w.shape != score.shape:
        raise R3Error("Stage B metric arrays are misaligned")
    if not np.isfinite(score).all() or not np.isfinite(w).all() or np.any(w < 0):
        raise R3Error("Stage B received nonfinite scores or invalid bootstrap weights")
    pos_total = float(w[y == 1].sum())
    neg_total = float(w[y == 0].sum())
    auc: float | None = None
    ap: float | None = None
    if pos_total > 0 and neg_total > 0:
        order = np.argsort(score, kind="mergesort")
        ss, yy, ww = score[order], y[order], w[order]
        starts = np.r_[0, np.flatnonzero(ss[1:] != ss[:-1]) + 1]
        pos_groups = np.add.reduceat(ww * (yy == 1), starts)
        neg_groups = np.add.reduceat(ww * (yy == 0), starts)
        neg_before = np.cumsum(neg_groups) - neg_groups
        auc = float(np.sum(pos_groups * (neg_before + 0.5 * neg_groups)) /
                    (pos_total * neg_total))

        order_desc = order[::-1]
        sd, yd, wd = score[order_desc], y[order_desc], w[order_desc]
        starts_d = np.r_[0, np.flatnonzero(sd[1:] != sd[:-1]) + 1]
        pos_d = np.add.reduceat(wd * (yd == 1), starts_d)
        neg_d = np.add.reduceat(wd * (yd == 0), starts_d)
        cumulative_pos = np.cumsum(pos_d)
        cumulative_all = np.cumsum(pos_d + neg_d)
        ap = float(np.sum((pos_d / pos_total) * cumulative_pos /
                          np.maximum(cumulative_all, 1e-300)))
    elif pos_total > 0:
        order_desc = np.argsort(-score, kind="mergesort")
        sd, yd, wd = score[order_desc], y[order_desc], w[order_desc]
        starts_d = np.r_[0, np.flatnonzero(sd[1:] != sd[:-1]) + 1]
        pos_d = np.add.reduceat(wd * (yd == 1), starts_d)
        neg_d = np.add.reduceat(wd * (yd == 0), starts_d)
        cumulative_pos = np.cumsum(pos_d)
        cumulative_all = np.cumsum(pos_d + neg_d)
        ap = float(np.sum((pos_d / pos_total) * cumulative_pos /
                          np.maximum(cumulative_all, 1e-300)))
    return {"auroc": auc, "average_precision": ap,
            "anomaly_prevalence": (pos_total / (pos_total + neg_total)
                                   if pos_total + neg_total else None)}


def _threshold_metrics(scores: Any, labels: Any, threshold: float,
                       weights: Any | None = None) -> dict[str, Any]:
    import numpy as np
    y = np.asarray(labels, dtype=np.int8)
    s = np.asarray(scores, dtype=np.float64)
    w = np.ones(len(y), dtype=np.float64) if weights is None else np.asarray(weights, dtype=np.float64)
    pred = (s >= float(threshold)).astype(np.int8)
    tn = float(w[(y == 0) & (pred == 0)].sum())
    fp = float(w[(y == 0) & (pred == 1)].sum())
    fn = float(w[(y == 1) & (pred == 0)].sum())
    tp = float(w[(y == 1) & (pred == 1)].sum())
    fpr = fp / (fp + tn) if fp + tn else None
    fnr = fn / (fn + tp) if fn + tp else None
    ba = (1.0 - (fpr + fnr) / 2.0) if fpr is not None and fnr is not None else None
    ranked = _weighted_rank_metrics(s, y, w)
    return {
        **ranked,
        "fpr": fpr, "fnr": fnr, "balanced_accuracy": ba,
        "confusion": {"tn": int(round(tn)), "fp": int(round(fp)),
                      "fn": int(round(fn)), "tp": int(round(tp))},
        "weighted_confusion": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
    }


def _factorial_contrasts(metrics_by_arm: Mapping[str, Mapping[str, Any]],
                         metric: str) -> dict[str, float | None]:
    def val(arm: str) -> float | None:
        value = metrics_by_arm[arm].get(metric)
        return None if value is None else float(value)
    s, sc, g, gc = (val(name) for name in ARMS)
    if any(value is None for value in (s, sc, g, gc)):
        return {"GC_minus_G": None, "GC_minus_SC": None,
                "SC_minus_S": None, "G_minus_S": None, "interaction": None}
    assert s is not None and sc is not None and g is not None and gc is not None
    return {"GC_minus_G": gc - g, "GC_minus_SC": gc - sc,
            "SC_minus_S": sc - s, "G_minus_S": g - s,
            "interaction": gc - g - sc + s}


def _paired_group_bootstrap(scores: Mapping[str, Any], labels: Any, groups: Any,
                            threshold: float, *, replicates: int = BOOTSTRAP_REPLICATES,
                            seed: int = BOOTSTRAP_SEED) -> dict[str, Any]:
    import numpy as np
    y = np.asarray(labels, dtype=np.int8)
    group_values, inverse = np.unique(np.asarray(groups, dtype=str), return_inverse=True)
    if len(group_values) < 2:
        raise R3Error("Source Group bootstrap requires at least two groups")
    rng = np.random.default_rng(int(seed))
    samples: dict[str, dict[str, list[float]]] = {
        metric: {contrast: [] for contrast in
                 ("GC_minus_G", "GC_minus_SC", "SC_minus_S", "G_minus_S", "interaction")}
        for metric in ("auroc", "average_precision", "balanced_accuracy")}
    valid = 0
    for _ in range(int(replicates)):
        sampled = rng.integers(0, len(group_values), size=len(group_values))
        cluster_weights = np.bincount(sampled, minlength=len(group_values)).astype(np.float64)
        weights = cluster_weights[inverse]
        metrics = {arm: _threshold_metrics(scores[arm], y, threshold, weights)
                   for arm in ARMS}
        point = {metric: _factorial_contrasts(metrics, metric) for metric in samples}
        if point["auroc"]["interaction"] is None:
            continue
        valid += 1
        for metric, contrasts in point.items():
            for name, value in contrasts.items():
                if value is not None and math.isfinite(float(value)):
                    samples[metric][name].append(float(value))
    output = {}
    for metric, contrasts in samples.items():
        output[metric] = {}
        for name, values in contrasts.items():
            output[metric][name] = {
                "valid_replicates": len(values),
                "ci_95_percentile": ([float(np.quantile(values, 0.025)),
                                      float(np.quantile(values, 0.975))]
                                     if values else None),
            }
    return {"method": "paired Source Group cluster bootstrap; percentile 95% interval",
            "replicates_requested": int(replicates), "valid_joint_replicates": valid,
            "seed": int(seed), "source_group_count": int(len(group_values)),
            "same_sampled_groups_for_all_arms": True,
            "factorial_interaction_computed_inside_each_replicate": True,
            "contrasts": output}


def _pair_credit_changes(scores_a: Any, scores_b: Any, labels: Any,
                         groups: Any) -> dict[str, Any]:
    import numpy as np
    y = np.asarray(labels, dtype=np.int8)
    group = np.asarray(groups, dtype=str)
    a, b = np.asarray(scores_a, dtype=np.float64), np.asarray(scores_b, dtype=np.float64)
    normal = np.flatnonzero(y == 0)
    defect = np.flatnonzero(y == 1)
    result = {}
    for label, restrict_disjoint in (("all_pairs", False), ("source_group_disjoint", True)):
        pair_count = improve = damage = 0
        net = 0.0
        ties_a = ties_b = 0
        for begin in range(0, len(defect), 128):
            di = defect[begin:begin + 128]
            diff_a = a[di, None] - a[normal][None, :]
            diff_b = b[di, None] - b[normal][None, :]
            credit_a = (diff_a > 0).astype(np.float64) + 0.5 * (diff_a == 0)
            credit_b = (diff_b > 0).astype(np.float64) + 0.5 * (diff_b == 0)
            mask = np.ones(diff_a.shape, dtype=bool)
            if restrict_disjoint:
                mask = group[di, None] != group[normal][None, :]
            delta = credit_b - credit_a
            pair_count += int(mask.sum())
            improve += int(((delta > 0) & mask).sum())
            damage += int(((delta < 0) & mask).sum())
            net += float(delta[mask].sum())
            ties_a += int(((diff_a == 0) & mask).sum())
            ties_b += int(((diff_b == 0) & mask).sum())
        result[label] = {
            "normal_defect_pair_denominator": pair_count,
            "candidate_credit_improvements": improve,
            "candidate_credit_damages": damage,
            "ties_in_reference": ties_a,
            "ties_in_candidate": ties_b,
            "net_ranking_credit": net,
            "net_ranking_credit_fraction_of_pairs": net / pair_count if pair_count else None,
            "ties_weighted_as_half": True,
        }
    return result


def _confusion_transitions(base: Any, candidate: Any, labels: Any) -> dict[str, Any]:
    import numpy as np
    y = np.asarray(labels, dtype=np.int8)
    p0 = np.asarray(base, dtype=np.int8)
    p1 = np.asarray(candidate, dtype=np.int8)
    if y.ndim != 1 or p0.shape != y.shape or p1.shape != y.shape:
        raise R3Error("Confusion transition inputs must be aligned one-dimensional arrays")
    counts = {
        "FN_to_TP_repaired": int(((y == 1) & (p0 == 0) & (p1 == 1)).sum()),
        "FP_to_TN_repaired": int(((y == 0) & (p0 == 1) & (p1 == 0)).sum()),
        "TP_to_FN_introduced": int(((y == 1) & (p0 == 1) & (p1 == 0)).sum()),
        "TN_to_FP_introduced": int(((y == 0) & (p0 == 0) & (p1 == 1)).sum()),
    }
    def confusion(pred: Any) -> dict[str, int]:
        pred = np.asarray(pred, dtype=np.int8)
        return {"tn": int(((y == 0) & (pred == 0)).sum()),
                "fp": int(((y == 0) & (pred == 1)).sum()),
                "fn": int(((y == 1) & (pred == 0)).sum()),
                "tp": int(((y == 1) & (pred == 1)).sum())}
    base_c, candidate_c = confusion(p0), confusion(p1)
    fntp, fptn = counts["FN_to_TP_repaired"], counts["FP_to_TN_repaired"]
    tpfn, tnfp = counts["TP_to_FN_introduced"], counts["TN_to_FP_introduced"]
    reconstructed = {"tn": base_c["tn"] - tnfp + fptn,
                     "fp": base_c["fp"] - fptn + tnfp,
                     "fn": base_c["fn"] - fntp + tpfn,
                     "tp": base_c["tp"] - tpfn + fntp}
    if reconstructed != candidate_c:
        raise R3Error("Error transition ledger does not reconstruct candidate confusion matrix")
    return {**counts, "base_confusion": base_c,
            "candidate_confusion": candidate_c,
            "reconstructed_candidate_confusion": reconstructed,
            "reconstruction_verified": True}


def _category_stage_b(category: str, rows: Sequence[Mapping[str, Any]],
                      labels: Any, threshold: float,
                      query_receipts: Sequence[Mapping[str, Any]],
                      category_receipt: Mapping[str, Any]) -> dict[str, Any]:
    import numpy as np
    y = np.asarray(labels, dtype=np.int8)
    if len(rows) != EXPECTED_COUNTS[category] or y.shape != (len(rows),):
        raise R3Error(f"{category}: score/outcome length differs from frozen stream")
    if np.any((y != 0) & (y != 1)):
        raise R3Error(f"{category}: evaluator returned a nonbinary label")
    positions = [int(row["position"]) for row in rows]
    if positions != list(range(len(rows))) or any(row.get("category") != category for row in rows):
        raise R3Error(f"{category}: score trajectory ordering/category identity failed")
    scores = {arm: np.asarray([float(row["scores"][arm]) for row in rows], dtype=np.float64)
              for arm in ARMS}
    groups = np.asarray([str(row["source_group_sha256"]) for row in rows], dtype=str)
    if any(not re.fullmatch(r"[0-9a-f]{64}", value) for value in groups):
        raise R3Error(f"{category}: a Stage A row has malformed Source Group identity")
    query_mask = np.asarray([bool(row["query_tape_decision"]) for row in rows], dtype=bool)
    query_positions = np.flatnonzero(query_mask)
    receipts = {int(row["position"]): row for row in query_receipts
                if row.get("category") == category and
                row.get("record_type") == "queried_feedback_and_updates"}
    if set(receipts) != set(query_positions.tolist()):
        raise R3Error(f"{category}: queried feedback receipts differ from the fixed tape")
    for position, receipt in receipts.items():
        if (receipt.get("event_id_sha256") != rows[position].get("event_id_sha256") or
                receipt.get("source_group_sha256") != groups[position] or
                int(receipt.get("label", -1)) != int(y[position]) or
                receipt.get("disclosure_after_four_score_seal") is not True):
            raise R3Error(f"{category}:{position}: queried feedback receipt does not bind sealed score")
        updates = receipt.get("correction_updates", {})
        if set(updates) != {"SC", "GC"}:
            raise R3Error(f"{category}:{position}: the two correction updates were not both executed")
        for branch, base_arm in (("SC", "S"), ("GC", "G")):
            update = updates[branch]
            prediction = int(rows[position]["predictions_at_frozen_threshold"][base_arm])
            outcome = {(0, 0): "TN", (0, 1): "FP",
                       (1, 0): "FN", (1, 1): "TP"}[(int(y[position]), prediction)]
            if (update.get("base_prediction") != prediction or
                    update.get("outcome") != outcome or
                    not _score_tolerance(update.get("base_score"), scores[base_arm][position])):
                raise R3Error(f"{category}:{position}: {branch} update used the wrong pre-event base")
        writer = receipt.get("greedy_writer_action", {})
        if (writer.get("input_score_source") != "common_C0" or
                writer.get("prediction") != int(rows[position]["predictions_at_frozen_threshold"]["G"]) or
                (int(y[position]) == 1 and writer.get("action") != "verified_defect_no_normal_write")):
            raise R3Error(f"{category}:{position}: Greedy write is not controlled by verified C0")
    metrics = {arm: _threshold_metrics(scores[arm], y, threshold) for arm in ARMS}
    predictions = {arm: (scores[arm] >= threshold).astype(np.int8) for arm in ARMS}
    contrasts = {name: _factorial_contrasts(metrics, name)
                 for name in ("auroc", "average_precision", "balanced_accuracy")}
    bootstrap = _paired_group_bootstrap(scores, y, groups, threshold)
    transition_ledgers = {
        "GC_vs_G": _confusion_transitions(predictions["G"], predictions["GC"], y),
        "GC_vs_SC": _confusion_transitions(predictions["SC"], predictions["GC"], y),
    }
    combination_only = {}
    wrong = {arm: predictions[arm] != y for arm in ARMS}
    right = {arm: predictions[arm] == y for arm in ARMS}
    for label, yvalue in (("normal", 0), ("anomalous", 1)):
        class_mask = y == yvalue
        combination_only[label] = {
            "repair": int((class_mask & wrong["S"] & wrong["G"] &
                           wrong["SC"] & right["GC"]).sum()),
            "harm": int((class_mask & right["S"] & right["G"] &
                         right["SC"] & wrong["GC"]).sum()),
        }
    ranking = {"GC_vs_G": _pair_credit_changes(scores["G"], scores["GC"], y, groups),
               "GC_vs_SC": _pair_credit_changes(scores["SC"], scores["GC"], y, groups)}

    unique_groups = set(groups.tolist())
    query_group_count = len(set(groups[query_mask].tolist()))
    if query_group_count != int(query_mask.sum()):
        raise R3Error(f"{category}: more than one queried view from a Source Group")
    query_labels = y[query_mask]
    query_feedback = {"total": int(query_mask.sum()),
                      "normal": int((query_labels == 0).sum()),
                      "anomalous": int((query_labels == 1).sum()),
                      "unqueried_label_access_in_stage_a": False}
    quartiles = []
    bounds = [int(math.floor(i * len(y) / 4)) for i in range(5)]
    for quartile in range(4):
        begin, end = bounds[quartile], bounds[quartile + 1]
        qlabels = y[begin:end]
        qentry: dict[str, Any] = {
            "quartile": quartile + 1, "start_position": begin,
            "end_position_exclusive": end, "events": end - begin,
            "normal_events": int((qlabels == 0).sum()),
            "anomalous_events": int((qlabels == 1).sum()),
            "normal_proportion": float((qlabels == 0).mean()) if len(qlabels) else None,
            "anomalous_proportion": float((qlabels == 1).mean()) if len(qlabels) else None,
            "query_events": int(query_mask[begin:end].sum()),
            "query_normal_feedback": int(((query_mask[begin:end]) & (qlabels == 0)).sum()),
            "query_anomalous_feedback": int(((query_mask[begin:end]) & (qlabels == 1)).sum()),
            "score_effects": {},
        }
        qscore_metrics = {}
        for arm in ARMS:
            qentry["score_effects"][arm] = {
                "mean_score": float(scores[arm][begin:end].mean()),
                "median_score": float(np.median(scores[arm][begin:end])),
            }
            qscore_metrics[arm] = _weighted_rank_metrics(scores[arm][begin:end], qlabels)
        qentry["score_effects"]["GC_minus_G_mean"] = float(
            scores["GC"][begin:end].mean() - scores["G"][begin:end].mean())
        qentry["score_effects"]["GC_minus_SC_mean"] = float(
            scores["GC"][begin:end].mean() - scores["SC"][begin:end].mean())
        qentry["ranking_metrics"] = qscore_metrics
        quartiles.append(qentry)

    first_anomaly = int(np.flatnonzero(y == 1)[0]) if np.any(y == 1) else None
    anomalous_queries = [int(pos) for pos in query_positions if y[pos] == 1]
    correction_summary = category_receipt.get("correction_stats", {})
    return {
        "events": len(y), "source_group_count": len(unique_groups),
        "source_group_disjointness": {
            "query_unique_groups": query_group_count,
            "queries_with_duplicate_source_group": 0,
            "support_c64_disjointness_verified_in_stage_a": True,
        },
        "outcomes": {"normal": int((y == 0).sum()),
                     "anomalous": int((y == 1).sum()),
                     "anomaly_prevalence": float((y == 1).mean())},
        "query_feedback": query_feedback,
        "runtime": {
            "wall_seconds": float(category_receipt.get("wall_seconds", 0.0)),
            "stream_image_encodings": int(category_receipt.get("actual_stream_image_encodings", 0)),
            "support_encoder_invocations": int(category_receipt.get("support_encoder_invocations", 0)),
            "peak_cuda_allocated_bytes": int(category_receipt.get("peak_cuda_allocated_bytes", 0)),
            "peak_cuda_reserved_bytes": int(category_receipt.get("peak_cuda_reserved_bytes", 0)),
            "peak_cpu_rss_bytes": category_receipt.get("peak_cpu_rss_bytes"),
        },
        "first_anomaly_event_position": first_anomaly,
        "first_queried_anomalous_feedback_position": min(anomalous_queries) if anomalous_queries else None,
        "frozen_threshold": float(threshold),
        "arms": metrics,
        "factorial_contrasts": contrasts,
        "paired_source_group_bootstrap": bootstrap,
        "error_transitions": transition_ledgers,
        "combination_only": combination_only,
        "ranking_pair_diagnostics": ranking,
        "normal_memory_and_correction_updates": {
            "greedy_write_events": int(category_receipt.get("greedy_write_events", 0)),
            "greedy_accepted_patch_count": int(category_receipt.get("greedy_accepted_patch_count", 0)),
            "greedy_skipped_patch_count": int(category_receipt.get("greedy_skipped_patch_count", 0)),
            "correction_update_stats": correction_summary,
            "correction_final_occupancy": category_receipt.get("correction_final_occupancy"),
            "correction_max_occupancy": category_receipt.get("correction_max_occupancy"),
        },
        "fixed_stream_quartiles": quartiles,
        "first512_exp2a_parity": category_receipt.get("first512_exp2a_parity"),
    }


def _read_stage_b_label_projection(path: Path, expected_count: int):
    """Read only image-level outcomes, tolerating whitespace before final `]`.

    The historical restricted reader skips every non-label field, but its final
    record boundary check rejects legal whitespace between the last TEST object
    and the array terminator. Reuse its incremental projection for preceding
    rows and its lexical skip helpers for the final row. Mask paths and files are
    never opened.
    """
    import numpy as np
    from experiments.calibration.stage1a_p12_label_provider import (
        _RestrictedTestArray, _decode_selected_label, _skip_value, _ws,
    )
    count = int(expected_count)
    if count <= 0:
        raise R3Error("Stage B label projection needs a positive frozen event count")
    reader = _RestrictedTestArray(Path(path))
    try:
        labels = np.empty(count, dtype=np.int8)
        for position in range(count - 1):
            try:
                labels[position] = reader.label_at(position)
            except Exception as exc:
                raise R3Error(
                    f"Stage B restricted label projection failed at frozen position {position}"
                ) from exc
        data = reader._data
        start = _ws(data, reader._cursor)
        if start >= len(data) or data[start] == ord("]"):
            raise R3Error("Stage B TEST array ended before the frozen event count")
        record_stop = _skip_value(data, start)
        delimiter = _ws(data, record_stop)
        if delimiter >= len(data) or data[delimiter] != ord("]"):
            raise R3Error("Final Stage B TEST record is not followed by the array terminator")
        labels[-1] = _decode_selected_label(data, start, delimiter, reader.normal_class)
        if not np.isin(labels, (0, 1)).all():
            raise R3Error("Stage B label projection contains an invalid binary outcome")
        return labels
    finally:
        reader.close()


def _run_stage_b_child(run_id: str) -> dict[str, Any]:
    import numpy as np
    import torch
    if torch.cuda.device_count() != 0 or torch.cuda.is_initialized():
        raise R3Error("Stage B must run with CUDA hidden and no initialized CUDA context")
    output = _existing_output_path(run_id)
    manifest = _read_json(output / "run_manifest.json")
    _verify_source_binding(manifest)
    seal_path = output / "stage_a_seal.json"
    scores_path = output / "stage_a_scores.jsonl"
    if (manifest.get("mode") != "FULL_STAGE_A" or not seal_path.is_file() or
            not scores_path.is_file()):
        raise R3Error("Stage B requires a completed full-stream Stage A run")
    stage_a_seal = _read_json(seal_path)
    if (stage_a_seal.get("schema") != STAGE_A_SEAL_SCHEMA or
            stage_a_seal.get("status") != "SEALED_COMPLETE" or
            stage_a_seal.get("score_rows_sha256") != sha256_file(scores_path) or
            int(stage_a_seal.get("event_count", -1)) != sum(EXPECTED_COUNTS.values()) or
            stage_a_seal.get("category_event_counts") != EXPECTED_COUNTS or
            set(stage_a_seal.get("category_seals", {})) != set(CATEGORIES)):
        raise R3Error("Stage A score trajectory is not integrity-verified and complete")

    rows_by_category: dict[str, list[dict[str, Any]]] = {cat: [] for cat in CATEGORIES}
    with scores_path.open("r", encoding="utf-8") as stream:
        for row_index, line in enumerate(stream, 1):
            row = json.loads(line)
            if (row.get("schema") != SCORE_SCHEMA or
                    "label" in row or "anomaly_class" in row or "mask_path" in row):
                raise R3Error(f"Stage A score row {row_index} violates label/mask firewall")
            category = str(row.get("category"))
            if category not in rows_by_category:
                raise R3Error("Stage A score row has an unauthorized category")
            rows_by_category[category].append(row)
    for category in CATEGORIES:
        rows = rows_by_category[category]
        if (len(rows) != EXPECTED_COUNTS[category] or
                [int(row.get("position", -1)) for row in rows] !=
                list(range(EXPECTED_COUNTS[category])) or
                int(stage_a_seal["category_seals"][category].get("event_count", -1)) !=
                EXPECTED_COUNTS[category]):
            raise R3Error(f"{category}: score rows are not the complete ordered stream")
    if sum(map(len, rows_by_category.values())) != int(stage_a_seal["event_count"]):
        raise R3Error("Stage A score row total differs from the completion seal")

    history_rows: list[dict[str, Any]] = []
    history_path = output / "attempt_history.jsonl"
    if history_path.is_file():
        with history_path.open("r", encoding="utf-8") as stream:
            history_rows = [json.loads(line) for line in stream if line.strip()]
    category_receipts = {str(row.get("category")): row for row in history_rows
                         if row.get("record_type") == "category_complete"}
    if set(category_receipts) != set(CATEGORIES):
        raise R3Error("Stage B requires four category completion receipts")

    contract = load_frozen_contract(verify_image_paths=False)
    for category in CATEGORIES:
        seal = stage_a_seal["category_seals"][category]
        rows = rows_by_category[category]
        expected_queries = list(contract["classes"][category]["query_positions"])
        actual_queries = [int(row["position"]) for row in rows if row["query_tape_decision"]]
        category_rows_digest = hashlib.sha256(
            b"".join(canonical_bytes(row) for row in rows)).hexdigest()
        feedback_rows = [row for row in history_rows
                         if row.get("record_type") == "queried_feedback_and_updates" and
                         row.get("category") == category]
        feedback_digest = hashlib.sha256(
            b"".join(canonical_bytes(row) for row in feedback_rows)).hexdigest()
        if (actual_queries != expected_queries or
                int(seal.get("query_count", -1)) != len(expected_queries) or
                seal.get("query_tape_sha256") != contract["classes"][category]["query_tape_sha256"] or
                seal.get("query_positions_sha256") != contract["classes"][category]["query_positions_sha256"] or
                seal.get("category_score_rows_sha256") != category_rows_digest or
                seal.get("query_receipts_sha256") != category_receipts[category].get("query_receipts_sha256") or
                seal.get("query_receipts_sha256") != feedback_digest or
                [int(row.get("position", -1)) for row in feedback_rows] != expected_queries):
            raise R3Error(f"{category}: Stage A category seals do not match source/tape receipts")
    category_results: dict[str, Any] = {}
    for category in CATEGORIES:
        info = contract["classes"][category]
        if sha256_file(JSON_ROOT / f"{category}.json") != info["category_json_sha256"]:
            raise R3Error(f"{category}: evaluator JSON source changed after preflight")
        # This restricted projection skips every TEST field except anomaly_class;
        # it never opens mask files and runs only after the Stage A seal checks.
        labels = _read_stage_b_label_projection(
            JSON_ROOT / f"{category}.json", EXPECTED_COUNTS[category])
        category_results[category] = _category_stage_b(
            category, rows_by_category[category], labels,
            float(info["theta_c64_q90"]), history_rows, category_receipts[category])

    macro_auroc = [category_results[c]["arms"]["GC"]["auroc"] for c in CATEGORIES]
    macro_defined = all(value is not None for value in macro_auroc)
    query_totals = {
        "all": sum(category_results[c]["query_feedback"]["total"] for c in CATEGORIES),
        "normal": sum(category_results[c]["query_feedback"]["normal"] for c in CATEGORIES),
        "anomalous": sum(category_results[c]["query_feedback"]["anomalous"] for c in CATEGORIES),
    }
    aggregate = {
        "schema": "r3.fullstream.2x2.aggregate.v1",
        "classification": "RETROSPECTIVE_FULL_STREAM_DEVELOPMENT_EVALUATION",
        "status": "COMPLETE",
        "dataset_protocol": {
            "categories": list(CATEGORIES),
            "events": sum(EXPECTED_COUNTS.values()),
            "event_order": "frozen prequential order from original Real-IAD TEST JSON; physical chronology not asserted",
            "labels_accessed_only_after_all_stage_a_scores_sealed": True,
            "test_masks_opened": False,
        },
        "query_feedback_total": query_totals,
        "arms": {"S": "Static normal memory, no correction",
                 "SC": "Static normal memory + original C1-Full",
                 "G": "Greedy normal memory, no correction",
                 "GC": "Greedy normal memory + original C1-Full"},
        "categories": category_results,
        "four_category_gc_auroc_mean": (
            float(sum(float(x) for x in macro_auroc) / 4) if macro_defined else None),
        "all_four_category_gc_aurocs_defined": macro_defined,
        "categories_with_defined_auroc": int(sum(value is not None for value in macro_auroc)),
        "interpretation_boundary": "Fixed exposed development streams; categories are not independent population samples.",
        "public_aggregate_redaction": {
            "raw_event_identities_present": False,
            "private_labels_present": False,
            "source_group_identifiers_present": False,
            "raw_scores_present": False,
        },
        "run_id": run_id,
        "source_parent_commit": manifest["source_parent_commit"],
        "source_fingerprints": manifest["source_fingerprints"],
        "model_and_query_identity": manifest["model_and_query_identity"],
        "stage_a_seal_sha256": sha256_file(seal_path),
        "score_rows_sha256": sha256_file(scores_path),
    }
    budget = _budget(output)
    budget.write_json("aggregate_results.json", aggregate)
    stage_b_seal = {
        "schema": STAGE_B_SEAL_SCHEMA, "status": "SEALED_COMPLETE",
        "run_id": run_id,
        "stage_a_seal_sha256": sha256_file(seal_path),
        "score_rows_sha256": sha256_file(scores_path),
        "aggregate_results_sha256": sha256_file(output / "aggregate_results.json"),
        "category_event_counts": dict(EXPECTED_COUNTS),
        "outcome_labels_read_after_stage_a_seal": True,
        "test_masks_opened": False,
        "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    budget.write_json("stage_b_seal.json", stage_b_seal)
    budget.write_json("complete_seal.json", {
        "schema": "r3.fullstream.2x2.complete.v1", "status": "COMPLETE",
        "run_id": run_id,
        "stage_a_seal_sha256": sha256_file(seal_path),
        "stage_b_seal_sha256": sha256_file(output / "stage_b_seal.json"),
        "aggregate_results_sha256": sha256_file(output / "aggregate_results.json"),
        "event_count": sum(EXPECTED_COUNTS.values()),
        "automatic_restart": False,
    })
    return aggregate


def _experiment_identity(contract: Mapping[str, Any], exp2a_identity: Mapping[str, Any],
                         sources: Mapping[str, str], model_receipt: Mapping[str, Any],
                         mode: str) -> dict[str, Any]:
    category_identity = {}
    for category in CATEGORIES:
        info = contract["classes"][category]
        category_identity[category] = {
            "stream_count": info["stream_count"],
            "category_json_sha256": info["category_json_sha256"],
            "support_image_sha256": info["support_image_sha256"],
            "support_group_sha256": info["support_group_sha256"],
            "c64_image_sha256": info["c64_image_sha256"],
            "c64_group_sha256": info["c64_group_sha256"],
            "c64_score_tape_sha256": info["c64_score_tape_sha256"],
            "fixed_support_sha256_expected": info["fixed_support_sha256_expected"],
            "theta_c64_q90": info["theta_c64_q90"],
            "sigma_c64_floor_002": info["sigma_c64_floor_002"],
            "query_positions_sha256": info["query_positions_sha256"],
            "query_tape_sha256": info["query_tape_sha256"],
        }
    config = {
        "profile": "R3-FULL-2X2", "mode": mode,
        "categories": list(CATEGORIES), "expected_counts": EXPECTED_COUNTS,
        "arms": list(ARMS), "support_seed": 0,
        "feature_fuse": "concat", "four_support_images": True,
        "four_support_augmentation_views": True,
        "fixed_capacity": 25000, "fixed_per_image": 5000,
        "topk_patches": 32, "image_aggregation": "max", "knn": 5,
        "c64_is_threshold_calibration_only": True,
        "greedy_dynamic_capacity": 256,
        "greedy_dynamic_state_safety_cap_bytes": 2 * 1024**3,
        "correction": dict(CORRECTION_CONFIG),
        "query_protocol": "frozen Stage 0.5 Q-A-TS-v1 rate=0.03 seed=37 bins=16",
        "score_seal_before_query_feedback": True,
        "stage_b_cpu_only_after_all_category_seals": True,
        "dino_checkpoint_downloads": False,
        "feature_encoding_per_test_event": 1,
        "output_limit_bytes": OUTPUT_LIMIT,
        "failure_reserve_bytes": FAILURE_RESERVE,
        "stage_a_category_deadline_seconds": CATEGORY_DEADLINE_SECONDS,
        "total_supervised_compute_deadline_seconds": TOTAL_DEADLINE_SECONDS,
        "stage_b_bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "stage_b_bootstrap_seed": BOOTSTRAP_SEED,
        "parity_atol": ATOL, "parity_rtol": RTOL,
    }
    identity = {
        "schema": "r3.fullstream.2x2.identity.v1",
        "source_parent_commit": EXPECTED_BASELINE,
        "source_fingerprints": dict(sources),
        "runner_source_sha256": sha256_file(Path(__file__).resolve()),
        "config": config,
        "category_identity": category_identity,
        "stage05": {key: contract[key] for key in (
            "stage05_run_manifest_sha256", "frozen_inputs_sha256",
            "score_tape_manifest_sha256", "query_tape_archive_sha256")},
        "exp2a_reference": dict(exp2a_identity),
        "verified_model": {key: model_receipt.get(key) for key in (
            "hub_source_tree_sha256", "checkpoint_sha256",
            "loaded_model_state_sha256", "profile_id", "source_commit")},
    }
    identity["identity_sha256"] = sha256_json(identity)
    return identity


def _initialize_run(run_id: str, mode: str,
                    contract: Mapping[str, Any],
                    exp2a_identity: Mapping[str, Any],
                    source_identity: Mapping[str, str],
                    model_receipt: Mapping[str, Any]) -> Path:
    root = _output_path(run_id)
    identity = _experiment_identity(contract, exp2a_identity,
                                    source_identity, model_receipt, mode)
    manifest = {
        "schema": RUN_SCHEMA, "run_id": run_id, "mode": mode,
        "authorization": AUTHORIZATION,
        "classification": "RETROSPECTIVE_FULL_STREAM_DEVELOPMENT_EVALUATION",
        "status": "INCOMPLETE",
        "source_parent_commit": EXPECTED_BASELINE,
        "branch": "calibration-stage0-contract-20261029",
        "source_fingerprints": dict(source_identity),
        "runner_source_sha256": identity["runner_source_sha256"],
        "identity_sha256": identity["identity_sha256"],
        "experiment_configuration": identity["config"],
        "stage05_input_hashes": identity["stage05"],
        "exp2a_first512_reference": dict(exp2a_identity),
        "model_and_query_identity": {
            "stage05_model_provenance": contract["stage05_model_provenance"],
            "stage05_score_source_mismatches": {
                rel: {"frozen_sha256": expected,
                      "current_sha256": contract["stage05_current_source_hashes"].get(rel)}
                for rel, expected in contract["stage05_scoring_source_hashes"].items()
                if contract["stage05_current_source_hashes"].get(rel) != expected
            },
            "query_tape_sha256_by_category": {
                category: contract["classes"][category]["query_tape_sha256"]
                for category in CATEGORIES},
            "query_position_digest_by_category": {
                category: contract["classes"][category]["query_positions_sha256"]
                for category in CATEGORIES},
            "fixed_support_sha256_expected_by_category": {
                category: contract["classes"][category]["fixed_support_sha256_expected"]
                for category in CATEGORIES},
            "c64_score_tape_sha256_by_category": {
                category: contract["classes"][category]["c64_score_tape_sha256"]
                for category in CATEGORIES},
        },
        "output_storage": {"max_bytes": OUTPUT_LIMIT,
                           "failure_reserve_bytes": FAILURE_RESERVE,
                           "max_files": MAX_FILES,
                           "os_hard_storage_isolation": False,
                           "all_writes_budgeted_prewrite": True,
                           "unique_non_overwriting_namespace": True},
        "offline_outcomes_accessed": False,
        "test_masks_accessed": False,
        "unqueried_test_labels_accessed": False,
        "automatic_restart": False,
    }
    budget = _budget(root, create=True)
    budget.write_json("run_manifest.json", manifest)
    return root


def _reuse_verified_categories(target_root: Path, source_run_id: str,
                              contract: Mapping[str, Any]) -> tuple[set[str], float]:
    """Copy only whole, identity-matched categories into this new run namespace."""
    source_root = _existing_output_path(source_run_id)
    source_manifest = _read_json(source_root / "run_manifest.json")
    target_manifest = _read_json(target_root / "run_manifest.json")
    source_runner_compatibility = _verify_reusable_stage_a_binding(source_manifest)
    if source_manifest.get("mode") != "FULL_STAGE_A":
        raise R3Error("Category reuse source does not have the exact same frozen experiment identity")
    exp2a_identity, _ = _load_exp2a_score_references()
    model_receipt = _get_verified_model_receipt()
    current_identity = _experiment_identity(
        contract, exp2a_identity, _source_fingerprints(), model_receipt, "FULL_STAGE_A")
    if current_identity["identity_sha256"] != target_manifest.get("identity_sha256"):
        raise R3Error("New run manifest identity does not match its frozen inputs")
    if source_runner_compatibility == "same-runner-source":
        if source_manifest.get("identity_sha256") != target_manifest.get("identity_sha256"):
            raise R3Error("Category reuse source does not have the exact same frozen experiment identity")
    elif _identity_sha_with_runner(
            current_identity, str(source_manifest.get("runner_source_sha256", ""))) != \
            source_manifest.get("identity_sha256"):
        raise R3Error("Stage A reuse identity differs beyond the approved Stage B-only runner repair")
    history = _read_history(source_root)
    receipts = _category_receipts(source_root)
    if not receipts:
        return set(), 0.0
    source_scores = source_root / "stage_a_scores.jsonl"
    if not source_scores.is_file():
        raise R3Error("Category reuse source lacks its score trajectory")
    rows_by_category: dict[str, list[dict[str, Any]]] = {cat: [] for cat in CATEGORIES}
    with source_scores.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            row = json.loads(line)
            category = str(row.get("category"))
            if (category not in rows_by_category or row.get("schema") != SCORE_SCHEMA or
                    any(key in row for key in ("label", "anomaly_class", "mask_path"))):
                raise R3Error(f"Category reuse source has a forbidden/malformed score row {line_number}")
            rows_by_category[category].append(row)
    reused: set[str] = set()
    reused_seconds = 0.0
    target_budget = _budget(target_root)
    for category in CATEGORIES:
        receipt = receipts.get(category)
        if receipt is None:
            # Ignore partial score rows. The target restarts that category at position zero.
            continue
        rows = rows_by_category[category]
        info = contract["classes"][category]
        expected_queries = list(info["query_positions"])
        actual_queries = [int(row["position"]) for row in rows
                          if bool(row.get("query_tape_decision"))]
        score_digest = hashlib.sha256(b"".join(canonical_bytes(row) for row in rows)).hexdigest()
        query_records = [row for row in history
                         if row.get("record_type") == "queried_feedback_and_updates" and
                         row.get("category") == category]
        query_digest = hashlib.sha256(b"".join(canonical_bytes(row)
                                              for row in query_records)).hexdigest()
        if (len(rows) != EXPECTED_COUNTS[category] or
                [int(row.get("position", -1)) for row in rows] !=
                list(range(EXPECTED_COUNTS[category])) or
                actual_queries != expected_queries or
                receipt.get("events_processed") != EXPECTED_COUNTS[category] or
                receipt.get("score_rows_sha256") != score_digest or
                receipt.get("query_receipts_sha256") != query_digest or
                [int(row.get("position", -1)) for row in query_records] != expected_queries or
                receipt.get("query_tape_sha256") != info["query_tape_sha256"] or
                receipt.get("fixed_support_sha256") != info["fixed_support_sha256_expected"] or
                receipt.get("first512_exp2a_parity", {}).get("checked_events") != 512):
            raise R3Error(f"{category}: completed category failed source/tape/score integrity verification")
        for feedback in query_records:
            if (feedback.get("disclosure_after_four_score_seal") is not True or
                    set(feedback.get("correction_updates", {})) != {"SC", "GC"}):
                raise R3Error(f"{category}: queried feedback receipt is not a complete four-arm disclosure")
        payload = b"".join(canonical_bytes(row) for row in rows)
        if "stage_a_scores.jsonl" in target_budget.created:
            target_budget.append("stage_a_scores.jsonl", payload)
        else:
            target_budget.write_once("stage_a_scores.jsonl", payload)
        history_payload = b"".join(canonical_bytes(row) for row in query_records + [receipt])
        if "attempt_history.jsonl" in target_budget.created:
            target_budget.append("attempt_history.jsonl", history_payload)
        else:
            target_budget.write_once("attempt_history.jsonl", history_payload)
        reuse_row = {
            "schema": HISTORY_SCHEMA, "record_type": "category_reused",
            "category": category, "source_run_id": source_run_id,
            "event_count": EXPECTED_COUNTS[category],
            "score_rows_sha256": score_digest,
            "query_receipts_sha256": query_digest,
            "query_tape_sha256": info["query_tape_sha256"],
            "same_identity_sha256": target_manifest["identity_sha256"],
            "source_stage_a_runner_sha256": source_manifest["runner_source_sha256"],
            "target_runner_sha256": target_manifest["runner_source_sha256"],
            "source_runner_compatibility": source_runner_compatibility,
            "automatic_restart": False,
        }
        _append_history(target_budget, reuse_row)
        reused.add(category)
        reused_seconds += float(receipt.get("wall_seconds", 0.0))
    target_budget.write_json("stage_a_reuse_receipt.json", {
        "schema": "r3.fullstream.2x2.reuse.v1",
        "source_run_id": source_run_id,
        "target_run_id": target_root.name,
        "identity_sha256": target_manifest["identity_sha256"],
        "source_stage_a_runner_sha256": source_manifest["runner_source_sha256"],
        "target_runner_sha256": target_manifest["runner_source_sha256"],
        "source_runner_compatibility": source_runner_compatibility,
        "reused_categories": [cat for cat in CATEGORIES if cat in reused],
        "reused_category_compute_seconds": reused_seconds,
        "automatic_restart": False,
    })
    return reused, reused_seconds


def _append_supervisor_result(root: Path, stage: str, category: str | None,
                              result: Any) -> dict[str, Any]:
    result_dict = asdict(result)
    row = {
        "schema": HISTORY_SCHEMA, "record_type": "supervisor_result",
        "stage": stage, "category": category, "result": result_dict,
        "process_tree_containment": "existing Windows Job Object supervisor; suspended child assignment; kill-on-close",
        "automatic_restart": False,
    }
    budget = _budget(root)
    _append_history(budget, row)
    return row


def _read_history(root: Path) -> list[dict[str, Any]]:
    path = root / "attempt_history.jsonl"
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def _category_receipts(root: Path) -> dict[str, dict[str, Any]]:
    receipts: dict[str, dict[str, Any]] = {}
    for row in _read_history(root):
        if row.get("record_type") == "category_complete":
            category = str(row.get("category"))
            if category in receipts:
                raise R3Error(f"Duplicate category completion receipt: {category}")
            receipts[category] = row
    return receipts


def _seal_stage_a(root: Path, contract: Mapping[str, Any]) -> dict[str, Any]:
    scores_path = root / "stage_a_scores.jsonl"
    history = _read_history(root)
    receipts = _category_receipts(root)
    if set(receipts) != set(CATEGORIES):
        raise R3Error("Cannot seal Stage A without all four completed category receipts")
    seen: dict[str, list[dict[str, Any]]] = {cat: [] for cat in CATEGORIES}
    digest_by_category = {cat: hashlib.sha256() for cat in CATEGORIES}
    with scores_path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            row = json.loads(line)
            category = str(row.get("category"))
            if category not in seen or row.get("schema") != SCORE_SCHEMA:
                raise R3Error(f"Stage A score identity malformed at row {line_number}")
            if "label" in row or "anomaly_class" in row or "mask_path" in row:
                raise R3Error("Stage A scores contain forbidden label or mask fields")
            seen[category].append(row)
            digest_by_category[category].update(canonical_bytes(row))
    category_seals = {}
    total_queries = 0
    for category in CATEGORIES:
        rows = seen[category]
        expected_n = EXPECTED_COUNTS[category]
        if [int(row["position"]) for row in rows] != list(range(expected_n)):
            raise R3Error(f"{category}: Stage A does not contain the exact full frozen stream")
        actual_queries = [int(row["position"]) for row in rows if row["query_tape_decision"]]
        expected_queries = list(contract["classes"][category]["query_positions"])
        if actual_queries != expected_queries:
            raise R3Error(f"{category}: Stage A query positions differ from frozen Q-A-TS tape")
        if digest_by_category[category].hexdigest() != receipts[category].get("score_rows_sha256"):
            raise R3Error(f"{category}: category score receipt digest does not match rows")
        query_records = [row for row in history
                         if row.get("record_type") == "queried_feedback_and_updates" and
                         row.get("category") == category]
        query_digest = hashlib.sha256()
        if [int(row["position"]) for row in query_records] != expected_queries:
            raise R3Error(f"{category}: private queried feedback receipt order differs from tape")
        for row in query_records:
            position = int(row["position"])
            score_row = rows[position]
            if (not score_row["query_tape_decision"] or
                    row.get("event_id_sha256") != score_row.get("event_id_sha256") or
                    row.get("source_group_sha256") != score_row.get("source_group_sha256") or
                    row.get("disclosure_after_four_score_seal") is not True or
                    set(row.get("correction_updates", {})) != {"SC", "GC"}):
                raise R3Error(f"{category}:{position}: queried feedback does not bind four sealed scores")
            predictions = score_row["predictions_at_frozen_threshold"]
            for branch, base_arm in (("SC", "S"), ("GC", "G")):
                update = row["correction_updates"][branch]
                expected_prediction = int(predictions[base_arm])
                expected_outcome = {(0, 0): "TN", (0, 1): "FP",
                                    (1, 0): "FN", (1, 1): "TP"}[
                                        (int(row["label"]), expected_prediction)]
                if (update.get("base_prediction") != expected_prediction or
                        update.get("outcome") != expected_outcome or
                        not _score_tolerance(update.get("base_score"),
                                             score_row["scores"][base_arm])):
                    raise R3Error(f"{category}:{position}: {branch} correction used non-native base outcome")
            writer_action = row.get("greedy_writer_action", {})
            if (writer_action.get("input_score_source") != "common_C0" or
                    writer_action.get("prediction") != int(predictions["G"]) or
                    (int(row["label"]) == 1 and
                     writer_action.get("action") != "verified_defect_no_normal_write")):
                raise R3Error(f"{category}:{position}: Greedy writer did not follow verified C0 feedback")
            query_digest.update(canonical_bytes(row))
        if query_digest.hexdigest() != receipts[category].get("query_receipts_sha256"):
            raise R3Error(f"{category}: feedback/update receipt digest mismatch")
        if any(set(row.get("correction_updates", {})) != {"SC", "GC"}
               for row in query_records):
            raise R3Error(f"{category}: SC and GC did not receive identical query opportunities")
        total_queries += len(expected_queries)
        category_seals[category] = {
            "event_count": expected_n,
            "query_count": len(expected_queries),
            "query_tape_sha256": contract["classes"][category]["query_tape_sha256"],
            "query_positions_sha256": contract["classes"][category]["query_positions_sha256"],
            "category_score_rows_sha256": digest_by_category[category].hexdigest(),
            "query_receipts_sha256": query_digest.hexdigest(),
            "fixed_support_sha256": receipts[category]["fixed_support_sha256"],
            "first512_exp2a_parity": receipts[category]["first512_exp2a_parity"],
            "consistency_checks": receipts[category]["consistency_checks"],
            "actual_stream_image_encodings": receipts[category]["actual_stream_image_encodings"],
            "greedy_write_events": receipts[category]["greedy_write_events"],
            "correction_final_occupancy": receipts[category]["correction_final_occupancy"],
            "correction_stats": receipts[category]["correction_stats"],
        }
    all_rows = sum(map(len, seen.values()))
    if all_rows != sum(EXPECTED_COUNTS.values()):
        raise R3Error("Stage A sealed trajectory does not contain exactly 15,295 events")
    seal = {
        "schema": STAGE_A_SEAL_SCHEMA,
        "status": "SEALED_COMPLETE",
        "run_id": root.name,
        "source_parent_commit": EXPECTED_BASELINE,
        "category_event_counts": dict(EXPECTED_COUNTS),
        "event_count": all_rows,
        "query_count": total_queries,
        "score_rows_sha256": sha256_file(scores_path),
        "category_seals": category_seals,
        "feedback_disclosure_order_verified": True,
        "current_event_score_fsynced_before_query_label": True,
        "unqueried_labels_accessed": False,
        "offline_outcome_labels_accessed": False,
        "test_masks_accessed": False,
        "automatic_restart": False,
        "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    budget = _budget(root)
    budget.write_json("stage_a_seal.json", seal)
    return seal


def _run_supervised_child(root: Path, *, stage: str,
                          category: str | None = None,
                          smoke_events: int = 0,
                          timeout_seconds: float,
                          authorized_cuda_visibility: str | None = None) -> Any:
    from experiments.calibration.r1_runtime_supervisor import run_supervised_child
    command = [sys.executable, "-m", "experiments.calibration.r3_fullstream_2x2"]
    command.append("--authorize-full-stream")
    if stage == "STAGE_A":
        command += ["--stage-a-child", "--run-id", root.name,
                    "--category", str(category)]
        if smoke_events:
            command += ["--smoke-events", str(int(smoke_events))]
    elif stage == "STAGE_B":
        command += ["--stage-b-child", "--run-id", root.name]
    else:
        raise R3Error(f"Unsupported supervised child stage: {stage}")
    child_env = _child_environment(os.environ, stage, authorized_cuda_visibility)
    return run_supervised_child(
        command, timeout_seconds=float(timeout_seconds), env=child_env,
        cwd=ROOT, output_dir=None,
    )


def _record_supervisor_status(root: Path, status: Mapping[str, Any]) -> None:
    budget = _budget(root)
    budget.write_json("supervisor_status.json", dict(status))


def _run_cpu_tests() -> dict[str, Any]:
    from experiments.calibration.r1_runtime_supervisor import run_supervised_child
    command = [sys.executable, "-m", "pytest", "-q",
               "tests/calibration/test_r3_fullstream_2x2.py"]
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = ""
    env["R3_SYNTHETIC_CPU_ONLY"] = "1"
    result = run_supervised_child(command, timeout_seconds=TEST_DEADLINE_SECONDS,
                                  env=env, cwd=ROOT)
    if result.status != "COMPLETED" or result.child_exit_code != 0:
        raise R3Error(f"Bounded synthetic CPU test suite failed: {asdict(result)}")
    return asdict(result)


def _write_public_deliverables(root: Path, aggregate: Mapping[str, Any],
                               report_context: Mapping[str, Any]) -> tuple[Path, Path]:
    aggregate_source = root / "aggregate_results.json"
    if sha256_file(aggregate_source) != sha256_bytes(canonical_bytes(aggregate)):
        raise R3Error("Public aggregate differs from its sealed Stage B payload")
    if REPORT_DIR.exists():
        raise FileExistsError(f"Refusing to overwrite an existing audit deliverable directory: {REPORT_DIR}")
    REPORT_DIR.mkdir(parents=True, exist_ok=False)
    report_path = REPORT_DIR / "REPORT.md"
    aggregate_path = REPORT_DIR / "aggregate_results.json"
    aggregate_path.write_bytes(aggregate_source.read_bytes())
    report_path.write_text(_render_report(aggregate, report_context),
                           encoding="utf-8", newline="\n")
    return report_path, aggregate_path


def _format_number(value: Any, digits: int = 6) -> str:
    if value is None:
        return "undefined"
    return f"{float(value):.{digits}f}"


def _render_report(aggregate: Mapping[str, Any], context: Mapping[str, Any]) -> str:
    categories = aggregate["categories"]
    lines = [
        "# R3-FULL-2X2 — Full-Stream Dual-Memory Complementarity Evaluation",
        "",
        "## Classification and status",
        "",
        "**Status: COMPLETE — `RETROSPECTIVE_FULL_STREAM_DEVELOPMENT_EVALUATION`.** "
        "Earlier development outcome summaries were already known. These results are "
        "not independent, blind, or confirmation evidence. The four arms, memory "
        "rules, correction policy, query tape and evaluation contrasts were frozen "
        "for this run; no post-result parameter search was performed.",
        "",
        "The Stage A online pass sealed all 15,295 event scores before Stage B read "
        "the full TEST outcome projection. Stage A decoded only the current legal "
        "Q-A-TS queried label after fsync of all four current event scores. TEST masks "
        "were not opened. The order is the frozen prequential JSON order; timestamps "
        "are retained as metadata and are not asserted to represent physical chronology.",
        "",
        "## Run identity and runtime",
        "",
        f"- Run ID: `{context['run_id']}`",
        f"- Source parent: `{EXPECTED_BASELINE}` on `{context['branch']}`",
        f"- Local commit: `{context.get('commit_sha', 'same scoped commit as this report')}`",
        f"- GPU: {context['gpu_name']}; CUDA runtime {context['cuda_runtime']}.",
        f"- Runtime controls: `{json.dumps(context['runtime_profile'], sort_keys=True)}`.",
        f"- Stage A stream encodings: {sum(EXPECTED_COUNTS.values()):,} — exactly one encoder invocation per TEST event.",
        f"- Support augmentation encodings: {context['support_encodings']:,} total across the four categories.",
        f"- Supervised Stage A wall time: {context['stage_a_wall_seconds']:.1f} seconds; Stage B CPU time: {context['stage_b_wall_seconds']:.1f} seconds.",
        f"- Peak CUDA allocated/reserved: {context['peak_cuda_allocated_bytes']:,} / {context['peak_cuda_reserved_bytes']:,} bytes.",
        f"- Local private output: `outputs/calibration/{context['run_id']}/` ({context['output_bytes']:,} bytes; 4 GiB application bound; 64 KiB failure reserve).",
        f"- Recovery history: {context['recovery_summary']}.",
        f"- Stage A score runner SHA-256: `{context['stage_a_runner_sha256']}`; Stage B and finalizer runner SHA-256: `{context['runner_sha256']}`.",
        f"- Stage A reuse provenance: source run `{context['stage_a_reuse_source_run_id']}`; compatibility `{context['stage_a_reuse_compatibility']}`. The final runner changes were limited to the Stage B final-record projection and parent/report bookkeeping; frozen Stage A scoring, parameters, identities, and query positions were unchanged.",
        f"- The 120-minute Windows Job Object supervisor was used separately for each category child and Stage B; the parent enforced an eight-hour total ceiling including tests, smoke and reused categories ({context['total_execution_wall_seconds']:.1f} seconds charged). No automatic restart of an incomplete category occurred.",
        "- The application byte accountant provides no OS/filesystem hard quota. Its pre-write accounting, unique namespace and disk-space preflight were active.",
        "",
        "## Frozen inputs and method configuration",
        "",
        f"- DINO source tree SHA-256: `{context['hub_source_tree_sha256']}`; checkpoint SHA-256: `{context['checkpoint_sha256']}`; loaded state SHA-256: `{context['loaded_model_state_sha256']}`.",
        f"- Support seed: 0; four deployed Support images with four augmentation views each. The Support bank used the native structured fixed-bank digest; C64's 64 disjoint normal TRAIN references supplied only each category's initial threshold and sigma.",
        f"- Fixed-bank structured SHA-256 by category: `{json.dumps(context['fixed_support_sha256'], sort_keys=True)}`.",
        f"- Support image and Source Group identity SHA-256 by category: `{json.dumps(context['support_image_sha256'], sort_keys=True)}` / `{json.dumps(context['support_group_sha256'], sort_keys=True)}`.",
        f"- C64 image and Source Group identity SHA-256 by category: `{json.dumps(context['c64_image_sha256'], sort_keys=True)}` / `{json.dumps(context['c64_group_sha256'], sort_keys=True)}`; C64 score tape SHA-256: `{json.dumps(context['c64_score_tape_sha256'], sort_keys=True)}`.",
        f"- Existing Stage 0.5 Q-A-TS-v1 tape: 3% rate, seed 37, 16 bins, official Source Group unit. Tape SHA-256 by category: `{json.dumps(context['query_tape_sha256'], sort_keys=True)}`.",
        f"- First-512 EXP2A score reference SHA-256: `{context['exp2a_score_rows_sha256']}`. Parity was checked on G vs EXP2A C0 and GC vs EXP2A C1-Full at atol={ATOL:g}, rtol={RTOL:g}.",
        "- S/SC share only the immutable initial Support representation; SC correction state is independent. G/GC share the same Greedy normal-memory trajectory; GC correction never gates memory writes. C1-Full uses each branch's own base map, Top-32, native dN, base outcome and beta update.",
        "- No new correction kernel, beta policy, admission gate, query controller, threshold sidecar, label-based query selection, or online threshold adaptation was introduced.",
        f"- Exact frozen C1-Full configuration: `{json.dumps(context['correction_config'], sort_keys=True)}`; each arm uses its own codebook, source-group history, native Top-32 and pre-event outcome.",
        "",
        "### Historical fingerprint reconciliation",
        "",
        f"Stage 0.5's historical score manifest records DINO source-tree SHA `{context['stage05_hub_source_tree_sha256']}`; the later verified R1/EXP2A local source used here records `{context['hub_source_tree_sha256']}`. The local R1 receipt and checkpoint were verified, and the actual rebuilt structured fixed-support digests matched the Stage 0.5 receipts for all four categories. Stage 0.5 source hashes for `stage05.py` and `stage05_runner.py` differ from current files; the archived seed-37 tape digest and event/source-group identities were validated directly, and no new tape was sampled.",
        "",
        "Exact Stage 0.5 source hash mismatches (frozen -> current):",
        "",
    ]
    if context["stage05_source_hash_mismatches"]:
        for rel, hashes in sorted(context["stage05_source_hash_mismatches"].items()):
            lines.append(f"- `{rel}`: `{hashes['frozen_sha256']}` -> `{hashes['current_sha256']}`")
    else:
        lines.append("- None recorded by the frozen score manifest.")
    lines += [
        "",
        "## Full-stream event and query coverage",
        "",
        "| Category | Events / encodings | GPU seconds | Queries | Normal query feedback | Anomalous query feedback | First anomaly | First queried anomaly | Greedy writes | SC/GC final occupancy |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for category in CATEGORIES:
        result = categories[category]
        updates = result["normal_memory_and_correction_updates"]
        lines.append(
            f"| {category} | {result['events']} / {result['runtime']['stream_image_encodings']} | "
            f"{result['runtime']['wall_seconds']:.1f} | {result['query_feedback']['total']} | "
            f"{result['query_feedback']['normal']} | {result['query_feedback']['anomalous']} | "
            f"{result['first_anomaly_event_position'] if result['first_anomaly_event_position'] is not None else 'none'} | "
            f"{result['first_queried_anomalous_feedback_position'] if result['first_queried_anomalous_feedback_position'] is not None else 'none'} | "
            f"{updates['greedy_write_events']} | "
            f"{updates['correction_final_occupancy']['SC']}/{updates['correction_final_occupancy']['GC']} |"
        )
    lines += [
        "",
        f"Total query feedback was {aggregate['query_feedback_total']['all']} events "
        f"({aggregate['query_feedback_total']['normal']} normal, "
        f"{aggregate['query_feedback_total']['anomalous']} anomalous). Each query "
        "was one unique Source Group and was delivered identically to all four arms.",
        "",
        "## Primary metrics and factorial effects",
        "",
        "| Category | Arm | AUROC | AP | FPR | FNR | Balanced accuracy | TN/FP/FN/TP |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for category in CATEGORIES:
        for arm in ARMS:
            m = categories[category]["arms"][arm]
            c = m["confusion"]
            lines.append(
                f"| {category} | {arm} | {_format_number(m['auroc'])} | "
                f"{_format_number(m['average_precision'])} | {_format_number(m['fpr'])} | "
                f"{_format_number(m['fnr'])} | {_format_number(m['balanced_accuracy'])} | "
                f"{c['tn']}/{c['fp']}/{c['fn']}/{c['tp']} |"
            )
        effects = categories[category]["factorial_contrasts"]
        ci = categories[category]["paired_source_group_bootstrap"]["contrasts"]["auroc"]
        interval = {key: ci[key]["ci_95_percentile"] or [None, None]
                    for key in ("GC_minus_G", "GC_minus_SC", "SC_minus_S", "G_minus_S", "interaction")}
        lines += [
            f"| {category} | AUROC GC-G | {_format_number(effects['auroc']['GC_minus_G'])} | - | - | - | - | CI [{_format_number(interval['GC_minus_G'][0])}, {_format_number(interval['GC_minus_G'][1])}] |",
            f"| {category} | AUROC GC-SC | {_format_number(effects['auroc']['GC_minus_SC'])} | - | - | - | - | CI [{_format_number(interval['GC_minus_SC'][0])}, {_format_number(interval['GC_minus_SC'][1])}] |",
            f"| {category} | AUROC SC-S | {_format_number(effects['auroc']['SC_minus_S'])} | - | - | - | - | CI [{_format_number(interval['SC_minus_S'][0])}, {_format_number(interval['SC_minus_S'][1])}] |",
            f"| {category} | AUROC G-S | {_format_number(effects['auroc']['G_minus_S'])} | - | - | - | - | CI [{_format_number(interval['G_minus_S'][0])}, {_format_number(interval['G_minus_S'][1])}] |",
            f"| {category} | AUROC interaction I | {_format_number(effects['auroc']['interaction'])} | - | - | - | - | CI [{_format_number(interval['interaction'][0])}, {_format_number(interval['interaction'][1])}] |",
        ]
    lines += [
        "",
        "| Category | Metric | GC−G | GC−SC | SC−S | G−S | Interaction I |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for category in CATEGORIES:
        for metric in ("auroc", "average_precision", "balanced_accuracy"):
            effect = categories[category]["factorial_contrasts"][metric]
            lines.append(
                f"| {category} | {metric} | {_format_number(effect['GC_minus_G'])} | "
                f"{_format_number(effect['GC_minus_SC'])} | {_format_number(effect['SC_minus_S'])} | "
                f"{_format_number(effect['G_minus_S'])} | {_format_number(effect['interaction'])} |"
            )
    lines += [
        "",
        "All five requested AUROC contrasts and their paired 95% intervals are shown above and in the aggregate: GC-G, GC-SC, SC-S, G-S, and factorial interaction I. The same contrasts for AP and balanced accuracy are reported as point estimates; the aggregate also carries paired intervals for them. The interaction was calculated inside each of 1,000 paired Source Group bootstrap replicates, with the same sampled groups reused across all four arms. These intervals are conditional on the four fixed development streams; they do not make the categories independent samples.",
        "",
        "## First-512 EXP2A parity",
        "",
        "| Category | Events checked | G vs C0 max absolute error | GC vs C1-Full max absolute error | Numeric mismatches |",
        "|---|---:|---:|---:|---:|",
    ]
    for category in CATEGORIES:
        p = categories[category]["first512_exp2a_parity"]
        lines.append(f"| {category} | {p['checked_events']} | {p['g_max_abs_error']:.9g} | {p['gc_max_abs_error']:.9g} | {p['numeric_mismatches']} |")
    lines += [
        "",
        "## Error repair, harm and ranking",
        "",
        "For both GC-vs-G and GC-vs-SC, the aggregate includes FN→TP and FP→TN repairs, TP→FN and TN→FP harms, the base/candidate confusion matrices, and verified transition reconstruction. It separately reports combination-only repairs and harms by normal/anomalous label. Ranking diagnostics count normal/defect pair credit changes for all pairs and Source Group-disjoint pairs; ties receive 0.5 credit, and denominators are explicit.",
        "",
        "### Transition counts",
        "",
        "| Category | Comparison | FN→TP | FP→TN | TP→FN | TN→FP |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for category in CATEGORIES:
        for comparison in ("GC_vs_G", "GC_vs_SC"):
            ledger = categories[category]["error_transitions"][comparison]
            lines.append(
                f"| {category} | {comparison} | {ledger['FN_to_TP_repaired']} | "
                f"{ledger['FP_to_TN_repaired']} | {ledger['TP_to_FN_introduced']} | "
                f"{ledger['TN_to_FP_introduced']} |"
            )
    lines += [
        "",
        "| Category | Outcome group | GC-only combination repair | GC-only combination harm |",
        "|---|---|---:|---:|",
    ]
    for category in CATEGORIES:
        for label in ("normal", "anomalous"):
            ledger = categories[category]["combination_only"][label]
            lines.append(f"| {category} | {label} | {ledger['repair']} | {ledger['harm']} |")
    lines += [
        "",
        "### Fixed-stream quartiles",
        "",
        "| Category | Quartile | Events | Anomaly proportion | Queries | Queried normal/anomaly | GC−G mean score | GC−SC mean score | G/GC AUROC |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for category in CATEGORIES:
        for q in categories[category]["fixed_stream_quartiles"]:
            lines.append(
                f"| {category} | {q['quartile']} | {q['events']} | "
                f"{_format_number(q['anomalous_proportion'])} | {q['query_events']} | "
                f"{q['query_normal_feedback']}/{q['query_anomalous_feedback']} | "
                f"{_format_number(q['score_effects']['GC_minus_G_mean'])} | "
                f"{_format_number(q['score_effects']['GC_minus_SC_mean'])} | "
                f"{_format_number(q['ranking_metrics']['G']['auroc'])}/"
                f"{_format_number(q['ranking_metrics']['GC']['auroc'])} |"
            )
    lines += [
        "",
        "## Interpretation",
        "",
        f"A four-category GC AUROC mean is {'defined at ' + _format_number(aggregate['four_category_gc_auroc_mean']) if aggregate['all_four_category_gc_aurocs_defined'] else 'undefined because at least one category has a single-class full stream'}. Category-specific conditional gains, intervals, false-positive/false-negative trade-offs, repair/harm ledgers and score-by-quartile effects govern the scientific reading. Complementarity and positive factorial interaction are evaluated separately; no positive interaction is required, and no synergy claim is made unless its paired interval excludes zero.",
        "",
        "Any AUROC gain accompanied by higher FPR is a ranking improvement with an operating-point cost. Results describe only these four previously exposed development streams and do not establish independent confirmation, population risk, or transfer of a statistical guarantee.",
        "",
        "## Provenance, tests and limitations",
        "",
        f"- Focused tests: {context['test_summary']} (Windows Job Object bounded to {TEST_DEADLINE_SECONDS} seconds); Python compilation passed.",
        f"- Source parent SHA: `{EXPECTED_BASELINE}`; runner SHA-256: `{context['runner_sha256']}`; test SHA-256: `{context['test_sha256']}`.",
        f"- Stage A score seal SHA-256: `{context['stage_a_seal_sha256']}`; Stage B seal SHA-256: `{context['stage_b_seal_sha256']}`; public aggregate SHA-256: `{context['aggregate_sha256']}`.",
        "- Stage A used the restricted queried-position label reader only after the four scores were fsynced. Stage B used the restricted label scanner after complete score sealing; it skipped mask fields and did not open mask images.",
        "- The 4 GiB OutputBudget is an application-level pre-write bound with a reserved failure record and unique output directory. Windows does not provide OS hard storage isolation here; unrelated same-user processes could bypass the accountant.",
        "- DINO/Support numerical fidelity was checked by the structured native fixed-bank digest and first-512 reference parity. Later-stream errors can reflect the frozen prequential state trajectory and are not independent trajectories.",
        "- No second algorithm search was performed. No GPU fallback model, label-driven query selection, per-category filtering, future feedback access or mask access was used.",
        "",
        "## Scientific conclusion",
        "",
        context["scientific_conclusion"],
        "",
        "See `aggregate_results.json` for the complete per-category metrics, confidence intervals, transitions, paired ranking diagnostics, quartiles and update telemetry.",
        "",
    ]
    return "\n".join(lines)


def run_experiment(run_id: str, *,
                   reuse_complete_category_from: str | None = None) -> dict[str, Any]:
    # Capture the launcher's authorized visibility before OutputBudget imports
    # r2_feedback_repair, whose CPU-only module initialization hides CUDA.
    authorized_cuda_visibility = os.environ.get("CUDA_VISIBLE_DEVICES")
    from experiments.calibration.r1_runtime_supervisor import run_supervised_child

    overall_started = time.monotonic()
    repo = _check_repository_identity()
    if REPORT_DIR.exists():
        raise FileExistsError(f"Refusing to overwrite existing audit deliverable directory: {REPORT_DIR}")
    _output_path(run_id)
    _output_path(run_id + "_smoke")
    if not Path(sys.executable).samefile(Path(r"D:\anaconda3\envs\torch\python.exe")):
        raise R3Error(f"Unexpected Python interpreter: {sys.executable}")
    free_bytes = shutil.disk_usage(OUTPUT_BASE).free
    if free_bytes < MIN_FREE_BYTES:
        raise R3Error(f"Insufficient free space: need {MIN_FREE_BYTES}, have {free_bytes}")
    contract = load_frozen_contract(verify_image_paths=True)
    exp2a_identity, _ = _load_exp2a_score_references()
    model_receipt = _get_verified_model_receipt()
    sources = _source_fingerprints()
    runner_sha = sha256_file(Path(__file__).resolve())
    tests_path = ROOT / "tests" / "calibration" / "test_r3_fullstream_2x2.py"
    if not tests_path.is_file():
        raise R3Error("Authorized focused CPU test file is missing")
    test_sha = sha256_file(tests_path)
    compiled = subprocess.run(
        [sys.executable, "-m", "py_compile", str(Path(__file__).resolve()), str(tests_path)],
        cwd=ROOT, capture_output=True, text=True, timeout=60)
    if compiled.returncode:
        raise R3Error(f"Python compilation failed: {compiled.stderr[-2000:]}")
    test_count = _collected_test_case_count(tests_path)
    test_result = _run_cpu_tests()

    smoke_id = run_id + "_smoke"
    smoke_root = _initialize_run(smoke_id, "SMOKE_ONLY", contract,
                                 exp2a_identity, sources, model_receipt)
    smoke_started = time.monotonic()
    try:
        smoke_result = _run_supervised_child(
            smoke_root, stage="STAGE_A", category="switch", smoke_events=128,
            timeout_seconds=SMOKE_DEADLINE_SECONDS,
            authorized_cuda_visibility=authorized_cuda_visibility)
        _append_supervisor_result(smoke_root, "STAGE_A_SMOKE", "switch", smoke_result)
        if smoke_result.status != "COMPLETED" or smoke_result.child_exit_code != 0:
            raise R3Error(f"GPU smoke process failed: {asdict(smoke_result)}")
        smoke_history = _read_history(smoke_root)
        smoke_receipt = next((row for row in smoke_history
                              if row.get("record_type") == "category_complete"), None)
        smoke_queries = [row for row in smoke_history
                         if row.get("record_type") == "queried_feedback_and_updates"]
        if (not smoke_receipt or smoke_receipt.get("events_processed") != 128 or
                not smoke_queries or any(set(row.get("correction_updates", {})) != {"SC", "GC"}
                                         for row in smoke_queries)):
            raise R3Error("GPU smoke did not exercise both native correction updates")
        _record_supervisor_status(smoke_root, {
            "schema": "r3.fullstream.2x2.supervisor_status.v1",
            "status": "GPU_SMOKE_PASS_NOT_EVALUATED",
            "run_id": smoke_id, "events": 128, "query_events": len(smoke_queries),
            "offline_evaluation_performed": False,
            "wall_seconds": time.monotonic() - smoke_started,
            "windows_job_object": True, "automatic_restart": False,
        })
    except BaseException as exc:
        _write_failure(smoke_root, "GPU_SMOKE", exc)
        raise

    # A fresh namespace is mandatory: the smoke output can never be resumed as the full run.
    root = _initialize_run(run_id, "FULL_STAGE_A", contract,
                           exp2a_identity, sources, model_receipt)
    full_started = time.monotonic()
    supervisor_rows = []
    stage_a_wall = 0.0
    try:
        reused_categories: set[str] = set()
        reused_seconds = 0.0
        if reuse_complete_category_from:
            reused_categories, reused_seconds = _reuse_verified_categories(
                root, reuse_complete_category_from, contract)
        for category in CATEGORIES:
            if category in reused_categories:
                continue
            elapsed = reused_seconds + time.monotonic() - overall_started
            remaining = TOTAL_DEADLINE_SECONDS - elapsed
            if remaining <= 0:
                raise R3Error("Eight-hour total execution ceiling reached before category launch")
            deadline = min(CATEGORY_DEADLINE_SECONDS, remaining)
            child = _run_supervised_child(root, stage="STAGE_A", category=category,
                                          timeout_seconds=deadline,
                                          authorized_cuda_visibility=authorized_cuda_visibility)
            _append_supervisor_result(root, "STAGE_A", category, child)
            supervisor_rows.append({"stage": "STAGE_A", "category": category,
                                    **asdict(child)})
            if child.status != "COMPLETED" or child.child_exit_code != 0:
                raise R3Error(f"Stage A child failed for {category}: {asdict(child)}")
            receipts = _category_receipts(root)
            if category not in receipts or receipts[category].get("events_processed") != EXPECTED_COUNTS[category]:
                raise R3Error(f"{category}: supervised child exited without a full category seal")
        stage_a_wall = reused_seconds + time.monotonic() - full_started
        stage_a_seal = _seal_stage_a(root, contract)
        remaining = TOTAL_DEADLINE_SECONDS - (reused_seconds + time.monotonic() - overall_started)
        if remaining <= 0:
            raise R3Error("Eight-hour total execution ceiling reached before Stage B")
        stage_b_started = time.monotonic()
        stage_b_child = _run_supervised_child(
            root, stage="STAGE_B", timeout_seconds=min(CATEGORY_DEADLINE_SECONDS, remaining))
        _append_supervisor_result(root, "STAGE_B", None, stage_b_child)
        supervisor_rows.append({"stage": "STAGE_B", **asdict(stage_b_child)})
        if stage_b_child.status != "COMPLETED" or stage_b_child.child_exit_code != 0:
            raise R3Error(f"CPU Stage B child failed: {asdict(stage_b_child)}")
        stage_b_wall = time.monotonic() - stage_b_started
        output_budget = _budget(root)
        output_audit = output_budget.audit()
        total_wall = _overall_wall_seconds(overall_started, reused_seconds)
        status_doc = {
            "schema": "r3.fullstream.2x2.supervisor_status.v1",
            "status": "COMPLETE", "run_id": run_id,
            "children": supervisor_rows,
            "total_supervised_deadline_seconds": TOTAL_DEADLINE_SECONDS,
            "overall_elapsed_seconds_including_tests_smoke_and_reused_categories": total_wall,
            "category_child_deadline_seconds": CATEGORY_DEADLINE_SECONDS,
            "termination_guarantee": "Windows Job Object process-tree containment",
            "automatic_restart": False,
            "output_budget": output_audit,
            "minimum_free_space_before_launch": MIN_FREE_BYTES,
            "free_space_before_launch": free_bytes,
        }
        output_budget.write_json("supervisor_status.json", status_doc)
        aggregate = _read_json(root / "aggregate_results.json")
        stage_b_seal = _read_json(root / "stage_b_seal.json")
        complete_seal = _read_json(root / "complete_seal.json")
        if (complete_seal.get("status") != "COMPLETE" or
                stage_b_seal.get("status") != "SEALED_COMPLETE"):
            raise R3Error("Stage B did not produce complete integrity seals")
        full_wall = time.monotonic() - full_started
        total_wall = _overall_wall_seconds(overall_started, reused_seconds)
        if total_wall > TOTAL_DEADLINE_SECONDS:
            raise R3Error("Total supervised experiment exceeded eight-hour deadline")
        budget_check = _budget(root)
        output_size = budget_check.audit()
        hist = _read_history(root)
        category_receipts = _category_receipts(root)
        support_encodings = sum(int(category_receipts[c].get("support_encoder_invocations", 0))
                                for c in CATEGORIES)
        # The exact category summary includes this count in its runtime receipt when present.
        if support_encodings != 64:
            raise R3Error(f"Expected 64 logged Support augmentation encodings, found {support_encodings}")
        runtime_rows = [category_receipts[c] for c in CATEGORIES]
        max_allocated = max(int(row.get("peak_cuda_allocated_bytes", 0)) for row in runtime_rows)
        max_reserved = max(int(row.get("peak_cuda_reserved_bytes", 0)) for row in runtime_rows)
        hub_sha = runtime_rows[0]["hub_source_tree_sha256"]
        checkpoint_sha = runtime_rows[0]["checkpoint_sha256"]
        model_state_sha = runtime_rows[0]["loaded_model_state_sha256"]
        gpu_name = runtime_rows[0]["runtime_profile"]["cuda_device_name"]
        cuda_runtime = runtime_rows[0]["runtime_profile"]["cuda_runtime_version"]
        fixed_sha = {c: category_receipts[c]["fixed_support_sha256"] for c in CATEGORIES}
        q_sha = {c: contract["classes"][c]["query_tape_sha256"] for c in CATEGORIES}
        support_image_sha = {c: contract["classes"][c]["support_image_sha256"] for c in CATEGORIES}
        support_group_sha = {c: contract["classes"][c]["support_group_sha256"] for c in CATEGORIES}
        c64_image_sha = {c: contract["classes"][c]["c64_image_sha256"] for c in CATEGORIES}
        c64_group_sha = {c: contract["classes"][c]["c64_group_sha256"] for c in CATEGORIES}
        c64_score_sha = {c: contract["classes"][c]["c64_score_tape_sha256"] for c in CATEGORIES}
        interpretation = _scientific_conclusion(aggregate)
        context = {
            "run_id": run_id, "branch": repo["branch"],
            "gpu_name": gpu_name, "cuda_runtime": cuda_runtime,
            "runtime_profile": runtime_rows[0]["runtime_profile"],
            "support_encodings": support_encodings,
            "stage_a_wall_seconds": stage_a_wall,
            "stage_b_wall_seconds": stage_b_wall,
            "total_execution_wall_seconds": total_wall,
            "peak_cuda_allocated_bytes": max_allocated,
            "peak_cuda_reserved_bytes": max_reserved,
            "output_bytes": output_size["bytes"],
            "recovery_summary": (
                f"Final run {run_id} passed a fresh 128-event GPU smoke and reused the four "
                f"complete, integrity-verified category trajectories from {reuse_complete_category_from}; "
                "no full-stream category was re-encoded. The first two smoke attempts stopped before "
                "DINO/event scoring because a CPU-only dependency hid CUDA. The first full Stage A "
                "sealed all categories but exposed a trailing-whitespace bug in Stage B; a later "
                "identity check and a parent wall-time bookkeeping error were repaired without "
                "changing scoring. All failed namespaces were preserved."
                if reused_categories else
                "One GPU smoke, then one full run; no failed category restart or category reuse."),
            "hub_source_tree_sha256": hub_sha,
            "checkpoint_sha256": checkpoint_sha,
            "loaded_model_state_sha256": model_state_sha,
            "fixed_support_sha256": fixed_sha,
            "support_image_sha256": support_image_sha,
            "support_group_sha256": support_group_sha,
            "c64_image_sha256": c64_image_sha,
            "c64_group_sha256": c64_group_sha,
            "c64_score_tape_sha256": c64_score_sha,
            "query_tape_sha256": q_sha,
            "correction_config": dict(CORRECTION_CONFIG),
            "exp2a_score_rows_sha256": exp2a_identity["score_rows_sha256"],
            "stage05_hub_source_tree_sha256": contract["stage05_model_provenance"].get("hub_source_tree_sha256"),
            "stage05_source_hash_mismatches": {
                rel: {"frozen_sha256": expected,
                      "current_sha256": contract["stage05_current_source_hashes"].get(rel)}
                for rel, expected in contract["stage05_scoring_source_hashes"].items()
                if contract["stage05_current_source_hashes"].get(rel) != expected
            },
            "test_summary": context_test_result(test_result, test_count),
            "runner_sha256": runner_sha,
            "test_sha256": test_sha,
            "stage_a_runner_sha256": runner_sha,
            "stage_a_reuse_source_run_id": None,
            "stage_a_reuse_source_runner_sha256": None,
            "stage_a_reuse_compatibility": "no-category-reuse",
            "stage_a_seal_sha256": sha256_file(root / "stage_a_seal.json"),
            "stage_b_seal_sha256": sha256_file(root / "stage_b_seal.json"),
            "aggregate_sha256": sha256_file(root / "aggregate_results.json"),
            "scientific_conclusion": interpretation,
        }
        reuse_receipt_path = root / "stage_a_reuse_receipt.json"
        if reuse_receipt_path.is_file():
            reuse_receipt = _read_json(reuse_receipt_path)
            context["stage_a_reuse_source_run_id"] = reuse_receipt.get("source_run_id")
            context["stage_a_reuse_source_runner_sha256"] = reuse_receipt.get(
                "source_stage_a_runner_sha256")
            context["stage_a_reuse_compatibility"] = reuse_receipt.get(
                "source_runner_compatibility")
            context["stage_a_runner_sha256"] = reuse_receipt.get(
                "source_stage_a_runner_sha256", runner_sha)
        report, aggregate_copy = _write_public_deliverables(root, aggregate, context)
        commit_sha = _git("rev-parse", "HEAD")
        print(json.dumps({"status": "COMPLETE", "run_id": run_id,
                          "stage_a_events": sum(EXPECTED_COUNTS.values()),
                          "query_total": aggregate["query_feedback_total"],
                          "output_bytes": output_size["bytes"],
                          "stage_a_seal_sha256": context["stage_a_seal_sha256"],
                          "stage_b_seal_sha256": context["stage_b_seal_sha256"],
                          "report_path": str(report),
                          "aggregate_path": str(aggregate_copy)},
                         sort_keys=True), flush=True)
        return {"aggregate": aggregate, "root": root, "context": context,
                "stage_a_seal": stage_a_seal, "test_result": test_result,
                "supervisors": supervisor_rows, "full_wall_seconds": full_wall}
    except BaseException as exc:
        _write_failure(root, "FULL_EXPERIMENT_PARENT", exc)
        raise


def _collected_test_case_count(path: Path) -> int:
    """Count simple pytest test functions and class methods in the focused file."""
    tree = ast.parse(Path(path).read_text(encoding="utf-8"), filename=str(path))
    count = 0
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name.startswith("test_"):
            count += 1
        elif isinstance(node, ast.ClassDef):
            count += sum(
                isinstance(member, (ast.FunctionDef, ast.AsyncFunctionDef)) and
                member.name.startswith("test_")
                for member in node.body)
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and \
                node.func.attr == "parametrize":
            raise R3Error("Focused test summary cannot count parametrized cases from syntax alone")
    return count


def context_test_result(result: Mapping[str, Any], test_count: int) -> str:
    if result.get("status") == "COMPLETED" and result.get("child_exit_code") == 0:
        return f"{int(test_count)} passed, 0 failed, 0 skipped"
    return f"failed: {result.get('status')} (child exit {result.get('child_exit_code')})"


def _scientific_conclusion(aggregate: Mapping[str, Any]) -> str:
    categories = aggregate["categories"]
    pieces = []
    complementarity = []
    positive_interaction = []
    negative_interaction = []
    cases = []
    operating_cost = []
    higher_fpr = []
    for category in CATEGORIES:
        effects = categories[category]["factorial_contrasts"]["auroc"]
        ci = categories[category]["paired_source_group_bootstrap"]["contrasts"]["auroc"]
        both_positive = (effects["GC_minus_G"] is not None and effects["GC_minus_SC"] is not None and
                         effects["GC_minus_G"] > 0 and effects["GC_minus_SC"] > 0)
        ci_gc_g = ci["GC_minus_G"]["ci_95_percentile"]
        ci_gc_sc = ci["GC_minus_SC"]["ci_95_percentile"]
        interaction_ci = ci["interaction"]["ci_95_percentile"]
        interaction_positive = (effects["interaction"] is not None and effects["interaction"] > 0 and
                                interaction_ci is not None and interaction_ci[0] > 0)
        interaction_negative = (effects["interaction"] is not None and effects["interaction"] < 0 and
                                interaction_ci is not None and interaction_ci[1] < 0)
        complementarity.append(both_positive)
        positive_interaction.append(interaction_positive)
        negative_interaction.append(interaction_negative)
        dg, ds = effects["GC_minus_G"], effects["GC_minus_SC"]
        if both_positive:
            case = "B" if interaction_positive else "A"
        elif dg is not None and ds is not None and dg > 0 and ds <= 0:
            case = "C"
        elif dg is not None and ds is not None and ds > 0 and dg <= 0:
            case = "D"
        else:
            case = "mixed/null/negative"
        gc_fpr = categories[category]["arms"]["GC"]["fpr"]
        g_fpr = categories[category]["arms"]["G"]["fpr"]
        if gc_fpr is not None and g_fpr is not None and gc_fpr > g_fpr:
            higher_fpr.append(category)
        fpr_tradeoff = (dg is not None and dg > 0 and gc_fpr is not None and
                        g_fpr is not None and gc_fpr > g_fpr)
        if fpr_tradeoff:
            case += "; E (GC AUROC gain with higher FPR than G)"
            operating_cost.append(category)
        cases.append(f"{category}: Case {case}")
        pieces.append(f"{category}: GC−G={_format_number(effects['GC_minus_G'])} "
                      f"(CI {_format_number(ci_gc_g[0] if ci_gc_g else None)} to "
                      f"{_format_number(ci_gc_g[1] if ci_gc_g else None)}), "
                      f"GC−SC={_format_number(effects['GC_minus_SC'])} "
                      f"(CI {_format_number(ci_gc_sc[0] if ci_gc_sc else None)} to "
                      f"{_format_number(ci_gc_sc[1] if ci_gc_sc else None)}), "
                      f"I={_format_number(effects['interaction'])}.")
    if all(complementarity) and any(positive_interaction):
        verdict = "Point estimates support complementary incremental value in every category; at least one category also has positive interaction with its paired interval above zero."
    elif any(complementarity):
        verdict = "Complementary point-estimate gains occur in some, but not all, categories; positive interaction is not established across the four fixed streams."
    elif all(negative_interaction):
        verdict = "GC did not add positive AUROC beyond both G and SC, and the factorial interaction was negative with a paired interval below zero in every category."
    else:
        verdict = "The category results do not show GC with positive incremental AUROC beyond both G and SC in every category; complementarity is mixed or unsupported."
    if operating_cost:
        tradeoff = " AUROC gains with higher FPR than G were observed for " + ", ".join(operating_cost) + "."
    elif higher_fpr:
        tradeoff = (" GC had higher FPR than G for " + ", ".join(higher_fpr) +
                    ", while GC AUROC did not exceed G in any category.")
    else:
        tradeoff = " No GC-vs-G increase in FPR was observed."
    return (verdict + " Category interpretation: " + "; ".join(cases) + tradeoff +
            " Conditional Source Group bootstrap intervals remain category-specific. " +
            " ".join(pieces))


def _parent_main(run_id: str, reuse_from: str | None = None) -> int:
    run_experiment(run_id, reuse_complete_category_from=reuse_from)
    return 0


def _require_windows_job_child() -> None:
    from experiments.calibration.r1_runtime_supervisor import current_process_is_in_job
    if not current_process_is_in_job():
        raise R3Error("R3 real-data child must be launched inside the Windows Job Object supervisor")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id")
    parser.add_argument("--authorize-full-stream", action="store_true")
    parser.add_argument("--stage-a-child", action="store_true")
    parser.add_argument("--stage-b-child", action="store_true")
    parser.add_argument("--category")
    parser.add_argument("--smoke-events", type=int, default=0)
    parser.add_argument("--reuse-complete-category-from")
    args = parser.parse_args(argv)
    try:
        if not args.authorize_full_stream:
            raise R3Error("Real-data execution requires --authorize-full-stream")
        if args.stage_a_child:
            if not args.run_id or not args.category:
                raise R3Error("Stage A child needs run ID and category")
            _require_windows_job_child()
            _check_repository_identity()
            result = _run_category_child(args.run_id, args.category,
                                         smoke_events=args.smoke_events)
            print(json.dumps(result, sort_keys=True), flush=True)
            return 0
        if args.stage_b_child:
            if not args.run_id:
                raise R3Error("Stage B child needs a run ID")
            _require_windows_job_child()
            _check_repository_identity()
            result = _run_stage_b_child(args.run_id)
            print(json.dumps({"status": "STAGE_B_COMPLETE",
                              "categories": list(result["categories"])}, sort_keys=True),
                  flush=True)
            return 0
        if not args.run_id:
            raise R3Error("A fresh unique --run-id is required")
        return _parent_main(args.run_id, args.reuse_complete_category_from)
    except BaseException as exc:
        print(f"R3_FULLSTREAM_2X2_ERROR: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
