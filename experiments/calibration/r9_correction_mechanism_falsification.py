"""R9 unified correction-mechanism falsification and resource-matched controls.

Stage A is supervised and label gated.  Its only outcome access is the exact
R7-identity-matched query set, disclosed after the current nine predictions
have been written and flushed.  Stage B is CPU-only and requires all eight
category/order seals plus the R7 shuffle reproduction gate.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import secrets
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[2]
AUTHORIZED_CUDA_VISIBILITY = os.environ.get("CUDA_VISIBLE_DEVICES")
EXPECTED_BRANCH = "calibration-stage0-contract-20261029"
EXPECTED_PARENT = "812e85407adb200b47e48ee811824e9394ed7416"
RUN_ID = "r9_correction_mechanism_20261103_a9"
PRESERVED_FAILED_ATTEMPTS: tuple[dict[str, Any], ...] = ({
    "run_id": "r9_correction_mechanism_20261103_a1",
    "status": "FAILED_NONZERO_EXIT",
    "failure_receipt_sha256": "cc61e74ef2bc077bc2ae03f3b5394dbd91e3adef1093ae7613c6ddb12fef76fe",
    "cause": "R1CausalityError: queried event completed correction and memory writes but omitted the final memory_updated phase",
    "automatic_restart": False,
}, {
    "run_id": "r9_correction_mechanism_20261103_a2",
    "status": "FAILED_PARENT_PARITY_CHECK",
    "failure_receipt_sha256": "9061d55f21d3b50cf668d15856c575199fb7292dff43a3fe5b1e450d6e0b0301",
    "cause": "NameError in parent score-parity checker after completed supervised smoke child",
    "automatic_restart": False,
}, {
    "run_id": "r9_correction_mechanism_20261103_a3",
    "status": "FAILED_PARENT_SMOKE_SUMMARY",
    "failure_receipt_sha256": "9d90021a4fc95223660e6c7bab92d577b5d6b014335a25fe3749b9e99bb9ccc3",
    "cause": "KeyError in parent smoke summary after supervised child, R7/R8 parity, feedback, and FIFO checks completed",
    "automatic_restart": False,
}, {
    "run_id": "r9_correction_mechanism_20261103_a4",
    "status": "FAILED_PARENT_OUTPUT_ALLOWLIST",
    "failure_receipt_sha256": "c9d1e3aee3ee6a62e3a00fd29da0df98fa36075370f6e7a742d7735289551111",
    "cause": "OutputBudget rejected the unregistered smoke_seal.json filename after the supervised smoke and parity gates passed",
    "automatic_restart": False,
}, {
    "run_id": "r9_correction_mechanism_20261103_a5",
    "status": "FAILED_STAGE_B_CHILD",
    "failure_receipt_sha256": "f3364cae3ca3bd12e7cb4a246bbe55891af0655d62a80cd2edc677283fe969ff",
    "cause": "Stage B raised KeyError for hub_source_tree_sha256 because Stage A stores model fingerprints nested under runtime_profile; additionally, the verified Support Seed 1 sensitivity condition was omitted from configuration.orders",
    "automatic_restart": False,
}, {
    "run_id": "r9_correction_mechanism_20261103_a6",
    "status": "FAILED_PRELAUNCH_IMPORT_PATH",
    "failure_receipt_sha256": None,
    "cause": "Invoked the module file directly instead of with python -m; package import failed before namespace creation and before any experiment or output access",
    "automatic_restart": False,
}, {
    "run_id": "r9_correction_mechanism_20261103_a7",
    "status": "FAILED_SUPPORT1_C64_CACHE_IDENTITY",
    "failure_receipt_sha256": "1064f01aa32d1503122647fe71673252768e886fdfcb135d9b4587afc4a38059",
    "cause": "Frozen R7 helper rejected the Support-0-bound R5 audiojack feature cache for the verified Support-1 fixed bank; failure occurred before any Support-1 event was scored",
    "automatic_restart": False,
}, {
    "run_id": "r9_correction_mechanism_20261103_a8",
    "status": "FAILED_SUPPORT1_CACHE_ROUTING",
    "failure_receipt_sha256": "5c8ad7d81590edd164693e530cd5326304369a084fd852aee0463605338ede6f",
    "cause": "R9 Support-1 C64 routing was added to the shadowed R8 helper rather than the active R9 child; active child therefore still called the Support-0-bound R7 cache",
    "automatic_restart": False,
})
RUN_NAMESPACE = ROOT / "outputs" / "calibration" / RUN_ID
OUTPUT_BASE = ROOT / "outputs" / "calibration"
R7_RUN_ID = "r7_shuffled_correction_4class_20261103_a4"
R8_RUN_ID = "r8_correction_mechanism_20261103_a5"
R7_FULL = OUTPUT_BASE / R7_RUN_ID / "full"
R7_REPORT = ROOT / "docs" / "calibration" / "audit" / "r7_shuffled_correction_4class_20261103" / "REPORT.md"
R7_AGGREGATE = ROOT / "docs" / "calibration" / "audit" / "r7_shuffled_correction_4class_20261103" / "aggregate_results.json"
REPORT_DIR = ROOT / "docs" / "calibration" / "audit" / "r9_correction_mechanism"
REPORT_PATH = REPORT_DIR / "REPORT.md"
AGGREGATE_PATH = REPORT_DIR / "aggregate_results.json"
CATEGORIES = ("audiojack", "regulator", "switch", "rolled_strip_base")
SHUFFLE_SEEDS = (0, 11, 23)
ORDERS = ("SHUFFLE0", "SHUFFLE11", "SHUFFLE23")
CAPACITIES = (1024,)
SCORERS = ("G1024", "GC1024", "SignedExemplar1024")
CONTROLLERS = ("Frozen", "B1", "B3")
ARMS = tuple(f"{scorer}-{controller}" for scorer in SCORERS for controller in CONTROLLERS)
ADDRESS_SEEDS = (17, 53, 97)
EXPECTED_COUNTS = {"audiojack": 3815, "regulator": 3855, "switch": 3820,
                   "rolled_strip_base": 3805}
EXPECTED_QUERIES = {"audiojack": 114, "regulator": 116, "switch": 115,
                    "rolled_strip_base": 114}
TOTAL_EVENTS_PER_ORDER = sum(EXPECTED_COUNTS.values())
TOTAL_QUERIES_PER_ORDER = sum(EXPECTED_QUERIES.values())
SMOKE_CATEGORY = "audiojack"
SMOKE_ORDER = "SHUFFLE0"
SMOKE_EVENTS = 3200
OUTPUT_LIMIT = 4 * 1024**3
FAILURE_RESERVE = 64 * 1024
MIN_FREE_BYTES = 10 * 1024**3
FULL_DEADLINE_SECONDS = 120 * 60
SMOKE_DEADLINE_SECONDS = 15 * 60
BOOTSTRAP_REPLICATES = 1000
BOOTSTRAP_SEED = 20261103
RUN_SCHEMA = "r9.correction_mechanism.run.v1"
IDENTITY_SCHEMA = "r9.correction_mechanism.identity.v1"
SCORE_SCHEMA = "r9.correction_mechanism.score.v1"
AUTHORIZATION = "R9 unified retrospective development correction-mechanism falsification"
SCORE_SOURCE_PATHS = (
    "experiments/calibration/r7_shuffled_correction_4class.py",
    "experiments/calibration/r6_capacity1024_audiojack_runner.py",
    "experiments/calibration/r3_fullstream_2x2.py",
    "experiments/calibration/r2_exp2a_c1_strength.py",
    "experiments/calibration/r1_matched_correction_pilot.py",
    "experiments/calibration/r1_runtime_supervisor.py",
    "experiments/calibration/query_tape.py",
    "experiments/calibration/controllers.py",
    "experiments/calibration/protocol.py",
    "experiments/calibration/stage1a_p12_label_provider.py",
    "experiments/calibration/native.py",
    "experiments/calibration/profiles.py",
    "experiments/calibration/stage05.py",
    "experiments/calibration/stage05_runner.py",
    "experiments/risk_safe/native_bridge.py",
    "experiments/p0_protocol_audit.py",
    "experiments/v02_memory.py",
    "vmb_visionad_new.py",
    "test_phase3_AIF_simplified_v22.py",
    "experiments/calibration/r8_correction_mechanism_attribution.py",
)


class R8Error(RuntimeError):
    pass


R9Error = R8Error
MANDATORY_ORDERS = ("SHUFFLE0", "SHUFFLE11", "SHUFFLE23")


def _registered_orders(support1_receipt: Mapping[str, Any] | None) -> tuple[str, ...]:
    return ORDERS + (("SUPPORT1_SHUFFLE0",) if support1_receipt else ())


def _registered_condition_configuration(
    support1_receipt: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Keep the frozen identity's order list aligned with executable conditions."""
    return {
        "orders": list(_registered_orders(support1_receipt)),
        "support_seeds": [0] + ([1] if support1_receipt else []),
    }


def _resource_and_state_receipt(completion: Mapping[str, Any]) -> dict[str, Any]:
    """Flatten sealed runtime fingerprints into the Stage B resource receipt."""
    ordinary_keys = (
        "first_normal_feedback_position", "first_defect_feedback_position",
        "c1_final_occupancy", "c1_first_full_position", "c1_update_stats",
        "simple_memory_final_occupancy", "simple_memory_insertions",
        "simple_memory_evictions", "simple_memory_first_full_position",
        "simple_memory_unavailable_sign_patch_events", "simple_memory_source_exclusions",
        "c64_rescore_costs", "mutable_memory_bytes", "timing_seconds",
        "peak_cuda_allocated_bytes", "peak_cuda_reserved_bytes", "peak_cpu_rss_bytes",
        "greedy_writer_audit", "c64_cache_receipt", "native_scorer_profile_sha256",
        "stream_image_encodings", "fixed_support_receipt",
    )
    missing = [key for key in ordinary_keys if key not in completion]
    if missing:
        raise R8Error(f"R9 Stage A resource receipt is missing fields: {missing}")
    receipt = {key: completion[key] for key in ordinary_keys}
    runtime = completion.get("runtime_profile")
    if not isinstance(runtime, Mapping):
        raise R8Error("R9 Stage A runtime profile is missing or malformed")
    model_keys = ("hub_source_tree_sha256", "checkpoint_sha256", "loaded_model_state_sha256")
    absent_model_keys = [key for key in model_keys if not runtime.get(key)]
    if absent_model_keys:
        raise R8Error(f"R9 Stage A runtime profile lacks model fingerprints: {absent_model_keys}")
    receipt["runtime_profile"] = runtime.get("runtime_profile")
    receipt.update({key: runtime[key] for key in model_keys})
    return receipt


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8") + b"\n"


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def _git(*args: str) -> str:
    result = subprocess.run(["git", *args], cwd=ROOT, check=True,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            text=True, encoding="utf-8")
    return result.stdout.strip()


def _source_fingerprints() -> dict[str, str]:
    result = {}
    for relative in SCORE_SOURCE_PATHS:
        path = ROOT / relative
        if not path.is_file():
            raise R8Error(f"required pinned source is missing: {relative}")
        result[relative] = _sha256_file(path)
    result["experiments/calibration/r9_correction_mechanism_falsification.py"] = _sha256_file(Path(__file__).resolve())
    test_path = ROOT / "tests" / "calibration" / "test_r9_correction_mechanism_falsification.py"
    if test_path.is_file():
        result["tests/calibration/test_r9_correction_mechanism_falsification.py"] = _sha256_file(test_path)
    return result


def _assert_repository_identity() -> None:
    branch, head = _git("branch", "--show-current"), _git("rev-parse", "HEAD")
    if branch != EXPECTED_BRANCH or head != EXPECTED_PARENT:
        raise R8Error(f"baseline mismatch: branch={branch!r}, HEAD={head}")
    dirty = _git("status", "--porcelain", "--", *SCORE_SOURCE_PATHS[1:])
    if dirty:
        raise R8Error("a pinned R8 dependency has tracked or staged local changes")


def make_order_events(shuffled_events: Sequence[Any], order: str) -> list[Any]:
    """Construct the registered deterministic group shuffle from public identities."""
    import re
    from experiments.calibration import r7_shuffled_correction_4class as r7
    match = re.fullmatch(r"(?:SUPPORT1_)?SHUFFLE(0|11|23)", order)
    if not match:
        raise R8Error(f"unsupported R9 stream order: {order}")
    seed = int(match.group(1))
    if seed == 0:
        result = list(shuffled_events)
        if [int(e.position) for e in result] != list(range(len(result))):
            raise R8Error("R7 Seed-0 shuffle positions are not contiguous")
        return result
    raw = sorted(shuffled_events, key=lambda event: int(event.original_position))
    if [int(event.original_position) for event in raw] != list(range(len(raw))):
        raise R8Error("R9 source stream is not a bijection over original positions")
    rows = [{"position": int(event.original_position),
             "opaque_group": str(event.source_group_sha256),
             "relative_id": str(event.relative_id),
             "camera_id": str(event.camera_id), "timestamp": str(event.timestamp),
             "image_path": str(event.image_path)} for event in raw]
    result = r7.deterministic_group_shuffle(rows, seed=seed)
    if len(result) != len(raw) or sorted(int(e.original_position) for e in result) != list(range(len(raw))):
        raise R8Error("R9 deterministic group shuffle is not a stream bijection")
    return result


def matched_query_positions(shuffled_events: Sequence[Any],
                            r7_query_positions: Sequence[int],
                            order_events: Sequence[Any]) -> tuple[int, ...]:
    """Map the frozen R7 query image identities to either arrival order."""
    qpos = tuple(int(value) for value in r7_query_positions)
    if len(qpos) != len(set(qpos)) or any(not 0 <= p < len(shuffled_events) for p in qpos):
        raise R8Error("R7 query schedule is malformed")
    selected = {str(shuffled_events[p].image_identity_sha256) for p in qpos}
    if len(selected) != len(qpos):
        raise R8Error("R7 query image identity set is not unique")
    positions = tuple(i for i, event in enumerate(order_events)
                      if str(event.image_identity_sha256) in selected)
    if len(positions) != len(qpos):
        raise R8Error("identity-matched order failed to preserve the R7 query set")
    selected_groups = [str(order_events[p].source_group_sha256) for p in positions]
    if len(set(selected_groups)) != len(selected_groups):
        raise R8Error("identity-matched R7 query set repeats a Source Group")
    if {str(order_events[p].image_identity_sha256) for p in positions} != selected:
        raise R8Error("identity-matched query set changed across stream orders")
    return positions


def map_outcomes_to_order(original_labels: Sequence[int], events: Sequence[Any]) -> list[int]:
    """Stage-B-only projection through sealed original JSON positions."""
    values = [int(value) for value in original_labels]
    positions = [int(event.original_position) for event in events]
    if (len(positions) != len(values) or sorted(positions) != list(range(len(values))) or
            any(value not in (0, 1) for value in values)):
        raise R8Error("outcome projection is not a complete identity bijection")
    return [values[position] for position in positions]


def assert_capacity_configs_differ_only_by_k(cfg256: Mapping[str, Any],
                                              cfg1024: Mapping[str, Any]) -> None:
    left, right = dict(cfg256), dict(cfg1024)
    left_k, right_k = left.pop("K_correction", None), right.pop("K_correction", None)
    left.pop("enabled", None)
    right.pop("enabled", None)
    if left_k != 256 or right_k != 1024 or left != right:
        raise R8Error("C256/C1024 configs differ beyond registered correction capacity")


def _r7_reference_and_contract() -> tuple[dict[str, Any], dict[str, Any], dict[str, list[dict[str, Any]]]]:
    """Verify sealed R7 artifacts against current pinned metadata, without labels."""
    from experiments.calibration import r7_shuffled_correction_4class as r7
    if not R7_FULL.is_dir():
        raise R8Error("BLOCKED_REFERENCE_INPUT: exact R7 full output directory is missing")
    required = ("run_manifest.json", "stage_a_scores.jsonl", "stage_a_seal.json",
                "stage_b_seal.json", "complete_seal.json", "aggregate_results.json")
    missing = [name for name in required if not (R7_FULL / name).is_file()]
    if missing:
        raise R8Error(f"BLOCKED_REFERENCE_INPUT: missing R7 artifact(s): {missing}")
    manifest = _read_json(R7_FULL / "run_manifest.json")
    stage_a = _read_json(R7_FULL / "stage_a_seal.json")
    stage_b = _read_json(R7_FULL / "stage_b_seal.json")
    complete = _read_json(R7_FULL / "complete_seal.json")
    score_path = R7_FULL / "stage_a_scores.jsonl"
    score_sha = _sha256_file(score_path)
    if (manifest.get("run_id") != R7_RUN_ID or
            stage_a.get("status") != "SEALED_COMPLETE" or
            stage_b.get("status") != "SEALED_COMPLETE" or
            complete.get("status") != "COMPLETE" or
            stage_a.get("score_rows_sha256") != score_sha or
            stage_b.get("stage_a_seal_sha256") != _sha256_file(R7_FULL / "stage_a_seal.json") or
            stage_b.get("aggregate_results_sha256") != _sha256_file(R7_FULL / "aggregate_results.json") or
            complete.get("stage_a_seal_sha256") != _sha256_file(R7_FULL / "stage_a_seal.json") or
            complete.get("stage_b_seal_sha256") != _sha256_file(R7_FULL / "stage_b_seal.json") or
            complete.get("aggregate_results_sha256") != _sha256_file(R7_FULL / "aggregate_results.json")):
        raise R8Error("BLOCKED_REFERENCE_INPUT: R7 full trajectory seals or score digest fail")
    contract = r7._frozen_contract()
    for category in CATEGORIES:
        info = contract["metadata"][category]
        support_groups = {
            hashlib.sha256(("official-sample-group\0" + str(group)).encode("utf-8")).hexdigest()
            for group in info["frozen_info"]["supports"]["0"]["groups"]}
        c64_groups = set(map(str, info["c64_groups"]))
        test_groups = {str(event.source_group_sha256) for event in info["events"]}
        if (len(support_groups) != 4 or len(c64_groups) != 64 or
                support_groups & c64_groups or support_groups & test_groups or
                c64_groups & test_groups):
            raise R8Error(f"{category}: Support, C64 and TEST Source Groups are not disjoint")
        if len(info["events"]) != EXPECTED_COUNTS[category]:
            raise R8Error(f"{category}: label-blind TEST metadata count differs from the frozen contract")
    identity = manifest.get("experiment_identity", {})
    if identity.get("identity_sha256") != manifest.get("identity_sha256"):
        raise R8Error("R7 identity manifest digest is inconsistent")
    reference_rows: dict[str, list[dict[str, Any]]] = {c: [] for c in CATEGORIES}
    with score_path.open("r", encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            category = row.get("category")
            if category not in reference_rows:
                raise R8Error("R7 reference contains an unauthorized category")
            reference_rows[category].append(row)
    for category in CATEGORIES:
        info = contract["metadata"][category]
        saved = identity.get("classes", {}).get(category, {})
        mapping = saved.get("shuffled_identity_map", [])
        query_receipt = saved.get("shuffled_query_tape", {})
        if (mapping != info["identity_receipt"]["rows"] or
                len(mapping) != EXPECTED_COUNTS[category] or
                int(saved.get("query_count", -1)) != EXPECTED_QUERIES[category] or
                tuple(query_receipt.get("positions", ())) != tuple(info["tape_receipt"]["positions"]) or
                query_receipt.get("tape_sha256") != info["tape_receipt"]["tape_sha256"]):
            raise R8Error(f"R7 identity/query receipt differs from current pinned metadata: {category}")
        rows = reference_rows[category]
        if len(rows) != EXPECTED_COUNTS[category] or any(
                int(row.get("position", -1)) != i or
                str(row.get("image_identity_sha256")) != str(mapping[i]["image_identity_sha256"]) or
                int(row.get("original_json_position", -1)) != int(mapping[i]["original_position"])
                for i, row in enumerate(rows)):
            raise R8Error(f"R7 score trajectory does not map to its sealed identity map: {category}")
        qset = set(int(x) for x in query_receipt["positions"])
        if any(bool(row.get("query_tape_decision")) != (i in qset)
               for i, row in enumerate(rows)):
            raise R8Error(f"R7 score query decisions differ from the pinned tape: {category}")
    return {"manifest": manifest, "stage_a": stage_a, "stage_b": stage_b,
            "complete": complete, "score_sha256": score_sha}, contract, reference_rows


def verify_support1_frozen_inputs(contract: Mapping[str, Any]) -> dict[str, Any]:
    """Verify only the already-frozen Support-1 identities and C64 score receipts."""
    import numpy as np
    from experiments.calibration import r3_fullstream_2x2 as r3
    stage = r3.STAGE05
    frozen_path = stage / "frozen_inputs.json"
    score_manifest_path = stage / "score_tape_manifest.json"
    checksums_path = stage / "checksums.json"
    frozen = _read_json(frozen_path)
    scores = _read_json(score_manifest_path)
    checksums = _read_json(checksums_path)
    if (frozen.get("schema") != "stage05.frozen_inputs.v1" or
            frozen.get("freeze_verification_status") != "PASS" or
            frozen.get("stage1_or_stage2_run") is not False or
            scores.get("status") != "PASS" or
            scores.get("controller_or_test_label_access") is not False):
        raise R8Error("Support-1 frozen data contract is not verified")
    checksum_files = checksums.get("files", {})
    for path in (frozen_path, score_manifest_path):
        rel = path.relative_to(stage).as_posix()
        if checksum_files.get(rel) != _sha256_file(path):
            raise R8Error(f"Support-1 Stage05 checksum receipt failed: {rel}")
    if set(frozen.get("classes", {})) != set(CATEGORIES):
        raise R8Error("Support-1 Stage05 category inventory differs from R9")
    receipts: dict[str, Any] = {}
    for category in CATEGORIES:
        fclass = frozen["classes"][category]
        info = contract["metadata"][category]
        support = fclass.get("supports", {}).get("1")
        c64 = fclass.get("c64", {})
        tape = scores.get("classes", {}).get(category, {}).get("score_tapes", {}).get("1")
        if not isinstance(support, Mapping) or not isinstance(tape, Mapping):
            raise R8Error(f"{category}: frozen Support-1 or C64 score receipt is missing")
        stream_groups = {str(event.source_group_sha256) for event in info["events"]}
        support_group_ids = [str(g) for g in support.get("groups", [])]
        c64_group_ids = [str(g) for g in c64.get("groups", [])]
        support_groups = {hashlib.sha256(("official-sample-group\0" + g).encode()).hexdigest()
                          for g in support_group_ids}
        c64_groups = {hashlib.sha256(("official-sample-group\0" + g).encode()).hexdigest()
                      for g in c64_group_ids}
        from experiments.calibration.stage05 import opaque_digest
        if (support.get("seed") != 1 or support.get("all_train_normal") is not True or
                support.get("c64_disjoint") is not True or len(support.get("images", [])) != 4 or
                len(support_group_ids) != 4 or len(support_groups) != 4 or
                len(c64.get("images", [])) != 64 or len(c64_groups) != 64 or
                support_groups & c64_groups or support_groups & stream_groups or
                c64_groups & stream_groups or
                support.get("group_sha256") != opaque_digest("support-groups", support_group_ids) or
                support.get("image_sha256") != opaque_digest("support-images", support["images"]) or
                c64.get("group_sha256") != opaque_digest("c64-groups", c64_group_ids) or
                c64.get("image_sha256") != opaque_digest("c64-images", c64["images"])):
            raise R8Error(f"{category}: Support-1/C64/TEST identity or disjointness failed")
        support0 = fclass["supports"]["0"]
        if (support.get("group_sha256") == support0.get("group_sha256") or
                support.get("image_sha256") == support0.get("image_sha256")):
            raise R8Error(f"{category}: Support-1 reuses Support-0 identity")
        if (tape.get("support_group_sha256") != support.get("group_sha256") or
                tape.get("c64_group_sha256") != c64.get("group_sha256") or
                int(tape.get("c64_score_count", -1)) != 64 or
                tape.get("native_parity", {}).get("passed") is not True or
                float(tape.get("native_parity", {}).get("direct_native_vs_p0_score_encoded_abs_delta", math.inf)) != 0.0):
            raise R8Error(f"{category}: Support-1 C64 commissioning receipt is incompatible")
        score_path = (stage / str(tape["file"])).resolve(strict=True)
        if stage.resolve(strict=True) not in score_path.parents:
            raise R8Error(f"{category}: Support-1 C64 score path escapes Stage05")
        expected_file_sha = str(tape.get("sha256", ""))
        if (not score_path.is_file() or _sha256_file(score_path) != expected_file_sha or
                checksum_files.get(score_path.relative_to(stage).as_posix()) != expected_file_sha):
            raise R8Error(f"{category}: Support-1 score archive checksum is invalid")
        with np.load(score_path, allow_pickle=False) as archive:
            c64_values = np.asarray(archive["c64_scores"], dtype=np.float64)
            c64_image_identity = str(archive["c64_identity_sha256"].item())
        if (c64_values.shape != (64,) or not np.isfinite(c64_values).all() or
                c64_image_identity != c64.get("image_sha256")):
            raise R8Error(f"{category}: Support-1 C64 scores do not bind to frozen C64 images")
        receipts[category] = {
            "support_seed": 1, "support_image_sha256": support["image_sha256"],
            "support_group_sha256": support["group_sha256"],
            "fixed_support_sha256": tape["fixed_support_sha256"],
            "c64_image_sha256": c64["image_sha256"],
            "c64_group_sha256": c64["group_sha256"],
            "c64_score_archive_sha256": expected_file_sha,
            "c64_score_count": 64,
            "theta_c64_q90": float(np.quantile(c64_values, .9, method="higher")),
            "sigma_c64_floor_002": max(float(c64_values.std(ddof=1)), .02),
            "stage05_frozen_inputs_sha256": _sha256_file(frozen_path),
            "stage05_score_tape_manifest_sha256": _sha256_file(score_manifest_path),
            "checksums_receipt_sha256": _sha256_file(checksums_path),
            "support_test_c64_group_disjoint": True,
        }
    return receipts


def support1_class_info(category: str, info: Mapping[str, Any]) -> dict[str, Any]:
    """Return the same frozen contract with only its deployed four supports replaced."""
    import numpy as np
    from experiments.calibration import r3_fullstream_2x2 as r3
    frozen = _read_json(r3.STAGE05 / "frozen_inputs.json")["classes"][category]
    manifest = _read_json(r3.STAGE05 / "score_tape_manifest.json")["classes"][category]
    support = frozen["supports"]["1"]
    score_row = manifest["score_tapes"]["1"]
    out = dict(info)
    out["support_seed"] = 1
    out["support_paths"] = [r3.DATASET_ROOT / "images" / frozen["prefix"] / rel
                            for rel in support["images"]]
    out["support_image_sha256"] = str(support["image_sha256"])
    out["support_group_sha256"] = str(support["group_sha256"])
    out["fixed_support_sha256_expected"] = str(score_row["fixed_support_sha256"])
    out["c64_scores"] = None
    score_path = (r3.STAGE05 / score_row["file"]).resolve(strict=True)
    with np.load(score_path, allow_pickle=False) as archive:
        out["c64_scores"] = np.asarray(archive["c64_scores"], dtype=np.float64).tolist()
    c64_values = np.asarray(out["c64_scores"], dtype=np.float64)
    out["theta_c64_q90"] = float(np.quantile(c64_values, .90, method="higher"))
    out["sigma_c64_floor_002"] = max(float(np.std(c64_values, ddof=1)), .02)
    if len(out["support_paths"]) != 4 or any(not path.is_file() for path in out["support_paths"]):
        raise R8Error(f"{category}: verified frozen Support-1 image path is unavailable")
    return out


def _encode_stage05_c64_items(category: str, info: Mapping[str, Any], p0: Any,
                              impl: Any, backbone: Any, cfg: Any,
                              *, expected_count: int = 64) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Encode an already frozen C64 set without crossing Support-bound cache identities."""
    import numpy as np
    import torch

    paths = list(info.get("c64_paths", ()))
    groups = list(info.get("c64_groups", ()))
    frozen_c64 = info.get("frozen_info", {}).get("c64", {})
    if (len(paths) != expected_count or len(groups) != expected_count or
            len(info.get("c64_scores", ())) != expected_count):
        raise R8Error(f"{category}: frozen C64 path, group, or score receipt count is invalid")
    items = []
    started = time.perf_counter()
    for index, path in enumerate(paths):
        image, features = p0.encode_query(impl, backbone, str(path), cfg)
        if (len(features) != 1 or tuple(features[0].shape) != (784, 4096) or
                features[0].dtype != torch.float32 or
                not torch.isfinite(features[0]).all().item()):
            raise R8Error(f"{category}: native C64 feature contract failed at index {index}")
        rgb = np.asarray(image.convert("RGB"), dtype=np.uint8).copy()
        feature = features[0].detach().to("cpu").contiguous().numpy().copy()
        items.append({"rgb": rgb, "features": [feature],
                      "source_group_sha256": str(groups[index])})
    return items, {
        "reused": False, "source_run_id": None,
        "count": len(items), "c64_group_sha256": frozen_c64.get("group_sha256"),
        "c64_image_sha256": frozen_c64.get("image_sha256"),
        "new_c64_dino_encodings": len(items),
        "encoding_seconds": time.perf_counter() - started,
        "feature_contract": {"shape": [784, 4096], "dtype": "float32"},
    }


def _make_r9_c64_items(category: str, info: Mapping[str, Any], p0: Any,
                       impl: Any, backbone: Any, cfg: Any,
                       r7: Any, *, support_seed: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Reuse R7's cache only for its Support-0-bound trajectory."""
    if support_seed == 0:
        return r7._make_c64_items(category, info, p0, impl, backbone, cfg)
    if support_seed == 1:
        return _encode_stage05_c64_items(category, info, p0, impl, backbone, cfg)
    raise R8Error(f"{category}: unregistered R9 support seed {support_seed}")


def _r8_reference_rows() -> tuple[dict[str, Any], dict[str, list[dict[str, Any]]]]:
    """Load the exact sealed R8 Shuffle-0 private trajectories for parity."""
    root = OUTPUT_BASE / R8_RUN_ID / "full"
    seal_dir = root / "stage_seals"
    manifest = _read_json(OUTPUT_BASE / R8_RUN_ID / "manifest" / "run_manifest.json")
    stage_a = _read_json(seal_dir / "stage_a_seal.json")
    stage_b = _read_json(seal_dir / "stage_b_seal.json")
    complete = _read_json(seal_dir / "complete_seal.json")
    if (manifest.get("run_id") != R8_RUN_ID or stage_a.get("status") != "SEALED_COMPLETE" or
            stage_b.get("status") != "SEALED_COMPLETE" or complete.get("status") != "COMPLETE" or
            stage_b.get("stage_a_seal_sha256") != _sha256_file(seal_dir / "stage_a_seal.json") or
            complete.get("stage_b_seal_sha256") != _sha256_file(seal_dir / "stage_b_seal.json")):
        raise R8Error("BLOCKED_REFERENCE_INPUT: exact R8 seals are absent or invalid")
    rows_by_category: dict[str, list[dict[str, Any]]] = {}
    for category in CATEGORIES:
        path = root / f"{category}_shuffle0" / "stage_a_scores.jsonl"
        seal_path = path.parent / "stage_a_seal.json"
        rows = _read_jsonl(path)
        if (len(rows) != EXPECTED_COUNTS[category] or
                _sha256_file(path) != _read_json(seal_path).get("score_rows_sha256")):
            raise R8Error(f"BLOCKED_REFERENCE_INPUT: R8 {category} rows fail the Stage A receipt")
        rows_by_category[category] = rows
    return {"run_id": R8_RUN_ID,
            "manifest_sha256": _sha256_file(OUTPUT_BASE / R8_RUN_ID / "manifest" / "run_manifest.json"),
            "stage_a_seal_sha256": _sha256_file(seal_dir / "stage_a_seal.json"),
            "stage_b_seal_sha256": _sha256_file(seal_dir / "stage_b_seal.json"),
            "complete_seal_sha256": _sha256_file(seal_dir / "complete_seal.json")}, rows_by_category


def build_identity(sources: Mapping[str, str], runner_sha256: str,
                   r7_receipts: Mapping[str, Any], contract: Mapping[str, Any],
                   support1_receipt: Mapping[str, Any] | None = None,
                   r8_receipts: Mapping[str, Any] | None = None) -> dict[str, Any]:
    from experiments.calibration import r7_shuffled_correction_4class as r7
    classes: dict[str, Any] = {}
    for category in CATEGORIES:
        info = contract["metadata"][category]
        shuffled = list(info["events"])
        query_ids = {str(shuffled[p].image_identity_sha256)
                     for p in info["tape_receipt"]["positions"]}
        order_records = {}
        for order in _registered_orders(support1_receipt):
            events = make_order_events(shuffled, order)
            positions = matched_query_positions(shuffled, info["tape_receipt"]["positions"], events)
            order_records[order] = {
                "event_count": len(events), "query_count": len(positions),
                "stream_shuffle_seed": int(order[-2:]) if order in {"SHUFFLE11", "SHUFFLE23"} else 0,
                "support_seed": 1 if order.startswith("SUPPORT1_") else 0,
                "query_positions": list(positions),
                "query_position_sha256": r7._sha256_json(list(positions)),
                "identity_map_sha256": r7._sha256_json([
                    {"position": int(e.position),
                     "original_position": int(e.original_position),
                     "image_identity_sha256": str(e.image_identity_sha256),
                     "source_group_sha256": str(e.source_group_sha256),
                     "camera_id": str(e.camera_id)} for e in events]),
            }
        if len(query_ids) != EXPECTED_QUERIES[category]:
            raise R8Error(f"{category}: R7 identity-matched query budget changed")
        classes[category] = {
            "category_json_sha256": info["contract"]["category_json_sha256"],
            "fixed_support_sha256": info["fixed_support_sha256_expected"],
            "support_image_sha256": info["support_image_sha256"],
            "support_group_sha256": info["support_group_sha256"],
            "c64_image_sha256": info["contract"]["c64_image_sha256"],
            "c64_group_sha256": info["contract"]["c64_group_sha256"],
            "c64_score_tape_sha256": info["contract"]["c64_score_tape_sha256"],
            "theta_c64_q90": info["theta_c64_q90"],
            "sigma_c64_floor_002": info["sigma_c64_floor_002"],
            "r7_shuffled_identity_map_sha256": info["identity_receipt"]["mapping_sha256"],
            "r7_query_tape_sha256": info["tape_receipt"]["tape_sha256"],
            "r7_query_identity_set_sha256": r7._sha256_json(sorted(query_ids)),
            "support_seed0": {
                "support_image_sha256": info["support_image_sha256"],
                "support_group_sha256": info["support_group_sha256"],
                "fixed_support_sha256": info["fixed_support_sha256_expected"],
            },
            "support_seed1": (dict(support1_receipt[category])
                              if support1_receipt is not None else None),
            "orders": order_records,
        }
    payload = {
        "schema": IDENTITY_SCHEMA, "run_id": RUN_ID,
        "classification": "RETROSPECTIVE_DEVELOPMENT_DIAGNOSTIC",
        "branch": EXPECTED_BRANCH, "source_parent_commit": EXPECTED_PARENT,
        "runner_source_sha256": runner_sha256, "source_fingerprints": dict(sources),
            "r7_reference": {
            "run_id": R7_RUN_ID,
            "source_parent_commit": r7_receipts["manifest"].get("source_parent_commit"),
            "launch_runner_source_sha256": r7_receipts["manifest"].get("runner_source_sha256"),
            "committed_runner_source_sha256": sources.get(
                "experiments/calibration/r7_shuffled_correction_4class.py"),
            "identity_sha256": r7_receipts["manifest"]["identity_sha256"],
            "score_rows_sha256": r7_receipts["score_sha256"],
            "stage_a_seal_sha256": _sha256_file(R7_FULL / "stage_a_seal.json"),
            "stage_b_seal_sha256": _sha256_file(R7_FULL / "stage_b_seal.json"),
            "complete_seal_sha256": _sha256_file(R7_FULL / "complete_seal.json"),
        },
            "r8_reference": dict(r8_receipts or {}),
            "configuration": {
            **_registered_condition_configuration(support1_receipt),
            "categories": list(CATEGORIES),
                "stream_shuffle_seeds": list(SHUFFLE_SEEDS),
                "fixed_normal_support_images_per_category": 4,
            "additional_protected_c64_normals_per_category": 64,
            "fixed_normal_capacity": 25000,
                "greedy_capacity": 1024, "correction_capacity": 1024,
            "backbone_updates": False, "unqueried_self_training": False,
            "masks_in_online_scoring": False,
                "query_identity_policy": "exact R7 Q-A-TS Seed37 image identity set remapped to each deterministic group-shuffle arrival; Shuffle11/23 are identity-matched stream-order interventions",
            "query_tape_sha256_by_category": {
                c: contract["metadata"][c]["tape_receipt"]["tape_sha256"] for c in CATEGORIES},
                "c1_formula": "frozen native C1-Full, original writer and readout at K=1024",
            "controller_settings": {"B1": {"window": 256, "minimum_normals": 16,
                                             "quantile": "higher"},
                                    "B3": {"delta": 0.01, "minimum_c64_normals": 16,
                                             "quantile": "higher"}},
            "address_shuffle_seeds": list(ADDRESS_SEEDS),
                "address_attribute_arrays": ["beta"],
                "signed_exemplar": {"capacity": 1024, "positive_slots": 512,
                    "negative_slots": 512, "retention": "independent deterministic FIFO",
                    "r0": 0.10, "beta_max": 0.20},
            "bootstrap_replicates": BOOTSTRAP_REPLICATES,
            "bootstrap_seed": BOOTSTRAP_SEED,
        },
        "classes": classes,
    }
    payload["identity_sha256"] = r7._sha256_json(payload)
    return payload


def make_address_permutation(seed: int, state_identity: str, eligible_count: int) -> tuple[int, ...]:
    """Local RNG keyed only by pre-score Correction state and registered seed."""
    if seed not in ADDRESS_SEEDS or eligible_count < 0:
        raise R8Error("address permutation seed or eligible count is invalid")
    raw = hashlib.sha256(f"r9-beta-only-v1\0{seed}\0{state_identity}\0{eligible_count}".encode("ascii")).digest()
    local_seed = int.from_bytes(raw[:8], "big", signed=False)
    import numpy as np
    return tuple(int(x) for x in np.random.default_rng(local_seed).permutation(eligible_count))


def permute_beta_only(codebook: Any, seed: int, state_identity: str) -> tuple[Any, tuple[int, ...]]:
    """Clone a source-filtered native codebook and permute only its beta values.

    The live C1 bank and the filtered prototype feature vectors remain untouched.
    Source-group exclusions must be applied before calling this helper.
    """
    import torch
    if not hasattr(codebook, "n") or not hasattr(codebook, "apply"):
        raise R8Error("address control requires a native filtered SignedCorrectionCodebook")
    clone = type(codebook)(int(codebook.K), int(codebook.protos.shape[1]),
                           codebook.device, codebook.cfg)
    clone.n = int(codebook.n)
    names = ("protos", "beta", "radius", "kappa", "count", "outcome_fp",
             "outcome_fn", "outcome_tn", "outcome_tp", "last_step")
    for name in names:
        getattr(clone, name)[:clone.n].copy_(getattr(codebook, name)[:clone.n])
    n = clone.n
    permutation = make_address_permutation(seed, state_identity, n)
    if n:
        p = torch.as_tensor(permutation, device=clone.device, dtype=torch.long)
        snapshot = clone.beta[:n].clone()
        clone.beta[:n].copy_(snapshot[p])
    return clone, permutation


class SignedExemplarMemory:
    """Bounded, source-excluding signed nearest-exemplar memory for R9."""

    def __init__(self, *, feature_dim: int = 4096, device: Any = "cuda",
                 positive_capacity: int = 512, negative_capacity: int = 512,
                 r0: float = 0.10, beta_max: float = 0.20):
        import numpy as np
        import torch
        if (positive_capacity != 512 or negative_capacity != 512 or
                not math.isclose(r0, .10) or not math.isclose(beta_max, .20)):
            raise R8Error("Signed Exemplar capacity and inherited readout parameters are frozen")
        self.positive_capacity = positive_capacity
        self.negative_capacity = negative_capacity
        self.capacity = positive_capacity + negative_capacity
        self.feature_dim = int(feature_dim)
        self.device = torch.device(device)
        self.r0 = float(r0)
        self.beta_max = float(beta_max)
        self.prototypes = torch.zeros((self.capacity, self.feature_dim),
                                      dtype=torch.float32, device=self.device)
        self.group_digests = np.zeros((self.capacity, 32), dtype=np.uint8)
        self.positions = np.full((self.capacity,), -1, dtype=np.int32)
        self.counts = {1: 0, -1: 0}
        self.cursors = {1: 0, -1: 0}
        self.insertions = {1: 0, -1: 0}
        self.evictions = {1: 0, -1: 0}
        self.unavailable_sign_patch_events = 0
        self.source_exclusions = 0

    def _base(self, sign: int) -> int:
        if sign not in (-1, 1):
            raise R8Error("Signed Exemplar sign must be verified normal (-1) or defect (+1)")
        return 0 if sign == 1 else self.positive_capacity

    def _active_indices(self, sign: int, *, logical: bool = False) -> list[int]:
        base = self._base(sign)
        count = int(self.counts[sign])
        capacity = self.positive_capacity if sign == 1 else self.negative_capacity
        if count < capacity:
            return list(range(base, base + count))
        if not logical:
            return list(range(base, base + capacity))
        cursor = int(self.cursors[sign])
        return [base + ((cursor + offset) % capacity) for offset in range(capacity)]

    def insert(self, selected_features: Any, *, sign: int, source_group_sha256: str,
               position: int) -> dict[str, int]:
        import numpy as np
        import torch
        if (len(source_group_sha256) != 64 or any(c not in "0123456789abcdef" for c in source_group_sha256)
                or int(position) < 0):
            raise R8Error("Signed Exemplar insertion provenance is malformed")
        rows = selected_features.detach().to(self.device, dtype=torch.float32)
        if rows.ndim != 2 or rows.shape[1] != self.feature_dim or not torch.isfinite(rows).all().item():
            raise R8Error("Signed Exemplar insertion feature tensor is malformed")
        norms = torch.linalg.vector_norm(rows, dim=1)
        if not torch.allclose(norms, torch.ones_like(norms), rtol=0, atol=2e-5):
            raise R8Error("Signed Exemplar requires normalized native Top-32 features")
        base = self._base(int(sign))
        capacity = self.positive_capacity if sign == 1 else self.negative_capacity
        inserted = evicted = 0
        group_row = np.frombuffer(bytes.fromhex(source_group_sha256), dtype=np.uint8)
        for row in rows:
            slot_in_partition = int(self.cursors[sign])
            slot = base + slot_in_partition
            if int(self.counts[sign]) == capacity:
                evicted += 1
                self.evictions[sign] += 1
            else:
                self.counts[sign] += 1
            self.prototypes[slot].copy_(row)
            self.group_digests[slot] = group_row
            self.positions[slot] = int(position)
            self.cursors[sign] = (slot_in_partition + 1) % capacity
            self.insertions[sign] += 1
            inserted += 1
        if not inserted:
            raise R8Error("Signed Exemplar feedback contained no native Top-32 features")
        return {"inserted": inserted, "evicted": evicted,
                "positive_occupancy": int(self.counts[1]),
                "negative_occupancy": int(self.counts[-1])}

    def residual(self, selected_features: Any, source_group_sha256: str) -> tuple[Any, dict[str, Any]]:
        import numpy as np
        import torch
        query = selected_features.detach().to(self.device, dtype=torch.float32)
        if (query.ndim != 2 or query.shape[1] != self.feature_dim or
                not torch.isfinite(query).all().item()):
            raise R8Error("Signed Exemplar query features are malformed")
        q_norm = torch.linalg.vector_norm(query, dim=1)
        if not torch.allclose(q_norm, torch.ones_like(q_norm), rtol=0, atol=2e-5):
            raise R8Error("Signed Exemplar query must use native normalized Top-32 features")
        try:
            source_bytes = np.frombuffer(bytes.fromhex(source_group_sha256), dtype=np.uint8)
        except ValueError as exc:
            raise R8Error("Signed Exemplar query Source Group digest is malformed") from exc
        if source_bytes.shape != (32,):
            raise R8Error("Signed Exemplar query Source Group digest is malformed")
        terms = {}
        eligible_counts = {}
        excluded = {}
        for sign, key in ((1, "positive"), (-1, "negative")):
            indices = self._active_indices(sign)
            eligible = [index for index in indices
                        if not np.array_equal(self.group_digests[index], source_bytes)]
            excluded[key] = len(indices) - len(eligible)
            self.source_exclusions += excluded[key]
            eligible_counts[key] = len(eligible)
            if not eligible:
                terms[key] = torch.zeros((query.shape[0],), device=self.device,
                                         dtype=torch.float32)
                continue
            bank_indices = torch.as_tensor(eligible, dtype=torch.long, device=self.device)
            similarity = torch.clamp(query @ self.prototypes[bank_indices].T, -1.0, 1.0)
            distance = torch.clamp(1.0 - similarity.max(dim=1).values, min=0.0)
            terms[key] = torch.exp(-(distance * distance) / (2.0 * self.r0 * self.r0))
        if not eligible_counts["positive"] and not eligible_counts["negative"]:
            self.unavailable_sign_patch_events += int(query.shape[0])
        delta = self.beta_max * (terms["positive"] - terms["negative"])
        return delta, {"eligible_positive": eligible_counts["positive"],
                       "eligible_negative": eligible_counts["negative"],
                       "same_source_excluded_positive": excluded["positive"],
                       "same_source_excluded_negative": excluded["negative"],
                       "positive_occupancy": int(self.counts[1]),
                       "negative_occupancy": int(self.counts[-1]),
                       "both_signs_unavailable": not eligible_counts["positive"] and not eligible_counts["negative"]}

    def state_sha256(self) -> str:
        import hashlib
        digest = hashlib.sha256()
        digest.update(b"r9-signed-exemplar-state-v1\0")
        for sign in (1, -1):
            active = self._active_indices(sign, logical=True)
            tensor = self.prototypes[active].detach().to("cpu").contiguous()
            digest.update(str((sign, tuple(tensor.shape), str(tensor.dtype))).encode("ascii"))
            digest.update(tensor.numpy().tobytes())
            digest.update(self.group_digests[active].tobytes())
            digest.update(self.positions[active].tobytes())
            digest.update(str((self.counts[sign], self.cursors[sign])).encode("ascii"))
        return digest.hexdigest()

    def resource_ledger(self) -> dict[str, int]:
        import sys
        dictionaries = (self.counts, self.cursors, self.insertions, self.evictions)
        shallow_python = sys.getsizeof(vars(self))
        for values in dictionaries:
            shallow_python += sys.getsizeof(values)
            shallow_python += sum(sys.getsizeof(key) + sys.getsizeof(value)
                                  for key, value in values.items())
        scalar_values = (self.positive_capacity, self.negative_capacity, self.capacity,
                         self.feature_dim, self.r0, self.beta_max,
                         self.unavailable_sign_patch_events, self.source_exclusions)
        shallow_python += sum(sys.getsizeof(value) for value in scalar_values)
        shallow_python += sys.getsizeof(self.device)
        return {"prototype_allocated_bytes": int(self.prototypes.numel() * self.prototypes.element_size()),
                "group_digest_allocated_bytes": int(self.group_digests.nbytes),
                "position_metadata_allocated_bytes": int(self.positions.nbytes),
                "measured_fixed_array_bytes": int(self.prototypes.numel() * self.prototypes.element_size() +
                                                    self.group_digests.nbytes + self.positions.nbytes),
                "per_exemplar_python_objects": 0,
                "python_metadata_shallow_bytes_separately_reported": int(shallow_python)}


def beta_only_control_score(*, impl: Any, bank: Any, r1: Any, exp2a: Any,
                            cfg: Any, codebook: Any,
                            source_groups: Mapping[int, set[str]],
                            group_sha256: str, position: int,
                            selected_features: Any, base_map: Any, top_ids: Any,
                            seed: int, state_identity: str) -> tuple[float, dict[str, Any]]:
    """Filter causal/source-ineligible anchors before constructing beta shuffles."""
    import numpy as np
    import torch
    prior_cfg, prior_cb = exp2a._switch_codebook(bank, "R9-beta-only", cfg, codebook)
    try:
        filtered, eligible = r1._filtered_c1_codebook(
            impl, bank, source_groups, group_sha256, event_position=position)
        control, permutation = permute_beta_only(filtered, seed, state_identity)
        query = impl.l2_normalize(selected_features.to(bank.device))
        if eligible:
            delta = control.apply(query).detach().float().cpu().numpy().astype(np.float32, copy=True)
        else:
            delta = np.zeros((32,), dtype=np.float32)
        score = float(r1.max_score(r1.corrected_dense_map(
            base_map, top_ids, delta)))
    finally:
        exp2a._restore_codebook(bank, cfg, prior_cfg, prior_cb)
    return score, {"eligible_count": len(eligible),
                   "permutation_sha256": _sha256_json(list(permutation)),
                   "seed": int(seed)}


def signed_exemplar_score(*, memory: SignedExemplarMemory, impl: Any, bank: Any,
                          r1: Any, base_map: Any, top_ids: Any,
                          selected_features: Any, source_group_sha256: str
                          ) -> tuple[float, dict[str, Any]]:
    delta, diagnostics = memory.residual(selected_features, source_group_sha256)
    delta_np = delta.detach().float().to("cpu").numpy().astype("float32", copy=True)
    score = float(r1.max_score(r1.corrected_dense_map(base_map, top_ids, delta_np)))
    return score, diagnostics


def scalar_control_score(base_score: float, c64_full: Sequence[float],
                         c64_base: Sequence[float]) -> tuple[float, float]:
    import numpy as np
    full, base = np.asarray(c64_full, dtype=np.float64), np.asarray(c64_base, dtype=np.float64)
    if full.shape != (64,) or base.shape != (64,) or not np.isfinite(full).all() or not np.isfinite(base).all():
        raise R8Error("Scalar control requires 64 finite current-state C64 score pairs")
    offset = float(np.mean(full - base))
    score = float(base_score) + offset
    if not math.isfinite(score):
        raise R8Error("Scalar control produced a nonfinite score")
    return score, offset


def _build_order_identity(contract: Mapping[str, Any], r7_manifest: Mapping[str, Any]) -> dict[str, Any]:
    classes = {}
    for category in CATEGORIES:
        info = contract["metadata"][category]
        shuffled = info["events"]
        original = make_order_events(shuffled, "RAW")
        qpos = r7_manifest["experiment_identity"]["classes"][category]["shuffled_query_tape"]["positions"]
        classes[category] = {order: {
            "events": [asdict(event) for event in events],
            "query_positions": list(matched_query_positions(shuffled, qpos, events)),
            "query_tape_sha256": info["tape_receipt"]["tape_sha256"],
        } for order, events in (("RAW", original), ("SHUFFLE0", list(shuffled)))}
    return classes


def _free_space(path: Path) -> int:
    import shutil
    path.mkdir(parents=True, exist_ok=True)
    free = int(shutil.disk_usage(path).free)
    if free < MIN_FREE_BYTES:
        raise R8Error(f"prelaunch free space {free} is below 10 GiB")
    return free


def _ensure_no_reparse(path: Path, stop: Path) -> None:
    from experiments.calibration.r2_feedback_repair import _is_reparse
    current, boundary = path.absolute(), stop.absolute()
    while True:
        if current.exists() and _is_reparse(current):
            raise R8Error(f"reparse point in R8 output chain: {current}")
        if current == boundary:
            return
        parent = current.parent
        if parent == current or (parent != boundary and boundary not in parent.parents):
            raise R8Error("R8 output path escaped the authorized output base")
        current = parent


def _budget(path: Path, *, create: bool = False) -> Any:
    from experiments.calibration.r2_feedback_repair import OutputBudget
    return OutputBudget(path, create=create, max_bytes=OUTPUT_LIMIT,
                        failure_reserve_bytes=FAILURE_RESERVE, max_files=64)


def _write_json(path_budget: Any, name: str, value: Any, *, failure_reserved: bool = False) -> None:
    path_budget.write_json(name, value, failure_reserved=failure_reserved)


def _write_failure(root: Path, stage: str, error: BaseException) -> None:
    try:
        import traceback
        budget = _budget(root)
        _write_json(budget, "failure.json", {
            "schema": "r8.failure.v1", "run_id": RUN_ID,
            "stage": stage, "error_type": type(error).__name__,
            "message": str(error)[:4000],
            "traceback": "".join(traceback.format_exception(error))[-12000:],
            "partial_outputs_preserved": True, "automatic_restart": False,
        }, failure_reserved=True)
    except Exception:
        pass


def _event_row(event: Any, category: str, order: str) -> dict[str, Any]:
    return {"category": category, "order": order,
            "position": int(event.position),
            "original_json_position": int(event.original_position),
            "image_identity_sha256": str(event.image_identity_sha256),
            "source_group_sha256": str(event.source_group_sha256),
            "camera_id": int(event.camera_id)}


def _corr_state(bank: Any, source_groups: Mapping[int, set[str]], cfg: Any,
                codebook: Any, exp2a: Any, r1: Any) -> tuple[str, str]:
    prior_cfg, prior_cb = exp2a._switch_codebook(bank, "R8", cfg, codebook)
    try:
        correction_sha = r1._correction_state_sha256(bank, source_groups)
    finally:
        exp2a._restore_codebook(bank, cfg, prior_cfg, prior_cb)
    greedy_sha = r1._memory_content_sha256(bank)
    return greedy_sha, correction_sha


def _scorer_digest(fixed_sha: str, greedy_sha: str, correction_sha: str | None = None) -> str:
    payload = {"fixed_support_sha256": fixed_sha,
               "greedy_memory_sha256": greedy_sha}
    if correction_sha is not None:
        payload["correction_state_sha256"] = correction_sha
    return _sha256_json(payload)


def _address_control_codebook(impl: Any, bank: Any, r1: Any, exp2a: Any,
                              cfg: Any, codebook: Any,
                              source_groups: Mapping[int, set[str]],
                              group_sha: str, position: int,
                              seed: int, state_identity: str
                              ) -> tuple[Any, list[int], tuple[int, ...]]:
    prior_cfg, prior_cb = exp2a._switch_codebook(bank, "R8-address", cfg, codebook)
    try:
        filtered, eligible = r1._filtered_c1_codebook(
            impl, bank, source_groups, group_sha, event_position=position)
        shuffled, permutation = permute_attribute_bundles(filtered, seed, state_identity)
    finally:
        exp2a._restore_codebook(bank, cfg, prior_cfg, prior_cb)
    return shuffled, list(eligible), permutation


def _address_score(impl: Any, bank: Any, r1: Any, exp2a: Any, cfg: Any,
                   codebook: Any, source_groups: Mapping[int, set[str]],
                   group_sha: str, selected: Any, position: int,
                   base_map: Any, top_ids: Any, seed: int,
                   state_identity: str) -> tuple[float, tuple[int, ...]]:
    import torch
    shuffled, eligible, permutation = _address_control_codebook(
        impl, bank, r1, exp2a, cfg, codebook, source_groups,
        group_sha, position, seed, state_identity)
    q = impl.l2_normalize(selected.to(bank.device))
    delta = shuffled.apply(q).detach().float().cpu().numpy().astype("float32", copy=True)
    if not eligible:
        delta = __import__("numpy").zeros((32,), dtype="float32")
    # The corrected map is computed by the same registered C1 native map adapter.
    from experiments.calibration.r1_matched_correction_pilot import corrected_dense_map, max_score
    return float(max_score(corrected_dense_map(base_map, top_ids, delta))), permutation


def _state_digest_for_codebook(bank: Any, fixed_sha: str, source_groups: Mapping[int, set[str]],
                               cfg: Any, codebook: Any, exp2a: Any, r1: Any) -> str:
    prior_cfg, prior_cb = exp2a._switch_codebook(bank, "R8-digest", cfg, codebook)
    try:
        correction_sha = r1._correction_state_sha256(bank, source_groups)
    finally:
        exp2a._restore_codebook(bank, cfg, prior_cfg, prior_cb)
    return _scorer_digest(fixed_sha, r1._memory_content_sha256(bank), correction_sha)


def _score_codebook(impl: Any, bank: Any, r1: Any, exp2a: Any, cfg: Any,
                    codebook: Any, source_groups: Mapping[int, set[str]],
                    selected: Any, group_sha: str, position: int,
                    base_map: Any, top_ids: Any) -> tuple[float, dict[str, Any]]:
    import numpy as np
    prior_cfg, prior_cb = exp2a._switch_codebook(bank, "R8-score", cfg, codebook)
    try:
        delta, diagnostics = r1._c1_score_delta(
            impl, bank, selected, source_groups, group_sha, event_position=position)
    finally:
        exp2a._restore_codebook(bank, cfg, prior_cfg, prior_cb)
    corrected = r1.corrected_dense_map(base_map, top_ids,
                                       np.asarray(delta, dtype=np.float32))
    return float(r1.max_score(corrected)), diagnostics


def _score_c64_for_codebook(impl: Any, bank: Any, r7: Any, r1: Any, exp2a: Any,
                            items: Sequence[Mapping[str, Any]], base_cache: Sequence[Any],
                            cfg: Any, correction_cfg: Any, correction_cb: Any,
                            source_groups: Mapping[int, set[str]], category: str,
                            event_position: int) -> tuple[list[float], float]:
    prior_cfg, prior_cb = exp2a._switch_codebook(
        bank, "R8-C64", correction_cfg, correction_cb)
    try:
        values, elapsed = r7._score_c64_corrected(
            impl, bank, items, base_cache, cfg, r1, source_groups, category,
            event_position=event_position)
    finally:
        exp2a._restore_codebook(bank, correction_cfg, prior_cfg, prior_cb)
    return values, elapsed


def _verify_r9_child_identity(output: Path, category: str, order: str,
                              mode: str, smoke_events: int) -> tuple[dict[str, Any], dict[str, Any]]:
    if category not in CATEGORIES or mode not in {"SMOKE", "FULL"}:
        raise R8Error("child condition is outside the frozen R9 design")
    if not output.is_dir():
        raise R8Error("child output directory was not pre-created by the parent")
    _ensure_no_reparse(output, RUN_NAMESPACE)
    manifest = _read_json(RUN_NAMESPACE / "manifest" / "run_manifest.json")
    identity = manifest.get("identity", {})
    if (manifest.get("run_id") != RUN_ID or manifest.get("identity_sha256") !=
            identity.get("identity_sha256") or
            identity.get("source_parent_commit") != EXPECTED_PARENT or
            identity.get("runner_source_sha256") != _sha256_file(Path(__file__).resolve()) or
            identity.get("identity_sha256") != _sha256_json({
                key: value for key, value in identity.items() if key != "identity_sha256"})):
        raise R8Error("R9 child identity manifest is malformed or its runner source changed")
    if _source_fingerprints() != identity.get("source_fingerprints"):
        raise R8Error("R9 child pinned source fingerprint changed")
    _assert_repository_identity()
    expected_dir = RUN_NAMESPACE / ("smoke" if mode == "SMOKE" else "full")
    if expected_dir not in output.parents:
        raise R8Error("R8 child output is outside its designated mode namespace")
    order_info = identity["classes"][category]["orders"].get(order)
    if order_info is None:
        raise R8Error("R9 child condition is not part of this identity")
    if mode == "SMOKE" and not 0 < smoke_events <= SMOKE_EVENTS:
        raise R8Error("R9 smoke event limit exceeds the fixed smoke bound")
    return identity, order_info


def _stage_a_child(category: str, order: str, mode: str,
                   smoke_events: int = 0) -> dict[str, Any]:
    inherited_cuda_visibility = os.environ.get("CUDA_VISIBLE_DEVICES")
    import numpy as np
    import torch
    from collections import Counter
    from types import SimpleNamespace
    from experiments import p0_protocol_audit as p0
    from experiments.calibration import r1_matched_correction_pilot as r1
    from experiments.calibration import r2_exp2a_c1_strength as exp2a
    from experiments.calibration import r3_fullstream_2x2 as r3
    from experiments.calibration import r6_capacity1024_audiojack_runner as r6
    from experiments.calibration import r7_shuffled_correction_4class as r7
    from experiments.calibration.controllers import RollingQ90QueriedNormal, StateConsistentQuantile
    from experiments.calibration.native import _fixed_bank_digest
    from experiments.calibration.protocol import Feedback, Observation
    from experiments.risk_safe.native_bridge import dynamic_codebooks
    from v02_memory import MemoryArm
    from experiments.calibration.r1_runtime_supervisor import current_process_is_in_job

    output = Path(os.environ.get("R8_CHILD_OUTPUT", ""))
    if not current_process_is_in_job():
        raise R8Error("R8 Stage A child is outside the Windows Job Object")
    identity, order_receipt = _verify_r8_child_identity(
        output, category, order, mode, smoke_events)
    budget = _budget(output)
    r3._restore_cuda_visibility(inherited_cuda_visibility)
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise R8Error("R8 Stage A requires the single authorized CUDA device")
    torch.set_num_threads(1)
    impl = p0.require_sources()
    model_receipt = r3._get_verified_model_receipt()
    restore_hub, runtime_info = r3._runtime_and_native_setup(torch, impl, model_receipt)
    cfg, scorer_profile = r3._native_config(
        p0, __import__("experiments.calibration.profiles", fromlist=["NativeScorerConfig"]), r1)
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    started = time.monotonic()
    backbone = None
    reader = None
    try:
        setup_started = time.perf_counter()
        try:
            backbone = impl.DINOv2MultiLayerBackbone(device="cuda")
        finally:
            restore_hub()
        loaded_model_sha = r1._model_state_sha256(backbone.model)
        if loaded_model_sha != model_receipt.get("loaded_model_state_sha256"):
            raise R8Error("R8 loaded model weights differ from verified local model receipt")
        contract = r7._frozen_contract()
        info = contract["metadata"][category]
        events = make_order_events(info["events"], order)
        if len(events) != EXPECTED_COUNTS[category]:
            raise R8Error("R8 event population differs from frozen TEST count")
        expected_map_sha = _sha256_json([
            {"position": int(event.position), "original_position": int(event.original_position),
             "image_identity_sha256": str(event.image_identity_sha256),
             "source_group_sha256": str(event.source_group_sha256),
             "camera_id": str(event.camera_id)} for event in events])
        if expected_map_sha != order_receipt["identity_map_sha256"]:
            raise R8Error("R8 child event identity map differs from parent-frozen receipt")
        query_positions = tuple(int(v) for v in order_receipt["query_positions"])
        if (len(query_positions) != EXPECTED_QUERIES[category] or
                query_positions != tuple(sorted(query_positions))):
            raise R8Error("R8 current-order query positions are invalid")

        support_paths = [str(path) for path in info["contract"]["support_paths"]]
        support_features = p0.encode_reference_features(impl, backbone, support_paths, cfg)
        impl.set_seed(0)
        bank = p0.build_vmb(impl, support_features, support_paths, cfg)
        fixed_receipt = r3._check_bank_receipt(
            bank, p0, info["fixed_support_sha256_expected"], r1)
        fixed_sha = str(fixed_receipt["actual_structured_sha256"])
        r3._validate_correction_config(bank)
        cfg1024, banks1024 = r6._configure_r6_correction_capacity(bank)
        cb1024 = bank.corr_cb["p0"]["id"][0]
        if (len(banks1024) != 1 or banks1024[0] is not cb1024 or cb1024.size() != 0 or
                int(cb1024.K) != 1024 or int(cfg1024.K_correction) != 1024):
            raise R8Error("R8 native K=1024 C1 state did not initialize independently")
        cfg256 = type(cfg1024)(**asdict(cfg1024))
        cfg256.K_correction = 256
        cfg256.enabled = False
        cb256 = type(cb1024)(256, int(cb1024.protos.shape[1]), cb1024.device, cfg256)
        if (cb256.size() != 0 or cb256.K != 256 or cb256 is cb1024 or
                cb256.protos.data_ptr() == cb1024.protos.data_ptr()):
            raise R8Error("R8 C256 and C1024 do not have independent native storage")
        cfg1024.enabled = False
        assert_capacity_configs_differ_only_by_k(asdict(cfg256), asdict(cfg1024))
        # Both states are swapped through the existing native C1 adapter; the
        # common Greedy writer always sees its frozen C0 bank and score.
        bank.corr_cfg = cfg1024
        bank.corr_cb["p0"]["id"][0] = cb1024
        settings = SimpleNamespace(flat=1024, stm=1024, ltm=0, buffer=0,
                                   min_events=2, promote_every=16, cap_bytes=2 * 1024**3)
        writer = MemoryArm("greedy_coreset", bank, settings, seed=0)
        if any(cb.size() != 0 or int(cb.K) != 1024 for cb in dynamic_codebooks(bank)):
            raise R8Error("R8 shared Greedy K=1024 bank did not initialize empty")
        bank.use_defect_bank = False
        if _fixed_bank_digest(bank) != fixed_sha:
            raise R8Error("R8 fixed Support changed during scorer initialization")

        c64_items, c64_cache_receipt = r7._make_c64_items(
            category, info, p0, impl, backbone, cfg)
        if len(c64_items) != 64 or any(
                c64_items[i].get("source_group_sha256") != info["c64_groups"][i]
                for i in range(64)):
            raise R8Error("R8 protected C64 feature identity or order differs from contract")
        c64_base, base_cache, c64_base_seconds = r7._score_c64_base(
            impl, bank, c64_items, cfg, r1, category)
        c64_expected = [float(v) for v in info["c64_scores"]]
        if any(not r3._score_tolerance(a, b) for a, b in zip(c64_base, c64_expected)):
            raise R8Error("R8 initial G C64 scores differ from the frozen score tape")
        if not math.isclose(float(np.quantile(c64_base, .9, method="higher")),
                            float(info["theta_c64_q90"]), rel_tol=0, abs_tol=1e-12):
            raise R8Error("R8 initial C64 q90 differs from frozen receipt")
        theta0 = float(info["theta_c64_q90"])
        sigma0 = float(info["sigma_c64_floor_002"])
        if not math.isclose(max(float(np.std(c64_expected, ddof=1)), .02), sigma0,
                            rel_tol=0, abs_tol=1e-12):
            raise R8Error("R8 frozen initial C64 scale differs from the pinned receipt")

        groups256: dict[int, set[str]] = {}
        groups1024: dict[int, set[str]] = {}
        c64_corr256, c64_seconds256 = _score_c64_for_codebook(
            impl, bank, r7, r1, exp2a, c64_items, base_cache, cfg,
            cfg256, cb256, groups256, category, 0)
        c64_corr1024, c64_seconds1024 = _score_c64_for_codebook(
            impl, bank, r7, r1, exp2a, c64_items, base_cache, cfg,
            cfg1024, cb1024, groups1024, category, 0)
        for values in (c64_corr256, c64_corr1024):
            if any(not r3._score_tolerance(a, b) for a, b in zip(values, c64_base)):
                raise R8Error("R8 empty Correction C64 scores do not equal G")
        scorer_cbs = {"G1024": (None, None, None, c64_base),
                      "GC256": (cfg256, cb256, groups256, c64_corr256),
                      "GC1024": (cfg1024, cb1024, groups1024, c64_corr1024)}
        state_versions = {name: 0 for name in SCORERS}
        greedy_state_sha = r1._memory_content_sha256(bank)
        state_digests = {
            "G1024": _scorer_digest(fixed_sha, greedy_state_sha),
            "GC256": _state_digest_for_codebook(bank, fixed_sha, groups256,
                                                 cfg256, cb256, exp2a, r1),
            "GC1024": _state_digest_for_codebook(bank, fixed_sha, groups1024,
                                                  cfg1024, cb1024, exp2a, r1),
        }
        controllers: dict[str, Any] = {}
        b3_history: dict[str, list[dict[str, Any]]] = {}
        c64_group_sha = info["frozen_info"]["c64"]["group_sha256"]
        for scorer in SCORERS:
            controllers[f"{scorer}-B1"] = RollingQ90QueriedNormal(
                theta0, window=256, min_count=16, name=f"{scorer}-B1")
            b3 = StateConsistentQuantile(theta0, delta=.01, min_normals=16,
                                         name=f"{scorer}-B3")
            controllers[f"{scorer}-B3"] = b3
            c64_scores = scorer_cbs[scorer][3]
            initial = r7._initialize_b3(b3, c64_scores, c64_group_sha,
                                        state_digests[scorer], theta0)
            b3_history[scorer] = [initial]

        tape_sha = info["tape_receipt"]["tape_sha256"]
        event_limit = min(len(events), smoke_events) if mode == "SMOKE" else len(events)
        allowed_queries = tuple(p for p in query_positions if p < event_limit)
        query_set = set(allowed_queries)
        reader = r7.IndexedQueriedLabelReader(
            info["json_path"], category, info["contract"]["category_json_sha256"],
            events, allowed_queries)
        score_writer = r7.BudgetJsonl(budget, "stage_a_scores.jsonl")
        score_gate = r7.ScoreDisclosureGate(lambda row: score_writer.write(row, flush=True))
        counters: Counter[str] = Counter()
        feedback_rows = 0
        queried_normal = queried_defect = 0
        greedy_changes = 0
        correction_changes = {"GC256": 0, "GC1024": 0}
        correction_updates = {"GC256": 0, "GC1024": 0}
        correction_update_stats: dict[str, Counter[str]] = {
            "GC256": Counter(), "GC1024": Counter()}
        correction_first_full: dict[str, int | None] = {"GC256": None, "GC1024": None}
        correction_score_delta: dict[str, list[float]] = {"GC256": [], "GC1024": []}
        scalar_deltas: list[float] = []
        scalar_offsets: list[float] = []
        scalar_state_identities: list[str] = []
        prior_scalar_state: str | None = None
        prior_scalar_offset: float | None = None
        address_scores: dict[str, list[float]] = {str(seed): [] for seed in ADDRESS_SEEDS}
        address_permutation_receipts: dict[str, list[str]] = {str(seed): [] for seed in ADDRESS_SEEDS}
        b1_history: dict[str, list[dict[str, Any]]] = {scorer: [] for scorer in SCORERS}
        c64_counts = {"G1024": {"rounds": 1, "score_calls": 64,
                                "seconds": c64_base_seconds},
                      "GC256": {"rounds": 1, "score_calls": 64,
                                "seconds": c64_seconds256},
                      "GC1024": {"rounds": 1, "score_calls": 64,
                                 "seconds": c64_seconds1024}}
        writer_audit_initial = writer.audit()
        feedback_digest = hashlib.sha256()
        query_group_set: set[str] = set()
        setup_seconds = time.perf_counter() - setup_started
        torch.cuda.synchronize()
        encode_seconds = score_seconds = 0.0
        for position, event in enumerate(events[:event_limit]):
            causal = r1.CausalEventOrder(position)
            image_path = Path(event.image_path)
            if not image_path.is_file():
                raise R8Error(f"{category}/{order}:{position}: frozen image disappeared")
            torch.cuda.synchronize()
            timer = time.perf_counter()
            image, features = p0.encode_query(impl, backbone, str(image_path), cfg)
            torch.cuda.synchronize()
            encode_seconds += time.perf_counter() - timer
            counters["stream_image_encodings"] += 1
            if (len(features) != 1 or tuple(features[0].shape) != (784, 4096) or
                    features[0].dtype != torch.float32 or not torch.isfinite(features[0]).all().item()):
                raise R8Error("R8 fused DINO descriptor contract failed")
            timer = time.perf_counter()
            base_map, top_ids, base_score, summary = r1._native_dense_score(
                impl, bank, image, features, cfg)
            top_ids = np.asarray(top_ids, dtype=np.int64)
            if len(top_ids) != 32 or not np.array_equal(
                    top_ids, np.asarray(summary["layer0"]["topk_patch_idx"], dtype=np.int64)):
                raise R8Error("R8 native Top-32 patch identity differs from receipt")
            selected = impl.l2_normalize(features[0].to(bank.device))[
                torch.as_tensor(top_ids, device=bank.device, dtype=torch.long)]
            group_sha = str(event.source_group_sha256)
            score_map = {"G1024": float(base_score)}
            diagnostics: dict[str, Any] = {}
            for scorer, cfg_i, cb_i, source_groups, _ in (
                    ("GC256", cfg256, cb256, groups256, None),
                    ("GC1024", cfg1024, cb1024, groups1024, None)):
                score_map[scorer], diagnostics[scorer] = _score_codebook(
                    impl, bank, r1, exp2a, cfg_i, cb_i, source_groups,
                    selected, group_sha, position, base_map, top_ids)
                correction_score_delta[scorer].append(score_map[scorer] - float(base_score))
            scalar_score = scalar_offset = None
            if order == "SHUFFLE0":
                scalar_score, scalar_offset = scalar_control_score(
                    float(base_score), scorer_cbs["GC1024"][3], c64_base)
                scalar_state = state_digests["GC1024"]
                if (prior_scalar_state == scalar_state and prior_scalar_offset is not None and
                        scalar_offset != prior_scalar_offset):
                    raise R8Error("R8 Scalar offset changed without a complete scorer-state change")
                scalar_offsets.append(scalar_offset)
                scalar_deltas.append(scalar_score - float(base_score))
                scalar_state_identities.append(scalar_state)
                prior_scalar_state, prior_scalar_offset = scalar_state, scalar_offset
                state_identity = _sha256_json({
                    "category": category, "order": order,
                    "scorer_state_sha256": state_digests["GC1024"]})
                for seed in ADDRESS_SEEDS:
                    value, permutation = _address_score(
                        impl, bank, r1, exp2a, cfg1024, cb1024, groups1024,
                        group_sha, selected, position, base_map, top_ids,
                        seed, state_identity)
                    address_scores[str(seed)].append(value)
                    address_permutation_receipts[str(seed)].append(_sha256_json(list(permutation)))
            torch.cuda.synchronize()
            score_seconds += time.perf_counter() - timer
            # Registered mutations happen only after queried feedback. Reuse
            # the last verified state receipt instead of hashing GPU banks at
            # every observation.
            current_digests = dict(state_digests)
            query_decision = position in query_set
            predictions: dict[str, tuple[Any, Any]] = {}
            decisions: dict[str, Any] = {}
            observations: dict[str, Any] = {}
            for scorer in SCORERS:
                score = score_map[scorer]
                decisions[f"{scorer}-Frozen"] = {
                    "prediction": int(score >= theta0), "threshold": theta0,
                    "status": "FROZEN_C64_Q90", "scorer_state_version": state_versions[scorer],
                    "scorer_state_sha256": state_digests[scorer]}
                obs = Observation(t=position, observation_id=event.event_id_sha256,
                                  camera_id=int(event.camera_id), score=score,
                                  memory_version=state_versions[scorer],
                                  score_state_digest=state_digests[scorer])
                observations[scorer] = obs
                b1 = controllers[f"{scorer}-B1"]
                b1.observe_score(obs)
                pred1 = b1.predict(obs)
                if pred1.label is None or not pred1.available:
                    raise R8Error("R8 B1 unexpectedly abstained")
                decisions[f"{scorer}-B1"] = {
                    "prediction": int(pred1.label), "threshold": float(pred1.threshold),
                    "status": "ROLLING_QUERIED_NORMAL_Q90",
                    "scorer_state_version": state_versions[scorer],
                    "scorer_state_sha256": state_digests[scorer]}
                predictions[f"{scorer}-B1"] = (obs, pred1)
                b3_record = r7._predict_b3(
                    controllers[f"{scorer}-B3"], position=position, event=event,
                    score=score, version=state_versions[scorer],
                    state_digest=state_digests[scorer])
                decisions[f"{scorer}-B3"] = b3_record
                predictions[f"{scorer}-B3"] = (
                    obs, controllers[f"{scorer}-B3"].predict(obs))
            row = {
                "schema": SCORE_SCHEMA, "run_id": RUN_ID, "category": category,
                "order": order, "position": position,
                "original_json_position": int(event.original_position),
                "event_id_sha256": str(event.event_id_sha256),
                "image_identity_sha256": str(event.image_identity_sha256),
                "source_group_sha256": group_sha, "camera_id": int(event.camera_id),
                "query_tape_decision": bool(query_decision),
                "query_identity_set_sha256": identity["classes"][category]["r7_query_identity_set_sha256"],
                "query_tape_sha256": identity["classes"][category]["r7_query_tape_sha256"],
                "scores": {**score_map,
                           **({"Scalar": scalar_score,
                              **{f"Address{seed}": address_scores[str(seed)][-1]
                                 for seed in ADDRESS_SEEDS}}
                              if order == "SHUFFLE0" else {})},
                "correction_score_delta": {
                    "GC256": score_map["GC256"] - float(base_score),
                    "GC1024": score_map["GC1024"] - float(base_score)},
                "scalar_offset": scalar_offset,
                "scalar_state_sha256": state_digests["GC1024"] if order == "SHUFFLE0" else None,
                "decisions": decisions,
                "score_state_versions": dict(state_versions),
                "score_state_sha256": dict(state_digests),
                "structured_fixed_support_sha256": fixed_sha,
                "native_c1_formula": "frozen native C1-Full SignedCorrection; original update/readout math",
            }
            causal.seal_scores()
            causal.decide_query(query_decision)
            score_gate.seal(position, row)
            label = None
            if query_decision:
                label = causal.disclose(lambda: reader.label_for(
                    position, sealed_and_fsynced=score_gate.is_sealed(position)))
                if group_sha in query_group_set:
                    raise R8Error("R8 identity-matched query tape repeats a Source Group")
                query_group_set.add(group_sha)
                queried_normal += int(label == 0)
                queried_defect += int(label == 1)
                feedback_rows += 1
                outcome = {(0, 0): "TN", (0, 1): "FP",
                           (1, 0): "FN", (1, 1): "TP"}[
                               label, decisions["G1024-Frozen"]["prediction"]]
                sign = 1 if label else -1
                before_corr = {scorer: current_digests[scorer] for scorer in ("GC256", "GC1024")}
                update_stats = {}
                for scorer, cfg_i, cb_i, source_groups in (
                        ("GC256", cfg256, cb256, groups256),
                        ("GC1024", cfg1024, cb1024, groups1024)):
                    _result, layer, trace = exp2a._capture_update(
                        impl, bank, r1, f"R8-{scorer}", cfg_i, cb_i,
                        features, top_ids, sign, float(base_score), theta0,
                        outcome, position, source_groups, group_sha)
                    update_stats[scorer] = {"native_stats": {
                        str(key): int(value) for key, value in layer.items()
                        if isinstance(value, (int, np.integer))},
                        "trace_integer_fields": {
                            str(key): int(value) for key, value in trace.items()
                            if isinstance(value, (int, np.integer))},
                        "occupancy_after": int(cb_i.size()),
                        "positive_after": int((cb_i.beta[:cb_i.size()] > 0).sum().item()),
                        "negative_after": int((cb_i.beta[:cb_i.size()] < 0).sum().item())}
                    correction_updates[scorer] += 1
                    correction_update_stats[scorer].update({
                        str(key): int(value) for key, value in layer.items()
                        if isinstance(value, (int, np.integer))})
                    if int(cb_i.size()) == int(cb_i.K) and correction_first_full[scorer] is None:
                        correction_first_full[scorer] = position
                causal.corrections_updated()
                writer_before = r1._memory_content_sha256(bank)
                audit_before = writer.audit()
                changed = {scorer: before_corr[scorer] != _state_digest_for_codebook(
                    bank, fixed_sha, source_groups,
                    cfg256 if scorer == "GC256" else cfg1024,
                    cb256 if scorer == "GC256" else cb1024, exp2a, r1)
                    for scorer, source_groups in (("GC256", groups256), ("GC1024", groups1024))}
                for scorer, did_change in changed.items():
                    correction_changes[scorer] += int(did_change)

                def writer_update(*, position: int, feature_layers: Sequence[Any],
                                  candidate_ids: Any, score: float, theta: float,
                                  sigma: float) -> Mapping[str, Any]:
                    before = r1._memory_content_sha256(bank)
                    writer.update(impl, list(feature_layers),
                                  {0: candidate_ids.tolist()}, position,
                                  score=score, theta=theta, sigma=sigma)
                    after = r1._memory_content_sha256(bank)
                    audit_after = writer.audit()
                    return {"accepted_patches_delta": int(audit_after["accepted_patches"]-
                                                             audit_before["accepted_patches"]),
                            "skipped_patches_delta": int(audit_after["skipped_patches"]-
                                                           audit_before["skipped_patches"]),
                            "_memory_content_changed": before != after}
                writer_action = r1.common_c0_greedy_writer_action(
                    position=position, c0_score=float(base_score), theta=theta0,
                    sigma=sigma0, c0_prediction=decisions["G1024-Frozen"]["prediction"],
                    candidate_ids=top_ids, candidate_features=selected.detach().float().cpu().numpy(),
                    feature_layers=features, queried=True, disclosed_label=label,
                    eligibility=impl.select_informative_normal_write, update=writer_update)
                writer_action.pop("_memory_content_changed", None)
                writer_after = r1._memory_content_sha256(bank)
                greedy_changed = writer_before != writer_after
                greedy_changes += int(greedy_changed)
                if (label == 1 and writer_action.get("action") != "verified_defect_no_normal_write"):
                    raise R8Error("R8 queried defect entered the normal Greedy memory")
                if writer_action.get("input_score_source") != "common_C0" or float(
                        writer_action.get("score", math.nan)) != float(base_score):
                    raise R8Error("R8 common Greedy writer was not controlled by common C0")
                after_digests = {
                    "G1024": _scorer_digest(fixed_sha, r1._memory_content_sha256(bank)),
                    "GC256": _state_digest_for_codebook(bank, fixed_sha, groups256,
                                                         cfg256, cb256, exp2a, r1),
                    "GC1024": _state_digest_for_codebook(bank, fixed_sha, groups1024,
                                                          cfg1024, cb1024, exp2a, r1),
                }
                changed_scorers = [name for name in SCORERS
                                   if after_digests[name] != state_digests[name]]
                if greedy_changed and "G1024" not in changed_scorers:
                    raise R8Error("R8 common Greedy change is absent from the base scorer identity")
                if changed["GC256"] and "GC256" not in changed_scorers:
                    raise R8Error("R8 C256 correction update is absent from its scorer identity")
                if changed["GC1024"] and "GC1024" not in changed_scorers:
                    raise R8Error("R8 C1024 correction update is absent from its scorer identity")
                causal.memory_updated()
                changed_c64: dict[str, list[float]] = {}
                if "G1024" in changed_scorers:
                    c64_base, base_cache, elapsed = r7._score_c64_base(
                        impl, bank, c64_items, cfg, r1, category)
                    scorer_cbs["G1024"] = (None, None, None, c64_base)
                    c64_counts["G1024"].update({"seconds": c64_counts["G1024"]["seconds"]+elapsed,
                                                "rounds": c64_counts["G1024"]["rounds"]+1,
                                                "score_calls": c64_counts["G1024"]["score_calls"]+64})
                for scorer, cfg_i, cb_i, source_groups in (
                        ("GC256", cfg256, cb256, groups256),
                        ("GC1024", cfg1024, cb1024, groups1024)):
                    if scorer in changed_scorers:
                        values, elapsed = _score_c64_for_codebook(
                            impl, bank, r7, r1, exp2a, c64_items, base_cache,
                            cfg, cfg_i, cb_i, source_groups, category, position + 1)
                        changed_c64[scorer] = values
                        scorer_cbs[scorer] = (cfg_i, cb_i, source_groups, values)
                        c64_counts[scorer].update({"seconds": c64_counts[scorer]["seconds"]+elapsed,
                                                   "rounds": c64_counts[scorer]["rounds"]+1,
                                                   "score_calls": c64_counts[scorer]["score_calls"]+64})
                for scorer in SCORERS:
                    if scorer in changed_scorers:
                        state_versions[scorer] += 1
                        trans = r7._refresh_b3(
                            controllers[f"{scorer}-B3"], scorer_cbs[scorer][3], c64_group_sha,
                            state_digests[scorer], after_digests[scorer],
                            state_versions[scorer]-1)
                        if trans is None:
                            raise R8Error("R8 B3 did not refresh from its complete scorer state")
                        trans.update({"position": position, "apply_from_position": position+1})
                        b3_history[scorer].append(trans)
                state_digests = after_digests
                for scorer in SCORERS:
                    obs, pred = predictions[f"{scorer}-B1"]
                    controllers[f"{scorer}-B1"].observe_feedback(
                        obs, pred, Feedback(t=position, queried=True, label=label))
                    theta_next = float(controllers[f"{scorer}-B1"].finalize_step())
                    b1_history[scorer].append({"position": position,
                                               "apply_from_position": position+1,
                                               "threshold": theta_next,
                                               "verified_normal_count": len(controllers[f"{scorer}-B1"].history)})
                    obs3, pred3 = predictions[f"{scorer}-B3"]
                    controllers[f"{scorer}-B3"].observe_feedback(
                        obs3, pred3, Feedback(t=position, queried=True, label=label))
                feedback = {
                    "schema": "r8.private.feedback.v1", "category": category,
                    "order": order, "position": position,
                    "original_json_position": int(event.original_position),
                    "event_id_sha256": str(event.event_id_sha256),
                    "source_group_sha256": group_sha, "label": int(label),
                    "queried_after_score_seal": score_gate.is_sealed(position),
                    "writer_outcome_from_G1024_C0": outcome,
                    "correction_updates": update_stats,
                    "correction_state_changed": changed,
                    "common_greedy_writer_action": writer_action,
                    "greedy_state_changed": greedy_changed,
                    "scorer_states_changed": changed_scorers,
                    "state_versions_after": dict(state_versions),
                    "state_digests_after": dict(state_digests),
                    "b3_refreshes_apply_from_next_event": True,
                }
                budget.append_jsonl("attempt_history.jsonl", feedback)
                feedback_digest.update(_canonical(feedback))
                causal.finish()
            else:
                causal.corrections_updated()
                def no_query_update(**_kwargs: Any) -> Mapping[str, Any]:
                    raise R8Error("Unqueried R8 event attempted a supervised Greedy write")
                writer_action = r1.common_c0_greedy_writer_action(
                    position=position, c0_score=float(base_score), theta=theta0,
                    sigma=sigma0, c0_prediction=decisions["G1024-Frozen"]["prediction"],
                    candidate_ids=top_ids, candidate_features=selected.detach().float().cpu().numpy(),
                    feature_layers=features, queried=False, disclosed_label=None,
                    eligibility=impl.select_informative_normal_write, update=no_query_update)
                if writer_action.get("action") != "no_normal_memory_write":
                    raise R8Error("Unqueried R8 event caused a supervised normal-memory write")
                causal.memory_updated()
                causal.finish()
            if (position+1) % 256 == 0:
                budget.append_jsonl("attempt_history.jsonl", {
                    "schema": "r8.private.progress.v1", "category": category,
                    "order": order, "sealed_through_position": position,
                    "query_feedback_count": feedback_rows,
                    "scorer_state_versions": dict(state_versions),
                    "automatic_restart": False})
        score_writer.flush()
        reader.finish()
        reader.close()
        reader = None
        if feedback_rows != len(allowed_queries) or len(query_group_set) != feedback_rows:
            raise R8Error("R8 Stage A did not consume the exact unique query allowlist")
        if mode == "FULL" and feedback_rows != EXPECTED_QUERIES[category]:
            raise R8Error("R8 full stream query count differs from the matched identity set")
        if _fixed_bank_digest(bank) != fixed_sha:
            raise R8Error("R8 fixed deployed Support bank changed during stream processing")
        writer_audit = writer.audit()
        import psutil
        torch.cuda.synchronize()
        memory256 = r7._mutable_state_bytes(bank, cb256, dynamic_codebooks)
        memory1024 = r7._mutable_state_bytes(bank, cb1024, dynamic_codebooks)
        completion = {
            "schema": "r8.stage_a.condition_complete.v1", "run_id": RUN_ID,
            "category": category, "order": order, "mode": mode,
            "event_count": event_limit, "query_count": feedback_rows,
            "queried_normal": queried_normal, "queried_defect": queried_defect,
            "score_rows_sha256": _sha256_file(output / "stage_a_scores.jsonl"),
            "feedback_rows_sha256": feedback_digest.hexdigest(),
            "query_positions": list(allowed_queries),
            "query_timing_fraction_quartiles": {
                str(q): float(np.quantile(np.asarray(allowed_queries, dtype=np.float64) /
                                         max(1, event_limit), q))
                for q in (.25, .50, .75, 1.0)} if allowed_queries else {},
            "query_identity_set_sha256": identity["classes"][category]["r7_query_identity_set_sha256"],
            "structured_fixed_support_sha256": fixed_sha,
            "correction_capacity": {"GC256": 256, "GC1024": 1024},
            "correction_final_occupancy": {"GC256": int(cb256.size()),
                                            "GC1024": int(cb1024.size())},
            "mutable_memory_bytes": {"GC256": memory256, "GC1024": memory1024},
            "correction_updates": correction_updates,
            "correction_update_stats": {name: dict(values)
                                         for name, values in correction_update_stats.items()},
            "correction_state_changes": correction_changes,
            "correction_first_capacity_reached_position": correction_first_full,
            "greedy_occupancy": [int(cb.size()) for cb in dynamic_codebooks(bank)],
            "greedy_writer_audit": writer_audit,
            "greedy_state_changes": greedy_changes,
            "writer_inputs_are_G1024_C0": True,
            "scorer_state_versions": dict(state_versions),
            "scorer_state_sha256": dict(state_digests),
            "b1_threshold_history": b1_history,
            "b3_threshold_history": b3_history,
            "c64_rescore_costs": c64_counts,
            "c64_cache_receipt": c64_cache_receipt,
            "correction_delta_summary": {
                scorer: {"mean": float(np.mean(values)) if values else None,
                         "active_fraction": float(np.mean(np.asarray(values) != 0)) if values else None}
                for scorer, values in correction_score_delta.items()},
            "scalar_offset_summary": {"mean": float(np.mean(scalar_offsets)) if scalar_offsets else None,
                                      "std": float(np.std(scalar_offsets)) if scalar_offsets else None,
                                      "minimum": float(min(scalar_offsets)) if scalar_offsets else None,
                                      "maximum": float(max(scalar_offsets)) if scalar_offsets else None,
                                      "distinct_full_state_count": len(set(scalar_state_identities))},
            "address_permutation_seed_receipts": {
                seed: _sha256_json(receipts) for seed, receipts in address_permutation_receipts.items()},
            "actual_stream_image_encodings": int(counters["stream_image_encodings"]),
            "timing_seconds": {"setup": setup_seconds, "encoding": encode_seconds,
                               "scoring": score_seconds, "wall": time.monotonic()-started},
            "peak_cuda_allocated_bytes": int(torch.cuda.max_memory_allocated()),
            "peak_cuda_reserved_bytes": int(torch.cuda.max_memory_reserved()),
            "peak_cpu_rss_bytes": int(psutil.Process().memory_info().rss),
            "runtime_profile": runtime_info["runtime_profile"],
            "hub_source_tree_sha256": runtime_info["hub_source_tree_sha256"],
            "checkpoint_sha256": runtime_info["checkpoint_sha256"],
            "loaded_model_state_sha256": loaded_model_sha,
            "scorer_profile": asdict(scorer_profile) if hasattr(scorer_profile, "__dataclass_fields__") else str(scorer_profile),
            "test_masks_opened": False, "unqueried_test_labels_accessed": False,
            "labels_were_revealed_only_for_registered_queries": True,
            "automatic_restart": False,
        }
        budget.write_json("stage_a_seal.json", completion)
        return completion
    except BaseException as exc:
        _write_failure(output, "STAGE_A_CHILD", exc)
        raise
    finally:
        if reader is not None:
            reader.close()
        if backbone is not None:
            del backbone


def _stage_a_child(category: str, order: str, mode: str,
                   smoke_events: int = 0) -> dict[str, Any]:
    """R9 causal child: one encoder/writer with Full, beta-only and simple readouts."""
    inherited_cuda_visibility = os.environ.get("CUDA_VISIBLE_DEVICES")
    import numpy as np
    import torch
    import psutil
    from collections import Counter
    from types import SimpleNamespace
    from experiments import p0_protocol_audit as p0
    from experiments.calibration import r1_matched_correction_pilot as r1
    from experiments.calibration import r2_exp2a_c1_strength as exp2a
    from experiments.calibration import r3_fullstream_2x2 as r3
    from experiments.calibration import r6_capacity1024_audiojack_runner as r6
    from experiments.calibration import r7_shuffled_correction_4class as r7
    from experiments.calibration.controllers import RollingQ90QueriedNormal, StateConsistentQuantile
    from experiments.calibration.native import _fixed_bank_digest
    from experiments.calibration.protocol import Feedback, Observation
    from experiments.risk_safe.native_bridge import dynamic_codebooks
    from v02_memory import MemoryArm
    from experiments.calibration.r1_runtime_supervisor import current_process_is_in_job

    output = Path(os.environ.get("R9_CHILD_OUTPUT", ""))
    if not current_process_is_in_job():
        raise R8Error("R9 child is outside the Windows Job Object")
    identity, order_info = _verify_r9_child_identity(output, category, order, mode, smoke_events)
    budget = _budget(output)
    r3._restore_cuda_visibility(inherited_cuda_visibility)
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise R8Error("R9 requires the single authorized CUDA device")
    torch.set_num_threads(1)
    impl = p0.require_sources()
    model_receipt = r3._get_verified_model_receipt()
    restore_hub, runtime_info = r3._runtime_and_native_setup(torch, impl, model_receipt)
    cfg, scorer_profile = r3._native_config(
        p0, __import__("experiments.calibration.profiles", fromlist=["NativeScorerConfig"]), r1)
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    started = time.monotonic()
    backbone = None
    reader = None
    try:
        setup_started = time.perf_counter()
        try:
            backbone = impl.DINOv2MultiLayerBackbone(device="cuda")
        finally:
            restore_hub()
        loaded_model_sha = r1._model_state_sha256(backbone.model)
        if loaded_model_sha != model_receipt.get("loaded_model_state_sha256"):
            raise R8Error("R9 loaded weights differ from verified model receipt")
        runtime_info["loaded_model_state_sha256"] = loaded_model_sha
        contract = r7._frozen_contract()
        info = contract["metadata"][category]
        events = make_order_events(info["events"], order)
        support_seed = int(order_info["support_seed"])
        if support_seed == 1:
            info = support1_class_info(category, info)
            frozen_support = identity["classes"][category]["support_seed1"]
            if (info["fixed_support_sha256_expected"] != frozen_support["fixed_support_sha256"] or
                    info["support_image_sha256"] != frozen_support["support_image_sha256"]):
                raise R8Error("R9 Support-1 inputs differ from the frozen preflight receipt")
        if len(events) != EXPECTED_COUNTS[category]:
            raise R8Error("R9 TEST event count differs from the frozen receipt")
        identity_map_sha = _sha256_json([{"position": int(e.position),
            "original_position": int(e.original_position),
            "image_identity_sha256": str(e.image_identity_sha256),
            "source_group_sha256": str(e.source_group_sha256),
            "camera_id": str(e.camera_id)} for e in events])
        if identity_map_sha != order_info["identity_map_sha256"]:
            raise R8Error("R9 stream event identities differ from the pre-launch map")
        qpositions = tuple(int(v) for v in order_info["query_positions"])
        if len(qpositions) != EXPECTED_QUERIES[category] or qpositions != tuple(sorted(qpositions)):
            raise R8Error("R9 identity-matched query positions are invalid")

        support_paths = [str(p) for p in info.get("support_paths", info["contract"]["support_paths"])]
        support_features = p0.encode_reference_features(impl, backbone, support_paths, cfg)
        impl.set_seed(0)
        bank = p0.build_vmb(impl, support_features, support_paths, cfg)
        fixed_receipt = r3._check_bank_receipt(bank, p0, info["fixed_support_sha256_expected"], r1)
        fixed_sha = str(fixed_receipt["actual_structured_sha256"])
        r3._validate_correction_config(bank)
        correction_cfg, correction_banks = r6._configure_r6_correction_capacity(bank)
        correction_cb = bank.corr_cb["p0"]["id"][0]
        if (len(correction_banks) != 1 or correction_cb is not correction_banks[0] or
                int(correction_cb.K) != 1024 or correction_cb.size() != 0):
            raise R8Error("R9 native C1-Full K=1024 did not start empty")
        correction_cfg.enabled = False
        writer = MemoryArm("greedy_coreset", bank,
            SimpleNamespace(flat=1024, stm=1024, ltm=0, buffer=0, min_events=2,
                            promote_every=16, cap_bytes=2 * 1024**3), seed=0)
        if any(int(cb.K) != 1024 or cb.size() != 0 for cb in dynamic_codebooks(bank)):
            raise R8Error("R9 shared Greedy K=1024 did not start empty")
        bank.use_defect_bank = False
        if _fixed_bank_digest(bank) != fixed_sha:
            raise R8Error("R9 fixed Support bank changed during setup")

        c64_items, c64_cache_receipt = _make_r9_c64_items(
            category, info, p0, impl, backbone, cfg, r7, support_seed=support_seed)
        if len(c64_items) != 64 or any(c64_items[i]["source_group_sha256"] != info["c64_groups"][i]
                                       for i in range(64)):
            raise R8Error("R9 C64 identity/cache receipt mismatch")
        c64_base, base_cache, base_seconds = r7._score_c64_base(
            impl, bank, c64_items, cfg, r1, category)
        c64_expected = [float(v) for v in info["c64_scores"]]
        if any(not r3._score_tolerance(a, b) for a, b in zip(c64_base, c64_expected)):
            raise R8Error("R9 initial native C64 scores fail frozen support receipt")
        theta0 = float(np.quantile(c64_base, .90, method="higher"))
        sigma0 = max(float(np.std(c64_base, ddof=1)), .02)
        if (not math.isclose(theta0, float(info["theta_c64_q90"]), rel_tol=0, abs_tol=1e-12) or
                not math.isclose(sigma0, float(info["sigma_c64_floor_002"]), rel_tol=0, abs_tol=1e-12)):
            raise R8Error("R9 initial C64 q90/sigma differs from the frozen commissioning receipt")
        source_groups: dict[int, set[str]] = {}
        c64_full, full_c64_seconds = _score_c64_for_codebook(
            impl, bank, r7, r1, exp2a, c64_items, base_cache, cfg, correction_cfg,
            correction_cb, source_groups, category, 0)
        if any(not r3._score_tolerance(a, b) for a, b in zip(c64_full, c64_base)):
            raise R8Error("R9 empty native C1 C64 scores differ from G")
        simple = SignedExemplarMemory(feature_dim=4096, device=bank.device)
        c64_scores = {"G1024": list(c64_base), "GC1024": list(c64_full),
                      "SignedExemplar1024": list(c64_base)}
        c64_group_sha = info["frozen_info"]["c64"]["group_sha256"]
        greedy_sha = r1._memory_content_sha256(bank)
        state_versions = {name: 0 for name in SCORERS}
        state_digests = {
            "G1024": _scorer_digest(fixed_sha, greedy_sha),
            "GC1024": _state_digest_for_codebook(bank, fixed_sha, source_groups,
                correction_cfg, correction_cb, exp2a, r1),
            "SignedExemplar1024": _sha256_json({"fixed_support_sha256": fixed_sha,
                "greedy_memory_sha256": greedy_sha, "signed_exemplar_sha256": simple.state_sha256()}),
        }
        controllers: dict[str, Any] = {}
        b3_history: dict[str, list[dict[str, Any]]] = {}
        for scorer in SCORERS:
            controllers[f"{scorer}-B1"] = RollingQ90QueriedNormal(
                theta0, window=256, min_count=16, name=f"{scorer}-B1")
            b3 = StateConsistentQuantile(theta0, delta=.01, min_normals=16, name=f"{scorer}-B3")
            controllers[f"{scorer}-B3"] = b3
            initial = r7._initialize_b3(b3, c64_scores[scorer], c64_group_sha,
                                       state_digests[scorer], theta0)
            b3_history[scorer] = [initial]

        event_limit = min(len(events), smoke_events) if mode == "SMOKE" else len(events)
        allowed_queries = tuple(p for p in qpositions if p < event_limit)
        query_set = set(allowed_queries)
        reader = r7.IndexedQueriedLabelReader(info["json_path"], category,
            info["contract"]["category_json_sha256"], events, allowed_queries)
        score_writer = r7.BudgetJsonl(budget, "stage_a_scores.jsonl")
        score_gate = r7.ScoreDisclosureGate(lambda row: score_writer.write(row, flush=True))
        counters: Counter[str] = Counter()
        feedback_digest = hashlib.sha256()
        query_groups: set[str] = set()
        first_feedback = {"normal": None, "defect": None}
        c1_changes = c1_updates = greedy_changes = simple_insertions = simple_evictions = 0
        c1_stats: Counter[str] = Counter()
        c1_first_full = None
        simple_first_full = {"positive": None, "negative": None}
        b1_history = {name: [] for name in SCORERS}
        c64_costs = {name: {"rounds": 1, "score_calls": 64, "seconds": base_seconds}
                     for name in SCORERS}
        c64_costs["GC1024"]["seconds"] += full_c64_seconds
        setup_seconds = time.perf_counter() - setup_started
        torch.cuda.synchronize()
        encoding_seconds = scoring_seconds = update_seconds = 0.0
        for position, event in enumerate(events[:event_limit]):
            causal = r1.CausalEventOrder(position)
            if not Path(event.image_path).is_file():
                raise R8Error(f"{category}/{order}:{position}: TEST image is missing")
            torch.cuda.synchronize(); timer = time.perf_counter()
            image, features = p0.encode_query(impl, backbone, str(event.image_path), cfg)
            torch.cuda.synchronize(); encoding_seconds += time.perf_counter() - timer
            counters["stream_image_encodings"] += 1
            if len(features) != 1 or tuple(features[0].shape) != (784, 4096) or features[0].dtype != torch.float32:
                raise R8Error("R9 DINO descriptor shape/dtype differs from native contract")
            timer = time.perf_counter()
            base_map, top_ids, base_score, summary = r1._native_dense_score(impl, bank, image, features, cfg)
            top_ids = np.asarray(top_ids, dtype=np.int64)
            if len(top_ids) != 32 or not np.array_equal(
                    top_ids, np.asarray(summary["layer0"]["topk_patch_idx"], dtype=np.int64)):
                raise R8Error("R9 native Top-32 feature selection mismatch")
            selected = impl.l2_normalize(features[0].to(bank.device))[
                torch.as_tensor(top_ids, device=bank.device, dtype=torch.long)]
            group_sha = str(event.source_group_sha256)
            full_score, full_diag = _score_codebook(impl, bank, r1, exp2a,
                correction_cfg, correction_cb, source_groups, selected, group_sha,
                position, base_map, top_ids)
            simple_score, simple_diag = signed_exemplar_score(memory=simple, impl=impl,
                bank=bank, r1=r1, base_map=base_map, top_ids=top_ids,
                selected_features=selected, source_group_sha256=group_sha)
            beta_scores, beta_receipts = {}, {}
            for seed in ADDRESS_SEEDS:
                beta_scores[f"BetaOnly{seed}"], beta_receipts[str(seed)] = beta_only_control_score(
                    impl=impl, bank=bank, r1=r1, exp2a=exp2a, cfg=correction_cfg,
                    codebook=correction_cb, source_groups=source_groups,
                    group_sha256=group_sha, position=position, selected_features=selected,
                    base_map=base_map, top_ids=top_ids, seed=seed,
                    state_identity=state_digests["GC1024"])
            torch.cuda.synchronize(); scoring_seconds += time.perf_counter() - timer
            score_map = {"G1024": float(base_score), "GC1024": float(full_score),
                         "SignedExemplar1024": float(simple_score), **beta_scores}
            predictions, decisions = {}, {}
            for scorer in SCORERS:
                score = score_map[scorer]
                decisions[f"{scorer}-Frozen"] = {"prediction": int(score >= theta0),
                    "threshold": theta0, "status": "FROZEN_C64_Q90",
                    "scorer_state_version": state_versions[scorer],
                    "scorer_state_sha256": state_digests[scorer]}
                obs = Observation(t=position, observation_id=event.event_id_sha256,
                    camera_id=int(event.camera_id), score=score, memory_version=state_versions[scorer],
                    score_state_digest=state_digests[scorer])
                b1 = controllers[f"{scorer}-B1"]; b1.observe_score(obs); pred1 = b1.predict(obs)
                if pred1.label is None or not pred1.available:
                    raise R8Error("R9 B1 unexpectedly abstained")
                decisions[f"{scorer}-B1"] = {"prediction": int(pred1.label),
                    "threshold": float(pred1.threshold), "status": "ROLLING_QUERIED_NORMAL_Q90",
                    "scorer_state_version": state_versions[scorer],
                    "scorer_state_sha256": state_digests[scorer]}
                predictions[f"{scorer}-B1"] = (obs, pred1)
                b3 = controllers[f"{scorer}-B3"]
                decisions[f"{scorer}-B3"] = r7._predict_b3(b3, position=position,
                    event=event, score=score, version=state_versions[scorer],
                    state_digest=state_digests[scorer])
                predictions[f"{scorer}-B3"] = (obs, b3.predict(obs))
            queried = position in query_set
            row = {"schema": SCORE_SCHEMA, "run_id": RUN_ID, "category": category,
                "order": order, "position": position,
                "original_json_position": int(event.original_position),
                "event_id_sha256": str(event.event_id_sha256),
                "image_identity_sha256": str(event.image_identity_sha256),
                "source_group_sha256": group_sha, "camera_id": int(event.camera_id),
                "query_tape_decision": bool(queried),
                "query_identity_set_sha256": identity["classes"][category]["r7_query_identity_set_sha256"],
                "query_tape_sha256": identity["classes"][category]["r7_query_tape_sha256"],
                "scores": score_map, "decisions": decisions,
                "score_state_versions": dict(state_versions),
                "score_state_sha256": dict(state_digests),
                "structured_fixed_support_sha256": fixed_sha,
                "beta_only_receipts": beta_receipts,
                "full_eligible_anchor_count": int(full_diag.get("eligible_anchor_slots", 0)),
                "simple_eligible_exemplars": {"positive": simple_diag["eligible_positive"],
                                               "negative": simple_diag["eligible_negative"]}}
            causal.seal_scores(); causal.decide_query(queried); score_gate.seal(position, row)
            if queried:
                label = causal.disclose(lambda: reader.label_for(position,
                    sealed_and_fsynced=score_gate.is_sealed(position)))
                if group_sha in query_groups:
                    raise R8Error("R9 query identity set repeats a Source Group")
                query_groups.add(group_sha)
                feedback_key = "normal" if label == 0 else "defect"
                if first_feedback[feedback_key] is None: first_feedback[feedback_key] = position
                counters["queried_normal"] += int(label == 0); counters["queried_defect"] += int(label == 1)
                counters["feedback_rows"] += 1
                outcome = {(0, 0): "TN", (0, 1): "FP", (1, 0): "FN", (1, 1): "TP"}[
                    label, decisions["G1024-Frozen"]["prediction"]]
                update_timer = time.perf_counter()
                c1_before = state_digests["GC1024"]
                _result, layer, trace = exp2a._capture_update(impl, bank, r1,
                    "R9-C1-Full", correction_cfg, correction_cb, features, top_ids,
                    (1 if label == 1 else -1), float(base_score), theta0, outcome,
                    position, source_groups, group_sha)
                c1_updates += 1
                c1_stats.update({str(k): int(v) for k, v in layer.items()
                                 if isinstance(v, (int, np.integer))})
                if correction_cb.size() == correction_cb.K and c1_first_full is None:
                    c1_first_full = position
                c1_after = _state_digest_for_codebook(bank, fixed_sha, source_groups,
                    correction_cfg, correction_cb, exp2a, r1)
                c1_changed = c1_before != c1_after; c1_changes += int(c1_changed)
                greedy_before = r1._memory_content_sha256(bank); writer_audit = writer.audit()
                def writer_update(*, position: int, feature_layers: Sequence[Any],
                                  candidate_ids: Any, score: float, theta: float,
                                  sigma: float) -> Mapping[str, Any]:
                    before = r1._memory_content_sha256(bank)
                    writer.update(impl, list(feature_layers), {0: candidate_ids.tolist()},
                                  position, score=score, theta=theta, sigma=sigma)
                    after = r1._memory_content_sha256(bank); audit_after = writer.audit()
                    return {"accepted_patches_delta": int(audit_after["accepted_patches"]-
                             writer_audit["accepted_patches"]),
                            "skipped_patches_delta": int(audit_after["skipped_patches"]-
                             writer_audit["skipped_patches"]),
                            "_memory_content_changed": before != after}
                writer_action = r1.common_c0_greedy_writer_action(position=position,
                    c0_score=float(base_score), theta=theta0, sigma=sigma0,
                    c0_prediction=decisions["G1024-Frozen"]["prediction"],
                    candidate_ids=top_ids, candidate_features=selected.detach().float().cpu().numpy(),
                    feature_layers=features, queried=True, disclosed_label=label,
                    eligibility=impl.select_informative_normal_write, update=writer_update)
                writer_action.pop("_memory_content_changed", None)
                greedy_changed = greedy_before != r1._memory_content_sha256(bank)
                greedy_changes += int(greedy_changed)
                if label == 1 and writer_action.get("action") != "verified_defect_no_normal_write":
                    raise R8Error("R9 queried defect entered shared normal Greedy memory")
                if writer_action.get("input_score_source") != "common_C0" or float(
                        writer_action.get("score", math.nan)) != float(base_score):
                    raise R8Error("R9 shared Greedy writer was not controlled by pre-feedback C0")
                simple_before = simple.state_sha256()
                insert = simple.insert(selected, sign=(1 if label == 1 else -1),
                    source_group_sha256=group_sha, position=position)
                simple_after = simple.state_sha256()
                if simple_before == simple_after:
                    raise R8Error("R9 queried feedback did not mutate the signed exemplar bank")
                simple_insertions += insert["inserted"]; simple_evictions += insert["evicted"]
                for key, sign in (("positive", 1), ("negative", -1)):
                    if label == sign and simple.counts[sign] == 512 and simple_first_full[key] is None:
                        simple_first_full[key] = position
                greedy_sha = r1._memory_content_sha256(bank)
                after_digests = {"G1024": _scorer_digest(fixed_sha, greedy_sha),
                    "GC1024": _state_digest_for_codebook(bank, fixed_sha, source_groups,
                        correction_cfg, correction_cb, exp2a, r1),
                    "SignedExemplar1024": _sha256_json({"fixed_support_sha256": fixed_sha,
                        "greedy_memory_sha256": greedy_sha,
                        "signed_exemplar_sha256": simple_after})}
                changed = [name for name in SCORERS if after_digests[name] != state_digests[name]]
                if greedy_changed and set(changed) != set(SCORERS):
                    raise R8Error("R9 common Greedy change diverged across scorer identities")
                causal.corrections_updated()
                if greedy_changed:
                    c64_base, base_cache, elapsed = r7._score_c64_base(impl, bank,
                        c64_items, cfg, r1, category)
                    c64_scores["G1024"] = list(c64_base)
                    for name in SCORERS:
                        c64_costs[name]["rounds"] += 1; c64_costs[name]["score_calls"] += 64
                        c64_costs[name]["seconds"] += elapsed
                if "GC1024" in changed:
                    values, elapsed = _score_c64_for_codebook(impl, bank, r7, r1,
                        exp2a, c64_items, base_cache, cfg, correction_cfg, correction_cb,
                        source_groups, category, position + 1)
                    c64_scores["GC1024"] = values; c64_costs["GC1024"]["seconds"] += elapsed
                    if not greedy_changed:
                        c64_costs["GC1024"]["rounds"] += 1; c64_costs["GC1024"]["score_calls"] += 64
                if "SignedExemplar1024" in changed:
                    values = []; c64_start = time.perf_counter()
                    for item, (amap, ids, _base) in zip(c64_items, base_cache):
                        cf = torch.from_numpy(np.ascontiguousarray(item["features"][0])).to(bank.device)
                        chosen = impl.l2_normalize(cf)[torch.as_tensor(ids, device=bank.device, dtype=torch.long)]
                        score, _diag = signed_exemplar_score(memory=simple, impl=impl, bank=bank,
                            r1=r1, base_map=amap, top_ids=ids, selected_features=chosen,
                            source_group_sha256=str(item["source_group_sha256"]))
                        values.append(score)
                    torch.cuda.synchronize(); elapsed = time.perf_counter() - c64_start
                    c64_scores["SignedExemplar1024"] = values
                    c64_costs["SignedExemplar1024"]["seconds"] += elapsed
                    if not greedy_changed:
                        c64_costs["SignedExemplar1024"]["rounds"] += 1
                        c64_costs["SignedExemplar1024"]["score_calls"] += 64
                for scorer in SCORERS:
                    if scorer in changed:
                        old_version = state_versions[scorer]; state_versions[scorer] += 1
                        transition = r7._refresh_b3(controllers[f"{scorer}-B3"], c64_scores[scorer],
                            c64_group_sha, state_digests[scorer], after_digests[scorer], old_version)
                        if transition is None:
                            raise R8Error(f"R9 {scorer} B3 did not refresh current C64 evidence")
                        transition.update({"position": position, "apply_from_position": position + 1})
                        b3_history[scorer].append(transition)
                state_digests = after_digests
                for scorer in SCORERS:
                    obs, pred = predictions[f"{scorer}-B1"]
                    controllers[f"{scorer}-B1"].observe_feedback(
                        obs, pred, Feedback(t=position, queried=True, label=label))
                    b1_history[scorer].append({"position": position,
                        "apply_from_position": position + 1,
                        "threshold": float(controllers[f"{scorer}-B1"].finalize_step()),
                        "verified_normal_count": len(controllers[f"{scorer}-B1"].history)})
                    obs3, pred3 = predictions[f"{scorer}-B3"]
                    controllers[f"{scorer}-B3"].observe_feedback(
                        obs3, pred3, Feedback(t=position, queried=True, label=label))
                feedback = {"schema": "r9.private.feedback.v1", "category": category,
                    "order": order, "position": position,
                    "original_json_position": int(event.original_position),
                    "event_id_sha256": event.event_id_sha256, "source_group_sha256": group_sha,
                    "label": int(label), "queried_after_score_seal": score_gate.is_sealed(position),
                    "writer_outcome_from_G1024_C0": outcome,
                    "c1_native_update": {"state_changed": c1_changed,
                        "integer_stats": {str(k): int(v) for k, v in layer.items()
                            if isinstance(v, (int, np.integer))},
                        "trace_integer_fields": {str(k): int(v) for k, v in trace.items()
                            if isinstance(v, (int, np.integer))}},
                    "common_greedy_writer_action": writer_action, "greedy_state_changed": greedy_changed,
                    "signed_exemplar_insert": insert, "scorer_states_changed": changed,
                    "state_versions_after": dict(state_versions), "state_digests_after": dict(state_digests)}
                budget.append_jsonl("attempt_history.jsonl", feedback); feedback_digest.update(_canonical(feedback))
                update_seconds += time.perf_counter() - update_timer
                causal.memory_updated()
                causal.finish()
            else:
                causal.corrections_updated()
                def deny_unqueried(**_kwargs: Any) -> Mapping[str, Any]:
                    raise R8Error("Unqueried R9 event attempted a supervised memory write")
                action = r1.common_c0_greedy_writer_action(position=position,
                    c0_score=float(base_score), theta=theta0, sigma=sigma0,
                    c0_prediction=decisions["G1024-Frozen"]["prediction"],
                    candidate_ids=top_ids, candidate_features=selected.detach().float().cpu().numpy(),
                    feature_layers=features, queried=False, disclosed_label=None,
                    eligibility=impl.select_informative_normal_write, update=deny_unqueried)
                if action.get("action") != "no_normal_memory_write":
                    raise R8Error("Unqueried R9 event caused a supervised write")
                causal.memory_updated(); causal.finish()
            if (position + 1) % 256 == 0:
                budget.append_jsonl("attempt_history.jsonl", {"schema": "r9.private.progress.v1",
                    "category": category, "order": order, "sealed_through_position": position,
                    "query_feedback_count": int(counters["feedback_rows"]),
                    "scorer_state_versions": dict(state_versions), "automatic_restart": False})
        score_writer.flush(); reader.finish(); reader.close(); reader = None
        if int(counters["feedback_rows"]) != len(allowed_queries) or len(query_groups) != int(counters["feedback_rows"]):
            raise R8Error("R9 did not consume the exact unique query allowlist")
        if mode == "FULL" and int(counters["feedback_rows"]) != EXPECTED_QUERIES[category]:
            raise R8Error("R9 query count differs from the frozen identity set")
        if int(counters["stream_image_encodings"]) != event_limit:
            raise R8Error("R9 did not perform exactly one DINO encoding per TEST event")
        if _fixed_bank_digest(bank) != fixed_sha:
            raise R8Error("R9 fixed Support bank changed during stream")
        writer_audit = writer.audit(); torch.cuda.synchronize()
        c1_memory = r7._mutable_state_bytes(bank, correction_cb, dynamic_codebooks)
        simple_memory = simple.resource_ledger()
        c1_budget = int(c1_memory["correction_allocated_prototype_bytes"] +
                        c1_memory["correction_allocated_metadata_bytes"])
        if simple_memory["measured_fixed_array_bytes"] > c1_budget:
            raise R8Error("R9 Signed Exemplar exceeds C1 allocated prototype-plus-metadata bytes")
        timing = {"setup_seconds": setup_seconds, "encoding_seconds": encoding_seconds,
            "scoring_seconds": scoring_seconds, "update_and_rescore_seconds": update_seconds,
            "child_wall_seconds": time.monotonic() - started}
        completion = {"schema": "r9.stage_a.condition_complete.v1", "run_id": RUN_ID,
            "status": "SEALED_COMPLETE" if mode == "FULL" else "SMOKE_PREFIX_COMPLETE",
            "category": category, "order": order, "support_seed": support_seed,
            "stream_shuffle_seed": int(order_info["stream_shuffle_seed"]),
            "event_count": event_limit, "query_count": int(counters["feedback_rows"]),
            "query_positions": list(allowed_queries),
            "query_identity_set_sha256": identity["classes"][category]["r7_query_identity_set_sha256"],
            "score_rows_sha256": _sha256_file(output / "stage_a_scores.jsonl"),
            "feedback_records_sha256": feedback_digest.hexdigest(),
            "queried_normal": int(counters["queried_normal"]), "queried_defect": int(counters["queried_defect"]),
            "first_normal_feedback_position": first_feedback["normal"],
            "first_defect_feedback_position": first_feedback["defect"],
            "greedy_state_changes": greedy_changes, "c1_state_changes": c1_changes,
            "c1_update_count": c1_updates, "c1_final_occupancy": int(correction_cb.size()),
            "c1_first_full_position": c1_first_full, "c1_update_stats": dict(c1_stats),
            "simple_memory_final_occupancy": {"positive": int(simple.counts[1]),
                                               "negative": int(simple.counts[-1])},
            "simple_memory_insertions": simple_insertions, "simple_memory_evictions": simple_evictions,
            "simple_memory_first_full_position": simple_first_full,
            "simple_memory_unavailable_sign_patch_events": simple.unavailable_sign_patch_events,
            "simple_memory_source_exclusions": simple.source_exclusions,
            "greedy_writer_audit": writer_audit, "b1_threshold_history": b1_history,
            "b3_threshold_history": b3_history, "c64_rescore_costs": c64_costs,
            "c64_initial_scores_sha256": _sha256_json(c64_expected),
            "c64_cache_receipt": c64_cache_receipt, "fixed_support_receipt": fixed_receipt,
            "fixed_support_sha256": fixed_sha, "native_scorer_profile": asdict(scorer_profile),
            "native_scorer_profile_sha256": r3.sha256_json(asdict(scorer_profile)),
            "runtime_profile": runtime_info,
            "support_encoder_invocations": sum(len(support_features[path]) for path in support_paths),
            "stream_image_encodings": int(counters["stream_image_encodings"]),
            "test_masks_opened": False, "unqueried_test_labels_accessed": False,
            "offline_outcome_vectors_accessed": False, "timing_seconds": timing,
            "peak_cuda_allocated_bytes": int(torch.cuda.max_memory_allocated()),
            "peak_cuda_reserved_bytes": int(torch.cuda.max_memory_reserved()),
            "peak_cpu_rss_bytes": int(psutil.Process().memory_info().rss),
            "mutable_memory_bytes": {"c1_native": c1_memory, "signed_exemplar": simple_memory,
                "signed_exemplar_cap_bytes": c1_budget,
                "signed_exemplar_within_c1_prototype_plus_metadata_budget": True,
                "greedy_common_state_excluded_from_additional_memory_comparison": True},
            "complete_scorer_state_sha256": dict(state_digests),
            "all_scores_sealed_before_feedback": True, "automatic_restart": False,
            "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
        budget.write_json("stage_a_seal.json", completion)
        return completion
    except BaseException as exc:
        if reader is not None: reader.close()
        _write_failure(output, "R9_STAGE_A_CHILD", exc)
        raise
    finally:
        if backbone is not None: del backbone
        if torch.cuda.is_available(): torch.cuda.empty_cache()


def _condition_output(root: Path, category: str, order: str) -> Path:
    return root / f"{category}_{order.lower()}"


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as stream:
        for line_no, line in enumerate(stream, start=1):
            if not line.endswith("\n"):
                raise R8Error(f"{path.name}:{line_no}: row is not newline sealed")
            row = json.loads(line)
            if not isinstance(row, dict):
                raise R8Error(f"{path.name}:{line_no}: row is not a JSON object")
            rows.append(row)
    return rows


def _condition_identity_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [{"position": int(row.get("position", -1)),
        "original_position": int(row.get("original_json_position", -1)),
        "image_identity_sha256": str(row.get("image_identity_sha256", "")),
        "source_group_sha256": str(row.get("source_group_sha256", "")),
        "camera_id": str(row.get("camera_id", ""))} for row in rows]


def _compare_reference_rows(category: str, actual_rows: Sequence[Mapping[str, Any]],
                            reference_rows: Sequence[Mapping[str, Any]],
                            reference: str) -> dict[str, Any]:
    import numpy as np
    if len(actual_rows) > len(reference_rows):
        raise R8Error(f"{category}: R9 exceeds the {reference} reference population")
    score_errors = {"G1024": [], "GC1024": []}
    threshold_errors = {f"{scorer}-{controller}": []
        for scorer in ("G1024", "GC1024") for controller in CONTROLLERS}
    identity_mismatches = query_mismatches = decision_mismatches = 0
    for index, (actual, expected) in enumerate(zip(actual_rows, reference_rows)):
        if (int(actual.get("position", -1)) != index or int(expected.get("position", -1)) != index or
                actual.get("image_identity_sha256") != expected.get("image_identity_sha256") or
                int(actual.get("original_json_position", -1)) != int(expected.get("original_json_position", -2)) or
                actual.get("source_group_sha256") != expected.get("source_group_sha256") or
                actual.get("event_id_sha256") != expected.get("event_id_sha256") or
                int(actual.get("camera_id", -1)) != int(expected.get("camera_id", -2)) or
                actual.get("structured_fixed_support_sha256") != expected.get("structured_fixed_support_sha256")):
            identity_mismatches += 1
            continue
        query_mismatches += int(bool(actual.get("query_tape_decision")) != bool(expected.get("query_tape_decision")))
        for scorer, r7_name, r8_name in (("G1024", "G", "G1024"), ("GC1024", "GC", "GC1024")):
            ref_name = r7_name if reference == "R7" else r8_name
            if ref_name not in expected.get("scores", {}):
                raise R8Error(f"{reference} reference lacks {ref_name} score")
            score_errors[scorer].append(float(actual["scores"][scorer]) - float(expected["scores"][ref_name]))
            for controller in CONTROLLERS:
                actual_d = actual["decisions"][f"{scorer}-{controller}"]
                if reference == "R7":
                    reference_d = expected["decisions"][f"{r7_name}-{controller}"]
                else:
                    reference_d = expected["decisions"][f"{r8_name}-{controller}"]
                decision_mismatches += int(int(actual_d["prediction"]) != int(reference_d["prediction"]))
                threshold_errors[f"{scorer}-{controller}"].append(
                    float(actual_d["threshold"]) - float(reference_d["threshold"]))
    if identity_mismatches or query_mismatches or decision_mismatches:
        raise R8Error(f"{reference} parity failed for {category}: identity={identity_mismatches}, "
                      f"query={query_mismatches}, decisions={decision_mismatches}")
    max_score = {k: float(np.max(np.abs(v))) if v else 0.0 for k, v in score_errors.items()}
    max_threshold = {k: float(np.max(np.abs(v))) if v else 0.0 for k, v in threshold_errors.items()}
    score_mismatch = False
    for actual, expected in zip(actual_rows, reference_rows):
        for scorer, r7_name, r8_name in (("G1024", "G", "G1024"),
                                          ("GC1024", "GC", "GC1024")):
            ref_name = r7_name if reference == "R7" else r8_name
            delta = abs(float(actual["scores"][scorer]) - float(expected["scores"][ref_name]))
            if delta > 1e-5 + 1e-4 * abs(float(expected["scores"][ref_name])):
                score_mismatch = True
                break
        if score_mismatch:
            break
    if score_mismatch:
        raise R8Error(f"{reference} score parity exceeds atol=1e-5 / rtol=1e-4")
    if any(value > 1e-12 for value in max_threshold.values()):
        raise R8Error(f"{reference} thresholds differ for identical trajectories")
    return {"reference": reference, "event_count": len(actual_rows),
        "identity_mismatches": identity_mismatches, "query_mismatches": query_mismatches,
        "decision_mismatches": decision_mismatches,
        "maximum_absolute_score_error": max_score,
        "maximum_absolute_threshold_error": max_threshold,
        "atol": 1e-5, "rtol": 1e-4, "identity_bijection_verified": True}


def _compare_r7_rows(category: str, r9_rows: Sequence[Mapping[str, Any]],
                     r7_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return _compare_reference_rows(category, r9_rows, r7_rows, "R7")


def _compare_r8_rows(category: str, r9_rows: Sequence[Mapping[str, Any]],
                     r8_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return _compare_reference_rows(category, r9_rows, r8_rows, "R8")


def _make_run_manifest(identity: Mapping[str, Any], source_sha: str,
                       free_bytes: int, r7_receipts: Mapping[str, Any],
                       cpu_test_receipt: Mapping[str, Any]) -> dict[str, Any]:
    return {"schema": RUN_SCHEMA, "run_id": RUN_ID,
            "classification": "RETROSPECTIVE_DEVELOPMENT_DIAGNOSTIC",
            "authorization": AUTHORIZATION,
            "branch": EXPECTED_BRANCH, "source_parent_commit": EXPECTED_PARENT,
            "runner_source_sha256": source_sha,
            "focused_cpu_test": dict(cpu_test_receipt),
            "identity_sha256": identity["identity_sha256"], "identity": dict(identity),
            "r7_reference_score_rows_sha256": r7_receipts["score_sha256"],
            "r8_reference": dict(identity.get("r8_reference", {})),
            "support1_condition_status": (
                "REGISTERED" if identity["configuration"]["support_seeds"] == [0, 1]
                else "NOT_RUN_INPUT_UNVERIFIED"),
            "output_limit_bytes_per_condition": OUTPUT_LIMIT,
            "reserved_failure_metadata_bytes": FAILURE_RESERVE,
            "minimum_prelaunch_free_bytes": MIN_FREE_BYTES,
            "prelaunch_free_bytes": int(free_bytes),
            "full_child_deadline_seconds": FULL_DEADLINE_SECONDS,
            "smoke_child_deadline_seconds": SMOKE_DEADLINE_SECONDS,
            "automatic_restart": False, "resume_incomplete_run": False,
            "source_fingerprints": dict(identity["source_fingerprints"]),
            "stage_a_permissions": {
                "support_and_c64": True, "test_images_and_label_blind_metadata": True,
                "queried_labels_after_fsync_only": True,
                "unqueried_test_labels": False, "test_masks": False,
                "offline_outcome_vectors": False},
            "output_namespace": str(RUN_NAMESPACE.relative_to(ROOT)).replace("\\", "/"),
            "private_outputs_non_overwriting": True,
            "preserved_prior_attempts": list(PRESERVED_FAILED_ATTEMPTS)}


def _run_focused_cpu_tests() -> dict[str, Any]:
    test_paths = [ROOT / "tests" / "calibration" / name for name in (
        "test_r9_correction_mechanism_falsification.py",
        "test_r8_correction_mechanism_attribution.py")]
    if any(not path.is_file() for path in test_paths):
        raise R8Error("focused R9/R8 CPU regression test file is missing")
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = ""
    env["R3_STAGE_B_CPU_ONLY"] = "1"
    env["PYTHONHASHSEED"] = "0"
    command = [sys.executable, "-m", "pytest", "-q", *(str(path) for path in test_paths)]
    result = subprocess.run(command, cwd=ROOT, env=env,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, encoding="utf-8", errors="replace")
    receipt = {"command": command, "return_code": int(result.returncode),
               "test_source_sha256": {path.name: _sha256_file(path) for path in test_paths},
               "output": result.stdout[-20000:]}
    if result.returncode != 0:
        raise R8Error(f"focused R9 CPU tests failed:\n{result.stdout[-8000:]}")
    return receipt


def _child_environment(stage: str) -> dict[str, str]:
    from experiments.calibration import r3_fullstream_2x2 as r3
    return r3._child_environment(dict(os.environ), stage,
                                 AUTHORIZED_CUDA_VISIBILITY if stage == "STAGE_A" else None)


def _run_supervised(stage: str, output: Path, *, category: str | None = None,
                    order: str | None = None, mode: str | None = None,
                    smoke_events: int = 0) -> dict[str, Any]:
    from experiments.calibration import r1_runtime_supervisor as supervisor
    env = _child_environment("STAGE_A" if stage in {"SMOKE", "STAGE_A"} else "STAGE_B")
    env["R9_CHILD_OUTPUT"] = str(output)
    env["R8_CHILD_OUTPUT"] = str(output)
    command = [str(Path(sys.executable)), "-m", "experiments.calibration.r9_correction_mechanism_falsification"]
    if stage in {"SMOKE", "STAGE_A"}:
        command.extend(["--stage-a-child", "--category", str(category),
                        "--order", str(order), "--mode", str(mode),
                        "--smoke-events", str(smoke_events)])
        timeout = SMOKE_DEADLINE_SECONDS if stage == "SMOKE" else FULL_DEADLINE_SECONDS
    elif stage == "STAGE_B":
        command.append("--stage-b-child")
        timeout = FULL_DEADLINE_SECONDS
    else:
        raise R8Error(f"unsupported supervised R8 stage: {stage}")
    result = supervisor.run_supervised_child(
        command, timeout_seconds=timeout, env=env, cwd=ROOT)
    record = asdict(result)
    if (record.get("child_exit_code") != 0 or record.get("status") != "COMPLETED" or
            not record.get("termination_confirmed") or record.get("active_job_processes") != 0):
        raise R8Error(f"supervised R9 {stage} child failed: {record}")
    return {"stage": stage, "category": category, "order": order,
            "result": record,
            "process_tree_containment": "Windows Job Object, suspended child assignment, kill-on-close",
            "automatic_restart": False}


def _create_leaf(path: Path) -> Any:
    _ensure_no_reparse(path.parent, OUTPUT_BASE)
    if path.exists():
        raise R8Error(f"refusing to reuse existing R8 output directory: {path}")
    budget = _budget(path, create=True)
    return budget


def _write_leaf_manifest(path: Path, record: Mapping[str, Any]) -> None:
    budget = _create_leaf(path)
    budget.write_json("run_manifest.json", dict(record))


def _verify_condition_output(path: Path, category: str, order: str,
                             expected_count: int, expected_query_positions: Sequence[int]
                             ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    required = ("run_manifest.json", "stage_a_scores.jsonl",
                "attempt_history.jsonl", "stage_a_seal.json")
    if any(not (path / name).is_file() for name in required):
        raise R8Error(f"{category}/{order}: complete Stage A child artifacts are missing")
    manifest = _read_json(path / "run_manifest.json")
    seal = _read_json(path / "stage_a_seal.json")
    rows = _read_jsonl(path / "stage_a_scores.jsonl")
    run_manifest = _read_json(RUN_NAMESPACE / "manifest" / "run_manifest.json")
    run_identity = run_manifest["identity"]
    order_identity = run_identity["classes"][category]["orders"].get(order)
    if order_identity is None:
        raise R8Error(f"{category}/{order}: output names an unregistered identity order")
    actual_identity_map = _condition_identity_rows(rows)
    query_rows = [row for row in rows if row.get("query_tape_decision") is True]
    query_positions = [int(row.get("position", -1)) for row in query_rows]
    query_groups = [str(row.get("source_group_sha256", "")) for row in query_rows]
    identity_ok = (manifest.get("mode") == "SMOKE" or
        _sha256_json(actual_identity_map) == order_identity.get("identity_map_sha256"))
    query_identity_ok = (manifest.get("mode") == "SMOKE" or
        seal.get("query_identity_set_sha256") ==
        _sha256_json(sorted(str(row.get("image_identity_sha256")) for row in query_rows)))
    if (manifest.get("run_id") != RUN_ID or
            manifest.get("identity_sha256") != run_identity.get("identity_sha256") or
            manifest.get("runner_source_sha256") != run_identity.get("runner_source_sha256") or
            manifest.get("category") != category or manifest.get("order") != order or
            seal.get("category") != category or seal.get("order") != order or
            seal.get("event_count") != expected_count or len(rows) != expected_count or
            seal.get("query_positions") != list(expected_query_positions) or
            query_positions != list(expected_query_positions) or
            len(query_groups) != len(set(query_groups)) or not identity_ok or
            not query_identity_ok or
            seal.get("score_rows_sha256") != _sha256_file(path / "stage_a_scores.jsonl") or
            any(row.get("category") != category or row.get("order") != order or
                int(row.get("position", -1)) != i or "label" in row or "anomaly_class" in row
                for i, row in enumerate(rows))):
        raise R8Error(f"{category}/{order}: Stage A seal or score rows fail identity checks")
    return rows, seal


def _compare_smoke(category: str, path: Path,
                   r7_rows: Sequence[Mapping[str, Any]],
                   r8_rows: Sequence[Mapping[str, Any]],
                   expected_query_positions: Sequence[int]) -> dict[str, Any]:
    rows, completion = _verify_condition_output(
        path, category, "SHUFFLE0", SMOKE_EVENTS,
        [p for p in expected_query_positions if p < SMOKE_EVENTS])
    parity_r7 = _compare_r7_rows(category, rows, r7_rows[:SMOKE_EVENTS])
    parity_r8 = _compare_r8_rows(category, rows, r8_rows[:SMOKE_EVENTS])
    if (completion.get("queried_normal", 0) <= 0 or
            completion.get("queried_defect", 0) <= 0 or
            completion.get("greedy_state_changes", 0) <= 0 or
            completion.get("c1_state_changes", 0) <= 0 or
            completion.get("simple_memory_insertions", 0) <= 0 or
            completion.get("simple_memory_evictions", 0) <= 0 or
            any(len(completion.get("b3_threshold_history", {}).get(s, [])) < 2
                for s in SCORERS)):
        raise R8Error("R9 smoke missed queried signs, C1/Greedy/SignedExemplar updates, FIFO eviction, or B3 refresh")
    if any(not all(name in row.get("scores", {}) for name in
                   ("BetaOnly17", "BetaOnly53", "BetaOnly97")) for row in rows):
        raise R8Error("R9 smoke omitted a predeclared beta-only control")
    return {"status": "PASS", "category": category, "order": "SHUFFLE0",
            "event_count": len(rows), "queried_normal": completion["queried_normal"],
            "queried_defect": completion["queried_defect"],
            "greedy_state_changes": completion["greedy_state_changes"],
            "correction_state_changes": completion["c1_state_changes"],
            "c64_rescore_costs": completion["c64_rescore_costs"],
            "r7_parity": parity_r7, "r8_parity": parity_r8,
            "control_scores_registered": True,
            "smoke_score_rows_sha256": completion["score_rows_sha256"]}


def _write_public_deliverables(aggregate: Mapping[str, Any],
                               source_sha: str, test_sha: str,
                               manifest: Mapping[str, Any],
                               stage_a: Mapping[str, Any],
                               stage_b_seal: Mapping[str, Any]) -> None:
    import numpy as np
    REPORT_DIR.mkdir(parents=True, exist_ok=False)
    h1_values = [float(aggregate["category_results"][c]["primary_contrasts"][
        "capacity_SHUFFLE0"]) for c in CATEGORIES]
    h2_values = [float(aggregate["category_results"][c]["primary_contrasts"][
        "order_C1024"]) for c in CATEGORIES]
    h3_scalar = [float(aggregate["category_results"][c]["orders"]["SHUFFLE0"][
        "mechanism"]["full_minus_control"]["Scalar"]["auroc"]) for c in CATEGORIES]
    h3_address = {str(seed): [float(aggregate["category_results"][c]["orders"]["SHUFFLE0"][
        "mechanism"]["full_minus_control"][f"Address{seed}"]["auroc"]) for c in CATEGORIES]
        for seed in ADDRESS_SEEDS}
    def classify(values: Sequence[float], positive_all: bool = False) -> str:
        positive = sum(value > 0 for value in values)
        negative = sum(value < 0 for value in values)
        if positive_all and positive == len(values):
            return "SUPPORTED_WITHIN_DEVELOPMENT_CONDITIONS"
        if positive >= 3 and negative == 0:
            return "SUPPORTED_WITHIN_DEVELOPMENT_CONDITIONS"
        if positive and negative:
            return "MIXED"
        return "CHALLENGED"
    pressure_categories = []
    for category in CATEGORIES:
        pressure = False
        for order in ORDERS:
            order_result = aggregate["category_results"][category]["orders"][order]
            first_full = order_result["correction_first_capacity_reached_position"].get("GC256")
            replacement_count = sum(int(value) for key, value in
                                    order_result["correction_update_stats"]["GC256"].items()
                                    if "replac" in key.lower())
            pressure = pressure or first_full is not None or replacement_count > 0
        if pressure:
            pressure_categories.append(category)
    if not pressure_categories:
        h1_status = "NOT_IDENTIFIABLE"
    else:
        pressured_effects = [float(aggregate["category_results"][c]["primary_contrasts"][key])
                             for c in pressure_categories
                             for key in ("capacity_RAW", "capacity_SHUFFLE0")]
        h1_status = classify(pressured_effects)
    nonzero_order = [value for value in h2_values if abs(value) > 1e-12]
    if not nonzero_order:
        h2_status = "CHALLENGED"
    elif len(nonzero_order) >= 3 and all(value > 0 for value in nonzero_order) or (
            len(nonzero_order) >= 3 and all(value < 0 for value in nonzero_order)):
        h2_status = "SUPPORTED_WITHIN_DEVELOPMENT_CONDITIONS"
    else:
        h2_status = "MIXED"
    h3_status = ("SUPPORTED_WITHIN_DEVELOPMENT_CONDITIONS"
                 if all(v > 0 for v in h3_scalar) and
                 all(v > 0 for values in h3_address.values() for v in values)
                 else "CHALLENGED" if all(v <= 0 for v in h3_scalar) and
                 all(v <= 0 for values in h3_address.values() for v in values)
                 else "MIXED")
    lines = [
        "# R8 Unified Correction Bank Mechanism Attribution",
        "",
        "## 1. Research question and hypotheses",
        "",
        "R8 tests Correction capacity, feedback order, and feature-address association in one matched retrospective development diagnostic. H1 is capacity retention; H2 is feedback timing; H3 is feature/prototype address association beyond a state-level scalar shift.",
        "",
        "## 2. Frozen protocol and causal permissions",
        "",
        f"Run `{RUN_ID}` used the four previously exposed categories, one Support Seed 0, the fixed four-image Support plus separate 64-normal C64 commissioning set, shared Greedy K=1024, Correction K=256/1024, and identity-matched R7 Seed 37 queries. This is RETROSPECTIVE_DEVELOPMENT_DIAGNOSTIC, not blind or independent confirmation. Stage A sealed all eight full conditions before Stage B read image-level outcomes; online labels were disclosed only for registered queries after score fsync; masks were not opened.",
        "",
        "## 3. R7 numerical reproduction",
        "",
        f"The immutable R7 run was launched from parent `{manifest['identity']['r7_reference']['source_parent_commit']}` with runner SHA `{manifest['identity']['r7_reference']['launch_runner_source_sha256']}`. The R7 report records that only trailing blank lines were later removed before commit; the committed runner SHA is `{manifest['identity']['r7_reference']['committed_runner_source_sha256']}`. R8 did not rerun R7 and binds to its sealed event rows and identity receipt.",
        "",
        json.dumps(stage_a["r7_parity"], indent=2, sort_keys=True),
        "",
        "## 4. Full Capacity × Order results",
        "",
        "Per-category scorer ranking and controller confusion results are in `aggregate_results.json`; the public aggregate contains no event rows, paths, feature vectors, or label vectors.",
        "",
        "| Category | Order | G AUROC | GC256 AUROC | GC1024 AUROC | G AP | GC256 AP | GC1024 AP |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for category in CATEGORIES:
        for order in ORDERS:
            stats = aggregate["category_results"][category]["orders"][order]["main_scorers"]
            lines.append("| {} | {} | {} | {} | {} | {} | {} | {} |".format(
                category, order,
                *(f"{stats[s]['ranking']['auroc']:.6f}" for s in SCORERS),
                *(f"{stats[s]['ranking']['average_precision']:.6f}" for s in SCORERS)))
    lines += ["", "## 5. Capacity and Order main effects and interaction", "",
              "Per-category primary contrasts and the equal-category macro are recorded in the aggregate. Paired whole-Source-Group bootstrap intervals are descriptive and conditional on this fixed trajectory.", "",
              json.dumps(aggregate["equal_category_macro"]["primary_contrasts"], indent=2, sort_keys=True),
              "",
              json.dumps(aggregate["equal_category_macro"]["factorial_source_group_bootstrap"], indent=2, sort_keys=True),
              "", "## 6. Full versus Scalar and Address-shuffled results", "",
              json.dumps(aggregate["equal_category_macro"]["mechanism_ranking"], indent=2, sort_keys=True),
              "", "## 7. Early/middle/late feedback diagnostics", "",
              "Early, middle, and late image-level ranking metrics are reported for the main arms and for Full, Scalar, and all three Address controls. Cumulative query timing is summarized at the 25th, 50th, 75th, and 100th percentile of stream progress for each category and order.",
              "", "## 8. Calibration and FPR/FNR trade-offs", "",
              "Frozen C64 q90, B1 queried-normal rolling q90, and B3 current-state q90 results include confusion counts, BA, FPR, FNR, accuracy, coverage, and worst complete 512-event windows in the aggregate.",
              "", "## 9. Memory and compute costs", "",
              "Per-condition runtime, measured CPU RSS, peak CUDA allocation/reservation, Greedy occupancy/writes, C1 occupancy, and C64 rescoring counts/times are recorded in private seals and selected aggregate summaries.",
              "", "## 10. Hypotheses supported, challenged or unresolved", "",
              f"- H1: **{h1_status}**; category SHUFFLE0 capacity contrasts: `{h1_values}`; capacity-pressure categories: `{pressure_categories}`.",
              f"- H2: **{h2_status}**; category matched-query order contrasts: `{h2_values}`.",
              f"- H3: **{h3_status}**; Full-minus-Scalar: `{h3_scalar}`; Full-minus-Address AUROC by seed: `{h3_address}`.",
              "", "## 11. Limitations and interpretation boundaries", "",
              "There is one Support Seed and one Shuffle Seed. Source Group bootstrap is not independent trajectory uncertainty. The categories and development outcomes were previously exposed. Results do not establish broad seed robustness, blind confirmation, strict Gaussian locality, semantic correctness, conformal coverage, or factory-chronology generalization.",
              "", "## 12. Final recommendation for Correction Bank", "",
              "Treat Correction capacity and address controls as condition-dependent development evidence. Do not infer a universal preferred capacity from this single matched run; use the category effects, operating-point trade-offs, memory cost, and conditional intervals together.",
              "", "## 13. Source commits, hashes, runtime and artifacts", "",
              f"- Branch: `{EXPECTED_BRANCH}`; source parent: `{EXPECTED_PARENT}`; run ID: `{RUN_ID}`.",
              f"- Executed R8 runner SHA-256: `{source_sha}`.",
              f"- Focused test source SHA-256: `{test_sha}`.",
              f"- R7 score rows SHA-256: `{manifest['r7_reference_score_rows_sha256']}`.",
              f"- R8 Stage A seal SHA-256: `{_sha256_json(stage_a)}`; Stage B seal SHA-256: `{_sha256_json(stage_b_seal)}`.",
              f"- Private artifacts: `outputs/calibration/{RUN_ID}/`; public aggregate: `aggregate_results.json`.",
              "- Preserved prior attempts (none resumed or overwritten): " + "; ".join(
                  f"`{item['run_id']}` {item['status']}: {item['reason']}"
                  for item in PRESERVED_FAILED_ATTEMPTS) + ".",
              "- CUDA numerical equivalence is verified event by event against the sealed R7 G/GC1024 trajectories and Frozen/B1/B3 decisions.",
              ""]
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
    AGGREGATE_PATH.write_text(json.dumps(aggregate, indent=2, sort_keys=True,
                                         allow_nan=False) + "\n", encoding="utf-8")


def _run_experiment() -> dict[str, Any]:
    if Path(sys.executable).resolve() != Path(r"D:\anaconda3\envs\torch\python.exe").resolve():
        raise R8Error(f"R8 must use the authorized Python environment: {sys.executable}")
    _assert_repository_identity()
    if RUN_NAMESPACE.exists():
        raise R8Error(f"R8 namespace already exists; resume/overwrite forbidden: {RUN_NAMESPACE}")
    if REPORT_DIR.exists() or REPORT_PATH.exists() or AGGREGATE_PATH.exists():
        raise R8Error("R8 public deliverable path already exists; overwrite is forbidden")
    cpu_test_receipt = _run_focused_cpu_tests()
    free_bytes = _free_space(OUTPUT_BASE)
    _ensure_no_reparse(RUN_NAMESPACE, OUTPUT_BASE)
    receipts, contract, reference_rows = _r7_reference_and_contract()
    source_sha = _sha256_file(Path(__file__).resolve())
    sources = _source_fingerprints()
    identity = build_identity(sources, source_sha, receipts, contract)
    # Construct the complete new namespace with no resume path. Every file is
    # subsequently written through an allowlisted leaf OutputBudget.
    RUN_NAMESPACE.mkdir(parents=False, exist_ok=False)
    for folder in ("smoke", "full"):
        (RUN_NAMESPACE / folder).mkdir()
    manifest_leaf = _create_leaf(RUN_NAMESPACE / "manifest")
    manifest = _make_run_manifest(identity, source_sha, free_bytes, receipts,
                                  cpu_test_receipt)
    manifest_leaf.write_json("run_manifest.json", manifest)
    active_output: Path | None = None
    active_stage = "R8_PREFLIGHT"
    try:
        smoke_dir = RUN_NAMESPACE / "smoke" / "audiojack_shuffle0"
        active_output, active_stage = smoke_dir, "R8_SMOKE"
        _write_leaf_manifest(smoke_dir, {"schema": "r8.condition.manifest.v1",
                                         "run_id": RUN_ID, "category": SMOKE_CATEGORY,
                                         "order": SMOKE_ORDER, "mode": "SMOKE",
                                         "identity_sha256": identity["identity_sha256"],
                                         "runner_source_sha256": source_sha})
        _free_space(OUTPUT_BASE)
        smoke_supervisor = _run_supervised(
            "SMOKE", smoke_dir, category=SMOKE_CATEGORY, order=SMOKE_ORDER,
            mode="SMOKE", smoke_events=SMOKE_EVENTS)
        smoke_budget = _budget(smoke_dir)
        smoke_budget.write_json("supervisor_status.json", smoke_supervisor)
        smoke_receipt = _compare_smoke(
            SMOKE_CATEGORY, smoke_dir, reference_rows[SMOKE_CATEGORY],
            identity["classes"][SMOKE_CATEGORY]["orders"][SMOKE_ORDER]["query_positions"])
        _create_leaf(RUN_NAMESPACE / "smoke" / "seal")
        smoke_seal = {"schema": "r8.smoke.seal.v1", "status": "PASS",
                      "run_id": RUN_ID, "smoke": smoke_receipt,
                      "child_supervisor": smoke_supervisor}
        _budget(RUN_NAMESPACE / "smoke" / "seal").write_json("stage_a_seal.json", smoke_seal)
        active_output = None

        full_root = RUN_NAMESPACE / "full"
        condition_receipts: dict[str, Any] = {}
        supervisor_rows: dict[str, Any] = {}
        parity: dict[str, Any] = {}
        runtimes: dict[str, Any] = {}
        for category in CATEGORIES:
            for order in ORDERS:
                key = f"{category}/{order}"
                leaf = _condition_output(full_root, category, order)
                active_output, active_stage = leaf, f"R8_STAGE_A:{key}"
                _write_leaf_manifest(leaf, {"schema": "r8.condition.manifest.v1",
                                             "run_id": RUN_ID, "category": category,
                                             "order": order, "mode": "FULL",
                                             "identity_sha256": identity["identity_sha256"],
                                             "runner_source_sha256": source_sha})
                _free_space(OUTPUT_BASE)
                supervised = _run_supervised(
                    "STAGE_A", leaf, category=category, order=order,
                    mode="FULL", smoke_events=0)
                child_budget = _budget(leaf)
                child_budget.write_json("supervisor_status.json", supervised)
                qpos = identity["classes"][category]["orders"][order]["query_positions"]
                rows, completion = _verify_condition_output(
                    leaf, category, order, EXPECTED_COUNTS[category], qpos)
                condition_receipts[key] = {
                    "score_rows_sha256": completion["score_rows_sha256"],
                    "query_count": completion["query_count"],
                    "query_positions_sha256": _sha256_json(qpos),
                    "query_identity_set_sha256": completion["query_identity_set_sha256"],
                    "event_count": completion["event_count"],
                    "completion_sha256": _sha256_file(leaf / "stage_a_seal.json"),
                }
                runtimes[key] = completion["timing_seconds"]
                supervisor_rows[key] = supervised
                active_output = None
                if order == "SHUFFLE0":
                    if len(rows) != len(reference_rows[category]):
                        raise R8Error("R8 full shuffled trajectory does not cover all R7 reference events")
                    parity[category] = _compare_r7_rows(category, rows, reference_rows[category])
        if len(condition_receipts) != 8:
            raise R8Error("R8 full Stage A did not execute all eight registered conditions")
        stage_a_seal = {
            "schema": "r8.stage_a.global.seal.v1", "status": "SEALED_COMPLETE",
            "run_id": RUN_ID, "identity_sha256": identity["identity_sha256"],
            "conditions": condition_receipts, "r7_parity": parity,
            "condition_runtimes": runtimes,
            "condition_supervisors": supervisor_rows,
            "event_count": sum(2 * EXPECTED_COUNTS[c] for c in CATEGORIES),
            "condition_count": 8,
            "query_count": sum(2 * EXPECTED_QUERIES[c] for c in CATEGORIES),
            "all_shuffled_g_gc1024_event_parity_verified": True,
            "all_six_frozen_b1_b3_decisions_verified": True,
            "unqueried_test_labels_accessed": False, "test_masks_opened": False,
            "offline_outcome_vectors_accessed": False,
            "automatic_restart": False,
            "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
        seal_leaf = _create_leaf(full_root / "stage_seals")
        active_output, active_stage = full_root / "stage_seals", "R8_STAGE_B"
        seal_leaf.write_json("stage_a_seal.json", stage_a_seal)
        stage_b_supervisor = _run_supervised("STAGE_B", full_root / "stage_seals")
        stage_b_budget = _budget(full_root / "stage_seals")
        stage_b_budget.write_json("supervisor_status.json", stage_b_supervisor)
        aggregate = _read_json(full_root / "stage_seals" / "aggregate_results.json")
        stage_b_seal = _read_json(full_root / "stage_seals" / "stage_b_seal.json")
        if (stage_b_seal.get("status") != "SEALED_COMPLETE" or
                stage_b_seal.get("stage_a_seal_sha256") !=
                _sha256_file(full_root / "stage_seals" / "stage_a_seal.json")):
            raise R8Error("R8 Stage B seal does not bind the complete Stage A seal")
        test_file = ROOT / "tests" / "calibration" / "test_r8_correction_mechanism_attribution.py"
        _write_public_deliverables(aggregate, source_sha, _sha256_file(test_file),
                                   manifest, stage_a_seal, stage_b_seal)
        active_output = None
        return {"status": "COMPLETE", "run_id": RUN_ID,
                "identity_sha256": identity["identity_sha256"],
                "stage_a_seal_sha256": _sha256_file(full_root / "stage_seals" / "stage_a_seal.json"),
                "stage_b_seal_sha256": _sha256_file(full_root / "stage_seals" / "stage_b_seal.json"),
                "public_aggregate_sha256": _sha256_file(AGGREGATE_PATH),
                "source_sha256": source_sha}
    except BaseException as exc:
        if active_output is not None and active_output.is_dir():
            _write_failure(active_output, active_stage, exc)
        _write_failure(RUN_NAMESPACE / "manifest", "R8_PARENT", exc)
        raise


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true", help="execute the single authorized R9 run")
    parser.add_argument("--stage-a-child", action="store_true")
    parser.add_argument("--stage-b-child", action="store_true")
    parser.add_argument("--category", choices=CATEGORIES)
    parser.add_argument("--order", choices=ORDERS + ("SUPPORT1_SHUFFLE0",))
    parser.add_argument("--mode", choices=("SMOKE", "FULL"))
    parser.add_argument("--smoke-events", type=int, default=0)
    args = parser.parse_args(argv)
    try:
        if args.stage_a_child:
            if args.category is None or args.order is None or args.mode is None:
                raise R8Error("Stage A child requires category, order and mode")
            _stage_a_child(args.category, args.order, args.mode, args.smoke_events)
            return 0
        if args.stage_b_child:
            output = Path(os.environ.get("R8_CHILD_OUTPUT", ""))
            try:
                _stage_b_evaluate()
            except BaseException as exc:
                _write_failure(output, "STAGE_B_CHILD", exc)
                raise
            return 0
        if args.run:
            print(json.dumps(_run_experiment(), indent=2, sort_keys=True))
            return 0
        parser.error("Choose --run, --stage-a-child or --stage-b-child")
    except BaseException as exc:
        print(f"R8_ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise
    return 2


def _metric_binary(labels: Any, predictions: Any) -> dict[str, Any]:
    import numpy as np
    y = np.asarray(labels, dtype=np.int8)
    p = np.asarray(predictions, dtype=np.int8)
    if y.shape != p.shape or y.ndim != 1 or np.any((y != 0) & (y != 1)) or np.any((p != 0) & (p != 1)):
        raise R8Error("binary metric inputs are malformed")
    tn = int(np.sum((y == 0) & (p == 0)))
    fp = int(np.sum((y == 0) & (p == 1)))
    fn = int(np.sum((y == 1) & (p == 0)))
    tp = int(np.sum((y == 1) & (p == 1)))
    fpr = fp / (fp + tn) if fp + tn else None
    fnr = fn / (fn + tp) if fn + tp else None
    tnr = tn / (tn + fp) if tn + fp else None
    tpr = tp / (tp + fn) if tp + fn else None
    return {"tn": tn, "fp": fp, "fn": fn, "tp": tp,
            "fpr": fpr, "fnr": fnr,
            "balanced_accuracy": ((tpr + tnr) / 2 if tpr is not None and tnr is not None else None),
            "accuracy": (tn + tp) / max(1, len(y)),
            "prediction_coverage": 1.0}


def _ranking_metrics(labels: Any, scores: Any) -> dict[str, float | None]:
    import numpy as np
    from sklearn.metrics import average_precision_score, roc_auc_score
    y = np.asarray(labels, dtype=np.int8)
    s = np.asarray(scores, dtype=np.float64)
    if y.ndim != 1 or s.shape != y.shape or not np.isfinite(s).all():
        raise R8Error("ranking metric inputs are malformed")
    if len(np.unique(y)) != 2:
        return {"auroc": None, "average_precision": None}
    return {"auroc": float(roc_auc_score(y, s)),
            "average_precision": float(average_precision_score(y, s))}


def _source_group_bootstrap(labels: Any, scores_a: Any, scores_b: Any,
                            groups: Sequence[str], seed: int) -> dict[str, Any]:
    import numpy as np
    from sklearn.metrics import average_precision_score, roc_auc_score
    y = np.asarray(labels, dtype=np.int8)
    a = np.asarray(scores_a, dtype=np.float64)
    b = np.asarray(scores_b, dtype=np.float64)
    g = np.asarray(groups, dtype=object)
    if y.shape != a.shape or y.shape != b.shape or y.shape != g.shape:
        raise R8Error("paired Source Group bootstrap arrays are misaligned")
    unique = np.unique(g)
    rng = np.random.default_rng(seed)
    auc_delta, ap_delta = [], []
    for _ in range(BOOTSTRAP_REPLICATES):
        draw = rng.choice(unique, size=len(unique), replace=True)
        counts = {value: int(np.sum(draw == value)) for value in unique}
        weights = np.asarray([counts[value] for value in g], dtype=np.float64)
        active = weights > 0
        if len(np.unique(y[active])) < 2:
            continue
        auc_delta.append(float(roc_auc_score(y[active], a[active], sample_weight=weights[active]) -
                               roc_auc_score(y[active], b[active], sample_weight=weights[active])))
        ap_delta.append(float(average_precision_score(y[active], a[active], sample_weight=weights[active]) -
                              average_precision_score(y[active], b[active], sample_weight=weights[active])))
    return {"method": "paired whole-Source-Group bootstrap, descriptive conditional on this trajectory",
            "replicates_requested": BOOTSTRAP_REPLICATES,
            "valid_replicates": len(auc_delta),
            "auroc_difference_ci_95_percentile": (
                [float(np.quantile(auc_delta, .025)), float(np.quantile(auc_delta, .975))]
                if auc_delta else None),
            "average_precision_difference_ci_95_percentile": (
                [float(np.quantile(ap_delta, .025)), float(np.quantile(ap_delta, .975))]
                if ap_delta else None)}


def _factorial_source_group_bootstrap(labels_by_order: Mapping[str, Any],
                                      rows_by_order: Mapping[str, Sequence[Mapping[str, Any]]],
                                      seed: int) -> tuple[dict[str, Any], dict[str, Any]]:
    """Paired group resampling for all factorial contrasts on one category."""
    import numpy as np
    from sklearn.metrics import roc_auc_score
    groups = {order: np.asarray([str(row["source_group_sha256"])
                                 for row in rows_by_order[order]], dtype=object)
              for order in ORDERS}
    unique_by_order = {order: set(groups[order].tolist()) for order in ORDERS}
    if unique_by_order["RAW"] != unique_by_order["SHUFFLE0"]:
        raise R8Error("Source Group bootstrap cannot pair RAW and SHUFFLE0 populations")
    unique = np.asarray(sorted(unique_by_order["RAW"]), dtype=object)
    rng = np.random.default_rng(seed)
    names = ("D_RAW_256", "D_RAW_1024", "D_SHUFFLE0_256", "D_SHUFFLE0_1024",
             "capacity_RAW", "capacity_SHUFFLE0", "order_C256", "order_C1024",
             "capacity_by_order_interaction")
    draws = {name: np.full(BOOTSTRAP_REPLICATES, np.nan, dtype=np.float64) for name in names}
    for replicate in range(BOOTSTRAP_REPLICATES):
        selected_groups = rng.choice(unique, size=len(unique), replace=True)
        counts = {group: int(np.sum(selected_groups == group)) for group in unique}
        auc = {}
        valid = True
        for order in ORDERS:
            row_groups = groups[order]
            weights = np.asarray([counts[group] for group in row_groups], dtype=np.float64)
            active = weights > 0
            labels = np.asarray(labels_by_order[order], dtype=np.int8)
            if len(np.unique(labels[active])) < 2:
                valid = False
                break
            rows = rows_by_order[order]
            auc[order] = {}
            for scorer in SCORERS:
                values = np.asarray([float(row["scores"][scorer]) for row in rows], dtype=np.float64)
                auc[order][scorer] = float(roc_auc_score(
                    labels[active], values[active], sample_weight=weights[active]))
        if not valid:
            continue
        d_raw_256 = auc["RAW"]["GC256"] - auc["RAW"]["G1024"]
        d_raw_1024 = auc["RAW"]["GC1024"] - auc["RAW"]["G1024"]
        d_shuffle_256 = auc["SHUFFLE0"]["GC256"] - auc["SHUFFLE0"]["G1024"]
        d_shuffle_1024 = auc["SHUFFLE0"]["GC1024"] - auc["SHUFFLE0"]["G1024"]
        values = {"D_RAW_256": d_raw_256, "D_RAW_1024": d_raw_1024,
                  "D_SHUFFLE0_256": d_shuffle_256, "D_SHUFFLE0_1024": d_shuffle_1024,
                  "capacity_RAW": d_raw_1024 - d_raw_256,
                  "capacity_SHUFFLE0": d_shuffle_1024 - d_shuffle_256,
                  "order_C256": d_shuffle_256 - d_raw_256,
                  "order_C1024": d_shuffle_1024 - d_raw_1024,
                  "capacity_by_order_interaction": (d_shuffle_1024 - d_shuffle_256) -
                                                   (d_raw_1024 - d_raw_256)}
        for name, value in values.items():
            draws[name][replicate] = value
    summaries = {}
    for name, values in draws.items():
        finite = values[np.isfinite(values)]
        summaries[name] = {"valid_replicates": int(len(finite)),
                           "auroc_difference_ci_95_percentile": (
                               [float(np.quantile(finite, .025)), float(np.quantile(finite, .975))]
                               if len(finite) else None)}
    return summaries, draws


def _controller_and_windows(labels: Any, rows: Sequence[Mapping[str, Any]],
                            scorer: str) -> dict[str, Any]:
    import numpy as np
    y = np.asarray(labels, dtype=np.int8)
    scores = np.asarray([float(row["scores"][scorer]) for row in rows], dtype=np.float64)
    predictions = {controller: np.asarray([
        int(row["decisions"][f"{scorer}-{controller}"]["prediction"]) for row in rows], dtype=np.int8)
        for controller in CONTROLLERS}
    thresholds = {controller: [float(row["decisions"][f"{scorer}-{controller}"]["threshold"])
                               for row in rows] for controller in CONTROLLERS}
    windows = []
    # The registered worst-window definition is every complete 512-event
    # window with stride one, matching R7's window audit.
    for start in range(0, len(y) - 511):
        end = start + 512
        metric_by_controller = {controller: _metric_binary(y[start:end], pred[start:end])
                                for controller, pred in predictions.items()}
        windows.append({"start": start, "end_exclusive": end,
                        "balanced_accuracy_by_controller": {
                            name: value["balanced_accuracy"]
                            for name, value in metric_by_controller.items()}})
    worst = {controller: min((row["balanced_accuracy_by_controller"][controller]
                              for row in windows
                              if row["balanced_accuracy_by_controller"][controller] is not None),
                             default=None) for controller in CONTROLLERS}
    thirds = {"early": (0, len(y)//3), "middle": (len(y)//3, 2*len(y)//3),
              "late": (2*len(y)//3, len(y))}
    phase = {name: _ranking_metrics(y[a:b], scores[a:b]) for name, (a, b) in thirds.items()}
    return {"ranking": _ranking_metrics(y, scores),
            "decisions": {controller: _metric_binary(y, pred)
                          for controller, pred in predictions.items()},
            "threshold_summary": {controller: {
                "mean": float(np.mean(value)), "minimum": float(min(value)),
                "maximum": float(max(value)), "movements": int(np.sum(np.diff(value) != 0))}
                for controller, value in thresholds.items()},
            "worst_complete_512_window_balanced_accuracy": worst,
            "complete_512_event_windows": len(windows),
            "early_middle_late_ranking": phase,
            "window_diagnostics": windows}


def _stage_b_mechanism_summary(order: str, labels: Sequence[int],
                               rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Summarize mechanism controls only for their predeclared SHUFFLE0 scope."""
    import numpy as np

    if order != "SHUFFLE0":
        return {"status": "NOT_APPLICABLE_BY_PREDECLARED_DESIGN"}
    names = ("GC1024", "Scalar", *(f"Address{s}" for s in ADDRESS_SEEDS))
    scores: dict[str, Any] = {}
    for name in names:
        if any(name not in row.get("scores", {}) for row in rows):
            raise R8Error(f"R8 SHUFFLE0 Stage B rows are missing registered mechanism score {name}")
        scores[name] = np.asarray([float(row["scores"][name]) for row in rows], dtype=np.float64)
    mechanism = {name: _ranking_metrics(labels, values)
                 for name, values in scores.items()}
    phase_bounds = {"early": (0, len(labels)//3),
                    "middle": (len(labels)//3, 2*len(labels)//3),
                    "late": (2*len(labels)//3, len(labels))}
    mechanism["early_middle_late"] = {
        name: {phase: _ranking_metrics(labels[start:end], values[start:end])
               for phase, (start, end) in phase_bounds.items()}
        for name, values in scores.items()}
    mechanism["full_minus_control"] = {
        name: {"auroc": (mechanism["GC1024"]["auroc"] - value["auroc"]
                         if mechanism["GC1024"]["auroc"] is not None and value["auroc"] is not None else None),
               "average_precision": (mechanism["GC1024"]["average_precision"] - value["average_precision"]
                                     if mechanism["GC1024"]["average_precision"] is not None and value["average_precision"] is not None else None)}
        for name, value in mechanism.items()
        if name in ("Scalar", *(f"Address{s}" for s in ADDRESS_SEEDS))}
    mechanism["address_seed_descriptive_mean"] = {
        metric: float(np.mean([mechanism[f"Address{s}"][metric] for s in ADDRESS_SEEDS]))
        for metric in ("auroc", "average_precision")}
    mechanism["address_seed_descriptive_spread"] = {
        metric: float(np.ptp([mechanism[f"Address{s}"][metric] for s in ADDRESS_SEEDS]))
        for metric in ("auroc", "average_precision")}
    return mechanism


def _stage_b_evaluate() -> dict[str, Any]:
    import numpy as np
    import torch
    from experiments.calibration import r3_fullstream_2x2 as r3
    if torch.cuda.device_count() != 0 or torch.cuda.is_initialized():
        raise R8Error("R8 Stage B must run with CUDA hidden and uninitialized")
    manifest = _read_json(RUN_NAMESPACE / "manifest" / "run_manifest.json")
    identity = manifest["identity"]
    full = RUN_NAMESPACE / "full"
    seal_dir = full / "stage_seals"
    _ensure_no_reparse(seal_dir, RUN_NAMESPACE)
    if (identity.get("runner_source_sha256") != _sha256_file(Path(__file__).resolve()) or
            identity.get("source_fingerprints") != _source_fingerprints()):
        raise R8Error("R8 Stage B source identity differs from the sealed Stage A execution")
    _assert_repository_identity()
    stage_a = _read_json(seal_dir / "stage_a_seal.json")
    if (stage_a.get("status") != "SEALED_COMPLETE" or
            stage_a.get("identity_sha256") != identity.get("identity_sha256") or
            len(stage_a.get("conditions", {})) != 8):
        raise R8Error("R8 Stage B refused labels: all eight full Stage A conditions are not sealed")
    contract = __import__("experiments.calibration.r7_shuffled_correction_4class",
                          fromlist=["_frozen_contract"])._frozen_contract()
    # All eight child score files and their identities are verified before the
    # first TEST outcome projection is opened.
    rows_by_condition: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for category in CATEGORIES:
        for order in ORDERS:
            path = _condition_output(full, category, order)
            expected_q = identity["classes"][category]["orders"][order]["query_positions"]
            rows, completion = _verify_condition_output(
                path, category, order, EXPECTED_COUNTS[category], expected_q)
            if (completion.get("score_rows_sha256") != stage_a["conditions"][f"{category}/{order}"][
                    "score_rows_sha256"] or
                    completion.get("query_identity_set_sha256") != identity["classes"][category][
                    "r7_query_identity_set_sha256"]):
                raise R8Error("R8 Stage B condition completion does not match the global seal")
            feedback = _read_jsonl(path / "attempt_history.jsonl")
            disclosed = [row for row in feedback
                         if row.get("schema") == "r8.private.feedback.v1"]
            query_rows = [row for row in rows if row.get("query_tape_decision")]
            if len(disclosed) != len(query_rows):
                raise R8Error("R8 query feedback count differs from sealed score tape")
            by_position = {int(row["position"]): row for row in disclosed}
            if set(by_position) != {int(row["position"]) for row in query_rows}:
                raise R8Error("R8 private feedback positions differ from query decisions")
            rows_by_condition[(category, order)] = rows

    category_results: dict[str, Any] = {}
    source_label_counts: dict[str, Any] = {}
    bootstrap_seed_receipts: dict[str, Any] = {}
    factorial_draws_by_category: dict[str, Any] = {}
    for cat_index, category in enumerate(CATEGORIES):
        class_info = contract["metadata"][category]
        original_labels = np.asarray(r3._read_stage_b_label_projection(
            class_info["json_path"], EXPECTED_COUNTS[category]), dtype=np.int8)
        order_results = {}
        labels_by_order = {}
        for order in ORDERS:
            rows = rows_by_condition[(category, order)]
            events = make_order_events(class_info["events"], order)
            labels = np.asarray(map_outcomes_to_order(original_labels, events), dtype=np.int8)
            labels_by_order[order] = labels
            source_groups = [str(row["source_group_sha256"]) for row in rows]
            if len(source_groups) != len(labels):
                raise R8Error("R8 Stage B score and label populations differ")
            queried = [row for row in rows if row.get("query_tape_decision")]
            feedback_by_position = {int(row["position"]): row for row in _read_jsonl(
                _condition_output(full, category, order) / "attempt_history.jsonl")
                if row.get("schema") == "r8.private.feedback.v1"}
            for row in queried:
                position = int(row["position"])
                if int(feedback_by_position[position]["label"]) != int(labels[position]):
                    raise R8Error("R8 queried feedback label disagrees with post-seal projection")
            main = {scorer: _controller_and_windows(labels, rows, scorer)
                    for scorer in SCORERS}
            mechanism = _stage_b_mechanism_summary(order, labels, rows)
            order_results[order] = {"main_scorers": main, "mechanism": mechanism,
                                    "query_count": len(queried),
                                    "query_feedback_normal": int(sum(labels[int(row["position"])] == 0 for row in queried)),
                                    "query_feedback_defect": int(sum(labels[int(row["position"])] == 1 for row in queried)),
                                    "query_timing_fraction_quartiles": _read_json(
                                        _condition_output(full, category, order) / "stage_a_seal.json")[
                                            "query_timing_fraction_quartiles"],
                                    "greedy_memory_final_occupancy": _read_json(
                                        _condition_output(full, category, order) / "stage_a_seal.json")["greedy_occupancy"],
                                    "correction_final_occupancy": _read_json(
                                        _condition_output(full, category, order) / "stage_a_seal.json")["correction_final_occupancy"],
                                    "c64_rescore_costs": _read_json(
                                        _condition_output(full, category, order) / "stage_a_seal.json")["c64_rescore_costs"],
                                    "correction_delta_summary": _read_json(
                                        _condition_output(full, category, order) / "stage_a_seal.json")["correction_delta_summary"],
                                    "scalar_offset_summary": _read_json(
                                        _condition_output(full, category, order) / "stage_a_seal.json")["scalar_offset_summary"],
                                    "correction_update_stats": _read_json(
                                        _condition_output(full, category, order) / "stage_a_seal.json")["correction_update_stats"],
                                    "correction_first_capacity_reached_position": _read_json(
                                        _condition_output(full, category, order) / "stage_a_seal.json")["correction_first_capacity_reached_position"],
                                    "mutable_memory_bytes": _read_json(
                                        _condition_output(full, category, order) / "stage_a_seal.json")["mutable_memory_bytes"],
                                    "resources": {key: _read_json(
                                        _condition_output(full, category, order) / "stage_a_seal.json")[key]
                                        for key in ("timing_seconds", "peak_cuda_allocated_bytes",
                                                    "peak_cuda_reserved_bytes", "peak_cpu_rss_bytes")}}
        # Matched identity set is checked again against labels after Stage A.
        if not np.array_equal(labels_by_order["RAW"][np.argsort([e.original_position for e in make_order_events(
                class_info["events"], "RAW")])], labels_by_order["SHUFFLE0"][np.argsort(
                    [e.original_position for e in make_order_events(class_info["events"], "SHUFFLE0")])]):
            raise R8Error("R8 Stage B raw/shuffle label projection mapping failed")
        auroc = {order: {scorer: order_results[order]["main_scorers"][scorer]["ranking"]["auroc"]
                         for scorer in SCORERS} for order in ORDERS}
        ap = {order: {scorer: order_results[order]["main_scorers"][scorer]["ranking"]["average_precision"]
                      for scorer in SCORERS} for order in ORDERS}
        contrasts = {}
        for order in ORDERS:
            d256 = auroc[order]["GC256"] - auroc[order]["G1024"]
            d1024 = auroc[order]["GC1024"] - auroc[order]["G1024"]
            contrasts[f"D_{order}_256"] = d256
            contrasts[f"D_{order}_1024"] = d1024
        contrasts["capacity_RAW"] = contrasts["D_RAW_1024"] - contrasts["D_RAW_256"]
        contrasts["capacity_SHUFFLE0"] = contrasts["D_SHUFFLE0_1024"] - contrasts["D_SHUFFLE0_256"]
        contrasts["order_C256"] = contrasts["D_SHUFFLE0_256"] - contrasts["D_RAW_256"]
        contrasts["order_C1024"] = contrasts["D_SHUFFLE0_1024"] - contrasts["D_RAW_1024"]
        contrasts["capacity_by_order_interaction"] = contrasts["capacity_SHUFFLE0"] - contrasts["capacity_RAW"]
        groups_by_order = {
            order: [str(row["source_group_sha256"]) for row in rows_by_condition[(category, order)]]
            for order in ORDERS}
        bootstrap = {
            key: _source_group_bootstrap(
                labels_by_order[order],
                [float(row["scores"][arm_a]) for row in rows_by_condition[(category, order)]],
                [float(row["scores"][arm_b]) for row in rows_by_condition[(category, order)]],
                groups_by_order[order], BOOTSTRAP_SEED + cat_index * 1009 + index)
            for index, (key, order, arm_a, arm_b) in enumerate((
                ("D_RAW_256", "RAW", "GC256", "G1024"),
                ("D_RAW_1024", "RAW", "GC1024", "G1024"),
                ("D_SHUFFLE0_256", "SHUFFLE0", "GC256", "G1024"),
                ("D_SHUFFLE0_1024", "SHUFFLE0", "GC1024", "G1024")))}
        address_bootstrap = {}
        shuffle_rows = rows_by_condition[(category, "SHUFFLE0")]
        for seed in ADDRESS_SEEDS:
            address_bootstrap[str(seed)] = _source_group_bootstrap(
                labels_by_order["SHUFFLE0"],
                [float(row["scores"]["GC1024"]) for row in shuffle_rows],
                [float(row["scores"][f"Address{seed}"]) for row in shuffle_rows],
                groups_by_order["SHUFFLE0"], BOOTSTRAP_SEED + cat_index * 101 + seed)
        full_scalar_bootstrap = _source_group_bootstrap(
            labels_by_order["SHUFFLE0"],
            [float(row["scores"]["GC1024"]) for row in shuffle_rows],
            [float(row["scores"]["Scalar"]) for row in shuffle_rows],
            groups_by_order["SHUFFLE0"], BOOTSTRAP_SEED + cat_index * 65537)
        factorial_summary, factorial_draws = _factorial_source_group_bootstrap(
            labels_by_order,
            {order: rows_by_condition[(category, order)] for order in ORDERS},
            BOOTSTRAP_SEED + cat_index * 1009)
        factorial_draws_by_category[category] = factorial_draws
        category_results[category] = {
            "event_count": EXPECTED_COUNTS[category],
            "orders": order_results, "primary_auroc": auroc,
            "average_precision": ap, "primary_contrasts": contrasts,
            "factorial_source_group_bootstrap": factorial_summary,
            "source_group_bootstrap": bootstrap,
            "mechanism_source_group_bootstrap": {
                "Full_minus_Scalar": full_scalar_bootstrap,
                "Full_minus_Address": address_bootstrap},
            "outcome_count": {"normal": int(np.sum(labels_by_order["RAW"] == 0)),
                              "defect": int(np.sum(labels_by_order["RAW"] == 1))}}
        source_label_counts[category] = category_results[category]["outcome_count"]
        bootstrap_seed_receipts[category] = {"primary": BOOTSTRAP_SEED + cat_index * 1009,
                                             "mechanism": BOOTSTRAP_SEED + cat_index * 101}
    # Macro effects are equal-category averages, never pooled image metrics.
    macro_contrasts = {key: float(np.mean([category_results[c]["primary_contrasts"][key]
                                           for c in CATEGORIES]))
                       for key in next(iter(category_results.values()))["primary_contrasts"]}
    macro_bootstrap = {}
    for contrast in next(iter(factorial_draws_by_category.values())):
        stack = np.stack([factorial_draws_by_category[c][contrast] for c in CATEGORIES], axis=0)
        valid = np.isfinite(stack).all(axis=0)
        macro_draws = stack[:, valid].mean(axis=0)
        macro_bootstrap[contrast] = {
            "valid_replicates": int(valid.sum()),
            "auroc_difference_ci_95_percentile": (
                [float(np.quantile(macro_draws, .025)), float(np.quantile(macro_draws, .975))]
                if len(macro_draws) else None)}
    macro_main = {}
    for order in ORDERS:
        macro_main[order] = {scorer: {
            metric: float(np.mean([category_results[c]["orders"][order]["main_scorers"][scorer]["ranking"][metric]
                                   for c in CATEGORIES]))
            for metric in ("auroc", "average_precision")}
            for scorer in SCORERS}
    macro_mechanism = {name: {
        metric: float(np.mean([category_results[c]["orders"]["SHUFFLE0"]["mechanism"][name][metric]
                               for c in CATEGORIES]))
        for metric in ("auroc", "average_precision")}
        for name in ("GC1024", "Scalar", *(f"Address{s}" for s in ADDRESS_SEEDS))}
    aggregate = {
        "schema": "r8.public.aggregate.v1", "run_id": RUN_ID,
        "classification": "RETROSPECTIVE_DEVELOPMENT_DIAGNOSTIC",
        "identity_sha256": identity["identity_sha256"],
        "source_parent_commit": EXPECTED_PARENT,
        "stage_a_seal_sha256": _sha256_file(seal_dir / "stage_a_seal.json"),
        "r7_reference_score_rows_sha256": manifest["r7_reference_score_rows_sha256"],
        "category_results": category_results,
        "equal_category_macro": {"main_ranking": macro_main,
                                  "primary_contrasts": macro_contrasts,
                                  "factorial_source_group_bootstrap": macro_bootstrap,
                                  "mechanism_ranking": macro_mechanism},
        "bootstrap": {"method": "paired whole physical Source Groups within each fixed trajectory",
                      "interpretation": "descriptive conditional-on-trajectory; not independent training/seed uncertainty",
                      "replicates": BOOTSTRAP_REPLICATES,
                      "seed": BOOTSTRAP_SEED,
                      "category_seed_receipts": bootstrap_seed_receipts},
        "interpretation_limits": {"independent_confirmation": False,
                                  "blinded_evaluation": False,
                                  "generalization_to_factory_chronology": False,
                                  "only_four_previously_exposed_development_categories": True,
                                  "one_support_seed_and_one_shuffle_seed": True,
                                  "masks_opened": False},
        "stage_a_information_flow": {"all_eight_full_conditions_sealed_before_stage_b": True,
                                     "unqueried_test_labels_in_stage_a": False,
                                     "queried_feedback_labels_checked_after_stage_b": True,
                                     "masks_opened": False},
        "preserved_prior_attempts": list(PRESERVED_FAILED_ATTEMPTS),
        "r7_parity": stage_a["r7_parity"],
        "runtimes": stage_a["condition_runtimes"],
        "private_outcome_count_receipt_sha256": _sha256_json(source_label_counts),
        "public_payload_excludes_event_rows_paths_features_and_label_vectors": True}
    _budget(seal_dir).write_json("aggregate_results.json", aggregate)
    stage_b = {"schema": "r8.stage_b.seal.v1", "status": "SEALED_COMPLETE",
               "run_id": RUN_ID,
               "stage_a_seal_sha256": _sha256_file(seal_dir / "stage_a_seal.json"),
               "aggregate_results_sha256": _sha256_file(seal_dir / "aggregate_results.json"),
               "all_test_outcomes_opened_after_global_stage_a_seal": True,
               "projection": "image-level anomaly_class only; no masks",
               "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    _budget(seal_dir).write_json("stage_b_seal.json", stage_b)
    _budget(seal_dir).write_json("complete_seal.json", {
        "schema": "r8.complete.seal.v1", "status": "COMPLETE", "run_id": RUN_ID,
        "stage_a_seal_sha256": stage_b["stage_a_seal_sha256"],
        "stage_b_seal_sha256": _sha256_file(seal_dir / "stage_b_seal.json"),
        "aggregate_results_sha256": stage_b["aggregate_results_sha256"]})
    return aggregate


def _r9_contrasts(labels: Sequence[int], rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Compute registered paired ranking effects on one sealed trajectory."""
    ranks = {name: _ranking_metrics(labels, [float(row["scores"][name]) for row in rows])
             for name in ("G1024", "GC1024", "SignedExemplar1024",
                          *(f"BetaOnly{s}" for s in ADDRESS_SEEDS))}
    pairs = {"Full_minus_G": ("GC1024", "G1024"),
             "SignedExemplar_minus_G": ("SignedExemplar1024", "G1024"),
             "Full_minus_SignedExemplar": ("GC1024", "SignedExemplar1024"),
             **{f"Full_minus_BetaOnly{s}": ("GC1024", f"BetaOnly{s}")
                for s in ADDRESS_SEEDS}}
    contrasts = {}
    for name, (left, right) in pairs.items():
        contrasts[name] = {
            metric: (None if ranks[left][metric] is None or ranks[right][metric] is None
                     else float(ranks[left][metric] - ranks[right][metric]))
            for metric in ("auroc", "average_precision")}
    return {"ranking": ranks, "paired_contrasts": contrasts}


def _r9_source_group_bootstrap(labels: Sequence[int], scores_a: Sequence[float],
                               scores_b: Sequence[float], groups: Sequence[str],
                               seed: int) -> dict[str, Any]:
    """Efficient paired bootstrap that resamples complete physical Source Groups."""
    import numpy as np
    from sklearn.metrics import average_precision_score, roc_auc_score
    y = np.asarray(labels, dtype=np.int8)
    a = np.asarray(scores_a, dtype=np.float64)
    b = np.asarray(scores_b, dtype=np.float64)
    g = np.asarray(groups, dtype=object)
    if y.ndim != 1 or y.shape != a.shape or y.shape != b.shape or y.shape != g.shape:
        raise R9Error("R9 paired Source Group bootstrap arrays are misaligned")
    unique, inverse = np.unique(g, return_inverse=True)
    if len(unique) < 2 or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise R9Error("R9 Source Group bootstrap lacks finite scores or multiple groups")
    rng = np.random.default_rng(int(seed))
    auc_delta: list[float] = []
    ap_delta: list[float] = []
    for _ in range(BOOTSTRAP_REPLICATES):
        group_counts = np.bincount(rng.integers(0, len(unique), size=len(unique)),
                                   minlength=len(unique))
        weights = group_counts[inverse].astype(np.float64, copy=False)
        active = weights > 0
        if len(np.unique(y[active])) < 2:
            continue
        auc_delta.append(float(roc_auc_score(y[active], a[active],
            sample_weight=weights[active]) - roc_auc_score(y[active], b[active],
            sample_weight=weights[active])))
        ap_delta.append(float(average_precision_score(y[active], a[active],
            sample_weight=weights[active]) - average_precision_score(y[active], b[active],
            sample_weight=weights[active])))
    return {"method": "paired whole-Source-Group bootstrap, conditional on this fixed trajectory",
        "replicates_requested": BOOTSTRAP_REPLICATES, "valid_replicates": len(auc_delta),
        "auroc_difference_ci_95_percentile": ([float(np.quantile(auc_delta, .025)),
            float(np.quantile(auc_delta, .975))] if auc_delta else None),
        "average_precision_difference_ci_95_percentile": ([float(np.quantile(ap_delta, .025)),
            float(np.quantile(ap_delta, .975))] if ap_delta else None)}


def _r9_stage_b_evaluate() -> dict[str, Any]:
    """R9 Stage B. No TEST outcomes are opened until every Stage A seal verifies."""
    import numpy as np
    import torch
    from experiments.calibration import r3_fullstream_2x2 as r3
    from experiments.calibration import r7_shuffled_correction_4class as r7
    from experiments.calibration.r1_runtime_supervisor import current_process_is_in_job

    if torch.cuda.device_count() != 0 or torch.cuda.is_initialized():
        raise R9Error("R9 Stage B must run CPU-only with CUDA hidden and uninitialized")
    if not current_process_is_in_job():
        raise R9Error("R9 Stage B child is outside the Windows Job Object")
    _assert_repository_identity()
    manifest = _read_json(RUN_NAMESPACE / "manifest" / "run_manifest.json")
    identity = manifest.get("identity", {})
    seal_dir = RUN_NAMESPACE / "full" / "stage_seals"
    _ensure_no_reparse(seal_dir, RUN_NAMESPACE)
    if (identity.get("runner_source_sha256") != _sha256_file(Path(__file__).resolve()) or
            identity.get("source_fingerprints") != _source_fingerprints()):
        raise R9Error("R9 Stage B source identity differs from the sealed execution")
    stage_a = _read_json(seal_dir / "stage_a_seal.json")
    registered_orders = tuple(identity["configuration"]["orders"])
    expected_keys = {f"{category}/{order}" for category in CATEGORIES
                     for order in registered_orders}
    if (stage_a.get("schema") != "r9.stage_a.global.seal.v1" or
            stage_a.get("status") != "SEALED_COMPLETE" or
            stage_a.get("identity_sha256") != identity.get("identity_sha256") or
            set(stage_a.get("conditions", {})) != expected_keys or
            stage_a.get("unqueried_test_labels_accessed") is not False or
            stage_a.get("test_masks_opened") is not False):
        raise R9Error("R9 Stage B refused outcomes: every registered Stage A condition is not sealed")

    # First pass verifies every sealed score trajectory without reading private
    # feedback labels or projecting any TEST outcome.
    rows_by_condition: dict[tuple[str, str], list[dict[str, Any]]] = {}
    completions: dict[tuple[str, str], dict[str, Any]] = {}
    for category in CATEGORIES:
        for order in registered_orders:
            expected = identity["classes"][category]["orders"][order]
            leaf = _condition_output(RUN_NAMESPACE / "full", category, order)
            rows, completion = _verify_condition_output(
                leaf, category, order, EXPECTED_COUNTS[category], expected["query_positions"])
            receipt = stage_a["conditions"][f"{category}/{order}"]
            if (completion.get("status") != "SEALED_COMPLETE" or
                    completion.get("support_seed") != expected["support_seed"] or
                    completion.get("stream_shuffle_seed") != expected["stream_shuffle_seed"] or
                    completion.get("score_rows_sha256") != receipt.get("score_rows_sha256") or
                    completion.get("query_identity_set_sha256") !=
                    identity["classes"][category]["r7_query_identity_set_sha256"] or
                    len(rows) != EXPECTED_COUNTS[category] or
                    any(not set(("G1024", "GC1024", "SignedExemplar1024",
                                 "BetaOnly17", "BetaOnly53", "BetaOnly97")) <=
                            set(row.get("scores", {})) for row in rows)):
                raise R9Error(f"R9 Stage B seal verification failed: {category}/{order}")
            rows_by_condition[(category, order)] = rows
            completions[(category, order)] = completion

    # Only after the complete seal gate above do we load label projections and
    # private queried feedback for offline checks.
    contract = r7._frozen_contract()
    category_results: dict[str, Any] = {}
    condition_private_labels: dict[tuple[str, str], list[int]] = {}
    bootstrap_seed_index = 0
    for cat_index, category in enumerate(CATEGORIES):
        class_info = contract["metadata"][category]
        original_labels = np.asarray(r3._read_stage_b_label_projection(
            class_info["json_path"], EXPECTED_COUNTS[category]), dtype=np.int8)
        condition_results = {}
        for order_index, order in enumerate(registered_orders):
            rows = rows_by_condition[(category, order)]
            events = make_order_events(class_info["events"], order)
            labels = np.asarray(map_outcomes_to_order(original_labels, events), dtype=np.int8)
            if len(labels) != len(rows):
                raise R9Error(f"R9 Stage B score and outcome populations differ: {category}/{order}")
            condition_private_labels[(category, order)] = labels.tolist()
            leaf = _condition_output(RUN_NAMESPACE / "full", category, order)
            feedback = [row for row in _read_jsonl(leaf / "attempt_history.jsonl")
                        if row.get("schema") == "r9.private.feedback.v1"]
            query_positions = [int(row["position"]) for row in rows
                               if row.get("query_tape_decision")]
            feedback_by_position = {int(row["position"]): row for row in feedback}
            if (len(feedback) != len(query_positions) or
                    set(feedback_by_position) != set(query_positions) or
                    any(int(feedback_by_position[pos]["label"]) != int(labels[pos])
                        for pos in query_positions)):
                raise R9Error(f"R9 queried feedback does not match the post-seal label projection: {category}/{order}")

            summary = _r9_contrasts(labels, rows)
            phase_bounds = {"early": (0, len(labels)//3),
                            "middle": (len(labels)//3, 2*len(labels)//3),
                            "late": (2*len(labels)//3, len(labels))}
            summary["early_middle_late_ranking"] = {
                scorer: {phase: _ranking_metrics(labels[a:b], [
                    float(row["scores"][scorer]) for row in rows[a:b]])
                    for phase, (a, b) in phase_bounds.items()}
                for scorer in summary["ranking"]}
            deployment = {}
            for scorer in SCORERS:
                detail = _controller_and_windows(labels, rows, scorer)
                detail.pop("window_diagnostics", None)
                deployment[scorer] = detail

            groups = [str(row["source_group_sha256"]) for row in rows]
            paired_bootstrap = {}
            pairs = (("Full_minus_G", "GC1024", "G1024"),
                     ("SignedExemplar_minus_G", "SignedExemplar1024", "G1024"),
                     ("Full_minus_SignedExemplar", "GC1024", "SignedExemplar1024"),
                     *((f"Full_minus_BetaOnly{s}", "GC1024", f"BetaOnly{s}")
                       for s in ADDRESS_SEEDS))
            for contrast_index, (name, left, right) in enumerate(pairs):
                paired_bootstrap[name] = _r9_source_group_bootstrap(
                    labels, [float(row["scores"][left]) for row in rows],
                    [float(row["scores"][right]) for row in rows], groups,
                    BOOTSTRAP_SEED + cat_index * 100_003 + order_index * 1009 + contrast_index)
            completion = completions[(category, order)]
            score_vector_sha = {scorer: _sha256_json([
                float(row["scores"][scorer]) for row in rows])
                for scorer in summary["ranking"]}
            condition_results[order] = {
                "support_seed": int(completion["support_seed"]),
                "stream_shuffle_seed": int(completion["stream_shuffle_seed"]),
                "event_count": len(rows),
                "query_count": int(completion["query_count"]),
                "query_feedback_normal": int(sum(labels[pos] == 0 for pos in query_positions)),
                "query_feedback_defect": int(sum(labels[pos] == 1 for pos in query_positions)),
                "outcome_count": {"normal": int(np.sum(labels == 0)),
                                  "defect": int(np.sum(labels == 1))},
                "ranking_and_contrasts": summary,
                "deployment_metrics": deployment,
                "source_group_paired_bootstrap": paired_bootstrap,
                "resource_and_state_receipt": _resource_and_state_receipt(completion),
                "score_vector_sha256": score_vector_sha,
                "score_rows_sha256": completion["score_rows_sha256"],
                "stage_a_seal_sha256": _sha256_file(leaf / "stage_a_seal.json"),
            }
            bootstrap_seed_index += 1
        category_results[category] = {"event_count": EXPECTED_COUNTS[category],
                                      "conditions": condition_results}

    def macro_for_order(order: str) -> dict[str, Any]:
        return {scorer: {metric: float(np.mean([
            category_results[c]["conditions"][order]["ranking_and_contrasts"]["ranking"][scorer][metric]
            for c in CATEGORIES])) for metric in ("auroc", "average_precision")}
            for scorer in ("G1024", "GC1024", "SignedExemplar1024",
                           *(f"BetaOnly{s}" for s in ADDRESS_SEEDS))}

    macro_by_order = {order: macro_for_order(order) for order in registered_orders}
    mandatory = list(MANDATORY_ORDERS)
    contrast_names = ("Full_minus_G", "SignedExemplar_minus_G", "Full_minus_SignedExemplar",
                      *(f"Full_minus_BetaOnly{s}" for s in ADDRESS_SEEDS))
    robustness: dict[str, Any] = {}
    for contrast in contrast_names:
        per_category = {}
        macro_effects = []
        for category in CATEGORIES:
            effects = {order: category_results[category]["conditions"][order][
                "ranking_and_contrasts"]["paired_contrasts"][contrast]["auroc"]
                for order in mandatory}
            values = list(effects.values())
            per_category[category] = {"by_shuffle_seed": effects,
                "mean": float(np.mean(values)), "minimum": float(np.min(values)),
                "maximum": float(np.max(values)),
                "positive_count": sum(value > 0 for value in values),
                "negative_count": sum(value < 0 for value in values),
                "zero_count": sum(value == 0 for value in values)}
            macro_effects.append(float(np.mean(values)))
        robustness[contrast] = {"per_category": per_category,
            "equal_category_mean_by_shuffle_seed": {
                order: float(np.mean([category_results[c]["conditions"][order][
                    "ranking_and_contrasts"]["paired_contrasts"][contrast]["auroc"]
                    for c in CATEGORIES])) for order in mandatory},
            "mean_across_category_seed_pairs": float(np.mean(macro_effects)),
            "range_of_category_seed_means": [float(np.min(macro_effects)), float(np.max(macro_effects))],
            "positive_category_count": sum(value > 0 for value in macro_effects),
            "negative_category_count": sum(value < 0 for value in macro_effects)}

    beta_effects = [category_results[c]["conditions"][order]["ranking_and_contrasts"][
        "paired_contrasts"][f"Full_minus_BetaOnly{s}"]["auroc"]
        for c in CATEGORIES for order in mandatory for s in ADDRESS_SEEDS]
    simple_effects = [category_results[c]["conditions"][order]["ranking_and_contrasts"][
        "paired_contrasts"]["Full_minus_SignedExemplar"]["auroc"]
        for c in CATEGORIES for order in mandatory]
    all_full_control_effects = simple_effects + beta_effects
    simple_vs_g = [category_results[c]["conditions"][order]["ranking_and_contrasts"][
        "paired_contrasts"]["SignedExemplar_minus_G"]["auroc"]
        for c in CATEGORIES for order in mandatory]

    def sign_status(values: Sequence[float]) -> str:
        if values and all(value > 0 for value in values):
            return "SUPPORTED_WITHIN_DEVELOPMENT_CONDITIONS"
        if values and all(value <= 0 for value in values):
            return "CHALLENGED"
        return "MIXED"

    support1_sensitivity = {}
    if "SUPPORT1_SHUFFLE0" in registered_orders:
        for category in CATEGORIES:
            a = category_results[category]["conditions"]["SHUFFLE0"]["ranking_and_contrasts"]["ranking"]
            b = category_results[category]["conditions"]["SUPPORT1_SHUFFLE0"]["ranking_and_contrasts"]["ranking"]
            support1_sensitivity[category] = {scorer: {
                metric: float(b[scorer][metric] - a[scorer][metric])
                for metric in ("auroc", "average_precision")}
                for scorer in ("G1024", "GC1024", "SignedExemplar1024")}
    stage_b_payload = {
        "schema": "r9.public.aggregate.v1", "run_id": RUN_ID,
        "classification": "RETROSPECTIVE_DEVELOPMENT_DIAGNOSTIC",
        "identity_sha256": identity["identity_sha256"],
        "source_parent_commit": EXPECTED_PARENT,
        "registered_conditions": registered_orders,
        "condition_count": len(expected_keys),
        "total_scored_events": sum(EXPECTED_COUNTS[c] * len(registered_orders) for c in CATEGORIES),
        "category_results": category_results,
        "equal_category_macro_ranking_by_order": macro_by_order,
        "contrast_stream_robustness": robustness,
        "support1_sensitivity_vs_support0_seed0": support1_sensitivity,
        "support1_status": ("EXECUTED" if "SUPPORT1_SHUFFLE0" in registered_orders
                            else "NOT_RUN_INPUT_UNVERIFIED"),
        "hypothesis_assessments": {
            "H9_A_beta_address_association": sign_status(beta_effects),
            "H9_B_complexity_necessity_vs_signed_exemplar": sign_status(simple_effects),
            "H9_C_stream_order_robustness": sign_status(all_full_control_effects),
        },
        "effect_direction_counts": {
            "full_minus_beta_only_auroc": {"positive": sum(v > 0 for v in beta_effects),
                "negative": sum(v < 0 for v in beta_effects), "total": len(beta_effects)},
            "full_minus_signed_exemplar_auroc": {"positive": sum(v > 0 for v in simple_effects),
                "negative": sum(v < 0 for v in simple_effects), "total": len(simple_effects)},
            "signed_exemplar_minus_g_auroc": {"positive": sum(v > 0 for v in simple_vs_g),
                "negative": sum(v < 0 for v in simple_vs_g), "total": len(simple_vs_g)}},
        "bootstrap": {"method": "paired whole physical Source Group resampling within each fixed trajectory",
            "interpretation": "descriptive conditional-on-trajectory; not independent category, support-seed, or training-run uncertainty",
            "replicates_requested": BOOTSTRAP_REPLICATES, "seed": BOOTSTRAP_SEED},
        "stage_a_information_flow": {"all_registered_conditions_sealed_before_stage_b": True,
            "unqueried_test_labels_in_stage_a": False,
            "queried_feedback_labels_checked_after_global_seal": True,
            "test_masks_opened": False, "offline_outcomes_opened_only_in_stage_b": True},
        "interpretation_limits": {"independent_confirmation": False,
            "blinded_evaluation": False, "external_validation": False,
            "four_previously_exposed_development_categories_only": True,
            "shuffle_seeds_are_not_independent_datasets": True,
            "support_and_stream_seed_hierarchies_not_pooled": True,
            "masks_opened": False},
        "public_payload_excludes_event_rows_paths_features_and_label_vectors": True,
    }
    _budget(seal_dir).write_json("aggregate_results.json", stage_b_payload)
    stage_b_seal = {"schema": "r9.stage_b.seal.v1", "status": "SEALED_COMPLETE",
        "run_id": RUN_ID, "identity_sha256": identity["identity_sha256"],
        "stage_a_seal_sha256": _sha256_file(seal_dir / "stage_a_seal.json"),
        "aggregate_results_sha256": _sha256_file(seal_dir / "aggregate_results.json"),
        "conditions_evaluated": len(expected_keys),
        "all_test_outcomes_opened_after_global_stage_a_seal": True,
        "projection": "image-level anomaly_class only; no masks",
        "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    _budget(seal_dir).write_json("stage_b_seal.json", stage_b_seal)
    return stage_b_payload


def _independent_cpu_reconciliation(aggregate: Mapping[str, Any],
                                   stage_a: Mapping[str, Any]) -> dict[str, Any]:
    """Second CPU pass over sealed rows and post-seal labels; emits only metrics/hashes."""
    import numpy as np
    from sklearn.metrics import average_precision_score, roc_auc_score
    from experiments.calibration import r3_fullstream_2x2 as r3
    from experiments.calibration import r7_shuffled_correction_4class as r7

    contract = r7._frozen_contract()
    orders = tuple(aggregate["registered_conditions"])
    compared_rank_metrics = compared_confusion_cells = 0
    max_abs_delta = 0.0
    verified_score_rows = 0
    for category in CATEGORIES:
        original = np.asarray(r3._read_stage_b_label_projection(
            contract["metadata"][category]["json_path"], EXPECTED_COUNTS[category]), dtype=np.int8)
        for order in orders:
            leaf = _condition_output(RUN_NAMESPACE / "full", category, order)
            rows = _read_jsonl(leaf / "stage_a_scores.jsonl")
            completion = _read_json(leaf / "stage_a_seal.json")
            receipt = stage_a["conditions"][f"{category}/{order}"]
            if (_sha256_file(leaf / "stage_a_scores.jsonl") != receipt["score_rows_sha256"] or
                    completion["score_rows_sha256"] != receipt["score_rows_sha256"]):
                raise R9Error(f"Independent reconciliation found a score-file hash mismatch: {category}/{order}")
            labels = np.asarray(map_outcomes_to_order(original,
                make_order_events(contract["metadata"][category]["events"], order)), dtype=np.int8)
            public = aggregate["category_results"][category]["conditions"][order]
            for scorer, expected_hash in public["score_vector_sha256"].items():
                values = np.asarray([float(row["scores"][scorer]) for row in rows], dtype=np.float64)
                if _sha256_json(values.tolist()) != expected_hash:
                    raise R9Error(f"Independent reconciliation score vector digest mismatch: {category}/{order}/{scorer}")
                if len(np.unique(labels)) == 2:
                    actual = {"auroc": float(roc_auc_score(labels, values)),
                              "average_precision": float(average_precision_score(labels, values))}
                else:
                    actual = {"auroc": None, "average_precision": None}
                expected = public["ranking_and_contrasts"]["ranking"][scorer]
                for metric in actual:
                    if actual[metric] is None or expected[metric] is None:
                        if actual[metric] is not expected[metric]:
                            raise R9Error("Independent ranking metric nullability differs")
                    else:
                        delta = abs(actual[metric] - float(expected[metric]))
                        max_abs_delta = max(max_abs_delta, delta)
                        if delta > 1e-12:
                            raise R9Error(f"Independent ranking metric differs: {category}/{order}/{scorer}/{metric}")
                    compared_rank_metrics += 1
            for scorer in SCORERS:
                for controller in CONTROLLERS:
                    pred = np.asarray([int(row["decisions"][f"{scorer}-{controller}"]["prediction"])
                                       for row in rows], dtype=np.int8)
                    cells = {"tn": int(np.sum((labels == 0) & (pred == 0))),
                             "fp": int(np.sum((labels == 0) & (pred == 1))),
                             "fn": int(np.sum((labels == 1) & (pred == 0))),
                             "tp": int(np.sum((labels == 1) & (pred == 1)))}
                    expected = public["deployment_metrics"][scorer]["decisions"][controller]
                    if any(cells[key] != int(expected[key]) for key in cells):
                        raise R9Error(f"Independent confusion matrix differs: {category}/{order}/{scorer}/{controller}")
                    compared_confusion_cells += 4
            verified_score_rows += len(rows)
    return {"schema": "r9.independent.cpu.reconciliation.v1", "status": "PASS",
        "verified_score_rows": verified_score_rows,
        "ranking_metric_cells_compared": compared_rank_metrics,
        "confusion_matrix_cells_compared": compared_confusion_cells,
        "maximum_absolute_ranking_metric_difference": max_abs_delta,
        "all_condition_score_file_hashes_reverified": True,
        "all_score_vector_digests_reverified": True,
        "all_deployable_controller_confusion_matrices_recomputed": True,
        "outcome_vectors_written": False, "masks_opened": False}


def _stage_b_evaluate() -> dict[str, Any]:
    # Keep the historical function name used by the isolated child CLI, but
    # bind it exclusively to the R9 Stage B protocol.
    return _r9_stage_b_evaluate()


def _write_public_deliverables(aggregate: Mapping[str, Any], source_sha: str,
                               test_hashes: Mapping[str, str],
                               identity: Mapping[str, Any],
                               stage_a: Mapping[str, Any],
                               stage_b_seal: Mapping[str, Any],
                               reconciliation: Mapping[str, Any],
                               complete_seal: Mapping[str, Any]) -> None:
    """Write sanitized, reviewable R9 results without event-level content."""
    import statistics
    if REPORT_DIR.exists():
        raise R9Error(f"refusing to overwrite existing R9 report directory: {REPORT_DIR}")
    REPORT_DIR.mkdir(parents=True, exist_ok=False)
    lines = [
        "# R9 Unified Correction Mechanism Falsification and Resource-Matched Controls", "",
        "## 1. Objective and competing hypotheses", "",
        "R9 is a retrospective development diagnostic on four previously exposed Real-IAD categories. H9-A tests whether Full C1's beta-to-feature association contributes beyond a beta-only address permutation. H9-B tests whether Full C1 adds ranking value over a bounded signed nearest-exemplar memory under the same queried feedback and comparable allocated external memory. H9-C tests sensitivity to three fixed stream group-shuffle seeds. This is not blinded, independent, or external confirmation.", "",
        "## 2. Historical R7/R8 evidence and provenance", "",
        f"R7 parent: `{identity['r7_reference']['source_parent_commit']}`; run `{R7_RUN_ID}`; sealed score rows SHA-256 `{identity['r7_reference']['score_rows_sha256']}`.",
        f"R8 run: `{R8_RUN_ID}`; manifest SHA-256 `{identity['r8_reference']['manifest_sha256']}`; Stage A seal SHA-256 `{identity['r8_reference']['stage_a_seal_sha256']}`; Stage B seal SHA-256 `{identity['r8_reference']['stage_b_seal_sha256']}`.",
        "R7/R8 outcomes are historical development evidence and were previously exposed. The current experiment did not use unqueried outcomes in online scoring.", "",
        "## 3. Registered experimental contract", "",
        f"Run `{RUN_ID}` used Support Seed 0 with identity-matched Shuffle Seeds 0, 11, and 23 across all four registered categories. Support Seed 1 was `{aggregate['support1_status']}`. The four-image normal Support bank and additional 64 protected C64 normals are distinct; this is not strict four-shot-total. The exact R7-selected query image identities were remapped to each arrival order. Seeds 11 and 23 are stream-order interventions, not independent Q-A-TS query-policy runs.",
        "Each condition used one image encoding, native G/Top-32 scoring, one shared C0-controlled Greedy writer, native Full C1, read-only beta-only controls, and a separate fixed 512-positive/512-negative FIFO Signed Exemplar memory. All scores were sealed before queried-label disclosure; Stage B outcomes were opened only after the complete Stage A condition set sealed. No masks were opened.", "",
        "## 4. Baseline numerical parity", "",
        f"Smoke status: `{stage_a['smoke']['status']}` over {stage_a['smoke']['event_count']} events; it exercised normal/defect queried feedback, Full C1 updates, common Greedy updates, Signed Exemplar FIFO eviction, beta-only controls, and B3 threshold refresh.",
        "Full Seed-0 parity against both sealed R7 and R8 is recorded below. Errors are measured over every event; tolerances were absolute 1e-5 / relative 1e-4 for scores, and exact query identities, predictions, and 1e-12 thresholds.",
        "| Category | Events | Reference | G max abs error | Full max abs error | Query mismatches | Decision mismatches |",
        "|---|---:|---|---:|---:|---:|---:|",
    ]
    for category in CATEGORIES:
        for reference in ("R7", "R8"):
            parity = stage_a["parity"][category][reference]
            lines.append(f"| {category} | {parity['event_count']} | {reference} | {parity['maximum_absolute_score_error']['G1024']:.9g} | {parity['maximum_absolute_score_error']['GC1024']:.9g} | {parity['query_mismatches']} | {parity['decision_mismatches']} |")
    lines += ["", "## 5. Beta-only association results", "",
        "The registered counterfactual filters same-source and causal anchors first, then permutes only beta over eligible slots. Prototypes, radius, kappa, counts, and timestamps remain unchanged. AUROC/AP are reported for all three local permutation seeds; no seed was selected after evaluation.",
        "| Category | Support | Shuffle | G AUROC/AP | Full AUROC/AP | Beta17 AUROC/AP | Beta53 AUROC/AP | Beta97 AUROC/AP |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for category in CATEGORIES:
        for order in aggregate["registered_conditions"]:
            result = aggregate["category_results"][category]["conditions"][order]
            ranks = result["ranking_and_contrasts"]["ranking"]
            fmt = lambda name: f"{ranks[name]['auroc']:.5f}/{ranks[name]['average_precision']:.5f}"
            lines.append(f"| {category} | {result['support_seed']} | {result['stream_shuffle_seed']} | {fmt('G1024')} | {fmt('GC1024')} | {fmt('BetaOnly17')} | {fmt('BetaOnly53')} | {fmt('BetaOnly97')} |")
    lines += ["", "| Category | Support | Shuffle | Full−Beta17 AUROC/AP | Full−Beta53 AUROC/AP | Full−Beta97 AUROC/AP |", "|---|---:|---:|---:|---:|---:|"]
    for category in CATEGORIES:
        for order in aggregate["registered_conditions"]:
            result = aggregate["category_results"][category]["conditions"][order]
            deltas = result["ranking_and_contrasts"]["paired_contrasts"]
            lines.append(f"| {category} | {result['support_seed']} | {result['stream_shuffle_seed']} | " + " | ".join(
                f"{deltas[f'Full_minus_BetaOnly{s}']['auroc']:.5f}/{deltas[f'Full_minus_BetaOnly{s}']['average_precision']:.5f}"
                for s in ADDRESS_SEEDS) + " |")
    lines += ["", "## 6. Simple Signed Exemplar comparison", "",
        "Signed Exemplar uses only normalized queried Top-32 features, verified sign, same-source exclusion, fixed inherited radius/amplitude, and independent per-sign FIFO. It shares the common Greedy trajectory; its additional mutable memory is compared against Full C1's allocated prototype-plus-metadata budget.",
        "| Category | Support | Shuffle | G AUROC/AP | Full AUROC/AP | Signed Exemplar AUROC/AP | Full−Simple AUROC/AP | Simple−G AUROC/AP |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for category in CATEGORIES:
        for order in aggregate["registered_conditions"]:
            result = aggregate["category_results"][category]["conditions"][order]
            ranks = result["ranking_and_contrasts"]["ranking"]
            deltas = result["ranking_and_contrasts"]["paired_contrasts"]
            fmt = lambda name: f"{ranks[name]['auroc']:.5f}/{ranks[name]['average_precision']:.5f}"
            fmt_delta = lambda name: f"{deltas[name]['auroc']:.5f}/{deltas[name]['average_precision']:.5f}"
            lines.append(f"| {category} | {result['support_seed']} | {result['stream_shuffle_seed']} | {fmt('G1024')} | {fmt('GC1024')} | {fmt('SignedExemplar1024')} | {fmt_delta('Full_minus_SignedExemplar')} | {fmt_delta('SignedExemplar_minus_G')} |")
    lines += ["", "## 7. Memory and computational costs", "",
        "| Category | Support | Shuffle | Full C1 occupancy | Exemplar ± occupancy | Exemplar insertions / FIFO evictions | Full allocated bytes | Exemplar fixed-array bytes | Exemplar cap bytes | Python metadata bytes (separate) | Encode s | Score s | Update/rescore s | C64 rounds/calls (Full; Exemplar) | Peak CUDA allocated bytes |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for category in CATEGORIES:
        for order in aggregate["registered_conditions"]:
            result = aggregate["category_results"][category]["conditions"][order]
            res = result["resource_and_state_receipt"]
            memory = res["mutable_memory_bytes"]
            simple = memory["signed_exemplar"]
            full_c1 = memory["c1_native"]
            full_c1_bytes = (full_c1["correction_allocated_prototype_bytes"] +
                             full_c1["correction_allocated_metadata_bytes"])
            costs = res["c64_rescore_costs"]
            t = res["timing_seconds"]
            lines.append(f"| {category} | {result['support_seed']} | {result['stream_shuffle_seed']} | {res['c1_final_occupancy']} | {res['simple_memory_final_occupancy']['positive']} / {res['simple_memory_final_occupancy']['negative']} | {res['simple_memory_insertions']} / {res['simple_memory_evictions']} | {full_c1_bytes} | {simple['measured_fixed_array_bytes']} | {memory['signed_exemplar_cap_bytes']} | {simple['python_metadata_shallow_bytes_separately_reported']} | {t['encoding_seconds']:.2f} | {t['scoring_seconds']:.2f} | {t['update_and_rescore_seconds']:.2f} | {costs['GC1024']['rounds']}/{costs['GC1024']['score_calls']}; {costs['SignedExemplar1024']['rounds']}/{costs['SignedExemplar1024']['score_calls']} | {res['peak_cuda_allocated_bytes']} |")
    lines += ["", "## 8. Category-level and cross-seed consistency", "",
        "The tables above provide the full category × stream-seed matrix. The following paired AUROC contrast summaries show the mean, range, and directional consistency across the three registered stream orders; bootstrap intervals are Source Group-aware and conditional on each fixed trajectory.", ""]
    for contrast, summary in aggregate["contrast_stream_robustness"].items():
        lines += [f"### {contrast}", "", "| Category | Seed 0 | Seed 11 | Seed 23 | Mean | Range | Positive / Negative / Zero |", "|---|---:|---:|---:|---:|---:|---:|"]
        for category, values in summary["per_category"].items():
            by = values["by_shuffle_seed"]
            lines.append(f"| {category} | {by['SHUFFLE0']:.5f} | {by['SHUFFLE11']:.5f} | {by['SHUFFLE23']:.5f} | {values['mean']:.5f} | [{values['minimum']:.5f}, {values['maximum']:.5f}] | {values['positive_count']} / {values['negative_count']} / {values['zero_count']} |")
        lines.append("")
    lines += ["## 9. Frozen, B1, and B3 operating trade-offs", "",
        "For G, Full C1, and Signed Exemplar, every available Frozen C64 q90, queried-normal B1, and state-consistent B3 controller reports BA, FPR/FNR, TP/TN/FP/FN, accuracy, coverage, and the worst complete 512-event window with stride one.", "",
        "| Category | Support | Shuffle | Scorer | Controller | BA | FPR | FNR | TP/TN/FP/FN | Accuracy | Coverage | Worst 512-event BA |", "|---|---:|---:|---|---|---:|---:|---:|---|---:|---:|---:|"]
    for category in CATEGORIES:
        for order in aggregate["registered_conditions"]:
            result = aggregate["category_results"][category]["conditions"][order]
            for scorer in SCORERS:
                for controller in CONTROLLERS:
                    m = result["deployment_metrics"][scorer]["decisions"][controller]
                    worst = result["deployment_metrics"][scorer]["worst_complete_512_window_balanced_accuracy"][controller]
                    lines.append(f"| {category} | {result['support_seed']} | {result['stream_shuffle_seed']} | {scorer} | {controller} | {m['balanced_accuracy']:.5f} | {m['fpr'] if m['fpr'] is not None else 'NA'} | {m['fnr'] if m['fnr'] is not None else 'NA'} | {m['tp']}/{m['tn']}/{m['fp']}/{m['fn']} | {m['accuracy']:.5f} | {m['prediction_coverage']:.3f} | {worst if worst is not None else 'NA'} |")
    lines += ["", "## 10. Failure modes and negative results", "",
        f"The GPU smoke and all registered Stage A conditions sealed: `{stage_a['status']}`. The CPU-only Stage B seal is `{stage_b_seal['status']}`. The independent CPU reprocessing receipt is `{reconciliation['status']}`. Any adverse category, seed, operating-point, or cost result remains in the tables; no scorer was tuned after Stage B.", "",
        "## 11. What the mechanism evidence establishes", "",
        f"- H9-A beta-to-feature association: **{aggregate['hypothesis_assessments']['H9_A_beta_address_association']}**.",
        f"- H9-B Full C1 complexity versus Signed Exemplar: **{aggregate['hypothesis_assessments']['H9_B_complexity_necessity_vs_signed_exemplar']}**.",
        f"- H9-C stream-order robustness: **{aggregate['hypothesis_assessments']['H9_C_stream_order_robustness']}**.",
        "A Full-versus-beta-only difference is conditional on preserving radius/kappa association and does not isolate beta sign from magnitude. It does not establish strict locality or Gaussian optimality. Full-versus-Signed Exemplar compares mechanisms under matched queried feedback and a byte-capped additional memory; it does not equate Python object overhead or end-to-end compute.", "",
        "## 12. What remains unidentified", "",
        "This retrospective development diagnostic does not estimate independent training-run uncertainty, new support-draw robustness beyond at most one Support-1 sensitivity, or generalization beyond four exposed categories. Shuffle seeds and camera views are not independent datasets. Bootstrap intervals describe Source Group resampling conditional on each fixed trajectory. No independent confirmation, external validation, factory-chronology generalization, or production readiness is established. The task defines no minimum practically meaningful AUROC margin; reported direction and magnitude should therefore be considered alongside AP, FPR/FNR, cost, and memory rather than a post-hoc utility score.", "",
        "## 13. Is Full C1 complexity justified?", "",
        f"Registered sign-based assessment: `{aggregate['hypothesis_assessments']['H9_B_complexity_necessity_vs_signed_exemplar']}`. The experiment reports measured ranking, operating, memory, update, scoring, and C64-rescore differences. Practical necessity remains conditional on these development data and the absence of a preregistered minimum-effect threshold.", "",
        "## 14. Decision", "",
        f"**{aggregate['recommendation']['decision']}** — {aggregate['recommendation']['rationale']}", "",
        "## 15. Code, model, data, query, output, and seal provenance", "",
        f"- Baseline branch: `{EXPECTED_BRANCH}`; parent commit: `{EXPECTED_PARENT}`.",
        f"- R9 source SHA-256: `{source_sha}`; R9 identity SHA-256: `{identity['identity_sha256']}`.",
        f"- Global Stage A seal SHA-256: `{_sha256_file(RUN_NAMESPACE / 'full' / 'stage_seals' / 'stage_a_seal.json')}`.",
        f"- Stage B seal SHA-256: `{stage_b_seal['aggregate_results_sha256']}` aggregate; seal SHA-256 `{_sha256_file(RUN_NAMESPACE / 'full' / 'stage_seals' / 'stage_b_seal.json')}`.",
        f"- Independent CPU reconciliation: `{json.dumps(reconciliation, sort_keys=True)}`.",
        f"- Complete private seal SHA-256: `{_sha256_file(RUN_NAMESPACE / 'full' / 'stage_seals' / 'complete_seal.json')}`.",
        f"- Runtime/model fingerprints: `{json.dumps(aggregate['runtime_model_fingerprints'], sort_keys=True)}`.",
        f"- Focused CPU test source hashes: `{json.dumps(test_hashes, sort_keys=True)}`.",
        f"- Stage A condition count: {stage_a['condition_count']}; total events: {stage_a['event_count']}; queried events: {stage_a['query_count']}; per-condition output cap: {OUTPUT_LIMIT} bytes.",
        "- Outcome vectors, raw event score rows, sample paths, feature vectors, and masks are excluded from this public report and aggregate; they remain in the private untracked output namespace.",
        "- This is retrospective development evidence, not independent confirmation, external validation, or production readiness.", "",
    ]
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
    AGGREGATE_PATH.write_text(json.dumps(aggregate, indent=2, sort_keys=True,
                                         allow_nan=False) + "\n", encoding="utf-8")


def _run_experiment() -> dict[str, Any]:
    if Path(sys.executable).resolve() != Path(r"D:\anaconda3\envs\torch\python.exe").resolve():
        raise R9Error(f"R9 requires the authorized Python environment: {sys.executable}")
    _assert_repository_identity()
    if RUN_NAMESPACE.exists():
        raise R9Error(f"R9 namespace already exists; resume/overwrite forbidden: {RUN_NAMESPACE}")
    if REPORT_DIR.exists() or REPORT_PATH.exists() or AGGREGATE_PATH.exists():
        raise R9Error("R9 public deliverable path exists; overwrite is forbidden")

    cpu_test_receipt = _run_focused_cpu_tests()
    free_bytes = _free_space(OUTPUT_BASE)
    _ensure_no_reparse(RUN_NAMESPACE, OUTPUT_BASE)
    r7_receipts, contract, r7_rows = _r7_reference_and_contract()
    r8_receipts, r8_rows = _r8_reference_rows()
    try:
        support1_receipt: dict[str, Any] | None = verify_support1_frozen_inputs(contract)
        support1_status = "VERIFIED"
    except Exception as exc:
        support1_receipt = None
        support1_status = "NOT_RUN_INPUT_UNVERIFIED"
        support1_reason = f"{type(exc).__name__}: {str(exc)[:2000]}"
    else:
        support1_reason = None
    source_sha = _sha256_file(Path(__file__).resolve())
    sources = _source_fingerprints()
    identity = build_identity(sources, source_sha, r7_receipts, contract,
                              support1_receipt, r8_receipts)
    from experiments.calibration import r3_fullstream_2x2 as r3
    model = r3._get_verified_model_receipt()
    model_fingerprints = {key: model.get(key) for key in (
        "profile_id", "hub_source_tree_sha256", "checkpoint_sha256",
        "loaded_model_state_sha256")}
    RUN_NAMESPACE.mkdir(parents=False, exist_ok=False)
    (RUN_NAMESPACE / "smoke").mkdir()
    (RUN_NAMESPACE / "full").mkdir()
    manifest_leaf = _create_leaf(RUN_NAMESPACE / "manifest")
    manifest = _make_run_manifest(identity, source_sha, free_bytes, r7_receipts,
                                  cpu_test_receipt)
    manifest["support1_preflight"] = {"status": support1_status, "reason": support1_reason,
        "receipts": support1_receipt}
    manifest["verified_model_fingerprints"] = model_fingerprints
    manifest["r8_reference_score_row_hashes"] = {
        category: _sha256_file(OUTPUT_BASE / R8_RUN_ID / "full" /
                               f"{category}_shuffle0" / "stage_a_scores.jsonl")
        for category in CATEGORIES}
    manifest_leaf.write_json("run_manifest.json", manifest)

    active_output: Path | None = None
    active_stage = "R9_PREFLIGHT"
    try:
        smoke_dir = RUN_NAMESPACE / "smoke" / "audiojack_shuffle0"
        active_output, active_stage = smoke_dir, "R9_GPU_SMOKE"
        _write_leaf_manifest(smoke_dir, {"schema": "r9.condition.manifest.v1",
            "run_id": RUN_ID, "category": SMOKE_CATEGORY, "order": SMOKE_ORDER,
            "mode": "SMOKE", "identity_sha256": identity["identity_sha256"],
            "runner_source_sha256": source_sha})
        _free_space(OUTPUT_BASE)
        smoke_supervisor = _run_supervised("SMOKE", smoke_dir, category=SMOKE_CATEGORY,
            order=SMOKE_ORDER, mode="SMOKE", smoke_events=SMOKE_EVENTS)
        smoke_budget = _budget(smoke_dir)
        smoke_budget.write_json("supervisor_status.json", smoke_supervisor)
        qpos = identity["classes"][SMOKE_CATEGORY]["orders"][SMOKE_ORDER]["query_positions"]
        smoke = _compare_smoke(SMOKE_CATEGORY, smoke_dir, r7_rows[SMOKE_CATEGORY],
                               r8_rows[SMOKE_CATEGORY], qpos)
        smoke_leaf = _create_leaf(RUN_NAMESPACE / "smoke" / "seal")
        smoke_leaf.write_json("stage_a_seal.json", {"schema": "r9.smoke.seal.v1",
            "status": "PASS", "run_id": RUN_ID, "smoke": smoke,
            "child_supervisor": smoke_supervisor})
        active_output = None

        registered_orders = tuple(identity["configuration"]["orders"])
        condition_receipts: dict[str, Any] = {}
        parity: dict[str, Any] = {}
        supervisors: dict[str, Any] = {}
        runtimes: dict[str, Any] = {}
        event_count = query_count = 0
        for category in CATEGORIES:
            for order in registered_orders:
                key = f"{category}/{order}"
                leaf = _condition_output(RUN_NAMESPACE / "full", category, order)
                active_output, active_stage = leaf, f"R9_STAGE_A:{key}"
                _write_leaf_manifest(leaf, {"schema": "r9.condition.manifest.v1",
                    "run_id": RUN_ID, "category": category, "order": order,
                    "mode": "FULL", "identity_sha256": identity["identity_sha256"],
                    "runner_source_sha256": source_sha})
                _free_space(OUTPUT_BASE)
                supervised = _run_supervised("STAGE_A", leaf, category=category,
                    order=order, mode="FULL", smoke_events=0)
                _budget(leaf).write_json("supervisor_status.json", supervised)
                qpos = identity["classes"][category]["orders"][order]["query_positions"]
                rows, completion = _verify_condition_output(
                    leaf, category, order, EXPECTED_COUNTS[category], qpos)
                condition_receipts[key] = {"event_count": completion["event_count"],
                    "query_count": completion["query_count"],
                    "score_rows_sha256": completion["score_rows_sha256"],
                    "query_positions_sha256": _sha256_json(qpos),
                    "query_identity_set_sha256": completion["query_identity_set_sha256"],
                    "condition_seal_sha256": _sha256_file(leaf / "stage_a_seal.json"),
                    "support_seed": completion["support_seed"],
                    "stream_shuffle_seed": completion["stream_shuffle_seed"]}
                runtimes[key] = completion["timing_seconds"]
                supervisors[key] = supervised
                event_count += int(completion["event_count"])
                query_count += int(completion["query_count"])
                active_output = None
                if order == "SHUFFLE0" and completion["support_seed"] == 0:
                    if len(rows) != len(r7_rows[category]) or len(rows) != len(r8_rows[category]):
                        raise R9Error(f"Seed-0 parity population differs for {category}")
                    parity[category] = {
                        "R7": _compare_r7_rows(category, rows, r7_rows[category]),
                        "R8": _compare_r8_rows(category, rows, r8_rows[category])}
        if len(condition_receipts) != len(CATEGORIES) * len(registered_orders):
            raise R9Error("R9 did not execute every registered full condition")
        if set(parity) != set(CATEGORIES):
            raise R9Error("R9 mandatory Seed-0 R7/R8 full-event parity gate was not completed")
        full_root = RUN_NAMESPACE / "full"
        seal_leaf = _create_leaf(full_root / "stage_seals")
        active_output, active_stage = full_root / "stage_seals", "R9_STAGE_B"
        stage_a = {"schema": "r9.stage_a.global.seal.v1", "status": "SEALED_COMPLETE",
            "run_id": RUN_ID, "identity_sha256": identity["identity_sha256"],
            "conditions": condition_receipts, "parity": parity, "smoke": smoke,
            "condition_runtimes": runtimes, "condition_supervisors": supervisors,
            "condition_count": len(condition_receipts), "event_count": event_count,
            "query_count": query_count,
            "support1_status": support1_status, "support1_preflight_reason": support1_reason,
            "all_registered_conditions_sealed": True,
            "all_shuffle0_support0_r7_r8_score_query_decision_parity_verified": True,
            "unqueried_test_labels_accessed": False, "test_masks_opened": False,
            "offline_outcome_vectors_accessed": False, "automatic_restart": False,
            "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
        seal_leaf.write_json("stage_a_seal.json", stage_a)
        stage_b_supervisor = _run_supervised("STAGE_B", full_root / "stage_seals")
        _budget(full_root / "stage_seals").write_json("supervisor_status.json",
                                                       stage_b_supervisor)
        aggregate = _read_json(full_root / "stage_seals" / "aggregate_results.json")
        stage_b_seal = _read_json(full_root / "stage_seals" / "stage_b_seal.json")
        if (stage_b_seal.get("status") != "SEALED_COMPLETE" or
                stage_b_seal.get("stage_a_seal_sha256") !=
                _sha256_file(full_root / "stage_seals" / "stage_a_seal.json") or
                stage_b_seal.get("aggregate_results_sha256") !=
                _sha256_file(full_root / "stage_seals" / "aggregate_results.json")):
            raise R9Error("R9 Stage B seal does not bind the global Stage A seal and aggregate")
        reconciliation = _independent_cpu_reconciliation(aggregate, stage_a)
        complete_seal = {"schema": "r9.complete.seal.v1", "status": "COMPLETE",
            "run_id": RUN_ID,
            "stage_a_seal_sha256": _sha256_file(full_root / "stage_seals" / "stage_a_seal.json"),
            "stage_b_seal_sha256": _sha256_file(full_root / "stage_seals" / "stage_b_seal.json"),
            "aggregate_results_sha256": _sha256_file(full_root / "stage_seals" / "aggregate_results.json"),
            "independent_reconciliation": reconciliation,
            "independent_reconciliation_sha256": _sha256_json(reconciliation),
            "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
        _budget(full_root / "stage_seals").write_json("complete_seal.json", complete_seal)
        aggregate["runtime_model_fingerprints"] = model_fingerprints
        aggregate["source_fingerprints"] = dict(identity["source_fingerprints"])
        aggregate["source_sha256"] = source_sha
        aggregate["support1_status"] = support1_status
        aggregate["support1_preflight_reason"] = support1_reason
        macro_full = aggregate["equal_category_macro_ranking_by_order"]
        full_effects = [aggregate["contrast_stream_robustness"]["Full_minus_G"][
            "equal_category_mean_by_shuffle_seed"][order] for order in MANDATORY_ORDERS]
        simple_effects_macro = [aggregate["contrast_stream_robustness"]["SignedExemplar_minus_G"][
            "equal_category_mean_by_shuffle_seed"][order] for order in MANDATORY_ORDERS]
        h9b = aggregate["hypothesis_assessments"]["H9_B_complexity_necessity_vs_signed_exemplar"]
        if h9b == "SUPPORTED_WITHIN_DEVELOPMENT_CONDITIONS":
            recommendation = {"decision": "RETAIN_FULL_C1_AS_DEVELOPMENT_CANDIDATE",
                "rationale": "Full C1 has consistently positive registered AUROC differences over Signed Exemplar; practical value still depends on AP, operating trade-offs, measured resources, and independent confirmation."}
        elif all(value <= 0 for value in full_effects) and all(value <= 0 for value in simple_effects_macro):
            recommendation = {"decision": "ABANDON_SIGNED_CORRECTION_RESEARCH_BRANCH",
                "rationale": "Neither Full C1 nor Signed Exemplar improves equal-category macro AUROC over shared Greedy on any registered shuffle seed."}
        else:
            recommendation = {"decision": "SIMPLIFY_TO_SIGNED_EXEMPLAR_FOR_FURTHER_DEVELOPMENT",
                "rationale": "Full C1 complexity did not show a consistently positive AUROC advantage over the byte-capped simple alternative across all category/order pairs; this conservative development choice does not establish independent confirmation."}
        aggregate["recommendation"] = recommendation
        aggregate["parity"] = parity
        aggregate["stage_a_seal_sha256"] = complete_seal["stage_a_seal_sha256"]
        aggregate["stage_b_seal_sha256"] = complete_seal["stage_b_seal_sha256"]
        aggregate["independent_cpu_reconciliation"] = reconciliation
        # Public aggregates may contain only opaque fingerprints and aggregate metrics.
        test_hashes = cpu_test_receipt["test_source_sha256"]
        _write_public_deliverables(aggregate, source_sha, test_hashes, identity,
            stage_a, stage_b_seal, reconciliation, complete_seal)
        active_output = None
        return {"status": "COMPLETE", "run_id": RUN_ID,
            "identity_sha256": identity["identity_sha256"],
            "stage_a_seal_sha256": complete_seal["stage_a_seal_sha256"],
            "stage_b_seal_sha256": complete_seal["stage_b_seal_sha256"],
            "independent_reconciliation_sha256": complete_seal["independent_reconciliation_sha256"],
            "public_aggregate_sha256": _sha256_file(AGGREGATE_PATH),
            "source_sha256": source_sha}
    except BaseException as exc:
        if active_output is not None and active_output.is_dir():
            _write_failure(active_output, active_stage, exc)
        _write_failure(RUN_NAMESPACE / "manifest", "R9_PARENT", exc)
        raise


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true", help="execute the single authorized R9 run")
    parser.add_argument("--stage-a-child", action="store_true")
    parser.add_argument("--stage-b-child", action="store_true")
    parser.add_argument("--category", choices=CATEGORIES)
    parser.add_argument("--order", choices=ORDERS + ("SUPPORT1_SHUFFLE0",))
    parser.add_argument("--mode", choices=("SMOKE", "FULL"))
    parser.add_argument("--smoke-events", type=int, default=0)
    args = parser.parse_args(argv)
    try:
        if args.stage_a_child:
            if args.category is None or args.order is None or args.mode is None:
                raise R9Error("R9 Stage A child requires category, order, and mode")
            _stage_a_child(args.category, args.order, args.mode, args.smoke_events)
            return 0
        if args.stage_b_child:
            try:
                _r9_stage_b_evaluate()
            except BaseException as exc:
                child_output = Path(os.environ.get("R9_CHILD_OUTPUT", ""))
                _write_failure(child_output, "R9_STAGE_B_CHILD", exc)
                raise
            return 0
        if args.run:
            print(json.dumps(_run_experiment(), indent=2, sort_keys=True))
            return 0
        parser.error("Choose --run, --stage-a-child, or --stage-b-child")
    except BaseException as exc:
        print(f"R9_ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise
    return 2


if __name__ == "__main__":
    raise SystemExit(main())


