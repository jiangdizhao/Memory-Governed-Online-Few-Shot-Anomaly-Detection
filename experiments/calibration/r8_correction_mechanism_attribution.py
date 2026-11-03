"""R8 matched capacity, order, and Correction-address attribution experiment.

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
EXPECTED_PARENT = "579fb7f5f4c590b95fda2b1c1c417f372aaf013f"
RUN_ID = "r8_correction_mechanism_20261103_a5"
PRESERVED_FAILED_ATTEMPTS = (
    {"run_id": "r8_correction_mechanism_20261103_a1",
     "status": "INITIALIZATION_FAILED_BEFORE_GPU",
     "reason": "parent pre-created manifest directory before OutputBudget; no experiment child launched",
     "namespace_reused": False},
    {"run_id": "r8_correction_mechanism_20261103_a2",
     "status": "SUPERVISED_SMOKE_FAILED_BEFORE_TEST_ENCODING",
     "reason": "the fixed-bank receipt evidence object was compared as if it were a digest string",
     "namespace_reused": False},
    {"run_id": "r8_correction_mechanism_20261103_a3",
     "status": "SMOKE_CHILD_COMPLETE_PARENT_GATE_FAILED",
     "reason": "the parent parity verifier was defined after the __main__ call; full conditions were not launched",
     "namespace_reused": False},
    {"run_id": "r8_correction_mechanism_20261103_a4",
     "status": "ALL_STAGE_A_SEALED_STAGE_B_FAILED",
     "reason": "Stage B attempted to score RAW rows with SHUFFLE0-only mechanism controls; KeyError Scalar. Artifacts preserved; no automatic restart.",
     "namespace_reused": False},
)
RUN_NAMESPACE = ROOT / "outputs" / "calibration" / RUN_ID
OUTPUT_BASE = ROOT / "outputs" / "calibration"
R7_RUN_ID = "r7_shuffled_correction_4class_20261103_a4"
R7_FULL = OUTPUT_BASE / R7_RUN_ID / "full"
R7_REPORT = ROOT / "docs" / "calibration" / "audit" / "r7_shuffled_correction_4class_20261103" / "REPORT.md"
R7_AGGREGATE = ROOT / "docs" / "calibration" / "audit" / "r7_shuffled_correction_4class_20261103" / "aggregate_results.json"
REPORT_DIR = ROOT / "docs" / "calibration" / "audit" / "r8_correction_mechanism_20261103"
REPORT_PATH = REPORT_DIR / "REPORT.md"
AGGREGATE_PATH = REPORT_DIR / "aggregate_results.json"
CATEGORIES = ("audiojack", "regulator", "switch", "rolled_strip_base")
ORDERS = ("RAW", "SHUFFLE0")
CAPACITIES = (256, 1024)
SCORERS = ("G1024", "GC256", "GC1024")
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
SMOKE_EVENTS = 512
OUTPUT_LIMIT = 4 * 1024**3
FAILURE_RESERVE = 64 * 1024
MIN_FREE_BYTES = 10 * 1024**3
FULL_DEADLINE_SECONDS = 120 * 60
SMOKE_DEADLINE_SECONDS = 15 * 60
BOOTSTRAP_REPLICATES = 1000
BOOTSTRAP_SEED = 20261103
RUN_SCHEMA = "r8.correction_mechanism.run.v1"
IDENTITY_SCHEMA = "r8.correction_mechanism.identity.v1"
SCORE_SCHEMA = "r8.correction_mechanism.score.v1"
AUTHORIZATION = "R8 unified retrospective four-class correction mechanism attribution"
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
    "experiments/risk_safe/native_bridge.py",
    "experiments/p0_protocol_audit.py",
    "experiments/v02_memory.py",
    "vmb_visionad_new.py",
    "test_phase3_AIF_simplified_v22.py",
)


class R8Error(RuntimeError):
    pass


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
    result["experiments/calibration/r8_correction_mechanism_attribution.py"] = _sha256_file(Path(__file__).resolve())
    return result


def _assert_repository_identity() -> None:
    branch, head = _git("branch", "--show-current"), _git("rev-parse", "HEAD")
    if branch != EXPECTED_BRANCH or head != EXPECTED_PARENT:
        raise R8Error(f"baseline mismatch: branch={branch!r}, HEAD={head}")
    dirty = _git("status", "--porcelain", "--", *SCORE_SOURCE_PATHS[1:])
    if dirty:
        raise R8Error("a pinned R8 dependency has tracked or staged local changes")


def make_order_events(shuffled_events: Sequence[Any], order: str) -> list[Any]:
    """Return a complete RAW or R7 SHUFFLE0 bijection without opening labels."""
    if order == "SHUFFLE0":
        events = list(shuffled_events)
        if [int(e.position) for e in events] != list(range(len(events))):
            raise R8Error("R7 shuffle positions are not contiguous")
        return events
    if order != "RAW":
        raise R8Error(f"unsupported stream order: {order}")
    by_original = sorted(shuffled_events, key=lambda e: int(e.original_position))
    if [int(e.original_position) for e in by_original] != list(range(len(by_original))):
        raise R8Error("raw order is not a bijection over original JSON positions")
    return [replace(event, position=index) for index, event in enumerate(by_original)]


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


def build_identity(sources: Mapping[str, str], runner_sha256: str,
                   r7_receipts: Mapping[str, Any], contract: Mapping[str, Any]) -> dict[str, Any]:
    from experiments.calibration import r7_shuffled_correction_4class as r7
    classes: dict[str, Any] = {}
    for category in CATEGORIES:
        info = contract["metadata"][category]
        shuffled = list(info["events"])
        raw = make_order_events(shuffled, "RAW")
        query_ids = {str(shuffled[p].image_identity_sha256)
                     for p in info["tape_receipt"]["positions"]}
        order_records = {}
        for order, events in (("RAW", raw), ("SHUFFLE0", shuffled)):
            positions = matched_query_positions(
                shuffled, info["tape_receipt"]["positions"], events)
            order_records[order] = {
                "event_count": len(events), "query_count": len(positions),
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
        "configuration": {
            "orders": list(ORDERS), "categories": list(CATEGORIES),
            "support_seed": 0, "fixed_normal_support_images_per_category": 4,
            "additional_protected_c64_normals_per_category": 64,
            "fixed_normal_capacity": 25000,
            "greedy_capacity": 1024, "correction_capacities": [256, 1024],
            "backbone_updates": False, "unqueried_self_training": False,
            "masks_in_online_scoring": False,
            "query_identity_policy": "exact R7 Q-A-TS Seed37 shuffled identity set, remapped by image identity in RAW",
            "query_tape_sha256_by_category": {
                c: contract["metadata"][c]["tape_receipt"]["tape_sha256"] for c in CATEGORIES},
            "c1_formula": "frozen native C1-Full, original writer and readout; only K differs",
            "controller_settings": {"B1": {"window": 256, "minimum_normals": 16,
                                             "quantile": "higher"},
                                    "B3": {"delta": 0.01, "minimum_c64_normals": 16,
                                             "quantile": "higher"}},
            "address_shuffle_seeds": list(ADDRESS_SEEDS),
            "address_attribute_arrays": ["beta", "radius", "kappa"],
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
    raw = hashlib.sha256(f"r8-address-v1\0{seed}\0{state_identity}".encode("ascii")).digest()
    local_seed = int.from_bytes(raw[:8], "big", signed=False)
    import numpy as np
    return tuple(int(x) for x in np.random.default_rng(local_seed).permutation(eligible_count))


def permute_attribute_bundles(codebook: Any, seed: int, state_identity: str) -> tuple[Any, tuple[int, ...]]:
    """Clone a source-filtered codebook and shuffle beta/radius/kappa as bundles.

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
        for name in ("beta", "radius", "kappa"):
            snapshot = getattr(clone, name)[:n].clone()
            getattr(clone, name)[:n].copy_(snapshot[p])
    return clone, permutation


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


def _verify_r8_child_identity(output: Path, category: str, order: str,
                              mode: str, smoke_events: int) -> tuple[dict[str, Any], dict[str, Any]]:
    if category not in CATEGORIES or order not in ORDERS or mode not in {"SMOKE", "FULL"}:
        raise R8Error("child condition is outside the frozen R8 design")
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
        raise R8Error("R8 child identity manifest is malformed or its runner source changed")
    if _source_fingerprints() != identity.get("source_fingerprints"):
        raise R8Error("R8 child pinned source fingerprint changed")
    _assert_repository_identity()
    expected_dir = RUN_NAMESPACE / ("smoke" if mode == "SMOKE" else "full")
    if expected_dir not in output.parents:
        raise R8Error("R8 child output is outside its designated mode namespace")
    order_info = identity["classes"][category]["orders"][order]
    if mode == "SMOKE" and not 0 < smoke_events <= SMOKE_EVENTS:
        raise R8Error("R8 smoke event limit exceeds the fixed smoke bound")
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

        c64_items, c64_cache_receipt = r7._make_c64_items(category, info, p0, impl, backbone, cfg)
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


def _compare_r7_rows(category: str, r8_rows: Sequence[Mapping[str, Any]],
                     r7_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    import numpy as np
    if len(r8_rows) > len(r7_rows):
        raise R8Error(f"{category}: R8 comparison exceeds the R7 reference population")
    score_errors = {"G1024": [], "GC1024": []}
    threshold_errors: dict[str, list[float]] = {
        f"{scorer}-{controller}": [] for scorer in ("G1024", "GC1024")
        for controller in CONTROLLERS}
    decision_mismatches = 0
    query_mismatches = 0
    identity_mismatches = 0
    for index, (actual, expected) in enumerate(zip(r8_rows, r7_rows)):
        if (int(actual.get("position", -1)) != index or
                int(expected.get("position", -1)) != index or
                actual.get("image_identity_sha256") != expected.get("image_identity_sha256") or
                int(actual.get("original_json_position", -1)) !=
                int(expected.get("original_json_position", -2)) or
                actual.get("source_group_sha256") != expected.get("source_group_sha256") or
                actual.get("event_id_sha256") != expected.get("event_id_sha256") or
                int(actual.get("camera_id", -1)) != int(expected.get("camera_id", -2)) or
                actual.get("structured_fixed_support_sha256") !=
                expected.get("structured_fixed_support_sha256")):
            identity_mismatches += 1
            continue
        query_mismatches += int(bool(actual.get("query_tape_decision")) !=
                                bool(expected.get("query_tape_decision")))
        for arm, key in (("G1024", "G"), ("GC1024", "GC")):
            if key not in expected.get("scores", {}):
                raise R8Error("R7 reference lacks the required G/GC score")
            score_errors[arm].append(float(actual["scores"][arm]) -
                                     float(expected["scores"][key]))
            for controller in CONTROLLERS:
                actual_decision = actual["decisions"][f"{arm}-{controller}"]
                reference_decision = expected["decisions"][f"{key}-{controller}"]
                if int(actual_decision["prediction"]) != int(reference_decision["prediction"]):
                    decision_mismatches += 1
                threshold_errors[f"{arm}-{controller}"].append(
                    float(actual_decision["threshold"]) - float(reference_decision["threshold"]))
    if identity_mismatches or query_mismatches or decision_mismatches:
        raise R8Error(
            f"R7 event parity failed for {category}: identity={identity_mismatches}, "
            f"query={query_mismatches}, decision={decision_mismatches}")
    max_score = {key: float(np.max(np.abs(values))) if values else 0.0
                 for key, values in score_errors.items()}
    max_threshold = {key: float(np.max(np.abs(values))) if values else 0.0
                     for key, values in threshold_errors.items()}
    for arm, errors in score_errors.items():
        for actual, error in zip(r8_rows, errors):
            ref_key = "G" if arm == "G1024" else "GC"
            expected_score = float(actual["scores"][arm]) - error
            if not math.isclose(float(actual["scores"][arm]), expected_score,
                                rel_tol=1e-4, abs_tol=1e-5):
                raise R8Error("R8 score parity exceeds the established atol/rtol")
    if any(value > 1e-12 for value in max_threshold.values()):
        raise R8Error("R8 R7 controller thresholds differ despite matched state trajectory")
    return {"event_count": len(r8_rows), "query_mismatches": query_mismatches,
            "decision_mismatches": decision_mismatches,
            "maximum_absolute_score_error": max_score,
            "maximum_absolute_threshold_error": max_threshold,
            "atol": 1e-5, "rtol": 1e-4,
            "identity_bijection_verified": True}


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
    test_path = ROOT / "tests" / "calibration" / "test_r8_correction_mechanism_attribution.py"
    if not test_path.is_file():
        raise R8Error("focused R8 CPU test file is missing")
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = ""
    env["R3_STAGE_B_CPU_ONLY"] = "1"
    env["PYTHONHASHSEED"] = "0"
    command = [sys.executable, "-m", "pytest", "-q", str(test_path)]
    result = subprocess.run(command, cwd=ROOT, env=env,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, encoding="utf-8", errors="replace")
    receipt = {"command": command, "return_code": int(result.returncode),
               "test_source_sha256": _sha256_file(test_path),
               "output": result.stdout[-20000:]}
    if result.returncode != 0:
        raise R8Error(f"focused R8 CPU tests failed:\n{result.stdout[-8000:]}")
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
    env["R8_CHILD_OUTPUT"] = str(output)
    command = [str(Path(sys.executable)), "-m", "experiments.calibration.r8_correction_mechanism_attribution"]
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
        raise R8Error(f"supervised R8 {stage} child failed: {record}")
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
    if (manifest.get("category") != category or manifest.get("order") != order or
            seal.get("category") != category or seal.get("order") != order or
            seal.get("event_count") != expected_count or len(rows) != expected_count or
            seal.get("query_positions") != list(expected_query_positions) or
            seal.get("score_rows_sha256") != _sha256_file(path / "stage_a_scores.jsonl") or
            any(row.get("category") != category or row.get("order") != order or
                int(row.get("position", -1)) != i or "label" in row or "anomaly_class" in row
                for i, row in enumerate(rows))):
        raise R8Error(f"{category}/{order}: Stage A seal or score rows fail identity checks")
    return rows, seal


def _compare_smoke(category: str, path: Path,
                   r7_rows: Sequence[Mapping[str, Any]],
                   expected_query_positions: Sequence[int]) -> dict[str, Any]:
    rows, completion = _verify_condition_output(
        path, category, "SHUFFLE0", SMOKE_EVENTS,
        [p for p in expected_query_positions if p < SMOKE_EVENTS])
    parity = _compare_r7_rows(category, rows, r7_rows[:SMOKE_EVENTS])
    if (completion.get("queried_normal", 0) <= 0 or
            completion.get("queried_defect", 0) <= 0 or
            completion.get("greedy_state_changes", 0) <= 0 or
            any(completion.get("correction_state_changes", {}).get(k, 0) <= 0
                for k in ("GC256", "GC1024")) or
            any(len(completion.get("b3_threshold_history", {}).get(s, [])) < 2
                for s in SCORERS)):
        raise R8Error("R8 smoke did not exercise both labels, both C1 banks, Greedy changes, and B3 refresh")
    if any(not all(name in row.get("scores", {}) for name in
                   ("Scalar", "Address17", "Address53", "Address97")) for row in rows):
        raise R8Error("R8 smoke did not execute every registered mechanism control")
    return {"status": "PASS", "category": category, "order": "SHUFFLE0",
            "event_count": len(rows), "queried_normal": completion["queried_normal"],
            "queried_defect": completion["queried_defect"],
            "greedy_state_changes": completion["greedy_state_changes"],
            "correction_state_changes": completion["correction_state_changes"],
            "c64_rescore_costs": completion["c64_rescore_costs"],
            "r7_parity": parity,
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
    parser.add_argument("--run", action="store_true", help="execute the single authorized R8 run")
    parser.add_argument("--stage-a-child", action="store_true")
    parser.add_argument("--stage-b-child", action="store_true")
    parser.add_argument("--category", choices=CATEGORIES)
    parser.add_argument("--order", choices=ORDERS)
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
    for start in range(0, len(y) - 511, 512):
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


if __name__ == "__main__":
    raise SystemExit(main())
