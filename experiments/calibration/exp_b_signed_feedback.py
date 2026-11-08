"""Integrated Experiment B: signed Correction causal attribution.

Implementation only; no result in this module implies a scientific outcome.
The four native factorial arms share one R9 Greedy writer and one event loop.
The external memory editors are independent secondary trajectories.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import time
import traceback
from typing import Any, Mapping, Sequence

# Keep the launcher's authorized device selection before any imported research
# helper can hide CUDA as part of a CPU-only utility.
AUTHORIZED_CUDA_VISIBILITY = os.environ.get("CUDA_VISIBLE_DEVICES")

import numpy as np

from experiments.calibration.exp_b_external_hitl import (
    CAPACITY as EXTERNAL_CAPACITY,
    ExternalMemoryError,
    ExternalNormalMemory,
)


ROOT = Path(__file__).resolve().parents[2]
CATEGORIES = ("audiojack", "regulator", "switch", "rolled_strip_base")
EXPECTED_COUNTS = {"audiojack": 3815, "regulator": 3855, "switch": 3820,
                   "rolled_strip_base": 3805}
EXPECTED_QUERIES = {"audiojack": 114, "regulator": 116, "switch": 115,
                    "rolled_strip_base": 114}
MAIN_ARMS = ("G", "N", "D", "ND", "SignedKNN", "HITL_FP", "HITL_FP_FN")
SCORE_SCHEMA = "exp-b.signed-feedback.score.v1"
AUTH_SCHEMA = "exp-b.full-run-authorization.v1"
PROTOCOL_PATH = ROOT / "docs" / "calibration" / "EXP_B_SCIENTIFIC_AND_IMPLEMENTATION_CONTRACT.md"
RUN_ROOT = ROOT / "outputs" / "calibration" / "experiment_b_signed_feedback"
MAX_OUTPUT_BYTES = 4 * 1024**3
FAILURE_RESERVE_BYTES = 64 * 1024
MIN_FREE_BYTES = 10 * 1024**3
CATEGORY_DEADLINE_SECONDS = 3600
STAGE_B_DEADLINE_SECONDS = 3600
PINNED_SOURCE_PATHS = (
    "experiments/p0_protocol_audit.py",
    "experiments/calibration/r1_matched_correction_pilot.py",
    "experiments/calibration/r1_runtime_supervisor.py",
    "experiments/calibration/r2_exp2a_c1_strength.py",
    "experiments/calibration/r3_fullstream_2x2.py",
    "experiments/calibration/r6_capacity1024_audiojack_runner.py",
    "experiments/calibration/r7_shuffled_correction_4class.py",
    "experiments/calibration/r9_correction_mechanism_falsification.py",
    "experiments/calibration/r10_signed_knn_baseline.py",
    "experiments/calibration/controllers.py",
    "experiments/calibration/native.py",
    "experiments/calibration/profiles.py",
    "experiments/calibration/stage05.py",
    "experiments/risk_safe/native_bridge.py",
    "experiments/v02_memory.py",
    "vmb_visionad_new.py",
    "test_phase3_AIF_simplified_v22.py",
)


class ExperimentBError(RuntimeError):
    pass


def canonical_json(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"),
                        ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json_exclusive(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(canonical_json(dict(value)).decode("utf-8"))
        stream.flush()
        os.fsync(stream.fileno())


def append_jsonl_fsync(path: Path, value: Mapping[str, Any]) -> None:
    with path.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(canonical_json(dict(value)).decode("utf-8"))
        stream.flush()
        os.fsync(stream.fileno())


def score_row_is_label_blind(row: Mapping[str, Any]) -> bool:
    forbidden = {"label", "anomaly_class", "ground_truth", "mask_path", "defect_mask"}
    return not bool(forbidden.intersection(row))


def accepted_c1_signs(arm: str, label: int) -> tuple[int, ...]:
    """Signs accepted by each native trajectory: +1 defect, -1 normal."""
    if arm not in {"G", "N", "D", "ND"} or int(label) not in (0, 1):
        raise ExperimentBError("Unknown factorial arm or nonbinary verified label")
    if arm == "G":
        return ()
    if arm == "N":
        return (-1,) if int(label) == 0 else ()
    if arm == "D":
        return (1,) if int(label) == 1 else ()
    return (-1,) if int(label) == 0 else (1,)


def sign_masked_c1_codebook(impl: Any, live: Any, eligible_slots: Sequence[int],
                            sign: int, cfg: Any) -> Any:
    """Clone only source-eligible anchors of one sign, including denominator filtering."""
    import torch

    if int(sign) not in (-1, 1):
        raise ExperimentBError("C1 readout sign must be +1 or -1")
    selected = [int(i) for i in eligible_slots if float(live.beta[int(i)].item()) * int(sign) > 0]
    clone_cfg = type(cfg)(**asdict(cfg))
    clone_cfg.enabled = True
    clone = impl.SignedCorrectionCodebook(
        max(1, len(selected)), int(live.protos.shape[1]), live.device, clone_cfg)
    clone.n = len(selected)
    if selected:
        ids = torch.as_tensor(selected, dtype=torch.long, device=live.device)
        for name in ("protos", "beta", "radius", "kappa", "count", "outcome_fp",
                     "outcome_fn", "outcome_tn", "outcome_tp", "last_step"):
            getattr(clone, name)[:len(selected)].copy_(getattr(live, name)[ids])
    return clone


def score_sign_readout(impl: Any, bank: Any, r1: Any, cfg: Any, codebook: Any,
                       source_groups: Mapping[int, set[str]], selected: Any,
                       group_sha256: str, position: int, base_map: Any,
                       top_ids: Any, sign: int) -> tuple[float, int]:
    """Apply R9 source/time filtering first, then a true sign-filtered clone."""
    prior_cfg, prior_cb = bank.corr_cfg, bank.corr_cb["p0"]["id"][0]
    bank.corr_cfg = cfg
    bank.corr_cb["p0"]["id"][0] = codebook
    try:
        filtered, eligible = r1._filtered_c1_codebook(
            impl, bank, source_groups, group_sha256, event_position=position)
        if not eligible:
            return float(r1.max_score(base_map)), 0
        signed = sign_masked_c1_codebook(impl, filtered, range(filtered.size()), sign, cfg)
        if signed.size() == 0:
            return float(r1.max_score(base_map)), 0
        delta = signed.apply(impl.l2_normalize(selected.to(bank.device))).detach().float().cpu().numpy()
        corrected = r1.corrected_dense_map(base_map, top_ids, np.asarray(delta, dtype=np.float32))
        return float(r1.max_score(corrected)), int(signed.size())
    finally:
        bank.corr_cb["p0"]["id"][0] = prior_cb
        bank.corr_cfg = prior_cfg


def state_digest(fixed_sha: str, greedy_sha: str, correction_sha: str | None = None,
                 extra_sha: str | None = None) -> str:
    if extra_sha is not None:
        value: dict[str, str] = {"fixed_support_sha256": fixed_sha,
                                 "adapter_state_sha256": extra_sha}
    else:
        value = {"fixed_support_sha256": fixed_sha,
                 "greedy_memory_sha256": greedy_sha}
    if correction_sha is not None:
        value["correction_state_sha256"] = correction_sha
    return sha256_bytes(canonical_json(value))


def validate_score_rows(rows: Sequence[Mapping[str, Any]], *, category: str,
                        expected_count: int, expected_queries: Sequence[int]) -> None:
    if len(rows) != int(expected_count):
        raise ExperimentBError(f"{category}: Stage A score row count differs from frozen events")
    query_positions = []
    for position, row in enumerate(rows):
        if not score_row_is_label_blind(row):
            raise ExperimentBError(f"{category}:{position}: Stage A score row disclosed a label field")
        if (int(row.get("position", -1)) != position or row.get("category") != category or
                row.get("schema") != SCORE_SCHEMA or set(row.get("scores", {})) != set(MAIN_ARMS)):
            raise ExperimentBError(f"{category}:{position}: Stage A score identity/arm schema mismatch")
        if row.get("query_tape_decision") is True:
            query_positions.append(position)
    if query_positions != list(expected_queries):
        raise ExperimentBError(f"{category}: sealed Query Tape differs from registered positions")


def _git(args: Sequence[str]) -> str:
    proc = subprocess.run(["git", *args], cwd=ROOT, check=True, capture_output=True, text=True)
    return proc.stdout.strip()


def _source_fingerprints() -> dict[str, str]:
    paths = list(PINNED_SOURCE_PATHS) + [
        "experiments/calibration/exp_b_signed_feedback.py",
        "experiments/calibration/exp_b_external_hitl.py",
    ]
    return {relative: sha256_file(ROOT / relative) for relative in paths}


def _verify_source_fingerprints(expected: Mapping[str, str]) -> None:
    actual = _source_fingerprints()
    if dict(expected) != actual:
        changed = sorted(key for key in set(expected) | set(actual)
                         if expected.get(key) != actual.get(key))
        raise ExperimentBError(f"Source fingerprint mismatch: {changed}")


def _tree_bytes(path: Path) -> int:
    total = 0
    for current, _dirs, files in os.walk(path):
        for name in files:
            candidate = Path(current) / name
            try:
                total += candidate.stat().st_size
            except FileNotFoundError:
                raise ExperimentBError(f"Output changed during resource audit: {candidate}")
    return total


def _registered_category_identity(contract: Mapping[str, Any], category: str,
                                  r9: Any, r7: Any) -> dict[str, Any]:
    info = contract["metadata"][category]
    events = r9.make_order_events(info["events"], "SHUFFLE0")
    positions = tuple(int(v) for v in info["tape_receipt"]["positions"])
    if (len(events) != EXPECTED_COUNTS[category] or
            len(positions) != EXPECTED_QUERIES[category] or
            tuple(sorted(positions)) != positions):
        raise ExperimentBError(f"{category}: frozen event or query identity count changed")
    query_ids = sorted(str(events[position].image_identity_sha256) for position in positions)
    if len(query_ids) != len(set(query_ids)):
        raise ExperimentBError(f"{category}: frozen Query Tape repeats an image identity")
    return {
        "event_count": len(events), "query_count": len(positions),
        "query_positions": list(positions),
        "query_positions_sha256": info["tape_receipt"]["positions_sha256"],
        "query_identity_set_sha256": r7._sha256_json(query_ids),
        "query_tape_sha256": info["tape_receipt"]["tape_sha256"],
        "event_identity_map_sha256": r7._sha256_json([
            {"position": int(e.position), "original_position": int(e.original_position),
             "event_id_sha256": str(e.event_id_sha256),
             "image_identity_sha256": str(e.image_identity_sha256),
             "source_group_sha256": str(e.source_group_sha256),
             "camera_id": str(e.camera_id)} for e in events]),
        "category_json_sha256": info["contract"]["category_json_sha256"],
        "fixed_support_sha256": info["fixed_support_sha256_expected"],
        "support_image_sha256": info["support_image_sha256"],
        "support_group_sha256": info["support_group_sha256"],
        "c64_image_sha256": info["frozen_info"]["c64"]["image_sha256"],
        "c64_group_sha256": info["frozen_info"]["c64"]["group_sha256"],
        "c64_score_archive_sha256": info["c64_score_archive_sha256"],
    }


def _preflight_historical_references(contract: Mapping[str, Any]) -> dict[str, Any]:
    """Verify sealed R9/R10 identity and score references before GPU execution."""
    from experiments.calibration import r10_signed_knn_baseline as r10

    r9refs = r10._load_r9_reference(contract, categories=CATEGORIES)
    manifest_path = r10.RUN_ROOT / "global" / "run_manifest.json"
    stage_path = r10.RUN_ROOT / "global" / "stage_a_seal.json"
    complete_path = r10.RUN_ROOT / "global" / "complete_seal.json"
    for path in (manifest_path, stage_path, complete_path):
        if not path.is_file():
            raise ExperimentBError(f"Historical R10 sealed reference is missing: {path.name}")
    manifest, stage, complete = (r10._read_json(manifest_path), r10._read_json(stage_path),
                                 r10._read_json(complete_path))
    identity = manifest.get("identity", {})
    if (stage.get("status") != "SEALED_COMPLETE" or
            stage.get("identity_sha256") != manifest.get("identity_sha256") or
            stage.get("event_count") != sum(EXPECTED_COUNTS.values()) or
            stage.get("query_count") != sum(EXPECTED_QUERIES.values()) or
            complete.get("status") != "COMPLETE" or
            complete.get("identity_sha256") != manifest.get("identity_sha256") or
            complete.get("stage_a_seal_sha256") != sha256_file(stage_path)):
        raise ExperimentBError("Historical R10 global Stage A/complete seals failed preflight")
    references: dict[str, Any] = {"r9": r9refs, "r10_rows": {}, "receipts": {}}
    for category in CATEGORIES:
        reference = r9refs["references"][category]
        r9_identity = r9refs["identity"]["classes"][category]
        r10_identity = identity.get("classes", {}).get(category, {})
        leaf = r10.RUN_ROOT / category
        score_path = leaf / "stage_a_scores.jsonl"
        category_seal_path = leaf / "stage_a_seal.json"
        if not score_path.is_file() or not category_seal_path.is_file():
            raise ExperimentBError(f"Historical R10 {category} score or seal is missing")
        rows = r10._read_jsonl(score_path)
        category_seal = r10._read_json(category_seal_path)
        receipt = stage.get("category_seals", {}).get(category, {})
        if (len(rows) != EXPECTED_COUNTS[category] or
                category_seal.get("status") != "SEALED_COMPLETE" or
                category_seal.get("score_rows_sha256") != sha256_file(score_path) or
                receipt.get("score_rows_sha256") != sha256_file(score_path) or
                int(category_seal.get("query_count", -1)) != EXPECTED_QUERIES[category] or
                int(r10_identity.get("event_count", -1)) != EXPECTED_COUNTS[category] or
                int(r10_identity.get("query_count", -1)) != EXPECTED_QUERIES[category] or
                r10_identity.get("fixed_support_sha256") != r9_identity.get("fixed_support_sha256") or
                r10_identity.get("query_identity_set_sha256") != r9_identity.get("r7_query_identity_set_sha256")):
            raise ExperimentBError(f"Historical R10 {category} category identity/seal failed preflight")
        for position, (row, r9row) in enumerate(zip(rows, reference["rows"])):
            if (not score_row_is_label_blind(row) or
                    row.get("event_id_sha256") != r9row.get("event_id_sha256") or
                    row.get("image_identity_sha256") != r9row.get("image_identity_sha256") or
                    row.get("source_group_sha256") != r9row.get("source_group_sha256") or
                    row.get("query_tape_decision") != r9row.get("query_tape_decision") or
                    row.get("score_state_sha256", {}).get("G1024") !=
                    r9row.get("score_state_sha256", {}).get("G1024")):
                raise ExperimentBError(f"Historical R10 identity/query/Greedy mismatch at {category}:{position}")
        references["r10_rows"][category] = rows
        references["receipts"][category] = {
            "r9_score_rows_sha256": reference["score_sha256"],
            "r10_score_rows_sha256": sha256_file(score_path),
            "r9_category_seal_sha256": reference["r9_condition_seal_sha256"],
            "r10_category_seal_sha256": sha256_file(category_seal_path),
            "event_count": len(rows), "query_count": EXPECTED_QUERIES[category],
        }
    references["r10_global_manifest_sha256"] = sha256_file(manifest_path)
    references["r10_global_stage_a_seal_sha256"] = sha256_file(stage_path)
    references["r10_complete_seal_sha256"] = sha256_file(complete_path)
    r9_aggregate_path = ROOT / "docs" / "calibration" / "audit" / "r9_correction_mechanism" / "aggregate_results.json"
    r9_aggregate = json.loads(r9_aggregate_path.read_text(encoding="utf-8"))
    if (r9_aggregate.get("run_id") != r9refs["manifest"].get("run_id") or
            r9_aggregate.get("identity_sha256") != r9refs["identity"].get("identity_sha256") or
            r9_aggregate.get("stage_a_seal_sha256") != r9refs["seal_hashes"].get("stage_a")):
        raise ExperimentBError("Historical R9 aggregate evidence is not bound to its sealed source run")
    references["historical_beta_permutation"] = {
        "source_run_id": r9_aggregate["run_id"],
        "aggregate_results_sha256": sha256_file(r9_aggregate_path),
        "status": "REUSED_HISTORICAL_EVIDENCE_NOT_RECOMPUTED",
        "H9_A_beta_address_association": r9_aggregate.get("hypothesis_assessments", {}).get(
            "H9_A_beta_address_association"),
        "interpretation": "prior R9 result; not a new Experiment B result"}
    return references


def _check_run_authorization(path: Path, *, run_id: str) -> dict[str, Any]:
    if not path.is_file():
        raise ExperimentBError("Full execution requires the review-complete researcher authorization file")
    try:
        auth = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ExperimentBError(f"Could not read full-run authorization: {exc}") from exc
    current_commit = _git(["rev-parse", "HEAD"])
    protocol_sha = sha256_file(PROTOCOL_PATH)
    if (auth.get("schema") != AUTH_SCHEMA or
            auth.get("authorization_status") != "AUTHORIZE_FULL_REAL_IAD_STAGE_A_B" or
            not str(auth.get("researcher", "")).strip() or
            auth.get("independent_review_status") != "PASS" or
            auth.get("reviewed_source_commit_sha") != current_commit or
            auth.get("allow_full_run") is not True or auth.get("run_id") != run_id or
            auth.get("source_commit_sha") != current_commit or
            auth.get("protocol_sha256") != protocol_sha or
            tuple(auth.get("categories", ())) != CATEGORIES or
            auth.get("confirmation_data_access") is not False or
            auth.get("test_masks_access") is not False or
            auth.get("stage_b_after_all_stage_a_seals") is not True):
        raise ExperimentBError("Full-run authorization is not bound to this source, run, protocol, and scope")
    if _git(["branch", "--show-current"]) != "calibration-stage0-contract-20261029":
        raise ExperimentBError("Experiment B full execution requires the registered calibration branch")
    if _git(["status", "--porcelain", "--untracked-files=no"]):
        raise ExperimentBError("Tracked source changes are present; full execution is not source-locked")
    return {"authorization_file_sha256": sha256_file(path),
            "authorization_status": auth["authorization_status"],
            "source_commit_sha": current_commit,
            "protocol_sha256": protocol_sha,
            "categories": list(CATEGORIES), "allow_full_run": True}


def _make_native_c1_states(bank: Any, impl: Any, base_cfg: Any, base_cb: Any) -> dict[str, dict[str, Any]]:
    """Keep four independent K1024 Correction trajectories over the shared bank."""
    states: dict[str, dict[str, Any]] = {}
    for arm in ("N", "D", "ND"):
        cfg = type(base_cfg)(**asdict(base_cfg))
        cfg.enabled = False
        cb = type(base_cb)(int(base_cb.K), int(base_cb.protos.shape[1]), base_cb.device, cfg)
        if int(cb.K) != 1024 or cb.size() != 0:
            raise ExperimentBError(f"{arm} native C1 codebook did not start empty at K1024")
        states[arm] = {"cfg": cfg, "cb": cb, "sources": {}, "version": 0,
                       "digest": "", "c64": [], "controller": None}
    states["ND"] = {"cfg": base_cfg, "cb": base_cb, "sources": {}, "version": 0,
                    "digest": "", "c64": [], "controller": None}
    base_cfg.enabled = False
    return states


def _score_c1(impl: Any, bank: Any, r9: Any, r1: Any, exp2a: Any,
              state: Mapping[str, Any], selected: Any, group_sha: str,
              position: int, base_map: Any, top_ids: Any) -> tuple[float, dict[str, Any]]:
    return r9._score_codebook(impl, bank, r1, exp2a, state["cfg"], state["cb"],
                              state["sources"], selected, group_sha, position,
                              base_map, top_ids)


def _refresh_threshold(controller: Any, scores: Sequence[float], group_sha: str,
                       old_digest: str, new_digest: str, old_version: int) -> dict[str, Any] | None:
    if old_digest == new_digest:
        return None
    from experiments.calibration import r7_shuffled_correction_4class as r7

    transition = r7._refresh_b3(controller, scores, group_sha, old_digest,
                                new_digest, old_version)
    if transition is None:
        raise ExperimentBError("B3 current-state refresh was omitted after a scorer state change")
    transition["movement_cap"] = 0.01
    return transition


def _jsonl_rows(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            if not line.endswith("\n"):
                raise ExperimentBError(f"{path.name}:{number}: row is not fsync/newline sealed")
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ExperimentBError(f"{path.name}:{number}: row is not an object")
            rows.append(value)
    return rows


def _stage_a_category(category: str, category_output: Path, *, run_id: str,
                      source_commit: str, identity_sha256: str,
                      authorized_cuda_visibility: str | None,
                      experiment_seed: int = 0) -> dict[str, Any]:
    """Run one frozen category through one encoder and all seven main arms."""
    inherited_visibility = authorized_cuda_visibility
    import torch
    import psutil
    from types import SimpleNamespace
    from collections import Counter

    from experiments import p0_protocol_audit as p0
    from experiments.calibration import r1_matched_correction_pilot as r1
    from experiments.calibration import r2_exp2a_c1_strength as exp2a
    from experiments.calibration import r3_fullstream_2x2 as r3
    from experiments.calibration import r6_capacity1024_audiojack_runner as r6
    from experiments.calibration import r7_shuffled_correction_4class as r7
    from experiments.calibration import r9_correction_mechanism_falsification as r9
    from experiments.calibration import r10_signed_knn_baseline as r10
    from experiments.calibration.controllers import StateConsistentQuantile
    from experiments.calibration.native import _fixed_bank_digest
    from experiments.calibration.r1_matched_correction_pilot import CausalEventOrder
    from experiments.risk_safe.native_bridge import dynamic_codebooks
    from v02_memory import MemoryArm

    if category not in CATEGORIES or category_output.exists():
        raise ExperimentBError("Category output must be a fresh leaf for a registered category")
    category_output.mkdir(parents=True, exist_ok=False)
    # Historical CPU-only helpers may hide CUDA at import time. Restore the
    # child environment captured before these imports, as R3/R9/R10 do.
    r3._restore_cuda_visibility(inherited_visibility)
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise ExperimentBError("Full Stage A requires exactly one authorized CUDA device")
    torch.set_num_threads(1)
    impl = p0.require_sources()
    model_receipt = r3._get_verified_model_receipt()
    restore_hub, runtime_info = r3._runtime_and_native_setup(torch, impl, model_receipt)
    cfg, scorer_profile = r3._native_config(
        p0, __import__("experiments.calibration.profiles", fromlist=["NativeScorerConfig"]), r1)
    backbone = None
    label_reader = None
    current_position = -1
    score_path = category_output / "stage_a_scores.jsonl"
    feedback_path = category_output / "private_feedback.jsonl"
    timing = Counter()
    counters = Counter()
    updates: list[dict[str, Any]] = []
    b3_history: dict[str, list[dict[str, Any]]] = {name: [] for name in MAIN_ARMS}
    try:
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        started = time.perf_counter()
        model_started = time.perf_counter()
        backbone = impl.DINOv2MultiLayerBackbone(device="cuda")
        loaded_sha = r1._model_state_sha256(backbone.model)
        model_load_seconds = time.perf_counter() - model_started
        if loaded_sha != model_receipt.get("loaded_model_state_sha256"):
            raise ExperimentBError("Loaded backbone differs from verified model receipt")
        contract = r7._frozen_contract()
        info = contract["metadata"][category]
        events = r9.make_order_events(info["events"], "SHUFFLE0")
        if len(events) != EXPECTED_COUNTS[category]:
            raise ExperimentBError(f"{category}: frozen event count differs from R9/R10 P0")
        query_positions = tuple(int(v) for v in info["tape_receipt"]["positions"])
        if len(query_positions) != EXPECTED_QUERIES[category] or tuple(sorted(query_positions)) != query_positions:
            raise ExperimentBError("Frozen R9 Query Tape positions are malformed")
        query_set = set(query_positions)
        query_identity_set_sha = r7._sha256_json(sorted(
            str(events[position].image_identity_sha256) for position in query_positions))

        support_paths = [str(path) for path in info.get("support_paths", info["contract"]["support_paths"])]
        support_features = p0.encode_reference_features(impl, backbone, support_paths, cfg)
        impl.set_seed(0)
        bank = p0.build_vmb(impl, support_features, support_paths, cfg)
        fixed_receipt = r3._check_bank_receipt(
            bank, p0, info["fixed_support_sha256_expected"], r1)
        fixed_sha = str(fixed_receipt["actual_structured_sha256"])
        r3._validate_correction_config(bank)
        correction_cfg, correction_cbs = r6._configure_r6_correction_capacity(bank)
        if len(correction_cbs) != 1:
            raise ExperimentBError("Native C1 source did not allocate its one selected bank")
        base_cb = correction_cbs[0]
        native_states = _make_native_c1_states(bank, impl, correction_cfg, base_cb)
        writer = MemoryArm("greedy_coreset", bank,
            SimpleNamespace(flat=1024, stm=1024, ltm=0, buffer=0, min_events=2,
                            promote_every=16, cap_bytes=2 * 1024**3), seed=0)
        if any(int(cb.K) != 1024 or cb.size() != 0 for cb in dynamic_codebooks(bank)):
            raise ExperimentBError("Common native Greedy codebooks did not initialize empty")
        bank.use_defect_bank = False
        bank.corr_cfg.enabled = False
        if _fixed_bank_digest(bank) != fixed_sha:
            raise ExperimentBError("Immutable four-Support bank changed during setup")

        c64_items, c64_receipt = r9._make_r9_c64_items(
            category, info, p0, impl, backbone, cfg, r7, support_seed=0)
        if len(c64_items) != 64:
            raise ExperimentBError("Registered C64 cache must contain exactly 64 additional normals")
        c64_scoring_started = time.perf_counter()
        c64_groups = [str(item["source_group_sha256"]) for item in c64_items]
        if c64_groups != [str(value) for value in info["c64_groups"]]:
            raise ExperimentBError("Cached C64 Source Groups differ from frozen registration")
        c64_g, base_cache, c64_base_seconds = r7._score_c64_base(
            impl, bank, c64_items, cfg, r1, category)
        expected_c64 = [float(v) for v in info["c64_scores"]]
        if any(not r3._score_tolerance(a, b) for a, b in zip(c64_g, expected_c64)):
            raise ExperimentBError("Native initial C64 scores fail the frozen Stage05 receipt")
        theta0 = float(np.quantile(c64_g, .90, method="higher"))
        sigma0 = max(float(np.std(c64_g, ddof=1)), .02)
        if not math.isclose(theta0, float(info["theta_c64_q90"]), rel_tol=0, abs_tol=1e-12):
            raise ExperimentBError("Native C64 initial q90 differs from the registered receipt")
        c64_group_sha = str(info["frozen_info"]["c64"]["group_sha256"])
        c64_feature_cache = [item["features"][0] for item in c64_items]

        fixed_cb = bank.normal_fixed["p0"][0]["id"][0]
        if int(fixed_cb.size()) != 8192:
            raise ExperimentBError("External adapters require the realized registered 8192-prototype fixed bank")
        fixed_features = fixed_cb.protos[:int(fixed_cb.size())]
        ext1 = ExternalNormalMemory(fixed_features, category=category,
                                    experiment_seed=experiment_seed, device=bank.device)
        ext2 = ExternalNormalMemory(fixed_features, category=category,
                                    experiment_seed=experiment_seed, device=bank.device)
        if ext1.state_sha256 != ext2.state_sha256:
            raise ExperimentBError("External E1/E2 initial banks are not identical copies")

        knn = r10.SignedKNNFIFO(capacity=1024)
        source_groups: dict[str, dict[int, set[str]]] = {
            arm: native_states[arm]["sources"] for arm in ("N", "D", "ND")}
        greedy_sha = r1._memory_content_sha256(bank)
        digests: dict[str, str] = {"G": r9._scorer_digest(fixed_sha, greedy_sha)}
        for arm in ("N", "D", "ND"):
            digests[arm] = r9._state_digest_for_codebook(
                bank, fixed_sha, native_states[arm]["sources"],
                native_states[arm]["cfg"], native_states[arm]["cb"], exp2a, r1)
        digests["SignedKNN"] = state_digest(
            fixed_sha, greedy_sha, correction_sha=knn.state_sha256())
        digests["HITL_FP"] = state_digest(fixed_sha, ext1.state_sha256)
        digests["HITL_FP_FN"] = state_digest(fixed_sha, ext2.state_sha256)
        versions = {arm: 0 for arm in MAIN_ARMS}

        c64_scores: dict[str, list[float]] = {arm: list(map(float, c64_g))
                                               for arm in ("G", "N", "D", "ND")}
        for arm in ("N", "D", "ND"):
            empty_scores, _elapsed = r9._score_c64_for_codebook(
                impl, bank, r7, r1, exp2a, c64_items, base_cache, cfg,
                native_states[arm]["cfg"], native_states[arm]["cb"],
                native_states[arm]["sources"], category, 0)
            if any(not r3._score_tolerance(actual, expected)
                   for actual, expected in zip(empty_scores, c64_g)):
                raise ExperimentBError(f"Empty {arm} C1 C64 readout differs from common G")
        c64_scores["SignedKNN"], _knn_c64_s, _knn_exclusions = r10._score_c2_c64(
            impl, bank, c64_items, base_cache, knn, event_position=0)
        if any(not r3._score_tolerance(a, b) for a, b in zip(c64_scores["SignedKNN"], c64_g)):
            raise ExperimentBError("Empty SignedKNN C64 scores do not equal G")
        c64_external: dict[str, list[float]] = {"HITL_FP": [], "HITL_FP_FN": []}
        external_novelty: dict[str, dict[str, Any]] = {}
        for name, memory in (("HITL_FP", ext1), ("HITL_FP_FN", ext2)):
            for item in c64_items:
                result = memory.score(item["features"][0],
                    source_group_sha256=str(item["source_group_sha256"]), event_position=0)
                c64_external[name].append(float(result.image_score))
            external_novelty[name] = memory.calibrate_novelty(
                c64_feature_cache, c64_source_groups=c64_groups)
            c64_scores[name] = c64_external[name]

        controllers: dict[str, Any] = {}
        frozen_thresholds: dict[str, float] = {}
        for arm in MAIN_ARMS:
            scores = c64_scores[arm]
            initial_theta = float(np.quantile(scores, .90, method="higher"))
            frozen_thresholds[arm] = initial_theta
            controller = StateConsistentQuantile(
                initial_theta, delta=.01, min_normals=16, name=f"{arm}-B3")
            initial = r7._initialize_b3(controller, scores, c64_group_sha,
                                        digests[arm], initial_theta)
            controllers[arm] = controller
            b3_history[arm].append(initial)
        if not math.isclose(float(controllers["HITL_FP"].theta),
                            float(controllers["HITL_FP_FN"].theta), rel_tol=0, abs_tol=0):
            raise ExperimentBError("E1/E2 initial C64 q90 thresholds are not identical")
        native_states["N"]["controller"] = controllers["N"]
        native_states["D"]["controller"] = controllers["D"]
        native_states["ND"]["controller"] = controllers["ND"]

        label_reader = r7.IndexedQueriedLabelReader(
            info["json_path"], category, info["contract"]["category_json_sha256"],
            events, query_positions)
        score_gate = r7.ScoreDisclosureGate(lambda row: append_jsonl_fsync(score_path, row))
        feedback_digest = hashlib.sha256()
        query_groups: set[str] = set()
        counters["c64_rescore_calls"] = 64 * len(MAIN_ARMS)
        counters["novelty_recalculations"] = 2
        timing["novelty_initial_seconds"] = sum(float(x["seconds"]) for x in external_novelty.values())
        timing["c64_initial_scoring_seconds"] = max(
            0.0, time.perf_counter() - c64_scoring_started - timing["novelty_initial_seconds"])
        timing["encoding_seconds"] = 0.0
        timing["scoring_seconds"] = 0.0
        timing["update_seconds"] = 0.0
        timing["c64_rescore_seconds"] = 0.0

        for position, event in enumerate(events):
            current_position = position
            if time.perf_counter() - started > CATEGORY_DEADLINE_SECONDS:
                raise ExperimentBError("Category Stage A exceeded the frozen one-hour worker deadline")
            if not Path(event.image_path).is_file():
                raise ExperimentBError(f"{category}:{position}: registered TEST image is missing")
            torch.cuda.synchronize(); timer = time.perf_counter()
            image, features = p0.encode_query(impl, backbone, str(event.image_path), cfg)
            torch.cuda.synchronize(); timing["encoding_seconds"] += time.perf_counter() - timer
            counters["stream_image_encodings"] += 1
            if len(features) != 1 or tuple(features[0].shape) != (784, 4096) or features[0].dtype != torch.float32:
                raise ExperimentBError("One-pass TEST encoder returned a descriptor outside the native shape contract")
            torch.cuda.synchronize(); timer = time.perf_counter()
            base_map, top_ids, g_score, summary = r1._native_dense_score(impl, bank, image, features, cfg)
            top_ids = np.asarray(top_ids, dtype=np.int64)
            if len(top_ids) != 32 or not np.array_equal(
                    top_ids, np.asarray(summary["layer0"]["topk_patch_idx"], dtype=np.int64)):
                raise ExperimentBError("Native G Top-32 differs from the dense-score selection receipt")
            selected = impl.l2_normalize(features[0].to(bank.device))[
                torch.as_tensor(top_ids, device=bank.device, dtype=torch.long)]
            group_sha = str(event.source_group_sha256)
            scores: dict[str, float] = {"G": float(g_score)}
            diagnostics: dict[str, Any] = {}
            for arm in ("N", "D", "ND"):
                scores[arm], diagnostics[arm] = _score_c1(
                    impl, bank, r9, r1, exp2a, native_states[arm], selected,
                    group_sha, position, base_map, top_ids)
            selected_np = selected.detach().float().cpu().numpy()
            scores["SignedKNN"], _knn_delta, diagnostics["SignedKNN"] = r10._corrected_c2_score(
                base_map, top_ids, selected_np, knn, position=position, group_sha=group_sha)
            patch_features = features[0]
            ext_results: dict[str, Any] = {}
            for arm, memory in (("HITL_FP", ext1), ("HITL_FP_FN", ext2)):
                ext_score = memory.score(patch_features, source_group_sha256=group_sha,
                                         event_position=position)
                scores[arm] = float(ext_score.image_score)
                ext_results[arm] = ext_score
            diag_pos, pos_count = score_sign_readout(
                impl, bank, r1, native_states["ND"]["cfg"], native_states["ND"]["cb"],
                native_states["ND"]["sources"], selected, group_sha, position,
                base_map, top_ids, 1)
            diag_neg, neg_count = score_sign_readout(
                impl, bank, r1, native_states["ND"]["cfg"], native_states["ND"]["cb"],
                native_states["ND"]["sources"], selected, group_sha, position,
                base_map, top_ids, -1)
            torch.cuda.synchronize(); timing["scoring_seconds"] += time.perf_counter() - timer

            decisions: dict[str, Any] = {}
            predictions: dict[str, int] = {}
            for arm in MAIN_ARMS:
                controller = controllers[arm]
                decision = r7._predict_b3(
                    controller, position=position, event=event, score=scores[arm],
                    version=versions[arm], state_digest=digests[arm])
                decisions[arm] = {**decision,
                    "frozen_c64_q90_threshold": frozen_thresholds[arm],
                    "frozen_c64_q90_prediction": int(scores[arm] >= frozen_thresholds[arm])}
                predictions[arm] = int(decision["prediction"])
            queried = position in query_set
            row = {
                "schema": SCORE_SCHEMA, "run_id": run_id, "category": category,
                "position": position, "original_json_position": int(event.original_position),
                "event_id_sha256": str(event.event_id_sha256),
                "image_identity_sha256": str(event.image_identity_sha256),
                "source_group_sha256": group_sha, "camera_id": int(event.camera_id),
                "query_tape_decision": bool(queried),
                "query_identity_set_sha256": query_identity_set_sha,
                "query_tape_sha256": info["tape_receipt"]["tape_sha256"],
                "query_tape_positions_sha256": info["tape_receipt"]["positions_sha256"],
                "scores": scores, "decisions": decisions,
                "score_state_versions": dict(versions),
                "score_state_sha256": dict(digests),
                "fixed_support_sha256": fixed_sha,
                "diagnostics": {"ND_positive_only_readout": diag_pos,
                    "ND_negative_only_readout": diag_neg,
                    "ND_positive_anchor_count": pos_count,
                    "ND_negative_anchor_count": neg_count,
                    "G_global_offset": float(np.mean(np.asarray(c64_scores["ND"]) -
                                                       np.asarray(c64_scores["G"]))),
                    "G_global_offset_score": float(g_score + np.mean(
                        np.asarray(c64_scores["ND"]) - np.asarray(c64_scores["G"]))),
                    "external_anomaly_patches": {k: int(v.anomaly_patch) for k, v in ext_results.items()},
                    "source_eligible_prototypes": {k: int(v.eligible_prototypes) for k, v in ext_results.items()}},
            }
            if not score_row_is_label_blind(row):
                raise ExperimentBError("Attempted to seal a score row containing an outcome field")
            causal = CausalEventOrder(position)
            causal.seal_scores()
            causal.decide_query(queried)
            score_gate.seal(position, row)
            if not score_gate.is_sealed(position):
                raise ExperimentBError("ScoreDisclosureGate did not durably seal current predictions")

            if queried:
                # This is the only label reveal call. The R7 reader binds the official JSON hash,
                # position, event identity and the already-fsynced score row.
                label = int(causal.disclose(lambda: label_reader.label_for(
                    position, sealed_and_fsynced=score_gate.is_sealed(position))))
                if group_sha in query_groups:
                    raise ExperimentBError("Registered Query Tape contains a repeated Source Group")
                query_groups.add(group_sha)
                counters["queried_labels"] += 1
                counters["queried_normal"] += int(label == 0)
                counters["queried_defect"] += int(label == 1)
                rescore_seconds_before = float(timing.get("c64_rescore_seconds", 0.0))
                novelty_seconds_before = float(timing.get("novelty_recalculation_seconds", 0.0))
                g_prediction = int(g_score >= theta0)
                common_outcome = {(0, 0): "TN", (0, 1): "FP", (1, 0): "FN", (1, 1): "TP"}[
                    (label, g_prediction)]
                timer = time.perf_counter()
                before_digests = dict(digests)
                c1_feedback: dict[str, Any] = {}
                for arm in ("N", "D", "ND"):
                    signs = accepted_c1_signs(arm, label)
                    if not signs:
                        c1_feedback[arm] = {"changed": False, "reason": "sign excluded by factorial arm"}
                        continue
                    sign = signs[0]
                    c1_before = r9._state_digest_for_codebook(
                        bank, fixed_sha, native_states[arm]["sources"],
                        native_states[arm]["cfg"], native_states[arm]["cb"], exp2a, r1)
                    _result, layer_stats, trace = exp2a._capture_update(
                        impl, bank, r1, f"EXP-B-{arm}", native_states[arm]["cfg"],
                        native_states[arm]["cb"], features, top_ids, sign,
                        float(g_score), theta0, common_outcome, position,
                        native_states[arm]["sources"], group_sha)
                    c1_after = r9._state_digest_for_codebook(
                        bank, fixed_sha, native_states[arm]["sources"],
                        native_states[arm]["cfg"], native_states[arm]["cb"], exp2a, r1)
                    c1_feedback[arm] = {"changed": True, "state_changed": c1_before != c1_after,
                        "accepted_sign": sign,
                        "common_G_outcome": common_outcome,
                        "layer_integer_stats": {str(k): int(v) for k, v in layer_stats.items()
                            if isinstance(v, (int, np.integer))},
                        "trace_integer_fields": {str(k): int(v) for k, v in trace.items()
                            if isinstance(v, (int, np.integer))}}

                greedy_before = r1._memory_content_sha256(bank)
                audit_before = writer.audit()
                def greedy_update(*, position: int, feature_layers: Sequence[Any],
                                  candidate_ids: Any, score: float, theta: float,
                                  sigma: float) -> Mapping[str, Any]:
                    writer.update(impl, list(feature_layers), {0: candidate_ids.tolist()},
                                  position, score=score, theta=theta, sigma=sigma)
                    audit_after = writer.audit()
                    return {"accepted_patches_delta": int(audit_after["accepted_patches"] - audit_before["accepted_patches"]),
                            "skipped_patches_delta": int(audit_after["skipped_patches"] - audit_before["skipped_patches"])}
                writer_action = r1.common_c0_greedy_writer_action(
                    position=position, c0_score=float(g_score), theta=theta0, sigma=sigma0,
                    c0_prediction=g_prediction, candidate_ids=top_ids,
                    candidate_features=selected.detach().float().cpu().numpy(),
                    feature_layers=features, queried=True, disclosed_label=label,
                    eligibility=impl.select_informative_normal_write, update=greedy_update)
                greedy_changed = greedy_before != r1._memory_content_sha256(bank)
                if label == 1 and writer_action.get("action") != "verified_defect_no_normal_write":
                    raise ExperimentBError("Queried defect entered the shared normal Greedy memory")
                if writer_action.get("input_score_source") != "common_C0":
                    raise ExperimentBError("Shared Greedy writer did not use the sealed common-G inputs")

                knn_before = knn.state_sha256()
                knn_insert = knn.insert(selected_np, sign=(1 if label == 1 else -1),
                    source_group_sha256=group_sha, position=position)
                if knn.state_sha256() == knn_before:
                    raise ExperimentBError("R10 SignedKNN did not record queried feedback")

                external_feedback: dict[str, Any] = {}
                for arm, memory, surgery in (("HITL_FP", ext1, False), ("HITL_FP_FN", ext2, True)):
                    ext_feedback = memory.apply_feedback(
                        patch_features, source_group_sha256=group_sha,
                        event_position=position, queried=True, label=label,
                        arm_prediction=predictions[arm],
                        anomaly_patch=ext_results[arm].anomaly_patch,
                        enable_fn_surgery=surgery)
                    external_feedback[arm] = ext_feedback

                new_greedy_sha = r1._memory_content_sha256(bank)
                after_digests: dict[str, str] = {"G": r9._scorer_digest(fixed_sha, new_greedy_sha)}
                for arm in ("N", "D", "ND"):
                    after_digests[arm] = r9._state_digest_for_codebook(
                        bank, fixed_sha, native_states[arm]["sources"],
                        native_states[arm]["cfg"], native_states[arm]["cb"], exp2a, r1)
                after_digests["SignedKNN"] = state_digest(
                    fixed_sha, new_greedy_sha, correction_sha=knn.state_sha256())
                after_digests["HITL_FP"] = state_digest(fixed_sha, ext1.state_sha256)
                after_digests["HITL_FP_FN"] = state_digest(fixed_sha, ext2.state_sha256)
                changed = [arm for arm in MAIN_ARMS if after_digests[arm] != before_digests[arm]]
                for arm in changed:
                    versions[arm] += 1

                c64_timer = time.perf_counter()
                novelty_seconds_this_round = 0.0
                if greedy_changed:
                    c64_g, base_cache, _elapsed = r7._score_c64_base(
                        impl, bank, c64_items, cfg, r1, category)
                    counters["c64_rescore_calls"] += 64
                for arm in ("N", "D", "ND"):
                    if arm in changed:
                        values, _elapsed = r9._score_c64_for_codebook(
                            impl, bank, r7, r1, exp2a, c64_items, base_cache,
                            cfg, native_states[arm]["cfg"], native_states[arm]["cb"],
                            native_states[arm]["sources"], category, position + 1)
                        c64_scores[arm] = values
                        counters["c64_rescore_calls"] += 64
                if "G" in changed:
                    c64_scores["G"] = list(map(float, c64_g))
                if "SignedKNN" in changed:
                    values, _elapsed, _excluded = r10._score_c2_c64(
                        impl, bank, c64_items, base_cache, knn, event_position=position + 1)
                    c64_scores["SignedKNN"] = values
                    counters["c64_rescore_calls"] += 64
                for arm, memory in (("HITL_FP", ext1), ("HITL_FP_FN", ext2)):
                    if arm in changed:
                        values = []
                        for item in c64_items:
                            result = memory.score(item["features"][0],
                                source_group_sha256=str(item["source_group_sha256"]),
                                event_position=position + 1)
                            values.append(result.image_score)
                        c64_scores[arm] = values
                        novelty = memory.calibrate_novelty(
                            c64_feature_cache, c64_source_groups=c64_groups)
                        external_novelty[arm] = novelty
                        counters["c64_rescore_calls"] += 64
                        counters["novelty_recalculations"] += 1
                        novelty_seconds_this_round += float(novelty["seconds"])
                        timing["novelty_recalculation_seconds"] = timing.get(
                            "novelty_recalculation_seconds", 0.0) + float(novelty["seconds"])
                for arm in changed:
                    previous_version = versions[arm] - 1
                    transition = _refresh_threshold(
                        controllers[arm], c64_scores[arm], c64_group_sha,
                        before_digests[arm], after_digests[arm], previous_version)
                    if transition is None:
                        raise ExperimentBError(f"{arm}: changed scorer failed to refresh B3")
                    transition.update({"position": position, "apply_from_position": position + 1,
                                       "score_scale": "mean-nine Euclidean" if arm.startswith("HITL") else "native registered score"})
                    b3_history[arm].append(transition)
                timing["c64_rescore_seconds"] += max(
                    0.0, time.perf_counter() - c64_timer - novelty_seconds_this_round)
                digests = after_digests

                feedback = {"schema": "exp-b.private.feedback.v1", "category": category,
                    "position": position, "event_id_sha256": str(event.event_id_sha256),
                    "source_group_sha256": group_sha, "label": label,
                    "queried_after_score_fsync": score_gate.is_sealed(position),
                    "common_G_prediction": g_prediction, "common_G_outcome": common_outcome,
                    "native_c1_updates": c1_feedback, "common_greedy_action": dict(writer_action),
                    "shared_greedy_state_changed": bool(greedy_changed),
                    "signed_knn_insert": knn_insert, "external_updates": external_feedback,
                    "changed_arms": changed, "state_versions_after": dict(versions),
                    "state_digests_after": dict(digests)}
                append_jsonl_fsync(feedback_path, feedback)
                feedback_digest.update(canonical_json(feedback))
                updates.append({"position": position, "label": label,
                                "native_c1": c1_feedback, "external": external_feedback})
                update_elapsed = time.perf_counter() - timer
                rescore_delta = float(timing.get("c64_rescore_seconds", 0.0)) - rescore_seconds_before
                novelty_delta = float(timing.get("novelty_recalculation_seconds", 0.0)) - novelty_seconds_before
                timing["update_seconds"] += max(0.0, update_elapsed - rescore_delta - novelty_delta)
                causal.corrections_updated()
                causal.memory_updated()
                causal.finish()
            else:
                if any(key in row for key in ("label", "anomaly_class")):
                    raise ExperimentBError("Unqueried event acquired label data")
                causal.corrections_updated()
                causal.memory_updated()
                causal.finish()

            if (position + 1) % 256 == 0:
                append_jsonl_fsync(category_output / "progress.jsonl", {
                    "schema": "exp-b.stage-a.progress.v1", "category": category,
                    "sealed_through_position": position,
                    "query_feedback_count": int(counters["queried_labels"]),
                    "state_versions": dict(versions)})

        label_reader.finish()
        label_reader.close()
        label_reader = None
        if counters["stream_image_encodings"] != len(events):
            raise ExperimentBError("TEST event was not encoded exactly once")
        if counters["queried_labels"] != len(query_positions) or len(query_groups) != len(query_positions):
            raise ExperimentBError("Stage A did not consume the exact unique registered Query Tape")
        if _fixed_bank_digest(bank) != fixed_sha:
            raise ExperimentBError("Fixed Support bank changed during Stage A")
        scores_rows = _jsonl_rows(score_path)
        validate_score_rows(scores_rows, category=category,
                            expected_count=len(events), expected_queries=query_positions)
        import psutil as _psutil
        torch.cuda.synchronize()
        correction_resource = {}
        for arm in ("N", "D", "ND"):
            correction_resource[arm] = r7._mutable_state_bytes(
                bank, native_states[arm]["cb"], dynamic_codebooks)
        greedy_codebooks = dynamic_codebooks(bank)
        greedy_prototype_bytes = int(sum(
            cb.protos.numel() * cb.protos.element_size() for cb in greedy_codebooks))
        greedy_metadata_bytes = int(sum(
            value.numel() * value.element_size()
            for cb in greedy_codebooks for name, value in vars(cb).items()
            if name != "protos" and isinstance(value, torch.Tensor)))
        c1_source_metadata_bytes = int(sum(
            len(group.encode("ascii")) for arm in ("N", "D", "ND")
            for groups in native_states[arm]["sources"].values() for group in groups))
        completion = {
            "schema": "exp-b.stage-a.category-seal.v1", "status": "SEALED_COMPLETE",
            "run_id": run_id, "identity_sha256": identity_sha256,
            "source_commit_sha": source_commit, "category": category,
            "support_seed": 0, "stream_shuffle_seed": 0,
            "event_count": len(events), "query_count": len(query_positions),
            "query_positions": list(query_positions),
            "query_identity_set_sha256": query_identity_set_sha,
            "category_json_sha256": info["contract"]["category_json_sha256"],
            "fixed_support_sha256": fixed_sha,
            "c64_image_sha256": info["frozen_info"]["c64"]["image_sha256"],
            "c64_group_sha256": info["frozen_info"]["c64"]["group_sha256"],
            "event_identity_map_sha256": r7._sha256_json([
                {"position": int(e.position), "original_position": int(e.original_position),
                 "event_id_sha256": str(e.event_id_sha256),
                 "image_identity_sha256": str(e.image_identity_sha256),
                 "source_group_sha256": str(e.source_group_sha256),
                 "camera_id": str(e.camera_id)} for e in events]),
            "query_feedback_sha256": feedback_digest.hexdigest(),
            "score_rows_sha256": sha256_file(score_path),
            "feedback_rows_sha256": sha256_file(feedback_path),
            "feedback_count": int(counters["queried_labels"]),
            "queried_normal_count": int(counters["queried_normal"]),
            "queried_defect_count": int(counters["queried_defect"]),
            "query_group_count": len(query_groups),
            "score_state_versions_final": dict(versions),
            "score_state_sha256_final": dict(digests),
            "c1_occupancy_final": {arm: int(native_states[arm]["cb"].size()) for arm in ("N", "D", "ND")},
            "c1_capacity": 1024,
            "signed_knn": {"capacity": 1024, "k": 5, "amplitude": 0.02,
                           "insertions": int(knn.insertions), "evictions": int(knn.evictions),
                           "state_sha256": knn.state_sha256()},
            "external_memory": {"HITL_FP": ext1.resource_ledger(),
                                "HITL_FP_FN": ext2.resource_ledger()},
            "external_novelty_recalculations": int(counters["novelty_recalculations"]),
            "external_fn_surgery_calls": int(ext2.fn_surgery_attempts),
            "external_fn_prototypes_removed": int(ext2.support_removed + ext2.online_removed),
            "b3_threshold_history": b3_history,
            "external_novelty_threshold_final": {k: v for k, v in external_novelty.items()},
            "resource_bytes": {
                "fixed_reference_realized": int(fixed_cb.protos[:int(fixed_cb.size())].numel() * fixed_cb.protos.element_size()),
                "fixed_reference_allocated": int(fixed_cb.protos.numel() * fixed_cb.protos.element_size()),
                "greedy_prototypes": greedy_prototype_bytes,
                "greedy_tensor_metadata": greedy_metadata_bytes,
                "native_correction_by_arm": correction_resource,
                "c1_source_group_digest_metadata": c1_source_metadata_bytes,
                "signed_knn_arrays": int(knn.array_bytes),
                "external_hitl_fp": ext1.resource_ledger(),
                "external_hitl_fp_fn": ext2.resource_ledger(),
                "c64_read_only_cache": int(sum(np.asarray(v).nbytes for item in c64_items for v in item["features"])),
            },
            "timing_seconds": {**dict(timing), "wall_seconds": time.perf_counter() - started,
                               "model_load_seconds": model_load_seconds,
                               "c64_encoding_seconds": float(c64_receipt.get("encoding_seconds", 0.0))},
            "stream_image_encodings": int(counters["stream_image_encodings"]),
            "c64_rescore_calls": int(counters["c64_rescore_calls"]),
            "peak_cuda_allocated_bytes": int(torch.cuda.max_memory_allocated()),
            "peak_cuda_reserved_bytes": int(torch.cuda.max_memory_reserved()),
            "peak_host_rss_bytes": int(psutil.Process().memory_info().rss),
            "unqueried_test_labels_accessed": False,
            "test_masks_opened": False,
            "confirmation_data_accessed": False,
            "source_file_sha256": {
                "exp_b_signed_feedback.py": sha256_file(Path(__file__).resolve()),
                "exp_b_external_hitl.py": sha256_file(Path(__file__).with_name("exp_b_external_hitl.py")),
                "r9_correction_mechanism_falsification.py": sha256_file(Path(r9.__file__).resolve()),
                "r10_signed_knn_baseline.py": sha256_file(Path(r10.__file__).resolve()),
            },
            "runtime_profile": runtime_info,
            "scorer_profile": asdict(scorer_profile),
        }
        write_json_exclusive(category_output / "stage_a_seal.json", completion)
        return completion
    except BaseException as exc:
        failure = {"schema": "exp-b.failure.v1", "phase": "stage_a",
            "category": category, "position": current_position,
            "exception_type": type(exc).__name__, "message": str(exc),
            "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "score_rows_sha256": sha256_file(score_path) if score_path.exists() else None,
            "feedback_rows_sha256": sha256_file(feedback_path) if feedback_path.exists() else None}
        try:
            write_json_exclusive(category_output / "failure.json", failure)
        except FileExistsError:
            pass
        raise
    finally:
        if label_reader is not None:
            label_reader.close()
        if backbone is not None:
            del backbone
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
        r3._restore_cuda_visibility(inherited_visibility)


def _verify_historical_parity(category: str, rows: Sequence[Mapping[str, Any]],
                              references: Mapping[str, Any]) -> dict[str, Any]:
    """Compare new G/ND/SignedKNN against preflighted sealed R9/R10 trajectories."""
    from experiments.calibration import r10_signed_knn_baseline as r10

    r9refs = references["r9"]
    historical = r9refs["references"][category]["rows"]
    if len(historical) != len(rows):
        raise ExperimentBError("R9 parity population differs in length")
    differences = {"G": 0.0, "ND": 0.0}
    checked = 0
    for i, (actual, expected) in enumerate(zip(rows, historical)):
        for arm, key in (("G", "G1024"), ("ND", "GC1024")):
            if (actual.get("event_id_sha256") != expected.get("event_id_sha256") or
                    actual.get("image_identity_sha256") != expected.get("image_identity_sha256") or
                    actual.get("source_group_sha256") != expected.get("source_group_sha256") or
                    actual.get("query_tape_decision") != expected.get("query_tape_decision")):
                raise ExperimentBError(f"R9 {arm} event/query identity mismatch at {category}:{i}")
            error = abs(float(actual["scores"][arm]) - float(expected["scores"][key]))
            differences[arm] = max(differences[arm], error)
            if not math.isclose(float(actual["scores"][arm]), float(expected["scores"][key]),
                                rel_tol=r10.RTOL, abs_tol=r10.ATOL):
                raise ExperimentBError(f"R9 {arm} native score parity failed at {category}:{i}")
        if actual["score_state_sha256"]["G"] != expected["score_state_sha256"]["G1024"]:
            raise ExperimentBError(f"R9 shared Greedy pre-event state mismatch at {category}:{i}")
        for arm, historical_arm in (("G", "G1024"), ("ND", "GC1024")):
            actual_decision = actual["decisions"][arm]
            expected_decision = expected["decisions"][f"{historical_arm}-B3"]
            if (int(actual_decision["prediction"]) != int(expected_decision["prediction"]) or
                    not math.isclose(float(actual_decision["threshold"]),
                                     float(expected_decision["threshold"]), rel_tol=0, abs_tol=1e-12)):
                raise ExperimentBError(f"R9 {arm} B3 decision/threshold parity failed at {category}:{i}")
        checked += 1
    # R10's Stage A rows are output-sealed and carry the same G state plus C2 score.
    r10_rows = references["r10_rows"][category]
    if len(r10_rows) != len(rows):
        raise ExperimentBError("R10 parity population differs in length")
    max_knn = 0.0
    for i, (actual, expected) in enumerate(zip(rows, r10_rows)):
        if (actual.get("event_id_sha256") != expected.get("event_id_sha256") or
                actual.get("image_identity_sha256") != expected.get("image_identity_sha256") or
                actual.get("source_group_sha256") != expected.get("source_group_sha256") or
                actual.get("query_tape_decision") != expected.get("query_tape_decision")):
            raise ExperimentBError(f"R10 SignedKNN event/query identity mismatch at {category}:{i}")
        if actual["score_state_sha256"]["G"] != expected["score_state_sha256"]["G1024"]:
            raise ExperimentBError(f"R10 shared Greedy pre-event state mismatch at {category}:{i}")
        if actual["score_state_sha256"]["SignedKNN"] != expected["score_state_sha256"]["C2-1024"]:
            raise ExperimentBError(f"R10 SignedKNN pre-event state mismatch at {category}:{i}")
        expected_b3 = expected["decisions"]["C2-1024-B3"]
        if (int(actual["decisions"]["SignedKNN"]["prediction"]) != int(expected_b3["prediction"]) or
                not math.isclose(float(actual["decisions"]["SignedKNN"]["threshold"]),
                                 float(expected_b3["threshold"]), rel_tol=0, abs_tol=1e-12)):
            raise ExperimentBError(f"R10 SignedKNN B3 decision/threshold parity failed at {category}:{i}")
        error = abs(float(actual["scores"]["SignedKNN"]) - float(expected["scores"]["C2-1024"]))
        max_knn = max(max_knn, error)
        if not math.isclose(float(actual["scores"]["SignedKNN"]),
                            float(expected["scores"]["C2-1024"]),
                            rel_tol=r10.RTOL, abs_tol=r10.ATOL):
            raise ExperimentBError(f"R10 SignedKNN parity failed at {category}:{i}")
    return {"status": "PASS", "events_checked": checked,
            "maximum_absolute_score_error": {**differences, "SignedKNN": max_knn},
            "r9_score_sha256": r9refs["references"][category]["score_sha256"],
            "r10_score_sha256": references["receipts"][category]["r10_score_rows_sha256"],
            "historical_outputs_used_as_parity_only": True}


def _binary_metrics(labels: Sequence[int], scores: Sequence[float],
                    predictions: Sequence[int]) -> dict[str, Any]:
    from sklearn.metrics import average_precision_score, roc_auc_score

    y = np.asarray(labels, dtype=np.int8)
    s = np.asarray(scores, dtype=np.float64)
    p = np.asarray(predictions, dtype=np.int8)
    if y.shape != s.shape or y.shape != p.shape or not np.isfinite(s).all():
        raise ExperimentBError("Stage B metric vectors are malformed")
    tn = int(np.sum((y == 0) & (p == 0)))
    fp = int(np.sum((y == 0) & (p == 1)))
    fn = int(np.sum((y == 1) & (p == 0)))
    tp = int(np.sum((y == 1) & (p == 1)))
    fpr = fp / (fp + tn) if fp + tn else None
    fnr = fn / (fn + tp) if fn + tp else None
    tnr = tn / (tn + fp) if tn + fp else None
    tpr = tp / (tp + fn) if tp + fn else None
    ranking = {"auroc": None, "average_precision": None}
    if len(np.unique(y)) == 2:
        ranking = {"auroc": float(roc_auc_score(y, s)),
                   "average_precision": float(average_precision_score(y, s))}
    return {**ranking, "balanced_accuracy": (tpr + tnr) / 2 if tpr is not None and tnr is not None else None,
            "ber": 1.0 - (tpr + tnr) / 2 if tpr is not None and tnr is not None else None,
            "fpr": fpr, "fnr": fnr, "tp": tp, "tn": tn, "fp": fp, "fn": fn,
            "event_count": int(len(y))}


def _operating_metrics(labels: Sequence[int], predictions: Sequence[int]) -> dict[str, Any]:
    y = np.asarray(labels, dtype=np.int8)
    p = np.asarray(predictions, dtype=np.int8)
    tn = int(np.sum((y == 0) & (p == 0)))
    fp = int(np.sum((y == 0) & (p == 1)))
    fn = int(np.sum((y == 1) & (p == 0)))
    tp = int(np.sum((y == 1) & (p == 1)))
    fpr = fp / (fp + tn) if fp + tn else None
    fnr = fn / (fn + tp) if fn + tp else None
    tnr = tn / (tn + fp) if tn + fp else None
    tpr = tp / (tp + fn) if tp + fn else None
    return {"ber": 1 - (tpr + tnr) / 2 if tpr is not None and tnr is not None else None,
            "fpr": fpr, "fnr": fnr}


def _source_group_bootstrap(labels: Sequence[int], rows: Sequence[Mapping[str, Any]],
                            *, seed: int, replicates: int = 500) -> dict[str, Any]:
    """Paired cluster bootstrap resampling whole five-view Source Groups."""
    from sklearn.metrics import average_precision_score, roc_auc_score

    y = np.asarray(labels, dtype=np.int8)
    groups = np.asarray([str(row["source_group_sha256"]) for row in rows], dtype=object)
    unique = np.asarray(sorted(set(groups.tolist())), dtype=object)
    if y.shape != groups.shape or len(unique) == 0:
        raise ExperimentBError("Source Group bootstrap identities do not align with score rows")
    rng = np.random.default_rng(int(seed))
    names = ("ND_minus_N", "D_minus_G", "N_minus_G", "factorial_sign_interaction")
    draws = {name: {metric: [] for metric in ("auroc", "average_precision")} for name in names}
    arm_scores = {arm: np.asarray([float(row["scores"][arm]) for row in rows], dtype=np.float64)
                  for arm in ("G", "N", "D", "ND")}
    for _ in range(int(replicates)):
        selected = rng.choice(unique, size=len(unique), replace=True)
        multiplicity = {group: int(np.sum(selected == group)) for group in unique}
        weights = np.asarray([multiplicity[group] for group in groups], dtype=np.float64)
        active = weights > 0
        if len(np.unique(y[active])) != 2:
            continue
        metrics: dict[str, dict[str, float]] = {}
        for arm in ("G", "N", "D", "ND"):
            metrics[arm] = {
                "auroc": float(roc_auc_score(y[active], arm_scores[arm][active],
                                               sample_weight=weights[active])),
                "average_precision": float(average_precision_score(
                    y[active], arm_scores[arm][active], sample_weight=weights[active]))}
        for metric in ("auroc", "average_precision"):
            contrasts = {
                "ND_minus_N": metrics["ND"][metric] - metrics["N"][metric],
                "D_minus_G": metrics["D"][metric] - metrics["G"][metric],
                "N_minus_G": metrics["N"][metric] - metrics["G"][metric],
                "factorial_sign_interaction": (metrics["ND"][metric] - metrics["N"][metric] -
                                               metrics["D"][metric] + metrics["G"][metric]),
            }
            for name, value in contrasts.items():
                draws[name][metric].append(float(value))
    summaries = {}
    for name, metrics in draws.items():
        summaries[name] = {}
        for metric, values in metrics.items():
            summaries[name][metric] = {
                "ci_95_percentile": ([float(np.quantile(values, .025)),
                                      float(np.quantile(values, .975))] if values else None),
                "valid_replicates": len(values)}
    return {"method": "paired bootstrap of whole official Source Groups; views remain clustered",
            "seed": int(seed), "replicates_requested": int(replicates),
            "valid_replicates": min((len(draws[n]["auroc"]) for n in names), default=0),
            "contrasts": summaries}


def _stage_b_category(category: str, category_dir: Path,
                      contract: Mapping[str, Any]) -> dict[str, Any]:
    """Open outcomes only after validating every category's Stage A seal globally."""
    from experiments.calibration import r3_fullstream_2x2 as r3
    rows = _jsonl_rows(category_dir / "stage_a_scores.jsonl")
    seal = json.loads((category_dir / "stage_a_seal.json").read_text(encoding="utf-8"))
    expected_q = tuple(int(v) for v in contract["metadata"][category]["tape_receipt"]["positions"])
    validate_score_rows(rows, category=category, expected_count=EXPECTED_COUNTS[category],
                        expected_queries=expected_q)
    if seal.get("status") != "SEALED_COMPLETE" or seal.get("score_rows_sha256") != sha256_file(category_dir / "stage_a_scores.jsonl"):
        raise ExperimentBError(f"{category}: Stage A category seal failed before Stage B")
    # This outcome projection is deliberately the first full TEST label read.
    labels_original = r3._read_stage_b_label_projection(
        contract["metadata"][category]["json_path"], EXPECTED_COUNTS[category])
    events = contract["metadata"][category]["events"]
    order_events = __import__("experiments.calibration.r9_correction_mechanism_falsification",
                              fromlist=["make_order_events"]).make_order_events(events, "SHUFFLE0")
    labels = np.asarray([int(labels_original[int(event.original_position)]) for event in order_events], dtype=np.int8)
    out: dict[str, Any] = {"event_count": len(labels), "query_count": len(expected_q),
                           "query_label_composition": {"normal": 0, "defect": 0},
                           "arms": {}, "worst_complete_512_windows": {},
                           "first_legal_defect_query_position": None}
    queried_rows = [row for row in rows if row["query_tape_decision"]]
    query_labels = [int(labels[int(row["position"])]) for row in queried_rows]
    out["query_label_composition"] = {"normal": int(query_labels.count(0)),
                                      "defect": int(query_labels.count(1))}
    defect_queries = [int(row["position"]) for row in queried_rows
                      if int(labels[int(row["position"])]) == 1]
    out["first_legal_defect_query_position"] = min(defect_queries) if defect_queries else None
    for arm in MAIN_ARMS:
        scores = [float(row["scores"][arm]) for row in rows]
        preds = [int(row["decisions"][arm]["prediction"]) for row in rows]
        out["arms"][arm] = _binary_metrics(labels, scores, preds)
        out["arms"][arm]["frozen_c64_q90_operating"] = _operating_metrics(
            labels, [int(row["decisions"][arm]["frozen_c64_q90_prediction"]) for row in rows])
        windows = []
        for start in range(0, len(labels) - 512 + 1):
            stop = start + 512
            metric = _operating_metrics(labels[start:stop], preds[start:stop])
            if metric["ber"] is not None:
                windows.append({"start": start, "stop": stop, "ber": metric["ber"],
                                "fpr": metric["fpr"], "fnr": metric["fnr"]})
        out["worst_complete_512_windows"][arm] = max(windows, key=lambda v: v["ber"]) if windows else None
    contrasts = {
        "ND_minus_N": ("ND", "N"), "D_minus_G": ("D", "G"),
        "N_minus_G": ("N", "G"), "ND_minus_SignedKNN": ("ND", "SignedKNN"),
        "HITL_FP_FN_minus_HITL_FP": ("HITL_FP_FN", "HITL_FP"),
        "ND_minus_HITL_FP_FN_method_level": ("ND", "HITL_FP_FN"),
    }
    out["contrasts"] = {}
    for name, (left, right) in contrasts.items():
        out["contrasts"][name] = {metric: (
            out["arms"][left][metric] - out["arms"][right][metric]
            if out["arms"][left][metric] is not None and out["arms"][right][metric] is not None else None)
            for metric in ("auroc", "average_precision", "balanced_accuracy", "ber", "fpr", "fnr")}
    out["contrasts"]["factorial_sign_interaction"] = {
        metric: (out["arms"]["ND"][metric] - out["arms"]["N"][metric] -
                 out["arms"]["D"][metric] + out["arms"]["G"][metric]
                 if all(out["arms"][a][metric] is not None for a in ("ND", "N", "D", "G")) else None)
        for metric in ("auroc", "average_precision", "balanced_accuracy", "ber", "fpr", "fnr")}
    feedback = _jsonl_rows(category_dir / "private_feedback.jsonl")
    by_pos = {int(row["position"]): row for row in feedback}
    if set(by_pos) != {int(row["position"]) for row in queried_rows}:
        raise ExperimentBError(f"{category}: private queried feedback does not match sealed tape")
    for row in queried_rows:
        if int(by_pos[int(row["position"])]["label"]) != int(labels[int(row["position"])]):
            raise ExperimentBError(f"{category}: disclosed queried label differs from official JSON projection")
    c1_activation: dict[str, Any] = {}
    for arm in ("N", "D", "ND"):
        entries = [by_pos[int(row["position"])].get("native_c1_updates", {}).get(arm, {})
                   for row in queried_rows]
        changed_positions = [int(row["position"]) for row, entry in zip(queried_rows, entries)
                             if entry.get("state_changed") is True]
        c1_activation[arm] = {
            "eligible_feedback_events": sum(bool(entry.get("changed")) for entry in entries),
            "state_change_events": len(changed_positions),
            "first_activation_position": min(changed_positions) if changed_positions else None,
            "final_occupancy": int(seal.get("c1_occupancy_final", {}).get(arm, 0)),
        }
    external_updates = {arm: {"fp_insert_events": 0, "inserted_prototypes": 0,
                              "fn_surgery_events": 0, "removed_prototypes": 0}
                        for arm in ("HITL_FP", "HITL_FP_FN")}
    for feedback_row in feedback:
        for arm, update in feedback_row.get("external_updates", {}).items():
            if arm not in external_updates:
                continue
            fp = update.get("fp", {})
            fn = update.get("fn", {})
            external_updates[arm]["fp_insert_events"] += int(fp.get("changed") is True)
            external_updates[arm]["inserted_prototypes"] += int(fp.get("inserted", 0))
            external_updates[arm]["fn_surgery_events"] += int(fn.get("changed") is True)
            external_updates[arm]["removed_prototypes"] += int(fn.get("removed_count", 0))
    out["state_activation_and_memory_changes"] = {
        "native_c1": c1_activation,
        "external_memory_edits": external_updates,
        "shared_greedy_write_events": sum(bool(row.get("shared_greedy_state_changed")) for row in feedback),
        "signed_knn_feedback_events": len(feedback),
    }
    out["source_group_cluster_bootstrap"] = _source_group_bootstrap(
        labels, rows, seed=20261108 + CATEGORIES.index(category), replicates=500)
    diagnostic_scores = {
        "ND_positive_only_readout": [float(row["diagnostics"]["ND_positive_only_readout"]) for row in rows],
        "ND_negative_only_readout": [float(row["diagnostics"]["ND_negative_only_readout"]) for row in rows],
        "G_global_offset": [float(row["diagnostics"]["G_global_offset_score"]) for row in rows],
    }
    from sklearn.metrics import average_precision_score, roc_auc_score
    out["read_only_diagnostic_ranking"] = {}
    for name, values in diagnostic_scores.items():
        out["read_only_diagnostic_ranking"][name] = {
            "auroc": float(roc_auc_score(labels, values)) if len(np.unique(labels)) == 2 else None,
            "average_precision": float(average_precision_score(labels, values)) if len(np.unique(labels)) == 2 else None,
        }
    out["read_only_diagnostic_contrasts_vs_ND"] = {
        name: {metric: (out["read_only_diagnostic_ranking"][name][metric] -
                        out["arms"]["ND"][metric]
                        if out["read_only_diagnostic_ranking"][name][metric] is not None and
                           out["arms"]["ND"][metric] is not None else None)
               for metric in ("auroc", "average_precision")}
        for name in diagnostic_scores}
    # Descriptive before/after first-defect-query summaries do not change the fixed-score ranking estimand.
    first = out["first_legal_defect_query_position"]
    out["before_after_first_defect_query"] = {}
    if first is None:
        spans = {"before": (0, None), "after": (None, len(labels))}
    else:
        first_feedback_effective = int(first) + 1
        spans = {"before": (0, first_feedback_effective),
                 "after": (first_feedback_effective, len(labels))}
    for arm in MAIN_ARMS:
        out["before_after_first_defect_query"][arm] = {}
        for name, (lo, hi) in spans.items():
            if lo is None or hi is None or hi <= lo:
                out["before_after_first_defect_query"][arm][name] = None
                continue
            out["before_after_first_defect_query"][arm][name] = _binary_metrics(
                labels[lo:hi], [float(row["scores"][arm]) for row in rows[lo:hi]],
                [int(row["decisions"][arm]["prediction"]) for row in rows[lo:hi]])
    return out


def _stage_b_global(run_root: Path, stage_a_seal: Mapping[str, Any],
                    contract: Mapping[str, Any]) -> dict[str, Any]:
    if stage_a_seal.get("status") != "SEALED_COMPLETE" or set(stage_a_seal.get("categories", {})) != set(CATEGORIES):
        raise ExperimentBError("Stage B refuses labels until all four Stage A category seals pass")
    import torch
    if torch.cuda.is_initialized() or torch.cuda.device_count() != 0:
        raise ExperimentBError("Stage B must run CPU-only after Stage A has exited")
    categories = {}
    for category in CATEGORIES:
        category_dir = run_root / "stage_a" / category
        seal = json.loads((category_dir / "stage_a_seal.json").read_text(encoding="utf-8"))
        global_record = stage_a_seal["categories"][category]
        if (seal.get("status") != "SEALED_COMPLETE" or
                seal.get("score_rows_sha256") != global_record.get("score_rows_sha256") or
                sha256_file(category_dir / "stage_a_scores.jsonl") != global_record.get("score_rows_sha256")):
            raise ExperimentBError(f"{category}: global Stage A seal does not match category output")
        categories[category] = _stage_b_category(category, category_dir, contract)
    macro = {}
    metrics = ("auroc", "average_precision", "balanced_accuracy", "ber", "fpr", "fnr")
    for arm in MAIN_ARMS:
        macro[arm] = {metric: (float(np.mean([categories[c]["arms"][arm][metric]
                    for c in CATEGORIES])) if all(categories[c]["arms"][arm][metric] is not None
                    for c in CATEGORIES) else None) for metric in metrics}
    contrasts = {}
    for name in next(iter(categories.values()))["contrasts"]:
        contrasts[name] = {metric: (float(np.mean([categories[c]["contrasts"][name][metric]
            for c in CATEGORIES])) if all(categories[c]["contrasts"][name][metric] is not None
            for c in CATEGORIES) else None) for metric in metrics}
    manifest = json.loads((run_root / "run_manifest.json").read_text(encoding="utf-8"))
    return {"schema": "exp-b.stage-b.result.v1", "status": "SEALED_COMPLETE",
            "classification": "development experiment; no scientific GO/NO-GO implied",
            "stage_a_seal_sha256": sha256_file(run_root / "stage_a" / "stage_a_seal.json"),
            "historical_r9_beta_permutation_evidence": manifest.get("historical_beta_permutation"),
            "category_results": categories, "category_equal_macro": macro,
            "category_equal_macro_contrasts": contrasts,
            "metric_direction": {"higher_preferred": ["auroc", "average_precision", "balanced_accuracy"],
                                 "lower_preferred": ["ber", "fpr", "fnr"]}}


def _verify_full_stage_a(run_root: Path, *, source_commit: str,
                         identity_sha256: str) -> dict[str, Any]:
    manifest = json.loads((run_root / "run_manifest.json").read_text(encoding="utf-8"))
    payload = {key: value for key, value in manifest.items() if key != "identity_sha256"}
    if (manifest.get("identity_sha256") != identity_sha256 or
            sha256_bytes(canonical_json(payload)) != identity_sha256 or
            manifest.get("source_commit_sha") != source_commit or
            manifest.get("source_fingerprints") != _source_fingerprints()):
        raise ExperimentBError("Run identity, source commit, or source fingerprint changed")
    category_receipts = {}
    for category in CATEGORIES:
        leaf = run_root / "stage_a" / category
        seal_path = leaf / "stage_a_seal.json"
        score_path = leaf / "stage_a_scores.jsonl"
        feedback_path = leaf / "private_feedback.jsonl"
        supervisor_path = leaf / "supervisor_receipt.json"
        if not all(path.is_file() for path in (seal_path, score_path, feedback_path, supervisor_path)):
            raise ExperimentBError(f"{category}: mandatory Stage A artifact missing")
        supervisor_receipt = json.loads(supervisor_path.read_text(encoding="utf-8"))
        if (supervisor_receipt.get("schema") != "exp-b.stage-a.supervisor-receipt.v1" or
                supervisor_receipt.get("status") != "COMPLETED" or
                supervisor_receipt.get("child_exit_code") != 0 or
                supervisor_receipt.get("termination_confirmed") is not True or
                supervisor_receipt.get("active_job_processes") != 0 or
                supervisor_receipt.get("run_id") != run_root.name or
                supervisor_receipt.get("category") != category or
                supervisor_receipt.get("source_commit_sha") != source_commit or
                supervisor_receipt.get("identity_sha256") != identity_sha256):
            raise ExperimentBError(f"{category}: Stage A process-tree receipt failed")
        seal = json.loads(seal_path.read_text(encoding="utf-8"))
        rows = _jsonl_rows(score_path)
        registration = manifest.get("registrations", {}).get(category, {})
        q = tuple(int(v) for v in registration.get("query_positions", ()))
        validate_score_rows(rows, category=category, expected_count=EXPECTED_COUNTS[category],
                            expected_queries=q)
        if (seal.get("status") != "SEALED_COMPLETE" or seal.get("run_id") != run_root.name or
                seal.get("identity_sha256") != identity_sha256 or
                seal.get("source_commit_sha") != source_commit or
                seal.get("query_count") != EXPECTED_QUERIES[category] or
                tuple(seal.get("query_positions", ())) != q or
                seal.get("query_identity_set_sha256") != registration.get("query_identity_set_sha256") or
                seal.get("category_json_sha256") != registration.get("category_json_sha256") or
                seal.get("fixed_support_sha256") != registration.get("fixed_support_sha256") or
                seal.get("c64_image_sha256") != registration.get("c64_image_sha256") or
                seal.get("c64_group_sha256") != registration.get("c64_group_sha256") or
                seal.get("event_identity_map_sha256") != registration.get("event_identity_map_sha256") or
                seal.get("score_rows_sha256") != sha256_file(score_path) or
                seal.get("feedback_rows_sha256") != sha256_file(feedback_path) or
                seal.get("unqueried_test_labels_accessed") is not False or
                seal.get("test_masks_opened") is not False):
            raise ExperimentBError(f"{category}: Stage A category seal/source/checksum failed")
        category_receipts[category] = {"seal_sha256": sha256_file(seal_path),
            "score_rows_sha256": seal["score_rows_sha256"],
            "feedback_rows_sha256": seal["feedback_rows_sha256"],
            "event_count": seal["event_count"], "query_count": seal["query_count"]}
    global_seal = {"schema": "exp-b.stage-a.global-seal.v1", "status": "SEALED_COMPLETE",
        "run_id": run_root.name, "identity_sha256": identity_sha256,
        "source_commit_sha": source_commit, "categories": category_receipts,
        "event_count": sum(EXPECTED_COUNTS.values()),
        "query_count": sum(EXPECTED_QUERIES.values()),
        "unqueried_test_labels_accessed": False, "test_masks_opened": False,
        "sealed_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    global_path = run_root / "stage_a" / "stage_a_seal.json"
    if global_path.exists():
        existing = json.loads(global_path.read_text(encoding="utf-8"))
        if (existing.get("status") != "SEALED_COMPLETE" or
                existing.get("identity_sha256") != identity_sha256 or
                existing.get("source_commit_sha") != source_commit or
                existing.get("categories") != category_receipts):
            raise ExperimentBError("Existing global Stage A seal differs from verified category seals")
        return existing
    write_json_exclusive(global_path, global_seal)
    return global_seal


def _run_supervised_stage_a(category: str, run_root: Path, *, run_id: str,
                            authorization_file: Path, source_commit: str,
                            identity_sha256: str, experiment_seed: int) -> dict[str, Any]:
    """Run one category in a Windows Job Object with a fixed deadline."""
    from dataclasses import asdict
    from experiments.calibration import r1_runtime_supervisor as supervisor
    from experiments.calibration import r3_fullstream_2x2 as r3

    env = r3._child_environment(os.environ, "STAGE_A", AUTHORIZED_CUDA_VISIBILITY)
    command = [sys.executable, "-m", "experiments.calibration.exp_b_signed_feedback",
               "--stage-a-child", "--run-id", run_id,
               "--authorization-file", str(authorization_file.resolve()),
               "--category", category, "--experiment-seed", str(int(experiment_seed))]
    result = supervisor.run_supervised_child(
        command, timeout_seconds=CATEGORY_DEADLINE_SECONDS, env=env, cwd=ROOT)
    receipt = asdict(result)
    receipt.update({"schema": "exp-b.stage-a.supervisor-receipt.v1",
                    "category": category, "run_id": run_id,
                    "source_commit_sha": source_commit,
                    "identity_sha256": identity_sha256,
                    "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
    category_output = run_root / "stage_a" / category
    if (result.status != "COMPLETED" or result.child_exit_code != 0 or
            not result.termination_confirmed or result.active_job_processes != 0):
        failure_candidates = (category_output / "startup_failure.json",
                              category_output / "failure.json",
                              category_output.parent / f"{category}.startup_failure.json")
        child_failure = None
        for failure_path in failure_candidates:
            if failure_path.is_file():
                failure_row = json.loads(failure_path.read_text(encoding="utf-8"))
                child_failure = {"receipt": str(failure_path.relative_to(run_root)),
                    "exception_type": failure_row.get("exception_type"),
                    "message": failure_row.get("message")}
                break
        raise ExperimentBError(
            f"{category}: supervised Stage A child did not complete cleanly: "
            f"supervisor={receipt}; child_failure={child_failure}")
    seal_path = category_output / "stage_a_seal.json"
    if not seal_path.is_file():
        raise ExperimentBError(f"{category}: supervised Stage A child exited without a category seal")
    write_json_exclusive(category_output / "supervisor_receipt.json", receipt)
    return json.loads(seal_path.read_text(encoding="utf-8"))


def _run_supervised_stage_b(run_id: str, authorization_file: Path) -> None:
    """Evaluate the sealed run in a fresh CPU-only, bounded child process."""
    from dataclasses import asdict
    from experiments.calibration import r1_runtime_supervisor as supervisor
    from experiments.calibration import r3_fullstream_2x2 as r3

    env = r3._child_environment(os.environ, "STAGE_B", None)
    command = [sys.executable, "-m", "experiments.calibration.exp_b_signed_feedback",
               "--stage-b-child", "--run-id", run_id,
               "--authorization-file", str(authorization_file.resolve())]
    result = supervisor.run_supervised_child(
        command, timeout_seconds=STAGE_B_DEADLINE_SECONDS, env=env, cwd=ROOT)
    if (result.status != "COMPLETED" or result.child_exit_code != 0 or
            not result.termination_confirmed or result.active_job_processes != 0):
        raise ExperimentBError(f"CPU Stage B child did not complete cleanly: {asdict(result)}")


def _record_stage_a_child_failure(run_id: str, category: str,
                                  exc: BaseException) -> Path | None:
    """Persist a child startup traceback without replacing any prior receipt."""
    if (not re.fullmatch(r"expb_[a-z0-9][a-z0-9_-]{5,63}", str(run_id)) or
            category not in CATEGORIES):
        return None
    stage_root = RUN_ROOT / run_id / "stage_a"
    if not stage_root.is_dir():
        return None
    category_root = stage_root / category
    failure_path = (category_root / "startup_failure.json" if category_root.is_dir() else
                    stage_root / f"{category}.startup_failure.json")
    failure = {
        "schema": "exp-b.stage-a.failure.v1", "status": "INCOMPLETE",
        "phase": "stage_a_child_startup", "run_id": run_id, "category": category,
        "exception_type": type(exc).__name__, "message": str(exc),
        "traceback": "".join(traceback.format_exception(
            type(exc), exc, exc.__traceback__)),
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    try:
        write_json_exclusive(failure_path, failure)
    except FileExistsError:
        # Failure records are immutable. Keep the first receipt intact.
        pass
    return failure_path


def _run_stage_a_child(args: Any, authorized_cuda_visibility: str | None) -> None:
    """Validate and run a category while preserving every startup traceback."""
    try:
        from experiments.calibration import r1_runtime_supervisor as supervisor
        if not supervisor.current_process_is_in_job():
            raise ExperimentBError(
                "Stage A category child must be contained in the authorized Windows Job Object")
        if args.category not in CATEGORIES:
            raise ExperimentBError("Stage A category child requires a registered category")
        auth = _check_run_authorization(args.authorization_file, run_id=args.run_id)
        run_root = (RUN_ROOT / args.run_id).resolve(strict=True)
        manifest = json.loads((run_root / "run_manifest.json").read_text(encoding="utf-8"))
        payload = {key: value for key, value in manifest.items() if key != "identity_sha256"}
        if (manifest.get("authorization", {}).get("authorization_file_sha256") != auth["authorization_file_sha256"] or
                manifest.get("source_commit_sha") != _git(["rev-parse", "HEAD"]) or
                sha256_bytes(canonical_json(payload)) != manifest.get("identity_sha256") or
                manifest.get("run_id") != args.run_id or args.category not in manifest.get("categories", ())):
            raise ExperimentBError(
                "Stage A child authorization, source revision or run identity differs from manifest")
        _verify_source_fingerprints(manifest["source_fingerprints"])
        category_output = run_root / "stage_a" / args.category
        if category_output.exists():
            raise ExperimentBError("Stage A child requires a fresh registered category output leaf")
        _stage_a_category(
            args.category, category_output, run_id=args.run_id,
            source_commit=manifest["source_commit_sha"],
            identity_sha256=manifest["identity_sha256"],
            authorized_cuda_visibility=authorized_cuda_visibility,
            experiment_seed=args.experiment_seed)
    except BaseException as exc:
        _record_stage_a_child_failure(args.run_id, str(args.category or ""), exc)
        raise


def run_full(*, run_id: str, authorization_file: Path, experiment_seed: int = 0) -> Path:
    """Explicitly authorized, non-resumable future full Stage A then Stage B command."""
    if not re.fullmatch(r"expb_[a-z0-9][a-z0-9_-]{5,63}", run_id):
        raise ExperimentBError("Run ID must be an immutable versioned expb_ identifier")
    auth = _check_run_authorization(authorization_file, run_id=run_id)
    run_root = RUN_ROOT / run_id
    if run_root.exists():
        raise ExperimentBError(f"Refusing to overwrite an existing or failed Experiment B run: {run_root}")
    ignored = subprocess.run(["git", "check-ignore", "-q", str((run_root / "future.txt").relative_to(ROOT))],
                             cwd=ROOT).returncode == 0
    if not ignored:
        raise ExperimentBError("Experiment B output namespace must be Git-ignored before execution")
    from experiments.calibration import r7_shuffled_correction_4class as r7
    from experiments.calibration import r9_correction_mechanism_falsification as r9
    from experiments.calibration import r10_signed_knn_baseline as r10
    from experiments.calibration import r3_fullstream_2x2 as r3

    source_commit = _git(["rev-parse", "HEAD"])
    source_fingerprints = _source_fingerprints()
    contract = r7._frozen_contract()
    registrations = {category: _registered_category_identity(contract, category, r9, r7)
                     for category in CATEGORIES}
    historical_refs = _preflight_historical_references(contract)
    model_receipt = r3._get_verified_model_receipt()
    model_identity = {"checkpoint_sha256": model_receipt.get("checkpoint_sha256"),
                      "loaded_model_state_sha256": model_receipt.get("loaded_model_state_sha256"),
                      "receipt_manifest_sha256": sha256_file(r3.MODEL_RECEIPT / "run_manifest.json")}
    resource_preflight = r10._verify_disk_and_gpu_preflight(RUN_ROOT.parent)
    if int(resource_preflight.get("disk_free_bytes", resource_preflight.get("free_bytes", 0))) < MIN_FREE_BYTES:
        raise ExperimentBError("Resource preflight reports less than the registered minimum free disk")
    protocol_sha = sha256_file(PROTOCOL_PATH)
    identity = {"schema": "exp-b.identity.v1", "run_id": run_id,
        "branch": _git(["branch", "--show-current"]), "source_commit_sha": source_commit,
        "parent_sha": _git(["rev-parse", "HEAD^"]), "source_fingerprints": source_fingerprints,
        "protocol_sha256": protocol_sha, "authorization": auth,
        "model_identity": model_identity, "historical_reference_receipts": historical_refs["receipts"],
        "historical_beta_permutation": historical_refs["historical_beta_permutation"],
        "historical_global_seals": {k: v for k, v in historical_refs.items()
                                     if k.endswith("_sha256")},
        "registrations": registrations,
        "categories": list(CATEGORIES), "support_seed": 0, "shuffle_seed": 0,
        "event_count": sum(EXPECTED_COUNTS.values()), "query_count": sum(EXPECTED_QUERIES.values()),
        "query_policy": "exact R9/R10 P0 Q-A-TS image identities; identity remapped to SHUFFLE0",
        "support_and_c64": "four fixed Supports plus 64 additional disjoint TRAIN normals",
        "encoder": "dinov2_vitl14_reg; resize448; center-crop392; blocks5/11/17/23; concat4096",
        "correction": "R9 native C1-Full; K1024; Top32; fixed coefficients and writer",
        "greedy": "one shared R9 native Greedy K1024 trajectory and common C0 writer",
        "external_memory_capacity": EXTERNAL_CAPACITY,
        "output_limit_bytes": MAX_OUTPUT_BYTES,
        "minimum_free_disk_bytes": MIN_FREE_BYTES,
        "failure_metadata_reserve_bytes": FAILURE_RESERVE_BYTES,
        "category_deadline_seconds": CATEGORY_DEADLINE_SECONDS,
        "confirmation_data_access": False, "test_masks_access": False,
        "automatic_restart": False, "resume_incomplete_run": False}
    identity["identity_sha256"] = sha256_bytes(canonical_json(identity))
    run_root.mkdir(parents=True, exist_ok=False)
    (run_root / "stage_a").mkdir()
    write_json_exclusive(run_root / "run_manifest.json", identity)
    try:
        write_json_exclusive(run_root / "resource_preflight.json", resource_preflight)
        for category in CATEGORIES:
            _verify_source_fingerprints(source_fingerprints)
            _run_supervised_stage_a(
                category, run_root, run_id=run_id, authorization_file=authorization_file,
                source_commit=source_commit, identity_sha256=identity["identity_sha256"],
                experiment_seed=experiment_seed)
            if _tree_bytes(run_root) > MAX_OUTPUT_BYTES - FAILURE_RESERVE_BYTES:
                raise ExperimentBError("Experiment B output exceeded its registered private output ceiling")
        stage_a_seal = _verify_full_stage_a(
            run_root, source_commit=source_commit, identity_sha256=identity["identity_sha256"])
        # Validate historical behavior before labels are projected in Stage B.
        parity = {}
        for category in CATEGORIES:
            parity[category] = _verify_historical_parity(
                category, _jsonl_rows(run_root / "stage_a" / category / "stage_a_scores.jsonl"),
                historical_refs)
        write_json_exclusive(run_root / "stage_a" / "historical_parity.json", {
            "schema": "exp-b.historical-parity.v1", "status": "PASS", "categories": parity})
        source_commit_now = _git(["rev-parse", "HEAD"])
        if source_commit_now != source_commit:
            raise ExperimentBError("Tracked source revision changed during Stage A")
        _verify_source_fingerprints(source_fingerprints)
        # A new process is required because CUDA cannot be uninitialized safely
        # after the Stage A backbone has been loaded.
        _run_supervised_stage_b(run_id, authorization_file)
        complete_path = run_root / "complete_seal.json"
        complete = json.loads(complete_path.read_text(encoding="utf-8"))
        if (complete.get("status") != "SEALED_COMPLETE" or
                complete.get("run_id") != run_id or
                complete.get("identity_sha256") != identity["identity_sha256"] or
                complete.get("stage_a_seal_sha256") != sha256_file(run_root / "stage_a" / "stage_a_seal.json")):
            raise ExperimentBError("Supervised Stage B child exited without a valid complete seal")
        if _tree_bytes(run_root) > MAX_OUTPUT_BYTES:
            raise ExperimentBError("Experiment B output exceeded its registered private output ceiling")
    except BaseException as exc:
        failure = {"schema": "exp-b.failure.v1", "status": "INCOMPLETE",
            "run_id": run_id, "phase": "full_run", "exception_type": type(exc).__name__,
            "message": str(exc), "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
        try:
            write_json_exclusive(run_root / "failure.json", failure)
        except FileExistsError:
            pass
        raise
    return run_root


def main(argv: Sequence[str] | None = None) -> int:
    stage_a_authorized_cuda_visibility = AUTHORIZED_CUDA_VISIBILITY
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full-run", action="store_true",
                        help="execute registered full Stage A and Stage B; requires researcher authorization")
    parser.add_argument("--stage-b-only", action="store_true",
                        help=argparse.SUPPRESS)
    parser.add_argument("--stage-a-child", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--stage-b-child", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--category", choices=CATEGORIES, help=argparse.SUPPRESS)
    parser.add_argument("--run-id", help="immutable expb_ run identifier")
    parser.add_argument("--authorization-file", type=Path,
                        help="review-complete researcher authorization JSON")
    parser.add_argument("--experiment-seed", type=int, default=0)
    args = parser.parse_args(argv)
    modes = (args.full_run, args.stage_b_only, args.stage_a_child, args.stage_b_child)
    if not args.run_id or not args.authorization_file or sum(bool(mode) for mode in modes) != 1:
        parser.error("select exactly one authorized full-run or supervised child mode with run ID and authorization")
    if args.stage_a_child:
        _run_stage_a_child(args, stage_a_authorized_cuda_visibility)
        return 0
    if args.stage_b_only or args.stage_b_child:
        if args.stage_b_only:
            parser.error("Stage B is available only as the supervised CPU child of --full-run")
        from experiments.calibration import r1_runtime_supervisor as supervisor
        if not supervisor.current_process_is_in_job():
            parser.error("Stage B child must be contained in the authorized Windows Job Object")
        auth = _check_run_authorization(args.authorization_file, run_id=args.run_id)
        run_root = (RUN_ROOT / args.run_id).resolve(strict=True)
        manifest = json.loads((run_root / "run_manifest.json").read_text(encoding="utf-8"))
        if (manifest.get("authorization", {}).get("authorization_file_sha256") != auth["authorization_file_sha256"] or
                manifest.get("source_commit_sha") != _git(["rev-parse", "HEAD"])):
            parser.error("Stage B authorization differs from the bound run manifest")
        stage_a_seal = _verify_full_stage_a(
            run_root, source_commit=str(manifest["source_commit_sha"]),
            identity_sha256=str(manifest["identity_sha256"]))
        parity_path = run_root / "stage_a" / "historical_parity.json"
        if not parity_path.is_file() or json.loads(parity_path.read_text(encoding="utf-8")).get("status") != "PASS":
            parser.error("Stage B requires successful historical parity verification")
        import torch
        if torch.cuda.is_initialized() or torch.cuda.device_count() != 0:
            parser.error("Stage B child must start with CUDA hidden")
        from experiments.calibration import r7_shuffled_correction_4class as r7
        contract = r7._frozen_contract()
        stage_b = _stage_b_global(run_root, stage_a_seal, contract)
        write_json_exclusive(run_root / "stage_b_results.json", stage_b)
        complete = {"schema": "exp-b.complete-seal.v1", "status": "SEALED_COMPLETE",
            "run_id": args.run_id, "identity_sha256": manifest["identity_sha256"],
            "stage_a_seal_sha256": sha256_file(run_root / "stage_a" / "stage_a_seal.json"),
            "stage_b_results_sha256": sha256_file(run_root / "stage_b_results.json"),
            "historical_parity_sha256": sha256_file(parity_path),
            "confirmation_data_accessed": False, "test_masks_opened": False}
        write_json_exclusive(run_root / "complete_seal.json", complete)
        output = run_root
    else:
        output = run_full(run_id=args.run_id, authorization_file=args.authorization_file,
                          experiment_seed=args.experiment_seed)
    print(json.dumps({"status": "SEALED_COMPLETE", "run_root": str(output)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
