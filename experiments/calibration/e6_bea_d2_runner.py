"""Authorized ten-arm D2 real-data Stage A and retrospective Stage B runner.

The only executable entry point is ``run-all`` with an exact committed HEAD,
explicit authorization, and the versioned D2 contract. There is no smoke,
truncation, automatic retry, GPU telemetry, or Stage A outcome evaluation.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from experiments.calibration.e6_bea_policy import (
    UTILITY_COLD_START_EVENTS,
    UTILITY_HISTORY_WINDOW,
)
from experiments.calibration import e6_bea_closed_loop as e6
from experiments.calibration.e6_bea_d2 import (
    D2_ARM_IDS,
    D2_BEA_TERMS,
    D2_EXPECTED_COUNTS,
    D2_PERCENTILE,
    D2_RANDOM_VIEW_PROBABILITY,
    D2_SIMPLE_ARMS,
    D2_STAGE_B_AUTHORIZATION_ID,
    D2_TARGET_RATE,
    D2Error,
    D2PolicyController,
    D2PrivateFeedbackSink,
    D2ScoreLedger,
    D2StageAClosedLoop,
    D2ValidatedRun,
    _write_json_once,
    authorize_d2_stage_b,
    compare_d2_to_frozen_e6,
    compare_public_trajectories,
    d2_stage_b_evaluate,
    event_stream_identity_sha256,
    factorial_and_policy_analysis,
    require_four_complete_d2_runs,
    validate_d2_stage_a,
)
from experiments.calibration.e6_bea_runner import (
    _support_cls_features,
    _support_features_batched,
    load_e6_stage05_category,
)
from experiments.calibration.e6_bea_closed_loop import (
    ArmRuntime,
    NormalCoverageMemory,
    OnDemandLabelReader,
    R7FrozenFeatureEncoder,
    StageBLabelSource,
    _assert_module_source,
    _canonical_json,
    _load_stage_b_identity_records,
    _reject_reparse_ancestors,
    _sha256_file,
    _validate_stage_b_identity_records,
    build_r7_native_arm_engines,
    load_r7_frozen_backbone,
    verify_e6_execution_identity,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
EXPECTED_BRANCH = "calibration-stage0-contract-20261029"
RUN_ID = "20261104_b1"
STAGE_A_AUTHORIZATION_ID = "E6-BEA-D2-STAGEA-20261104-A1"
STAGE_B_AUTHORIZATION_ID = D2_STAGE_B_AUTHORIZATION_ID
OUTPUT_BASE = REPOSITORY_ROOT / "outputs" / "calibration"
RUN_ROOT = OUTPUT_BASE / f"e6_bea_d2_{RUN_ID}"
STAGE_B_DIR = OUTPUT_BASE / f"e6_bea_d2_stage_b_{RUN_ID}"
STAGE_A_DIRS = {
    category: OUTPUT_BASE / f"e6_bea_d2_{RUN_ID}_{category}"
    for category in ("audiojack", "regulator", "switch", "rolled_strip_base")}
CONTRACT_PATH = RUN_ROOT / "D2_contract.json"

SOURCE_FILES = (
    "experiments/calibration/e6_bea_d2.py",
    "experiments/calibration/e6_bea_d2_runner.py",
    "experiments/calibration/e6_bea_policy.py",
    "experiments/calibration/e6_bea_closed_loop.py",
    "experiments/calibration/e6_bea_runner.py",
    "experiments/calibration/r7_shuffled_correction_4class.py",
    "experiments/calibration/r1_matched_correction_pilot.py",
    "experiments/calibration/r2_exp2a_c1_strength.py",
    "experiments/calibration/r3_fullstream_2x2.py",
    "experiments/calibration/r6_capacity1024_audiojack_runner.py",
    "experiments/calibration/native.py",
    "experiments/calibration/profiles.py",
    "experiments/calibration/controllers.py",
    "experiments/calibration/stage05.py",
    "experiments/p0_protocol_audit.py",
    "experiments/v02_memory.py",
    "test_phase3_AIF_simplified_v22.py",
    "vmb_visionad_new.py",
)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise D2Error(f"required D2 metadata is unreadable: {path}") from exc
    if not isinstance(value, dict):
        raise D2Error(f"required D2 metadata must be an object: {path}")
    return value


def _git(args: list[str]) -> str:
    try:
        return subprocess.run(args, cwd=REPOSITORY_ROOT, check=True,
                              capture_output=True, text=True,
                              encoding="utf-8").stdout.strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise D2Error(f"D2 cannot verify Git identity using {args}") from exc


def verify_d2_execution_identity(approved_commit: str) -> dict[str, Any]:
    if (len(approved_commit) != 40 or
            any(ch not in "0123456789abcdef" for ch in approved_commit.lower())):
        raise D2Error("D2 execution requires the exact full committed HEAD SHA")
    branch = _git(["git", "branch", "--show-current"])
    head = _git(["git", "rev-parse", "HEAD"]).lower()
    if branch != EXPECTED_BRANCH or head != approved_commit.lower():
        raise D2Error("D2 execution branch/HEAD differs from the separately authorized identity")
    scoped = [*SOURCE_FILES, "tests/calibration/test_e6_bea_d2.py"]
    dirty = _git(["git", "status", "--porcelain", "--", *scoped])
    if dirty:
        raise D2Error("D2 implementation or E6 source paths have uncommitted changes")
    native_identity = verify_e6_execution_identity(head, root=REPOSITORY_ROOT)
    return {"branch": branch, "commit": head,
            "reviewed_e6_identity": native_identity}


def _source_fingerprints() -> dict[str, str]:
    result = {}
    for relative in SOURCE_FILES:
        path = REPOSITORY_ROOT / relative
        if not path.is_file():
            raise D2Error(f"D2 source identity file is missing: {relative}")
        result[relative] = _sha256_file(path)
    return result


def _stream_receipt(info: Mapping[str, Any]) -> dict[str, Any]:
    events = tuple(info["events"])
    contract = info["contract"]
    return {
        "event_count": len(events),
        "event_stream_identity_sha256": event_stream_identity_sha256(events),
        "category_json_sha256": str(contract["category_json_sha256"]),
        "support_image_sha256": str(info["support_image_sha256"]),
        "support_group_sha256": str(info["support_group_sha256"]),
        "fixed_bank_sha256": str(info["fixed_support_sha256_expected"]),
        "c64_count": len(info["c64_scores"]),
        "c64_image_sha256": str(info["frozen_info"]["c64"]["image_sha256"]),
        "c64_group_sha256": str(info["contract"]["c64_group_sha256"]),
        "c64_score_archive_sha256": str(info["c64_score_archive_sha256"]),
        "c64_score_vector_sha256": hashlib.sha256(_canonical_json(
            [float(x) for x in info["c64_scores"]])).hexdigest(),
    }


def _build_contract(*, run_id: str, approved_commit: str,
                    authorization_id: str, source_hashes: Mapping[str, str],
                    model_receipt: Mapping[str, Any],
                    category_receipts: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    import torch

    return {
        "schema": "e6.bea_v2.d2.contract.v1",
        "method_version": "e6-bea-v2-d2-integrated-matched-supervision-v1",
        "status": "FROZEN_BEFORE_STAGE_A",
        "classification": "DEVELOPMENT_EXPLORATORY",
        "run_id": run_id,
        "authorization": {
            "stage_a_id": authorization_id,
            "stage_b_id": STAGE_B_AUTHORIZATION_ID,
            "authorized_by": "explicit researcher authorization in D2 task",
            "confirmation_categories_authorized": False,
        },
        "execution_identity": {"branch": EXPECTED_BRANCH,
                               "approved_commit": approved_commit},
        "dataset": {
            "protocol": "Real-IAD Regime B; four independent normal Supports; separate C64 commissioning normals",
            "categories": dict(category_receipts),
            "total_test_events": sum(D2_EXPECTED_COUNTS.values()),
            "shuffle": {"source": "original deterministic R7 Source Group shuffle",
                        "seed": 0},
            "support_seed": 0,
            "c64_initialization": "authenticated Stage 0.5 C64; initial C0 scores remain version-zero B3 evidence",
        },
        "model_and_native_protocol": {
            "model_receipt": dict(model_receipt),
            "model": "frozen local DINOv2 ViT-L/14 registers",
            "native_scoring": "original R7 identity-view native scoring and Top-32 aggregation",
            "greedy_capacity": 1024,
            "c1_capacity": 1024,
            "c1_active_in_every_arm": True,
            "b3": "current-state calibration; each arm owns independent B3 state",
            "normal_coverage_memory": "one independently mutable instance per arm",
            "greedy_writer": "original frozen C0 writer eligibility and candidate selection",
            "max_executed_queries_per_source_group_per_arm": 1,
            "test_image_encoding": "one frozen DINO encoding shared as immutable event features across the ten arm scorers",
        },
        "runtime": {
            "python_executable": sys.executable,
            "python_version": sys.version,
            "platform": platform.platform(),
            "torch_version": str(torch.__version__),
            "cuda_runtime_build": str(torch.version.cuda),
            "execution": "authorized full-stream CUDA Stage A; CPU Stage B",
            "gpu_monitoring": False,
            "nvidia_smi": False,
            "automatic_retry": False,
            "event_limits": {category: count for category, count in D2_EXPECTED_COUNTS.items()},
            "truncated_streams": False,
        },
        "arm_registry": list(D2_ARM_IDS),
        "bea_arms": {
            arm: {"objective_terms": list(terms),
                  "objective": "sum of existing BEAComponents weighted fields"}
            for arm, terms in D2_BEA_TERMS.items()},
        "bea_components": {
            "B": "R + 0.30 H", "N": "0.35 U_N",
            "C": "0.15 U_C", "D": "0.35 U_D",
            "implementation": "import existing compute_bea_components; no formula edits",
        },
        "bea_query_price": {
            "initial": 0.80, "target_rate": 0.03, "learning_rate": 0.02,
            "executed_query_window": 256, "bounds": [0.05, 3.0],
            "decision": "objective > current price (strict)",
            "update": "past and current executed flags only; update affects event t+1",
        },
        "simple_policies": {
            "RAND37": {"query_rate_target": 0.037,
                       "group_view_count": 5,
                       "per_view_probability": D2_RANDOM_VIEW_PROBABILITY,
                       "determinism": "seed 37 and identity-hashed per-event Bernoulli",
                       "budget_choice_rationale": "known development query expenditures; modest supervision advantage, not outcome-optimized"},
            "PERIOD27": {"proposal_rule": "(position + 1) modulo 27 equals zero"},
            "MARGIN37": {"utility": "existing standardized margin psi",
                         "target_rate": D2_TARGET_RATE},
            "ENTROPY37": {"utility": "existing Bernoulli entropy H from BEAComponents",
                          "target_rate": D2_TARGET_RATE},
            "NOVELTY_NN37": {"utility": "raw distance_nn from the arm's NormalCoverageMemory novelty_percentile receipt",
                             "not_clipped_percentile": True,
                             "target_rate": D2_TARGET_RATE},
            "percentile_policies": {
                "cold_start_events": UTILITY_COLD_START_EVENTS,
                "cold_start": "seeded Bernoulli p=0.037; shared identity hash for MARGIN37/ENTROPY37",
                "past_only_history_window": UTILITY_HISTORY_WINDOW,
                "percentile": D2_PERCENTILE,
                "history_order": "propose from prior utilities; append current utility after final action",
                "comparison": "utility > threshold proposes; utility < threshold does not",
                "tie_rule": "identity-hashed lottery with p=clip((0.037*n_history - n_gt)/n_eq,0,1), computed only from prior history",
                "no_forced_queries": True,
            },
        },
        "causal_execution": {
            "sequence": [
                "encode each TEST image once",
                "compute all ten PRE-EVENT native scores/components",
                "propose from each arm's own pre-event state",
                "apply per-arm Source Group eligibility",
                "jointly append and fsync the ten-arm score/query row",
                "disclose only executed-query labels to their arms",
                "apply only corresponding feedback; effects first visible at t+1",
            ],
            "unqueried_outcome_disclosure": False,
            "self_training_on_unqueried": False,
            "future_length_or_hard_query_quota": False,
            "retrospective_query_insertion": False,
            "mutable_state_independent_per_arm": True,
        },
        "primary_comparisons": {
            "primary_bea": "BND, selected before this D2 using prior exposed development evidence",
            "simple_comparators": list(D2_SIMPLE_ARMS),
            "strongest_simple_selection": "lowest category-equal Macro BER; report as best-of-five comparison",
            "candidate_practical_effect_threshold": 0.01,
            "threshold_is_not_significance": True,
            "budget_match_tolerance_absolute_rate": 0.003,
            "budget_dominance_rule": "if a simple policy executes more queries and BEA has lower BER, report BUDGET_DOMINANCE evidence separately",
            "confounding_rule": "if supervision differs beyond tolerance and neither policy dominates quality and expenditure, report BUDGET_CONFOUNDED",
            "no_query_count_normalization": True,
        },
        "factorial": {
            "arms": ["B", "BN", "BD", "BND"],
            "N_main_improvement": "0.5*((L_B-L_BN)+(L_BD-L_BND))",
            "D_main_improvement": "0.5*((L_B-L_BD)+(L_BN-L_BND))",
            "interaction": "L_BND-L_BN-L_BD+L_B; negative is synergistic on additive BER scale",
        },
        "c_on_off": {"comparison": "BND versus BNCD",
                      "primary": "MacroBER(BNCD)-MacroBER(BND)",
                      "secondary": ["AUROC", "AP"],
                      "c1_active_both": True},
        "recommendation_rule": {
            "GO": "BND improves Macro BER by at least 0.01 against the strongest simple baseline, the macro budget status is APPROXIMATELY_BUDGET_MATCHED or BUDGET_DOMINANCE_EVIDENCE_IN_A_FAVOR, and every registered B/BND/BNCD frozen-E6 parity comparison passes.",
            "SIMPLIFY": "GO is not met, parity passes, and at least one of B/BN/BD has Macro BER no more than 0.01 above BND while using no more than BND's macro query rate plus the registered 0.003 tolerance.",
            "DROP": "Neither GO nor SIMPLIFY is met. A parity failure means dropping the current implementation from the paper narrative, not evidence against BEA in general.",
            "candidate_effect_threshold_is_not_significance": True,
        },
        "stage_b": {
            "authorization_id": STAGE_B_AUTHORIZATION_ID,
            "gate": "all four full Stage A seals and all input/hash/identity checks pass before any TEST outcome load",
            "label_source": "existing authenticated Stage B identity mapping and decoder",
            "primary_metric": "category-equal Macro Prequential BER",
            "worst_256": "original nonoverlapping complete 256-event window implementation",
            "uncertainty": "category-level paired differences; no naive independent-image significance and no seed sweep",
            "stage_a_label_output": False,
            "public_event_level_outcome_output": False,
        },
        "frozen_e6_parity": {
            "references": {category:
                f"outputs/calibration/e6_bea_v2_E6FULL_20261104_{category}_fix1"
                for category in D2_EXPECTED_COUNTS},
            "arm_map": dict({"B": "Q4", "BND": "Q5", "BNCD": "Q6"}),
            "required_fields": ["initial_model_support_c64_fixed_bank_identities",
                                "pre_event_scores", "B3_thresholds",
                                "query_proposals", "executed_queries",
                                "source_group_identities", "scorer_state_versions",
                                "scorer_state_digests", "realized_query_counts"],
            "label_blind": True,
            "divergences_are_reported_not_corrected": True,
        },
        "frozen_source_sha256": dict(source_hashes),
        "output_identity": {
            "root": str(RUN_ROOT),
            "stage_a_directories": {cat: str(path) for cat, path in STAGE_A_DIRS.items()},
            "stage_b_directory": str(STAGE_B_DIR),
            "non_overwrite": True,
        },
    }


def _write_failure(run_dir: Path, stage: str, error: BaseException) -> None:
    failure = Path(run_dir) / "failure.json"
    if failure.exists():
        raise D2Error(f"D2 failure receipt already exists and will not be overwritten: {failure}")
    payload = {
        "schema": "e6.bea_v2.d2.failure.v1", "status": "INCOMPLETE",
        "run_id": RUN_ID, "stage": str(stage),
        "failure_type": type(error).__name__, "message": str(error)[:2000],
        "automatic_retry": False,
        "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    with failure.open("xb") as stream:
        stream.write(_canonical_json(payload))
        stream.flush()
        os.fsync(stream.fileno())


def _write_text_once(path: Path, value: str) -> str:
    path = Path(path)
    _reject_reparse_ancestors(path)
    payload = str(value).encode("utf-8")
    with path.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    return hashlib.sha256(payload).hexdigest()


def _initial_states(engines: Mapping[str, Any],
                    coverages: Mapping[str, NormalCoverageMemory]) -> dict[str, Any]:
    states: dict[str, Any] = {}
    for arm in D2_ARM_IDS:
        engine = engines[arm]
        states[arm] = {
            "fixed_bank_sha256": str(engine.fixed_support_sha256),
            "scorer_state_sha256": str(engine.initial_state_sha256),
            "scorer_state_version": int(engine.state_version),
            "coverage_state_sha256": coverages[arm].state_sha256(),
            "c64_group_sha256": str(engine.c64_group_sha256),
            "c64_score_archive_sha256": str(engine.input_identity["c64_score_archive_sha256"]),
            "c64_initial_score_vector_sha256": hashlib.sha256(_canonical_json(
                [float(x) for x in engine.c64_initial_scores])).hexdigest(),
            "initial_b3_c64_initialization": dict(engine.initial_b3_c64_initialization),
        }
    if len({row["scorer_state_sha256"] for row in states.values()}) != 1:
        raise D2Error("D2 native arms do not share equivalent initial scorer state")
    if len({row["fixed_bank_sha256"] for row in states.values()}) != 1:
        raise D2Error("D2 native arms do not share the authenticated fixed support bank")
    if len({row["coverage_state_sha256"] for row in states.values()}) != 1:
        raise D2Error("D2 coverage memories do not start in equivalent support state")
    mutable_ids = [set(engines[arm].owned_mutable_state_ids) for arm in D2_ARM_IDS]
    for i, left in enumerate(mutable_ids):
        for right in mutable_ids[i + 1:]:
            if left & right:
                raise D2Error("D2 native arms share a mutable native state object")
    if len({id(coverages[arm]) for arm in D2_ARM_IDS}) != len(D2_ARM_IDS):
        raise D2Error("D2 arms share a mutable Coverage Memory object")
    return states


def _run_category(*, category: str, run_id: str, approved_commit: str,
                  authorization_id: str, contract_sha256: str,
                  source_hashes: Mapping[str, str],
                  info: Mapping[str, Any]) -> D2ValidatedRun:
    run_dir = STAGE_A_DIRS[category]
    if run_dir.exists() or run_dir.is_symlink():
        raise D2Error(f"D2 refuses to resume or overwrite an existing Stage A directory: {run_dir}")
    run_dir.mkdir(parents=False, exist_ok=False)
    _reject_reparse_ancestors(run_dir)
    namespace = {
        "schema": "e6.bea_v2.d2.run_namespace.v1",
        "status_at_creation": "INCOMPLETE",
        "run_id": run_id, "category": category,
        "approved_commit": approved_commit,
        "authorization_id": authorization_id,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    _write_json_once(run_dir / "run_namespace.json", namespace)
    try:
        import torch
        from types import SimpleNamespace
        from experiments import p0_protocol_audit as p0
        from experiments.calibration import r1_matched_correction_pilot as r1
        from experiments.calibration import r2_exp2a_c1_strength as exp2a
        from experiments.calibration import r3_fullstream_2x2 as r3
        from experiments.calibration import r6_capacity1024_audiojack_runner as r6
        from experiments.calibration import r7_shuffled_correction_4class as r7
        from experiments.calibration import native, profiles

        identity = verify_e6_execution_identity(approved_commit, root=REPOSITORY_ROOT)
        impl = p0.require_sources()
        visionad = importlib.import_module("vmb_visionad_new")
        backbone, model_receipt, runtime_info = load_r7_frozen_backbone(
            impl=impl, visionad=visionad, r1=r1, r3=r3,
            approved_e6_commit=identity["head"], repository_root=REPOSITORY_ROOT)
        cfg, native_profile = r3._native_config(p0, profiles, r1)
        support_features = p0.encode_reference_features(
            impl, backbone, list(info["support_paths"]), cfg)
        support_cls = _support_cls_features(
            impl, backbone, list(info["support_paths"]), cfg.feature_fuse)
        c64_items, c64_feature_receipt = r7._make_c64_items(
            category, info, p0, impl, backbone, cfg)
        engines = build_r7_native_arm_engines(
            category=category, info=info, impl=impl, visionad=visionad,
            p0=p0, r1=r1, r3=r3, r6=r6, r7=r7, exp2a=exp2a,
            native_module=native, cfg=cfg,
            support_features=support_features,
            support_paths=info["support_paths"], c64_items=c64_items,
            approved_e6_commit=approved_commit,
            repository_root=REPOSITORY_ROOT, arm_ids=D2_ARM_IDS)
        if set(engines) != set(D2_ARM_IDS):
            raise D2Error("native factory did not construct exactly the ten D2 arms")
        coverages = {arm: NormalCoverageMemory(support_cls) for arm in D2_ARM_IDS}
        initial_states = _initial_states(engines, coverages)
        input_identity = engines["B"].input_identity
        if any(engine.input_identity != input_identity for engine in engines.values()):
            raise D2Error("D2 arms were constructed from different Support/C64 identities")
        model_state_sha = str(model_receipt["loaded_model_state_sha256"])
        if (not model_state_sha or
                model_state_sha != runtime_info.get("loaded_model_state_sha256")):
            raise D2Error("D2 loaded model fingerprint differs from its registered receipt")
        event_receipt = _stream_receipt(info)
        output_files = {
            "run_namespace.json", "run_manifest.json", "sealed_scores.jsonl",
            "stage_a_seal.json", *[f"private_feedback_{arm}.jsonl" for arm in D2_ARM_IDS]}
        manifest = {
            "schema": "e6.bea_v2.d2.run_manifest.v1",
            "status_at_creation": "INCOMPLETE",
            "stage_b_enabled": False,
            "run_id": run_id, "category": category,
            "registered_categories": list(D2_EXPECTED_COUNTS),
            "registered_arm_ids": list(D2_ARM_IDS),
            "stream_event_count": len(info["events"]),
            "stream_identity_sha256": event_receipt["event_stream_identity_sha256"],
            "approved_commit": approved_commit,
            "branch": EXPECTED_BRANCH,
            "execution_authorization_id": authorization_id,
            "contract_sha256": contract_sha256,
            "namespace_record_sha256": _sha256_file(run_dir / "run_namespace.json"),
            "source_sha256": dict(source_hashes),
            "model_identity": {
                "loaded_model_state_sha256": model_state_sha,
                "model_receipt": dict(model_receipt),
                "runtime_profile": dict(runtime_info),
                "verified_frozen_backbone": True,
            },
            "support_and_c64_identity": dict(event_receipt),
            "input_identity": dict(input_identity),
            "initial_arm_states": initial_states,
            "native_scoring_configuration": {
                "p0": engines["B"].native_scorer_config,
                "r7": engines["B"].native_profile_config,
                "c1": engines["B"].correction_config_record,
                "greedy_writer": engines["B"].greedy_settings_record,
            },
            "initial_c64_numerical_parity": {
                arm: dict(engines[arm].c64_parity_receipt) for arm in D2_ARM_IDS},
            "c64_feature_cache_receipt": dict(c64_feature_receipt),
            "runtime": {
                "python": sys.version,
                "platform": platform.platform(),
                "torch": str(torch.__version__),
                "cuda_execution": True,
                "gpu_monitoring": False,
                "automatic_retry": False,
            },
            "query_policies": {
                "B": {"kind": "BEA", "terms": ["B"]},
                "BN": {"kind": "BEA", "terms": ["B", "N"]},
                "BD": {"kind": "BEA", "terms": ["B", "D"]},
                "BND": {"kind": "BEA", "terms": ["B", "N", "D"]},
                "BNCD": {"kind": "BEA", "terms": ["B", "N", "C", "D"]},
                "RAND37": {"kind": "identity_hashed_bernoulli", "seed": 37},
                "PERIOD27": {"kind": "periodic_27"},
                "MARGIN37": {"kind": "past_only_q963", "seed": 37},
                "ENTROPY37": {"kind": "past_only_q963", "seed": 37},
                "NOVELTY_NN37": {"kind": "raw_distance_nn_past_only_q963", "seed": 37},
            },
            "greedy_writer_semantics": "frozen C0 eligibility and candidate selection; queried verified normals only",
            "c1_correction": {"active": True, "capacity": 1024,
                              "native_r7_top32_residual": True},
            "output_files": sorted(output_files),
        }
        manifest_path = run_dir / "run_manifest.json"
        _write_json_once(manifest_path, manifest)
        manifest_sha = _sha256_file(manifest_path)
        ledger = D2ScoreLedger(run_dir / "sealed_scores.jsonl")
        arms: dict[str, ArmRuntime] = {}
        readers = []
        sinks = []
        try:
            for arm in D2_ARM_IDS:
                reader = OnDemandLabelReader(
                    info["json_path"], category,
                    info["contract"]["category_json_sha256"],
                    info["events"], arm, ledger,
                    registered_arm_ids=D2_ARM_IDS)
                sink = D2PrivateFeedbackSink(
                    run_dir / f"private_feedback_{arm}.jsonl", arm)
                readers.append(reader)
                sinks.append(sink)
                arms[arm] = ArmRuntime(
                    arm_id=arm, engine=engines[arm],
                    controller=D2PolicyController(arm, seed=37),
                    coverage=coverages[arm], label_reader=reader,
                    private_feedback=sink)
            encoder = R7FrozenFeatureEncoder(
                impl=impl, visionad=visionad, repository_root=REPOSITORY_ROOT,
                backbone=backbone, feature_fuse=cfg.feature_fuse, r1=r1,
                expected_model_state_sha256=model_state_sha,
                runtime_info=runtime_info)
            coordinator = D2StageAClosedLoop(
                category=category, run_id=run_id, events=info["events"],
                feature_encoder=encoder, arms=arms, ledger=ledger,
                output_dir=run_dir, manifest_path=manifest_path,
                manifest_sha256=manifest_sha,
                execution_authorization_id=authorization_id)
            created_seal = coordinator.run()
        except BaseException:
            ledger.close()
            for reader in readers:
                reader.close()
            for sink in sinks:
                sink.close()
            raise
        validated = validate_d2_stage_a(
            run_dir=run_dir, category=category, run_id=run_id,
            expected_commit=approved_commit, contract_sha256=contract_sha256,
            expected_events=info["events"])
        if (created_seal.seal_sha256 != validated.seal_sha256 or
                created_seal.manifest_sha256 != validated.manifest_sha256 or
                created_seal.score_log_sha256 != validated.score_log_sha256 or
                dict(created_seal.private_feedback_sha256) != {
                    arm: validated.seal["feedback_artifacts"][arm]["sha256"]
                    for arm in D2_ARM_IDS}):
            raise D2Error(f"{category}: Stage A seal changed between creation and validation")
        return validated
    except BaseException as exc:
        _write_failure(run_dir, "stage_a", exc)
        raise


def _render_report(evaluation: Mapping[str, Any],
                   analysis: Mapping[str, Any],
                   parity: Mapping[str, Any]) -> str:
    lines = [
        "# E6-BEA-v2 D2: Integrated Matched-Supervision Mechanism Experiment",
        "",
        "- Status: **COMPLETE**",
        "- Classification: DEVELOPMENT_EXPLORATORY; not independent confirmation.",
        "- Primary metric: category-equal Macro Prequential BER.",
        "- Stage A used one TEST encoding and ten jointly sealed arm decisions per event.",
        "- Stage B outcomes were loaded only after all four full Stage A seals and identities verified.",
        "- No confirmation category, image-level outcome table, or per-event label output was used.",
        "",
        "## Macro results",
        "",
        "| Arm | BER | FPR | FNR | BA | AUROC | AP | Worst-256 BER | Query labels | Query rate |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    macro = evaluation["category_equal_macro"]
    for arm in D2_ARM_IDS:
        lines.append(
            f"| {arm} | {macro[arm]['ber']:.5f} | {macro[arm]['fpr']:.5f} | "
            f"{macro[arm]['fnr']:.5f} | {macro[arm]['ba']:.5f} | "
            f"{macro[arm]['auroc']:.5f} | {macro[arm]['ap']:.5f} | "
            f"{macro[arm]['worst_256_window_ber']:.5f} | "
            f"{macro[arm]['query_count_sum']} | {macro[arm]['query_rate']:.4%} |")
    primary = analysis["primary_BND_vs_simple"]
    lines.extend(["", "## Primary BND versus simple policies", "",
                  f"Strongest simple policy by Macro BER: **{primary['strongest_simple_by_macro_ber']}** "
                  f"(Macro BER {primary['strongest_simple_macro_ber']:.5f}); BND Macro BER "
                  f"{primary['BND_macro_ber']:.5f}; BND minus comparator "
                  f"{primary['macro_ber_difference_BND_minus_strongest_simple']:+.5f}. "
                  f"The 0.01 threshold is candidate practical-effect context, not significance. ",
                  "", "| Simple arm | BND category wins | BND category losses | Macro query-count difference |",
                  "|---|---:|---:|---:|"])
    for arm in D2_SIMPLE_ARMS:
        record = primary["comparisons"][arm]
        lines.append(f"| {arm} | {record['category_wins_for_BND']} | "
                     f"{record['category_losses_for_BND']} | "
                     f"{record['macro_label_count_difference_BND_minus_simple']:+d} |")
    lines.extend(["", "Per-category query rate/count, budget classification, and FPR/FNR trade-offs are retained in `D2_evaluation.json`.",
                  "", "## N x D factorial and C-on/off", "",
                  "| Scope | N main improvement | D main improvement | N x D interaction | C delta BER (BNCD-BND) |",
                  "|---|---:|---:|---:|---:|"])
    factorial = analysis["N_by_D_factorial"]
    for scope in (*D2_EXPECTED_COUNTS.keys(), "CATEGORY_EQUAL_MACRO"):
        record = factorial[scope]
        lines.append(f"| {scope} | {record['N_main_improvement']:+.5f} | "
                     f"{record['D_main_improvement']:+.5f} | "
                     f"{record['N_by_D_interaction']:+.5f} | "
                     f"{record['C_delta_BER_BNCD_minus_BND']:+.5f} |")
    c_result = analysis["C_on_vs_C_off"]
    primary_budget = primary["macro_budget_classification"]
    lines.extend(["", f"Macro BND versus strongest-simple budget classification: **{primary_budget}**.",
                  f"Macro C-on minus C-off: BER **{c_result['macro_delta_BER_BNCD_minus_BND']:+.5f}**, "
                  f"AUROC {c_result['macro_delta_AUROC_BNCD_minus_BND']:+.5f}, "
                  f"AP {c_result['macro_delta_AP_BNCD_minus_BND']:+.5f}; label-count difference "
                  f"{c_result['macro_label_count_difference_BNCD_minus_BND']:+d}; budget status "
                  f"**{c_result['macro_budget_classification']}**; resource-controlled benefit "
                  f"established={c_result['resource_controlled_benefit_established']}. "
                  "C1 Correction Memory remained active in both BND and BNCD.",
                  "", "Budget comparisons use the registered absolute realized-rate tolerance of 0.003. "
                  "Unequal budgets are reported as budget-dominance evidence only when the policy with more labels has worse BER; otherwise the comparison is classified as budget-confounded where neither policy dominates.",
                  "", "## Novelty and entropy diagnostics", ""])
    for category in D2_EXPECTED_COUNTS:
        novelty = analysis["novelty_nn_diagnostics"][category]
        equiv = analysis["margin_entropy_equivalence"][category]
        lines.append(f"- **{category} NOVELTY_NN37:** {novelty['query_count']} queries "
                     f"({novelty['query_rate']:.3%}); proposals={novelty['proposal_count']}, "
                     f"ties={novelty['quantile_ties']}, Source-Group-suppressed="
                     f"{novelty['source_group_suppressed']}, longest query-free interval="
                     f"{novelty['max_query_free_interval_events']} events; BER="
                     f"{novelty['ber']:.5f}, AUROC={novelty['auroc']:.5f}, AP={novelty['ap']:.5f}.")
        lines.append(f"  MARGIN37/ENTROPY37 complete trajectory identical: "
                     f"{equiv['complete_logged_trajectory_identical']}.")
    lines.extend(["", "## Frozen E6 parity", "",
                  "D2 B, BND, and BNCD were compared label-blind against original E6 Q4, Q5, and Q6. Exact field mismatches and registered score/threshold tolerance results are in `D2_evaluation.json`."])
    for category, record in parity.items():
        lines.append(f"- {category}: **{record.get('status', 'UNCLASSIFIED')}**")
    lines.extend([
                  "", "## Evidence limits and decision", "",
                  "The four classes remain development data. Category-level paired differences are descriptive; 15,295 views are not treated as independent observations. No fresh online-run uncertainty, confirmation evidence, parameter tuning, or seed sweep is claimed.",
                  "", f"Recommendation: **{analysis['recommendation']['decision']}**. "
                  f"Reason: {analysis['recommendation']['reason']} This is a research contribution recommendation, not a statistical significance claim.",
                  ""])
    return "\n".join(lines)


def _recommend_d2_contribution(evaluation: Mapping[str, Any],
                               analysis: Mapping[str, Any],
                               parity: Mapping[str, Any]) -> dict[str, Any]:
    primary = analysis["primary_BND_vs_simple"]
    macro = evaluation["category_equal_macro"]
    parity_pass = all(
        row.get("status") == "PASS" and all(
            all(value == 0 for value in arm["exact_field_mismatch_counts"].values()) and
            arm["score_values_outside_registered_c64_tolerance"] == 0 and
            arm["threshold_values_outside_registered_tolerance"] == 0 and
            arm["initial_identities_all_match"]
            for arm in row["arms"].values())
        for row in parity.values())
    fair_or_dominance = primary["macro_budget_classification"] in {
        "APPROXIMATELY_BUDGET_MATCHED", "BUDGET_DOMINANCE_EVIDENCE_IN_A_FAVOR"}
    go = (float(primary["macro_ber_difference_BND_minus_strongest_simple"]) <= -0.01 and
          fair_or_dominance and parity_pass)
    bnd_ber = float(macro["BND"]["ber"])
    bnd_rate = float(macro["BND"]["query_rate"])
    simpler = [arm for arm in ("B", "BN", "BD")
               if float(macro[arm]["ber"]) <= bnd_ber + 0.01 and
               float(macro[arm]["query_rate"]) <= bnd_rate + 0.003]
    if go:
        decision = "GO"
        reason = ("BND exceeds the registered candidate practical-effect threshold against the strongest simple baseline at a matched or BEA-favorable budget, and frozen E6 parity passes.")
    elif parity_pass and simpler:
        decision = "SIMPLIFY"
        reason = (f"BND did not meet the registered GO rule; simpler arm(s) {', '.join(simpler)} are within 0.01 Macro BER of BND at no more than the registered query-rate tolerance.")
    else:
        decision = "DROP"
        reason = ("The registered GO and SIMPLIFY rules were not met." if parity_pass else
                  "Frozen E6 parity failed, so the current implementation is not suitable for a paper contribution claim.")
    return {"decision": decision, "reason": reason,
            "frozen_e6_parity_pass": parity_pass,
            "simpler_bea_arms_within_rule": simpler,
            "candidate_effect_threshold_is_not_significance": True}


def _freeze_contract_and_preflight(*, approved_commit: str,
                                   authorization_id: str) -> tuple[dict[str, Any], dict[str, Any], dict[str, dict[str, Any]], str]:
    if not authorization_id:
        raise D2Error("D2 Stage A requires the explicit experiment authorization ID")
    if RUN_ROOT.exists() or RUN_ROOT.is_symlink():
        raise D2Error(f"D2 refuses to reuse the existing run root: {RUN_ROOT}")
    if STAGE_B_DIR.exists() or STAGE_B_DIR.is_symlink():
        raise D2Error(f"D2 refuses to reuse the Stage B directory: {STAGE_B_DIR}")
    if any(path.exists() or path.is_symlink() for path in STAGE_A_DIRS.values()):
        raise D2Error("D2 refuses to reuse a category Stage A directory")
    identity = verify_d2_execution_identity(approved_commit)
    source_hashes = _source_fingerprints()
    from experiments import p0_protocol_audit as p0
    from experiments.calibration import r3_fullstream_2x2 as r3
    # Public metadata/projection checks only. No image decoding and no TEST
    # anomaly_class field is decoded in this pre-execution contract phase.
    category_info: dict[str, dict[str, Any]] = {}
    receipts: dict[str, dict[str, Any]] = {}
    for category in D2_EXPECTED_COUNTS:
        info = load_e6_stage05_category(category, repository_root=REPOSITORY_ROOT)
        if len(info["events"]) != D2_EXPECTED_COUNTS[category]:
            raise D2Error(f"{category}: preflight event count differs from the frozen contract")
        receipts[category] = _stream_receipt(info)
        category_info[category] = info
    model_receipt = dict(r3._get_verified_model_receipt())
    root_info = {"identity": identity, "source_hashes": source_hashes,
                 "model_receipt": model_receipt}
    RUN_ROOT.mkdir(parents=False, exist_ok=False)
    _reject_reparse_ancestors(RUN_ROOT)
    contract = _build_contract(
        run_id=RUN_ID, approved_commit=approved_commit,
        authorization_id=authorization_id, source_hashes=source_hashes,
        model_receipt=model_receipt, category_receipts=receipts)
    _write_json_once(CONTRACT_PATH, contract)
    contract_sha = _sha256_file(CONTRACT_PATH)
    return root_info, category_info, receipts, contract_sha


def _execute_stage_b(*, runs: Mapping[str, D2ValidatedRun],
                     category_info: Mapping[str, Mapping[str, Any]],
                     parity: Mapping[str, Any], authorization_id: str,
                     approved_commit: str, contract_sha: str,
                     source_hashes: Mapping[str, str]) -> dict[str, Any]:
    labels: dict[str, np.ndarray] = {}
    validated_runs = dict(runs)
    # Revalidate every namespace/hash immediately before Stage B outcome access.
    for category in D2_EXPECTED_COUNTS:
        latest = validate_d2_stage_a(
            run_dir=STAGE_A_DIRS[category], category=category, run_id=RUN_ID,
            expected_commit=approved_commit, contract_sha256=contract_sha,
            expected_events=category_info[category]["events"])
        prior = runs[category]
        if (latest.manifest_sha256 != prior.manifest_sha256 or
                latest.seal_sha256 != prior.seal_sha256 or
                latest.score_log_sha256 != prior.score_log_sha256):
            raise D2Error(f"{category}: Stage A hashes changed before Stage B label access")
        validated_runs[category] = latest

    # This is the first point at which TEST outcomes are loaded, after all
    # four full seals and the exact independent Stage B authorization pass.
    authorize_d2_stage_b(runs=runs,
                         authorization_id=STAGE_B_AUTHORIZATION_ID)
    for category in D2_EXPECTED_COUNTS:
        info = category_info[category]
        source = StageBLabelSource(
            Path(info["json_path"]),
            str(info["contract"]["category_json_sha256"]))
        records = _load_stage_b_identity_records(
            source, category, validated_runs[category].rows)
        labels[category] = _validate_stage_b_identity_records(
            validated_runs[category].rows, records)

    evaluation = d2_stage_b_evaluate(
        runs=validated_runs, labels_by_category=labels,
        authorization_id=STAGE_B_AUTHORIZATION_ID)
    analysis = factorial_and_policy_analysis(evaluation)
    analysis["margin_entropy_equivalence"] = {
        category: compare_public_trajectories(
            validated_runs[category], "MARGIN37", "ENTROPY37")
        for category in D2_EXPECTED_COUNTS}
    analysis["recommendation"] = _recommend_d2_contribution(
        evaluation, analysis, parity)
    evaluation["registered_analysis"] = analysis
    evaluation["frozen_e6_parity"] = parity
    evaluation["stage_b_authorization_id"] = STAGE_B_AUTHORIZATION_ID
    evaluation["stage_a_authorization_id"] = authorization_id
    evaluation["stage_a_seals_verified_before_outcomes"] = True
    evaluation["private_feedback_contents_read_for_verification"] = False
    evaluation["test_labels_saved_or_printed"] = False

    _write_json_once(STAGE_B_DIR / "D2_evaluation.json", evaluation)
    report = _render_report(evaluation, analysis, parity)
    _write_text_once(STAGE_B_DIR / "D2_report.md", report)
    provenance = {
        "schema": "e6.bea_v2.d2.provenance.v1",
        "status": "COMPLETE",
        "classification": "DEVELOPMENT_EXPLORATORY",
        "branch": EXPECTED_BRANCH,
        "commit": approved_commit,
        "run_id": RUN_ID,
        "stage_a_authorization_id": authorization_id,
        "stage_b_authorization_id": STAGE_B_AUTHORIZATION_ID,
        "contract_path": str(CONTRACT_PATH),
        "contract_sha256": contract_sha,
        "source_sha256": dict(source_hashes),
        "stage_a": {cat: {
            "run_dir": str(STAGE_A_DIRS[cat]),
            "manifest_sha256": validated_runs[cat].manifest_sha256,
            "score_log_sha256": validated_runs[cat].score_log_sha256,
            "seal_sha256": validated_runs[cat].seal_sha256,
            "private_feedback_sha256": {
                arm: validated_runs[cat].seal["feedback_artifacts"][arm]["sha256"]
                for arm in D2_ARM_IDS},
            "query_counts": dict(validated_runs[cat].seal["arm_query_counts"]),
        } for cat in D2_EXPECTED_COUNTS},
        "stage_b_evaluation_sha256": _sha256_file(STAGE_B_DIR / "D2_evaluation.json"),
        "stage_b_labels_saved": False,
        "frozen_e6_parity": parity,
    }
    _write_json_once(STAGE_B_DIR / "D2_provenance.json", provenance)
    labels.clear()
    return {"status": "COMPLETE", "run_root": str(RUN_ROOT),
            "stage_b_dir": str(STAGE_B_DIR),
            "stage_a_seals": {cat: str(runs[cat].run_dir / "stage_a_seal.json")
                               for cat in D2_EXPECTED_COUNTS},
            "stage_b_evaluation": str(STAGE_B_DIR / "D2_evaluation.json"),
            "stage_b_report": str(STAGE_B_DIR / "D2_report.md"),
            "stage_b_provenance": str(STAGE_B_DIR / "D2_provenance.json"),
            "frozen_commit": approved_commit}


def execute_d2(*, approved_commit: str, authorization_id: str) -> dict[str, Any]:
    """Run one no-retry D2 Stage A and Stage B execution under the frozen contract."""
    root_info, category_info, _receipts, contract_sha = _freeze_contract_and_preflight(
        approved_commit=approved_commit, authorization_id=authorization_id)
    source_hashes = root_info["source_hashes"]
    runs: dict[str, D2ValidatedRun] = {}
    for category in ("audiojack", "regulator", "switch", "rolled_strip_base"):
        try:
            runs[category] = _run_category(
                category=category, run_id=RUN_ID, approved_commit=approved_commit,
                authorization_id=authorization_id, contract_sha256=contract_sha,
                source_hashes=source_hashes, info=category_info[category])
        except BaseException as exc:
            # The failed category keeps its partial files and one failure receipt.
            # No automatic restart and no later category is launched.
            raise D2Error(f"D2 stopped after {category} Stage A failure: {type(exc).__name__}: {exc}") from exc
    require_four_complete_d2_runs(runs)

    # Revalidate exact stream identities after all Stage A files are sealed.
    for category in D2_EXPECTED_COUNTS:
        revalidated = validate_d2_stage_a(
            run_dir=STAGE_A_DIRS[category], category=category, run_id=RUN_ID,
            expected_commit=approved_commit, contract_sha256=contract_sha,
            expected_events=category_info[category]["events"])
        previous = runs[category]
        if (revalidated.manifest_sha256 != previous.manifest_sha256 or
                revalidated.seal_sha256 != previous.seal_sha256 or
                revalidated.score_log_sha256 != previous.score_log_sha256):
            raise D2Error(f"{category}: Stage A artifacts changed after initial validation")
        runs[category] = revalidated

    # This parity phase reads only immutable score receipts and old public
    # score logs. It does not decode Stage B outcome vectors.
    parity = compare_d2_to_frozen_e6(d2_runs=runs, e6_root=OUTPUT_BASE)
    for category in D2_EXPECTED_COUNTS:
        if any(any(value != 0 for value in arm["exact_field_mismatch_counts"].values()) or
               arm["score_values_outside_registered_c64_tolerance"] or
               arm["threshold_values_outside_registered_tolerance"] or
               not arm["initial_identities_all_match"]
               for arm in parity[category]["arms"].values()):
            parity[category]["status"] = "DIVERGENCE_RECORDED"
        else:
            parity[category]["status"] = "PASS"

    authorize_d2_stage_b(runs=runs,
                         authorization_id=STAGE_B_AUTHORIZATION_ID)
    if STAGE_B_DIR.exists() or STAGE_B_DIR.is_symlink():
        raise D2Error("D2 refuses to reuse an existing Stage B output directory")
    STAGE_B_DIR.mkdir(parents=False, exist_ok=False)
    _reject_reparse_ancestors(STAGE_B_DIR)
    try:
        return _execute_stage_b(
            runs=runs, category_info=category_info, parity=parity,
            authorization_id=authorization_id, approved_commit=approved_commit,
            contract_sha=contract_sha, source_hashes=source_hashes)
    except BaseException as exc:
        _write_failure(STAGE_B_DIR, "stage_b", exc)
        raise


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run-all")
    run.add_argument("--approved-commit", required=True)
    run.add_argument("--authorization-id", required=True)
    run.add_argument("--human-authorized", action="store_true")
    args = parser.parse_args(argv)
    if args.command != "run-all":
        parser.error("unknown D2 action")
    if not args.human_authorized:
        parser.error("D2 full-stream execution requires --human-authorized")
    try:
        result = execute_d2(approved_commit=args.approved_commit,
                            authorization_id=args.authorization_id)
    except BaseException as exc:
        print(json.dumps({"status": "INCOMPLETE", "failure_type": type(exc).__name__,
                          "message": str(exc), "automatic_retry": False},
                         indent=2, sort_keys=True))
        return 1
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
