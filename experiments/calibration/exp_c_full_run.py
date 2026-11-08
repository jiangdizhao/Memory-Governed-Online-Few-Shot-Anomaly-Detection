"""Future authorized Experiment C Stage A/B entry point; never run during implementation."""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import time
import traceback
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping

# Capture before importing research helpers that may change process CUDA visibility.
AUTHORIZED_CUDA_VISIBILITY = os.environ.get("CUDA_VISIBLE_DEVICES")

from experiments.calibration.e6_bea_closed_loop import (
    ArmRuntime,
    NormalCoverageMemory,
    OnDemandLabelReader,
    R7FrozenFeatureEncoder,
    StageBLabelSource,
    _canonical_json,
    _load_stage_b_identity_records,
    _reject_reparse_ancestors,
    _sha256_file,
    _validate_stage_b_identity_records,
    build_r7_native_arm_engines,
    load_r7_frozen_backbone,
    validate_data_role_group_disjointness,
    verify_e6_execution_identity,
)
from experiments.calibration.e6_bea_d2 import (
    D2_EXPECTED_COUNTS,
    D2_STAGE_B_AUTHORIZATION_ID,
    D2PolicyController,
    d2_stage_b_evaluate,
)
from experiments.calibration.exp_c_memory_evidence import (
    ARM_IDS,
    C_MANIFEST_SCHEMA,
    CATEGORIES,
    CBEAPriceController,
    CPrivateFeedbackSink,
    MAX_CATEGORY_OUTPUT_BYTES,
    MIN_FREE_DISK_BYTES,
    COutputBudget,
    CScoreLedger,
    CStageAClosedLoop,
    CONTRASTS,
    ExperimentCError,
    build_stage_b_metrics,
)
from experiments.calibration.exp_c_historical_parity import (
    validate_c_stage_a,
    verify_c_d2_public_parity,
    verify_d2_metric_reproduction,
)


ROOT = Path(__file__).resolve().parents[2]
OUTPUT_BASE = ROOT / "outputs" / "calibration"
RUN_PREFIX = "exp_c_memory_evidence_"
AUTH_SCHEMA = "exp-c.full-run-authorization.v1"
PROTOCOL_PATH = ROOT / "docs" / "calibration" / "EXP_C_MEMORY_CONDITIONED_EVIDENCE.md"
D2_RUN_ID = "20261104_b1"
D2_STAGE_B_DIR = OUTPUT_BASE / f"e6_bea_d2_stage_b_{D2_RUN_ID}"
CATEGORIES_ORDERED = ("audiojack", "regulator", "switch", "rolled_strip_base")
CATEGORY_DEADLINE_SECONDS = 7 * 24 * 3600
STAGE_B_DEADLINE_SECONDS = 24 * 3600
SOURCE_FILES = (
    "experiments/calibration/exp_c_memory_evidence.py",
    "experiments/calibration/exp_c_historical_parity.py",
    "experiments/calibration/exp_c_full_run.py",
    "experiments/calibration/e6_bea_policy.py",
    "experiments/calibration/e6_bea_d2.py",
    "experiments/calibration/e6_bea_d2_runner.py",
    "experiments/calibration/e6_bea_closed_loop.py",
    "experiments/calibration/e6_bea_runner.py",
    "experiments/calibration/r1_matched_correction_pilot.py",
    "experiments/calibration/r1_runtime_supervisor.py",
    "experiments/calibration/r2_exp2a_c1_strength.py",
    "experiments/calibration/r3_fullstream_2x2.py",
    "experiments/calibration/r6_capacity1024_audiojack_runner.py",
    "experiments/calibration/r7_shuffled_correction_4class.py",
    "experiments/calibration/native.py",
    "experiments/calibration/profiles.py",
    "experiments/calibration/controllers.py",
    "experiments/calibration/stage05.py",
    "experiments/p0_protocol_audit.py",
    "experiments/v02_memory.py",
    "test_phase3_AIF_simplified_v22.py",
    "vmb_visionad_new.py",
    "docs/calibration/EXP_C_MEMORY_CONDITIONED_EVIDENCE.md",
)


def _write_once(path: Path, value: Mapping[str, Any]) -> str:
    _reject_reparse_ancestors(path)
    payload = _canonical_json(dict(value))
    with Path(path).open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    return hashlib.sha256(payload).hexdigest()


def _write_text_once(path: Path, value: str) -> str:
    _reject_reparse_ancestors(path)
    payload = value.encode("utf-8")
    with Path(path).open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    return hashlib.sha256(payload).hexdigest()


def _git(args: list[str]) -> str:
    result = subprocess.run(args, cwd=ROOT, check=True, capture_output=True,
                            text=True, encoding="utf-8")
    return result.stdout.strip()


def _source_fingerprints() -> dict[str, str]:
    values = {}
    for relative in SOURCE_FILES:
        path = ROOT / relative
        if not path.is_file():
            raise ExperimentCError(f"registered C source is missing: {relative}")
        values[relative] = _sha256_file(path)
    return values


def _verify_source_identity(approved_commit: str) -> dict[str, Any]:
    if not re.fullmatch(r"[0-9a-f]{40}", approved_commit):
        raise ExperimentCError("C full-run authorization requires a full commit SHA")
    branch = _git(["git", "branch", "--show-current"])
    head = _git(["git", "rev-parse", "HEAD"]).lower()
    if branch != "calibration-stage0-contract-20261029" or head != approved_commit:
        raise ExperimentCError("C authorized branch or HEAD differs from the source commit")
    dirty = _git(["git", "status", "--porcelain", "--", *SOURCE_FILES])
    if dirty:
        raise ExperimentCError("C execution source paths have uncommitted changes")
    return {"branch": branch, "commit": head,
            "e6_native_identity": verify_e6_execution_identity(head, root=ROOT)}


def _load_authorization(path: Path, run_id: str) -> dict[str, Any]:
    try:
        auth = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ExperimentCError("C execution authorization is unreadable") from exc
    protocol_sha = _sha256_file(PROTOCOL_PATH)
    if (auth.get("schema") != AUTH_SCHEMA or
            auth.get("authorization_status") != "AUTHORIZE_FULL_REAL_IAD_STAGE_A_B" or
            auth.get("independent_review_status") != "PASS" or
            not isinstance(auth.get("authorized_by"), str) or
            not auth.get("authorized_by", "").strip() or
            auth.get("source_commit") != _git(["git", "rev-parse", "HEAD"]).lower() or
            auth.get("protocol_sha256") != protocol_sha or auth.get("run_id") != run_id or
            auth.get("categories") != list(CATEGORIES_ORDERED) or
            auth.get("allow_full_run") is not True or
            auth.get("stage_b_after_all_stage_a_seals") is not True or
            auth.get("confirmation_data_access") is not False or
            auth.get("test_masks_access") is not False or
            auth.get("historical_d2_run_id") != D2_RUN_ID):
        raise ExperimentCError("C execution authorization is not bound to this source, protocol, run, and scope")
    return {"schema": AUTH_SCHEMA, "authorization_status": auth["authorization_status"],
            "authorization_file_sha256": _sha256_file(path), "source_commit": auth["source_commit"],
            "independent_review_status": "PASS", "authorized_by": auth["authorized_by"],
            "protocol_sha256": protocol_sha, "run_id": run_id,
            "categories": list(CATEGORIES_ORDERED), "allow_full_run": True,
            "stage_b_after_all_stage_a_seals": True, "confirmation_data_access": False,
            "test_masks_access": False, "historical_d2_run_id": D2_RUN_ID}


def _run_root(run_id: str) -> Path:
    if not re.fullmatch(r"exp_c_[a-z0-9][a-z0-9_-]{5,63}", run_id):
        raise ExperimentCError("C Run ID must be immutable and match the registered pattern")
    return OUTPUT_BASE / f"{RUN_PREFIX}{run_id}"


def _category_receipt(info: Mapping[str, Any]) -> dict[str, Any]:
    from experiments.calibration.e6_bea_d2_runner import _stream_receipt
    return _stream_receipt(info)


def _preflight_categories() -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    from experiments.calibration.e6_bea_runner import load_e6_stage05_category
    from experiments.calibration import r3_fullstream_2x2 as r3

    category_info = {}
    receipts = {}
    for category in CATEGORIES_ORDERED:
        info = load_e6_stage05_category(category, repository_root=ROOT)
        if len(info["events"]) != D2_EXPECTED_COUNTS[category]:
            raise ExperimentCError(f"{category}: preflight event count differs from frozen D2")
        category_info[category] = info
        receipts[category] = _category_receipt(info)
    return {"model_receipt": dict(r3._get_verified_model_receipt()),
            "source_hashes": _source_fingerprints()}, category_info | {"_receipts": receipts}


def _initial_state_receipt(engines: Mapping[str, Any], coverages: Mapping[str, Any]) -> dict[str, Any]:
    states = {}
    for arm in ARM_IDS:
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
    if (len({row["scorer_state_sha256"] for row in states.values()}) != 1 or
            len({row["fixed_bank_sha256"] for row in states.values()}) != 1 or
            len({row["coverage_state_sha256"] for row in states.values()}) != 1):
        raise ExperimentCError("C initial seven-arm state/input parity failed")
    mutable = [set(engines[arm].owned_mutable_state_ids) for arm in ARM_IDS]
    if any(mutable[i] & mutable[j] for i in range(len(mutable)) for j in range(i + 1, len(mutable))):
        raise ExperimentCError("C arms share mutable native state")
    return states


def _resource_snapshot() -> dict[str, Any]:
    receipt: dict[str, Any] = {"host_rss_bytes": None,
                               "cuda_peak_allocated_bytes": None,
                               "cuda_peak_reserved_bytes": None,
                               "cuda_free_bytes": None, "cuda_total_bytes": None,
                               "cuda_device_name": None}
    try:
        import psutil
        receipt["host_rss_bytes"] = int(psutil.Process().memory_info().rss)
    except (ImportError, OSError):
        pass
    try:
        import torch
        if torch.cuda.is_available():
            free_bytes, total_bytes = torch.cuda.mem_get_info()
            receipt["cuda_free_bytes"] = int(free_bytes)
            receipt["cuda_total_bytes"] = int(total_bytes)
            receipt["cuda_device_name"] = str(torch.cuda.get_device_name())
            receipt["cuda_peak_allocated_bytes"] = int(torch.cuda.max_memory_allocated())
            receipt["cuda_peak_reserved_bytes"] = int(torch.cuda.max_memory_reserved())
    except Exception:
        pass
    return receipt


def _stage_a_child(run_id: str, category: str, authorization_file: Path) -> int:
    root = _run_root(run_id)
    run_dir = root / "stage_a" / category
    failure_path = run_dir / "startup_failure.json"
    try:
        auth = _load_authorization(authorization_file, run_id)
        source_identity = _verify_source_identity(auth["source_commit"])
        contract_path = root / "run_contract.json"
        contract = json.loads(contract_path.read_text(encoding="utf-8"))
        if (contract.get("run_id") != run_id or contract.get("authorization") != auth or
                contract.get("protocol_sha256") != auth["protocol_sha256"] or
                contract.get("source_commit") != auth["source_commit"]):
            raise ExperimentCError("C child identity differs from the frozen parent contract")
        if _source_fingerprints() != contract.get("source_hashes"):
            raise ExperimentCError("C source fingerprint changed after parent preflight")

        # Import all historical GPU helpers before restoring the authorized device view.
        from experiments import p0_protocol_audit as p0
        from experiments.calibration import r1_matched_correction_pilot as r1
        from experiments.calibration import r2_exp2a_c1_strength as exp2a
        from experiments.calibration import r3_fullstream_2x2 as r3
        from experiments.calibration import r6_capacity1024_audiojack_runner as r6
        from experiments.calibration import r7_shuffled_correction_4class as r7
        from experiments.calibration import native, profiles
        from experiments.calibration.e6_bea_runner import (
            _support_cls_features, load_e6_stage05_category,
        )
        r3._restore_cuda_visibility(AUTHORIZED_CUDA_VISIBILITY)
        import torch
        if not torch.cuda.is_available():
            raise ExperimentCError("authorized Stage A CUDA device is not available")
        resource_start = _resource_snapshot()

        info = load_e6_stage05_category(category, repository_root=ROOT)
        if len(info["events"]) != D2_EXPECTED_COUNTS[category]:
            raise ExperimentCError(f"{category}: child event count differs from frozen stream")
        impl = p0.require_sources()
        visionad = importlib.import_module("vmb_visionad_new")
        r3._restore_cuda_visibility(AUTHORIZED_CUDA_VISIBILITY)
        if not torch.cuda.is_available():
            raise ExperimentCError("authorized Stage A CUDA device became unavailable after imports")
        backbone, model_receipt, runtime_info = load_r7_frozen_backbone(
            impl=impl, visionad=visionad, r1=r1, r3=r3,
            approved_e6_commit=auth["source_commit"], repository_root=ROOT)
        cfg, _native_profile = r3._native_config(p0, profiles, r1)
        support_features = p0.encode_reference_features(
            impl, backbone, list(info["support_paths"]), cfg)
        support_cls = _support_cls_features(impl, backbone,
                                            list(info["support_paths"]), cfg.feature_fuse)
        c64_items, c64_feature_receipt = r7._make_c64_items(
            category, info, p0, impl, backbone, cfg)
        engines = build_r7_native_arm_engines(
            category=category, info=info, impl=impl, visionad=visionad,
            p0=p0, r1=r1, r3=r3, r6=r6, r7=r7, exp2a=exp2a,
            native_module=native, cfg=cfg, support_features=support_features,
            support_paths=info["support_paths"], c64_items=c64_items,
            approved_e6_commit=auth["source_commit"], repository_root=ROOT,
            arm_ids=ARM_IDS)
        validate_data_role_group_disjointness(
            info["events"], engines["B"].input_identity["support_source_group_sha256"],
            engines["B"].input_identity["c64_source_group_sha256"])
        coverages = {arm: NormalCoverageMemory(support_cls) for arm in ARM_IDS}
        states = _initial_state_receipt(engines, coverages)
        input_identity = engines["B"].input_identity
        if any(engine.input_identity != input_identity for engine in engines.values()):
            raise ExperimentCError("C arm input identities differ")
        if (str(model_receipt["loaded_model_state_sha256"]) !=
                runtime_info.get("loaded_model_state_sha256")):
            raise ExperimentCError("C model receipt differs from loaded backbone state")

        output_files = {"run_manifest.json", "sealed_scores.jsonl", "stage_a_seal.json",
                        "resource_receipt.json", "supervisor_receipt.json",
                        *[f"private_feedback_{arm}.jsonl" for arm in ARM_IDS]}
        manifest = {
            "schema": C_MANIFEST_SCHEMA, "status_at_creation": "INCOMPLETE",
            "stage_b_enabled": False, "run_id": run_id, "category": category,
            "registered_categories": list(CATEGORIES_ORDERED),
            "registered_arm_ids": list(ARM_IDS), "stream_event_count": len(info["events"]),
            "stream_identity_sha256": _category_receipt(info)["event_stream_identity_sha256"],
            "approved_commit": auth["source_commit"], "branch": source_identity["branch"],
            "protocol_sha256": auth["protocol_sha256"],
            "authorization_file_sha256": auth["authorization_file_sha256"],
            "source_sha256": dict(contract["source_hashes"]),
            "model_identity": {"loaded_model_state_sha256": model_receipt["loaded_model_state_sha256"],
                               "model_receipt": dict(model_receipt),
                               "runtime_profile": dict(runtime_info),
                               "verified_frozen_backbone": True},
            "support_and_c64_identity": _category_receipt(info),
            "input_identity": dict(input_identity),
            "initial_arm_states": states,
            "native_scoring_configuration": {
                "p0": engines["B"].native_scorer_config,
                "r7_profile": engines["B"].native_profile_config,
                "c1": engines["B"].correction_config_record,
                "greedy": engines["B"].greedy_settings_record},
            "initial_c64_numerical_parity": {
                arm: dict(engines[arm].c64_parity_receipt) for arm in ARM_IDS},
            "c64_feature_cache_receipt": dict(c64_feature_receipt),
            "arm_objectives": {
                "RAND37": "exact D2 identity-hashed Bernoulli",
                "MARGIN37": "exact D2 cold-start, q=.963, tie and identity-hash policy",
                "B": ["B"], "BD": ["B", "D"],
                "BN_BAND": ["B", "N", "D_band"],
                "BN_AVG_GATE": ["B", "N", "g_bar(t-1)*D_band"],
                "BND": ["B", "N", "D"]},
            "average_gate": {"window": 256, "weight": "D_band",
                             "empty_or_zero_denominator": 1.0,
                             "history_is_arm_local": True,
                             "current_record_append_phase": "after decision seal and feedback completion"},
            "bea_price": {"initial": 0.80, "target_rate": 0.03,
                          "executed_query_window": 256, "learning_rate": 0.02,
                          "bounds": [0.05, 3.0], "proposal": "objective > current price",
                          "update_phase": "after current action, affects future events"},
            "unchanged_d2_arms": list(UNCHANGED_D2_ARMS),
            "runtime": {"python": sys.version, "platform": platform.platform(),
                        "torch": str(torch.__version__), "cuda_execution": True,
                        "gpu_monitoring": False, "automatic_retry": False},
            "output_files": sorted(output_files),
        }
        manifest_path = run_dir / "run_manifest.json"
        manifest_sha = _write_once(manifest_path, manifest)
        budget = COutputBudget(run_dir)
        ledger = CScoreLedger(run_dir / "sealed_scores.jsonl", budget)
        readers, sinks = [], []
        arms = {}
        try:
            for arm in ARM_IDS:
                reader = OnDemandLabelReader(
                    info["json_path"], category, info["contract"]["category_json_sha256"],
                    info["events"], arm, ledger, registered_arm_ids=ARM_IDS)
                sink = CPrivateFeedbackSink(run_dir / f"private_feedback_{arm}.jsonl", arm, budget)
                readers.append(reader)
                sinks.append(sink)
                controller = (CBEAPriceController(arm) if arm in {"BN_BAND", "BN_AVG_GATE"}
                              else D2PolicyController(arm, seed=37))
                arms[arm] = ArmRuntime(arm, engines[arm], controller, coverages[arm],
                                       reader, sink)
            encoder = R7FrozenFeatureEncoder(
                impl=impl, visionad=visionad, repository_root=ROOT, backbone=backbone,
                feature_fuse=cfg.feature_fuse, r1=r1,
                expected_model_state_sha256=model_receipt["loaded_model_state_sha256"],
                runtime_info=runtime_info)
            coordinator = CStageAClosedLoop(
                category=category, run_id=run_id, events=info["events"],
                feature_encoder=encoder, arms=arms, ledger=ledger, run_dir=run_dir,
                manifest_path=manifest_path, manifest_sha256=manifest_sha,
                authorization_id=auth["authorization_file_sha256"],
                expected_event_count=D2_EXPECTED_COUNTS[category])
            seal = coordinator.run()
            if not seal.complete:
                raise ExperimentCError("C coordinator did not return a complete seal")
        # Resource samples are separate so the durable scientific seal is immutable.
            _write_once(run_dir / "resource_receipt.json", {
                "schema": "exp-c.resource-receipt.v1", "category": category,
            "run_id": run_id, "score_and_feedback_bytes": budget.bytes_written,
            "score_and_feedback_budget_bytes": budget.max_bytes,
            "resource_start": resource_start,
            "resource_end_and_cuda_peak": _resource_snapshot()})
        except BaseException:
            ledger.close()
            for reader in readers:
                reader.close()
            for sink in sinks:
                sink.close()
            raise
        return 0
    except BaseException as exc:
        receipt = {"schema": "exp-c.stage-a-startup-failure.v1", "run_id": run_id,
                   "category": category, "exception_type": type(exc).__name__,
                   "message": str(exc), "traceback": traceback.format_exc(),
                   "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
        try:
            target = failure_path if run_dir.is_dir() else root / "stage_a" / f"{category}.startup_failure.json"
            _write_once(target, receipt)
        except FileExistsError:
            pass
        return 1


def _supervise_category(run_id: str, category: str, authorization_file: Path,
                        run_dir: Path, source_commit: str,
                        authorization_sha256: str) -> dict[str, Any]:
    from experiments.calibration import r1_runtime_supervisor as supervisor
    from experiments.calibration import r3_fullstream_2x2 as r3

    env = r3._child_environment(os.environ, "STAGE_A", AUTHORIZED_CUDA_VISIBILITY)
    command = [sys.executable, "-m", "experiments.calibration.exp_c_full_run",
               "--stage-a-child", "--run-id", run_id, "--category", category,
               "--authorization-file", str(authorization_file.resolve())]
    result = supervisor.run_supervised_child(command, timeout_seconds=CATEGORY_DEADLINE_SECONDS,
                                              env=env, cwd=ROOT)
    receipt = asdict(result)
    receipt.update({"schema": "exp-c.stage-a-supervisor-receipt.v1", "run_id": run_id,
                    "category": category, "source_commit": source_commit,
                    "authorization_sha256": authorization_sha256,
                    "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
    if (result.status != "COMPLETED" or result.child_exit_code != 0 or
            not result.termination_confirmed or result.active_job_processes != 0):
        failure = None
        for path in (run_dir / "startup_failure.json",
                     run_dir.parent / f"{category}.startup_failure.json"):
            if path.is_file():
                failure = json.loads(path.read_text(encoding="utf-8"))
                break
        raise ExperimentCError(f"{category}: supervised Stage A failed; result={receipt}; child_failure={failure}")
    _write_once(run_dir / "supervisor_receipt.json", receipt)
    return receipt


def run_full(*, run_id: str, authorization_file: Path) -> Path:
    auth = _load_authorization(authorization_file, run_id)
    identity = _verify_source_identity(auth["source_commit"])
    root = _run_root(run_id)
    if root.exists() or root.is_symlink():
        raise ExperimentCError(f"C refuses to reuse an existing run namespace: {root}")
    preflight, bundle = _preflight_categories()
    receipts = bundle.pop("_receipts")
    free_bytes = shutil.disk_usage(OUTPUT_BASE).free
    if free_bytes < MIN_FREE_DISK_BYTES:
        raise ExperimentCError(
            f"C resource preflight found {free_bytes} free bytes; registered minimum is {MIN_FREE_DISK_BYTES}")
    root.mkdir(parents=False, exist_ok=False)
    stage_a = root / "stage_a"
    stage_a.mkdir()
    for category in CATEGORIES_ORDERED:
        (stage_a / category).mkdir()
    contract = {
        "schema": "exp-c.run-contract.v1", "status_at_creation": "INCOMPLETE",
        "run_id": run_id, "source_commit": auth["source_commit"],
        "protocol_sha256": auth["protocol_sha256"], "authorization": auth,
        "source_hashes": preflight["source_hashes"],
        "source_identity": identity, "model_receipt": preflight["model_receipt"],
        "categories": receipts, "historical_d2_run_id": D2_RUN_ID,
        "stage_b_enabled": False, "confirmation_data_access": False,
        "test_masks_access": False, "automatic_retry": False,
        "resource_limits": {"max_score_and_feedback_bytes_per_category": MAX_CATEGORY_OUTPUT_BYTES,
                            "aggregate_score_and_feedback_upper_bound":
                                len(CATEGORIES_ORDERED) * MAX_CATEGORY_OUTPUT_BYTES,
                            "minimum_free_disk_bytes_at_start": MIN_FREE_DISK_BYTES,
                            "stage_a_child_deadline_seconds": CATEGORY_DEADLINE_SECONDS,
                            "stage_b_child_deadline_seconds": STAGE_B_DEADLINE_SECONDS,
                            "gpu_memory_measurement": "torch CUDA peak counters when available",
                            "host_memory_measurement": "optional psutil RSS snapshot"},
    }
    contract_sha = _write_once(root / "run_contract.json", contract)
    _write_once(root / "run_namespace.json", {
        "schema": "exp-c.run-namespace.v1", "status_at_creation": "INCOMPLETE",
        "run_id": run_id, "source_commit": auth["source_commit"],
        "protocol_sha256": auth["protocol_sha256"], "contract_sha256": contract_sha})
    category_runs = {}
    try:
        for category in CATEGORIES_ORDERED:
            run_dir = stage_a / category
            _supervise_category(run_id, category, authorization_file, run_dir,
                                auth["source_commit"], auth["authorization_file_sha256"])
            from experiments.calibration.e6_bea_runner import load_e6_stage05_category
            from experiments.calibration.exp_c_historical_parity import validate_c_stage_a
            info = load_e6_stage05_category(category, repository_root=ROOT)
            category_runs[category] = validate_c_stage_a(
                run_dir=run_dir, category=category, run_id=run_id,
                expected_commit=auth["source_commit"], protocol_sha256=auth["protocol_sha256"],
                expected_events=info["events"])
        parity = {}
        for category in CATEGORIES_ORDERED:
            d2_dir = OUTPUT_BASE / f"e6_bea_d2_{D2_RUN_ID}_{category}"
            info = bundle[category]
            parity[category], _d2run = verify_c_d2_public_parity(
                c_run=category_runs[category], d2_run_dir=d2_dir,
                expected_events=info["events"])
        public_parity_sha = _write_once(root / "historical_public_parity.json", {
            "schema": "exp-c.d2-public-parity.v1", "status": "PASS",
            "run_id": run_id, "historical_d2_run_id": D2_RUN_ID,
            "categories": parity})
        stage_b_dir = root / "stage_b"
        stage_b_dir.mkdir()
        _supervise_stage_b(run_id, authorization_file, stage_b_dir,
                           auth["source_commit"], auth["authorization_file_sha256"])
        evaluation_path = stage_b_dir / "evaluation.json"
        evaluation_sha = _sha256_file(evaluation_path)
        report_sha = _sha256_file(stage_b_dir / "report.md")
        evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
        parity_path = stage_b_dir / "d2_metric_reproduction.json"
        metric_parity = json.loads(parity_path.read_text(encoding="utf-8"))
        if metric_parity.get("status") != "PASS":
            raise ExperimentCError("C Stage B completed without required D2 metric reproduction")
        complete = {"schema": "exp-c.campaign-seal.v1", "status": "SEALED_COMPLETE",
                    "run_id": run_id, "source_commit": auth["source_commit"],
                    "protocol_sha256": auth["protocol_sha256"], "contract_sha256": contract_sha,
                    "stage_a": {category: {
                        "manifest_sha256": category_runs[category].manifest_sha256,
                        "seal_sha256": category_runs[category].seal_sha256,
                        "score_sha256": category_runs[category].score_sha256,
                        "resource_receipt_sha256": category_runs[category].resource_sha256,
                        "supervisor_receipt_sha256": category_runs[category].supervisor_sha256}
                                for category in CATEGORIES_ORDERED},
                    "historical_public_parity": "PASS",
                    "historical_public_parity_sha256": public_parity_sha,
                    "historical_metric_reproduction": "PASS",
                    "stage_b_evaluation_sha256": evaluation_sha,
                    "stage_b_report_sha256": report_sha,
                    "outcomes_saved_only_as_aggregates": True,
                    "confirmation_data_access": False, "test_masks_access": False,
                    "completed_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
        _write_once(root / "campaign_seal.json", complete)
        return root
    except BaseException as exc:
        try:
            _write_once(root / "campaign_failure.json", {
                "schema": "exp-c.campaign-failure.v1", "run_id": run_id,
                "exception_type": type(exc).__name__, "message": str(exc),
                "traceback": traceback.format_exc(), "automatic_retry": False})
        except FileExistsError:
            pass
        raise


def _supervise_stage_b(run_id: str, authorization_file: Path, stage_b_dir: Path,
                       source_commit: str, authorization_sha256: str) -> None:
    from experiments.calibration import r1_runtime_supervisor as supervisor
    from experiments.calibration import r3_fullstream_2x2 as r3
    env = r3._child_environment(os.environ, "STAGE_B", None)
    command = [sys.executable, "-m", "experiments.calibration.exp_c_full_run",
               "--stage-b-child", "--run-id", run_id,
               "--authorization-file", str(authorization_file.resolve())]
    result = supervisor.run_supervised_child(command, timeout_seconds=STAGE_B_DEADLINE_SECONDS,
                                              env=env, cwd=ROOT)
    receipt = asdict(result)
    receipt.update({"schema": "exp-c.stage-b-supervisor-receipt.v1", "run_id": run_id,
                    "source_commit": source_commit, "authorization_sha256": authorization_sha256})
    if (result.status != "COMPLETED" or result.child_exit_code != 0 or
            not result.termination_confirmed or result.active_job_processes != 0):
        raise ExperimentCError(f"C CPU Stage B failed under Windows Job Object: {receipt}")
    _write_once(stage_b_dir / "supervisor_receipt.json", receipt)


def _render_report(evaluation: Mapping[str, Any]) -> str:
    lines = ["# Experiment C — Memory-Conditioned Evidence Acquisition", "",
             "- Classification: DEVELOPMENT_EXPLORATORY; four development categories only.",
             "- Primary outcome: category-equal macro prequential BER.",
             "- Budget matching requires absolute query-rate difference <= 0.003, per category and macro.",
             "- Source Group bootstrap intervals are descriptive conditional-trajectory intervals.",
             "- No independent online-history or confirmation-set claim is made.", "",
             "## Category-equal macro", "",
             "| Arm | BER | BA | AUROC | AP | FPR | FNR | Query rate |",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for arm in ARM_IDS:
        row = evaluation["category_equal_macro"][arm]
        lines.append("| {arm} | {ber:.6f} | {ba:.6f} | {auroc:.6f} | {ap:.6f} | "
                     "{fpr:.6f} | {fnr:.6f} | {rate:.6f} |".format(
                         arm=arm, ber=row["ber"], ba=row["ba"], auroc=row["auroc"],
                         ap=row["ap"], fpr=row["fpr"], fnr=row["fnr"], rate=row["query_rate"]))
    lines.extend(["", "## Registered BER contrasts", "",
                  "| Contrast | BER difference | Query-rate difference | Budget classification |",
                  "|---|---:|---:|---|"])
    for name, _left, _right in CONTRASTS:
        row = evaluation["contrasts"]["CATEGORY_EQUAL_MACRO"][name]
        lines.append(f"| {name} | {row['ber_difference_left_minus_right']:.6f} | "
                     f"{row['query_rate_difference_left_minus_right']:.6f} | "
                     f"{row['budget_classification']} |")
    lines.extend(["", "## Mechanism diagnostics", "", "```json",
                  json.dumps(evaluation["mechanism_diagnostics"], sort_keys=True, indent=2),
                  "```", ""])
    return "\n".join(lines)


def _stage_b_child(run_id: str, authorization_file: Path) -> int:
    root = _run_root(run_id)
    stage_b_dir = root / "stage_b"
    try:
        auth = _load_authorization(authorization_file, run_id)
        _verify_source_identity(auth["source_commit"])
        from experiments.calibration import r3_fullstream_2x2 as r3
        r3._restore_cuda_visibility("")
        contract = json.loads((root / "run_contract.json").read_text(encoding="utf-8"))
        if contract.get("authorization") != auth or contract.get("stage_b_enabled") is not False:
            raise ExperimentCError("C Stage B authorization differs from the frozen run contract")
        from experiments.calibration.e6_bea_runner import load_e6_stage05_category
        from experiments.calibration.exp_c_historical_parity import (
            validate_c_stage_a, verify_c_d2_public_parity,
        )
        from experiments.calibration.e6_bea_d2_runner import STAGE_A_DIRS as D2_STAGE_A_DIRS
        from experiments.calibration.e6_bea_d2 import d2_stage_b_evaluate

        runs, d2_runs, infos = {}, {}, {}
        for category in CATEGORIES_ORDERED:
            info = load_e6_stage05_category(category, repository_root=ROOT)
            infos[category] = info
            c_run = validate_c_stage_a(
                run_dir=root / "stage_a" / category, category=category, run_id=run_id,
                expected_commit=auth["source_commit"], protocol_sha256=auth["protocol_sha256"],
                expected_events=info["events"])
            parity, d2_run = verify_c_d2_public_parity(
                c_run=c_run,
                d2_run_dir=OUTPUT_BASE / f"e6_bea_d2_{D2_RUN_ID}_{category}",
                expected_events=info["events"])
            if parity.get("status") != "PASS":
                raise ExperimentCError(f"{category}: D2 public parity did not pass before Stage B")
            runs[category], d2_runs[category] = c_run, d2_run
        if len(runs) != len(CATEGORIES_ORDERED) or len(d2_runs) != len(CATEGORIES_ORDERED):
            raise ExperimentCError("C Stage B requires all four C and D2 seals before labels")

        # This is the first operation that decodes complete TEST labels.
        labels = {}
        for category in CATEGORIES_ORDERED:
            source = StageBLabelSource(Path(infos[category]["json_path"]),
                                       infos[category]["contract"]["category_json_sha256"])
            records = _load_stage_b_identity_records(source, category, runs[category].rows)
            labels[category] = _validate_stage_b_identity_records(runs[category].rows, records)
        counts = {cat: runs[cat].seal["arm_query_counts"] for cat in CATEGORIES_ORDERED}
        evaluation = build_stage_b_metrics(
            labels, {cat: runs[cat].rows for cat in CATEGORIES_ORDERED}, counts)
        evaluation["state_and_resource_costs"] = {
            category: {
                "stage_a_runtime_seconds": float(runs[category].seal["runtime_seconds"]),
                "score_and_feedback_bytes": int(runs[category].seal["score_and_feedback_bytes"]),
                "resource_receipt": json.loads((runs[category].run_dir /
                                                 "resource_receipt.json").read_text(encoding="utf-8")),
                "per_arm": {
                    arm: {
                        **dict(runs[category].seal["state_change_and_feedback_cost"][arm]),
                        "final_state": dict(runs[category].seal["final_arm_state"][arm]),
                        "greedy_writer_final": dict(runs[category].seal["greedy_writer_final"][arm]),
                        "correction_bank_final_size": int(
                            runs[category].seal["correction_bank_final_size"][arm]),
                    } for arm in ARM_IDS},
                "shared_encoding_seconds_once": float(sum(
                    float(row["arm_rows"]["B"]["encoding_seconds"])
                    for row in runs[category].rows)),
                "native_score_seconds_all_arms": float(sum(
                    float(row["arm_rows"][arm]["native_score_seconds"])
                    for row in runs[category].rows for arm in ARM_IDS)),
            } for category in CATEGORIES_ORDERED}
        # Re-evaluate the unchanged D2 trajectories with C's authorized official-label projection.
        d2_eval = d2_stage_b_evaluate(
            runs=d2_runs, labels_by_category=labels,
            authorization_id=D2_STAGE_B_AUTHORIZATION_ID)
        metric_parity = verify_d2_metric_reproduction(
            d2_run_id=D2_RUN_ID, c_evaluation=evaluation,
            d2_evaluation=d2_eval, d2_stage_b_dir=D2_STAGE_B_DIR)
        evaluation["historical_d2_public_trajectory_reproduction"] = metric_parity
        evaluation["stage_a_seals_verified_before_outcomes"] = True
        evaluation["historical_public_parity_passed_before_outcomes"] = True
        evaluation["test_labels_saved_or_printed"] = False
        evaluation["test_masks_opened"] = False
        evaluation["classification"] = "DEVELOPMENT_EXPLORATORY"
        _write_once(stage_b_dir / "evaluation.json", evaluation)
        _write_once(stage_b_dir / "d2_metric_reproduction.json", metric_parity)
        _write_text_once(stage_b_dir / "report.md", _render_report(evaluation))
        # Keep only aggregate evaluation output; discard in-memory label vectors.
        labels.clear()
        return 0
    except BaseException as exc:
        try:
            _write_once(stage_b_dir / "failure.json", {
                "schema": "exp-c.stage-b-failure.v1", "run_id": run_id,
                "exception_type": type(exc).__name__, "message": str(exc),
                "traceback": traceback.format_exc(), "automatic_retry": False})
        except FileExistsError:
            pass
        return 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--full-run", action="store_true")
    group.add_argument("--stage-a-child", action="store_true")
    group.add_argument("--stage-b-child", action="store_true")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--authorization-file", required=True, type=Path)
    parser.add_argument("--category", choices=CATEGORIES_ORDERED)
    args = parser.parse_args(argv)
    try:
        if args.full_run:
            if args.category:
                parser.error("--category is only valid for supervised Stage A children")
            print(str(run_full(run_id=args.run_id, authorization_file=args.authorization_file)))
        elif args.stage_a_child:
            if not args.category:
                parser.error("--stage-a-child requires --category")
            return _stage_a_child(args.run_id, args.category, args.authorization_file)
        else:
            if args.category:
                parser.error("--category is not valid for Stage B")
            return _stage_b_child(args.run_id, args.authorization_file)
    except BaseException as exc:
        print(json.dumps({"status": "INCOMPLETE", "exception_type": type(exc).__name__,
                          "message": str(exc), "automatic_retry": False}, sort_keys=True))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
