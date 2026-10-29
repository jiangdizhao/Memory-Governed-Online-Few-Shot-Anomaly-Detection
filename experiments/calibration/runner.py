"""One-command Stage 0 validation and audit artifact writer.

Execution is synthetic CPU validation plus metadata-only preflight over the
four preregistered development classes. When CUDA, the existing cached DINOv2
source/checkpoint, and Real-IAD train metadata are available, it also runs a
bounded audiojack train-only native score/update parity smoke. It never opens a
test image, mask, or confirmation-category metadata.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import unittest
from typing import Any

import torch

from . import PROTOCOL_VERSION
from .controllers import theta0_from_c64
from .engine import PrequentialEngine
from .method_registry import METHOD_CARDS, build_controllers
from .metrics import PrivateOutcomeLedger
from .profiles import DEFAULT_PROFILE, canonical_json
from .protocol import (EvidenceAnchor, EvidenceSnapshot, ScoreResult,
                       StreamEvent, TapeRecord)
from .query_tape import build_qa_ts, build_qb
from .real_iad_preflight import run_metadata_preflight
from .native_event_replay import run_native_event_replay
from .native_greedy_matrix import run_native_greedy_matrix
from .native_parity_smoke import run_native_parity_smoke
from .controller_activation import run_controller_activation
from .b7_engine_activation import run_b7_engine_activation
from .sun_cs import (EMPIRICAL_EXTREMA_SENSITIVITY_POLICY,
                     PRIMARY_ENDPOINT_POLICY, algorithm3_reference,
                     run_class_scale_synthetic_diagnostic)
from .sun_cs_diagnostics import (b7_query_budget_feasibility,
                                build_diagnostic_suite,
                                clopper_pearson_zero_error_minimum)
from .source_provenance import (PROTECTED_BASELINE_PATH, source_fingerprint_paths,
                                verify_protected_sources)


ROOT = Path(__file__).resolve().parents[2]
EXPECTED_PYTHON = Path(r"D:\anaconda3\envs\torch\python.exe")
HISTORICAL_SOURCES = (
    "vmb_visionad_new.py", "test_phase3_AIF_simplified_v22.py",
    "experiments/p0_protocol_audit.py", "experiments/a0_a1_strict4_matched.py",
    "experiments/v02_memory.py", "experiments/risk_safe/native_bridge.py",
    "experiments/risk_safe/preflight.py", "experiments/risk_safe/query_protocol.py",
    "experiments/risk_safe/d2_threshold_calibration.py",
    "experiments/risk_safe/p01_falsification_native.py",
)
ARTIFACT_FILES = frozenset({
    "run_identity.json", "INCOMPLETE", "protocol_manifest.json",
    "dependency_environment.json", "gates.json",
    "controller_method_cards.json", "synthetic_event_trace.jsonl",
    "synthetic_summary.json", "test_results.json", "source_hashes.json",
    "protected_source_verification.json", "controller_activation_matrix.json",
    "sun_cs_reference_status.json",
    "b7_engine_activation.json",
    "sun_cs_class_scale_synthetic_diagnostic.json",
    "sun_cs_continuous_diagnostics.json",
    "sun_cs_endpoint_policy_comparison.json",
    "b7_query_budget_feasibility.json",
    "stage0_final_scientific_protocol_review.md",
    "stage1_evaluation_protocol_candidate.md",
    "stage0_gate_amendment_candidate.md",
    "sun_cs_a3_declared_conventions.md",
    "sun_cs_a3_conservative_endpoints_v2.md",
    "b7_activation_feasibility_reporting_v1.md",
    "real_iad_metadata_preflight.json", "native_parity_smoke.json",
    "native_greedy_parity_matrix.json", "native_event_replay.json",
    "stage0_summary.md", "checksums.json",
})


def _sha(data: bytes) -> str: return hashlib.sha256(data).hexdigest()
def _hash_file(path: Path) -> str: return _sha(path.read_bytes())


def _json_write(path: Path, value: Any) -> None:
    payload = json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n"
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(payload, encoding="utf-8")
    temp.replace(path)


def _check_environment() -> dict[str, Any]:
    actual = Path(sys.executable).resolve()
    expected = EXPECTED_PYTHON.resolve()
    if str(actual).casefold() != str(expected).casefold():
        raise RuntimeError(f"Wrong Python executable: {actual}; expected {expected}")
    if sys.prefix.casefold() != r"d:\anaconda3\envs\torch".casefold():
        raise RuntimeError(f"Wrong Python environment prefix: {sys.prefix}")
    return {"python_executable": str(actual), "python_version": sys.version,
            "environment_prefix": sys.prefix, "torch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "synthetic_execution_device": "cpu",
            "optional_native_smoke_device": (
                "cuda" if torch.cuda.is_available() else "unavailable")}


def _synthetic_run() -> dict[str, Any]:
    n = 160
    records = [TapeRecord(i, f"synthetic-group-{i:04d}", i % 5 + 1,
                          f"synthetic-view-{i:04d}", phase=0) for i in range(n)]
    tape = build_qa_ts(records)
    outcomes_raw = [int((i % 17) in (0, 1, 2)) for i in range(n)]
    outcomes = PrivateOutcomeLedger(outcomes_raw)
    c64_scores = [0.01 + i * 0.001 for i in range(64)]
    theta0 = theta0_from_c64(c64_scores)
    controllers = build_controllers(theta0, c64_scores)

    class Memory:
        version = 0
        digest = _sha(b"synthetic-empty-memory")
    memory = Memory()
    evidence_extra: list[tuple[EvidenceAnchor, int]] = []
    events = [StreamEvent(record, (i % 23) / 100.0) for i, record in enumerate(records)]

    def score_fn(event: StreamEvent) -> ScoreResult:
        score = float(event.payload) + memory.version * 0.001
        state = _sha(canonical_json({"version": memory.version,
                                     "kind": "synthetic_only"}).encode())
        return ScoreResult(score, memory.version, state)

    def write_fn(event: StreamEvent, score: ScoreResult, feedback: Any) -> dict[str, Any]:
        before = memory.digest
        if feedback.queried and feedback.label == 0:
            memory.version += 1
            memory.digest = _sha(f"synthetic-memory-{memory.version}".encode())
            changed = True
        else:
            changed = False
        return {"queried_normal_only": bool(feedback.queried and feedback.label == 0),
                "changed": changed, "before": before, "after": memory.digest}

    def evidence_fn(event: StreamEvent | None, feedback: Any) -> EvidenceSnapshot:
        if event is not None and feedback is not None:
            label = int(feedback.label)
            evidence_extra.append((EvidenceAnchor(
                f"query:{event.record.position}",
                float(event.payload) + memory.version * 0.001, memory.version,
                False, 0, "queried_normal_all_layer_leave_source_out"
                if label == 0 else "queried_defect_not_written"), label))
        c64 = tuple(EvidenceAnchor(f"c64:{i:03d}", value + memory.version * 0.001,
                                   memory.version, True, 0, "C64_PROTECTED")
                    for i, value in enumerate(c64_scores))
        def current_version(anchor: EvidenceAnchor) -> EvidenceAnchor:
            shift = (memory.version - anchor.memory_version) * 0.001
            return EvidenceAnchor(anchor.opaque_anchor_id, anchor.score + shift,
                                  memory.version, anchor.protected,
                                  anchor.source_excluded_rows,
                                  anchor.source_exclusion_policy)
        normal_extra = tuple(current_version(a) for a, label in evidence_extra if label == 0)
        defect_extra = tuple(current_version(a) for a, label in evidence_extra if label == 1)
        return EvidenceSnapshot(memory.version, c64 + normal_extra, defect_extra,
                                {"synthetic": True, "c64_mutable": False},
                                evidence_revision=len(evidence_extra) + memory.version)

    engine = PrequentialEngine(
        tape=tape, events=events, outcomes=outcomes, controllers=controllers,
        score_fn=score_fn,
        memory_state_fn=lambda: (memory.version, memory.digest),
        write_fn=write_fn, evidence_fn=evidence_fn,
        provenance_id=f"synthetic-stage0-v1:{tape.stream_digest}")
    result = engine.run()
    qb = build_qb([TapeRecord(i, f"qb-group-{i:04d}", i % 5 + 1,
                              f"qb-view-{i:04d}") for i in range(1000)])
    return {"execution_scope": "synthetic_mechanism_smoke_only",
            "stage1_or_stage2_run": False,
            "synthetic_stream_n": n,
            "theta0_synthetic_c64_higher_q90": theta0,
            "qa_ts": {"digest": tape.digest, "query_count": len(tape.events),
                      "query_fraction_report": {
                          "images": len(tape.events) / n,
                          "groups": len(tape.queried_groups) / n}},
            "qb": {"digest": qb.digest, "query_count": len(qb.events),
                   "early_budget": qb.early_budget,
                   "early_queries": sum(e.position < 400 for e in qb.events)},
            "final_memory_version": memory.version,
            "metrics": result["metrics"],
            "scores": result["scores"], "predictions": result["predictions"],
            "trace": result["trace"],
            "controller_states": result["controller_states"]}


def _run_tests() -> dict[str, Any]:
    loader = unittest.TestLoader()
    suite = loader.discover(str(ROOT / "tests" / "calibration"))
    stream = __import__("io").StringIO()
    class TrackingResult(unittest.TextTestResult):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.passed_test_ids: list[str] = []
        def addSuccess(self, test):
            self.passed_test_ids.append(test.id())
            super().addSuccess(test)
    runner = unittest.TextTestRunner(stream=stream, verbosity=2,
                                     resultclass=TrackingResult)
    started = time.time()
    result = runner.run(suite)
    details = {
        "tests_run": result.testsRun,
        "passed": result.testsRun - len(result.failures) - len(result.errors) - len(result.skipped),
        "failed": len(result.failures), "errors": len(result.errors),
        "skipped": len(result.skipped), "duration_seconds": time.time() - started,
        "passed_test_ids": result.passed_test_ids,
        "output": stream.getvalue(),
        "failures": [str(t) + "\n" + msg for t, msg in result.failures],
        "errors_detail": [str(t) + "\n" + msg for t, msg in result.errors],
        "skipped_detail": [str(t) + ": " + msg for t, msg in result.skipped],
    }
    return details


def _source_hashes() -> dict[str, str]:
    paths = [ROOT / p for p in HISTORICAL_SOURCES]
    paths.extend(ROOT / p for p in source_fingerprint_paths())
    paths.extend(sorted((ROOT / "experiments" / "calibration").glob("*.py")))
    paths.extend(sorted((ROOT / "tests" / "calibration").glob("*.py")))
    paths.extend(path for path in sorted(
        (ROOT / "docs" / "calibration").glob("*.md"))
        if path.name not in {"STAGE0_IMPLEMENTATION_REPORT.md",
                             "STAGE0_SOURCE_PROVENANCE.md",
                             "STAGE0_FIFTHPASS_IMPLEMENTATION_REPORT.md",
                             "STAGE0_FIFTHPASS_SOURCE_PROVENANCE.md",
                             "STAGE0_SIXTHPASS_IMPLEMENTATION_REPORT.md",
                             "STAGE0_SIXTHPASS_SOURCE_PROVENANCE.md"})
    paths.append(ROOT / "docs" /
                 "Designing and Prioritizing Memory-Dependent _ State-Consistent Calibration for Dynamic-Memory Indust.pdf")
    paths.append(PROTECTED_BASELINE_PATH)
    return {str(path.relative_to(ROOT)).replace("\\", "/"): _hash_file(path)
            for path in paths if path.is_file()}


def _git_info() -> tuple[str, list[str]]:
    branch = subprocess.run(["git", "branch", "--show-current"], cwd=ROOT,
                            capture_output=True, text=True, check=True).stdout.strip()
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                            capture_output=True, text=True, check=True).stdout.strip()
    status = subprocess.run(["git", "status", "--short"], cwd=ROOT,
                            capture_output=True, text=True, check=True).stdout.splitlines()
    return branch + "@" + commit, status


def _verify_checksums(out_dir: Path) -> bool:
    path = out_dir / "checksums.json"
    if not path.exists(): return False
    manifest = json.loads(path.read_text(encoding="utf-8"))
    actual = {p.name for p in out_dir.iterdir() if p.is_file() and
              p.name != "checksums.json"}
    if set(manifest) != actual:
        return False
    for rel, expected in manifest.items():
        file = out_dir / rel
        if not file.is_file() or _hash_file(file) != expected:
            return False
    return True


def _make_artifacts(out_dir: Path) -> dict[str, Any]:
    marker = out_dir / "INCOMPLETE"
    marker.write_text("WIP / STAGE0_BLOCKED: mandatory fidelity gates and scientific validation remain unresolved.\n",
                      encoding="utf-8")
    env = _check_environment()
    git, status = _git_info()
    branch, source_commit = git.split("@", 1)
    source_hashes = _source_hashes()
    protected_sources = verify_protected_sources()
    tests = _run_tests()
    tests_ok = (tests["failed"] == 0 and tests["errors"] == 0 and
                tests["skipped"] == 0)
    synthetic = (_synthetic_run() if tests_ok else {
        "execution_scope": "skipped_after_unit_test_failure",
        "trace": [], "controller_states": {}, "scores": [], "predictions": {},
    })
    dataset_root = Path(os.environ.get("REALIAD_ROOT", r"G:\dataset\realiad"))
    json_root = Path(os.environ.get("REALIAD_JSON_ROOT",
                                   r"G:\dataset\realiad\jsons\realiad_jsons"))
    real_preflight = run_metadata_preflight(dataset_root, json_root)
    if tests_ok:
        try:
            sun_continuous_diagnostics = build_diagnostic_suite(n=512)
            expected_development = {
                "audiojack", "regulator", "switch", "rolled_strip_base"}
            counts = {}
            for row in real_preflight.get("classes", []):
                n_images = int(row["test_stream"]["images"])
                query_count = int(row["Q-A-TS"]["query_count"])
                if query_count != math.floor(.03 * n_images + .5):
                    raise RuntimeError(
                        f"Metadata Q-A-TS budget mismatch: {row['class']}")
                counts[row["class"]] = query_count
            if (real_preflight.get("status") != "PASS" or
                    set(counts) != expected_development):
                raise RuntimeError(
                    "Metadata preflight did not validate exactly four development classes")
            b7_query_feasibility = {
                "schema": "calibration.b7_query_feasibility.v1",
                "per_class_query_budgets": counts,
                "total_query_budget": sum(counts.values()),
                "zero_error_sample_size": clopper_pearson_zero_error_minimum(),
                "hypothetical_scenarios": b7_query_budget_feasibility(counts),
                "source": "Runner metadata-only development-class Q-A-TS preflight",
                "real_iad_labels_or_prevalence_inspected": False,
            }
        except Exception as exc:
            sun_continuous_diagnostics = {
                "schema": "calibration.sun_cs_continuous_diagnostics.v1",
                "status": "FAIL",
                "reason": f"Continuous diagnostic raised: {type(exc).__name__}: {exc}",
                "stage1_or_stage2_executed": False,
                "real_iad_data_or_outcomes_used": False,
            }
            b7_query_feasibility = {
                "schema": "calibration.b7_query_feasibility.v1",
                "status": "BLOCKED_METADATA_OR_QUERY_BUDGET_CHECK",
                "reason": f"Query feasibility raised: {type(exc).__name__}: {exc}",
                "real_iad_labels_or_prevalence_inspected": False,
            }
    else:
        sun_continuous_diagnostics = {
            "schema": "calibration.sun_cs_continuous_diagnostics.v1",
            "status": "SKIP_TESTS_NOT_PASSING",
            "stage1_or_stage2_executed": False,
            "real_iad_data_or_outcomes_used": False,
        }
        b7_query_feasibility = {
            "schema": "calibration.b7_query_feasibility.v1",
            "status": "SKIP_TESTS_NOT_PASSING",
            "real_iad_labels_or_prevalence_inspected": False,
        }
    native_smoke = run_native_parity_smoke(dataset_root, json_root)
    native_matrix = run_native_greedy_matrix()
    native_replay = run_native_event_replay(dataset_root, json_root)
    try:
        activation = run_controller_activation()
    except Exception as exc:
        activation = {"schema": "calibration.controller_activation_matrix.v1",
                      "status": "FAIL", "reason":
                      f"Activation harness raised: {type(exc).__name__}: {exc}",
                      "methods": {}, "stage1_or_stage2_run": False,
                      "confirmation_outcomes_opened": False}
    try:
        b7_engine = run_b7_engine_activation()
    except Exception as exc:
        b7_engine = {
            "schema": "calibration.b7_synthetic_full_engine_activation.v1",
            "pass_name": "B7_SYNTHETIC_FULL_ENGINE_ACTIVATION-v1.0.0",
            "status": "FAIL",
            "reason": f"Full-engine activation raised: {type(exc).__name__}: {exc}",
            "stage1_or_stage2_run": False,
            "confirmation_outcomes_opened": False,
            "scenarios": {},
        }
    sun_reference_test_ids = [test_id for test_id in tests.get("passed_test_ids", [])
                              if ("test_sun_cs_reference." in test_id or
                                  "test_sun_cs_diagnostics." in test_id)]
    sun_tests_ok = (
        tests["failed"] == 0 and tests["errors"] == 0 and tests["skipped"] == 0 and
        any("test_algorithm2_optimized_matches_independent_reference" in item
            for item in sun_reference_test_ids) and
        any("test_algorithm3_event_trace_matches_independent_reference" in item
            for item in sun_reference_test_ids) and
        any("test_empirical_extrema_false_reset_pathology_and_conservative_control" in item
            for item in sun_reference_test_ids) and
        any("test_conservative_upper_endpoint_clipping_precludes_anomaly" in item
            for item in sun_reference_test_ids) and
        any("test_seeded_continuous_streams_are_reproducible_and_distinct" in item
            for item in sun_reference_test_ids) and
        any("test_checkpoint_restore_reproduces_remaining_event_trace" in item
            for item in sun_reference_test_ids) and
        any("test_engine_event_boundary_checkpoint_resume_is_identical" in item
            for item in sun_reference_test_ids))
    if sun_tests_ok:
        try:
            sun_class_scale = run_class_scale_synthetic_diagnostic()
        except Exception as exc:
            sun_class_scale = {
                "schema": "calibration.sun_cs_class_scale_synthetic_diagnostic.v1",
                "status": "FAIL",
                "reason": f"Class-scale diagnostic raised: {type(exc).__name__}: {exc}",
                "stage1_or_stage2_run": False,
                "real_iad_data_or_outcomes_used": False,
            }
    else:
        sun_class_scale = {
            "schema": "calibration.sun_cs_class_scale_synthetic_diagnostic.v1",
            "status": "SKIP_TESTS_NOT_PASSING",
            "stage1_or_stage2_run": False,
            "real_iad_data_or_outcomes_used": False,
        }
    sun_reference_status = {
        "method": "B6 Sun-CS",
        "primary_candidate": "SUN_CS_A3_CONSERVATIVE_EXTENDED_REAL",
        "sensitivity_candidate": "SUN_CS_A3_EMPIRICAL_EXTREMA",
        "status": ("PAPER_EQUATIONS_WITH_DECLARED_ENDPOINT_CONVENTIONS"
                    if sun_tests_ok else "FAIL_REFERENCE_TESTS"),
        "source_fidelity_spec": "docs/calibration/SUN_CS_SOURCE_FIDELITY_SPEC.md",
        "registered_conventions": "docs/calibration/SUN_CS_A3_CONSERVATIVE_ENDPOINTS_v2.md",
        "method_version": PRIMARY_ENDPOINT_POLICY.method_version,
        "endpoint_policy_schema": "sun_cs_endpoint_policy.v2",
        "checkpoint_schema": "calibration.sun_cs_checkpoint.v2",
        "sensitivity_method_version": EMPIRICAL_EXTREMA_SENSITIVITY_POLICY.method_version,
        "primary_paper": "https://proceedings.mlr.press/v235/sun24h.html",
        "full_pdf": "https://raw.githubusercontent.com/mlresearch/v235/main/assets/sun24h/sun24h.pdf",
        "full_paper_and_appendices_reviewed": True,
        "verified_author_implementation": None,
        "author_implementation_search_status": "NO_VERIFIED_AUTHOR_IMPLEMENTATION_LOCATED",
        "implementation_scope": "Registered causal Algorithm 3 controller; Algorithm 2 exact optimized implementation checked against an independent small-sequence reference.",
        "reference_tests_passed": sun_reference_test_ids,
        "reference_test_count": len(sun_reference_test_ids),
        "author_code_parity": "NOT_CLAIMED_NO_VERIFIED_AUTHOR_IMPLEMENTATION",
        "algorithm2_reference_parity": "PASS" if sun_tests_ok else "FAIL_OR_SKIPPED",
        "algorithm3_event_trace_parity": "PASS" if sun_tests_ok else "FAIL_OR_SKIPPED",
        "checkpoint_resume": "PASS" if sun_tests_ok else "FAIL_OR_SKIPPED",
        "implementation_conventions": [
            "Primary clipped q=0 endpoint is negative infinity and clipped q=1 endpoint is positive infinity.",
            "Empirical-extrema q=0/sample-minimum and q=1/sample-maximum behavior is a separately named sensitivity version.",
            "Interior midpoint empirical quantiles use the declared one-based floor/ceiling ranks; a zero floor rank extends to the first order statistic.",
            "Empty historical prefix abstains before the current score is appended.",
            "At a detected change event: abstain, reset, then append that event as the first event in the new segment.",
            "Both endpoint variants preserve decision-before-update and Algorithm 3 reset behavior; neither claims author-code parity.",
        ],
        "synthetic_reference_sequence_coverage": {},
        "continuous_synthetic_diagnostics": sun_continuous_diagnostics,
        "query_budget_feasibility": b7_query_feasibility,
        "complexity": {
            "algorithm2_optimized": "O(n log n) time and O(n log n) peak auxiliary storage per invocation.",
            "algorithm3_recomputes_algorithm2_on_each_history": "O(T^2 log T) cumulative worst-case time and O(T log T) peak auxiliary storage for a segment of length T; no truncation or rolling window.",
            "class_scale_runtime_claim": "Not established by the Stage 0 short-stream tests; no long Real-IAD sequence was run.",
        },
        "dynamic_memory_guarantee_transfer": False,
        "physical_defect_guarantee": False,
        "stage1_or_stage2_run": False,
    }
    for sequence_name, sequence in (
            ("stationary_short", [0.5 + .08 * math.sin(i * .19)
                                  for i in range(80)]),
            ("piecewise_stationary_short",
             [0.1 + .02 * math.sin(i * .37) for i in range(40)] +
             [0.8 + .02 * math.sin(i * .37) for i in range(40)])):
        for policy in (PRIMARY_ENDPOINT_POLICY,
                       EMPIRICAL_EXTREMA_SENSITIVITY_POLICY):
            trace = algorithm3_reference(sequence, .99, .01,
                                          endpoint_policy=policy)
            reasons = [row["reason"] for row in trace]
            predicted = sum(row["prediction"] is not None for row in trace)
            sun_reference_status["synthetic_reference_sequence_coverage"][
                f"{sequence_name}:{policy.name}"] = {
                    "endpoint_policy": policy.name,
                    "method_version": policy.method_version,
                    "n": len(trace), "binary_predictions": predicted,
                    "abstentions_including_cold_start": len(trace) - predicted,
                    "coverage": predicted / len(trace),
                    "anomalies": sum(row["prediction"] == 1 for row in trace),
                    "benign": sum(row["prediction"] == 0 for row in trace),
                    "cold_start_abstentions": reasons.count("ABSTAIN_COLD_START"),
                    "change_detection_abstentions": reasons.count("ABSTAIN_CHANGE_DETECTED"),
                    "interval_abstentions": reasons.count("SUN_CS_ABSTAIN"),
                    "synthetic_scores_only_no_physical_labels": True,
                }
    correct_branch = branch == "calibration-stage0-contract-20261029"
    native_all_controller_ok = (
        native_replay.get("status") == "PASS" and
        native_replay.get("all_controller_matrix_status") ==
        "PASS_BOUNDED_NATIVE_DIAGNOSTIC")
    activation_rows = set(activation.get("methods", {}))
    b7_engine_rows = set(b7_engine.get("scenarios", {}).get(
        "two_sided_evidence_stress", {}).get("controller_names", []))
    all_registered_engine_rows = bool(activation_rows and
                                      activation_rows == b7_engine_rows)
    g0_status = ("PASS" if protected_sources.get("status") == "PASS" and
                 correct_branch else "BLOCKED")
    gates = {
        "G0": {"status": g0_status,
               "reason": ("Transitive protected-source baseline and byte-level historical shadow_memory check passed on the authorized branch."
                          if g0_status == "PASS" else
                          "Protected-source verification or authorized-branch check is blocked; see protected_source_verification.json."),
               "protected_source_status": protected_sources.get("status"),
               "authorized_branch": correct_branch},
        "G1": {"status": "PASS" if tests_ok else "FAIL", "reason": "Synthetic Q-A-TS/Q-B metadata tests passed." if tests_ok else "Synthetic query tests failed; see test_results.json."},
        "G1_real_data": {"status": real_preflight["status"],
                         "reason": ("Official sample identities, train-only support/C64 selection, and label-blind Q-A-TS/Q-B were checked using development-category JSON metadata; no image or mask was opened."
                                    if real_preflight["status"] == "PASS" else
                                    real_preflight.get("reason", "See per-class preflight status.")),
                         "classes": {item["class"]: item["status"]
                                     for item in real_preflight.get("classes", [])},
                         "dataset_root_present": dataset_root.is_dir(),
                         "json_root_present": json_root.is_dir()},
        "G2": {"status": native_smoke["status"],
               "reason": ("Actual development train-normal score parity passed across live native v22, untouched clone, and calibration adapter."
                          if native_smoke["status"] == "PASS" else
                          native_smoke.get("reason", "See native_parity_smoke.json."))},
        "G3": {"status": "PASS" if native_smoke["status"] == "PASS" and
               native_matrix["status"] == "PASS" else "BLOCKED",
               "reason": ("Synthetic native operator matrix passed APPEND/REPLACE/SKIP, repeated-slot source attribution, prototype-content audit, LSO filtering and exact checkpoint restore; bounded CUDA v22 smoke passed live-score and append parity."
                          if native_smoke["status"] == "PASS" and
                          native_matrix["status"] == "PASS" else
                          f"Native matrix={native_matrix.get('status')}; CUDA smoke={native_smoke.get('status')}: {native_smoke.get('reason', '')}")},
        "G4": {"status": "PASS" if tests_ok else "FAIL", "reason": "Synthetic predict-before-disclosure and queried-only feedback tests passed." if tests_ok else "Synthetic causal tests failed; see test_results.json."},
        "G5": {"status": "PASS" if tests_ok else "FAIL", "reason": "Synthetic protected C64, current-version evidence, and all-layer LSO source attribution tests passed." if tests_ok else "Synthetic evidence tests failed; see test_results.json."},
        "G6": {"status": "BLOCKED_FIDELITY", "reason": "Existing G6 acceptance remains in force pending researcher review of the proposed gate amendment: external author-code parity is not claimed for SPOT/dSPOT/Sun-CS, B2 is an explicitly adapted historical sidecar, and B7 remains provisional. B6 itself is registered as a paper-equation implementation with disclosed finite-sample conventions, not labeled BLOCKED solely for those conventions."},
        "G7": {"status": "BLOCKED", "reason": "Existing G7 source-fidelity acceptance remains in force: mathematical reference tests are reported separately, while verified author-code trace parity is unavailable for the external implementations and theoretical assumptions/guarantee transfer are not established for this score, feedback, and memory process."},
        "G8": {"status": "PASS" if any("test_checkpoint_resume_matches_uninterrupted_run_and_restores_rng" in item for item in tests.get("passed_test_ids", [])) else "FAIL", "reason": "Event-boundary checkpoint/resume is compared with an uninterrupted synthetic run, including memory, controller, trace, private-label identity, and Python/NumPy/Torch RNG restoration." if any("test_checkpoint_resume_matches_uninterrupted_run_and_restores_rng" in item for item in tests.get("passed_test_ids", [])) else "Checkpoint/resume equivalence test did not pass; see test_results.json."},
        "G9_native_core": {"status": native_replay["status"],
               "reason": ("Bounded multi-event actual v22/DINO train-normal trajectory passed causal query/write, current-version C64 refresh, LSO, native checkpoint-resume equivalence and synthetic-label AUROC invariance."
                          if native_replay["status"] == "PASS" else
                          native_replay.get("reason", "See native_event_replay.json."))},
        "G9_controller_activation": {"status": activation.get("status", "BLOCKED"),
               "reason": ("All registered original methods and separate fallback variants are represented in the causal synthetic controller activation matrix; see per-method rows." if activation.get("status") == "PASS" else activation.get("reason", "Activation harness failed or is partial."))},
        "G9_full_engine_integration": {"status": b7_engine.get("status", "BLOCKED"),
               "reason": "B7 two-scenario multi-event PrequentialEngine integration, deterministic resume, current-memory C64 refresh, queried-normal LSO, and score/query/memory counterfactual invariance; synthetic only."},
        "G9_all_controller_matrix": {"status": "PASS" if native_all_controller_ok else
                                     native_replay.get("status", "BLOCKED"),
               "reason": ("Every registered original and operational controller participated in the same bounded native-v22 score/memory/query replay; this is an engineering diagnostic, not an empirical performance result." if native_all_controller_ok else
                          "The bounded native all-controller replay did not pass; see native_event_replay.json.")},
        "G9": {"status": "PASS" if (native_all_controller_ok and
                                      activation.get("status") == "PASS" and
                                      b7_engine.get("status") == "PASS" and tests_ok) else "BLOCKED",
               "reason": ("Stage 0 native score/memory/AUROC invariance, all-controller synthetic activation, and B7 full-engine resume/counterfactual checks passed. This establishes engineering parity only, not Real-IAD scientific validation." if native_all_controller_ok and activation.get("status") == "PASS" and b7_engine.get("status") == "PASS" and tests_ok else
                          "One or more bounded native or synthetic controller-parity checks did not pass; see the scoped G9 rows and evidence.")},
    }
    # Candidate dimensions remain proposals until independently approved.
    # They are deliberately additive and do not overwrite the operative G0-G9
    # results above.
    spot_reference_ok = any("test_grimshaw_matches_independent_scipy_gpd_mle" in item
                            for item in tests.get("passed_test_ids", []))
    fallback_reference_ok = any(
        "test_operational_fallback_records_startup_and_abstention_without_changing_source" in item
        for item in tests.get("passed_test_ids", []))
    gates.update({
        "G6_CORE_METHODS": {
            "status": "PASS" if tests_ok else "FAIL",
            "reason": "B0-B4 and provisional B7 registered paths are covered by the passing calibration suite; B7 remains explicitly provisional."},
        "G6_EXTERNAL_PAPER_EQUATION_IMPLEMENTATIONS": {
            "status": "PASS" if sun_tests_ok and spot_reference_ok else "FAIL_OR_SKIPPED",
            "reason": "B5a/B5b and B6 paper-equation implementations have independent equation/stream references; no author-code parity is claimed."},
        "G6_OPERATIONAL_VARIANTS": {
            "status": "PASS" if fallback_reference_ok and activation.get("status") == "PASS" else "FAIL_OR_SKIPPED",
            "reason": "SPOT, dSPOT, and Sun-CS Frozen-Theta wrappers are separate registered rows with startup/abstention mode accounting."},
        "G6_UNRESOLVED_METHODS": {
            "status": "BLOCKED_RESEARCH_REVIEW",
            "method_rows": ["B2 full native-loop parity is not claimed", "B7 remains provisional pending researcher review"],
            "reason": "Unresolved limitations remain visible; no method row is removed."},
        "G7_MATHEMATICAL_REFERENCE_TESTS": {
            "status": "PASS" if tests_ok else "FAIL",
            "reason": "Calibration reference suite status; includes independent Sun-CS A2/A3 reference comparisons and SPOT/dSPOT equations."},
        "G7_SOURCE_IMPLEMENTATION_PARITY": {
            "status": "BLOCKED_UNAVAILABLE",
            "reason": "No verified author implementation trace was available for external SPOT/dSPOT/Sun-CS parity; independent equation tests are not a substitute."},
        "G7_ASSUMPTION_COMPATIBILITY": {
            "status": "NOT_ESTABLISHED",
            "reason": "Score dependence, physical-defect target, selective feedback, and dynamic memory differ from external source assumptions; disclosures are in the frozen candidate."},
        "G7_THEORETICAL_GUARANTEE_TRANSFER": {
            "status": "NOT_ESTABLISHED",
            "reason": "No transfer proof for physical-defect risk under sparse feedback and adaptively changing memory is claimed."},
        "G9_NATIVE_CORE": {
            "status": native_replay.get("status", "SKIP"),
            "reason": "Bounded native score/update/replay evidence; scope and data availability are in native_event_replay.json."},
        "G9_CONTROLLER_SYNTHETIC_ACTIVATION": {
            "status": activation.get("status", "BLOCKED"),
            "reason": "Causal shared-engine controller activation is synthetic mechanism evidence only."},
        "G9_FULL_ENGINE_INTEGRATION": {
            "status": b7_engine.get("status", "BLOCKED"),
            "reason": "Named B7 two-scenario full PrequentialEngine activation with checkpoint resume and shared-path invariance; synthetic only."},
        "G9_ALL_REGISTERED_METHODS": {
            "status": "PASS" if (activation.get("status") == "PASS" and
                                  b7_engine.get("status") == "PASS" and
                                  all_registered_engine_rows) else "FAIL_OR_SKIPPED",
            "all_registered_names_match": all_registered_engine_rows,
            "activation_method_names": sorted(activation_rows),
            "full_engine_method_names": sorted(b7_engine_rows),
            "reason": "Every original controller and separate fallback row participates in the full PrequentialEngine test; B7 is covered by both two-scenario full-engine integration and the complete method activation matrix."},
        "G9_SCIENTIFIC_VALIDATION_NOT_STAGE0": {
            "status": "OUT_OF_SCOPE_NOT_RUN",
            "reason": "Stage 0 does not establish Real-IAD scientific performance or Stage 1/2 authorization."},
    })
    scoped = {
        "G0_PROTECTED_SOURCE_INTEGRITY": "G0",
        "G1_QUERY_PROTOCOL": "G1",
        "G2_NATIVE_SCORING": "G2",
        "G3_NATIVE_GREEDY_OPERATOR": "G3",
        "G4_CAUSAL_FIREWALL": "G4",
        "G5_ANCHOR_AND_LSO": "G5",
        "G6_ALL_METHOD_FAMILIES": "G6",
        "G7_EXTERNAL_SOURCE_FIDELITY": "G7",
        "G8_REPRODUCIBILITY": "G8",
        "G9_NATIVE_CORE": "G9_native_core",
        "G9_CONTROLLER_ACTIVATION": "G9_controller_activation",
        "G9_FULL_ENGINE_INTEGRATION": "G9_full_engine_integration",
        "G9_FULL_MATRIX": "G9_all_controller_matrix",
    }
    for scoped_name, source_name in scoped.items():
        gates[scoped_name] = {"status": gates[source_name]["status"],
                              "reason": gates[source_name]["reason"]}
    # Keep one canonical serialized spelling for G9 subdimensions. The
    # lowercase keys above are internal aggregation inputs; emitting both
    # mixed-case aliases makes some JSON consumers treat them as duplicates.
    for internal_name in ("G9_native_core", "G9_controller_activation",
                          "G9_full_engine_integration",
                          "G9_all_controller_matrix"):
        gates.pop(internal_name, None)
    manifest = {
        "stage": 0, "overall_status": "BLOCKED_WIP" if tests_ok else "FAIL",
        "branch": branch, "source_commit": source_commit,
        "protocol_version": PROTOCOL_VERSION,
        "profile": asdict(DEFAULT_PROFILE), "profile_sha256": DEFAULT_PROFILE.sha256,
        "method_cards_sha256": _sha(canonical_json(METHOD_CARDS).encode()),
        "execution": "synthetic_cpu_dev_metadata_plus_optional_audiojack_train_only_native_smoke_greedy_matrix_and_multievent_replay",
        "stage1_run": False, "stage2_run": False,
        "confirmation_outcomes_opened": False,
        "support_regime": "four deployed supports plus 64 additional protected C64 normals",
        "method_cards": METHOD_CARDS,
        "query_tape_versions": ["Q-A-TS-v1", "Q-B via Q-A-TS-v1 allocation"],
    }
    _json_write(out_dir / "protocol_manifest.json", manifest)
    _json_write(out_dir / "dependency_environment.json", env)
    _json_write(out_dir / "gates.json", gates)
    _json_write(out_dir / "real_iad_metadata_preflight.json", real_preflight)
    _json_write(out_dir / "native_parity_smoke.json", native_smoke)
    _json_write(out_dir / "native_greedy_parity_matrix.json", native_matrix)
    _json_write(out_dir / "native_event_replay.json", native_replay)
    _json_write(out_dir / "protected_source_verification.json", protected_sources)
    _json_write(out_dir / "controller_activation_matrix.json", activation)
    _json_write(out_dir / "sun_cs_reference_status.json", sun_reference_status)
    _json_write(out_dir / "b7_engine_activation.json", b7_engine)
    _json_write(out_dir / "sun_cs_class_scale_synthetic_diagnostic.json",
                sun_class_scale)
    _json_write(out_dir / "sun_cs_continuous_diagnostics.json",
                sun_continuous_diagnostics)
    _json_write(out_dir / "sun_cs_endpoint_policy_comparison.json", {
        "schema": "calibration.sun_cs_endpoint_policy_comparison.v1",
        "primary": {
            "method_version": sun_continuous_diagnostics.get("method_version"),
            "policy": "conservative_extended_real",
            "status": "PAPER_EQUATIONS_WITH_DECLARED_CONVENTIONS",
        },
        "sensitivity": {
            "method_version": EMPIRICAL_EXTREMA_SENSITIVITY_POLICY.method_version,
            "policy": "empirical_extrema",
            "status": "SENSITIVITY_ONLY_NOT_PRIMARY",
        },
        "comparison_basis": "same seeded synthetic streams, causal score order, p=.99, delta=.01; offline source-target scoring",
        "endpoint_identifiability": sun_continuous_diagnostics.get(
            "finite_stream_identifiability", []),
        "upper_endpoint_identifiability_horizon": sun_continuous_diagnostics.get(
            "upper_endpoint_identifiability_horizon"),
        "runs": sun_continuous_diagnostics.get("runs", []),
        "real_iad_data_or_outcomes_used": False,
        "stage1_or_stage2_executed": False,
    })
    _json_write(out_dir / "b7_query_budget_feasibility.json",
                b7_query_feasibility)
    for source_name, artifact_name in (
            ("STAGE1_EVALUATION_PROTOCOL_FROZEN_CANDIDATE.md",
             "stage1_evaluation_protocol_candidate.md"),
            ("STAGE0_GATE_AMENDMENT_CANDIDATE.md",
             "stage0_gate_amendment_candidate.md"),
            ("SUN_CS_A3_DECLARED_CONVENTIONS_v1.md",
             "sun_cs_a3_declared_conventions.md"),
            ("SUN_CS_A3_CONSERVATIVE_ENDPOINTS_v2.md",
             "sun_cs_a3_conservative_endpoints_v2.md"),
            ("B7_ACTIVATION_FEASIBILITY_REPORTING_v1.md",
             "b7_activation_feasibility_reporting_v1.md"),
            ("STAGE0_FINAL_SCIENTIFIC_PROTOCOL_REVIEW.md",
             "stage0_final_scientific_protocol_review.md")):
        source_path = ROOT / "docs" / "calibration" / source_name
        (out_dir / artifact_name).write_text(source_path.read_text(encoding="utf-8"),
                                             encoding="utf-8")
    _json_write(out_dir / "controller_method_cards.json", METHOD_CARDS)
    with (out_dir / "synthetic_event_trace.jsonl").open("w", encoding="utf-8") as f:
        for row in synthetic["trace"]:
            f.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
    compact = {k: v for k, v in synthetic.items()
               if k not in {"trace", "scores", "predictions", "controller_states"}}
    compact["controller_names"] = list(synthetic["controller_states"])
    _json_write(out_dir / "synthetic_summary.json", compact)
    _json_write(out_dir / "test_results.json", tests)
    _json_write(out_dir / "source_hashes.json", {
        "stage0_source_hashes": source_hashes,
        "source_commit": source_commit,
        "source_fingerprint_sha256": _sha(canonical_json(source_hashes).encode()),
        "historical_sources_unchanged_by_this_work": protected_sources.get("status") == "PASS",
        "historical_source_restoration": "experiments/risk_safe/shadow_memory.py restored byte-for-byte to protected baseline",
        "protected_sources_match_reviewed_baseline": protected_sources.get("status") == "PASS",
        "protected_source_baseline_status": protected_sources.get("status")})
    summary = [
        "# Stage 0 synthetic runner summary", "",
        f"Status: **{'BLOCKED_WIP' if tests_ok else 'FAIL'}; not scientifically validated**.", "",
        f"Branch/commit: `{branch}` / `{source_commit}`.",
        "Synthetic mechanisms ran in the configured torch environment on CPU.",
        f"Real-IAD metadata preflight: {real_preflight['status']}; four development-category JSON indexes were checked without decoding images or masks.",
        f"Native parity smoke: {native_smoke['status']}; see `native_parity_smoke.json` for the bounded train-only scope.",
        f"Native Greedy operator matrix: {native_matrix['status']}; see `native_greedy_parity_matrix.json`.",
        f"Native event replay: {native_replay['status']}; controller activation: {activation.get('status', 'BLOCKED')}; B7 full-engine synthetic activation: {b7_engine.get('status', 'BLOCKED')}; native-image all-controller matrix: {gates['G9_FULL_MATRIX']['status']}.",
        "No test image, mask, or confirmation-category metadata/outcomes were opened.",
        "Stage 1 and Stage 2 were not run.", "",
        "Gate statuses: " + ", ".join(f"{k}={v['status']}" for k, v in gates.items()),
        "", "The independent review should use `controller_method_cards.json`, `gates.json`,",
        "`source_hashes.json`, `test_results.json`, `native_greedy_parity_matrix.json`,",
        "`native_event_replay.json`, `controller_activation_matrix.json`,",
        "`b7_engine_activation.json`, and the frozen protocol candidates,",
        "`protected_source_verification.json`, and the synthetic trace.",
    ]
    (out_dir / "stage0_summary.md").write_text("\n".join(summary) + "\n", encoding="utf-8")
    files = [p for p in out_dir.iterdir() if p.is_file() and
             p.name != "checksums.json"]
    _json_write(out_dir / "checksums.json", {p.name: _hash_file(p) for p in sorted(files)})
    acceptance = all(gates[f"G{i}"]["status"] == "PASS" for i in range(10))
    return {"gates": gates, "tests": tests, "source_commit": source_commit,
            "branch": branch, "synthetic": compact,
            "status": "PASS" if acceptance else "BLOCKED_WIP" if tests_ok else "FAIL"}


def _verify_artifact(out_dir: Path) -> dict[str, Any]:
    if not out_dir.is_dir() or not _verify_checksums(out_dir):
        raise RuntimeError("Artifact missing or checksum verification failed")
    manifest = json.loads((out_dir / "protocol_manifest.json").read_text(encoding="utf-8"))
    hashes = json.loads((out_dir / "source_hashes.json").read_text(encoding="utf-8"))
    identity_path = out_dir / "run_identity.json"
    if not identity_path.is_file():
        raise RuntimeError("Artifact lacks a resumable run identity record")
    identity = json.loads(identity_path.read_text(encoding="utf-8"))
    current_hashes = _source_hashes()
    current_fp = _sha(canonical_json(current_hashes).encode())
    protected = verify_protected_sources()
    if (manifest.get("profile_sha256") != DEFAULT_PROFILE.sha256 or
            manifest.get("method_cards_sha256") !=
            _sha(canonical_json(METHOD_CARDS).encode()) or
            manifest.get("protocol_version") != PROTOCOL_VERSION or
            hashes.get("stage0_source_hashes") != current_hashes or
            identity.get("source_fingerprint_sha256") != current_fp or
            identity.get("profile_sha256") != DEFAULT_PROFILE.sha256 or
            protected.get("status") != "PASS"):
        raise RuntimeError("Artifact source/configuration provenance does not match this checkout")
    return {"checksums_verified": True, "source_and_config_match": True,
            "protected_source_status": protected.get("status"),
            "artifact_status": manifest.get("overall_status", "UNKNOWN"),
            "run_key": identity.get("run_key")}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", type=int, required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--all", action="store_true")
    mode.add_argument("--verify-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args(argv)
    if args.stage != 0:
        parser.error("Only Stage 0 is implemented; Stage 1/2 are prohibited")
    _check_environment()
    if args.verify_only:
        if args.output_dir is None:
            parser.error("--verify-only requires --output-dir for an exact artifact")
        output = args.output_dir.resolve()
        result = _verify_artifact(output)
        print(json.dumps({"mode": "verify-only",
                          "verification_status": "ARTIFACT_INTEGRITY_VERIFIED",
                          "overall_status": result["artifact_status"],
                          "output_dir": str(output),
                          "checksums_verified": result["checksums_verified"],
                          "source_and_config_match": result["source_and_config_match"],
                          "protected_source_status": result["protected_source_status"]}, indent=2))
        return 0
    git, _ = _git_info()
    source_commit = git.split("@", 1)[1]
    source_fingerprint = _sha(canonical_json(_source_hashes()).encode())
    default_key = _sha((DEFAULT_PROFILE.sha256 + source_fingerprint).encode())[:16]
    out_dir = (args.output_dir or ROOT / "outputs" / "calibration" / "stage0" /
               f"stage0_{default_key}").resolve()
    if not out_dir.is_relative_to(ROOT):
        raise RuntimeError("Output directory must remain inside the repository")
    if out_dir.exists():
        if not args.resume:
            raise FileExistsError(f"Refusing to overwrite existing Stage 0 output: {out_dir}")
        existing_names = {p.name for p in out_dir.iterdir()}
        if existing_names - ARTIFACT_FILES:
            raise RuntimeError("Resume refused: unexpected files/directories in output")
        identity_file = out_dir / "run_identity.json"
        if not identity_file.is_file():
            raise RuntimeError("Resume refused: output has no Stage 0 run identity")
        identity = json.loads(identity_file.read_text(encoding="utf-8"))
        expected_identity = {"schema": "calibration.stage0_run_identity.v1",
            "run_key": default_key, "profile_sha256": DEFAULT_PROFILE.sha256,
            "source_fingerprint_sha256": source_fingerprint}
        if identity != expected_identity:
            raise RuntimeError("Resume refused: source/configuration provenance changed")
        if _verify_checksums(out_dir):
            result = _verify_artifact(out_dir)
            print(json.dumps({"mode": "resume-verified",
                              "verification_status": "ARTIFACT_INTEGRITY_VERIFIED",
                              "overall_status": result["artifact_status"],
                              "output_dir": str(out_dir), "checksums_verified": True,
                              "source_and_config_match": True}, indent=2))
            return 0
        if not (out_dir / "INCOMPLETE").exists():
            raise RuntimeError("Resume refused: incomplete marker is missing")
        # Rebuild deterministic smoke files in place only after the immutable
        # source/config identity matches; research streams are never resumed.
    else:
        out_dir.mkdir(parents=True, exist_ok=False)
        _json_write(out_dir / "run_identity.json", {
            "schema": "calibration.stage0_run_identity.v1",
            "run_key": default_key, "profile_sha256": DEFAULT_PROFILE.sha256,
            "source_fingerprint_sha256": source_fingerprint})
    result = _make_artifacts(out_dir)
    print(json.dumps({"mode": "all", "status": result["status"],
                      "output_dir": str(out_dir), "branch": result["branch"],
                      "source_commit": result["source_commit"],
                      "gates": {k: v["status"] for k, v in result["gates"].items()},
                      "tests_run": result["tests"]["tests_run"],
                      "passed": result["tests"]["passed"]}, indent=2))
    return 0 if result["status"] == "PASS" else 2 if result["status"] == "BLOCKED_WIP" else 1


if __name__ == "__main__":
    raise SystemExit(main())
