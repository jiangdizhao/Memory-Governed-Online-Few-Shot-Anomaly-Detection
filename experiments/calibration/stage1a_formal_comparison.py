"""Execute the frozen Stage 1-A controller-only comparison.

Prediction and outcome evaluation are deliberately separate phases.  The
prediction phase imports the frozen replay adapter and reads only its
precomputed scores, memory transitions, and already-queried P1.2 evidence.
The evaluator opens development test labels only after all 20 prediction
streams and their query-response chains have been sealed.

Run with the repository's existing ``torch`` Conda interpreter:

    python -m experiments.calibration.stage1a_formal_comparison run

Outputs are written to a new namespace and never overwrite an existing run.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict, replace
import gzip
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time
from typing import Any, Iterable, Mapping, Sequence

from . import stage1a_final_readiness as frozen
from .controllers import theta0_from_c64
from .metrics import PrivateOutcomeLedger, summarize_prediction_modes


ROOT = frozen.ROOT
OUTPUT = ROOT / "outputs/calibration/stage1a_formal_controller_comparison_v4_20261030"
REQUIRED_HEAD = "89e2cf4992408cbc5e86facb621749aa7210642b"
METHODS = (
    "B0", "B1", "B2", "B3", "B4", "B5a", "B5a_FROZEN_FALLBACK",
    "B5b", "B5b_FROZEN_FALLBACK", "B7", "Stage075BOffset",
)
DISPLAY_NAMES = {
    "B5a_FROZEN_FALLBACK": "B5a-Frozen",
    "B5b_FROZEN_FALLBACK": "B5b-Frozen",
}
METRIC_NAMES = (
    "balanced_accuracy", "fpr", "fnr", "defect_recall", "coverage",
    "normal_coverage", "defect_coverage", "selective_balanced_accuracy",
    "worst_defined_full_window_balanced_accuracy",
)
SOURCE_FILES = (
    "experiments/calibration/stage1a_final_readiness.py",
    "experiments/calibration/controllers.py",
    "experiments/calibration/spot_dspot.py",
    "experiments/calibration/scdac_g.py",
    "experiments/calibration/stage075b_controllers.py",
    "experiments/calibration/metrics.py",
    "experiments/calibration/protocol.py",
    "experiments/calibration/profiles.py",
    "experiments/calibration/stage1a_p11_production.py",
    "experiments/calibration/stage1a_p12_sharded_recovery.py",
    "test_phase3_AIF_simplified_v22.py",
)


class ComparisonError(RuntimeError):
    """Frozen-input, controller, metric, or output-integrity failure."""


def _sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _bind_query_source_identity(source_key: str, expected_group_sha256: str) -> str:
    """Rebuild P1's response identity from label-blind frozen source metadata."""
    if not isinstance(source_key, str) or not source_key:
        raise ComparisonError("Frozen queried source key is missing")
    group_sha = _sha_bytes(
        ("official-sample-group\0" + source_key).encode("utf-8"))
    if group_sha != expected_group_sha256:
        raise ComparisonError("Frozen source key does not match its query-tape group digest")
    return _sha_bytes(source_key.encode("utf-8"))


def _canonical(value: Any) -> bytes:
    return json.dumps(_json_safe(value), sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False).encode("ascii")


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Mapping):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if hasattr(value, "item"):
        return _json_safe(value.item())
    raise TypeError(f"Not JSON serializable in Stage 1-A artifact: {type(value)!r}")


def _write_json_once(path: Path, value: Any, *, pretty: bool = True) -> str:
    if path.exists():
        raise ComparisonError(f"Refusing to overwrite immutable result: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(_json_safe(value), sort_keys=True, indent=2,
                          ensure_ascii=True, allow_nan=False) + "\n").encode("utf-8")
    with path.open("xb") as stream:
        stream.write(payload)
    return _sha_bytes(payload)


def _write_progress(path: Path, value: Mapping[str, Any]) -> None:
    """Update only the run-control file; result artifacts themselves are write-once."""
    temporary = path.with_suffix(path.suffix + ".partial")
    payload = json.dumps(_json_safe(value), sort_keys=True, indent=2,
                         ensure_ascii=True, allow_nan=False).encode("utf-8") + b"\n"
    with temporary.open("wb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _write_gzip_jsonl_once(path: Path, rows: Iterable[Mapping[str, Any]]) -> str:
    if path.exists():
        raise ComparisonError(f"Refusing to overwrite immutable prediction stream: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".partial")
    if partial.exists():
        raise ComparisonError(f"Partial artifact already exists; inspect manually: {partial}")
    with partial.open("xb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, compresslevel=6,
                           mtime=0) as compressed:
            for row in rows:
                compressed.write(_canonical(row) + b"\n")
    if path.exists():
        raise ComparisonError(f"Immutable prediction stream appeared during write: {path}")
    partial.replace(path)
    return _sha_file(path)


def _git(*args: str) -> str:
    result = subprocess.run(["git", *args], cwd=ROOT, check=True,
                            capture_output=True, text=True)
    return result.stdout.strip()


def _frozen_input_preflight() -> dict[str, Any]:
    branch = _git("branch", "--show-current")
    head = _git("rev-parse", "HEAD")
    if branch != frozen.BRANCH:
        raise ComparisonError(f"Expected branch {frozen.BRANCH}; found {branch}")
    if _git("merge-base", "--is-ancestor", REQUIRED_HEAD, head) == "":
        # git merge-base --is-ancestor communicates by exit status; this branch
        # is reached only when it succeeds.
        pass
    intervening = _git("diff", "--name-only", f"{REQUIRED_HEAD}..{head}").splitlines()
    if set(intervening) - {"tests/calibration/test_stage1a_final_readiness.py"}:
        raise ComparisonError("Post-required-HEAD changes include non-test implementation files")

    audit = frozen.verify_audit_package(frozen.AUDIT_DIR)
    readiness_manifest = json.loads(
        (frozen.AUDIT_DIR / "stage1a_execution_manifest.json").read_text(encoding="utf-8"))
    p12_complete = json.loads((frozen.P12_OUTPUT / "complete.json").read_text(encoding="utf-8"))
    if p12_complete.get("decision") != "P12_EXACT_PARITY_AND_EVIDENCE_PASS":
        raise ComparisonError("P1.2 exact parity receipt is not passing")

    current_sources = {name: _sha_file(ROOT / name) for name in SOURCE_FILES}
    frozen_input_path = frozen.STAGE05 / "frozen_inputs.json"
    expected_frozen_input_sha = readiness_manifest["stage05_source_artifacts"].get(
        "frozen_inputs.json")
    if (not expected_frozen_input_sha or
            _sha_file(frozen_input_path) != expected_frozen_input_sha):
        raise ComparisonError("Frozen label-blind Stage 0.5 metadata hash differs from readiness")
    method_identities = readiness_manifest["controller_config_identities"]
    for method in METHODS:
        for name, expected in method_identities[method]["source_sha256"].items():
            if current_sources.get(name) != expected:
                raise ComparisonError(f"Frozen source fingerprint mismatch for {method}: {name}")
    return {
        "branch": branch,
        "actual_head": head,
        "required_head": REQUIRED_HEAD,
        "required_head_is_ancestor": True,
        "intervening_commits_touch_only_test_fixture": intervening,
        "tracked_worktree_diff": _git("diff", "--name-only"),
        "frozen_readiness_audit": audit,
        "readiness_execution_manifest_sha256": _sha_file(
            frozen.AUDIT_DIR / "stage1a_execution_manifest.json"),
        "readiness_controller_config_identities": {
            method: method_identities[method] for method in METHODS},
        "current_source_sha256": current_sources,
        "frozen_label_blind_metadata_sha256": expected_frozen_input_sha,
        "stage075a_source_provenance_caveat": readiness_manifest.get(
            "p12_source_provenance_caveat"),
        "historical_g2_status": readiness_manifest.get("historical_g2_status"),
        "p12_completion_sha256": _sha_file(frozen.P12_OUTPUT / "complete.json"),
        "p12_performance_evaluation_previously_executed": p12_complete.get(
            "stage1a_performance_evaluation"),
        "query_tape_binding_sha256": frozen.TAPE_BINDING_SHA256,
        "trajectory_count": len(readiness_manifest["trajectories"]),
        "event_count": p12_complete.get("event_count"),
        "query_rate": 0.03,
        "query_seed": 37,
        "classes": list(frozen.CLASSES),
        "support_seeds": list(frozen.SEEDS),
    }


def _controller_snapshot(method: str, controller: Any, prediction: Any,
                         post_threshold: Any, *, include_diagnostics: bool) -> dict[str, Any]:
    """Whitelisted diagnostics; deliberately excludes score buffers and labels."""
    row: dict[str, Any] = {
        "prediction": prediction.label,
        "available": prediction.available,
        "reason": prediction.reason,
        "threshold_pre": prediction.threshold,
        "threshold_post": post_threshold,
    }
    if not include_diagnostics:
        return _json_safe(row)
    if method == "B1":
        count = len(controller.history)
        row["diagnostics"] = {
            "eligible_queried_normals": count,
            "minimum_required": controller.min_count,
            "status": "HOLD_INSUFFICIENT_QUERIED_NORMALS" if count < controller.min_count
                      else "ROLLING_Q90_ACTIVE",
        }
    elif method == "B3":
        row["diagnostics"] = {
            "status": controller.status,
            "fnr_evidence_status": controller.fnr_status,
            "protected_c64_count": len(controller.normal_scores),
            "evidence_memory_version": controller.evidence_version,
        }
    elif method == "B4":
        row["diagnostics"] = {
            "evidence_status": controller.last_status,
            "conflict": controller.conflict,
            "normal_anchor_count": len(controller.normal_scores),
            "defect_anchor_count": len(controller.defect_scores),
            "last_prediction_error": controller.last_prediction_error,
        }
    elif method == "B5a":
        row["diagnostics"] = {
            "startup_state": "READY" if controller.core.ready else "WARMUP",
            "observations": controller.observations,
            "initial_observations": len(controller.core.initial),
            "strict_peak_count": len(controller.core.peaks),
            "alarms": controller.core.alarms,
        }
    elif method == "B5b":
        row["diagnostics"] = {
            "startup_state": "READY" if controller.core.ready else "WARMUP",
            "observations": controller.observations,
            "local_window_count": len(controller.local_window),
            "residual_initialization_count": len(controller.initial_residuals),
            "strict_peak_count": len(controller.core.peaks),
            "alarms": controller.alarms,
        }
    elif method == "B5a_FROZEN_FALLBACK":
        row["diagnostics"] = {
            "source_reason": prediction.details.get("source_reason"),
            "fallback_policy": "theta0_on_unavailable_or_abstention",
            "source_threshold": prediction.details.get("source_threshold"),
        }
    elif method == "B5b_FROZEN_FALLBACK":
        row["diagnostics"] = {
            "source_reason": prediction.details.get("source_reason"),
            "fallback_policy": "theta0_on_unavailable_or_abstention",
            "source_threshold": prediction.details.get("source_threshold"),
        }
    elif method == "B7":
        diag = controller.diagnostics()
        row["diagnostics"] = {
            key: diag.get(key) for key in (
                "status", "decision", "normal_count", "defect_count",
                "raw_normal_count", "raw_defect_count", "fpr_upper", "fnr_upper",
                "constraint_diagnostic_reason", "support_feasible_candidate_count",
                "empirical_feasible_candidate_count",
                "executed_threshold_in_descriptive_feasible_region",
                "candidate_count", "candidate_evaluations", "bound_evaluations",
                "total_candidate_evaluations", "total_bound_evaluations",
                "confidence_delta", "last_two_sided_supported_interval",
                "interval_stale", "guarantee") if key in diag
        }
    elif method == "Stage075BOffset":
        row["diagnostics"] = {
            "name": controller.name,
            "updates": controller.updates,
            "normal_evidence_count": len(controller.normal_scores),
            "defect_evidence_count": len(controller.defect_scores),
            "last_update": controller.diagnostics_last,
            "missing_side_behavior": "hold_own_threshold",
            "is_additive_b3_offset": False,
        }
    return _json_safe(row)


def _prediction_event_row(event: Any, predictions: Mapping[str, Any],
                          replay: Any) -> dict[str, Any]:
    trace = replay.trace[-1]
    include_diagnostics = event.t == 0 or event.queried
    methods = {}
    for method in METHODS:
        controller = replay.controllers[method]
        methods[method] = _controller_snapshot(
            method, controller, predictions[method],
            trace["threshold_post"].get(method),
            include_diagnostics=include_diagnostics)
    writer = event.writer_record or {}
    result = {
        "t": event.t,
        "query_event": event.queried,
        "dynamic_score": event.observation.score,
        "static_score": event.static_score,
        "memory_version_before": event.memory_before.version,
        "memory_version_after": event.memory_after.version,
        "memory_sha256_before": event.memory_before.memory_sha256,
        "memory_sha256_after": event.memory_after.memory_sha256,
        "source_map_sha256_after": event.memory_after.source_map_sha256,
        "source_revision_after": event.memory_after.source_revision,
        "legacy_greedy_version_after": event.memory_after.legacy_greedy_version,
        "evidence_revision": trace["evidence_revision"],
        "writer_action": {
            key: writer.get(key) for key in (
                "eligible", "changed", "reason", "keep", "appended", "replaced",
                "skipped", "source_map_changed") if key in writer
        },
        "methods": methods,
    }
    if include_diagnostics:
        result["controller_diagnostics_at_evidence_boundary"] = {
            method: value.get("diagnostics") for method, value in methods.items()
            if "diagnostics" in value}
    return result


def _summarize_threshold_dynamics(pre_values: Sequence[float | None],
                                  post_values: Sequence[float | None]) -> dict[str, Any]:
    if len(pre_values) != len(post_values):
        raise ComparisonError("Threshold pre/post trajectories differ in length")
    finite_pre = [float(value) for value in pre_values if value is not None]
    finite_post = [float(value) for value in post_values if value is not None]
    changes = [abs(float(before) - float(after))
               for before, after in zip(pre_values, post_values)
               if before is not None and after is not None]
    return {
        "events": len(pre_values),
        "pre_available_events": len(finite_pre),
        "post_available_events": len(finite_post),
        "first_pre": finite_pre[0] if finite_pre else None,
        "last_post": finite_post[-1] if finite_post else None,
        "minimum_post": min(finite_post) if finite_post else None,
        "maximum_post": max(finite_post) if finite_post else None,
        "threshold_change_events": sum(delta != 0 for delta in changes),
        "absolute_threshold_movement": math.fsum(changes),
        "maximum_single_event_movement": max(changes, default=0.0),
    }


def _run_prediction_phase(output: Path, preflight: Mapping[str, Any]) -> dict[str, Any]:
    if output.exists():
        raise ComparisonError(f"New immutable output namespace already exists: {output}")
    output.mkdir(parents=True, exist_ok=False)
    (output / "predictions").mkdir()
    progress_path = output / "RUN_STATE.json"
    progress: dict[str, Any] = {
        "schema": "stage1a.formal_comparison.run_state.v1",
        "status": "PREDICTION_PHASE_RUNNING",
        "completed_trajectories": [],
        "total_trajectories": 20,
        "completed_events": 0,
        "expected_events": 76475,
        "outcome_labels_accessed": False,
    }
    _write_progress(progress_path, progress)
    _write_json_once(output / "preflight.json", preflight)
    readiness_manifest = json.loads(
        (frozen.AUDIT_DIR / "stage1a_execution_manifest.json").read_text(encoding="utf-8"))
    receipts_manifest = {(row["category"], int(row["support_seed"])): row
                        for row in readiness_manifest["trajectories"]}
    frozen_inputs = json.loads((frozen.STAGE05 / "frozen_inputs.json").read_text(
        encoding="utf-8"))
    stream_metadata = frozen_inputs["classes"]
    from .stage1a_p11_production import verify_stage05_qats_artifacts
    tapes = verify_stage05_qats_artifacts(frozen.STAGE05)
    result_rows: list[dict[str, Any]] = []
    trajectory_total = 0
    start = time.perf_counter()

    for category in frozen.CLASSES:
        for seed in frozen.SEEDS:
            key = (category, seed)
            identity = receipts_manifest[key]
            receipt_path = frozen.P12_OUTPUT / "shards" / category / f"seed{seed}.receipt.json"
            receipt_sha = _sha_file(receipt_path)
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            trajectory = frozen.load_frozen_trajectory_input(
                category=category, support_seed=seed, receipt=receipt, root=ROOT)
            c64_scores = [anchor.score for anchor in trajectory.initial_evidence.normal_anchors
                          if anchor.protected]
            theta0 = theta0_from_c64(c64_scores)
            suite = frozen.build_controller_suite(theta0, c64_scores)
            if tuple(suite) != METHODS:
                raise ComparisonError(f"Frozen method registry order differs: {tuple(suite)}")
            input_binding = frozen.sha256_json({
                "category": category, "support_seed": seed,
                "receipt_sha256": receipt_sha,
                "query_tape_binding_sha256": frozen.TAPE_BINDING_SHA256,
                "controller_configuration_identities": {
                    name: preflight["readiness_controller_config_identities"][name]
                    ["configuration_sha256"] for name in METHODS},
            })
            replay = frozen.ControllerOnlyReplay(
                controllers=suite,
                query_positions=trajectory.query_positions,
                input_binding_sha256=input_binding,
                initial_memory=trajectory.initial_memory,
                initial_evidence=trajectory.initial_evidence,
            )
            prediction_path = output / "predictions" / f"{category}_seed{seed}.jsonl.gz"
            event_counter = 0
            b7_timeline: list[dict[str, Any]] = []
            hold_timeline: list[dict[str, Any]] = []

            def rows():
                nonlocal event_counter
                for event in trajectory.events():
                    if event.queried:
                        metadata = stream_metadata[category]["stream"][event.t]
                        source_identity = _bind_query_source_identity(
                            str(metadata["sample_key"]),
                            str(tapes.group_identity_by_class_position[category][event.t]))
                        event = replace(event,
                                        query_source_identity_sha256=source_identity)
                    predicted = replay.step(event)
                    event_counter += 1
                    row = _prediction_event_row(event, predicted, replay)
                    if event.t == 0 or event.queried:
                        diagnostics = row.get("controller_diagnostics_at_evidence_boundary", {})
                        b7_timeline.append({
                            "t": event.t,
                            "query_event": event.queried,
                            "memory_version": event.memory_after.version,
                            "evidence_revision": replay.trace[-1]["evidence_revision"],
                            "threshold": replay.trace[-1]["threshold_post"].get("B7"),
                            "diagnostics": diagnostics.get("B7"),
                        })
                        hold_timeline.append({
                            "t": event.t,
                            "query_event": event.queried,
                            "evidence_revision": replay.trace[-1]["evidence_revision"],
                            "controllers": {
                                name: diagnostics[name] for name in
                                ("B1", "B3", "B4", "B5a", "B5b",
                                 "B5a_FROZEN_FALLBACK", "B5b_FROZEN_FALLBACK",
                                 "Stage075BOffset") if name in diagnostics},
                        })
                    yield row

            prediction_sha = _write_gzip_jsonl_once(prediction_path, rows())
            if event_counter != trajectory.event_count or replay.cursor != trajectory.event_count:
                raise ComparisonError(f"Incomplete replay for {category}/seed{seed}")
            response_sha = replay.verify_query_response_chain(
                trajectory.query_response_chain_sha256)
            if trajectory.event_count != int(identity["event_count"]):
                raise ComparisonError(f"Readiness receipt event count differs: {category}/seed{seed}")

            trace = replay.trace
            b7 = replay.controllers["B7"]
            threshold_summaries = {}
            for name in METHODS:
                pre_values = [prediction.threshold
                              for prediction in replay.predictions[name]]
                post_values = [r["threshold_post"].get(name) for r in trace]
                threshold_summaries[name] = _summarize_threshold_dynamics(
                    pre_values, post_values)
            modes = {
                name: summarize_prediction_modes([p.reason for p in replay.predictions[name]])
                for name in METHODS
            }
            native_available = {}
            for name in ("B5a", "B5b"):
                native_available[name] = sum(
                    p.available and p.label is not None for p in replay.predictions[name])
            for name in ("B5a_FROZEN_FALLBACK", "B5b_FROZEN_FALLBACK"):
                native_available[name] = modes[name]["original_method"]
            trajectory_row = {
                "trajectory_id": f"{category}/seed{seed}",
                "category": category, "support_seed": seed,
                "event_count": trajectory.event_count,
                "query_count": len(trajectory.query_positions),
                "receipt_sha256": receipt_sha,
                "evidence_file_sha256": identity["evidence_sha256"],
                "input_binding_sha256": input_binding,
                "query_response_chain_sha256": response_sha,
                "final_memory": identity["final_memory"],
                "prediction_file": prediction_path.relative_to(output).as_posix(),
                "prediction_file_sha256": prediction_sha,
                "threshold_dynamics": threshold_summaries,
                "prediction_modes": modes,
                "source_native_decisions": native_available,
                "b7_diagnostics_final": b7.diagnostics(),
                "b7_evidence_revision_timeline": b7_timeline,
                "hold_and_evidence_diagnostics_timeline": hold_timeline,
                "controller_final_thresholds": {
                    name: trace[-1]["threshold_post"].get(name) for name in METHODS},
                "writer_transition_verified_by_adapter": True,
                "query_chain_verified_against_p12_receipt": True,
                "native_scorer_or_gpu_rerun": False,
            }
            result_rows.append(trajectory_row)
            trajectory_total += 1
            progress["completed_trajectories"] = [r["trajectory_id"] for r in result_rows]
            progress["completed_events"] += trajectory.event_count
            progress["last_completed_trajectory"] = trajectory_row["trajectory_id"]
            progress["last_prediction_sha256"] = prediction_sha
            _write_progress(progress_path, progress)
            print(f"PREDICTIONS {trajectory_total}/20 {category}/seed{seed}: "
                  f"{trajectory.event_count} events, {prediction_sha}", flush=True)

    if trajectory_total != 20 or progress["completed_events"] != 76475:
        raise ComparisonError("Formal replay did not complete all 20 frozen trajectories")
    receipt_rows = {
        "schema": "stage1a.formal_comparison.prediction_receipts.v1",
        "status": "PASS_ALL_20_CAUSAL_REPLAYS",
        "trajectory_count": trajectory_total,
        "event_count": progress["completed_events"],
        "query_tape_binding_sha256": frozen.TAPE_BINDING_SHA256,
        "controller_rows": [DISPLAY_NAMES.get(name, name) for name in METHODS],
        "native_or_gpu_execution": False,
        "outcome_labels_accessed": False,
        "trajectories": result_rows,
        "prediction_phase_elapsed_seconds": time.perf_counter() - start,
    }
    receipts_sha = _write_json_once(output / "prediction_receipts.json", receipt_rows)
    prediction_inventory = {
        row["prediction_file"]: row["prediction_file_sha256"] for row in result_rows}
    seal = {
        "schema": "stage1a.formal_comparison.prediction_phase_seal.v1",
        "status": "PREDICTION_PHASE_COMPLETE_LABEL_FIREWALL_INTACT",
        "trajectory_count": 20,
        "event_count": 76475,
        "prediction_receipts_sha256": receipts_sha,
        "prediction_files": prediction_inventory,
        "outcome_labels_accessed": False,
        "confirmation_or_mask_access": False,
        "native_scorer_or_gpu_rerun": False,
    }
    seal_sha = _write_json_once(output / "prediction_phase_complete.json", seal)
    progress.update({"status": "PREDICTION_PHASE_COMPLETE",
                     "prediction_phase_complete_sha256": seal_sha,
                     "outcome_labels_accessed": False})
    _write_progress(progress_path, progress)
    return {"prediction_receipts": receipt_rows,
            "prediction_receipts_sha256": receipts_sha,
            "prediction_phase_seal_sha256": seal_sha}


def _load_prediction_stream(path: Path) -> dict[str, Any]:
    scores: list[float] = []
    static_scores: list[float] = []
    predictions = {method: [] for method in METHODS}
    availability = {method: [] for method in METHODS}
    reasons = {method: [] for method in METHODS}
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        for expected_t, line in enumerate(stream):
            row = json.loads(line)
            if row.get("t") != expected_t or set(row.get("methods", {})) != set(METHODS):
                raise ComparisonError(f"Malformed ordered prediction stream: {path.name}:{expected_t}")
            scores.append(float(row["dynamic_score"]))
            static_scores.append(float(row["static_score"]))
            for method in METHODS:
                prediction = row["methods"][method]
                predictions[method].append(prediction["prediction"])
                availability[method].append(bool(prediction["available"]))
                reasons[method].append(str(prediction["reason"]))
    return {"scores": scores, "static_scores": static_scores,
            "predictions": predictions, "availability": availability,
            "reasons": reasons}


def _enrich_method_metrics(row: Mapping[str, Any], labels: Sequence[int],
                           predictions: Sequence[int | None],
                           availability: Sequence[bool], reasons: Sequence[str],
                           method: str) -> dict[str, Any]:
    output = dict(row)
    output.update({"TP": row["tp"], "TN": row["tn"], "FP": row["fp"], "FN": row["fn"],
                   "BA": row["balanced_accuracy"], "FPR": row["fpr"], "FNR": row["fnr"],
                   "defect_recall": None if row["fnr"] is None else 1.0 - row["fnr"]})
    normal_n = sum(y == 0 for y in labels)
    defect_n = sum(y == 1 for y in labels)
    normal_answered = sum(y == 0 and p is not None and a
                          for y, p, a in zip(labels, predictions, availability))
    defect_answered = sum(y == 1 and p is not None and a
                          for y, p, a in zip(labels, predictions, availability))
    output["normal_coverage"] = normal_answered / normal_n if normal_n else None
    output["defect_coverage"] = defect_answered / defect_n if defect_n else None
    selective_fpr = row["fpr_answered_only"]
    selective_fnr = row["fnr_answered_only"]
    output["selective_fpr"] = selective_fpr
    output["selective_fnr"] = selective_fnr
    output["selective_balanced_accuracy"] = (
        1.0 - (selective_fpr + selective_fnr) / 2.0
        if selective_fpr is not None and selective_fnr is not None else None)
    full_windows = [item for item in row["windows"]["windows"] if item["n"] == 256]
    output["full_256_image_windows_only"] = full_windows
    output["worst_defined_full_window_balanced_accuracy"] = min(
        (item["balanced_accuracy"] for item in full_windows
         if item["balanced_accuracy"] is not None), default=None)
    output["worst_defined_full_window_fpr"] = max(
        (item["fpr"] for item in full_windows if item["fpr"] is not None), default=None)
    output["worst_defined_full_window_fnr"] = max(
        (item["fnr"] for item in full_windows if item["fnr"] is not None), default=None)
    output["prediction_modes"] = summarize_prediction_modes(reasons)
    if method in {"B5a_FROZEN_FALLBACK", "B5b_FROZEN_FALLBACK"}:
        output["source_native_decisions"] = sum(reason == "ORIGINAL_METHOD"
                                                 for reason in reasons)
    else:
        output["source_native_decisions"] = sum(
            bool(a) and p is not None for a, p in zip(availability, predictions))
    output["source_native_decision_fraction"] = (
        output["source_native_decisions"] / len(labels) if labels else None)
    output["startup_unavailable_count"] = sum(not a for a in availability)
    output["available_abstention_count"] = sum(a and p is None
                                                for a, p in zip(availability, predictions))
    output["fallback_fraction"] = output["prediction_modes"]["fallback_fraction"]
    output["original_method_decision_fraction"] = (
        output["prediction_modes"]["original_method"] / len(labels) if labels else None)
    output["startup_fallback_fraction"] = (
        output["prediction_modes"]["startup_fallback"] / len(labels) if labels else None)
    output["abstention_fallback_fraction"] = (
        output["prediction_modes"]["abstention_fallback"] / len(labels) if labels else None)
    return output


def _evaluate_one_stream(labels: Sequence[int], score_stream: Mapping[str, Any],
                         *, horizon_start: int = 1050) -> dict[str, Any]:
    ledger = PrivateOutcomeLedger(labels)
    predictions = score_stream["predictions"]
    availability = score_stream["availability"]
    reasons = score_stream["reasons"]
    dynamic = ledger.evaluate(score_stream["scores"], predictions,
                              prediction_availability=availability)
    static_rank = PrivateOutcomeLedger(labels).evaluate(
        score_stream["static_scores"], {"static_score": [0] * len(labels)})
    method_rows = {
        method: _enrich_method_metrics(dynamic["methods"][method], labels,
                                       predictions[method], availability[method], reasons[method],
                                       method)
        for method in METHODS
    }
    horizon = ledger.evaluate_common_horizon(
        score_stream["scores"], predictions, start=horizon_start,
        prediction_availability=availability, prediction_reasons=reasons)
    horizon_static = PrivateOutcomeLedger(labels[horizon_start:]).evaluate(
        score_stream["static_scores"][horizon_start:],
        {"static_score": [0] * max(0, len(labels) - horizon_start)})
    horizon_rank = horizon.get("metrics") or {}
    horizon_static_rank = horizon_static if len(labels) > horizon_start else {}
    return {
        "event_count": len(labels),
        "label_denominators": {"normal": sum(y == 0 for y in labels),
                               "defect": sum(y == 1 for y in labels)},
        "methods": method_rows,
        "shared_score_ranking": {
            "dynamic_score_auroc": dynamic["shared_score_auroc"],
            "dynamic_score_auprc": dynamic["shared_score_auprc"],
            "dynamic_status": dynamic["ranking_status"],
            "static_score_auroc": static_rank["shared_score_auroc"],
            "static_score_auprc": static_rank["shared_score_auprc"],
            "static_status": static_rank["ranking_status"],
            "controller_count_using_same_dynamic_score": len(METHODS),
        },
        "common_horizon_t_ge_1050": {
            "start_zero_based": horizon_start,
            "stop_exclusive": len(labels),
            "status": horizon["status"],
            "event_count": horizon["n"],
            "methods": {
                method: _enrich_method_metrics(
                    horizon_rank.get("methods", {}).get(method, _empty_metrics()),
                    labels[horizon_start:], predictions[method][horizon_start:],
                    availability[method][horizon_start:], reasons[method][horizon_start:],
                    method)
                for method in METHODS
            } if horizon_rank else {method: None for method in METHODS},
            "shared_score_ranking": {
                "dynamic_score_auroc": horizon_rank.get("shared_score_auroc"),
                "dynamic_score_auprc": horizon_rank.get("shared_score_auprc"),
                "dynamic_status": horizon_rank.get("ranking_status", "UNAVAILABLE_EMPTY_HORIZON"),
                "static_score_auroc": horizon_static_rank.get("shared_score_auroc"),
                "static_score_auprc": horizon_static_rank.get("shared_score_auprc"),
                "static_status": horizon_static_rank.get("ranking_status", "UNAVAILABLE_EMPTY_HORIZON"),
            },
        },
    }


def _empty_metrics() -> dict[str, Any]:
    return {
        "tp": 0, "tn": 0, "fp": 0, "fn": 0, "abstentions": 0,
        "normal_abstentions": 0, "defect_abstentions": 0,
        "missing_predictions": 0, "normal_missing_predictions": 0,
        "defect_missing_predictions": 0, "withheld_predictions": 0,
        "prediction_availability_fraction": None, "coverage": None,
        "normal_denominator": 0, "defect_denominator": 0,
        "fpr": None, "fnr": None, "balanced_accuracy": None,
        "fpr_answered_only": None, "fnr_answered_only": None,
        "longest_consecutive_errors": 0,
        "balanced_accuracy_status": "UNAVAILABLE_EMPTY_HORIZON",
        "windows": {"window_size": 256, "windows": []},
    }


def _mean_defined(values: Iterable[Any]) -> tuple[float | None, int]:
    finite = [float(x) for x in values if isinstance(x, (int, float)) and
              not isinstance(x, bool) and math.isfinite(float(x))]
    return (statistics.fmean(finite), len(finite)) if finite else (None, 0)


def _summarize_metrics(trajectory_metrics: Mapping[str, Any]) -> dict[str, Any]:
    by_category: dict[str, Any] = {}
    overall: dict[str, Any] = {}
    for method in METHODS:
        by_category[method] = {}
        for category in frozen.CLASSES:
            rows = [m for (c, _s), m in trajectory_metrics.items() if c == category]
            by_category[method][category] = {
                metric: {"mean": _mean_defined(
                    row["methods"][method].get(metric) for row in rows)[0],
                         "n_defined": _mean_defined(
                    row["methods"][method].get(metric) for row in rows)[1]}
                for metric in METRIC_NAMES
            }
        rows = list(trajectory_metrics.values())
        overall[method] = {
            metric: {"macro_mean_over_20_class_seed_trajectories": _mean_defined(
                row["methods"][method].get(metric) for row in rows)[0],
                     "n_defined": _mean_defined(
                row["methods"][method].get(metric) for row in rows)[1]}
            for metric in METRIC_NAMES
        }
    return {"aggregation_note": (
                "Macro means over class/support-seed trajectories; repeated support-seed "
                "runs are not pooled as independent images."),
            "by_class_support_seed_macro": by_category,
            "macro_over_20_trajectories": overall}


def _paired_comparisons(trajectory_metrics: Mapping[str, Any]) -> dict[str, Any]:
    rows = []
    for (category, seed), result in sorted(trajectory_metrics.items()):
        methods = result["methods"]
        for left, right in itertools.combinations(METHODS, 2):
            for metric in METRIC_NAMES:
                lv = methods[left].get(metric)
                rv = methods[right].get(metric)
                valid = (isinstance(lv, (int, float)) and not isinstance(lv, bool) and
                         isinstance(rv, (int, float)) and not isinstance(rv, bool))
                rows.append({
                    "category": category, "support_seed": seed,
                    "method_a": DISPLAY_NAMES.get(left, left),
                    "method_b": DISPLAY_NAMES.get(right, right),
                    "metric": metric,
                    "method_a_minus_method_b": float(lv) - float(rv) if valid else None,
                    "status": "PAIRED_DEFINED" if valid else "UNDEFINED_FOR_AT_LEAST_ONE_METHOD",
                })
    summary: dict[str, Any] = {}
    group_keys = sorted({(row["method_a"], row["method_b"], row["metric"], row["category"])
                         for row in rows})
    for a, b, metric, category in group_keys:
        selected = [row for row in rows if row["method_a"] == a and
                    row["method_b"] == b and row["metric"] == metric and
                    row["category"] == category]
        mean, n = _mean_defined(row["method_a_minus_method_b"] for row in selected)
        summary.setdefault(f"{a} vs {b}", {}).setdefault(category, {})[metric] = {
            "mean_paired_difference_across_support_seeds": mean,
            "n_defined_pairs": n,
            "support_seeds": list(frozen.SEEDS),
        }
    return {
        "schema": "stage1a.formal_comparison.paired_results.v1",
        "reference_design": "paired within identical class/support-seed/event stream",
        "inference": "descriptive differences only; no significance or risk guarantee",
        "metrics": list(METRIC_NAMES),
        "pairwise_class_seed_rows": rows,
        "per_class_support_seed_summaries": summary,
    }


def _evaluate_phase(output: Path, json_root: Path) -> dict[str, Any]:
    seal_path = output / "prediction_phase_complete.json"
    if not seal_path.is_file():
        raise ComparisonError("Refusing label access before prediction-phase seal exists")
    seal = json.loads(seal_path.read_text(encoding="utf-8"))
    receipts_path = output / "prediction_receipts.json"
    if (seal.get("status") != "PREDICTION_PHASE_COMPLETE_LABEL_FIREWALL_INTACT" or
            seal.get("outcome_labels_accessed") is not False or
            seal.get("trajectory_count") != 20 or seal.get("event_count") != 76475):
        raise ComparisonError("Prediction phase is not completely sealed")
    receipts = json.loads(receipts_path.read_text(encoding="utf-8"))
    if _sha_file(receipts_path) != seal.get("prediction_receipts_sha256"):
        raise ComparisonError("Prediction receipts changed after the phase seal")
    for rel, expected in seal["prediction_files"].items():
        path = output / rel
        if _sha_file(path) != expected:
            raise ComparisonError(f"Prediction file changed after phase seal: {rel}")

    # Only this post-prediction evaluator imports the official outcome reader.
    from .stage05 import test_labels_for_class

    trajectory_metrics: dict[tuple[str, int], Any] = {}
    label_sources: dict[str, Any] = {}
    summaries = {(row["category"], int(row["support_seed"])): row
                 for row in receipts["trajectories"]}
    for category in frozen.CLASSES:
        labels = test_labels_for_class(json_root, category)
        expected_length = next(
            int(row["event_count"]) for row in receipts["trajectories"]
            if row["category"] == category)
        if len(labels) != expected_length:
            raise ComparisonError(f"Evaluator label length differs for {category}")
        label_sources[category] = {
            "test_image_count": len(labels),
            "official_json_sha256": _sha_file(json_root / f"{category}.json"),
            "labels_retained_in_output": False,
            "mask_files_opened": False,
        }
        # The same official class outcome vector is applied separately to the
        # five registered support-seed score/controller trajectories.
        for seed in frozen.SEEDS:
            receipt = summaries[(category, seed)]
            stream = _load_prediction_stream(output / receipt["prediction_file"])
            result = _evaluate_one_stream(labels, stream)
            trajectory_metrics[(category, seed)] = result
            print(f"EVALUATED {category}/seed{seed}: n={len(labels)}, "
                  f"dynamic AUROC={result['shared_score_ranking']['dynamic_score_auroc']}",
                  flush=True)
        del labels

    metrics_out = {
        "schema": "stage1a.formal_comparison.metrics.v1",
        "scope": {"classes": list(frozen.CLASSES),
                  "support_seeds": list(frozen.SEEDS),
                  "trajectory_count": 20,
                  "query_protocol": "Q-A-TS-v1", "query_rate": 0.03,
                  "query_seed": 37,
                  "controller_rows": [DISPLAY_NAMES.get(name, name) for name in METHODS]},
        "evaluation_contract": {
            "target": "physical image-level defect",
            "full_stream": True,
            "common_horizon_start_zero_based": 1050,
            "window_size": 256,
            "window_stride": 256,
            "worst_window_requires_full_256_images": True,
            "source_native_spot_dspot_ba_null_when_startup_missing": True,
            "missing_denominators": "undefined/null",
            "outcome_labels_are_not_saved": True,
            "masks_or_confirmation_outcomes_accessed": False,
        },
        "label_source_provenance": label_sources,
        "per_class_support_seed": {
            f"{category}/seed{seed}": trajectory_metrics[(category, seed)]
            for category in frozen.CLASSES for seed in frozen.SEEDS},
        "descriptive_macro_summaries": _summarize_metrics(trajectory_metrics),
        "shared_score_ranking_consistency": {
            "one_shared_dynamic_score_tape_per_class_seed": True,
            "threshold_only_methods": list(METHODS),
            "dynamic_and_static_score_metrics_reported_per_trajectory": True,
        },
    }
    metrics_sha = _write_json_once(output / "per_method_metrics.json", metrics_out)
    paired = _paired_comparisons(trajectory_metrics)
    paired_sha = _write_json_once(output / "paired_results.json", paired)
    audit = {
        "schema": "stage1a.formal_comparison.audit.v1",
        "status": "FORMAL_STAGE1A_COMPARISON_COMPLETE",
        "prediction_phase_complete_sha256": _sha_file(seal_path),
        "prediction_receipts_sha256": _sha_file(receipts_path),
        "per_method_metrics_sha256": metrics_sha,
        "paired_results_sha256": paired_sha,
        "prediction_files_sha256": seal["prediction_files"],
        "metrics_file_contains_raw_label_vector": False,
        "controller_input_contains_unqueried_labels": False,
        "confirmation_or_mask_access": False,
        "native_scorer_or_gpu_rerun": False,
        "b6_applicability_only": True,
        "b8_historical_only": True,
        "source_native_spot_dspot_keep_missing_predictions": True,
    }
    audit_sha = _write_json_once(output / "formal_comparison_audit.json", audit)
    manifest = {
        "schema": "stage1a.formal_comparison.run_manifest.v1",
        "status": "COMPLETE",
        "created_utc_date": "2026-10-30",
        "requested_head": REQUIRED_HEAD,
        "actual_head": json.loads((output / "preflight.json").read_text())[
            "actual_head"],
        "branch": frozen.BRANCH,
        "preflight_sha256": _sha_file(output / "preflight.json"),
        "prediction_phase_complete_sha256": _sha_file(seal_path),
        "prediction_receipts_sha256": _sha_file(receipts_path),
        "per_method_metrics_sha256": metrics_sha,
        "paired_results_sha256": paired_sha,
        "formal_comparison_audit_sha256": audit_sha,
        "controller_rows": [DISPLAY_NAMES.get(name, name) for name in METHODS],
        "trajectory_count": 20,
        "event_count": 76475,
        "query_tape_binding_sha256": frozen.TAPE_BINDING_SHA256,
        "outcome_label_vectors_saved": False,
        "mask_or_confirmation_accessed": False,
        "runner_source_sha256": _sha_file(Path(__file__)),
    }
    manifest_sha = _write_json_once(output / "run_manifest.json", manifest)
    _write_json_once(output / "checksums.json", {
        "schema": "stage1a.formal_comparison.checksums.v1",
        "self_excluded_manifest": "checksums.json",
        "files": {path.relative_to(output).as_posix(): _sha_file(path)
                  for path in sorted(output.rglob("*"))
                  if path.is_file() and path.name not in {
                      "checksums.json", "RUN_STATE.json"}},
        "run_manifest_sha256": manifest_sha,
    })
    _write_progress(output / "RUN_STATE.json", {
        "schema": "stage1a.formal_comparison.run_state.v1",
        "status": "COMPLETE",
        "completed_trajectories": 20,
        "completed_events": 76475,
        "outcome_labels_accessed": True,
        "outcome_labels_saved": False,
    })
    return {"status": "COMPLETE", "output": str(output),
            "run_manifest_sha256": manifest_sha,
            "metrics_sha256": metrics_sha, "paired_results_sha256": paired_sha}


def _verify_outputs(output: Path) -> dict[str, Any]:
    checks_path = output / "checksums.json"
    checks = json.loads(checks_path.read_text(encoding="utf-8"))
    failures = []
    for rel, expected in checks["files"].items():
        path = output / rel
        actual = _sha_file(path) if path.is_file() else None
        if actual != expected:
            failures.append({"path": rel, "expected": expected, "actual": actual})
    manifest = json.loads((output / "run_manifest.json").read_text(encoding="utf-8"))
    if _sha_file(output / "run_manifest.json") != checks.get("run_manifest_sha256"):
        failures.append({"path": "run_manifest.json", "expected": checks.get(
            "run_manifest_sha256"), "actual": _sha_file(output / "run_manifest.json")})
    return {"status": "PASS" if not failures and manifest.get("status") == "COMPLETE"
            else "FAIL", "checked_files": len(checks["files"]),
            "failures": failures, "run_manifest": manifest}


def run(*, output: Path = OUTPUT, json_root: Path = Path(
        r"G:\dataset\realiad\jsons\realiad_jsons")) -> dict[str, Any]:
    if output.exists():
        raise ComparisonError(f"New immutable output namespace already exists: {output}")
    preflight = _frozen_input_preflight()
    try:
        _run_prediction_phase(output, preflight)
        result = _evaluate_phase(output, json_root)
        result["output_verification"] = _verify_outputs(output)
        if result["output_verification"]["status"] != "PASS":
            raise ComparisonError("Final artifact checksum verification failed")
        return result
    except Exception as exc:
        if output.exists() and not (output / "FAILURE.json").exists():
            try:
                _write_json_once(output / "FAILURE.json", {
                    "schema": "stage1a.formal_comparison.failure.v1",
                    "status": "INCOMPLETE",
                    "exception_type": type(exc).__name__,
                    "exception": str(exc),
                    "outcome_evaluator_phase_started": (output / "prediction_phase_complete.json").exists(),
                    "output_namespace_is_preserved": True,
                })
            except Exception:
                pass
        raise


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("run", "verify"), nargs="?", default="run")
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--json-root", type=Path,
                        default=Path(r"G:\dataset\realiad\jsons\realiad_jsons"))
    args = parser.parse_args(argv)
    try:
        result = run(output=args.output, json_root=args.json_root) if args.command == "run" \
            else _verify_outputs(args.output)
        print(json.dumps(_json_safe(result), sort_keys=True, indent=2, allow_nan=False))
        return 0 if result.get("status") == "COMPLETE" or result.get("status") == "PASS" else 1
    except Exception as exc:
        print(f"STAGE1A_COMPARISON_FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
