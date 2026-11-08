"""Read-only Experiment C validation and historical D2 parity checks."""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from experiments.calibration.e6_bea_d2 import (
    D2_EXPECTED_COUNTS,
    D2ValidatedRun,
    validate_d2_stage_a,
)
from experiments.calibration.e6_bea_closed_loop import (
    EventIdentity,
    _canonical_json,
    _reject_public_label_fields,
    _reject_reparse_ancestors,
    _sha256_file,
)
from experiments.calibration.exp_c_memory_evidence import (
    ARM_IDS,
    C_MANIFEST_SCHEMA,
    C_SCORE_SCHEMA,
    C_SEAL_SCHEMA,
    CStageASeal,
    CATEGORIES,
    ExperimentCError,
    MAX_CATEGORY_OUTPUT_BYTES,
    UNCHANGED_D2_ARMS,
)


@dataclass(frozen=True)
class CValidatedRun:
    category: str
    run_id: str
    run_dir: Path
    manifest: Mapping[str, Any]
    seal: Mapping[str, Any]
    rows: tuple[dict[str, Any], ...]
    manifest_sha256: str
    seal_sha256: str
    score_sha256: str
    resource_sha256: str
    supervisor_sha256: str


def validate_c_stage_a(*, run_dir: Path, category: str, run_id: str,
                       expected_commit: str, protocol_sha256: str,
                       expected_events: Sequence[EventIdentity] | None = None) -> CValidatedRun:
    run_dir = Path(run_dir).resolve(strict=True)
    _reject_reparse_ancestors(run_dir)
    if category not in CATEGORIES or (run_dir / "failure.json").exists():
        raise ExperimentCError(f"{category}: failed or unregistered C run cannot be promoted")
    paths = {name: run_dir / name for name in
             ("run_manifest.json", "stage_a_seal.json", "sealed_scores.jsonl")}
    for path in paths.values():
        _reject_reparse_ancestors(path)
    try:
        manifest = json.loads(paths["run_manifest.json"].read_text(encoding="utf-8"))
        seal = json.loads(paths["stage_a_seal.json"].read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ExperimentCError(f"{category}: C manifest or seal is unreadable") from exc
    manifest_sha = _sha256_file(paths["run_manifest.json"])
    seal_sha = _sha256_file(paths["stage_a_seal.json"])
    score_sha = _sha256_file(paths["sealed_scores.jsonl"])
    expected_files = {"run_manifest.json", "stage_a_seal.json", "sealed_scores.jsonl",
                      "resource_receipt.json", "supervisor_receipt.json",
                      *[f"private_feedback_{arm}.jsonl" for arm in ARM_IDS]}
    if (manifest.get("schema") != C_MANIFEST_SCHEMA or
            manifest.get("status_at_creation") != "INCOMPLETE" or
            manifest.get("stage_b_enabled") is not False or
            manifest.get("run_id") != run_id or manifest.get("category") != category or
            manifest.get("approved_commit") != expected_commit or
            manifest.get("protocol_sha256") != protocol_sha256 or
            manifest.get("stream_event_count") != D2_EXPECTED_COUNTS[category] or
            manifest.get("registered_arm_ids") != list(ARM_IDS) or
            set(manifest.get("output_files", ())) != expected_files or
            {item.name for item in run_dir.iterdir()} != expected_files):
        raise ExperimentCError(f"{category}: C manifest identity or output namespace is invalid")
    if (seal.get("schema") != C_SEAL_SCHEMA or seal.get("status") != "COMPLETE" or
            seal.get("run_id") != run_id or seal.get("category") != category or
            seal.get("event_count") != D2_EXPECTED_COUNTS[category] or
            seal.get("manifest_sha256") != manifest_sha or
            seal.get("score_log") != {"file": "sealed_scores.jsonl", "sha256": score_sha} or
            set(seal.get("arm_query_counts", {})) != set(ARM_IDS) or
            set(seal.get("arm_controller_diagnostics", {})) != set(ARM_IDS) or
            set(seal.get("feedback_artifacts", {})) != set(ARM_IDS)):
        raise ExperimentCError(f"{category}: C completion seal does not match its inputs")
    rows: list[dict[str, Any]] = []
    counts = {arm: 0 for arm in ARM_IDS}
    with paths["sealed_scores.jsonl"].open("r", encoding="utf-8") as stream:
        for line in stream:
            if not line.endswith("\n"):
                raise ExperimentCError(f"{category}: C score row lacks durable newline")
            row = json.loads(line)
            _reject_public_label_fields(row)
            if (row.get("schema") != C_SCORE_SCHEMA or row.get("position") != len(rows) or
                    set(row.get("arm_rows", {})) != set(ARM_IDS)):
                raise ExperimentCError(f"{category}: C event order or seven-arm set is invalid")
            if expected_events is not None:
                if len(rows) >= len(expected_events):
                    raise ExperimentCError(f"{category}: C log contains extra events")
                event = expected_events[len(rows)]
                for key, value in (("position", event.position),
                                   ("original_json_position", event.original_position),
                                   ("event_id_sha256", event.event_id_sha256),
                                   ("image_identity_sha256", event.image_identity_sha256),
                                   ("source_group_sha256", event.source_group_sha256)):
                    if row.get(key) != value:
                        raise ExperimentCError(f"{category}: event {len(rows)} identity differs at {key}")
            for arm in ARM_IDS:
                value = row["arm_rows"][arm].get("query_executed")
                if not isinstance(value, bool):
                    raise ExperimentCError(f"{category}/{arm}: invalid query execution flag")
                counts[arm] += int(value)
            rows.append(row)
    if len(rows) != D2_EXPECTED_COUNTS[category] or (expected_events is not None and
            len(rows) != len(expected_events)):
        raise ExperimentCError(f"{category}: incomplete C Stage A event ledger")
    if counts != seal["arm_query_counts"] or counts != {
            arm: int(seal["arm_controller_diagnostics"][arm]["executed_queries"])
            for arm in ARM_IDS}:
        raise ExperimentCError(f"{category}: C query counts disagree across ledger and seal")
    for arm in ARM_IDS:
        item = seal["feedback_artifacts"][arm]
        feedback_path = run_dir / f"private_feedback_{arm}.jsonl"
        _reject_reparse_ancestors(feedback_path)
        if item != {"file": feedback_path.name, "sha256": _sha256_file(feedback_path)}:
            raise ExperimentCError(f"{category}/{arm}: private feedback checksum mismatch")
    resource_path = run_dir / "resource_receipt.json"
    supervisor_path = run_dir / "supervisor_receipt.json"
    try:
        resource = json.loads(resource_path.read_text(encoding="utf-8"))
        supervisor = json.loads(supervisor_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ExperimentCError(f"{category}: resource or supervisor receipt is unreadable") from exc
    if (resource.get("schema") != "exp-c.resource-receipt.v1" or
            resource.get("run_id") != run_id or resource.get("category") != category or
            resource.get("score_and_feedback_budget_bytes") != MAX_CATEGORY_OUTPUT_BYTES or
            not 0 <= int(resource.get("score_and_feedback_bytes", -1)) <= MAX_CATEGORY_OUTPUT_BYTES):
        raise ExperimentCError(f"{category}: C resource receipt does not match its registered budget")
    if (supervisor.get("status") != "COMPLETED" or supervisor.get("child_exit_code") != 0 or
            not supervisor.get("termination_confirmed") or supervisor.get("active_job_processes") != 0 or
            supervisor.get("category") != category or supervisor.get("run_id") != run_id or
            supervisor.get("source_commit") != expected_commit):
        raise ExperimentCError(f"{category}: Stage A supervisor receipt is not a clean Job Object exit")
    return CValidatedRun(category, run_id, run_dir, manifest, seal, tuple(rows),
                         manifest_sha, seal_sha, score_sha,
                         _sha256_file(resource_path), _sha256_file(supervisor_path))


def verify_c_d2_public_parity(*, c_run: CValidatedRun,
                              d2_run_dir: Path,
                              expected_events: Sequence[EventIdentity]) -> tuple[dict[str, Any], D2ValidatedRun]:
    """Validate historical D2 seals and compare five common arms without labels."""
    d2_run_dir = Path(d2_run_dir).resolve(strict=True)
    _reject_reparse_ancestors(d2_run_dir)
    try:
        d2_manifest = json.loads((d2_run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ExperimentCError(f"{c_run.category}: historical D2 manifest is unavailable") from exc
    d2 = validate_d2_stage_a(
        run_dir=d2_run_dir, category=c_run.category,
        run_id=str(d2_manifest.get("run_id", "")),
        expected_commit=str(d2_manifest.get("approved_commit", "")),
        contract_sha256=str(d2_manifest.get("contract_sha256", "")),
        expected_events=expected_events)
    if len(c_run.rows) != len(d2.rows):
        raise ExperimentCError(f"{c_run.category}: D2 and C event counts differ")
    c_inputs = c_run.manifest.get("support_and_c64_identity", {})
    d_inputs = d2.manifest.get("support_and_c64_identity", {})
    input_keys = ("event_stream_identity_sha256", "category_json_sha256",
                  "support_image_sha256", "support_group_sha256", "fixed_bank_sha256",
                  "c64_count", "c64_image_sha256", "c64_group_sha256",
                  "c64_score_archive_sha256", "c64_score_vector_sha256")
    input_match = {key: c_inputs.get(key) == d_inputs.get(key) for key in input_keys}
    if not all(input_match.values()):
        raise ExperimentCError(f"{c_run.category}: Support/C64 identity differs from historical D2")
    tolerance = d2.manifest.get("input_identity", {}).get("c64_score_tolerance", {})
    atol, rtol = float(tolerance.get("atol", 0.0)), float(tolerance.get("rtol", 0.0))
    exact_fields = ("query_proposed", "query_executed", "group_suppressed",
                    "scorer_state_version", "scorer_state_sha256", "correction_bank_size",
                    "query_reason", "query_utility", "query_threshold", "query_objective",
                    "query_cost_runtime", "query_random_u", "query_tie_probability",
                    "query_tie_u", "query_n_gt", "query_n_eq", "utility_history_count")
    numeric_fields = ("gc_score", "c0_score", "b3_threshold", "c64_sigma")
    per_arm: dict[str, Any] = {}
    for arm in UNCHANGED_D2_ARMS:
        c_initial = c_run.manifest.get("initial_arm_states", {}).get(arm, {})
        d_initial = d2.manifest.get("initial_arm_states", {}).get(arm, {})
        initial_fields = ("fixed_bank_sha256", "scorer_state_sha256",
                          "scorer_state_version", "coverage_state_sha256",
                          "c64_group_sha256", "c64_score_archive_sha256",
                          "c64_initial_score_vector_sha256")
        initial_match = {field: c_initial.get(field) == d_initial.get(field)
                         for field in initial_fields}
        c_final = c_run.seal.get("final_arm_state", {}).get(arm, {})
        d_final = d2.seal.get("final_arm_state", {}).get(arm, {})
        final_fields = ("scorer_state_version", "scorer_state_sha256",
                        "coverage_state_sha256", "queried_source_group_count")
        final_match = {field: c_final.get(field) == d_final.get(field)
                       for field in final_fields}
        if not all(initial_match.values()) or not all(final_match.values()):
            raise ExperimentCError(f"{c_run.category}/{arm}: initial or final state identity differs from D2")
        mismatch = {field: 0 for field in exact_fields}
        numeric_max = {field: 0.0 for field in numeric_fields}
        outside = {field: 0 for field in numeric_fields}
        for c_row, d_row in zip(c_run.rows, d2.rows):
            for field in ("position", "original_json_position", "event_id_sha256",
                          "image_identity_sha256", "source_group_sha256"):
                if c_row.get(field) != d_row.get(field):
                    raise ExperimentCError(f"{c_run.category}: {arm} event identity mismatch at {field}")
            c_arm, d_arm = c_row["arm_rows"][arm], d_row["arm_rows"][arm]
            for field in exact_fields:
                mismatch[field] += int(c_arm.get(field) != d_arm.get(field))
            for field in numeric_fields:
                a, b = float(c_arm[field]), float(d_arm[field])
                numeric_max[field] = max(numeric_max[field], abs(a - b))
                outside[field] += int(not math.isclose(a, b, rel_tol=rtol, abs_tol=atol))
            if c_arm.get("components") != d_arm.get("components"):
                mismatch["components"] = mismatch.get("components", 0) + 1
        c_count = int(c_run.seal["arm_query_counts"][arm])
        d_count = int(d2.seal["arm_query_counts"][arm])
        if any(mismatch.values()) or any(outside.values()) or c_count != d_count:
            raise ExperimentCError(f"{c_run.category}/{arm}: historical D2 public trajectory diverged")
        per_arm[arm] = {"status": "PASS", "exact_mismatch_counts": mismatch,
                        "initial_state_match": initial_match,
                        "final_state_match": final_match,
                        "numeric_max_abs_difference": numeric_max,
                        "numeric_values_outside_d2_tolerance": outside,
                        "query_count_C": c_count, "query_count_D2": d_count}
    return {"status": "PASS", "category": c_run.category,
            "d2_run_dir": str(d2_run_dir), "d2_manifest_sha256": d2.manifest_sha256,
            "d2_seal_sha256": d2.seal_sha256, "d2_score_sha256": d2.score_log_sha256,
            "support_c64_identity_match": input_match,
            "registered_tolerance": {"atol": atol, "rtol": rtol},
            "arms": per_arm}, d2


def verify_d2_metric_reproduction(*, d2_run_id: str,
                                  c_evaluation: Mapping[str, Any],
                                  d2_evaluation: Mapping[str, Any] | None = None,
                                  d2_stage_b_dir: Path | None = None) -> dict[str, Any]:
    """Compare common-arm metrics using C's official-label projection only.

    D2's saved evaluation contains aggregate metrics, not labels. No historical
    private feedback or TEST outcome file is opened by this function.
    """
    if d2_stage_b_dir is None:
        raise ExperimentCError("historical D2 Stage B metric artifact path is required")
    stored_d2_evaluation = None
    if d2_stage_b_dir is not None:
        stage_b = Path(d2_stage_b_dir).resolve(strict=True)
        provenance_path = stage_b / "D2_provenance.json"
        evaluation_path = stage_b / "D2_evaluation.json"
        try:
            provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
            stored_d2_evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ExperimentCError("sealed historical D2 metric artifacts are missing or unreadable") from exc
        if (provenance.get("status") != "COMPLETE" or provenance.get("run_id") != d2_run_id or
                provenance.get("stage_b_evaluation_sha256") != _sha256_file(evaluation_path)):
            raise ExperimentCError("historical D2 Stage B evaluation checksum or identity failed")
    if d2_evaluation is None:
        d2_evaluation = stored_d2_evaluation
    if d2_evaluation is None or stored_d2_evaluation is None:
        raise ExperimentCError("historical D2 Stage B evaluation is unavailable")
    if stored_d2_evaluation is not d2_evaluation:
        # Verify that D2's registered saved metrics reproduce under the same
        # current official-label projection without opening historical labels.
        for category in CATEGORIES:
            for arm in UNCHANGED_D2_ARMS:
                for metric in ("ber", "ba", "fpr", "fnr", "auroc", "ap",
                               "worst_256_window_ber", "query_rate"):
                    a = float(d2_evaluation["category_metrics"][category][arm][metric])
                    b = float(stored_d2_evaluation["category_metrics"][category][arm][metric])
                    if not math.isclose(a, b, rel_tol=0.0, abs_tol=1e-12):
                        raise ExperimentCError(
                            f"{category}/{arm}: recomputed D2 metric differs from saved historical {metric}")
    d2_categories = d2_evaluation.get("category_metrics", {})
    d2_macro = d2_evaluation.get("category_equal_macro", {})
    c_categories = c_evaluation.get("per_category", {})
    c_macro = c_evaluation.get("category_equal_macro", {})
    if set(d2_categories) != set(CATEGORIES) or set(c_categories) != set(CATEGORIES):
        raise ExperimentCError("D2/C evaluation does not contain all four categories")
    compared = {}
    metrics = ("ber", "ba", "fpr", "fnr", "auroc", "ap", "worst_256_window_ber", "query_rate")
    for category in CATEGORIES:
        compared[category] = {}
        for arm in UNCHANGED_D2_ARMS:
            fields = {}
            for metric in metrics:
                left = float(c_categories[category][arm][metric])
                right = float(d2_categories[category][arm][metric])
                fields[metric] = {"C": left, "D2": right, "difference": left - right}
                if not math.isclose(left, right, rel_tol=0.0, abs_tol=1e-12):
                    raise ExperimentCError(f"{category}/{arm}: D2 metric reproduction differs at {metric}")
            if int(c_categories[category][arm]["query_count"]) != int(
                    d2_categories[category][arm]["query_count"]):
                raise ExperimentCError(f"{category}/{arm}: D2 query count reproduction differs")
            compared[category][arm] = fields
    macro_comparison = {}
    for arm in UNCHANGED_D2_ARMS:
        macro_comparison[arm] = {}
        for metric in ("ber", "ba", "fpr", "fnr", "auroc", "ap", "worst_256_window_ber", "query_rate"):
            left = float(c_macro[arm][metric])
            right = float(d2_macro[arm][metric])
            if not math.isclose(left, right, rel_tol=0.0, abs_tol=1e-12):
                raise ExperimentCError(f"{arm}: D2 macro metric reproduction differs at {metric}")
            macro_comparison[arm][metric] = {"C": left, "D2": right, "difference": left - right}
    return {"status": "PASS", "historical_d2_run_id": d2_run_id,
            "metric_source": "D2 stored aggregate metrics; re-evaluated from C Stage B official-label projection",
            "historical_label_files_opened": False,
            "per_category": compared, "category_equal_macro": macro_comparison}
