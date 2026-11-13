"""Non-overwriting aggregate output; event identities and labels never serialize."""
from __future__ import annotations
import csv
import json
import math
import re
import subprocess
from pathlib import Path
from typing import Any, Mapping, Sequence
from .core import ARMS, AuditError

ROW_KEYS = {"segment", "arm", "metric", "category_count", "defined_category_count", "mean", "ci95_low", "ci95_high", "status"}
MANIFEST_KEYS = {"schema", "status", "source_commits", "input_artifact_sha256", "authorization_sha256", "bootstrap_seed", "bootstrap_resamples", "claim_scope"}


def validate_destination(destination: Path, *, repository: Path, synthetic_test_only: bool = False) -> Path:
    repository = Path(repository).resolve()
    destination = Path(destination).resolve()
    allowed = (repository / ("tests/calibration/bea_temporal_audit" if synthetic_test_only else
                              "outputs/calibration/bea_temporal_audit")).resolve()
    if synthetic_test_only:
        try:
            destination.relative_to(allowed)
        except ValueError as exc:
            raise AuditError("synthetic output must remain under the BEA-T1 test directory") from exc
    elif destination.parent != allowed:
        raise AuditError("audit output must be a fresh direct child of its registered root")
    if destination.exists():
        raise FileExistsError("audit output directory already exists")
    if not destination.parent.is_dir():
        raise AuditError("registered audit output root does not exist")
    if not synthetic_test_only:
        probe = subprocess.run(["git", "check-ignore", "-q", "--no-index", str(destination / "audit_manifest.json")],
                               cwd=repository, check=False, capture_output=True)
        if probe.returncode != 0:
            raise AuditError("real audit output directory is not Git-ignored")
    return destination


def write_outputs(destination: Path, *, aggregate: Sequence[Mapping[str, Any]],
                  integrity: Mapping[str, Any], manifest: Mapping[str, Any],
                  repository: Path, synthetic_test_only: bool = False) -> None:
    """Only fixed aggregate schemas are accepted; directories are exclusively created."""
    allowed_integrity = {"trajectory_count", "category_count", "event_count", "executed_queries", "cap_suppressed", "status", "metric_parity"}
    if set(manifest) != MANIFEST_KEYS or not set(integrity).issubset(allowed_integrity):
        raise AuditError("output manifest or integrity schema contains forbidden fields")
    for name in ("executed_queries", "cap_suppressed"):
        if name in integrity and (not isinstance(integrity[name], Mapping) or set(integrity[name]) != set(ARMS) or
                any(type(v) is not int or v < 0 for v in integrity[name].values())):
            raise AuditError("integrity accounting is not aggregate arm-level data")
    if (any(type(integrity.get(k)) is not int or integrity[k] < 0 for k in ("trajectory_count", "category_count", "event_count")) or
            integrity.get("status") != "PASS" or integrity.get("metric_parity") != "PASS_ALL_TRAJECTORIES"):
        raise AuditError("integrity summary is not a verified aggregate")
    if not synthetic_test_only and (
            (integrity["trajectory_count"], integrity["category_count"], integrity["event_count"]) != (90, 30, 343755) or
            integrity.get("executed_queries") != {"F-B":10272, "K-B":10272, "F-M":10121, "K-M":10068} or
            integrity.get("cap_suppressed", {}).get("F-B") != 3287 or
            integrity.get("cap_suppressed", {}).get("K-B") != 3492):
        raise AuditError("formal campaign accounting does not meet the registered totals")
    if any(set(row) != ROW_KEYS for row in aggregate):
        raise AuditError("aggregate row contains a forbidden or missing field")
    allowed_arms = set(ARMS) | {"STATIC-4+64", "N-shadow", "F-B minus F-M", "K-B minus K-M",
        "F-B minus STATIC-4+64", "K-B minus STATIC-4+64", "F-B minus N-shadow"}
    allowed_metrics = {"proposed", "executed", "group_suppressed", "cap_suppressed", "groups_with_proposal",
        "groups_with_execution", "groups_with_cap_suppression", "fpr", "fnr", "ber", "auroc", "ap",
        "normal_count", "defect_count", "predicted_normal", "predicted_defect", "group_count", "image_count", "cap_exhaustion_fraction", "fixed_price_proposals",
        *(x + "_" + y for x in ("predicted_normal", "predicted_defect") for y in ("normal_count", "defect_count", "defect_prevalence")),
        "fixed_price_disagreements", *("cap_suppressed_" + x for x in ("count", "normal_count", "defect_count", "defect_prevalence", "hidden_false_negatives")),
        *(x + "_" + y for x in ("objective", "price", "objective_minus_price", "pre_cap_price", "post_cap_price") for y in ("mean", "p10", "median", "p90"))}
    for row in aggregate:
        if (row["segment"] not in {"early", "middle", "late", "overall", *(f"decile_{i:02d}" for i in range(1, 11))} or
                row["arm"] not in allowed_arms or row["metric"] not in allowed_metrics or
                row["status"] not in {"DEFINED", "UNDEFINED_CATEGORY_OR_CONDITION"} or
                any(isinstance(row[k], bool) or not isinstance(row[k], int) or row[k] < 0
                    for k in ("category_count", "defined_category_count")) or
                row["defined_category_count"] > row["category_count"] or
                any(row[k] is not None and (isinstance(row[k], bool) or not isinstance(row[k], (int, float)) or not math.isfinite(row[k]))
                    for k in ("mean", "ci95_low", "ci95_high"))):
            raise AuditError("invalid aggregate output identifier")
    hashes = manifest.get("input_artifact_sha256")
    if (not isinstance(hashes, Mapping) or not set(hashes).issubset({"original_plan", "continuation_plan", "amendment", "official_metrics", "audit_configuration", "stage_b_authorization"}) or
            not hashes or any(not isinstance(v, str) or not re.fullmatch(r"[0-9a-f]{64}", v) for v in hashes.values()) or
            not isinstance(manifest.get("source_commits"), list) or not 1 <= len(manifest["source_commits"]) <= 3 or
            any(not isinstance(v, str) or not re.fullmatch(r"[0-9a-f]{40}", v) for v in manifest["source_commits"]) or
            not isinstance(manifest.get("authorization_sha256"), str) or not re.fullmatch(r"[0-9a-f]{64}", manifest["authorization_sha256"]) or
            manifest.get("bootstrap_seed") != 20261114 or manifest.get("bootstrap_resamples") != 10000 or
            manifest.get("schema") != "bea-t1.audit-manifest.v1" or manifest.get("status") != "COMPLETE" or
            manifest.get("claim_scope") != "EXPLORATORY_POST_HOC"):
        raise AuditError("sanitized output manifest is invalid")
    # Refuse paths, raw per-event identities and labels even in nested supplied values.
    def check(value: Any) -> None:
        if isinstance(value, Mapping):
            for k, v in value.items():
                if any(x in str(k).lower() for x in ("event_id", "image_identity", "path", "label", "receipt", "position")):
                    raise AuditError("private or image-level data cannot enter public output")
                check(v)
        elif isinstance(value, (list, tuple)):
            for x in value: check(x)
        elif isinstance(value, str) and ("\\" in value or "/" in value):
            raise AuditError("paths cannot enter public output")
    check(integrity); check(manifest); check(aggregate)
    destination = validate_destination(destination, repository=repository, synthetic_test_only=synthetic_test_only)
    destination.mkdir(parents=False, exist_ok=False)
    def write(name: str, value: Any) -> None:
        with (destination / name).open("x", encoding="utf-8", newline="\n") as f:
            json.dump(value, f, sort_keys=True, indent=2, allow_nan=False); f.write("\n")
    write("integrity_checks.json", dict(integrity))
    write("audit_summary.json", {"schema": "bea-t1.aggregate-summary.v1", "results": list(aggregate),
        "interpretation": "Exploratory post-hoc diagnostics; no causal or independent-generalization claim."})
    write("audit_manifest.json", dict(manifest))
    with (destination / "temporal_aggregate.csv").open("x", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=sorted(ROW_KEYS)); writer.writeheader(); writer.writerows(aggregate)
    with (destination / "decision_report.md").open("x", encoding="utf-8", newline="\n") as f:
        f.write("# BEA-T1 exploratory decision report\n\nOBSERVED: see aggregate tables.\n\n"
                "COMPATIBLE_WITH_HYPOTHESIS: descriptive patterns require scientific interpretation.\n\n"
                "INCONCLUSIVE — H1 learning saturation: no counterfactual future update is available.\n\n"
                "INCONCLUSIVE — H2 budget censoring: cap-suppressed outcomes describe hidden errors, not query value.\n\n"
                "INCONCLUSIVE — H3 price feedback: replay measures the registered price response, not its causal effect on future detector state.\n\n"
                "NOT_IDENTIFIABLE: individual B/N/D contributions and query counterfactuals are not archived.\n\n"
                "BLOCKED: any failed authorization, seal, accounting or metric-parity check stops output.\n")
