"""Source-bound, post-seal BEA-T1 command; no default real-data access."""
from __future__ import annotations
import argparse
from dataclasses import fields
import json
from pathlib import Path
from typing import Any, Mapping

from experiments.calibration.final_benchmark.evaluation import LabelProjectionAuthorization
from experiments.calibration.final_benchmark.execution_support import sha256_file
from experiments.calibration.final_benchmark.launch import validate_config, registered_trajectory
from .campaign import NEW, OLD, VerifiedCampaign
from .core import AuditError
from .integrity import load_verified
from .orchestrate import run_verified, verify_audit_source
from .output import validate_destination
from .reference import validate_stage_b_summary


def _read(path: str | Path, description: str) -> Mapping[str, Any]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception as exc:
        raise AuditError(f"{description} is missing or unreadable") from exc
    if not isinstance(value, Mapping):
        raise AuditError(f"{description} must be a JSON object")
    return value


def _path(value: Any, description: str) -> Path:
    if not isinstance(value, str) or not Path(value).is_absolute():
        raise AuditError(f"{description} must be an absolute path")
    return Path(value)


def _sha(path: str | Path, description: str) -> str:
    try:
        return sha256_file(path)
    except Exception as exc:
        raise AuditError(f"{description} is missing or unreadable") from exc


def run_config(config_path: str | Path, *, repository: str | Path) -> None:
    """Verify grants, registration and all 90 seals before loading any label file."""
    repo = Path(repository).resolve()
    config_path = _path(str(config_path), "audit configuration")
    config = _read(config_path, "audit configuration")
    config_sha = sha256_file(config_path)
    if config.get("schema") != "bea-t1.execution-config.v1":
        raise AuditError("audit configuration schema is invalid")
    audit_commit = config.get("audit_source_commit")
    auth = _read(_path(config.get("audit_authorization_path"), "audit authorization"), "audit authorization")
    if (auth.get("schema") != "bea-t1.audit-authorization.v1" or
            auth.get("status") != "RESEARCHER_AUTHORIZED_POST_SEAL_AUDIT" or
            auth.get("independent_review_status") != "PASS" or auth.get("audit_source_commit") != audit_commit or
            auth.get("configuration_sha256") != config_sha or auth.get("trajectory_count") != 90 or
            auth.get("mask_access") is not False or auth.get("output_scope") != "AGGREGATE_ONLY"):
        raise AuditError("reviewed, configuration-bound BEA-T1 authorization is missing or invalid")
    verify_audit_source(repo, str(audit_commit))
    destination = _path(config.get("output_directory"), "audit output directory")
    validate_destination(destination, repository=repo)
    reference_path = _path(config.get("official_stage_b_summary_path"), "official Stage B summary")
    if _sha(reference_path, "official Stage B summary") != auth.get("official_metrics_sha256"):
        raise AuditError("official Stage B summary hash differs from the audit authorization")
    records = validate_stage_b_summary(_read(reference_path, "official Stage B summary"))

    plans_raw = config.get("campaign_plans")
    if not isinstance(plans_raw, list) or len(plans_raw) != 2:
        raise AuditError("exactly two historical source-bound campaign plans are required")
    plans: dict[str, tuple[Mapping[str, Any], str]] = {}
    for item in plans_raw:
        if not isinstance(item, Mapping) or item.get("source_commit") not in {OLD, NEW} or item["source_commit"] in plans:
            raise AuditError("historical source plan registrations are invalid")
        plan_path = _path(item.get("path"), "historical campaign plan")
        plan = _read(plan_path, "historical campaign plan")
        try:
            report = validate_config(plan, current_head=None, require_source_binding=True)
        except Exception as exc:
            raise AuditError("historical campaign plan failed its frozen validation") from exc
        if report.get("source_commit") != item["source_commit"]:
            raise AuditError("historical plan source commit is mismatched")
        plans[item["source_commit"]] = (plan, _sha(plan_path, "historical campaign plan"))
    if set(plans) != {OLD, NEW} or sorted(x[1] for x in plans.values()) != auth.get("source_plan_sha256"):
        raise AuditError("historical plan file hashes differ from the audit authorization")

    entries = config.get("trajectories")
    if not isinstance(entries, list) or len(entries) != 90 or any(not isinstance(x, Mapping) for x in entries):
        raise AuditError("audit configuration must contain 90 trajectory inputs")
    pairs = [(e.get("category"), e.get("condition")) for e in entries]
    if len(set(pairs)) != 90 or set(pairs) != set(records):
        raise AuditError("audit trajectory inputs do not match the Stage B schedule")
    verified = []
    stage_b_grants: dict[tuple[str, str], LabelProjectionAuthorization] = {}
    projection_grants: dict[tuple[str, str], Mapping[str, Any]] = {}
    projection_paths: dict[tuple[str, str], Path] = {}
    for entry in entries:
        pair = (entry["category"], entry["condition"])
        record = records[pair]
        source = record["source_commit"]
        plan, plan_sha = plans[source]
        row = registered_trajectory(plan, *pair)
        identity = entry.get("expected_attempt_identity")
        if not isinstance(identity, Mapping):
            raise AuditError("attempt identity is missing from the authorized configuration")
        expected = {"source_commit": source, "campaign_run_id": plan["source_identity"]["campaign_run_id"],
            "execution_plan_sha256": plan_sha, "identity_manifest_sha256": plan["dataset"]["identity_manifest_sha256"],
            "dataset_release_sha256": plan["dataset"]["dataset_release_sha256"],
            "role_manifest_sha256": row["role_manifest_sha256"], "stream_manifest_sha256": row["stream_manifest_sha256"],
            "category": pair[0], "condition": pair[1]}
        if any(identity.get(k) != v for k, v in expected.items()):
            raise AuditError("attempt identity differs from its registered historical plan")
        stream = _read(_path(entry.get("stream_manifest_path"), "registered stream manifest"), "registered stream manifest")
        try:
            verified.append(load_verified(status_path=_path(entry.get("attempt_status_path"), "Stage A attempt receipt"),
                expected_identity=identity, registered_stream_manifest=stream,
                expected_event_count=row["test_event_count"], hard_cap=row["query_cap_per_principal_arm"],
                stage_b_record=record))
        except AuditError as exc:
            raise AuditError(f"{pair[0]}/{pair[1]}: {exc}") from exc
        sb_path = _path(entry.get("stage_b_authorization_path"), "Stage B authorization")
        pg_path = _path(entry.get("projection_authorization_path"), "audit-only projection authorization")
        if _sha(sb_path, "Stage B authorization") != entry.get("stage_b_authorization_file_sha256"):
            raise AuditError("Stage B authorization file digest differs from the approved audit configuration")
        sb = _read(sb_path, "Stage B authorization")
        try:
            names = {f.name for f in fields(LabelProjectionAuthorization)}
            stage_b_grants[pair] = LabelProjectionAuthorization(**{k: v for k, v in sb.items() if k in names})
        except Exception as exc:
            raise AuditError("Stage B authorization fields are incomplete") from exc
        projection_grants[pair] = _read(pg_path, "audit-only projection authorization")
        projection_paths[pair] = _path(entry.get("projection_file_path"), "authorized Stage B identity projection")

    campaign = VerifiedCampaign(verified)
    run_verified(campaign=campaign, audit_authorization=auth, audit_source_commit=str(audit_commit),
        official_reference_path=reference_path, stage_b_grants=stage_b_grants,
        projection_grants=projection_grants, projection_paths=projection_paths,
        configuration_sha256=config_sha, destination=destination, repository=repo,
        artifact_hashes={"original_plan": plans[OLD][1], "continuation_plan": plans[NEW][1],
                         "official_metrics": _sha(reference_path, "official Stage B summary"), "audit_configuration": config_sha})


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="BEA-T1 source-bound CPU-only temporal audit")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--check-execution-boundary", action="store_true")
    group.add_argument("--run", action="store_true", help="run only with all reviewed authorizations and registered inputs")
    parser.add_argument("--config", help="absolute path of the separately authorized private audit configuration")
    args = parser.parse_args(argv)
    if args.check_execution_boundary:
        print(json.dumps({"status": "BLOCKED", "reason": "reviewed source, audit and Stage B projection grants, verified identity projections, and Git-ignored output are required"}))
        return 2
    if not args.run:
        print(json.dumps({"status": "BLOCKED", "reason": "an authorized --run and private --config are required"}))
        return 2
    if not args.config:
        print(json.dumps({"status": "BLOCKED", "reason": "an authorized private --config is required"}))
        return 2
    try:
        run_config(args.config, repository=Path(__file__).resolve().parents[3])
    except Exception as exc:
        reason = str(exc) if isinstance(exc, AuditError) else f"{type(exc).__name__}: required input could not be verified"
        print(json.dumps({"status": "BLOCKED", "reason": reason}))
        return 2
    print(json.dumps({"status": "AUDIT_COMPLETE"}))
    return 0
