"""Metadata preflight and explicitly authorized post-seal BEA-T1 execution."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from typing import Any, Mapping
from experiments.calibration.final_benchmark.execution_support import sha256_file, verify_attempt_receipt
from experiments.calibration.final_benchmark.launch import validate_config, registered_trajectory, load_principal_bundle
from .artifacts import validate_combined_grant, validate_embedded_stream
from .campaign import NEW, OLD, VerifiedCampaign
from .core import AuditError
from .integrity import load_verified
from .orchestrate import run_verified, verify_audit_source
from .output import validate_destination
from .projection import validate_audit_authorization
from .reference import read_stage_b_metadata, validate_stage_b_summary


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


def _load_inputs(config_path: str | Path, *, repository: Path, preflight: bool):
    config_path = _path(str(config_path), "audit configuration")
    config = _read(config_path, "audit configuration")
    if config.get("schema") != "bea-t1.execution-config.v2":
        raise AuditError("audit configuration schema is invalid")
    raw_plans = config.get("campaign_plans")
    if not isinstance(raw_plans, list) or len(raw_plans) != 2:
        raise AuditError("exactly two historical plans are required")
    plans, roots = {}, {}
    for item in raw_plans:
        if not isinstance(item, Mapping) or item.get("source_commit") not in {OLD, NEW} or item["source_commit"] in plans:
            raise AuditError("historical campaign plan identities are invalid")
        path = _path(item.get("path"), "historical plan")
        plan = _read(path, "historical plan")
        report = validate_config(plan, current_head=None, require_source_binding=True)
        if report.get("source_commit") != item["source_commit"]:
            raise AuditError("historical plan source is mismatched")
        plans[item["source_commit"]] = (plan, sha256_file(path))
        roots[item["source_commit"]] = _path(item.get("private_binding_root"), "private binding root")
    reference_path = _path(config.get("official_stage_b_summary_path"), "Stage B summary")
    metadata = read_stage_b_metadata(str(reference_path))
    grant_path = _path(config.get("stage_b_authorization_path"), "Stage B authorization")
    grant = _read(grant_path, "Stage B authorization")
    grant_sha = sha256_file(grant_path)
    combined = validate_combined_grant(grant=grant, grant_sha256=grant_sha, summary=metadata,
        plans=plans, repository=repository, metadata_only=True)
    source = config.get("audit_source_commit")
    config_sha = sha256_file(config_path)
    reference_sha = sha256_file(reference_path)
    auth_path = config.get("audit_authorization_path")
    auth = None
    if auth_path and _path(auth_path, "audit authorization").is_file():
        auth = _read(auth_path, "audit authorization")
        validate_audit_authorization(auth, audit_source_commit=source,
            seal_hashes=[e["stage_a_category_seal_sha256"] for e in combined.entries.values()],
            reference_sha256=reference_sha, source_plan_sha256=[p[1] for p in plans.values()],
            configuration_sha256=config_sha, stage_b_authorization_sha256=grant_sha)
    elif not preflight:
        raise AuditError("separate BEA-T1 audit authorization is missing")
    return config, config_sha, plans, roots, reference_path, reference_sha, combined, auth


def _trajectory_inputs(plans, roots, combined, records, *, full: bool):
    verified, annotations = [], {}
    for pair, entry in combined.entries.items():
        plan, plan_sha = plans[entry["source_commit"]]
        row = registered_trajectory(plan, *pair)
        bundle_path = roots[entry["source_commit"]] / f"{pair[0]}_{pair[1]}_private_bundle.json"
        payload, identity = load_principal_bundle(bundle_path, config=plan, category=pair[0], condition=pair[1])
        stream = validate_embedded_stream(bundle=payload["category_bundle"], plan_row=row,
            release_sha256=plan["dataset"]["dataset_release_sha256"])
        if identity["annotation_sha256"] != entry["annotation_sha256"]:
            raise AuditError("annotation metadata differs from Stage B authorization")
        identity.update(execution_plan_sha256=plan_sha, execution_plan_file_sha256=plan_sha,
            authorization_file_sha256=entry["stage_a_authorization_file_sha256"],
            category_bundle_sha256=identity["category_bundle_file_sha256"])
        status = Path(entry["stage_a_status_path"])
        if full:
            item = load_verified(status_path=status, expected_identity=identity,
                registered_stream_manifest=stream, expected_event_count=row["test_event_count"],
                hard_cap=row["query_cap_per_principal_arm"], stage_b_record=records[pair])
            if item.journal_sha256 != entry["stage_a_journal_sha256"]:
                raise AuditError("Stage A journal differs from original Stage B grant")
            verified.append(item)
        else:
            receipt = verify_attempt_receipt(status, expected_identity=identity, expected_kind="principal_stage_a")
            if receipt["status"].get("completion_seal_sha256") != entry["stage_a_category_seal_sha256"]:
                raise AuditError("Stage A seal differs from original Stage B grant")
            for filename in ("stage_a_category_report.json", "stage_a_journal.jsonl"):
                if not (status.parent / filename).is_file():
                    raise AuditError("Stage A report or journal is missing")
        annotations[pair] = payload["annotation_path"]
    return verified, annotations


def preflight_config(config_path: str | Path, *, repository: str | Path) -> Mapping[str, Any]:
    repo = Path(repository).resolve()
    config, _, plans, roots, reference, _, combined, auth = _load_inputs(config_path, repository=repo, preflight=True)
    report = {"status": "NOT_READY", "plans": "AVAILABLE", "stage_b_authorization": "AVAILABLE",
              "stage_b_summary_metadata": "AVAILABLE", "audit_authorization": "AVAILABLE" if auth else "MISSING"}
    try:
        verify_audit_source(repo, str(config.get("audit_source_commit")))
        report["source"] = "AVAILABLE"
    except AuditError:
        report["source"] = "INCOMPATIBLE"
    try:
        validate_destination(_path(config.get("output_directory"), "audit output directory"), repository=repo)
        report["output"] = "AVAILABLE"
    except (AuditError, OSError):
        report["output"] = "INCOMPATIBLE"
    _trajectory_inputs(plans, roots, combined, read_stage_b_metadata(str(reference)), full=False)
    report["stage_a_and_embedded_streams"] = "AVAILABLE"
    if all(report[k] == "AVAILABLE" for k in ("source", "output", "audit_authorization")):
        report["status"] = "READY_FOR_AUTHORIZED_RUN"
    return report


def run_config(config_path: str | Path, *, repository: str | Path) -> None:
    repo = Path(repository).resolve()
    config, config_sha, plans, roots, reference, reference_sha, combined, auth = _load_inputs(config_path, repository=repo, preflight=False)
    verify_audit_source(repo, str(config.get("audit_source_commit")))
    destination = _path(config.get("output_directory"), "audit output directory")
    validate_destination(destination, repository=repo)
    summary = _read(reference, "Stage B summary")
    records = validate_stage_b_summary(summary)
    combined = validate_combined_grant(grant=combined.grant, grant_sha256=combined.sha256,
        summary=summary, plans=plans, repository=repo)
    verified, annotations = _trajectory_inputs(plans, roots, combined, records, full=True)
    campaign = VerifiedCampaign(verified)
    run_verified(campaign=campaign, audit_authorization=auth, audit_source_commit=config["audit_source_commit"],
        official_reference_path=reference, stage_b=combined, annotation_paths=annotations,
        configuration_sha256=config_sha, destination=destination, repository=repo,
        artifact_hashes={"original_plan": plans[OLD][1], "continuation_plan": plans[NEW][1],
            "official_metrics": reference_sha, "stage_b_authorization": combined.sha256, "audit_configuration": config_sha})


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="BEA-T1 source-bound CPU-only temporal audit")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--check-execution-boundary", action="store_true")
    group.add_argument("--preflight", action="store_true")
    group.add_argument("--run", action="store_true")
    parser.add_argument("--config", help="absolute path to private execution configuration")
    args = parser.parse_args(argv)
    if args.check_execution_boundary or not (args.preflight or args.run) or not args.config:
        print(json.dumps({"status": "BLOCKED", "reason": "explicit mode and private configuration are required"}))
        return 2
    try:
        repo = Path(__file__).resolve().parents[3]
        if args.preflight:
            report = preflight_config(args.config, repository=repo)
            print(json.dumps(report))
            return 0 if report["status"] == "READY_FOR_AUTHORIZED_RUN" else 2
        run_config(args.config, repository=repo)
    except Exception as exc:
        reason = str(exc) if isinstance(exc, AuditError) else f"{type(exc).__name__}: required input could not be verified"
        print(json.dumps({"status": "BLOCKED", "reason": reason}))
        return 2
    print(json.dumps({"status": "AUDIT_COMPLETE"}))
    return 0
