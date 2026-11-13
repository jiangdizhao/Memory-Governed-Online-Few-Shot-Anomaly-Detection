"""In-memory integration: complete verification precedes any label projection."""
from __future__ import annotations

from pathlib import Path
import json
import subprocess
from typing import Any, Mapping
from experiments.calibration.final_benchmark.evaluation import evaluate_sealed_category
from experiments.calibration.final_benchmark.execution_support import sha256_file
from .aggregate import RESAMPLES, SEED, aggregate_campaign
from .artifacts import VerifiedStageBGrant, category_projection_authorization
from .campaign import VerifiedCampaign
from .core import AuditError, analyze_trajectory
from .integrity import digest, load_verified
from .output import validate_destination, write_outputs
from .projection import verify_evaluation_parity
from .projector import ProjectionAuthority, project_annotation
from .reference import validate_stage_b_summary


def verify_audit_source(repository: str | Path, audit_source_commit: str) -> None:
    repo = Path(repository).resolve()
    try:
        head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo, check=True,
                              capture_output=True, text=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=all", "--",
            "experiments/calibration/bea_temporal_audit", "experiments/calibration/final_benchmark"],
            cwd=repo, check=True, capture_output=True, text=True).stdout
    except (OSError, subprocess.SubprocessError) as exc:
        raise AuditError("audit source identity cannot be verified") from exc
    if head != audit_source_commit or dirty.strip():
        raise AuditError("audit/frozen source differs from the reviewed committed revision")


def run_verified(*, campaign: VerifiedCampaign, audit_authorization: Mapping[str, Any],
                 audit_source_commit: str, official_reference_path: str | Path,
                 stage_b: VerifiedStageBGrant,
                 annotation_paths: Mapping[tuple[str, str], str | Path],
                 configuration_sha256: str,
                 artifact_hashes: Mapping[str, str], destination: str | Path,
                 repository: str | Path) -> None:
    """Analyze in memory and publish aggregate files only after every parity check."""
    if not isinstance(campaign, VerifiedCampaign):
        raise AuditError("all formal campaign seals must be verified before running")
    verify_audit_source(repository, audit_source_commit)
    validate_destination(Path(destination), repository=Path(repository))
    pairs = {(t.category, t.condition) for t in campaign.trajectories}
    if set(annotation_paths) != pairs:
        raise AuditError("all 90 registered annotation sources are required")
    reference_sha = sha256_file(official_reference_path)
    if (artifact_hashes.get("official_metrics") != reference_sha or
            artifact_hashes.get("audit_configuration") != configuration_sha256 or
            artifact_hashes.get("stage_b_authorization") != stage_b.sha256 or
            {artifact_hashes.get("original_plan"), artifact_hashes.get("continuation_plan")} != set(campaign.source_plan_sha256)):
        raise AuditError("official Stage B metrics file does not match the registered reference")
    authority = ProjectionAuthority(campaign=campaign, stage_b=stage_b, audit_authorization=audit_authorization,
        audit_source_commit=audit_source_commit, summary_sha256=reference_sha, configuration_sha256=configuration_sha256)
    try:
        records = validate_stage_b_summary(json.loads(Path(official_reference_path).read_text(encoding="utf-8")))
    except Exception as exc:
        raise AuditError("official Stage B summary is invalid") from exc
    # Recheck every seal/report/journal and source path before opening an annotation.
    for trajectory in campaign.trajectories:
        pair = (trajectory.category, trajectory.condition)
        if records[pair] != trajectory.stage_b_record:
            raise AuditError("Stage B record does not match the verified trajectory")
        current = load_verified(status_path=trajectory.status_path, expected_identity=trajectory.expected_identity,
            registered_stream_manifest=trajectory.registered_stream_manifest,
            expected_event_count=trajectory.seal.expected_event_count, hard_cap=trajectory.hard_cap,
            stage_b_record=records[pair])
        if (current.seal != trajectory.seal or current.journal_sha256 != trajectory.journal_sha256 or
                current.report_sha256 != trajectory.report_sha256):
            raise AuditError("a sealed trajectory changed after campaign verification")
        if not Path(annotation_paths[pair]).is_file():
            raise AuditError(f"registered annotation file is unavailable for {pair[0]}/{pair[1]}")
    results: dict[tuple[str, str], Mapping[str, Any]] = {}
    for trajectory in campaign.trajectories:
        pair = (trajectory.category, trajectory.condition)
        labels = project_annotation(authority=authority, trajectory=trajectory, annotation_path=annotation_paths[pair])
        try:
            evaluated = evaluate_sealed_category(dataset_id="realiad", category=trajectory.category,
                rows=trajectory.rows, seal=trajectory.seal, group_seals=trajectory.groups,
                labels_by_event_id=labels, label_projection=category_projection_authorization(stage_b, pair))
            verify_evaluation_parity(evaluated, trajectory.stage_b_record)
            results[pair] = analyze_trajectory(trajectory.rows, cap=int(len(trajectory.rows) * .03),
                labels=[labels[row["event_id_sha256"]] for row in trajectory.rows])
        finally:
            del labels
    aggregate = aggregate_campaign(results)
    integrity = {**campaign.integrity, "metric_parity": "PASS_ALL_TRAJECTORIES"}
    manifest = {"schema": "bea-t1.audit-manifest.v1", "status": "COMPLETE",
        "source_commits": sorted({t.source_commit for t in campaign.trajectories} | {audit_source_commit}),
        "input_artifact_sha256": dict(artifact_hashes), "authorization_sha256": digest(audit_authorization),
        "bootstrap_seed": SEED, "bootstrap_resamples": RESAMPLES, "claim_scope": "EXPLORATORY_POST_HOC"}
    write_outputs(Path(destination), aggregate=aggregate, integrity=integrity, manifest=manifest, repository=Path(repository))
