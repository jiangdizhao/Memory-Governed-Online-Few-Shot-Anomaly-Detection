"""Fail-closed post-seal label projection and frozen metric parity boundary."""
from __future__ import annotations

import math
import json
import re
import hashlib
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence
from experiments.calibration.final_benchmark.evaluation import LabelProjectionAuthorization, evaluate_sealed_category
from .core import AuditError
from .campaign import IDENTITY, RELEASE, VerifiedCampaign
from .integrity import VerifiedTrajectory, digest, load_verified
from .reference import validate_stage_b_summary
from experiments.calibration.final_benchmark.execution_support import sha256_file


def validate_audit_authorization(authorization: Mapping[str, Any], *, audit_source_commit: str,
                                 seal_hashes: Sequence[str], reference_sha256: str,
                                 source_plan_sha256: Sequence[str], configuration_sha256: str) -> None:
    """The separate audit grant must be issued externally after independent review."""
    if (not re.fullmatch(r"[0-9a-f]{40}", audit_source_commit) or
            not re.fullmatch(r"[0-9a-f]{64}", reference_sha256) or
            not re.fullmatch(r"[0-9a-f]{64}", configuration_sha256) or
            any(not re.fullmatch(r"[0-9a-f]{64}", str(x)) for x in seal_hashes) or
            len(set(source_plan_sha256)) != 2 or any(not re.fullmatch(r"[0-9a-f]{64}", str(x)) for x in source_plan_sha256) or
            authorization.get("schema") != "bea-t1.audit-authorization.v1" or
            authorization.get("status") != "RESEARCHER_AUTHORIZED_POST_SEAL_AUDIT" or
            authorization.get("independent_review_status") != "PASS" or
            authorization.get("audit_source_commit") != audit_source_commit or
            authorization.get("seal_set_sha256") != digest(sorted(seal_hashes)) or
            authorization.get("source_plan_sha256") != sorted(source_plan_sha256) or
            authorization.get("dataset_release_sha256") != RELEASE or
            authorization.get("identity_manifest_sha256") != IDENTITY or
            authorization.get("official_metrics_sha256") != reference_sha256 or
            authorization.get("configuration_sha256") != configuration_sha256 or
            authorization.get("trajectory_count") != 90 or len(seal_hashes) != 90 or
            len(set(seal_hashes)) != 90 or authorization.get("mask_access") is not False or
            authorization.get("output_scope") != "AGGREGATE_ONLY"):
        raise AuditError("separate reviewed BEA-T1 audit authorization is missing or mismatched")


def verify_metric_parity(actual: Mapping[str, Any], reference: Mapping[str, Any]) -> None:
    def compare(left: Any, right: Any) -> None:
        if isinstance(left, Mapping):
            if not isinstance(right, Mapping) or set(left) != set(right):
                raise AuditError("full-stream metric schema parity failed")
            for key in left:
                compare(left[key], right[key])
        elif left is None or right is None:
            if left is not right:
                raise AuditError("full-stream metric parity failed")
        elif isinstance(left, (int, float)) and not isinstance(left, bool) and isinstance(right, (int, float)) and not isinstance(right, bool):
            if not math.isfinite(left) or not math.isfinite(right) or not math.isclose(left, right, rel_tol=1e-12, abs_tol=1e-12):
                raise AuditError("full-stream metric parity failed")
        elif type(left) is not type(right) or left != right:
            raise AuditError("full-stream metric parity failed")
    compare(actual, reference)


def validate_projection_grant(*, trajectory: VerifiedTrajectory, stage_b_authorization: LabelProjectionAuthorization,
                              projection_grant: Mapping[str, Any], audit_authorization: Mapping[str, Any]) -> None:
    """Check both independently issued grants without touching the label file."""
    t, a, g = trajectory, stage_b_authorization, projection_grant
    if (not isinstance(audit_authorization, Mapping) or
            audit_authorization.get("schema") != "bea-t1.audit-authorization.v1" or
            audit_authorization.get("status") != "RESEARCHER_AUTHORIZED_POST_SEAL_AUDIT" or
            audit_authorization.get("independent_review_status") != "PASS" or
            audit_authorization.get("trajectory_count") != 90 or
            audit_authorization.get("dataset_release_sha256") != RELEASE or
            audit_authorization.get("identity_manifest_sha256") != IDENTITY or
            audit_authorization.get("mask_access") is not False or
            audit_authorization.get("output_scope") != "AGGREGATE_ONLY" or
            not isinstance(a, LabelProjectionAuthorization) or
            a.status != "AUTHORIZED_POST_STAGE_A_LABEL_PROJECTION" or a.scope != "stage_b_category_evaluation" or
            a.dataset_id != "realiad" or a.category != t.category or a.stage_a_seal_sha256 != t.seal.durable_seal_sha256 or
            not re.fullmatch(r"[0-9a-f]{64}", str(a.source_sha256)) or
            not isinstance(g, Mapping) or g.get("schema") != "bea-t1.label-projection-authorization.v1" or
            g.get("status") != "RESEARCHER_AUTHORIZED_BEA_T1_LABEL_PROJECTION" or
            g.get("scope") != "realiad_post_seal_event_labels" or
            g.get("projection_provenance") != "AUTHORIZED_STAGE_B_IDENTITY_PROJECTION" or
            g.get("audit_authorization_sha256") != digest(audit_authorization) or
            g.get("stage_b_authorization_sha256") != digest(asdict(a)) or
            g.get("dataset_id") != "realiad" or g.get("category") != t.category or g.get("condition") != t.condition or
            g.get("campaign_run_id") != t.expected_identity.get("campaign_run_id") or
            g.get("source_commit") != t.source_commit or
            g.get("stage_a_category_seal_sha256") != t.seal.durable_seal_sha256 or
            g.get("annotation_sha256") != t.expected_identity.get("annotation_sha256") or
            not re.fullmatch(r"[0-9a-f]{64}", str(g.get("annotation_sha256", ""))) or
            g.get("source_sha256") != a.source_sha256 or
            t.stage_b_record.get("label_projection_source_sha256", a.source_sha256) != a.source_sha256 or
            not re.fullmatch(r"[0-9a-f]{64}", str(g.get("projection_file_sha256", ""))) or
            g.get("mask_access") is not False):
        raise AuditError("Stage B or audit-only label projection authorization is missing or mismatched")


def _load_authorized_projection(path: str | Path, *, trajectory: VerifiedTrajectory,
                                projection_grant: Mapping[str, Any]) -> Mapping[str, int]:
    """Load only an externally generated, explicitly authorized identity projection."""
    try:
        raw = Path(path).read_bytes()
        if hashlib.sha256(raw).hexdigest() != projection_grant["projection_file_sha256"]:
            raise AuditError("authorized label projection file hash does not match its grant")
        payload = json.loads(raw)
        keys = {"schema", "dataset_id", "category", "condition", "campaign_run_id", "source_commit",
                "stage_a_category_seal_sha256", "annotation_sha256", "source_sha256", "labels_by_event_id"}
        if (not isinstance(payload, Mapping) or set(payload) != keys or payload.get("schema") != "realiad.stage-b-event-label-projection.v1" or
                payload.get("dataset_id") != "realiad" or payload.get("category") != trajectory.category or
                payload.get("condition") != trajectory.condition or
                payload.get("campaign_run_id") != trajectory.expected_identity.get("campaign_run_id") or
                payload.get("source_commit") != trajectory.source_commit or
                payload.get("stage_a_category_seal_sha256") != trajectory.seal.durable_seal_sha256 or
                payload.get("annotation_sha256") != trajectory.expected_identity.get("annotation_sha256") or
                payload.get("source_sha256") != projection_grant.get("source_sha256") or
                not isinstance(payload.get("labels_by_event_id"), Mapping)):
            raise AuditError("authorized identity projection metadata does not match the sealed trajectory")
        labels = payload["labels_by_event_id"]
        if (set(labels) != {row["event_id_sha256"] for row in trajectory.rows} or
                any(type(v) is not int or v not in (0, 1) for v in labels.values())):
            raise AuditError("authorized label projection has missing, extra or nonbinary event labels")
        return labels
    except AuditError:
        raise
    except Exception as exc:
        raise AuditError("authorized label projection is unavailable or unreadable") from exc


def _evaluate_authorized_trajectory(*, trajectory: VerifiedTrajectory,
                                    stage_b_authorization: LabelProjectionAuthorization,
                                    projection_grant: Mapping[str, Any], audit_authorization: Mapping[str, Any],
                                    projection_path: str | Path) -> list[int]:
    """Audit-only adapter for the historical persisted-seal contract."""
    validate_projection_grant(trajectory=trajectory, stage_b_authorization=stage_b_authorization,
                              projection_grant=projection_grant, audit_authorization=audit_authorization)
    current = load_verified(status_path=trajectory.status_path, expected_identity=trajectory.expected_identity,
        registered_stream_manifest=trajectory.registered_stream_manifest,
        expected_event_count=trajectory.seal.expected_event_count, hard_cap=trajectory.hard_cap,
        stage_b_record=trajectory.stage_b_record)
    if (current.seal != trajectory.seal or current.journal_sha256 != trajectory.journal_sha256 or
            current.report_sha256 != trajectory.report_sha256):
        raise AuditError("sealed artifacts changed after campaign verification")
    labels = _load_authorized_projection(projection_path, trajectory=current, projection_grant=projection_grant)
    try:
        evaluated = evaluate_sealed_category(dataset_id="realiad", category=current.category,
            rows=current.rows, seal=current.seal, group_seals=current.groups,
            labels_by_event_id=labels, label_projection=stage_b_authorization)
        if evaluated["query_count"] != current.stage_b_record.get("query_count"):
            raise AuditError("Stage B query-count parity failed")
        reference_metrics = current.stage_b_record["metrics"]
        required = {"F-B", "K-B", "F-M", "K-M", "N-shadow", "STATIC-4+64"}
        if not required.issubset(reference_metrics) or not set(reference_metrics).issubset(evaluated["arm_metrics"]):
            raise AuditError("Stage B metric reference has missing or unknown arms")
        verify_metric_parity({arm: evaluated["arm_metrics"][arm] for arm in reference_metrics}, reference_metrics)
        return [labels[row["event_id_sha256"]] for row in current.rows]
    except AuditError:
        raise
    except Exception as exc:
        raise AuditError("existing Stage B evaluator rejected the authorized projection") from exc


def project_after_all_seals(*, trajectory: VerifiedTrajectory,
                            campaign: VerifiedCampaign,
                            audit_authorization: Mapping[str, Any], audit_source_commit: str,
                            reference_sha256: str, official_reference_path: str | Path, configuration_sha256: str,
                            stage_b_authorization: LabelProjectionAuthorization,
                            projection_grant: Mapping[str, Any], projection_path: str | Path) -> list[int]:
    """Verify the complete campaign and apply the audit-only Stage B adapter."""
    if not isinstance(campaign, VerifiedCampaign):
        raise AuditError("the complete campaign must be verified before label projection")
    all_verified = campaign.trajectories
    pairs = {(t.category, t.condition) for t in all_verified}
    categories = {t.category for t in all_verified}
    if len(all_verified) != 90 or len(categories) != 30 or pairs != {(c, p) for c in categories for p in ("P0", "P1", "P2")}:
        raise AuditError("all 90 distinct category-condition seals must verify before label projection")
    if not any(t is trajectory for t in all_verified):
        raise AuditError("trajectory is not a member of the verified campaign")
    validate_audit_authorization(audit_authorization, audit_source_commit=audit_source_commit,
        seal_hashes=[t.seal.durable_seal_sha256 for t in all_verified], reference_sha256=reference_sha256,
        source_plan_sha256=campaign.source_plan_sha256, configuration_sha256=configuration_sha256)
    try:
        if sha256_file(official_reference_path) != reference_sha256:
            raise AuditError("official Stage B metric reference hash differs from audit authorization")
        reference = json.loads(Path(official_reference_path).read_text(encoding="utf-8"))
    except AuditError:
        raise
    except Exception as exc:
        raise AuditError("official Stage B metric reference is unreadable") from exc
    records = validate_stage_b_summary(reference)
    if records[(trajectory.category, trajectory.condition)] != trajectory.stage_b_record:
        raise AuditError("Stage B record changed after source-bound campaign verification")
    return _evaluate_authorized_trajectory(trajectory=trajectory, stage_b_authorization=stage_b_authorization,
        projection_grant=projection_grant, audit_authorization=audit_authorization, projection_path=projection_path)
