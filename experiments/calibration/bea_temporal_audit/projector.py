"""Authorized, in-memory Real-IAD image-level projection; mask values stay opaque."""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Mapping

from experiments.calibration.final_benchmark import identity
from experiments.calibration.final_benchmark.label_broker import _test_record_bounds
from .artifacts import VerifiedStageBGrant
from .campaign import VerifiedCampaign
from .core import AuditError
from .integrity import VerifiedTrajectory, load_verified
from .projection import validate_audit_authorization


class ProjectionAuthority:
    """A complete verified campaign plus both original and audit authorizations."""
    def __init__(self, *, campaign: VerifiedCampaign, stage_b: VerifiedStageBGrant,
                 audit_authorization: Mapping[str, Any], audit_source_commit: str,
                 summary_sha256: str, configuration_sha256: str):
        if not isinstance(campaign, VerifiedCampaign) or not isinstance(stage_b, VerifiedStageBGrant):
            raise AuditError("projection requires the complete verified campaign and original Stage B grant")
        validate_audit_authorization(audit_authorization, audit_source_commit=audit_source_commit,
            seal_hashes=[t.seal.durable_seal_sha256 for t in campaign.trajectories],
            reference_sha256=summary_sha256, source_plan_sha256=campaign.source_plan_sha256,
            configuration_sha256=configuration_sha256, stage_b_authorization_sha256=stage_b.sha256)
        if len(campaign.trajectories) != 90 or len(stage_b.entries) != 90:
            raise AuditError("projection requires all 90 verified Stage A trajectories")
        for t in campaign.trajectories:
            entry = stage_b.entries.get((t.category, t.condition))
            if (entry is None or entry.get("source_commit") != t.source_commit or
                    entry.get("campaign_run_id") != t.expected_identity.get("campaign_run_id") or
                    entry.get("stage_a_category_seal_sha256") != t.seal.durable_seal_sha256 or
                    entry.get("stage_a_journal_sha256") != t.journal_sha256 or
                    entry.get("annotation_sha256") != t.expected_identity.get("annotation_sha256") or
                    entry.get("stream_manifest_sha256") != t.expected_identity.get("stream_manifest_sha256") or
                    entry.get("role_manifest_sha256") != t.expected_identity.get("role_manifest_sha256") or
                    entry.get("stage_a_authorization_file_sha256") != t.expected_identity.get("authorization_file_sha256")):
                raise AuditError("original Stage B grant differs from a verified Stage A trajectory")
        self.campaign = campaign
        self.stage_b = stage_b


def _decode_annotation(data: bytes, *, trajectory: VerifiedTrajectory, annotation_sha256: str) -> dict[str, int]:
    """Pure decoder: only image identity, normal class and image-level anomaly class."""
    from experiments.calibration import r7_shuffled_correction_4class as r7
    if hashlib.sha256(data).hexdigest() != annotation_sha256:
        raise AuditError("annotation SHA-256 differs from its registered identity")
    try:
        top = identity._identity_object_fields(data, 0, len(data))
        if "meta" not in top or "test" not in top:
            raise AuditError("registered annotation lacks metadata or TEST records")
        meta = identity._identity_object_fields(data, *top["meta"])
        if "normal_class" not in meta:
            raise AuditError("registered annotation lacks normal-class metadata")
        normal = r7._decode_field(data, meta["normal_class"])
        if not isinstance(normal, str) or not normal:
            raise AuditError("registered normal-class metadata is invalid")
        bounds = _test_record_bounds(data, top["test"])
        stream = trajectory.registered_stream_manifest
        records = stream["events"]
        if len(bounds) != len(records) or len(records) != len(trajectory.rows):
            raise AuditError("annotation TEST count differs from the sealed stream")
        originals = [r.get("original_position") for r in records]
        if any(type(x) is not int for x in originals) or sorted(originals) != list(range(len(records))):
            raise AuditError("registered original TEST positions are missing or duplicated")
        labels: dict[str, int] = {}
        seen_images: set[str] = set()
        for row, event in zip(records, trajectory.rows):
            fields = identity._identity_object_fields(data, *bounds[row["original_position"]])
            if "image_path" not in fields or "anomaly_class" not in fields:
                raise AuditError("registered TEST record lacks its image identity or image-level class")
            image_key = r7._decode_field(data, fields["image_path"])
            anomaly_class = r7._decode_field(data, fields["anomaly_class"])
            if not isinstance(image_key, str) or not image_key or not isinstance(anomaly_class, str) or not anomaly_class:
                raise AuditError("registered image identity or class is not text")
            group, camera = identity._realiad_identity_key(trajectory.category, image_key)
            image_sha = identity._image_identity_digest(image_key)
            event_sha = identity._domain_hash("Gate0-event-v1", str(stream["dataset_release_sha256"]),
                trajectory.category, str(stream["stream_seed"]), group, str(camera), image_key, str(camera))
            if (image_sha != row.get("image_identity_sha256") or image_sha in seen_images or
                    camera != row.get("camera_id") or camera != row.get("within_group_ordinal") or
                    identity._source_group_digest(group) != row.get("source_group_sha256") or
                    event_sha != row.get("event_id_sha256") or event_sha != event.get("event_id_sha256")):
                raise AuditError("annotation image identity, view or event differs from the sealed stream")
            seen_images.add(image_sha)
            if event_sha in labels:
                raise AuditError("duplicate event identity in annotation projection")
            labels[event_sha] = int(anomaly_class != normal)
        if len(labels) != len(trajectory.rows):
            raise AuditError("annotation projection does not cover exactly the sealed stream")
        return labels
    except AuditError:
        raise
    except Exception as exc:
        raise AuditError("registered annotation projection failed") from exc


def project_annotation(*, authority: ProjectionAuthority, trajectory: VerifiedTrajectory,
                       annotation_path: str | Path) -> dict[str, int]:
    """Read outcomes only after the complete campaign and both grants pass."""
    if (not isinstance(authority, ProjectionAuthority) or
            not any(t is trajectory for t in authority.campaign.trajectories)):
        raise AuditError("annotation access requires an authorized complete campaign")
    entry = authority.stage_b.entries[(trajectory.category, trajectory.condition)]
    current = load_verified(status_path=trajectory.status_path, expected_identity=trajectory.expected_identity,
        registered_stream_manifest=trajectory.registered_stream_manifest,
        expected_event_count=trajectory.seal.expected_event_count, hard_cap=trajectory.hard_cap,
        stage_b_record=trajectory.stage_b_record)
    if (current.seal != trajectory.seal or current.journal_sha256 != trajectory.journal_sha256 or
            current.report_sha256 != trajectory.report_sha256 or current.journal_sha256 != entry["stage_a_journal_sha256"]):
        raise AuditError("sealed artifacts changed before annotation projection")
    try:
        data = Path(annotation_path).read_bytes()
    except OSError as exc:
        raise AuditError("registered annotation file is unavailable") from exc
    return _decode_annotation(data, trajectory=current, annotation_sha256=entry["annotation_sha256"])
