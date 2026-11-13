"""Adapters for the existing cross-campaign grant and embedded stream manifests."""
from __future__ import annotations
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from experiments.calibration.final_benchmark.evaluation import LabelProjectionAuthorization
from experiments.calibration.final_benchmark.execution_support import sha256_file
from experiments.calibration.final_benchmark.launch import registered_trajectory
from experiments.calibration.final_benchmark.sink import FsyncGroupSink
from .campaign import CATEGORIES, IDENTITY, NEW, OLD, RELEASE
from .core import AuditError
from .integrity import digest
from .reference import validate_stage_b_metadata, validate_stage_b_summary

HEX = re.compile(r"[0-9a-f]{64}\Z")
SEEDS = {"P0": (0, 0), "P1": (0, 11), "P2": (1, 0)}


@dataclass(frozen=True)
class VerifiedStageBGrant:
    grant: Mapping[str, Any]
    sha256: str
    entries: Mapping[tuple[str, str], Mapping[str, Any]]


def validate_combined_grant(*, grant: Mapping[str, Any], grant_sha256: str,
                            summary: Mapping[str, Any], plans: Mapping[str, tuple[Mapping[str, Any], str]],
                            repository: str | Path, metadata_only: bool = False) -> VerifiedStageBGrant:
    """Validate the existing authorization; no category grant is written or issued."""
    records = (validate_stage_b_metadata if metadata_only else validate_stage_b_summary)(summary)
    root = Path(repository).resolve()
    source_dir = root / "experiments/calibration/final_benchmark"
    if (not isinstance(grant, Mapping) or grant.get("schema") != "realiad.stage-b-cross-campaign-authorization.v1" or
            grant.get("status") != "AUTHORIZED_POST_STAGE_A_LABEL_PROJECTION" or
            grant.get("scope") != "stage_b_category_evaluation" or grant.get("dataset_id") != "realiad" or
            grant.get("registered_category_count") != 30 or grant.get("registered_trajectory_count") != 90 or
            grant.get("registered_test_image_events") != 343755 or grant.get("test_mask_access") is not False or
            grant.get("stage_a_reexecution") is not False or grant.get("model_loading") is not False or
            grant.get("evaluation_code_commit") != NEW or
            grant.get("evaluation_file_sha256") != sha256_file(source_dir / "evaluation.py") or
            grant.get("metrics_file_sha256") != sha256_file(source_dir / "metrics.py") or
            not HEX.fullmatch(str(grant_sha256)) or summary.get("stage_b_authorization_sha256") != grant_sha256 or
            set(plans) != {OLD, NEW}):
        raise AuditError("cross-campaign Stage B authorization metadata is invalid")
    provenance = {p[0]["source_identity"]["campaign_run_id"]: source for source, p in plans.items()}
    if grant.get("scientific_execution_provenance") != provenance:
        raise AuditError("Stage B grant does not bind the two historical campaign sources")
    entries = grant.get("authorizations")
    if not isinstance(entries, list) or len(entries) != 90:
        raise AuditError("Stage B grant must contain exactly 90 category-condition identities")
    result = {}
    for entry in entries:
        if not isinstance(entry, Mapping):
            raise AuditError("Stage B grant entry is malformed")
        pair = (entry.get("category"), entry.get("condition"))
        if pair not in records or pair in result:
            raise AuditError("Stage B grant coverage has a duplicate or unknown trajectory")
        record = records[pair]
        source = record["source_commit"]
        plan, plan_sha = plans[source]
        row = registered_trajectory(plan, *pair)
        expected = {"source_commit": source, "campaign_run_id": plan["source_identity"]["campaign_run_id"],
            "category": pair[0], "condition": pair[1], "execution_plan_file_sha256": plan_sha,
            "role_manifest_sha256": row["role_manifest_sha256"], "stream_manifest_sha256": row["stream_manifest_sha256"],
            "expected_event_count": row["test_event_count"], "expected_source_groups": row["source_group_count"],
            "stage_a_category_seal_sha256": record["stage_a_category_seal_sha256"],
            "annotation_sha256": record["annotation_sha256"]}
        if (any(entry.get(k) != v for k, v in expected.items()) or
                any(not HEX.fullmatch(str(entry.get(k, ""))) for k in
                    ("annotation_sha256", "stage_a_authorization_file_sha256", "stage_a_category_seal_sha256", "stage_a_journal_sha256"))):
            raise AuditError("Stage B grant entry differs from its plan or scientific record")
        status_path = Path(str(entry.get("stage_a_status_path", "")))
        expected_parent = root / "outputs/calibration/final_benchmark" / expected["campaign_run_id"] / "stage_a" / pair[0] / pair[1]
        if (not status_path.is_absolute() or status_path.name != "attempt_status.json" or
                status_path.parent.parent.resolve() != expected_parent.resolve() or
                not re.fullmatch(r"attempt_[0-9]{2,}", status_path.parent.name)):
            raise AuditError("Stage B grant points outside its registered attempt namespace")
        result[pair] = dict(entry)
    if set(result) != set(records):
        raise AuditError("Stage B grant does not cover the complete reference")
    return VerifiedStageBGrant(dict(grant), grant_sha256, result)


def category_projection_authorization(verified: VerifiedStageBGrant, pair: tuple[str, str]) -> LabelProjectionAuthorization:
    """Construct an in-memory view only after the combined grant has been verified."""
    if not isinstance(verified, VerifiedStageBGrant) or pair not in verified.entries:
        raise AuditError("category authorization requires the verified combined Stage B grant")
    grant, entry = verified.grant, verified.entries[pair]
    return LabelProjectionAuthorization(status=grant["status"], scope=grant["scope"],
        source_sha256=entry["annotation_sha256"], dataset_id=grant["dataset_id"],
        category=entry["category"], stage_a_seal_sha256=entry["stage_a_category_seal_sha256"])


def validate_embedded_stream(*, bundle: Mapping[str, Any], plan_row: Mapping[str, Any],
                             release_sha256: str = RELEASE) -> Mapping[str, Any]:
    """Check the registered manifest in place, including original TEST positions."""
    category, condition = plan_row.get("category"), plan_row.get("condition")
    stream = bundle.get("stream_manifest")
    if (condition not in SEEDS or not isinstance(stream, Mapping) or
            bundle.get("schema") != "final-benchmark.category-condition-identity.v1" or
            bundle.get("category") != category or bundle.get("condition_id") != condition or
            bundle.get("dataset_id") != "realiad" or
            bundle.get("stream_manifest_sha256") != plan_row.get("stream_manifest_sha256") or
            digest(stream) != plan_row.get("stream_manifest_sha256") or
            stream.get("schema") != "final-benchmark.stream-manifest.v1" or
            stream.get("dataset_id") != "realiad" or stream.get("category") != category or
            stream.get("condition_id") != condition or stream.get("dataset_release_sha256") != release_sha256 or
            (stream.get("support_seed"), stream.get("stream_seed")) != SEEDS[condition] or
            (bundle.get("support_seed"), bundle.get("stream_seed")) != SEEDS[condition] or
            stream.get("event_count") != plan_row.get("test_event_count") or
            not isinstance(stream.get("events"), list) or len(stream["events"]) != plan_row.get("test_event_count")):
        raise AuditError("embedded stream manifest differs from the registered plan")
    try:
        FsyncGroupSink._reject_outcomes(stream)
        events = stream["events"]
        if len(events) % 5 or [r.get("position") for r in events] != list(range(len(events))):
            raise AuditError("embedded stream event positions are not contiguous")
        original = [r.get("original_position") for r in events]
        if any(type(x) is not int for x in original) or sorted(original) != list(range(len(events))):
            raise AuditError("embedded stream original positions are missing or duplicated")
        ids = [r.get("event_id_sha256") for r in events]
        images = [r.get("image_identity_sha256") for r in events]
        if any(not HEX.fullmatch(str(x)) for x in ids + images) or len(set(ids)) != len(ids) or len(set(images)) != len(images):
            raise AuditError("embedded stream has duplicate or malformed event identities")
        groups = set()
        for start in range(0, len(events), 5):
            part = events[start:start+5]
            group = part[0].get("source_group_sha256")
            if (not HEX.fullmatch(str(group)) or group in groups or
                    any(r.get("source_group_sha256") != group for r in part) or
                    [r.get("camera_id") for r in part] != [1, 2, 3, 4, 5] or
                    [r.get("within_group_ordinal") for r in part] != [1, 2, 3, 4, 5]):
                raise AuditError("embedded stream five-view group order is invalid")
            groups.add(group)
        if len(groups) != plan_row.get("source_group_count"):
            raise AuditError("embedded stream group count differs from the plan")
    except AuditError:
        raise
    except Exception as exc:
        raise AuditError("embedded stream is malformed or contains outcome fields") from exc
    return stream
