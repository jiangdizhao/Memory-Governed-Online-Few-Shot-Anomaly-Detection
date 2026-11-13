"""Read-only reconstruction of sealed Stage A rows; no feedback files are read."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from experiments.calibration.final_benchmark.execution_support import ExecutionBoundaryError, verify_attempt_receipt, sha256_file
from experiments.calibration.final_benchmark.runner import StageASeal
from experiments.calibration.final_benchmark.sink import FsyncGroupSink
from .core import AuditError, validate_rows


def canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value)).hexdigest()


@dataclass(frozen=True)
class VerifiedTrajectory:
    category: str
    condition: str
    source_commit: str
    rows: tuple[Mapping[str, Any], ...]
    groups: tuple[Mapping[str, Any], ...]
    seal: StageASeal
    status_path: Path
    expected_identity: Mapping[str, Any]
    journal_sha256: str
    report_sha256: str
    registered_stream_manifest: Mapping[str, Any]
    hard_cap: int
    stage_b_record: Mapping[str, Any]


def load_verified(*, status_path: str | Path, expected_identity: Mapping[str, Any],
                  registered_stream_manifest: Mapping[str, Any], expected_event_count: int,
                  hard_cap: int, stage_b_record: Mapping[str, Any]) -> VerifiedTrajectory:
    """Require registered identities and independently verify every sealed envelope."""
    status_path = Path(status_path)
    identity = dict(expected_identity)
    if identity.get("dataset_id", "realiad") != "realiad" or identity.get("condition") not in {"P0", "P1", "P2"}:
        raise AuditError("invalid registered dataset or condition")
    if identity.get("source_commit") not in {"40d61c3e948bdb89088f70845d2562444066d6e3", "a3688e87ac165866fa4c4ddb3501f96258500534"}:
        raise AuditError("unregistered scientific source commit")
    if (not isinstance(registered_stream_manifest, Mapping) or
            digest(registered_stream_manifest) != identity.get("stream_manifest_sha256") or
            not isinstance(registered_stream_manifest.get("events"), list)):
        raise AuditError("registered stream manifest does not match the source-bound identity")
    try:
        FsyncGroupSink._reject_outcomes(registered_stream_manifest)
    except Exception as exc:
        raise AuditError("registered stream manifest contains outcome fields") from exc
    expected_rows = registered_stream_manifest["events"]
    if len(expected_rows) != expected_event_count:
        raise AuditError("registered stream event count differs from the schedule")
    try:
        receipt = verify_attempt_receipt(status_path, expected_identity=identity, expected_kind="principal_stage_a")
        s = receipt["seal"]
        if (receipt["seal_path"].name != "stage_a_category_seal.json" or
                s.get("schema") != "final-benchmark.stage-a-category-seal.v1" or s.get("dataset_id") != "realiad" or
                s.get("category") != identity.get("category") or s.get("run_condition") != identity.get("condition") or
                s.get("expected_event_count") != expected_event_count or s.get("completed_event_count") != expected_event_count or
                receipt["status"].get("expected_event_count") != expected_event_count):
            raise AuditError("seal category, condition or event count differs from registration")
        report_path = status_path.parent / "stage_a_category_report.json"
        if not report_path.is_file():
            raise AuditError("Stage A category report is missing")
        report = json.loads(report_path.read_text(encoding="utf-8"))
        seal_sha = receipt["status"]["completion_seal_sha256"]
        if (report.get("schema") != "final-benchmark.stage-a-category-report.v1" or
                report.get("durable_seal_sha256") != seal_sha or
                any(report.get(k) != s.get(k) for k in ("dataset_id", "category", "run_condition", "status",
                    "expected_event_count", "completed_event_count", "group_count", "score_rows_sha256", "group_seals_sha256")) or
                report.get("source_commit") != identity["source_commit"] or
                any(report.get(k) != identity.get(k) for k in ("identity_manifest_sha256", "role_manifest_sha256", "dataset_release_sha256")) or
                stage_b_record.get("category") != identity["category"] or
                stage_b_record.get("condition") != identity["condition"] or
                stage_b_record.get("source_commit") != identity["source_commit"] or
                stage_b_record.get("campaign_run_id") != identity.get("campaign_run_id") or
                stage_b_record.get("stage_a_category_seal_sha256") != seal_sha or
                stage_b_record.get("stage_a_score_rows_sha256", s["score_rows_sha256"]) != s["score_rows_sha256"] or
                stage_b_record.get("stage_a_group_seals_sha256", s["group_seals_sha256"]) != s["group_seals_sha256"] or
                not isinstance(stage_b_record.get("metrics"), Mapping) or
                any(isinstance(m, Mapping) and "n" in m and m["n"] != expected_event_count
                    for m in stage_b_record["metrics"].values())):
            raise AuditError("Stage A report, attempt and Stage B record do not bind the same seal")
        journal = status_path.parent / "stage_a_journal.jsonl"
        if not journal.is_file():
            raise AuditError("Stage A journal is missing")
        rows: list[Mapping[str, Any]] = []
        groups: list[Mapping[str, Any]] = []
        with journal.open("rb") as stream:
            header_line = stream.readline()
            if not header_line.endswith(b"\n"):
                raise AuditError("incomplete journal header")
            header = json.loads(header_line)
            FsyncGroupSink._reject_outcomes(header)
            expected_header = {"schema": "final-benchmark.stage-a-journal-header.v1", "dataset_id": "realiad",
                "category": identity["category"], "condition_id": identity["condition"],
                "source_commit": identity["source_commit"], "identity_manifest_sha256": identity["identity_manifest_sha256"],
                "role_manifest_sha256": identity["role_manifest_sha256"], "dataset_release_sha256": identity["dataset_release_sha256"]}
            if any(header.get(k) != v for k, v in expected_header.items()):
                raise AuditError("journal header differs from registered identity")
            for line in stream:
                if not line.endswith(b"\n"):
                    raise AuditError("incomplete journal envelope")
                env = json.loads(line)
                FsyncGroupSink._reject_outcomes(env)
                if env.get("schema") != "final-benchmark.stage-a-group.v1":
                    raise AuditError("unexpected Stage A journal envelope")
                local, group = env["score_rows"], env["group_seal"]
                if (not isinstance(local, list) or len(local) != 5 or group.get("group_index") != len(groups) or
                        group.get("start_position") != len(rows) or group.get("end_position_exclusive") != len(rows) + 5 or
                        group.get("status") != "SEALED_BEFORE_LABEL_DISCLOSURE" or
                        group.get("score_rows_sha256") != digest(local) or
                        group.get("event_order_sha256") != digest([r.get("event_id_sha256") for r in local]) or
                        any(r.get("source_group_sha256") != group.get("source_group_sha256") for r in local)):
                    raise AuditError("journal envelope differs from its group seal")
                rows.extend(local)
                groups.append({**group, "durable_group_receipt_sha256": hashlib.sha256(line).hexdigest()})
        if (len(rows) != expected_event_count or len(groups) != s.get("group_count") or
                digest(rows) != s.get("score_rows_sha256") or digest(groups) != s.get("group_seals_sha256")):
            raise AuditError("journal content does not match the completed category seal")
        for row, expected in zip(rows, expected_rows):
            if any(row.get(field) != expected.get(field) for field in
                   ("position", "event_id_sha256", "image_identity_sha256", "source_group_sha256", "camera_id")):
                raise AuditError("sealed event identity or view differs from the source-bound stream")
        accounting = validate_rows(rows, cap=hard_cap, expected_event_ids=[x.get("event_id_sha256") for x in expected_rows])
        if stage_b_record.get("query_count") != accounting["executed"]:
            raise AuditError("Stage B query counts differ from the sealed actions")
        seal = StageASeal("realiad", identity["category"], identity["condition"], expected_event_count,
            expected_event_count, len(groups), s["score_rows_sha256"], s["group_seals_sha256"],
            receipt["status"]["completion_seal_sha256"], "SEALED_COMPLETE")
        return VerifiedTrajectory(identity["category"], identity["condition"], identity["source_commit"],
            tuple(rows), tuple(groups), seal, status_path, identity, sha256_file(journal), sha256_file(report_path),
            dict(registered_stream_manifest), hard_cap, dict(stage_b_record))
    except AuditError:
        raise
    except ExecutionBoundaryError as exc:
        raise AuditError(f"Stage A attempt receipt rejected: {exc}") from exc
    except Exception as exc:
        raise AuditError("sealed Stage A input failed verification") from exc
