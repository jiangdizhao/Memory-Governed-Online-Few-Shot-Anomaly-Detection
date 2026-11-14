"""Post-seal VisA P0 label projection and Stage B evaluation."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping

from . import identity
from .evaluation import LabelProjectionAuthorization, evaluate_sealed_category
from .execution_support import ExecutionBoundaryError, sha256_file, verify_attempt_receipt
from .runner import StageASeal


class VisaStageBError(RuntimeError):
    pass


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")


def _write_exclusive(path: Path, value: Any) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = _canonical(value) + b"\n"
    try:
        with path.open("xb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
    except FileExistsError:
        raise VisaStageBError("Stage B output is non-overwriting and already exists") from None
    return hashlib.sha256(payload).hexdigest()


def _load_stage_a(attempt_root: Path, expected_identity: Mapping[str, Any],
                  stream_records: list[Mapping[str, Any]], candidate: Mapping[str, Any]):
    try:
        verified = verify_attempt_receipt(attempt_root / "attempt_status.json",
            expected_identity=expected_identity, expected_kind="principal_stage_a")
    except ExecutionBoundaryError as exc:
        raise VisaStageBError("VisA Stage B requires a matching durable SEALED_COMPLETE Stage A receipt") from exc
    status = verified["status"]
    seal_obj = verified["seal"]
    journal_path = attempt_root / "stage_a_journal.jsonl"
    if not journal_path.is_file():
        raise VisaStageBError("sealed Stage A journal is missing")
    rows: list[dict[str, Any]] = []
    unit_seals: list[dict[str, Any]] = []
    with journal_path.open("rb") as stream:
        header_line = stream.readline()
        try:
            header = json.loads(header_line)
        except Exception as exc:
            raise VisaStageBError("Stage A journal header is malformed") from exc
        if (header.get("schema") != "final-benchmark.stage-a-journal-header.v1" or
                any(header.get(key) != value for key, value in expected_identity.items())):
            raise VisaStageBError("Stage A journal identity differs from its authorization")
        for line in stream:
            try:
                envelope = json.loads(line)
            except Exception as exc:
                raise VisaStageBError("Stage A journal contains a malformed durable unit") from exc
            if envelope.get("schema") != "final-benchmark.stage-a-atomic-unit.v1":
                raise VisaStageBError("VisA journal contains a non-unit or unregistered envelope")
            unit_seal = envelope.get("unit_seal")
            score_rows = envelope.get("score_rows")
            if not isinstance(unit_seal, Mapping) or not isinstance(score_rows, list) or not score_rows:
                raise VisaStageBError("VisA journal unit envelope is incomplete")
            digest = hashlib.sha256(line).hexdigest()
            unit_seals.append({**dict(unit_seal), "durable_group_receipt_sha256": digest})
            rows.extend(dict(row) for row in score_rows)
    if len(rows) != len(stream_records) or status.get("completed_event_count") != len(stream_records):
        raise VisaStageBError("Stage A event count differs from the registered P0 stream")
    candidate_sha = candidate.get("candidate_sha256")
    approval_sha = seal_obj.get("unit_approval_sha256")
    if (seal_obj.get("unit_candidate_sha256") != candidate_sha or
            not isinstance(approval_sha, str) or len(approval_sha) != 64):
        raise VisaStageBError("Stage A completion seal differs from the candidate-bound VisA approval")
    for position, (row, registered) in enumerate(zip(rows, stream_records)):
        if (row.get("position") != position or
                row.get("event_id_sha256") != registered.get("event_id_sha256") or
                row.get("image_identity_sha256") != registered.get("image_identity_sha256") or
                row.get("source_group_sha256") != registered.get("source_group_sha256") or
                row.get("atomic_unit_sha256") != registered.get("atomic_unit_sha256") or
                row.get("atomic_unit_kind") != registered.get("atomic_unit_kind")):
            raise VisaStageBError("Stage A journal event identity/order differs from the registered P0 stream")
    # Verify every private query receipt against the immutable, sealed action rows.
    expected_queries: dict[tuple[str, int], tuple[str, str]] = {}
    for row in rows:
        for arm, action in row.get("query_actions", {}).items():
            if action.get("executed"):
                expected_queries[(str(arm), int(row["position"]))] = (
                    str(row["event_id_sha256"]), str(action.get("query_receipt_sha256")))
    private_path = attempt_root / "private_feedback.jsonl"
    disclosed: dict[tuple[str, int], dict[str, Any]] = {}
    if not private_path.is_file():
        if expected_queries:
            raise VisaStageBError("Stage A query ledger is missing despite executed query actions")
    else:
        with private_path.open("r", encoding="utf-8") as feedback:
            for line in feedback:
                item = json.loads(line)
                key = (str(item.get("arm_id")), int(item.get("position", -1)))
                if key in disclosed or key not in expected_queries:
                    raise VisaStageBError("private VisA query ledger has an extra or duplicate disclosure")
                if (item.get("event_id_sha256") != expected_queries[key][0] or
                        item.get("query_receipt_sha256") != expected_queries[key][1] or
                        item.get("schema") != "final-benchmark.visa-private-feedback.v1"):
                    raise VisaStageBError("private VisA disclosure does not match a durable executed receipt")
                disclosed[key] = item
    if set(disclosed) != set(expected_queries):
        raise VisaStageBError("Stage A private query ledger does not cover all executed query receipts")
    seal = StageASeal(
        dataset_id=str(seal_obj["dataset_id"]), category=str(seal_obj["category"]),
        run_condition=str(seal_obj["run_condition"]),
        expected_event_count=int(seal_obj["expected_event_count"]),
        completed_event_count=int(seal_obj["completed_event_count"]),
        group_count=int(seal_obj["group_count"]),
        score_rows_sha256=str(seal_obj["score_rows_sha256"]),
        group_seals_sha256=str(seal_obj["group_seals_sha256"]),
        durable_seal_sha256=sha256_file(verified["seal_path"]),
        status=str(seal_obj["status"]),
        atomic_unit_kind=str(seal_obj["atomic_unit_kind"]),
        unit_candidate_sha256=str(seal_obj["unit_candidate_sha256"]),
        unit_approval_sha256=str(seal_obj["unit_approval_sha256"]))
    return status, seal_obj, seal, rows, unit_seals, disclosed


def _project_all_test_labels(*, dataset_root: Path, source_csv_sha256: str,
                             category: str, records: list[Mapping[str, Any]],
                             rows: list[Mapping[str, Any]]) -> dict[str, int]:
    """Decode official image-level labels only after Stage A receipt/journal checks."""
    source_path = dataset_root / identity.VISA_1CLS_SOURCE
    actual_sha, _ = identity._safe_file_hash(source_path, identity="visa-stage-b-source")
    if actual_sha != source_csv_sha256:
        raise VisaStageBError("official VisA CSV source hash differs from the Stage A identity registration")
    expected_by_original = {int(record["original_position"]): (record, row)
                            for record, row in zip(records, rows)}
    if len(expected_by_original) != len(records):
        raise VisaStageBError("registered VisA original positions are not unique")
    labels: dict[str, int] = {}
    with source_path.open("rb") as stream:
        import csv
        header = next(csv.reader([stream.readline().decode("utf-8-sig")]))
        for original_position, line in enumerate(stream):
            if not line.strip():
                continue
            expected = expected_by_original.get(original_position)
            if expected is None:
                continue
            registered, score_row = expected
            projected = identity._parse_csv_projected_line(
                line, header, train_label_allowed=False, queried_test_label_allowed=True)
            image_key = identity._nfc_key(projected.get("image", ""))
            image_id = identity._image_identity_digest(image_key)
            if (projected.get("split") != "test" or projected.get("object") != category or
                    image_id != registered.get("image_identity_sha256") or
                    projected.get("label") not in {"normal", "anomaly"}):
                raise VisaStageBError("official post-seal TEST row differs from its sealed image identity")
            labels[str(score_row["event_id_sha256"])] = int(projected["label"] == "anomaly")
    if set(labels) != {str(row["event_id_sha256"]) for row in rows}:
        raise VisaStageBError("official post-seal label projection does not cover the exact event stream")
    return labels


def evaluate_visa_attempt(*, attempt_root: str | Path, dataset_root: str | Path,
                          identity_manifest: Mapping[str, Any],
                          stream_manifest: Mapping[str, Any],
                          candidate: Mapping[str, Any], approval: Mapping[str, Any],
                          authorization: Mapping[str, Any],
                          expected_attempt_identity: Mapping[str, Any],
                          output_directory: str | Path) -> dict[str, Any]:
    attempt, root = Path(attempt_root), Path(dataset_root).resolve(strict=True)
    category = str(stream_manifest.get("category", ""))
    records = list(stream_manifest.get("stream_order", ()))
    candidate_sha = str(candidate.get("candidate_sha256", ""))
    approval_sha = hashlib.sha256(_canonical(approval)).hexdigest()
    expected_identity = dict(expected_attempt_identity)
    if (expected_identity.get("source_commit") != stream_manifest.get("source_commit") or
            expected_identity.get("identity_manifest_sha256") != stream_manifest.get("identity_manifest_sha256") or
            expected_identity.get("role_manifest_sha256") != stream_manifest.get("role_manifest_sha256") or
            expected_identity.get("dataset_release_sha256") != stream_manifest.get("dataset_release_sha256") or
            expected_identity.get("unit_candidate_sha256") != candidate_sha or
            expected_identity.get("unit_approval_sha256") != approval_sha or
            expected_identity.get("stream_manifest_sha256") != stream_manifest.get("stream_manifest_sha256")):
        raise VisaStageBError("VisA source-bound execution identity differs from its stream and authorization")
    status, seal_obj, seal, rows, unit_seals, disclosed = _load_stage_a(
        attempt, expected_identity, records, candidate)
    source_rows = [row for row in identity_manifest.get("source_artifacts", ())
                   if row.get("artifact_id") == "official-split:1cls.csv"]
    if len(source_rows) != 1:
        raise VisaStageBError("VisA identity registration lacks a unique official split-table hash")
    source_sha = str(source_rows[0]["sha256"])
    if (authorization.get("schema") != "final-benchmark.visa-stage-b-label-projection-authorization.v1" or
            authorization.get("status") != "RESEARCHER_AUTHORIZED_POST_SEAL_VISA_STAGE_B" or
            authorization.get("scope") != f"{category}_P0_post_seal_only" or
            authorization.get("source_commit") != expected_identity["source_commit"] or
            authorization.get("dataset_id") != "visa_1cls" or
            authorization.get("category") != category or
            authorization.get("stage_a_seal_sha256") != seal.durable_seal_sha256 or
            authorization.get("unit_candidate_sha256") != candidate_sha or
            authorization.get("unit_approval_sha256") != approval_sha or
            authorization.get("source_csv_sha256") != source_sha or
            authorization.get("pixel_masks_allowed") is not False):
        raise VisaStageBError("post-seal label projection authorization is not bound to this sealed category")
    test_records = [row for row in identity_manifest["identity_records"]
                    if row.get("category") == category and row.get("split") == "test"]
    if len(test_records) != len(rows):
        raise VisaStageBError("VisA Stage A event count differs from registered image identities")
    labels = _project_all_test_labels(dataset_root=root, source_csv_sha256=source_sha,
                                      category=category, records=records,
                                      rows=rows)
    for (arm, position), item in disclosed.items():
        event_id = str(rows[position]["event_id_sha256"])
        if item.get("label") != labels[event_id]:
            raise VisaStageBError("queried VisA label disclosure differs from its official post-seal image label")
    projection = LabelProjectionAuthorization(
        status="AUTHORIZED_POST_STAGE_A_LABEL_PROJECTION",
        scope="stage_b_category_evaluation", source_sha256=source_sha,
        dataset_id="visa_1cls", category=category,
        stage_a_seal_sha256=seal.durable_seal_sha256,
        unit_candidate_sha256=candidate_sha,
        unit_approval_sha256=approval_sha)
    result = evaluate_sealed_category(dataset_id="visa_1cls", category=category,
        rows=rows, seal=seal, group_seals=unit_seals,
        labels_by_event_id=labels, label_projection=projection)
    result.update({"schema": "final-benchmark.visa-stage-b-result.v1",
        "source_commit": expected_identity["source_commit"],
        "stage_a_attempt_status_sha256": sha256_file(attempt / "attempt_status.json"),
        "stage_a_seal_sha256": seal.durable_seal_sha256,
        "stage_a_journal_sha256": sha256_file(attempt / "stage_a_journal.jsonl"),
        "unit_candidate_sha256": candidate_sha, "unit_approval_sha256": approval_sha,
        "official_split_csv_sha256": source_sha,
        "stage_b_authorization_sha256": hashlib.sha256(_canonical(authorization)).hexdigest(),
        "outcome_access_phase": "STAGE_B_AFTER_VALID_STAGE_A_SEAL",
        "pixel_masks_accessed": False,
        "physical_independence_claim": "NOT_ESTABLISHED"})
    target = Path(output_directory)
    if target.exists():
        raise VisaStageBError("VisA Stage B result directory already exists; refusing overwrite")
    target.mkdir(parents=True, exist_ok=False)
    _write_exclusive(target / "stage_b_metrics.json", result)
    return result
