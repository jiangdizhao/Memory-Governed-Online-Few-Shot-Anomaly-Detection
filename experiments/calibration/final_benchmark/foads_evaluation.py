"""CPU-only post-seal evaluation for the adapted FOADS/K-NG baseline.

TEST outcomes are read only after a complete Stage A status, seal, journal,
registration, and category-bound authorization have all been verified.
"""
from __future__ import annotations

import hashlib
import json
import math
import numbers
import os
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

import numpy as np

from .execution_support import canonical_json, sha256_file
from .foads_execution import FOADSExecutionError, verify_foads_category_seal
from .launch import CONFIG_SCHEMA, object_sha256, read_json, validate_config
from .metrics import _auc_ap


class FOADSEvaluationError(RuntimeError):
    """Fail-closed error for post-seal FOADS evaluation."""


FOADS_STAGE_B_AUTH_SCHEMA = "final-benchmark.foads-stage-b-label-projection-authorization.v1"
FOADS_STAGE_B_RESULT_SCHEMA = "final-benchmark.foads-stage-b-evaluation.v1"
FOADS_STAGE_B_RECEIPT_SCHEMA = "final-benchmark.foads-stage-b-evaluation-receipt.v1"
FOADS_STAGE_B_AGGREGATE_SCHEMA = "final-benchmark.foads-stage-b-macro.v1"
FOADS_BASELINE = "FOADS-KNG-PAPER-CONSISTENT-PROTOCOL-ADAPTED"
_HEX40 = re.compile(r"[0-9a-f]{40}\Z")
_HEX64 = re.compile(r"[0-9a-f]{64}\Z")


def _read_object(path: str | Path, description: str) -> dict[str, Any]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception as exc:
        raise FOADSEvaluationError(f"{description} is missing or unreadable") from None
    if not isinstance(value, dict):
        raise FOADSEvaluationError(f"{description} must be a JSON object")
    return value


def _read_jsonl(path: Path, description: str) -> list[dict[str, Any]]:
    try:
        rows: list[dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as stream:
            for line in stream:
                if not line.endswith("\n"):
                    raise FOADSEvaluationError(f"{description} has a truncated record")
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise FOADSEvaluationError(f"{description} contains a non-object record")
                rows.append(value)
        return rows
    except FOADSEvaluationError:
        raise
    except Exception:
        raise FOADSEvaluationError(f"{description} is missing or unreadable") from None


def _is_sha(value: Any) -> bool:
    return isinstance(value, str) and _HEX64.fullmatch(value) is not None


def _verify_status_and_seal(attempt_root: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], str]:
    status = _read_object(attempt_root / "attempt_status.json", "FOADS attempt status")
    identity = status.get("identity")
    if (status.get("schema") != "final-benchmark.attempt-status.v1" or
            status.get("execution_kind") != "foads_kng_p0" or
            status.get("status") != "SEALED_COMPLETE" or
            not isinstance(identity, dict) or
            identity.get("category") is None or identity.get("condition") != "P0" or
            identity.get("dataset_release_sha256") is None or
            not _HEX40.fullmatch(str(identity.get("source_commit", ""))) or
            status.get("completed_event_count") != status.get("expected_event_count") or
            isinstance(status.get("expected_event_count"), bool) or
            not isinstance(status.get("expected_event_count"), int) or
            status["expected_event_count"] <= 0 or
            status.get("completion_seal_path") != "foads_category_seal.json"):
        raise FOADSEvaluationError("FOADS attempt status is not a complete registered P0 attempt")
    try:
        seal = verify_foads_category_seal(attempt_root, expected_identity=identity)
    except (FOADSExecutionError, OSError, ValueError) as exc:
        raise FOADSEvaluationError(f"FOADS completion seal or journals failed verification: {exc}") from exc
    seal_path = attempt_root / "foads_category_seal.json"
    seal_file_sha = sha256_file(seal_path)
    if (seal.get("baseline") != FOADS_BASELINE or
            seal.get("scored_images") != status["expected_event_count"] or
            seal.get("expected_event_count") != status["expected_event_count"] or
            seal.get("paid_labels") != 0 or seal.get("test_outcome_or_mask_access") is not False):
        raise FOADSEvaluationError("FOADS seal does not establish complete zero-label Stage A")
    return status, identity, seal, seal_file_sha


def _verify_category_registration(attempt_root: Path, identity: Mapping[str, Any],
                                  seal: Mapping[str, Any], category_bundle_path: str | Path
                                  ) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], str]:
    bundle_path = Path(category_bundle_path)
    if not bundle_path.is_file() or sha256_file(bundle_path) != identity.get("category_bundle_file_sha256"):
        raise FOADSEvaluationError("category bundle file does not match the sealed Stage A identity")
    bundle = _read_object(bundle_path, "registered category bundle")
    if (bundle.get("schema") != "final-benchmark.category-condition-identity.v1" or
            bundle.get("dataset_id") != "realiad" or
            bundle.get("category") != identity.get("category") or
            bundle.get("condition_id") != "P0"):
        raise FOADSEvaluationError("category bundle does not identify the sealed Real-IAD P0 trajectory")
    release = bundle.get("release_identity")
    role = bundle.get("role_manifest")
    stream = bundle.get("stream_manifest")
    if not isinstance(release, dict) or not isinstance(role, dict) or not isinstance(stream, dict):
        raise FOADSEvaluationError("category bundle lacks its registered release, role, or stream")
    events = stream.get("events")
    selected = role.get("selected")
    annotation_sha = role.get("annotation_sha256")
    category = str(identity["category"])
    if (release.get("dataset_release_sha256") != identity.get("dataset_release_sha256") or
            role.get("category") != category or role.get("condition_id") != "P0" or
            role.get("dataset_id") != "realiad" or
            stream.get("category") != category or stream.get("condition_id") != "P0" or
            stream.get("dataset_id") != "realiad" or
            object_sha256(role) != identity.get("role_manifest_sha256") or
            object_sha256(stream) != identity.get("stream_manifest_sha256") or
            not _is_sha(annotation_sha) or
            not isinstance(selected, list) or len(selected) != 68 or
            sum(row.get("role") == "SUPPORT" for row in selected if isinstance(row, dict)) != 4 or
            sum(row.get("role") == "C64" for row in selected if isinstance(row, dict)) != 64 or
            not isinstance(events, list) or len(events) != seal.get("scored_images") or
            stream.get("event_count") != len(events) or
            bundle.get("stream_seed") != stream.get("stream_seed") or
            role.get("stream_seed") != stream.get("stream_seed") or
            role.get("support_seed") != bundle.get("support_seed")):
        raise FOADSEvaluationError("category release, roles, or stream differ from the sealed identity")
    if identity.get("foads_registration_projection_sha256") is None:
        raise FOADSEvaluationError("sealed identity lacks its FOADS registration projection digest")

    scores = _read_jsonl(attempt_root / "foads_prediction_journal.jsonl", "FOADS prediction journal")
    updates = _read_jsonl(attempt_root / "foads_update_journal.jsonl", "FOADS update journal")
    if len(scores) != len(events):
        raise FOADSEvaluationError("prediction journal length differs from the registered stream")
    group_ids: list[str] = []
    original_positions: list[int] = []
    group_events: list[list[dict[str, Any]]] = []
    for position, (event, score) in enumerate(zip(events, scores)):
        if not isinstance(event, dict):
            raise FOADSEvaluationError("registered stream contains a malformed event")
        for field in ("event_id_sha256", "image_identity_sha256", "source_group_sha256"):
            if not _is_sha(event.get(field)):
                raise FOADSEvaluationError("registered stream contains a malformed identity digest")
        original_position = event.get("original_position")
        camera = event.get("camera_id")
        ordinal = event.get("within_group_ordinal")
        if (event.get("position") != position or
                isinstance(original_position, bool) or not isinstance(original_position, int) or original_position < 0 or
                isinstance(camera, bool) or not isinstance(camera, int) or camera not in (1, 2, 3, 4, 5) or
                isinstance(ordinal, bool) or not isinstance(ordinal, int) or ordinal not in (1, 2, 3, 4, 5) or
                score.get("position") != position or
                score.get("event_id_sha256") != event.get("event_id_sha256") or
                score.get("image_identity_sha256") != event.get("image_identity_sha256") or
                score.get("source_group_sha256") != event.get("source_group_sha256") or
                score.get("model_update_index") != position // 10):
            raise FOADSEvaluationError("prediction journal does not match registered event order and identity")
        raw_score = score.get("raw_image_score")
        if (isinstance(raw_score, bool) or not isinstance(raw_score, numbers.Real) or
                not math.isfinite(float(raw_score))):
            raise FOADSEvaluationError("prediction journal contains an invalid raw image score")
        original_positions.append(original_position)
        if not group_ids or group_ids[-1] != event["source_group_sha256"]:
            group_ids.append(str(event["source_group_sha256"]))
            group_events.append([])
        group_events[-1].append(event)
    if (len(set(original_positions)) != len(original_positions) or
            any(len(group) != 5 or [e["camera_id"] for e in group] != [1, 2, 3, 4, 5] or
                [e["within_group_ordinal"] for e in group] != [1, 2, 3, 4, 5]
                for group in group_events) or
            len(group_events) != seal.get("source_group_count")):
        raise FOADSEvaluationError("registered five-view group or original-position accounting is invalid")
    expected_updates = len(group_events) // 2
    if len(updates) != expected_updates or seal.get("completed_batch10_updates") != expected_updates:
        raise FOADSEvaluationError("native update journal does not match registered Batch-10 accounting")
    for index, update in enumerate(updates):
        start, stop = index * 10, (index + 1) * 10
        expected_groups = [group_ids[index * 2], group_ids[index * 2 + 1]]
        if (update.get("schema") != "final-benchmark.foads-native-update.v1" or
                update.get("update_index") != index + 1 or
                update.get("start_position") != start or update.get("end_position_exclusive") != stop or
                update.get("event_identity_sha256s") != [e["image_identity_sha256"] for e in events[start:stop]] or
                update.get("source_group_sha256s") != expected_groups or
                not _is_sha(update.get("model_state_after_update_sha256"))):
            raise FOADSEvaluationError("native update journal differs from registered Batch-10 identities")
    return bundle, events, scores, annotation_sha


def _load_authorization(stage_b_authorization: str | Path | Mapping[str, Any]
                        ) -> tuple[dict[str, Any], str]:
    if isinstance(stage_b_authorization, Mapping):
        authorization = dict(stage_b_authorization)
        digest = hashlib.sha256(canonical_json(authorization)).hexdigest()
    else:
        path = Path(stage_b_authorization)
        try:
            raw = path.read_bytes()
            authorization = json.loads(raw)
        except Exception:
            raise FOADSEvaluationError("post-seal evaluation authorization is missing or unreadable") from None
        if not isinstance(authorization, dict):
            raise FOADSEvaluationError("post-seal evaluation authorization must be a JSON object")
        digest = hashlib.sha256(raw).hexdigest()
    return authorization, digest


def _validate_authorization(auth: Mapping[str, Any], *, identity: Mapping[str, Any],
                            seal: Mapping[str, Any], seal_file_sha: str, category_bundle_sha: str,
                            annotation_sha: str) -> None:
    required = {
        "status": "AUTHORIZED_POST_STAGE_A_LABEL_PROJECTION",
        "scope": "stage_b_category_evaluation",
        "dataset_id": "realiad",
        "category": identity.get("category"),
        "condition_id": "P0",
        "campaign_run_id": identity.get("campaign_run_id"),
        "stage_a_source_commit": identity.get("source_commit"),
        "stage_a_seal_sha256": seal_file_sha,
        "category_bundle_sha256": category_bundle_sha,
        "dataset_release_sha256": identity.get("dataset_release_sha256"),
        "role_manifest_sha256": identity.get("role_manifest_sha256"),
        "stream_manifest_sha256": identity.get("stream_manifest_sha256"),
        "foads_registration_projection_sha256": identity.get("foads_registration_projection_sha256"),
        "prediction_journal_sha256": seal.get("prediction_journal_sha256"),
        "update_journal_sha256": seal.get("update_journal_sha256"),
        "source_sha256": annotation_sha,
        "test_outcomes_after_valid_seal": True,
        "test_masks_authorized": False,
        "online_outcome_access": False,
    }
    if (auth.get("schema") != FOADS_STAGE_B_AUTH_SCHEMA or
            any(auth.get(key) != value for key, value in required.items()) or
            not isinstance(auth.get("authorization_basis"), str) or
            not auth["authorization_basis"].startswith("RESEARCHER_EXPLICIT_")):
        raise FOADSEvaluationError("post-seal evaluation authorization is missing or bound to different inputs")


def _read_official_annotation_bytes(path: str | Path) -> bytes:
    try:
        return Path(path).read_bytes()
    except OSError:
        raise FOADSEvaluationError("registered official annotation is unavailable") from None


def _project_labels(data: bytes, *, expected_sha256: str, category: str,
                    dataset_release_sha256: str, stream_seed: int,
                    events: Sequence[Mapping[str, Any]]) -> list[int]:
    if hashlib.sha256(data).hexdigest() != expected_sha256:
        raise FOADSEvaluationError("official annotation SHA-256 differs from the registered annotation digest")
    try:
        annotation = json.loads(data)
    except Exception:
        raise FOADSEvaluationError("official annotation JSON is malformed") from None
    if not isinstance(annotation, dict) or not isinstance(annotation.get("meta"), dict):
        raise FOADSEvaluationError("official annotation lacks registered metadata")
    meta = annotation["meta"]
    normal_class = meta.get("normal_class")
    if (not isinstance(normal_class, str) or not normal_class or
            str(meta.get("category", category)).casefold() != category.casefold()):
        raise FOADSEvaluationError("official annotation category or normal-class metadata is invalid")
    dataset_name = str(meta.get("dataset", "realiad")).replace("-", "").replace("_", "").casefold()
    if dataset_name != "realiad":
        raise FOADSEvaluationError("official annotation dataset metadata differs from Real-IAD")
    test_rows = annotation.get("test")
    if not isinstance(test_rows, list) or len(test_rows) != len(events):
        raise FOADSEvaluationError("official annotation TEST array length differs from the sealed stream")
    positions = [event.get("original_position") for event in events]
    if set(positions) != set(range(len(test_rows))):
        raise FOADSEvaluationError("registered original positions do not cover the official TEST array exactly")

    from .identity import (
        _domain_hash, _image_identity_digest, _realiad_identity_key, _source_group_digest,
    )

    labels: list[int] = []
    seen_paths: set[str] = set()
    seen_events: set[int] = set()
    for event in events:
        original_position = int(event["original_position"])
        row = test_rows[original_position]
        if not isinstance(row, dict):
            raise FOADSEvaluationError("official annotation TEST entry is malformed")
        image_key = row.get("image_path")
        anomaly_class = row.get("anomaly_class")
        if (not isinstance(image_key, str) or not image_key or
                not isinstance(anomaly_class, str) or not anomaly_class or
                ("category" in row and str(row["category"]).casefold() != category.casefold())):
            raise FOADSEvaluationError("official TEST entry lacks its registered image identity or image-level class")
        try:
            group_key, camera = _realiad_identity_key(category, image_key)
        except Exception:
            raise FOADSEvaluationError("official image path fails the registered identity grammar") from None
        ordinal = event.get("within_group_ordinal")
        expected_identity = _image_identity_digest(image_key)
        expected_group = _source_group_digest(group_key)
        expected_event = _domain_hash(
            "Gate0-event-v1", dataset_release_sha256, category, str(stream_seed),
            group_key, str(camera), image_key, str(ordinal))
        event_position = event.get("position")
        if (isinstance(event_position, bool) or not isinstance(event_position, int) or
                camera != event.get("camera_id") or
                expected_identity != event.get("image_identity_sha256") or
                expected_group != event.get("source_group_sha256") or
                expected_event != event.get("event_id_sha256") or
                event_position in seen_events or image_key in seen_paths):
            raise FOADSEvaluationError("official image identity/path does not match the sealed registered event")
        seen_events.add(event_position)
        seen_paths.add(image_key)
        labels.append(int(anomaly_class != normal_class))
    if len(labels) != len(events) or len(seen_events) != len(events):
        raise FOADSEvaluationError("official image-level labels do not align one-to-one with sealed predictions")
    return labels


def _calculate_ranking_metrics(labels: Sequence[int], scores: Sequence[float]) -> dict[str, Any]:
    y = np.asarray(labels, dtype=np.int8)
    s = np.asarray(scores, dtype=np.float64)
    if (y.ndim != 1 or s.ndim != 1 or len(y) == 0 or len(y) != len(s) or
            not np.isin(y, (0, 1)).all() or not np.isfinite(s).all()):
        raise FOADSEvaluationError("labels and raw scores do not form a finite aligned binary ranking")
    positive = int(y.sum())
    negative = int(len(y) - positive)
    if positive == 0 or negative == 0:
        return {"status": "UNDEFINED_SINGLE_CLASS", "auroc": None, "average_precision": None,
                "crosscheck": "NOT_APPLICABLE", "positive_count": positive,
                "negative_count": negative, "anomaly_prevalence": positive / len(y)}
    auc, ap = _auc_ap(y, s)
    if auc is None or ap is None or not math.isfinite(auc) or not math.isfinite(ap):
        raise FOADSEvaluationError("registered ranking metric implementation returned an invalid value")
    crosscheck = "UNAVAILABLE"
    try:
        from sklearn.metrics import average_precision_score, roc_auc_score
        sklearn_auc = float(roc_auc_score(y, s))
        sklearn_ap = float(average_precision_score(y, s))
        if (not math.isclose(float(auc), sklearn_auc, rel_tol=0.0, abs_tol=1e-10) or
                not math.isclose(float(ap), sklearn_ap, rel_tol=0.0, abs_tol=1e-10)):
            raise FOADSEvaluationError("ranking metrics disagree with scikit-learn cross-check")
        crosscheck = "PASS"
    except ImportError:
        pass
    return {"status": "DEFINED", "auroc": float(auc), "average_precision": float(ap),
            "crosscheck": crosscheck, "positive_count": positive,
            "negative_count": negative, "anomaly_prevalence": positive / len(y)}


def _write_exclusive(path: Path, data: bytes) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())


def _render_report(summary: Mapping[str, Any]) -> str:
    metrics = summary["metrics"]
    lines = [
        "# FOADS-KNG Real-IAD Stage B Evaluation",
        "",
        f"**Status:** `{summary['evaluation_status']}`  ",
        f"**Category/condition:** `{summary['category']}/P0`  ",
        f"**Scored events:** {summary['scored_event_count']}",
        "",
        "## Image-level ranking metrics",
        "",
        f"- AUROC: {metrics['auroc']}",
        f"- Average Precision: {metrics['average_precision']}",
        f"- Normal/anomaly counts: {summary['negative_count']}/{summary['positive_count']}",
        f"- Anomaly prevalence: {summary['anomaly_prevalence']:.12g}",
        "- Score direction: higher `raw_image_score` means more anomalous.",
        "- No threshold metrics are reported because FOADS has no registered decision threshold.",
        "",
        "TEST outcomes were decoded only in Stage B after the complete FOADS seal and the bound post-seal grant were verified. No masks or event-level labels are included in this report.",
        "",
        f"Stage A source commit: `{summary['stage_a_source_commit']}`  ",
        f"Stage A seal SHA-256: `{summary['stage_a_seal_sha256']}`",
        "",
    ]
    return "\n".join(lines)


def evaluate_foads_sealed_category(*, attempt_root: str | Path,
                                   category_bundle_path: str | Path,
                                   annotation_path: str | Path,
                                   stage_b_authorization: str | Path | Mapping[str, Any],
                                   output_directory: str | Path) -> dict[str, Any]:
    """Verify a sealed FOADS P0 attempt and create an immutable aggregate result.

    The function intentionally does not compare the historical Stage A source
    commit to the current Git HEAD. The attempt's own source-bound status and
    seal remain the authority for its numerical trajectory.
    """
    root = Path(attempt_root).resolve()
    output = Path(output_directory).resolve()
    if output.exists():
        raise FOADSEvaluationError("Stage B output directory already exists; refusing to overwrite")
    if output.is_relative_to(root):
        raise FOADSEvaluationError("Stage B output must be outside the sealed Stage A attempt")

    # No official annotation path is opened above this line.
    status, identity, seal, seal_file_sha = _verify_status_and_seal(root)
    bundle, events, scores, annotation_sha = _verify_category_registration(
        root, identity, seal, category_bundle_path)
    category_bundle_sha = sha256_file(category_bundle_path)
    authorization, authorization_sha = _load_authorization(stage_b_authorization)
    _validate_authorization(authorization, identity=identity, seal=seal,
                            seal_file_sha=seal_file_sha,
                            category_bundle_sha=category_bundle_sha,
                            annotation_sha=annotation_sha)

    # Outcome bytes are first read after the status, seal, journal, registration,
    # output location, and post-seal authorization checks have passed.
    annotation_bytes = _read_official_annotation_bytes(annotation_path)
    labels = _project_labels(
        annotation_bytes, expected_sha256=annotation_sha,
        category=str(identity["category"]),
        dataset_release_sha256=str(identity["dataset_release_sha256"]),
        stream_seed=int(bundle["stream_manifest"]["stream_seed"]),
        events=events)
    scores_values = [float(row["raw_image_score"]) for row in scores]
    ranked = _calculate_ranking_metrics(labels, scores_values)
    summary = {
        "schema": FOADS_STAGE_B_RESULT_SCHEMA,
        "evaluation_status": "EVALUATION_COMPLETE",
        "dataset_id": "realiad",
        "category": identity["category"],
        "condition_id": "P0",
        "baseline": FOADS_BASELINE,
        "stage_a_campaign_run_id": identity["campaign_run_id"],
        "stage_a_source_commit": identity["source_commit"],
        "stage_a_seal_sha256": seal_file_sha,
        "stage_a_seal_body_sha256": seal["seal_sha256"],
        "prediction_journal_sha256": seal["prediction_journal_sha256"],
        "update_journal_sha256": seal["update_journal_sha256"],
        "category_bundle_sha256": category_bundle_sha,
        "dataset_release_sha256": identity["dataset_release_sha256"],
        "role_manifest_sha256": identity["role_manifest_sha256"],
        "stream_manifest_sha256": identity["stream_manifest_sha256"],
        "foads_registration_projection_sha256": identity["foads_registration_projection_sha256"],
        "annotation_sha256": annotation_sha,
        "post_seal_authorization_sha256": authorization_sha,
        "post_seal_authorization_status": authorization["status"],
        "post_seal_authorization_scope": authorization["scope"],
        "scored_event_count": len(scores),
        "positive_count": ranked["positive_count"],
        "negative_count": ranked["negative_count"],
        "anomaly_prevalence": ranked["anomaly_prevalence"],
        "outcome_access_phase": "STAGE_B_AFTER_VALID_FOADS_SEAL",
        "test_masks_read": False,
        "threshold_metrics_status": "NOT_REGISTERED_FOR_FOADS",
        "metric_status": ranked["status"],
        "metrics": {
            "auroc": ranked["auroc"],
            "average_precision": ranked["average_precision"],
            "score_field": "raw_image_score",
            "positive_class": "image-level defect (anomaly_class != meta.normal_class)",
            "score_direction": "higher_is_more_anomalous",
            "auroc_definition": "Mann-Whitney ranking AUROC with average ranks for exact score ties",
            "average_precision_definition": "noninterpolated step-wise average precision over descending score thresholds",
            "crosscheck": ranked["crosscheck"],
        },
    }
    report = _render_report(summary).encode("utf-8")
    summary_bytes = json.dumps(summary, sort_keys=True, indent=2, ensure_ascii=False,
                               allow_nan=False).encode("utf-8") + b"\n"
    receipt = {
        "schema": FOADS_STAGE_B_RECEIPT_SCHEMA,
        "category": identity["category"],
        "condition_id": "P0",
        "stage_a_seal_sha256": seal_file_sha,
        "annotation_sha256": annotation_sha,
        "evaluation_summary_sha256": hashlib.sha256(summary_bytes).hexdigest(),
        "evaluation_report_sha256": hashlib.sha256(report).hexdigest(),
        "evaluator_source_sha256": sha256_file(Path(__file__)),
    }
    receipt_bytes = json.dumps(receipt, sort_keys=True, indent=2).encode("utf-8") + b"\n"
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        output.mkdir(exist_ok=False)
        _write_exclusive(output / "evaluation_summary.json", summary_bytes)
        _write_exclusive(output / "EVALUATION_REPORT.md", report)
        _write_exclusive(output / "evaluation_receipt.json", receipt_bytes)
    except Exception as exc:
        raise FOADSEvaluationError(f"could not create non-overwriting Stage B output ({type(exc).__name__})") from None
    return {"evaluation_status": summary["evaluation_status"],
            "output_directory": str(output), "summary": summary,
            "evaluation_receipt_sha256": hashlib.sha256(receipt_bytes).hexdigest()}


def _registered_p0_categories(config_path: str | Path) -> dict[str, dict[str, Any]]:
    config = _read_object(config_path, "registered execution plan")
    if config.get("schema") != CONFIG_SCHEMA:
        raise FOADSEvaluationError("aggregation plan schema is not the registered final-benchmark schema")
    try:
        validate_config(config, require_source_binding=False)
    except Exception as exc:
        raise FOADSEvaluationError(f"aggregation plan failed registered validation: {exc}") from exc
    rows = config.get("trajectories")
    if not isinstance(rows, list):
        raise FOADSEvaluationError("aggregation plan lacks its registered trajectories")
    p0 = [row for row in rows if isinstance(row, dict) and row.get("condition") == "P0"]
    result = {str(row.get("category")): row for row in p0}
    if len(p0) != 30 or len(result) != 30:
        raise FOADSEvaluationError("aggregation plan must contain exactly 30 unique registered P0 categories")
    return result


def _load_verified_result(summary_path: str | Path) -> dict[str, Any]:
    path = Path(summary_path)
    summary = _read_object(path, "Stage B evaluation summary")
    receipt = _read_object(path.parent / "evaluation_receipt.json", "Stage B evaluation receipt")
    try:
        report_sha = sha256_file(path.parent / "EVALUATION_REPORT.md")
        summary_sha = sha256_file(path)
    except OSError:
        raise FOADSEvaluationError("Stage B result artifacts are incomplete") from None
    if (receipt.get("schema") != FOADS_STAGE_B_RECEIPT_SCHEMA or
            receipt.get("evaluation_summary_sha256") != summary_sha or
            receipt.get("evaluation_report_sha256") != report_sha or
            receipt.get("category") != summary.get("category") or
            receipt.get("condition_id") != summary.get("condition_id") or
            receipt.get("stage_a_seal_sha256") != summary.get("stage_a_seal_sha256") or
            receipt.get("annotation_sha256") != summary.get("annotation_sha256")):
        raise FOADSEvaluationError("Stage B summary and evaluation receipt do not match")
    return summary


def aggregate_foads_stage_b(*, config_path: str | Path,
                            evaluation_summaries: Sequence[str | Path],
                            output_directory: str | Path) -> dict[str, Any]:
    """Create a P0 category-equal macro only from valid registered result files."""
    output = Path(output_directory).resolve()
    if output.exists():
        raise FOADSEvaluationError("macro output directory already exists; refusing to overwrite")
    expected = _registered_p0_categories(config_path)
    accepted: dict[str, dict[str, Any]] = {}
    for summary_path in evaluation_summaries:
        summary = _load_verified_result(summary_path)
        category = summary.get("category")
        if (summary.get("schema") != FOADS_STAGE_B_RESULT_SCHEMA or
                summary.get("evaluation_status") != "EVALUATION_COMPLETE" or
                summary.get("dataset_id") != "realiad" or
                summary.get("condition_id") != "P0" or
                summary.get("baseline") != FOADS_BASELINE or
                summary.get("outcome_access_phase") != "STAGE_B_AFTER_VALID_FOADS_SEAL" or
                summary.get("test_masks_read") is not False or
                category not in expected or category in accepted):
            raise FOADSEvaluationError("Stage B result is invalid, duplicated, or outside the registered P0 categories")
        registration = expected[category]
        metrics = summary.get("metrics")
        if (not isinstance(metrics, dict) or
                summary.get("scored_event_count") != registration.get("test_event_count") or
                summary.get("role_manifest_sha256") != registration.get("role_manifest_sha256") or
                summary.get("stream_manifest_sha256") != registration.get("stream_manifest_sha256") or
                summary.get("foads_registration_projection_sha256") != registration.get("foads_p0_registration_projection_sha256") or
                summary.get("metric_status") != "DEFINED" or
                not isinstance(metrics.get("auroc"), (int, float)) or
                not isinstance(metrics.get("average_precision"), (int, float)) or
                not 0.0 <= float(metrics["auroc"]) <= 1.0 or
                not 0.0 <= float(metrics["average_precision"]) <= 1.0 or
                not _is_sha(summary.get("stage_a_seal_sha256")) or
                not _is_sha(summary.get("annotation_sha256")) or
                not _is_sha(summary.get("post_seal_authorization_sha256"))):
            raise FOADSEvaluationError("Stage B result metrics or registered identities are invalid")
        accepted[category] = summary

    missing = sorted(set(expected) - set(accepted))
    complete = not missing and len(accepted) == 30
    macro = None
    status = "INCOMPLETE_MISSING_CATEGORIES"
    if complete:
        macro = {
            "category_equal_macro_auroc": float(np.mean([accepted[c]["metrics"]["auroc"] for c in sorted(accepted)])),
            "category_equal_macro_average_precision": float(np.mean([accepted[c]["metrics"]["average_precision"] for c in sorted(accepted)])),
        }
        status = "COMPLETE_30_CATEGORY_P0"
    result = {
        "schema": FOADS_STAGE_B_AGGREGATE_SCHEMA,
        "status": status,
        "dataset_id": "realiad",
        "condition_id": "P0",
        "baseline": FOADS_BASELINE,
        "registered_category_count": 30,
        "valid_category_count": len(accepted),
        "missing_categories": missing,
        "complete_30_category_benchmark": complete,
        "macro_metrics": macro,
        "category_results": {category: {
            "stage_a_source_commit": accepted[category]["stage_a_source_commit"],
            "stage_a_seal_sha256": accepted[category]["stage_a_seal_sha256"],
            "annotation_sha256": accepted[category]["annotation_sha256"],
            "scored_event_count": accepted[category]["scored_event_count"],
            "auroc": accepted[category]["metrics"]["auroc"],
            "average_precision": accepted[category]["metrics"]["average_precision"],
        } for category in sorted(accepted)},
        "aggregation_rule": "unweighted arithmetic mean across the 30 registered P0 categories",
    }
    payload = json.dumps(result, sort_keys=True, indent=2, ensure_ascii=False,
                         allow_nan=False).encode("utf-8") + b"\n"
    report = ("# FOADS-KNG Real-IAD P0 Macro\n\n"
              f"**Status:** `{status}`\n\n"
              f"Valid categories: {len(accepted)}/30.\n\n"
              +(f"Category-equal macro AUROC: {macro['category_equal_macro_auroc']:.12g}\n\n"
                f"Category-equal macro AP: {macro['category_equal_macro_average_precision']:.12g}\n"
                if macro is not None else "No completed 30-category macro is reported until all registered categories have valid results.\n"))
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        output.mkdir(exist_ok=False)
        _write_exclusive(output / "macro_summary.json", payload)
        _write_exclusive(output / "MACRO_REPORT.md", report.encode("utf-8"))
    except Exception as exc:
        raise FOADSEvaluationError(f"could not write non-overwriting macro output ({type(exc).__name__})") from None
    return result
