"""Sealed Stage B evaluator; Gate 1 validation uses synthetic labels only."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import numbers
import re
from typing import Any, Mapping, Sequence

from .metrics import binary_metrics, worst_256_window_ber
from .runner import StageASeal, static_arms_for_condition, CampaignProtocolError


class EvaluationProtocolError(RuntimeError):
    pass


@dataclass(frozen=True)
class LabelProjectionAuthorization:
    status: str
    scope: str
    source_sha256: str
    dataset_id: str
    category: str
    stage_a_seal_sha256: str


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def evaluate_sealed_category(*, dataset_id: str, category: str,
                             rows: Sequence[Mapping[str, Any]], seal: StageASeal,
                             group_seals: Sequence[Mapping[str, Any]],
                             labels_by_event_id: Mapping[str, int],
                             label_projection: LabelProjectionAuthorization) -> dict[str, Any]:
    """Evaluate one category only after a valid Stage A seal and separate label grant."""
    if (label_projection.status != "AUTHORIZED_POST_STAGE_A_LABEL_PROJECTION" or
            label_projection.scope != "stage_b_category_evaluation" or
            not re.fullmatch(r"[0-9a-f]{64}", label_projection.source_sha256) or
            label_projection.dataset_id != dataset_id or
            label_projection.category != category or
            label_projection.stage_a_seal_sha256 != seal.durable_seal_sha256):
        raise EvaluationProtocolError("Stage B requires a separate authorized label projection")
    if (seal.status != "SEALED_COMPLETE" or seal.dataset_id != dataset_id or
            seal.category != category or seal.completed_event_count != seal.expected_event_count or
            seal.expected_event_count != len(rows) or
            not re.fullmatch(r"[0-9a-f]{64}", seal.durable_seal_sha256)):
        raise EvaluationProtocolError("Stage B cannot evaluate an incomplete or mismatched Stage A seal")
    if (_digest(list(rows)) != seal.score_rows_sha256 or
            _digest(list(group_seals)) != seal.group_seals_sha256 or
            len(group_seals) != seal.group_count):
        raise EvaluationProtocolError("Stage A score rows differ from their completion seal")
    cursor = 0
    for group in group_seals:
        start, stop = int(group.get("start_position", -1)), int(group.get("end_position_exclusive", -1))
        if (start != cursor or stop - start != 5 or
                group.get("status") != "SEALED_BEFORE_LABEL_DISCLOSURE"):
            raise EvaluationProtocolError("Stage A group seal order is incomplete or noncontiguous")
        local = rows[start:stop]
        if ([int(row.get("camera_id", -1)) for row in local] != [1, 2, 3, 4, 5] or
                any(row.get("source_group_sha256") != group.get("source_group_sha256") for row in local)):
            raise EvaluationProtocolError("Stage A event rows do not match their sealed five-view group")
        cursor = stop
    if cursor != len(rows):
        raise EvaluationProtocolError("Stage A group seals do not cover all sealed score rows")
    event_ids = [str(row.get("event_id_sha256", "")) for row in rows]
    if len(set(event_ids)) != len(event_ids) or set(event_ids) != set(labels_by_event_id):
        raise EvaluationProtocolError("label projection does not match exactly the sealed event identities")
    labels = [int(labels_by_event_id[event_id]) for event_id in event_ids]
    if any(isinstance(labels_by_event_id[event_id], bool) or
           not isinstance(labels_by_event_id[event_id], numbers.Integral) or
           labels_by_event_id[event_id] not in (0, 1)
           for event_id in event_ids):
        raise EvaluationProtocolError("label projection contains a nonbinary image-level outcome")

    scores: dict[str, list[float]] = {}
    predictions: dict[str, list[int]] = {}
    for arm in ("F-B", "K-B", "F-M", "K-M"):
        scores[arm] = [float(row["arm_scores"][arm]) for row in rows]
        predictions[arm] = [int(row["arm_predictions"][arm]) for row in rows]
    scores["N-shadow"] = [float(row["n_shadow_score"]) for row in rows]
    # N-shadow reports its registered prequential AUROC/AP diagnostic only.
    predictions["N-shadow"] = [0] * len(rows)
    try:
        static_arms = static_arms_for_condition(seal.run_condition)
    except CampaignProtocolError as exc:
        raise EvaluationProtocolError("Stage A seal has an invalid condition") from exc
    for row in rows:
        for field in ("static_scores", "static_predictions", "static_thresholds"):
            if not isinstance(row.get(field), Mapping) or set(row[field]) != set(static_arms):
                raise EvaluationProtocolError("static baseline schema does not match the registered condition")
    for baseline in static_arms:
        values = [float(row["static_scores"][baseline]) for row in rows]
        thresholds = [float(row["static_thresholds"][baseline]) for row in rows]
        raw_predictions = [row["static_predictions"][baseline] for row in rows]
        if (not all(math.isfinite(value) for value in values + thresholds) or
                len(set(thresholds)) != 1 or
                any(isinstance(value, bool) or not isinstance(value, numbers.Integral) or value not in (0, 1)
                    for value in raw_predictions) or
                any(int(pred) != int(score >= threshold)
                    for pred, score, threshold in zip(raw_predictions, values, thresholds))):
            raise EvaluationProtocolError("static predictions must use their own unchanged frozen threshold")
        scores[baseline] = values
        predictions[baseline] = [int(value) for value in raw_predictions]

    results: dict[str, Any] = {}
    for arm, values in scores.items():
        metric = binary_metrics(labels, values, predictions[arm])
        if arm == "N-shadow":
            metric = {key: metric[key] for key in ("status", "n", "positive_count", "negative_count", "auroc", "ap")}
        else:
            metric["worst_256"] = worst_256_window_ber(dataset_id, labels, predictions[arm])
        results[arm] = metric
    queried_label_composition: dict[str, dict[str, int]] = {}
    query_counts: dict[str, int] = {}
    first_queried_defect_position: dict[str, int | None] = {}
    for arm in ("F-B", "K-B", "F-M", "K-M"):
        queried = [index for index, row in enumerate(rows)
                   if bool(row["query_actions"][arm]["executed"])]
        query_counts[arm] = len(queried)
        queried_label_composition[arm] = {
            "normal": sum(labels[index] == 0 for index in queried),
            "defect": sum(labels[index] == 1 for index in queried),
        }
        defect_positions = [int(rows[index]["position"]) for index in queried if labels[index] == 1]
        first_queried_defect_position[arm] = min(defect_positions) if defect_positions else None
    return {
        "dataset_id": dataset_id, "category": category,
        "condition_id": seal.run_condition,
        "stage_a_seal_status": seal.status,
        "stage_a_score_rows_sha256": seal.score_rows_sha256,
        "label_projection_source_sha256": label_projection.source_sha256,
        "arm_metrics": results,
        "query_count": query_counts,
        "queried_label_composition": queried_label_composition,
        "first_queried_defect_position": first_queried_defect_position,
        "outcome_access_phase": "STAGE_B_AFTER_VALID_STAGE_A_SEAL",
    }
