"""Pure, in-memory analysis of synthetic or separately verified score rows."""
from __future__ import annotations

from collections import deque
import math
import numbers
import re
from typing import Any, Mapping, Sequence

import numpy as np
from experiments.calibration.final_benchmark.metrics import binary_metrics

ARMS = ("F-B", "K-B", "F-M", "K-M")
BEA = ("F-B", "K-B")
HEX = re.compile(r"[0-9a-f]{64}\Z")


class AuditError(ValueError):
    """Audit inputs violate the registered analysis contract."""


def _digest(value: Any) -> bool:
    return isinstance(value, str) and bool(HEX.fullmatch(value))


def intervals(group_index: int, group_count: int) -> tuple[str, str]:
    if not 0 <= group_index < group_count:
        raise AuditError("invalid group index")
    decile = min(9, 10 * group_index // group_count)
    return f"decile_{decile + 1:02d}", ("early" if decile < 3 else "middle" if decile < 7 else "late")


def validate_rows(rows: Sequence[Mapping[str, Any]], *, cap: int,
                  expected_event_ids: Sequence[str] | None = None) -> dict[str, Any]:
    """Validate group structure, accounting and a past-only BEA price replay."""
    if not rows or len(rows) % 5 or isinstance(cap, bool) or not isinstance(cap, int) or not 0 <= cap <= len(rows):
        raise AuditError("a stream must contain complete five-view groups and a valid hard cap")
    ids = [r.get("event_id_sha256") for r in rows]
    if any(not _digest(x) for x in ids) or len(set(ids)) != len(ids):
        raise AuditError("event identities must be unique SHA-256 values")
    if expected_event_ids is not None and list(expected_event_ids) != ids:
        raise AuditError("event identities differ from the registered stream order")
    seen_groups: set[str] = set()
    static_threshold: float | None = None
    prices = {arm: 0.8 for arm in BEA}
    flags = {arm: deque(maxlen=256) for arm in BEA}
    counts = {arm: 0 for arm in ARMS}
    first_cap = {arm: (0 if cap == 0 else None) for arm in ARMS}
    for start in range(0, len(rows), 5):
        group = rows[start:start + 5]
        g = group[0].get("source_group_sha256")
        if not _digest(g) or g in seen_groups or any(r.get("source_group_sha256") != g for r in group):
            raise AuditError("duplicate or malformed Source Group")
        seen_groups.add(g)
        if [r.get("position") for r in group] != list(range(start, start + 5)) or [r.get("camera_id") for r in group] != [1, 2, 3, 4, 5]:
            raise AuditError("group position or camera order differs from the registered stream")
        for arm in ARMS:
            used = False
            actions: list[bool] = []
            for row in group:
                try:
                    score = row["arm_scores"][arm]
                    prediction = row["arm_predictions"][arm]
                    threshold = row["arm_thresholds"][arm]
                    action = row["query_actions"][arm]
                    values = [action[x] for x in ("proposed", "executed", "group_suppressed", "cap_suppressed")]
                except (KeyError, TypeError) as exc:
                    raise AuditError("missing score, prediction, threshold or query-action field") from exc
                if (isinstance(score, bool) or not isinstance(score, (float, int)) or not math.isfinite(score) or
                        isinstance(threshold, bool) or not isinstance(threshold, (float, int)) or not math.isfinite(threshold) or
                        type(prediction) is not int or prediction not in (0, 1) or prediction != int(score >= threshold)):
                    raise AuditError("score, prediction or threshold is invalid")
                if any(type(x) is not bool for x in values):
                    raise AuditError("query actions must be explicit booleans")
                proposed, executed, group_suppressed, cap_suppressed = values
                expected = (proposed and not used and counts[arm] < cap,
                            proposed and used,
                            proposed and not used and counts[arm] >= cap)
                if (executed, group_suppressed, cap_suppressed) != expected:
                    raise AuditError("executed and suppression flags violate group/cap ordering")
                if arm in BEA:
                    objective, price = action.get("proposal_objective"), action.get("proposal_price")
                    if (isinstance(objective, bool) or isinstance(price, bool) or
                            not isinstance(objective, (int, float)) or not isinstance(price, (int, float)) or
                            not math.isfinite(objective) or not math.isfinite(price) or
                            not math.isclose(price, prices[arm], rel_tol=1e-12, abs_tol=1e-12) or
                            proposed != (objective > price)):
                        raise AuditError("BEA objective, pre-group price or proposal does not replay")
                if executed:
                    used = True
                    counts[arm] += 1
                    if counts[arm] == cap:
                        first_cap[arm] = row["position"]
                actions.append(executed)
            if arm in BEA:
                for action in actions:
                    flags[arm].append(int(action))
                    prices[arm] = min(3.0, max(0.05, prices[arm] + 0.02 * (sum(flags[arm]) / len(flags[arm]) - 0.03)))
        for row in group:
            try:
                shadow = row["n_shadow_score"]
                score = row["static_scores"]["STATIC-4+64"]
                threshold = row["static_thresholds"]["STATIC-4+64"]
                prediction = row["static_predictions"]["STATIC-4+64"]
                if (any(isinstance(x, bool) or not isinstance(x, (float, int)) or not math.isfinite(x)
                        for x in (shadow, score, threshold)) or type(prediction) is not int or
                        prediction not in (0, 1) or prediction != int(score >= threshold)):
                    raise AuditError("invalid shadow or static score")
                if static_threshold is None:
                    static_threshold = float(threshold)
                elif static_threshold != float(threshold):
                    raise AuditError("static threshold changed during the trajectory")
            except (KeyError, TypeError) as exc:
                raise AuditError("missing shadow or static score") from exc
    return {"cap": cap, "executed": counts, "first_cap_exhaustion_position": first_cap,
            "final_price": prices}


def _distribution(values: Sequence[float]) -> dict[str, Any]:
    if not values:
        return {"n": 0, "mean": None, "p10": None, "median": None, "p90": None}
    arr = np.asarray(values, dtype=float)
    return {"n": len(values), "mean": float(arr.mean()), "p10": float(np.quantile(arr, .1)),
            "median": float(np.quantile(arr, .5)), "p90": float(np.quantile(arr, .9))}


def _metric(labels: Sequence[int], scores: Sequence[float], predictions: Sequence[int] | None) -> dict[str, Any]:
    if not labels:
        return {"status": "UNDEFINED_EMPTY", "n": 0, "normal_count": 0, "defect_count": 0,
                "fpr": None, "fnr": None, "ber": None, "auroc": None, "ap": None}
    p = list(predictions) if predictions is not None else [0] * len(labels)
    m = binary_metrics(labels, scores, p)
    normals, defects = labels.count(0), labels.count(1)
    fpr = sum(y == 0 and pred == 1 for y, pred in zip(labels, p)) / normals if normals and predictions is not None else None
    fnr = sum(y == 1 and pred == 0 for y, pred in zip(labels, p)) / defects if defects and predictions is not None else None
    return {"status": m["status"], "n": len(labels), "normal_count": normals, "defect_count": defects,
            "fpr": fpr, "fnr": fnr, "ber": (fpr + fnr) / 2 if fpr is not None and fnr is not None else None,
            "auroc": m["auroc"], "ap": m["ap"]}


def analyze_trajectory(rows: Sequence[Mapping[str, Any]], *, cap: int,
                       labels: Sequence[int] | None = None,
                       expected_event_ids: Sequence[str] | None = None) -> dict[str, Any]:
    """Return aggregate-only diagnostics. Labels are never included in the result."""
    replay = validate_rows(rows, cap=cap, expected_event_ids=expected_event_ids)
    if labels is not None and (len(labels) != len(rows) or any(isinstance(x, bool) or not isinstance(x, numbers.Integral) or x not in (0, 1) for x in labels)):
        raise AuditError("labels must be an aligned binary vector")
    if labels is not None:
        labels = [int(x) for x in labels]
    group_count = len(rows) // 5
    result: dict[str, Any] = {"accounting": replay, "segments": {},
        "component_availability": {"objective": "ARCHIVED", "B": "NOT_IDENTIFIABLE", "N": "NOT_IDENTIFIABLE", "D": "NOT_IDENTIFIABLE"}}
    names = [f"decile_{i:02d}" for i in range(1, 11)] + ["early", "middle", "late"]
    for name in names:
        group_indices = [g for g in range(group_count) if name in intervals(g, group_count)]
        indices = [g * 5 + view for g in group_indices for view in range(5)]
        seg: dict[str, Any] = {"group_count": len(group_indices), "image_count": len(indices), "arms": {}}
        for arm in ARMS:
            acts = [rows[i]["query_actions"][arm] for i in indices]
            item: dict[str, Any] = {key: sum(int(a[key]) for a in acts) for key in ("proposed", "executed", "group_suppressed", "cap_suppressed")}
            item["groups_with_proposal"] = sum(any(rows[g * 5 + j]["query_actions"][arm]["proposed"] for j in range(5)) for g in group_indices)
            item["groups_with_execution"] = sum(any(rows[g * 5 + j]["query_actions"][arm]["executed"] for j in range(5)) for g in group_indices)
            item["groups_with_cap_suppression"] = sum(any(rows[g * 5 + j]["query_actions"][arm]["cap_suppressed"] for j in range(5)) for g in group_indices)
            if arm in BEA:
                obj = [float(a["proposal_objective"]) for a in acts]
                price = [float(a["proposal_price"]) for a in acts]
                item.update(objective=_distribution(obj), price=_distribution(price), objective_minus_price=_distribution([x-y for x,y in zip(obj,price)]),
                            fixed_price_proposals=sum(x > .8 for x in obj),
                            fixed_price_disagreements=sum((x > .8) != a["proposed"] for x,a in zip(obj,acts)))
                exhaustion = replay["first_cap_exhaustion_position"][arm]
                item["pre_cap_price"] = _distribution([float(rows[i]["query_actions"][arm]["proposal_price"]) for i in indices if cap > 0 and (exhaustion is None or i <= exhaustion)])
                item["post_cap_price"] = _distribution([float(rows[i]["query_actions"][arm]["proposal_price"]) for i in indices if exhaustion is not None and (cap == 0 or i > exhaustion)])
            if labels is not None:
                ys = [labels[i] for i in indices]
                scores = [float(rows[i]["arm_scores"][arm]) for i in indices]
                preds = [int(rows[i]["arm_predictions"][arm]) for i in indices]
                item["metrics"] = _metric(ys, scores, preds)
                item["prediction_counts"] = {"predicted_normal": preds.count(0), "predicted_defect": preds.count(1)}
                item["prediction_groups"] = {}
                for prediction, key in ((0, "predicted_normal"), (1, "predicted_defect")):
                    outcomes = [y for y, p in zip(ys, preds) if p == prediction]
                    defects = outcomes.count(1)
                    item["prediction_groups"][key] = {"count": len(outcomes), "normal_count": len(outcomes)-defects,
                        "defect_count": defects, "defect_prevalence": defects / len(outcomes) if outcomes else None}
                suppressed = [i for i in indices if rows[i]["query_actions"][arm]["cap_suppressed"]]
                defect = sum(labels[i] == 1 for i in suppressed)
                item["cap_suppressed_outcomes"] = {"count": len(suppressed), "normal_count": len(suppressed)-defect,
                    "defect_count": defect, "defect_prevalence": defect / len(suppressed) if suppressed else None,
                    "hidden_false_negatives": sum(labels[i] == 1 and int(rows[i]["arm_predictions"][arm]) == 0 for i in suppressed)}
            seg["arms"][arm] = item
        if labels is not None:
            ys = [labels[i] for i in indices]
            seg["n_shadow"] = _metric(ys, [float(rows[i]["n_shadow_score"]) for i in indices], None)
            seg["static_4_64"] = _metric(ys, [float(rows[i]["static_scores"]["STATIC-4+64"]) for i in indices],
                [int(rows[i]["static_predictions"]["STATIC-4+64"]) for i in indices])
        result["segments"][name] = seg
    return result
