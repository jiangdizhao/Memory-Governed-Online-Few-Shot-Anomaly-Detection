"""Independent CPU-only metrics and temporal evaluator for the frozen FS4 campaign."""
from __future__ import annotations

import hashlib
import json
import math
from typing import Any, Mapping, Sequence

from .fs4_joint_campaign_contract import EXPECTED_CATEGORIES, EXPECTED_EVENTS, load_spec


def _finite_vector(values: Sequence[Any], name: str) -> list[float]:
    result = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            raise ValueError(f"{name} contains a non-finite or non-numeric value")
        result.append(float(value))
    return result


def _auc(labels: Sequence[int], scores: Sequence[float]) -> float | None:
    positives = sum(labels)
    negatives = len(labels) - positives
    if not positives or not negatives:
        return None
    ordered = sorted(zip(scores, labels), key=lambda pair: pair[0])
    rank_sum = 0.0
    index = 0
    while index < len(ordered):
        end = index + 1
        while end < len(ordered) and ordered[end][0] == ordered[index][0]:
            end += 1
        average_rank = ((index + 1) + end) / 2.0
        rank_sum += average_rank * sum(label for _, label in ordered[index:end])
        index = end
    return (rank_sum - positives * (positives + 1) / 2.0) / (positives * negatives)


def _average_precision(labels: Sequence[int], scores: Sequence[float]) -> float | None:
    positives = sum(labels)
    if not positives:
        return None
    ordered = sorted(zip(scores, labels), key=lambda pair: pair[0], reverse=True)
    true_positive = false_positive = 0
    previous_recall = 0.0
    area = 0.0
    index = 0
    while index < len(ordered):
        end = index + 1
        while end < len(ordered) and ordered[end][0] == ordered[index][0]:
            end += 1
        block_positive = sum(label for _, label in ordered[index:end])
        block_negative = end - index - block_positive
        true_positive += block_positive
        false_positive += block_negative
        recall = true_positive / positives
        precision = true_positive / (true_positive + false_positive)
        area += (recall - previous_recall) * precision
        previous_recall = recall
        index = end
    return area


def metrics(labels: Sequence[int], scores: Sequence[Any], predictions: Sequence[int] | None = None) -> dict[str, Any]:
    """Compute image-level binary metrics with explicit undefined denominators."""
    if len(labels) != len(scores) or not labels:
        raise ValueError("metric inputs must have one or more aligned events")
    if any(type(label) is not int or label not in (0, 1) for label in labels):
        raise ValueError("outcomes must be binary integer labels")
    score_values = _finite_vector(scores, "scores")
    if predictions is None:
        raise ValueError("prequential predictions must be supplied from the sealed Stage A projection")
    if (len(predictions) != len(labels) or
            any(type(value) is not int or value not in (0, 1) for value in predictions)):
        raise ValueError("prequential predictions must be aligned binary integer decisions")
    tp = sum(y == 1 and p == 1 for y, p in zip(labels, predictions))
    tn = sum(y == 0 and p == 0 for y, p in zip(labels, predictions))
    fp = sum(y == 0 and p == 1 for y, p in zip(labels, predictions))
    fn = sum(y == 1 and p == 0 for y, p in zip(labels, predictions))
    normal = tn + fp
    defect = tp + fn
    fpr = fp / normal if normal else None
    fnr = fn / defect if defect else None
    ber = (fpr + fnr) / 2.0 if fpr is not None and fnr is not None else None
    return {
        "n": len(labels), "confusion": {"tp": tp, "tn": tn, "fp": fp, "fn": fn},
        "fpr": fpr, "fnr": fnr, "ber": ber,
        "balanced_accuracy": 1.0 - ber if ber is not None else None,
        "auroc": _auc(labels, score_values),
        "average_precision": _average_precision(labels, score_values),
        "undefined_reasons": {
            "fpr": "no normal outcomes" if normal == 0 else None,
            "fnr": "no defect outcomes" if defect == 0 else None,
            "ber": "one ground-truth class absent" if not normal or not defect else None,
            "balanced_accuracy": "one ground-truth class absent" if not normal or not defect else None,
            "auroc": "one ground-truth class absent" if not normal or not defect else None,
            "average_precision": "no defect outcomes" if defect == 0 else None,
        },
    }


def _mean_defined(values: Sequence[float | None], *, metric_name: str, category_count: int) -> dict[str, Any]:
    if len(values) != category_count or any(value is None for value in values):
        return {"value": None, "undefined_reason": f"requires all {category_count} category values for {metric_name}"}
    return {"value": sum(float(value) for value in values) / category_count, "undefined_reason": None}


def _readout_names(trajectory_id: str) -> list[str]:
    if trajectory_id == "S":
        return [f"{controller}_{scorer}" for controller in ("L", "E", "FrozenTheta")
                for scorer in ("G", "GC")]
    controller = "L" if trajectory_id == "CL" else "E"
    return [f"{controller}_{scorer}" for scorer in ("G", "GC")]


def _controller_name(readout: str) -> str:
    return readout.rsplit("_", 1)[0]


def _score_name(readout: str) -> str:
    return readout.rsplit("_", 1)[1]


def _grouped(category_rows: Sequence[Mapping[str, Any]]) -> tuple[list[str], dict[str, list[Mapping[str, Any]]]]:
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    order: list[str] = []
    for row in sorted(category_rows, key=lambda item: item["position"]):
        group = row["group"]
        if group not in grouped:
            grouped[group] = []
            order.append(group)
        grouped[group].append(row)
    for group in order:
        views = [row["view"] for row in grouped[group]]
        if views != [1, 2, 3, 4, 5]:
            raise ValueError("Stage A event projection does not preserve five ordered camera views")
    return order, grouped


def _max_false_run(flags: Sequence[bool]) -> int:
    best = current = 0
    for flag in flags:
        current = 0 if flag else current + 1
        best = max(best, current)
    return best


def _controller_audit(row: Mapping[str, Any], controller: str) -> Mapping[str, Any]:
    audit = row.get("transition", {}).get("audit")
    controllers = audit.get("controllers") if isinstance(audit, Mapping) else None
    if controller == "FrozenTheta" and isinstance(controllers, Mapping) and "L" in controllers:
        frozen = {"revision": 0, "suppression": {}, "threshold_tokens": {"G": None, "GC": None},
            "status": {"G": "FROZEN", "GC": "FROZEN"}, "effective_sample_size": 0.0,
            "normal_evidence_groups": 0, "cache_bytes": 0, "cache_groups": 0,
            "rescore_invocations": 0, "rescore_seconds_total": 0.0,
            "thresholds": {"G": row.get("thresholds", {}).get("FrozenTheta_G"),
                           "GC": row.get("thresholds", {}).get("FrozenTheta_GC")}}
        return frozen
    state = controllers.get(controller) if isinstance(controllers, Mapping) else None
    if not isinstance(state, Mapping):
        raise ValueError("Stage A transition lacks a controller audit snapshot")
    return state


def _prediction_threshold(row: Mapping[str, Any], controller: str, scorer: str) -> float:
    thresholds = row.get("thresholds", {})
    value = thresholds.get(f"{controller}_{scorer}") if isinstance(thresholds, Mapping) else None
    if value is None and isinstance(thresholds, Mapping):
        nested = thresholds.get(controller)
        value = nested.get(scorer) if isinstance(nested, Mapping) else None
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise ValueError("pre-event threshold is missing or non-finite")
    return float(value)


def _controller_update_summary(rows: Sequence[Mapping[str, Any]], controller: str,
                               prior_state: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Count accepted estimator tokens, independently of evidence revisions."""
    if not rows:
        return {"evidence_revision_start": 0, "evidence_revision": 0,
            "accepted_update_count_G": 0, "accepted_update_count_GC": 0,
            "earliest_accepted_update_position_G": None,
            "earliest_accepted_update_position_GC": None,
            "events_with_any_accepted_update": 0, "threshold_holds_by_reason": {},
            "total_threshold_movement_G": 0.0, "total_threshold_movement_GC": 0.0}
    if prior_state is None:
        first_state = _controller_audit(rows[0], controller)
        prior: dict[str, Any] = {
            "revision": 0,
            "threshold_tokens": {"G": None, "GC": None},
            "thresholds": {scorer: _prediction_threshold(rows[0], controller, scorer)
                           for scorer in ("G", "GC")},
            # The first sealed snapshot has no predecessor in this output slice;
            # use its hold counters as the baseline, as the prior evaluator did.
            "suppression": dict(first_state.get("suppression", {})),
        }
    else:
        prior = dict(prior_state)
    if set(prior.get("thresholds", {})) != {"G", "GC"}:
        raise ValueError("prior controller audit lacks both scorer thresholds")
    tokens = prior.get("threshold_tokens", {"G": None, "GC": None})
    if not isinstance(tokens, Mapping) or set(tokens) != {"G", "GC"}:
        raise ValueError("prior controller audit lacks successful threshold tokens")
    prior["threshold_tokens"] = dict(tokens)
    prior["suppression"] = dict(prior.get("suppression", {}))
    prior["revision"] = int(prior.get("revision", 0))

    counts = {"G": 0, "GC": 0}
    earliest: dict[str, int | None] = {"G": None, "GC": None}
    movements = {"G": 0.0, "GC": 0.0}
    hold_counts: dict[str, int] = {}
    event_positions: set[int] = set()
    seen_positions: set[int] = set()
    revision_start = prior["revision"]
    revision = revision_start
    for row in rows:
        position = row.get("position")
        if type(position) is not int or position in seen_positions:
            raise ValueError("controller audit event position is invalid or duplicated")
        seen_positions.add(position)
        state = _controller_audit(row, controller)
        current_revision = state.get("revision")
        if type(current_revision) is not int or current_revision < revision:
            raise ValueError("controller evidence revision regressed")
        revision = current_revision
        current_tokens = state.get("threshold_tokens")
        if not isinstance(current_tokens, Mapping) or set(current_tokens) != {"G", "GC"}:
            raise ValueError("controller audit lacks successful threshold tokens")
        thresholds = state.get("thresholds")
        suppression = state.get("suppression", {})
        if not isinstance(thresholds, Mapping) or set(thresholds) != {"G", "GC"} or not isinstance(suppression, Mapping):
            raise ValueError("controller audit thresholds or hold counters are malformed")
        accepted_here = False
        for scorer in ("G", "GC"):
            old_token, new_token = prior["threshold_tokens"][scorer], current_tokens[scorer]
            if old_token is not None and new_token is None:
                raise ValueError("successful threshold token was cleared")
            if new_token is not None and new_token != old_token:
                counts[scorer] += 1
                earliest[scorer] = position if earliest[scorer] is None else earliest[scorer]
                accepted_here = True
            old_theta = prior["thresholds"][scorer]
            new_theta = thresholds[scorer]
            if (isinstance(new_theta, bool) or not isinstance(new_theta, (int, float)) or
                    not math.isfinite(float(new_theta))):
                raise ValueError("controller audit threshold is non-finite")
            movements[scorer] += abs(float(new_theta) - float(old_theta))
        if accepted_here:
            event_positions.add(position)
        for reason in set(prior["suppression"]) | set(suppression):
            previous_count = prior["suppression"].get(reason, 0)
            current_count = suppression.get(reason, 0)
            if (type(previous_count) is not int or type(current_count) is not int or
                    current_count < previous_count):
                raise ValueError("threshold hold counter regressed or is malformed")
            hold_counts[reason] = hold_counts.get(reason, 0) + current_count - previous_count
        prior = {"revision": current_revision, "threshold_tokens": dict(current_tokens),
                 "thresholds": dict(thresholds), "suppression": dict(suppression)}
    return {"evidence_revision_start": revision_start, "evidence_revision": revision,
        "accepted_update_count_G": counts["G"], "accepted_update_count_GC": counts["GC"],
        "earliest_accepted_update_position_G": earliest["G"],
        "earliest_accepted_update_position_GC": earliest["GC"],
        "events_with_any_accepted_update": len(event_positions),
        "threshold_holds_by_reason": {key: value for key, value in sorted(hold_counts.items()) if value},
        "total_threshold_movement_G": movements["G"],
        "total_threshold_movement_GC": movements["GC"]}


def _query_summary(rows: Sequence[Mapping[str, Any]], labels: Mapping[str, int]) -> dict[str, Any]:
    receipts = [row["query"] for row in rows]
    executed = [bool(receipt["executed"]) for receipt in receipts]
    queried = [row for row, flag in zip(rows, executed) if flag]
    groups = {row["group"] for row in rows}
    queried_groups = {row["group"] for row in queried}
    candidates: dict[str, int] = {}
    ordinals: dict[str, int] = {}
    for row, receipt in zip(rows, receipts):
        candidates[str(receipt["candidate"])] = candidates.get(str(receipt["candidate"]), 0) + 1
        ordinals[str(receipt["ordinal"])] = ordinals.get(str(receipt["ordinal"]), 0) + 1
    design = [float(row["query"]["design_probability"]) for row in rows
              if row["query"]["design_probability"] is not None]
    normal_groups = {row["group"] for row in queried if labels[row["event"]] == 0}
    return {
        "image_events": len(rows), "source_groups": len(groups), "executed_queries": len(queried),
        "image_query_rate": len(queried) / len(rows),
        "group_query_fraction": len(queried_groups) / len(groups) if groups else None,
        "candidate_ordinal_distribution": candidates, "event_ordinal_distribution": ordinals,
        "design_probability_distribution": {
            "n_candidate_events": len(design), "min": min(design) if design else None,
            "median": sorted(design)[len(design) // 2] if design else None,
            "max": max(design) if design else None,
            "clipped_weight_count": sum((1.0 / value) > 50.0 for value in design),
        },
        "query_cost_utility_history": [{"position": row["position"], "cost": row["query"]["cost"],
            "utility": row["query"]["utility"], "executed": row["query"]["executed"],
            "candidate": row["query"]["candidate"], "ordinal": row["query"]["ordinal"],
            "design_probability": row["query"]["design_probability"]} for row in rows],
        "queried_verified_normal_groups": len(normal_groups),
        "queried_verified_normal_events": sum(labels[row["event"]] == 0 for row in queried),
        "longest_query_free_event_interval": _max_false_run(executed),
    }


def _readout_temporal(rows: Sequence[Mapping[str, Any]], readout: str,
                      labels: Mapping[str, int]) -> dict[str, Any]:
    groups, grouped = _grouped(rows)
    n = len(groups)
    group_indices = {group: index for index, group in enumerate(groups)}
    windows: list[dict[str, Any]] = []
    for start in range(0, max(0, n - 50) + 1, 10):
        selected = set(groups[start:start + 50])
        window_rows = [row for row in rows if row["group"] in selected]
        window_labels = [labels[row["event"]] for row in window_rows]
        score_key, prediction_key = _score_name(readout), readout
        values = metrics(window_labels, [row["scores"][score_key] for row in window_rows],
                         [row["decisions"][prediction_key] for row in window_rows])
        query_stats = _query_summary(window_rows, labels)
        last_audit = _controller_audit(window_rows[-1], _controller_name(readout))
        prior_rows = [row for row in rows if group_indices[row["group"]] < start]
        prior_row = max(prior_rows, key=lambda row: row["position"]) if prior_rows else None
        first_audit = (_controller_audit(prior_row, _controller_name(readout)) if prior_row is not None else
                       {"rescore_invocations": 0, "rescore_seconds_total": 0.0})
        update_summary = _controller_update_summary(window_rows, _controller_name(readout),
            _controller_audit(prior_row, _controller_name(readout)) if prior_row is not None else None)
        first_rescore = int(first_audit.get("rescore_invocations", 0))
        last_rescore = int(last_audit.get("rescore_invocations", 0))
        first_seconds = float(first_audit.get("rescore_seconds_total", 0.0))
        last_seconds = float(last_audit.get("rescore_seconds_total", 0.0))
        windows.append({"start_group_index": start, "end_group_index_exclusive": start + 50,
            "groups": 50, "events": len(window_rows), "class_counts": {"normal": window_labels.count(0),
            "defect": window_labels.count(1)}, "ber": values["ber"], "fpr": values["fpr"],
            "fnr": values["fnr"], "ber_undefined_reason": values["undefined_reasons"]["ber"],
            "executed_query_count": query_stats["executed_queries"], "query_rate": query_stats["image_query_rate"],
            "normal_feedback_count": query_stats["queried_verified_normal_events"],
            "distinct_normal_feedback_groups": query_stats["queried_verified_normal_groups"],
            "effective_sample_size": last_audit.get("effective_sample_size"),
            "ess_sufficient": float(last_audit.get("effective_sample_size", 0.0)) >= 12.0,
            "e_rescore_invocations": max(0, last_rescore - first_rescore),
            "e_rescore_seconds": max(0.0, last_seconds - first_seconds),
            "e_cache_bytes_at_window_end": last_audit.get("cache_bytes", 0),
            "longest_query_free_event_interval": query_stats["longest_query_free_event_interval"],
            "threshold_update_diagnostics": update_summary})
    valid = [row for row in windows if row["ber"] is not None]
    valid_fpr = [row["fpr"] for row in windows if row["fpr"] is not None]
    valid_fnr = [row["fnr"] for row in windows if row["fnr"] is not None]
    return {"window_width_groups": 50, "stride_groups": 10, "total_windows": len(windows),
        "valid_ber_windows": len(valid), "excluded_one_class_ber_windows": len(windows) - len(valid),
        "worst_window_ber": max((row["ber"] for row in valid), default=None),
        "worst_window_fpr": max(valid_fpr, default=None), "worst_window_fnr": max(valid_fnr, default=None),
        "windows": windows}


def evaluate_campaign(stage_a_by_arm: Mapping[str, Mapping[str, Any]], labels: Mapping[str, int],
                      telemetry_by_arm: Mapping[str, Sequence[Mapping[str, Any]]]) -> dict[str, Any]:
    """Build the preregistered metrics from independently verified sealed event projections."""
    if set(stage_a_by_arm) != {"S", "CL", "CE"} or set(telemetry_by_arm) != {"S", "CL", "CE"}:
        raise ValueError("all three mandatory sealed trajectories and telemetry streams are required")
    spec = load_spec()
    expected_total = sum(EXPECTED_EVENTS.values())
    if len(labels) != expected_total or any(type(value) is not int or value not in (0, 1) for value in labels.values()):
        raise ValueError("Stage B outcome projection must contain all and only registered binary outcomes")
    all_event_ids: set[str] = set()
    common_identity: list[tuple[str, str, int, int]] | None = None
    per_arm: dict[str, Any] = {}
    for trajectory in ("S", "CL", "CE"):
        projection = stage_a_by_arm[trajectory]
        events = projection.get("events")
        if (projection.get("trajectory_id") != trajectory or not isinstance(events, list) or
                len(events) != expected_total):
            raise ValueError(f"{trajectory}: Stage A projection identity or full event count differs")
        local_ids = [row["event"] for row in events]
        if len(set(local_ids)) != expected_total or set(local_ids) - labels.keys():
            raise ValueError(f"{trajectory}: event projection is duplicated or absent from Stage B outcomes")
        identity = [(row["event"], row["group"], row["position"], row["view"]) for row in events]
        if common_identity is None:
            common_identity = identity
        elif identity != common_identity:
            raise ValueError("S, CL and CE do not share the same opaque registration and stream order")
        if not set(EXPECTED_CATEGORIES).issubset({row["category"] for row in events}):
            raise ValueError(f"{trajectory}: a frozen category is absent")
        by_category: dict[str, list[Mapping[str, Any]]] = {}
        for category in EXPECTED_CATEGORIES:
            rows = sorted((row for row in events if row["category"] == category), key=lambda row: row["position"])
            if len(rows) != EXPECTED_EVENTS[category]:
                raise ValueError(f"{trajectory}/{category}: event count differs from the frozen population")
            by_category[category] = rows
        category_metrics: dict[str, Any] = {}
        temporal: dict[str, Any] = {}
        diagnostics: dict[str, Any] = {}
        readouts = _readout_names(trajectory)
        for category in EXPECTED_CATEGORIES:
            rows = by_category[category]
            ys = [labels[row["event"]] for row in rows]
            category_metrics[category] = {}
            temporal[category] = {}
            diagnostics[category] = _query_summary(rows, labels)
            group_order, grouped = _grouped(rows)
            n_groups = len(group_order)
            group_indices = {group: index for index, group in enumerate(group_order)}
            boundaries = (0, n_groups // 3, (2 * n_groups) // 3, n_groups)
            segment_rows = {}
            for name, low, high in zip(("early", "middle", "late"), boundaries[:-1], boundaries[1:]):
                chosen = set(group_order[low:high])
                segment = [row for row in rows if row["group"] in chosen]
                segment_rows[name] = (low, high, segment)
            for readout in readouts:
                scorer = _score_name(readout)
                preds = [row["decisions"][readout] for row in rows]
                values = metrics(ys, [row["scores"][scorer] for row in rows], preds)
                category_metrics[category][readout] = values
                segments = {}
                for segment_name, (low, high, segment) in segment_rows.items():
                    segment_labels = [labels[row["event"]] for row in segment]
                    query_summary = _query_summary(segment, labels)
                    segment_metric = metrics(segment_labels,
                        [row["scores"][scorer] for row in segment],
                        [row["decisions"][readout] for row in segment]) if segment else None
                    state = _controller_audit(segment[-1], _controller_name(readout)) if segment else None
                    before_rows = [row for row in rows if group_indices[row["group"]] < low]
                    before_row = max(before_rows, key=lambda row: row["position"]) if before_rows else None
                    before = (_controller_audit(before_row, _controller_name(readout))
                              if before_row is not None else None)
                    update_summary = _controller_update_summary(segment, _controller_name(readout), before)
                    segments[segment_name] = {"group_index_bounds": [low, high], "groups": high - low,
                        "events": len(segment), "query_rate": query_summary["image_query_rate"],
                        "group_query_fraction": query_summary["group_query_fraction"],
                        "executed_queries": query_summary["executed_queries"],
                        "normal_feedback_count": query_summary["queried_verified_normal_events"],
                        "ber": segment_metric["ber"] if segment_metric else None,
                        "fpr": segment_metric["fpr"] if segment_metric else None,
                        "fnr": segment_metric["fnr"] if segment_metric else None,
                        "ber_undefined_reason": (segment_metric["undefined_reasons"]["ber"]
                                                 if segment_metric else "empty Source Group segment"),
                        "effective_sample_size": state.get("effective_sample_size") if state else None,
                        "distinct_verified_normal_groups": state.get("normal_evidence_groups") if state else 0,
                        "active_threshold_update_count": update_summary[
                            f"accepted_update_count_{_score_name(readout)}"],
                        **update_summary}
                temporal[category][readout] = {"segments": segments,
                    "sliding_windows": _readout_temporal(rows, readout, labels)}
            diagnostics[category]["threshold_trajectories"] = {
                readout: [{"position": row["position"], "threshold": row["thresholds"][readout]}
                          for row in rows] for readout in readouts}
            diagnostics[category]["native_state_version_trajectory"] = [
                {"position": row["position"], "normal_version": row["transition"]["audit"]["normal_version"],
                 "correction_version": row["transition"]["audit"]["correction_version"],
                 "greedy_dynamic_prototypes": row["transition"]["audit"]["greedy_dynamic_prototypes"],
                 "correction_prototypes": row["transition"]["audit"]["correction_prototypes"]}
                for row in rows]
            controller_diagnostics = {}
            for controller in sorted({_controller_name(name) for name in readouts}):
                states = [_controller_audit(row, controller) for row in rows]
                update_summary = _controller_update_summary(rows, controller)
                controller_diagnostics[controller] = {
                    "distinct_verified_normal_groups": states[-1].get("normal_evidence_groups", 0),
                    "effective_sample_size": states[-1].get("effective_sample_size", 0.0),
                    **update_summary,
                    "accepted_threshold_updates_after_first_event": (update_summary["accepted_update_count_G"] +
                        update_summary["accepted_update_count_GC"]),
                    "earliest_accepted_threshold_update_position": min(
                        (value for value in (update_summary["earliest_accepted_update_position_G"],
                                             update_summary["earliest_accepted_update_position_GC"])
                         if value is not None), default=None),
                    "total_threshold_movement": {"G": update_summary["total_threshold_movement_G"],
                                                  "GC": update_summary["total_threshold_movement_GC"]},
                    "e_cache_peak_bytes": max((state.get("cache_bytes", 0) for state in states), default=0),
                    "e_cache_peak_groups": max((state.get("cache_groups", 0) for state in states), default=0),
                    "e_rescore_invocations": states[-1].get("rescore_invocations", 0) - states[0].get("rescore_invocations", 0),
                    "e_rescore_seconds": states[-1].get("rescore_seconds_total", 0.0) - states[0].get("rescore_seconds_total", 0.0),
                }
            diagnostics[category]["controllers"] = controller_diagnostics
        macro: dict[str, Any] = {}
        for readout in readouts:
            macro[readout] = {metric_name: _mean_defined(
                [category_metrics[category][readout][metric_name] for category in EXPECTED_CATEGORIES],
                metric_name=metric_name, category_count=4)
                for metric_name in ("ber", "fpr", "fnr", "balanced_accuracy", "auroc", "average_precision")}
        per_arm[trajectory] = {"category_metrics": category_metrics, "four_category_macro": macro,
            "temporal": temporal, "query_and_controller_diagnostics": diagnostics,
            "runtime_telemetry": list(telemetry_by_arm[trajectory])}

    shared_rows = per_arm["S"]["category_metrics"]
    for category in EXPECTED_CATEGORIES:
        for scorer in ("G", "GC"):
            left, right = shared_rows[category][f"L_{scorer}"], shared_rows[category][f"E_{scorer}"]
            if left["auroc"] != right["auroc"] or left["average_precision"] != right["average_precision"]:
                raise ValueError("shared L/E score-ranking sanity check failed")
    h1_category = {category: (shared_rows[category]["E_GC"]["ber"] - shared_rows[category]["L_GC"]["ber"]
                              if shared_rows[category]["E_GC"]["ber"] is not None and
                              shared_rows[category]["L_GC"]["ber"] is not None else None)
                   for category in EXPECTED_CATEGORIES}
    h1 = _mean_defined(list(h1_category.values()), metric_name="Shared E-GC minus L-GC BER", category_count=4)
    coupled_category = {}
    for category in EXPECTED_CATEGORIES:
        cl = per_arm["CL"]["category_metrics"][category]["L_GC"]["ber"]
        ce = per_arm["CE"]["category_metrics"][category]["E_GC"]["ber"]
        coupled_category[category] = ce - cl if ce is not None and cl is not None else None
    h2 = _mean_defined(list(coupled_category.values()), metric_name="Coupled CE/E-GC minus CL/L-GC BER", category_count=4)
    return {
        "schema": "FS4_FINAL_JOINT_CAMPAIGN_METRICS_V1", "scientific_spec_sha256": spec["spec_sha256"],
        "population": {"categories": EXPECTED_EVENTS, "total_events_per_trajectory": expected_total,
                       "source_groups": sum(row["source_groups"] for row in spec["categories"])},
        "estimands": {
            "H1_shared_threshold_only": {"definition": "MacroBER(S/E/GC) - MacroBER(S/L/GC)",
                "category_paired_differences": h1_category, "four_category_macro_difference": h1,
                "interpretation": "negative values favour E; paired shared-score threshold-only contrast"},
            "H2_coupled_system": {"definition": "MacroBER(CE/E/GC) - MacroBER(CL/L/GC)",
                "category_paired_differences": coupled_category, "four_category_macro_difference": h2,
                "interpretation": "whole-system query and memory trajectory contrast; not a threshold-only effect"},
        },
        "arms": per_arm,
        "interpretation_limits": ["four development categories, one query seed", "Source Group views are clustered",
            "no inferential confidence interval is registered", "no finite-sample population-risk guarantee",
            "shared and coupled estimands are separate", "results are not independent external validation"],
    }


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()
