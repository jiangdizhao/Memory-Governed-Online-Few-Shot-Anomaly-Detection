"""Outcome evaluation and paired category bootstrap for sealed Gate 1 rows."""
from __future__ import annotations

from collections import defaultdict
import math
import numbers
from typing import Any, Mapping, Sequence

import numpy as np


class MetricContractError(RuntimeError):
    pass


def _binary(values: Sequence[int], name: str) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim != 1 or len(array) == 0 or not np.isin(array, (0, 1)).all():
        raise MetricContractError(f"{name} must be a nonempty binary vector")
    return array.astype(np.int8, copy=False)


def _auc_ap(labels: np.ndarray, scores: np.ndarray) -> tuple[float | None, float | None]:
    if len(np.unique(labels)) != 2:
        return None, None
    order = np.argsort(scores, kind="mergesort")
    sorted_scores, sorted_labels = scores[order], labels[order]
    # Average ranks for exact ties produce the Mann-Whitney AUROC.
    ranks = np.arange(1, len(scores) + 1, dtype=np.float64)
    start = 0
    while start < len(scores):
        stop = start + 1
        while stop < len(scores) and sorted_scores[stop] == sorted_scores[start]:
            stop += 1
        ranks[start:stop] = (start + 1 + stop) / 2.0
        start = stop
    positive = labels == 1
    n_pos = int(positive.sum())
    n_neg = len(labels) - n_pos
    auc = float((ranks[order.argsort()][positive].sum() - n_pos * (n_pos + 1) / 2) /
                (n_pos * n_neg))

    descending = np.argsort(-scores, kind="mergesort")
    y_desc = labels[descending]
    s_desc = scores[descending]
    tp = 0
    seen = 0
    ap = 0.0
    i = 0
    while i < len(y_desc):
        j = i + 1
        while j < len(y_desc) and s_desc[j] == s_desc[i]:
            j += 1
        tp += int(y_desc[i:j].sum())
        seen = j
        group_positives = int(y_desc[i:j].sum())
        ap += (group_positives / n_pos) * (tp / seen)
        i = j
    return auc, float(ap)


def binary_metrics(labels: Sequence[int], scores: Sequence[float],
                   predictions: Sequence[int]) -> dict[str, Any]:
    y = _binary(labels, "labels")
    pred = _binary(predictions, "predictions")
    score = np.asarray(scores, dtype=np.float64)
    if score.ndim != 1 or score.shape != y.shape or pred.shape != y.shape or not np.isfinite(score).all():
        raise MetricContractError("labels, scores and predictions must be aligned and finite")
    positives, negatives = y == 1, y == 0
    if not positives.any() or not negatives.any():
        undefined = "UNDEFINED_MISSING_POSITIVE_CLASS" if not positives.any() else "UNDEFINED_MISSING_NEGATIVE_CLASS"
        return {"status": undefined, "n": len(y), "positive_count": int(positives.sum()),
                "negative_count": int(negatives.sum()), "auroc": None, "ap": None,
                "balanced_accuracy": None, "ber": None, "fpr": None, "fnr": None}
    fpr = float(np.mean(pred[negatives] == 1))
    fnr = float(np.mean(pred[positives] == 0))
    auc, ap = _auc_ap(y, score)
    return {"status": "DEFINED", "n": len(y), "positive_count": int(positives.sum()),
            "negative_count": int(negatives.sum()), "auroc": auc, "ap": ap,
            "balanced_accuracy": 1.0 - (fpr + fnr) / 2.0,
            "ber": (fpr + fnr) / 2.0, "fpr": fpr, "fnr": fnr}


def worst_256_window_ber(dataset_id: str, labels: Sequence[int],
                         predictions: Sequence[int]) -> dict[str, Any]:
    y, pred = _binary(labels, "labels"), _binary(predictions, "predictions")
    if y.shape != pred.shape:
        raise MetricContractError("window labels and predictions are not aligned")
    if dataset_id == "visa_1cls":
        return {"status": "NOT_APPLICABLE", "worst_ber": None,
                "undefined_window_count": 0,
                "trailing_incomplete_event_count": int(len(y) % 256)}
    if dataset_id != "realiad":
        raise MetricContractError("dataset ID is not registered")
    values: list[float] = []
    undefined = 0
    for start in range(0, len(y) - 255, 256):
        local_y, local_pred = y[start:start + 256], pred[start:start + 256]
        if len(np.unique(local_y)) != 2:
            undefined += 1
            continue
        fpr = float(np.mean(local_pred[local_y == 0] == 1))
        fnr = float(np.mean(local_pred[local_y == 1] == 0))
        values.append((fpr + fnr) / 2.0)
    return {"status": "DEFINED" if values else "UNDEFINED_NO_COMPLETE_TWO_CLASS_WINDOW",
            "worst_ber": max(values) if values else None,
            "defined_window_count": len(values), "undefined_window_count": undefined,
            "trailing_incomplete_event_count": int(len(y) % 256)}


def category_equal_macro(category_metrics: Mapping[str, Mapping[str, Any]],
                         metric_names: Sequence[str] = ("auroc", "ap", "balanced_accuracy", "ber", "fpr", "fnr")) -> dict[str, Any]:
    categories = sorted(category_metrics)
    if not categories:
        raise MetricContractError("macro requires at least one category")
    result: dict[str, Any] = {"category_count": len(categories)}
    for name in metric_names:
        values = [category_metrics[category].get(name) for category in categories]
        result[name] = (None if any(value is None for value in values)
                        else float(np.mean(np.asarray(values, dtype=np.float64))))
        result[f"{name}_status"] = ("UNDEFINED_CATEGORY_PRESENT" if result[name] is None else "DEFINED")
    result["categories"] = categories
    return result


def compare_query_budgets(*, dataset_id: str, rho: float, total_events: Mapping[str, int],
                          left_queries: Mapping[str, int],
                          right_queries: Mapping[str, int],
                          tolerance: float = 0.003) -> dict[str, Any]:
    """Report equal maximum caps separately from realized-rate matching."""
    categories = sorted(total_events)
    expected_rho = {"realiad": 0.03, "visa_1cls": 0.10}.get(dataset_id)
    if (expected_rho is None or float(rho) != expected_rho or not categories or
            set(left_queries) != set(categories) or
            set(right_queries) != set(categories) or not 0.0 < float(rho) < 1.0 or
            float(tolerance) != 0.003):
        raise MetricContractError("budget comparison inputs differ from the registered category contract")
    by_category: dict[str, dict[str, Any]] = {}
    left_rates, right_rates = [], []
    for category in categories:
        raw_total, raw_left, raw_right = (total_events[category], left_queries[category], right_queries[category])
        if any(isinstance(value, bool) or not isinstance(value, numbers.Integral)
               for value in (raw_total, raw_left, raw_right)):
            raise MetricContractError("budget counts must be integers")
        total, left, right = int(raw_total), int(raw_left), int(raw_right)
        if total <= 0 or not 0 <= left <= math.floor(rho * total) or not 0 <= right <= math.floor(rho * total):
            raise MetricContractError("query count exceeds its category hard cap")
        left_rate, right_rate = left / total, right / total
        left_rates.append(left_rate)
        right_rates.append(right_rate)
        by_category[category] = {
            "total_events": total,
            "category_cap": math.floor(rho * total),
            "left_queries": left, "right_queries": right,
            "left_rate": left_rate, "right_rate": right_rate,
            "query_count_difference_left_minus_right": left - right,
            "rate_difference_left_minus_right": left_rate - right_rate,
            "realized_rate_match_pass": abs(left_rate - right_rate) <= tolerance,
        }
    left_macro, right_macro = float(np.mean(left_rates)), float(np.mean(right_rates))
    macro_difference = left_macro - right_macro
    return {
        "equal_maximum_category_caps": True,
        "all_category_realized_rate_matches_pass": all(
            value["realized_rate_match_pass"] for value in by_category.values()),
        "category_equal_macro_left_rate": left_macro,
        "category_equal_macro_right_rate": right_macro,
        "category_equal_macro_rate_difference_left_minus_right": macro_difference,
        "category_equal_macro_realized_rate_match_pass": abs(macro_difference) <= tolerance,
        "approximately_rate_matched_description_pass": (
            all(value["realized_rate_match_pass"] for value in by_category.values()) and
            abs(macro_difference) <= tolerance),
        "tolerance": float(tolerance), "categories": by_category,
    }


def paired_category_bootstrap(category_conditions: Mapping[str, Mapping[str, Mapping[str, Mapping[str, Any]]]],
                              contrast: str, *, replicates: int = 10_000,
                              seed: int = 20_261_111) -> dict[str, Any]:
    """Bootstrap category means while keeping all three conditions paired."""
    definitions = {
        "H_Q": ("ber", "F-M", "F-B"),
        "H_Q_transfer": ("ber", "K-M", "K-B"),
        "H_D": ("auroc", "F-B", "N-shadow"),
        "H_C": ("ber", "K-B", "F-B"),
    }
    if contrast not in definitions or replicates < 1:
        raise MetricContractError("contrast or bootstrap count is not registered")
    metric, left, right = definitions[contrast]
    categories = sorted(category_conditions)
    if not categories:
        raise MetricContractError("bootstrap requires complete category results")
    conditions = ("P0", "P1", "P2")
    category_values = []
    for category in categories:
        if set(category_conditions[category]) != set(conditions):
            raise MetricContractError("bootstrap category is missing a paired P0/P1/P2 condition")
        values = []
        for condition in conditions:
            arms = category_conditions[category][condition]
            a, b = arms[left].get(metric), arms[right].get(metric)
            if a is None or b is None:
                raise MetricContractError("undefined category metric prevents the registered macro/bootstrap")
            values.append(float(a) - float(b))
        category_values.append(float(np.mean(values)))
    values_array = np.asarray(category_values, dtype=np.float64)
    rng = np.random.Generator(np.random.PCG64(seed))
    samples = np.empty(replicates, dtype=np.float64)
    for index in range(replicates):
        draw = rng.integers(0, len(values_array), size=len(values_array))
        samples[index] = float(np.mean(values_array[draw]))
    return {"contrast": contrast, "metric": metric,
            "category_equal_estimate": float(values_array.mean()),
            "two_sided_95_percentile_interval": [float(np.quantile(samples, 0.025)),
                                                  float(np.quantile(samples, 0.975))],
            "replicates": int(replicates), "rng": "NumPy PCG64",
            "seed": int(seed), "category_count": len(categories),
            "category_values": dict(zip(categories, map(float, values_array)))}
