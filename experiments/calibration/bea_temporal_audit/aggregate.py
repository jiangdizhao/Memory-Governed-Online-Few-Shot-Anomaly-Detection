"""Condition-within-category, category-macro exploratory aggregation."""
from __future__ import annotations

from collections import defaultdict
from typing import Any, Mapping
import numpy as np
from .core import ARMS, BEA, AuditError

SEED = 20261114
RESAMPLES = 10000


def _values(segment: Mapping[str, Any]) -> dict[tuple[str, str], float | None]:
    out: dict[tuple[str, str], float | None] = {}
    arms = segment["arms"]
    for arm in ARMS:
        a = arms[arm]
        out[(arm, "group_count")] = segment["group_count"]
        out[(arm, "image_count")] = segment["image_count"]
        for key in ("proposed", "executed", "group_suppressed", "cap_suppressed", "groups_with_proposal", "groups_with_execution", "groups_with_cap_suppression"):
            out[(arm, key)] = a[key]
        for key in ("fpr", "fnr", "ber", "auroc", "ap", "normal_count", "defect_count"):
            out[(arm, key)] = a.get("metrics", {}).get(key)
        for key in ("predicted_normal", "predicted_defect"):
            out[(arm, key)] = a.get("prediction_counts", {}).get(key)
            for sub in ("normal_count", "defect_count", "defect_prevalence"):
                out[(arm, key + "_" + sub)] = a.get("prediction_groups", {}).get(key, {}).get(sub)
        for key in ("count", "normal_count", "defect_count", "defect_prevalence", "hidden_false_negatives"):
            out[(arm, "cap_suppressed_" + key)] = a.get("cap_suppressed_outcomes", {}).get(key)
        if arm in BEA:
            for key in ("objective", "price", "objective_minus_price", "pre_cap_price", "post_cap_price"):
                for stat in ("mean", "p10", "median", "p90"):
                    out[(arm, key + "_" + stat)] = a[key][stat]
            for key in ("fixed_price_proposals", "fixed_price_disagreements"):
                out[(arm, key)] = a[key]
    for arm, key in (("STATIC-4+64", "static_4_64"), ("N-shadow", "n_shadow")):
        for metric in ("fpr", "fnr", "ber", "auroc", "ap", "normal_count", "defect_count"):
            out[(arm, metric)] = segment.get(key, {}).get(metric)
    for left, right in (("F-B", "F-M"), ("K-B", "K-M"), ("F-B", "STATIC-4+64"),
                        ("K-B", "STATIC-4+64"), ("F-B", "N-shadow")):
        for metric in ("fpr", "fnr", "ber", "auroc", "ap"):
            a, b = out.get((left, metric)), out.get((right, metric))
            out[(left + " minus " + right, metric)] = None if a is None or b is None else a - b
    return out


def aggregate_campaign(results: Mapping[tuple[str, str], Mapping[str, Any]], *,
                       expected_categories: int = 30) -> list[dict[str, Any]]:
    """Each category contributes one P0/P1/P2 mean to each macro and bootstrap."""
    cats = sorted({c for c, _ in results})
    if len(cats) != expected_categories or set(results) != {(c, p) for c in cats for p in ("P0", "P1", "P2")}:
        raise AuditError("aggregation requires every registered category and exactly P0/P1/P2")
    segments = sorted(next(iter(results.values()))["segments"])
    rng = np.random.default_rng(SEED)
    indices = rng.integers(0, len(cats), size=(RESAMPLES, len(cats)))
    output = []
    for segment in segments:
        flat = {key: _values(value["segments"][segment]) for key, value in results.items()}
        keys = sorted(set().union(*(v.keys() for v in flat.values())))
        for arm, metric in keys:
            category_values = []
            for cat in cats:
                v = [flat[(cat, cond)].get((arm, metric)) for cond in ("P0", "P1", "P2")]
                category_values.append(None if any(x is None for x in v) else sum(v) / 3.0)
            defined = sum(v is not None for v in category_values)
            if defined == len(cats):
                arr = np.asarray(category_values, dtype=float)
                boot = arr[indices].mean(axis=1)
                mean, low, high, status = float(arr.mean()), float(np.quantile(boot, .025)), float(np.quantile(boot, .975)), "DEFINED"
            else:
                mean = low = high = None
                status = "UNDEFINED_CATEGORY_OR_CONDITION"
            output.append({"segment": segment, "arm": arm, "metric": metric,
                           "category_count": len(cats), "defined_category_count": defined,
                           "mean": mean, "ci95_low": low, "ci95_high": high, "status": status})
    for arm in ARMS:
        category_values = []
        for cat in cats:
            positions = []
            for cond in ("P0", "P1", "P2"):
                trajectory = results[(cat, cond)]
                pos = trajectory["accounting"]["first_cap_exhaustion_position"][arm]
                n = sum(trajectory["segments"][f"decile_{i:02d}"]["image_count"] for i in range(1, 11))
                positions.append(None if pos is None else pos / n)
            category_values.append(None if any(x is None for x in positions) else sum(positions) / 3)
        defined = sum(x is not None for x in category_values)
        if defined == len(cats):
            arr = np.asarray(category_values, dtype=float); boot = arr[indices].mean(axis=1)
            mean, low, high, status = float(arr.mean()), float(np.quantile(boot, .025)), float(np.quantile(boot, .975)), "DEFINED"
        else:
            mean = low = high = None; status = "UNDEFINED_CATEGORY_OR_CONDITION"
        output.append({"segment": "overall", "arm": arm, "metric": "cap_exhaustion_fraction",
                       "category_count": len(cats), "defined_category_count": defined,
                       "mean": mean, "ci95_low": low, "ci95_high": high, "status": status})
    return output
