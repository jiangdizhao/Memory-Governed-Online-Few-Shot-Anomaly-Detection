"""R2-EXP1 development analysis for the 512-event switch tape.

Stage A is label blind: it verifies and replays the existing R1 tape, adds
counterfactual C2 patch amplitudes, and seals event scores. Stage B maps only
the first 512 TEST identities to outcomes after the Stage A seal, then computes
descriptive outcome-aware mechanism diagnostics. Raw event outcomes and query
feedback stay in the new untracked run directory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import secrets
import subprocess
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import numpy as np

from experiments.calibration.r2_0_label_blind_mechanism_audit import (
    ReadOnlyR1Evidence,
    corrected_dense_map,
    max_score,
    verify_public_tape,
)
from experiments.calibration.r2_perf1_common import (
    OUTPUT_PARENT,
    ROOT,
    OutputBudget,
    create_output_namespace,
    git_identity,
    new_run_id,
    require_plain_path,
    sha256_file,
)
from experiments.calibration.r2_perf1_stage_a import (
    _QueriedLedgerReader,
    append_queried_c2_anchors,
)
from experiments.calibration.r2_perf1_stage_b import _metric_record, _read_outcomes
from experiments.calibration.r1_matched_correction_pilot import (
    _c2_state_sha256,
    _validate_c2_anchor_state,
    signed_knn_delta,
)
from experiments.calibration.r1_runtime_supervisor import (
    current_process_is_in_job as r1_current_process_is_in_job,
    run_supervised_child,
)


TASK_ID = "R2-EXP1"
EXPECTED_BRANCH = "calibration-stage0-contract-20261029"
EXPECTED_PARENT_HEAD = "4a718090790812c91a90c8ab6d6c1a347593b8de"
R1_RUN_ID = "engineering_r1s4b-switch-s0-512-20261031-a1"
R1_ROOT = OUTPUT_PARENT / f"r1_digest_contract_v1_{R1_RUN_ID}"
PERF_RUN_ID = "20261101T025600_70c8a20b"
PERF_ROOT = OUTPUT_PARENT / f"r2_perf1_switch_s0_512_{PERF_RUN_ID}"
THETA = 0.5165398120880127
AMPLITUDES = (0.02, 0.04, 0.08, 0.20)
BOOTSTRAP_REPLICATES = 1000
BOOTSTRAP_SEED = 20261101
PAIR_MARGIN_BINS = (0.0, 0.01, 0.05, 0.10, 0.25, math.inf)
NEAR_THRESHOLD_BINS = (0.0, 0.02, 0.05, 0.10, math.inf)
QUARTILE_SIZE = 128
RUN_PREFIX = "r2_perf1_switch_s0_512_"
NONCE_ENV = "R2_EXP1_SUPERVISED_NONCE"
ROLE_ENV = "R2_EXP1_WORKER_ROLE"
OUTPUT_FILENAMES = {
    "run_manifest.json",
    "stage_a_scores.jsonl",
    "stage_a_c2_parity.json",
    "stage_a_diagnostics.json",
    "stage_a_sealed.json",
    "stage_b_metrics.json",
    "stage_b_sensitivity.json",
    "stage_b_diagnostics.json",
    "stage_b_report.json",
    "orchestrator_complete.json",
    "supervisor_status.json",
    "failure.json",
}


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode("utf-8")


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.write_bytes(_canonical_bytes(value) + b"\n")


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                raise ValueError(f"blank JSONL row at {path}:{line_number}")
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"non-object JSONL row at {path}:{line_number}")
            rows.append(row)
    return rows


def _json_safe(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _require_finite_vector(values: Sequence[float], name: str) -> np.ndarray:
    result = np.asarray(values, dtype=np.float64)
    if result.ndim != 1 or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be a finite one-dimensional vector")
    return result


def _binary_labels(values: Sequence[int]) -> np.ndarray:
    labels = np.asarray(values, dtype=np.int8)
    if labels.ndim != 1 or not np.isin(labels, (0, 1)).all():
        raise ValueError("labels must be a one-dimensional binary vector")
    return labels


def exact_pairwise_attribution(
    labels: Sequence[int], baseline: Sequence[float], candidate: Sequence[float]
) -> dict[str, Any]:
    """Return exact strict/tied AUROC pair attribution for one candidate arm."""
    y = _binary_labels(labels)
    base = _require_finite_vector(baseline, "baseline")
    cand = _require_finite_vector(candidate, "candidate")
    if len(y) != len(base) or len(y) != len(cand):
        raise ValueError("labels and scores must have equal lengths")
    pos = np.flatnonzero(y == 1)
    neg = np.flatnonzero(y == 0)
    if not len(pos) or not len(neg):
        raise ValueError("pairwise attribution requires both outcome classes")
    m0 = base[pos, None] - base[None, neg]
    m1 = cand[pos, None] - cand[None, neg]
    c0 = np.where(m0 > 0, 1.0, np.where(m0 == 0, 0.5, 0.0))
    c1 = np.where(m1 > 0, 1.0, np.where(m1 == 0, 0.5, 0.0))
    contribution = c1 - c0
    counts = {
        "baseline_strict_incorrect": int(np.count_nonzero(m0 < 0)),
        "baseline_strict_correct": int(np.count_nonzero(m0 > 0)),
        "baseline_ties": int(np.count_nonzero(m0 == 0)),
        "strict_incorrect_to_correct_repairs": int(np.count_nonzero((m0 < 0) & (m1 > 0))),
        "strict_correct_to_incorrect_damage": int(np.count_nonzero((m0 > 0) & (m1 < 0))),
        "baseline_incorrect_to_tie": int(np.count_nonzero((m0 < 0) & (m1 == 0))),
        "baseline_correct_to_tie": int(np.count_nonzero((m0 > 0) & (m1 == 0))),
        "baseline_tie_resolved_correctly": int(np.count_nonzero((m0 == 0) & (m1 > 0))),
        "baseline_tie_resolved_incorrectly": int(np.count_nonzero((m0 == 0) & (m1 < 0))),
        "baseline_tie_remaining": int(np.count_nonzero((m0 == 0) & (m1 == 0))),
    }
    net = float(contribution.sum(dtype=np.float64))
    n_pairs = int(m0.size)
    auc0 = float(c0.mean(dtype=np.float64))
    auc1 = float(c1.mean(dtype=np.float64))
    if not math.isclose(net / n_pairs, auc1 - auc0, rel_tol=0.0, abs_tol=2e-15):
        raise AssertionError("pairwise concordance contribution does not equal AUROC change")
    return {
        "pair_count": n_pairs,
        **counts,
        "net_concordance_change": net,
        "baseline_auc_from_pairs": auc0,
        "candidate_auc_from_pairs": auc1,
        "auc_difference_from_pairs": net / n_pairs,
    }


def _pair_margin_decomposition(
    labels: np.ndarray,
    baseline: np.ndarray,
    candidate: np.ndarray,
    candidate_name: str,
) -> list[dict[str, Any]]:
    pos = np.flatnonzero(labels == 1)
    neg = np.flatnonzero(labels == 0)
    m0 = baseline[pos, None] - baseline[None, neg]
    m1 = candidate[pos, None] - candidate[None, neg]
    contrib = np.where(m1 > 0, 1.0, np.where(m1 == 0, 0.5, 0.0)) - np.where(
        m0 > 0, 1.0, np.where(m0 == 0, 0.5, 0.0)
    )
    rows: list[dict[str, Any]] = []
    for lower, upper in zip(PAIR_MARGIN_BINS[:-1], PAIR_MARGIN_BINS[1:]):
        mask = (np.abs(m0) >= lower) & (np.abs(m0) < upper)
        rows.append(
            {
                "candidate": candidate_name,
                "abs_c0_margin_range": [lower, None if math.isinf(upper) else upper],
                "pair_count": int(mask.sum()),
                "net_concordance_change": float(contrib[mask].sum(dtype=np.float64)),
            }
        )
    return rows


def _movement_pair_decomposition(
    labels: np.ndarray, baseline: np.ndarray, candidate: np.ndarray, candidate_name: str
) -> dict[str, Any]:
    pos = np.flatnonzero(labels == 1)
    neg = np.flatnonzero(labels == 0)
    m0 = baseline[pos, None] - baseline[None, neg]
    m1 = candidate[pos, None] - candidate[None, neg]
    contrib = np.where(m1 > 0, 1.0, np.where(m1 == 0, 0.5, 0.0)) - np.where(
        m0 > 0, 1.0, np.where(m0 == 0, 0.5, 0.0)
    )
    ddelta = candidate[pos] - baseline[pos]
    ndelta = candidate[neg] - baseline[neg]
    # A positive anomaly movement and negative normal movement are favorable.
    def category(delta: np.ndarray, *, anomaly: bool) -> np.ndarray:
        favorable = delta > 0 if anomaly else delta < 0
        harmful = delta < 0 if anomaly else delta > 0
        return np.where(favorable, "favorable", np.where(harmful, "harmful", "zero"))

    dcat = category(ddelta, anomaly=True)[:, None]
    ncat = category(ndelta, anomaly=False)[None, :]
    cells = []
    for dname in ("favorable", "zero", "harmful"):
        for nname in ("favorable", "zero", "harmful"):
            mask = (dcat == dname) & (ncat == nname)
            cells.append(
                {
                    "anomaly_score_movement": dname,
                    "normal_score_movement": nname,
                    "pair_count": int(mask.sum()),
                    "net_concordance_change": float(contrib[mask].sum(dtype=np.float64)),
                }
            )
    return {"candidate": candidate_name, "cells": cells}


def _confusion(y: np.ndarray, prediction: np.ndarray) -> dict[str, int]:
    return {
        "tp": int(np.count_nonzero((y == 1) & (prediction == 1))),
        "tn": int(np.count_nonzero((y == 0) & (prediction == 0))),
        "fp": int(np.count_nonzero((y == 0) & (prediction == 1))),
        "fn": int(np.count_nonzero((y == 1) & (prediction == 0))),
    }


def correction_movement_summary(
    labels: Sequence[int],
    baseline_scores: Sequence[float],
    candidate_scores: Sequence[float],
    theta: float,
) -> dict[str, Any]:
    """Outcome direction, fixed-threshold flips, and near-threshold movements."""
    y = _binary_labels(labels)
    base = _require_finite_vector(baseline_scores, "baseline_scores")
    cand = _require_finite_vector(candidate_scores, "candidate_scores")
    if len(y) != len(base) or len(y) != len(cand) or not math.isfinite(theta):
        raise ValueError("movement inputs have incompatible lengths or threshold")
    delta = cand - base
    pred0 = base >= theta
    pred1 = cand >= theta
    favorable = np.where(y == 1, delta > 0, delta < 0)
    harmful = np.where(y == 1, delta < 0, delta > 0)
    neutral = delta == 0
    c0 = _confusion(y, pred0)
    c1 = _confusion(y, pred1)
    quantiles = (0.025, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99)

    def magnitude(array: np.ndarray) -> dict[str, Any]:
        absolute = np.abs(array)
        return {
            "count": int(array.size),
            "mean_absolute": float(absolute.mean()) if array.size else None,
            "quantiles": {
                str(q): float(np.quantile(absolute, q)) if array.size else None
                for q in quantiles
            },
        }

    near = []
    distance = np.abs(base - theta)
    for lower, upper in zip(NEAR_THRESHOLD_BINS[:-1], NEAR_THRESHOLD_BINS[1:]):
        mask = (distance >= lower) & (distance < upper)
        near.append(
            {
                "abs_c0_margin_range": [lower, None if math.isinf(upper) else upper],
                "events": int(mask.sum()),
                "favorable": int(np.count_nonzero(mask & favorable)),
                "harmful": int(np.count_nonzero(mask & harmful)),
                "prediction_flips": int(np.count_nonzero(mask & (pred0 != pred1))),
                "confusion_c0": _confusion(y[mask], pred0[mask]),
                "confusion_candidate": _confusion(y[mask], pred1[mask]),
            }
        )
    return {
        "events": int(len(y)),
        "image_score_delta": {
            "mean": float(delta.mean()),
            "mean_absolute": float(np.abs(delta).mean()),
            "median_absolute": float(np.median(np.abs(delta))),
            "p90_absolute": float(np.quantile(np.abs(delta), 0.90)),
            "absolute_magnitude_quantiles": magnitude(delta)["quantiles"],
            "positive": int(np.count_nonzero(delta > 0)),
            "negative": int(np.count_nonzero(delta < 0)),
            "zero": int(np.count_nonzero(delta == 0)),
        },
        "image_score_delta_by_outcome": {
            "anomalies": magnitude(delta[y == 1]),
            "normals": magnitude(delta[y == 0]),
        },
        "outcome_direction": {
            "favorable": int(np.count_nonzero(favorable)),
            "harmful": int(np.count_nonzero(harmful)),
            "neutral": int(np.count_nonzero(neutral)),
            "favorable_anomalies": int(np.count_nonzero((y == 1) & (delta > 0))),
            "harmful_anomalies": int(np.count_nonzero((y == 1) & (delta < 0))),
            "favorable_normals": int(np.count_nonzero((y == 0) & (delta < 0))),
            "harmful_normals": int(np.count_nonzero((y == 0) & (delta > 0))),
        },
        "prediction_changes": {
            "c0": c0,
            "candidate": c1,
            "flips": int(np.count_nonzero(pred0 != pred1)),
            "new_fp": int(np.count_nonzero((y == 0) & ~pred0 & pred1)),
            "repaired_fp": int(np.count_nonzero((y == 0) & pred0 & ~pred1)),
            "new_fn": int(np.count_nonzero((y == 1) & pred0 & ~pred1)),
            "repaired_fn": int(np.count_nonzero((y == 1) & ~pred0 & pred1)),
        },
        "near_c0_threshold": near,
    }


def patch_magnitude_summary(
    labels: Sequence[int], patch_deltas: Sequence[np.ndarray]
) -> dict[str, Any]:
    y = _binary_labels(labels)
    if len(y) != len(patch_deltas):
        raise ValueError("patch deltas and labels must have equal lengths")
    stacked = np.stack([np.asarray(row, dtype=np.float64).reshape(-1) for row in patch_deltas])
    if not np.isfinite(stacked).all():
        raise ValueError("patch deltas must be finite")

    def summarize(array: np.ndarray) -> dict[str, Any]:
        absolute = np.abs(array)
        return {
            "values": int(array.size),
            "mean_absolute": float(absolute.mean()) if array.size else 0.0,
            "median_absolute": float(np.median(absolute)) if array.size else 0.0,
            "p90_absolute": float(np.quantile(absolute, 0.90)) if array.size else 0.0,
            "max_absolute": float(absolute.max()) if array.size else 0.0,
            "absolute_magnitude_quantiles": {
                str(q): float(np.quantile(absolute, q)) if array.size else 0.0
                for q in (0.025, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99)
            },
            "positive": int(np.count_nonzero(array > 0)),
            "negative": int(np.count_nonzero(array < 0)),
            "zero": int(np.count_nonzero(array == 0)),
            "nonzero_fraction": float(np.count_nonzero(array != 0) / array.size) if array.size else 0.0,
        }

    return {
        "all": summarize(stacked),
        "anomalies": summarize(stacked[y == 1]),
        "normals": summarize(stacked[y == 0]),
        "per_event_mean_absolute": np.abs(stacked).mean(axis=1).tolist(),
    }


def query_interval_ids(positions: Sequence[int], query_positions: Sequence[int]) -> np.ndarray:
    """A query event belongs to the old interval; feedback affects later events."""
    q = np.asarray(sorted(int(value) for value in query_positions), dtype=np.int64)
    if len(set(q.tolist())) != len(q) or np.any(q < 0):
        raise ValueError("query positions must be unique, nonnegative values")
    p = np.asarray(positions, dtype=np.int64)
    if p.ndim != 1 or (len(p) and (np.any(p < 0) or np.any(np.diff(p) <= 0))):
        raise ValueError("event positions must be nonnegative and strictly increasing")
    return np.searchsorted(q, p, side="left").astype(np.int16)


def _auc_or_missing(labels: np.ndarray, scores: np.ndarray) -> tuple[float | None, str | None]:
    if not np.any(labels == 0) or not np.any(labels == 1):
        return None, "interval does not contain both outcome classes"
    return float(_metric_record(labels, scores, THETA)["image_auroc"]), None


def _weighted_auc(labels: np.ndarray, scores: np.ndarray, weights: np.ndarray) -> float:
    pos_w = weights * (labels == 1)
    neg_w = weights * (labels == 0)
    total = float(pos_w.sum() * neg_w.sum())
    if total <= 0:
        return float("nan")
    margins = scores[labels == 1, None] - scores[None, labels == 0]
    pair_weights = pos_w[labels == 1, None] * neg_w[labels == 0][None, :]
    conc = np.where(margins > 0, 1.0, np.where(margins == 0, 0.5, 0.0))
    return float((conc * pair_weights).sum(dtype=np.float64) / total)


def source_group_cluster_bootstrap(
    labels: Sequence[int],
    source_groups: Sequence[str],
    scores: Mapping[str, Sequence[float]],
    *,
    replicates: int = BOOTSTRAP_REPLICATES,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, Any]:
    """Paired percentile intervals; source groups, not events, are resampled."""
    y = _binary_labels(labels)
    groups = np.asarray([str(group) for group in source_groups], dtype=object)
    if len(groups) != len(y) or not len(y) or replicates < 1:
        raise ValueError("invalid source-group bootstrap inputs")
    unique = np.unique(groups)
    if len(unique) < 2:
        raise ValueError("at least two Source Groups are required")
    arms = {name: _require_finite_vector(values, name) for name, values in scores.items()}
    if any(len(values) != len(y) for values in arms.values()):
        raise ValueError("each score arm must align with labels")
    group_indices = {group: np.flatnonzero(groups == group) for group in unique}
    rng = np.random.default_rng(seed)
    samples: dict[str, list[float]] = {name: [] for name in arms}
    sample_gaps: dict[str, list[float]] = {
        name: [] for name in arms if name != "C0"
    }
    for _ in range(replicates):
        draw = rng.choice(unique, size=len(unique), replace=True)
        multiplicity = Counter(draw.tolist())
        weights = np.asarray([multiplicity[group] for group in groups], dtype=np.float64)
        aucs = {name: _weighted_auc(y, arm, weights) for name, arm in arms.items()}
        if not all(math.isfinite(value) for value in aucs.values()):
            continue
        for name, value in aucs.items():
            samples[name].append(value)
        for name in sample_gaps:
            sample_gaps[name].append(aucs[name] - aucs["C0"])

    def interval(values: Sequence[float]) -> list[float] | None:
        if not values:
            return None
        return [float(x) for x in np.quantile(np.asarray(values), [0.025, 0.975])]

    return {
        "method": "paired Source Group bootstrap with replacement",
        "seed": seed,
        "requested_replicates": replicates,
        "valid_replicates": min((len(values) for values in samples.values()), default=0),
        "source_group_count": int(len(unique)),
        "auc_percentile_ci_95": {name: interval(values) for name, values in samples.items()},
        "paired_auc_difference_vs_c0_ci_95": {
            name: interval(values) for name, values in sample_gaps.items()
        },
        "interpretation": "descriptive uncertainty under source-group resampling; not causal evidence",
    }


def _per_interval_summary(
    labels: np.ndarray,
    scores: Mapping[str, np.ndarray],
    positions: np.ndarray,
    interval_ids: np.ndarray,
    query_positions: Sequence[int],
) -> list[dict[str, Any]]:
    rows = []
    n_intervals = len(query_positions) + 1
    for interval_id in range(n_intervals):
        mask = interval_ids == interval_id
        if interval_id == 0:
            start = int(positions[mask][0]) if mask.any() else 0
            prev_query = None
        else:
            prev_query = int(query_positions[interval_id - 1])
            start = prev_query + 1
        end = int(query_positions[interval_id]) if interval_id < len(query_positions) else (
            int(positions[-1]) if len(positions) else start - 1
        )
        subset_labels = labels[mask]
        arm_metrics: dict[str, Any] = {}
        for name, values in scores.items():
            auc, reason = _auc_or_missing(subset_labels, values[mask])
            arm_metrics[name] = {"auroc": auc, "unavailable_reason": reason}
        rows.append(
            {
                "interval_id": interval_id,
                "start_position_inclusive": start,
                "end_position_inclusive": end,
                "previous_query_position": prev_query,
                "events": int(mask.sum()),
                "normal_events": int(np.count_nonzero(subset_labels == 0)),
                "anomaly_events": int(np.count_nonzero(subset_labels == 1)),
                "metrics": arm_metrics,
            }
        )
    return rows


def _query_age_summary(
    labels: np.ndarray,
    scores: Mapping[str, np.ndarray],
    positions: np.ndarray,
    query_positions: Sequence[int],
) -> list[dict[str, Any]]:
    q = np.asarray(sorted(int(value) for value in query_positions), dtype=np.int64)
    prior_indices = np.searchsorted(q, positions, side="left") - 1
    has_prior = prior_indices >= 0
    age = np.full(len(positions), -1, dtype=np.int64)
    age[has_prior] = positions[has_prior] - q[prior_indices[has_prior]]
    masks = [
        ("before_first_query", age < 0),
        ("0_to_15_events_after_previous_query", (age >= 0) & (age < 16)),
        ("16_to_63_events_after_previous_query", (age >= 16) & (age < 64)),
        ("64_or_more_events_after_previous_query", age >= 64),
    ]
    result = []
    for name, mask in masks:
        subset_labels = labels[mask]
        metrics = {}
        for arm, values in scores.items():
            auc, reason = _auc_or_missing(subset_labels, values[mask])
            metrics[arm] = {"auroc": auc, "unavailable_reason": reason}
        result.append(
            {
                "query_age_bin": name,
                "events": int(mask.sum()),
                "normal_events": int(np.count_nonzero(subset_labels == 0)),
                "anomaly_events": int(np.count_nonzero(subset_labels == 1)),
                "metrics": metrics,
            }
        )
    return result


def _pairwise_by_latest_interval(
    labels: np.ndarray,
    baseline: np.ndarray,
    candidate: np.ndarray,
    interval_ids: np.ndarray,
    candidate_name: str,
) -> list[dict[str, Any]]:
    pos = np.flatnonzero(labels == 1)
    neg = np.flatnonzero(labels == 0)
    m0 = baseline[pos, None] - baseline[None, neg]
    m1 = candidate[pos, None] - candidate[None, neg]
    contrib = np.where(m1 > 0, 1.0, np.where(m1 == 0, 0.5, 0.0)) - np.where(
        m0 > 0, 1.0, np.where(m0 == 0, 0.5, 0.0)
    )
    latest = np.maximum(interval_ids[pos, None], interval_ids[None, neg])
    result = []
    for interval_id in np.unique(latest):
        mask = latest == interval_id
        result.append(
            {
                "candidate": candidate_name,
                "latest_of_defect_or_normal_query_interval": int(interval_id),
                "pair_count": int(mask.sum()),
                "net_concordance_change": float(contrib[mask].sum(dtype=np.float64)),
            }
        )
    return result


def _source_group_contribution_summary(
    labels: np.ndarray,
    source_groups: Sequence[str],
    baseline: np.ndarray,
    candidate: np.ndarray,
    candidate_name: str,
) -> dict[str, Any]:
    pos = np.flatnonzero(labels == 1)
    neg = np.flatnonzero(labels == 0)
    m0 = baseline[pos, None] - baseline[None, neg]
    m1 = candidate[pos, None] - candidate[None, neg]
    contrib = np.where(m1 > 0, 1.0, np.where(m1 == 0, 0.5, 0.0)) - np.where(
        m0 > 0, 1.0, np.where(m0 == 0, 0.5, 0.0)
    )
    groups = np.asarray(source_groups, dtype=object)
    role_net: dict[tuple[str, str], float] = defaultdict(float)
    for row, image_index in enumerate(pos):
        for col, normal_index in enumerate(neg):
            amount = float(contrib[row, col])
            role_net[("anomaly", str(groups[image_index]))] += amount
            role_net[("normal", str(groups[normal_index]))] += amount
    output = {}
    for role in ("anomaly", "normal"):
        values = [value for (this_role, _), value in role_net.items() if this_role == role]
        output[role] = {
            "source_groups_with_positive_net_pair_contribution": int(sum(value > 0 for value in values)),
            "source_groups_with_negative_net_pair_contribution": int(sum(value < 0 for value in values)),
            "source_groups_with_zero_net_pair_contribution": int(sum(value == 0 for value in values)),
        }
    return {
        "candidate": candidate_name,
        "role_attribution": output,
        "caveat": "each cross-class pair is attributed to both endpoint groups; descriptive only",
    }


def _grouped_query_stats(query_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    by_outcome: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    by_c0_prediction: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    by_c0_outcome: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    feedback_signs = Counter()
    c1_keys = ("candidates_examined", "native_gate_kept", "inserted", "merged", "replaced", "skipped")
    for row in query_rows:
        outcome = str(row["queried_outcome"])
        prediction = str(row["c0_prediction"])
        c0_outcome = str(row.get("outcome", "NOT_RECORDED"))
        feedback_signs[outcome] += 1
        for key in c1_keys:
            by_outcome[outcome][key] += int(row.get(key, 0))
            by_c0_prediction[prediction][key] += int(row.get(key, 0))
            by_c0_outcome[c0_outcome][key] += int(row.get(key, 0))
        by_outcome[outcome]["queries"] += 1
        by_c0_prediction[prediction]["queries"] += 1
        by_c0_outcome[c0_outcome]["queries"] += 1
    return {
        "by_queried_outcome": {key: dict(value) for key, value in sorted(by_outcome.items())},
        "by_c0_prediction": {key: dict(value) for key, value in sorted(by_c0_prediction.items())},
        "by_c0_confusion_outcome": {key: dict(value) for key, value in sorted(by_c0_outcome.items())},
        "queried_feedback_balance": dict(sorted(feedback_signs.items())),
        "total": {
            "queries": len(query_rows),
            **{key: sum(int(row.get(key, 0)) for row in query_rows) for key in c1_keys},
            "c2_additions": sum(int(row.get("c2_additions", 0)) for row in query_rows),
            "c2_retained_after_query": max((int(row.get("c2_retained_after_query", 0)) for row in query_rows), default=0),
        },
    }


def analyze_outcomes(
    *,
    labels: Sequence[int],
    scores: Mapping[str, Sequence[float]],
    source_groups: Sequence[str],
    positions: Sequence[int],
    query_positions: Sequence[int],
    threshold: float,
    patch_deltas: Mapping[str, Sequence[np.ndarray]],
    dense_maps: Sequence[np.ndarray],
    top_patch_ids: Sequence[np.ndarray],
    query_details: Sequence[Mapping[str, Any]],
    bootstrap_replicates: int = BOOTSTRAP_REPLICATES,
) -> dict[str, Any]:
    """Pure CPU analysis. Per-event outcomes are returned separately by caller."""
    y = _binary_labels(labels)
    group_values = [str(item) for item in source_groups]
    pos = np.asarray(positions, dtype=np.int64)
    if any(len(values) != len(y) for values in (group_values, pos, dense_maps, top_patch_ids)):
        raise ValueError("event-level inputs do not align")
    arms = {name: _require_finite_vector(values, name) for name, values in scores.items()}
    if "C0" not in arms or any(len(values) != len(y) for values in arms.values()):
        raise ValueError("score arms must include aligned C0 scores")
    baseline = arms["C0"]
    arm_metrics = {name: _metric_record(y, value, threshold) for name, value in arms.items()}
    movement = {
        name: correction_movement_summary(y, baseline, value, threshold)
        for name, value in arms.items()
        if name != "C0"
    }
    pairwise = {
        name: exact_pairwise_attribution(y, baseline, value)
        for name, value in arms.items()
        if name != "C0"
    }
    c1_vs_c2_pairwise = exact_pairwise_attribution(y, arms["C2"], arms["C1"])
    margin_decomposition = [
        row
        for name, value in arms.items()
        if name != "C0"
        for row in _pair_margin_decomposition(y, baseline, value, name)
    ]
    movement_pair_decomposition = {
        name: _movement_pair_decomposition(y, baseline, value, name)
        for name, value in arms.items()
        if name != "C0"
    }
    interval_ids = query_interval_ids(pos, query_positions)
    interval_summary = _per_interval_summary(y, arms, pos, interval_ids, query_positions)
    age_summary = _query_age_summary(y, arms, pos, query_positions)
    pairwise_latest_interval = [
        row
        for name, value in arms.items()
        if name != "C0"
        for row in _pairwise_by_latest_interval(y, baseline, value, interval_ids, name)
    ]
    by_group = {
        name: _source_group_contribution_summary(y, group_values, baseline, value, name)
        for name, value in arms.items()
        if name != "C0"
    }
    quartile_rows = []
    for start in range(0, len(y), QUARTILE_SIZE):
        stop = min(start + QUARTILE_SIZE, len(y))
        mask = (pos >= start) & (pos < stop)
        subset_y = y[mask]
        metrics = {}
        for name, values in arms.items():
            auc, reason = _auc_or_missing(subset_y, values[mask])
            metrics[name] = {"auroc": auc, "unavailable_reason": reason}
        quartile_rows.append(
            {
                "position_start_inclusive": start,
                "position_end_exclusive": stop,
                "events": int(mask.sum()),
                "normal_events": int(np.count_nonzero(subset_y == 0)),
                "anomaly_events": int(np.count_nonzero(subset_y == 1)),
                "metrics": metrics,
            }
        )

    patch_summary = {
        name: patch_magnitude_summary(y, values)
        for name, values in patch_deltas.items()
    }
    argmax = _argmax_summary(dense_maps, top_patch_ids, patch_deltas)
    bootstrap = source_group_cluster_bootstrap(
        y,
        group_values,
        arms,
        replicates=bootstrap_replicates,
        seed=BOOTSTRAP_SEED,
    )
    c1_auc = float(arm_metrics["C1"]["image_auroc"])
    c2_auc = float(arm_metrics["C2"]["image_auroc"])
    gap = c1_auc - c2_auc
    amplitude_rows = []
    for amplitude in AMPLITUDES:
        arm_name = f"C2_a{amplitude:.2f}"
        auc = float(arm_metrics[arm_name]["image_auroc"])
        amplitude_rows.append(
            {
                "amplitude": amplitude,
                "auroc": auc,
                "auc_gain_vs_c2_a020": auc - c2_auc,
                "fraction_of_c1_minus_c2_auc_gap_recovered": (auc - c2_auc) / gap if gap else None,
                "movement": movement[arm_name],
                "pairwise": pairwise[arm_name],
            }
        )

    return _json_safe(
        {
            "schema": "r2-exp1.outcome_analysis.v1",
            "status": "COMPLETE",
            "development_only": True,
            "outcome_counts": {"normal": int(np.count_nonzero(y == 0)), "anomaly": int(np.count_nonzero(y == 1))},
            "event_count": len(y),
            "source_group_count": len(set(group_values)),
            "threshold": threshold,
            "metrics": arm_metrics,
            "movement": movement,
            "pairwise": pairwise,
            "c1_vs_c2_pairwise": c1_vs_c2_pairwise,
            "pairwise_by_abs_c0_margin": margin_decomposition,
            "pairwise_by_score_movement": movement_pair_decomposition,
            "pairwise_by_latest_query_interval": pairwise_latest_interval,
            "source_group_pair_contribution": by_group,
            "source_group_bootstrap": bootstrap,
            "query_intervals": interval_summary,
            "query_age_buckets": age_summary,
            "position_quartiles": quartile_rows,
            "patch_magnitude": patch_summary,
            "argmax_and_masking": argmax,
            "c2_amplitude_sensitivity": amplitude_rows,
            "query_feedback": _grouped_query_stats(query_details),
        }
    )


def _argmax_summary(
    dense_maps: Sequence[np.ndarray],
    top_patch_ids: Sequence[np.ndarray],
    patch_deltas: Mapping[str, Sequence[np.ndarray]],
) -> dict[str, Any]:
    if len(dense_maps) != len(top_patch_ids):
        raise ValueError("dense maps and Top32 ids must align")
    result = {}
    for name, rows in patch_deltas.items():
        if len(rows) != len(dense_maps):
            raise ValueError("patch delta arm does not align with dense maps")
        changed = 0
        any_patch_changed = 0
        unchanged_image_score = 0
        base_argmax_inside_top32 = 0
        candidate_argmax_changed = 0
        masked_by_unchanged_base_argmax = 0
        for base_map, ids, delta in zip(dense_maps, top_patch_ids, rows):
            base = np.asarray(base_map).reshape(-1)
            ids = np.asarray(ids, dtype=np.int64).reshape(-1)
            delta = np.asarray(delta, dtype=np.float64).reshape(-1)
            if len(ids) != len(delta) or np.any(ids < 0) or np.any(ids >= len(base)):
                raise ValueError("Top32 patch ids and deltas are inconsistent")
            corrected = base.copy()
            corrected[ids] += delta
            base_argmax = int(np.argmax(base))
            candidate_argmax = int(np.argmax(corrected))
            base_image = float(np.max(base))
            candidate_image = float(np.max(corrected))
            active = delta != 0
            has_patch_change = bool(active.any())
            any_patch_changed += int(has_patch_change)
            changed += int(candidate_image != base_image)
            unchanged_image_score += int(candidate_image == base_image)
            base_argmax_inside_top32 += int(base_argmax in set(ids.tolist()))
            candidate_argmax_changed += int(candidate_argmax != base_argmax)
            base_position = np.flatnonzero(ids == base_argmax)
            base_argmax_unchanged = not len(base_position) or delta[int(base_position[0])] == 0
            masked_by_unchanged_base_argmax += int(
                has_patch_change and candidate_image == base_image and candidate_argmax == base_argmax and base_argmax_unchanged
            )
        result[name] = {
            "events_with_any_nonzero_top32_patch_delta": any_patch_changed,
            "image_score_changed": changed,
            "image_score_unchanged": unchanged_image_score,
            "baseline_argmax_inside_top32": base_argmax_inside_top32,
            "corrected_argmax_changed": candidate_argmax_changed,
            "score_unchanged_due_to_unchanged_baseline_argmax": masked_by_unchanged_base_argmax,
        }
    return result


def _verify_repository_identity() -> dict[str, str]:
    identity = git_identity()
    allowed_heads = {EXPECTED_PARENT_HEAD}
    if identity["branch"] != EXPECTED_BRANCH or identity["head"] not in allowed_heads:
        raise RuntimeError(
            f"repository identity rejected: expected {EXPECTED_BRANCH}@{EXPECTED_PARENT_HEAD}, "
            f"found {identity['branch']}@{identity['head']}"
        )
    return identity


def _source_fingerprints(r1_manifest: Mapping[str, Any], perf_manifest: Mapping[str, Any]) -> dict[str, str]:
    expected_maps = (r1_manifest.get("source_hashes"), perf_manifest.get("source_fingerprints"))
    fingerprints: dict[str, str] = {}
    for expected in expected_maps:
        if not isinstance(expected, Mapping):
            raise RuntimeError("historical run is missing a source fingerprint map")
        for relative, expected_sha in expected.items():
            if not isinstance(relative, str) or not isinstance(expected_sha, str):
                raise RuntimeError("historical source fingerprint record is malformed")
            path = ROOT / relative
            if not path.is_file():
                raise RuntimeError(f"fingerprinted source is missing: {relative}")
            observed = sha256_file(path)
            if observed != expected_sha:
                raise RuntimeError(f"historical source fingerprint mismatch: {relative}")
            prior = fingerprints.setdefault(relative, observed)
            if prior != observed:
                raise RuntimeError(f"conflicting historical source fingerprints: {relative}")
    module_relative = Path(__file__).resolve().relative_to(ROOT).as_posix()
    fingerprints[module_relative] = sha256_file(Path(__file__))
    return dict(sorted(fingerprints.items()))


def _load_public_inputs() -> dict[str, Any]:
    """Verify R1, prior R2-PERF-1, identities, scores and query schedule."""
    reader = ReadOnlyR1Evidence(R1_ROOT)
    r1_manifest, r1_manifest_raw = reader.read_manifest()
    if (
        r1_manifest.get("schema") != "calibration.r1_evidence_tape.v1"
        or r1_manifest.get("events_processed") != 512
        or r1_manifest.get("query_events") != 17
        or r1_manifest.get("authorization_scope", {}).get("support_seed") != 0
        or r1_manifest.get("fixed_support_sha256") != "8a710d028509350568e976bc3ede66aa59603499cbea525bdd8579322b1a1558"
        or r1_manifest.get("fixed_support_raw_prototype_sha256") != "087feded0d780bbc3a56d45404b4126991b1aeb54f1ae3724bce025caf0d380e"
        or r1_manifest.get("query_positions_sha256") != "7f4aedf49f3e3bffd2482bd00d9e16512553b7cef8c0af7e002b47b60ddcb17f"
        or float(r1_manifest.get("c64_theta0_higher_q90", -1)) != THETA
    ):
        raise RuntimeError("R1 run manifest does not match the authorized switch/seed-0/512 receipt")
    r1_sources = r1_manifest.get("source_hashes")
    if not isinstance(r1_sources, Mapping):
        raise RuntimeError("R1 manifest has no historical source fingerprint map")
    for relative, expected in r1_sources.items():
        actual_path = ROOT / relative
        if not actual_path.is_file() or sha256_file(actual_path) != expected:
            raise RuntimeError(f"R1 source identity changed after the historical run: {relative}")

    events, tape_receipt = verify_public_tape(
        reader,
        expected_count=512,
        expected_query_digest=str(r1_manifest["query_positions_sha256"]),
        expected_fixed_bank_digest=str(r1_manifest["fixed_support_sha256"]),
    )
    sealed_identity_rows = [
        json.loads(line)
        for line in reader.read_authorized_bytes("sealed_scores.jsonl").splitlines()
        if line.strip()
    ]
    if len(sealed_identity_rows) != 512:
        raise RuntimeError("R1 public identity tape does not contain exactly 512 rows")
    for position, (row, event) in enumerate(zip(sealed_identity_rows, events)):
        if (
            row.get("position") != position
            or row.get("official_group_sha256") != event["source_group_sha256"]
            or row.get("fixed_support_sha256") != tape_receipt["fixed_support_sha256"]
            or not re.fullmatch(r"[0-9a-f]{64}", str(row.get("event_id", "")))
            or not re.fullmatch(r"[0-9a-f]{64}", str(row.get("source_image_identity_sha256", "")))
        ):
            raise RuntimeError(f"R1 public event identity is malformed or misaligned at {position}")

    perf_manifest_path = PERF_ROOT / "run_manifest.json"
    perf_manifest = _read_json(perf_manifest_path)
    if (
        perf_manifest.get("schema") != "calibration.r2_perf1.run_manifest.v1"
        or perf_manifest.get("run_id") != PERF_RUN_ID
        or perf_manifest.get("event_count") != 512
        or perf_manifest.get("execution_scope") != "switch only; seed 0; first 512 frozen TEST positions"
        or perf_manifest.get("historical_evidence_manifest_sha256") != hashlib.sha256(r1_manifest_raw).hexdigest()
        or perf_manifest.get("cuda_visible_devices") != ""
        or perf_manifest.get("gpu_inference") is not False
    ):
        raise RuntimeError("Prior R2-PERF-1 manifest fails its recorded evidence/scope contract")
    perf_seal = _read_json(PERF_ROOT / "stage_a_sealed.json")
    perf_parity = _read_json(PERF_ROOT / "stage_a_c2_parity.json")
    complete = _read_json(PERF_ROOT / "orchestrator_complete.json")
    supervisor = _read_json(PERF_ROOT / "supervisor_status.json")
    if (
        perf_seal.get("status") != "SEALED"
        or perf_seal.get("outcome_labels_read") is not False
        or perf_seal.get("event_count") != 512
        or perf_seal.get("r1_evidence_manifest_sha256") != hashlib.sha256(r1_manifest_raw).hexdigest()
        or perf_seal.get("fixed_support_sha256") != tape_receipt["fixed_support_sha256"]
        or perf_seal.get("query_positions_sha256") != tape_receipt["query_positions_sha256"]
        or perf_parity.get("status") != "PASS"
        or perf_parity.get("events_reconstructed") != 512
        or perf_parity.get("queried_records_consumed") not in (None, 17)
        or complete.get("status") != "COMPLETE"
        or supervisor.get("status") != "COMPLETED"
        or supervisor.get("termination_confirmed") is not True
        or supervisor.get("child_exit_code") != 0
    ):
        raise RuntimeError("Prior R2-PERF-1 completion receipts are inconsistent")
    if sha256_file(PERF_ROOT / "stage_a_scores.jsonl") != perf_seal.get("score_table_sha256"):
        raise RuntimeError("Prior R2-PERF-1 score table does not match its Stage A seal")
    perf_rows = _read_jsonl(PERF_ROOT / "stage_a_scores.jsonl")
    if len(perf_rows) != 512:
        raise RuntimeError("Prior R2-PERF-1 score table does not contain 512 records")
    for position, (prior, row, event) in enumerate(zip(perf_rows, sealed_identity_rows, events)):
        if (
            prior.get("position") != position
            or prior.get("event_id") != row["event_id"]
            or prior.get("source_image_identity_sha256") != row["source_image_identity_sha256"]
            or prior.get("source_group_sha256") != row["official_group_sha256"]
            or prior.get("scores", {}).get("c0") != event["scores"]["c0"]
            or prior.get("scores", {}).get("c1") != event["scores"]["c1"]
            or prior.get("scores", {}).get("c2") != event["scores"]["c2"]
        ):
            raise RuntimeError(f"Prior R2-PERF-1 score/identity trajectory differs at event {position}")
    if tape_receipt["query_count"] != 17 or tape_receipt["unique_source_group_count"] != 103:
        raise RuntimeError("Verified R1 query or Source Group count differs from prior receipt")
    source_fingerprints = _source_fingerprints(r1_manifest, perf_manifest)
    return {
        "reader": reader,
        "r1_manifest": r1_manifest,
        "r1_manifest_raw": r1_manifest_raw,
        "r1_manifest_sha256": hashlib.sha256(r1_manifest_raw).hexdigest(),
        "events": events,
        "sealed_identity_rows": sealed_identity_rows,
        "tape_receipt": tape_receipt,
        "perf_manifest": perf_manifest,
        "perf_seal": perf_seal,
        "perf_parity": perf_parity,
        "perf_rows": perf_rows,
        "source_fingerprints": source_fingerprints,
        "input_hashes": {
            "r1_run_manifest_sha256": hashlib.sha256(r1_manifest_raw).hexdigest(),
            "r1_sealed_scores_sha256": tape_receipt["sealed_scores_file_sha256"],
            "r1_transition_projection_sha256": tape_receipt["transition_structural_projection_sha256"],
            "r1_shard_receipt_list_sha256": _canonical_sha256(tape_receipt["verified_shards"]),
            "r2_perf1_run_manifest_sha256": sha256_file(perf_manifest_path),
            "r2_perf1_stage_a_seal_sha256": sha256_file(PERF_ROOT / "stage_a_sealed.json"),
            "r2_perf1_score_table_sha256": perf_seal["score_table_sha256"],
            "r2_perf1_c2_parity_receipt_sha256": sha256_file(PERF_ROOT / "stage_a_c2_parity.json"),
            "r2_perf1_stage_b_metrics_sha256": sha256_file(PERF_ROOT / "stage_b_metrics.json"),
            "r2_perf1_orchestrator_receipt_sha256": sha256_file(PERF_ROOT / "orchestrator_complete.json"),
        },
    }


def _current_process_peak_working_set() -> int | None:
    if os.name != "nt":
        return None
    import ctypes
    from ctypes import wintypes

    class PROCESS_MEMORY_COUNTERS(ctypes.Structure):
        _fields_ = [
            ("cb", wintypes.DWORD),
            ("PageFaultCount", wintypes.DWORD),
            ("PeakWorkingSetSize", ctypes.c_size_t),
            ("WorkingSetSize", ctypes.c_size_t),
            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
            ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
            ("PagefileUsage", ctypes.c_size_t),
            ("PeakPagefileUsage", ctypes.c_size_t),
        ]

    counters = PROCESS_MEMORY_COUNTERS()
    counters.cb = ctypes.sizeof(counters)
    psapi = ctypes.WinDLL("psapi", use_last_error=True)
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    handle = kernel32.GetCurrentProcess()
    if not psapi.GetProcessMemoryInfo(handle, ctypes.byref(counters), counters.cb):
        return None
    return int(counters.PeakWorkingSetSize)


def _require_worker(role: str, output_root: Path) -> tuple[OutputBudget, dict[str, Any]]:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    if sys.executable.lower() != r"d:\anaconda3\envs\torch\python.exe".lower():
        raise RuntimeError(f"worker must use the authorized interpreter, got {sys.executable}")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise RuntimeError("CUDA visibility is not disabled")
    if os.environ.get(ROLE_ENV) != role or not os.environ.get(NONCE_ENV):
        raise RuntimeError("worker role or supervised nonce is missing")
    if not r1_current_process_is_in_job():
        raise RuntimeError("worker is not contained in the Windows Job Object")
    identity = _verify_repository_identity()
    budget = OutputBudget(output_root, allow_existing=True)
    manifest = _read_json(output_root / "run_manifest.json")
    if (
        manifest.get("schema") != "calibration.r2_exp1.run_manifest.v1"
        or manifest.get("expected_repository") != identity
        or manifest.get("run_id") != output_root.name.removeprefix(RUN_PREFIX)
    ):
        raise RuntimeError("R2-EXP1 output manifest identity is invalid")
    return budget, manifest


def _stage_a(output_root: Path) -> dict[str, Any]:
    started = time.perf_counter()
    budget, run_manifest = _require_worker("stage-a", output_root)
    inputs = _load_public_inputs()
    _verify_launch_input_hashes(run_manifest)
    if inputs["source_fingerprints"] != run_manifest.get("source_fingerprints"):
        raise RuntimeError("source fingerprints differ from the launch manifest")
    reader = inputs["reader"]
    events = inputs["events"]
    sealed_rows = inputs["sealed_identity_rows"]
    transitions, transition_sha = reader.read_transition_structure()
    query_positions = list(inputs["tape_receipt"]["query_positions"])
    query_set = set(query_positions)
    feedback_path = require_plain_path(R1_ROOT / "private_feedback.jsonl", must_exist=True)
    correction_path = require_plain_path(R1_ROOT / "correction_anchor_events.jsonl", must_exist=True)
    ledgers = _QueriedLedgerReader(feedback_path, correction_path)
    anchors: list[dict[str, Any]] = []
    query_details: list[dict[str, Any]] = []
    c2_delta_max_abs = 0.0
    c2_score_max_abs = 0.0
    c2_amplitude_020_exact_matches = 0
    rows_written = 0

    for position, (event, identity_row, transition, prior_row) in enumerate(
        zip(events, sealed_rows, transitions, inputs["perf_rows"])
    ):
        arrays = event["arrays"]
        group = str(event["source_group_sha256"])
        if _c2_state_sha256(anchors) != identity_row["c2_state_sha256_before"]:
            raise RuntimeError(f"C2 pre-event state differs from historical R1 at {position}")
        eligible_slots = _validate_c2_anchor_state(
            anchors, event_position=position, phase="pre_score", event_group=group
        )
        replay_delta, replay_eligible = signed_knn_delta(
            np.asarray(arrays["top32_features"], dtype=np.float32),
            anchors,
            event_position=position,
            event_group=group,
            neighbors=5,
            amplitude=0.20,
        )
        if replay_eligible != len(eligible_slots):
            raise RuntimeError(f"C2 eligible-anchor count differs at {position}")
        delta_error = float(
            np.max(np.abs(replay_delta.astype(np.float64) - arrays["c2_delta"].astype(np.float64)))
        )
        c2_delta_max_abs = max(c2_delta_max_abs, delta_error)
        if delta_error > 1e-7:
            raise RuntimeError(f"historical C2 patch vector differs at event {position}")
        replay_score = max_score(
            corrected_dense_map(arrays["base_patch_map"], arrays["top32_ids"], replay_delta)
        )
        score_error = abs(replay_score - float(event["scores"]["c2"]))
        c2_score_max_abs = max(c2_score_max_abs, score_error)
        if score_error > 1e-7 or replay_score != float(prior_row["scores"]["c2"]):
            raise RuntimeError(f"historical C2 score differs from R1/R2-PERF-1 at {position}")

        score_record: dict[str, Any] = {
            "schema": "calibration.r2_perf1.score_row.v1",
            "position": position,
            "event_id": str(identity_row["event_id"]),
            "source_image_identity_sha256": str(identity_row["source_image_identity_sha256"]),
            "source_group_sha256": group,
            "scores": {
                "c0": float(event["scores"]["c0"]),
                "c1": float(event["scores"]["c1"]),
                "c2": float(event["scores"]["c2"]),
            },
            "argmax_patch": {
                "c0": int(np.argmax(arrays["base_patch_map"])),
                "c1": int(np.argmax(event["maps"]["c1"])),
                "c2": int(np.argmax(event["maps"]["c2"])),
            },
        }
        for amplitude in AMPLITUDES:
            scale = np.float32(float(amplitude) / 0.20)
            scaled_delta = np.asarray(arrays["c2_delta"], dtype=np.float32) * scale
            corrected = corrected_dense_map(arrays["base_patch_map"], arrays["top32_ids"], scaled_delta)
            score = max_score(corrected)
            key = f"C2_a{amplitude:.2f}"
            score_record["scores"][key] = score
            score_record["argmax_patch"][key] = int(np.argmax(corrected))
            if amplitude == 0.20:
                c2_amplitude_020_exact_matches += int(
                    score == float(event["scores"]["c2"])
                    and np.array_equal(scaled_delta, arrays["c2_delta"])
                )
        # Seal this event's score before consuming any queried feedback at this
        # position, preserving the original predict -> query -> reveal order.
        budget.write(
            "stage_a_scores.jsonl",
            (json.dumps(score_record, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode(),
            append=True,
        )
        rows_written += 1
        if position in query_set:
            feedback, correction, feedback_sha = ledgers.reveal(
                position=position,
                expected_event_id=str(identity_row["event_id"]),
                expected_image_identity=str(identity_row["source_image_identity_sha256"]),
                expected_group=group,
                expected_c0_score=float(event["scores"]["c0"]),
                theta=THETA,
            )
            outcome = str(correction["outcome"])
            stats = correction.get("c1_stats")
            if not isinstance(stats, Mapping):
                raise RuntimeError(f"C1 update record lacks existing statistics at query {position}")
            query_details.append(
                {
                    "position": position,
                    "event_id": str(identity_row["event_id"]),
                    "source_image_identity_sha256": str(identity_row["source_image_identity_sha256"]),
                    "source_group_sha256": group,
                    "queried_outcome": "anomaly" if int(feedback["label"]) == 1 else "normal",
                    "queried_label": int(feedback["label"]),
                    "c0_prediction": int(correction["c0_prediction"]),
                    "c0_score": float(correction["c0_score"]),
                    "threshold": float(correction["c0_threshold"]),
                    "outcome": outcome,
                    "candidates_examined": 32,
                    "native_gate_kept": int(correction.get("c1_kept_by_native_gate", stats.get("kept", 0))),
                    "inserted": int(stats.get("inserted", 0)),
                    "merged": int(stats.get("merged", 0)),
                    "replaced": int(stats.get("replaced", 0)),
                    "skipped": int(stats.get("skipped", 0)),
                    "same_sign_merged": int(stats.get("same_sign_merged", 0)),
                    "same_sign_replaced": int(stats.get("same_sign_replaced", 0)),
                    "cross_sign_replaced": int(stats.get("cross_sign_replaced", 0)),
                    "corr_bank_size": int(stats.get("corr_bank_size", 0)),
                    "c1_stats_recorded": {str(key): int(value) for key, value in stats.items()},
                    "c1_same_source_excluded_anchor_slots": int(event["c1_same_source_excluded_anchor_slots"]),
                    "c1_eligible_anchor_slots": int(event["eligible"]["c1"]),
                    "c2_eligible_anchor_count_before": int(event["eligible"]["c2"]),
                    "c2_additions": int(correction["c2_added_patch_rows"]),
                    "c2_retained_after_query": min(256, len(query_details) * 32),
                    "c2_retention": str(correction["c2_retention"]),
                    "c2_state_sha256_after": str(transition["c2_state_sha256_after"]),
                    "feedback_record_sha256": feedback_sha,
                }
            )
            # The historical R2-PERF-1 writer flushed and sealed the score before
            # reveal. Here the complete Stage A tape is sealed before Stage B can
            # open offline outcomes; query feedback remains restricted to its own
            # record at the original query position.
            sign = 1 if int(feedback["label"]) == 1 else -1
            anchors = append_queried_c2_anchors(
                anchors,
                event_position=position,
                event_group=group,
                top32_ids=arrays["top32_ids"],
                top32_features=arrays["top32_features"],
                sign=sign,
                capacity=256,
            )
        if _c2_state_sha256(anchors) != transition["c2_state_sha256_after"]:
            raise RuntimeError(f"C2 post-event state differs from historical R1 at {position}")
        if len(anchors) > 256:
            raise RuntimeError("C2 FIFO state exceeded its frozen capacity")
    ledger_receipt = ledgers.finish(17)
    if len(query_details) != 17 or rows_written != 512:
        raise RuntimeError("Stage A event/query output counts do not match the frozen tape")
    query_digest = _canonical_sha256(query_positions)
    if query_digest != inputs["tape_receipt"]["query_positions_sha256"]:
        raise RuntimeError("Query position digest changed during Stage A")
    raw_score_path = output_root / "stage_a_scores.jsonl"
    score_sha = sha256_file(raw_score_path)
    c2_report = {
        "schema": "r2-exp1.stage_a.c2_replay.v1",
        "status": "PASS",
        "events_reconstructed": rows_written,
        "query_events": len(query_details),
        "c2_patch_delta_max_absolute_error": c2_delta_max_abs,
        "c2_image_score_max_absolute_error": c2_score_max_abs,
        "c2_a020_exact_score_and_patch_matches": c2_amplitude_020_exact_matches,
        "c2_final_state_sha256": _c2_state_sha256(anchors),
        "c2_final_retained_anchors": len(anchors),
        "query_positions_sha256": query_digest,
        "ledger_read_receipt": ledger_receipt,
        "query_event_details_private": query_details,
        "offline_test_outcomes_read": False,
        "queried_feedback_labels_read": 17,
    }
    budget.write_json("stage_a_c2_parity.json", c2_report)
    stage_a_diagnostics = {
        "schema": "r2-exp1.stage_a.input_verification.v1",
        "status": "PASS",
        "event_count": 512,
        "query_count": 17,
        "source_group_count": 103,
        "fixed_support_sha256": inputs["tape_receipt"]["fixed_support_sha256"],
        "fixed_support_raw_prototype_sha256": inputs["r1_manifest"]["fixed_support_raw_prototype_sha256"],
        "r1_score_reconstruction_exact": True,
        "prior_r2_perf1_score_identity_alignment_exact": True,
        "prior_r2_perf1_initial_manifest_status": inputs["perf_manifest"].get("status"),
        "prior_r2_perf1_orchestrator_status": "COMPLETE",
        "r1_tape_receipt": {
            key: value for key, value in inputs["tape_receipt"].items() if key != "verified_shards"
        },
        "input_hashes": inputs["input_hashes"],
        "transition_projection_sha256": transition_sha,
        "source_fingerprints": inputs["source_fingerprints"],
        "stage_a_score_table_sha256": score_sha,
        "stage_a_elapsed_seconds": time.perf_counter() - started,
        "stage_a_peak_working_set_bytes": _current_process_peak_working_set(),
        "offline_test_outcomes_read": False,
        "image_decoding_or_dino_loaded": False,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }
    budget.write_json("stage_a_diagnostics.json", stage_a_diagnostics)
    seal = {
        "schema": "r2-exp1.stage_a_score_seal.v1",
        "status": "SEALED",
        "event_count": rows_written,
        "offline_test_outcomes_read": False,
        "queried_feedback_records_read_at_frozen_positions": 17,
        "query_positions_sha256": query_digest,
        "fixed_support_sha256": inputs["tape_receipt"]["fixed_support_sha256"],
        "r1_manifest_sha256": inputs["r1_manifest_sha256"],
        "r1_transition_projection_sha256": transition_sha,
        "score_table_sha256": score_sha,
        "stage_a_diagnostics_sha256": sha256_file(output_root / "stage_a_diagnostics.json"),
        "stage_a_c2_parity_sha256": sha256_file(output_root / "stage_a_c2_parity.json"),
        "source_fingerprints": inputs["source_fingerprints"],
        "sealed_utc_epoch": time.time(),
    }
    budget.write_json("stage_a_sealed.json", seal)
    return seal


def _expected_prior_primary_metrics() -> Mapping[str, Any]:
    previous = _read_json(PERF_ROOT / "stage_b_metrics.json")
    primary = previous.get("primary_methods")
    if not isinstance(primary, Mapping) or not {"C0", "C1", "C2"}.issubset(primary):
        raise RuntimeError("prior R2-PERF-1 primary metric receipt is malformed")
    if (
        previous.get("event_count") != 512
        or previous.get("normal_count") != 153
        or previous.get("defect_count") != 359
        or previous.get("unique_source_group_count") != 103
        or float(previous.get("frozen_threshold", -1)) != THETA
    ):
        raise RuntimeError("prior R2-PERF-1 outcome summary differs from its frozen scope")
    return primary


def _compare_primary_metrics(current: Mapping[str, Any], previous: Mapping[str, Any]) -> None:
    for method in ("C0", "C1", "C2"):
        left = current[method]
        right = previous[method]
        for field in ("image_auroc", "image_auprc_average_precision"):
            if not math.isclose(float(left[field]), float(right[field]), rel_tol=0.0, abs_tol=2e-15):
                raise RuntimeError(f"{method} historical metric differs from prior R2-PERF-1: {field}")
        for field in ("tp", "fp", "fn", "tn", "fpr", "fnr", "balanced_accuracy"):
            observed = left["fixed_threshold"][field]
            expected = right["fixed_threshold"][field]
            if isinstance(observed, float) or isinstance(expected, float):
                if not math.isclose(float(observed), float(expected), rel_tol=0.0, abs_tol=2e-15):
                    raise RuntimeError(f"{method} historical threshold metric differs: {field}")
            elif observed != expected:
                raise RuntimeError(f"{method} historical threshold count differs: {field}")


def _stage_b(output_root: Path, *, bootstrap_replicates: int = BOOTSTRAP_REPLICATES) -> dict[str, Any]:
    started_wall = time.perf_counter()
    started_cpu = time.process_time()
    budget, run_manifest = _require_worker("stage-b", output_root)
    seal = _read_json(output_root / "stage_a_sealed.json")
    if (
        seal.get("schema") != "r2-exp1.stage_a_score_seal.v1"
        or seal.get("status") != "SEALED"
        or seal.get("offline_test_outcomes_read") is not False
        or seal.get("event_count") != 512
        or sha256_file(output_root / "stage_a_scores.jsonl") != seal.get("score_table_sha256")
    ):
        raise RuntimeError("Stage B refused an absent or invalid Stage A score seal")
    _verify_repository_identity()
    source_fingerprints = seal.get("source_fingerprints")
    if not isinstance(source_fingerprints, Mapping):
        raise RuntimeError("Stage A seal has no source fingerprint map")
    for relative, expected in source_fingerprints.items():
        path = ROOT / str(relative)
        if not path.is_file() or sha256_file(path) != expected:
            raise RuntimeError(f"source changed after Stage A seal: {relative}")
    score_rows = _read_jsonl(output_root / "stage_a_scores.jsonl")
    if len(score_rows) != 512:
        raise RuntimeError("Stage A score seal does not contain exactly 512 rows")
    for position, row in enumerate(score_rows):
        if (
            row.get("schema") != "calibration.r2_perf1.score_row.v1"
            or row.get("position") != position
            or len(str(row.get("event_id", ""))) != 64
            or len(str(row.get("source_image_identity_sha256", ""))) != 64
            or len(str(row.get("source_group_sha256", ""))) != 64
            or not {"c0", "c1", "c2", *(f"C2_a{a:.2f}" for a in AMPLITUDES)}.issubset(row.get("scores", {}))
        ):
            raise RuntimeError(f"malformed Stage A score/identity row at {position}")

    # This is the first point at which EXP1 reads offline TEST outcomes. The
    # existing R2-PERF-1 projection checks each first-512 TEST identity/group
    # against this sealed Stage A score table before exposing anomaly_class.
    labels_list, groups_list, alignment = _read_outcomes(
        score_rows,
        test_json_path=Path(r"G:\dataset\realiad\jsons\realiad_jsons\switch.json"),
    )
    labels = np.asarray(labels_list, dtype=np.int8)
    source_groups = [str(group) for group in groups_list]
    if (
        len(labels) != 512
        or int(np.count_nonzero(labels == 0)) != 153
        or int(np.count_nonzero(labels == 1)) != 359
        or len(set(source_groups)) != 103
        or alignment.get("status") != "512_TEST_OUTCOMES_IDENTITY_POSITION_VERIFIED"
        or alignment.get("train_labels_materialized") is not False
        or alignment.get("test_labels_materialized_outside_target_prefix") is not False
        or alignment.get("mask_fields_materialized") is not False
    ):
        raise RuntimeError("Stage B exact outcome alignment/count contract failed")
    if source_groups != [str(row["source_group_sha256"]) for row in score_rows]:
        raise RuntimeError("Outcome evaluator source groups differ from sealed Stage A source groups")
    if int(seal.get("queried_feedback_records_read_at_frozen_positions", 0)) != 17:
        raise RuntimeError("Stage A query feedback receipt is incomplete")

    private_query_record = _read_json(output_root / "stage_a_c2_parity.json")
    if (
        private_query_record.get("schema") != "r2-exp1.stage_a.c2_replay.v1"
        or private_query_record.get("status") != "PASS"
        or sha256_file(output_root / "stage_a_c2_parity.json") != seal.get("stage_a_c2_parity_sha256")
    ):
        raise RuntimeError("Stage A queried-feedback detail receipt failed its seal")
    query_details = private_query_record.get("query_event_details_private")
    if not isinstance(query_details, list) or len(query_details) != 17:
        raise RuntimeError("Stage A private query detail table does not have 17 rows")
    query_positions = [int(row["position"]) for row in query_details]
    if _canonical_sha256(query_positions) != seal.get("query_positions_sha256"):
        raise RuntimeError("Private query detail positions differ from the frozen query digest")
    for query in query_details:
        position = int(query["position"])
        if (
            int(labels[position]) != int(query["queried_label"])
            or query["event_id"] != score_rows[position]["event_id"]
            or query["source_image_identity_sha256"] != score_rows[position]["source_image_identity_sha256"]
            or query["source_group_sha256"] != score_rows[position]["source_group_sha256"]
        ):
            raise RuntimeError(f"queried feedback and offline identity/outcome differ at {position}")

    inputs = _load_public_inputs()
    _verify_launch_input_hashes(run_manifest)
    if inputs["source_fingerprints"] != run_manifest.get("source_fingerprints"):
        raise RuntimeError("Stage B sources differ from the original launch fingerprints")
    events = inputs["events"]
    score_arms = {
        "C0": np.asarray([float(row["scores"]["c0"]) for row in score_rows], dtype=np.float64),
        "C1": np.asarray([float(row["scores"]["c1"]) for row in score_rows], dtype=np.float64),
        "C2": np.asarray([float(row["scores"]["c2"]) for row in score_rows], dtype=np.float64),
    }
    for amplitude in AMPLITUDES:
        arm = f"C2_a{amplitude:.2f}"
        score_arms[arm] = np.asarray([float(row["scores"][arm]) for row in score_rows], dtype=np.float64)
    if not np.array_equal(score_arms["C2"], score_arms["C2_a0.20"]):
        raise RuntimeError("C2 amplitude 0.20 score path differs from historical C2")
    current_metrics = {name: _metric_record(labels, scores, THETA) for name, scores in score_arms.items()}
    _compare_primary_metrics(current_metrics, _expected_prior_primary_metrics())

    patch_deltas: dict[str, list[np.ndarray]] = {name: [] for name in score_arms if name != "C0"}
    dense_maps: list[np.ndarray] = []
    top_ids: list[np.ndarray] = []
    positions = np.arange(512, dtype=np.int64)
    private_event_rows = []
    for position, (event, score_row) in enumerate(zip(events, score_rows)):
        arrays = event["arrays"]
        dense_maps.append(np.asarray(arrays["base_patch_map"], dtype=np.float32))
        top_ids.append(np.asarray(arrays["top32_ids"], dtype=np.int64))
        patch_deltas["C1"].append(np.asarray(arrays["c1_delta"], dtype=np.float32))
        patch_deltas["C2"].append(np.asarray(arrays["c2_delta"], dtype=np.float32))
        for amplitude in AMPLITUDES:
            arm = f"C2_a{amplitude:.2f}"
            patch_deltas[arm].append(
                np.asarray(arrays["c2_delta"], dtype=np.float32) * np.float32(amplitude / 0.20)
            )
        last_query_index = int(np.searchsorted(query_positions, position, side="left") - 1)
        interval_id = int(np.searchsorted(query_positions, position, side="left"))
        prediction = int(float(score_row["scores"]["c0"]) >= THETA)
        label = int(labels[position])
        outcome = "TN" if label == 0 and prediction == 0 else (
            "FP" if label == 0 else "FN" if prediction == 0 else "TP"
        )
        private_event_rows.append(
            {
                "position": position,
                "event_id": score_row["event_id"],
                "source_image_identity_sha256": score_row["source_image_identity_sha256"],
                "source_group_sha256": score_row["source_group_sha256"],
                "offline_label": label,
                "c0_prediction": prediction,
                "c0_outcome": outcome,
                "query_interval_id": interval_id,
                "previous_query_position": query_positions[last_query_index] if last_query_index >= 0 else None,
                "scores": score_row["scores"],
                "image_score_deltas_vs_c0": {
                    arm: float(value[position] - score_arms["C0"][position])
                    for arm, value in score_arms.items()
                    if arm != "C0"
                },
            }
        )
    analysis = analyze_outcomes(
        labels=labels,
        scores=score_arms,
        source_groups=source_groups,
        positions=positions,
        query_positions=query_positions,
        threshold=THETA,
        patch_deltas=patch_deltas,
        dense_maps=dense_maps,
        top_patch_ids=top_ids,
        query_details=query_details,
        bootstrap_replicates=bootstrap_replicates,
    )
    if analysis["pairwise"]["C1"]["pair_count"] != 359 * 153:
        raise RuntimeError("Exact C1 AUROC pair audit did not produce 54,927 pairs")
    for arm, result in analysis["pairwise"].items():
        if not math.isclose(
            float(result["candidate_auc_from_pairs"]),
            float(analysis["metrics"][arm]["image_auroc"]),
            rel_tol=0.0,
            abs_tol=2e-15,
        ):
            raise RuntimeError(f"pairwise AUROC disagrees with evaluator for {arm}")
    amplitude_metrics = []
    c0_auc = float(analysis["metrics"]["C0"]["image_auroc"])
    c1_auc = float(analysis["metrics"]["C1"]["image_auroc"])
    for row in analysis["c2_amplitude_sensitivity"]:
        arm = f"C2_a{float(row['amplitude']):.2f}"
        fixed = analysis["metrics"][arm]["fixed_threshold"]
        amplitude_metrics.append(
            {
                **row,
                "image_auprc_average_precision": analysis["metrics"][arm]["image_auprc_average_precision"],
                "auroc_difference_vs_c0": float(row["auroc"] - c0_auc),
                "auroc_difference_vs_c1": float(row["auroc"] - c1_auc),
                "fpr": fixed["fpr"],
                "fnr": fixed["fnr"],
                "balanced_accuracy": fixed["balanced_accuracy"],
                "fixed_threshold_confusion": {
                    key: fixed[key] for key in ("tp", "tn", "fp", "fn", "threshold")
                },
                "patch_magnitude": analysis["patch_magnitude"][arm],
                "image_magnitude": analysis["movement"][arm]["image_score_delta"],
            }
        )
    analysis["c2_amplitude_sensitivity"] = amplitude_metrics
    analysis["c1_vs_c2_magnitude_comparison"] = {
        "C1_image_magnitude": analysis["movement"]["C1"]["image_score_delta"],
        "C2_image_magnitude": analysis["movement"]["C2"]["image_score_delta"],
        "C1_top32_patch_magnitude": analysis["patch_magnitude"]["C1"],
        "C2_top32_patch_magnitude": analysis["patch_magnitude"]["C2"],
        "image_and_patch_magnitudes_are_distinct_quantities": True,
    }
    analysis["conditional_fixed_trajectory"] = True
    analysis["independent_experiment_samples"] = 103
    analysis["correlated_image_pairs_not_counted_as_independent_samples"] = True
    analysis["source_codebook_state_evidence"] = {
        "per_anchor_beta": "NOT_IDENTIFIABLE_FROM_EXISTING_TAPE",
        "per_anchor_kappa": "NOT_IDENTIFIABLE_FROM_EXISTING_TAPE",
        "per_anchor_radius": "NOT_IDENTIFIABLE_FROM_EXISTING_TAPE",
        "hidden_codebook_prototypes": "NOT_IDENTIFIABLE_FROM_EXISTING_TAPE",
        "unrecorded_state_changes": "NOT_IDENTIFIABLE_FROM_EXISTING_TAPE",
        "query_event_aggregate_statistics": "RECORDED_IN_CORRECTION_EVENT_LEDGER",
    }
    analysis["outcome_alignment"] = alignment
    analysis["prior_metric_reproduction"] = {
        "status": "PASS",
        "methods": {name: current_metrics[name] for name in ("C0", "C1", "C2")},
        "compared_with_existing_r2_perf1_primary_metrics": True,
    }
    analysis["historical_final_c1_active_anchors_by_sign"] = {
        "positive": int(inputs["r1_manifest"].get("c1_active_positive_anchors", 0)),
        "negative": int(inputs["r1_manifest"].get("c1_active_negative_anchors", 0)),
    }
    query_intervals = analysis["query_intervals"]
    query_index_by_position = {position: index for index, position in enumerate(query_positions)}
    for query in query_details:
        next_interval = min(
            query_index_by_position[int(query["position"])] + 1,
            len(query_intervals) - 1,
        )
        query["subsequent_interval_id"] = next_interval
        query["subsequent_interval_metrics"] = query_intervals[next_interval]["metrics"]

    analysis["hypothesis_ranking"] = _rank_hypotheses(analysis, query_details, events)
    private_diagnostics = {
        "schema": "r2-exp1.private_event_diagnostics.v1",
        "analysis": analysis,
        "event_level_outcomes_and_identities_private": private_event_rows,
        "query_event_details_private": query_details,
        "not_for_version_control": True,
    }
    budget.write_json("stage_b_diagnostics.json", private_diagnostics)
    budget.write_json("stage_b_metrics.json", analysis["metrics"])
    budget.write_json("stage_b_sensitivity.json", analysis["c2_amplitude_sensitivity"])

    sanitized = _sanitized_aggregate(analysis, inputs, run_manifest, output_root)
    sanitized["stage_b_runtime"] = {
        "wall_seconds": time.perf_counter() - started_wall,
        "cpu_seconds": time.process_time() - started_cpu,
        "peak_working_set_bytes": _current_process_peak_working_set(),
    }
    budget.write_json("stage_b_report.json", sanitized)
    sanitized["report_sha256"] = sha256_file(output_root / "stage_b_report.json")
    sanitized["analysis_runtime_seconds"] = time.perf_counter() - started_wall
    sanitized["analysis_cpu_seconds"] = time.process_time() - started_cpu
    sanitized["peak_working_set_bytes"] = _current_process_peak_working_set()
    # The file itself was sealed by OutputBudget before the counters were
    # finalized; runtime counters are carried in orchestrator_complete.json.
    return {
        "sanitized_report": sanitized,
        "analysis": analysis,
        "runtime": {
            "wall_seconds": time.perf_counter() - started_wall,
            "cpu_seconds": time.process_time() - started_cpu,
            "peak_working_set_bytes": _current_process_peak_working_set(),
        },
    }


def _rank_hypotheses(
    analysis: Mapping[str, Any],
    query_details: Sequence[Mapping[str, Any]],
    events: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    c1_auc = float(analysis["metrics"]["C1"]["image_auroc"])
    c2_auc = float(analysis["metrics"]["C2"]["image_auroc"])
    gap = c1_auc - c2_auc
    query_totals = analysis["query_feedback"]["total"]
    gate_kept = int(query_totals["native_gate_kept"])
    candidates = int(query_totals["candidates_examined"])
    admission_fraction = gate_kept / candidates if candidates else None
    amplitude_fractions = [
        float(row["fraction_of_c1_minus_c2_auc_gap_recovered"])
        for row in analysis["c2_amplitude_sensitivity"]
        if float(row["amplitude"]) < 0.20
    ]
    max_uniform_amplitude_fraction = max(amplitude_fractions, default=None)
    governance_changes = int(query_totals["merged"] + query_totals["replaced"])
    far_patch_total = sum(int(event["c1_far_field_patch_count"]) for event in events)
    far_events = sum(int(event["c1_far_field_patch_count"] > 0) for event in events)
    active_coverage = analysis["patch_magnitude"]

    h1_present = bool(admission_fraction is not None and admission_fraction < 1.0)
    h2_explains_most = bool(
        max_uniform_amplitude_fraction is not None and max_uniform_amplitude_fraction >= 0.5
    )
    h3_present = governance_changes > 0
    h4_present = far_patch_total > 0
    hypotheses = [
        {
            "id": "H1",
            "name": "Selective Anchor Admission",
            "evidence_class": "SUPPORTED_BUT_CONFOUNDED" if h1_present and gap > 0 else "WEAK_EVIDENCE",
            "direct_identifiability_label": "DIRECTLY_OBSERVED",
            "directly_observed": {
                "top32_candidates_examined": candidates,
                "native_gate_kept": gate_kept,
                "native_gate_keep_fraction": admission_fraction,
                "inserted_merged_replaced_skipped_counts": {
                    key: query_totals[key] for key in ("inserted", "merged", "replaced", "skipped")
                },
            },
            "supporting_empirical_evidence": (
                "C1 kept a strict subset of each queried Top-32 candidate set while C1 exceeded C2 AUROC."
                if h1_present and gap > 0 else "The tape does not show a selective gate associated with a C1 AUROC advantage."
            ),
            "contradictory_or_limit_evidence": (
                "Admission counts do not reveal whether rejected or retained anchors were useful; C1 also differs in magnitude and retention."
            ),
            "not_identifiable": [
                "per-anchor reliability counterfactuals",
                "C1 score path with admission disabled and all other state held fixed",
            ],
            "experiment_2_intervention_required": True,
        },
        {
            "id": "H2",
            "name": "Outcome-aware and margin-dependent Correction Magnitudes",
            "evidence_class": (
                "SUPPORTED_BUT_CONFOUNDED" if h2_explains_most else
                "CONTRADICTED_IN_THIS_DEVELOPMENT_RUN"
            ),
            "direct_identifiability_label": "DIRECTLY_OBSERVED",
            "directly_observed": {
                "predeclared_uniform_c2_amplitude_sweep": analysis["c2_amplitude_sensitivity"],
                "maximum_fraction_of_c1_minus_c2_auc_gap_recovered_by_a_lt_0_20": max_uniform_amplitude_fraction,
                "per_anchor_beta": "NOT_IDENTIFIABLE_FROM_EXISTING_TAPE",
                "per_anchor_kappa": "NOT_IDENTIFIABLE_FROM_EXISTING_TAPE",
            },
            "supporting_empirical_evidence": (
                "Uniformly reducing C2 amplitude recovered at least half of the C1-minus-C2 AUROC gap on this fixed trajectory."
                if h2_explains_most else
                "The frozen-amplitude counterfactuals directly test whether simple uniform downscaling explains most of the gap."
            ),
            "contradictory_or_limit_evidence": (
                "Uniform scaling is not the same intervention as C1's outcome- and margin-dependent beta policy; per-anchor beta is not recorded."
                if h2_explains_most else
                "No predeclared amplitude below 0.20 recovered half of the observed C1-minus-C2 AUROC gap; this only contradicts uniform downscaling as a sufficient explanation."
            ),
            "not_identifiable": [
                "per-anchor beta values and their outcome/margin dependence",
                "C1 trajectory with only correction strength replaced by a fixed value",
            ],
            "experiment_2_intervention_required": True,
        },
        {
            "id": "H3",
            "name": "Sign-aware Merging and Retention",
            "evidence_class": "SUPPORTED_BUT_CONFOUNDED" if h3_present and gap > 0 else "WEAK_EVIDENCE",
            "direct_identifiability_label": "DIRECTLY_OBSERVED",
            "directly_observed": {
                "merged": int(query_totals["merged"]),
                "replaced": int(query_totals["replaced"]),
                "same_sign_merged": int(sum(int(row.get("same_sign_merged", 0)) for row in query_details)),
                "same_sign_replaced": int(sum(int(row.get("same_sign_replaced", 0)) for row in query_details)),
                "cross_sign_replaced": int(sum(int(row.get("cross_sign_replaced", 0)) for row in query_details)),
                "c2_fifo_capacity": 256,
                "c2_final_retained": 256,
            },
            "supporting_empirical_evidence": (
                "The correction ledger records sign-aware merges/replacements during a trajectory where C1 exceeds C2 AUROC."
                if h3_present and gap > 0 else "No sign-aware merge/replacement activity was recorded."
            ),
            "contradictory_or_limit_evidence": "The tape does not include matched C1 FIFO or C2 sign-aware-retention counterfactuals.",
            "not_identifiable": [
                "which retained prototype caused a later score change",
                "C1 memory with retention policy swapped while all other components remain fixed",
            ],
            "experiment_2_intervention_required": True,
        },
        {
            "id": "H4",
            "name": "Broader Gaussian Influence Propagation",
            "evidence_class": "SUPPORTED_BUT_CONFOUNDED" if h4_present and gap > 0 else "WEAK_EVIDENCE",
            "direct_identifiability_label": "DIRECTLY_OBSERVED",
            "directly_observed": {
                "events_with_recorded_c1_far_field_patches": far_events,
                "c1_far_field_patch_count_total": far_patch_total,
                "c1_top32_patch_nonzero_fraction": active_coverage["C1"]["all"]["nonzero_fraction"],
                "c2_top32_patch_nonzero_fraction": active_coverage["C2"]["all"]["nonzero_fraction"],
            },
            "supporting_empirical_evidence": (
                "The historical tape records nonzero C1 far-field patch activity while C1 exceeds C2 AUROC."
                if h4_present and gap > 0 else "No C1 far-field patch activity was recorded."
            ),
            "contradictory_or_limit_evidence": "Activation coverage and AUROC are trajectory-level associations, not a propagation-only ablation.",
            "not_identifiable": [
                "the score trajectory with C1 influence truncated to the local neighborhood",
                "the unique AUROC contribution of the Gaussian kernel",
            ],
            "experiment_2_intervention_required": True,
        },
    ]

    # The predeclared uniform amplitude table decides whether H2 explains most
    # of the gap. H1 is the other mechanism with directly recorded gate behavior.
    ordered = []
    for condition, hypothesis_id in (
        (h2_explains_most, "H2"),
        (h1_present, "H1"),
        (h3_present, "H3"),
        (h4_present, "H4"),
    ):
        if condition and hypothesis_id not in ordered:
            ordered.append(hypothesis_id)
    for hypothesis_id in ("H1", "H2", "H3", "H4"):
        if hypothesis_id not in ordered:
            ordered.append(hypothesis_id)
    recommended = []
    if h1_present:
        recommended.append("H1")
    if h2_explains_most and len(recommended) < 2:
        recommended.append("H2")
    if not h2_explains_most and h3_present and len(recommended) < 2:
        recommended.append("H3")
    if h4_present and len(recommended) < 2:
        recommended.append("H4")
    rank_by_id = {hypothesis_id: rank for rank, hypothesis_id in enumerate(ordered, start=1)}
    for rank, hypothesis in enumerate(hypotheses, start=1):
        hypothesis["rank"] = rank_by_id[hypothesis["id"]]
        hypothesis["experiment_2_priority"] = (
            "PRIMARY" if hypothesis["id"] in recommended else
            "LOWER"
        )
    return {
        "ranking": ordered,
        "evidence_labels_allowed": [
            "DIRECTLY_OBSERVED", "SUPPORTED_BUT_CONFOUNDED", "WEAK_EVIDENCE",
            "NOT_IDENTIFIABLE", "CONTRADICTED_IN_THIS_DEVELOPMENT_RUN",
        ],
        "causal_confirmation_claim": False,
        "recommended_experiment_2_primary_mechanisms": recommended,
        "hypotheses": hypotheses,
        "uniform_amplitude_explains_most_of_c1_advantage": h2_explains_most,
        "interpretation": "mechanism evidence is development-set and conditional on one historical trajectory",
    }


def _sanitized_aggregate(
    analysis: Mapping[str, Any],
    inputs: Mapping[str, Any],
    run_manifest: Mapping[str, Any],
    output_root: Path,
) -> dict[str, Any]:
    sanitized_analysis = json.loads(json.dumps(_json_safe(analysis), allow_nan=False))

    private_event_fields = {
        "event_id",
        "source_image_identity_sha256",
        "source_group_sha256",
        "offline_label",
        "queried_label",
        "feedback_record_sha256",
        "query_event_details_private",
        "event_level_outcomes_and_identities_private",
    }

    def strip_private_event_values(value: Any) -> Any:
        if isinstance(value, dict):
            return {
                key: strip_private_event_values(item)
                for key, item in value.items()
                if key not in private_event_fields
                and not key.startswith("per_event_")
                and not key.startswith("event_level_")
            }
        if isinstance(value, list):
            return [strip_private_event_values(item) for item in value]
        return value

    sanitized_analysis = strip_private_event_values(sanitized_analysis)
    return _json_safe(
        {
            "schema": "r2-exp1.sanitized_aggregate.v1",
            "status": "COMPLETE_DEVELOPMENT_ONLY",
            "task_id": TASK_ID,
            "repository": run_manifest["expected_repository"],
            "run_id": run_manifest["run_id"],
            "local_private_output_path": str(output_root),
            "scientific_scope": "switch, support seed 0, frozen first 512 TEST events; development-only outcome-aware analysis",
            "analysis": sanitized_analysis,
            "input_hashes": inputs["input_hashes"],
            "source_fingerprints": inputs["source_fingerprints"],
            "original_r1_fixed_bank_sha256": inputs["tape_receipt"]["fixed_support_sha256"],
            "original_r1_raw_prototype_sha256": inputs["r1_manifest"]["fixed_support_raw_prototype_sha256"],
            "historical_c2_amplitude_020_replay": "EXACT_SCORE_AND_PATCH_PARITY",
            "private_event_outcomes_and_feedback_committed": False,
            "images_masks_dino_gpu_or_new_categories_accessed": False,
            "no_causal_confirmation_claim": True,
            "independent_confirmation_claim": False,
        }
    )


def _initial_run_manifest(run_id: str, repo_identity: Mapping[str, str]) -> dict[str, Any]:
    r1_manifest_path = R1_ROOT / "run_manifest.json"
    perf_manifest_path = PERF_ROOT / "run_manifest.json"
    r1_manifest = _read_json(r1_manifest_path)
    perf_manifest = _read_json(perf_manifest_path)
    sources = _source_fingerprints(r1_manifest, perf_manifest)
    expected_inputs = {
        "r1_run_manifest_sha256": sha256_file(r1_manifest_path),
        "r1_sealed_scores_sha256": sha256_file(R1_ROOT / "sealed_scores.jsonl"),
        "r1_transition_file_sha256": sha256_file(R1_ROOT / "event_transitions.jsonl"),
        "r2_perf1_run_manifest_sha256": sha256_file(perf_manifest_path),
        "r2_perf1_stage_a_seal_sha256": sha256_file(PERF_ROOT / "stage_a_sealed.json"),
        "r2_perf1_score_table_sha256": sha256_file(PERF_ROOT / "stage_a_scores.jsonl"),
        "r2_perf1_c2_parity_receipt_sha256": sha256_file(PERF_ROOT / "stage_a_c2_parity.json"),
        "r2_perf1_stage_b_metrics_sha256": sha256_file(PERF_ROOT / "stage_b_metrics.json"),
        "r2_perf1_orchestrator_receipt_sha256": sha256_file(PERF_ROOT / "orchestrator_complete.json"),
    }
    return {
        "schema": "calibration.r2_exp1.run_manifest.v1",
        "status": "INCOMPLETE",
        "run_id": run_id,
        "task_id": TASK_ID,
        "expected_repository": dict(repo_identity),
        "expected_parent_head": EXPECTED_PARENT_HEAD,
        "authorization_basis": "explicit bounded R2-EXP1 task in the active user request",
        "execution_scope": "Real-IAD switch only, support seed 0, first 512 frozen TEST events",
        "development_only": True,
        "outcome_aware_stage_b_authorized": True,
        "offline_test_outcomes_read": "Stage B only after Stage A score seal",
        "queried_feedback_read_policy": "one private feedback/correction row at each of the 17 frozen query positions after its score is budget-written",
        "query_positions_sha256": r1_manifest.get("query_positions_sha256"),
        "fixed_support_sha256": r1_manifest.get("fixed_support_sha256"),
        "event_count": 512,
        "query_count": 17,
        "source_group_count_expected": 103,
        "threshold": THETA,
        "c2_amplitudes": list(AMPLITUDES),
        "c2_historical_k": 5,
        "c2_historical_capacity": 256,
        "c2_historical_amplitude": 0.20,
        "python_executable": r"D:\anaconda3\envs\torch\python.exe",
        "cuda_visible_devices": "",
        "gpu_inference": False,
        "dino_loaded": False,
        "image_decoding": False,
        "mask_access": False,
        "automatic_restart": False,
        "runtime_deadline_seconds": 7200,
        "output_budget_bytes": 2 * 1024**3,
        "input_hashes_at_launch": expected_inputs,
        "source_fingerprints": sources,
        "r1_historical_status": r1_manifest.get("status"),
        "r2_perf1_initial_manifest_status": perf_manifest.get("status"),
        "r2_perf1_completion_receipt_status": _read_json(PERF_ROOT / "orchestrator_complete.json").get("status"),
        "private_event_level_outcomes_and_feedback": "new run output only; not version controlled",
    }


def _verify_launch_input_hashes(manifest: Mapping[str, Any]) -> None:
    expected = manifest.get("input_hashes_at_launch")
    if not isinstance(expected, Mapping):
        raise RuntimeError("launch manifest has no input hash table")
    current = {
        "r1_run_manifest_sha256": sha256_file(R1_ROOT / "run_manifest.json"),
        "r1_sealed_scores_sha256": sha256_file(R1_ROOT / "sealed_scores.jsonl"),
        "r1_transition_file_sha256": sha256_file(R1_ROOT / "event_transitions.jsonl"),
        "r2_perf1_run_manifest_sha256": sha256_file(PERF_ROOT / "run_manifest.json"),
        "r2_perf1_stage_a_seal_sha256": sha256_file(PERF_ROOT / "stage_a_sealed.json"),
        "r2_perf1_score_table_sha256": sha256_file(PERF_ROOT / "stage_a_scores.jsonl"),
        "r2_perf1_c2_parity_receipt_sha256": sha256_file(PERF_ROOT / "stage_a_c2_parity.json"),
        "r2_perf1_stage_b_metrics_sha256": sha256_file(PERF_ROOT / "stage_b_metrics.json"),
        "r2_perf1_orchestrator_receipt_sha256": sha256_file(PERF_ROOT / "orchestrator_complete.json"),
    }
    if dict(expected) != current:
        raise RuntimeError("an authorized historical input changed after launch")


def _run_worker_process(role: str, output_root: Path, *, nonce: str) -> None:
    if role not in {"stage-a", "stage-b"}:
        raise ValueError("unknown worker role")
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = ""
    env[NONCE_ENV] = nonce
    env[ROLE_ENV] = role
    command = [
        sys.executable,
        "-m",
        "experiments.calibration.r2_exp1_c1_mechanism_discovery",
        "--worker",
        role,
        "--output-root",
        str(output_root),
    ]
    completed = subprocess.run(command, cwd=str(ROOT), env=env, check=False)
    if completed.returncode != 0:
        raise RuntimeError(f"{role} worker exited with {completed.returncode}")


def _run_orchestrator(output_root: Path) -> dict[str, Any]:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    if sys.executable.lower() != r"d:\anaconda3\envs\torch\python.exe".lower():
        raise RuntimeError("orchestrator must use the authorized Python interpreter")
    if os.environ.get(ROLE_ENV) != "orchestrator" or not os.environ.get(NONCE_ENV):
        raise RuntimeError("orchestrator worker identity is missing")
    if not r1_current_process_is_in_job():
        raise RuntimeError("orchestrator is not contained in the Windows Job Object")
    _verify_repository_identity()
    nonce = os.environ[NONCE_ENV]
    _run_worker_process("stage-a", output_root, nonce=nonce)
    # Stage B is spawned only after the Stage A worker has created a seal and
    # the score-table checksum can be verified here without opening labels.
    seal = _read_json(output_root / "stage_a_sealed.json")
    if (
        seal.get("status") != "SEALED"
        or seal.get("offline_test_outcomes_read") is not False
        or sha256_file(output_root / "stage_a_scores.jsonl") != seal.get("score_table_sha256")
    ):
        raise RuntimeError("orchestrator refused to launch Stage B before a valid score seal")
    _run_worker_process("stage-b", output_root, nonce=nonce)
    return {
        "status": "COMPLETE",
        "stage_a_seal_sha256": sha256_file(output_root / "stage_a_sealed.json"),
        "stage_b_report_sha256": sha256_file(output_root / "stage_b_report.json"),
        "private_event_diagnostics_sha256": sha256_file(output_root / "stage_b_diagnostics.json"),
    }


def _write_docs_deliverables(
    output_root: Path, run_id: str, supervisor: Mapping[str, Any], output_bytes: int
) -> Path:
    from datetime import datetime, timezone

    report_root = ROOT / "docs" / "calibration" / "audit" / f"r2_exp1_c1_mechanism_discovery_20261101_{run_id}"
    if report_root.exists():
        raise FileExistsError(f"Refusing to overwrite an existing EXP1 report directory: {report_root}")
    report_root.mkdir(parents=True, exist_ok=False)
    aggregate = _read_json(output_root / "stage_b_report.json")
    private_query = _read_json(output_root / "stage_a_c2_parity.json")
    stage_a_diag = _read_json(output_root / "stage_a_diagnostics.json")
    complete = _read_json(output_root / "orchestrator_complete.json")
    supervisor_status = _read_json(output_root / "supervisor_status.json")
    aggregate["execution"] = {
        "supervisor": supervisor_status,
        "completed": complete,
        "stage_b_runtime": aggregate.get("stage_b_runtime"),
        "output_bytes": int(output_bytes),
        "output_budget_bytes": 2 * 1024**3,
        "stage_a_elapsed_seconds": stage_a_diag.get("stage_a_elapsed_seconds"),
        "stage_a_peak_working_set_bytes": stage_a_diag.get("stage_a_peak_working_set_bytes"),
        "private_query_event_details_path": str(output_root / "stage_a_c2_parity.json"),
        "private_event_diagnostics_path": str(output_root / "stage_b_diagnostics.json"),
        "private_event_level_rows_written": 512,
        "private_event_level_rows_committed": False,
    }
    aggregate_path = report_root / "aggregate_results.json"
    aggregate_path.write_bytes(_canonical_bytes(aggregate) + b"\n")
    report_path = report_root / "R2_EXP1_C1_MECHANISM_DISCOVERY_REPORT.md"
    report_path.write_text(_format_scientific_report(aggregate), encoding="utf-8", newline="\n")
    return report_root


def _fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "NA"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _fmt_count_metric(metric: Mapping[str, Any]) -> str:
    if metric.get("auroc") is None:
        return "NA"
    return _fmt(metric["auroc"], 4)


def _format_scientific_report(aggregate: Mapping[str, Any]) -> str:
    analysis = aggregate["analysis"]
    metrics = analysis["metrics"]
    movement = analysis["movement"]
    pairwise = analysis["pairwise"]
    query = analysis["query_feedback"]
    amp_rows = analysis["c2_amplitude_sensitivity"]
    hypothesis = analysis["hypothesis_ranking"]
    source_boot = analysis["source_group_bootstrap"]
    execution = aggregate.get("execution", {})
    receipts = execution.get("verification_receipts", {})
    methods = ("C0", "C1", "C2")

    lines = [
        "# R2-EXP1 — C1 Mechanism Discovery",
        "",
        "**Status: COMPLETE_DEVELOPMENT_ONLY.** This is an outcome-aware analysis of the already exposed switch, support-seed-0, 512-event development stream. All 512 outcomes had already been used by R2-PERF-1. The results are descriptive and conditional on one frozen historical trajectory; they do not establish independent confirmation or causal mechanism effects.",
        "",
        "## Integrated finding",
        "",
        f"C1 image AUROC is {_fmt(metrics['C1']['image_auroc'], 6)} versus {_fmt(metrics['C2']['image_auroc'], 6)} for historical C2, a difference of {_fmt(metrics['C1']['image_auroc'] - metrics['C2']['image_auroc'], 6)}. The predeclared C2 amplitude sweep tested uniform downscaling on the same queried-anchor trajectory. At most {_fmt(max((float(r['fraction_of_c1_minus_c2_auc_gap_recovered']) for r in amp_rows if float(r['amplitude']) < 0.20), default=0.0) * 100, 1)}% of that AUROC gap was recovered by a predeclared amplitude below 0.20. The strongest evidence for a follow-up is summarized under the Experiment 2 priorities below; no mechanism is causally confirmed.",
        "",
        "## A. Performance and correction movement",
        "",
        "All threshold metrics use the frozen C64 threshold 0.5165398120880127 and the historical `score >= threshold` rule. Directionally favorable movement is upward for defects and downward for normals; those event counts are separate from ranking-pair changes.",
        "",
        "| Method | Image AUROC | AP | FPR | FNR | Balanced accuracy | TP | TN | FP | FN |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name in methods:
        metric = metrics[name]
        fixed = metric["fixed_threshold"]
        lines.append(
            f"| {name} | {_fmt(metric['image_auroc'], 6)} | {_fmt(metric['image_auprc_average_precision'], 6)} | {_fmt(fixed['fpr'], 4)} | {_fmt(fixed['fnr'], 4)} | {_fmt(fixed['balanced_accuracy'], 4)} | {fixed['tp']} | {fixed['tn']} | {fixed['fp']} | {fixed['fn']} |"
        )
    lines += [
        "",
        "| Correction | Mean absolute image-score change | P90 absolute image-score change | Positive | Negative | Zero | Favorable | Harmful | Image score changed | Prediction flips | New FP | Repaired FP | New FN | Repaired FN |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name in ("C1", "C2"):
        record = movement[name]
        amount = record["image_score_delta"]
        direction = record["outcome_direction"]
        changes = record["prediction_changes"]
        lines.append(
            f"| {name} | {_fmt(amount['mean_absolute'], 6)} | {_fmt(amount['p90_absolute'], 6)} | {amount['positive']} | {amount['negative']} | {amount['zero']} | {direction['favorable']} | {direction['harmful']} | {amount['positive'] + amount['negative']} | {changes['flips']} | {changes['new_fp']} | {changes['repaired_fp']} | {changes['new_fn']} | {changes['repaired_fn']} |"
        )
    mag = analysis["c1_vs_c2_magnitude_comparison"]
    lines += [
        "",
        "### C1 versus C2 magnitude",
        "",
        "| Method | Mean absolute image delta | Median absolute image delta | P90 absolute image delta | Mean absolute Top-32 patch delta | Median absolute Top-32 patch delta | P90 absolute Top-32 patch delta | Nonzero Top-32 fraction |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name in ("C1", "C2"):
        image = mag[f"{name}_image_magnitude"]
        patch = mag[f"{name}_top32_patch_magnitude"]["all"]
        lines.append(
            f"| {name} | {_fmt(image['mean_absolute'], 7)} | {_fmt(image['median_absolute'], 7)} | {_fmt(image['p90_absolute'], 7)} | {_fmt(patch['mean_absolute'], 7)} | {_fmt(patch['median_absolute'], 7)} | {_fmt(patch['p90_absolute'], 7)} | {_fmt(patch['nonzero_fraction'], 4)} |"
        )
    lines += [
        "",
        "| Method | Defect mean absolute image delta | Normal mean absolute image delta | C0-margin <0.02: events / favorable / harmful / flips | Corrected argmax changed | Unchanged score masked by unchanged baseline argmax |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name in ("C1", "C2"):
        record = movement[name]
        by_outcome = record["image_score_delta_by_outcome"]
        nearest = record["near_c0_threshold"][0]
        argmax = analysis["argmax_and_masking"][name]
        lines.append(
            f"| {name} | {_fmt(by_outcome['anomalies']['mean_absolute'], 7)} | {_fmt(by_outcome['normals']['mean_absolute'], 7)} | {nearest['events']} / {nearest['favorable']} / {nearest['harmful']} / {nearest['prediction_flips']} | {argmax['corrected_argmax_changed']} | {argmax['score_unchanged_due_to_unchanged_baseline_argmax']} |"
        )
    lines += [
        "",
        "Top-32 patch-level magnitude and image-score movement are not interchangeable: image scoring takes the maximum over the corrected dense map, so changes on non-maximal patches can have no image-score effect. The argmax/masking counts and normal/anomaly magnitude quantiles are in `aggregate_results.json`.",
        "",
        "## B. Exact AUROC pairwise decomposition",
        "",
        f"The verified labels contain {analysis['outcome_counts']['anomaly']} defects and {analysis['outcome_counts']['normal']} normals, giving exactly {359 * 153:,} anomaly-normal pairs. AUROC gives ties weight 0.5. Strict repairs/damage exclude ties; tie creation and resolution are reported separately.",
        "",
        "| Candidate vs C0 | Strict incorrect→correct | Strict correct→incorrect | C0 incorrect→tie | C0 correct→tie | C0 tie→correct | C0 tie→incorrect | Net concordance change | AUROC difference |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name in ("C1", "C2"):
        row = pairwise[name]
        lines.append(
            f"| {name} | {row['strict_incorrect_to_correct_repairs']} | {row['strict_correct_to_incorrect_damage']} | {row['baseline_incorrect_to_tie']} | {row['baseline_correct_to_tie']} | {row['baseline_tie_resolved_correctly']} | {row['baseline_tie_resolved_incorrectly']} | {_fmt(row['net_concordance_change'], 1)} | {_fmt(row['auc_difference_from_pairs'], 6)} |"
        )
    direct = analysis["c1_vs_c2_pairwise"]
    lines += [
        "",
        f"Direct C1-versus-C2 pairwise comparison: C1 strictly repairs {direct['strict_incorrect_to_correct_repairs']} C2-incorrect pairs and strictly damages {direct['strict_correct_to_incorrect_damage']} C2-correct pairs; net concordance change is {_fmt(direct['net_concordance_change'], 1)} across {direct['pair_count']:,} pairs (AUROC difference {_fmt(direct['auc_difference_from_pairs'], 6)}).",
        "",
        f"For each arm, `AUROC_m − AUROC_C0 = net concordance change / {359 * 153:,}` was verified against the evaluator within 2e-15. The decomposition by original C0 margin, favorable/harmful normal and anomaly movement, pair time interval, and Source Group participation is in `aggregate_results.json`. Source Group bootstrap intervals are paired and conditional on this fixed trajectory; the 54,927 pairs are not treated as independent samples.",
        "",
        "| Arm | Source Group bootstrap 95% AUROC interval | Paired AUROC difference vs C0 95% interval |",
        "|---|---:|---:|",
    ]
    for name in ("C1", "C2"):
        auc_ci = source_boot["auc_percentile_ci_95"][name]
        diff_ci = source_boot["paired_auc_difference_vs_c0_ci_95"][name]
        lines.append(
            f"| {name} | [{_fmt(auc_ci[0], 6)}, {_fmt(auc_ci[1], 6)}] | [{_fmt(diff_ci[0], 6)}, {_fmt(diff_ci[1], 6)}] |"
        )
    lines += [
        "",
        "## C. Magnitude-controlled C2",
        "",
        f"C2 a=0.20 reproduced the historical C2 patch vectors and all 512 image scores exactly ({_fmt(analysis['prior_metric_reproduction']['methods']['C2']['image_auroc'], 6)} AUROC). Each other row rescales the stored Top-32 patch delta before the exact `corrected_dense_map` and max aggregation; no image-score delta was scaled.",
        "",
        "| Amplitude | AUROC | AP | ΔAUROC vs C0 | ΔAUROC vs C1 | Gap recovered | FPR | FNR | Balanced accuracy | FP | FN | Strict pairs repaired | Strict pairs damaged |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in amp_rows:
        fixed = row["fixed_threshold_confusion"]
        lines.append(
            f"| {row['amplitude']:.2f} | {_fmt(row['auroc'], 6)} | {_fmt(row['image_auprc_average_precision'], 6)} | {_fmt(row['auroc_difference_vs_c0'], 6)} | {_fmt(row['auroc_difference_vs_c1'], 6)} | {_fmt(row['fraction_of_c1_minus_c2_auc_gap_recovered'] * 100, 1)}% | {_fmt(row['fpr'], 4)} | {_fmt(row['fnr'], 4)} | {_fmt(row['balanced_accuracy'], 4)} | {fixed['fp']} | {fixed['fn']} | {row['pairwise']['strict_incorrect_to_correct_repairs']} | {row['pairwise']['strict_correct_to_incorrect_damage']} |"
        )
    max_fraction = max((float(row["fraction_of_c1_minus_c2_auc_gap_recovered"]) for row in amp_rows if row["amplitude"] < 0.20), default=0.0)
    lines += [
        "",
        f"**Can reducing C2 magnitude explain most of C1’s observed AUROC advantage?** {'Yes, on this fixed trajectory: at least half of the C1−C2 gap was recovered by a predeclared amplitude below 0.20.' if max_fraction >= 0.5 else 'No, not by uniform amplitude scaling alone: every predeclared amplitude below 0.20 recovered less than half of the C1−C2 AUROC gap.'} This diagnostic does not rule out C1’s outcome-aware or margin-dependent strength policy, because its per-anchor beta values were not recorded.",
        "",
        "## D. Query timing and memory evidence",
        "",
        f"The query ledger confirms {query['total']['queries']} feedback events: {query['queried_feedback_balance'].get('anomaly', 0)} defect and {query['queried_feedback_balance'].get('normal', 0)} normal. Across the {query['total']['candidates_examined']} Top-32 candidates, C1’s native gate kept {query['total']['native_gate_kept']} ({_fmt(query['total']['native_gate_kept'] / max(query['total']['candidates_examined'], 1), 4)}), with {query['total']['inserted']} inserts, {query['total']['merged']} merges, {query['total']['replaced']} replacements and {query['total']['skipped']} skipped updates. C2 added {query['total']['c2_additions']} patch anchors and ended at its FIFO capacity of {query['total']['c2_retained_after_query']}.",
        "",
        "",
        "| Queried C0 outcome | Queries | Top-32 kept | Inserted | Merged | Replaced | Skipped |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for outcome_name in ("TP", "TN", "FP", "FN"):
        row = query["by_c0_confusion_outcome"].get(outcome_name, {})
        lines.append(
            f"| {outcome_name} | {row.get('queries', 0)} | {row.get('native_gate_kept', 0)} | {row.get('inserted', 0)} | {row.get('merged', 0)} | {row.get('replaced', 0)} | {row.get('skipped', 0)} |"
        )
    lines += [
        "",
        "Per-query positions, feedback labels, identities and ledger rows are stored only in the private run output. Query-interval AUROCs are descriptive; an interval without both outcomes is reported as unavailable rather than imputed. The query-age buckets and early/late position quartiles are also descriptive and do not establish an update effect.",
        "",
        "| Query interval | Position range | Events (normal/anomaly) | C0 AUROC | C1 AUROC | C2 AUROC |",
        "|---:|---|---:|---:|---:|---:|",
    ]
    for interval in analysis["query_intervals"]:
        values = interval["metrics"]
        lines.append(
            f"| {interval['interval_id']} | {interval['start_position_inclusive']}–{interval['end_position_inclusive']} | {interval['events']} ({interval['normal_events']}/{interval['anomaly_events']}) | {_fmt_count_metric(values['C0'])} | {_fmt_count_metric(values['C1'])} | {_fmt_count_metric(values['C2'])} |"
        )
    lines += [
        "",
        "Query age and early/late position breakdowns are available in `aggregate_results.json`; the query event itself is assigned to the interval before its feedback disclosure.",
        "",
        "The tape does not identify per-anchor beta, kappa, radius, hidden Codebook prototypes, or unrecorded state changes: `NOT_IDENTIFIABLE_FROM_EXISTING_TAPE`. The R1 manifest reports final C1 positive/negative anchor counts, while the correction-event ledger provides only its recorded aggregate update statistics.",
        "",
        "## E. Mechanism ranking and Experiment 2 priorities",
        "",
        "| Rank | Hypothesis | Evidence label | Directly observed | Causal limitation | Experiment 2 priority |",
        "|---:|---|---|---|---|---|",
    ]
    for h in sorted(hypothesis["hypotheses"], key=lambda item: int(item["rank"])):
        direct_summary = h["supporting_empirical_evidence"]
        limitation = h["contradictory_or_limit_evidence"]
        lines.append(
            f"| {h['rank']} | {h['id']} — {h['name']} | `{h['evidence_class']}` | {direct_summary} | {limitation} | {h['experiment_2_priority']} |"
        )
    lines += [
        "",
        "**Recommended primary Experiment 2 mechanisms:** " + ", ".join(hypothesis["recommended_experiment_2_primary_mechanisms"]) + ". These priorities follow the observed admission, amplitude, retention and far-field records plus interpretability. All require randomized or otherwise controlled component intervention; none is causally confirmed here.",
        "",
        "## F. Reproducibility and limits",
        "",
        f"- Repository: `{aggregate['repository']['branch']}@{aggregate['repository']['head']}`; task code SHA-256: `{aggregate['source_fingerprints'].get('experiments/calibration/r2_exp1_c1_mechanism_discovery.py', 'not recorded')}`.",
        f"- R1 manifest SHA-256: `{aggregate['input_hashes']['r1_run_manifest_sha256']}`; sealed scores SHA-256: `{aggregate['input_hashes']['r1_sealed_scores_sha256']}`; structural transition projection SHA-256: `{aggregate['input_hashes']['r1_transition_projection_sha256']}`.",
        f"- Prior R2-PERF-1 score table SHA-256: `{aggregate['input_hashes']['r2_perf1_score_table_sha256']}`; prior Stage A seal SHA-256: `{aggregate['input_hashes']['r2_perf1_stage_a_seal_sha256']}`.",
        f"- R1 public tape check: {receipts.get('r1_event_count', 'NA')} event NPZ shard and array hashes verified; all recorded C0/C1/C2 scores reconstructed exactly; 512-entry shard receipt-list SHA-256: `{receipts.get('r1_shard_receipt_list_sha256', 'NA')}`.",
        f"- EXP1 Stage A: {receipts.get('exp1_event_count', 'NA')} scores sealed with score-table SHA-256 `{receipts.get('exp1_score_table_sha256', 'NA')}`; seal SHA-256 `{receipts.get('exp1_stage_a_seal_sha256', 'NA')}`; query positions SHA-256 `{receipts.get('query_positions_sha256', 'NA')}`; event/identity alignment with R1 and R2-PERF-1 was exact.",
        f"- Stage B identity mapping: `{receipts.get('outcome_alignment_status', 'NA')}` for TEST positions 0–511; {receipts.get('anomaly_count', 'NA')} defects and {receipts.get('normal_count', 'NA')} normals. The full-file source SHA remains `NOT_COMPUTED`.",
        f"- New private run: `{aggregate['local_private_output_path']}`; unique run ID `{aggregate['run_id']}`; new outputs are separate from all historical artifacts.",
        f"- Supervisor: `{execution.get('supervisor', {}).get('status', 'unknown')}`, elapsed {_fmt(execution.get('supervisor', {}).get('elapsed_seconds'), 2)} seconds, confirmed termination `{execution.get('supervisor', {}).get('termination_confirmed', 'unknown')}`; output byte count is enforced by the existing pre-write `OutputBudget`.",
        f"- New output size: {_fmt(execution.get('output_bytes'), 0)} bytes under the 2 GiB application limit.",
        f"- Stage workers: Stage A {_fmt(execution.get('stage_a_elapsed_seconds'), 2)} seconds; Stage B {_fmt(execution.get('stage_b_runtime', {}).get('wall_seconds'), 2)} wall seconds and {_fmt(execution.get('stage_b_runtime', {}).get('cpu_seconds'), 2)} CPU seconds; peak Stage B working set {_fmt(execution.get('stage_b_runtime', {}).get('peak_working_set_bytes'), 0)} bytes.",
        f"- Source Group bootstrap: {source_boot['requested_replicates']} paired resamples, seed {source_boot['seed']}, {source_boot['source_group_count']} groups, {source_boot['valid_replicates']} valid replicates; intervals describe uncertainty conditional on the historical trajectory.",
        "- The existing R2-PERF-1 projection read authorized switch.json as UTF-8 text, identity-checked the first 512 TEST records, and parsed only those outcome fields. It structurally skipped TRAIN and non-prefix TEST outcomes and did not open mask files or other categories. The full-file source SHA was not computed, so no independent full-JSON fingerprint is claimed.",
        "- No DINO load, feature extraction, image decoding, GPU/CUDA inference, new category, mask, independent confirmation set, or algorithm modification was performed.",
        "- Raw per-event labels, event/image identities and queried feedback are retained only in the private untracked run directory. The committed report and JSON contain aggregate results only.",
        "- This development-set analysis does not validate a final hyperparameter and does not prove that any component caused C1’s observed advantage. Experiment 2 must isolate components on an appropriately authorized and frozen design.",
        "",
    ]
    sanitization = aggregate.get("post_run_sanitization")
    if isinstance(sanitization, Mapping):
        anchor = next(
            index for index, line in enumerate(lines)
            if line.startswith("- Raw per-event labels, event/image identities")
        )
        lines.insert(
            anchor,
            "- The committed aggregate was re-emitted from the sealed Stage B report using recursive removal of event-level fields; sanitizer SHA-256: `" + str(sanitization.get("sanitizer_source_sha256")) + "`. The analysis worker source SHA above remains the run-time fingerprint.",
        )
    test_results = execution.get("cpu_regression_tests")
    if isinstance(test_results, Mapping):
        output_anchor = next(
            index for index, line in enumerate(lines)
            if line.startswith("- New output size:")
        )
        lines.insert(
            output_anchor + 1,
            f"- Focused CPU regression: `{test_results.get('command')}`; {test_results.get('result')}; test source SHA-256: `{test_results.get('test_source_sha256')}`.",
        )
    return "\n".join(lines)


def _execute_integrated_run() -> dict[str, Any]:
    if sys.executable.lower() != r"d:\anaconda3\envs\torch\python.exe".lower():
        raise RuntimeError(f"use the authorized interpreter D:\\anaconda3\\envs\\torch\\python.exe; got {sys.executable}")
    identity = _verify_repository_identity()
    run_id = new_run_id()
    manifest = _initial_run_manifest(run_id, identity)
    output_root = create_output_namespace(run_id)
    budget = OutputBudget(output_root, allow_existing=True)
    budget.write_json("run_manifest.json", manifest)
    nonce = secrets.token_hex(32)
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = ""
    env[NONCE_ENV] = nonce
    env[ROLE_ENV] = "orchestrator"
    command = [
        sys.executable,
        "-m",
        "experiments.calibration.r2_exp1_c1_mechanism_discovery",
        "--worker",
        "orchestrator",
        "--output-root",
        str(output_root),
    ]
    supervisor_result = None
    try:
        supervisor_result = run_supervised_child(
            command,
            timeout_seconds=7200,
            env=env,
            cwd=ROOT,
            output_dir=None,
        )
        supervisor_record = {
            "schema": "r2-exp1.supervisor_status.v1",
            **supervisor_result.__dict__,
            "automatic_restart": False,
            "cuda_visible_devices": "",
            "python_executable": sys.executable,
        }
        if (
            supervisor_result.status != "COMPLETED"
            or supervisor_result.child_exit_code != 0
            or supervisor_result.termination_confirmed is not True
            or supervisor_result.active_job_processes != 0
        ):
            budget.write_json("supervisor_status.json", supervisor_record, failure_reserved=True)
            raise RuntimeError(f"supervised child did not complete cleanly: {supervisor_result.status}")
        seal = _read_json(output_root / "stage_a_sealed.json")
        report = _read_json(output_root / "stage_b_report.json")
        parity = _read_json(output_root / "stage_a_c2_parity.json")
        if (
            seal.get("status") != "SEALED"
            or parity.get("status") != "PASS"
            or report.get("status") != "COMPLETE_DEVELOPMENT_ONLY"
            or report.get("analysis", {}).get("prior_metric_reproduction", {}).get("status") != "PASS"
        ):
            budget.write_json("supervisor_status.json", supervisor_record, failure_reserved=True)
            raise RuntimeError("supervised workers exited zero but EXP1 completion receipts are incomplete")
        budget.write_json("supervisor_status.json", supervisor_record)
        orchestrator_complete = {
            "schema": "r2-exp1.orchestrator_complete.v1",
            "status": "COMPLETE",
            "run_id": run_id,
            "automatic_restart": False,
            "stage_a_sealed": True,
            "stage_b_started_after_score_seal": True,
            "c2_a020_exact_score_parity": parity.get("c2_a020_exact_score_and_patch_matches") == 512,
            "stage_a_seal_sha256": sha256_file(output_root / "stage_a_sealed.json"),
            "stage_b_report_sha256": sha256_file(output_root / "stage_b_report.json"),
            "supervisor_status": supervisor_result.status,
            "supervisor_elapsed_seconds": supervisor_result.elapsed_seconds,
            "termination_confirmed": supervisor_result.termination_confirmed,
        }
        budget.write_json("orchestrator_complete.json", orchestrator_complete)
        return {
            "run_id": run_id,
            "output_root": output_root,
            "supervisor": supervisor_record,
            "orchestrator_complete": orchestrator_complete,
            "output_bytes": budget._audit(),
        }
    except Exception as exc:
        failure = {
            "schema": "r2-exp1.failure.v1",
            "status": "INCOMPLETE",
            "run_id": run_id,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "failed_utc_epoch": time.time(),
            "automatic_restart": False,
            "partial_output_preserved": True,
            "supervisor_result": supervisor_result.__dict__ if supervisor_result is not None else None,
        }
        try:
            if not (output_root / "failure.json").exists():
                budget.write_json("failure.json", failure, failure_reserved=True)
        except Exception:
            pass
        raise


def main(argv: Sequence[str] | None = None) -> int:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    parser = argparse.ArgumentParser(description="Integrated CPU-only R2-EXP1 mechanism discovery")
    parser.add_argument("--worker", choices=("orchestrator", "stage-a", "stage-b"))
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--bootstrap-replicates", type=int, default=BOOTSTRAP_REPLICATES)
    args = parser.parse_args(argv)
    try:
        if args.worker:
            if args.output_root is None:
                raise ValueError("worker requires --output-root")
            output_root = Path(args.output_root)
            if args.worker == "stage-a":
                result = _stage_a(output_root)
                print(
                    json.dumps(
                        {
                            "status": result["status"],
                            "event_count": result["event_count"],
                            "score_table_sha256": result["score_table_sha256"],
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
            elif args.worker == "stage-b":
                if args.bootstrap_replicates != BOOTSTRAP_REPLICATES:
                    raise ValueError("R2-EXP1 bootstrap replicate count is fixed at 1000")
                result = _stage_b(output_root, bootstrap_replicates=args.bootstrap_replicates)
                print(
                    json.dumps(
                        {
                            "status": result["sanitized_report"]["status"],
                            "event_count": result["analysis"]["event_count"],
                            "pair_count": result["analysis"]["pairwise"]["C1"]["pair_count"],
                            "wall_seconds": result["runtime"]["wall_seconds"],
                            "cpu_seconds": result["runtime"]["cpu_seconds"],
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
            else:
                result = _run_orchestrator(output_root)
                print(json.dumps(result, sort_keys=True), flush=True)
            return 0

        result = _execute_integrated_run()
        report_root = _write_docs_deliverables(
            result["output_root"], result["run_id"], result["supervisor"], result["output_bytes"]
        )
        print(
            json.dumps(
                {
                    "status": "COMPLETE_DEVELOPMENT_ONLY",
                    "run_id": result["run_id"],
                    "output_root": str(result["output_root"]),
                    "report_root": str(report_root),
                    "supervisor": result["supervisor"],
                    "output_bytes": result["output_bytes"],
                },
                sort_keys=True,
            ),
            flush=True,
        )
        return 0
    except Exception as exc:
        print(f"R2_EXP1_FAILURE {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
