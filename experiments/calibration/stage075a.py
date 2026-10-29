"""Stage 0.75-A causal Native Greedy ranking and C64 transport primitives.

The GPU runner lives in :mod:`stage075a_runner`.  This module contains only
deterministic, testable protocol operations and metric helpers.  It does not
implement B8 or alter the historical scorer/writer.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np


SCHEMA = "calibration.stage075a.v1"
DEVELOPMENT_CLASSES = ("audiojack", "regulator", "switch", "rolled_strip_base")
SUPPORT_SEEDS = (0, 1, 2, 3, 4)
QUERY_SEED = 37
QUERY_RATE = 0.03
SNAPSHOT_POSITIONS = (0, 400, 800, 1600, 2400, 3200)
SNAPSHOT_NEXT_WIDTH = 256
C64_DIAGNOSTIC_EPSILON = 1e-12
UNIFORMITY_RATIO_BANDS = (0.01, 0.05, 0.10)


class Stage075AError(RuntimeError):
    """Raised when a frozen Stage 0.75-A invariant is violated."""


@dataclass(frozen=True)
class CausalStep:
    position: int
    static_score: float
    dynamic_score: float
    memory_version_before: int
    memory_sha256_before: str
    queried: bool
    writer_result: Any
    memory_version_after: int
    memory_sha256_after: str


def higher_quantile(values: Sequence[float], q: float) -> float:
    """NumPy ``method='higher'`` order statistic, independent of NumPy version."""
    if not values or not 0.0 <= q <= 1.0:
        raise ValueError("higher_quantile requires finite data and q in [0, 1]")
    ordered = sorted(float(value) for value in values)
    if not all(math.isfinite(value) for value in ordered):
        raise ValueError("higher_quantile received nonfinite data")
    return ordered[math.ceil(q * (len(ordered) - 1))]


def causal_prequential_step(
        *, position: int, queried: bool,
        memory_state: Callable[[], tuple[int, str]],
        score_pair: Callable[[], tuple[float, float]],
        static_map: Callable[[], Any],
        reveal_label: Callable[[int], int],
        apply_writer: Callable[..., Any],
        source_key: str, features: Sequence[Any]) -> CausalStep:
    """Execute one strict score -> query -> disclose -> shared-writer step.

    The callback boundary is intentional: no hidden label is requested unless
    ``queried`` is already fixed by the label-blind tape.  Both scores and the
    pre-event memory identity are captured before writer mutation.
    """
    version_before, digest_before = memory_state()
    static_score_value, dynamic_score_value = score_pair()
    if not (math.isfinite(float(static_score_value)) and
            math.isfinite(float(dynamic_score_value))):
        raise Stage075AError("Native event scores must be finite")
    score_static_map = static_map() if queried else None
    label = int(reveal_label(position)) if queried else None
    if label not in (None, 0, 1):
        raise Stage075AError("Queried label must be verified physical 0/1")
    writer_result = apply_writer(
        queried=queried, label=label, source_key=source_key,
        score=float(static_score_value), static_amap=score_static_map,
        full_features=features, step=position)
    version_after, digest_after = memory_state()
    return CausalStep(
        position=int(position), static_score=float(static_score_value),
        dynamic_score=float(dynamic_score_value),
        memory_version_before=int(version_before),
        memory_sha256_before=str(digest_before), queried=bool(queried),
        writer_result=writer_result, memory_version_after=int(version_after),
        memory_sha256_after=str(digest_after))


def prior_group_status(source_key: str, ever_admitted: set[str],
                       resident_sources: set[str]) -> dict[str, bool]:
    """Return pre-event same-group flags; callers update sets after the write."""
    return {"previously_admitted": source_key in ever_admitted,
            "currently_resident": source_key in resident_sources}


def snapshot_cohort_mask(source_keys: Sequence[str], *, snapshot_position: int,
                         stream_length: int, prior_sources: set[str],
                         cohort: str) -> np.ndarray:
    """Position/group-only fixed-snapshot cohort; labels/scores are never read."""
    if cohort not in ("next256", "all_remaining"):
        raise ValueError(f"Unknown snapshot cohort: {cohort}")
    if not 0 <= snapshot_position <= stream_length or len(source_keys) != stream_length:
        raise ValueError("Snapshot boundary or source-key length is invalid")
    stop = (min(stream_length, snapshot_position + SNAPSHOT_NEXT_WIDTH)
            if cohort == "next256" else stream_length)
    mask = np.zeros(stream_length, dtype=bool)
    for position in range(snapshot_position, stop):
        mask[position] = source_keys[position] not in prior_sources
    return mask


def score_distribution(values: Sequence[float]) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return {"n": 0, "quantiles_linear": None, "mean": None, "sd_sample": None}
    if not np.isfinite(array).all():
        raise Stage075AError("Metric score vector contains nonfinite values")
    qs = (0.0, 0.05, 0.25, 0.5, 0.75, 0.95, 1.0)
    return {"n": int(array.size),
            "quantiles_linear": {str(q): float(np.quantile(array, q, method="linear"))
                                 for q in qs},
            "mean": float(array.mean()),
            "sd_sample": float(array.std(ddof=1)) if array.size > 1 else 0.0}


def confusion_metrics(labels: Sequence[int], scores: Sequence[float],
                      thresholds: float | Sequence[float]) -> dict[str, Any]:
    y = np.asarray(labels, dtype=np.int8)
    s = np.asarray(scores, dtype=np.float64)
    th = np.asarray(thresholds, dtype=np.float64)
    if y.ndim != 1 or s.shape != y.shape or th.shape not in ((), y.shape):
        raise ValueError("Confusion metric inputs must be aligned 1D vectors")
    if not np.isin(y, (0, 1)).all() or not np.isfinite(s).all() or not np.isfinite(th).all():
        raise ValueError("Confusion metrics require finite scores/thresholds and binary labels")
    pred = s >= th
    positive, negative = y == 1, y == 0
    tp = int(np.count_nonzero(pred & positive))
    fp = int(np.count_nonzero(pred & negative))
    fn = int(np.count_nonzero(~pred & positive))
    tn = int(np.count_nonzero(~pred & negative))
    fpr = fp / (fp + tn) if fp + tn else None
    fnr = fn / (fn + tp) if fn + tp else None
    ba = (0.5 * ((1.0 - fpr) + (1.0 - fnr))
          if fpr is not None and fnr is not None else None)
    return {"n": int(len(y)), "positive": int(positive.sum()),
            "negative": int(negative.sum()), "tp": tp, "fp": fp,
            "fn": fn, "tn": tn, "fpr": fpr, "fnr": fnr,
            "balanced_accuracy": ba,
            "threshold_kind": "per_event" if th.ndim else "scalar"}


def ranking_metrics(labels: Sequence[int], scores: Sequence[float]) -> dict[str, Any]:
    """AUROC/AP/tie fraction with undefined single-class AUROC made explicit."""
    from sklearn.metrics import average_precision_score, roc_auc_score

    y = np.asarray(labels, dtype=np.int8)
    s = np.asarray(scores, dtype=np.float64)
    if y.ndim != 1 or s.shape != y.shape or not np.isin(y, (0, 1)).all():
        raise ValueError("Ranking vectors must be aligned and labels binary")
    if not np.isfinite(s).all():
        raise Stage075AError("Ranking score vector contains nonfinite values")
    n_pos, n_neg = int(np.sum(y == 1)), int(np.sum(y == 0))
    auc = (float(roc_auc_score(y, s)) if n_pos and n_neg else None)
    ap = (float(average_precision_score(y, s)) if n_pos else None)
    pos = s[y == 1]
    neg = s[y == 0]
    tied = sum(int(np.count_nonzero(pos_value == neg)) for pos_value in pos)
    pairs = int(n_pos * n_neg)
    return {"n": int(len(y)), "positive": n_pos, "negative": n_neg,
            "auroc": auc, "auprc_average_precision": ap,
            "positive_negative_pairs": pairs,
            "cross_class_tied_pairs": tied,
            "cross_class_tie_fraction": tied / pairs if pairs else None,
            "auroc_status": "DEFINED" if auc is not None else "UNDEFINED_SINGLE_CLASS",
            "auprc_status": "DEFINED" if ap is not None else "UNDEFINED_NO_POSITIVES"}


def paired_score_metrics(labels: Sequence[int], static: Sequence[float],
                         dynamic: Sequence[float], *, group_mask: Sequence[bool],
                         theta0: float, current_theta: Sequence[float]) -> dict[str, Any]:
    y = np.asarray(labels, dtype=np.int8)
    s0 = np.asarray(static, dtype=np.float64)
    s1 = np.asarray(dynamic, dtype=np.float64)
    mask = np.asarray(group_mask, dtype=bool)
    theta_t = np.asarray(current_theta, dtype=np.float64)
    if not (y.shape == s0.shape == s1.shape == mask.shape == theta_t.shape):
        raise ValueError("Paired metric vectors must have identical lengths")
    out: dict[str, Any] = {"n_total": int(len(y)), "n_included": int(mask.sum()),
                           "n_excluded": int((~mask).sum())}
    yy, a, b, tt = y[mask], s0[mask], s1[mask], theta_t[mask]
    out["static"] = ranking_metrics(yy, a)
    out["dynamic"] = ranking_metrics(yy, b)
    out["delta_auroc"] = (out["dynamic"]["auroc"] - out["static"]["auroc"]
                          if out["dynamic"]["auroc"] is not None and
                          out["static"]["auroc"] is not None else None)
    out["delta_auprc"] = (
        out["dynamic"]["auprc_average_precision"] -
        out["static"]["auprc_average_precision"]
        if out["dynamic"]["auprc_average_precision"] is not None and
        out["static"]["auprc_average_precision"] is not None else None)
    out["score_shift_dynamic_minus_static"] = score_distribution(b - a)
    out["static_normal"] = score_distribution(a[yy == 0])
    out["static_defect"] = score_distribution(a[yy == 1])
    out["dynamic_normal"] = score_distribution(b[yy == 0])
    out["dynamic_defect"] = score_distribution(b[yy == 1])
    out["frozen_theta0_static"] = confusion_metrics(yy, a, float(theta0))
    out["frozen_theta0_dynamic"] = confusion_metrics(yy, b, float(theta0))
    out["current_C64_q90_static"] = confusion_metrics(yy, a, tt)
    out["current_C64_q90_dynamic"] = confusion_metrics(yy, b, tt)
    return out


def c64_transport_diagnostics(c_before: Sequence[float],
                              c_after: Sequence[float], *,
                              theta_previous: float,
                              epsilon: float = C64_DIAGNOSTIC_EPSILON
                              ) -> dict[str, Any]:
    """Paired protected-C64 shift diagnostics; values never alter predictions."""
    before = np.asarray(c_before, dtype=np.float64)
    after = np.asarray(c_after, dtype=np.float64)
    if before.shape != (64,) or after.shape != (64,) or not (
            np.isfinite(before).all() and np.isfinite(after).all()):
        raise ValueError("C64 transport requires two aligned 64-score finite vectors")
    if not math.isfinite(theta_previous) or epsilon <= 0:
        raise ValueError("Invalid C64 threshold or diagnostic scale floor")
    shift = after - before
    median = float(np.median(shift))
    q25, q75 = np.quantile(shift, [0.25, 0.75], method="linear")
    heterogeneity = float(q75 - q25)
    theta_after = higher_quantile(after.tolist(), 0.90)
    q25_after, q75_after = np.quantile(after, [0.25, 0.75], method="linear")
    iqr_after = float(q75_after - q25_after)

    # Tie-aware pairwise rank changes over the 2,016 C64 pairs.
    bdiff = before[:, None] - before[None, :]
    adiff = after[:, None] - after[None, :]
    upper = np.triu(np.ones((64, 64), dtype=bool), k=1)
    bsign, asign = np.sign(bdiff[upper]), np.sign(adiff[upper])
    discordant = int(np.count_nonzero((bsign * asign) < 0))
    newly_tied = int(np.count_nonzero((bsign != 0) & (asign == 0)))
    newly_ordered = int(np.count_nonzero((bsign == 0) & (asign != 0)))
    from scipy.stats import rankdata
    rank_move = np.abs(rankdata(after, method="average") -
                       rankdata(before, method="average"))
    relative_residual = np.abs(shift - median) / max(iqr_before(before), epsilon)
    scale = max(1.0, float(np.max(np.abs(before))), float(np.max(np.abs(after))))
    machine_uniform = bool(np.max(np.abs(shift - median)) <= max(1e-8, 1e-6 * scale))
    uniform_bands = {str(ratio): bool(heterogeneity <= ratio * max(iqr_before(before), epsilon))
                     for ratio in UNIFORMITY_RATIO_BANDS}
    return {
        "n_c64": 64,
        "theta_previous": float(theta_previous),
        "median_paired_shift": median,
        "shift_q25_linear": float(q25), "shift_q75_linear": float(q75),
        "shift_heterogeneity_H_IQR": heterogeneity,
        "theta_transport_candidate": float(theta_previous + median),
        "theta_SCQ_candidate_higher_q90": float(theta_after),
        "theta_q90_change": float(theta_after - theta_previous),
        "disagreement_normalized_by_after_IQR": float(
            abs(theta_previous + median - theta_after) / max(iqr_after, epsilon)),
        "after_C64_IQR_linear": iqr_after,
        "diagnostic_epsilon": float(epsilon),
        "quantile_conventions": {
            "q90": "higher order statistic, index ceil(0.90*(n-1))",
            "q25_q75_iqr": "linear interpolation",
            "median": "NumPy median (average of middle pair for even n)"},
        "rank_reordering": {"pairs_total": 64 * 63 // 2,
                            "strict_discordant_pairs": discordant,
                            "newly_tied_pairs": newly_tied,
                            "newly_ordered_pairs": newly_ordered,
                            "mean_absolute_anchor_rank_movement": float(rank_move.mean()),
                            "max_absolute_anchor_rank_movement": float(rank_move.max())},
        "uniformity_diagnostic": {
            "numeric_uniform_shift": machine_uniform,
            "heterogeneity_to_before_IQR_cutoffs": uniform_bands,
            "heterogeneity_ratio": float(heterogeneity / max(iqr_before(before), epsilon)),
            "interpretation": "descriptive only; not a B8 parameter or guarantee"},
        "c64_score_shift_quantiles_linear": {
            str(q): float(np.quantile(shift, q, method="linear"))
            for q in (0.0, 0.25, 0.5, 0.75, 1.0)},
    }


def iqr_before(values: Sequence[float]) -> float:
    array = np.asarray(values, dtype=np.float64)
    q25, q75 = np.quantile(array, [0.25, 0.75], method="linear")
    return float(q75 - q25)


def summarize_numeric(rows: Sequence[float]) -> dict[str, Any]:
    values = np.asarray(rows, dtype=np.float64)
    if values.size == 0:
        return {"n": 0, "mean": None, "sd_sample": None, "min": None,
                "median": None, "max": None, "quantiles_linear": None}
    if not np.isfinite(values).all():
        raise Stage075AError("Cannot summarize nonfinite values")
    return {"n": int(values.size), "mean": float(values.mean()),
            "sd_sample": float(values.std(ddof=1)) if values.size > 1 else 0.0,
            "min": float(values.min()), "median": float(np.median(values)),
            "max": float(values.max()),
            "quantiles_linear": {str(q): float(np.quantile(values, q, method="linear"))
                                 for q in (0.0, .25, .5, .75, 1.0)}}
