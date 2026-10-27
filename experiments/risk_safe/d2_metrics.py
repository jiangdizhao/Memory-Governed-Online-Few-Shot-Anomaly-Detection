"""D2 independent-probe metrics; no metric may enter an online controller."""
from __future__ import annotations

from typing import Mapping, Sequence
import math


def image_metrics(labels: Sequence[int], scores: Sequence[float],
                  theta: float) -> dict:
    import numpy as np
    from sklearn.metrics import roc_auc_score, average_precision_score
    if len(labels) != len(scores) or not labels:
        raise ValueError("Probe labels/scores missing or mismatched")
    y = np.asarray(labels, dtype=np.int32)
    s = np.asarray(scores, dtype=np.float64)
    if not np.isfinite(s).all() or set(y.tolist()) != {0, 1}:
        raise ValueError("Probe requires finite scores and both image-level labels")
    # Match a0_a1_strict4_matched.metrics: predictions at score >= theta.
    pred = s >= float(theta)
    tp = int(((pred == 1) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    return {
        "N": len(y), "N_normal": tn + fp, "N_defect": tp + fn,
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "fpr": fp / (tn + fp), "fnr": fn / (tp + fn),
        "auroc": float(roc_auc_score(y, s)),
        "auprc": float(average_precision_score(y, s)),
        "theta0": float(theta),
    }


def probe_deltas(
        scores: Sequence[float], labels: Sequence[int],
        reference_scores: Sequence[float],
        current: Mapping[str, float],
        reference: Mapping[str, float]) -> dict:
    if len(scores) != len(reference_scores) or len(scores) != len(labels):
        raise ValueError("Unequal paired probe sizes")
    harms = [max(0., float(old) - float(new))
             for old, new, lab in zip(reference_scores, scores, labels)
             if lab == 1]
    if not harms:
        raise ValueError("No defect probe score")
    return {
        "defect_erosion_positive_mean": sum(harms) / len(harms),
        "defect_erosion_positive_max": max(harms),
        "fnr_delta": float(current["fnr"]) - float(reference["fnr"]),
        "fnr_degradation_positive": max(
            0., float(current["fnr"]) - float(reference["fnr"])),
        "fpr_gain": float(reference["fpr"]) - float(current["fpr"]),
    }


def risk_prediction_pairs(rows: Sequence[Mapping[str, object]]) -> dict:
    """Diagnostic only; never use held-out probe damage to tune on same probe."""
    usable = [r for r in rows
              if r.get("defect_risk") is not None and
              r.get("independent_probe_erosion") is not None]
    if len(usable) < 3:
        return {"status": "insufficient_evidence",
                "N_available_pairs": len(usable),
                "spearman": None,
                "reason": "At least three distinct paired proposals are needed"}
    x = [float(r["defect_risk"]) for r in usable]
    y = [float(r["independent_probe_erosion"]) for r in usable]
    if len(set(x)) < 2 or len(set(y)) < 2:
        return {"status": "insufficient_variation",
                "N_available_pairs": len(usable),
                "spearman": None}
    from scipy.stats import spearmanr
    corr, p = spearmanr(x, y)
    return {"status": "descriptive_only_not_significance_test",
            "N_available_pairs": len(usable),
            "spearman": float(corr),
            "p_value_exploratory_not_iid": float(p)}

def online_gate_elapsed(total_seconds: float, offline_probe_seconds: float) -> float:
    """Exclude retrospective heldout-probe rescoring from online gate cost.

    Timings are exploratory host-wall times, NOT synchronized CUDA kernel
    benchmarks. The offline scorer is diagnostic-only and MUST NOT affect
    admission latency comparisons to ordinary Greedy/novelty baselines.
    """
    total, offline = float(total_seconds), float(offline_probe_seconds)
    if not (math.isfinite(total) and math.isfinite(offline)):
        raise ValueError("Nonfinite timing")
    if total < 0 or offline < 0 or offline > total + 1e-8:
        raise ValueError("Invalid online/offline timing components")
    return max(0., total - offline)
