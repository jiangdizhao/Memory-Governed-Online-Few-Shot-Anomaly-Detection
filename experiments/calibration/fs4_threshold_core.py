"""Shared selection-aware threshold mathematics and verified evidence types."""
from __future__ import annotations

from dataclasses import dataclass, replace
import math
import re
from typing import Mapping, Sequence

import numpy as np

from .fs4_stochastic_bea import POLICY, QueryReceipt, StochasticBEAGroupSlot, action_uniform, candidate_ordinal, receipt_digest

SCORERS = ("G", "GC")


def _identity(value: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
        raise ValueError("complete SHA-256 source/event provenance is required")
    return value


@dataclass(frozen=True)
class NormalEvidence:
    group: str
    event: str
    position: int
    pre_scores: Mapping[str, float]
    scorer_states: Mapping[str, str]
    normal_version: int
    correction_version: int
    receipt: QueryReceipt
    issuer: StochasticBEAGroupSlot
    feature_handle: str | None = None

    def validate(self) -> None:
        receipt, p, policy = self.receipt, self.receipt.prediction, self.issuer
        _identity(self.group); _identity(self.event)
        if (policy.receipts.get(p.position) != receipt or self.group != p.group or self.event != p.event or
                self.position != p.position or dict(self.pre_scores) != {"G": p.score_g, "GC": p.score_gc} or
                dict(self.scorer_states) != {"G": p.scorer_g, "GC": p.scorer_gc} or
                receipt.policy != POLICY or not receipt.query_seal or not receipt.disclosed or not receipt.executed or receipt.label != 0 or
                receipt.query_seal != receipt_digest(replace(receipt, query_seal=None, disclosed=False, label=None)) or
                receipt.ordinal != receipt.candidate or receipt.candidate != candidate_ordinal(receipt.seed, p.group) or
                receipt.seed != policy.seed or receipt.exploitation != (receipt.utility > receipt.cost) or
                receipt.exploration_probability != 0.05 or
                receipt.gate_probability != (1.0 if receipt.exploitation else 0.05) or
                receipt.slot_probability != 0.2 or receipt.gate_probability not in (0.05, 1.0) or
                receipt.design_probability != receipt.slot_probability * receipt.gate_probability or
                receipt.random_draw != action_uniform(receipt.seed, p.group, p.event) or
                receipt.random_draw >= receipt.gate_probability or
                self.normal_version < 0 or self.correction_version < 0):
            raise ValueError("verified-normal evidence lacks genuine selection provenance")

    @classmethod
    def from_disclosure(cls, policy: StochasticBEAGroupSlot, receipt: QueryReceipt,
                        *, normal_version: int, correction_version: int,
                        feature_handle: str | None = None) -> "NormalEvidence":
        p = receipt.prediction
        _identity(p.group); _identity(p.event)
        if normal_version < 0 or correction_version < 0:
            raise ValueError("negative state version")
        evidence = cls(p.group, p.event, p.position, {"G": p.score_g, "GC": p.score_gc},
                       {"G": p.scorer_g, "GC": p.scorer_gc}, int(normal_version),
                       int(correction_version), receipt, policy, feature_handle)
        evidence.validate()
        return evidence


def weights(probabilities: Sequence[float], w_max: float = 50.0) -> tuple[np.ndarray, np.ndarray, float]:
    p = np.asarray(probabilities, dtype=np.float64)
    if p.ndim != 1 or not len(p) or not np.isfinite(p).all() or np.any(p <= 0) or np.any(p > 1):
        raise ValueError("SELECTION_PROBABILITY_UNAVAILABLE")
    if not math.isfinite(w_max) or w_max <= 0:
        raise ValueError("invalid weight cap")
    raw = 1.0 / p
    bounded = np.minimum(raw, w_max)
    ess = float(bounded.sum() ** 2 / np.square(bounded).sum())
    return raw, bounded, ess


def weighted_higher_quantile(values: Sequence[float], sample_weights: Sequence[float], q: float = 0.90) -> float:
    """First sorted value whose inclusive weighted empirical CDF reaches q.

    Ties share a value and therefore have the same returned value; stable sorting
    makes order deterministic. Equal weights implement nearest-rank ceil(q*n).
    """
    x = np.asarray(values, dtype=np.float64)
    w = np.asarray(sample_weights, dtype=np.float64)
    if x.ndim != 1 or not len(x) or x.shape != w.shape or not np.isfinite(x).all() or not np.isfinite(w).all() or np.any(w <= 0) or not 0 < q <= 1:
        raise ValueError("invalid weighted empirical quantile inputs")
    order = np.argsort(x, kind="stable")
    index = int(np.searchsorted(np.cumsum(w[order]), q * float(w.sum()), side="left"))
    return float(x[order[min(index, len(x) - 1)]])


def support_self_reference(scores: Sequence[float]) -> dict[str, float | str]:
    x = np.asarray(scores, dtype=np.float64)
    if x.shape != (4,) or not np.isfinite(x).all():
        raise ValueError("exactly four finite native Support self-scores required")
    mu = float(x.mean())
    sample_std = float(x.std(ddof=1))
    effective_std = max(sample_std, 1e-6)  # historical P0 initializer convention
    return {"theta": mu + 2.5 * effective_std, "mean": mu,
            "sample_std_ddof1": sample_std, "initializer_sigma": effective_std,
            "writer_sigma": max(sample_std, 0.02),
            "status": "COLD_START_UNCALIBRATED",
            "uncertainty": "four self-scores; no population quantile or coverage guarantee"}


class WeightedThresholds:
    """Identical estimator for both controllers, with independent G/GC state."""

    def __init__(self, initial_theta: float, *, minimum_groups: int = 12,
                 minimum_ess: float = 12.0, w_max: float = 50.0) -> None:
        if not math.isfinite(float(initial_theta)) or minimum_groups < 12 or minimum_ess < 12 or w_max <= 0:
            raise ValueError("invalid cold start or weakened evidence gate")
        self.theta = {j: float(initial_theta) for j in SCORERS}
        self.tokens: dict[str, object] = {j: None for j in SCORERS}
        self.status = {j: "COLD_START_UNCALIBRATED" for j in SCORERS}
        self.minimum_groups, self.minimum_ess, self.w_max = minimum_groups, minimum_ess, w_max
        self.suppression: dict[str, int] = {}

    def update(self, scorer: str, evidence: Sequence[NormalEvidence], values: Sequence[float],
               token: object) -> dict[str, object]:
        if scorer not in SCORERS or token is None:
            raise ValueError("scorer and genuine evidence/state token required")
        def held(reason: str) -> dict[str, object]:
            self.status[scorer] = reason
            self.suppression[reason] = self.suppression.get(reason, 0) + 1
            return {"status": reason, "theta": self.theta[scorer]}
        if token == self.tokens[scorer]:
            return held("THRESHOLD_HELD")
        if len(evidence) != len(values) or len({e.group for e in evidence}) != len(evidence):
            return held("SOURCE_PROVENANCE_INCOMPLETE")
        if any(e.receipt.design_probability is None for e in evidence):
            return held("SELECTION_PROBABILITY_UNAVAILABLE")
        try:
            for item in evidence:
                item.validate()
        except (ValueError, TypeError, AttributeError):
            return held("SOURCE_PROVENANCE_INCOMPLETE")
        if len(evidence) < self.minimum_groups:
            return held("INSUFFICIENT_VERIFIED_NORMALS")
        try:
            raw, clipped, ess = weights([e.receipt.design_probability for e in evidence], self.w_max)
        except (ValueError, TypeError):
            return held("SELECTION_PROBABILITY_UNAVAILABLE")
        if ess + 1e-12 < self.minimum_ess:
            return held("LOW_EFFECTIVE_SAMPLE_SIZE")
        try:
            quantile = weighted_higher_quantile(values, clipped)
        except (ValueError, TypeError):
            return held("CURRENT_STATE_RESCORING_UNAVAILABLE")
        old = self.theta[scorer]
        self.theta[scorer] = min(old + 0.01, max(old - 0.01, quantile))
        self.tokens[scorer] = token
        self.status[scorer] = "UPDATE_ACCEPTED"
        return {"status": "UPDATE_ACCEPTED", "theta": self.theta[scorer], "prior_theta": old,
                "quantile": quantile, "ess": ess, "raw_weights": raw.tolist(),
                "clipped_weights": clipped.tolist(), "applies": "future_events_only"}
