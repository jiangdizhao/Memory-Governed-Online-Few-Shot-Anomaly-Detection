"""Historical and current-state controllers sharing one weighted estimator."""
from __future__ import annotations

from collections import Counter
from typing import Any

from .fs4_threshold_core import NormalEvidence, SCORERS, WeightedThresholds, weights
from .fs4_source_excluded_scorer import NativeSourceExcludedScorer, VerifiedNormalFeatureCache
from .fs4_stochastic_bea import StochasticBEAGroupSlot


def diagnostics(evidence: list[NormalEvidence], policy: StochasticBEAGroupSlot,
                *, current_position: int, current_scorer_states: dict[str, str],
                thresholds: dict[str, float], suppression: dict[str, int]) -> dict[str, Any]:
    receipts = list(policy.receipts.values())
    raw: list[float] = []
    clipped: list[float] = []
    ess = None
    if evidence:
        r, c, ess = weights([e.receipt.design_probability for e in evidence])
        raw, clipped = r.tolist(), c.tolist()
    coverage: dict[str, dict[str, dict[str, int]]] = {j: {} for j in SCORERS}
    for receipt in receipts:
        for j, score in (("G", receipt.prediction.score_g), ("GC", receipt.prediction.score_gc)):
            region = "below" if score < thresholds[j] - 0.02 else (
                "above" if score > thresholds[j] + 0.02 else "near")
            row = coverage[j].setdefault(region, {"observed": 0, "queried": 0, "verified_normal": 0})
            row["observed"] += 1
            row["queried"] += int(receipt.executed)
            row["verified_normal"] += int(receipt.disclosed and receipt.label == 0)
    return {"raw_weights": raw, "clipped_weights": clipped, "effective_sample_size": ess,
            "distinct_normal_groups": len({e.group for e in evidence}),
            "score_region_coverage": coverage, "query": policy.diagnostics(),
            "evidence_age_events": [current_position - e.position for e in evidence],
            "normal_versions": [e.normal_version for e in evidence],
            "correction_versions": [e.correction_version for e in evidence],
            "historical_stale_counts": {j: sum(e.scorer_states[j] != current_scorer_states[j] for e in evidence) for j in SCORERS},
            "fraction_without_calibratable_label": (1 - len(evidence) / len(receipts)) if receipts else None,
            "threshold_suppression_reasons": dict(suppression)}


class SelectionAwareHistoricalThreshold:
    def __init__(self, initial_theta: float, *, capacity: int = 1024) -> None:
        self.engine = WeightedThresholds(initial_theta)
        self.capacity = capacity
        self.evidence: list[NormalEvidence] = []
        self.groups: set[str] = set()
        self.revision = 0

    def add(self, evidence: NormalEvidence) -> dict[str, Any]:
        evidence.validate()
        if evidence.group in self.groups:
            raise ValueError("duplicate source group in historical evidence")
        if len(self.evidence) >= self.capacity:
            raise OverflowError("historical evidence capacity reached; no silent replacement")
        self.evidence.append(evidence)
        self.groups.add(evidence.group)
        self.revision += 1
        return {j: self.engine.update(j, self.evidence,
                    [e.pre_scores[j] for e in self.evidence], self.revision) for j in SCORERS}

    def observe_state_change(self) -> dict[str, str]:
        # Historical scalar scores are stale and cannot be silently refreshed.
        return {j: "THRESHOLD_HELD" for j in SCORERS}

    def diagnostics(self, policy: StochasticBEAGroupSlot, *, current_position: int,
                    current_scorer_states: dict[str, str]) -> dict[str, Any]:
        return diagnostics(self.evidence, policy, current_position=current_position,
            current_scorer_states=current_scorer_states, thresholds=self.engine.theta,
            suppression=self.engine.suppression)


class SelectionAwareSourceExcludedThreshold:
    def __init__(self, initial_theta: float, scorer: NativeSourceExcludedScorer,
                 cache: VerifiedNormalFeatureCache | None = None) -> None:
        self.engine = WeightedThresholds(initial_theta)
        self.scorer = scorer
        self.cache = cache if cache is not None else VerifiedNormalFeatureCache()
        self.evidence: list[NormalEvidence] = []
        self.revision = 0
        self.rescore_calls = Counter()
        self.current_scores: dict[str, dict[str, float]] = {j: {} for j in SCORERS}
        self.attempt_tokens: dict[str, object] = {j: None for j in SCORERS}

    def add(self, evidence: NormalEvidence, features: Any, *, height: int,
            width: int, normalization: str) -> None:
        evidence.validate()
        self.cache.add(evidence, features, height=height, width=width, normalization=normalization)
        self.evidence.append(evidence)
        self.revision += 1

    def diagnostics(self, policy: StochasticBEAGroupSlot, *, current_position: int,
                    current_scorer_states: dict[str, str]) -> dict[str, Any]:
        return {**diagnostics(self.evidence, policy, current_position=current_position,
            current_scorer_states=current_scorer_states, thresholds=self.engine.theta,
            suppression=self.engine.suppression), "cache": self.cache.bytes_report(),
            "rescore_calls": dict(self.rescore_calls)}

    def refresh(self, *, state_g: str, state_gc: str, event_position: int) -> dict[str, Any]:
        states = {"G": state_g, "GC": state_gc}
        if not all(states.values()):
            raise ValueError("complete current scorer state identities required")
        tokens = {j: (self.revision, states[j]) for j in SCORERS}
        changed = [j for j in SCORERS if tokens[j] != self.attempt_tokens[j]]
        if not changed:
            return {j: {"status": "THRESHOLD_HELD", "theta": self.engine.theta[j]} for j in SCORERS}
        if len(self.evidence) < self.engine.minimum_groups:
            for j in changed:
                self.attempt_tokens[j] = tokens[j]
                self.engine.status[j] = "INSUFFICIENT_VERIFIED_NORMALS"
                self.engine.suppression["INSUFFICIENT_VERIFIED_NORMALS"] = self.engine.suppression.get("INSUFFICIENT_VERIFIED_NORMALS", 0) + 1
            return {j: {"status": self.engine.status[j], "theta": self.engine.theta[j]} for j in SCORERS}
        try:
            for item in self.evidence:
                item.validate()
        except (ValueError, TypeError, AttributeError):
            for j in changed:
                self.attempt_tokens[j] = tokens[j]
                self.engine.status[j] = "SOURCE_PROVENANCE_INCOMPLETE"
                self.engine.suppression["SOURCE_PROVENANCE_INCOMPLETE"] = self.engine.suppression.get("SOURCE_PROVENANCE_INCOMPLETE", 0) + 1
            return {j: {"status": self.engine.status[j], "theta": self.engine.theta[j]} for j in SCORERS}
        try:
            _raw, _bounded, ess = weights([e.receipt.design_probability for e in self.evidence], self.engine.w_max)
        except (ValueError, TypeError):
            ess = 0.0
            reason = "SELECTION_PROBABILITY_UNAVAILABLE"
        else:
            reason = "LOW_EFFECTIVE_SAMPLE_SIZE"
        if ess + 1e-12 < self.engine.minimum_ess:
            for j in changed:
                self.attempt_tokens[j] = tokens[j]
                self.engine.status[j] = reason
                self.engine.suppression[reason] = self.engine.suppression.get(reason, 0) + 1
            return {j: {"status": self.engine.status[j], "theta": self.engine.theta[j]} for j in SCORERS}
        try:
            refreshed = {j: {} for j in changed}
            for e in self.evidence:
                result = self.scorer.score(self.cache.records[e.group], event_position=event_position)
                for j in changed:
                    refreshed[j][e.group] = float(result[j])
                    self.rescore_calls[j] += 1
        except Exception as exc:
            for j in changed:
                self.attempt_tokens[j] = tokens[j]
                self.engine.status[j] = "CURRENT_STATE_RESCORING_UNAVAILABLE"
                self.engine.suppression["CURRENT_STATE_RESCORING_UNAVAILABLE"] = self.engine.suppression.get("CURRENT_STATE_RESCORING_UNAVAILABLE", 0) + 1
            return {j: {"status": self.engine.status[j], "theta": self.engine.theta[j], "reason": str(exc)} for j in SCORERS}
        output: dict[str, Any] = {}
        for j in SCORERS:
            if j not in changed:
                output[j] = {"status": "THRESHOLD_HELD", "theta": self.engine.theta[j]}
                continue
            self.current_scores[j] = refreshed[j]
            self.attempt_tokens[j] = tokens[j]
            output[j] = self.engine.update(j, self.evidence,
                [refreshed[j][e.group] for e in self.evidence], tokens[j])
        return output
