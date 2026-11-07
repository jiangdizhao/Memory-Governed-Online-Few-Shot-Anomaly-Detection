"""Historical and current-state controllers sharing one weighted estimator."""
from __future__ import annotations

from collections import Counter
import copy
import math
import time
from typing import Any

from .fs4_threshold_core import NormalEvidence, SCORERS, WeightedThresholds, weights
from .fs4_source_excluded_scorer import NativeSourceExcludedScorer, VerifiedNormalFeatureCache
from .fs4_stochastic_bea import StochasticBEAGroupSlot
from .fs4_errors import (FS4FatalIntegrityError, FS4NativeScoringIntegrityError,
                         FS4ProvenanceIntegrityError, FS4RecoverableAvailabilityError)


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
        self.rescore_invocations = 0
        self.current_scores: dict[str, dict[str, float]] = {j: {} for j in SCORERS}
        self.attempt_tokens: dict[str, object] = {j: None for j in SCORERS}
        self.successful_tokens: dict[str, object] = {j: None for j in SCORERS}
        self.rescore_seconds: list[float] = []
        self.fatal_error: FS4FatalIntegrityError | None = None

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
            "rescore_calls": dict(self.rescore_calls), "rescore_invocations": self.rescore_invocations,
            "rescore_seconds": list(self.rescore_seconds)}

    def refresh(self, *, state_g: str, state_gc: str, event_position: int) -> dict[str, Any]:
        states = {"G": state_g, "GC": state_gc}
        if not all(states.values()):
            raise ValueError("complete current scorer state identities required")
        tokens = {j: (self.revision, states[j]) for j in SCORERS}
        if self.fatal_error is not None:
            raise self.fatal_error
        changed = [j for j in SCORERS if tokens[j] != self.successful_tokens[j]]
        if not changed:
            return {j: {"status": "THRESHOLD_HELD", "theta": self.engine.theta[j]} for j in SCORERS}
        if len(self.evidence) < self.engine.minimum_groups:
            for j in changed:
                self.engine.status[j] = "INSUFFICIENT_VERIFIED_NORMALS"
                self.engine.suppression["INSUFFICIENT_VERIFIED_NORMALS"] = self.engine.suppression.get("INSUFFICIENT_VERIFIED_NORMALS", 0) + 1
            return {j: {"status": self.engine.status[j], "theta": self.engine.theta[j]} for j in SCORERS}
        try:
            for item in self.evidence:
                item.validate()
        except (ValueError, TypeError, AttributeError) as exc:
            self.fatal_error = FS4ProvenanceIntegrityError("invalid normal evidence provenance")
            raise self.fatal_error from exc
        try:
            _raw, _bounded, ess = weights([e.receipt.design_probability for e in self.evidence], self.engine.w_max)
        except (ValueError, TypeError):
            ess = 0.0
            reason = "SELECTION_PROBABILITY_UNAVAILABLE"
        else:
            reason = "LOW_EFFECTIVE_SAMPLE_SIZE"
        if ess + 1e-12 < self.engine.minimum_ess:
            for j in changed:
                self.engine.status[j] = reason
                self.engine.suppression[reason] = self.engine.suppression.get(reason, 0) + 1
            return {j: {"status": self.engine.status[j], "theta": self.engine.theta[j]} for j in SCORERS}
        for j in changed:
            self.attempt_tokens[j] = tokens[j]
        try:
            refreshed = {j: {} for j in changed}
            for e in self.evidence:
                if e.group not in self.cache.records:
                    raise FS4ProvenanceIntegrityError("missing verified-normal cached feature")
                if self.cache.records[e.group].evidence != e:
                    raise FS4ProvenanceIntegrityError("cached feature belongs to different normal evidence")
                started = time.perf_counter()
                self.rescore_invocations += 1
                try:
                    result = self.scorer.score(self.cache.records[e.group], event_position=event_position)
                finally:
                    self.rescore_seconds.append(time.perf_counter() - started)
                for j in changed:
                    self.rescore_calls[j] += 1
                    value = float(result[j])
                    if not math.isfinite(value):
                        raise FS4NativeScoringIntegrityError("nonfinite source-excluded native score")
                    refreshed[j][e.group] = value
        except FS4RecoverableAvailabilityError as exc:
            for j in changed:
                self.engine.status[j] = "CURRENT_STATE_RESCORING_UNAVAILABLE"
                self.engine.suppression["CURRENT_STATE_RESCORING_UNAVAILABLE"] = self.engine.suppression.get("CURRENT_STATE_RESCORING_UNAVAILABLE", 0) + 1
            return {j: {"status": self.engine.status[j], "theta": self.engine.theta[j], "reason": str(exc)} for j in SCORERS}
        except FS4FatalIntegrityError as exc:
            self.fatal_error = exc
            raise
        except Exception as exc:
            self.fatal_error = FS4NativeScoringIntegrityError("unexpected native source-excluded scoring failure")
            raise self.fatal_error from exc
        # Stage both updates and publish only if both are valid; never expose half a transaction.
        proposed_engine = copy.deepcopy(self.engine)
        output: dict[str, Any] = {}
        for j in SCORERS:
            if j not in changed:
                output[j] = {"status": "THRESHOLD_HELD", "theta": self.engine.theta[j]}
                continue
            output[j] = proposed_engine.update(j, self.evidence,
                [refreshed[j][e.group] for e in self.evidence], tokens[j])
            if output[j]["status"] != "UPDATE_ACCEPTED":
                self.fatal_error = FS4ProvenanceIntegrityError("validated rescore could not be atomically applied")
                raise self.fatal_error
        self.engine = proposed_engine
        for j in changed:
            self.current_scores[j] = refreshed[j]
            self.successful_tokens[j] = tokens[j]
        return output
