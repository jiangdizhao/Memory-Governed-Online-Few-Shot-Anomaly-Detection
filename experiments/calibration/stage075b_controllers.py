"""Synthetic-only B8 controller arms for Stage 0.75-B.

These controllers consume the existing causal protocol types. They do not
modify the Stage 0 B3/B7 algorithms or enter the approved method registry.
"""
from __future__ import annotations

from dataclasses import dataclass
from collections import Counter
import math
from typing import Any

from .controllers import StateConsistentQuantile
from .protocol import EvidenceSnapshot, Observation, Prediction


METHOD_VERSION = "stage075b-ecmsc-candidate-v3"
MOVE_LIMIT = 0.01
IQR_FLOOR = 1e-12
DELTA_HEURISTIC = 0.05
LAMBDA_BASE = 0.05
LAMBDA_MOVE = 0.05


def _linear_quantile(values: tuple[float, ...] | list[float], p: float) -> float:
    if not values or not 0.0 <= p <= 1.0:
        raise ValueError("linear quantile needs finite values and p in [0,1]")
    ordered = sorted(float(x) for x in values)
    h = (len(ordered) - 1) * p
    lo, hi = math.floor(h), math.ceil(h)
    return ordered[lo] + (h - lo) * (ordered[hi] - ordered[lo])


def _median(values: tuple[float, ...] | list[float]) -> float:
    if not values:
        raise ValueError("median requires at least one value")
    ordered = sorted(float(x) for x in values)
    n = len(ordered)
    if n % 2:
        return ordered[n // 2]
    return ordered[n // 2 - 1] / 2.0 + ordered[n // 2] / 2.0


def _cdf(sorted_c64: tuple[float, ...], value: float) -> float:
    """Right-continuous finite empirical CDF with ties counted on the right."""
    if len(sorted_c64) != 64:
        raise ValueError("C64 coordinate requires exactly 64 scores")
    if not math.isfinite(float(value)):
        raise ValueError("Nonfinite CDF input")
    # Deliberately exact binary64 comparison; prediction has the same tie rule.
    return sum(x <= value for x in sorted_c64) / 64.0


def _higher_c64(sorted_c64: tuple[float, ...], q: float = 0.90) -> float:
    index = math.ceil(q * (len(sorted_c64) - 1))
    return sorted_c64[index]


def _c64_scale(sorted_c64: tuple[float, ...]) -> float:
    q1 = _linear_quantile(sorted_c64, 0.25)
    q3 = _linear_quantile(sorted_c64, 0.75)
    return max(q3 - q1, IQR_FLOOR)


def _finite_candidates(theta_b3: float, theta_prev: float,
                       c64: tuple[float, ...], normals: tuple[float, ...],
                       defects: tuple[float, ...], *, restrict_movement: bool = False
                       ) -> tuple[float, ...]:
    scores = tuple(sorted(set((*c64, *normals, *defects))))
    values = {float(theta_b3), float(theta_prev)}
    for score in scores:
        values.add(score)
        low, high = math.nextafter(score, -math.inf), math.nextafter(score, math.inf)
        if math.isfinite(low): values.add(low)
        if math.isfinite(high): values.add(high)
    for left, right in zip(scores, scores[1:]):
        mid = left / 2.0 + right / 2.0
        if math.isfinite(mid): values.add(mid)
    values.update((theta_prev - MOVE_LIMIT, theta_prev + MOVE_LIMIT))
    if restrict_movement:
        lo, hi = theta_prev - MOVE_LIMIT, theta_prev + MOVE_LIMIT
        values = {x for x in values if lo <= x <= hi}
    return tuple(sorted(x for x in values if math.isfinite(x)))


def _empirical_j(theta: float, normals: tuple[float, ...],
                 defects: tuple[float, ...]) -> float:
    if not normals or not defects:
        raise ValueError("J is undefined when either queried class is missing")
    fpr = sum(x >= theta for x in normals) / len(normals)
    fnr = sum(x < theta for x in defects) / len(defects)
    return 1.0 - fpr - fnr


def _risk_penalty(candidate_count: int, n_normal: int, n_defect: int) -> float:
    if n_normal <= 0 or n_defect <= 0:
        raise ValueError("Reliability penalty is undefined for a missing class")
    k = max(1, candidate_count)
    log_term = math.log(2.0 * k / DELTA_HEURISTIC)
    return (math.sqrt(log_term / (2.0 * n_normal)) +
            math.sqrt(log_term / (2.0 * n_defect)))


def _map_percentile(sorted_c64: tuple[float, ...], target_u: float,
                    raw_reference: float, candidates: tuple[float, ...]
                    ) -> tuple[float, dict[str, Any]]:
    """Map a rank target to an explicit raw candidate without changing ties."""
    if not 0.0 <= target_u <= 1.0:
        raise ValueError("Percentile target must be in [0,1]")
    mapped = min(candidates,
                 key=lambda x: (abs(_cdf(sorted_c64, x) - target_u),
                                abs(x - raw_reference), x))
    q = sorted_c64[max(0, min(63, math.ceil(64.0 * target_u) - 1))]
    return mapped, {
        "target_percentile": target_u,
        "selected_raw_threshold": mapped,
        "selected_percentile": _cdf(sorted_c64, mapped),
        "generalized_inverse_diagnostic": q,
        "saturated": target_u in (0.0, 1.0),
        "raw_boundary_preserved": mapped == raw_reference,
    }


@dataclass
class Stage075BOffset:
    theta: float
    name: str = "B3+Offset"
    version: str = "stage075b-b3-raw-median-offset-v1"
    max_step: float = MOVE_LIMIT
    evidence_digest: str = ""
    normal_scores: tuple[float, ...] = ()
    defect_scores: tuple[float, ...] = ()
    updates: int = 0
    diagnostics_last: dict[str, Any] | None = None
    pending: bool = False

    def predict(self, observation: Observation) -> Prediction:
        return Prediction(int(observation.score >= self.theta), self.theta,
                          "RAW_MEDIAN_OFFSET")

    def observe_score(self, observation: Observation) -> None: return
    def observe_feedback(self, observation: Observation, prediction: Prediction,
                         feedback: Any) -> None: return

    def on_memory_change(self, evidence: EvidenceSnapshot) -> None:
        normals, defects = _eligible_scores(evidence)
        digest = _evidence_digest(evidence)
        if digest == self.evidence_digest:
            return
        self.evidence_digest = digest
        self.normal_scores, self.defect_scores = normals, defects
        self.pending = True

    def finalize_step(self) -> float:
        if not self.pending:
            return self.theta
        self.pending = False
        if self.normal_scores and self.defect_scores:
            desired = (_median(self.normal_scores) / 2.0 +
                       _median(self.defect_scores) / 2.0)
        else:
            desired = self.theta
        old = self.theta
        self.theta = _clip(old, desired, self.max_step)
        self.updates += 1
        self.diagnostics_last = {"desired": desired, "executed": self.theta,
                                 "movement": self.theta - old}
        return self.theta

    def state_dict(self) -> dict[str, Any]:
        return {"version": self.version, "theta": self.theta,
                "max_step": self.max_step, "evidence_digest": self.evidence_digest,
                "normal_scores": list(self.normal_scores),
                "defect_scores": list(self.defect_scores), "updates": self.updates,
                "diagnostics_last": self.diagnostics_last,
                "pending": self.pending}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if state.get("version") != self.version or state.get("max_step") != self.max_step:
            raise ValueError("B3+Offset checkpoint identity/configuration mismatch")
        self.theta = _finite(state["theta"])
        self.evidence_digest = str(state["evidence_digest"])
        self.normal_scores = tuple(_finite(x) for x in state["normal_scores"])
        self.defect_scores = tuple(_finite(x) for x in state["defect_scores"])
        self.updates = int(state["updates"])
        self.diagnostics_last = state.get("diagnostics_last")
        self.pending = bool(state["pending"])


class Stage075BController:
    """Common stateful shell for the four candidate B8 ablation arms."""

    name = "B8"
    version = METHOD_VERSION

    def __init__(self, theta: float, arm: str):
        if arm not in {"M", "S", "P", "Full"}:
            raise ValueError(f"Unknown Stage 0.75-B arm: {arm}")
        self.name = f"B8-{arm}"
        self.version = f"{METHOD_VERSION}-{arm}"
        self.arm = arm
        self.theta = _finite(theta)
        self.shadow = StateConsistentQuantile(theta=self.theta, delta=0.01,
                                              min_normals=16)
        self.evidence_digest = ""
        self.memory_version = -1
        self.evidence_revision = -1
        self.c64: tuple[float, ...] = ()
        self.normals: tuple[float, ...] = ()
        self.defects: tuple[float, ...] = ()
        self.normal_ids: tuple[str, ...] = ()
        self.defect_ids: tuple[str, ...] = ()
        self.previous_c64: tuple[float, ...] = ()
        self.previous_b3 = self.theta
        self.previous_theta = self.theta
        self.pending = False
        self.departed = False
        self.rejoin_updates = 0
        self.total_updates = 0
        self.c64_rescores = 0
        self.defect_rescores = 0
        self.last_diagnostics: dict[str, Any] = {
            "status": "INITIAL_HOLD", "deviation": 0.0,
            "desired_b3": self.theta, "reachable_theta": self.theta,
            "remaining_b3_distance": 0.0, "rejoin_blocked": False,
        }

    def predict(self, observation: Observation) -> Prediction:
        return Prediction(int(observation.score >= self.theta), self.theta,
                          str(self.last_diagnostics.get("status", "")),
                          dict(self.last_diagnostics))

    def observe_score(self, observation: Observation) -> None: return
    def observe_feedback(self, observation: Observation, prediction: Prediction,
                         feedback: Any) -> None: return

    def on_memory_change(self, evidence: EvidenceSnapshot) -> None:
        digest = _evidence_digest(evidence)
        if digest == self.evidence_digest:
            return
        normals, defects = _eligible_scores(evidence)
        protected = tuple(sorted(float(a.score) for a in evidence.normal_anchors
                                 if a.protected))
        if len(protected) != 64:
            raise ValueError(f"{self.name} requires exactly 64 protected C64 scores")
        if any(a.memory_version != evidence.memory_version
               for a in (*evidence.normal_anchors, *evidence.defect_anchors)):
            raise ValueError("Stage 0.75-B rejected stale evidence")
        self.shadow.on_memory_change(evidence)
        self.previous_c64 = self.c64 or protected
        self.previous_b3 = self.shadow.theta
        self.previous_theta = self.theta
        self.c64 = protected
        self.normals, self.defects = normals, defects
        self.normal_ids = tuple(a.opaque_anchor_id for a in evidence.normal_anchors
                                if not a.protected and
                                a.source_exclusion_policy ==
                                "queried_normal_all_layer_leave_source_out")
        self.defect_ids = tuple(a.opaque_anchor_id for a in evidence.defect_anchors
                                if a.source_exclusion_policy ==
                                "queried_defect_not_written")
        self.memory_version = evidence.memory_version
        self.evidence_revision = evidence.evidence_revision
        self.evidence_digest = digest
        self.c64_rescores += 64 if self.memory_version >= 0 else 0
        self.defect_rescores += len(defects)
        self.pending = True

    def finalize_step(self) -> float:
        b3 = self.shadow.finalize_step()
        if not self.pending:
            self.last_diagnostics["b3_shadow_theta"] = b3
            return self.theta
        self.pending = False
        old = self.theta
        old_b3 = self.previous_b3
        desired = b3
        details: dict[str, Any] = {}
        cdf_ref = tuple(sorted(self.c64)) if self.c64 else ()
        q90 = _higher_c64(cdf_ref) if cdf_ref else b3
        scale = _c64_scale(cdf_ref) if cdf_ref else IQR_FLOOR
        candidates = _finite_candidates(b3, old, cdf_ref,
                                         self.normals, self.defects)

        if self.arm == "M":
            median_delta = (_median(self.c64) - _median(self.previous_c64)
                            if self.previous_c64 else 0.0)
            desired = b3 + (old - old_b3) + median_delta
            details = {"median_shift": median_delta}
        elif self.arm == "S":
            old_q = _higher_c64(tuple(sorted(self.previous_c64))) if self.previous_c64 else q90
            old_scale = _c64_scale(tuple(sorted(self.previous_c64))) if self.previous_c64 else scale
            kappa_prev = (old - old_q) / old_scale
            transported = q90 + kappa_prev * scale
            desired = transported
            if self.normals and self.defects:
                desired, details = self._optimize(candidates, b3, old, scale,
                                                  standardized=True)
                details["transported_standardized_threshold"] = transported
                details["kappa_previous"] = kappa_prev
            else:
                details = {"transported_standardized_threshold": transported,
                           "kappa_previous": kappa_prev,
                           "missing_class": True}
                desired = b3
        elif self.arm == "P":
            if self.defects:
                target_u = _cdf(cdf_ref, _median(self.defects))
                desired, details = _map_percentile(cdf_ref, target_u, b3, candidates)
                details["defect_median"] = _median(self.defects)
            else:
                desired = b3
                details = {"status": "DEFECT_EVIDENCE_UNAVAILABLE"}
        else:  # Full
            if self.normals and self.defects:
                desired, details = self._optimize(candidates, b3, old, scale,
                                                  standardized=False)
            else:
                desired = b3
                details = {"status": "MISSING_CLASS_EVIDENCE",
                           "conflict_reason": None,
                           "normal_count": len(self.normals),
                           "defect_count": len(self.defects)}

        # Exact no-departure fallback follows the real B3 shadow bit-for-bit.
        if (self.arm == "Full" and not self.departed and
                abs(old - old_b3) <= 1e-15 and
                abs(desired - b3) <= 1e-15):
            new_theta = b3
        else:
            new_theta = _clip(old, desired, MOVE_LIMIT)
        if abs(new_theta - b3) > 1e-15:
            self.departed = True
        if desired == b3 and new_theta != b3 and self.departed:
            self.rejoin_updates += 1
        elif new_theta == b3:
            self.rejoin_updates = 0
        self.theta = new_theta
        self.total_updates += 1
        self.last_diagnostics = {
            "status": "UPDATED", "arm": self.arm,
            "b3_shadow_theta": b3, "desired_b3": desired,
            "reachable_theta": new_theta,
            "remaining_b3_distance": abs(new_theta - b3),
            "rejoin_blocked": desired == b3 and new_theta != b3,
            "rejoin_updates": self.rejoin_updates,
            "movement": new_theta - old,
            "movement_limit": MOVE_LIMIT,
            "normal_source_count": len(self.normals),
            "defect_source_count": len(self.defects),
            "evidence_state": ("BOTH_VALID" if self.normals and self.defects else
                               "NORMAL_ONLY" if self.normals else
                               "DEFECT_ONLY" if self.defects else "NONE"),
            "evidence_validity": "CURRENT_VERSION",
            "c64_rescores": self.c64_rescores,
            "defect_anchor_rescores": self.defect_rescores,
            **details,
        }
        return self.theta

    def _optimize(self, candidates: tuple[float, ...], theta_b3: float,
                  theta_prev: float, scale: float, *, standardized: bool
                  ) -> tuple[float, dict[str, Any]]:
        if not self.normals or not self.defects:
            return theta_b3, {"status": "MISSING_CLASS_EVIDENCE"}
        penalty = _risk_penalty(len(candidates), len(self.normals), len(self.defects))
        base_q = _higher_c64(tuple(sorted(self.c64)))
        prior_q = (_higher_c64(tuple(sorted(self.previous_c64)))
                   if self.previous_c64 else base_q)
        prior_scale = (_c64_scale(tuple(sorted(self.previous_c64)))
                       if self.previous_c64 else scale)
        kappa_prev = (theta_prev - prior_q) / prior_scale
        current_cdf = tuple(sorted(self.c64))
        previous_cdf = tuple(sorted(self.previous_c64 or self.c64))
        u_b3 = _cdf(current_cdf, theta_b3)
        rho_previous = (_cdf(previous_cdf, self.previous_theta) -
                        _cdf(previous_cdf, self.previous_b3))
        u_transported = min(1.0, max(0.0, u_b3 + rho_previous))
        scored = []
        baseline_j = _empirical_j(theta_b3, self.normals, self.defects)
        for candidate in candidates:
            delta_j = _empirical_j(candidate, self.normals, self.defects) - baseline_j
            if standardized:
                candidate_k = (candidate - base_q) / scale
                b3_k = (theta_b3 - base_q) / scale
                objective = (delta_j - penalty -
                             LAMBDA_BASE * abs(candidate_k - b3_k) -
                             LAMBDA_MOVE * abs(candidate_k - kappa_prev))
            else:
                objective = (delta_j - penalty -
                             LAMBDA_BASE * abs(_cdf(current_cdf, candidate) - u_b3) -
                             LAMBDA_MOVE * abs(_cdf(current_cdf, candidate) - u_transported))
            scored.append((objective, candidate))
        best_objective = max(value for value, _ in scored)
        tied = [candidate for value, candidate in scored if value == best_objective]
        best = min(tied, key=lambda x: (abs(x - theta_b3), abs(x - theta_prev), x))
        accepted = best_objective > 0.0
        same_distribution = _same_normalized_empirical_distribution(self.normals, self.defects)
        conflict_reason = ("UNIDENTIFIABLE_SCORE_DISTRIBUTIONS" if same_distribution else
                           "NONPOSITIVE_PENALIZED_GAIN" if not accepted else None)
        diagnostics = {
            "status": ("POSITIVE_PENALIZED_GAIN" if accepted else
                       "CONFLICT_B3_FALLBACK" if same_distribution else "B3_FALLBACK"),
            "conflict_reason": conflict_reason,
            "candidate_count": len(candidates),
            "best_penalized_objective": best_objective,
            "best_empirical_delta_j": _empirical_j(best, self.normals, self.defects) - baseline_j,
            "reliability_penalty_heuristic": penalty,
            "reliability_penalty_is_confidence_bound": False,
            "lambda_base": LAMBDA_BASE, "lambda_move": LAMBDA_MOVE,
            "scale": scale, "standardized_objective": standardized,
            "candidate_threshold": best,
        }
        if not standardized:
            diagnostics.update({
                "b3_percentile": u_b3,
                "previous_b8_relative_percentile": rho_previous,
                "transported_percentile_target": u_transported,
                "selected_candidate_percentile": _cdf(current_cdf, best),
                "percentile_regularization": True,
                "raw_score_candidate_retained": True,
            })
        return (best if accepted else theta_b3), diagnostics

    def state_dict(self) -> dict[str, Any]:
        return {
            "version": self.version, "arm": self.arm, "theta": self.theta,
            "shadow": self.shadow.state_dict(), "evidence_digest": self.evidence_digest,
            "memory_version": self.memory_version,
            "evidence_revision": self.evidence_revision,
            "c64": list(self.c64), "normals": list(self.normals),
            "defects": list(self.defects), "normal_ids": list(self.normal_ids),
            "defect_ids": list(self.defect_ids),
            "previous_c64": list(self.previous_c64),
            "previous_b3": self.previous_b3, "previous_theta": self.previous_theta,
            "pending": self.pending, "departed": self.departed,
            "rejoin_updates": self.rejoin_updates, "total_updates": self.total_updates,
            "c64_rescores": self.c64_rescores,
            "defect_rescores": self.defect_rescores,
            "last_diagnostics": self.last_diagnostics,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if state.get("version") != self.version or state.get("arm") != self.arm:
            raise ValueError("B8 checkpoint method identity mismatch")
        self.theta = _finite(state["theta"])
        self.shadow.load_state_dict(state["shadow"])
        self.evidence_digest = str(state["evidence_digest"])
        self.memory_version = int(state["memory_version"])
        self.evidence_revision = int(state["evidence_revision"])
        self.c64 = tuple(_finite(x) for x in state["c64"])
        self.normals = tuple(_finite(x) for x in state["normals"])
        self.defects = tuple(_finite(x) for x in state["defects"])
        self.normal_ids = tuple(map(str, state["normal_ids"]))
        self.defect_ids = tuple(map(str, state["defect_ids"]))
        self.previous_c64 = tuple(_finite(x) for x in state["previous_c64"])
        self.previous_b3 = _finite(state["previous_b3"])
        self.previous_theta = _finite(state["previous_theta"])
        self.pending = bool(state["pending"])
        self.departed = bool(state["departed"])
        self.rejoin_updates = int(state["rejoin_updates"])
        self.total_updates = int(state["total_updates"])
        self.c64_rescores = int(state["c64_rescores"])
        self.defect_rescores = int(state["defect_rescores"])
        self.last_diagnostics = dict(state["last_diagnostics"])


def _eligible_scores(evidence: EvidenceSnapshot) -> tuple[tuple[float, ...], tuple[float, ...]]:
    normals = tuple(float(a.score) for a in evidence.normal_anchors if not a.protected and
                    a.source_exclusion_policy ==
                    "queried_normal_all_layer_leave_source_out")
    defects = tuple(float(a.score) for a in evidence.defect_anchors
                    if a.source_exclusion_policy == "queried_defect_not_written")
    if any(not math.isfinite(x) for x in (*normals, *defects)):
        raise ValueError("Nonfinite eligible evidence")
    return normals, defects


def _same_normalized_empirical_distribution(
        normals: tuple[float, ...], defects: tuple[float, ...]) -> bool:
    if not normals or not defects:
        return False
    n_hist, d_hist = Counter(normals), Counter(defects)
    if set(n_hist) != set(d_hist):
        return False
    return all(n_hist[value] / len(normals) == d_hist[value] / len(defects)
               for value in n_hist)


def _evidence_digest(evidence: EvidenceSnapshot) -> str:
    import hashlib
    from .profiles import canonical_json
    payload = {
        "memory_version": evidence.memory_version,
        "evidence_revision": evidence.evidence_revision,
        "normals": [(a.opaque_anchor_id, a.score, a.memory_version,
                     a.source_exclusion_policy) for a in evidence.normal_anchors],
        "defects": [(a.opaque_anchor_id, a.score, a.memory_version,
                     a.source_exclusion_policy) for a in evidence.defect_anchors],
        "audit": dict(evidence.source_exclusion_audit),
    }
    return hashlib.sha256(canonical_json(payload).encode()).hexdigest()


def _clip(previous: float, desired: float, step: float) -> float:
    return min(max(float(desired), float(previous) - step), float(previous) + step)


def _finite(value: Any) -> float:
    value = float(value)
    if not math.isfinite(value):
        raise ValueError("Nonfinite threshold or score in controller state")
    return value
