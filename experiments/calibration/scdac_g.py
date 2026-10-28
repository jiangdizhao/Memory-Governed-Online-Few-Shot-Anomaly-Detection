"""Provisional, executable SCDAC-G v0 mechanism.

The internal proposal specifies the objective family and causal/evidence
semantics but leaves several numeric choices open. Defaults below are therefore
versioned provisional design choices, not a uniquely prescribed algorithm and
not statistical safety guarantees.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from typing import Any

from .protocol import EvidenceSnapshot, Observation, Prediction


class SCDACGError(RuntimeError):
    pass


@dataclass(frozen=True)
class SCDACGConfig:
    version: str = "SCDAC-G-v0-PROVISIONAL_METHOD_SPEC"
    min_normals: int = 16
    min_defects: int = 8
    confidence_delta: float = 0.05
    support_fpr: float = 0.20
    support_fnr: float = 0.20
    diagnostic_fpr_targets: tuple[float, ...] = (0.05, 0.10, 0.20)
    movement_penalty: float = 0.05
    max_threshold_step: float = 0.01
    score_scale_epsilon: float = 1e-12

    def __post_init__(self) -> None:
        if self.version != "SCDAC-G-v0-PROVISIONAL_METHOD_SPEC":
            raise ValueError("SCDAC-G method version must be explicit")
        if self.min_normals < 1 or self.min_defects < 1:
            raise ValueError("Minimum evidence counts must be positive")
        if not 0.0 < self.confidence_delta < 1.0:
            raise ValueError("confidence_delta must be inside (0,1)")
        if not 0.0 < self.support_fpr < 1.0 or not 0.0 < self.support_fnr < 1.0:
            raise ValueError("Descriptive support risks must be inside (0,1)")
        if (tuple(sorted(self.diagnostic_fpr_targets)) != self.diagnostic_fpr_targets or
                self.diagnostic_fpr_targets != (0.05, 0.10, 0.20)):
            raise ValueError("The diagnostic FPR grid is frozen at .05/.10/.20")
        if self.movement_penalty < 0 or self.max_threshold_step <= 0:
            raise ValueError("Movement penalty/step must be nonnegative/positive")
        if self.score_scale_epsilon <= 0:
            raise ValueError("Score scale epsilon must be positive")


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False)


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def _cp_one_sided_upper(errors: int, n: int, delta: float) -> float:
    """One-sided Clopper-Pearson upper endpoint, used descriptively only.

    The IID binomial assumptions and simultaneous post-selection validity do
    not hold automatically for selectively disclosed, dependent labels and
    repeatedly rescored anchors. This function is a small-sample guard only.
    """
    if n <= 0 or not 0 <= errors <= n:
        raise ValueError("Invalid binomial error count")
    if errors == n:
        return 1.0
    try:
        from scipy.stats import beta
    except Exception as exc:  # fail closed rather than use a different bound
        raise SCDACGError("SciPy is required for the registered CP descriptive bound") from exc
    return min(1.0, float(beta.ppf(1.0 - delta, errors + 1, n - errors)))


class SCDACGController:
    """B7 controller with current-state evidence, explicit states and bounded moves."""

    name = "B7"

    def __init__(self, theta: float, config: SCDACGConfig = SCDACGConfig()):
        if not math.isfinite(float(theta)):
            raise ValueError("Initial SCDAC-G threshold must be finite")
        self.theta = float(theta)
        self.config = config
        self.memory_version = -1
        self.evidence_revision = -1
        self.state = "NONE"
        self.decision = "INITIAL_HOLD"
        self.normal_scores: tuple[tuple[str, float], ...] = ()
        self.defect_scores: tuple[tuple[str, float], ...] = ()
        self.normal_count_raw = 0
        self.defect_count_raw = 0
        self.evidence_digest = ""
        self.pending = False
        self.last_two_sided_supported_interval: dict[str, Any] | None = None
        self.interval_stale = False
        self.last_diagnostics: dict[str, Any] = {
            "status": "NONE", "decision": self.decision,
            "fpr_upper": None, "fnr_upper": None,
            "risk_constrained": [], "candidate_count": 0,
            "candidate_evaluations": 0,
            "bound_evaluations": 0,
            "bound_name": "one-sided Clopper-Pearson descriptive_upper",
            "guarantee": "none; dependent/selective/dynamic-state assumptions are unverified",
        }
        self.total_candidate_evaluations = 0
        self.total_bound_evaluations = 0

    def predict(self, observation: Observation) -> Prediction:
        return Prediction(int(observation.score >= self.theta), self.theta,
                          self.state, self.diagnostics())

    def observe_score(self, observation: Observation) -> None:
        # The method uses anchors only; the stream score is not retrospectively
        # allowed to change the current prediction.
        return

    def observe_feedback(self, observation: Observation, prediction: Prediction,
                         feedback: Any) -> None:
        # Feedback reaches B7 through the versioned ledger, never as an
        # unversioned scalar score buffer.
        return

    def on_memory_change(self, evidence: EvidenceSnapshot) -> None:
        if (evidence.memory_version == self.memory_version and
                evidence.evidence_revision == self.evidence_revision):
            return
        normals = []
        for anchor in evidence.normal_anchors:
            if anchor.memory_version != evidence.memory_version:
                raise SCDACGError("Stale normal anchor rejected")
            if anchor.protected:
                normals.append((anchor.opaque_anchor_id, float(anchor.score)))
            elif anchor.source_exclusion_policy == "queried_normal_all_layer_leave_source_out":
                normals.append((anchor.opaque_anchor_id, float(anchor.score)))
        defects = []
        for anchor in evidence.defect_anchors:
            if anchor.memory_version != evidence.memory_version:
                raise SCDACGError("Stale defect anchor rejected")
            if anchor.source_exclusion_policy == "queried_defect_not_written":
                defects.append((anchor.opaque_anchor_id, float(anchor.score)))

        self.memory_version = evidence.memory_version
        self.evidence_revision = evidence.evidence_revision
        self.normal_count_raw = len(evidence.normal_anchors)
        self.defect_count_raw = len(evidence.defect_anchors)
        self.normal_scores = tuple(sorted(normals))
        self.defect_scores = tuple(sorted(defects))
        self.evidence_digest = _digest({
            "memory_version": evidence.memory_version,
            "normal": self.normal_scores,
            "defect": self.defect_scores,
            "source_exclusion_audit": dict(evidence.source_exclusion_audit),
        })
        n_ok = len(self.normal_scores) >= self.config.min_normals
        d_ok = len(self.defect_scores) >= self.config.min_defects
        self.state = ("BOTH" if n_ok and d_ok else
                      "NORMAL_ONLY" if n_ok else
                      "DEFECT_ONLY" if d_ok else "NONE")
        self.pending = True
        interval = self.last_two_sided_supported_interval
        self.interval_stale = bool(interval and
                                   (interval["memory_version"] != self.memory_version or
                                    interval["evidence_digest"] != self.evidence_digest))

    @staticmethod
    def _grid(theta: float, scores: tuple[float, ...]) -> tuple[float, ...]:
        unique = sorted(set(float(s) for s in scores))
        candidates = {float(theta)}
        for value in unique:
            candidates.add(value)
            low = math.nextafter(value, -math.inf)
            high = math.nextafter(value, math.inf)
            if math.isfinite(low):
                candidates.add(low)
            if math.isfinite(high):
                candidates.add(high)
        for left, right in zip(unique, unique[1:]):
            midpoint = left / 2.0 + right / 2.0
            if math.isfinite(midpoint):
                candidates.add(midpoint)
        return tuple(sorted(x for x in candidates if math.isfinite(x)))

    def _upper_risks(self, theta: float, normals: tuple[tuple[str, float], ...],
                     defects: tuple[tuple[str, float], ...]) -> tuple[float, float]:
        n_fp = sum(score >= theta for _, score in normals)
        n_fn = sum(score < theta for _, score in defects)
        return (_cp_one_sided_upper(n_fp, len(normals), self.config.confidence_delta),
                _cp_one_sided_upper(n_fn, len(defects), self.config.confidence_delta))

    def _score_scale(self) -> float:
        values = [s for _, s in (*self.normal_scores, *self.defect_scores)]
        if len(values) < 2:
            return 0.0
        mean = math.fsum(values) / len(values)
        variance = math.fsum((x - mean) ** 2 for x in values) / len(values)
        return math.sqrt(max(0.0, variance))

    def _objective(self, theta: float, fpr_u: float, fnr_u: float) -> float:
        scale = self._score_scale()
        return (0.5 * fpr_u + 0.5 * fnr_u +
                self.config.movement_penalty * abs(theta - self.theta) /
                (scale + self.config.score_scale_epsilon))

    def _risk_diagnostics(self, candidates: tuple[float, ...],
                          rows: dict[float, tuple[float, float]]) -> list[dict[str, Any]]:
        reports = []
        for target in self.config.diagnostic_fpr_targets:
            feasible = [(self._objective(theta, *risk), abs(theta - self.theta), theta, risk)
                        for theta, risk in rows.items() if risk[0] <= target]
            if not feasible:
                reports.append({"fpr_target": target, "status": "NO_DESCRIPTIVE_FEASIBLE_THRESHOLD",
                                "threshold": None, "fpr_upper": None, "fnr_upper": None})
                continue
            _, _, chosen, (fpr_u, fnr_u) = min(feasible)
            reports.append({"fpr_target": target, "status": "AVAILABLE",
                            "threshold": chosen, "fpr_upper": fpr_u,
                            "fnr_upper": fnr_u})
        return reports

    def finalize_step(self) -> float:
        if not self.pending:
            return self.theta
        self.pending = False
        if self.state != "BOTH":
            unavailable = ("_FNR_UNAVAILABLE" if self.state == "NORMAL_ONLY" else
                           "_FPR_UNAVAILABLE" if self.state == "DEFECT_ONLY" else
                           "_FPR_FNR_UNAVAILABLE")
            self.decision = f"HOLD_{self.state}{unavailable}"
            self.last_diagnostics = {
                "status": self.state, "decision": self.decision,
                "normal_count": len(self.normal_scores), "defect_count": len(self.defect_scores),
                "raw_normal_count": self.normal_count_raw,
                "raw_defect_count": self.defect_count_raw,
                "fpr_upper": None if not self.normal_scores else "NOT_OPTIMIZED_WITHOUT_BOTH_SIDES",
                "fnr_upper": None if not self.defect_scores else "NOT_OPTIMIZED_WITHOUT_BOTH_SIDES",
                "risk_constrained": [], "candidate_count": 0,
                "candidate_evaluations": 0, "bound_evaluations": 0,
                "bound_name": "one-sided Clopper-Pearson descriptive_upper",
                "guarantee": "none; dependent/selective/dynamic-state assumptions are unverified",
                "last_two_sided_supported_interval": self.last_two_sided_supported_interval,
                "interval_stale": self.interval_stale,
            }
            return self.theta

        normal_values = tuple(score for _, score in self.normal_scores)
        defect_values = tuple(score for _, score in self.defect_scores)
        candidates = self._grid(self.theta, normal_values + defect_values)
        rows = {candidate: self._upper_risks(candidate, self.normal_scores,
                                             self.defect_scores)
                for candidate in candidates}
        self.total_candidate_evaluations += len(candidates)
        self.total_bound_evaluations += 2 * len(candidates)
        feasible = [theta for theta, (fp, fn) in rows.items()
                    if fp <= self.config.support_fpr and fn <= self.config.support_fnr]
        if feasible:
            self.last_two_sided_supported_interval = {
                "lower": min(feasible), "upper": max(feasible),
                "memory_version": self.memory_version,
                "evidence_revision": self.evidence_revision,
                "evidence_digest": self.evidence_digest,
                "normal_count": len(self.normal_scores),
                "defect_count": len(self.defect_scores),
                "support_fpr": self.config.support_fpr,
                "support_fnr": self.config.support_fnr,
                "bound_label": "descriptive_upper_not_certified",
                "active": True,
            }
            self.interval_stale = False
            self.state = "BOTH"
        else:
            self.state = "CONFLICT"
            if self.last_two_sided_supported_interval:
                self.last_two_sided_supported_interval = {
                    **self.last_two_sided_supported_interval, "active": False}
                self.interval_stale = True

        ranked = [(self._objective(theta, *risk), abs(theta - self.theta), theta, risk)
                  for theta, risk in rows.items()]
        _, _, proposed, proposed_risk = min(ranked)
        lower_step = self.theta - self.config.max_threshold_step
        upper_step = self.theta + self.config.max_threshold_step
        updated = min(max(proposed, lower_step), upper_step)
        self.theta = float(updated)
        self.decision = ("BA_MODE_MINIMUM_DESCRIPTIVE_RISK" if self.state == "BOTH"
                         else "CONFLICT_BOUNDED_MINIMUM_DESCRIPTIVE_RISK")
        selected_risk = self._upper_risks(self.theta, self.normal_scores,
                                          self.defect_scores)
        self.total_bound_evaluations += 2
        self.last_diagnostics = {
            "status": self.state, "decision": self.decision,
            "normal_count": len(self.normal_scores),
            "defect_count": len(self.defect_scores),
            "fpr_upper": selected_risk[0], "fnr_upper": selected_risk[1],
            "proposed_threshold": proposed, "threshold": self.theta,
            "objective": self._objective(self.theta, *selected_risk),
            "score_scale": self._score_scale(),
            "risk_constrained": self._risk_diagnostics(candidates, rows),
            "candidate_count": len(candidates),
            "candidate_evaluations": len(candidates),
            "bound_evaluations": 2 * len(candidates) + 2,
            "total_candidate_evaluations": self.total_candidate_evaluations,
            "total_bound_evaluations": self.total_bound_evaluations,
            "bound_name": "one-sided Clopper-Pearson descriptive_upper",
            "confidence_delta": self.config.confidence_delta,
            "guarantee": "none; selective labels, serial dependence, memory rescore and post-selection invalidate transfer",
            "memory_version": self.memory_version,
            "evidence_revision": self.evidence_revision,
            "evidence_digest": self.evidence_digest,
            "last_two_sided_supported_interval": self.last_two_sided_supported_interval,
            "interval_stale": self.interval_stale,
        }
        return self.theta

    def diagnostics(self) -> dict[str, Any]:
        return dict(self.last_diagnostics)

    def state_dict(self) -> dict[str, Any]:
        return {
            "name": self.name, "config": asdict(self.config),
            "theta": self.theta, "memory_version": self.memory_version,
            "evidence_revision": self.evidence_revision, "state": self.state,
            "decision": self.decision, "normal_scores": list(self.normal_scores),
            "defect_scores": list(self.defect_scores),
            "normal_count_raw": self.normal_count_raw,
            "defect_count_raw": self.defect_count_raw,
            "evidence_digest": self.evidence_digest, "pending": self.pending,
            "last_two_sided_supported_interval": self.last_two_sided_supported_interval,
            "interval_stale": self.interval_stale,
            "last_diagnostics": self.last_diagnostics,
            "total_candidate_evaluations": self.total_candidate_evaluations,
            "total_bound_evaluations": self.total_bound_evaluations,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if state.get("name") != self.name or state.get("config") != asdict(self.config):
            raise SCDACGError("Checkpoint method/config mismatch")
        theta = float(state["theta"])
        if not math.isfinite(theta):
            raise SCDACGError("Checkpoint threshold is nonfinite")
        self.theta = theta
        for key in ("memory_version", "evidence_revision", "state", "decision",
                    "normal_count_raw", "defect_count_raw", "evidence_digest",
                    "pending", "last_two_sided_supported_interval", "interval_stale",
                    "last_diagnostics", "total_candidate_evaluations",
                    "total_bound_evaluations"):
            setattr(self, key, state[key])
        self.normal_scores = tuple((str(k), float(v)) for k, v in state["normal_scores"])
        self.defect_scores = tuple((str(k), float(v)) for k, v in state["defect_scores"])
