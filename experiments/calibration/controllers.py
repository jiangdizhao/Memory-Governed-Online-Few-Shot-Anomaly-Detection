"""Stage 0 threshold controllers with explicit information permissions."""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
import math
from typing import Any, Mapping, Protocol

from .protocol import EvidenceSnapshot, Feedback, Observation, Prediction


def higher_quantile(values: list[float] | tuple[float, ...], q: float) -> float:
    """NumPy ``method='higher'`` order statistic, including boundary ties."""
    if not values or not 0 <= q <= 1:
        raise ValueError("higher_quantile needs data and q in [0,1]")
    ordered = sorted(float(x) for x in values)
    index = math.ceil(q * (len(ordered) - 1))
    return ordered[index]


def theta0_from_c64(c64_scores: list[float] | tuple[float, ...]) -> float:
    """D2 q90 threshold: C64 are 64 additional normals, not four-shot total."""
    if len(c64_scores) != 64 or any(not math.isfinite(float(x)) for x in c64_scores):
        raise ValueError("theta0 requires exactly 64 finite additional C64 scores")
    # method='higher' at q=.90 selects zero-based index ceil(.9*63)=57.
    return higher_quantile(list(c64_scores), .90)


class Controller(Protocol):
    name: str
    theta: float
    def predict(self, observation: Observation) -> Prediction: ...
    def observe_score(self, observation: Observation) -> None: ...
    def observe_feedback(self, observation: Observation, prediction: Prediction,
                         feedback: Feedback) -> None: ...
    def on_memory_change(self, evidence: EvidenceSnapshot) -> None: ...
    def finalize_step(self) -> float: ...
    def state_dict(self) -> dict[str, Any]: ...


@dataclass
class FrozenTheta:
    theta: float
    name: str = "B0"
    def predict(self, observation: Observation) -> Prediction:
        return Prediction(int(observation.score >= self.theta), self.theta)
    def observe_score(self, observation: Observation) -> None: pass
    def observe_feedback(self, observation: Observation, prediction: Prediction,
                         feedback: Feedback) -> None: pass
    def on_memory_change(self, evidence: EvidenceSnapshot) -> None: pass
    def finalize_step(self) -> float: return self.theta
    def state_dict(self) -> dict[str, Any]: return {"name": self.name, "theta": self.theta}


@dataclass
class RollingQ90QueriedNormal:
    """B1 stale-score rolling q90; only queried verified normals are eligible."""
    theta: float
    window: int = 256
    min_count: int = 16
    name: str = "B1"
    history: deque[float] = field(default_factory=deque)
    def __post_init__(self) -> None:
        if self.window < 1 or not 1 <= self.min_count <= self.window:
            raise ValueError("Invalid fixed B1 window/min_count")
        self.history = deque(self.history, maxlen=self.window)
    def predict(self, observation: Observation) -> Prediction:
        return Prediction(int(observation.score >= self.theta), self.theta)
    def observe_score(self, observation: Observation) -> None: pass
    def observe_feedback(self, observation: Observation, prediction: Prediction,
                         feedback: Feedback) -> None:
        if feedback.queried and feedback.label == 0:
            self.history.append(float(observation.score))
            if len(self.history) >= self.min_count:
                self.theta = higher_quantile(list(self.history), .90)
    def on_memory_change(self, evidence: EvidenceSnapshot) -> None: pass
    def finalize_step(self) -> float: return self.theta
    def state_dict(self) -> dict[str, Any]:
        return {"name": self.name, "theta": self.theta, "window": self.window,
                "min_count": self.min_count, "history": list(self.history),
                "score_semantics": "stale_at_arrival"}


@dataclass
class StateConsistentQuantile:
    """B3 protected-C64 current-memory q90 with a preregistered step clip."""
    theta: float
    delta: float = 0.01
    min_normals: int = 16
    name: str = "B3"
    evidence_version: int = -1
    normal_scores: tuple[float, ...] = ()
    status: str = "NORMAL_EVIDENCE_INSUFFICIENT"
    fnr_status: str = "FNR_UNAVAILABLE"
    _update_pending: bool = False
    def predict(self, observation: Observation) -> Prediction:
        return Prediction(int(observation.score >= self.theta), self.theta,
                          self.status)
    def observe_score(self, observation: Observation) -> None: pass
    def observe_feedback(self, observation: Observation, prediction: Prediction,
                         feedback: Feedback) -> None: pass
    def on_memory_change(self, evidence: EvidenceSnapshot) -> None:
        if evidence.memory_version == self.evidence_version:
            return
        self.evidence_version = evidence.memory_version
        self._update_pending = True
        self.normal_scores = tuple(a.score for a in evidence.normal_anchors
                                   if a.protected)
        self.fnr_status = "AVAILABLE" if evidence.defect_anchors else "FNR_UNAVAILABLE"
        if len(self.normal_scores) < self.min_normals:
            self.status = "NORMAL_EVIDENCE_INSUFFICIENT"
        else:
            self.status = "CURRENT_C64_Q90"
    def finalize_step(self) -> float:
        if self._update_pending and self.status == "CURRENT_C64_Q90":
            raw = higher_quantile(list(self.normal_scores), .90)
            self.theta = min(max(raw, self.theta - self.delta), self.theta + self.delta)
        self._update_pending = False
        return self.theta
    def state_dict(self) -> dict[str, Any]:
        return {"name": self.name, "theta": self.theta, "delta": self.delta,
                "min_normals": self.min_normals,
                "evidence_version": self.evidence_version,
                "normal_scores": list(self.normal_scores), "status": self.status,
                "fnr_status": self.fnr_status,
                "update_pending": self._update_pending}


class OriginalGuardedSidecar:
    """B2 wrapper over the historical v22 controller with declared adaptations.

    The original score-only normal fallback is disabled because unqueried labels
    are unavailable in this protocol. It receives only queried feedback and
    initializes from the additional C64 scalar scores, using its native NumPy
    interpolation convention; this is explicitly not the theta0 D2 estimator.
    """
    name = "B2"
    def __init__(self, c64_scores: list[float], theta: float):
        try:
            import test_phase3_AIF_simplified_v22 as v22
            cfg = v22.GuardedThetaSidecarConfig(enable_normal_fallback=False)
            fallback = v22.ThresholdConfig(target_fpr=0.10)
            self.source = v22.GuardedDualAnchorThresholdController(cfg, fallback)
            self.source.init_from_normal_calibration([], list(c64_scores),
                                                     mode="raw_quantile")
        except Exception as exc:
            raise RuntimeError(f"Could not initialize original v22 sidecar: {exc}") from exc
        # Keep the source method's own q90 initialization. The supplied D2
        # theta is recorded for comparison, not imposed on this comparator.
        self.theta = float(self.source.theta())
        self.adaptation = {
            "score_only_normal_fallback": "disabled; no unqueried feedback",
            "initial_source": "additional C64 scores under M0",
            "source_quantile": "native numpy.quantile interpolation",
            "d2_theta0_for_comparison": float(theta),
            "query_feedback": "queried verified labels only",
        }
    def predict(self, observation: Observation) -> Prediction:
        self.theta = float(self.source.theta())
        return Prediction(int(observation.score >= self.theta), self.theta,
                          str(self.source.last_mode))
    def observe_score(self, observation: Observation) -> None:
        # Historical source explicitly ignores unlabeled observations.
        self.source.observe_unlabeled(float(observation.score), 0.0,
                                      bool(observation.score >= self.theta))
    def observe_feedback(self, observation: Observation, prediction: Prediction,
                         feedback: Feedback) -> None:
        if feedback.queried:
            self.source.observe_labeled(float(observation.score), int(feedback.label))
            self.theta = float(self.source.theta())
    def on_memory_change(self, evidence: EvidenceSnapshot) -> None:
        # Original sidecar operates on observed scalar scores and does not
        # refresh historical anchors under a changed mutable bank.
        return
    def finalize_step(self) -> float:
        self.theta = float(self.source.theta())
        return self.theta
    def state_dict(self) -> dict[str, Any]:
        state = self.source.debug_state()
        state.update({"name": self.name, "theta": self.theta,
                      "adaptation": self.adaptation})
        return state


@dataclass
class AdaptedDualACI:
    """B4: queried-label class-conditional ACI adaptation (no ACI guarantee)."""
    theta: float
    alpha_target: float = 0.10
    beta_target: float = 0.10
    gamma_normal: float = 0.005
    gamma_defect: float = 0.005
    alpha_normal: float = 0.10
    alpha_defect: float = 0.10
    name: str = "B4"
    evidence_version: int = -1
    evidence_revision: int = -1
    normal_scores: tuple[float, ...] = ()
    defect_scores: tuple[float, ...] = ()
    conflict: bool = False
    last_status: str = "NONE"
    last_prediction_error: bool | None = None
    def predict(self, observation: Observation) -> Prediction:
        return Prediction(int(observation.score >= self.theta), self.theta,
                          "CONFLICT_HOLD" if self.conflict else self.last_status)
    def observe_score(self, observation: Observation) -> None: pass
    def observe_feedback(self, observation: Observation, prediction: Prediction,
                         feedback: Feedback) -> None:
        self.last_prediction_error = None
        if not feedback.queried:
            return
        if feedback.label == 0:
            error = int(prediction.label == 1)
            self.alpha_normal = min(1 - 1e-6, max(1e-6,
                self.alpha_normal + self.gamma_normal * (self.alpha_target - error)))
            self.last_prediction_error = bool(error)
        else:
            error = int(prediction.label == 0)
            self.alpha_defect = min(1 - 1e-6, max(1e-6,
                self.alpha_defect + self.gamma_defect * (self.beta_target - error)))
            self.last_prediction_error = bool(error)
    def on_memory_change(self, evidence: EvidenceSnapshot) -> None:
        if (evidence.memory_version == self.evidence_version and
                evidence.evidence_revision == self.evidence_revision):
            return
        self.evidence_version = evidence.memory_version
        self.evidence_revision = evidence.evidence_revision
        self.normal_scores = tuple(x.score for x in evidence.normal_anchors)
        self.defect_scores = tuple(x.score for x in evidence.defect_anchors)
        self.last_status = evidence.state
    def finalize_step(self) -> float:
        lower = (higher_quantile(list(self.defect_scores), self.alpha_defect)
                 if self.defect_scores else None)
        upper = (higher_quantile(list(self.normal_scores), 1 - self.alpha_normal)
                 if self.normal_scores else None)
        if lower is not None and upper is not None:
            self.conflict = lower > upper
            if not self.conflict:
                self.theta = min(max(self.theta, lower), upper)
        elif upper is not None:
            self.conflict = False
            self.theta = max(self.theta, upper)
            self.last_status = "NORMAL_ONLY"
        elif lower is not None:
            self.conflict = False
            self.theta = min(self.theta, lower)
            self.last_status = "DEFECT_ONLY"
        else:
            self.conflict = False
            self.last_status = "NONE"
        return self.theta
    def state_dict(self) -> dict[str, Any]:
        return {"name": self.name, "theta": self.theta,
                "alpha_target": self.alpha_target, "beta_target": self.beta_target,
                "gamma_normal": self.gamma_normal, "gamma_defect": self.gamma_defect,
                "alpha_normal": self.alpha_normal, "alpha_defect": self.alpha_defect,
                "evidence_version": self.evidence_version,
                "evidence_revision": self.evidence_revision,
                "normal_scores": list(self.normal_scores),
                "defect_scores": list(self.defect_scores),
                "conflict": self.conflict, "status": self.last_status}


class BlockedController:
    """Registered method whose exact behavior is unresolved; never predicts."""
    def __init__(self, name: str, reason: str, theta: float):
        self.name, self.reason, self.theta = name, reason, float(theta)
    def predict(self, observation: Observation) -> Prediction:
        raise RuntimeError(f"{self.name} BLOCKED_FIDELITY: {self.reason}")
    def observe_score(self, observation: Observation) -> None: pass
    def observe_feedback(self, observation: Observation, prediction: Prediction,
                         feedback: Feedback) -> None: pass
    def on_memory_change(self, evidence: EvidenceSnapshot) -> None: pass
    def finalize_step(self) -> float: return self.theta
    def state_dict(self) -> dict[str, Any]:
        return {"name": self.name, "status": "BLOCKED_FIDELITY", "reason": self.reason}
