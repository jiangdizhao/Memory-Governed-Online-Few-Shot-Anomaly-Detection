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
    def load_state_dict(self, state: dict[str, Any]) -> None:
        if state.get("name") != self.name:
            raise ValueError("Frozen controller identity mismatch")
        self.theta = float(state["theta"])


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
    def load_state_dict(self, state: dict[str, Any]) -> None:
        if (state.get("name") != self.name or state.get("window") != self.window or
                state.get("min_count") != self.min_count or
                state.get("score_semantics") != "stale_at_arrival"):
            raise ValueError("B1 checkpoint configuration mismatch")
        history = [float(x) for x in state["history"]]
        if len(history) > self.window:
            raise ValueError("B1 checkpoint exceeds configured history window")
        self.theta = float(state["theta"])
        self.history = deque(history, maxlen=self.window)


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
    def load_state_dict(self, state: dict[str, Any]) -> None:
        if (state.get("name") != self.name or state.get("delta") != self.delta or
                state.get("min_normals") != self.min_normals):
            raise ValueError("B3 checkpoint configuration mismatch")
        self.theta = float(state["theta"])
        self.evidence_version = int(state["evidence_version"])
        self.normal_scores = tuple(float(x) for x in state["normal_scores"])
        self.status = str(state["status"])
        self.fnr_status = str(state["fnr_status"])
        self._update_pending = bool(state["update_pending"])


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
                      "adaptation": self.adaptation,
                      "source_checkpoint": {
                          key: value for key, value in self.source.__dict__.items()
                          if key not in {"cfg", "fallback_cfg"}}})
        return state
    def load_state_dict(self, state: dict[str, Any]) -> None:
        if state.get("name") != self.name or state.get("adaptation") != self.adaptation:
            raise ValueError("B2 checkpoint identity/adaptation mismatch")
        source_state = state.get("source_checkpoint")
        if not isinstance(source_state, dict):
            raise ValueError("B2 checkpoint lacks native source state")
        allowed = set(self.source.__dict__) - {"cfg", "fallback_cfg"}
        if set(source_state) != allowed:
            raise ValueError("B2 checkpoint fields differ from historical source")
        for key, value in source_state.items():
            setattr(self.source, key, value)
        self.theta = float(state["theta"])


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
        # For predict_defect = score >= theta, normals impose a *lower*
        # threshold bound and defects impose an *upper* bound. Use exact
        # empirical error-count boundaries instead of interpolated quantiles:
        # equality at a normal score is a false positive.
        lower = (_normal_fpr_lower_bound(self.normal_scores, self.alpha_normal)
                 if self.normal_scores else None)
        upper = (_defect_fnr_upper_bound(self.defect_scores, self.alpha_defect)
                 if self.defect_scores else None)
        if lower is not None and upper is not None:
            self.conflict = lower > upper
            if self.conflict:
                self.last_status = "CONFLICT"
            else:
                self.theta = min(max(self.theta, lower), upper)
                self.last_status = "BOTH"
        elif lower is not None:
            self.conflict = False
            self.theta = max(self.theta, lower)
            self.last_status = "NORMAL_ONLY"
        elif upper is not None:
            self.conflict = False
            self.theta = min(self.theta, upper)
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
                "conflict": self.conflict, "status": self.last_status,
                "last_prediction_error": self.last_prediction_error}
    def load_state_dict(self, state: dict[str, Any]) -> None:
        immutable = {"alpha_target": self.alpha_target,
                     "beta_target": self.beta_target,
                     "gamma_normal": self.gamma_normal,
                     "gamma_defect": self.gamma_defect}
        if state.get("name") != self.name or any(
                state.get(key) != value for key, value in immutable.items()):
            raise ValueError("B4 checkpoint configuration mismatch")
        self.theta = float(state["theta"])
        self.alpha_normal = float(state["alpha_normal"])
        self.alpha_defect = float(state["alpha_defect"])
        self.evidence_version = int(state["evidence_version"])
        self.evidence_revision = int(state["evidence_revision"])
        self.normal_scores = tuple(float(x) for x in state["normal_scores"])
        self.defect_scores = tuple(float(x) for x in state["defect_scores"])
        self.conflict = bool(state["conflict"])
        self.last_status = str(state["status"])
        self.last_prediction_error = state.get("last_prediction_error")


def _normal_fpr_lower_bound(scores: tuple[float, ...], target: float) -> float:
    """Smallest threshold whose observed normal FP count meets target.

    The rule is exact for ``score >= theta`` including ties. If k errors are
    permitted, theta must be strictly above the (n-k-1)th order statistic.
    """
    ordered = sorted(float(x) for x in scores)
    if not ordered or not 0.0 <= target <= 1.0:
        raise ValueError("Normal FPR bound requires scores and target in [0,1]")
    allowed = min(len(ordered), math.floor(target * len(ordered)))
    if allowed == len(ordered):
        return math.nextafter(ordered[0], -math.inf)
    boundary = ordered[len(ordered) - allowed - 1]
    return math.nextafter(boundary, math.inf)


def _defect_fnr_upper_bound(scores: tuple[float, ...], target: float) -> float:
    """Largest threshold whose observed defect FN count meets target.

    Since a tie at theta predicts defect, the (k)th ascending score is an
    inclusive upper boundary when k misses are permitted.
    """
    ordered = sorted(float(x) for x in scores)
    if not ordered or not 0.0 <= target <= 1.0:
        raise ValueError("Defect FNR bound requires scores and target in [0,1]")
    allowed = min(len(ordered) - 1, math.floor(target * len(ordered)))
    return ordered[allowed]


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


class FrozenThresholdFallback:
    """Explicit operational wrapper for a source method's unavailable/abstain.

    The source controller continues to receive exactly the same score,
    feedback, and evidence calls as its unwrapped counterpart. The wrapper
    changes only the emitted operational prediction and records its provenance
    in ``Prediction.reason`` and ``Prediction.details``.
    """

    def __init__(self, source: Controller, theta0: float, *, name: str | None = None):
        if not math.isfinite(float(theta0)):
            raise ValueError("Fallback threshold must be finite")
        self.source = source
        self.theta0 = float(theta0)
        self.name = name or f"{source.name}_FROZEN_FALLBACK"
        if self.name == source.name:
            raise ValueError("Operational fallback must be a separate method row")

    @property
    def theta(self) -> float:
        value = getattr(self.source, "theta", self.theta0)
        return self.theta0 if value is None else float(value)

    def predict(self, observation: Observation) -> Prediction:
        native = self.source.predict(observation)
        if not native.available:
            reason = "STARTUP_FALLBACK"
        elif native.label is None:
            reason = "ABSTENTION_FALLBACK"
        else:
            return Prediction(native.label, native.threshold, "ORIGINAL_METHOD",
                              {"source_method": self.source.name,
                               "source_reason": native.reason,
                               "source_details": dict(native.details)}, True)
        return Prediction(
            int(observation.score >= self.theta0), self.theta0, reason,
            {"source_method": self.source.name,
             "source_label": native.label,
             "source_available": native.available,
             "source_threshold": native.threshold,
             "source_reason": native.reason,
             "source_details": dict(native.details),
             "fallback_threshold_source": "theta0_frozen_C64"}, True)

    def observe_score(self, observation: Observation) -> None:
        self.source.observe_score(observation)

    def observe_feedback(self, observation: Observation, prediction: Prediction,
                         feedback: Feedback) -> None:
        # Reconstruct the pure source decision captured during wrapper.predict.
        # This avoids evaluating an expensive source predictor twice and never
        # feeds the operational fallback decision back as the source decision.
        details = prediction.details
        native = Prediction(
            details.get("source_label"), details.get("source_threshold"),
            str(details.get("source_reason", "")),
            dict(details.get("source_details", {})),
            bool(details.get("source_available", True)))
        self.source.observe_feedback(observation, native, feedback)

    def on_memory_change(self, evidence: EvidenceSnapshot) -> None:
        self.source.on_memory_change(evidence)

    def finalize_step(self) -> float:
        value = self.source.finalize_step()
        return self.theta0 if value is None else float(value)

    def state_dict(self) -> dict[str, Any]:
        return {"name": self.name, "source_name": self.source.name,
                "theta0": self.theta0, "source": self.source.state_dict(),
                "fallback_policy": "theta0_on_unavailable_or_abstention"}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if (state.get("name") != self.name or
                state.get("source_name") != self.source.name or
                state.get("theta0") != self.theta0 or
                state.get("fallback_policy") !=
                "theta0_on_unavailable_or_abstention"):
            raise ValueError("Operational fallback checkpoint identity mismatch")
        self.source.load_state_dict(state["source"])
