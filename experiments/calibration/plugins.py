"""Future-compatible interfaces; Stage 0 activates only calibration isolation."""
from __future__ import annotations

from typing import Any, Mapping, Protocol, runtime_checkable

from .protocol import Feedback, Observation, ScoreResult


@runtime_checkable
class FeatureProvider(Protocol):
    def extract(self, event: Any) -> Any: ...


@runtime_checkable
class ScoreService(Protocol):
    def score(self, event: Any, memory_snapshot: Any) -> ScoreResult: ...


@runtime_checkable
class MemoryPolicy(Protocol):
    def update_verified_normal(self, event: Any, score: ScoreResult,
                               feedback: Feedback) -> Any: ...


@runtime_checkable
class QueryPolicy(Protocol):
    def decide(self, observation: Observation,
               decisions_snapshot: Mapping[str, Any]) -> bool: ...


@runtime_checkable
class CorrectionPolicy(Protocol):
    def apply(self, score: ScoreResult, correction_snapshot: Any) -> ScoreResult: ...


class CorrectionOff:
    """Calibration-isolated Stage 0 correction policy: identity on raw scores."""
    name = "OFF"
    def apply(self, score: ScoreResult, correction_snapshot: Any = None) -> ScoreResult:
        if correction_snapshot not in (None, "OFF"):
            raise ValueError("Stage 0 profile fixes correction OFF")
        return score


class BEAPlugin(Protocol):
    """Interface reserved for a separately registered future query experiment."""
    name: str
    def decide(self, observation: Observation,
               decisions_snapshot: Mapping[str, Any]) -> bool: ...
