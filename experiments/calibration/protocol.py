"""Causal event types and capability-limited controller inputs."""
from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, Mapping


class ProtocolError(RuntimeError):
    pass


@dataclass(frozen=True)
class TapeRecord:
    """Trusted, label-blind metadata used by tape construction only."""

    position: int
    official_sample_key: str
    camera_id: int
    relative_id: str
    timestamp: str = ""
    phase: int = 0

    def __post_init__(self) -> None:
        if self.position < 0 or not self.official_sample_key:
            raise ProtocolError("Invalid stream position or sample grouping key")
        if self.camera_id not in (1, 2, 3, 4, 5):
            raise ProtocolError("Camera id must be one of Real-IAD cameras 1..5")
        # Keys may encode OK/NG. They are permitted only in the trusted tape,
        # source-attribution and private evaluator layers. Observation below
        # intentionally has no key/path field.


@dataclass(frozen=True)
class Observation:
    """Only score/state metadata visible to calibration controllers.

    It intentionally has no official sample key, path, image, feature tensor,
    mask, or label. Observation ids are opaque and do not encode class state.
    """

    t: int
    observation_id: str
    camera_id: int
    score: float
    memory_version: int
    score_state_digest: str

    def __post_init__(self) -> None:
        if self.t < 0 or not self.observation_id:
            raise ProtocolError("Invalid public observation identity")
        if not math.isfinite(float(self.score)):
            raise ProtocolError("Nonfinite score cannot enter a controller")
        if self.memory_version < 0 or not self.score_state_digest:
            raise ProtocolError("Score state is not provenance-addressed")


@dataclass(frozen=True)
class Feedback:
    """Label disclosure capability; labels are present only after a query."""

    t: int
    queried: bool
    label: int | None = None

    def __post_init__(self) -> None:
        if self.queried and self.label not in (0, 1):
            raise ProtocolError("Queried feedback must contain a verified binary label")
        if not self.queried and self.label is not None:
            raise ProtocolError("Unqueried feedback must not carry a label")


@dataclass(frozen=True)
class Prediction:
    """A binary decision or source-faithful abstention."""

    label: int | None
    threshold: float | None
    reason: str = ""
    details: Mapping[str, Any] = field(default_factory=dict)
    available: bool = True

    def __post_init__(self) -> None:
        if self.label not in (0, 1, None):
            raise ProtocolError("Prediction label must be 0, 1, or abstain (None)")
        if self.threshold is not None and not math.isfinite(float(self.threshold)):
            raise ProtocolError("Prediction threshold must be finite or unavailable")
        if not self.available and self.label is not None:
            raise ProtocolError("An unavailable prediction cannot carry a binary label")


@dataclass(frozen=True)
class ScoreResult:
    score: float
    memory_version: int
    score_state_digest: str
    score_map: Any = None

    def __post_init__(self) -> None:
        if not math.isfinite(float(self.score)):
            raise ProtocolError("Native score is nonfinite")
        if self.memory_version < 0 or not self.score_state_digest:
            raise ProtocolError("Score result lacks a valid memory-state identity")


@dataclass(frozen=True)
class StreamEvent:
    """Trusted engine input; labels remain in a separate evaluator object."""

    record: TapeRecord
    payload: Any
    anchor_features: Any = None


@dataclass(frozen=True)
class EvidenceAnchor:
    opaque_anchor_id: str
    score: float
    memory_version: int
    protected: bool
    source_excluded_rows: int = 0
    source_exclusion_policy: str = "protected_no_selfwrite"

    def __post_init__(self) -> None:
        if (not self.opaque_anchor_id or not math.isfinite(float(self.score)) or
                self.memory_version < 0 or self.source_excluded_rows < 0):
            raise ProtocolError("Invalid or stale evidence anchor")


@dataclass(frozen=True)
class EvidenceSnapshot:
    memory_version: int
    normal_anchors: tuple[EvidenceAnchor, ...]
    defect_anchors: tuple[EvidenceAnchor, ...]
    source_exclusion_audit: Mapping[str, Any]
    evidence_revision: int = 0

    def __post_init__(self) -> None:
        if self.memory_version < 0 or self.evidence_revision < 0:
            raise ProtocolError("Invalid evidence snapshot version")
        if any(anchor.memory_version != self.memory_version
               for anchor in (*self.normal_anchors, *self.defect_anchors)):
            raise ProtocolError("Evidence snapshot contains a stale anchor score")

    @property
    def available_normal(self) -> bool:
        return bool(self.normal_anchors)

    @property
    def available_defect(self) -> bool:
        return bool(self.defect_anchors)

    @property
    def state(self) -> str:
        if self.available_normal and self.available_defect:
            return "BOTH"
        if self.available_normal:
            return "NORMAL_ONLY"
        if self.available_defect:
            return "DEFECT_ONLY"
        return "NONE"
