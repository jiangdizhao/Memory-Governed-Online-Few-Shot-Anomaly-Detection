"""Query-only, version-checked local gain measurement primitives.

This module has no evaluator-label or future-cohort API. It contains no
controller updates and never writes to Native Greedy memory.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from typing import Any, Iterable, Mapping, Sequence

import torch

from experiments.risk_safe.shadow_memory import GreedyBank
from .memory_state import CalibrationGreedyState


PRIMARY_OFFSETS = (-0.01, 0.0, 0.01)
SECONDARY_OFFSETS = (-0.005, 0.0, 0.005)
DEFECT_COUNT_BINS = ((0, 0, "0"), (1, 2, "1-2"), (3, 7, "3-7"),
                     (8, 13, "8-13"), (14, None, "14+"))


class IdentifiabilityError(ValueError):
    """A query score or evidence record violates the frozen measurement contract."""


@dataclass(frozen=True)
class AnchorScore:
    """One verified query label paired to an exact frozen snapshot score."""

    group_sha256: str
    position: int
    camera_id: int
    label: int
    score: float
    memory_version: int
    snapshot_memory_sha256: str
    scoring_memory_sha256: str
    policy: str
    excluded_rows_by_layer: tuple[int, ...] = ()
    arrival_score: float | None = None
    arrival_memory_version: int | None = None
    arrival_source_was_resident: bool | None = None

    def validate(self, *, expected_version: int,
                 expected_snapshot_sha256: str) -> None:
        if self.label not in (0, 1):
            raise IdentifiabilityError("anchor label must be binary")
        if not self.group_sha256 or len(self.group_sha256) != 64:
            raise IdentifiabilityError("anchor group identity must be SHA-256")
        if self.memory_version != expected_version:
            raise IdentifiabilityError("anchor score Memory Version differs from snapshot")
        if self.snapshot_memory_sha256 != expected_snapshot_sha256:
            raise IdentifiabilityError("anchor score is bound to another snapshot digest")
        if not self.scoring_memory_sha256:
            raise IdentifiabilityError("anchor lacks the actual scoring-memory digest")
        if not math.isfinite(float(self.score)):
            raise IdentifiabilityError("anchor score is nonfinite")
        if self.position < 0 or self.camera_id not in (1, 2, 3, 4, 5):
            raise IdentifiabilityError("anchor position or camera provenance is invalid")
        if self.arrival_score is not None and not math.isfinite(float(self.arrival_score)):
            raise IdentifiabilityError("arrival score is nonfinite")


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False)


def digest_json(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def distinct_group_count(group_ids: Iterable[str]) -> int:
    """Count official groups, never camera views or rescoring rows."""
    return len(set(group_ids))


def leave_source_out(memory: CalibrationGreedyState,
                    source_rows: Sequence[Sequence[str]], source_key: str
                    ) -> tuple[CalibrationGreedyState, tuple[int, ...]]:
    """Remove every source-attributed row in every dynamic layer.

    The returned state preserves the frozen memory version and legacy version;
    only a private scoring view is created. The supplied state is not mutated.
    """
    if len(memory.layers) != len(source_rows):
        raise IdentifiabilityError("source-map layer count differs from memory")
    filtered: list[torch.Tensor] = []
    removed: list[int] = []
    for layer, rows in zip(memory.layers, source_rows):
        if len(layer) != len(rows):
            raise IdentifiabilityError("source map is not aligned with prototype rows")
        keep = torch.tensor([str(key) != source_key for key in rows],
                            dtype=torch.bool, device=layer.device)
        removed.append(int((~keep).sum().item()))
        filtered.append(layer[keep].detach().clone().contiguous())
    legacy = GreedyBank(filtered, memory.capacities,
                        version=memory.legacy_greedy_version)
    result = CalibrationGreedyState(legacy, memory.version)
    if result.version != memory.version:
        raise IdentifiabilityError("LSO view changed logical snapshot version")
    return result, tuple(removed)


def _validate_unique(anchors: Sequence[AnchorScore], label: int, *,
                     expected_version: int, expected_snapshot_sha256: str) -> None:
    seen: set[str] = set()
    for anchor in anchors:
        anchor.validate(expected_version=expected_version,
                        expected_snapshot_sha256=expected_snapshot_sha256)
        if anchor.label != label:
            raise IdentifiabilityError("anchor is in the wrong class collection")
        if anchor.group_sha256 in seen:
            raise IdentifiabilityError("official sample group is counted more than once")
        seen.add(anchor.group_sha256)


def local_gain_table(normals: Sequence[AnchorScore],
                     defects: Sequence[AnchorScore], *, theta: float,
                     offsets: Sequence[float], expected_version: int,
                     expected_snapshot_sha256: str,
                     invalid_required_scores: int = 0) -> dict[str, Any]:
    """Estimate paired empirical Youden gain using only verified query anchors."""
    if not math.isfinite(float(theta)):
        raise IdentifiabilityError("B3 threshold is nonfinite")
    if invalid_required_scores < 0:
        raise IdentifiabilityError("invalid score count cannot be negative")
    _validate_unique(normals, 0, expected_version=expected_version,
                     expected_snapshot_sha256=expected_snapshot_sha256)
    _validate_unique(defects, 1, expected_version=expected_version,
                     expected_snapshot_sha256=expected_snapshot_sha256)
    overlap = {a.group_sha256 for a in normals}.intersection(
        a.group_sha256 for a in defects)
    if overlap:
        raise IdentifiabilityError("one official group has contradictory class labels")
    n_n, n_d = len(normals), len(defects)
    if invalid_required_scores:
        return {"status": "SCORE_INCOMPLETE", "reason": "INVALID_REQUIRED_SCORE",
                "normal_groups": n_n, "defect_groups": n_d,
                "invalid_required_scores": invalid_required_scores,
                "gains": None}
    if n_n == 0 or n_d == 0:
        return {"status": "TWO_CLASS_GAIN_UNIDENTIFIABLE",
                "reason": "MISSING_NORMAL_OR_DEFECT_EVIDENCE",
                "normal_groups": n_n, "defect_groups": n_d,
                "invalid_required_scores": invalid_required_scores,
                "gains": None}
    normal_scores = [float(row.score) for row in normals]
    defect_scores = [float(row.score) for row in defects]

    def objective(threshold: float) -> dict[str, float]:
        fpr = sum(score >= threshold for score in normal_scores) / n_n
        fnr = sum(score < threshold for score in defect_scores) / n_d
        return {"fpr": float(fpr), "fnr": float(fnr),
                "j": float(1.0 - fpr - fnr)}

    base = objective(float(theta))
    table: dict[str, Any] = {}
    for offset in offsets:
        h = float(offset)
        if not math.isfinite(h):
            raise IdentifiabilityError("candidate offset is nonfinite")
        value = objective(float(theta) + h)
        delta_j = value["j"] - base["j"]
        table[f"{h:+.6f}"] = {
            "offset": h, "threshold": float(theta) + h,
            "fpr": value["fpr"], "fnr": value["fnr"], "j": value["j"],
            "delta_j_vs_keep": float(delta_j),
            "delta_ba_vs_keep": float(delta_j / 2.0),
            "delta_fpr": float(value["fpr"] - base["fpr"]),
            "delta_fnr": float(value["fnr"] - base["fnr"]),
        }
    return {"status": "IDENTIFIABLE_TWO_CLASS_EMPIRICAL",
            "reason": "COMPLETE_VERSION_MATCHED_TWO_CLASS_EVIDENCE",
            "normal_groups": n_n, "defect_groups": n_d,
            "invalid_required_scores": 0, "base": base, "gains": table}


def select_primary_action(table: Mapping[str, Any]) -> dict[str, str]:
    """Apply the frozen exact-tie LOWER/KEEP/RAISE/ABSTAIN rule."""
    if table.get("status") != "IDENTIFIABLE_TWO_CLASS_EMPIRICAL":
        return {"action": "ABSTAIN", "reason": str(table.get("reason", "UNIDENTIFIABLE"))}
    gains = table.get("gains")
    if not isinstance(gains, Mapping) or "-0.010000" not in gains or "+0.010000" not in gains:
        return {"action": "ABSTAIN", "reason": "PRIMARY_GAIN_MISSING"}
    lower = float(gains["-0.010000"]["delta_j_vs_keep"])
    raise_gain = float(gains["+0.010000"]["delta_j_vs_keep"])
    if not (math.isfinite(lower) and math.isfinite(raise_gain)):
        return {"action": "ABSTAIN", "reason": "INVALID_PRIMARY_GAIN"}
    if lower > 0.0 and raise_gain > 0.0 and lower == raise_gain:
        return {"action": "ABSTAIN", "reason": "UNRESOLVED_POSITIVE_GAIN_TIE"}
    if lower > 0.0 and lower > raise_gain:
        return {"action": "LOWER", "reason": "LOWER_STRICTLY_BETTER_POSITIVE_GAIN"}
    if raise_gain > 0.0 and raise_gain > lower:
        return {"action": "RAISE", "reason": "RAISE_STRICTLY_BETTER_POSITIVE_GAIN"}
    if lower <= 0.0 and raise_gain <= 0.0:
        return {"action": "KEEP", "reason": "BOTH_NONPOSITIVE"}
    return {"action": "ABSTAIN", "reason": "UNRESOLVED_ACTION_COMPARISON"}


def defect_evidence_bin(count: int) -> str:
    value = int(count)
    if value < 0:
        raise ValueError("defect evidence count cannot be negative")
    for lower, upper, label in DEFECT_COUNT_BINS:
        if value >= lower and (upper is None or value <= upper):
            return label
    raise AssertionError("defect evidence bin table is incomplete")


def action_digest(rows: Sequence[Mapping[str, Any]]) -> str:
    """Hash only frozen query-derived actions/estimates, never future labels."""
    ordered = sorted((dict(row) for row in rows),
                     key=lambda row: (row["category"], int(row["seed"]),
                                      int(row["snapshot_position"])))
    return digest_json(ordered)
