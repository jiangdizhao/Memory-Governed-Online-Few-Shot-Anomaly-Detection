"""Protected C64 and queried-feedback evidence with memory-versioned scores."""
from __future__ import annotations

from dataclasses import dataclass
import copy
from typing import Any, Callable, Sequence

import torch

from experiments.risk_safe.shadow_memory import GreedyBank
from .protocol import EvidenceAnchor, EvidenceSnapshot


class AnchorLedgerError(RuntimeError):
    pass


@dataclass
class _Anchor:
    anchor_id: str
    source_key: str
    camera_id: int
    label: int
    features: Any
    protected: bool
    arrival_version: int
    score: float | None = None
    score_version: int | None = None
    excluded_rows: int = 0


class AnchorLedger:
    """Features are private; controllers receive only current version scores."""

    def __init__(self, c64: Sequence[tuple[str, int, Any]], *,
                 deployed_support_keys: Sequence[str]):
        if len(c64) != 64:
            raise AnchorLedgerError("C64 requires exactly 64 additional normal anchors")
        if len(deployed_support_keys) != 4 or len(set(deployed_support_keys)) != 4:
            raise AnchorLedgerError("Exactly four distinct deployed support keys are required")
        ids = [item[0] for item in c64]
        if len(set(ids)) != 64:
            raise AnchorLedgerError("C64 anchor identities must be unique")
        overlap = set(ids).intersection(deployed_support_keys)
        if overlap:
            raise AnchorLedgerError("C64 must be disjoint from the four deployed supports")
        self._anchors = [
            _Anchor(f"c64:{index:03d}", key, int(camera), 0, features,
                    True, 0)
            for index, (key, camera, features) in enumerate(c64)
        ]
        self._source_sets = set(ids)
        self._deployed_support_keys = frozenset(deployed_support_keys)
        self._revision = 0
        self._last_refresh_version: int | None = None

    def add_queried(self, *, source_key: str, camera_id: int, label: int,
                    features: Any, t: int, memory_version: int) -> str:
        if label not in (0, 1):
            raise AnchorLedgerError("Only queried verified binary labels are eligible")
        if camera_id not in (1, 2, 3, 4, 5) or t < 0 or memory_version < 0:
            raise AnchorLedgerError("Queried anchor event provenance is invalid")
        if source_key in self._source_sets:
            raise AnchorLedgerError("Stream source overlaps protected C64")
        if source_key in self._deployed_support_keys:
            raise AnchorLedgerError("Queried stream source overlaps deployed support")
        if any(not a.protected and a.source_key == source_key for a in self._anchors):
            raise AnchorLedgerError("Official sample group has already contributed a queried anchor")
        anchor_id = f"query:{t}:{camera_id}"
        if any(a.anchor_id == anchor_id for a in self._anchors):
            raise AnchorLedgerError("Duplicate queried anchor event")
        self._anchors.append(_Anchor(anchor_id, source_key, camera_id, label,
                                     features, False, memory_version))
        self._revision += 1
        return anchor_id

    @property
    def protected_source_keys(self) -> frozenset[str]:
        return frozenset(self._source_sets)

    @staticmethod
    def _without_source(bank: GreedyBank, source_rows: Sequence[Sequence[str]],
                        source_key: str) -> tuple[GreedyBank, int]:
        if len(bank.layers) != len(source_rows):
            raise AnchorLedgerError("Source map layer count differs from memory")
        filtered, removed = [], 0
        for tensor, rows in zip(bank.layers, source_rows):
            if len(tensor) != len(rows):
                raise AnchorLedgerError("Source map is not aligned with prototype rows")
            keep = torch.tensor([key != source_key for key in rows],
                                dtype=torch.bool, device=tensor.device)
            removed += int((~keep).sum().item())
            filtered.append(tensor[keep].contiguous())
        return GreedyBank(filtered, bank.capacities, version=bank.version), removed

    def refresh(self, *, memory: GreedyBank,
                source_rows: Sequence[Sequence[str]],
                scorer: Callable[[Any, GreedyBank], Any]) -> EvidenceSnapshot:
        """Rescore every eligible anchor after actual memory content changes.

        Queried normal anchors are scored leave-source-out over every layer.
        If attribution is misaligned, this fails closed. C64 is protected and
        remains in eligibility even though C64 never enters mutable memory.
        """
        if self._last_refresh_version != memory.version:
            self._revision += 1
            self._last_refresh_version = memory.version
        normals, defects = [], []
        excluded_total = 0
        for anchor in self._anchors:
            score_bank = memory
            excluded = 0
            if not anchor.protected and anchor.label == 0:
                score_bank, excluded = self._without_source(
                    memory, source_rows, anchor.source_key)
            result = scorer(anchor.features, score_bank)
            if hasattr(result, "memory_version") and result.memory_version != memory.version:
                raise AnchorLedgerError("Scorer returned an anchor score from another memory version")
            score = float(result.score if hasattr(result, "score") else result)
            if not torch.isfinite(torch.tensor(score)).item():
                raise AnchorLedgerError("Nonfinite anchor rescore")
            anchor.score, anchor.score_version = score, memory.version
            anchor.excluded_rows = excluded
            excluded_total += excluded
            record = EvidenceAnchor(anchor.anchor_id, score, memory.version,
                                    anchor.protected, excluded,
                                    "C64_PROTECTED" if anchor.protected else
                                    "queried_normal_all_layer_leave_source_out"
                                    if anchor.label == 0 else
                                    "queried_defect_not_written")
            (normals if anchor.label == 0 else defects).append(record)
        return EvidenceSnapshot(
            memory.version, tuple(normals), tuple(defects),
            {"queried_normal_source_rows_excluded": excluded_total,
             "policy": "all-layer leave-source-out; C64 protected",
             "c64_mutable": False}, self._revision)

    def assert_current(self, snapshot: EvidenceSnapshot, version: int) -> None:
        if snapshot.memory_version != version:
            raise AnchorLedgerError("Evidence snapshot belongs to stale memory")
        if any(a.memory_version != version
               for a in (*snapshot.normal_anchors, *snapshot.defect_anchors)):
            raise AnchorLedgerError("Stale anchor score cannot enter calibration")

    def state_dict(self) -> dict[str, Any]:
        """Private checkpoint state; features are intentionally included.

        Callers must keep this object in a local checkpoint artifact. The
        public evidence snapshots continue to expose only opaque ids and scores.
        """
        return copy.deepcopy({"schema": "calibration.anchor_ledger.v1",
                "c64_count": 64, "queried_count": len(self._anchors) - 64,
                "deployed_support_keys": sorted(self._deployed_support_keys),
                "evidence_revision": self._revision,
                "last_refresh_version": self._last_refresh_version,
                "anchors": [{"id": a.anchor_id, "source_key": a.source_key,
                             "camera_id": a.camera_id, "label": a.label,
                             "protected": a.protected,
                             "arrival_version": a.arrival_version,
                             "score": a.score, "score_version": a.score_version,
                             "excluded_rows": a.excluded_rows,
                             "features": a.features}
                            for a in self._anchors]})

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if (state.get("schema") != "calibration.anchor_ledger.v1" or
                state.get("c64_count") != 64 or
                state.get("deployed_support_keys") !=
                sorted(self._deployed_support_keys)):
            raise AnchorLedgerError("Anchor checkpoint support/protocol identity mismatch")
        rows = state.get("anchors")
        if not isinstance(rows, list) or len(rows) != 64 + state.get("queried_count", -1):
            raise AnchorLedgerError("Anchor checkpoint count is inconsistent")
        base = rows[:64]
        expected_keys = [a.source_key for a in self._anchors[:64]]
        if any(not row.get("protected") or row.get("label") != 0
               for row in base) or [row.get("source_key") for row in base] != expected_keys:
            raise AnchorLedgerError("Protected C64 anchor identity/order mismatch")
        restored: list[_Anchor] = []
        seen_ids: set[str] = set()
        seen_sources = set(self._source_sets)
        for index, row in enumerate(rows):
            anchor_id = str(row.get("id", ""))
            source_key = str(row.get("source_key", ""))
            label = row.get("label")
            protected = bool(row.get("protected"))
            if (not anchor_id or anchor_id in seen_ids or label not in (0, 1) or
                    not source_key or (index < 64) != protected):
                raise AnchorLedgerError("Invalid or duplicate anchor checkpoint row")
            seen_ids.add(anchor_id)
            features = copy.deepcopy(row.get("features"))
            if index < 64:
                if anchor_id != self._anchors[index].anchor_id:
                    raise AnchorLedgerError("Protected C64 anchor id mismatch")
                if not _features_equal(features, self._anchors[index].features):
                    raise AnchorLedgerError("Protected C64 feature payload changed")
            else:
                if source_key in seen_sources or source_key in self._deployed_support_keys:
                    raise AnchorLedgerError("Queried anchor overlaps or duplicates a source")
                if not anchor_id.startswith("query:"):
                    raise AnchorLedgerError("Queried anchor id is not event-addressed")
                seen_sources.add(source_key)
            if features is None and (index >= 64 or
                    self._anchors[index].features is not None):
                raise AnchorLedgerError("Anchor checkpoint omits private feature payload")
            restored.append(_Anchor(
                anchor_id, source_key, int(row["camera_id"]), int(label), features,
                protected, int(row["arrival_version"]),
                None if row.get("score") is None else float(row["score"]),
                None if row.get("score_version") is None else int(row["score_version"]),
                int(row.get("excluded_rows", 0))))
        revision = int(state.get("evidence_revision", -1))
        refresh_version = state.get("last_refresh_version")
        if revision < 0 or (refresh_version is not None and int(refresh_version) < 0):
            raise AnchorLedgerError("Invalid anchor checkpoint revision")
        self._anchors = restored
        self._source_sets = set(self._source_sets)
        self._revision = revision
        self._last_refresh_version = (None if refresh_version is None
                                      else int(refresh_version))


def _features_equal(left: Any, right: Any) -> bool:
    if isinstance(left, torch.Tensor) or isinstance(right, torch.Tensor):
        return (isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor) and
                left.dtype == right.dtype and tuple(left.shape) == tuple(right.shape) and
                torch.equal(left.detach().cpu(), right.detach().cpu()))
    if (type(left).__module__.startswith("PIL.") or
            type(right).__module__.startswith("PIL.")):
        return (type(left).__module__.startswith("PIL.") and
                type(right).__module__.startswith("PIL.") and
                getattr(left, "mode", None) == getattr(right, "mode", None) and
                getattr(left, "size", None) == getattr(right, "size", None) and
                left.tobytes() == right.tobytes())
    if isinstance(left, dict) or isinstance(right, dict):
        return (isinstance(left, dict) and isinstance(right, dict) and
                left.keys() == right.keys() and
                all(_features_equal(left[key], right[key]) for key in left))
    if isinstance(left, (tuple, list)) or isinstance(right, (tuple, list)):
        return (type(left) is type(right) and len(left) == len(right) and
                all(_features_equal(a, b) for a, b in zip(left, right)))
    try:
        import numpy as np
        if isinstance(left, np.ndarray) or isinstance(right, np.ndarray):
            return (isinstance(left, np.ndarray) and isinstance(right, np.ndarray) and
                    left.dtype == right.dtype and np.array_equal(left, right))
    except ImportError:
        pass
    return left == right
