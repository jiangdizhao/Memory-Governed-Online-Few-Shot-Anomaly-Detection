"""Calibration-owned version view over the unchanged historical GreedyBank.

The historical ``GreedyBank.version`` retains its original operation-count
semantics. Calibration instead versions score state only when the stored
prototype bytes change. Source attribution is versioned separately by the
shared writer.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Sequence

import torch

from experiments.risk_safe.shadow_memory import GreedyBank, GreedyAudit


class CalibrationMemoryError(RuntimeError):
    pass


def prototype_content_sha256(layers: Sequence[torch.Tensor],
                             capacities: Sequence[int]) -> str:
    """Hash stored prototype bytes, shapes, dtypes and capacity, excluding versions."""
    h = hashlib.sha256()
    h.update(str(tuple(int(value) for value in capacities)).encode("ascii"))
    for tensor in layers:
        value = tensor.detach().cpu().contiguous()
        h.update(str((tuple(value.shape), str(value.dtype))).encode("ascii"))
        h.update(value.numpy().tobytes())
    return h.hexdigest()


@dataclass(frozen=True)
class CalibrationGreedyState:
    """Duck-typed native-bank view with independent calibration versioning.

    ``version`` is the calibration memory-content version expected by the
    scorer, AnchorLedger and prequential engine. ``legacy_greedy_version``
    retains the version stored by the historical GreedyBank implementation.
    """

    legacy_bank: GreedyBank
    memory_content_version: int

    def __post_init__(self) -> None:
        if self.memory_content_version < 0:
            raise CalibrationMemoryError("Negative calibration memory-content version")

    @property
    def version(self) -> int:
        """Version consumed by calibration score/evidence contracts."""
        return self.memory_content_version

    @property
    def legacy_greedy_version(self) -> int:
        return self.legacy_bank.version

    @property
    def layers(self) -> tuple[torch.Tensor, ...]:
        return self.legacy_bank.layers

    @property
    def capacities(self) -> tuple[int, ...]:
        return self.legacy_bank.capacities

    def content_sha256(self) -> str:
        return prototype_content_sha256(self.layers, self.capacities)

    def sha256(self) -> str:
        """Digest compatible with a GreedyBank snapshot at calibration version."""
        h = hashlib.sha256()
        h.update(str((self.capacities, self.memory_content_version)).encode("ascii"))
        for tensor in self.layers:
            value = tensor.detach().cpu().contiguous()
            h.update(str((tuple(value.shape), str(value.dtype))).encode("ascii"))
            h.update(value.numpy().tobytes())
        return h.hexdigest()

    def transition(self, legacy_proposal: GreedyBank
                   ) -> tuple["CalibrationGreedyState", bool]:
        """Apply one historical proposal and advance calibration version on bytes only."""
        if legacy_proposal.capacities != self.capacities:
            raise CalibrationMemoryError("Historical proposal changed memory capacities")
        if legacy_proposal.version not in (self.legacy_greedy_version,
                                           self.legacy_greedy_version + 1):
            raise CalibrationMemoryError("Historical GreedyBank version did not advance by at most one")
        changed = (prototype_content_sha256(legacy_proposal.layers,
                                             legacy_proposal.capacities) !=
                   self.content_sha256())
        if changed and legacy_proposal.version != self.legacy_greedy_version + 1:
            raise CalibrationMemoryError(
                "Prototype content changed without historical GreedyBank advancement")
        return (CalibrationGreedyState(
            legacy_proposal,
            self.memory_content_version + int(changed)), changed)

    def proposed(self, candidate_layers: Sequence[torch.Tensor], *,
                 already_normalized: bool = False
                 ) -> tuple["CalibrationGreedyState", GreedyAudit, bool]:
        legacy_proposal, audit = self.legacy_bank.proposed(
            candidate_layers, already_normalized=already_normalized)
        state, changed = self.transition(legacy_proposal)
        return state, audit, changed

    def clone(self) -> "CalibrationGreedyState":
        return CalibrationGreedyState(self.legacy_bank.clone(),
                                      self.memory_content_version)
