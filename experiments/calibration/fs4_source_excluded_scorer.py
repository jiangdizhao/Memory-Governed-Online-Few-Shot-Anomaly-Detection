"""Read-only full-patch source-excluded native Greedy and C1 scoring."""
from __future__ import annotations

import copy
from dataclasses import dataclass
import hashlib
from types import SimpleNamespace
from typing import Any, Sequence

import torch

from .fs4_greedy_provenance import GreedyProvenance, _native_modules
from .fs4_threshold_core import NormalEvidence, support_self_reference


@dataclass(frozen=True)
class FullFeatureRecord:
    evidence: NormalEvidence
    features: torch.Tensor
    height: int
    width: int
    normalization: str
    dtype: str
    feature_sha256: str


def _feature_digest(value: torch.Tensor) -> str:
    x = value.detach().to("cpu").contiguous()
    h = hashlib.sha256(f"{tuple(x.shape)}|{x.dtype}".encode())
    h.update(x.view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


class VerifiedNormalFeatureCache:
    def __init__(self, capacity: int = 1024) -> None:
        if capacity != 1024:
            raise ValueError("FS4 cache capacity is fixed at 1024 groups")
        self.capacity = capacity
        self.records: dict[str, FullFeatureRecord] = {}

    def add(self, evidence: NormalEvidence, features: torch.Tensor, *, height: int,
            width: int, normalization: str) -> FullFeatureRecord:
        evidence.validate()
        if evidence.receipt.label != 0 or not evidence.receipt.disclosed or not evidence.receipt.executed:
            raise ValueError("only legally queried verified normals may be cached")
        if evidence.group in self.records:
            raise ValueError("one calibration image per source group")
        if len(self.records) >= self.capacity:
            raise OverflowError("verified-normal feature cache is full; no replacement permitted")
        if (not isinstance(features, torch.Tensor) or tuple(features.shape) != (784, 4096) or
                features.dtype != torch.float32 or not torch.isfinite(features).all() or
                height <= 0 or width <= 0 or not normalization):
            raise ValueError("complete finite fused (784,4096) native features and geometry required")
        saved = features.detach().to("cpu").contiguous().clone()
        record = FullFeatureRecord(evidence, saved, int(height), int(width), normalization,
                                   str(saved.dtype), _feature_digest(saved))
        self.records[evidence.group] = record
        return record

    def bytes_report(self) -> dict[str, int]:
        actual = sum(r.features.numel() * r.features.element_size() for r in self.records.values())
        return {"count": len(self.records), "capacity_groups": self.capacity,
                "actual_tensor_bytes": actual, "allocated_tensor_bytes": actual,
                "capacity_if_fp32_bytes": self.capacity * 784 * 4096 * 4}


def _correction_digest(r1: Any, cb: Any, sources: dict[int, set[str]]) -> str:
    view = SimpleNamespace(corr_cb={"p0": {"id": [cb]}})
    return r1._correction_state_sha256(view, sources)


def initialize_native_support(*, impl: Any, bank: Any, cfg: Any,
                              support: Sequence[tuple[str, Any, torch.Tensor]],
                              correction_cb: Any) -> dict[str, Any]:
    """Score exactly four registered Supports under their frozen native bank."""
    from experiments.calibration import r1_matched_correction_pilot as r1
    from experiments.calibration.native import _fixed_bank_digest
    bridge, _ = _native_modules()
    if len(support) != 4 or len({x[0] for x in support}) != 4:
        raise ValueError("exactly four distinct Support source groups required")
    if any(cb.size() != 0 or int(cb.K) != 1024 for cb in bridge.dynamic_codebooks(bank)) or correction_cb.size() != 0 or int(correction_cb.K) != 1024:
        raise ValueError("cold start requires empty K=1024 Greedy and C1")
    fixed = _fixed_bank_digest(bank)
    scores = [r1._native_dense_score(impl, bank, image, [features], cfg)[2]
              for _, image, features in support]
    if _fixed_bank_digest(bank) != fixed:
        raise ValueError("native Support self-scoring changed the fixed bank")
    result = support_self_reference(scores)
    return {**result, "support_scores": scores, "fixed_support_sha256": fixed,
            "theta_g": result["theta"], "theta_gc": result["theta"],
            "initial_c1_empty": True, "initial_gc_equals_g": True,
            "writer_reference": {"theta": result["theta"], "sigma": result["writer_sigma"]}}


class NativeSourceExcludedScorer:
    def __init__(self, *, impl: Any, bank: Any, cfg: Any,
                 correction_cfg: Any, correction_cb: Any,
                 correction_sources: dict[int, set[str]], provenance: GreedyProvenance,
                 normalization: str) -> None:
        if int(correction_cb.K) != 1024 or float(correction_cfg.corr_weight) != 1.0 or not normalization:
            raise ValueError("native C1 K=1024 and feature normalization identity required")
        self.impl, self.bank, self.cfg = impl, bank, cfg
        self.correction_cfg, self.correction_cb = correction_cfg, correction_cb
        self.correction_sources, self.provenance = correction_sources, provenance
        self.normalization = normalization

    @torch.no_grad()
    def score(self, record: FullFeatureRecord, *, event_position: int) -> dict[str, Any]:
        from experiments.calibration import r1_matched_correction_pilot as r1
        from experiments.calibration.native import _fixed_bank_digest
        bridge, _ = _native_modules()
        record.evidence.validate()
        self.provenance.verify(self.bank)
        if (record.normalization != self.normalization or record.dtype != str(record.features.dtype) or
                _feature_digest(record.features) != record.feature_sha256 or
                event_position <= record.evidence.position):
            raise ValueError("CURRENT_STATE_RESCORING_UNAVAILABLE: feature convention mismatch")
        before_fixed = _fixed_bank_digest(self.bank)
        before_g = bridge.snapshot(self.bank, version=0).sha256()
        before_provenance = self.provenance.digest()
        before_c = _correction_digest(r1, self.correction_cb, self.correction_sources)
        eligible = self.provenance.eligible_indices(self.bank, record.evidence.group)
        try:
            clone = copy.deepcopy(self.bank)
            for cb, indices in zip(bridge.dynamic_codebooks(clone), eligible):
                slots = torch.as_tensor(indices, dtype=torch.long, device=cb.protos.device)
                cb.protos = cb.protos.index_select(0, slots).clone()
            clone.corr_cfg.enabled = False
            image = SimpleNamespace(height=record.height, width=record.width)
            features = [record.features.to(clone.device)]
            base, ids, g_score, summary = r1._native_dense_score(self.impl, clone, image, features, self.cfg)
            selected = self.impl.l2_normalize(features[0].to(clone.device))[
                torch.as_tensor(ids, device=clone.device, dtype=torch.long)]
            clone.corr_cfg = copy.deepcopy(self.correction_cfg)
            clone.corr_cfg.enabled = True
            clone.corr_cb["p0"]["id"][0] = copy.deepcopy(self.correction_cb)
            delta, diagnostics = r1._c1_score_delta(self.impl, clone, selected,
                copy.deepcopy(self.correction_sources), record.evidence.group,
                event_position=int(event_position))
            corrected = r1.corrected_dense_map(base, ids, delta)
            return {"G": float(g_score), "GC": r1.max_score(corrected),
                    "top32_ids": ids.tolist(), "excluded_greedy_slots": sum(len(x) for x in self.provenance.rows) - sum(len(x) for x in eligible),
                    "correction": diagnostics, "summary": summary}
        finally:
            if (_fixed_bank_digest(self.bank) != before_fixed or
                    bridge.snapshot(self.bank, version=0).sha256() != before_g or
                    self.provenance.digest() != before_provenance or
                    _correction_digest(r1, self.correction_cb, self.correction_sources) != before_c):
                raise RuntimeError("source-excluded scoring mutated a live native state")
