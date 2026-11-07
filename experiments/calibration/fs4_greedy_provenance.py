"""Mirror the unmodified V0.2 Greedy writer and retain conservative origins."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import re
import sys
from typing import Any, Mapping, Sequence

import torch

from .fs4_threshold_core import NormalEvidence
from .fs4_stochastic_bea import StochasticBEAGroupSlot
from .fs4_errors import FS4ProvenanceIntegrityError


def _native_modules() -> tuple[Any, Any]:
    experiments = str(Path(__file__).resolve().parents[1])
    if experiments not in sys.path:
        sys.path.insert(0, experiments)
    from experiments.risk_safe import native_bridge
    from v02_memory import greedy_slot
    return native_bridge, greedy_slot


def _feature_id(tensor: torch.Tensor, group: str, position: int, layer: int, patch: int) -> str:
    row = tensor.detach().to("cpu").contiguous()
    h = hashlib.sha256(f"{group}|{position}|{layer}|{patch}|{row.dtype}|{tuple(row.shape)}".encode())
    h.update(row.numpy().tobytes())
    return h.hexdigest()


class GreedyProvenance:
    """One sidecar per live bank; a failed parity check permanently blocks use."""

    def __init__(self, bank: Any) -> None:
        bridge, _ = _native_modules()
        cbs = bridge.dynamic_codebooks(bank)
        if any(cb.size() != 0 or int(cb.K) != 1024 for cb in cbs):
            raise ValueError("FS4 requires empty native Greedy K=1024 codebooks")
        self.rows: list[list[dict[str, Any]]] = [[] for _ in cbs]
        self.memory_digest = bridge.snapshot(bank, version=0).sha256()
        self.valid = True
        self.revision = 0

    def verify(self, bank: Any) -> None:
        bridge, _ = _native_modules()
        cbs = bridge.dynamic_codebooks(bank)
        if (not self.valid or len(cbs) != len(self.rows) or
                any(cb.size() != len(rows) or int(cb.K) != 1024 for cb, rows in zip(cbs, self.rows)) or
                bridge.snapshot(bank, version=0).sha256() != self.memory_digest or
                any(not row.get("groups") or not row.get("history") for layer in self.rows for row in layer)):
            self.valid = False
            raise FS4ProvenanceIntegrityError("SOURCE_PROVENANCE_INCOMPLETE: Greedy sidecar does not match live bank")
        for layer_index, (cb, rows) in enumerate(zip(cbs, self.rows)):
            active = cb.protos.detach().to("cpu").contiguous()
            for slot, row in enumerate(rows):
                history = row["history"]
                if (any(not re.fullmatch(r"[0-9a-f]{64}", item.get("group", "")) or
                        not re.fullmatch(r"[0-9a-f]{64}", item.get("feature_id", "")) or
                        not isinstance(item.get("position"), int) or item["position"] < 0 or
                        not isinstance(item.get("patch"), int) or item["patch"] < 0
                        for item in history) or
                        row["groups"] != sorted({item["group"] for item in history})):
                    self.valid = False
                    raise FS4ProvenanceIntegrityError("SOURCE_PROVENANCE_INCOMPLETE: malformed source history")
                last = history[-1]
                if _feature_id(active[slot].reshape(1, -1), last["group"],
                               last["position"], layer_index, last["patch"]) != last["feature_id"]:
                    self.valid = False
                    raise FS4ProvenanceIntegrityError("SOURCE_PROVENANCE_INCOMPLETE: active prototype identity differs")

    def eligible_indices(self, bank: Any, group: str) -> list[list[int]]:
        self.verify(bank)
        return [[i for i, row in enumerate(layer) if group not in row["groups"]] for layer in self.rows]

    def digest(self) -> str:
        payload = json.dumps({"memory": self.memory_digest, "rows": self.rows,
                              "revision": self.revision, "valid": self.valid},
                             sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode()).hexdigest()

    @torch.no_grad()
    def write_normal(self, *, writer: Any, impl: Any, evidence: NormalEvidence,
                     policy: StochasticBEAGroupSlot,
                     features: Sequence[torch.Tensor], candidates: Mapping[int, Sequence[int]],
                     score: float, theta_ref: float, sigma_ref: float) -> dict[str, Any]:
        bank = writer.bank
        self.verify(bank)
        evidence.validate()
        if (policy.receipts.get(evidence.position) != evidence.receipt or
                evidence.receipt.label != 0 or not evidence.receipt.query_seal or
                not evidence.receipt.disclosed or not evidence.receipt.executed or
                float(score) != evidence.pre_scores["G"]):
            raise ValueError("only legally queried verified normals may enter Greedy")
        bridge, greedy_slot = _native_modules()
        cbs = bridge.dynamic_codebooks(bank)
        if len(features) != len(cbs):
            raise ValueError("native feature/codebook layer mismatch")
        allow, keep, reason, _ = impl.select_informative_normal_write(
            score=float(score), theta=float(theta_ref), sigmaN=max(float(sigma_ref), 0.02),
            pred_defect=bool(score >= theta_ref), novelty=0.0)
        if not allow or keep not in (8, 16, 24):
            raise ValueError("native verified-normal selection did not return 8/16/24")
        mirrors = [copy.deepcopy(cb) for cb in cbs]
        proposed = copy.deepcopy(self.rows)
        accepted = skipped = 0
        for layer, (mirror, feat) in enumerate(zip(mirrors, features)):
            indices = list(candidates.get(layer, ()))[:keep]
            if indices and (min(indices) < 0 or max(indices) >= len(feat)):
                raise ValueError("native candidate index outside feature tensor")
            normalized = impl.l2_normalize(feat.to(mirror.device))
            for index in indices:
                val = normalized[index].reshape(1, -1).to(mirror.device, dtype=mirror.dtype)
                record = {"group": evidence.group, "position": evidence.position,
                          "feature_id": _feature_id(val, evidence.group, evidence.position, layer, int(index)),
                          "patch": int(index)}
                if mirror.size() < mirror.K:
                    mirror.protos = torch.cat((mirror.protos, val), dim=0)
                    proposed[layer].append({"groups": [evidence.group], "history": [record]})
                    accepted += 1
                else:
                    slot = greedy_slot(mirror, val)
                    if slot is None:
                        skipped += 1
                        continue
                    mirror.protos[slot] = val.reshape(-1)
                    prior = proposed[layer][slot]
                    prior["groups"] = sorted(set(prior["groups"]) | {evidence.group})
                    prior["history"].append(record)
                    accepted += 1
        before = self.memory_digest
        try:
            writer.update(impl, list(features), {int(k): list(v) for k, v in candidates.items()},
                          evidence.position, score=float(score), theta=float(theta_ref), sigma=float(sigma_ref))
            actual = bridge.dynamic_codebooks(bank)
            if any(not torch.equal(x.protos, y.protos) for x, y in zip(actual, mirrors)):
                raise FS4ProvenanceIntegrityError("native writer and disposable Greedy mirror differ")
            self.rows = proposed
            self.memory_digest = bridge.snapshot(bank, version=0).sha256()
            self.revision += int(accepted > 0)
            self.verify(bank)
        except Exception:
            self.valid = False
            raise
        return {"before_sha256": before, "after_sha256": self.memory_digest,
                "changed": before != self.memory_digest, "accepted": accepted,
                "skipped": skipped, "keep": keep, "native_reason": reason,
                "provenance_revision": self.revision}
