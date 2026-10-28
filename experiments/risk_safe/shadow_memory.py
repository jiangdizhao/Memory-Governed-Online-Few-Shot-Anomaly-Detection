"""D1-C: isolated Greedy counterfactual normal memory (CPU-first).

The append/replacement operator calls the exact V0.2 greedy_slot, not a
surrogate novelty test. This is a tensor-level adapter for synthetic audits;
the production VisionAD/v22 image scorer must be checked separately.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import sys
from pathlib import Path
from typing import Callable, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[2]
for location in (ROOT, ROOT / "experiments"):
    if str(location) not in sys.path:
        sys.path.insert(0, str(location))

import torch
import torch.nn.functional as F
from v02_memory import greedy_slot as native_greedy_slot


class ShadowError(RuntimeError):
    pass


def _copies(layers: Sequence[torch.Tensor]) -> Tuple[torch.Tensor, ...]:
    return tuple(t.detach().clone().contiguous() for t in layers)


def _validate(layers: Sequence[torch.Tensor], caps: Sequence[int]) -> None:
    if not layers or len(layers) != len(caps):
        raise ShadowError("Layer count/capacity mismatch")
    for tensor, cap in zip(layers, caps):
        if tensor.ndim != 2 or tensor.shape[1] == 0 or cap < 1:
            raise ShadowError("Expected nonempty feature dimension and positive capacity")
        if len(tensor) > cap or not tensor.is_floating_point():
            raise ShadowError("Invalid bank dtype or capacity")
        if not torch.isfinite(tensor).all().item():
            raise ShadowError("Nonfinite memory feature")


class _Codebook:
    """Minimal Codebook facade accepted by unchanged V0.2 greedy_slot."""
    def __init__(self, protos: torch.Tensor, cap: int):
        self.protos = protos
        self.K = cap

    def size(self) -> int:
        return len(self.protos)


@dataclass(frozen=True)
class GreedyAudit:
    appended: int
    replaced: int
    skipped: int
    candidate_patches: int
    old_version: int
    new_version: int


class GreedyBank:
    """Dynamic normal patch bank; static support is maintained by the scorer.

    Copies are independent. Only commit() may modify the live bank. An
    otherwise rejected shadow must not change tensor bytes or bank version.
    """
    def __init__(self, layers: Sequence[torch.Tensor], capacities: Sequence[int],
                 *, version: int = 0):
        self.layers = _copies(layers)
        self.capacities = tuple(int(x) for x in capacities)
        self.version = int(version)
        _validate(self.layers, self.capacities)
        if self.version < 0:
            raise ShadowError("Negative bank version")

    def clone(self) -> "GreedyBank":
        return GreedyBank(self.layers, self.capacities, version=self.version)

    def sha256(self) -> str:
        h = hashlib.sha256()
        h.update(str((self.capacities, self.version)).encode("ascii"))
        for t in self.layers:
            x = t.detach().cpu().contiguous()
            h.update(str((tuple(x.shape), str(x.dtype))).encode("ascii"))
            h.update(x.numpy().tobytes())
        return h.hexdigest()

    def content_sha256(self) -> str:
        """Hash prototype content without the event/version counter."""
        h = hashlib.sha256()
        h.update(str(self.capacities).encode("ascii"))
        for tensor in self.layers:
            value = tensor.detach().cpu().contiguous()
            h.update(str((tuple(value.shape), str(value.dtype))).encode("ascii"))
            h.update(value.numpy().tobytes())
        return h.hexdigest()

    def bytes(self) -> int:
        return sum(t.numel() * t.element_size() for t in self.layers)

    @torch.no_grad()
    def proposed(self, candidate_layers: Sequence[torch.Tensor],
                 *, already_normalized: bool = False
                 ) -> tuple["GreedyBank", GreedyAudit]:
        """Stage the exact native Greedy append/swap, preserving dtype.

        Default candidates are raw patches. The native v22 bridge instead
        supplies rows indexed *after normalizing the FULL original feature
        matrix with impl.l2_normalize*, matching MemoryArm.update exactly.
        Those rows MUST NOT be normalized a second time: half precision,
        shape-dependent reduction and (norm + eps) are not bitwise invariant.
        """
        if len(candidate_layers) != len(self.layers):
            raise ShadowError("Candidate layer count mismatch")
        shadow = self.clone()
        added = replaced = skipped = examined = 0
        next_layers = []
        for source, proposed, cap in zip(shadow.layers, candidate_layers, self.capacities):
            if proposed.ndim != 2 or proposed.shape[1] != source.shape[1]:
                raise ShadowError("Candidate dimension mismatch")
            if proposed.device != source.device:
                raise ShadowError("Candidate and memory device mismatch")
            if not torch.isfinite(proposed).all().item():
                raise ShadowError("Nonfinite candidate")
            cb = _Codebook(source, cap)
            # Native v22 uses x/(x.norm(dim=-1, keepdim=True)+1e-12)
            # in the INPUT dtype; F.normalize(proposed.float()) is NOT
            # bitwise equivalent (the observed error was 7.629e-6).
            # When the bridge gives already-normalized full-matrix rows,
            # neither promote dtype nor renormalize.
            q = (proposed if already_normalized else
                 proposed / (proposed.norm(dim=-1, keepdim=True) + 1e-12))
            for row in q:
                examined += 1
                val = row.reshape(1, -1).to(cb.protos.device, dtype=cb.protos.dtype)
                if cb.size() < cb.K:
                    cb.protos = torch.cat((cb.protos, val), dim=0)
                    added += 1
                else:
                    slot = native_greedy_slot(cb, val)
                    if slot is None:
                        skipped += 1
                    else:
                        # Clone was constructed before candidate application.
                        cb.protos[slot] = val.reshape(-1)
                        replaced += 1
            next_layers.append(cb.protos.contiguous())
        shadow.layers = tuple(next_layers)
        _validate(shadow.layers, self.capacities)
        content_changed = any(not torch.equal(old, new)
                              for old, new in zip(self.layers, next_layers))
        if content_changed:
            shadow.version = self.version + 1
        return shadow, GreedyAudit(
            added, replaced, skipped, examined, self.version, shadow.version)

    def commit(self, shadow: "GreedyBank", *, expected_version: int,
               expected_sha256: str) -> None:
        if self.version != expected_version or self.sha256() != expected_sha256:
            raise ShadowError("Stale shadow update: live bank changed")
        if (shadow.capacities != self.capacities or
                shadow.version not in (self.version, self.version + 1)):
            raise ShadowError("Invalid version or capacity-incompatible shadow")
        _validate(shadow.layers, self.capacities)
        for old, new in zip(self.layers, shadow.layers):
            if old.shape[1] != new.shape[1] or old.device != new.device:
                raise ShadowError("Shadow feature/device mismatch")
        content_changed = any(not torch.equal(old, new)
                              for old, new in zip(self.layers, shadow.layers))
        if content_changed != (shadow.version == self.version + 1):
            raise ShadowError("Memory version must advance exactly on content change")
        if not content_changed:
            return
        self.layers = _copies(shadow.layers)
        self.version = shadow.version


class CosinePatchScorer:
    """Synthetic exact nearest-cosine prototype scorer, not v22 image scorer.

    Only for CPU mechanism tests. Production use must inject the native v22
    score_bank-based callback under frozen static support/scorer settings.
    """
    def __init__(self, static_layers: Sequence[torch.Tensor]):
        self.static_layers = _copies(static_layers)

    @torch.no_grad()
    def __call__(self, anchor_layers: Sequence[torch.Tensor],
                 bank: GreedyBank) -> float:
        if len(anchor_layers) != len(bank.layers) or len(anchor_layers) != len(self.static_layers):
            raise ShadowError("Scoring layer mismatch")
        scores = []
        for z, fixed, dyn in zip(anchor_layers, self.static_layers, bank.layers):
            if z.ndim != 2 or z.shape[1] != fixed.shape[1] or z.shape[0] == 0:
                raise ShadowError("Scored feature shape mismatch")
            protos = torch.cat((fixed, dyn), dim=0)
            if not len(protos):
                raise ShadowError("No normal prototype for scoring")
            zn = F.normalize(z.float(), dim=1)
            pn = F.normalize(protos.float(), dim=1)
            dist = 1 - (zn @ pn.T).clamp(-1, 1)
            scores.append(float(dist.min(dim=1).values.max().item()))
        return float(sum(scores) / len(scores))
