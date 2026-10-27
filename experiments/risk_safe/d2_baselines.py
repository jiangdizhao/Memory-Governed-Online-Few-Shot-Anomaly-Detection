"""D2 controlled, verified-label memory admission baselines.

Every method sees exactly the same preselected candidate image/patch set and
the same Q-A human label. Scoring/feature extraction are delegated to native
v22. All state is per-arm; no evaluation probes are used for decisions.

'causal_novelty' is a HITL-style ADAPTATION, not the authors' original pool
calibration/oracle review selection. 'reservoir' uses dynamic-patch uniform
reservoir replacement. 'append' rejects an entire event if it would overflow
the shared capacity, never expands memory silently.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable, Dict, Optional, Sequence, Tuple
import math
import random
import torch
import torch.nn.functional as F

from .shadow_memory import GreedyBank
from .native_bridge import NativeParityError

BASELINES = ("static", "append", "greedy", "reservoir", "causal_novelty")


@dataclass
class Outcome:
    method: str
    decision: str
    reason: str
    bank: GreedyBank
    added: int = 0
    replaced: int = 0
    skipped: int = 0
    novelty: Optional[float] = None
    novelty_cutoff: Optional[float] = None
    normal_utility: Optional[float] = None
    defect_risk: Optional[float] = None
    extra: dict = field(default_factory=dict)


@dataclass
class BaselineState:
    name: str
    seed: int
    seen: list[int] = field(default_factory=list)
    prior_novelties: list[float] = field(default_factory=list)
    attempts: int = 0
    admissions: int = 0
    patches_added: int = 0
    patches_replaced: int = 0
    patches_skipped: int = 0
    _rng: random.Random = field(init=False, repr=False)

    def __post_init__(self):
        if self.name not in BASELINES:
            raise ValueError(f"Unsupported D2 baseline: {self.name}")
        self._rng = random.Random(self.seed)

    def record(self, outcome: Outcome) -> None:
        self.attempts += 1
        if outcome.decision == "admit":
            self.admissions += 1
        self.patches_added += outcome.added
        self.patches_replaced += outcome.replaced
        self.patches_skipped += outcome.skipped


@torch.no_grad()
def candidate_novelty(
        raw_candidate_layers: Sequence[torch.Tensor],
        native_bank: object, *,
        chunk: int = 1024) -> float:
    """Cosine novelty of candidate features to *prior* fixed + dynamic bank.

    Uses no future class labels, no independently evaluated probe, and no
    original HITL oracle review pool. Mismatch with original HITL algorithm is
    deliberate and documented. Chunked to avoid 4096D full NxK allocation.
    """
    fixed = native_bank.normal_fixed["p0"][0]["id"]
    dyn = native_bank.normal_dyn["p0"][0]["id"]
    if len(raw_candidate_layers) != len(fixed) or len(fixed) != len(dyn):
        raise NativeParityError("Novelty layer mismatch")
    accum = []
    for candidates, fixed_cb, dyn_cb in zip(raw_candidate_layers, fixed, dyn):
        if not len(candidates):
            raise NativeParityError("Empty novelty candidate set")
        sources = [fixed_cb.protos, dyn_cb.protos]
        sources = [x for x in sources if len(x)]
        if not sources:
            raise NativeParityError("Novelty requires nonempty static reference")
        c = F.normalize(candidates.float(), dim=-1)
        best = torch.full((len(c),), -1., dtype=torch.float32,
                          device=c.device)
        for bank in sources:
            for offset in range(0, len(bank), chunk):
                p = F.normalize(bank[offset:offset + chunk].float(), dim=-1)
                best = torch.maximum(best, (c @ p.T).max(dim=1).values)
        accum.append(float((1 - best.clamp(-1, 1)).mean().item()))
    return sum(accum) / len(accum)


def _counts(out: Outcome, *, added=0, replaced=0, skipped=0) -> Outcome:
    out.added, out.replaced, out.skipped = added, replaced, skipped
    return out


@torch.no_grad()
def propose_baseline(state: BaselineState, bank: GreedyBank,
                     normalized_candidate_layers: Sequence[torch.Tensor],
                     *, raw_candidate_layers: Sequence[torch.Tensor] = (),
                     native_bank: Optional[object] = None) -> Outcome:
    """Compute an event-level proposal WITHOUT modifying current bank.

    normalized_candidate_layers must be selected from the original full
    native feature matrix AFTER impl.l2_normalize on the full matrix.
    """
    name = state.name
    if name == "static":
        return Outcome(name, "reject", "static_no_dynamic_write", bank.clone())
    if len(normalized_candidate_layers) != len(bank.layers):
        raise NativeParityError("Candidate layer mismatch")
    if not all(x.ndim == 2 and len(x) > 0 and
               x.shape[1] == old.shape[1]
               for x, old in zip(normalized_candidate_layers, bank.layers)):
        raise NativeParityError("Malformed candidate layers")
    novelty = cutoff = None
    if name == "causal_novelty":
        if native_bank is None or not raw_candidate_layers:
            raise NativeParityError("Causal novelty requires preupdate native reference")
        novelty = candidate_novelty(raw_candidate_layers, native_bank)
        previous = sorted(state.prior_novelties)
        if previous:
            n = len(previous)
            cutoff = ((previous[(n-1)//2] + previous[n//2]) / 2)
        # Strictly historical feedback: add current observed novelty only
        # AFTER deciding, irrespective of admit/reject. Cold first normal
        # candidate is accepted (not a proof of defect safety).
        state.prior_novelties.append(novelty)
        if cutoff is not None and novelty <= cutoff:
            return Outcome(name, "reject", "causal_median_novelty_gate",
                           bank.clone(), novelty=novelty,
                           novelty_cutoff=cutoff)
    if name == "append":
        # Event is indivisible; reject rather than partially appending.
        if any(len(old) + len(c) > cap for old, c, cap in zip(
                bank.layers, normalized_candidate_layers, bank.capacities)):
            return Outcome(name, "reject", "append_capacity_exceeded",
                           bank.clone())
        out = bank.clone()
        out.layers = tuple(torch.cat((
            old, c.detach().to(device=old.device, dtype=old.dtype)), 0)
            for old, c in zip(out.layers, normalized_candidate_layers))
        out.version += 1
        return _counts(Outcome(name, "admit", "append",
                               out), added=sum(len(c) for c in normalized_candidate_layers))
    if name == "reservoir":
        out = bank.clone()
        next_layers = []
        appended = replaced = skipped = 0
        if not state.seen:
            state.seen = [len(x) for x in bank.layers]
        if len(state.seen) != len(bank.layers):
            raise NativeParityError("Reservoir state/layer count mismatch")
        for i, (old, cand, cap) in enumerate(zip(
                out.layers, normalized_candidate_layers, out.capacities)):
            store = old
            for row in cand:
                state.seen[i] += 1
                value = row.reshape(1, -1).to(device=old.device, dtype=old.dtype)
                if len(store) < cap:
                    store = torch.cat((store, value), 0)
                    appended += 1
                else:
                    j = state._rng.randrange(state.seen[i])
                    if j < cap:
                        store[j] = value[0]
                        replaced += 1
                    else:
                        skipped += 1
            next_layers.append(store)
        out.layers = tuple(next_layers)
        if appended or replaced:
            out.version += 1
        return _counts(Outcome(name, "admit" if (appended or replaced) else "reject",
                               "reservoir_sampling" if (appended or replaced)
                               else "reservoir_no_change", out),
                       added=appended, replaced=replaced, skipped=skipped)
    # Both greedy and accepted novelty use exact unchanged V0.2 Greedy logic.
    out, audit = bank.proposed(normalized_candidate_layers,
                               already_normalized=True)
    changed = audit.appended + audit.replaced
    return _counts(Outcome(name, "admit" if changed else "reject",
                           "greedy_native_slot" if changed else "greedy_no_change",
                           out, novelty=novelty, novelty_cutoff=cutoff),
                   added=audit.appended, replaced=audit.replaced,
                   skipped=audit.skipped)
