"""V0.2 bounded normal memory arms, retaining native v22 hierarchy operators.

The online Greedy control uses the same redundancy-swap criterion as A0/A1,
with chunked on-demand pair-gap evaluation rather than a materialized KxK
matrix. Same current-image candidates and label timing as the hierarchy.
"""
from __future__ import annotations
import time
import torch
import torch.nn.functional as F
import tmra_compact_memory as tmra


codebooks = tmra.codebooks
allocated_state_bytes = tmra.allocated_state_bytes


@torch.no_grad()
def greedy_slot(cb, candidate):
    """A0 greedy min-redundancy swap without O(K^2) transient allocation.

    Exact for the current set (up to roundoff); chunking controls peak VRAM.
    This is an online heuristic, NOT an official ContCore reproduction.
    """
    p = F.normalize(cb.protos.float(), dim=1)
    q = F.normalize(candidate.float().view(1, -1), dim=1)
    nearest = float((1 - (q @ p.T).clamp(-1,1)).min().item())
    n = len(p)
    if n == 0: return 0
    if n == 1: return 0 if nearest > 1e-7 else None
    best_gap = float("inf")
    slot = -1
    for begin in range(0, n, 128):
        block = p[begin:begin+128]
        dist = 1 - (block @ p.T).clamp(-1,1)
        index = torch.arange(len(block), device=p.device)
        dist[index, begin+index] = float("inf")
        smallest, _ = dist.min(dim=1)
        val, pos = smallest.min(dim=0)
        if float(val.item()) < best_gap:
            best_gap, slot = float(val.item()), begin + int(pos.item())
    return slot if nearest > best_gap + 1e-7 else None


class MemoryArm:
    def __init__(self, name, bank, settings, seed):
        if name not in ("greedy_coreset", "hierarchy"):
            raise ValueError(name)
        self.name, self.bank, self.settings = name, bank, settings
        self.events = self.patches = self.promotions = 0
        self.accepted = self.skipped = 0
        self.update_seconds = 0.0
        b, x = bank, settings
        hierarchy = name == "hierarchy"
        b.use_defect_bank = False
        b.corr_cfg.enabled = False
        b.enable_maturation_buffer = hierarchy
        b.enable_ltm_promotion = hierarchy
        b.enable_ltm_retrieval = hierarchy
        b.enable_maturation_buffer_retrieval = hierarchy
        b.enable_utility_aware_stm_retention = hierarchy
        b.enable_stm_retirement = False
        b.shadow_ltm = False
        b.K_normal_dyn = x.stm if hierarchy else x.flat
        b.K_normal_ltm = x.ltm if hierarchy else 1
        b.ltm_candidate_capacity = x.buffer if hierarchy else 1
        b.mature_max_candidates = b.ltm_candidate_capacity
        b.mature_min_events = x.min_events
        b.ltm_min_repeat = x.min_events
        b.mature_promote_every = x.promote_every
        b.mature_min_usefulness = .15
        b.mature_score_threshold = .45
        b.mature_max_event_protos = 8
        b.mature_event_merge_eps = .03
        b.mature_retrieval_min_events = 2
        b.mature_retrieval_min_usefulness = .1
        b.mature_retrieval_score_thr = .25
        b.mature_retrieval_penalty = .01
        b.mature_retrieval_k = 1
        b.ltm_retrieval_k = 1
        b.stm_retention_replace_margin = .03
        stm, ltm, buf = codebooks(b)
        if any(c.size() for group in (stm, ltm, buf) for c in group):
            raise AssertionError("Mutable bank must be empty at start")
        for c in stm: c.K = b.K_normal_dyn
        for c in ltm: c.K = b.K_normal_ltm
        for i, old in enumerate(buf):
            buf[i] = type(old)(
                b.ltm_candidate_capacity, old.feat_dim, old.device,
                merge_eps=.06, max_event_prototypes=8, event_merge_eps=.03)
        self._check_budget()

    def _check_budget(self):
        bank = self.bank
        stm, ltm, buf = codebooks(bank)
        if sum(c.size() for c in stm) > bank.K_normal_dyn:
            raise AssertionError("STM cap exceeded")
        if sum(c.size() for c in ltm) > bank.K_normal_ltm:
            raise AssertionError("LTM cap exceeded")
        if sum(c.size() for c in buf) > bank.ltm_candidate_capacity:
            raise AssertionError("Buffer cap exceeded")
        amount = allocated_state_bytes(bank)
        if amount > self.settings.cap_bytes:
            raise AssertionError(f"Dynamic persistent tensor state {amount} > {self.settings.cap_bytes}")
        return amount

    @torch.no_grad()
    def update(self, impl, features, candidates, step, *, score, theta, sigma):
        """Called only AFTER a queried verified NORMAL or charged history event.

        Derive v22 usefulness from SAME static pre-query score for both arms.
        'novelty=0' explicitly isolates coverage-memory feedback from this
        matched memory test. Use variable boundary/FP signals, not neutral .5.
        """
        if torch.cuda.is_available() and features[0].is_cuda:
            torch.cuda.synchronize()
        started = time.perf_counter()
        allow, keep, reason, info = impl.select_informative_normal_write(
            score=float(score), theta=float(theta),
            sigmaN=float(max(sigma, .02)), pred_defect=bool(score >= theta),
            novelty=0.0)
        if not allow: raise AssertionError("Verified-normal write refused")
        selected = {k:list(v)[:keep] for k,v in candidates.items()}
        self.events += 1
        self.patches += sum(len(x) for x in selected.values())
        if self.name == "hierarchy":
            report = self.bank.update_from_topk(
                cls_name="p0", feats_per_layer=features, which="normal_dyn",
                mode=0, view="id", layer_to_patch_idx=selected,
                tau_insert=.15, tau_replace=.25, event_id=step,
                write_value=float(info["write_value"]),
                outcome=("FP" if score >= theta else "TN"),
                boundary_value=float(info["boundary"]), coverage_gain=0.0)
            if report.get("consolidation",{}).get("consolidated"):
                self.promotions += 1
            for stat in report.get("per_layer",{}).values():
                if isinstance(stat,dict):
                    self.accepted += int(stat.get("n_added",0)) + int(stat.get("n_replaced",0))
        else:
            for layer, cb in enumerate(codebooks(self.bank)[0]):
                idx = selected.get(layer,[])
                if not idx: continue
                q = impl.l2_normalize(features[layer].to(cb.device))
                if min(idx)<0 or max(idx)>=len(q):
                    raise AssertionError("Invalid common candidate index")
                for j in idx:
                    val = q[j].reshape(1,-1).to(cb.device, dtype=cb.dtype)
                    if cb.size() < cb.K:
                        cb.protos = torch.cat((cb.protos, val), dim=0)
                        self.accepted += 1
                    else:
                        slot = greedy_slot(cb, val)
                        if slot is None: self.skipped += 1
                        else:
                            cb.protos[slot] = val.reshape(-1)
                            self.accepted += 1
        if torch.cuda.is_available() and features[0].is_cuda:
            torch.cuda.synchronize()
        self.update_seconds += time.perf_counter() - started
        self._check_budget()

    def audit(self):
        stm, ltm, buf = codebooks(self.bank)
        return {
            "stm":sum(c.size() for c in stm),
            "ltm":sum(c.size() for c in ltm),
            "buffer":sum(c.size() for c in buf),
            "allocated_state_bytes":self._check_budget(),
            "byte_ceiling":self.settings.cap_bytes,
            "feedback_events":self.events,
            "candidate_patches":self.patches,
            "accepted_patches":self.accepted,
            "skipped_patches":self.skipped,
            "promotions":self.promotions,
            "update_seconds":self.update_seconds,
        }
