"""A1-H memory-only intervention adapters. Reuse actual v22 maturation and LTM code.

No feature extractor, query policy, threshold controller, correction or defect
memory modifications. Explicitly isolate retention, promotion and retrieval.
"""
from __future__ import annotations

from dataclasses import dataclass
from types import MethodType
import math

import a0_a1_memory_methods as a1

FLATS = ("static", "reservoir", "online_greedy_coreset", "v22_normal_stm")
HIERARCHIES = ("H1_track", "H2_shadow_promotion", "H3_ltm_retrieval", "H4_utility")
ARMS = FLATS + HIERARCHIES


@dataclass(frozen=True)
class Budget:
    total: int

    def __post_init__(self):
        if self.total not in (128, 256):
            raise ValueError("Pre-registered total prototype budgets are 128 or 256")

    @property
    def stm(self):
        return self.total // 2

    @property
    def ltm(self):
        return self.total // 4

    @property
    def buffer(self):
        return self.total // 4


def install_unified_retrieval(bank, *, include_ltm: bool, k: int = 5):
    """Shared global top-k over Fixed + STM [+ LTM]; no hierarchy-specific k.

    PatchVMB.score_single_view calls this method. The base fixed bank, image
    aggregation, and top-k hyperparameter remain identical across arms.
    """
    def dN(self, cls_name, mode, view, layer, q):
        import torch
        cbf = self.normal_fixed[cls_name][mode][view][layer]
        cbs = self.normal_dyn[cls_name][mode][view][layer]
        cbl = self.normal_dyn_ltm[cls_name][mode][view][layer]
        chunks = [cbf.protos.float()]
        if cbs.size():
            chunks.append(cbs.protos.float())
        if include_ltm and cbl.size():
            chunks.append(cbl.protos.float())
        p = torch.nn.functional.normalize(torch.cat(chunks, dim=0), dim=1)
        qn = torch.nn.functional.normalize(q.float(), dim=1)
        sim = torch.clamp(qn @ p.T, -1.0, 1.0)
        top, idx = sim.topk(k=min(k, p.shape[0]), dim=1, largest=True)
        return 1.0 - top.mean(dim=1), idx[:, 0]
    bank._dN_min_fixed_dyn = MethodType(dN, bank)


def _buffers(bank):
    return bank.normal_dyn_candidates["p0"][0]["id"]


def _codebooks(bank):
    return (
        bank.normal_dyn["p0"][0]["id"],
        bank.normal_dyn_ltm["p0"][0]["id"],
        _buffers(bank),
    )


class Hierarchy:
    def __init__(self, name, bank, *, budget: Budget):
        if name not in HIERARCHIES:
            raise ValueError(name)
        self.name, self.bank, self.budget = name, bank, budget
        self.candidate_events = 0
        self.candidate_patches = 0
        self.promotion_triggers = 0
        self.promotion_attempts = 0
        self.promotion_successes = 0
        self.promoted_prototypes = 0
        self.ltm_retrieval_effect_events = 0
        self.retired = 0
        self.update_seconds = 0.
        self._configure()

    def _configure(self):
        b = self.bank
        budget = self.budget
        stm, ltm, buffers = _codebooks(b)
        b.K_normal_dyn = budget.stm
        b.K_normal_ltm = budget.ltm
        b.ltm_candidate_capacity = budget.buffer
        b.mature_max_candidates = budget.buffer
        b.enable_maturation_buffer = True
        b.enable_ltm_promotion = self.name != "H1_track"
        b.shadow_ltm = False  # shadow promotion implemented as physical but nonretrieved LTM
        b.enable_ltm_retrieval = self.name in ("H3_ltm_retrieval", "H4_utility")
        b.enable_utility_aware_stm_retention = self.name == "H4_utility"
        b.enable_maturation_buffer_retrieval = False
        b.enable_stm_retirement = False
        b.use_defect_bank = False
        b.corr_cfg.enabled = False
        b.mature_promote_every = 16  # historical v22, not performance tuned
        b.ltm_min_pending_updates = 64
        b.mature_min_events = 3
        b.ltm_min_repeat = 3
        b.mature_score_threshold = .45
        b.mature_min_usefulness = .10
        b.mature_promote_max_per_layer = max(4, budget.ltm // 8)
        b.ltm_retrieval_k = 1

        for cs, cl, old in zip(stm, ltm, buffers):
            if cs.size() or cl.size() or old.size():
                raise AssertionError("Hierarchy must begin with empty dynamic states")
            cs.K = budget.stm
            cl.K = budget.ltm
        for layer, old in enumerate(buffers):
            # Critical: v22 builds tracker tensors with K=4096 at construction;
            # resetting the object is necessary to enforce real allocated budget.
            buffers[layer] = type(old)(
                budget.buffer, old.feat_dim, old.device,
                merge_eps=b.ltm_candidate_merge_eps,
                max_event_prototypes=b.mature_max_event_protos,
                event_merge_eps=b.mature_event_merge_eps)
        install_unified_retrieval(b, include_ltm=b.enable_ltm_retrieval)

    def observe_verified_normal(self, impl, features, patch_indices, *,
                                step, score, theta, sigma):
        import time
        import torch
        if torch.cuda.is_available() and features[0].is_cuda:
            torch.cuda.synchronize()
        start = time.perf_counter()
        self.candidate_events += 1
        _allow, _topk, _reason, info = impl.select_informative_normal_write(
            score=score, theta=theta, sigmaN=sigma,
            pred_defect=score >= theta, novelty=0.0)
        for layer, indices in patch_indices.items():
            if not indices:
                continue
            self.candidate_patches += len(indices)
        before = [cb.size() for cb in self.bank.normal_dyn_ltm["p0"][0]["id"]]
        report = self.bank.update_from_topk(
            cls_name="p0", feats_per_layer=features,
            which="normal_dyn", mode=0, view="id",
            layer_to_patch_idx=patch_indices, tau_insert=.15,
            tau_replace=.25, event_id=step, write_value=info["write_value"],
            outcome="FP" if score >= theta else "TN",
            boundary_value=info["boundary"], coverage_gain=0.0)
        trigger = report.get("ltm_trigger_debug", {})
        if trigger.get("mature_trigger") or trigger.get("pressure_trigger"):
            self.promotion_triggers += 1
            if self.bank.enable_ltm_promotion:
                self.promotion_attempts += 1
        if report.get("consolidation", {}).get("consolidated"):
            self.promotion_successes += 1
        after = [cb.size() for cb in self.bank.normal_dyn_ltm["p0"][0]["id"]]
        self.promoted_prototypes += sum(max(0, y-x) for x,y in zip(before,after))
        if torch.cuda.is_available() and features[0].is_cuda:
            torch.cuda.synchronize()
        self.update_seconds += time.perf_counter()-start
        return report

    def state(self):
        stm, ltm, buffer = _codebooks(self.bank)
        ns = sum(c.size() for c in stm)
        nl = sum(c.size() for c in ltm)
        nb = sum(c.size() for c in buffer)
        occupied = ns+nl+nb
        cap = self.budget.total * len(stm)
        if occupied > cap:
            raise AssertionError(f"Hierarchy {self.name} exceeded budget {occupied}>{cap}")
        feats = sum(c.protos[:c.size()].numel() * c.protos.element_size()
                    for group in (stm, ltm, buffer) for c in group)
        allocated = sum(c.protos.numel() * c.protos.element_size()
                        for group in (stm, ltm, buffer) for c in group)
        # The tracker preallocates additional metadata vectors; account for
        # them rather than falsely claiming only feature storage.
        tracker_meta = sum(
            getattr(c, fld).numel() * getattr(c, fld).element_size()
            for c in buffer
            for fld in ("count","distinct_events","usefulness_sum",
                        "boundary_sum","fp_rescue_sum","coverage_sum",
                        "last_event","first_step","last_step","promoted_count"))
        return {
            "method":self.name,"stm":ns,"ltm":nl,"buffer":nb,
            "occupied_total":occupied,"capacity_total":cap,
            "feature_bytes_occupied":feats,
            "feature_bytes_allocated":allocated,
            "tracker_metadata_bytes_allocated":tracker_meta,
            "candidate_events":self.candidate_events,
            "candidate_patches":self.candidate_patches,
            "promotion_triggers":self.promotion_triggers,
            "promotion_attempts":self.promotion_attempts,
            "promotion_successes":self.promotion_successes,
            "promoted_new_prototypes":self.promoted_prototypes,
            "enable_ltm_retrieval":bool(self.bank.enable_ltm_retrieval),
            "enable_utility":bool(self.bank.enable_utility_aware_stm_retention),
            "update_seconds":self.update_seconds,
        }


class Flat:
    """Existing A1 verified-normal retention primitives, with common scorer."""
    def __init__(self, name, bank, *, budget: Budget, seed: int):
        self.name, self.bank, self.budget = name, bank, budget
        self.update_seconds = 0.0
        self.inner = a1.MemoryArm(name, bank, dynamic_capacity=budget.total, seed=seed)
        # Flat reference must not accidentally allocate a 4096-track buffer:
        # zero-capacity conceptual sidecars use 1 allocated dummy slot.
        for layer, cb in enumerate(_buffers(bank)):
            _buffers(bank)[layer] = type(cb)(
                1, cb.feat_dim, cb.device,
                merge_eps=bank.ltm_candidate_merge_eps)
        install_unified_retrieval(bank, include_ltm=False)

    def observe_verified_normal(self, impl, features, patch_indices, *,
                                step, score, theta, sigma):
        import time
        import torch
        if torch.cuda.is_available() and features[0].is_cuda:
            torch.cuda.synchronize()
        started = time.perf_counter()
        out = self.inner.observe_verified_normal(
            impl, features, patch_indices, step=step, score=score, theta=theta,
            sigma=sigma)
        if torch.cuda.is_available() and features[0].is_cuda:
            torch.cuda.synchronize()
        self.update_seconds += time.perf_counter()-started
        return out

    def state(self):
        d = self.inner.audit()
        cbs = self.bank.normal_dyn["p0"][0]["id"]
        occupied_bytes = sum(cb.protos.numel()*cb.protos.element_size()
                             for cb in cbs)
        return {
            "method":self.name, "stm":d["mutable_prototypes"],
            "ltm":0,"buffer":0,
            "occupied_total":d["mutable_prototypes"],
            "capacity_total":self.budget.total * len(self.inner.sizes()),
            "feature_bytes_occupied":occupied_bytes,
            "feature_bytes_allocated":occupied_bytes,
            "tracker_metadata_bytes_allocated":0,
            "update_seconds":self.update_seconds,
            "candidate_events":d["candidate_events"],
            "candidate_patches":d["candidate_patches"],
            "accepted_patches":d["accepted_patches"],
            "promotion_attempts":0, "promotion_successes":0,
            "promoted_new_prototypes":0,
            "promotion_triggers":0,
            "enable_ltm_retrieval":False, "enable_utility":False,
        }


def make_arm(name, bank, budget, seed):
    if name in FLATS:
        return Flat(name, bank, budget=budget, seed=seed)
    return Hierarchy(name, bank, budget=budget)


def counterfactual_ltm_scores(score_fn, arm, img, feat, cfg):
    """Same memory state: toggle LTM retrieval without changing any update."""
    if arm.name not in ("H3_ltm_retrieval", "H4_utility"):
        return None
    b = arm.bank
    install_unified_retrieval(b, include_ltm=True)
    with_ltm = score_fn(b, img, feat, cfg)
    install_unified_retrieval(b, include_ltm=False)
    without = score_fn(b, img, feat, cfg)
    install_unified_retrieval(b, include_ltm=True)
    return {"with_ltm":with_ltm, "without_ltm":without,
            "delta":with_ltm-without}
