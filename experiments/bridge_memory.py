"""Historical-to-matched normal-memory audit: actual v22 operators, independent arms.

This does NOT reproduce the original manuscript's pool-assisted supports,
endogenous BEA queries, signed correction or threshold sidecar. It freezes
supervision so historical memory mechanisms can be attributed causally.
"""
from __future__ import annotations

from dataclasses import dataclass
import time

ARM_NAMES=(
    "flat_stm",
    "flat_utility",
    "flat_equal_total",
    "track_utility",
    "shadow_utility",
    "ltm_utility",
    "full_utility",
    "full_no_utility",
)
HIERARCHIES=set(ARM_NAMES[3:])

@dataclass(frozen=True)
class BridgeConfig:
    stm: int=512
    ltm: int=512
    buffer: int=4096
    min_events: int=2
    promote_every: int=32
    patches: int=24
    evidence_mode: str="neutral"

    def __post_init__(self):
        if self.stm not in (512,768):
            raise ValueError("Historical STM pressure sweep is 512/768")
        if self.ltm!=512 or self.buffer!=4096:
            raise ValueError("Bridge history profile requires LTM=512 and Buffer=4096")
        if self.min_events!=2:
            raise ValueError("Historical E_min=2, do not tune on evaluation labels")
        if self.promote_every!=32:
            raise ValueError("Original v22 CLI maturation period=32")
        if self.patches!=24:
            raise ValueError("Bridge fixed candidate budget is 24 per feedback")
        if self.evidence_mode not in ("neutral","loso_margin"):
            raise ValueError("Unsupported evidence mode")

    @property
    def total(self):
        return self.stm+self.ltm+self.buffer


def memory_books(bank):
    return (bank.normal_dyn["p0"][0]["id"],
            bank.normal_dyn_ltm["p0"][0]["id"],
            bank.normal_dyn_candidates["p0"][0]["id"])


class BridgeArm:
    def __init__(self,name,bank,config:BridgeConfig):
        if name not in ARM_NAMES:
            raise ValueError(name)
        self.name,self.bank,self.config=name,bank,config
        self.events=0
        self.patches_seen=0
        self.attempts=0
        self.successes=0
        self.promoted_total=0
        self.eligible_peak=0
        self.update_seconds=0.
        self._configure()

    def _configure(self):
        b=self.bank
        cfg=self.config
        flat=self.name not in HIERARCHIES
        active_promote=self.name not in (
            "flat_stm","flat_utility","flat_equal_total","track_utility")
        # IMPORTANT: deliberately use original v22 native scoring. No monkeypatch.
        b.enable_maturation_buffer=not flat
        b.enable_ltm_promotion=active_promote
        b.shadow_ltm=False
        b.enable_ltm_retrieval=self.name in (
            "ltm_utility","full_utility","full_no_utility")
        b.enable_maturation_buffer_retrieval=self.name in (
            "full_utility","full_no_utility")
        b.enable_utility_aware_stm_retention=self.name in (
            "flat_utility","flat_equal_total","track_utility",
            "shadow_utility","ltm_utility","full_utility")
        b.enable_stm_retirement=False
        b.use_defect_bank=False
        b.corr_cfg.enabled=False
        b.mature_min_events=cfg.min_events
        b.ltm_min_repeat=cfg.min_events
        b.mature_promote_every=cfg.promote_every
        b.mature_score_threshold=.45
        b.mature_min_usefulness=.10
        b.mature_rescue_min_fp=.15
        b.mature_event_merge_eps=.03
        b.ltm_candidate_merge_eps=.06
        b.mature_max_event_protos=8
        b.mature_promote_max_per_layer=64
        b.mature_retrieval_min_events=2
        b.mature_retrieval_min_usefulness=.10
        b.mature_retrieval_score_thr=.25
        b.mature_retrieval_penalty=.01
        b.mature_retrieval_max_protos=256
        b.mature_retrieval_k=1
        b.ltm_retrieval_k=1
        b.stm_retention_replace_margin=.03
        b.K_normal_dyn=cfg.total if self.name=="flat_equal_total" else cfg.stm
        b.K_normal_ltm=cfg.ltm if not flat else 1
        b.ltm_candidate_capacity=cfg.buffer if not flat else 1
        b.mature_max_candidates=b.ltm_candidate_capacity
        stm,ltm,buf=memory_books(b)
        if any(x.size()!=0 for group in (stm,ltm,buf) for x in group):
            raise AssertionError("Nonempty mutable memory at start")
        for cb in stm:cb.K=b.K_normal_dyn
        for cb in ltm:cb.K=b.K_normal_ltm
        # Underlying MaturationBuffer preallocates K*D float32 tensors;
        # reconstruct it to enforce declared capacity, never leave P0 defaults.
        for i,old in enumerate(buf):
            buf[i]=type(old)(
                b.ltm_candidate_capacity,old.feat_dim,old.device,
                merge_eps=.06,replace_eps=.12,max_event_prototypes=8,
                event_merge_eps=.03)

    def observe_verified_normal(self,impl,feat,candidates,*,step,score,theta,sigma):
        import torch
        self.events+=1
        self.patches_seen+=sum(len(x) for x in candidates.values())
        if self.config.evidence_mode=="neutral":
            weight,boundary,outcome=.5,.5,"TN"
        else:
            _a,_b,_c,info=impl.select_informative_normal_write(
                score=score,theta=theta,sigmaN=sigma,
                pred_defect=score>=theta,novelty=0.0)
            weight=float(info["write_value"])
            boundary=float(info["boundary"])
            outcome="FP" if score>=theta else "TN"
        if torch.cuda.is_available() and feat[0].is_cuda:torch.cuda.synchronize()
        start=time.perf_counter()
        before=[cb.size() for cb in memory_books(self.bank)[1]]
        report=self.bank.update_from_topk(
            cls_name="p0",feats_per_layer=feat,
            which="normal_dyn",mode=0,view="id",
            layer_to_patch_idx=candidates,
            tau_insert=.15,tau_replace=.25,event_id=step,
            write_value=weight,outcome=outcome,
            boundary_value=boundary,coverage_gain=0.0)
        if "consolidation" in report:
            self.attempts+=1
            detail=report["consolidation"]
            if detail.get("consolidated"):
                self.successes+=1
            for layer in detail.get("layers",{}).values():
                self.eligible_peak=max(self.eligible_peak,int(layer.get("eligible_count",0)))
        after=[cb.size() for cb in memory_books(self.bank)[1]]
        self.promoted_total+=sum(max(0,b-a) for a,b in zip(before,after))
        if torch.cuda.is_available() and feat[0].is_cuda:torch.cuda.synchronize()
        self.update_seconds+=time.perf_counter()-start
        return report

    def state(self):
        stm,ltm,buf=memory_books(self.bank)
        nstm=sum(x.size() for x in stm)
        nltm=sum(x.size() for x in ltm) if self.name in HIERARCHIES else 0
        nbuf=sum(x.size() for x in buf) if self.name in HIERARCHIES else 0
        maxcap=self.config.total if self.name in HIERARCHIES or self.name=="flat_equal_total" else self.config.stm
        if nstm+nltm+nbuf>maxcap:
            raise AssertionError(f"Capacity overflow for {self.name}")
        feature_alloc=sum(x.protos.numel()*x.protos.element_size()
                          for group in (stm,ltm,buf) for x in group)
        # Reflect real allocation, not just valid entries. Buffer stores fp32;
        # flat 1-slot disabled buffers count in physical bytes below.
        metadata=0
        for x in stm+ltm:
            for field in ("meta_count","meta_utility","meta_boundary",
                          "meta_fp_rescue","meta_coverage","meta_first_step",
                          "meta_last_step","meta_last_event","meta_distinct_events"):
                t=getattr(x,field)
                metadata+=t.numel()*t.element_size()
        for x in buf:
            for field in ("count","distinct_events","usefulness_sum","boundary_sum",
                          "fp_rescue_sum","coverage_sum","last_event",
                          "first_step","last_step","promoted_count"):
                t=getattr(x,field)
                metadata+=t.numel()*t.element_size()
        counts={"stm":nstm,"ltm":nltm,"buffer":nbuf,
                "occupied_total":nstm+nltm+nbuf,"capacity_total":maxcap,
                "allocated_feature_bytes":feature_alloc,
                "allocated_metadata_bytes":metadata,
                "candidate_events":self.events,
                "candidate_patches":self.patches_seen,
                "consolidation_attempts":self.attempts,
                "consolidation_successes":self.successes,
                "promoted_prototypes":self.promoted_total,
                "max_eligible_candidates":self.eligible_peak,
                "update_seconds":self.update_seconds,
                "ltm_retrieval":self.bank.enable_ltm_retrieval,
                "buffer_retrieval":self.bank.enable_maturation_buffer_retrieval,
                "utility_retention":self.bank.enable_utility_aware_stm_retention}
        return counts


def assert_same_stm_and_ltm(arms,name_a="shadow_utility",name_b="ltm_utility"):
    """Shadow vs LTM retrieval: same update rules, SAME physical state."""
    import torch
    a,b=arms[name_a].bank,arms[name_b].bank
    for fam in ("normal_dyn","normal_dyn_ltm","normal_dyn_candidates"):
        group_a=getattr(a,fam)["p0"][0]["id"]
        group_b=getattr(b,fam)["p0"][0]["id"]
        for x,y in zip(group_a,group_b):
            if x.size()!=y.size() or not torch.equal(
                    x.protos[:x.size()],y.protos[:y.size()]):
                raise AssertionError(f"State coupling: {name_a}/{name_b}/{fam}")
