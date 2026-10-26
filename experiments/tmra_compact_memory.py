"""TMRA-II compact matched BYTE-CEILING external normal memory arms.

All arms obey a *common upper byte bound*, not falsely equal actual bytes.
The 4096-dimensional FP32 maturation tracker is counted at allocation.
"""
from __future__ import annotations
import time
import tmra_recurrence_core as core
import a0_a1_memory_methods as a0

NAMES=("static","greedy_coreset","flat_utility","hierarchy")

def codebooks(bank):
    return (
        bank.normal_dyn["p0"][0]["id"],
        bank.normal_dyn_ltm["p0"][0]["id"],
        bank.normal_dyn_candidates["p0"][0]["id"],
    )

def allocated_state_bytes(bank):
    """Count mutable PyTorch tensors, including tracker preallocation.

    Does not include Python object memory or immutable fixed reference bank.
    Does not claim physical CUDA allocator residency equals this figure.
    """
    import torch
    total=0
    seen=set()
    for group in codebooks(bank):
        for cb in group:
            for value in vars(cb).values():
                if isinstance(value,torch.Tensor):
                    storage=value.untyped_storage()
                    ptr=(str(value.device),storage.data_ptr(),storage.nbytes())
                    if ptr not in seen:
                        seen.add(ptr)
                        total+=storage.nbytes()
    return total

class Arm:
    def __init__(self,name,bank,settings:core.Settings,seed:int):
        if name not in NAMES or name=="static":
            raise ValueError("Static bank is handled directly")
        self.name,self.bank,self.settings=name,bank,settings
        self.events=0
        self.patches=0
        self.promotions=0
        self.update_seconds=0.
        self.greedy=None
        self._configure(seed)
    def _configure(self,seed):
        import torch
        b=self.bank
        x=self.settings
        b.use_defect_bank=False
        b.corr_cfg.enabled=False
        hierarchy=self.name=="hierarchy"
        b.enable_maturation_buffer=hierarchy
        b.enable_ltm_promotion=hierarchy
        b.enable_ltm_retrieval=hierarchy
        b.enable_maturation_buffer_retrieval=hierarchy
        b.enable_utility_aware_stm_retention=self.name!="greedy_coreset"
        b.enable_stm_retirement=False
        b.shadow_ltm=False
        b.K_normal_dyn=x.stm if hierarchy else x.flat
        b.K_normal_ltm=x.ltm if hierarchy else 1
        b.ltm_candidate_capacity=x.buffer if hierarchy else 1
        b.mature_max_candidates=b.ltm_candidate_capacity
        b.mature_min_events=x.min_events
        b.ltm_min_repeat=x.min_events
        b.mature_promote_every=x.promote_every
        b.mature_min_usefulness=.15
        b.mature_score_threshold=.45
        b.mature_max_event_protos=8
        b.mature_event_merge_eps=.03
        b.mature_retrieval_min_events=2
        b.mature_retrieval_min_usefulness=.1
        b.mature_retrieval_score_thr=.25
        b.mature_retrieval_penalty=.01
        b.mature_retrieval_k=1
        b.ltm_retrieval_k=1
        b.stm_retention_replace_margin=.03
        stm,ltm,buf=codebooks(b)
        if any(v.size() for group in (stm,ltm,buf) for v in group):
            raise AssertionError("Mutable memory not empty at start")
        for c in stm:c.K=b.K_normal_dyn
        for c in ltm:c.K=b.K_normal_ltm
        for i,old in enumerate(buf):
            buf[i]=type(old)(
                b.ltm_candidate_capacity,old.feat_dim,old.device,
                merge_eps=.06,max_event_prototypes=8,event_merge_eps=.03)
        if self.name=="greedy_coreset":
            # Its original a0 memory method uses a precise simple online greedy
            # replacement rule. All mutable stores were resized BEFORE init.
            self.greedy=a0.MemoryArm(
                "online_greedy_coreset",b,
                dynamic_capacity=x.flat,seed=seed)
        self._assert_bound()
    def _assert_bound(self):
        total=allocated_state_bytes(self.bank)
        if total>self.settings.cap_bytes:
            raise AssertionError(f"{self.name}: {total} bytes > {self.settings.cap_bytes}")
        stm,ltm,buf=codebooks(self.bank)
        if (sum(c.size() for c in stm)>self.bank.K_normal_dyn
            or sum(c.size() for c in ltm)>self.bank.K_normal_ltm
            or sum(c.size() for c in buf)>self.bank.ltm_candidate_capacity):
            raise AssertionError("Prototype capacity overrun")
        return total
    def update(self,impl,feats,candidates,step):
        import torch
        self.events+=1
        self.patches+=sum(len(v) for v in candidates.values())
        if torch.cuda.is_available() and feats[0].is_cuda:torch.cuda.synchronize()
        begin=time.perf_counter()
        if self.greedy is not None:
            self.greedy.observe_verified_normal(
                impl,feats,candidates,step=step,score=0.5,theta=0.5,sigma=.1)
        else:
            report=self.bank.update_from_topk(
                cls_name="p0",feats_per_layer=feats,which="normal_dyn",
                mode=0,view="id",layer_to_patch_idx=candidates,
                tau_insert=.15,tau_replace=.25,event_id=step,
                write_value=.5,outcome="TN",boundary_value=.5,
                coverage_gain=0.)
            if report.get("consolidation",{}).get("consolidated"):
                self.promotions+=1
        if torch.cuda.is_available() and feats[0].is_cuda:torch.cuda.synchronize()
        self.update_seconds+=time.perf_counter()-begin
        self._assert_bound()
    def audit(self):
        stm,ltm,buf=codebooks(self.bank)
        return dict(
            stm=sum(c.size() for c in stm),
            ltm=sum(c.size() for c in ltm),
            buffer=sum(c.size() for c in buf),
            allocated_state_bytes=self._assert_bound(),
            byte_ceiling=self.settings.cap_bytes,
            feedback_events=self.events,candidate_patches=self.patches,
            promotions=self.promotions,update_seconds=self.update_seconds)
