#!/usr/bin/env python3
"""CPU preflight of actual v22 MaturationBuffer recurrence/promotion semantics.

Run before any Real-IAD GPU execution. This deliberately uses synthetic normal
events, only to verify the code path. No Real-IAD performance is inferred.
"""
from __future__ import annotations
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def main():
    import torch
    import test_phase3_AIF_simplified_v22 as impl
    assert hasattr(impl,"MaturationBuffer")
    b=impl.MaturationBuffer(8,4,torch.device("cpu"),merge_eps=.06,
                           max_event_prototypes=1)
    v=torch.tensor([[1.,0.,0.,0.]])
    for t in (10,40,75):
        out=b.update(v,usefulness=1.,event_id=t,step=t,
                     outcome="FP",boundary_value=.9,coverage_gain=.8)
        assert out["n_event_protos"]==1
    assert b.size()==1
    assert int(b.distinct_events[0].item())==3
    chosen,report=b.get_promotion_candidates(
        min_repeat=3,min_usefulness=.1,score_threshold=.45,max_promote=4)
    assert report["eligible_count"]>=1,report
    assert chosen.shape[0]>=1
    before=chosen.clone()
    # A single event with multiple similar patches must not create recurrence.
    b2=impl.MaturationBuffer(8,4,torch.device("cpu"),merge_eps=.06)
    b2.update(v.repeat(8,1),usefulness=1.,event_id=5,step=5,
              outcome="FP",boundary_value=.9,coverage_gain=.8)
    assert max(int(x) for x in b2.distinct_events[:b2.size()])==1
    fail,_=b2.get_promotion_candidates(
        min_repeat=3,min_usefulness=.1,score_threshold=.45,max_promote=4)
    assert fail.shape[0]==0
    # End-to-end: actual PatchVMB.update_from_topk must WRITE LTM and use it.
    support=torch.tensor([[1.,0.,0.,0.],[0.,1.,0.,0.],
                          [0.,0.,1.,0.],[1.,1.,0.,0.],
                          [0.,1.,1.,0.],[1.,0.,1.,0.]])
    bank=impl.PatchVMB(
        device="cpu",K_normal_fixed=16,K_normal_dyn=8,K_normal_ltm=4,
        K_defect=4,K_concept=8,knn_k_normal=5,
        corr_cfg=impl.CorrectionConfig(enabled=False),use_defect_bank=False)
    bank.bank_dtype=torch.float32
    bank.build_fixed_normal_from_support(
        "p0",[[support]],mode=0,view="id",normal_seed_patches=6,seed=0)
    bank.shadow_ltm=False
    bank.enable_maturation_buffer=True
    bank.enable_ltm_promotion=True
    bank.enable_ltm_retrieval=True
    bank.mature_promote_every=3
    bank.mature_min_events=2
    bank.mature_min_usefulness=0.0
    bank.mature_score_threshold=.1
    bank.mature_max_event_protos=1
    for t in range(1,7):
        report=bank.update_from_topk(
            cls_name="p0",feats_per_layer=[v.repeat(8,1)],
            which="normal_dyn",mode=0,view="id",
            layer_to_patch_idx={0:[0,1,2,3]},tau_insert=.15,
            tau_replace=.25,event_id=t,write_value=1.,
            outcome="FP",boundary_value=.9,coverage_gain=.8)
    ltm=bank.normal_dyn_ltm["p0"][0]["id"][0]
    assert ltm.size()>0, (
        "Real v22 update→maturation→consolidation failed to store LTM")
    # Isolate the LTM retrieval contribution: STM deliberately cleared AFTER
    # promotion in this synthetic functionality check only.
    bank.normal_dyn["p0"][0]["id"][0].clear()
    bank.enable_ltm_retrieval=False
    d0,_=bank._dN_min_fixed_dyn("p0",0,"id",0,v)
    bank.enable_ltm_retrieval=True
    d1,_=bank._dN_min_fixed_dyn("p0",0,"id",0,v)
    assert float(d1.mean())+1e-5<float(d0.mean()), (
        "LTM was stored but did not influence native retrieval")
    print("A1-H native PatchVMB maturation→promotion→retrieval CPU preflight PASS")
    print("A1-H v22 maturation/recurrence/promotion CPU mechanism preflight PASS")


if __name__=="__main__":
    main()
