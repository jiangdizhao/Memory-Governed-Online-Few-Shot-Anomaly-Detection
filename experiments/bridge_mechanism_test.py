#!/usr/bin/env python3
"""CPU fail-closed preflight of real v22 and Bridge memory flags.

This is synthetic operator verification, not Real-IAD performance evidence.
"""
from __future__ import annotations
import copy
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
for p in (ROOT,ROOT/"experiments"):
    if str(p) not in sys.path:sys.path.insert(0,str(p))

def main():
    import torch
    import a1h_mechanism_test
    from bridge_memory import BridgeArm,BridgeConfig,assert_same_stm_and_ltm
    import test_phase3_AIF_simplified_v22 as impl
    # Already locally verified in A1-H. Tests actual native update->LTM->score.
    a1h_mechanism_test.main()
    support=torch.tensor([
        [1.,0.,0.,0.],[0.,1.,0.,0.],[0.,0.,1.,0.],
        [0.,1.,1.,0.],[1.,0.,1.,0.],[1.,1.,0.,0.]])
    bank=impl.PatchVMB(
        device="cpu",K_normal_fixed=16,K_normal_dyn=512,K_normal_ltm=512,
        K_defect=4,K_concept=8,knn_k_normal=5,
        corr_cfg=impl.CorrectionConfig(enabled=False),use_defect_bank=False)
    bank.bank_dtype=torch.float32
    bank.build_fixed_normal_from_support(
        "p0",[[support]],mode=0,view="id",normal_seed_patches=6,seed=0)
    cfg=BridgeConfig(stm=512)
    arms={name:BridgeArm(name,copy.deepcopy(bank),cfg)
          for name in ("shadow_utility","ltm_utility","full_utility")}
    for name,arm in arms.items():
        assert arm.bank.mature_min_events==2
        assert arm.bank.mature_promote_every==32
        assert arm.bank.mature_retrieval_k==1
        assert arm.bank.ltm_retrieval_k==1
        assert arm.bank.normal_dyn_candidates["p0"][0]["id"][0].K==4096
    assert not arms["shadow_utility"].bank.enable_ltm_retrieval
    assert arms["ltm_utility"].bank.enable_ltm_retrieval
    assert arms["full_utility"].bank.enable_maturation_buffer_retrieval
    v=torch.tensor([[0.,0.,0.,1.]])
    for event in range(1,45):
        feats=[v.repeat(24,1)]
        for arm in arms.values():
            arm.observe_verified_normal(
                impl,feats,{0:list(range(24))},step=event,
                score=.8,theta=.7,sigma=.1)
    assert_same_stm_and_ltm(arms)
    for name in ("ltm_utility","full_utility"):
        st=arms[name].state()
        if st["ltm"]<1 or st["consolidation_successes"]<1:
            raise AssertionError(
                f"Historical E_min=2 / period=32 synthetic LTM failed: {name}, {st}")
    print("[BRIDGE] Historical STM/LTM/maturation code paths PASS")
    print("[BRIDGE] Bridge arm state equivalence PASS")

if __name__=="__main__":main()
