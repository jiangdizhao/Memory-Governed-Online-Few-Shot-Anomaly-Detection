#!/usr/bin/env python3
"""Native CPU mechanisms and compact TMRA memory ceiling preflight."""
from __future__ import annotations
from pathlib import Path
import copy
import sys
ROOT=Path(__file__).resolve().parents[1]
for p in (ROOT,ROOT/"experiments"):
    if str(p) not in sys.path:sys.path.insert(0,str(p))

def main():
    import torch
    import test_phase3_AIF_simplified_v22 as impl
    import a1h_mechanism_test as native
    from tmra_compact_memory import Arm,allocated_state_bytes
    from tmra_recurrence_core import Settings
    from tmra_audit_core import shapley
    native.main()   # actual v22 Maturation -> promotion -> retrieval CPU
    support=torch.tensor([[1.,0.,0.,0.],[0.,1.,0.,0.],
                          [0.,0.,1.,0.],[1.,1.,0.,0.]])
    bank=impl.PatchVMB(device="cpu",K_normal_fixed=16,K_normal_dyn=384,
                       K_normal_ltm=64,K_defect=4,K_concept=8,
                       knn_k_normal=5,
                       corr_cfg=impl.CorrectionConfig(enabled=False),
                       use_defect_bank=False)
    # Match the actual DINO deployment memory storage dtype (fp16).
    bank.bank_dtype=torch.float16
    bank.build_fixed_normal_from_support(
        "p0",[[support]],mode=0,view="id",
        normal_seed_patches=4,seed=0)
    settings=Settings(phase_length=64)
    arms={name:Arm(name,copy.deepcopy(bank),settings,seed=37)
          for name in ("greedy_coreset","flat_utility","hierarchy")}
    feat=[torch.tensor([[0.,0.,0.,1.]]).repeat(24,1)]
    for t in range(1,35):
        for arm in arms.values():
            arm.update(impl,feat,{0:list(range(24))},t)
    for a in arms.values():
        report=a.audit()
        assert report["feedback_events"]==34
        assert report["allocated_state_bytes"]<=3_500_000
        assert report["candidate_patches"]==816
    assert arms["hierarchy"].audit()["buffer"]>0
    assert set(shapley({"S0":.8,"SL":.7,"SB":.6,"SLB":.55}))=={
        "phi_ltm","phi_buffer","combined_score_reduction","interaction"}
    print("[TMRA] Actual v22 CPU mechanism + 3.5 MB ceiling PASS")

if __name__=="__main__":
    main()
