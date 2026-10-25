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
    print("A1-H v22 maturation/recurrence/promotion CPU mechanism preflight PASS")


if __name__=="__main__":
    main()
