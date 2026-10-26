"""TMRA-I read-only four-source attribution and patch-level native-distance audit."""
from __future__ import annotations
from contextlib import contextmanager

SOURCES=("fixed","stm","ltm","buffer")

@contextmanager
def source_flags(bank,ltm,buffer):
    old=(bank.enable_ltm_retrieval,bank.enable_maturation_buffer_retrieval)
    bank.enable_ltm_retrieval=bool(ltm)
    bank.enable_maturation_buffer_retrieval=bool(buffer)
    try:
        yield bank
    finally:
        bank.enable_ltm_retrieval,bank.enable_maturation_buffer_retrieval=old

def score_counterfactual(score_one,bank):
    """No writes; four scores measured from exactly the same bank object."""
    result={}
    for key,has_ltm,has_buf in (
        ("S0",False,False),
        ("SL",True,False),
        ("SB",False,True),
        ("SLB",True,True),
    ):
        with source_flags(bank,has_ltm,has_buf):
            result[key]=float(score_one())
    return result

def shapley(scores):
    """Marginal reduction in anomaly score from a normal-memory source.

    Positive value = decreases anomaly score (potential hard-normal rescue
    AND potential missed-defect risk). No labels enter this computation.
    """
    s0,sl,sb,slb=(float(scores[k]) for k in ("S0","SL","SB","SLB"))
    pl=.5*((s0-sl)+(sb-slb))
    pb=.5*((s0-sb)+(sl-slb))
    assert abs((pl+pb)-(s0-slb))<1e-5
    return {"phi_ltm":pl,"phi_buffer":pb,
            "combined_score_reduction":s0-slb,
            "interaction":sl+sb-s0-slb}

def patch_source_details(bank,impl,features,top_n=12):
    """Native dN distances per patch, do not claim these are semantic regions.

    A source wins only when it has the smallest *effective* native distance.
    The per-patch dN map exactly equals the image score map only when
    defect and correction banks are disabled (checked here).
    """
    import torch
    if bank.use_defect_bank or bank.corr_cfg.enabled:
        raise RuntimeError("Patch-only normal distance audit expects no defect/correction")
    records=[]
    src_counts=dict.fromkeys(SOURCES,0)
    selected=[]
    for layer,feat in enumerate(features):
        q=impl.l2_normalize(feat.to(bank.device))
        fixed=bank.normal_fixed["p0"][0]["id"][layer]
        stm=bank.normal_dyn["p0"][0]["id"][layer]
        ltm=bank.normal_dyn_ltm["p0"][0]["id"][layer]
        tracker=bank.normal_dyn_candidates["p0"][0]["id"][layer]
        df,_=fixed.nearest(q,relax=True,k=bank.knn_k_normal)
        ds,_=stm.nearest(q,relax=True,k=bank.knn_k_normal) if stm.size() else (torch.full_like(df,10.),None)
        dl,_=ltm.nearest(q,relax=False) if ltm.size() else (torch.full_like(df,10.),None)
        db=torch.full_like(df,10.)
        eligible=0
        if tracker.size():
            p,weights,info=tracker.get_retrieval_prototypes(
                min_repeat=int(bank.mature_retrieval_min_events),
                min_usefulness=float(bank.mature_retrieval_min_usefulness),
                score_threshold=float(bank.mature_retrieval_score_thr),
                max_protos=int(bank.mature_retrieval_max_protos))
            eligible=int(p.shape[0])
            if eligible:
                P=impl.l2_normalize(p.to(q.device).float())
                Q=impl.l2_normalize(q.float())
                distances=1-torch.clamp(Q @ P.T,-1.,1.)
                adjusted=distances+float(bank.mature_retrieval_penalty)*(
                    1-weights.to(q.device).float().view(1,-1))
                k=max(1,min(int(bank.mature_retrieval_k),eligible))
                db=adjusted.topk(k,dim=1,largest=False).values.mean(dim=1)
        matrix=torch.stack((df,ds,dl,db))
        winning=matrix.argmin(dim=0)
        without=torch.minimum(df,ds)
        with_all=matrix.min(dim=0).values
        benefit=(without-with_all).clamp_min(0)
        for source in SOURCES:
            src_counts[source]+=int((winning==SOURCES.index(source)).sum().item())
        # Store only patches that can change score, plus most anomalous patch.
        wanted=set(torch.topk(benefit,k=min(top_n,len(benefit))).indices.tolist())
        wanted.add(int(with_all.argmax().item()))
        for i in sorted(wanted):
            dsrow=[float(v[i].item()) for v in (df,ds,dl,db)]
            records.append({
                "layer":int(layer),"patch_idx":int(i),
                "source_winner":SOURCES[int(winning[i].item())],
                "d_fixed":dsrow[0],"d_stm":dsrow[1],
                "d_ltm":dsrow[2] if ltm.size() else None,
                "d_buffer_eff":dsrow[3] if eligible else None,
                "d_native_full":float(with_all[i].item()),
                "d_no_extra":float(without[i].item()),
                "d_extra_reduction":float(benefit[i].item()),
                "eligible_buffer_prototypes":eligible,
            })
    return records,src_counts
