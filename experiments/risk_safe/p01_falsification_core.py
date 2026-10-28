"""P0/P1 controlled calibration and admission-matching primitives.

No probe/test labels enter admission selectors. Top-K selectors below are
RETROSPECTIVE, using future queried-normal metadata; not causal online policies.
"""
from __future__ import annotations

import hashlib
import math
from typing import Sequence


class FalsificationProtocolError(RuntimeError):
    pass


def deterministic_hash(key: str, seed: int) -> str:
    return hashlib.sha256(f"p01.v1|seed={seed}|{key}".encode("utf-8")).hexdigest()


def select_matched(records: Sequence[dict], target: int, *, seed: int) -> dict:
    """Exactly matched normal EVENT count, NOT patch count or final bank size.

    records: in chronological verified-normal order; each has identity,
    static_score and static_patch_novelty computed from frozen features.
    Hindsight selections are causally invalid as online controllers.
    """
    n = len(records)
    if not 0 < target <= n:
        raise FalsificationProtocolError("target must be <= verified normal candidates")
    keys = [str(row["sample_key"]) for row in records]
    if len(set(keys)) != n:
        raise FalsificationProtocolError("Query tape normal sample keys repeated")
    for row in records:
        for name in ("static_score", "static_patch_novelty"):
            if not math.isfinite(float(row[name])):
                raise FalsificationProtocolError("Nonfinite selector signal")
    indices = list(range(n))
    return {
        "first_k": set(indices[:target]),
        "hash_k": set(sorted(indices, key=lambda i: (
            deterministic_hash(keys[i], seed), i))[:target]),
        "static_novelty_top_k": set(sorted(indices, key=lambda i: (
            -float(records[i]["static_patch_novelty"]), i))[:target]),
        "static_score_top_k": set(sorted(indices, key=lambda i: (
            -float(records[i]["static_score"]), i))[:target]),
    }


def normal_q90(scores: Sequence[float]) -> float:
    """Same linear 0.9 quantile convention as original D2 threshold fitter."""
    z = sorted(float(x) for x in scores)
    if len(z) != 64 or any(not math.isfinite(x) for x in z):
        raise FalsificationProtocolError("Require exactly 64 normal training scores")
    pos = .90 * (len(z) - 1)
    i = int(math.floor(pos))
    return z[i] + (pos-i) * (z[min(i+1,len(z)-1)]-z[i])


def metrics(y: Sequence[int], scores: Sequence[float], theta: float) -> dict:
    if len(y)!=len(scores) or not y or set(y)!={0,1}:
        raise FalsificationProtocolError("Need same-length binary probe labels both classes")
    if not math.isfinite(theta):
        raise FalsificationProtocolError("Nonfinite threshold")
    vals = [float(x) for x in scores]
    if any(not math.isfinite(x) for x in vals):
        raise FalsificationProtocolError("Nonfinite scores")
    tp=tn=fp=fn=0
    for label,score in zip(y, vals):
        positive = score >= theta
        if label == 1:
            tp += int(positive)
            fn += int(not positive)
        else:
            fp += int(positive)
            tn += int(not positive)
    n0, n1 = tn+fp, tp+fn
    # Mann–Whitney AUROC with average ranks for ties.
    ranked = sorted(zip(vals,y), key=lambda x:x[0])
    rank_sum=0.
    i=0
    while i<len(ranked):
        j=i+1
        while j<len(ranked) and ranked[j][0]==ranked[i][0]:
            j+=1
        mid_rank=(i+1+j)/2
        rank_sum+=mid_rank*sum(label for _,label in ranked[i:j])
        i=j
    auc = (rank_sum-n1*(n1+1)/2)/(n0*n1)
    # Average precision, grouped ties treated as one shared score threshold.
    descending = sorted(zip(vals,y), key=lambda x:x[0], reverse=True)
    total=positives=0
    ap=0.
    i=0
    while i<len(descending):
        j=i+1
        while j<len(descending) and descending[j][0]==descending[i][0]:
            j+=1
        group_pos=sum(v for _,v in descending[i:j])
        total+=j-i
        positives+=group_pos
        ap+=group_pos/n1*(positives/total)
        i=j
    return dict(n=len(y),n_normal=n0,n_defect=n1,tp=tp,tn=tn,fp=fp,fn=fn,
        fpr=fp/n0,fnr=fn/n1,balanced_accuracy=(tn/n0+tp/n1)/2,
        auroc=auc,auprc=ap,theta=float(theta))


def paired_changes(y: Sequence[int], base: Sequence[float], alt: Sequence[float],
                   base_theta: float, alt_theta: float) -> dict:
    rescued_fp=new_fp=new_fn=rescued_fn=0
    for label, b, a in zip(y, base, alt):
        old=float(b)>=base_theta
        new=float(a)>=alt_theta
        if label==0:
            rescued_fp+=int(old and not new)
            new_fp+=int(not old and new)
        else:
            new_fn+=int(old and not new)
            rescued_fn+=int(not old and new)
    return dict(rescued_false_positives=rescued_fp,new_false_positives=new_fp,
                new_false_negatives=new_fn,rescued_false_negatives=rescued_fn)


def select_oracle_bits(summary: dict, *, tape_sha: str, n_normal: int,
                       target: int, capacity: int) -> set[int]:
    cfg=summary.get("config",{})
    if summary.get("status")!="COMPLETE" or summary.get("globally_exhaustive") is not False:
        raise FalsificationProtocolError("Required completed heuristic R1 beam, not an exact oracle")
    if cfg.get("budget")!=128 or cfg.get("capacity")!=capacity or (
       summary.get("oracle_provenance",{}).get("tape_sha256") != tape_sha):
        raise FalsificationProtocolError("R1 Oracle protocol or tape mismatch")
    bits=summary.get("oracle_decision_bits","")
    if len(bits)!=n_normal or any(x not in "01" for x in bits):
        raise FalsificationProtocolError("R1 Oracle decision length mismatch")
    chosen={i for i,v in enumerate(bits) if v=="1"}
    if len(chosen)!=target:
        raise FalsificationProtocolError("R1 Oracle selected count mismatch")
    return chosen
