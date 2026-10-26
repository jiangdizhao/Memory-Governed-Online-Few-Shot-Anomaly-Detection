"""FINAL hierarchy decision protocol. Frozen before GPU evidence.

No grid-search, no threshold tuning, no sequential extension. Two contestants:
legacy v22 hierarchy and A0/A1 online Greedy Coreset. Image-level feedback is
3% label-blind, maximum one queried view per real physical instance.
"""
from __future__ import annotations
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import random
import re

SCHEMA="final.hierarchy.decision.v1"
CLASSES=("switch","audiojack")
SEEDS=(0,1)
ORDERS=("ABA","AAA")
GROUPS_PER_PHASE=175
A_PROBE_GROUPS=24
B_PROBE_GROUPS=24
QUERY_RATE=.03
BYTE_CEILING=2_300_000
# Constraints frozen, NOT a grid. Our recorded approx deployed state allocations
# are 2.11 MB Greedy(256), 2.11-2.13 MB Hierarchy(128+64+64).
@dataclass(frozen=True)
class MemorySettings:
    stm:int=128
    ltm:int=64
    buffer:int=64
    flat:int=256
    cap_bytes:int=BYTE_CEILING
    promote_every:int=32
    min_events:int=2

    def __post_init__(self):
        if (self.stm,self.ltm,self.buffer,self.flat,
            self.cap_bytes,self.promote_every,self.min_events) != (
                128,64,64,256,BYTE_CEILING,32,2):
            raise ValueError("FINAL memory protocol is immutable")

@dataclass(frozen=True)
class DatasetSettings:
    a_probe_groups:int=A_PROBE_GROUPS
    b_probe_groups:int=B_PROBE_GROUPS
    per_phase:int=GROUPS_PER_PHASE

    def __post_init__(self):
        if (self.a_probe_groups,self.b_probe_groups,self.per_phase)!=(24,24,175):
            raise ValueError("FINAL instance-group counts are immutable")

def digest(x):
    return hashlib.sha256(json.dumps(x,sort_keys=True,ensure_ascii=False).encode("utf-8")).hexdigest()

# Group = same part ID + same OK or NG defect type; camera C1-5 / timestamp
# may vary. Fail closed if naming differs, never silently infer independence.
_RE=re.compile(r"^(?P<instance>.+)_C[1-5]_\d{14}\.(?:jpg|jpeg|png)$",re.I)
def object_id(path):
    m=_RE.match(Path(str(path).replace("\\","/")).name)
    if m is None:
        raise ValueError(f"Cannot prove Real-IAD physical identity: {path}")
    return m.group("instance")

def group_samples(samples):
    by={}
    paths=set()
    for s in samples:
        p=s.path
        if p in paths:raise AssertionError(f"Duplicate image path {p}")
        paths.add(p)
        by.setdefault(object_id(p),[]).append(s)
    return {k:sorted(v,key=lambda x:x.path) for k,v in sorted(by.items())}

def choose_supports(train,seed:int):
    """Exactly four normal original images from four distinct objects."""
    groups=group_samples(s for s in train if s.is_good)
    if len(groups)<4:raise ValueError("Fewer than four normal physical objects")
    group_ids=sorted(random.Random(seed).sample(list(groups),4))
    rng=random.Random(seed+891)
    supports=[rng.choice(groups[k]).path for k in group_ids]
    if len(supports)!=4 or len(set(map(object_id,supports)))!=4:
        raise AssertionError("Four-shot instance isolation failed")
    return sorted(supports)

def make_split(test,support_paths,seed:int):
    by=group_samples(test)
    excluded={object_id(p) for p in support_paths}
    remaining=sorted(set(by)-excluded)
    plan=DatasetSettings()
    n=plan.a_probe_groups+plan.b_probe_groups+3*plan.per_phase
    if len(remaining)<n:
        raise ValueError(f"Physical-instance grouping leaves {len(remaining)} groups; "
                         f"final frozen trial requires {n}; do not shorten by tuning")
    rng=random.Random(seed+2201)
    rng.shuffle(remaining)
    a_ids=remaining[:plan.a_probe_groups]
    b_ids=remaining[plan.a_probe_groups:plan.a_probe_groups+plan.b_probe_groups]
    rest=remaining[plan.a_probe_groups+plan.b_probe_groups:n]
    phase_ids=[rest[i*plan.per_phase:(i+1)*plan.per_phase] for i in range(3)]
    all_ids=a_ids+b_ids+sum(phase_ids,[])
    if len(set(all_ids))!=len(all_ids):raise AssertionError("Physical identity leakage")
    def images(ids):return [sample for k in ids for sample in by[k]]
    probes={"A":images(a_ids),"B":images(b_ids)}
    phases=[images(x) for x in phase_ids]
    return probes,phases,{"A":a_ids,"B":b_ids,"phases":phase_ids,
                         "excluded_support_objects":sorted(excluded)}

def query_tape(phases,rate=QUERY_RATE,seed=37):
    """Exactly round(3% of image observations) one-label-per-object tape.

    Choose physical objects uniformly, then a single camera view for each.
    These indices are generated without using normality or defect labels.
    """
    if rate!=QUERY_RATE:raise ValueError("Query rate cannot be retuned")
    flat=sum(phases,[])
    by={}
    for i,s in enumerate(flat,1):
        by.setdefault(object_id(s.path),[]).append(i)
    if len(by)!=len(set(by)):raise AssertionError("Internal group error")
    n=int(round(rate*len(flat)))
    if n<1 or n>len(by):raise ValueError("Insufficient independent query groups")
    rng=random.Random(seed)
    groups=rng.sample(sorted(by),n)
    selected=sorted(rng.choice(by[g]) for g in groups)
    assert len(selected)==len({object_id(flat[t-1].path) for t in selected})
    return set(selected)

def auroc(y,s):
    from sklearn.metrics import roc_auc_score
    if set(y)!={0,1}:raise ValueError("Probe must contain both labels")
    return float(roc_auc_score(y,s))

def phase_modes(order):
    return {"ABA":("A","B","A"),"AAA":("A","A","A")}[order]

def recovery_count(points,reference,margin=.01):
    target=reference-margin
    for labels,value in points:
        if value>=target:return labels
    return None

def pass_decision(rows):
    """One hard final engineering decision, NOT a statistical theorem.

    Any missing/invalid/unconvincing hierarchy proof chooses simpler Greedy.
    Positive hierarchy decision requires a clear repeated, meaningful gain.
    """
    if len(rows)!=4:
        return {"keep_hierarchy":False,"reason":"missing paired conditions"}
    diffs=[r["hierarchy_shift_advantage"] for r in rows]
    mean=sum(diffs)/4
    wins=sum(z>=.01 for z in diffs)
    mean_quality=sum(r["hierarchy_end2_minus_greedy_end2"] for r in rows)/4
    mean_B=sum(r["hierarchy_B_end2_minus_greedy_B_end2"] for r in rows)/4
    ltm_active=sum(bool(r["A_retrieval_effect"]) for r in rows)
    # Sensitivity / FNR is reported, never silently optimized on Probe labels.
    keep=bool(mean>=.015 and wins>=3 and mean_quality>=-.005
              and mean_B>=-.01 and ltm_active>=2)
    failed=[
        *(["mean shift-retention gain <0.015"] if mean<.015 else []),
        *(["fewer than 3/4 paired gains >=0.01"] if wins<3 else []),
        *(["end-II A discrimination inferior by >0.005"] if mean_quality<-.005 else []),
        *(["end-II B discrimination inferior by >0.01"] if mean_B<-.01 else []),
        *(["insufficient same-state LTM/Buffer retrieval evidence"] if ltm_active<2 else []),
    ]
    return {"keep_hierarchy":keep,
            "decision":("RETAIN_HIERARCHY" if keep else "USE_GREEDY_WITH_RISK_SAFE_ADMISSION"),
            "avg_shift_retention_advantage":mean,
            "positive_pairs_ge_0.01":wins,
            "avg_end2_A_quality_difference":mean_quality,
            "avg_end2_B_quality_difference":mean_B,
            "active_retrieval_pairs":ltm_active,
            "failed_gates":failed}
