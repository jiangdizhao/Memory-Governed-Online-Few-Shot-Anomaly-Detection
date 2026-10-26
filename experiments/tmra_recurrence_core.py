"""TMRA-II predeclared synthetic acquisition shift and mode-recurrence design.

No operations on training supports, defect masks, or probe labels. Does not
represent a measured real-world machine operating mode.
"""
from __future__ import annotations
from dataclasses import dataclass
import hashlib
import json
import random

@dataclass(frozen=True)
class Settings:
    phase_length: int=900
    probe_count: int=96
    q_rate: float=.03
    stm: int=128
    ltm: int=64
    buffer: int=64
    flat: int=384
    promote_every: int=16
    min_events: int=2
    cap_bytes: int=3_500_000
    def __post_init__(self):
        if self.phase_length<64 or self.probe_count<16:
            raise ValueError("Need meaningful stream/probe lengths")
        if (self.stm,self.ltm,self.buffer,self.flat)!=(128,64,64,384):
            raise ValueError("TMRA-II capacities are frozen; do not tune")
        if (self.promote_every,self.min_events)!=(16,2):
            raise ValueError("TMRA-II maturation gates are preregistered")
        if self.q_rate!=.03 or self.cap_bytes!=3_500_000:
            raise ValueError("Query/bytes budget is preregistered")

def hash_manifest(x):
    return hashlib.sha256(json.dumps(x,sort_keys=True).encode()).hexdigest()

def label_blind_partition(samples,settings:Settings,seed:int):
    """Original-identity-disjoint fixed evaluation and three phases.

    Derive all from TEST-only dataset items. No stratification by label:
    class labels are reserved for post-prediction scoring and feedback.
    """
    rng=random.Random(seed)
    pool=sorted(samples,key=lambda x:x.path)
    rng.shuffle(pool)
    total=2*settings.probe_count+3*settings.phase_length
    if len(pool)<total:
        raise ValueError(f"Need {total} original images, have {len(pool)}")
    A=pool[:settings.probe_count]
    B=pool[settings.probe_count:2*settings.probe_count]
    raw=pool[2*settings.probe_count:total]
    phases=[raw[i*settings.phase_length:(i+1)*settings.phase_length]
            for i in range(3)]
    all_paths=[z.path for z in A+B+raw]
    if len(set(all_paths))!=len(all_paths):
        raise AssertionError("Raw image leakage/repetition")
    return A,B,phases

def phase_modes(order):
    if order=="ABA":return ("A","B","A")
    if order=="AAA":return ("A","A","A")
    raise ValueError(order)

def transform_image(img,mode):
    """Fixed, bounded covariate shift: linear channel gains + brightness.

    Mode A is exact identity. Mode B simulates altered illumination and
    color balance without changing object geometry or defect labels.
    """
    if mode=="A":return img
    if mode!="B":raise ValueError(mode)
    import numpy as np
    from PIL import Image
    x=np.asarray(img.convert("RGB"),dtype=np.float32)
    gains=np.array([0.84,0.94,1.08],dtype=np.float32)
    y=np.clip(x*gains.reshape(1,1,3)+8.0,0,255).astype(np.uint8)
    return Image.fromarray(y,"RGB")

def first_recovery(values,steps,reference,tolerance=.01):
    """Measured only at post-feedback checkpoints in Phase III.

    Return new confirmed normal labels until reaching Phase-I AUROC
    within tolerance; null means target was not regained.
    """
    if reference is None:return None
    target=reference-tolerance
    for val,n in zip(values,steps):
        if val is not None and val>=target:
            return int(n)
    return None
