"""D4 exact-equal-win retrieval-rights / feedback attribution (development only).

This module contains NO dataset labels, training split access, or GPU feature
extractors. It selects the same number K of real native dynamic winners per
query among all arms, then permits ALL original dynamic codebook prototypes
at selected query patches. The native v22 scorer recomputes exactly the same
fixed top-k distance for enabled patches; all other patches use Static.

The nonparametric credit model is based exclusively on B16 *revealed* source
images. Its defect-local evidence is weakly supervised, not pixel GT. This is
a falsifiable baseline / mechanism probe, not a safety guarantee.
"""
from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Sequence

import torch

from .native_bridge import NativeParityError

METHODS=("random","spatial","normal_feedback",
         "normal_defect_feedback","shuffled_defect_feedback")
REPLICATES=(0,1,2)
RHO=.02
BANDWIDTH=.10
SPATIAL_BANDWIDTH=.25
RISK_MULTIPLIER=1.
TOP_SOURCE_PATCHES=32
MAX_EXAMPLES_PER_LABEL=32


def stable_seed(*parts) -> int:
    v="\x1f".join(str(x) for x in parts)
    return int.from_bytes(hashlib.sha256(v.encode()).digest()[:8],
                          "big")%(2**63-1)


@dataclass(frozen=True)
class Evidence:
    embedding: torch.Tensor   # [n,d], CPU float32 normalized
    coordinate: torch.Tensor  # [n,2], CPU float32
    camera: torch.Tensor      # [n], CPU int64
    weight: torch.Tensor      # [n], CPU float32 nonnegative
    source: tuple[str,...]    # unique human-labeled Q-A images, NOT Probe

    def __post_init__(self):
        n=len(self.embedding)
        if (self.embedding.ndim!=2 or
            self.coordinate.shape!=(n,2) or self.camera.shape!=(n,) or
            self.weight.shape!=(n,) or len(self.source)!=n):
            raise NativeParityError("Invalid D4 feedback evidence shapes")
        if not (torch.isfinite(self.embedding).all() and
                torch.isfinite(self.coordinate).all() and
                torch.isfinite(self.weight).all()):
            raise NativeParityError("Nonfinite D4 feedback evidence")
        if bool((self.weight<0).any()):
            raise NativeParityError("Negative feedback credit not allowed")


def make_evidence(chunks:Sequence[tuple[torch.Tensor,torch.Tensor,int,
                                        torch.Tensor,str]],
                  feature_dim:int) -> Evidence:
    if not chunks:
        return Evidence(torch.empty(0,feature_dim),torch.empty(0,2),
                        torch.empty(0,dtype=torch.long),torch.empty(0),
                        ())
    e=torch.cat([x[0].detach().cpu().float() for x in chunks],dim=0)
    uv=torch.cat([x[1].detach().cpu().float() for x in chunks],dim=0)
    cams=torch.cat([torch.full((len(x[0]),),int(x[2]),dtype=torch.long)
                    for x in chunks])
    weights=torch.cat([x[3].detach().cpu().float() for x in chunks])
    ids=tuple(j for x in chunks for j in (x[4],)*len(x[0]))
    e=torch.nn.functional.normalize(e,dim=1)
    return Evidence(e,uv,cams,weights,ids)


@torch.no_grad()
def credit(query:torch.Tensor,coordinates:torch.Tensor,query_camera:int,
           evidence:Evidence) -> torch.Tensor:
    """Weighted prior-shrunk empirical influence in DINO feature+context."""
    n=len(query)
    if len(evidence.embedding)==0:
        return torch.zeros(n,device=query.device,dtype=torch.float32)
    z=torch.nn.functional.normalize(query.float(),dim=1)
    x=evidence.embedding.to(device=z.device)
    uv=coordinates.to(device=z.device)
    ev=evidence.coordinate.to(device=z.device)
    # Every permission method gets the same frozen DINO feature and (u,v).
    cosine=(z@x.T).clamp(-1.,1.)
    location=(uv[:,None,:]-ev[None,:,:]).square().sum(-1)
    kernel=torch.exp(-(1.-cosine)/BANDWIDTH -
                     location/(2.*SPATIAL_BANDWIDTH**2))
    camera=(evidence.camera.to(z.device)==query_camera)
    kernel*=torch.where(camera,1.,.5)[None,:]
    # Prior count=1 protects against a single high-weight near match.
    numerator=kernel@evidence.weight.to(z.device)
    return numerator/(kernel.sum(dim=1)+1.)


@torch.no_grad()
def shuffled_risk(risk:Evidence,replicate:int) -> Evidence:
    """Destroy risk attribution to feature/context while preserving marginals."""
    n=len(risk.embedding)
    if n<=1:
        return risk
    generator=torch.Generator(device="cpu")
    generator.manual_seed(stable_seed("D4", "shuffled_defect",replicate))
    perm=torch.randperm(n,generator=generator)
    # Original evidential weights/defect labels unchanged, only the
    # feature/space/camera association for each evidence weight is scrambled.
    return Evidence(risk.embedding[perm].clone(),
                    risk.coordinate[perm].clone(),
                    risk.camera[perm].clone(),
                    risk.weight.clone(),risk.source)


@torch.no_grad()
def native_winners(impl,bank,features:Sequence[torch.Tensor]):
    """Exact native Global per-patch fixed-vs-dynamic mean-top5 winner set."""
    if len(features)!=1:
        raise NativeParityError("D4 needs one concat feature layer")
    q=impl.l2_normalize(features[0].to(bank.device))
    fixed=bank.normal_fixed["p0"][0]["id"][0]
    dyn=bank.normal_dyn["p0"][0]["id"][0]
    if dyn.size()<5 or bank.knn_k_normal!=5:
        raise NativeParityError("D4 requires native dynamic 5NN")
    df=fixed.nearest(q,relax=True,k=5)[0]
    dd=dyn.nearest(q,relax=True,k=5)[0]
    if len(df)!=len(dd):
        raise NativeParityError("Native fixed/dynamic patch mismatch")
    winners=(dd<df)
    # Spatial baseline uses TOP5 winning source prototype coordinates.
    _,top5=dyn.nearest(q,relax=True,k=5)
    if top5.ndim!=2 or top5.shape!=(len(q),5):
        raise NativeParityError("Native top5 source-index shape mismatch")
    return q.detach(),winners.detach(),top5.detach()


def select_indices(candidate:torch.Tensor,priority:torch.Tensor, *,
                   rho:float=RHO, method:str, image_id:str,
                   replicate:int=0) -> tuple[torch.Tensor,int]:
    """Exact same K across policies, no current/probe label needed."""
    if method not in METHODS or not image_id:
        raise NativeParityError("Unknown D4 method or missing image identity")
    if candidate.ndim!=1 or priority.shape!=candidate.shape:
        raise NativeParityError("D4 winner/priorities shape mismatch")
    if not 0<rho<=1:
        raise NativeParityError("Invalid equal-win budget")
    if not torch.isfinite(priority).all():
        raise NativeParityError("Nonfinite rights priority")
    ids=torch.nonzero(candidate.cpu(),as_tuple=False).reshape(-1)
    k=min(len(ids),max(1,math.ceil(len(candidate)*rho)))
    selected=torch.zeros(len(candidate),dtype=torch.bool)
    if k==0:
        return selected,0
    if method=="random":
        gen=torch.Generator(device="cpu")
        gen.manual_seed(stable_seed("D4",method,replicate,image_id))
        order=torch.randperm(len(ids),generator=gen)
        selected[ids[order[:k]]]=True
    else:
        # Priority rank stable across hardware; native scores are only
        # selection features, never current ground-truth labels.
        p=priority.detach().cpu().tolist()
        ranked=sorted(ids.tolist(),key=lambda i:(-p[i],i))
        selected[ranked[:k]]=True
    if int(selected.sum())!=k or bool((selected & ~candidate.cpu()).any()):
        raise NativeParityError("Equal-win selection cardinality violated")
    return selected,k


def spatial_priority(top5:torch.Tensor,uv:torch.Tensor,
                     prototype_u_v:torch.Tensor)->torch.Tensor:
    if len(top5)!=len(uv) or top5.ndim!=2 or top5.shape[1]!=5:
        raise NativeParityError("D4 spatial source top5 mismatch")
    source=prototype_u_v.to(top5.device)[top5.long()]
    return -(source-uv.to(top5.device)[:,None,:]).abs().amax(dim=2).mean(dim=1)


def action_priority(method:str,query:torch.Tensor,uv:torch.Tensor,
                    camera:int,normal:Evidence,risk:Evidence, *,
                    top5:torch.Tensor,proto_uv:torch.Tensor,
                    replicate:int=0)->torch.Tensor:
    if method=="spatial" or method=="random":
        return spatial_priority(top5,uv,proto_uv)
    b=credit(query,uv,camera,normal)
    if method=="normal_feedback":
        return b
    r=shuffled_risk(risk,replicate) if method=="shuffled_defect_feedback" else risk
    penalty=credit(query,uv,camera,r)
    return b-RISK_MULTIPLIER*penalty


class ExactWinMask:
    def __init__(self):
        self.image_id=None
        self.allow=None

    def set(self,image_id:str,selection:torch.Tensor):
        if not image_id or selection.ndim!=1 or selection.dtype!=torch.bool:
            raise NativeParityError("Invalid D4 exact-win mask request")
        self.image_id=image_id
        self.allow=selection.detach().clone().cpu()

    def __call__(self,provenance,policy,*,query_camera,
                 query_coords,image_id):
        if (policy!="spatial_0125" or
            image_id!=self.image_id or
            self.allow is None or len(self.allow)!=len(query_coords)):
            raise NativeParityError("D4 mask applied to wrong image/context")
        return self.allow.to(query_coords.device)[:,None].expand(
            len(query_coords),len(provenance))
