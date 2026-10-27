"""Read-only conditional retrieval for unmodified v22 PatchVMB scorers.

This is a diagnostic adapter, NOT a learned policy. Static codebooks remain
globally eligible; only DYNAMIC STM prototypes receive provenance eligibility.
The adapter replaces the per-layer nearest-distance hook ON A PRIVATE CLONE,
so v22 score_single_view, score_sample_multiview, topk, fusion and aggregation
remain canonical. Stage-1 requires append-only dynamic banks (no replacements).

Critical mathematical point: native nearest(..., relax=True,k=5) computes
MEAN of the top-min(5,eligible) cosine distances, NOT one nearest neighbour.
Restricted groups smaller than five therefore have different k; this is
explicitly logged and must not be mistaken for learned influence.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Mapping, Sequence
from types import MethodType
import math
import torch

from .native_bridge import NativeParityError

POLICIES = ("global", "camera", "spatial_0125", "spatial_025",
            "camera_spatial_0125", "camera_spatial_025")
RADII = {"spatial_0125": .125, "spatial_025": .25,
         "camera_spatial_0125": .125, "camera_spatial_025": .25}


def grid_hw(impl: Any, n: int, image_hw: tuple[int, int]) -> tuple[int, int]:
    h, w = impl._grid_hw_from_npatches(n, aspect=image_hw[0] / image_hw[1])
    if h * w != n:
        raise NativeParityError("Patch grid and native spatial dimensions mismatch")
    return int(h), int(w)


def coordinates(impl: Any, n: int, image_hw: tuple[int, int]) -> torch.Tensor:
    h, w = grid_hw(impl, n, image_hw)
    idx = torch.arange(n, dtype=torch.long)
    return torch.stack(((idx // w + .5) / h, (idx % w + .5) / w), dim=1)


def assert_provenance(provenance: Sequence[Mapping],
                      expected_dynamic_size: int) -> None:
    if len(provenance) != expected_dynamic_size:
        raise NativeParityError(
            f"Dynamic provenance rows {len(provenance)} != {expected_dynamic_size}")
    for i, p in enumerate(provenance):
        if int(p["prototype_index"]) != i:
            raise NativeParityError("Dynamic prototype provenance index mismatch")
        if int(p["camera_id"]) not in range(1, 6):
            raise NativeParityError("Invalid physical Real-IAD camera ID")
        if not (0 < float(p["u"]) < 1 and 0 < float(p["v"]) < 1):
            raise NativeParityError("Invalid dynamic prototype patch coordinates")


def eligible_mask(
        provenance: Sequence[Mapping], policy: str, *,
        query_camera: int, query_coords: torch.Tensor) -> torch.Tensor:
    if policy not in POLICIES:
        raise NativeParityError(f"Unknown preregistered retrieval policy {policy}")
    if query_camera not in range(1, 6):
        raise NativeParityError("Unknown physical camera identity")
    n = len(query_coords)
    m = len(provenance)
    if not torch.isfinite(query_coords).all() or query_coords.shape != (n, 2):
        raise NativeParityError("Malformed query patch coordinates")
    out = torch.ones((n, m), dtype=torch.bool, device=query_coords.device)
    if "camera" in policy:
        cam = torch.tensor([int(p["camera_id"]) for p in provenance],
                           device=query_coords.device)
        out &= cam.reshape(1, -1) == query_camera
    if policy in RADII:
        pv = torch.tensor([[float(p["u"]), float(p["v"])] for p in provenance],
                          device=query_coords.device, dtype=torch.float32)
        dist = (query_coords[:, None, :] - pv[None, :, :]).abs().amax(dim=2)
        out &= dist <= RADII[policy]
    return out


@torch.no_grad()
def conditional_dynamic_distance(
        impl: Any, codebook: Any, q: torch.Tensor,
        mask: torch.Tensor, *, k: int) -> torch.Tensor:
    """Returns [N] or +inf if an individual patch has NO eligible dynamic refs."""
    if mask.shape != (len(q), codebook.size()):
        raise NativeParityError("Eligibility shape does not match native codebook")
    if codebook.size() == 0:
        return torch.full((len(q),), float("inf"), device=q.device)
    if bool(mask.all()):
        return codebook.nearest(q, relax=True, k=k)[0]
    # Reproduce native fp32 cosine path but use mask with per-query valid k.
    q32 = impl.l2_normalize(q).float()
    p32 = impl.l2_normalize(codebook.protos.float())
    similarity = q32 @ p32.T
    allowed = mask.to(q.device)
    top = min(int(k), codebook.size())
    if top <= 0:
        raise NativeParityError("Invalid native nearest k")
    values = similarity.masked_fill(~allowed, float("-inf")).topk(
        k=top, dim=1).values
    counts = allowed.sum(dim=1).clamp(max=top)
    safe = torch.where(torch.isfinite(values), 1.0 - values,
                       torch.zeros_like(values))
    out = safe.sum(dim=1) / counts.clamp(min=1).to(safe.dtype)
    return torch.where(counts > 0, out,
                       torch.full_like(out, float("inf")))


class ScopedRetrieval:
    """One physical camera and one image's patch grid at a time.

    attach() modifies ONLY a fresh private clone's method. No changes to
    frozen code, global class, the support bank, or prototype values.
    """
    def __init__(self, impl: Any, native_bank: Any,
                 provenance: Sequence[Mapping], policy: str):
        if policy not in POLICIES:
            raise NativeParityError("Invalid policy")
        self.impl = impl
        self.bank = native_bank
        self.provenance = tuple(provenance)
        self.policy = policy
        self.camera: int | None = None
        self.image_hw: tuple[int, int] | None = None
        self._original = native_bank._dN_min_fixed_dyn
        if len(native_bank.normal_dyn["p0"][0]["id"]) != 1:
            raise NativeParityError("Pilot requires one concat layer")
        cb = native_bank.normal_dyn["p0"][0]["id"][0]
        assert_provenance(self.provenance, cb.size())
        self.attach()

    def set_query(self, camera_id: int, image_hw: tuple[int, int]):
        if int(camera_id) not in range(1, 6):
            raise NativeParityError("Camera must refer to physical C1-C5")
        if not image_hw[0] or not image_hw[1]:
            raise NativeParityError("Missing image dimensions")
        self.camera = int(camera_id)
        self.image_hw = (int(image_hw[0]), int(image_hw[1]))

    def attach(self):
        def override(native_self, cls_name, mode, view, layer, q):
            if self.camera is None or self.image_hw is None:
                raise NativeParityError("Scoped scorer requires explicit query context")
            if (cls_name, mode, view, layer) != ("p0", 0, "id", 0):
                raise NativeParityError("Unsupported scoring route in controlled pilot")
            if self.policy == "global":
                return self._original(cls_name, mode, view, layer, q)
            cbf = native_self.normal_fixed[cls_name][mode][view][layer]
            cbs = native_self.normal_dyn[cls_name][mode][view][layer]
            fixed = cbf.nearest(q, relax=True, k=native_self.knn_k_normal)[0]
            if not cbs.size():
                return fixed, torch.full((len(q),), -1, device=q.device,
                                         dtype=torch.long)
            uv = coordinates(self.impl, len(q), self.image_hw).to(q.device)
            mask = eligible_mask(self.provenance, self.policy,
                                 query_camera=self.camera,
                                 query_coords=uv)
            dyn = conditional_dynamic_distance(
                self.impl, cbs, q, mask, k=native_self.knn_k_normal)
            return torch.minimum(fixed, dyn), torch.full(
                (len(q),), -1, device=q.device, dtype=torch.long)
        self.bank._dN_min_fixed_dyn = MethodType(override, self.bank)

    @torch.no_grad()
    def score(self, image: Any, features: Sequence[torch.Tensor],
              cfg: Any, *, camera_id: int) -> tuple[float, dict]:
        from .native_bridge import native_score
        self.set_query(camera_id, (image.height, image.width))
        return native_score(self.impl, self.bank, image, features, cfg)

    @torch.no_grad()
    def patch_map(self, image: Any, features: Sequence[torch.Tensor],
                  cfg: Any, *, camera_id: int) -> torch.Tensor:
        self.set_query(camera_id, (image.height, image.width))
        maps, _ = self.bank.score_single_view(
            "p0", list(features), mode=0, view="id",
            topk_patches=cfg.topk_patches, mode_scoring="boost",
            lam=.8, tau_close=.20, gamma=1.)
        if len(maps) != 1:
            raise NativeParityError("Pilot requires one concat feature layer")
        return maps[0].detach()


@torch.no_grad()
def patch_collision_rows(
        impl, image, features, cfg, *,
        global_scorer: ScopedRetrieval,
        static_scorer: ScopedRetrieval,
        provenance: Sequence[Mapping], query_camera: int,
        label: int, image_id: str, source: str,
        query_index: int) -> list[dict]:
    """Offline full-image/patch comparison, never a gate input.

    'winning_prototype' is only TOP-1 diagnostic for the native mean-top5
    distance; the native winner is identified using native mean-top5 distance.
    """
    if not global_scorer.provenance == tuple(provenance):
        raise NativeParityError("Collision provenance mismatch")
    n = features[0].shape[0]
    q = impl.l2_normalize(features[0].to(global_scorer.bank.device))
    fixed = global_scorer.bank.normal_fixed["p0"][0]["id"][0]
    dyn = global_scorer.bank.normal_dyn["p0"][0]["id"][0]
    fdist = fixed.nearest(q, relax=True, k=global_scorer.bank.knn_k_normal)[0]
    if not dyn.size():
        return []
    ddist = dyn.nearest(q, relax=True, k=global_scorer.bank.knn_k_normal)[0]
    top1 = dyn.nearest(q, relax=False)[1]
    ga = global_scorer.patch_map(image, features, cfg, camera_id=query_camera)
    sa = static_scorer.patch_map(image, features, cfg, camera_id=query_camera)
    uv = coordinates(impl, n, (image.height, image.width))
    harm = (sa - ga).clamp_min(0).detach().cpu()
    winners = (ddist < fdist).detach().cpu()
    idxs = top1.detach().cpu()
    result = []
    for j in torch.where(winners & (harm > 0))[0].tolist():
        proto = provenance[int(idxs[j].item())]
        distance = max(abs(float(uv[j, 0]) - float(proto["u"])),
                       abs(float(uv[j, 1]) - float(proto["v"])))
        result.append({
            "source": source, "image": image_id,
            "query_index": query_index, "label_offline_only": label,
            "query_camera": query_camera, "patch_index": j,
            "u": float(uv[j, 0]), "v": float(uv[j, 1]),
            "global_patch_score": float(ga[j]), "static_patch_score": float(sa[j]),
            "score_erosion": float(harm[j]),
            "fixed_top5_distance": float(fdist[j]),
            "dynamic_top5_distance": float(ddist[j]),
            "winning_dynamic_top1_index_diagnostic": int(idxs[j]),
            "winning_dynamic_top1_camera_diagnostic": int(proto["camera_id"]),
            "winning_dynamic_top1_spatial_linf_diagnostic": distance,
            "cross_camera_top1_diagnostic": int(proto["camera_id"]) != query_camera,
        })
    return result
