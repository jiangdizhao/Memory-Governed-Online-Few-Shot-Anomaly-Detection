"""D1-D native-v22 bank bridge, with fail-closed scorer/update parity.

No replacement scoring formula: production scores always use the unchanged
a0_a1_strict4_matched.score_bank -> v22 score_sample_multiview path.
No modification to immutable static support; shadow proposals use the unchanged
v02_memory.greedy_slot for full-capacity decisions.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Sequence, Tuple
import copy
import math
import torch

from .shadow_memory import GreedyBank, ShadowError


class NativeParityError(RuntimeError):
    pass


def dynamic_codebooks(native_bank: Any) -> Tuple[Any, ...]:
    from v02_memory import codebooks
    layers = tuple(codebooks(native_bank)[0])
    if not layers:
        raise NativeParityError("No dynamic Greedy codebooks")
    return layers


def snapshot(native_bank: Any, *, version: int = 0) -> GreedyBank:
    layers = dynamic_codebooks(native_bank)
    return GreedyBank([cb.protos for cb in layers],
                      [int(cb.K) for cb in layers], version=version)


@torch.no_grad()
def install_on_clone(native_bank: Any, shadow: GreedyBank) -> Any:
    """Clone entire VMB, then install ONLY shadow dynamic prototype tensors.

    Using the full cloned v22 bank preserves its static memory, concept
    prototypes and scoring configuration. The original object is never
    modified by this function.
    """
    clone = copy.deepcopy(native_bank)
    cbs = dynamic_codebooks(clone)
    if len(cbs) != len(shadow.layers):
        raise NativeParityError("Native/shadow layer counts differ")
    for cb, tensor, capacity in zip(cbs, shadow.layers, shadow.capacities):
        if cb.feat_dim != tensor.shape[1] or int(cb.K) != int(capacity):
            raise NativeParityError("Native/shadow feature dimension or cap mismatch")
        if cb.protos.device != tensor.device or cb.protos.dtype != tensor.dtype:
            raise NativeParityError("Native/shadow tensor device or dtype mismatch")
        if len(tensor) > int(cb.K):
            raise NativeParityError("Shadow exceeds native capacity")
        cb.protos = tensor.detach().clone()
    return clone


@torch.no_grad()
def install_on_live(native_bank: Any, new: GreedyBank, *,
                    expected_digest: str, version: int) -> None:
    """Commit gate-approved new dynamic state, never touch static support."""
    current = snapshot(native_bank, version=version)
    if current.sha256() != expected_digest or new.version <= version:
        raise NativeParityError("Stale/invalid live-memory commit")
    layers = dynamic_codebooks(native_bank)
    if len(layers) != len(new.layers):
        raise NativeParityError("Commit dimension mismatch")
    for cb, tensor, cap in zip(layers, new.layers, new.capacities):
        if cb.feat_dim != tensor.shape[1] or int(cb.K) != cap:
            raise NativeParityError("Commit capacity/dimension mismatch")
        if cb.protos.dtype != tensor.dtype or cb.protos.device != tensor.device:
            raise NativeParityError("Commit dtype/device mismatch")
    for cb, tensor in zip(layers, new.layers):
        cb.protos = tensor.detach().clone()


def extract_v22_candidates(
        impl: Any, static_map: Mapping[str, Any],
        full_features: Sequence[torch.Tensor], *,
        keep: int) -> Tuple[Tuple[torch.Tensor, ...], Dict[int, list[int]]]:
    """Same per-layer top-k patch indices as V0.2 native MemoryArm.update."""
    indices = impl.layer_to_topk_from_amap(static_map)
    selected = []
    for i, layer in enumerate(full_features):
        k = list(indices.get(i, []))[:keep]
        if not k or min(k) < 0 or max(k) >= layer.shape[0]:
            raise NativeParityError(f"Invalid/missing native candidate layer {i}")
        selected.append(layer[k].detach().clone())
    return tuple(selected), {int(i): list(v) for i, v in indices.items()}


@torch.no_grad()
def native_score(impl: Any, bank: Any, image: Any,
                 features: Sequence[torch.Tensor], cfg: Any
                 ) -> Tuple[float, Any]:
    from a0_a1_strict4_matched import score_bank
    score, amap = score_bank(impl, bank, image, features, cfg)
    if not math.isfinite(float(score)):
        raise NativeParityError("Nonfinite native anomaly score")
    return float(score), amap


@dataclass
class NativeScoreAdapter:
    """Map full image+features to v22 scoring with a proposed GreedyBank."""
    impl: Any
    live_native_bank: Any
    cfg: Any

    @torch.no_grad()
    def __call__(self, full_image_bundle: Any, shadow: GreedyBank) -> float:
        image, features = full_image_bundle
        shadow_native = install_on_clone(self.live_native_bank, shadow)
        return native_score(self.impl, shadow_native, image, features, self.cfg)[0]


def parity_tolerance(x: float, y: float, *, atol: float = 1e-5,
                     rtol: float = 1e-4) -> float:
    delta = abs(float(x) - float(y))
    if not (math.isfinite(x) and math.isfinite(y)):
        raise NativeParityError("Nonfinite parity values")
    if delta > atol + rtol * max(abs(x), abs(y)):
        raise NativeParityError(
            f"Score parity failed: direct={x:.9g}, shadow={y:.9g}, "
            f"|diff|={delta:.9g}, atol={atol}, rtol={rtol}")
    return delta


@torch.no_grad()
def verify_untouched_parity(impl: Any, bank: Any, img: Any,
                            feat: Sequence[torch.Tensor], cfg: Any,
                            *, atol: float = 1e-5,
                            rtol: float = 1e-4) -> Dict[str, Any]:
    """Exact native score path on live vs untouched deep-copy shadow."""
    before = snapshot(bank)
    digest = before.sha256()
    s1, _ = native_score(impl, bank, img, feat, cfg)
    native_copy = install_on_clone(bank, before)
    s2, _ = native_score(impl, native_copy, img, feat, cfg)
    delta = parity_tolerance(s1, s2, atol=atol, rtol=rtol)
    if snapshot(bank).sha256() != digest:
        raise NativeParityError("Scoring mutated live Greedy bank")
    return {"native_score": s1, "untouched_shadow_score": s2,
            "abs_difference": delta, "max_allowed_abs": atol,
            "relative_tolerance": rtol, "passed": True,
            "native_original_sha256": digest}


@torch.no_grad()
def verify_native_update_parity(
        impl: Any, native_memory_arm: Any,
        features: Sequence[torch.Tensor],
        static_amap: Mapping[str, Any],
        *, score: float, theta: float, sigma: float,
        step: int) -> Dict[str, Any]:
    """Compare ACTUAL v02_memory.MemoryArm.update vs GreedyBank.proposed.

    Mutates only the dedicated parity MemoryArm. Caller must never use this
    diagnostic bank as an online evaluation/admission bank afterward.
    """
    if getattr(native_memory_arm, "name", None) != "greedy_coreset":
        raise NativeParityError("Requires the genuine v02 Greedy memory arm")
    allow, keep, reason, _ = impl.select_informative_normal_write(
        score=float(score), theta=float(theta), sigmaN=float(max(sigma, .02)),
        pred_defect=bool(score >= theta), novelty=0.0)
    if not allow or keep not in (8, 16, 24):
        raise NativeParityError("Unexpected native verified-normal patch selection")
    selected, selected_indices = extract_v22_candidates(
        impl, static_amap, features, keep=keep)
    before = snapshot(native_memory_arm.bank)
    before_digest = before.sha256()
    shadow, audit = before.proposed(selected)
    native_memory_arm.update(
        impl, features, selected_indices, step,
        score=float(score), theta=float(theta), sigma=float(sigma))
    after = snapshot(native_memory_arm.bank,
                     version=shadow.version)
    if len(after.layers) != len(shadow.layers):
        raise NativeParityError("Post-update layer mismatch")
    exact = []
    for i, (actual, predicted) in enumerate(zip(after.layers, shadow.layers)):
        equal = torch.equal(actual, predicted)
        exact.append(bool(equal))
        if not equal:
            absmax = float((actual.float() - predicted.float()).abs().max()) \
                     if actual.shape == predicted.shape else None
            raise NativeParityError(
                f"Layer {i}: genuine V0.2 Greedy != shadow. "
                f"actual={tuple(actual.shape)}, shadow={tuple(predicted.shape)}, "
                f"absmax={absmax}")
    if after.sha256() != shadow.sha256():
        raise NativeParityError("Greedy memory content digest mismatch")
    return {"keep_selected": keep, "native_selection_reason": reason,
            "before_sha256": before_digest,
            "native_after_sha256": after.sha256(),
            "shadow_after_sha256": shadow.sha256(),
            "equal_each_layer": exact, "passed": True,
            "greedy": {"appended": audit.appended,
                       "replaced": audit.replaced,
                       "skipped": audit.skipped},
            "comparison": "full native MemoryArm.update vs GreedyBank.proposed"}
