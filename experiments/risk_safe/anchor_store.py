"""D1-C causal verified-label normal/defect anchor ledger.

Defect image labels are NOT pixel-mask supervision; defect anchors retain only
the highest model-anomaly-scored patches (uncertain proxies for true defects).
Trusted image IDs are never given to the pre-query policy.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple
import torch


class AnchorProtocolError(RuntimeError):
    pass


@dataclass(frozen=True)
class VerifiedImage:
    query_id: str
    image_id: str
    sample_key: str
    camera_id: int
    label: str  # "normal" or "defect", independently verified for THIS image only
    query_decision_step: int
    label_revealed_step: int
    queried: bool = True
    query_propensity: Optional[float] = None

    def __post_init__(self) -> None:
        if not self.query_id or not self.image_id or not self.sample_key:
            raise AnchorProtocolError("Missing verified identity")
        if self.label not in ("normal", "defect") or not self.queried:
            raise AnchorProtocolError("Only an explicitly queried verified image can enter anchors")
        if self.query_decision_step < 1 or self.label_revealed_step < self.query_decision_step:
            raise AnchorProtocolError("Label revealed before query decision")
        if self.camera_id not in (1, 2, 3, 4, 5):
            raise AnchorProtocolError("Invalid camera")
        if self.query_propensity is not None and not (0 < self.query_propensity <= 1):
            raise AnchorProtocolError("Invalid query propensity")


@dataclass(frozen=True)
class Anchor:
    event: VerifiedImage
    features: Tuple[torch.Tensor, ...]
    selected_patch_indices: Tuple[Tuple[int, ...], ...]
    effective_step: int
    weight: float = 1.0

    def __post_init__(self) -> None:
        if self.effective_step <= self.event.label_revealed_step:
            raise AnchorProtocolError("Anchor must become effective on a later step")


class AnchorLedger:
    def __init__(self, max_normal_images: int = 64,
                 max_defect_images: int = 32,
                 patches_per_image: int = 8):
        if min(max_normal_images, max_defect_images, patches_per_image) < 1:
            raise AnchorProtocolError("Anchor capacities must be positive")
        self.max_normal_images = int(max_normal_images)
        self.max_defect_images = int(max_defect_images)
        self.patches_per_image = int(patches_per_image)
        self._normal: list[Anchor] = []
        self._defect: list[Anchor] = []
        self._query_ids: set[str] = set()
        self._image_ids: set[str] = set()
        self._last_step = 0

    def record(self, event: VerifiedImage,
               patch_layers: Sequence[torch.Tensor],
               anomaly_patch_scores: Optional[Sequence[torch.Tensor]] = None,
               *, weight: float = 1.0) -> Anchor:
        """Record one acquired image label; never propagate to sister cameras.

        The pre-query anomaly patch score determines which defect patches are
        retained. It is NOT a human pixel mask, so anchor uncertainty persists.
        """
        if event.query_id in self._query_ids or event.image_id in self._image_ids:
            raise AnchorProtocolError("Duplicate feedback or duplicate image label")
        if event.label_revealed_step < self._last_step:
            raise AnchorProtocolError("Out-of-order feedback")
        if not patch_layers or not (weight > 0 and weight < float("inf")):
            raise AnchorProtocolError("Missing features or invalid anchor weight")
        if event.label == "defect" and anomaly_patch_scores is None:
            raise AnchorProtocolError("Defect anchors require pre-label model patch scores")
        if anomaly_patch_scores is not None and len(anomaly_patch_scores) != len(patch_layers):
            raise AnchorProtocolError("Patch scores/layer mismatch")

        features = []
        indices = []
        for layer, tensor in enumerate(patch_layers):
            if tensor.ndim != 2 or tensor.shape[0] == 0 or not torch.isfinite(tensor).all().item():
                raise AnchorProtocolError("Invalid/nonfinite patch features")
            k = min(len(tensor), self.patches_per_image)
            if event.label == "defect":
                score = anomaly_patch_scores[layer]
                if score.ndim != 1 or len(score) != len(tensor) or not torch.isfinite(score).all().item():
                    raise AnchorProtocolError("Invalid model anomaly patch scores")
                # Stable tie-break by patch index: no defect ground-truth masks.
                ranked = sorted(range(len(tensor)), key=lambda j: (-float(score[j]), j))
                keep = tuple(ranked[:k])
            else:
                keep = tuple(range(k))
            indices.append(keep)
            features.append(tensor[list(keep)].detach().clone().contiguous())

        anchor = Anchor(event=event, features=tuple(features),
                        selected_patch_indices=tuple(indices),
                        effective_step=event.label_revealed_step + 1,
                        weight=float(weight))
        target = self._normal if event.label == "normal" else self._defect
        target.append(anchor)
        limit = self.max_normal_images if event.label == "normal" else self.max_defect_images
        if len(target) > limit:
            del target[:len(target) - limit]
        self._query_ids.add(event.query_id)
        self._image_ids.add(event.image_id)
        self._last_step = event.label_revealed_step
        return anchor

    def prior(self, step: int) -> tuple[Tuple[Anchor, ...], Tuple[Anchor, ...]]:
        """Return only anchors available strictly before the candidate step."""
        if step < 1:
            raise AnchorProtocolError("Invalid online step")
        normal = tuple(a for a in self._normal if a.effective_step <= step
                       and a.event.label_revealed_step < step)
        defect = tuple(a for a in self._defect if a.effective_step <= step
                       and a.event.label_revealed_step < step)
        return normal, defect

    @property
    def counts(self) -> dict:
        return {"verified_normal_anchor_images": len(self._normal),
                "verified_defect_anchor_images": len(self._defect),
                "verified_query_events": len(self._query_ids)}
