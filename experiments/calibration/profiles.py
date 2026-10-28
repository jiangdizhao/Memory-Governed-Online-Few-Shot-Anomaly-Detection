"""Immutable, versioned Stage 0 protocol configuration."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from typing import Any


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False)


def digest_json(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class NativeScorerConfig:
    """Values inherited from P0/P1 and the frozen v22 score path."""

    device: str = "cpu"
    augmentation_seed: int = 0
    bank_seed: int = 0
    fixed_capacity: int = 25000
    fixed_per_image: int = 5000
    augmentations: int = 4
    include_identity: bool = False
    multiplier: float = 2.5
    feature_fuse: str = "concat"
    topk_patches: int = 32
    image_aggregation: str = "max"
    image_topk: int = 64
    image_quantile: float = 0.95
    knn: int = 5
    model_name: str = "dinov2_vitl14_reg"
    image_size: int = 392
    resize_size: int = 448
    use_fp16: bool = True
    view_names: tuple[str, ...] = ("id",)
    mode_scoring: str = "boost"
    boost_lambda: float = 0.8
    tau_close: float = 0.20
    gamma: float = 1.0
    layer_fusion: str = "mean"
    view_fusion: str = "mean"
    concept_assign_view: str = "id"
    fixed_support_count: int = 4
    mutable_capacity_per_layer: int = 256

    def __post_init__(self) -> None:
        locked = {
            "device": "cpu", "augmentation_seed": 0, "bank_seed": 0,
            "fixed_capacity": 25000, "fixed_per_image": 5000,
            "augmentations": 4, "multiplier": 2.5,
            "feature_fuse": "concat", "topk_patches": 32,
            "image_aggregation": "max", "image_topk": 64,
            "image_quantile": 0.95, "knn": 5,
            "model_name": "dinov2_vitl14_reg", "image_size": 392,
            "resize_size": 448, "use_fp16": True,
            "view_names": ("id",), "mode_scoring": "boost",
            "boost_lambda": 0.8, "tau_close": 0.20, "gamma": 1.0,
            "layer_fusion": "mean", "view_fusion": "mean",
            "concept_assign_view": "id", "fixed_support_count": 4,
            "mutable_capacity_per_layer": 256,
        }
        changed = {key: (getattr(self, key), expected)
                   for key, expected in locked.items()
                   if getattr(self, key) != expected}
        if changed:
            raise ValueError(f"Native Stage 0 scorer profile is locked: {changed}")
        if self.fixed_support_count != 4:
            raise ValueError("Stage 0 protocol fixes exactly four deployed supports")
        if self.mutable_capacity_per_layer != 256:
            raise ValueError("Stage 0 memory capacity must be K=256 per layer")
        if self.image_aggregation != "max" or self.knn != 5:
            raise ValueError("Native scorer settings differ from the locked P0 profile")
        if self.feature_fuse != "concat" or self.topk_patches != 32:
            raise ValueError("Native feature fusion/top-k settings changed")
        if self.include_identity:
            raise ValueError("Support augmentation excludes the identity reference")

    @property
    def sha256(self) -> str:
        return digest_json(asdict(self))


@dataclass(frozen=True)
class QueryProtocolConfig:
    version: str = "Q-A-TS-v1"
    bins: int = 16
    fraction: float = 0.03
    early_observations: int = 400
    q_b_early_fraction: float = 0.20
    seed: int = 37
    rounding: str = "half_up_floor_0.03N_plus_0.5"
    group_unit: str = "official_sample_key"

    def __post_init__(self) -> None:
        if (self.version != "Q-A-TS-v1" or self.bins != 16 or
                self.fraction != 0.03 or self.early_observations != 400 or
                self.q_b_early_fraction != 0.20 or self.seed != 37 or
                self.group_unit != "official_sample_key"):
            raise ValueError("Q-A-TS/Q-B protocol fields are locked")
        if self.rounding != "half_up_floor_0.03N_plus_0.5":
            raise ValueError("Unregistered query-budget rounding rule")

    @property
    def sha256(self) -> str:
        return digest_json(asdict(self))


@dataclass(frozen=True)
class CalibrationIsolatedProfile:
    name: str = "calibration_isolated"
    stage: int = 0
    support_regime: str = "strict4_support_plus_additional_C64"
    c64_size: int = 64
    c64_mutable: bool = False
    memory_capacity_per_layer: int = 256
    reference_theta_policy: str = "fixed_theta0_higher_q90_at_M0"
    reference_sigma_floor: float = 0.02
    scorer: NativeScorerConfig = NativeScorerConfig()
    query: QueryProtocolConfig = QueryProtocolConfig()

    def __post_init__(self) -> None:
        if self.stage != 0 or self.name != "calibration_isolated":
            raise ValueError("Only Stage 0 calibration_isolated execution is enabled")
        if self.support_regime != "strict4_support_plus_additional_C64":
            raise ValueError("C64 must remain explicitly additional to four supports")
        if self.c64_size != 64 or self.c64_mutable:
            raise ValueError("C64 is 64 protected, read-only normal anchors")
        if self.memory_capacity_per_layer != 256:
            raise ValueError("The mutable memory capacity is locked at K=256")
        if self.reference_theta_policy != "fixed_theta0_higher_q90_at_M0":
            raise ValueError("Reference threshold policy is immutable")
        if self.reference_sigma_floor != 0.02:
            raise ValueError("Reference sigma floor is immutable")

    @property
    def sha256(self) -> str:
        return digest_json(asdict(self))


DEFAULT_PROFILE = CalibrationIsolatedProfile()
