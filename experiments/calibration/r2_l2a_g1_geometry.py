"""R2-L2A exact streaming G1 geometry reference (NumPy, CPU only).

The public API accepts in-memory, provenance-validated descriptor tables. It
does not load models, images, dataset metadata, or feature-cache files.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
import re
from typing import Any, Mapping, Sequence

import numpy as np


G1_ENGINE_VERSION = "r2_l2a.streaming_exact.v1"
G1A_SAMPLING_POLICY = "r1_grid28x28_cell_centers_7x7_v1"
G1B_SELECTION_RULE = "r1_native_c0_top32_indices_v1"
G1B_BASELINE_MEMORY_CONTRACT = "R1_M0_FIXED_SUPPORT_PLUS_EMPTY_GREEDY_V1"
R1_FIXED_SUPPORT_SHA256 = "8a710d028509350568e976bc3ede66aa59603499cbea525bdd8579322b1a1558"
R1_FIXED_SUPPORT_DIGEST_ALGORITHM = (
    "experiments.calibration.native._fixed_bank_digest.v1:path-shape-dtype-bytes")
K_NEIGHBORS = 5
DEFAULT_RADIUS_GRID_POINTS = 101
DEFAULT_QUERY_BLOCK_SIZE = 64
DEFAULT_REFERENCE_BLOCK_SIZE = 512
DEFAULT_MAX_PAIRWISE_ELEMENTS = 65_536
MAX_RADIUS_GRID_POINTS = 1_001
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class GeometryContractError(ValueError):
    """A feature, cache, role, or execution parameter is incompatible."""


def _sha256(value: str, name: str) -> str:
    value = str(value)
    if not SHA256_RE.fullmatch(value):
        raise GeometryContractError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"),
                         ensure_ascii=False, allow_nan=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True)
class R1DescriptorProfile:
    """R1 receipt-backed identity for selected native patch descriptors."""

    profile_id: str
    source_commit: str
    source_hashes: tuple[tuple[str, str], ...]
    model_variant: str
    checkpoint_file: str
    checkpoint_sha256: str
    hub_source_tree_sha256: str
    loaded_model_state_sha256: str
    hook_layers: tuple[int, ...]
    layer_fusion: str
    feature_concatenation: str
    preprocessing: str
    view: str
    encoder_fingerprint_sha256: str
    preprocess_fingerprint_sha256: str
    native_scorer_profile_sha256: str
    fixed_support_sha256: str
    feature_shape: tuple[int, int]
    feature_dtype: str
    l2_normalized: bool

    @property
    def sha256(self) -> str:
        return _canonical_sha256(asdict(self))


# Values are projected from the R1 run manifest and checked against frozen R1
# and vmb source. No feature arrays or private feedback/anchor ledgers are used.
R1_DESCRIPTOR_PROFILE = R1DescriptorProfile(
    profile_id="r1_switch_s0_512_digest_contract_v1",
    source_commit="c86d5b89fa0062b87d80adb324eaa6bc3c94448e",
    source_hashes=(
        ("experiments/calibration/r1_matched_correction_pilot.py",
         "3208e2c52277ad76351eb86968e362aa315effe9e3c734bfe2b6ff7bab0db448"),
        ("experiments/calibration/native.py",
         "3a347d65aaf4bfd6b41eff2ea7339d878065bb2e08ed64795f57b0458135dd45"),
        ("experiments/p0_protocol_audit.py",
         "b83918ec5aceb3ef63cca942fc9fb0e241559d9672d20bed422b15fcd48135e7"),
        ("vmb_visionad_new.py",
         "bd6545bf5ff63acce01792e5d42aa964c88e73fb59670febeb55f2d1139a409a"),
        ("test_phase3_AIF_simplified_v22.py",
         "bfc871d5a9a32b2f5bdb9d7aab5177e702dec2e1110a6632f20bd13c20fcba30"),
    ),
    model_variant="DINOv2 ViT-L/14 with four register tokens (dinov2_vitl14_reg)",
    checkpoint_file="dinov2_vitl14_reg4_pretrain.pth",
    checkpoint_sha256="36e4deffbaef061a2576705b0c36f93621e2ae20bf6274694821b0b492551b51",
    hub_source_tree_sha256="b7c0037f90671c4932393bd56de8c8a205b29a6eb22dea563b6c8a88c471da15",
    loaded_model_state_sha256="1f26464ba4055403e550f1d5c1333fb012631d98661df6993cabea730d17e3f9",
    hook_layers=(5, 11, 17, 23),
    layer_fusion="concatenate four selected 1024D ViT-L layer outputs in hook order",
    feature_concatenation="concat_then_l2_normalize",
    preprocessing=("PIL RGB; resize 448x448; center crop 392x392; PIL BICUBIC; "
                   "model mean/std normalization"),
    view="id",
    encoder_fingerprint_sha256="859c5270f1b9e82888a29055273718ef35ad92b143c9f6362ec46dd8566eb5c0",
    preprocess_fingerprint_sha256="ac0cf9ae583f0eeb766a39ad4c21049bbb4135b2d001c6dd43d6d35568d04a8d",
    native_scorer_profile_sha256="ba3fa736734c2c6c3789f485f3288c5ae77177c68b6b3e24de7e257e447ced87",
    fixed_support_sha256=R1_FIXED_SUPPORT_SHA256,
    feature_shape=(784, 4096),
    feature_dtype="float32",
    l2_normalized=True,
)


@dataclass(frozen=True)
class FeatureCacheReceipt:
    """Metadata-only cache receipt; it contains no feature tensor or data path."""

    cache_sha256: str
    inventory_sha256: str
    profile_sha256: str
    encoder_fingerprint_sha256: str
    preprocess_fingerprint_sha256: str
    native_scorer_profile_sha256: str
    checkpoint_sha256: str
    hub_source_tree_sha256: str
    feature_shape: tuple[int, int]
    feature_dtype: str
    view: str
    l2_normalized: bool
    role: str
    population: str
    source_group_identity_contract: str
    contains_labels_or_masks: bool = False
    contains_image_paths: bool = False
    contains_private_feedback: bool = False
    contains_correction_anchors: bool = False


def validate_feature_cache_receipt(
    receipt: FeatureCacheReceipt,
    expected: R1DescriptorProfile = R1_DESCRIPTOR_PROFILE,
    *,
    verified_cache_sha256: str | None = None,
    verified_inventory_sha256: str | None = None,
) -> dict[str, Any]:
    """Fail closed unless every representation/provenance field matches R1."""
    _sha256(receipt.cache_sha256, "feature cache digest")
    _sha256(receipt.inventory_sha256, "inventory digest")
    wanted = {
        "profile_sha256": expected.sha256,
        "encoder_fingerprint_sha256": expected.encoder_fingerprint_sha256,
        "preprocess_fingerprint_sha256": expected.preprocess_fingerprint_sha256,
        "native_scorer_profile_sha256": expected.native_scorer_profile_sha256,
        "checkpoint_sha256": expected.checkpoint_sha256,
        "hub_source_tree_sha256": expected.hub_source_tree_sha256,
        "feature_shape": expected.feature_shape,
        "feature_dtype": expected.feature_dtype,
        "view": expected.view,
        "l2_normalized": True,
        "role": "A",
        "population": "authorized_normal_reference_only",
        "source_group_identity_contract": "realid-offical-sample-key-v2",
        "contains_labels_or_masks": False,
        "contains_image_paths": False,
        "contains_private_feedback": False,
        "contains_correction_anchors": False,
    }
    mismatch = {
        field: {"expected": value, "actual": getattr(receipt, field)}
        for field, value in wanted.items() if getattr(receipt, field) != value
    }
    if mismatch:
        raise GeometryContractError(f"incompatible feature cache receipt: {mismatch}")
    if verified_cache_sha256 is None or verified_inventory_sha256 is None:
        return {
            "status": "PENDING_EXTERNAL_CACHE_AND_INVENTORY_RECEIPT_REVIEW",
            "cache_sha256": receipt.cache_sha256,
            "inventory_sha256": receipt.inventory_sha256,
            "profile_sha256": expected.sha256,
        }
    if (_sha256(verified_cache_sha256, "externally verified cache digest")
            != receipt.cache_sha256
            or _sha256(verified_inventory_sha256, "externally verified inventory digest")
            != receipt.inventory_sha256):
        raise GeometryContractError("external cache/inventory review digests do not match receipt")
    return {
        "status": "COMPATIBLE_EXTERNALLY_VERIFIED_METADATA_RECEIPT",
        "cache_sha256": receipt.cache_sha256,
        "inventory_sha256": receipt.inventory_sha256,
        "profile_sha256": expected.sha256,
        "feature_shape": list(expected.feature_shape),
        "feature_dtype": expected.feature_dtype,
        "role": "A",
    }


@dataclass(frozen=True)
class DescriptorBatch:
    """Rows share one category, role, representation, image, and group ledger."""

    category_id: str
    features: np.ndarray
    source_group_sha256: tuple[str, ...]
    image_sha256: tuple[str, ...]
    representation_sha256: str
    role: str = "A"
    access_status: str = "AUTHORIZED_A_IDENTITY_RECEIPT"
    inventory_sha256: str = "0" * 64
    cache_receipt_sha256: str = "0" * 64
    contains_labels_or_masks: bool = False
    contains_private_feedback: bool = False
    contains_correction_anchors: bool = False


def validate_descriptor_batch(batch: DescriptorBatch,
                              expected_dimension: int | None = None) -> np.ndarray:
    if (not isinstance(batch.category_id, str)
            or not re.fullmatch(r"[a-z][a-z0-9_]*", batch.category_id)
            or batch.role != "A"):
        raise GeometryContractError("G1 geometry accepts only an explicit category A role")
    if not isinstance(batch.access_status, str) or not batch.access_status:
        raise GeometryContractError("descriptor batch requires an explicit access status")
    if batch.contains_labels_or_masks or batch.contains_private_feedback \
            or batch.contains_correction_anchors:
        raise GeometryContractError("descriptor input contains a forbidden information class")
    _sha256(batch.representation_sha256, "representation fingerprint")
    _sha256(batch.inventory_sha256, "inventory receipt digest")
    _sha256(batch.cache_receipt_sha256, "cache receipt digest")
    x = np.asarray(batch.features)
    if x.ndim != 2 or x.shape[1] == 0:
        raise GeometryContractError("descriptor features must have shape (n, d), d > 0")
    if expected_dimension is not None and x.shape[1] != expected_dimension:
        raise GeometryContractError(
            f"descriptor dimension mismatch: expected {expected_dimension}, found {x.shape[1]}")
    if x.dtype not in (np.dtype("float32"), np.dtype("float64")):
        raise GeometryContractError("descriptor dtype must be float32 or float64")
    if len(batch.source_group_sha256) != len(x) or len(batch.image_sha256) != len(x):
        raise GeometryContractError("every descriptor requires image and source-group identity")
    for value in batch.source_group_sha256:
        _sha256(value, "source-group digest")
    for value in batch.image_sha256:
        _sha256(value, "image identity digest")
    if not np.isfinite(x).all():
        raise GeometryContractError("descriptor features must be finite")
    if len(x):
        norms = np.linalg.norm(x.astype(np.float64, copy=False), axis=1)
        tolerance = 1e-4 if x.dtype == np.float32 else 1e-8
        if not np.all(np.abs(norms - 1.0) <= tolerance):
            raise GeometryContractError("descriptor vectors must already be L2-normalized")
    return x


def g1a_spatial_indices() -> np.ndarray:
    """49 fixed cell-center patches on R1's 28x28 grid; no score-based selection."""
    coordinates = np.arange(2, 28, 4, dtype=np.int64)
    indices = np.asarray([row * 28 + col for row in coordinates for col in coordinates],
                         dtype=np.int64)
    indices.flags.writeable = False
    return indices


@dataclass(frozen=True)
class Top32SelectionReceipt:
    """C0 native Top-32 indices tied to an explicit pre-score memory state."""

    query_image_sha256: str
    scorer_profile_sha256: str
    fixed_support_sha256: str
    score_state_memory_sha256: str
    source_receipt_sha256: str
    baseline_memory_contract: str
    baseline_memory_certificate_sha256: str
    selection_rule: str
    selected_indices: tuple[int, ...]
    role: str = "A"
    contains_labels_or_masks: bool = False


def validate_top32_selection(receipt: Top32SelectionReceipt) -> None:
    for field in ("query_image_sha256", "scorer_profile_sha256",
                  "fixed_support_sha256", "score_state_memory_sha256", "source_receipt_sha256",
                  "baseline_memory_certificate_sha256"):
        _sha256(getattr(receipt, field), field)
    if receipt.role != "A" or receipt.contains_labels_or_masks:
        raise GeometryContractError("Top-32 selection receipt is not a permitted A-only projection")
    if receipt.selection_rule != G1B_SELECTION_RULE:
        raise GeometryContractError("Top-32 selection rule is not the frozen R1 native C0 rule")
    if receipt.scorer_profile_sha256 != R1_DESCRIPTOR_PROFILE.native_scorer_profile_sha256:
        raise GeometryContractError("Top-32 selection uses an incompatible native scorer profile")
    if receipt.fixed_support_sha256 != R1_FIXED_SUPPORT_SHA256:
        raise GeometryContractError("Top-32 selection uses an incompatible fixed support bank")
    if receipt.baseline_memory_contract != G1B_BASELINE_MEMORY_CONTRACT:
        raise GeometryContractError("Top-32 selection lacks the frozen R1 M0 memory contract")
    if (len(receipt.selected_indices) != 32
            or len(set(receipt.selected_indices)) != 32
            or any(isinstance(i, bool) or not isinstance(i, (int, np.integer))
                   or i < 0 or i >= 784 for i in receipt.selected_indices)):
        raise GeometryContractError("Top-32 receipt must contain 32 unique indices in [0, 783]")


def g1b_identifiability_status(
    receipt: Top32SelectionReceipt | None,
    *,
    verified_receipt_sha256: str | None = None,
    verified_baseline_certificate_sha256: str | None = None,
) -> str:
    if receipt is None:
        return "NOT_YET_IDENTIFIABLE"
    validate_top32_selection(receipt)
    if verified_receipt_sha256 is None or verified_baseline_certificate_sha256 is None:
        return "NOT_YET_IDENTIFIABLE"
    if (_sha256(verified_receipt_sha256, "externally verified Top-32 receipt digest")
            != receipt.source_receipt_sha256
            or _sha256(verified_baseline_certificate_sha256,
                       "externally verified baseline certificate digest")
            != receipt.baseline_memory_certificate_sha256):
        raise GeometryContractError("external Top-32/M0 review digests do not match receipt")
    return "IDENTIFIABLE_FROM_EXTERNALLY_VERIFIED_FROZEN_NATIVE_C0_RECEIPT"


def select_g1b_features(features: np.ndarray,
                        receipt: Top32SelectionReceipt,
                        *,
                        verified_receipt_sha256: str | None = None,
                        verified_baseline_certificate_sha256: str | None = None) -> np.ndarray:
    """Select only indices emitted by the frozen scorer; never rerank locally."""
    validate_top32_selection(receipt)
    if g1b_identifiability_status(
            receipt,
            verified_receipt_sha256=verified_receipt_sha256,
            verified_baseline_certificate_sha256=verified_baseline_certificate_sha256
    ) != "IDENTIFIABLE_FROM_EXTERNALLY_VERIFIED_FROZEN_NATIVE_C0_RECEIPT":
        raise GeometryContractError("G1-B is NOT_YET_IDENTIFIABLE without external receipt review")
    x = np.asarray(features)
    if x.shape != (784, 4096) or x.dtype != np.float32:
        raise GeometryContractError("G1-B input must match R1 float32 shape (784, 4096)")
    if not np.isfinite(x).all():
        raise GeometryContractError("G1-B feature matrix must be finite")
    if not np.all(np.abs(np.linalg.norm(x.astype(np.float64), axis=1) - 1.0) <= 1e-4):
        raise GeometryContractError("G1-B feature rows must be L2-normalized")
    result = x[np.asarray(receipt.selected_indices, dtype=np.int64)].copy()
    result.flags.writeable = False
    return result


def _validate_radius_grid(radii: Sequence[float] | None) -> np.ndarray:
    if radii is None:
        result = np.linspace(0.0, 2.0, DEFAULT_RADIUS_GRID_POINTS, dtype=np.float64)
    else:
        result = np.asarray(radii, dtype=np.float64)
    if (result.ndim != 1 or not 2 <= len(result) <= MAX_RADIUS_GRID_POINTS
            or not np.isfinite(result).all() or result[0] != 0.0 or result[-1] != 2.0
            or np.any(np.diff(result) <= 0.0)):
        raise GeometryContractError("radius grid must be strictly increasing from 0 through 2")
    return result


@dataclass(frozen=True)
class DistanceStats:
    category_id: str
    nearest_distance: np.ndarray
    fifth_nearest_distance: np.ndarray
    nearest_reference_index: np.ndarray
    fifth_reference_index: np.ndarray
    eligible_reference_count: np.ndarray
    local_reference_count_by_radius: np.ndarray
    boundary_equal_count_by_radius: np.ndarray
    radius_grid: np.ndarray
    query_source_groups: tuple[str, ...]
    reference_count: int
    query_image_count: int
    reference_image_count: int
    query_source_group_count: int
    reference_source_group_count: int
    representation_sha256: str
    query_inventory_sha256: str
    reference_inventory_sha256: str
    query_cache_receipt_sha256: str
    reference_cache_receipt_sha256: str
    query_feature_dtype: str
    reference_feature_dtype: str
    feature_dimension: int
    block_pairwise_elements: int
    metadata: Mapping[str, Any]


def compare_source_disjoint(
    query: DescriptorBatch,
    reference: DescriptorBatch,
    *,
    radii: Sequence[float] | None = None,
    query_block_size: int = DEFAULT_QUERY_BLOCK_SIZE,
    reference_block_size: int = DEFAULT_REFERENCE_BLOCK_SIZE,
    max_pairwise_elements: int = DEFAULT_MAX_PAIRWISE_ELEMENTS,
) -> DistanceStats:
    """Compute exact nearest/fifth distances with bounded blockwise matmul.

    Candidate rows sharing a source group or image identity are masked before
    top-k selection. Equal distances are ordered by original reference index.
    Only Q-block by R-block distance arrays are materialized.
    """
    q = validate_descriptor_batch(query)
    r = validate_descriptor_batch(reference)
    if query.category_id != reference.category_id:
        raise GeometryContractError("query and reference must belong to the same category")
    if query.representation_sha256 != reference.representation_sha256:
        raise GeometryContractError("query and reference representation fingerprints differ")
    if q.shape[1] != r.shape[1]:
        raise GeometryContractError("query/reference descriptor dimensions differ")
    for name, value in (("query_block_size", query_block_size),
                        ("reference_block_size", reference_block_size),
                        ("max_pairwise_elements", max_pairwise_elements)):
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value <= 0:
            raise GeometryContractError(f"{name} must be a positive integer")
    block_product = int(query_block_size) * int(reference_block_size)
    if block_product > int(max_pairwise_elements):
        raise GeometryContractError("configured pairwise distance block exceeds its resource cap")
    grid = _validate_radius_grid(radii)
    if len(r) > np.iinfo(np.int32).max:
        raise GeometryContractError("reference count exceeds bounded local-count representation")

    n_query, n_reference, dimension = len(q), len(r), q.shape[1]
    nearest = np.full(n_query, np.nan, dtype=np.float64)
    fifth = np.full(n_query, np.nan, dtype=np.float64)
    nearest_index = np.full(n_query, -1, dtype=np.int64)
    fifth_index = np.full(n_query, -1, dtype=np.int64)
    eligible_count = np.zeros(n_query, dtype=np.int64)
    local_count = np.zeros((n_query, len(grid)), dtype=np.int32)
    boundary_count = np.zeros((n_query, len(grid)), dtype=np.int32)
    best_dist = np.full((n_query, K_NEIGHBORS), np.inf, dtype=np.float64)
    best_index = np.full((n_query, K_NEIGHBORS), -1, dtype=np.int64)

    q_groups = np.asarray(query.source_group_sha256, dtype=object)
    q_images = np.asarray(query.image_sha256, dtype=object)
    r_groups = np.asarray(reference.source_group_sha256, dtype=object)
    r_images = np.asarray(reference.image_sha256, dtype=object)

    for q_start in range(0, n_query, int(query_block_size)):
        q_stop = min(n_query, q_start + int(query_block_size))
        q_block = q[q_start:q_stop].astype(np.float64, copy=False)
        for r_start in range(0, n_reference, int(reference_block_size)):
            r_stop = min(n_reference, r_start + int(reference_block_size))
            r_block = r[r_start:r_stop].astype(np.float64, copy=False)
            similarity = q_block @ r_block.T
            distance = 1.0 - np.clip(similarity, -1.0, 1.0)
            excluded = ((q_groups[q_start:q_stop, None] == r_groups[None, r_start:r_stop])
                        | (q_images[q_start:q_stop, None] == r_images[None, r_start:r_stop]))
            eligible = ~excluded
            eligible_count[q_start:q_stop] += np.sum(eligible, axis=1, dtype=np.int64)
            distance[excluded] = np.inf
            reference_indices = np.arange(r_start, r_stop, dtype=np.int64)

            for local_row, global_row in enumerate(range(q_start, q_stop)):
                row_dist = distance[local_row]
                row_eligible = eligible[local_row]
                candidates = row_dist[row_eligible]
                candidate_indices = reference_indices[row_eligible]
                if len(candidates):
                    previous = best_index[global_row] >= 0
                    merged_dist = np.concatenate((best_dist[global_row, previous], candidates))
                    merged_index = np.concatenate((best_index[global_row, previous],
                                                   candidate_indices))
                    order = np.lexsort((merged_index, merged_dist))[:K_NEIGHBORS]
                    take = len(order)
                    best_dist[global_row] = np.inf
                    best_index[global_row] = -1
                    best_dist[global_row, :take] = merged_dist[order]
                    best_index[global_row, :take] = merged_index[order]
                    sorted_distances = np.sort(candidates)
                    local_count[global_row] += np.searchsorted(
                        sorted_distances, grid, side="left").astype(np.int32)
                    boundary_count[global_row] += (
                        np.searchsorted(sorted_distances, grid, side="right")
                        - np.searchsorted(sorted_distances, grid, side="left")
                    ).astype(np.int32)

    has_one = eligible_count >= 1
    has_five = eligible_count >= K_NEIGHBORS
    nearest[has_one] = best_dist[has_one, 0]
    nearest_index[has_one] = best_index[has_one, 0]
    fifth[has_five] = best_dist[has_five, K_NEIGHBORS - 1]
    fifth_index[has_five] = best_index[has_five, K_NEIGHBORS - 1]

    for array in (nearest, fifth, nearest_index, fifth_index, eligible_count,
                  local_count, boundary_count, grid):
        array.flags.writeable = False
    return DistanceStats(
        category_id=query.category_id,
        nearest_distance=nearest,
        fifth_nearest_distance=fifth,
        nearest_reference_index=nearest_index,
        fifth_reference_index=fifth_index,
        eligible_reference_count=eligible_count,
        local_reference_count_by_radius=local_count,
        boundary_equal_count_by_radius=boundary_count,
        radius_grid=grid,
        query_source_groups=query.source_group_sha256,
        reference_count=n_reference,
        query_image_count=len(set(query.image_sha256)),
        reference_image_count=len(set(reference.image_sha256)),
        query_source_group_count=len(set(query.source_group_sha256)),
        reference_source_group_count=len(set(reference.source_group_sha256)),
        representation_sha256=query.representation_sha256,
        query_inventory_sha256=query.inventory_sha256,
        reference_inventory_sha256=reference.inventory_sha256,
        query_cache_receipt_sha256=query.cache_receipt_sha256,
        reference_cache_receipt_sha256=reference.cache_receipt_sha256,
        query_feature_dtype=str(q.dtype),
        reference_feature_dtype=str(r.dtype),
        feature_dimension=dimension,
        block_pairwise_elements=block_product,
        metadata={
            "engine_version": G1_ENGINE_VERSION,
            "distance": "1 - clip(float64_dot, -1, 1)",
            "neighbor_count": K_NEIGHBORS,
            "source_group_exclusion": "exact_digest_equality_before_topk",
            "self_image_exclusion": "exact_image_digest_equality_before_topk",
            "tie_break": "distance_then_original_reference_row_index",
            "radius_boundary": "strict_distance_less_than_radius",
            "query_block_size": int(query_block_size),
            "reference_block_size": int(reference_block_size),
            "max_pairwise_elements": int(max_pairwise_elements),
            "max_distance_matrix_elements": block_product,
            "dense_query_by_reference_matrix_allocated": False,
        },
    )


def _distance_quantiles(values: np.ndarray) -> dict[str, float | int | None]:
    valid = values[np.isfinite(values)]
    return {
        "valid_count": int(len(valid)),
        "minimum": float(np.min(valid)) if len(valid) else None,
        "q05": float(np.quantile(valid, 0.05)) if len(valid) else None,
        "q25": float(np.quantile(valid, 0.25)) if len(valid) else None,
        "median": float(np.quantile(valid, 0.50)) if len(valid) else None,
        "q75": float(np.quantile(valid, 0.75)) if len(valid) else None,
        "q95": float(np.quantile(valid, 0.95)) if len(valid) else None,
    }


def conditional_query_group_bootstrap(
    stats: DistanceStats,
    *,
    n_resamples: int = 1_000,
    seed: int = 0,
) -> dict[str, Any]:
    """Cluster-bootstrap query groups with the reference set held fixed."""
    if (isinstance(n_resamples, bool) or not isinstance(n_resamples, (int, np.integer))
            or n_resamples < 1):
        raise GeometryContractError("n_resamples must be a positive integer")
    if not isinstance(seed, (int, np.integer)) or isinstance(seed, bool) or seed < 0:
        raise GeometryContractError("bootstrap seed must be a nonnegative integer")
    groups = np.asarray(stats.query_source_groups, dtype=object)
    unique, inverse = np.unique(groups, return_inverse=True)
    if len(unique) < 2:
        return {
            "status": "CI_UNDEFINED_INSUFFICIENT_QUERY_SOURCE_GROUPS",
            "independent_query_source_groups": int(len(unique)),
            "uncertainty_target": "conditional_on_fixed_reference_set",
            "n_resamples": int(n_resamples),
            "coverage_ci": None,
            "five_neighbor_coverage_ci": None,
        }
    n_groups = len(unique)
    sizes = np.bincount(inverse, minlength=n_groups).astype(np.int64)
    nearest_hits = np.zeros((n_groups, len(stats.radius_grid)), dtype=np.int64)
    five_hits = np.zeros_like(nearest_hits)
    nearest_valid = np.isfinite(stats.nearest_distance)
    for group_index in range(n_groups):
        rows = inverse == group_index
        nearest_hits[group_index] = np.sum(
            stats.nearest_distance[rows, None] < stats.radius_grid[None, :], axis=0)
        five_hits[group_index] = np.sum(
            stats.local_reference_count_by_radius[rows] >= K_NEIGHBORS, axis=0)
    rng = np.random.default_rng(int(seed))
    sampled = rng.integers(0, n_groups, size=(int(n_resamples), n_groups))
    denominators = sizes[sampled].sum(axis=1)
    coverage_draws = nearest_hits[sampled].sum(axis=1) / denominators[:, None]
    five_draws = five_hits[sampled].sum(axis=1) / denominators[:, None]
    coverage_ci = np.quantile(coverage_draws, [0.025, 0.975], axis=0)
    five_ci = np.quantile(five_draws, [0.025, 0.975], axis=0)
    return {
        "status": "DESCRIPTIVE_CLUSTER_BOOTSTRAP_CONDITIONAL_ON_REFERENCE",
        "independent_query_source_groups": int(n_groups),
        "uncertainty_target": "conditional_on_fixed_reference_set",
        "reference_group_resampling": False,
        "n_resamples": int(n_resamples),
        "seed": int(seed),
        "coverage_ci": [
            {"radius": float(radius), "lower_95": float(coverage_ci[0, i]),
             "upper_95": float(coverage_ci[1, i])}
            for i, radius in enumerate(stats.radius_grid)
        ],
        "five_neighbor_coverage_ci": [
            {"radius": float(radius), "lower_95": float(five_ci[0, i]),
             "upper_95": float(five_ci[1, i])}
            for i, radius in enumerate(stats.radius_grid)
        ],
        "nearest_valid_query_descriptors": int(np.count_nonzero(nearest_valid)),
        "limitation": "does not include reference-set uncertainty or adaptive selection",
    }


def summarize_category_geometry(
    query: DescriptorBatch,
    reference: DescriptorBatch,
    stats: DistanceStats,
    *,
    bootstrap_resamples: int = 1_000,
    bootstrap_seed: int = 0,
) -> dict[str, Any]:
    if stats.category_id != query.category_id or stats.category_id != reference.category_id:
        raise GeometryContractError("distance result category provenance mismatch")
    q_count = len(stats.nearest_distance)
    valid = np.isfinite(stats.nearest_distance)
    curve: list[dict[str, Any]] = []
    for i, radius in enumerate(stats.radius_grid):
        hits = int(np.count_nonzero(stats.nearest_distance < radius))
        local = stats.local_reference_count_by_radius[:, i]
        curve.append({
            "radius": float(radius),
            "n_nearest_strictly_inside": hits,
            "nearest_distance_lt_radius_fraction": float(hits / q_count) if q_count else None,
            "at_least_five_local_references_fraction":
                float(np.mean(local >= K_NEIGHBORS)) if q_count else None,
            "zero_local_references_fraction": float(np.mean(local == 0)) if q_count else None,
            "local_reference_count_mean": float(np.mean(local)) if q_count else None,
            "local_reference_count_q50": float(np.quantile(local, 0.50)) if q_count else None,
            "local_reference_count_q95": float(np.quantile(local, 0.95)) if q_count else None,
            "support_overlap_multiplicity_mean": float(np.mean(local)) if q_count else None,
            "support_overlap_multiplicity_q95": float(np.quantile(local, 0.95)) if q_count else None,
            "exact_boundary_equality_count": int(
                np.sum(stats.boundary_equal_count_by_radius[:, i], dtype=np.int64)),
            "fifth_neighbor_defined_fraction": float(np.mean(
                stats.eligible_reference_count >= K_NEIGHBORS)) if q_count else None,
        })
    bootstrap = conditional_query_group_bootstrap(
        stats, n_resamples=bootstrap_resamples, seed=bootstrap_seed)
    return {
        "category_id": stats.category_id,
        "data_role": "A",
        "access_status": query.access_status,
        "status": "DESCRIPTIVE_SYNTHETIC_OR_AUTHORIZED_A_ONLY" if q_count else "NO_QUERY_DESCRIPTORS",
        "independent_query_source_group_count": stats.query_source_group_count,
        "independent_reference_source_group_count": stats.reference_source_group_count,
        "query_image_count": stats.query_image_count,
        "reference_image_count": stats.reference_image_count,
        "query_descriptor_count": q_count,
        "reference_descriptor_count": stats.reference_count,
        "eligible_comparison_count": int(np.sum(stats.eligible_reference_count, dtype=np.int64)),
        "excluded_comparison_count": int(q_count * stats.reference_count
                                          - np.sum(stats.eligible_reference_count, dtype=np.int64)),
        "queries_without_eligible_references": int(np.count_nonzero(~valid)),
        "queries_with_fewer_than_five_eligible_references": int(
            np.count_nonzero(stats.eligible_reference_count < K_NEIGHBORS)),
        "nearest_distance_quantiles": _distance_quantiles(stats.nearest_distance),
        "fifth_nearest_distance_quantiles": _distance_quantiles(
            stats.fifth_nearest_distance),
        "coverage_radius_curve": curve,
        "uncertainty": bootstrap,
        "provenance": {
            "representation_sha256": stats.representation_sha256,
            "query_inventory_sha256": stats.query_inventory_sha256,
            "reference_inventory_sha256": stats.reference_inventory_sha256,
            "query_cache_receipt_sha256": stats.query_cache_receipt_sha256,
            "reference_cache_receipt_sha256": stats.reference_cache_receipt_sha256,
            "query_feature_dtype": stats.query_feature_dtype,
            "reference_feature_dtype": stats.reference_feature_dtype,
            "feature_dimension": stats.feature_dimension,
            "engine": dict(stats.metadata),
        },
    }


def summarize_30_category_coverage(
    category_reports: Sequence[Mapping[str, Any]],
    expected_categories: Sequence[str],
    radius: float,
) -> dict[str, Any]:
    """Macro, descriptor-weighted pooled, worst, and lower-tail coverage.

    The all-category denominator is fail-closed: missing/undefined categories
    make all 30-category summaries unavailable instead of silently dropping.
    """
    if (len(expected_categories) != 30 or len(set(expected_categories)) != 30
            or not math.isfinite(radius) or radius < 0.0 or radius > 2.0):
        raise GeometryContractError("summary requires 30 unique categories and radius in [0, 2]")
    by_category = {str(row.get("category_id")): row for row in category_reports}
    if len(by_category) != len(category_reports):
        raise GeometryContractError("duplicate category geometry report")
    missing = sorted(set(expected_categories) - set(by_category))
    extra = sorted(set(by_category) - set(expected_categories))
    if missing or extra:
        return {
            "status": "INCOMPLETE_30_CATEGORY_DENOMINATOR",
            "radius": float(radius),
            "expected_category_count": 30,
            "reported_category_count": len(by_category),
            "missing_category_ids": missing,
            "unexpected_category_ids": extra,
            "category_equal_macro_coverage": None,
            "pooled_coverage": None,
            "worst_category": None,
            "lower_tail": None,
        }
    indexed: list[tuple[str, float, int]] = []
    for category in expected_categories:
        row = by_category[category]
        curve = row.get("coverage_radius_curve", ())
        match = next((point for point in curve if point.get("radius") == radius), None)
        if match is None or row.get("query_descriptor_count", 0) <= 0:
            return {
                "status": "INCOMPLETE_30_CATEGORY_DENOMINATOR",
                "radius": float(radius),
                "expected_category_count": 30,
                "reported_category_count": len(by_category),
                "missing_or_undefined_category_ids": [category],
                "category_equal_macro_coverage": None,
                "pooled_coverage": None,
                "worst_category": None,
                "lower_tail": None,
            }
        indexed.append((category,
                        float(match["nearest_distance_lt_radius_fraction"]),
                        int(row["query_descriptor_count"])))
    values = np.asarray([item[1] for item in indexed], dtype=np.float64)
    weights = np.asarray([item[2] for item in indexed], dtype=np.int64)
    order = sorted(indexed, key=lambda item: (item[1], item[0]))
    lower_n = int(math.ceil(0.10 * len(indexed)))
    return {
        "status": "DESCRIPTIVE_COMPLETE_30_CATEGORY_COVERAGE",
        "radius": float(radius),
        "expected_category_count": 30,
        "reported_category_count": 30,
        "category_equal_macro_coverage": float(np.mean(values)),
        "pooled_coverage": {
            "value": float(np.average(values, weights=weights)),
            "weighting_unit": "query_descriptor",
            "total_query_descriptors": int(np.sum(weights)),
        },
        "worst_category": {"coverage": float(order[0][1]),
                           "category_ids": [item[0] for item in order
                                            if item[1] == order[0][1]]},
        "lower_tail": {
            "definition": "mean coverage of the lowest ceil(0.10*30) categories",
            "category_count": lower_n,
            "category_ids": [item[0] for item in order[:lower_n]],
            "mean_coverage": float(np.mean([item[1] for item in order[:lower_n]])),
        },
        "category_values": [
            {"category_id": category, "coverage": coverage,
             "query_descriptor_count": count}
            for category, coverage, count in indexed
        ],
        "inference_note": "descriptive weighting; descriptors are not independent units",
    }
