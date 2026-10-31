"""CPU-only mathematical references for compactly supported local correction.

This module is deliberately disconnected from every experiment runner. Features
must be finite, nonzero, L2-normalized vectors. Anchor provenance is validated
before scoring: an anchor at the current/future position is an error, and an
anchor from the query's source group is excluded.

The ``support_radius`` fields are prospective effective supports. They are not
aliases for ``gaussian_width`` and no values are selected by this module.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import re
from typing import Sequence

import numpy as np


LEGACY_C1_DENOMINATOR_EPSILON = 1e-6
_SOURCE_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_UNIT_NORM_TOLERANCE = 1e-5


def _finite_scalar(value: object, name: str, *, positive: bool = False) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
            value, (int, float, np.integer, np.floating)):
        raise ValueError(f"{name} must be a finite real scalar")
    result = float(value)
    if not math.isfinite(result) or (positive and result <= 0.0):
        qualifier = "finite and positive" if positive else "finite"
        raise ValueError(f"{name} must be {qualifier}")
    return result


def _position(value: object, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    result = int(value)
    if result < 0:
        raise ValueError(f"{name} must be nonnegative")
    return result


def _source_group(value: object, name: str) -> str:
    if not isinstance(value, str) or _SOURCE_SHA256.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256 source-group digest")
    return value


def _as_real_array(value: object, name: str, *, ndim: int | None = None) -> np.ndarray:
    array = np.asarray(value)
    if array.dtype.kind not in "iuf" or (ndim is not None and array.ndim != ndim):
        expectation = f"a real numeric {ndim}-D array" if ndim is not None else "a real numeric array"
        raise ValueError(f"{name} must be {expectation}")
    try:
        result = np.asarray(array, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} cannot be represented as float64") from exc
    if not np.isfinite(result).all():
        raise ValueError(f"{name} must contain only finite values")
    return result


def l2_normalize(features: object) -> np.ndarray:
    """Normalize one vector or a matrix of row vectors, rejecting zero rows.

    Scaled normalization avoids overflow for large finite inputs. Float32 input
    remains float32; other real numeric inputs return float64.
    """
    raw = np.asarray(features)
    if raw.dtype.kind not in "iuf" or raw.ndim not in (1, 2):
        raise ValueError("features must be a real numeric vector or row matrix")
    values = _as_real_array(raw, "features")
    if values.ndim == 1:
        rows = values[None, :]
        was_vector = True
    else:
        rows = values
        was_vector = False
    if rows.shape[1] == 0 or rows.shape[0] == 0:
        raise ValueError("features must contain at least one nonempty row")
    scales = np.max(np.abs(rows), axis=1)
    if np.any(scales == 0.0):
        raise ValueError("zero-norm feature vectors are invalid")
    scaled = rows / scales[:, None]
    norms = np.sqrt(np.sum(scaled * scaled, axis=1))
    normalized = scaled / norms[:, None]
    out_dtype = np.float32 if raw.dtype == np.dtype(np.float32) else np.float64
    normalized = normalized.astype(out_dtype, copy=False)
    return normalized[0].copy() if was_vector else normalized.copy()


def _unit_vector(value: object, name: str) -> np.ndarray:
    vector = _as_real_array(value, name, ndim=1)
    if vector.size == 0:
        raise ValueError(f"{name} must not be empty")
    scale = float(np.max(np.abs(vector)))
    if scale == 0.0:
        raise ValueError(f"{name} must have nonzero norm")
    norm = scale * math.sqrt(float(np.sum((vector / scale) ** 2)))
    if not math.isfinite(norm) or not math.isclose(
            norm, 1.0, rel_tol=0.0, abs_tol=_UNIT_NORM_TOLERANCE):
        raise ValueError(f"{name} must be L2-normalized within {_UNIT_NORM_TOLERANCE:g}")
    return vector


@dataclass(frozen=True)
class CorrectionAnchor:
    """Immutable synthetic anchor with separate legacy and candidate radii.

    ``beta`` is the signed real correction for C1. ``c2_sign`` is the discrete
    sign used by Signed kNN. ``gaussian_width`` reproduces legacy C1; the
    distinct ``support_radius`` is used by C1-Gated/C1-Compact.
    """

    feature: np.ndarray
    beta: float
    kappa: float
    gaussian_width: float
    support_radius: float
    position: int
    source_group: str
    c2_sign: int | None = None

    def __post_init__(self) -> None:
        feature = _unit_vector(self.feature, "anchor feature").copy()
        feature.flags.writeable = False
        beta = _finite_scalar(self.beta, "beta")
        kappa = _finite_scalar(self.kappa, "kappa", positive=True)
        gaussian_width = _finite_scalar(self.gaussian_width, "gaussian_width", positive=True)
        support_radius = _finite_scalar(self.support_radius, "support_radius", positive=True)
        position = _position(self.position, "anchor position")
        source_group = _source_group(self.source_group, "anchor source group")
        c2_sign = self.c2_sign
        if c2_sign is not None:
            if (isinstance(c2_sign, (bool, np.bool_)) or
                    not isinstance(c2_sign, (int, np.integer)) or int(c2_sign) not in (-1, 1)):
                raise ValueError("c2_sign must be -1, +1, or None")
            c2_sign = int(c2_sign)
        object.__setattr__(self, "feature", feature)
        object.__setattr__(self, "beta", beta)
        object.__setattr__(self, "kappa", kappa)
        object.__setattr__(self, "gaussian_width", gaussian_width)
        object.__setattr__(self, "support_radius", support_radius)
        object.__setattr__(self, "position", position)
        object.__setattr__(self, "source_group", source_group)
        object.__setattr__(self, "c2_sign", c2_sign)


def eligible_anchor_indices(
    anchors: Sequence[CorrectionAnchor], *, query_position: int, query_source_group: str
) -> tuple[int, ...]:
    """Return strictly-past, source-disjoint indexes; malformed state fails closed.

    Any current/future anchor is rejected rather than silently ignored. This
    guards the pre-score contract even if that anchor would otherwise be
    source-excluded.
    """
    position = _position(query_position, "query position")
    group = _source_group(query_source_group, "query source group")
    if not isinstance(anchors, Sequence):
        raise ValueError("anchors must be a sequence")
    eligible: list[int] = []
    for index, anchor in enumerate(anchors):
        if not isinstance(anchor, CorrectionAnchor):
            raise ValueError(f"anchor {index} lacks validated provenance")
        # Revalidate mutable object-level invariants at the interface boundary.
        _position(anchor.position, f"anchor {index} position")
        _source_group(anchor.source_group, f"anchor {index} source group")
        _unit_vector(anchor.feature, f"anchor {index} feature")
        _finite_scalar(anchor.beta, f"anchor {index} beta")
        _finite_scalar(anchor.kappa, f"anchor {index} kappa", positive=True)
        _finite_scalar(anchor.gaussian_width, f"anchor {index} gaussian_width", positive=True)
        _finite_scalar(anchor.support_radius, f"anchor {index} support_radius", positive=True)
        if anchor.c2_sign is not None and (
                isinstance(anchor.c2_sign, (bool, np.bool_)) or
                not isinstance(anchor.c2_sign, (int, np.integer)) or int(anchor.c2_sign) not in (-1, 1)):
            raise ValueError(f"anchor {index} c2_sign must be -1, +1, or None")
        if anchor.position >= position:
            raise ValueError(f"anchor {index} is current/future at query position {position}")
        if anchor.source_group != group:
            eligible.append(index)
    return tuple(eligible)


def cosine_distances(query_feature: object, anchor_features: object) -> np.ndarray:
    """Compute ``1 - clip(q dot x, -1, 1)`` for normalized features."""
    query = _unit_vector(query_feature, "query feature")
    anchors = _as_real_array(anchor_features, "anchor features")
    if anchors.ndim != 2 or anchors.shape[0] == 0 or anchors.shape[1] != query.size:
        raise ValueError("anchor features must be a nonempty row matrix matching query dimension")
    for row in range(anchors.shape[0]):
        _unit_vector(anchors[row], f"anchor feature {row}")
    similarities = np.clip(anchors @ query, -1.0, 1.0)
    return 1.0 - similarities


def compact_support_phi(normalized_distance: object) -> np.ndarray | float:
    """Evaluate ``(1-u^2)^2`` for ``0 <= u < 1`` and zero otherwise."""
    values = _as_real_array(normalized_distance, "normalized_distance")
    if np.any(values < 0.0):
        raise ValueError("normalized distances must be nonnegative")
    inside = values < 1.0
    output = np.zeros_like(values, dtype=np.float64)
    u = values[inside]
    output[inside] = (1.0 - u * u) ** 2
    if output.ndim == 0:
        return float(output)
    return output


def _query_and_eligible(
    query_feature: object,
    anchors: Sequence[CorrectionAnchor],
    *,
    query_position: int,
    query_source_group: str,
) -> tuple[np.ndarray, tuple[int, ...], np.ndarray]:
    query = _unit_vector(query_feature, "query feature")
    eligible = eligible_anchor_indices(
        anchors, query_position=query_position, query_source_group=query_source_group)
    if not eligible:
        return query, eligible, np.empty((0,), dtype=np.float64)
    features = np.stack([anchors[index].feature for index in eligible]).astype(np.float64, copy=False)
    if features.shape[1] != query.size:
        raise ValueError("query and anchor feature dimensions differ")
    return query, eligible, cosine_distances(query, features)


def _regularized_weighted_sum(weights: np.ndarray, values: np.ndarray, regularizer: float) -> float:
    """Stable form of ``sum(w*y)/(regularizer+sum(w))`` for nonnegative w."""
    if weights.size == 0:
        return 0.0
    scale = max(regularizer, float(np.max(weights)))
    scaled_weights = weights / scale
    denominator = regularizer / scale + math.fsum(float(v) for v in scaled_weights)
    coefficients = scaled_weights / denominator
    value = math.fsum(float(c) * float(y) for c, y in zip(coefficients, values))
    if not math.isfinite(value):
        raise FloatingPointError("weighted correction was nonfinite")
    return value


def _legacy_gaussian_weights(distances: np.ndarray, anchors: Sequence[CorrectionAnchor],
                             indexes: tuple[int, ...]) -> np.ndarray:
    weights = np.zeros((len(indexes),), dtype=np.float64)
    for row, index in enumerate(indexes):
        distance = float(distances[row])
        width = anchors[index].gaussian_width
        # A ratio above 40 already underflows the Gaussian below useful float64
        # precision. Avoid squaring a potentially overflowing distance/width.
        if distance == 0.0:
            attenuation = 1.0
        elif distance / 40.0 > width:
            attenuation = 0.0
        else:
            ratio = distance / width
            attenuation = math.exp(-0.5 * ratio * ratio)
        weights[row] = anchors[index].kappa * attenuation
    return weights


def legacy_c1_delta(
    query_feature: object,
    anchors: Sequence[CorrectionAnchor],
    *,
    query_position: int,
    query_source_group: str,
) -> float:
    """Frozen normalized Gaussian RBF formula with its ``1e-6`` denominator.

    This deliberately has no compact cutoff. The fixed 1e-6 denominator is
    retained from the specified legacy equation.
    """
    _, indexes, distances = _query_and_eligible(
        query_feature, anchors, query_position=query_position,
        query_source_group=query_source_group)
    if not indexes:
        return 0.0
    weights = _legacy_gaussian_weights(distances, anchors, indexes)
    values = np.asarray([anchors[index].beta for index in indexes], dtype=np.float64)
    result = _regularized_weighted_sum(weights, values, LEGACY_C1_DENOMINATOR_EPSILON)
    return float(np.clip(result, -float(np.max(np.abs(values))),
                         float(np.max(np.abs(values)))))


def legacy_signed_knn_delta(
    query_feature: object,
    anchors: Sequence[CorrectionAnchor],
    *,
    query_position: int,
    query_source_group: str,
    k: int = 5,
    amplitude: float = 0.20,
) -> float:
    """Frozen ungated Signed kNN; ``k_eff=min(k, eligible_count)``.

    Neighbors are ordered by cosine distance and then original anchor index.
    Distance affects retrieval only, not the signed mean, so a distant anchor
    can still supply the full-amplitude correction.
    """
    if isinstance(k, (bool, np.bool_)) or not isinstance(k, (int, np.integer)) or int(k) < 1:
        raise ValueError("k must be a positive integer")
    magnitude = _finite_scalar(amplitude, "amplitude", positive=True)
    _, indexes, distances = _query_and_eligible(
        query_feature, anchors, query_position=query_position,
        query_source_group=query_source_group)
    if not indexes:
        return 0.0
    if any(anchors[index].c2_sign not in (-1, 1) for index in indexes):
        raise ValueError("every eligible Signed kNN anchor needs c2_sign -1 or +1")
    order = sorted(range(len(indexes)), key=lambda row: (float(distances[row]), indexes[row]))
    selected = order[:min(int(k), len(order))]
    mean_sign = math.fsum(float(anchors[indexes[row]].c2_sign) for row in selected) / len(selected)
    return magnitude * mean_sign


def c1_gated_delta(
    query_feature: object,
    anchors: Sequence[CorrectionAnchor],
    *,
    query_position: int,
    query_source_group: str,
) -> float:
    """Legacy C1 multiplied by a maximum compact-support gate.

    The gate restricts where the total correction can be nonzero. Inside that
    union, the legacy Gaussian mixture still includes every eligible anchor.
    """
    _, indexes, distances = _query_and_eligible(
        query_feature, anchors, query_position=query_position,
        query_source_group=query_source_group)
    if not indexes:
        return 0.0
    gate_values = np.zeros((len(indexes),), dtype=np.float64)
    for row, index in enumerate(indexes):
        radius = anchors[index].support_radius
        distance = float(distances[row])
        if distance < radius:
            gate_values[row] = float(compact_support_phi(distance / radius))
    gate = float(np.max(gate_values))
    if gate == 0.0:
        return 0.0
    weights = _legacy_gaussian_weights(distances, anchors, indexes)
    values = np.asarray([anchors[index].beta for index in indexes], dtype=np.float64)
    old_delta = _regularized_weighted_sum(
        weights, values, LEGACY_C1_DENOMINATOR_EPSILON)
    return gate * old_delta


def c1_compact_delta(
    query_feature: object,
    anchors: Sequence[CorrectionAnchor],
    *,
    query_position: int,
    query_source_group: str,
    evidence_regularizer: float,
) -> float:
    """Strictly compact C1 weighted signed evidence with absolute shrinkage."""
    regularizer = _finite_scalar(evidence_regularizer, "evidence_regularizer", positive=True)
    _, indexes, distances = _query_and_eligible(
        query_feature, anchors, query_position=query_position,
        query_source_group=query_source_group)
    if not indexes:
        return 0.0
    weights = np.zeros((len(indexes),), dtype=np.float64)
    values = np.asarray([anchors[index].beta for index in indexes], dtype=np.float64)
    for row, index in enumerate(indexes):
        radius = anchors[index].support_radius
        distance = float(distances[row])
        if distance < radius:
            weights[row] = anchors[index].kappa * float(
                compact_support_phi(distance / radius))
    result = _regularized_weighted_sum(weights, values, regularizer)
    bound = float(np.max(np.abs(values)))
    return float(np.clip(result, -bound, bound))


def gated_signed_knn_delta(
    query_feature: object,
    anchors: Sequence[CorrectionAnchor],
    *,
    query_position: int,
    query_source_group: str,
    support_radius: float,
    evidence_regularizer: float,
    k: int = 5,
    amplitude: float = 0.20,
) -> float:
    """Compact-support, distance-weighted Signed kNN with a strict local cutoff.

    First retain only legal anchors with distance strictly below the global
    support radius, then choose the nearest ``min(k, local_count)`` with stable
    index tie-breaking. The compact weights vanish continuously at the radius;
    changes to the kNN set inside the radius may still be discontinuous.
    """
    radius = _finite_scalar(support_radius, "support_radius", positive=True)
    regularizer = _finite_scalar(evidence_regularizer, "evidence_regularizer", positive=True)
    magnitude = _finite_scalar(amplitude, "amplitude", positive=True)
    if isinstance(k, (bool, np.bool_)) or not isinstance(k, (int, np.integer)) or int(k) < 1:
        raise ValueError("k must be a positive integer")
    _, indexes, distances = _query_and_eligible(
        query_feature, anchors, query_position=query_position,
        query_source_group=query_source_group)
    if not indexes:
        return 0.0
    if any(anchors[index].c2_sign not in (-1, 1) for index in indexes):
        raise ValueError("every eligible Signed kNN anchor needs c2_sign -1 or +1")
    local = [row for row, distance in enumerate(distances) if float(distance) < radius]
    if not local:
        return 0.0
    local.sort(key=lambda row: (float(distances[row]), indexes[row]))
    selected = local[:min(int(k), len(local))]
    weights = np.asarray([
        float(compact_support_phi(float(distances[row]) / radius)) for row in selected
    ], dtype=np.float64)
    signs = np.asarray([anchors[indexes[row]].c2_sign for row in selected], dtype=np.float64)
    signed_fraction = _regularized_weighted_sum(weights, signs, regularizer)
    # The exact expression is in [-1,1]. Clamp only floating-point roundoff so
    # a finite declared amplitude remains a hard numerical bound.
    return magnitude * float(np.clip(signed_fraction, -1.0, 1.0))
