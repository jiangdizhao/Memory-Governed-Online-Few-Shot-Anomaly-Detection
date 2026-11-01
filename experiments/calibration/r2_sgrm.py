"""Source-Group-aware residual aggregation, first frozen CPU version.

The numerical core assumes temporal and query-source eligibility have already
been enforced.  The anchor wrapper validates those rules for one category's
single-source anchors and never mutates the supplied descriptors or anchors.
"""
from __future__ import annotations

import math
import numbers
import os
from typing import Any, Sequence

# Set before any optional Torch tensor is handled.  Torch is never imported by
# this module; CPU Torch tensors are converted lazily through their own methods.
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import numpy as np

ALGORITHM_ID = "r2.sgrm.group_max_relevance.v1"
DEFAULT_TAU = 0.10
DEFAULT_EPSILON = 1e-6
FEATURE_NORM_EPSILON = 1e-12
DISTANCE_ROUNDOFF_TOLERANCE = 1e-6
ANCHOR_CONTRACT_TOLERANCE = 1e-6
FLOAT32_MAX = float(np.finfo(np.float32).max)


class SGRMInputError(ValueError):
    """Input violates the frozen SGRM numerical or anchor contract."""


def _positive_finite(value: Any, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Real):
        raise SGRMInputError(f"{name} must be a real scalar")
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise SGRMInputError(f"{name} must be positive and finite")
    return result


def _float_array(value: Any, name: str, *, ndim: int) -> np.ndarray:
    try:
        array = np.asarray(value)
    except Exception as exc:
        raise SGRMInputError(f"{name} is not a numeric array") from exc
    if array.ndim != ndim or array.dtype.kind != "f":
        raise SGRMInputError(f"{name} must be a {ndim}D floating-point array")
    result = np.asarray(array, dtype=np.float64)
    if not np.isfinite(result).all():
        raise SGRMInputError(f"{name} must contain only finite values")
    return result


def _group_layout(group_ids: Sequence[str], beta: np.ndarray,
                  anchor_count: int) -> tuple[list[str], dict[str, list[int]]]:
    if isinstance(group_ids, (str, bytes)):
        raise SGRMInputError("group_ids must be a sequence of Source Group identifiers")
    try:
        groups = list(group_ids)
    except Exception as exc:
        raise SGRMInputError("group_ids must be a sequence of strings") from exc
    if len(groups) != anchor_count:
        raise SGRMInputError("group_ids length must equal the anchor count")
    if any(not isinstance(group, str) or not group.strip() for group in groups):
        raise SGRMInputError("every Source Group identifier must be a nonempty string")
    indices: dict[str, list[int]] = {}
    signs: dict[str, set[int]] = {}
    for index, (group, value) in enumerate(zip(groups, beta)):
        indices.setdefault(group, []).append(index)
        sign = int(np.sign(value))
        if sign:
            signs.setdefault(group, set()).add(sign)
    conflicts = [group for group, group_signs in signs.items() if len(group_signs) > 1]
    if conflicts:
        raise SGRMInputError("conflicting nonzero beta signs within a Source Group")
    return sorted(indices), indices


def sgrm_delta_from_distances(
    distances: Any,
    beta: Any,
    group_ids: Sequence[str],
    *,
    tau: float = DEFAULT_TAU,
    epsilon: float = DEFAULT_EPSILON,
    return_diagnostics: bool = False,
) -> np.ndarray | tuple[np.ndarray, dict[str, Any]]:
    """Aggregate signed residuals from an already eligible ``(Q, A)`` matrix.

    ``distances`` are cosine distances in [0, 2], ``beta`` has one value per
    anchor, and ``group_ids`` contains that anchor's official Source Group.
    This core does not know query time/category/source identity: callers must
    supply only anchors with ``position < current_position`` and exclude the
    query Source Group first.  Distances in [-1e-6, 0) or (2, 2+1e-6] are
    accepted as roundoff and preserved; larger violations are rejected.

    The output is float32. Optional aggregate diagnostics use float64 arrays
    and deliberately omit Source Group identifiers.
    """
    tau_value = _positive_finite(tau, "tau")
    epsilon_value = _positive_finite(epsilon, "epsilon")
    d = _float_array(distances, "distances", ndim=2)
    b = _float_array(beta, "beta", ndim=1)
    query_count, anchor_count = d.shape
    if b.shape != (anchor_count,):
        raise SGRMInputError("beta shape must be (A,) and match distances.shape[1]")
    if np.any(d < -DISTANCE_ROUNDOFF_TOLERANCE) or np.any(d > 2.0 + DISTANCE_ROUNDOFF_TOLERANCE):
        raise SGRMInputError("cosine distances must lie in [0, 2] up to the documented roundoff tolerance")
    group_order, group_indices = _group_layout(group_ids, b, anchor_count)
    group_count = len(group_order)
    means = np.zeros((query_count, group_count), dtype=np.float64)
    log_relevance = np.full((query_count, group_count), -np.inf, dtype=np.float64)
    log_epsilon = math.log(epsilon_value)

    # A within-group log shift computes mu_g without adding epsilon to its
    # denominator. If every weight underflows in the log domain, that group's
    # absolute relevance is zero and it contributes no evidence.
    for group_column, group in enumerate(group_order):
        anchor_indices = group_indices[group]
        with np.errstate(over="ignore", under="ignore", invalid="ignore"):
            scaled = d[:, anchor_indices] / tau_value
            log_weights = -0.5 * scaled * scaled
        for query_index in range(query_count):
            row = log_weights[query_index]
            maximum = float(np.max(row))
            if not math.isfinite(maximum):
                continue
            shifted_weights = [math.exp(float(value) - maximum) for value in row]
            denominator = math.fsum(shifted_weights)
            numerator = math.fsum(
                weight * float(b[anchor_index])
                for weight, anchor_index in zip(shifted_weights, anchor_indices))
            means[query_index, group_column] = numerator / denominator
            log_relevance[query_index, group_column] = maximum

    relevance = np.zeros((query_count, group_count), dtype=np.float64)
    outer_weights = np.zeros_like(relevance)
    delta64 = np.zeros((query_count,), dtype=np.float64)
    total_relevance = np.zeros((query_count,), dtype=np.float64)
    zero_evidence = np.ones((query_count,), dtype=bool)
    for query_index in range(query_count):
        row_logs = log_relevance[query_index]
        finite_logs = [float(value) for value in row_logs if math.isfinite(float(value))]
        group_masses = [math.exp(value) for value in finite_logs]
        total_relevance[query_index] = math.fsum(group_masses)
        if group_count == 0:
            continue
        scale_log = max(log_epsilon, max(finite_logs, default=-math.inf))
        epsilon_scaled = math.exp(log_epsilon - scale_log)
        scaled_masses: list[float] = []
        for group_column, log_mass in enumerate(row_logs):
            mass = (math.exp(float(log_mass) - scale_log)
                    if math.isfinite(float(log_mass)) else 0.0)
            scaled_masses.append(mass)
            relevance[query_index, group_column] = (
                math.exp(float(log_mass)) if math.isfinite(float(log_mass)) else 0.0)
        denominator = epsilon_scaled + math.fsum(scaled_masses)
        weights = [mass / denominator for mass in scaled_masses]
        outer_weights[query_index] = weights
        zero_evidence[query_index] = not any(mass > 0.0 for mass in scaled_masses)
        delta64[query_index] = math.fsum(
            weight * float(means[query_index, column])
            for column, weight in enumerate(weights))

    if not np.isfinite(delta64).all() or np.any(np.abs(delta64) > FLOAT32_MAX):
        raise SGRMInputError("result is not representable as finite float32")
    delta = delta64.astype(np.float32)
    if not return_diagnostics:
        return delta
    diagnostics = {
        "algorithm_id": ALGORITHM_ID,
        "eligible_patch_count": anchor_count,
        "eligible_source_group_count": group_count,
        "group_patch_counts": tuple(len(group_indices[group]) for group in group_order),
        "per_query_total_group_relevance": total_relevance,
        "maximum_absolute_correction": float(np.max(np.abs(delta64))) if query_count else 0.0,
        "zero_evidence_flags": zero_evidence,
        "group_means": means,
        "group_max_relevance": relevance,
        "outer_group_weights": outer_weights,
    }
    return delta, diagnostics


def _cpu_float32(value: Any, name: str) -> np.ndarray:
    device = getattr(value, "device", None)
    if device is not None and getattr(device, "type", None) != "cpu":
        raise SGRMInputError(f"{name} must reside on CPU")
    if hasattr(value, "detach"):
        try:
            value = value.detach().numpy()
        except Exception as exc:
            raise SGRMInputError(f"{name} cannot be read as a CPU array") from exc
    try:
        array = np.asarray(value)
    except Exception as exc:
        raise SGRMInputError(f"{name} is not a numeric array") from exc
    if array.dtype.kind != "f":
        raise SGRMInputError(f"{name} must be floating point")
    with np.errstate(over="ignore", invalid="ignore"):
        result = np.array(array, dtype=np.float32, copy=True, order="C")
    if not np.isfinite(result).all():
        raise SGRMInputError(f"{name} must contain only finite values representable as float32")
    return result


def _anchor_scalar(anchor: Any, field: str) -> float:
    try:
        value = getattr(anchor, field)
    except Exception as exc:
        raise SGRMInputError(f"anchor is missing required {field!r} metadata") from exc
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Real):
        raise SGRMInputError(f"anchor {field} must be a real scalar")
    result = float(value)
    if not math.isfinite(result):
        raise SGRMInputError(f"anchor {field} must be finite")
    return result


def sgrm_delta_from_anchors(
    query_features: Any,
    anchors: Sequence[Any],
    *,
    current_position: int,
    current_source_group: str,
    tau: float = DEFAULT_TAU,
    epsilon: float = DEFAULT_EPSILON,
    return_diagnostics: bool = False,
) -> np.ndarray | tuple[np.ndarray, dict[str, Any]]:
    """Validate existing Anchor-like objects, then compute CPU cosine SGRM.

    Each anchor must expose ``feature``, ``beta``, ``radius``, ``kappa``,
    ``position`` and ``source_group``. All provided anchors must be prior to
    the current event; same-Source-Group anchors are validated then excluded.
    This wrapper is for a single category bank with one unambiguous source group
    per anchor; merged/multi-source prototypes are unsupported and rejected by
    the caller's representation contract.
    """
    tau_value = _positive_finite(tau, "tau")
    epsilon_value = _positive_finite(epsilon, "epsilon")
    if isinstance(current_position, (bool, np.bool_)) or not isinstance(current_position, numbers.Integral):
        raise SGRMInputError("current_position must be an integer")
    if int(current_position) < 0:
        raise SGRMInputError("current_position must be nonnegative")
    if not isinstance(current_source_group, str) or not current_source_group.strip():
        raise SGRMInputError("current_source_group must be a nonempty string")
    query = _cpu_float32(query_features, "query_features")
    if query.ndim != 2 or query.shape[1] == 0:
        raise SGRMInputError("query_features must have shape (Q, D) with D > 0")
    try:
        anchor_list = list(anchors)
    except Exception as exc:
        raise SGRMInputError("anchors must be a sequence of Anchor-like objects") from exc

    valid_features: list[np.ndarray] = []
    valid_beta: list[float] = []
    valid_groups: list[str] = []
    sign_by_group: dict[str, set[int]] = {}
    excluded_same_source = 0
    for anchor in anchor_list:
        try:
            feature_value = getattr(anchor, "feature")
            group = getattr(anchor, "source_group")
            position = getattr(anchor, "position")
        except Exception as exc:
            raise SGRMInputError("anchor is missing feature, position, or source_group") from exc
        if not isinstance(group, str) or not group.strip():
            raise SGRMInputError("every anchor requires one nonempty Source Group identifier")
        if isinstance(position, (bool, np.bool_)) or not isinstance(position, numbers.Integral):
            raise SGRMInputError("anchor position must be an integer")
        if int(position) < 0:
            raise SGRMInputError("anchor position must be nonnegative")
        if int(position) >= int(current_position):
            raise SGRMInputError("current/future anchors are not causally eligible")
        beta_value = _anchor_scalar(anchor, "beta")
        radius_value = _anchor_scalar(anchor, "radius")
        kappa_value = _anchor_scalar(anchor, "kappa")
        if kappa_value <= 0.0 or abs(kappa_value - 1.0) > ANCHOR_CONTRACT_TOLERANCE:
            raise SGRMInputError("initial SGRM adapter requires kappa == 1 within 1e-6")
        if radius_value <= 0.0 or abs(radius_value - tau_value) > ANCHOR_CONTRACT_TOLERANCE:
            raise SGRMInputError("initial SGRM adapter requires anchor radius == tau within 1e-6")
        if abs(beta_value) > FLOAT32_MAX:
            raise SGRMInputError("anchor beta is outside the existing float32 replay range")
        feature = _cpu_float32(feature_value, "anchor feature")
        if feature.ndim != 1 or feature.shape[0] != query.shape[1]:
            raise SGRMInputError("each anchor feature must have shape (D,) matching the query")
        sign = int(np.sign(beta_value))
        if sign:
            sign_by_group.setdefault(group, set()).add(sign)
        if group == current_source_group:
            excluded_same_source += 1
            continue
        valid_features.append(feature)
        valid_beta.append(beta_value)
        valid_groups.append(group)
    if any(len(signs) > 1 for signs in sign_by_group.values()):
        raise SGRMInputError("conflicting nonzero beta signs within a Source Group")

    if valid_features:
        feature_matrix = np.stack(valid_features).astype(np.float32, copy=False)
    else:
        feature_matrix = np.empty((0, query.shape[1]), dtype=np.float32)
    # Match the existing v22 feature contract: float32 L2 normalization with
    # norm + 1e-12, clipped cosine similarity, then cosine distance.
    query_norm = query / (np.linalg.norm(query, axis=1, keepdims=True) + FEATURE_NORM_EPSILON)
    anchor_norm = feature_matrix / (
        np.linalg.norm(feature_matrix, axis=1, keepdims=True) + FEATURE_NORM_EPSILON)
    similarities = np.clip(query_norm @ anchor_norm.T, -1.0, 1.0)
    distances = (1.0 - similarities).astype(np.float32, copy=False)
    result = sgrm_delta_from_distances(
        distances, np.asarray(valid_beta, dtype=np.float32), valid_groups,
        tau=tau_value, epsilon=epsilon_value, return_diagnostics=return_diagnostics)
    if return_diagnostics:
        delta, diagnostics = result
        diagnostics["validated_anchor_count"] = len(anchor_list)
        diagnostics["excluded_same_source_anchor_count"] = excluded_same_source
        return delta, diagnostics
    return result


__all__ = [
    "ALGORITHM_ID", "DEFAULT_EPSILON", "DEFAULT_TAU", "SGRMInputError",
    "sgrm_delta_from_anchors", "sgrm_delta_from_distances",
]
